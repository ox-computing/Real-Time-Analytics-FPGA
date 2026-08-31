"""Time-series-architecture baselines for the WESAD 3-class task, sized for an iCE40UP5K.

The DWN reads 123 hand-engineered features. The obvious question a reviewer asks is
whether a model designed for sequences -- a CNN, an RNN, a Conformer -- does better by
learning its own representation straight off the sampled signals. This script answers it
under exactly the DWN's protocol: the same 1015 windows (row-paired with the feature
cache), leave-one-subject-out, class-weighted loss, several seeds reported as
mean +/- sigma, because LOSO noise here is ~2.3 points balanced.

Every raw-signal model shares one multi-rate convolutional stem that carries each
modality from its own sensor ODR down to a common 1 Hz / 60-step trunk grid, so swapping
the trunk isolates the temporal architecture rather than the input plumbing. Two controls
bracket the result: a stem-only pooling head (how much the trunk actually buys) and an MLP
on the DWN's own 123 features (same information, conventional arithmetic).

The ARIMA family enters as AR(p) fitted by Yule-Walker/Levinson-Durbin on the raw and
first-differenced signal plus the dominant seasonal lag. That is the member of the family
a fixed-latency FPGA pipeline can actually run -- it is linear prediction, all
autocorrelation and back-substitution. Per-window SARIMAX maximum likelihood is an
iterative optimiser with data-dependent runtime and is not implementable here; it is left
out rather than approximated and reported as if it were the same thing.

Parameter and MAC counts are reported next to every result against the UP5K budget:
5280 LUT4, 8 SB_MAC16, 120 kbit EBR, and one free 256 kbit SPRAM block (the other three
hold the 40920-word sensor ring the frontend already needs).
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut

REPO = Path(__file__).resolve().parents[2]

# Temp is dropped to match the locked TD feature set the DWN is trained on.
MODS = ["ACC", "ECG", "EDA", "Resp", "EMG"]
FS = {"ACC": 32, "ECG": 250, "EMG": 350, "EDA": 25, "Resp": 25}
WIN_SEC = 60
TRUNK_LEN = WIN_SEC            # one trunk step per second of window
NCLS = 3

# iCE40UP5K weight-storage budget, in int8 parameters.
EBR_BYTES = 30 * 4096 // 8                 # 30 x 4 kbit block RAMs
FREE_SPRAM_BYTES = 256 * 1024 // 8         # 1 of 4 SPRAM blocks; 3 hold the sample ring
DSP_MACS_PER_INFERENCE = 8 * 12_000_000    # 8 SB_MAC16 x 12 MHz x 1 s hop


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- Data -------------------------------------------------------------------

def load_windows(path: str, device: str):
    """Raw per-window signals at true sensor ODR -> {mod: (N, L) tensor}, y, groups.

    Row order replays extract_features.main(), so row i here is the same window as row i
    of the feature cache the DWN trains on: every comparison below is paired.
    """
    d = np.load(path)
    sig = {m: torch.from_numpy(d[m]).float().to(device) for m in MODS}
    return sig, d["y_multi"].astype(int), d["groups"].astype(int)


def load_features(path: str, drop_mods, drop_feats):
    """The DWN's own feature matrix, selected with baseline.py's column semantics."""
    d = np.load(path, allow_pickle=True)
    names = d["feature_names"]
    mask = np.ones(len(names), bool)
    prefixes = np.array([str(n).split("_")[0] for n in names])
    if drop_mods:
        mask &= ~np.isin(prefixes, drop_mods)
    for sub in drop_feats or []:
        mask &= np.array([sub not in str(n) for n in names])
    return d["X"].astype(float)[:, mask], d["y_multi"].astype(int), d["groups"].astype(int)


def fold_stats(sig, idx):
    """Per-modality mean/std over the training rows only.

    One scalar pair per modality rather than per timestep: the tonic level of EDA and the
    absolute scale of ACC carry class information, and a per-window normalisation would
    delete exactly that. This mirrors the thermometer being fit on training rows only.
    """
    stats = {}
    for m in MODS:
        x = sig[m][idx]
        stats[m] = (x.mean(), x.std().clamp_min(1e-6))
    return stats


def take(sig, idx, stats, aug=None):
    """(B, 1, L) normalised tensors for one batch of rows, optionally augmented.

    Augmentation is applied AFTER normalisation and only on training batches. Time jitter
    is deliberately absent: the dense-hop training cache already supplies genuine
    time-shifted views, which beats rolling a window and splicing its end onto its start.
    What is left is sensor-plausible -- gain error, additive noise, and losing a channel.
    """
    out = {}
    for m in MODS:
        mu, sd = stats[m]
        x = (sig[m][idx] - mu) / sd
        if aug is not None:
            b = x.shape[0]
            if aug["scale"] > 0:
                g = 1.0 + aug["scale"] * torch.randn(b, 1, device=x.device)
                x = x * g
            if aug["noise"] > 0:
                x = x + aug["noise"] * torch.randn_like(x)
            if aug["drop"] > 0:
                keep = (torch.rand(b, 1, device=x.device) >= aug["drop"]).float()
                x = x * keep
        out[m] = x.unsqueeze(1)
    return out


def quantize_weights_int8(model):
    """Per-output-channel symmetric INT8 fake-quantisation of every weight tensor.

    Returns the original tensors so the caller can restore them. This measures the
    WEIGHT half of an integer build only -- activations stay float. That distinction
    matters most for the Conformer, whose LayerNorm and softmax are the operators that
    actually break under activation quantisation, so a clean result here is a necessary
    but not sufficient condition for an integer deployment.
    """
    saved = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() < 2:                      # biases / norm gains stay float
                continue
            saved[name] = p.detach().clone()
            flat = p.reshape(p.shape[0], -1)
            scale = flat.abs().amax(dim=1, keepdim=True) / 127.0
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            p.copy_((torch.round(flat / scale).clamp(-127, 127) * scale).reshape(p.shape))
    return saved


def restore_weights(model, saved):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in saved:
                p.copy_(saved[name])


class ActQuantizer:
    """Per-tensor symmetric INT8 fake-quantisation of module OUTPUTS.

    torch.ao's QAT/PTQ path covers Conv/Linear/BN and little else, so the two architectures
    whose activations are most at risk -- the GRU's gate states and the Conformer's
    LayerNorm and attention output -- would silently stay in float under it. Hooks catch
    every module type instead.

    Ranges are calibrated on TRAINING batches only, at a high percentile rather than the
    max so one outlier window cannot set the scale for the whole tensor.

    This is deliberately pessimistic: a real integer build fuses conv+BN and folds ReLU
    into a clamp, so the tensor is quantised once per fused block, whereas hooking Conv and
    BatchNorm separately quantises twice. Read the result as a lower bound on integer
    accuracy, not an estimate of it.
    """

    TARGETS = (nn.Conv1d, nn.Linear, nn.BatchNorm1d, nn.LayerNorm, nn.GRU,
               nn.MultiheadAttention)

    def __init__(self, model, pct=99.9):
        self.pct = pct
        self.ranges = {}
        self.mode = "off"
        self.handles = []
        for name, m in model.named_modules():
            if isinstance(m, self.TARGETS):
                self.handles.append(m.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(t):
                return None
            if self.mode == "calibrate":
                v = t.detach().abs().flatten().float()
                if v.numel() > 100_000:                 # torch.quantile has a size ceiling
                    v = v[torch.randint(0, v.numel(), (100_000,), device=v.device)]
                a = float(torch.quantile(v, self.pct / 100.0))
                self.ranges[name] = max(self.ranges.get(name, 0.0), a)
                return None
            if self.mode == "quantize":
                a = self.ranges.get(name, 0.0)
                if a <= 0:
                    return None
                s = a / 127.0
                q = (torch.round(t / s).clamp(-127, 127) * s).to(t.dtype)
                return (q,) + tuple(out[1:]) if isinstance(out, tuple) else q
            return None
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def calibrate_activations(model, batches, n_rows, batch_size, max_batches=8):
    """Push a few training batches through so every hook sees a representative range."""
    model.eval()
    seen = 0
    for i in range(0, n_rows, batch_size):
        idx = torch.arange(i, min(i + batch_size, n_rows),
                           device=next(model.parameters()).device)
        model(batches(idx))
        seen += 1
        if seen >= max_batches:
            break


# --- Multi-rate stem --------------------------------------------------------

def stride_factors(total: int):
    """Factorise a decimation ratio into per-layer strides, largest first.

    ECG 250 -> [5,5,5,2], EMG 350 -> [7,5,5,2], ACC 32 -> [4,4,2], EDA/Resp 25 -> [5,5].
    Largest stride first keeps the widest layers shortest, which is what a hardware
    implementation would want too.
    """
    facs, n = [], total
    for p in (7, 5, 4, 3, 2):
        while n % p == 0:
            facs.append(p)
            n //= p
    if n != 1:
        raise ValueError(f"cannot factorise stride {total} into factors <= 7")
    return sorted(facs, reverse=True)


class CausalConv1d(nn.Module):
    """Left-padded convolution: output t depends only on inputs <= t.

    This is what makes a model streamable. A centred convolution needs the whole window
    resident before it can produce its first output, which is why the batch design needs a
    60 s SPRAM ring; a causal one needs only dilation*(k-1) past samples -- a line buffer of
    a few tens of words per stage. Output length matches the centred form exactly, so
    swapping causality in does not move any downstream shape.
    """

    def __init__(self, c_in, c_out, k, stride=1, dilation=1, groups=1):
        super().__init__()
        self.pad = dilation * (k - 1)
        self.conv = nn.Conv1d(c_in, c_out, k, stride=stride, dilation=dilation, groups=groups)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class MultiRateStem(nn.Module):
    """Per-modality strided conv stacks landing every signal on the same 60-step grid.

    Each modality owns its stack because the rates differ by 14x; sharing weights across
    them would be meaningless. Kernel is 2*stride+1 so successive layers see overlapping
    support and the receptive field grows to cover the whole second by the last layer.
    """

    def __init__(self, ch: int, causal: bool = False):
        super().__init__()
        self.ch = ch
        self.causal = causal
        self.branches = nn.ModuleDict()
        for m in MODS:
            layers, c_in = [], 1
            for s in stride_factors(FS[m]):
                k = 2 * s + 1
                conv = (CausalConv1d(c_in, ch, k, stride=s) if causal
                        else nn.Conv1d(c_in, ch, k, stride=s, padding=s))
                layers += [conv, nn.BatchNorm1d(ch), nn.ReLU(inplace=True)]
                c_in = ch
            self.branches[m] = nn.Sequential(*layers)

    def line_buffer_words(self) -> int:
        """Past samples a causal stem must hold, summed over every stage and modality.

        This is the number that replaces the 60 s ring: the streaming design keeps this
        many words instead of 40,920.
        """
        total = 0
        for m in MODS:
            c = 1
            for s in stride_factors(FS[m]):
                total += (2 * s) * c      # dilation=1, k-1 = 2s past samples, c channels
                c = self.ch
        return total

    @property
    def out_channels(self) -> int:
        return self.ch * len(MODS)

    def forward(self, x):
        return torch.cat([self.branches[m](x[m]) for m in MODS], dim=1)


# --- Trunks (B, C, 60) -> logits --------------------------------------------

def readout(h, causal):
    """Pool a (B, C, L) trunk output.

    Causal mode takes the LAST step only: mean/max over the whole sequence is a
    non-causal operation and would smuggle future context back into a design that is
    supposed to emit a class the instant the newest sample lands.
    """
    if causal:
        return h[..., -1]
    return torch.cat([h.mean(-1), h.amax(-1)], dim=-1)


class PoolHead(nn.Module):
    """Control arm: no temporal model at all, just pool the stem and classify."""

    def __init__(self, c_in, causal=False, **_):
        super().__init__()
        self.causal = causal
        self.fc = nn.Linear(c_in if causal else 2 * c_in, NCLS)

    def forward(self, x):
        return self.fc(readout(x, self.causal))


class CNNTrunk(nn.Module):
    """Dilated 1-D conv stack -- the standard TCN-flavoured sequence classifier."""

    def __init__(self, c_in, hidden=32, blocks=2, dropout=0.2, causal=False, **_):
        super().__init__()
        self.causal = causal
        layers, c = [], c_in
        for i in range(blocks):
            d = 2 ** i
            conv = (CausalConv1d(c, hidden, 5, dilation=d) if causal
                    else nn.Conv1d(c, hidden, 5, padding=2 * d, dilation=d))
            layers += [conv, nn.BatchNorm1d(hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            c = hidden
        self.body = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden if causal else 2 * hidden, NCLS)

    def forward(self, x):
        return self.fc(readout(self.body(x), self.causal))


class GRUTrunk(nn.Module):
    """Recurrent arm. GRU rather than LSTM: 3 gates instead of 4 at equal hidden size,
    and the state is a single vector, which is the cheaper thing to hold in registers."""

    def __init__(self, c_in, hidden=32, layers=1, dropout=0.2, causal=False, **_):
        super().__init__()
        self.causal = causal
        self.rnn = nn.GRU(c_in, hidden, num_layers=layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden if causal else 2 * hidden, NCLS)

    def forward(self, x):
        h, _ = self.rnn(x.transpose(1, 2))
        pooled = h[:, -1] if self.causal else torch.cat([h.mean(1), h[:, -1]], dim=-1)
        return self.fc(self.drop(pooled))

    def state_words(self) -> int:
        """Hidden state carried between steps -- what a streaming build stores instead
        of a 60 s sample ring."""
        return self.rnn.num_layers * self.rnn.hidden_size


class ConvModule(nn.Module):
    """Conformer convolution module: pointwise-GLU -> depthwise -> BN -> SiLU -> pointwise."""

    def __init__(self, d, kernel=15, dropout=0.1, causal=False):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = (CausalConv1d(d, d, kernel, groups=d) if causal
                   else nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d))
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                     # x: (B, L, d)
        h = self.norm(x).transpose(1, 2)
        h = F.glu(self.pw1(h), dim=1)
        h = F.silu(self.bn(self.dw(h)))
        return self.drop(self.pw2(h).transpose(1, 2))


class ConformerBlock(nn.Module):
    """Half-step FFN, self-attention, convolution, half-step FFN, then a final norm."""

    def __init__(self, d, heads=4, ff_mult=2, kernel=15, dropout=0.1, causal=False):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff_mult * d), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(ff_mult * d, d),
                                 nn.Dropout(dropout))
        self.attn_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.conv = ConvModule(d, kernel, dropout, causal=causal)
        self.ff2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff_mult * d), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(ff_mult * d, d),
                                 nn.Dropout(dropout))
        self.out_norm = nn.LayerNorm(d)

    def forward(self, x, attn_mask=None):
        x = x + 0.5 * self.ff1(x)
        h = self.attn_norm(x)
        x = x + self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.out_norm(x)


class ConformerTrunk(nn.Module):
    """Transformer arm, in its convolution-augmented form -- the variant that is actually
    used for signals, since plain self-attention has no locality prior and 60 steps of
    biosignal is far too little data to learn one."""

    def __init__(self, c_in, d=40, blocks=2, heads=4, kernel=15, dropout=0.1,
                 causal=False, **_):
        super().__init__()
        self.causal = causal
        self.proj = nn.Linear(c_in, d)
        self.pos = nn.Parameter(torch.zeros(1, TRUNK_LEN, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([ConformerBlock(d, heads, kernel=kernel, dropout=dropout,
                                                    causal=causal) for _ in range(blocks)])
        self.fc = nn.Linear(d, NCLS)
        if causal:
            mask = torch.triu(torch.ones(TRUNK_LEN, TRUNK_LEN, dtype=torch.bool), diagonal=1)
            self.register_buffer("attn_mask", mask)
        else:
            self.attn_mask = None

    def forward(self, x):
        h = self.proj(x.transpose(1, 2)) + self.pos
        for blk in self.blocks:
            h = blk(h, attn_mask=self.attn_mask)
        return self.fc(h[:, -1] if self.causal else h.mean(1))


class TSNet(nn.Module):
    def __init__(self, stem_ch, trunk_cls, trunk_kw, causal=False):
        super().__init__()
        self.causal = causal
        self.stem = MultiRateStem(stem_ch, causal=causal)
        self.trunk = trunk_cls(self.stem.out_channels, causal=causal, **trunk_kw)

    def forward(self, x):
        return self.trunk(self.stem(x))


class FeatMLP(nn.Module):
    """Control arm: the DWN's own 123 features through a conventional MLP."""

    def __init__(self, n_in, hidden=64, layers=2, dropout=0.3, **_):
        super().__init__()
        mods, c = [], n_in
        for _ in range(layers):
            mods += [nn.Linear(c, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                     nn.Dropout(dropout)]
            c = hidden
        mods.append(nn.Linear(c, NCLS))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


# --- Cost accounting --------------------------------------------------------

def count_macs(model, sample):
    """Multiply-accumulates for one inference.

    Conv1d and Linear are caught with forward hooks. nn.GRU and nn.MultiheadAttention are
    added by formula: their internal matmuls use functional calls on flat Parameters, so
    no child module fires a hook and a pure-hook count would silently under-report the two
    architectures whose cost matters most.
    """
    total = [0]

    def conv_hook(mod, _in, out):
        total[0] += out.shape[-1] * mod.out_channels * (mod.in_channels // mod.groups) * mod.kernel_size[0]

    def lin_hook(mod, _in, out):
        per_sample = int(np.prod(out.shape[:-1])) // out.shape[0]
        total[0] += per_sample * mod.in_features * mod.out_features

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(lin_hook))
    model.eval()
    with torch.no_grad():
        model(sample)
    for h in handles:
        h.remove()

    extra = 0
    for m in model.modules():
        if isinstance(m, nn.GRU):
            for layer in range(m.num_layers):
                c_in = m.input_size if layer == 0 else m.hidden_size
                extra += TRUNK_LEN * 3 * (c_in * m.hidden_size + m.hidden_size ** 2)
        elif isinstance(m, nn.MultiheadAttention):
            d = m.embed_dim
            extra += TRUNK_LEN * 4 * d * d + 2 * TRUNK_LEN * TRUNK_LEN * d
    return total[0] + extra


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def budget_note(params, macs):
    """Where an int8 build of this model would have to keep its weights, and whether the
    8 hard MACs can finish it inside the 1 s hop."""
    if params <= EBR_BYTES:
        where = "EBR"
    elif params <= EBR_BYTES + FREE_SPRAM_BYTES:
        where = "EBR+SPRAM"
    else:
        where = "OVER"
    return where, macs / DSP_MACS_PER_INFERENCE


# --- ARIMA-family features --------------------------------------------------

def _acf(x, nlags):
    """Normalised autocorrelation to nlags, via FFT, for a whole (N, L) block at once."""
    x = x - x.mean(axis=1, keepdims=True)
    n = x.shape[1]
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    spec = np.fft.rfft(x, nfft, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), nfft, axis=1)[:, :nlags + 1]
    r0 = ac[:, :1].copy()
    r0[r0 <= 0] = 1.0
    return ac / r0, ac[:, 0] / n


def _levinson(r, p):
    """Yule-Walker AR(p) by Levinson-Durbin, vectorised over the window axis.

    r is the normalised ACF (r[:,0] == 1), so the returned prediction error is already the
    fraction of variance the AR model fails to explain.
    """
    n = r.shape[0]
    a = np.zeros((n, p + 1))
    a[:, 0] = 1.0
    err = np.ones(n)
    for i in range(1, p + 1):
        acc = r[:, i].copy()
        if i > 1:
            acc += np.einsum("nj,nj->n", a[:, 1:i], r[:, i - 1:0:-1])
        k = -acc / np.where(err > 1e-12, err, 1e-12)
        k = np.clip(k, -0.999, 0.999)
        if i > 1:
            prev = a[:, 1:i].copy()
            a[:, 1:i] = prev + k[:, None] * prev[:, ::-1]
        a[:, i] = k
        err = err * (1.0 - k ** 2)
    return a[:, 1:], np.maximum(err, 1e-12)


def ar_features(x, fs, p):
    """AR coefficients on the level and on the first difference, plus the seasonal lag.

    The differenced fit is the 'I' of ARIMA -- a stress-driven EDA ramp is non-stationary
    in level and the differenced series is what an ARIMA(p,1,0) would actually model. The
    seasonal term is read off the ACF peak between 0.3 s and 8 s, which spans both the
    cardiac and the respiratory period, instead of being fixed a priori.
    Level and scale are carried explicitly because a classifier needs them and the fitted
    values of an ARIMA carry them implicitly.
    """
    lag_lo, lag_hi = int(0.3 * fs), int(8 * fs)
    nlags = max(lag_hi, p + 1)
    r, var = _acf(x, nlags)
    a, err = _levinson(r, p)

    dx = np.diff(x, axis=1)
    rd, vard = _acf(dx, p + 1)
    ad, errd = _levinson(rd, p)

    band = r[:, lag_lo:lag_hi + 1]
    peak = band.argmax(axis=1)
    seas_lag = (peak + lag_lo) / fs
    seas_val = band[np.arange(band.shape[0]), peak]

    return np.column_stack([
        a, ad,
        np.log(err), np.log(errd),
        np.log(var + 1e-12), np.log(vard + 1e-12),
        seas_lag, seas_val,
        x.mean(axis=1),
    ])


def build_ar_matrix(sig, p):
    cols, names = [], []
    for m in MODS:
        x = sig[m].cpu().numpy().astype(np.float64)
        f = ar_features(x, FS[m], p)
        cols.append(f)
        names += [f"{m}_ar{i+1}" for i in range(p)] + [f"{m}_dar{i+1}" for i in range(p)]
        names += [f"{m}_logerr", f"{m}_dlogerr", f"{m}_logvar", f"{m}_dlogvar",
                  f"{m}_seas_lag", f"{m}_seas_acf", f"{m}_mean"]
    X = np.hstack(cols)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


# --- Training ---------------------------------------------------------------

def class_weights(y, k=NCLS):
    counts = np.bincount(y, minlength=k).astype(float)
    with np.errstate(divide="ignore"):
        w = y.size / (k * counts)
    w[~np.isfinite(w)] = 0.0
    return torch.tensor(w, dtype=torch.float32)


def train_one(model, batches, y_tr, args, weight, device):
    """Fixed epoch budget, identical across architectures -- no per-arch early stopping,
    which would need a validation subject and quietly change the protocol."""
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    milestones = [int(args.epochs * 0.5), int(args.epochs * 0.8)]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)
    weight = weight.to(device)
    n = y_tr.numel()
    model.train()
    for _ in range(args.epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            if idx.numel() < 2:
                continue
            opt.zero_grad()
            loss = F.cross_entropy(model(batches(idx)), y_tr[idx], weight=weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def predict(model, batches, n, batch_size):
    model.eval()
    out = []
    for i in range(0, n, batch_size):
        idx = torch.arange(i, min(i + batch_size, n), device=next(model.parameters()).device)
        out.append(model(batches(idx)).argmax(dim=1).cpu())
    return torch.cat(out).numpy()


def metrics(y_true, y_pred):
    return (accuracy_score(y_true, y_pred),
            balanced_accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro"))


def run_loso_raw(data, make_model, args, seed, device):
    """One seed over the 15 subject folds, pooled predictions, on the raw windows.

    data carries a TRAIN set and an EVAL set that may be different caches: training can
    use a denser window hop for more data while evaluation stays on the original 30 s grid,
    so the test rows remain the ones the DWN was scored on and the comparison stays paired.
    Held-out subjects are excluded from the training cache by subject id, never by row, so
    a denser hop cannot leak a neighbouring window of the test subject into training.
    """
    set_seed(seed)
    sig_tr, y_tr_all, g_tr = data["train"]
    sig_ev, y_ev_all, g_ev = data["eval"]
    aug = args.aug_params
    y_true, y_pred, y_pred_q, y_pred_qa = [], [], [], []

    for subj in np.unique(g_ev):
        tr = np.flatnonzero(g_tr != subj)
        te = np.flatnonzero(g_ev == subj)
        tr_t = torch.from_numpy(tr).to(device)
        te_t = torch.from_numpy(te).to(device)
        stats = fold_stats(sig_tr, tr_t)
        model = make_model().to(device)
        train_one(model, lambda i: take(sig_tr, tr_t[i], stats, aug),
                  torch.from_numpy(y_tr_all[tr]).long().to(device), args,
                  class_weights(y_tr_all[tr]), device)
        y_true.append(y_ev_all[te])
        y_pred.append(predict(model, lambda i: take(sig_ev, te_t[i], stats), len(te),
                              args.batch_size))
        if args.ptq:
            saved = quantize_weights_int8(model)
            y_pred_q.append(predict(model, lambda i: take(sig_ev, te_t[i], stats), len(te),
                                    args.batch_size))
            if args.ptq_act:
                # Weights stay quantised while activation ranges are calibrated, so the
                # scales are observed on the tensors the integer build actually produces.
                aq = ActQuantizer(model, args.ptq_pct)
                aq.mode = "calibrate"
                calibrate_activations(model, lambda i: take(sig_tr, tr_t[i], stats),
                                      len(tr), args.batch_size)
                aq.mode = "quantize"
                y_pred_qa.append(predict(model, lambda i: take(sig_ev, te_t[i], stats),
                                         len(te), args.batch_size))
                aq.remove()
            restore_weights(model, saved)

    y_true = np.concatenate(y_true)
    out = metrics(y_true, np.concatenate(y_pred))
    if args.ptq:
        q = metrics(y_true, np.concatenate(y_pred_q))
        qa = metrics(y_true, np.concatenate(y_pred_qa)) if args.ptq_act else None
        return out, q, qa
    return out, None, None


def run_loso_tabular(X, y, groups, make_model, args, seed, device, train_set=None):
    """Same protocol for the feature-vector arms; train-median impute then train z-score.

    train_set, when given, is a separate (X, y, groups) drawn from the denser-hop cache:
    the tabular arms then get the same extra training data the sequence models get, while
    still being scored on the original grid. Folds are formed by subject id so the held-out
    subject contributes nothing to either the fit or the imputation medians.
    """
    set_seed(seed)
    Xt, yt, gt = train_set if train_set is not None else (X, y, groups)
    y_true, y_pred = [], []
    for subj in np.unique(groups):
        tr = np.flatnonzero(gt != subj)
        te = np.flatnonzero(groups == subj)
        Xtr, Xte = Xt[tr], X[te]
        y_tr_fold = yt[tr]
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(Xtr), Xtr, med)
        Xte = np.where(np.isfinite(Xte), Xte, med)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd < 1e-9] = 1.0
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        if make_model is None:
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=args.logreg_c)
            clf.fit(Xtr, y_tr_fold)
            pred = clf.predict(Xte)
        else:
            t_tr = torch.from_numpy(Xtr).float().to(device)
            t_te = torch.from_numpy(Xte).float().to(device)
            model = make_model(X.shape[1]).to(device)
            train_one(model, lambda i: t_tr[i],
                      torch.from_numpy(y_tr_fold).long().to(device),
                      args, class_weights(y_tr_fold), device)
            pred = predict(model, lambda i: t_te[i], len(te), args.batch_size)
        y_true.append(y[te])
        y_pred.append(pred)
    return metrics(np.concatenate(y_true), np.concatenate(y_pred))


# --- Architecture registry --------------------------------------------------

TRUNKS = {"stem": PoolHead, "cnn": CNNTrunk, "gru": GRUTrunk, "conformer": ConformerTrunk}

PRESETS = {
    "stem":       [("stem_c8", dict(stem_ch=8), {})],
    "cnn":        [("cnn_c8_h16", dict(stem_ch=8), dict(hidden=16, blocks=2)),
                   ("cnn_c8_h32", dict(stem_ch=8), dict(hidden=32, blocks=2)),
                   ("cnn_c16_h32", dict(stem_ch=16), dict(hidden=32, blocks=3))],
    "gru":        [("gru_c8_h16", dict(stem_ch=8), dict(hidden=16)),
                   ("gru_c8_h32", dict(stem_ch=8), dict(hidden=32)),
                   ("gru_c16_h48", dict(stem_ch=16), dict(hidden=48))],
    "conformer":  [("conf_c8_d24b1", dict(stem_ch=8), dict(d=24, blocks=1, heads=4)),
                   ("conf_c8_d40b2", dict(stem_ch=8), dict(d=40, blocks=2, heads=4)),
                   ("conf_c16_d64b2", dict(stem_ch=16), dict(d=64, blocks=2, heads=4))],
}

FEAT_PRESETS = [("mlp_h32", dict(hidden=32, layers=2)),
                ("mlp_h64", dict(hidden=64, layers=2)),
                ("mlp_h128", dict(hidden=128, layers=2)),
                ("mlp_h256", dict(hidden=256, layers=2)),
                ("mlp_h64x3", dict(hidden=64, layers=3)),
                ("mlp_h128x3", dict(hidden=128, layers=3))]


def make_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", default=str(REPO / "data/wesad_cache/wesad_windows_resampled6.npz"))
    ap.add_argument("--features", default=str(REPO / "data/wesad_cache/wesad_features_dt2048k13_6mod.npz"))
    ap.add_argument("--arch", default="stem,cnn,gru,conformer,ar,mlpfeat",
                    help="comma list: stem,cnn,gru,conformer,ar,arlin,mlpfeat")
    ap.add_argument("--only", default=None, help="comma list of preset names to restrict to")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--ar-order", type=int, default=8)
    ap.add_argument("--logreg-c", type=float, default=0.1)
    ap.add_argument("--train-windows", default=None,
                    help="separate (denser-hop) cache to TRAIN on; eval stays on --windows")
    ap.add_argument("--causal", action="store_true",
                    help="causal stem/trunk + last-step readout: the streamable variant")
    ap.add_argument("--ptq", action="store_true",
                    help="also report per-channel INT8 weight-quantised accuracy")
    ap.add_argument("--ptq-act", action="store_true",
                    help="additionally fake-quantise activations to INT8 (implies --ptq)")
    ap.add_argument("--ptq-pct", type=float, default=99.9,
                    help="percentile of |activation| used to set each tensor's scale")
    ap.add_argument("--aug-scale", type=float, default=0.0, help="per-channel gain jitter sd")
    ap.add_argument("--aug-noise", type=float, default=0.0, help="additive noise sd")
    ap.add_argument("--aug-drop", type=float, default=0.0, help="per-channel dropout prob")
    ap.add_argument("--tag", default="")
    ap.add_argument("--results-root", default=str(REPO / "results"))
    return ap


def summarise(name, per_seed, extra):
    def stat(key):
        v = np.array([r[key] for r in per_seed])
        return float(v.mean()), float(v.std())
    acc_m, acc_s = stat("acc")
    bal_m, bal_s = stat("balanced")
    f1_m, f1_s = stat("macro_f1")
    rec = dict(name=name, acc_mean=acc_m, acc_std=acc_s, balanced_mean=bal_m,
               balanced_std=bal_s, macro_f1_mean=f1_m, macro_f1_std=f1_s,
               per_seed=per_seed, **extra)
    print(f"  {name:<16} acc {acc_m:.4f}+/-{acc_s:.4f}  bal {bal_m:.4f}+/-{bal_s:.4f}  "
          f"mF1 {f1_m:.4f}+/-{f1_s:.4f}")
    return rec


def main():
    args = make_arg_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    archs = [a.strip() for a in args.arch.split(",") if a.strip()]
    only = set(x.strip() for x in args.only.split(",")) if args.only else None
    seeds = list(range(args.seed, args.seed + args.seeds))

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.results_root) / f"wesad-tsbase{('-' + args.tag) if args.tag else ''}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ptq_act:
        args.ptq = True
    args.aug_params = None
    if max(args.aug_scale, args.aug_noise, args.aug_drop) > 0:
        args.aug_params = dict(scale=args.aug_scale, noise=args.aug_noise, drop=args.aug_drop)

    sig, y, groups = load_windows(args.windows, device)
    print(f"eval windows {len(y)}  classes {np.bincount(y).tolist()}  "
          f"subjects {len(np.unique(groups))}")
    if args.train_windows:
        sig_tr, y_tr, g_tr = load_windows(args.train_windows, device)
        print(f"train windows {len(y_tr)} (dense)  classes {np.bincount(y_tr).tolist()}")
    else:
        sig_tr, y_tr, g_tr = sig, y, groups
    raw_data = {"train": (sig_tr, y_tr, g_tr), "eval": (sig, y, groups)}
    if args.causal:
        print("causal mode: streamable stem + last-step readout")
    if args.aug_params:
        print(f"augmentation {args.aug_params}")
    results = []

    for arch in archs:
        if arch in TRUNKS:
            for name, stem_kw, trunk_kw in PRESETS[arch]:
                if only and name not in only:
                    continue
                def make_model(stem_kw=stem_kw, trunk_kw=trunk_kw, arch=arch):
                    return TSNet(stem_kw["stem_ch"], TRUNKS[arch], trunk_kw,
                                 causal=args.causal)
                probe = make_model().to(device)
                idx0 = torch.arange(2, device=device)
                stats0 = fold_stats(sig, torch.arange(len(y), device=device))
                params = count_params(probe)
                macs = count_macs(probe, take(sig, idx0, stats0))
                where, util = budget_note(params, macs)
                # What a streaming build holds instead of the 60 s / 40920-word ring.
                buf = probe.stem.line_buffer_words() if args.causal else 40920
                if args.causal and hasattr(probe.trunk, "state_words"):
                    buf += probe.trunk.state_words()
                del probe
                label = name + ("_causal" if args.causal else "")
                print(f"\n[{label}] params {params:,}  MAC/inf {macs:,}  int8 weights -> {where}"
                      f"  DSP util {util:.3f}  sample buffer {buf:,} words")
                per_seed = []
                for s in seeds:
                    t0 = time.time()
                    (acc, bal, mf1), q, qa = run_loso_raw(raw_data, make_model, args,
                                                          s, device)
                    rec = dict(seed=s, acc=acc, balanced=bal, macro_f1=mf1,
                               elapsed_sec=round(time.time() - t0, 1))
                    if q:
                        rec.update(acc_int8=q[0], balanced_int8=q[1], macro_f1_int8=q[2])
                    if qa:
                        rec.update(acc_int8act=qa[0], balanced_int8act=qa[1],
                                   macro_f1_int8act=qa[2])
                    per_seed.append(rec)
                    extra = f"  | w8 bal {q[1]:.4f}" if q else ""
                    extra += f"  | w8a8 bal {qa[1]:.4f}" if qa else ""
                    print(f"    seed {s}  acc {acc:.4f}  bal {bal:.4f}  mF1 {mf1:.4f}{extra}  "
                          f"({rec['elapsed_sec']:.0f}s)")
                meta = dict(family=arch, params=params, macs=macs, weight_store=where,
                            dsp_util=util, input="raw", causal=args.causal,
                            buffer_words=buf)
                if args.ptq:
                    meta["balanced_int8_mean"] = float(
                        np.mean([r["balanced_int8"] for r in per_seed]))
                if args.ptq_act:
                    meta["balanced_int8act_mean"] = float(
                        np.mean([r["balanced_int8act"] for r in per_seed]))
                results.append(summarise(label, per_seed, meta))

        elif arch in ("ar", "arlin"):
            t0 = time.time()
            Xar, ar_names = build_ar_matrix(sig, args.ar_order)
            ar_train = None
            if args.train_windows:
                Xar_tr, _ = build_ar_matrix(sig_tr, args.ar_order)
                ar_train = (Xar_tr, y_tr, g_tr)
            print(f"\n[ar] AR({args.ar_order}) features {Xar.shape} in {time.time()-t0:.1f}s"
                  + (f"  (+{Xar_tr.shape[0]} dense train rows)" if ar_train else ""))
            if arch == "arlin":
                per_seed = []
                for s in seeds:
                    acc, bal, mf1 = run_loso_tabular(Xar, y, groups, None, args, s, device, ar_train)
                    per_seed.append(dict(seed=s, acc=acc, balanced=bal, macro_f1=mf1))
                    break   # logistic regression is deterministic; one pass is the answer
                results.append(summarise("ar_logreg", per_seed, dict(
                    family="ar", params=Xar.shape[1] * NCLS + NCLS,
                    macs=Xar.shape[1] * NCLS, weight_store="EBR", dsp_util=0.0,
                    input="ar_features", ar_order=args.ar_order)))
            else:
                for name, mlp_kw in FEAT_PRESETS:
                    if only and name not in only:
                        continue
                    def make_mlp(n_in, mlp_kw=mlp_kw):
                        return FeatMLP(n_in, **mlp_kw)
                    probe = FeatMLP(Xar.shape[1], **mlp_kw)
                    params, macs = count_params(probe), count_macs(probe, torch.zeros(2, Xar.shape[1]))
                    where, util = budget_note(params, macs)
                    print(f"\n[ar_{name}] params {params:,}  MAC/inf {macs:,} -> {where}")
                    per_seed = []
                    for s in seeds:
                        acc, bal, mf1 = run_loso_tabular(Xar, y, groups, make_mlp, args, s, device, ar_train)
                        per_seed.append(dict(seed=s, acc=acc, balanced=bal, macro_f1=mf1))
                        print(f"    seed {s}  acc {acc:.4f}  bal {bal:.4f}  mF1 {mf1:.4f}")
                    results.append(summarise(f"ar_{name}", per_seed, dict(
                        family="ar", params=params, macs=macs, weight_store=where,
                        dsp_util=util, input="ar_features", ar_order=args.ar_order)))
                per_seed = []
                acc, bal, mf1 = run_loso_tabular(Xar, y, groups, None, args, seeds[0], device, ar_train)
                per_seed.append(dict(seed=seeds[0], acc=acc, balanced=bal, macro_f1=mf1))
                results.append(summarise("ar_logreg", per_seed, dict(
                    family="ar", params=Xar.shape[1] * NCLS + NCLS,
                    macs=Xar.shape[1] * NCLS, weight_store="EBR", dsp_util=0.0,
                    input="ar_features", ar_order=args.ar_order)))

        elif arch == "mlpfeat":
            Xf, yf, gf = load_features(args.features, ["Temp"], ["skew", "kurtosis", "cvrr"])
            assert np.array_equal(yf, y) and np.array_equal(gf, groups), "feature cache not row-paired"
            print(f"\n[mlpfeat] DWN feature matrix {Xf.shape}")
            for name, mlp_kw in FEAT_PRESETS:
                if only and name not in only:
                    continue
                def make_mlp(n_in, mlp_kw=mlp_kw):
                    return FeatMLP(n_in, **mlp_kw)
                probe = FeatMLP(Xf.shape[1], **mlp_kw)
                params, macs = count_params(probe), count_macs(probe, torch.zeros(2, Xf.shape[1]))
                where, util = budget_note(params, macs)
                print(f"[{name}] params {params:,}  MAC/inf {macs:,} -> {where}")
                per_seed = []
                for s in seeds:
                    acc, bal, mf1 = run_loso_tabular(Xf, y, groups, make_mlp, args, s, device)
                    per_seed.append(dict(seed=s, acc=acc, balanced=bal, macro_f1=mf1))
                    print(f"    seed {s}  acc {acc:.4f}  bal {bal:.4f}  mF1 {mf1:.4f}")
                results.append(summarise(f"feat_{name}", per_seed, dict(
                    family="mlpfeat", params=params, macs=macs, weight_store=where,
                    dsp_util=util, input="dwn_features")))
        else:
            raise SystemExit(f"unknown arch {arch!r}")

    summary = dict(run_id=run_id, args=vars(args), seeds=seeds, results=results)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
