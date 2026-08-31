"""Multimodal fusion arms for the WESAD DWN: constrained layer-1 mappings and
cross-modal features.

The flat-vector baseline hands the DWN 123 per-channel marginal statistics as one
492-bit thermometer vector and lets a learnable layer-1 mapping decide what to read.
Two things follow from that, and this module builds the arms that test both:

1. The mapping is the only inductive bias a DWN has, and on the exported baseline it
   lands close to a random draw on the modality axis (62% of LUT4s span 3-4 modalities
   vs 76% for a random mapping). Since a mapping is only wires, a structural prior on
   it is free in silicon -- so it is worth forcing the two structures the free mapping
   is not choosing: within-modality specialists, and guaranteed cross-modal tuples.

2. Every base feature is computed from a single channel, and a LUT4 reading thermometer
   bits cannot recover a ratio or a product across two channels -- with 4 bits each it
   sees at most a coarse rectangle in a 5x5 grid. Cross-modal quantities therefore have
   to be handed to it explicitly, either as scalar features before the thermometer or as
   popcount combinations of two features' thermometer codes after it.
"""

import numpy as np
import torch
from torch import nn

import torch_dwn as dwn
from torch_dwn.mapping import LearnableMapping, LearnableMappingFunction


# --- Constrained mappings ---------------------------------------------------

MASK_NEG = -1e9  # finite, so a fully-masked column can never produce a NaN softmax


class MaskedLearnableMapping(LearnableMapping):
    """LearnableMapping restricted to a per-pin subset of the input bits.

    torch-dwn picks pin j's input as weights[:, j].argmax(), so confining a pin to a
    modality is just forcing every disallowed row of that column to lose. The weights
    stay trainable within the allowed set, which is the whole point: the constrained
    arms must differ from the free arm by the constraint alone, not by also giving up
    the learnable mapping. masked_fill's backward zeroes the gradient on masked entries,
    so they never drift back into contention.
    """

    def __init__(self, input_size, output_size, mask, tau=0.001):
        super().__init__(input_size, output_size, tau)
        assert mask.shape == (input_size, output_size)
        assert bool(mask.any(dim=0).all()), "every pin needs at least one allowed input bit"
        self.register_buffer("mask", mask)

    def forward(self, x):
        weights = self.weights.masked_fill(~self.mask, MASK_NEG)
        return LearnableMappingFunction.apply(x, weights, torch.tensor(self.tau))


def modality_bit_blocks(feature_names, num_bits):
    """modality -> input-bit indices, from the '<MOD>_' prefix convention.

    The thermometer is flattened feature-major, so feature i owns bits
    [i*num_bits, (i+1)*num_bits). Returns an insertion-ordered dict so the pin plans
    below are deterministic given the cache's column order.
    """
    blocks = {}
    for i, name in enumerate(feature_names):
        mod = str(name).split("_")[0]
        blocks.setdefault(mod, []).extend(range(i * num_bits, (i + 1) * num_bits))
    return {m: np.asarray(v, dtype=np.int64) for m, v in blocks.items()}


def pin_modality_plan(variant, n_luts, n, modalities):
    """(n_luts, n) array naming the modality each pin may read.

    within  -- all n pins of a LUT on one modality; LUTs dealt round-robin over the
               modalities, so every modality gets an equal share of layer 1 regardless
               of how many features it contributed. This is the grouped sub-DWN: at
               100 LUTs and 5 modalities it is exactly 5 blocks of 20 feeding the
               fusion layer, and it is also the only arm that guarantees no modality
               is left unread.
    cross4  -- n distinct modalities per LUT, cycling the 4-subsets. Maximum breadth,
               but at n=4 it buys one threshold per channel and nothing more.
    cross2  -- two modalities, n/2 pins each, cycling all unordered pairs. Half the
               breadth of cross4 but two bits per channel, which is the cheapest tuple
               that can express a 2-D region rather than a pair of thresholds.
    """
    mods = list(modalities)
    plan = np.empty((n_luts, n), dtype=object)
    if variant == "within":
        for i in range(n_luts):
            plan[i, :] = mods[i % len(mods)]
    elif variant == "cross4":
        subsets = [mods[j:] + mods[:j] for j in range(len(mods))]
        for i in range(n_luts):
            pick = subsets[i % len(subsets)][:n]
            plan[i, :] = [pick[j % len(pick)] for j in range(n)]
    elif variant == "cross2":
        pairs = [(a, b) for ai, a in enumerate(mods) for b in mods[ai + 1:]]
        for i in range(n_luts):
            a, b = pairs[i % len(pairs)]
            plan[i, :] = [a] * (n // 2) + [b] * (n - n // 2)
    else:
        raise ValueError(f"unknown mapping variant {variant!r}")
    return plan


def build_mask(plan, blocks, input_size):
    """(input_size, n_luts*n) bool: True where pin p is allowed to read input bit b."""
    n_luts, n = plan.shape
    mask = torch.zeros(input_size, n_luts * n, dtype=torch.bool)
    for i in range(n_luts):
        for j in range(n):
            mask[torch.from_numpy(blocks[plan[i, j]]), i * n + j] = True
    return mask


def build_model_variant(input_size, layer_sizes, n, tau, k, variant,
                        feature_names=None, num_bits=None):
    """train_dwn.build_model with layer 0's mapping swapped for the arm's.

    Layers after the first keep the random fixed mapping the baseline uses -- the
    experiment is about how layer 1 reads the thermometer, and changing the interior
    too would confound it. 'free' and 'randfix' are the two controls: free is the
    locked baseline, randfix drops the learnable mapping so the constrained arms can
    be read against what learnability alone is worth.
    """
    layers, in_size = [], input_size
    for i, size in enumerate(layer_sizes):
        first = i == 0
        mode = "random" if (first and variant == "randfix") else \
               ("learnable" if first else "random")
        layer = dwn.LUTLayer(in_size, size, n=n, mapping=mode)
        if first and variant not in ("free", "randfix"):
            blocks = modality_bit_blocks(feature_names, num_bits)
            plan = pin_modality_plan(variant, size, n, blocks.keys())
            layer.mapping = MaskedLearnableMapping(
                in_size, size * n, build_mask(plan, blocks, in_size))
        layers.append(layer)
        in_size = size
    layers.append(dwn.GroupSum(k=k, tau=tau))
    return nn.Sequential(*layers)


# --- Cross-modal features ---------------------------------------------------

# (name, op, numerator/left column, denominator/right column). 'div' is scale-free by
# construction and 'mul' is monotone in both operands, so neither needs a normaliser --
# which matters because any normaliser would have to be re-fit per fold to stay
# leak-free. The per-feature quantile thermometer is rank-based, so a heavy-tailed
# ratio costs nothing.
CROSS_MODAL = {
    # Cardio-respiratory coupling: vagal tone read per breath rather than per window.
    "rsa": [
        ("X_rsa_rmssd_per_breath", "div", "ECG_hrv_rmssd", "Resp_psd_peak_freq"),
        ("X_rsa_rr_x_respfreq", "mul", "ECG_hrv_mean_rr", "Resp_psd_peak_freq"),
        ("X_rsa_centroid_ratio", "div", "ECG_psd_centroid", "Resp_psd_centroid"),
    ],
    # Electrodermal response per unit of movement: separates stress sweat from exertion.
    "eda_acc": [
        ("X_eda_std_per_motion", "div", "EDA_t_std", "ACC_t_std"),
        ("X_eda_slope_per_motion", "div", "EDA_t_slope", "ACC_t_rms"),
        ("X_eda_power_per_motion", "div", "EDA_psd_total", "ACC_psd_total"),
    ],
    # Muscle activity not explained by gross movement.
    "emg_acc": [
        ("X_emg_rms_per_motion", "div", "EMG_t_rms", "ACC_t_rms"),
        ("X_emg_power_per_motion", "div", "EMG_psd_total", "ACC_psd_total"),
    ],
    # Sympathetic co-activation: both branches rising together is the stress signature.
    "ecg_eda": [
        ("X_hr_x_scl", "mul", "ECG_hrv_mean_hr", "EDA_t_mean"),
        ("X_hr_x_eda_slope", "mul", "ECG_hrv_mean_hr", "EDA_t_slope"),
    ],
    # Heart rate not explained by movement -- the stress-vs-effort confound.
    "ecg_acc": [
        ("X_hr_per_motion", "div", "ECG_hrv_mean_hr", "ACC_t_rms"),
        ("X_sdnn_per_motion", "div", "ECG_hrv_sdnn", "ACC_t_std"),
    ],
}

FAMILIES = list(CROSS_MODAL)


def resolve_specs(families, feature_names):
    """Specs for the requested families, skipping any whose columns were ablated away.

    The prune ablations (drop Temp, drop skew/kurtosis/cvrr) run before this, so a spec
    can reference a column that no longer exists; dropping the spec is the right
    behaviour and the caller reports what survived.
    """
    index = {str(n): i for i, n in enumerate(feature_names)}
    out, missing = [], []
    for fam in families:
        for name, op, a, b in CROSS_MODAL[fam]:
            if a in index and b in index:
                out.append((name, op, index[a], index[b]))
            else:
                missing.append(name)
    return out, missing


def add_scalar_cross_modal(X, feature_names, families):
    """Append scalar cross-modal columns to X (pre-thermometer, pre-imputation).

    Non-finite results are written as NaN rather than clipped, which routes them into
    the same per-fold train-median imputation the HRV columns already use instead of
    inventing a value. Returns (X, names, applied_specs).
    """
    specs, missing = resolve_specs(families, feature_names)
    if not specs:
        return X, list(feature_names), [], missing
    cols = []
    for _, op, ia, ib in specs:
        a, b = X[:, ia], X[:, ib]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            v = a / b if op == "div" else a * b
        cols.append(np.where(np.isfinite(v), v, np.nan))
    X = np.hstack([X, np.column_stack(cols)])
    names = list(feature_names) + [s[0] for s in specs]
    return X, names, specs, missing


def popcount_pair_bits(b_tr, b_te, specs, num_bits):
    """Cross-modal features built in the binary domain, after the thermometer.

    popcount of a distributive thermometer code is the feature's quantile bin, so the
    difference of two popcounts is a rank difference -- a monotone stand-in for log(a/b)
    -- and their sum stands in for log(a*b). In hardware that is a popcount subtractor
    and num_bits comparators per pair, with no divider and no extra input register bits
    beyond the pair's own. The second thermometer is fit on the train fold only.
    """
    if not specs:
        return b_tr, b_te, None

    def pops(bits, i):
        return bits[:, i * num_bits:(i + 1) * num_bits].sum(dim=1)

    tr = torch.stack([pops(b_tr, ia) - pops(b_tr, ib) if op == "div"
                      else pops(b_tr, ia) + pops(b_tr, ib)
                      for _, op, ia, ib in specs], dim=1).float()
    te = torch.stack([pops(b_te, ia) - pops(b_te, ib) if op == "div"
                      else pops(b_te, ia) + pops(b_te, ib)
                      for _, op, ia, ib in specs], dim=1).float()
    therm = dwn.DistributiveThermometer(num_bits).fit(tr)
    extra_tr = therm.binarize(tr).flatten(start_dim=1)
    extra_te = therm.binarize(te).flatten(start_dim=1)
    return (torch.cat([b_tr, extra_tr], dim=1),
            torch.cat([b_te, extra_te], dim=1), therm)


def mi_drop_mask(X_tr, y_tr, n_drop, seed):
    """Columns to keep after dropping the n_drop lowest-MI base features.

    Used by the constant-width arm: adding cross-modal features widens the input
    register, which is 41% of the placed design, so the hardware-neutral question is
    whether they earn their bits against the base features they would displace. MI is
    computed on the training fold only.
    """
    from sklearn.feature_selection import mutual_info_classif
    keep = np.ones(X_tr.shape[1], bool)
    if n_drop <= 0:
        return keep
    finite = np.where(np.isfinite(X_tr), X_tr, np.nanmedian(X_tr, axis=0))
    finite = np.where(np.isfinite(finite), finite, 0.0)
    mi = mutual_info_classif(finite, y_tr, random_state=seed)
    keep[np.argsort(mi)[:n_drop]] = False
    return keep
