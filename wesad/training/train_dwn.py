"""Train DWN models on the extracted WESAD features under LOSO, the first DWN trained on
real biosignal features and evaluated the way the device is deployed, against a wearer it
never saw. The thermometer and NaN imputation are re-fit per fold on the training
subjects only, cross-entropy is class-weighted, and every config runs over several seeds
reported as mean +/- sigma because LOSO noise is ~1 point. --fit-all retrains one config
on all 15 subjects and saves the checkpoint the fixed-point reference and the RTL
exporter consume.
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.functional import cross_entropy
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

import torch_dwn as dwn


# --- Reproducibility --------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix Python / NumPy / Torch (CPU+CUDA) RNGs.

    Same caveat as the MNIST script: torch-dwn's EFD backward is a custom CUDA
    kernel with non-deterministic atomics, so two runs at one seed can still
    differ slightly -- which is exactly why this script averages over seeds
    rather than trusting any single run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- Data -------------------------------------------------------------------

def load_cache(path: str, task: str):
    """Return (X float64, y int, groups int, feature_names) for the chosen task.

    X keeps its NaNs here (HRV features are NaN on flat windows); imputation is
    deliberately deferred to per-fold so the held-out subject never contributes
    a median. task selects y_multi (3-class) or y_binary (stress vs rest).
    """
    d = np.load(path, allow_pickle=True)
    y = d["y_multi"] if task == "multi" else d["y_binary"]
    raw = d["raw_time"] if "raw_time" in d.files else None
    return d["X"].astype(float), y.astype(int), d["groups"], d["feature_names"], raw


def select_columns(feature_names, keep, drop, drop_feat=None):
    """Boolean column mask over feature_names, matching baseline.py's semantics.

    keep/drop act on the modality prefix (text before the first '_'); drop_feat
    drops any feature whose name contains one of the given substrings. Mirrored
    verbatim from baseline.py so a DWN prune ablation and the tabular baseline
    select exactly the same columns -- the Phase-3 prune gate compares the two
    directly, and a mismatch in what "drop Temp" means would silently break it.
    """
    mask = np.ones(len(feature_names), bool)
    prefixes = np.array([n.split("_")[0] for n in feature_names])
    if keep:
        mask &= np.isin(prefixes, keep)
    if drop:
        mask &= ~np.isin(prefixes, drop)
    if drop_feat:
        for sub in drop_feat:
            mask &= np.array([sub not in n for n in feature_names])
    return mask


def impute_train_median(X_tr, X_te, mode="median"):
    """Fill NaN with the stand-in the hardware would emit for a degenerate window.

    "median" fills with per-column medians from the TRAIN rows only -- lifted
    verbatim from baseline.py so the DWN sees exactly the imputation the tabular
    baseline was measured under. A column that is NaN for every training row
    falls back to 0.

    "zero" fills with 0 instead, which is what the datapath produces for free:
    the accumulators are already zero, so there is no constant to store and no
    forced-bit logic at the comparators. The two are not equivalent. For
    total/peak_power/band* a flat window's 0 is the *true* value rather than a
    stand-in, so those columns want 0 either way; for centroid/bandwidth/entropy/
    rolloff85/peak_freq a 0 is a sentinel that collides with the bottom of the
    learned quantile ladder, making a degenerate window indistinguishable from a
    genuinely low one on that feature. Which costs less is measured, not assumed.
    """
    if mode == "zero":
        return (np.where(np.isfinite(X_tr), X_tr, 0.0),
                np.where(np.isfinite(X_te), X_te, 0.0))
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return (np.where(np.isfinite(X_tr), X_tr, med),
            np.where(np.isfinite(X_te), X_te, med))


def quantise_feature_word(X_tr, X_te, width, names):
    """Round every column onto the W-bit grid the feature store would hold it on.

    Power-like columns (total, peak_power, band*) are quantised in log2, which is
    how the hardware holds an (acc, exponent) pair; everything else is linear.
    The grid is derived from the TRAINING rows only -- it is a per-feature
    constant baked into the bitstream, so fitting it on the held-out subject
    leaks exactly the way fitting a scaler on test data does. Values are written
    back dequantised so the thermometer's thresholds land on the same grid the
    comparator sees.
    """
    is_log = np.array([bool(re.search("_psd_total|_psd_peak_power|_psd_band", str(n)))
                       for n in names])
    out_tr, out_te = X_tr.copy(), X_te.copy()
    for j in range(X_tr.shape[1]):
        ok_tr = np.isfinite(X_tr[:, j])
        if not ok_tr.any():
            continue
        v = X_tr[ok_tr, j]
        if is_log[j]:
            pos = v[v > 0]
            if pos.size == 0:
                continue
            floor = pos.min()
            v = np.log2(np.maximum(v, floor))
        lo, hi = v.min(), v.max()
        if hi == lo:
            continue
        step = (hi - lo) / (2 ** width - 1)
        for src, dst in ((X_tr, out_tr), (X_te, out_te)):
            ok = np.isfinite(src[:, j])
            w = src[ok, j]
            if is_log[j]:
                w = np.log2(np.maximum(w, floor))
            # clip to the fitted grid: the hardware register cannot hold a code
            # outside the range it was scaled for.
            q = lo + np.rint((np.clip(w, lo, hi) - lo) / step) * step
            dst[ok, j] = np.exp2(q) if is_log[j] else q
    return out_tr, out_te


def narrow_time_fold(X_tr, X_te, raw_tr, raw_te, names):
    """Refit the time narrowing tables on train rows, rewrite the time columns.

    Same argument as quantise_feature_word: (shift, offset) are constants in the
    bitstream, so they are fit where the bitstream's calibration data would come
    from -- the training subjects -- and never from the held-out one. The fit and
    the narrowing both come from the golden model, so the columns this produces
    are bit-identical to what the RTL computes under the same tables.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "golden_models"))
    from time_stats_golden_model import narrow_raw_columns

    out_tr, out_te = X_tr.copy(), X_te.copy()
    cols = [i for i, n in enumerate(names) if "_t_" in str(n)]
    narrowed_tr, narrowed_te = narrow_raw_columns(raw_tr, raw_te)
    out_tr[:, cols] = narrowed_tr
    out_te[:, cols] = narrowed_te
    return out_tr, out_te


def binarize_fold(X_tr, X_te, num_bits: int, impute="median"):
    """Impute (train-median), fit a per-feature thermometer on train, binarize both.

    DistributiveThermometer(feature_wise=True) learns a separate quantile ladder
    per feature -- the distributive (not uniform) form the thermometer ablation
    showed is worth ~4 points on WESAD. Fit on the imputed training rows only,
    then applied to both splits. Returns flat float tensors (N, F*num_bits) ready
    for the LUT layers, plus the fitted thermometer (needed by --fit-all).
    """
    X_tr, X_te = impute_train_median(X_tr, X_te, impute)
    t_tr = torch.from_numpy(X_tr).float()
    t_te = torch.from_numpy(X_te).float()
    thermometer = dwn.DistributiveThermometer(num_bits).fit(t_tr)
    b_tr = thermometer.binarize(t_tr).flatten(start_dim=1)
    b_te = thermometer.binarize(t_te).flatten(start_dim=1)
    return b_tr, b_te, thermometer


# --- Model ------------------------------------------------------------------

def build_model(input_size: int, layer_sizes: list, n: int, tau: float, k: int) -> nn.Module:
    """LUT-layer stack into a GroupSum -- identical construction to train_mnist.

    Learnable mapping on the first layer only (it reads the thermometer bits,
    where which inputs a LUT sees is worth learning); random fixed mapping after.
    Kept byte-for-byte compatible with the MNIST build so verification/export
    read the same checkpoint layout; only k (class count) is parameterised.
    """
    layers, in_size = [], input_size
    for i, size in enumerate(layer_sizes):
        layers.append(dwn.LUTLayer(in_size, size, n=n,
                                   mapping="learnable" if i == 0 else "random"))
        in_size = size
    layers.append(dwn.GroupSum(k=k, tau=tau))
    return nn.Sequential(*layers)


def parse_configs(spec: str, k: int) -> list:
    """'100-51,200-102' -> [('100_51', [100, 51]), ...]; underscores name the dir.

    Rejects a final layer that does not divide by k up front -- GroupSum needs
    final % k == 0, and catching it here stops a sweep from dying on config 3.
    """
    configs = []
    for item in spec.split(","):
        sizes = [int(s) for s in item.strip().split("-") if s]
        if not sizes:
            raise SystemExit(f"empty config in --configs: {spec!r}")
        if sizes[-1] % k:
            raise SystemExit(f"config {item!r}: final layer {sizes[-1]} must divide "
                             f"into k={k} GroupSum classes")
        configs.append(("_".join(str(s) for s in sizes), sizes))
    return configs


# --- Train / evaluate -------------------------------------------------------

def class_weights(y_tr: np.ndarray, k: int) -> torch.Tensor:
    """Inverse-frequency CE weights from the fold's own training labels.

    weight_c = N / (k * count_c): a class half as common gets twice the loss.
    An absent class (possible on a small held-out fold's complement, in theory)
    gets weight 0 rather than an infinity.
    """
    counts = np.bincount(y_tr, minlength=k).astype(float)
    with np.errstate(divide="ignore"):
        w = y_tr.size / (k * counts)
    w[~np.isfinite(w)] = 0.0
    return torch.tensor(w, dtype=torch.float32)


def train_one(model, x_tr, y_tr, epochs, batch_size, lr, weight):
    """Train a single model in place for a fixed epoch budget (no test peeking).

    Same optimiser and staged 0.1x decay schedule as the MNIST script. y_tr is a
    CPU long tensor; batches are moved to CUDA per step (the dataset is tiny).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    milestones = [int(epochs * 0.3), int(epochs * 0.6), int(epochs * 0.9)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    weight = weight.cuda()
    n_samples = x_tr.size(0)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            bx, by = x_tr[idx].cuda(), y_tr[idx].cuda()
            optimizer.zero_grad()
            loss = cross_entropy(model(bx), by, weight=weight)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


@torch.no_grad()
def predict(model, x, batch_size):
    """Predicted class ids for x, batched onto CUDA."""
    model.eval()
    out = []
    for i in range(0, x.size(0), batch_size):
        out.append(model(x[i:i + batch_size].cuda()).argmax(dim=1).cpu())
    return torch.cat(out).numpy()


def run_loso(X, y, groups, layer_sizes, args, k, seed):
    """One seed: pooled LOSO predictions -> (acc, balanced, macro-F1).

    Re-seeds first so every config sees the same RNG stream at a given seed and
    is judged on equal footing. For each held-out subject the thermometer and
    imputation are re-fit on the training subjects only, a fresh model is trained
    for the fixed budget, and its predictions on the held-out subject are
    collected. Metrics are computed once over the concatenation of all 15 folds.
    """
    set_seed(seed)
    y_true, y_pred = [], []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        X_tr, X_te = X[tr], X[te]
        if args.raw_time is not None:
            X_tr, X_te = narrow_time_fold(X_tr, X_te, args.raw_time[tr],
                                          args.raw_time[te], args.feature_names)
        if args.feature_word:
            X_tr, X_te = quantise_feature_word(X_tr, X_te, args.feature_word,
                                               args.feature_names)
        b_tr, b_te, _ = binarize_fold(X_tr, X_te, args.num_bits, args.impute)
        model = build_model(b_tr.size(1), layer_sizes, args.n, args.tau, k).cuda()
        train_one(model, b_tr, torch.from_numpy(y[tr]).long(),
                  args.epochs, args.batch_size, args.lr, class_weights(y[tr], k))
        y_true.append(y[te])
        y_pred.append(predict(model, b_te, args.batch_size))
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    return (accuracy_score(y_true, y_pred),
            balanced_accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro"))


def run_config(X, y, groups, name, layer_sizes, args, k, seeds):
    """Run one config across seeds; return a record with per-seed and mean+/-sigma."""
    per_seed = []
    for s in seeds:
        t0 = time.time()
        acc, bal, mf1 = run_loso(X, y, groups, layer_sizes, args, k, s)
        per_seed.append({"seed": s, "acc": acc, "balanced": bal, "macro_f1": mf1,
                         "elapsed_sec": round(time.time() - t0, 1)})
        print(f"    seed {s:>3d}  acc {acc:.4f}  bal {bal:.4f}  mF1 {mf1:.4f}  "
              f"({per_seed[-1]['elapsed_sec']:.0f}s)")

    def stat(key):
        v = np.array([r[key] for r in per_seed])
        return float(v.mean()), float(v.std())

    acc_m, acc_s = stat("acc")
    bal_m, bal_s = stat("balanced")
    f1_m, f1_s = stat("macro_f1")
    print(f"  {name:<12} acc {acc_m:.4f}+/-{acc_s:.4f}  "
          f"bal {bal_m:.4f}+/-{bal_s:.4f}  mF1 {f1_m:.4f}+/-{f1_s:.4f}")
    return {
        "name": name, "layer_sizes": layer_sizes, "num_layers": len(layer_sizes),
        "total_luts": sum(layer_sizes), "k": k, "n": args.n, "tau": args.tau,
        "num_bits": args.num_bits, "epochs": args.epochs,
        "acc_mean": acc_m, "acc_std": acc_s,
        "balanced_mean": bal_m, "balanced_std": bal_s,
        "macro_f1_mean": f1_m, "macro_f1_std": f1_s,
        "per_seed": per_seed,
    }


def fit_all(X, y, groups, feature_names, layer_sizes, args, k, out_dir):
    """Retrain one config on ALL 15 subjects and save a shippable checkpoint.

    The locked decision: LOSO gives the honest estimate, the deployed weights are
    a retrain on everyone. Thermometer + imputation are fit on the full set here
    (correct -- there is no held-out subject to protect), and the checkpoint +
    thermometer + meta are written in the same layout the MNIST exporter reads.
    """
    set_seed(args.seed)
    # Full-set impute+thermometer: pass X as both train and test to reuse the path;
    # the test copy is discarded.
    b_all, _, thermometer = binarize_fold(X, X, args.num_bits, args.impute)
    model = build_model(b_all.size(1), layer_sizes, args.n, args.tau, k).cuda()
    train_one(model, b_all, torch.from_numpy(y).long(),
              args.epochs, args.batch_size, args.lr, class_weights(y, k))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "checkpoint.pt")
    torch.save(thermometer, out_dir / "thermometer.pt")
    meta = {"layer_sizes": layer_sizes, "num_layers": len(layer_sizes),
            "input_size": int(b_all.size(1)), "k": k, "n": args.n, "tau": args.tau,
            "num_bits": args.num_bits, "num_features": int(X.shape[1]),
            "feature_names": [str(n) for n in feature_names],
            "keep": args.keep, "drop": args.drop, "drop_feat": args.drop_feat,
            "task": args.task, "epochs": args.epochs, "seed": args.seed}
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  fit-all checkpoint -> {out_dir}")


# --- Plotting ---------------------------------------------------------------

def plot_comparison(results, out_path, task):
    """Balanced-accuracy mean with +/-sigma error bars vs total LUT-node count."""
    order = sorted(range(len(results)), key=lambda i: results[i]["total_luts"])
    names = [results[i]["name"] for i in order]
    luts = [results[i]["total_luts"] for i in order]
    means = [results[i]["balanced_mean"] for i in order]
    stds = [results[i]["balanced_std"] for i in order]

    plt.figure(figsize=(7.5, 4.5))
    plt.errorbar(luts, means, yerr=stds, fmt="o-", capsize=4)
    for name, x, ym in zip(names, luts, means):
        plt.annotate(name, (x, ym), textcoords="offset points", xytext=(6, 6), fontsize=9)
    plt.xlabel("total DWN LUT nodes (all layers)")
    plt.ylabel("LOSO balanced accuracy (mean +/- sigma over seeds)")
    plt.title(f"WESAD DWN architecture sweep ({task}, k={results[0]['k']})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# --- Main -------------------------------------------------------------------

def main():
    root = Path(__file__).resolve().parent.parent.parent
    ap = argparse.ArgumentParser(description="WESAD DWN LOSO training / architecture sweep")
    # Default is the decimated cache: each modality band-limited to its sensor's
    # real ODR Nyquist (Phase 1 showed this costs ~0 accuracy, 3-class ET even
    # +2.6). That is the signal the FPGA actually sees, so it is what the DWN
    # trains on; --cache overrides for the plain/causal caches.
    ap.add_argument("--cache",
                    default=str(root / "data" / "wesad_cache" / "wesad_features_decimated.npz"))
    ap.add_argument("--task", choices=["multi", "binary"], default="multi",
                    help="multi = 3-class (k=3); binary = stress vs rest (k=2)")
    # Phase-3 prune ablations. Same semantics as baseline.py: keep/drop on
    # modality prefix, drop-feat on name substring. Default None = no-op, so
    # existing invocations (arch_sweep.sh, the Phase-2 configs) are unchanged.
    ap.add_argument("--keep", nargs="*", default=None, help="modality prefixes to keep")
    ap.add_argument("--drop", nargs="*", default=None, help="modality prefixes to drop")
    ap.add_argument("--drop-feat", nargs="*", default=None, dest="drop_feat",
                    help="drop any feature whose name contains one of these substrings")
    ap.add_argument("--configs", default="100-51,50-51,100-102,200-102,400-201,200-100-51",
                    help="comma-separated layer stacks, widths joined by '-'")
    ap.add_argument("--n", type=int, default=4, help="inputs per LUT (LUT-n; 4 = iCE40 fabric)")
    ap.add_argument("--num-bits", type=int, default=3, help="thermometer bits per feature (z)")
    ap.add_argument("--impute", default="median", choices=["median", "zero"],
                    help="stand-in for a degenerate window's NaN features")
    ap.add_argument("--feature-word", type=int, default=None, dest="feature_word",
                    help="round every feature onto a W-bit grid before the thermometer, "
                         "as the feature store holds it (fit per fold on train rows)")
    ap.add_argument("--tau", type=float, default=1.3, help="GroupSum softmax temperature")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of seeds; runs base-seed .. base-seed+seeds-1")
    ap.add_argument("--seed", type=int, default=0, help="base seed (also the --fit-all seed)")
    ap.add_argument("--fit-all", action="store_true",
                    help="skip LOSO; retrain each config on all 15 subjects and save a checkpoint")
    ap.add_argument("--results-root", default=str(root / "results"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: torch-dwn's EFD kernel has no CPU implementation.")

    k = 3 if args.task == "multi" else 2
    configs = parse_configs(args.configs, k)
    X, y, groups, feature_names, raw_time = load_cache(args.cache, args.task)
    mask = select_columns(feature_names, args.keep, args.drop, args.drop_feat)
    if not mask.all():
        if raw_time is not None and int(mask[["_t_" in str(n) for n in feature_names]].sum()) \
                != int(sum("_t_" in str(n) for n in feature_names)):
            print("  dropping time columns: per-fold narrowing disabled")
            raw_time = None
        X = X[:, mask]
        feature_names = feature_names[mask]
        print(f"feature ablation: {int(mask.sum())}/{mask.size} features kept"
              + (f"  keep={args.keep}" if args.keep else "")
              + (f"  drop={args.drop}" if args.drop else "")
              + (f"  drop_feat={args.drop_feat}" if args.drop_feat else ""))
    args.feature_names = feature_names
    args.raw_time = raw_time
    if raw_time is not None:
        print(f"per-fold narrowing: refitting {raw_time.shape[1]} time tables on train rows")
    seeds = list(range(args.seed, args.seed + args.seeds))

    run_id = time.strftime("%Y%m%d-%H%M%S")
    tag = "fitall" if args.fit_all else "loso"
    run_dir = Path(args.results_root) / f"wesad-dwn-{tag}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"cache: {args.cache}")
    print(f"X {X.shape}  subjects {np.unique(groups).size}  task {args.task} (k={k})  "
          f"classes {np.bincount(y).tolist()}")
    print(f"input width after z={args.num_bits} thermometer: {X.shape[1] * args.num_bits} bits")

    if args.fit_all:
        for name, layer_sizes in configs:
            print(f"\n=== fit-all {name}  (LUTs {layer_sizes}, tau {args.tau}) ===")
            fit_all(X, y, groups, feature_names, layer_sizes, args, k, run_dir / name)
        print(f"\nCheckpoints written to: {run_dir}")
        return

    results = []
    for name, layer_sizes in configs:
        print(f"\n=== config {name}  (LUTs {layer_sizes}; n={args.n}, tau={args.tau}, "
              f"z={args.num_bits}, {args.seeds} seeds) ===")
        rec = run_config(X, y, groups, name, layer_sizes, args, k, seeds)
        with open(run_dir / f"{name}.json", "w") as f:
            json.dump(rec, f, indent=2)
        results.append(rec)

    plot_comparison(results, run_dir / "comparison.png", args.task)
    # feature_names/raw_time ride on args so the fold loop can reach them; they
    # are arrays, so they cannot go into the JSON record of the run.
    arg_record = {k: v for k, v in vars(args).items()
                  if k not in ("feature_names", "raw_time")}
    summary = {"run_id": run_id, "args": arg_record, "seeds": seeds,
               "configs": [{ky: r[ky] for ky in (
                   "name", "layer_sizes", "total_luts", "num_layers", "tau",
                   "acc_mean", "acc_std", "balanced_mean", "balanced_std",
                   "macro_f1_mean", "macro_f1_std")} for r in results]}
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== summary (ranked by balanced-accuracy mean) ===")
    for r in sorted(results, key=lambda r: r["balanced_mean"], reverse=True):
        print(f"  {r['name']:<12} LUTs={r['total_luts']:<5} "
              f"acc {r['acc_mean']:.4f}+/-{r['acc_std']:.4f}  "
              f"bal {r['balanced_mean']:.4f}+/-{r['balanced_std']:.4f}  "
              f"mF1 {r['macro_f1_mean']:.4f}+/-{r['macro_f1_std']:.4f}")
    print(f"\nResults written to: {run_dir}")


if __name__ == "__main__":
    main()
