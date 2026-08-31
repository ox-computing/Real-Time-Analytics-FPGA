import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch_dwn as dwn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "wesad" / "training"))

from train_dwn import select_columns  # noqa: E402

CACHE = REPO / "data" / "wesad_cache" / "wesad_features_dt2048k13.npz"


def codes(V, lo, step, live):
    return np.where(live, np.rint((V - lo) / step), V)


def margin_report(X, thr):
    d = np.abs(X[:, :, None] - thr[None, :, :])          
    d = np.where(d > 0, d, np.inf).min(axis=2)     
    span = X.max(axis=0) - X.min(axis=0)
    span = np.where(span > 0, span, 1.0)
    return d / span


def analytic_width(min_margin):
    out = np.zeros_like(min_margin, dtype=int)
    for i, m in enumerate(min_margin):
        out[i] = 99 if (not np.isfinite(m) or m <= 0) else \
            int(np.ceil(np.log2(1.0 / m + 1.0)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--num-bits", type=int, default=4, help="thermometer bits per feature (z)")
    ap.add_argument("--drop", nargs="*", default=["Temp", "EMG"])
    ap.add_argument("--drop-feat", nargs="*", default=["skew", "kurtosis", "cvrr"],
                    dest="drop_feat")
    ap.add_argument("--widths", nargs="*", type=int,
                    default=list(range(2, 25)), help="candidate widths to scan")
    ap.add_argument("--log-cols", nargs="*", default=None, dest="log_cols",
                    help="name patterns quantised in log2 (the power-like features)")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="flip-rate tolerance for the relaxed width (needs LOSO confirmation)")
    ap.add_argument("--out", type=Path, default=None, help="write per-feature results as json")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    X = d["X"].astype(np.float64)
    names = np.array([str(s) for s in d["feature_names"]])
    mask = select_columns(names, None, args.drop, args.drop_feat)
    X, names = X[:, mask], names[mask]
    print(f"cache {args.cache.name}   X {X.shape}   z={args.num_bits}")


    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isfinite(X), X, med)

    t = torch.from_numpy(X).float()
    therm = dwn.DistributiveThermometer(args.num_bits).fit(t)

    thr = therm.thresholds.numpy()   
    X = X.astype(np.float32)
    ref = (X[:, :, None] > thr[None, :, :]).reshape(len(X), -1)
    assert np.array_equal(ref, therm.binarize(t).flatten(start_dim=1).numpy().astype(bool)), \
        "reference does not reproduce DistributiveThermometer.binarize"

    ties = (X[:, :, None] == thr[None, :, :]).reshape(len(X), -1)
    live_bits = ~ties
    n_bits = int(live_bits.sum())
    print(f"exact ties on a threshold: {int(ties.sum())} bits "
          f"({ties.sum() / ties.size:.2e}) -- excluded, not resolvable by width")
    print(f"thermometer: {thr.shape[0]} features x {thr.shape[1]} thresholds"
          f"  -> {ref.shape[1]}-bit input, {n_bits} bits over the dataset\n")

    marg = margin_report(X, thr)
    mn = marg.min(axis=0)
    aw = analytic_width(mn)
    print("margin to nearest threshold (fraction of feature range):")
    print(f"  min over all features/windows : {mn.min():.2e}")
    print(f"  median feature's min-margin   : {np.median(mn):.2e}")
    print(f"  median over all values        : {np.median(marg):.3f}")
    tight = np.argsort(mn)[:8]
    print("  tightest features (these set the width):")
    for i in tight:
        print(f"    {names[i]:28s} min-margin {mn[i]:.2e}  -> analytic W >= {aw[i]}")
    print()

    # Power-like features are stored as log2 in hardware, so their code step is
    # proportional, not absolute. Quantise those columns in that domain; the
    # reference bits are unchanged because log2 is monotone.
    log_col = np.zeros(len(names), dtype=bool)
    if args.log_cols:
        pat = re.compile("|".join(args.log_cols))
        log_col = np.array([bool(pat.search(n)) for n in names])
        floor = np.where(X > 0, X, np.inf).min(axis=0)
        floor = np.where(np.isfinite(floor), floor, np.float32(1.0))
        Q = np.where(log_col, np.log2(np.maximum(X, floor)), X)
        thrq = np.where(log_col[:, None], np.log2(np.maximum(thr, floor[:, None])), thr)
        print(f"log-domain columns: {int(log_col.sum())} of {len(names)}\n")
    else:
        Q, thrq = X, thr

    lo, hi = Q.min(axis=0), Q.max(axis=0)
    span = hi - lo
    live = span > 0
    zero_w = np.full(len(names), -1, dtype=int)                  # narrowest zero-flip width
    tol_w = np.full(len(names), -1, dtype=int)
    rows = []
    print(f"{'W':>3}  {'flips':>8}  {'flip rate':>10}  {'feats exact':>12}")
    for w in sorted(args.widths):
        step = np.where(live, span / (2 ** w - 1), 1.0)
        xc = codes(Q, lo, step, live)                            # (N, F)
        tc = codes(thrq, lo[:, None], step[:, None], live[:, None])
        bits = (xc[:, :, None] > tc[None, :, :]).reshape(len(X), -1)
        diff = (bits != ref) & live_bits
        per_feat = diff.reshape(len(X), len(names), args.num_bits).any(axis=2).mean(axis=0)
        exact = per_feat == 0
        zero_w = np.where((zero_w < 0) & exact, w, zero_w)
        tol_w = np.where((tol_w < 0) & (per_feat <= args.tol), w, tol_w)
        rate = diff.sum() / n_bits
        rows.append({"width": w, "flips": int(diff.sum()), "flip_rate": float(rate),
                     "features_exact": int(exact.sum())})
        print(f"{w:3d}  {diff.sum():8d}  {rate:10.3e}  {exact.sum():7d}/{len(names)}")

    ok = zero_w > 0
    print(f"\nzero-flip width: max {zero_w[ok].max() if ok.any() else '-'} bits "
          f"(the binding feature), median {int(np.median(zero_w[ok])) if ok.any() else '-'}")
    print(f"at tol={args.tol:g}: max {tol_w[tol_w > 0].max() if (tol_w > 0).any() else '-'} bits")
    if ok.any():
        worst = np.argsort(-zero_w)[:8]
        print("\nwidest features (narrow these last):")
        for i in worst:
            z = int(zero_w[i]) if zero_w[i] > 0 else None
            tl = int(tol_w[i]) if tol_w[i] > 0 else None
            print(f"  {names[i]:28s} zero-flip W={z}  tol W={tl}  analytic {aw[i]}")
    const = [names[i] for i in range(len(names)) if hi[i] == lo[i]]
    if const:
        print(f"\nconstant features ({len(const)}, free at any width): {const}")

    if args.out:
        args.out.write_text(json.dumps({
            "cache": str(args.cache), "num_bits": args.num_bits,
            "sweep": rows,
            "per_feature": [
                {"name": str(names[i]), "zero_flip_width": int(zero_w[i]),
                 "tol_width": int(tol_w[i]), "analytic_width": int(aw[i]),
                 "min_margin": float(mn[i])} for i in range(len(names))],
        }, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
