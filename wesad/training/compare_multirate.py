"""Paired LOSO comparison of multi-rate feature caches against the 60 s control.

The window-length arms differ by which cache they read, not by an arm flag, so they
cannot sit in one exp_multimodal run. build_multirate_cache keeps the row set identical
across caches (same window_starts, same labels, same groups), so the folds and the seeds
line up and the comparison is paired on both -- this script trains the same config on
each cache over the same seeds and reports the paired-t against the control cache.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_dwn import load_cache, select_columns  # noqa: E402
from exp_multimodal import arm, paired_vs, run_arm  # noqa: E402


def main():
    root = Path(__file__).resolve().parent.parent.parent
    cache_dir = root / "data" / "wesad_cache"
    ap = argparse.ArgumentParser(description="Multi-rate window-length comparison")
    ap.add_argument("--control", default=str(cache_dir / "wesad_features_mr_ctrl60s.npz"),
                    help="the 60 s cache built through build_multirate_cache")
    ap.add_argument("--caches", nargs="+", required=True, help="multi-rate caches to test")
    ap.add_argument("--task", default="multi", choices=["multi", "binary"])
    ap.add_argument("--drop", nargs="*", default=["Temp"])
    ap.add_argument("--drop-feat", nargs="*", dest="drop_feat",
                    default=["skew", "kurtosis", "cvrr"])
    ap.add_argument("--config", default="100-51")
    ap.add_argument("--mapping", default="free", help="layer-1 mapping variant for every arm")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--num-bits", type=int, default=4)
    ap.add_argument("--tau", type=float, default=3.5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-root", default=str(root / "results"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: torch-dwn's EFD kernel has no CPU implementation.")

    args.layer_sizes = [int(s) for s in args.config.split("-") if s]
    k = 3 if args.task == "multi" else 2
    seeds = list(range(args.seed, args.seed + args.seeds))
    run_dir = Path(args.results_root) / f"wesad-mm-multirate_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    results, rows = [], None
    for path in [args.control] + list(args.caches):
        name = Path(path).stem.replace("wesad_features_mr_", "")
        X, y, groups, feature_names, _raw = load_cache(path, args.task)
        mask = select_columns(feature_names, None, args.drop, args.drop_feat)
        X, names = X[:, mask], [str(n) for n in feature_names[mask]]
        # Row alignment is the whole basis of the pairing; refuse rather than
        # silently compare caches built over different window sets.
        if rows is None:
            rows, ref_y, ref_g = X.shape[0], y.copy(), np.asarray(groups).copy()
        elif X.shape[0] != rows or not np.array_equal(y, ref_y) \
                or not np.array_equal(np.asarray(groups), ref_g):
            raise SystemExit(f"{name}: row set differs from the control cache; "
                             "the caches are not paired and cannot be compared")

        print(f"\n=== cache {name}  X {X.shape}  ({len(seeds)} seeds) ===")
        rec = run_arm(X, names, y, groups, arm(name, mapping=args.mapping), args, k, seeds)
        rec["cache"] = str(path)
        with open(run_dir / f"{name}.json", "w") as fh:
            json.dump(rec, fh, indent=2)
        results.append(rec)

    control = results[0]
    for r in results:
        r["vs_control"] = paired_vs(control, r) if r is not control else None
    with open(run_dir / "summary.json", "w") as fh:
        json.dump({"args": vars(args), "seeds": seeds, "control": control["name"],
                   "arms": [{kk: r[kk] for kk in ("name", "cache", "input_bits",
                                                  "balanced_mean", "balanced_std",
                                                  "acc_mean", "macro_f1_mean",
                                                  "vs_control")} for r in results]},
                  fh, indent=2)

    print(f"\n=== multi-rate: balanced accuracy, paired vs '{control['name']}' ===")
    for r in sorted(results, key=lambda r: r["balanced_mean"], reverse=True):
        v = r["vs_control"]
        if r is control:
            cell = "control"
        elif v is None:
            cell = "n/a (needs >=2 seeds)"
        else:
            cell = (f"{v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] p={v['p']:.4f} "
                    f"{v['wins']}/{v['n']}")
        print(f"  {r['name']:<28} {r['balanced_mean']:.4f}+/-{r['balanced_std']:.4f}  {cell}")
    print(f"\nResults written to: {run_dir}")


if __name__ == "__main__":
    main()
