"""Multimodal fusion experiments for the WESAD DWN: constrained layer-1 mappings,
explicit cross-modal features, and the grouped per-modality architecture.

Every arm runs the same LOSO protocol as train_dwn.py -- the per-fold imputation,
per-fold distributive thermometer, class-weighted CE and fixed epoch budget are
imported from it rather than restated, so the control arm reproduces the locked
100-51 number exactly and any movement is attributable to the arm alone. All arms
see the same seed list and each re-seeds before its first fold, so the seeds are
paired and the summary reports a paired-t against the control.

  python wesad/training/exp_multimodal.py --exp mapping  --seeds 5
  python wesad/training/exp_multimodal.py --exp features --seeds 5
  python wesad/training/exp_multimodal.py --exp grouped  --seeds 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_dwn import (  # noqa: E402
    binarize_fold, class_weights, impute_train_median, load_cache, predict,
    select_columns, set_seed, train_one,
)
from multimodal_arms import (  # noqa: E402
    FAMILIES, add_scalar_cross_modal, build_model_variant, mi_drop_mask,
    popcount_pair_bits, resolve_specs,
)


# --- Arm definitions --------------------------------------------------------

def arm(name, mapping="free", families=(), popcount=False, constant_width=False,
        layer_sizes=None):
    return {"name": name, "mapping": mapping, "families": list(families),
            "popcount": popcount, "constant_width": constant_width,
            "layer_sizes": layer_sizes}


def experiment_arms(exp, layer_sizes):
    """The arm table for one experiment. Arm 0 is always the control."""
    if exp == "mapping":
        # Free is the locked baseline. randfix removes the learnable mapping so the
        # constrained arms can be read against what learnability alone is worth,
        # rather than against a moving target.
        return [arm("free"), arm("randfix", mapping="randfix"),
                arm("within", mapping="within"),
                arm("cross4", mapping="cross4"),
                arm("cross2", mapping="cross2")]

    if exp == "features":
        # Families alone, then the pairs that target different confounds (motion vs
        # respiration), then everything. 'all_cw' holds the input register width fixed
        # by dropping the same number of lowest-MI base features, which is the only
        # version of the question that is hardware-neutral. 'all_pc' builds the same
        # relations in the binary domain instead, where they cost a popcount subtractor
        # rather than a divider.
        arms = [arm("none")]
        arms += [arm(f, families=[f]) for f in FAMILIES]
        arms += [arm("motion", families=["eda_acc", "emg_acc", "ecg_acc"]),
                 arm("resp_symp", families=["rsa", "ecg_eda"]),
                 arm("all", families=FAMILIES),
                 arm("all_cw", families=FAMILIES, constant_width=True),
                 arm("all_pc", families=FAMILIES, popcount=True)]
        return arms

    if exp == "grouped":
        # The grouped sub-DWN is the 'within' mapping: 100 layer-1 LUTs dealt 20 per
        # modality feeding one fusion layer. Sweeping the fusion width and the block
        # width against the free-mapping model of identical size separates "grouping
        # helps" from "this shape helps".
        out = []
        for sizes in ([100, 51], [100, 102], [150, 51], [150, 102]):
            tag = "_".join(map(str, sizes))
            out.append(arm(f"free_{tag}", layer_sizes=sizes))
            out.append(arm(f"within_{tag}", mapping="within", layer_sizes=sizes))
        return out

    if exp == "combined":
        # Populated from the mapping/features results; kept explicit so a combined run
        # is never an accident of defaults.
        return [arm("none"),
                arm("within", mapping="within"),
                arm("all", families=FAMILIES),
                arm("within_all", mapping="within", families=FAMILIES),
                arm("within_all_pc", mapping="within", families=FAMILIES, popcount=True)]

    raise SystemExit(f"unknown --exp {exp!r}")


# --- LOSO for one arm -------------------------------------------------------

def run_loso_arm(X_base, names_base, y, groups, a, args, k, seed):
    """Pooled LOSO predictions for one arm -> (acc, balanced, macro_f1, widths).

    Scalar cross-modal features are appended outside the fold loop on purpose: they are
    per-sample ratios and products with no fitted parameter, so they carry nothing from
    the held-out subject. Everything that IS fitted -- imputation, both thermometers,
    the MI drop -- stays inside the fold on the training subjects only.
    """
    set_seed(seed)
    X, names, specs, _ = (add_scalar_cross_modal(X_base, names_base, a["families"])
                          if a["families"] and not a["popcount"]
                          else (X_base, list(names_base), [], []))
    # The popcount arm needs the pair's column indices in the BASE matrix, since it
    # combines their thermometer codes rather than their scalar values.
    pc_specs = resolve_specs(a["families"], names_base)[0] if a["popcount"] else []
    n_drop = len(specs) if a["constant_width"] else 0

    layer_sizes = a["layer_sizes"] or args.layer_sizes
    y_true, y_pred, widths = [], [], []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        Xtr, Xte, fold_names = X[tr], X[te], names
        if n_drop:
            imp_tr, _ = impute_train_median(X[tr], X[te])
            keep = mi_drop_mask(imp_tr, y[tr], n_drop, seed)
            Xtr, Xte = Xtr[:, keep], Xte[:, keep]
            fold_names = [n for n, kp in zip(names, keep) if kp]

        b_tr, b_te, _ = binarize_fold(Xtr, Xte, args.num_bits)
        if pc_specs:
            b_tr, b_te, _ = popcount_pair_bits(b_tr, b_te, pc_specs, args.num_bits)
            # The popcount columns are their own modality block for the mapping arms.
            fold_names = list(fold_names) + [s[0] for s in pc_specs]
        widths.append(int(b_tr.size(1)))

        model = build_model_variant(b_tr.size(1), layer_sizes, args.n, args.tau, k,
                                    a["mapping"], fold_names, args.num_bits).cuda()
        train_one(model, b_tr, torch.from_numpy(y[tr]).long(),
                  args.epochs, args.batch_size, args.lr, class_weights(y[tr], k))
        y_true.append(y[te])
        y_pred.append(predict(model, b_te, args.batch_size))

    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    return (accuracy_score(y_true, y_pred),
            balanced_accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro"),
            int(np.max(widths)))


def run_arm(X, names, y, groups, a, args, k, seeds):
    """One arm across every seed; returns a record with per-seed rows and mean+/-sigma."""
    per_seed = []
    for s in seeds:
        t0 = time.time()
        acc, bal, mf1, width = run_loso_arm(X, names, y, groups, a, args, k, s)
        per_seed.append({"seed": s, "acc": acc, "balanced": bal, "macro_f1": mf1,
                         "input_bits": width, "elapsed_sec": round(time.time() - t0, 1)})
        print(f"    seed {s:>3d}  acc {acc:.4f}  bal {bal:.4f}  mF1 {mf1:.4f}  "
              f"bits {width}  ({per_seed[-1]['elapsed_sec']:.0f}s)")

    def stat(key):
        v = np.array([r[key] for r in per_seed])
        return float(v.mean()), float(v.std())

    rec = {"name": a["name"], "arm": a, "per_seed": per_seed,
           "input_bits": per_seed[0]["input_bits"],
           "layer_sizes": a["layer_sizes"] or args.layer_sizes}
    for key, tag in (("acc", "acc"), ("balanced", "balanced"), ("macro_f1", "macro_f1")):
        m, sd = stat(key)
        rec[f"{tag}_mean"], rec[f"{tag}_std"] = m, sd
    print(f"  {a['name']:<16} acc {rec['acc_mean']:.4f}+/-{rec['acc_std']:.4f}  "
          f"bal {rec['balanced_mean']:.4f}+/-{rec['balanced_std']:.4f}  "
          f"mF1 {rec['macro_f1_mean']:.4f}+/-{rec['macro_f1_std']:.4f}")
    return rec


def paired_vs(control, other, key="balanced"):
    """Mean delta (other - control), 95% CI and paired-t p over shared seeds.

    Same convention as compare_ts_baselines.paired so the two sets of numbers can sit
    in the same table; only the sign is oriented the other way, since here the arm is
    the thing on trial.
    """
    cb = {r["seed"]: r for r in control["per_seed"]}
    ob = {r["seed"]: r for r in other["per_seed"]}
    common = sorted(set(cb) & set(ob))
    if len(common) < 2:
        return None
    diff = np.array([ob[s][key] - cb[s][key] for s in common])
    n = len(common)
    se = diff.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    _, p = stats.ttest_rel([ob[s][key] for s in common], [cb[s][key] for s in common])
    return {"n": n, "delta": float(diff.mean()),
            "lo": float(diff.mean() - tcrit * se), "hi": float(diff.mean() + tcrit * se),
            "p": float(p), "wins": int((diff > 0).sum())}


# --- Main -------------------------------------------------------------------

def main():
    root = Path(__file__).resolve().parent.parent.parent
    ap = argparse.ArgumentParser(description="WESAD DWN multimodal fusion experiments")
    ap.add_argument("--exp", required=True,
                    choices=["mapping", "features", "grouped", "combined"])
    ap.add_argument("--cache", default=str(root / "data" / "wesad_cache"
                                           / "wesad_features_dt2048k13_6mod.npz"))
    ap.add_argument("--task", choices=["multi", "binary"], default="multi")
    ap.add_argument("--keep", nargs="*", default=None)
    ap.add_argument("--drop", nargs="*", default=["Temp"])
    ap.add_argument("--drop-feat", nargs="*", dest="drop_feat",
                    default=["skew", "kurtosis", "cvrr"])
    ap.add_argument("--config", default="100-51", help="layer stack, widths joined by '-'")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--num-bits", type=int, default=4)
    ap.add_argument("--tau", type=float, default=3.5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None, help="run only these arm names")
    ap.add_argument("--results-root", default=str(root / "results"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: torch-dwn's EFD kernel has no CPU implementation.")

    args.layer_sizes = [int(s) for s in args.config.split("-") if s]
    k = 3 if args.task == "multi" else 2
    if args.layer_sizes[-1] % k:
        raise SystemExit(f"final layer {args.layer_sizes[-1]} must divide into k={k}")

    X, y, groups, feature_names, _raw = load_cache(args.cache, args.task)
    mask = select_columns(feature_names, args.keep, args.drop, args.drop_feat)
    X, feature_names = X[:, mask], [str(n) for n in feature_names[mask]]
    seeds = list(range(args.seed, args.seed + args.seeds))

    arms = experiment_arms(args.exp, args.layer_sizes)
    if args.only:
        arms = [a for a in arms if a["name"] in args.only]
        if not arms:
            raise SystemExit(f"--only {args.only} matched no arm in --exp {args.exp}")

    run_dir = Path(args.results_root) / f"wesad-mm-{args.exp}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"cache: {args.cache}")
    print(f"X {X.shape}  subjects {np.unique(groups).size}  task {args.task} (k={k})  "
          f"classes {np.bincount(y).tolist()}")
    print(f"base input width at z={args.num_bits}: {X.shape[1] * args.num_bits} bits")
    _, missing = resolve_specs(FAMILIES, feature_names)
    if missing:
        print(f"cross-modal specs unavailable after ablation: {missing}")

    results = []
    for a in arms:
        print(f"\n=== arm {a['name']}  (mapping={a['mapping']}, families={a['families']}, "
              f"popcount={a['popcount']}, cw={a['constant_width']}, "
              f"layers={a['layer_sizes'] or args.layer_sizes}, {len(seeds)} seeds) ===")
        rec = run_arm(X, feature_names, y, groups, a, args, k, seeds)
        with open(run_dir / f"{a['name']}.json", "w") as f:
            json.dump(rec, f, indent=2)
        results.append(rec)

    control = results[0]
    for r in results:
        r["vs_control"] = paired_vs(control, r) if r is not control else None
    summary = {"exp": args.exp, "args": {kk: vv for kk, vv in vars(args).items()},
               "seeds": seeds, "control": control["name"],
               "arms": [{kk: r[kk] for kk in ("name", "layer_sizes", "input_bits",
                                              "acc_mean", "acc_std", "balanced_mean",
                                              "balanced_std", "macro_f1_mean",
                                              "macro_f1_std", "vs_control")}
                        for r in results]}
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {args.exp}: balanced accuracy, paired vs control '{control['name']}' ===")
    print(f"  {'arm':<16} {'bits':>5} {'bal':>16}  {'delta':>8} {'95% CI':>18} "
          f"{'p':>8} {'wins':>6}")
    for r in sorted(results, key=lambda r: r["balanced_mean"], reverse=True):
        v = r["vs_control"]
        if r is control:
            cell = "control"
        elif v is None:
            cell = "n/a (needs >=2 seeds)"
        else:
            cell = (f"{v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] {v['p']:8.4f} "
                    f"{v['wins']:>3d}/{v['n']:<3d}")
        print(f"  {r['name']:<16} {r['input_bits']:>5} "
              f"{r['balanced_mean']:.4f}+/-{r['balanced_std']:.4f}  {cell}")
    print(f"\nResults written to: {run_dir}")


if __name__ == "__main__":
    main()
