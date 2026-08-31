import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

SEED = 42
N_TREES = 300


def make_models():
    """The two ensembles, fixed for reproducibility (n_estimators=300)."""
    return {
        "RF": RandomForestClassifier(n_estimators=N_TREES, class_weight="balanced",
                                     random_state=SEED, n_jobs=-1),
        "ET": ExtraTreesClassifier(n_estimators=N_TREES, class_weight="balanced",
                                   random_state=SEED, n_jobs=-1),
    }


def select_columns(feature_names, keep, drop, drop_feat=None):
    """Boolean column mask. keep/drop act on modality prefix; drop_feat drops any
    feature whose name contains one of the given substrings (spectral ablation)."""
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


def impute_train_median(X_tr, X_te):
    """Fill NaN with per-column medians taken from the TRAIN rows only."""
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)          # column NaN everywhere -> 0
    return np.where(np.isfinite(X_tr), X_tr, med), np.where(np.isfinite(X_te), X_te, med)


def _scores(y_true, y_pred):
    return (accuracy_score(y_true, y_pred),
            balanced_accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro"))


def eval_split(X, y, folds, make_model):
    """Aggregate acc/balanced/macro-F1 over a fold iterator, imputing per fold."""
    yt, yp = [], []
    for tr, te in folds:
        X_tr, X_te = impute_train_median(X[tr], X[te])
        m = make_model()
        m.fit(X_tr, y[tr])
        yt.append(y[te])
        yp.append(m.predict(X_te))
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return _scores(yt, yp)


def importances(X, y, feature_names):
    """RF+ET averaged impurity importance over the full set, sorted desc."""
    Xi = np.where(np.isfinite(X), X, np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0))
    Xi = np.where(np.isfinite(Xi), Xi, 0.0)
    imp = np.zeros(X.shape[1])
    for m in make_models().values():
        m.fit(Xi, y)
        imp += m.feature_importances_
    imp /= 2.0
    order = np.argsort(imp)[::-1]
    return [(feature_names[i], float(imp[i])) for i in order]


def main():
    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="WESAD tabular baseline (RF/ET)")
    ap.add_argument("--cache", default=str(root / "data" / "wesad_cache" / "wesad_features.npz"))
    ap.add_argument("--task", choices=["binary", "multi", "both"], default="both")
    ap.add_argument("--split", choices=["loso", "pooled", "both"], default="both")
    ap.add_argument("--keep", nargs="*", default=None, help="modality prefixes to keep")
    ap.add_argument("--drop", nargs="*", default=None, help="modality prefixes to drop")
    ap.add_argument("--drop-feat", nargs="*", default=None, dest="drop_feat",
                    help="drop any feature whose name contains one of these substrings")
    ap.add_argument("--importances", type=int, default=0,
                    help="print the top-N importance ranking (0 = skip)")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    X, groups, feature_names = d["X"], d["groups"], d["feature_names"]
    mask = select_columns(feature_names, args.keep, args.drop, args.drop_feat)
    X = X[:, mask]
    fnames = feature_names[mask]
    print(f"cache: {args.cache}")
    print(f"X {X.shape}  subjects {np.unique(groups).size}  "
          f"features {mask.sum()}/{mask.size}"
          + (f"  keep={args.keep}" if args.keep else "")
          + (f"  drop={args.drop}" if args.drop else ""))

    tasks = ["binary", "multi"] if args.task == "both" else [args.task]
    splits = ["loso", "pooled"] if args.split == "both" else [args.split]

    for task in tasks:
        y = d["y_binary"] if task == "binary" else d["y_multi"]
        counts = np.bincount(y)
        floor = counts.max() / counts.sum()
        print(f"\n=== {task}  (classes {counts.tolist()}, majority floor {floor:.3f}) ===")
        print(f"{'model':<5} {'split':<7} {'acc':>7} {'bal':>7} {'macroF1':>8}")
        for split in splits:
            if split == "loso":
                folds = list(LeaveOneGroupOut().split(X, y, groups))
            else:
                folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
            for name, make in [("RF", lambda: make_models()["RF"]),
                               ("ET", lambda: make_models()["ET"])]:
                acc, bal, mf1 = eval_split(X, y, folds, make)
                print(f"{name:<5} {split:<7} {acc:>7.3f} {bal:>7.3f} {mf1:>8.3f}")

    if args.importances:
        y = d["y_binary"] if "binary" in tasks else d[f"y_{tasks[0]}"]
        ranking = importances(X, y, fnames)
        print(f"\n=== top {args.importances} importances "
              f"({'binary' if 'binary' in tasks else tasks[0]}, RF+ET avg) ===")
        for nm, sc in ranking[:args.importances]:
            print(f"  {sc:.4f}  {nm}")


if __name__ == "__main__":
    main()
