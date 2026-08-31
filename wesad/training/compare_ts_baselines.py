"""Pair every time-series baseline against the DWN 100-51 reference, seed for seed.

Both sides ran the same LOSO folds on the same windows, so a paired test is the right
comparison: it cancels the fold-to-fold difficulty that dominates this dataset's variance
and asks only whether a given architecture beat the DWN on the SAME seed.

A preset may appear across several run directories (screening did seeds 0-1, the
confirmation run 2-4). Records are merged by preset name and the means recomputed from the
merged seed set rather than trusting any single file's summary.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
# Renamed from wesad-dwn-loso_20260803-110511 by commit 2b18012 (results-tree partition);
# glob on the timestamp so a future rename does not silently break this.
DWN_GLOB = str(REPO / "results" / "*_20260803-110511" / "100_51.json")


def load_dwn(path=None):
    if path is None:
        hits = glob.glob(DWN_GLOB)
        if not hits:
            raise SystemExit(f"DWN reference not found: {DWN_GLOB}")
        path = hits[0]
    d = json.load(open(path))
    return {r["seed"]: r for r in d["per_seed"]}, d


def paired(base, dwn, key):
    """Mean delta, 95% CI and paired-t p over the seeds the two runs share."""
    common = sorted(set(base) & set(dwn))
    if len(common) < 2:
        return None
    b = np.array([base[s][key] for s in common])
    d = np.array([dwn[s][key] for s in common])
    diff = b - d
    n = len(common)
    se = diff.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    _, p = stats.ttest_rel(b, d)
    return dict(n=n, delta=diff.mean(), lo=diff.mean() - tcrit * se,
                hi=diff.mean() + tcrit * se, p=p, wins=int((diff > 0).sum()))


def condition_of(run_dir: str) -> str:
    """Group runs by the training CONDITION they were produced under.

    A preset name alone is not a key: cnn_c8_h16 exists at the 30 s hop without
    augmentation and again on the dense hop with it. Merging those seed sets would
    average two different experiments into one number, so the run tag joins the key and
    only genuinely identical conditions pool (screening seeds 0-1 with confirmation 2-4).
    """
    tag = Path(run_dir).name.split("_")[0].replace("wesad-tsbase-", "")
    return {"screen": "base", "confirm": "base", "tab5": "base", "mlpwide": "base",
            "bestshot": "dense", "tabdense": "dense", "stream": "stream"}.get(tag, tag)


def collect(patterns):
    merged = {}
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            cond = condition_of(str(Path(f).parent))
            s = json.load(open(f))
            for r in s["results"]:
                key = (r["name"], cond)
                slot = merged.setdefault(key, {"meta": {**r, "condition": cond}, "per": {}})
                for x in r["per_seed"]:
                    slot["per"].setdefault(x["seed"], x)
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("patterns", nargs="*",
                    default=[str(REPO / "results/wesad-tsbase*/summary.json")])
    ap.add_argument("--dwn", default=None)
    args = ap.parse_args()

    dwn_seeds, dwn = load_dwn(args.dwn)
    merged = collect(args.patterns)

    rows = []
    for (name, cond), slot in merged.items():
        r, per = dict(slot["meta"]), slot["per"]
        r["name"] = f"{name}[{cond}]"
        order = sorted(per)
        for key in ("acc", "balanced", "macro_f1"):
            r[f"{key}_mean"] = float(np.mean([per[s][key] for s in order]))
        if all("balanced_int8" in per[s] for s in order):
            r["balanced_int8_mean"] = float(np.mean([per[s]["balanced_int8"] for s in order]))
        r["n_seeds"] = len(order)
        rows.append((r, per))

    print(f"DWN 100-51 reference ({len(dwn_seeds)} seeds): acc {dwn['acc_mean']:.4f}  "
          f"bal {dwn['balanced_mean']:.4f}  mF1 {dwn['macro_f1_mean']:.4f}   152 SB_LUT4, 0 DSP\n")
    hdr = (f"{'model':<30}{'in':<9}{'n':>3}{'params':>9}{'MAC/inf':>12}{'buf':>8}"
           f"{'acc':>8}{'bal':>8}{'int8':>8}{'d_bal':>9}{'95% CI':>20}{'p':>7}{'w':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r, per in sorted(rows, key=lambda x: -x[0]["balanced_mean"]):
        pb = paired(per, dwn_seeds, "balanced")
        ci = f"[{pb['lo']:+.4f},{pb['hi']:+.4f}]" if pb else ""
        dbal = f"{pb['delta']:+.4f}" if pb else ""
        pv = f"{pb['p']:.3f}" if pb else ""
        w = f"{pb['wins']}/{pb['n']}" if pb else ""
        q = f"{r['balanced_int8_mean']:.4f}" if "balanced_int8_mean" in r else "-"
        buf = f"{r['buffer_words']:,}" if r.get("buffer_words") else "-"
        print(f"{r['name']:<30}{r.get('input','?'):<9}{r['n_seeds']:>3}{r['params']:>9,}"
              f"{r['macs']:>12,}{buf:>8}{r['acc_mean']:>8.4f}{r['balanced_mean']:>8.4f}"
              f"{q:>8}{dbal:>9}{ci:>20}{pv:>7}{w:>5}")


if __name__ == "__main__":
    main()
