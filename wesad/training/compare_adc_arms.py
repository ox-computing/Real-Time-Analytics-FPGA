"""Pair each ADC full-scale arm seed-for-seed against a reference arm
(--ref, default perwindow) and report delta-balanced with 95% CI, p and wins."""

import argparse
import json
import math
from pathlib import Path

ARMS = ["perwindow", "respiban", "part12", "emgtight", "part12_acc3",
        "part12_b10", "part12_b8", "part12_b7", "part12_b6", "part12_b5", "part12_b4"]


def load_arm(runroot, arm):
    """{seed: balanced} for one arm, or None if it did not run."""
    dirfile = runroot / f"{arm}.dir"
    if not dirfile.exists():
        return None
    run = Path(dirfile.read_text().strip())
    if not run.is_absolute():
        run = runroot.parents[1] / run
    js = run / "100_51.json"
    if not js.exists():
        return None
    d = json.loads(js.read_text())
    return {s["seed"]: s["balanced"] for s in d["per_seed"]}


def paired(a, b):
    """(mean diff, lo, hi, p, wins, n) for a - b over their common seeds."""
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    n = len(d)
    if n < 2:
        return None
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0:
        return mean, mean, mean, 1.0, sum(x > 0 for x in d), n
    t = mean / se
    # Two-sided p from the t distribution via its incomplete-beta form; scipy is
    # not imported here so the comparison runs without the training environment.
    df = n - 1
    x = df / (df + t * t)
    p = _betainc_half(df / 2.0, 0.5, x)
    crit = _t_crit_95(df)
    return mean, mean - crit * se, mean + crit * se, p, sum(x > 0 for x in d), n


def _betainc_half(a, b, x):
    """Regularised incomplete beta I_x(a, b) by continued fraction (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


# Two-sided 95% t critical values; the sweep runs a fixed 15 seeds, so a small
# table beats pulling in scipy for one number.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 24: 2.064, 29: 2.045}


def _t_crit_95(df):
    if df in _T95:
        return _T95[df]
    keys = sorted(_T95)
    return _T95[min(keys, key=lambda k: abs(k - df))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runroot", type=Path)
    ap.add_argument("--ref", default="perwindow",
                    help="arm to pair against (e.g. part12_b8 to isolate width steps)")
    ap.add_argument("--arms", nargs="+", default=ARMS,
                    help="arms to report; defaults to the ADC full-scale set")
    args = ap.parse_args()

    ref = load_arm(args.runroot, args.ref)
    if ref is None:
        raise SystemExit(f"reference arm {args.ref} not found under {args.runroot}")
    rm = sum(ref.values()) / len(ref)
    print(f"reference {args.ref}: balanced {rm:.4f} over {len(ref)} seeds\n")
    print(f"{'arm':14} {'bal':>7} {'delta':>8} {'95% CI':>18} {'p':>7} {'wins':>7}")
    for arm in args.arms:
        if arm == args.ref:
            continue
        a = load_arm(args.runroot, arm)
        if a is None:
            print(f"{arm:14} {'--':>7}   (not run)")
            continue
        am = sum(a.values()) / len(a)
        st = paired(a, ref)
        if st is None:
            print(f"{arm:14} {am:7.4f}   (too few common seeds)")
            continue
        mean, lo, hi, p, wins, n = st
        flag = "" if lo <= 0 <= hi else "  <-- CI excludes 0"
        print(f"{arm:14} {am:7.4f} {mean:+8.4f} [{lo:+.4f},{hi:+.4f}] {p:7.3f} "
              f"{wins:3d}/{n:<3d}{flag}")


if __name__ == "__main__":
    main()
