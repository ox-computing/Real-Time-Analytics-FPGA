#!/usr/bin/env python3
"""Plot a replay_wesad.py CSV: per-hop DWN scores and the class the FPGA returned.

    ~/miniforge3/envs/dwn/bin/python3 wesad/scripts/analysis/plot_replay.py \
        replay_s2.csv --out replay_s2.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NAMES = ["baseline", "stress", "amusement"]
COLOURS = ["#2c7fb8", "#d95f0e", "#31a354"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, default=Path("replay.png"))
    ap.add_argument("--title", default="WESAD replay on ECP5, one inference per second")
    a = ap.parse_args()

    rows = [{k: int(v) for k, v in r.items()}
            for r in csv.DictReader(a.csv.open())]
    t = np.array([r["t"] for r in rows])
    truth = np.array([r["truth"] for r in rows])
    pred = np.array([r["class"] for r in rows])
    scores = np.array([[r["s0"], r["s1"], r["s2"]] for r in rows])

    wrong = pred != truth
    band = (t[wrong].min(), t[wrong].max() + 1) if wrong.any() else None

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.15], "hspace": 0.12})

    for i in range(3):
        ax0.plot(t, scores[:, i], lw=1.6, color=COLOURS[i], label=NAMES[i])
    if band:
        for ax in (ax0, ax1):
            ax.axvspan(*band, color="0.85", zorder=0)
        ax0.text((band[0] + band[1]) / 2, ax0.get_ylim()[1] * 0.97,
                 f"window refilling\n{band[1] - band[0]} s", ha="center",
                 va="top", fontsize=9, color="0.35")

    ax0.set_ylabel("DWN group score")
    ax0.legend(loc="upper left", frameon=False, ncol=3)
    ax0.set_title(a.title)
    ax0.grid(alpha=0.25)

    ax1.step(t, truth, where="post", lw=2.4, color="0.6", label="protocol label")
    ax1.step(t, pred, where="post", lw=1.4, color="k", label="FPGA class")
    ax1.set_yticks(range(3))
    ax1.set_yticklabels(NAMES, fontsize=8)
    ax1.set_ylim(-0.4, 2.4)
    ax1.set_xlabel("replay time (s)")
    ax1.legend(loc="upper left", frameon=False, ncol=2, fontsize=8)
    ax1.grid(alpha=0.25)

    agree = int((~wrong).sum())
    steady = int((~wrong[(t < band[0]) | (t >= band[1])]).sum()) if band else agree
    total_steady = int(((t < band[0]) | (t >= band[1])).sum()) if band else len(t)
    fig.text(0.995, 0.015,
             f"{agree}/{len(t)} hops match  |  {steady}/{total_steady} "
             f"outside the transition", ha="right", fontsize=8.5, color="0.35")

    fig.savefig(a.out, dpi=160, bbox_inches="tight")
    print(f"wrote {a.out}  ({agree}/{len(t)} match, "
          f"{steady}/{total_steady} outside the transition)")


if __name__ == "__main__":
    main()
