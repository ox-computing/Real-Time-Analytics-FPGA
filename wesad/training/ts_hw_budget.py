"""Resource accounting for the time-series baselines against the iCE40UP5K budget.

Everything here is derived from the actual model objects, not hand-typed, so it cannot
drift from what was trained. Three separate costs are reported because they bind at
different places and conflating them is how a "it fits" claim goes wrong:

  1. WEIGHT storage      - int8 parameter bytes vs EBR + the one free SPRAM block.
  2. ACTIVATION storage  - the largest intermediate a stem stage must hold.
  3. SAMPLE storage      - the 60 s ring a windowed design needs, versus the line
                           buffers and recurrent state a causal design needs instead.

MAC counts are exact. LUT counts are NOT estimated here on purpose: a parameter budget
says nothing about the fabric a datapath, its sequencer and its requantisation logic
actually cost, and a made-up LUT number is worse than none. The one honest anchor is a
measurement on this exact part -- Rahman et al. 2026 built an INT8 1-D CNN with a 6-PE
systolic array on an iCE40UP5K and measured 2,861 LUT4 (54%), 7/8 DSP, 4/4 SPRAM. Read
that as the realistic floor for the CNN arm, and treat every "fits" below as a necessary
condition, never a sufficient one.
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "wesad" / "training"))

from train_ts_baselines import (  # noqa: E402
    MODS, FS, TRUNK_LEN, EBR_BYTES, FREE_SPRAM_BYTES, DSP_MACS_PER_INFERENCE,
    PRESETS, TRUNKS, TSNet, MultiRateStem, stride_factors, count_params, count_macs,
    fold_stats, take, load_windows,
)

RING_WORDS = sum(60 * FS[m] for m in MODS)     # the windowed design's 60 s sample ring

OPERATORS = {
    "dwn": ["LUT lookup", "popcount", "compare"],
    "cnn": ["MAC", "requantise (mul+shift)", "clamp"],
    "gru": ["MAC", "requantise", "sigmoid x2/step", "tanh x2/step", "elementwise mul x3/step"],
    "conformer": ["MAC", "requantise", "LayerNorm mean+var+rsqrt (x5/block)",
                  "softmax exp+reciprocal (LxL per head)", "SiLU", "GLU (sigmoid+mul)"],
    "ar": ["MAC (autocorrelation)", "divide xp (Levinson)", "log", "argmax over lags"],
}


def stem_activation_peak(ch):
    """Largest single per-stage activation buffer across all modality branches, int8."""
    worst, detail = 0, {}
    for m in MODS:
        cur, stages = 60 * FS[m], []
        for s in stride_factors(FS[m]):
            cur //= s
            stages.append(cur * ch)
            worst = max(worst, cur * ch)
        detail[m] = stages
    return worst, detail


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sig, y, _ = load_windows(str(REPO / "data/wesad_cache/wesad_windows_resampled6.npz"), device)
    stats = fold_stats(sig, torch.arange(len(y), device=device))
    sample = take(sig, torch.arange(2, device=device), stats)

    print(f"iCE40UP5K budget: 5,280 LUT4 | 8 SB_MAC16 | EBR {EBR_BYTES:,} B | "
          f"free SPRAM {FREE_SPRAM_BYTES:,} B")
    print(f"MAC budget at 12 MHz x 1 s hop, 8 MACs: {DSP_MACS_PER_INFERENCE:,}")
    print(f"windowed sample ring: {RING_WORDS:,} words (3 of 4 SPRAM blocks)\n")

    hdr = (f"{'model':<22}{'params':>9}{'int8 KB':>9}{'store':>11}{'MAC/inf':>12}"
           f"{'DSP%':>7}{'act pk B':>10}{'sample words':>14}")
    print(hdr)
    print("-" * len(hdr))

    for causal in (False, True):
        for fam, presets in PRESETS.items():
            for name, stem_kw, trunk_kw in presets:
                net = TSNet(stem_kw["stem_ch"], TRUNKS[fam], trunk_kw, causal=causal).to(device)
                params = count_params(net)
                macs = count_macs(net, sample)
                kb = params / 1024.0
                store = ("EBR" if params <= EBR_BYTES else
                         "EBR+SPRAM" if params <= EBR_BYTES + FREE_SPRAM_BYTES else "OVER")
                act, _ = stem_activation_peak(stem_kw["stem_ch"])
                if causal:
                    words = net.stem.line_buffer_words()
                    if hasattr(net.trunk, "state_words"):
                        words += net.trunk.state_words()
                else:
                    words = RING_WORDS
                label = name + ("_causal" if causal else "")
                print(f"{label:<22}{params:>9,}{kb:>9.1f}{store:>11}{macs:>12,}"
                      f"{100*macs/DSP_MACS_PER_INFERENCE:>7.1f}{act:>10,}{words:>14,}")
                del net
        print()

    print("Stem activation detail (int8 bytes per stage, 8 channels):")
    _, detail = stem_activation_peak(8)
    for m, stages in detail.items():
        print(f"  {m:<5} " + " -> ".join(f"{v:,}" for v in stages))
    print(f"  EBR is {EBR_BYTES:,} B: the first ECG and EMG stages exceed it, so those")
    print("  branches must be fused and streamed rather than materialised per layer.\n")

    print("Operators each family needs in fabric beyond the multiplier:")
    for k, v in OPERATORS.items():
        print(f"  {k:<10} {', '.join(v)}")
    print("\nMeasured anchor on this part (Rahman et al. 2026, INT8 1-D CNN + 6-PE array):")
    print("  2,861 LUT4 (54%), 7/8 DSP, 4/4 SPRAM, 95.5 ms/inference.")
    print("Measured DWN 100-51 classifier core: 152 SB_LUT4, 151 DFF, 0 DSP, 0 EBR, 0 SPRAM.")


if __name__ == "__main__":
    main()
