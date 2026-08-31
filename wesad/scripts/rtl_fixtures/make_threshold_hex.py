"""Convert a DWN export's thermometer thresholds into the signed-16-bit hex file that feature_threshold_storage loads with $readmemh."""

import argparse
import json
import sys
from pathlib import Path

INT16_MIN = -32768
INT16_MAX = 32767


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("thresholds.hex"))
    ap.add_argument("--width", type=int, default=8,
                    help="threshold word width feature_threshold_storage holds")
    args = ap.parse_args()

    meta = json.loads(args.model.read_text())
    thresholds = meta["thermometer_thresholds"]
    n_feature = len(thresholds)
    z = len(thresholds[0])

    overflow = []
    collapsed = []
    words = []
    for f, row in enumerate(thresholds):
        rounded = [int(round(t)) for t in row]
        for k, t in enumerate(rounded):
            if t < INT16_MIN or t > INT16_MAX:
                overflow.append((f, k, row[k]))
        if len(set(rounded)) != len(rounded):
            collapsed.append((f, row, rounded))
        words.extend(rounded)

    if overflow:
        print(f"ERROR: {len(overflow)} threshold(s) do not fit signed 16 bits:", file=sys.stderr)
        for f, k, t in overflow[:10]:
            print(f"  feature {f} bit {k}: {t}", file=sys.stderr)
        print("The model must be trained on time_feats_fixed integer features.", file=sys.stderr)
        return 1

    for f, row, rounded in collapsed:
        print(f"WARNING: feature {f} thresholds tie after rounding: {row} -> {rounded}")

    digits = args.width // 4
    mask = (1 << args.width) - 1
    args.out.write_text("".join(f"{w & mask:0{digits}x}\n" for w in words))
    print(f"{n_feature} features x {z} bits = {len(words)} thresholds -> {args.out}")
    print(f"range {min(words)} .. {max(words)}, input vector {len(words)} bits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
