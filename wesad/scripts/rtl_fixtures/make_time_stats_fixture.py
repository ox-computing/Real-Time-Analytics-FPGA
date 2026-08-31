"""Emit one window's per-sensor sample stream and the 50 expected time feature words."""

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "wesad" / "scripts" / "golden_models"))
sys.path.insert(0, str(REPO / "wesad" / "scripts" / "cache_builders"))

from fixed_frontend import ADC_PROFILES, acc_magnitude_fixed, adc_quantize  # noqa: E402
from time_stats_golden_model import (FEATURES, SENSORS, load_narrowing,  # noqa: E402
                                     time_stats_fixed)

OUT = REPO / "wesad" / "sim_fixtures"
WINDOWS = REPO / "data" / "wesad_cache" / "wesad_windows_resampled6.npz"


def main():
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    win = np.load(WINDOWS)
    spans, bits = ADC_PROFILES["respiban"]
    codes = {m: adc_quantize(win[m], m, spans, bits) for m in SENSORS}
    qx, qy, qz = (adc_quantize(win[f"ACC_{a}"], "ACC_axis", spans, bits) for a in "xyz")
    codes["ACC"] = np.vstack([acc_magnitude_fixed(qx[i], qy[i], qz[i])
                              for i in range(qx.shape[0])])

    shift, offset = load_narrowing()
    OUT.mkdir(parents=True, exist_ok=True)
    expected = []
    for s, mod in enumerate(SENSORS):
        row = codes[mod][w]
        (OUT / f"ts_{mod.lower()}.hex").write_text(
            "".join(f"{int(v) & 0xFFFF:04x}\n" for v in row))
        f = time_stats_fixed(row, s, shift, offset)
        expected.extend(f[k] for k in FEATURES)
    (OUT / "ts_feat.hex").write_text("".join(f"{v & 0xFF:02x}\n" for v in expected))
    print(f"window {w}: {len(expected)} feature words -> {OUT}/ts_feat.hex")
    for s, mod in enumerate(SENSORS):
        print(f"  {mod:5} " + " ".join(f"{expected[s * 10 + i]:5d}" for i in range(10)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
