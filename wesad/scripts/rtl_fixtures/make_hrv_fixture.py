"""Emit one ECG window's converter codes and the eight expected HRV feature words."""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "wesad" / "scripts" / "golden_models"))
sys.path.insert(0, str(REPO / "wesad" / "scripts" / "cache_builders"))

from fixed_frontend import (ADC_PROFILES, HRV_KEYS, adc_quantize,  # noqa: E402
                            biquad_cascade_fixed, hrv_feats_fixed,
                            hrv_peaks_stream)
from time_stats_golden_model import wrap  # noqa: E402

OUT = REPO / "wesad" / "sim_fixtures"
WINDOWS = REPO / "data" / "wesad_cache" / "wesad_windows_resampled6.npz"


def main():
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    win = np.load(WINDOWS)
    spans, bits = ADC_PROFILES["respiban"]
    codes = adc_quantize(win["ECG"][w : w + 1], "ECG", spans, bits)[0]
    filtered = biquad_cascade_fixed(codes)
    peaks = hrv_peaks_stream(filtered, 250)
    feats = hrv_feats_fixed(peaks, 250)

    tab = json.loads((REPO / "wesad" / "scripts" / "golden_models"
                      / "hrv_narrowing.json").read_text())
    words = [wrap((feats[k] >> tab["shift"][j]) - tab["offset"][j], 8)
             for j, k in enumerate(HRV_KEYS)]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hrv_in.hex").write_text("".join(f"{int(v) & 0xFFFF:04x}\n" for v in codes))
    (OUT / "hrv_feat.hex").write_text("".join(f"{v & 0xFF:02x}\n" for v in words))
    print(f"window {w}: {len(peaks)} peaks")
    for j, k in enumerate(HRV_KEYS):
        print(f"  {k:10s} raw {feats[k]:8d}  word {words[j]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
