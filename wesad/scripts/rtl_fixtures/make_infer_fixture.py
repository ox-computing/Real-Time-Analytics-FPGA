"""Emit one window's converter codes and the 115 feature words, 460 thermometer bits and class the full design should produce.

Everything here is computed the way the hardware computes it, on that window alone.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "wesad" / "scripts" / "golden_models"))
sys.path.insert(0, str(REPO / "wesad" / "scripts" / "cache_builders"))
sys.path.insert(0, str(REPO / "verification"))

from extract_features import MODALITY_FS, PSD_BANDS  # noqa: E402
from fft_fixed_bfp import ACC_BITS, HANN_BITS, TW_BITS, psd_power_bfp  # noqa: E402
from fixed_frontend import ADC_PROFILES, acc_magnitude_fixed, adc_quantize  # noqa: E402
from hw_model import HWModel  # noqa: E402
from psd_features_golden_model import PSD_KEYS, band_edges, psd_features_fixed  # noqa: E402
from time_stats_golden_model import (FEATURES, SENSORS, load_narrowing,  # noqa: E402
                                     time_stats_fixed, wrap)

WINDOWS = REPO / "data" / "wesad_cache" / "wesad_windows_resampled6.npz"
GOLDEN = REPO / "wesad" / "scripts" / "golden_models"
NFFT = 2048
SEG_LEN = {"ECG": 2048, "EMG": 2048}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "wesad" / "sim_fixtures")
    ap.add_argument("--adc", default="respiban", choices=sorted(ADC_PROFILES))
    ap.add_argument("--model", type=Path,
                    default=REPO / "verification" / "exported"
                    / "dwn_wesad_hw115_100_51_tau3.5.json")
    a = ap.parse_args()

    d = np.load(WINDOWS)
    w = a.window
    spans, bits = ADC_PROFILES[a.adc]
    a.out.mkdir(parents=True, exist_ok=True)

    codes = {}
    for mod in ("ECG", "Resp", "EDA", "EMG"):
        codes[mod] = adc_quantize(d[mod][w : w + 1], mod, spans, bits)[0]
    axes = [adc_quantize(d[f"ACC_{c}"][w : w + 1], "ACC_axis", spans, bits)[0]
            for c in "xyz"]
    codes["ACC"] = acc_magnitude_fixed(*axes)

    for mod in SENSORS:
        (a.out / f"{mod.lower()}.hex").write_text(
            "".join(f"{int(v) & 0xFFFF:04x}\n" for v in codes[mod]))
    for c, ax in zip("xyz", axes):
        (a.out / f"acc_{c}.hex").write_text(
            "".join(f"{int(v) & 0xFFFF:04x}\n" for v in ax))

    shift, offset = load_narrowing()
    row = []
    for s, mod in enumerate(SENSORS):
        f = time_stats_fixed(codes[mod], s, shift, offset)
        row.extend(int(f[k]) for k in FEATURES)

    ptab = json.loads((GOLDEN / "psd_narrowing.json").read_text())
    for s, mod in enumerate(SENSORS):
        edges = band_edges(MODALITY_FS[mod], NFFT, PSD_BANDS)
        acc, pe, _ = psd_power_bfp(codes[mod][None, :], SEG_LEN.get(mod), NFFT, 8,
                                   detrend=True, tw_bits=TW_BITS, hann_bits=HANN_BITS,
                                   acc_bits=ACC_BITS, count_overflow=True, cap_mode="sat")
        f = psd_features_fixed(acc[0], pe[0], edges)
        row.extend(wrap((f[k] >> ptab["shift"][s][j]) - ptab["offset"][s][j], 8)
                   for j, k in enumerate(PSD_KEYS))

    assert len(row) == 115, len(row)

    model = HWModel(a.model)
    model.thresholds = np.rint(model.thresholds)
    r = model.infer(np.array([row], dtype=float), return_all=True)
    bits_out, cls = r["bin"][0], int(r["preds"][0])

    (a.out / "feat.hex").write_text("".join(f"{v & 0xFF:02x}\n" for v in row))
    (a.out / "bits.hex").write_text("".join(f"{b}\n" for b in bits_out))
    (a.out / "class.hex").write_text(f"{cls:04x}\n")
    (a.out / "scores.hex").write_text(
        "".join(f"{int(v) & 0xFF:02x}\n" for v in r["scores"][0]))

    print(f"window {w}: subject S{d['groups'][w]}, label {d['y_multi'][w]}, "
          f"predicted {cls}, scores {r['scores'][0]}")
    lo, hi = -(1 << 7), (1 << 7) - 1
    adrift = int(np.count_nonzero((model.thresholds < lo) | (model.thresholds > hi)))
    if adrift:
        print(f"WARNING: {adrift} of {model.thresholds.size} thresholds fall outside "
              f"the 8-bit feature word", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
