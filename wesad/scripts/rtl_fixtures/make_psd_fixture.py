"""Emit one modality's power spectrum and the 13 expected spectral feature values."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "wesad" / "scripts" / "golden_models"))
sys.path.insert(0, str(REPO / "wesad" / "scripts" / "cache_builders"))

from extract_features import MODALITY_FS, PSD_BANDS  # noqa: E402
from fft_fixed_bfp import psd_power_bfp  # noqa: E402
from fixed_frontend import ADC_PROFILES, adc_quantize  # noqa: E402
import json  # noqa: E402

from psd_features_golden_model import (PSD_KEYS, band_edges,  # noqa: E402
                                       psd_features_fixed)
from time_stats_golden_model import wrap  # noqa: E402

OUT = REPO / "wesad" / "sim_fixtures"
WINDOWS = REPO / "data" / "wesad_cache" / "wesad_windows_resampled6.npz"
SEG_LEN = {"ECG": 2048, "EMG": 2048}


def main():
    mod = sys.argv[1] if len(sys.argv) > 1 else "ECG"
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    win = np.load(WINDOWS)
    spans, bits = ADC_PROFILES["respiban"]
    codes = adc_quantize(win[mod][w : w + 1], mod, spans, bits)
    acc, pe, _ = psd_power_bfp(codes, SEG_LEN.get(mod), 2048, 8, detrend=True)
    edges = band_edges(MODALITY_FS[mod], 2048, PSD_BANDS)
    feats = psd_features_fixed(acc[0], pe[0], edges)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "psd_acc.hex").write_text("".join(f"{int(v) & 0xFFFF:04x}\n" for v in acc[0]))
    tab = json.loads((REPO / "wesad" / "scripts" / "golden_models"
                      / "psd_narrowing.json").read_text())
    si = ["ECG", "ACC", "Resp", "EDA", "EMG"].index(mod)
    words = [wrap((feats[k] >> tab["shift"][si][j]) - tab["offset"][si][j], 8)
             for j, k in enumerate(PSD_KEYS)]
    (OUT / "psd_feat.hex").write_text("".join(f"{v & 0xFF:02x}\n" for v in words))
    (OUT / "psd_pe.hex").write_text(f"{int(pe[0]) & 0x7F:02x}\n")
    (OUT / "psd_mod.hex").write_text(f"{['ECG', 'ACC', 'Resp', 'EDA', 'EMG'].index(mod):02x}\n")
    print(f"{mod} window {w}: pe={int(pe[0])} bins={acc.shape[1]}")
    for k in PSD_KEYS:
        print(f"  {k:11s} {feats[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
