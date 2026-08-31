"""Emit one 2048-sample segment and the power spectrum the golden FFT produces from it."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "wesad" / "scripts" / "golden_models"))
sys.path.insert(0, str(REPO / "wesad" / "scripts" / "cache_builders"))

from fft_fixed_bfp import psd_power_bfp  # noqa: E402
from fixed_frontend import ADC_PROFILES, adc_quantize  # noqa: E402

OUT = REPO / "wesad" / "sim_fixtures"
WINDOWS = REPO / "data" / "wesad_cache" / "wesad_windows_resampled6.npz"


def main():
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mod = sys.argv[2] if len(sys.argv) > 2 else "ECG"
    off = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    win = np.load(WINDOWS)
    spans, bits = ADC_PROFILES["respiban"]
    seg = adc_quantize(win[mod][w : w + 1, off : off + 2048], mod, spans, bits)
    acc, pe, _ = psd_power_bfp(seg, None, 2048, 8, detrend=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fft_in.hex").write_text("".join(f"{int(v) & 0xFFFF:04x}\n" for v in seg[0]))
    (OUT / "fft_power.hex").write_text("".join(f"{int(v) & 0xFFFFFF:06x}\n" for v in acc[0]))
    (OUT / "fft_pe.hex").write_text(f"{int(pe[0]) & 0xFF:02x}\n")
    print(f"window {w} {mod}+{off}: pe={int(pe[0])} peak={int(acc[0].max())} bins={acc.shape[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
