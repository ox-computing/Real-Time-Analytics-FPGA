"""Emit the twiddle and Hann ROM contents for psd_features from the golden model.

Both tables are read straight out of fft_fixed_bfp so the RTL cannot drift from
the reference: the twiddle words are exactly what fft_bfp multiplies by, and the
Hann words exactly what psd_power_bfp windows with.

  twiddle.hex  1024 words, {w_imaginary, w_real}, two's complement Q1.7
  hann.hex     2734 words, Q0.7, the three half-windows concatenated
               2048 -> [0, 1024)   1920 -> [1024, 1984)   1500 -> [1984, 2734)
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "wesad" / "scripts" / "golden_models"))

from fft_fixed_bfp import _hann, _twiddles, HANN_BITS, TW_BITS  # noqa: E402

RTL = REPO / "wesad" / "rtl"
NFFT = 2048
HANN_LENGTHS = [2048, 1920, 1500]


def main():
    cos_t, sin_t = _twiddles(NFFT, TW_BITS)
    w_real, w_imaginary = cos_t, -sin_t
    assert w_real.min() >= -128 and w_real.max() <= 127
    assert w_imaginary.min() >= -128 and w_imaginary.max() <= 127

    with open(RTL / "twiddle.hex", "w") as f:
        for k in range(NFFT // 2):
            word = ((int(w_imaginary[k]) & 0xFF) << 8) | (int(w_real[k]) & 0xFF)
            f.write(f"{word:04x}\n")
    print(f"twiddle.hex  {NFFT // 2} words  Q1.{TW_BITS}  "
          f"re [{w_real.min()}, {w_real.max()}]  im [{w_imaginary.min()}, {w_imaginary.max()}]")

    words = 0
    with open(RTL / "hann.hex", "w") as f:
        for length in HANN_LENGTHS:
            h = _hann(length, HANN_BITS)
            assert np.array_equal(h, h[::-1]), f"hann {length} not symmetric"
            half = h[:length // 2]
            assert half.min() >= 0 and half.max() <= 127
            for v in half:
                f.write(f"{int(v):02x}\n")
            words += half.size
            print(f"hann {length:5d} -> {half.size:4d} stored  peak {int(h.max())}")
    print(f"hann.hex     {words} words  Q0.{HANN_BITS}")


if __name__ == "__main__":
    main()
