import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "wesad" / "scripts" / "golden_models"))
from fixed_frontend import ADC_PROFILES, adc_quantize  # noqa: E402

CACHE = REPO / "data" / "wesad_cache" / "wesad_windows_resampled6.npz"
MODS = {"ECG": 15000, "ACC": 1920, "Resp": 1500, "EDA": 1500, "EMG": 21000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--adc", default="respiban", choices=sorted(ADC_PROFILES))
    a = ap.parse_args()

    d = np.load(CACHE)
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"window {a.window}: subject S{d['groups'][a.window]}, class {d['y_multi'][a.window]}")

    for mod, n in MODS.items():
        x = d[mod][a.window]
        assert x.shape == (n,), f"{mod}: {x.shape} != {(n,)}"
        spans, bits = ADC_PROFILES[a.adc]
        q = adc_quantize(x, mod, spans, bits)
        assert q.min() >= -32768 and q.max() <= 32767, f"{mod} out of int16 range"
        if mod == "ACC":
            assert q.min() >= 0, "ACC magnitude must be non-negative"
        path = a.out / f"{mod.lower()}.hex"
        path.write_text("".join(f"{v & 0xFFFF:04x}\n" for v in q))
        print(f"  {path.name}: {n} words, min {q.min()}, max {q.max()}")


if __name__ == "__main__":
    main()
