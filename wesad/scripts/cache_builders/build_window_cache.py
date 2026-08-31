"""Cache the raw ODR windows every fixed-point arm reads.

Decimation happens once per recording and windows are cut out of the decimated
stream, so a sample decimates identically no matter which window it lands in --
see extract_features.resample_recording. --hop-sec picks the window cadence: the
frontend runs at 1 s, the training caches at a coarser hop because adjacent 1 s
windows share 59 of their 60 seconds.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "wesad" / "scripts" / "cache_builders"))

from extract_features import (  # noqa: E402
    FS, SHIFT_SEC, SUBJECTS, MODALITY_FS, load_chest, resample_recording,
    window_slice, window_starts,
)

MODS = ["ACC", "ECG", "EMG", "EDA", "Temp", "Resp", "ACC_x", "ACC_y", "ACC_z"]
DATA_ROOT = REPO / "data" / "WESAD" / "WESAD"
CACHE = REPO / "data" / "wesad_cache"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hop-sec", type=int, default=SHIFT_SEC, dest="hop_sec")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out_path = args.out or CACHE / (
        "wesad_windows_resampled6.npz" if args.hop_sec == SHIFT_SEC
        else f"wesad_windows_resampled6_hop{args.hop_sec}.npz")

    parts = {m: [] for m in MODS}
    ys, grps, starts = [], [], []
    t0 = time.time()
    for sid in SUBJECTS:
        path = DATA_ROOT / f"S{sid}" / f"S{sid}.pkl"
        if not path.exists():
            print(f"  skip S{sid}: missing")
            continue
        sigs, label = load_chest(str(path))
        dec = resample_recording(sigs)
        n = 0
        for s, cls in window_starts(label, args.hop_sec * FS):
            for m in MODS:
                parts[m].append(window_slice(dec, m, s).astype(np.float32))
            ys.append(cls)
            grps.append(sid)
            starts.append(s)
            n += 1
        print(f"  S{sid}: {n} windows  ({time.time() - t0:.0f}s)", flush=True)

    out = {m: np.vstack(parts[m]) for m in MODS}
    out["y_multi"] = np.asarray(ys, int)
    out["groups"] = np.asarray(grps, int)
    out["starts"] = np.asarray(starts, int)
    for m in MODS:
        exp = int(round(60 * MODALITY_FS[m]))
        print(f"{m}: {out[m].shape}  (expect n_win x {exp})")
        assert out[m].shape[1] == exp, f"{m} length {out[m].shape[1]} != {exp}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f"\nwrote {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB, "
          f"{time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
