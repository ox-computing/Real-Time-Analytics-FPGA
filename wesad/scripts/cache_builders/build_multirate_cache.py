"""Build a feature cache with a per-modality window length (multi-rate fusion).

The flat design computes every modality's features over the same 60 s window. That is
the wrong window for at least two of them: an EDA window holds only a handful of skin
conductance responses and its tonic level drifts over minutes, and respiratory rate
variability needs many breaths before it means anything, while ACC/EMG/ECG carry their
information over seconds. Re-encoding a slow channel at the fast rate spends input bits
on a quantity that has barely moved.

This builder gives each modality its own window, all ending at the same instant, and
holds everything else fixed:

  - window_starts is imported unmodified, so the row set, labels and groups are
    IDENTICAL to the 60 s cache built through this same file. The two are row-paired,
    which is what makes the LOSO comparison a paired one.
  - a long window is clamped to the start of its own label segment rather than allowed
    to run back into the previous condition, so a longer window never buys accuracy by
    importing label information. The clamp rate is reported per modality.
  - the control arm (--slow-sec 60) runs the same code path, so the A/B isolates the
    window length. Absolute numbers will not match wesad_features_dt2048k13_6mod.npz,
    which recomputes the PSD columns on the 2048-point engine; this file uses scipy
    welch throughout, on both sides.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "wesad" / "scripts" / "cache_builders"))

from extract_features import (  # noqa: E402
    FS, KEEP_LABELS, MODALITIES, MODALITY_FS, MULTI_MAP, RESAMPLE_UPDOWN, SUBJECTS,
    WIN, WIN_SEC, hrv_feats, load_chest, psd_feats, sos_bandpass, time_feats,
    window_starts,
)

DEFAULT_SLOW = ("EDA", "Resp", "Temp")


def segment_starts(label):
    """For every sample, the index at which its contiguous same-label run began.

    A 240 s EDA window sitting near the front of a stress block would otherwise reach
    back into the baseline block and read a condition change as an EDA feature. Clamping
    to the run start keeps every sample in the window drawn from the labelled condition,
    at the cost of a shorter effective window for the first few windows of each block --
    which is also what a real device sees while its buffer fills.
    """
    starts = np.zeros(label.size, dtype=np.int64)
    run_start = 0
    for i in range(1, label.size):
        if label[i] != label[i - 1]:
            run_start = i
        starts[i] = run_start
    return starts


def window_for(sig, seg_start, end, win_samples, updown):
    """One modality's window, clamped to its segment and resampled to its ODR.

    Returns (x at the modality's ODR, clamped?). Slicing happens at 700 Hz and the
    polyphase resample runs on the slice, matching how features_for_window builds the
    --resample path.
    """
    start = max(seg_start, end - win_samples)
    return resample_poly(sig[start:end], *updown), start > end - win_samples


def features_for_multirate_window(sigs, seg_start, end, win_samples):
    """Flat feature dict for one window, each modality on its own window length.

    Mirrors extract_features.features_for_window's --resample branch feature-for-feature
    -- same time_feats, same psd_feats at the modality ODR, same HRV from a 5-15 Hz
    bandpass of the downsampled ECG -- so the only difference between this cache and the
    resampled 60 s cache is how far back each window reaches.
    """
    feats, clamped = {}, {}
    for mod in MODALITIES:
        x, was_clamped = window_for(sigs[mod], seg_start, end, win_samples[mod],
                                    RESAMPLE_UPDOWN[mod])
        clamped[mod] = was_clamped
        for k, v in time_feats(x).items():
            feats[f"{mod}_t_{k}"] = v
        for k, v in psd_feats(x, MODALITY_FS[mod]).items():
            feats[f"{mod}_psd_{k}"] = v
    ecg, _ = window_for(sigs["ECG"], seg_start, end, win_samples["ECG"],
                        RESAMPLE_UPDOWN["ECG"])
    hrv_src = sos_bandpass(ecg, 5.0, 15.0, MODALITY_FS["ECG"], causal=False)
    for k, v in hrv_feats(hrv_src, MODALITY_FS["ECG"]).items():
        feats[f"ECG_hrv_{k}"] = v
    return feats, clamped


def extract_subject_multirate(path, win_samples):
    """(X rows, y_multi, names, clamp counts, n) for one subject."""
    sigs, label = load_chest(path)
    seg = segment_starts(label)
    rows, ys, names = [], [], None
    clamp = {m: 0 for m in MODALITIES}
    for s, cls in window_starts(label):
        end = s + WIN
        feats, clamped = features_for_multirate_window(sigs, int(seg[s]), end, win_samples)
        if names is None:
            names = list(feats.keys())
        rows.append([feats[k] for k in names])
        ys.append(cls)
        for m, c in clamped.items():
            clamp[m] += int(c)
    if not rows:
        return None
    return np.asarray(rows, float), np.asarray(ys, int), names, clamp, len(ys)


def main():
    ap = argparse.ArgumentParser(description="WESAD multi-rate (per-modality window) cache")
    ap.add_argument("--data-root", default=str(REPO / "data" / "WESAD" / "WESAD"))
    ap.add_argument("--slow-sec", type=int, default=240,
                    help="window length for the slow modalities (60 = control arm)")
    ap.add_argument("--slow-mods", nargs="*", default=list(DEFAULT_SLOW),
                    help="modalities that get the long window")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    win_sec = {m: (args.slow_sec if m in args.slow_mods else WIN_SEC) for m in MODALITIES}
    win_samples = {m: int(v * FS) for m, v in win_sec.items()}
    if args.out is None:
        tag = f"{args.slow_sec}s_" + "-".join(sorted(args.slow_mods)) if args.slow_sec != WIN_SEC \
              else "ctrl60s"
        args.out = str(REPO / "data" / "wesad_cache" / f"wesad_features_mr_{tag}.npz")

    print(f"window lengths (s): {win_sec}")
    print(f"buffer samples at each sensor ODR: "
          f"{ {m: int(win_sec[m] * MODALITY_FS[m]) for m in MODALITIES} }")

    X_parts, ym_parts, grp_parts, names = [], [], [], None
    clamp_tot = {m: 0 for m in MODALITIES}
    n_tot, t0 = 0, time.time()
    for sid in SUBJECTS:
        path = Path(args.data_root) / f"S{sid}" / f"S{sid}.pkl"
        if not path.exists():
            print(f"  skip S{sid}: {path} missing")
            continue
        out = extract_subject_multirate(str(path), win_samples)
        if out is None:
            print(f"  skip S{sid}: no valid windows")
            continue
        Xs, ys, names, clamp, n = out
        X_parts.append(Xs)
        ym_parts.append(ys)
        grp_parts.append(np.full(ys.size, sid))
        n_tot += n
        for m, c in clamp.items():
            clamp_tot[m] += c
        print(f"  S{sid}: {Xs.shape[0]} windows  ({time.time() - t0:.0f}s)")

    X = np.vstack(X_parts)
    y_multi = np.concatenate(ym_parts)
    groups = np.concatenate(grp_parts)
    y_binary = (y_multi == MULTI_MAP[2]).astype(int)
    feature_names = np.array(names)

    all_nan = np.where(np.all(~np.isfinite(X), axis=0))[0]
    print(f"\nX shape: {X.shape}  ({len(feature_names)} features, {X.shape[0]} windows)")
    print(f"class counts (multi 0/1/2): {np.bincount(y_multi).tolist()}")
    print("clamped windows (long window hit its segment start): "
          + ", ".join(f"{m} {clamp_tot[m]}/{n_tot} ({clamp_tot[m] / max(n_tot, 1):.1%})"
                      for m in MODALITIES if clamp_tot[m]))
    print(f"all-NaN columns: {all_nan.size}"
          + (f"  -> {feature_names[all_nan].tolist()}" if all_nan.size else ""))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=X, y_binary=y_binary, y_multi=y_multi,
             groups=groups, feature_names=feature_names)
    print(f"\nwrote {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
