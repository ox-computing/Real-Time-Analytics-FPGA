import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.signal import butter, sosfilt, sosfiltfilt, welch, find_peaks, resample_poly

FS = 700                      # chest RespiBAN sample rate (Hz)
WIN_SEC = 60
# The hardware emits a class every HOP_SEC (hop_timer, locked at 1 s). Training
# does not need that cadence: at a 1 s hop adjacent windows share 59 of their 60
# seconds, so the rows are ~30x the compute for almost no extra information.
# SHIFT_SEC is therefore the *training* hop and HOP_SEC the frontend's; the
# replay fixtures and the golden model run at HOP_SEC.
HOP_SEC = 1
SHIFT_SEC = 30
WIN = WIN_SEC * FS            # 42000 samples
SHIFT = SHIFT_SEC * FS        # 21000 samples

KEEP_LABELS = (1, 2, 3)                     # baseline, stress, amusement
MULTI_MAP = {1: 0, 2: 1, 3: 2}              # -> 0-indexed class ids
SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
MODALITIES = ["ACC", "ECG", "EMG", "EDA", "Temp", "Resp"]

# Per-modality target output data rate (Hz), bandwidth-matched. Used by --decimate
# to band-limit each signal to its target Nyquist -- modelling what the sensor's
# on-chip ODR/decimation filter delivers, so we can check the rates cost no accuracy.
MODALITY_FS = {"ACC": 32, "ECG": 250, "EMG": 350, "EDA": 25, "Temp": 8, "Resp": 25,
               "ACC_x": 32, "ACC_y": 32, "ACC_z": 32}

# Rational resample factors (up, down) from the 700 Hz source to each modality's
# ODR, in lowest terms. Used by --resample: resample_poly's polyphase anti-alias
# FIR models the sensor AFE's on-chip decimation filter -- a single causal pass
# that both band-limits AND downsamples, unlike the zero-phase --decimate proxy
# above which only band-limits and keeps the 700 Hz grid.
RESAMPLE_UPDOWN = {"ACC": (8, 175), "ECG": (5, 14), "EMG": (1, 2),
                   "EDA": (1, 28), "Temp": (2, 175), "Resp": (1, 28),
                   "ACC_x": (8, 175), "ACC_y": (8, 175), "ACC_z": (8, 175)}

# Fixed PSD sub-bands (Hz). Deliberately span all modalities' ranges so one
# function serves every signal; bands above a signal's content just read ~0.
PSD_BANDS = [(0.0, 0.1), (0.1, 0.5), (0.5, 2.0), (2.0, 8.0), (8.0, 30.0), (30.0, 100.0)]


# --- Filters ----------------------------------------------------------------

def sos_bandpass(x, lo, hi, fs, order=4, causal=False):
    """Bandpass in SOS form (see module docstring on the NaN trap).

    causal=False: zero-phase sosfiltfilt (offline reference).
    causal=True:  single forward sosfilt, the hardware-realistic path -- a live
    stream cannot run the backward pass, so this is what the FPGA implements.
    """
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfilt(sos, x) if causal else sosfiltfilt(sos, x)


def band_limit(x, target_fs, fs, order=4):
    """Low-pass to a target rate's Nyquist -- models what the sensor's on-chip
    decimation filter passes at that ODR. 90% of Nyquist leaves anti-alias margin."""
    cutoff = 0.9 * (target_fs / 2.0)
    sos = butter(order, cutoff, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


# --- Feature functions (each returns an ordered dict {stat: value}) ----------

def time_feats(x):
    """12 time-domain statistics of a 1-D window."""
    p25, p50, p75 = np.percentile(x, [25, 50, 75])
    slope = np.polyfit(np.arange(x.size), x, 1)[0]
    return {
        "mean": x.mean(),
        "std": x.std(),
        "min": x.min(),
        "max": x.max(),
        "range": x.max() - x.min(),
        "median": p50,
        "iqr": p75 - p25,
        "rms": np.sqrt(np.mean(x * x)),
        "mad": np.mean(np.abs(x - x.mean())),
        "skew": stats.skew(x),
        "kurtosis": stats.kurtosis(x),
        "slope": slope,
    }


def psd_feats(x, fs):
    """13 Welch-PSD statistics: 7 spectral-shape + 6 band powers."""
    f, p = welch(x, fs=fs, nperseg=min(x.size, 8192))
    total = p.sum()
    if total <= 0 or not np.isfinite(total):
        # Flat/degenerate window: leave everything NaN, imputed downstream.
        base = {k: np.nan for k in
                ("total", "peak_freq", "peak_power", "centroid", "bandwidth",
                 "entropy", "rolloff85")}
        base.update({f"band{i}": np.nan for i in range(len(PSD_BANDS))})
        return base
    pn = p / total                                   # normalised spectrum
    centroid = np.sum(f * pn)
    bandwidth = np.sqrt(np.sum(((f - centroid) ** 2) * pn))
    entropy = -np.sum(pn * np.log(pn + 1e-12))
    rolloff = f[np.searchsorted(np.cumsum(pn), 0.85)] if np.any(np.cumsum(pn) >= 0.85) else f[-1]
    feats = {
        "total": total,
        "peak_freq": f[np.argmax(p)],
        "peak_power": p.max(),
        "centroid": centroid,
        "bandwidth": bandwidth,
        "entropy": entropy,
        "rolloff85": rolloff,
    }
    for i, (lo, hi) in enumerate(PSD_BANDS):
        feats[f"band{i}"] = p[(f >= lo) & (f < hi)].sum()
    return feats


def hrv_feats(ecg_qrs_window, fs):
    """9 time-domain HRV statistics from R-peaks in a filtered ECG window."""
    # min RR 0.4 s caps HR at 150 bpm; height gate rejects sub-threshold noise.
    peaks, _ = find_peaks(ecg_qrs_window, distance=int(0.4 * fs),
                          height=np.std(ecg_qrs_window))
    nan9 = {k: np.nan for k in
            ("mean_hr", "mean_rr", "sdnn", "rmssd", "pnn50",
             "min_rr", "max_rr", "median_rr", "cvrr")}
    if peaks.size < 3:
        return nan9
    rr = np.diff(peaks) / fs                          # RR intervals (seconds)
    drr = np.diff(rr)
    return {
        "mean_hr": 60.0 / rr.mean(),
        "mean_rr": rr.mean(),
        "sdnn": rr.std(),
        "rmssd": np.sqrt(np.mean(drr * drr)) if drr.size else np.nan,
        "pnn50": np.mean(np.abs(drr) > 0.05) if drr.size else np.nan,
        "min_rr": rr.min(),
        "max_rr": rr.max(),
        "median_rr": np.median(rr),
        "cvrr": rr.std() / rr.mean(),
    }


# --- Per-subject extraction -------------------------------------------------

def load_chest(path):
    """Load one pickle -> ({modality: 1-D signal}, per-sample label array)."""
    with open(path, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    ch = d["signal"]["chest"]
    axes = ch["ACC"].astype(float)
    acc = np.sqrt((axes ** 2).sum(axis=1))
    # The three axes ride along so the fixed-point path can convert each one at
    # the accelerometer's own full scale and reduce afterwards, which is what
    # acc_magnitude.sv does. Taking the magnitude in float first and quantising
    # that reduces before it converts -- the opposite order, and magnitude is
    # nonlinear, so the two do not agree.
    sigs = {
        "ACC": acc,
        "ACC_x": axes[:, 0],
        "ACC_y": axes[:, 1],
        "ACC_z": axes[:, 2],
        "ECG": ch["ECG"].astype(float).ravel(),
        "EMG": ch["EMG"].astype(float).ravel(),
        "EDA": ch["EDA"].astype(float).ravel(),
        "Temp": ch["Temp"].astype(float).ravel(),
        "Resp": ch["Resp"].astype(float).ravel(),
    }
    return sigs, d["label"].astype(int)


def window_starts(label, shift=SHIFT):
    """Yield (start, class_id) for every window lying inside one kept segment."""
    for s in range(0, label.size - WIN + 1, shift):
        seg = label[s:s + WIN]
        lab = seg[0]
        if lab in KEEP_LABELS and np.all(seg == lab):
            yield s, MULTI_MAP[lab]


def resample_recording(sigs):
    """Decimate each modality once over the WHOLE recording, then window from it.

    Windowing first and resampling each window (which is what this module used to
    do) makes the decimation depend on where the window boundary fell: the FIR
    pads with zeros at both ends, so a sample near a boundary is filtered against
    zeros rather than against the samples that really preceded it, and the same
    physical sample decimates to different values in two overlapping windows. A
    sensor's on-chip decimation filter has no window boundaries at all. Doing it
    once over the recording removes both, and is also ~30x less work at a 1 s hop.

    The rates are chosen so every integer-second offset lands on an exact output
    sample (700*up/down is an integer for all five modalities), so a window that
    starts at second t in the 700 Hz stream starts at sample t*fs here with no
    rounding.
    """
    return {m: resample_poly(x, *RESAMPLE_UPDOWN[m]) for m, x in sigs.items()}


def window_slice(dec, mod, start_700, win_sec=WIN_SEC):
    """The decimated window whose 700 Hz start index is start_700."""
    fs = MODALITY_FS[mod]
    lo = start_700 * fs // FS
    assert start_700 * fs % FS == 0, f"{mod}: start {start_700} is not on the {fs} Hz grid"
    return dec[mod][lo:lo + win_sec * fs]


def features_for_window(sigs, ecg_qrs, s, causal=False, resample=False):
    """Flat feature dict for one window across all modalities.

    resample=True reads each modality from the already-decimated recording (see
    resample_recording) and computes every feature at the sensor's true rate --
    the exact signal the FPGA frontend sees, including the coarser R-peak timing
    the 700 Hz band-limit proxy could not show. HRV then comes from the
    decimated ECG, bandpassed once over the whole recording at the ODR.
    """
    e = s + WIN
    feats = {}
    for mod in MODALITIES:
        if resample:
            x = window_slice(sigs, mod, s)
            fs = MODALITY_FS[mod]
        else:
            x, fs = sigs[mod][s:e], FS
        for k, v in time_feats(x).items():
            feats[f"{mod}_t_{k}"] = v
        for k, v in psd_feats(x, fs).items():
            feats[f"{mod}_psd_{k}"] = v
    if resample:
        fs = MODALITY_FS["ECG"]
        lo = s * fs // FS
        hrv_src, hrv_fs = ecg_qrs[lo:lo + WIN_SEC * fs], fs
    else:
        hrv_src, hrv_fs = ecg_qrs[s:e], FS
    for k, v in hrv_feats(hrv_src, hrv_fs).items():
        feats[f"ECG_hrv_{k}"] = v
    return feats


def extract_subject(path, causal=False, decimate=False, resample=False, shift=SHIFT):
    """(X rows, y_multi, subject-id array) for one subject."""
    sigs, label = load_chest(path)
    if decimate:                                   # model sensor-side ODR band-limiting
        sigs = {m: band_limit(x, MODALITY_FS[m], FS) for m, x in sigs.items()}
    if resample:
        sigs = resample_recording(sigs)
        ecg_qrs = sos_bandpass(sigs["ECG"], 5.0, 15.0, MODALITY_FS["ECG"], causal=causal)
    else:
        ecg_qrs = sos_bandpass(sigs["ECG"], 5.0, 15.0, FS, causal=causal)
    rows, ys = [], []
    names = None
    for s, cls in window_starts(label, shift):
        feats = features_for_window(sigs, ecg_qrs, s, causal=causal, resample=resample)
        if names is None:
            names = list(feats.keys())
        rows.append([feats[k] for k in names])
        ys.append(cls)
    if not rows:
        return None
    return np.asarray(rows, float), np.asarray(ys, int), names


# --- Main -------------------------------------------------------------------

def main():
    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="WESAD chest feature extraction")
    ap.add_argument("--data-root", default=str(root / "data" / "WESAD" / "WESAD"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--causal", action="store_true",
                    help="causal sosfilt (hardware-realistic) instead of zero-phase sosfiltfilt")
    ap.add_argument("--decimate", action="store_true",
                    help="band-limit each modality to its target-ODR Nyquist (models sensor-side decimation)")
    ap.add_argument("--resample", action="store_true",
                    help="decimate each modality to its sensor ODR once over the whole recording "
                         "(resample_poly = the AFE's on-chip decimation filter) and compute every "
                         "feature at that true rate")
    ap.add_argument("--hop-sec", type=int, default=SHIFT_SEC, dest="hop_sec",
                    help=f"window hop in seconds (default {SHIFT_SEC}; the frontend runs at "
                         f"{HOP_SEC})")
    args = ap.parse_args()
    if args.decimate and args.resample:
        raise SystemExit("--decimate and --resample are mutually exclusive: both model sensor-side "
                         "band-limiting, and resampling after a band-limit filters the signal twice")
    if args.out is None:
        name = ("wesad_features_resampled.npz" if args.resample
                else "wesad_features_decimated.npz" if args.decimate
                else "wesad_features_causal.npz" if args.causal
                else "wesad_features.npz")
        args.out = str(root / "data" / "wesad_cache" / name)

    X_parts, ym_parts, grp_parts, names = [], [], [], None
    for sid in SUBJECTS:
        path = Path(args.data_root) / f"S{sid}" / f"S{sid}.pkl"
        if not path.exists():
            print(f"  skip S{sid}: {path} missing")
            continue
        out = extract_subject(str(path), causal=args.causal, decimate=args.decimate,
                              resample=args.resample, shift=args.hop_sec * FS)
        if out is None:
            print(f"  skip S{sid}: no valid windows")
            continue
        Xs, ys, names = out
        X_parts.append(Xs)
        ym_parts.append(ys)
        grp_parts.append(np.full(ys.size, sid))
        print(f"  S{sid}: {Xs.shape[0]} windows")

    X = np.vstack(X_parts)
    y_multi = np.concatenate(ym_parts)
    groups = np.concatenate(grp_parts)
    y_binary = (y_multi == MULTI_MAP[2]).astype(int)        # stress vs rest
    feature_names = np.array(names)

    # The check that would have caught the silent all-NaN filter run: a column
    # NaN for every window is a dead feature, not just occasional flat windows.
    all_nan = np.where(np.all(~np.isfinite(X), axis=0))[0]
    print(f"\nX shape: {X.shape}  ({len(feature_names)} features, {X.shape[0]} windows)")
    print(f"class counts (multi 0/1/2): {np.bincount(y_multi).tolist()}")
    print(f"class counts (binary 0/1):  {np.bincount(y_binary).tolist()}")
    print(f"all-NaN columns: {all_nan.size}"
          + (f"  -> {feature_names[all_nan].tolist()}" if all_nan.size else ""))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=X, y_binary=y_binary, y_multi=y_multi,
             groups=groups, feature_names=feature_names)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
