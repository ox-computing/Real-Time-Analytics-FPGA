"""Assemble the 123-column contract cache entirely from the hardware datapaths.

50 time columns from time_stats_golden_model (8-bit sample, 8-bit feature word,
exact variance, fitted shift+offset narrowing), 65 PSD columns from the integer
BFP FFT, 8 HRV columns from the streaming A2 detector. Nothing here comes from
the float extractor, so the cache the thermometer is fit on is the cache the
silicon produces.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "wesad" / "scripts" / "cache_builders"))
sys.path.insert(0, str(REPO / "wesad" / "scripts" / "golden_models"))

from extract_features import MODALITY_FS, PSD_BANDS  # noqa: E402
from fixed_frontend import (ADC_PROFILES, HRV_KEYS, acc_magnitude_fixed,  # noqa: E402
                            adc_quantize, biquad_cascade_fixed,
                            hrv_feats_fixed, hrv_peaks_stream)
from fft_fixed_bfp import (ACC_BITS, HANN_BITS, TW_BITS,  # noqa: E402
                           psd_power_bfp)
from psd_features_golden_model import (PSD_KEYS as PSD_FEATURE_KEYS,  # noqa: E402
                                       band_edges, psd_features_fixed)
from time_stats_golden_model import (FEATURES as TIME_FEATURES,  # noqa: E402
                                     SENSORS, fit_narrowing, load_narrowing,
                                     raw_columns, time_stats_fixed, wrap)

PSD_NARROWING = (REPO / "wesad" / "scripts" / "golden_models" / "psd_narrowing.json")

CACHE = REPO / "data" / "wesad_cache"
WINDOWS = CACHE / "wesad_windows_resampled6.npz"

PSD_KEYS = PSD_FEATURE_KEYS

NFFT = 2048
# ECG/EMG segment at the engine size; the others are shorter than it and fall
# out as K=1 (the dt2048k13 plan).
SEG_LEN = {"ECG": 2048, "EMG": 2048}
CHUNK = 256


def report_sample_wrap(codes, bits):
    """How often the top byte the datapath reads comes back the wrong sign.

    `time_stats.sv` takes `stream_data[15:8]` into a signed word, so a one-sided
    channel whose codes run 0..2**bits-1 arrives negative for every code at or
    above half scale. That is not a rounding difference -- it folds the top half
    of the range onto the bottom and the feature stops being monotone in the
    signal. Reported per modality so the rate is visible rather than inferred.
    """
    shift = bits - 8
    print(f"\n{'channel':8} {'samples wrapped':>16} {'windows touched':>16} {'max code':>10}")
    for m in SENSORS:
        c = np.asarray(codes[m])
        top = c >> shift
        wrapped = top > 127
        print(f"{m:8} {100.0 * wrapped.mean():15.3f}% "
              f"{100.0 * wrapped.any(axis=1).mean():15.1f}% {int(c.max()):10d}")


def column_names():
    names = [f"{m}_t_{f}" for m in SENSORS for f in TIME_FEATURES]
    names += [f"{m}_psd_{k}" for m in SENSORS for k in PSD_KEYS]
    names += [f"ECG_hrv_{k}" for k in HRV_KEYS]
    return names


def time_columns(codes, shift, offset):
    n_win = codes[SENSORS[0]].shape[0]
    out = np.zeros((n_win, len(SENSORS) * len(TIME_FEATURES)))
    for s, mod in enumerate(SENSORS):
        base = s * len(TIME_FEATURES)
        for w in range(n_win):
            f = time_stats_fixed(codes[mod][w], s, shift, offset)
            out[w, base:base + len(TIME_FEATURES)] = [f[k] for k in TIME_FEATURES]
    return out


def psd_columns(codes, width, tw_bits, hann_bits, acc_bits, cap_mode):
    """65 spectral columns, integer from the transform to the feature word."""
    n_win = codes[SENSORS[0]].shape[0]
    out = np.zeros((n_win, len(SENSORS) * len(PSD_KEYS)))
    over_total, widths = 0, {}
    tab = json.loads(PSD_NARROWING.read_text())
    for s, mod in enumerate(SENSORS):
        base = s * len(PSD_KEYS)
        edges = band_edges(MODALITY_FS[mod], NFFT, PSD_BANDS)
        for lo in range(0, n_win, CHUNK):
            blk = codes[mod][lo:lo + CHUNK]
            acc, pe, over = psd_power_bfp(blk, SEG_LEN.get(mod), NFFT, width,
                                          detrend=True, tw_bits=tw_bits,
                                          hann_bits=hann_bits, acc_bits=acc_bits,
                                          count_overflow=True, cap_mode=cap_mode)
            over_total += over
            for r in range(blk.shape[0]):
                feats = psd_features_fixed(acc[r], pe[r], edges, widths)
                # narrowed here so the cache holds the 8-bit word psd_calc emits
                out[lo + r, base:base + len(PSD_KEYS)] = [
                    wrap((feats[k] >> tab["shift"][s][j]) - tab["offset"][s][j], 8)
                    for j, k in enumerate(PSD_KEYS)]
    print("  psd accumulator widths (bits): "
          + "  ".join(f"{k} {v}" for k, v in sorted(widths.items(), key=lambda t: -t[1])))
    return out, over_total


def hrv_columns(win, adc):
    """HRV from peaks detected once over each subject's whole ECG stream.

    The window cache stores windows, not recordings, so the stream is
    reconstructed per subject from the non-overlapping part of each window --
    exact whenever the hop divides the window length, which every cache here
    satisfies. Peaks are then taken per window from that continuous detection.
    """
    spans, bits = ADC_PROFILES[adc]
    fs = MODALITY_FS["ECG"]
    n_win, n_samp = win["ECG"].shape
    groups, starts = win["groups"], win["starts"]
    out = np.full((n_win, len(HRV_KEYS)), np.nan)

    for sid in np.unique(groups):
        rows = np.nonzero(groups == sid)[0]
        order = rows[np.argsort(starts[rows])]
        # Windows only exist inside a kept label segment, so the starts come in
        # runs with gaps between segments. Splicing across a gap would feed the
        # detector a discontinuity that is not in the recording, and the EWMA
        # gate would carry the artefact ~1 s into the next window; each run gets
        # its own stream instead.
        hop = int(np.min(np.diff(starts[order]))) if order.size > 1 else 0
        runs, cur = [], [order[0]]
        for a, b in zip(order, order[1:]):
            if starts[b] - starts[a] == hop:
                cur.append(b)
            else:
                runs.append(cur)
                cur = [b]
        runs.append(cur)

        for run in runs:
            base = starts[run[0]] * fs // 700
            span = (starts[run[-1]] - starts[run[0]]) * fs // 700 + n_samp
            stream = np.zeros(span)
            for r in run:
                lo = starts[r] * fs // 700 - base
                stream[lo:lo + n_samp] = win["ECG"][r]
            # the converter comes before the digital filter in hardware
            filtered = biquad_cascade_fixed(adc_quantize(stream, "ECG", spans, bits))
            peaks = hrv_peaks_stream(filtered, fs)
            for r in run:
                lo = starts[r] * fs // 700 - base
                inside = peaks[(peaks >= lo) & (peaks < lo + n_samp)] - lo
                feats = hrv_feats_fixed(inside, fs)
                out[r] = [feats[k] for k in HRV_KEYS]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adc", default="respiban", choices=sorted(ADC_PROFILES))
    ap.add_argument("--sample-bits", type=int, default=None, dest="sample_bits",
                    help="keep only the top N bits of each converter code")
    ap.add_argument("--fft-width", type=int, default=8, dest="width")
    ap.add_argument("--tw-bits", type=int, default=TW_BITS, dest="tw_bits")
    ap.add_argument("--hann-bits", type=int, default=HANN_BITS, dest="hann_bits")
    ap.add_argument("--acc-bits", type=int, default=ACC_BITS, dest="acc_bits")
    ap.add_argument("--cap-mode", default="sat", choices=["renorm", "sat"],
                    dest="cap_mode")
    ap.add_argument("--windows", type=Path, default=WINDOWS)
    ap.add_argument("--refit-narrowing", action="store_true", dest="refit",
                    help="refit the time narrowing tables over these windows first")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    tag = args.tag or (f"{args.adc}_fftw{args.width}"
                       + ("" if args.sample_bits is None else f"_sb{args.sample_bits}")
                       + ("" if args.cap_mode == "renorm" else "_sat"))
    out = args.out or CACHE / f"wesad_features_hw123_{tag}.npz"

    win = np.load(args.windows)
    spans, bits = ADC_PROFILES[args.adc]
    codes = {m: adc_quantize(win[m], m, spans, bits) for m in SENSORS}

    # ACC: convert each axis at the accelerometer's own full scale and reduce
    # afterwards, which is the order acc_magnitude.sv works in. Quantising the
    # float magnitude instead reduces before it converts, and magnitude is
    # nonlinear, so the two orders do not agree. Measured accuracy-free
    # (+0.0069, p=0.367) -- taken here for datapath fidelity, not accuracy.
    if all(f"ACC_{a}" in win.files for a in "xyz"):
        qx, qy, qz = (adc_quantize(win[f"ACC_{a}"], "ACC_axis", spans, bits) for a in "xyz")
        codes["ACC"] = np.vstack([acc_magnitude_fixed(qx[i], qy[i], qz[i])
                                  for i in range(qx.shape[0])])
        print("ACC reduced from per-axis codes (quantise-then-reduce)")
    else:
        print("WARNING: window cache has no ACC axes -- ACC is a quantised float "
              "magnitude, which is not the order acc_magnitude.sv reduces in")
    if args.sample_bits is not None:
        if args.sample_bits > bits:
            raise SystemExit(f"--sample-bits {args.sample_bits} exceeds the converter's {bits}")
        codes = {m: c >> (bits - args.sample_bits) for m, c in codes.items()}

    report_sample_wrap(codes, bits if args.sample_bits is None else args.sample_bits)

    if args.refit:
        shift, offset = fit_narrowing(codes)
        print("refit narrowing tables")
    else:
        shift, offset = load_narrowing()

    t0 = time.time()
    Xt = time_columns(codes, shift, offset)
    print(f"  time  {Xt.shape[1]} cols  ({time.time() - t0:.0f}s)", flush=True)
    Xp, over = psd_columns(codes, args.width, args.tw_bits, args.hann_bits,
                           args.acc_bits, args.cap_mode)
    print(f"  psd   {Xp.shape[1]} cols  datapath overflows {over}  "
          f"({time.time() - t0:.0f}s)", flush=True)
    Xh = hrv_columns(win, args.adc)
    print(f"  hrv   {Xh.shape[1]} cols  ({time.time() - t0:.0f}s)", flush=True)

    X = np.hstack([Xt, Xp, Xh])
    names = column_names()
    assert X.shape[1] == len(names) == 123, f"{X.shape[1]} columns against {len(names)}"

    # raw_time carries the pre-narrow accumulators so train_dwn can refit the
    # narrowing tables per LOSO fold instead of inheriting one global fit.
    raw_time = raw_columns(codes)

    y_multi = win["y_multi"]
    np.savez_compressed(out, X=X, y_multi=y_multi,
                        y_binary=(y_multi == 1).astype(int),
                        groups=win["groups"], feature_names=np.array(names),
                        raw_time=raw_time)
    dead = np.where(np.all(~np.isfinite(X), axis=0))[0]
    const = [names[i] for i in range(X.shape[1])
             if np.unique(X[np.isfinite(X[:, i]), i]).size <= 1]
    print(f"\nwrote {out}  ({X.shape[0]} x {X.shape[1]})")
    print(f"  dead columns {dead.size}" + (f" {[names[i] for i in dead]}" if dead.size else ""))
    print(f"  constant columns {len(const)}" + (f" {const}" if const else ""))
    print(f"  NaN rate {100.0 * np.mean(~np.isfinite(X)):.2f}%")


if __name__ == "__main__":
    main()
