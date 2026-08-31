"""Sensor model and HRV block of the fixed-point golden model.

The converter (adc_quantize), the ACC axis reduction and the streaming R-peak
detector live here. The time path is time_stats_golden_model.py and the spectral
path is fft_fixed_bfp.py; between the three there is exactly one implementation
of each stage, and none of them reads a float feature.
"""

import math
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

# Windowing, rates and the float reference features come from the extractor so
# the fixed path is compared against the same windows it was derived from.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cache_builders"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from psd_features_golden_model import QUOT_FRAC, divide  # noqa: E402
from extract_features import (load_chest, resample_recording, window_starts,
                              MODALITY_FS, sos_bandpass, hrv_feats, PSD_BANDS)

# The 10 time features kept in the frozen contract (skew/kurtosis dropped).
# time_stats_golden_model.py owns how they are computed; this list is only the
# contract's column order.
TIME_KEYS = ("mean", "std", "min", "max", "range",
             "median", "iqr", "rms", "mad", "slope")

# The kept modalities under the TD prune set (Temp dropped). HRV is ECG-only.
KEPT_MODS = ("ACC", "ECG", "EDA", "Resp", "EMG")


# --- Sensor model: a fixed full scale, the way a real converter works ---------
#
# Full scale is a configuration, not a property of the signal: a range register
# (LIS3DH FS[1:0]), a PGA code (ADS1292R gain against its 2.42 V reference), a
# reference pin (MCP3008) or a fixed analog gain (AD8232 at 100 V/V). It is set
# once at boot and never moves.
#
# The spans below are not chosen, they are recovered. WESAD stores physical
# units because the authors applied RespiBAN's transfer functions before
# publishing, but the converter grid survives that multiplication: every raw
# value in the pickles is an integer multiple of span/2**16. Measured on S2,
# 4,255,300 samples per channel:
#
#   ECG   41266 distinct values, step 3/65536       -> 3 V span, +/-1.5 V
#   EMG    4324 distinct values, step 3/65536       -> 3 V span, +/-1.5 V
#   EDA   18583 distinct values, step 25/65536      -> 25 uS span, 0-25 uS
#   Resp  26834 distinct values, step 100/65536     -> 100 % span, +/-50 %
#   ACC          per axis, step 0.0002 g            -> 13.1072 g span, +/-6.5536 g
#
# ECG and EMG hold to that grid within 5.5e-9 relative, so the original 16-bit
# words can be recovered exactly rather than estimated. ACC and Temp sit on an
# offset grid (ACC maps counts 28000-38000 onto -1..+1 g) and Temp's transfer is
# nonlinear, which is one more reason Temp stays dropped.
#
# ACC needs both entries. The pipeline converts the three axes at +/-6.5536 g
# ("ACC_axis") and reduces them with acc_magnitude_fixed, which is the order
# acc_magnitude.sv works in. The "ACC" entry converts a magnitude already taken
# in floating point at 700 Hz -- the opposite order, kept only so the two can be
# compared. The spans are set so the LSB matches either way (13.1072/4095 against
# 6.5536/2047, 0.0032 g), leaving the ordering as the only difference.

RESPIBAN_SPAN = {"ECG": 3.0, "EMG": 3.0, "EDA": 25.0, "Resp": 100.0,
                 "ACC": 13.1072, "ACC_axis": 13.1072}

# Every channel is converted signed, because the feature engine reads
# `stream_data[15:8]` into a signed word: a one-sided channel whose codes run
# 0..2**bits-1 folds negative for every code at or above half scale, and the
# feature stops being monotone in the signal. EDA measured 2.04 % of samples and
# 2.2 % of windows wrapping that way.
#
# ADC_MIDPOINT is what the converter's reference is tied to: 0 for an AC-coupled
# channel, the middle of the range for a one-sided one. Referencing EDA to 12.5 uS
# instead of 0 takes the wrap to 0.000 % and still uses exactly 181 distinct
# top-byte codes -- same 25 uS across the same 2**16 steps, so it is a reference
# pin, not a resolution trade. The ACC magnitude keeps its one-sided reference for
# now; it crosses half scale on 0.2 % of windows and the same fix applies if that
# is judged worth closing.
ADC_SIGNED = {"ECG": True, "EMG": True, "Resp": True, "ACC_axis": True,
              "EDA": True, "ACC": False}

ADC_MIDPOINT = {"ECG": 0.0, "EMG": 0.0, "Resp": 0.0, "ACC_axis": 0.0,
                "EDA": 12.5, "ACC": 0.0}

# Named knob settings. Each is (span table, ADC bits); the reference arm
# "perwindow" is handled separately since it has no fixed scale at all.
ADC_PROFILES = {
    "respiban": (RESPIBAN_SPAN, 16),
    "part12": (RESPIBAN_SPAN, 12),
    "part10": (RESPIBAN_SPAN, 10),
    "part8": (RESPIBAN_SPAN, 8),
    "emgtight": ({**RESPIBAN_SPAN, "EMG": 0.5}, 12),
}


def adc_quantize(x, mod, spans=RESPIBAN_SPAN, bits=16):
    """Map a float window to the integer codes a fixed-scale converter emits.

    span is the whole input range: +/-span/2 for a bipolar channel, 0..span for a
    one-sided one. Out-of-range input saturates, because that is what a converter
    does -- it does not wrap and it does not rescale. mod may be "ACC_axis" to
    convert one accelerometer axis rather than a precomputed magnitude.
    """
    span = spans[mod]
    x = np.asarray(x, dtype=np.float64) - ADC_MIDPOINT[mod]
    if ADC_SIGNED[mod]:
        full = (1 << (bits - 1)) - 1
        code = np.rint(x / (span / 2.0) * full)
        return np.clip(code, -full - 1, full).astype(np.int64)
    full = (1 << bits) - 1
    code = np.rint(x / span * full)
    return np.clip(code, 0, full).astype(np.int64)


def acc_magnitude_fixed(qx, qy, qz):
    """Axis codes -> magnitude code, matching rtl/frontend/acc_magnitude.sv.

    The RTL accumulates sample*sample into a 32-bit sumsq across the three-word
    burst and hands it to isqrt32, so the magnitude carries the axes' own LSB.
    isqrt truncates, which is what math.isqrt does.
    """
    ss = qx.astype(np.int64) ** 2 + qy.astype(np.int64) ** 2 + qz.astype(np.int64) ** 2
    return np.array([math.isqrt(int(v)) for v in ss], dtype=np.int64)


# --- PSD features: shared by every spectral path -----------------------------

def _psd_features_from_power(p, f):
    """The 13 spectral features from a power spectrum p and its frequency axis f.

    p comes from fft_fixed_bfp via psd_power_float, the only spectral engine left.
    The feature math here is still float; making it fixed-point is the #5 step.
    """
    p = p.astype(np.float64)
    total = p.sum()
    keys = ("total", "peak_freq", "peak_power", "centroid", "bandwidth",
            "entropy", "rolloff85")
    if total <= 0 or not np.isfinite(total):    # flat/degenerate window
        base = {k: np.nan for k in keys}
        base.update({f"band{i}": np.nan for i in range(len(PSD_BANDS))})
        return base
    pn = p / total
    centroid = np.sum(f * pn)
    cumsum = np.cumsum(pn)
    feats = {
        "total": total,
        "peak_freq": f[np.argmax(p)],
        "peak_power": p.max(),
        "centroid": centroid,
        "bandwidth": np.sqrt(np.sum(((f - centroid) ** 2) * pn)),
        "entropy": -np.sum(pn * np.log(pn + 1e-12)),
        "rolloff85": f[np.searchsorted(cumsum, 0.85)] if cumsum[-1] >= 0.85 else f[-1],
    }
    for i, (lo, hi) in enumerate(PSD_BANDS):
        feats[f"band{i}"] = p[(f >= lo) & (f < hi)].sum()
    return feats


# --- HRV block: streaming A2 R-peak detector -> RR statistics ----------------
#
# The float reference finds R-peaks with scipy.find_peaks(distance=0.4*fs,
# height=std(window)) -- a non-causal, global "keep the tallest within distance"
# search over an isolated window. The hardware detector runs on the stream: it
# never sees a window boundary, so refractory state and the amplitude gate are
# continuous across hops and a beat near an edge is detected once, not twice and
# not differently depending on which window it lands in.
#
# That forces the gate to change. A per-window std needs the whole window before
# it can classify anything inside it, which a streaming detector does not have.
# The replacement is an exponentially-weighted mean of |x|, updated once per
# sample: acc += |x| - (acc >> K), so mean|x| = acc >> K -- one adder, one
# subtract, no multiplier and no second pass. For a zero-mean bandpassed signal
# E|x| = sigma*sqrt(2/pi), so sigma ~= 1.25*mean|x| and the std gate comes back
# as thr = m + (m >> 2): two adds, still no multiplier.
#
# All eight HRV features derive from R-peak *timing*, so the converter's
# amplitude scaling does not affect them; RR intervals are kept in integer
# samples and 1/fs is folded into the output units only.

# The 8 HRV features kept in the frozen contract (cvrr dropped).
HRV_KEYS = ("mean_hr", "mean_rr", "sdnn", "rmssd", "pnn50",
            "min_rr", "max_rr", "median_rr")

RMSSD_FRAC = 6                   # guard bits for the small mean(drr^2) sqrt
EWMA_SHIFT = 8                   # K in acc += |x| - (acc >> K); ~1 s at 250 Hz

# 5-15 Hz Butterworth as four biquads with Q14 feedback and b = +/-[1 2 1].
# scipy designs this in float64, which no fixed-point datapath can reproduce, so
# the reference runs the same cascade the RTL runs. The overall gain is dropped:
# the detector's gate is a 1.25x EWMA of this signal's own magnitude, so the
# filter's scale cancels and only its shape matters.
BIQUAD_A = ((-27486, 12232), (-29701, 13745), (-28590, 14238), (-31694, 15577))
BIQUAD_FRAC = 14
BIQUAD_W = 24
BIQUAD_SHIFT = 3                 # keeps the gainless cascade inside BIQUAD_W


def _wrap(v, bits):
    half = 1 << (bits - 1)
    return ((int(v) + half) & ((1 << bits) - 1)) - half


def biquad_cascade_fixed(x_int):
    """The 5-15 Hz cascade in the exact integer arithmetic hrv_features.sv uses."""
    w1 = [0] * 4
    w2 = [0] * 4
    out = np.zeros(len(x_int), dtype=np.int64)
    for n in range(len(x_int)):
        v = _wrap(int(x_int[n]), BIQUAD_W)
        for k in range(4):
            a1, a2 = BIQUAD_A[k]
            w0 = _wrap(v - _wrap((a1 * w1[k] + a2 * w2[k]) >> BIQUAD_FRAC, BIQUAD_W), BIQUAD_W)
            if k < 2:
                y = w0 + 2 * w1[k] + w2[k]
            else:
                y = w0 - 2 * w1[k] + w2[k]
            w2[k] = w1[k]
            w1[k] = w0
            v = _wrap(y >> BIQUAD_SHIFT, BIQUAD_W)
        out[n] = v
    return out


def running_threshold(x_int, shift=EWMA_SHIFT):
    """Per-sample amplitude gate: 1.25 * EWMA(|x|), integer, no multiplier.

    Returned as a full-length array so the detector can be expressed as a
    vectorised comparison here while the RTL keeps one accumulator register.
    """
    x = np.abs(x_int.astype(np.int64))
    acc = np.zeros(x.size + 1, dtype=np.int64)
    a = 0                        # a streaming engine has no lookahead to seed with
    for i, v in enumerate(x):
        acc[i] = a
        a = a + int(v) - (a >> shift)
    acc[-1] = a
    m = acc[:-1] >> shift
    return m + (m >> 2)


def hrv_peaks_stream(x_int, fs, shift=EWMA_SHIFT):
    """R-peak indices over a whole recording, single causal pass.

    Candidates are strict local maxima at or above the running gate. One forward
    pass then enforces the refractory period ref = 0.4*fs samples: while
    successive candidates fall within ref of the pending peak the taller one
    wins and the refractory window slides to it; the first candidate ref samples
    past the pending peak commits it and opens the next.
    """
    thr = running_threshold(x_int, shift)
    body = x_int[1:-1]
    # strict local max on the left, non-strict on the right (scipy's edge rule).
    cand = np.nonzero((body >= thr[1:-1]) & (body > x_int[:-2]) & (body >= x_int[2:]))[0] + 1

    ref = int(0.4 * fs)
    warmup = 1 << shift          # the gate is still ramping over its own window
    peaks = []
    pend_i, pend_v = None, None
    for i in cand[cand >= warmup]:
        v = x_int[i]
        if pend_i is None:
            pend_i, pend_v = int(i), int(v)
        elif i - pend_i < ref:
            if v > pend_v:
                pend_i, pend_v = int(i), int(v)       # taller peak inside refractory
        else:
            peaks.append(pend_i)                      # pending peak clears refractory
            pend_i, pend_v = int(i), int(v)
    if pend_i is not None:
        peaks.append(pend_i)
    return np.asarray(peaks, dtype=np.int64)


def hrv_feats_fixed(peaks, fs):
    """8 HRV features from R-peak sample indices, integer throughout.

    RR intervals stay in integer samples and every output is either an integer
    sample count or a quotient off the SHARED DIVIDER carrying QUOT_FRAC
    fractional bits -- the same unit the PSD features use. The per-column
    constants that would turn samples into seconds (1/fs) are dropped, because
    the thermometer learns a ladder per column and a constant scale cannot move
    a quantile. mean_hr keeps its 60*fs so it lands in bpm, where its own
    resolution is natural; that constant is a shift-add, not a divide.

    <3 peaks -> all zero. On this dataset that never happens: a 60 s window at a
    resting heart rate holds 60-100 beats, and the streaming detector has no
    per-window gate to fail on, so the measured NaN rate over 1015 windows x 5
    modalities is 0.
    """
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.size < 3:
        return {k: 0 for k in HRV_KEYS}

    rr = np.diff(peaks)                                # RR intervals, samples
    drr = np.diff(rr)
    n = int(rr.size)
    s = int(rr.sum())                                  # == peaks[-1]-peaks[0]
    ss = int((rr.astype(np.int64) ** 2).sum())

    # sdnn*n = isqrt(n*ss - s^2) is exact and non-negative; the divide by n is
    # the same restoring divider the PSD features use.
    sdnn = divide(math.isqrt(max(0, n * ss - s * s)), n)

    if drr.size:
        # mean(drr^2) is small (single-digit sample counts), so carry RMSSD_FRAC
        # guard bits into the isqrt or the floor swamps its precision; the result
        # is then re-aligned to QUOT_FRAC.
        num = (int((drr.astype(np.int64) ** 2).sum()) << (2 * RMSSD_FRAC)) // drr.size
        rmssd = math.isqrt(num) << (QUOT_FRAC - RMSSD_FRAC)
        # |drr| > 0.05 s, compared as 2*|drr| > 0.1*fs to stay integer
        pnn50 = divide(int(np.count_nonzero(2 * np.abs(drr) > int(round(0.1 * fs)))),
                       int(drr.size))
    else:
        rmssd = pnn50 = 0

    rr_sorted = np.sort(rr)                            # tiny array; nearest-rank
    return {
        "mean_hr": divide(n * 60 * fs, s),             # bpm
        "mean_rr": divide(s, n),                       # samples
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "min_rr": int(rr.min()),
        "max_rr": int(rr.max()),
        "median_rr": int(rr_sorted[n // 2]),
    }


# --- Self-test: streaming detector vs the scipy per-window reference ---------

def _selftest_hrv(subject_pkl):
    """Compare streaming-A2 HRV against scipy find_peaks on the same ECG.

    The reference filters and detects per window; the fixed path filters the
    whole recording once and detects on the stream, then takes the peaks lying
    inside each window. Both read the true-ODR ECG. Peaks near a window edge are
    expected to move -- that is the point of the change, not a regression.
    """
    sigs, label = load_chest(subject_pkl)
    fs = MODALITY_FS["ECG"]
    dec = resample_recording(sigs)["ECG"]
    spans, bits = ADC_PROFILES["respiban"]

    filtered = sos_bandpass(dec, 5.0, 15.0, fs, causal=True)
    codes = adc_quantize(filtered, "ECG", spans, bits)
    peaks = hrv_peaks_stream(codes, fs)

    err = {k: [] for k in HRV_KEYS}
    nwin = matched = total = 0
    for s0, _cls in window_starts(label):
        nwin += 1
        lo = int(round(s0 * fs / 700.0))
        hi = lo + 60 * fs
        ref = hrv_feats(filtered[lo:hi], fs)
        win_peaks = peaks[(peaks >= lo) & (peaks < hi)] - lo
        fix = hrv_feats_fixed(win_peaks, fs)
        ref_peaks, _ = find_peaks(filtered[lo:hi], distance=int(0.4 * fs),
                                  height=np.std(filtered[lo:hi]))
        total += ref_peaks.size
        matched += int(np.isin(ref_peaks, win_peaks).sum())
        for k in HRV_KEYS:
            a, b = ref[k], fix[k]
            if np.isnan(a) and np.isnan(b):
                continue
            if np.isnan(a) or np.isnan(b):
                err[k].append(1.0)
                continue
            err[k].append(abs(a - b) / max(abs(a), abs(b), 1e-9))
    print(f"{subject_pkl}  HRV  ({nwin} windows)")
    print(f"  peaks matching scipy exactly: {matched}/{total} ({100.0 * matched / total:.1f}%)")
    print(f"{'feature':>10}  {'median rel err':>15}  {'max rel err':>12}")
    for k in HRV_KEYS:
        e = np.array(err[k]) if err[k] else np.array([0.0])
        print(f"{k:>10}  {np.median(e):>15.2e}  {np.max(e):>12.2e}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    _selftest_hrv(str(root / "data" / "WESAD" / "WESAD" / "S2" / "S2.pkl"))
