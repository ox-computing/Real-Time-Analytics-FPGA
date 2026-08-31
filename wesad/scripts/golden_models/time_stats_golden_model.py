"""Bit-faithful Python model of time_stats.sv that builds the integer feature cache the thermometer thresholds are fit on, and serves as the golden reference for hardware comparison."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "wesad" / "scripts" / "golden_models"))
sys.path.insert(0, str(REPO / "verification"))
from fixed_frontend import ADC_PROFILES, adc_quantize  # noqa: E402
from hw_model import HWModel  # noqa: E402

SENSORS = ["ECG", "ACC", "Resp", "EDA", "EMG"]
FEATURES = ["mean", "std", "rms", "min", "max", "range", "median", "iqr", "mad", "slope"]
WINDOW_LEN = [15000, 1920, 1500, 1500, 21000]

# Every width and shift below mirrors time_stats.sv one for one. That module is
# the only source for them, so a change there has to be copied here or the
# reference silently drifts from the hardware it is meant to check.
#
# The stream word stays 16 bits all the way through the SPRAM and the shift
# register, but the accumulators only ever see its top byte -- time_stats has
# carried `sample = stream_data[15:8]` since the feature datapath was narrowed.
# So the model truncates its input the same way instead of running the full
# codes through a datapath that no longer holds them.
SAMPLE_SHIFT = 8
SAMPLE_W = 8
MEAN_SUM_W = 24
SQ_SUM_W = 32
PRE_SUM_W = 48
MEAN_FOR_SLOPE_W = 18
SLOPE_W = 34
VARIANCE_W = 46
SQRT_ARG_W = 32
FEATURE_W = 8
RANGE_W = 9                      # max - min of two signed 8-bit samples reaches 255

# Narrowing tables: every feature reaches the 8-bit word as (raw >>> shift) - offset.
#
# The shift alone is not enough. Sized so the widest window in the set cannot
# overflow, it leaves the typical window in a handful of codes -- under a fixed
# converter span a median window occupies 8.6-51% of full scale, so a feature
# whose between-window spread is small lands in a few adjacent codes and carries
# almost no information. The offset is a per-feature constant, so it costs one
# shared 8-bit subtract in the write path and folds into nothing else; with it
# the shift can be chosen to fill the word with the spread rather than the range.
#
# Fitted by fit_narrowing() on training windows only and written to
# narrowing.json; the tables below are the defaults that file overrides.
NARROWING_FILE = Path(__file__).resolve().parent / "narrowing.json"

DEFAULT_SHIFT = {
    "mean": [6, 4, 5, 11, 6], "std": [8, 31, 31, 31, 31], "rms": [14, 11, 11, 11, 14],
    "range": [1, 1, 1, 1, 1], "median": [0, 0, 0, 0, 0], "iqr": [0, 0, 0, 0, 0],
    "mad": [8, 8, 8, 8, 8], "slope": [6, 3, 3, 3, 7],
}
DEFAULT_OFFSET = {k: [0, 0, 0, 0, 0] for k in DEFAULT_SHIFT}

CACHE = REPO / "data" / "wesad_cache" / "wesad_windows_resampled6.npz"
EXPORT = REPO / "verification" / "exported" / "dwn_wesad_timeonly35_int_100_51_tau3.5.json"

# The converter the reference runs through. A per-window divisor is not a
# converter setting, so there is no AGC arm here any more.
ADC = "respiban"


def wrap(v, bits):
    half = 1 << (bits - 1)
    return ((v + half) & ((1 << bits) - 1)) - half


def trunc(v, bits):
    return v & ((1 << bits) - 1)


def load_narrowing(path=NARROWING_FILE):
    """(shift, offset) tables, from narrowing.json when it exists."""
    if Path(path).exists():
        d = json.loads(Path(path).read_text())
        return d["shift"], d["offset"]
    return DEFAULT_SHIFT, DEFAULT_OFFSET


def narrow(v, feat, sensor, shift, offset):
    """raw -> the signed 8-bit feature word the thermometer compares."""
    return wrap((v >> shift[feat][sensor]) - offset[feat][sensor], FEATURE_W)


def accumulators(x, sensor):
    """The pre-narrow quantities, exactly as the RTL's registers hold them.

    variance is n*sum(x^2) - sum(x)^2, computed with no pre-shift on either
    term. At 8-bit samples both products reach 2**43 at EMG's 21000 samples and
    VARIANCE_W is already 46, so the exact difference fits the register that is
    there -- and being exact it is non-negative by construction. The `>> 14` and
    `>> 7` this replaces were sized for a 16-bit sample and were never rescaled
    when the datapath narrowed; at 8 bits they left ~11 bits for a subtraction
    that cancels almost everything, which drove the difference negative on
    896/1015 ACC and 638/1015 EDA windows and pinned std to two distinct values.
    """
    n = WINDOW_LEN[sensor]
    # `sample = stream_data[15:8]` into a `logic signed [7:0]`, so the top byte is
    # read as two's complement whatever the converter meant it as. EDA and the ACC
    # magnitude are one-sided channels whose codes run 0..2**16-1, so any code at
    # or above half scale reaches the datapath negative. That is what the RTL
    # does; modelling it faithfully is the point, but it makes those two features
    # non-monotone exactly like `range` did.
    sample = np.array([wrap(int(v) >> SAMPLE_SHIFT, SAMPLE_W) for v in x], dtype=np.int64)

    mean_sum = wrap(int(sample.sum()), MEAN_SUM_W)
    sq_sum = wrap(int((sample * sample).sum()), SQ_SUM_W)
    # pre_sum adds mean_sum before the current sample joins it, which closes to
    # sum((n-1-i)*sample[i]). That closed form only holds while mean_sum itself
    # never wraps: a wrapped partial sum would carry into the wider pre_sum.
    prefix = np.cumsum(sample)
    assert int(np.abs(prefix).max()) < 1 << (MEAN_SUM_W - 1), f"{SENSORS[sensor]}: mean_sum wraps"
    pre_sum = wrap(int(prefix[:-1].sum()), PRE_SUM_W)

    mean_for_slope = wrap(mean_sum >> 14, MEAN_FOR_SLOPE_W)
    pre_for_slope = wrap(pre_sum >> 13, SLOPE_W)
    absolute_mean = abs(mean_sum)

    # MULT_SQ and MULT_MEANSQ share one 48-bit shift-add accumulator, so the
    # n*sq and -mean^2 terms land in the same register before variance takes its
    # low 46 bits. Both fit, so the wrap is a no-op and variance is exact.
    accumulator = wrap(n * sq_sum - absolute_mean * absolute_mean, PRE_SUM_W)
    assert accumulator >= 0, f"{SENSORS[sensor]}: variance subtract negative"
    variance = trunc(accumulator, VARIANCE_W)
    slope = wrap((n - 1) * mean_for_slope - pre_for_slope, SLOPE_W)

    return sample, mean_sum, sq_sum, variance, slope


def tally_stats(sample, n):
    """median, iqr and mad from one complete tally over the sample alphabet.

    A fixed converter span bounds the alphabet at 2**SAMPLE_W, so a full tally is
    one streaming pass and exact -- the same structure the RTL bin-select uses.
    Ranks are n/4, n/2, n/2+n/4: every window length divides by 4, so they are
    pure shifts of window_length with no rounding case. mad comes off the same
    tally multiplier-free as sum_{j<m} cum(j) + sum_{j>=m} (n - cum(j)), which is
    exact only from a complete tally, so it costs no second pass.
    """
    lo = -(1 << (SAMPLE_W - 1))
    counts = np.bincount((sample - lo).astype(np.int64), minlength=1 << SAMPLE_W)
    cum = np.cumsum(counts)

    def rank(r):
        return lo + int(np.searchsorted(cum, r))

    p25, p50, p75 = rank(n >> 2), rank(n >> 1), rank((n >> 1) + (n >> 2))
    m = p50 - lo
    mad_n = int(cum[:m].sum() + (n - cum[m:]).sum())
    return p50, p75 - p25, mad_n


def time_stats_fixed(x, sensor, shift=None, offset=None):
    """The 10 stored time features for one window, in write_index order."""
    if shift is None:
        shift, offset = load_narrowing()
    n = WINDOW_LEN[sensor]
    sample, mean_sum, sq_sum, variance, slope = accumulators(x, sensor)

    xmin = int(sample.min())
    xmax = int(sample.max())
    rng = wrap(xmax - xmin, RANGE_W)          # 9 bits: the difference reaches 255
    median, iqr, mad_n = tally_stats(sample, n)

    # isqrt32 hands back a 16-bit root that rms_result and std_result narrow to
    # the feature word, and thermometer_encoder reads every feature as signed --
    # std, rms and range included, none of which the RTL clamps.
    std_root = _apply(variance, shift["std"][sensor], "sqrt")
    rms_root = _apply(sq_sum, shift["rms"][sensor], "sqrt_rms")

    return {
        "mean": narrow(mean_sum, "mean", sensor, shift, offset),
        "std": wrap(std_root - offset["std"][sensor], FEATURE_W),
        "rms": wrap(rms_root - offset["rms"][sensor], FEATURE_W),
        "min": wrap(xmin, FEATURE_W),
        "max": wrap(xmax, FEATURE_W),
        "range": narrow(rng, "range", sensor, shift, offset),
        "median": narrow(median, "median", sensor, shift, offset),
        "iqr": narrow(iqr, "iqr", sensor, shift, offset),
        "mad": narrow(mad_n, "mad", sensor, shift, offset),
        "slope": narrow(slope, "slope", sensor, shift, offset),
    }


# --- Fitting the narrowing tables -------------------------------------------

FEATURE_MAX = (1 << (FEATURE_W - 1)) - 1
FEATURE_MIN = -(1 << (FEATURE_W - 1))


def _apply(v, shift, kind):
    """Narrow one raw quantity exactly as time_stats_fixed does.

    The three kinds are not interchangeable: `std` truncates the shifted variance
    to the sqrt argument width, while `rms` truncates sq_sum *first* and shifts
    after, which is the order the RTL's `sqrt_data` mux presents them in. They
    agree while sq_sum stays under 2**32, and diverge silently the moment it does
    not -- hence one function rather than a shared "sqrt" branch.
    """
    if kind == "sqrt":
        return math.isqrt(trunc(v >> shift, SQRT_ARG_W))
    if kind == "sqrt_rms":
        return math.isqrt(trunc(v, SQRT_ARG_W) >> shift)
    return v >> shift


def _fit_one(values, kind):
    """Smallest shift whose spread fits the word, with the offset that centres it.

    Chosen on the spread rather than the range, which is the whole point of the
    offset: a feature whose windows all sit near a large constant needs a small
    shift and a large offset, not a shift big enough to bring the constant itself
    inside 8 bits.
    """
    for s in range(48):
        out = [_apply(v, s, kind) for v in values]
        off = (max(out) + min(out)) // 2
        if all(FEATURE_MIN <= o - off <= FEATURE_MAX for o in out):
            return s, off
    raise AssertionError("no shift narrows this feature into the word")


def fit_narrowing(codes, out_path=NARROWING_FILE):
    """Fit (shift, offset) per sensor and feature over the given windows.

    codes is {modality: (n_win, n_samp)} and must be TRAINING windows only --
    these constants are baked into the bitstream, so fitting them on the held-out
    subject is the same leak as fitting a scaler on test data.
    """
    keys = ("mean", "std", "rms", "range", "median", "iqr", "mad", "slope")
    shift = {k: [0] * len(SENSORS) for k in keys}
    offset = {k: [0] * len(SENSORS) for k in keys}

    for s, mod in enumerate(SENSORS):
        n = WINDOW_LEN[s]
        raw = {k: [] for k in keys}
        for c in codes[mod]:
            sample, mean_sum, sq_sum, variance, slope = accumulators(c, s)
            median, iqr, mad_n = tally_stats(sample, n)
            raw["mean"].append(mean_sum)
            raw["std"].append(variance)
            raw["rms"].append(sq_sum)
            raw["range"].append(wrap(int(sample.max()) - int(sample.min()), RANGE_W))
            raw["median"].append(median)
            raw["iqr"].append(iqr)
            raw["mad"].append(mad_n)
            raw["slope"].append(slope)
        for k in keys:
            kind = "sqrt_rms" if k == "rms" else "sqrt" if k == "std" else "shift"
            shift[k][s], offset[k][s] = _fit_one(raw[k], kind)

    if out_path is not None:
        Path(out_path).write_text(json.dumps({"shift": shift, "offset": offset}, indent=2) + "\n")
    return shift, offset


RAW_KEYS = ("mean", "std", "rms", "min", "max", "range", "median", "iqr", "mad", "slope")


def raw_columns(codes):
    """(n_win, 5*10) pre-narrow quantities, in the same column order as the cache.

    Saved beside the narrowed features so the narrowing tables can be refit per
    LOSO fold on training rows only -- they are bitstream constants, so fitting
    them across the held-out subject is the same leak as fitting a scaler on it.
    """
    n_win = codes[SENSORS[0]].shape[0]
    out = np.zeros((n_win, len(SENSORS) * len(RAW_KEYS)), dtype=np.int64)
    for s, mod in enumerate(SENSORS):
        n = WINDOW_LEN[s]
        base = s * len(RAW_KEYS)
        for w in range(n_win):
            sample, mean_sum, sq_sum, variance, slope = accumulators(codes[mod][w], s)
            median, iqr, mad_n = tally_stats(sample, n)
            xmin, xmax = int(sample.min()), int(sample.max())
            out[w, base:base + len(RAW_KEYS)] = (
                mean_sum, variance, sq_sum, xmin, xmax,
                wrap(xmax - xmin, RANGE_W), median, iqr, mad_n, slope)
    return out


def narrow_raw_columns(raw_tr, raw_te):
    """Fit narrowing on raw_tr, apply to both -> the 8-bit feature words.

    min and max are already sample-domain 8-bit values and reach the word
    unnarrowed, matching time_stats.sv's write_index 3 and 4.
    """
    nf = len(RAW_KEYS)
    out_tr = np.zeros_like(raw_tr)
    out_te = np.zeros_like(raw_te)
    for s in range(len(SENSORS)):
        for j, key in enumerate(RAW_KEYS):
            c = s * nf + j
            if key in ("min", "max"):
                out_tr[:, c] = raw_tr[:, c]
                out_te[:, c] = raw_te[:, c]
                continue
            kind = "sqrt_rms" if key == "rms" else "sqrt" if key == "std" else "shift"
            sh, off = _fit_one([int(v) for v in raw_tr[:, c]], kind)
            for src, dst in ((raw_tr, out_tr), (raw_te, out_te)):
                dst[:, c] = [wrap(_apply(int(v), sh, kind) - off, FEATURE_W)
                             for v in src[:, c]]
    return out_tr, out_te


# --- Cache building ---------------------------------------------------------

def quantise(win, adc=ADC):
    """{modality: converter codes} -- the sensor model, never a per-window AGC."""
    spans, bits = ADC_PROFILES[adc]
    return {m: adc_quantize(win[m], m, spans, bits) for m in SENSORS}


def features_for_window(codes, w, shift, offset):
    out = []
    for s, mod in enumerate(SENSORS):
        feats = time_stats_fixed(codes[mod][w], s, shift, offset)
        out.extend(feats[f] for f in FEATURES)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO / "data" / "wesad_cache" / "wesad_features_timeonly50_int.npz")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--adc", default=ADC, choices=sorted(ADC_PROFILES))
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--emit", type=Path, default=None)
    ap.add_argument("--fit", action="store_true",
                    help="refit the narrowing tables over every window and rewrite narrowing.json")
    args = ap.parse_args()

    npz = np.load(args.cache)
    codes = quantise(npz, args.adc)

    if args.fit:
        shift, offset = fit_narrowing(codes)
        print(f"wrote {NARROWING_FILE}")
        for k in shift:
            print(f"  {k:7s} shift {shift[k]}  offset {offset[k]}")
    else:
        shift, offset = load_narrowing()

    names = [f"{m}_t_{f}" for m in SENSORS for f in FEATURES]

    if args.emit is not None:
        w = args.window or 0
        row = features_for_window(codes, w, shift, offset)
        model = HWModel(EXPORT)
        model.thresholds = np.rint(model.thresholds)
        r = model.infer(np.array([row], dtype=float), return_all=True)
        bits, cls = r["bin"][0], int(r["preds"][0])
        # thermometer_encoder compares a FEATURE_W-bit feature against a
        # FEATURE_W-bit threshold. A threshold the word cannot hold is not a
        # threshold the hardware can apply, so say so rather than writing a
        # reference nothing can reproduce -- that is what left the testbench
        # failing against fixtures that looked freshly generated.
        low, high = FEATURE_MIN, FEATURE_MAX
        adrift = int(np.count_nonzero((model.thresholds < low) | (model.thresholds > high)))
        if adrift:
            print(f"WARNING: {adrift} of {model.thresholds.size} thresholds fall outside the "
                  f"{FEATURE_W}-bit feature word [{low}, {high}]. bits.hex and class.hex below "
                  f"are the {EXPORT.name} model's own answer, not one the comparator can reach; "
                  f"the export has to be refit on {FEATURE_W}-bit features first.", file=sys.stderr)
        args.emit.mkdir(parents=True, exist_ok=True)
        (args.emit / "feat.hex").write_text("".join(f"{v & 0xFF:02x}\n" for v in row))
        (args.emit / "bits.hex").write_text("".join(f"{b}\n" for b in bits))
        (args.emit / "class.hex").write_text(f"{cls:04x}\n")
        print(f"window {w}: subject S{npz['groups'][w]}, label {npz['y_multi'][w]}, "
              f"predicted {cls}, scores {r['scores'][0]}")
        print(f"  -> {args.emit}/feat.hex ({len(row)}), bits.hex, class.hex")
        return 0

    if args.window is not None:
        row = features_for_window(codes, args.window, shift, offset)
        print(f"window {args.window}: subject S{npz['groups'][args.window]}, "
              f"class {npz['y_multi'][args.window]}")
        for i, (nm, v) in enumerate(zip(names, row)):
            print(f"  {i:2d} {nm:14s} {v:7d}  {v & 0xFF:02x}")
        return 0

    n_win = int(npz["y_multi"].shape[0])
    X = np.zeros((n_win, len(names)), dtype=np.int64)
    for w in range(n_win):
        X[w] = features_for_window(codes, w, shift, offset)

    np.savez_compressed(args.out, X=X.astype(float), y_multi=npz["y_multi"],
                        groups=npz["groups"], feature_names=np.array(names))
    print(f"{n_win} windows x {len(names)} features -> {args.out}")
    print(f"range {X.min()} .. {X.max()}")
    for i, nm in enumerate(names):
        col = X[:, i]
        print(f"  {i:2d} {nm:14s} min {col.min():5d}  max {col.max():5d}  "
              f"distinct {len(np.unique(col)):4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
