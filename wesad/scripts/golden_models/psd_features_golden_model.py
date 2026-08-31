"""The 13 spectral features in integer arithmetic, from the BFP FFT's output.

fft_fixed_bfp produces (acc, pe) with true power = acc * 2**pe. Everything here
works on `acc` and folds `pe` in only where it matters, so no float appears
between the transform and the feature word.

Three things are shared across all 13 and are the only non-trivial units:

  * one sequential restoring DIVIDER -- centroid, entropy and bandwidth each
    normalise by `total`, a per-window sum that is neither constant nor a power
    of two, so it cannot be a shift or folded into a learned threshold;
  * one integer SQRT, already in the design for the time path;
  * one LOG2 = leading-zero count + a small mantissa ROM.

Constant scale factors are free because the thermometer learns a ladder per
column, so peak_freq/centroid/bandwidth/rolloff85 stay in BIN units rather than
Hz and entropy is in bits rather than nats.
"""

import math

import numpy as np

# --- D1: divider ------------------------------------------------------------
# Quotients carry QUOT_FRAC fractional bits. Measured spread of `centroid` is
# ~50-450 bins across modalities, so 1/256 of a bin resolves about 3-4 bits finer
# than the 8-bit feature word can hold once shift+offset narrowing is applied --
# the divide is deliberately not the limiting stage. A sequential restoring
# divider costs one cycle per quotient bit; at ~15 divides per window that is a
# few hundred cycles against the 12e6 in a 1 s hop, so the width is bought with
# area, not time. Widths are asserted against the data in feature_widths().
QUOT_FRAC = 8

# --- D2: log2 ---------------------------------------------------------------
# log2(v) = (bit_length-1) + log2(mantissa), the integer part being a
# leading-zero count and the fraction a ROM on the top LOG2_ROM_BITS bits after
# the leading 1. 32 entries leave ~0.022 bits of sawtooth against z=4 thresholds
# roughly 0.4 bits apart -- an 18x margin -- and 32 x 8 bits is 256 bits of
# distributed ROM rather than an EBR.
LOG2_ROM_BITS = 5
LOG2_FRAC = 8
_LOG2_ROM = [int(round(math.log2(1.0 + i / (1 << LOG2_ROM_BITS)) * (1 << LOG2_FRAC)))
             for i in range(1 << LOG2_ROM_BITS)]

# --- D3: which features are stored in the log domain -------------------------
# total/peak_power/band* span >6 orders of magnitude across the dataset, so a
# linear 8-bit code spends nearly all of its range on the top decade. In the log
# domain the code is uniform in dB, which is what a quantile thermometer wants,
# and it is nearly free: the datapath already holds acc * 2**pe, so
# log2 = pe + log2(acc) reuses the unit D2 already pays for.
LOG_DOMAIN = ("total", "peak_power", "band0", "band1", "band2",
              "band3", "band4", "band5")

PSD_KEYS = ("total", "peak_freq", "peak_power", "centroid", "bandwidth",
            "entropy", "rolloff85", "band0", "band1", "band2", "band3",
            "band4", "band5")


def log2_fixed(v):
    """log2 of a positive integer, Q<int>.LOG2_FRAC, via count + ROM."""
    if v <= 0:
        return None
    e = v.bit_length() - 1
    if e >= LOG2_ROM_BITS:
        m = (v >> (e - LOG2_ROM_BITS)) & ((1 << LOG2_ROM_BITS) - 1)
    else:
        m = (v << (LOG2_ROM_BITS - e)) & ((1 << LOG2_ROM_BITS) - 1)
    return (e << LOG2_FRAC) + _LOG2_ROM[m]


def divide(num, den, frac=QUOT_FRAC):
    """num/den with `frac` fractional bits, truncating -- one restoring divider."""
    if den <= 0:
        return 0
    return (int(num) << frac) // int(den)


def band_edges(fs, nfft, bands):
    """Bin index of each band boundary; precomputed offline, per modality."""
    return [(int(math.ceil(lo * nfft / fs)), int(math.ceil(hi * nfft / fs)))
            for lo, hi in bands]


def psd_features_fixed(acc, pe, edges, widths=None):
    """The 13 features for one window's spectrum, integer throughout.

    acc is the per-bin power mantissa, pe the shared exponent. `edges` comes from
    band_edges() for that modality. `widths`, if given, is a dict the function
    updates with the peak bit-width every accumulator actually reached, so the
    registers get sized by measurement rather than by worst case.
    """
    acc = [int(v) for v in acc]
    n = len(acc)

    # --- pass 1: one read -> total, peak, sum(k*p), sum(p*log2 p) -----------
    total = 0
    peak_power = -1
    peak_freq = 0
    num_centroid = 0
    sum_plogp = 0
    for k, p in enumerate(acc):
        total += p
        if p > peak_power:                     # strict: argmax keeps the first
            peak_power, peak_freq = p, k
        num_centroid += k * p
        if p > 0:                              # p == 0 contributes exactly 0
            sum_plogp += p * log2_fixed(p)

    if total <= 0:
        return {k: 0 for k in PSD_KEYS}

    centroid_q = divide(num_centroid, total)           # Q.QUOT_FRAC, bin units
    centroid_i = centroid_q >> QUOT_FRAC               # integer bin
    d = centroid_q - (centroid_i << QUOT_FRAC)         # fractional remainder

    # entropy = log2(T) - (sum p*log2 p)/T, in bits
    entropy = log2_fixed(total) - divide(sum_plogp, total, 0)

    # --- pass 2: centred second moment, cumsum for rolloff and the bands ----
    # Centring on an INTEGER bin keeps the loop exact; the fractional part comes
    # off afterwards as -d^2, which is the identity
    # sum((k-c)^2 p) = sum((k-c_i)^2 p) - d^2 * T.
    m2 = 0
    for k, p in enumerate(acc):
        dk = k - centroid_i
        m2 += dk * dk * p

    var_q = divide(m2, total) - ((d * d) >> QUOT_FRAC)
    bandwidth = math.isqrt(max(0, var_q) << QUOT_FRAC)  # Q.QUOT_FRAC bin units

    # rolloff85: first bin whose cumulative power reaches 85 % of total, as the
    # integer compare 20*cum >= 17*total -- shift-adds, no divide.
    cum = 0
    rolloff85 = n - 1
    target = 17 * total
    cumsum = [0] * n
    for k, p in enumerate(acc):
        cum += p
        cumsum[k] = cum
        if 20 * cum >= target:
            rolloff85 = k
            break
    if cum < total:                                    # finish the cumsum for bands
        for k in range(rolloff85 + 1, n):
            cum += acc[k]
            cumsum[k] = cum

    feats = {
        "total": total,
        "peak_freq": peak_freq,
        "peak_power": peak_power,
        "centroid": centroid_q,
        "bandwidth": bandwidth,
        "entropy": entropy,
        "rolloff85": rolloff85,
    }
    for i, (lo, hi) in enumerate(edges):
        lo = min(max(lo, 0), n)
        hi = min(max(hi, 0), n)
        feats[f"band{i}"] = (cumsum[hi - 1] if hi > 0 else 0) - (cumsum[lo - 1] if lo > 0 else 0)

    # Scale-dependent features carry the block exponent; the bin-domain ones
    # (peak_freq/centroid/bandwidth/rolloff85) and the normalised one (entropy)
    # are exponent-invariant by construction.
    out = {}
    for k, v in feats.items():
        if k in LOG_DOMAIN:
            lv = log2_fixed(v)
            # log2(acc * 2**pe) = pe + log2(acc); pe is already an integer power.
            out[k] = (int(pe) << LOG2_FRAC) + lv if lv is not None else 0
        else:
            out[k] = v

    if widths is not None:
        for name, v in (("total", total), ("sum_k_p", num_centroid),
                        ("sum_p_log_p", sum_plogp), ("m2", m2),
                        ("peak_power", peak_power), ("cum_x20", 20 * cum)):
            widths[name] = max(widths.get(name, 0), int(abs(v)).bit_length())
    return out
