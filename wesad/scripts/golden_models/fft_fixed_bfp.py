"""Integer FFT to PSD power on a fixed-width datapath with block floating point."""

import numpy as np

TW_BITS = 7                      # twiddle table -> Q1.7, matches RTL W_WIDTH = 8
HANN_BITS = 7                    # Hann coefficient fractional bits, same word
SHIFT_NUM, SHIFT_DEN = 5, 2      # shift when (SHIFT_NUM/SHIFT_DEN)*M > lim
ACC_BITS = 16                    # K-segment power accumulator, matches RTL ACC_WIDTH

_BITREV, _TW, _HANN = {}, {}, {}


def _bitrev(n):
    if n not in _BITREV:
        b = n.bit_length() - 1
        _BITREV[n] = np.array([int(f"{i:0{b}b}"[::-1], 2) for i in range(n)])
    return _BITREV[n]


def _twiddles(n, bits):
    if (n, bits) not in _TW:
        t = np.arange(n // 2)
        s = 1 << bits
        hi = s - 1
        _TW[(n, bits)] = (
            np.clip(np.round(np.cos(2 * np.pi * t / n) * s), -hi - 1, hi).astype(np.int64),
            np.clip(np.round(np.sin(2 * np.pi * t / n) * s), -hi - 1, hi).astype(np.int64))
    return _TW[(n, bits)]


def _hann(length, bits):
    if (length, bits) not in _HANN:
        _HANN[(length, bits)] = np.round(
            np.hanning(length) * ((1 << bits) - 1)).astype(np.int64)
    return _HANN[(length, bits)]


def rshift(v, sh):
    """Round-half-up arithmetic right shift. Negative sh shifts left.

    sh is either a scalar or one value per row of v, which is what a per-segment
    block exponent looks like when the whole dataset is run as a batch.
    """
    if np.isscalar(sh):
        if sh == 0:
            return v
        return (v + (1 << (sh - 1))) >> sh if sh > 0 else v << (-sh)
    sh = np.asarray(sh, dtype=np.int64).reshape(-1, 1)
    out = v.copy()
    pos, neg = sh.ravel() > 0, sh.ravel() < 0
    if pos.any():
        s = sh[pos]
        out[pos] = (v[pos] + (np.int64(1) << (s - 1))) >> s
    if neg.any():
        out[neg] = v[neg] << (-sh[neg])
    return out


def signed_bits(v):
    """Bits a signed two's-complement word needs to hold magnitude |v|."""
    v = int(abs(v))
    return v.bit_length() + 1 if v else 1


def row_bits(a):
    """signed_bits of each row's peak magnitude, as an (n_rows, 1) column."""
    return np.array([[signed_bits(v)] for v in np.abs(a).max(axis=1).ravel()],
                    dtype=np.int64)


def normalise(x, width):
    """Shift each row so its peak fills `width` signed bits. -> (x, exponent).

    Sheds LSBs when the row is wider than the datapath and normalises up when it
    is narrower, which is what recovers the precision a narrow converter would
    otherwise lose -- the butterfly's rounding error is half an LSB per stage
    regardless of signal size, so an under-filled datapath carries proportionally
    more of it.

    Rounding up at the boundary can carry the peak to +2**(width-1), which the
    signed datapath cannot hold, so the result is clamped the way the register
    that receives it is.
    """
    sh = row_bits(x) - width
    lim = (1 << (width - 1)) - 1
    return np.clip(rshift(x, sh), -lim - 1, lim), sh.ravel()


def fft_bfp(re, im, nfft, width, tw_bits=TW_BITS, count_overflow=False):
    """Radix-2 DIT FFT on a `width`-bit datapath, block floating point.

    re/im are (batch, nfft) int64 and must already be in bit-reversed order at
    `width` bits. Returns re, im, the per-row shift count, and (optionally) how
    many stored values left signed `width`-bit range -- which must be 0.
    """
    b = re.shape[0]
    cos_t, sin_t = _twiddles(nfft, tw_bits)
    rnd = 1 << (tw_bits - 1)
    lim = (1 << (width - 1)) - 1
    e = np.zeros((b, 1), dtype=np.int64)
    m = np.abs(re).max(axis=1, keepdims=True)
    over = 0
    half = 1
    while half < nfft:
        stride = nfft // (2 * half)
        cols = np.arange(half) * stride
        wc, ws = cos_t[cols], -sin_t[cols]
        r = re.reshape(b, -1, 2 * half)
        i = im.reshape(b, -1, 2 * half)
        # one rounding on the combined product-difference, not one per product
        tr = (wc * r[:, :, half:] - ws * i[:, :, half:] + rnd) >> tw_bits
        ti = (wc * i[:, :, half:] + ws * r[:, :, half:] + rnd) >> tw_bits
        top_r, top_i = r[:, :, :half].copy(), i[:, :, :half].copy()
        r[:, :, :half], i[:, :, :half] = top_r + tr, top_i + ti
        r[:, :, half:], i[:, :, half:] = top_r - tr, top_i - ti
        re, im = r.reshape(b, nfft), i.reshape(b, nfft)
        sh = (SHIFT_NUM * m > SHIFT_DEN * lim).astype(np.int64)
        re, im = rshift(re, sh), rshift(im, sh)
        re, im = np.clip(re, -lim - 1, lim), np.clip(im, -lim - 1, lim)
        e = e + sh
        if count_overflow:
            over += int(((re > lim) | (re < -lim - 1)).sum()
                        + ((im > lim) | (im < -lim - 1)).sum())
        m = np.maximum(np.abs(re).max(axis=1, keepdims=True),
                       np.abs(im).max(axis=1, keepdims=True))
        half *= 2
    return re, im, e.ravel(), over


def segments(n, seg_len):
    """(start, stop) pairs at 50% overlap; a window shorter than the engine is K=1."""
    if seg_len is None or seg_len >= n:
        return [(0, n)]
    step = max(seg_len // 2, 1)
    return [(s, s + seg_len) for s in range(0, n - seg_len + 1, step)]


def accumulate(acc, pe, p, pe_seg, cap=ACC_BITS, cap_mode="sat"):
    """Sum segment power under one shared exponent: true power = acc * 2^pe.

    cap_mode picks what happens when a bin outgrows `cap`. "renorm" rescales the
    whole spectrum and bumps the exponent, keeping every bin's value relative to
    the others. "sat" is what the RTL writeback does instead -- clip that bin at
    the register's maximum and leave the exponent alone, which loses the largest
    bin (i.e. peak_power/peak_freq) rather than shrinking everything together.
    """
    if acc is None:
        acc, pe = p.copy(), pe_seg.copy()
    else:
        up = np.maximum(pe, pe_seg)
        acc = rshift(acc, up - pe) + rshift(p, up - pe_seg)
        pe = up
    if cap is None:
        return acc, pe
    if cap_mode == "sat":
        return np.minimum(acc, (1 << cap) - 1), pe
    over = np.maximum(np.array([[int(v).bit_length()] for v in acc.max(axis=1)],
                               dtype=np.int64) - cap, 0)
    if over.any():
        acc = rshift(acc, over)
        pe = pe + over.ravel()
    return acc, pe


def psd_power_bfp(codes, seg_len, nfft, width, detrend=True, hann_bits=HANN_BITS,
                  acc_bits=ACC_BITS, tw_bits=TW_BITS, count_overflow=False,
                  cap_mode="sat"):
    """One-sided integer power spectrum for a batch of windows.

    codes is (batch, n) converter codes. Mirrors the dt2048k13 arm: per-segment
    detrend, Hann, 50% overlap, power accumulated over the segments. The sample
    is narrowed to `width` BEFORE the window multiply, so preprocessing is a
    width x 16 multiply like the butterfly rather than a wider one.

    -> (acc, pe, overflow_count) with true power = acc * 2^pe, acc (batch, nfft/2+1).
    """
    codes = np.asarray(codes, dtype=np.int64)
    n = codes.shape[1]
    segs = segments(n, seg_len)
    hlen = segs[0][1] - segs[0][0]
    hann = _hann(hlen, hann_bits)
    idx = _bitrev(nfft)
    acc, pe, over = None, None, 0
    for a, b in segs:
        seg = codes[:, a:b]
        if detrend:
            seg = seg - np.rint(seg.mean(axis=1)).astype(np.int64)[:, None]
        x, e0 = normalise(seg, width)
        xw = (x * hann + (1 << (hann_bits - 1))) >> hann_bits
        buf = np.zeros((codes.shape[0], nfft), dtype=np.int64)
        buf[:, :xw.shape[1]] = xw
        re = np.ascontiguousarray(buf[:, idx])
        re, im, e, o = fft_bfp(re, np.zeros_like(re), nfft, width,
                               tw_bits=tw_bits, count_overflow=count_overflow)
        over += o
        p = re[:, :nfft // 2 + 1] ** 2 + im[:, :nfft // 2 + 1] ** 2
        acc, pe = accumulate(acc, pe, p, 2 * (e + e0), acc_bits, cap_mode)
    return acc, pe, over


def psd_power_float(acc, pe):
    """(mantissa, exponent) -> float power, for the feature formulas."""
    return acc.astype(np.float64) * np.exp2(pe.astype(np.float64))[:, None]
