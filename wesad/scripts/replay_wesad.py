#!/usr/bin/env python3
"""Replay contiguous WESAD signal through the FPGA over I2C, one class per hop second.

The design ingests in hop-sized blocks: 250 ECG, 32 ACC triples, 25 Resp, 25 EDA
and 350 EMG words, which is one second of every modality. Each completed block
advances the circular buffers by a second and fires one inference on the trailing
60 s window, so after the first 60 blocks prime the buffers every block yields a
class.

libusb needs the FTDI claimed, so run it as root with the environment's own
interpreter rather than a bare `sudo python3`:

    sudo "$(which python3)" wesad/scripts/replay_wesad.py --subject 2 --seconds 300

Source signal is reconstructed from the cached 60 s windows, which overlap by 30 s,
by taking the leading 30 s of each consecutive window.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from pyftdi.i2c import I2cController, I2cNackError

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "wesad" / "scripts" / "cache_builders"))
sys.path.insert(0, str(REPO / "wesad" / "scripts" / "golden_models"))

from fixed_frontend import ADC_PROFILES, acc_magnitude_fixed, adc_quantize  # noqa: E402

WINDOWS = REPO / "data" / "wesad_cache" / "wesad_windows_resampled6.npz"

HOP_ECG, HOP_ACC, HOP_RESP, HOP_EDA, HOP_EMG = 250, 32, 25, 25, 350
WIN_SECONDS = 60
OVERLAP_SECONDS = 30
CHUNK = 4096
LABEL_NAME = {0: "baseline", 1: "stress", 2: "amusement"}


def build_signal(subject, seconds, adc, start=0):
    d = np.load(WINDOWS)
    sel = np.where(d["groups"] == subject)[0]
    if len(sel) == 0:
        sys.exit(f"subject {subject} not in the cache")
    order = sel[np.argsort(d["starts"][sel])]

    need = -(-seconds // OVERLAP_SECONDS)
    if start + need > len(order):
        sys.exit(f"subject {subject} has {len(order)} windows, "
                 f"start {start} plus {seconds} s needs {start + need}")
    order = order[start : start + need]

    spans, bits = ADC_PROFILES[adc]
    rates = {"ECG": HOP_ECG, "Resp": HOP_RESP, "EDA": HOP_EDA, "EMG": HOP_EMG}
    keep = {m: r * OVERLAP_SECONDS for m, r in rates.items()}

    parts = {m: [] for m in rates}
    axes_parts = {c: [] for c in "xyz"}
    for w in order:
        for mod in rates:
            q = adc_quantize(d[mod][w : w + 1], mod, spans, bits)[0]
            parts[mod].append(q[: keep[mod]])
        for c in "xyz":
            q = adc_quantize(d[f"ACC_{c}"][w : w + 1], "ACC_axis", spans, bits)[0]
            axes_parts[c].append(q[: HOP_ACC * OVERLAP_SECONDS])

    sig = {m: np.concatenate(v) for m, v in parts.items()}
    axes = [np.concatenate(axes_parts[c]) for c in "xyz"]
    labels = np.repeat(d["y_multi"][order], OVERLAP_SECONDS)
    return sig, axes, labels


def hop_block(sig, axes, k):
    words = list(sig["ECG"][k * HOP_ECG : (k + 1) * HOP_ECG])
    for i in range(k * HOP_ACC, (k + 1) * HOP_ACC):
        words += [axes[0][i], axes[1][i], axes[2][i]]
    words += list(sig["Resp"][k * HOP_RESP : (k + 1) * HOP_RESP])
    words += list(sig["EDA"][k * HOP_EDA : (k + 1) * HOP_EDA])
    words += list(sig["EMG"][k * HOP_EMG : (k + 1) * HOP_EMG])

    out = bytearray()
    for w in words:
        out += (int(w) & 0xFFFF).to_bytes(2, "little")
    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ftdi://ftdi:2232h/2")
    ap.add_argument("--address", type=lambda v: int(v, 0), default=0x40)
    ap.add_argument("--freq", type=int, default=400_000)
    ap.add_argument("--settle", type=float, default=0.02)
    ap.add_argument("--subject", type=int, default=2)
    ap.add_argument("--seconds", type=int, default=300)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--adc", default="respiban")
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    sig, axes, labels = build_signal(args.subject, args.seconds, args.adc,
                                     args.start)
    total = min(args.seconds, len(labels))
    print(f"S{args.subject}: {total} s of signal, {WIN_SECONDS} s priming "
          f"then {max(0, total - WIN_SECONDS)} classified hops")

    i2c = I2cController()
    i2c.configure(args.url, frequency=args.freq)
    rows = []
    t0 = time.time()
    try:
        port = i2c.get_port(args.address)
        for k in range(total):
            block = hop_block(sig, axes, k)
            try:
                for off in range(0, len(block), CHUNK):
                    last = off + CHUNK >= len(block)
                    port.write(block[off:off + CHUNK],
                               start=(off == 0), relax=last)
            except I2cNackError:
                print(f"t={k:3d}s  FAIL  slave stopped acknowledging")
                return 1

            if k < WIN_SECONDS:
                continue

            time.sleep(args.settle)
            try:
                resp = port.read(4, start=True, relax=True)
            except I2cNackError:
                print(f"t={k:3d}s  FAIL  no response")
                return 1

            cls, s0, s1, s2 = resp
            truth = int(labels[k])
            rows.append((k, cls, truth, s0, s1, s2))
            mark = "ok " if cls == truth else "MISS"
            print(f"t={k:3d}s  class {cls} ({LABEL_NAME[cls]:9s}) "
                  f"truth {truth} ({LABEL_NAME[truth]:9s}) "
                  f"scores [{s0:2d} {s1:2d} {s2:2d}]  {mark}")
    finally:
        i2c.terminate()

    if not rows:
        print("no classified hops: raise --seconds above 60")
        return 1

    agree = sum(1 for _, c, t, *_ in rows if c == t)
    print(f"\n{agree}/{len(rows)} hops match the window label "
          f"({100 * agree / len(rows):.1f}%) in {time.time() - t0:.1f} s")

    if args.csv:
        args.csv.write_text("t,class,truth,s0,s1,s2\n" +
                            "".join(f"{a},{b},{c},{d},{e},{f}\n"
                                    for a, b, c, d, e, f in rows))
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
