#!/usr/bin/env python3
"""Load-phase vs inference-phase switching, with the boundary taken from the design.

A naive boundary at the testbench's "finished sending" timestamp is WRONG:
uart_rx raises valid at the middle of the last byte's stop bit and the FSM
reaches S_TX roughly 21 clocks later, so the inference is already finished by
the time the send loop returns. Placing the boundary there puts the whole
inference on the load side of the split and the comparison says nothing.

uart_tx_o gives an exact, observable marker instead. The first falling edge of
each response is the start bit of resp[0], which the FSM can only drive once
S_SETTLE -> S_GROUPSUM -> S_ARGMAX have completed. So the ~21 clocks immediately
before that edge ARE the inference, and everything earlier in the run is serial
load.

Reports toggles per block in the inference window and, for comparison, a
per-clock rate over a load window later in the same run.

Usage: vcd_phase.py <file.vcd> [inference_window_clocks]
"""
import re
import sys
from collections import defaultdict

CLK_PS = 83333            # 12 MHz period in ps
FRAME_PS = 1040 * CLK_PS  # one 8N1 byte at DIV=104

BUCKETS = [
    # any depth -- was layer1/layer2 literal, missed layer3+ on a >2-layer model
    ("layers",   re.compile(r"layer\d+_bits")),
    ("groupsum", re.compile(r"u_groupsum|(^|[/.])scores\[|gs_start|gs_done")),
    ("argmax",   re.compile(r"u_argmax|class_idx|best_lo|best_hi|idx_lo|idx_hi")),
    ("pixels",   re.compile(r"(^|[/.])pixels")),
    ("uart_rx",  re.compile(r"u_rx|uart_rx_i")),
    ("uart_tx",  re.compile(r"u_tx|uart_tx_o|tx_start|tx_data")),
    ("fsm_glue", re.compile(r"rx_timer|byte_cnt|resp|(^|[/.])state|settle")),
]


def bucket_of(path):
    for name, pat in BUCKETS:
        if pat.search(path):
            return name
    return "other"


def read_ids(path):
    ids, scope = {}, []
    var_re = re.compile(r"^\$var\s+\S+\s+\d+\s+(\S+)\s+(.+?)\s*\$end")
    with open(path, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("$scope"):
                scope.append(s.split()[2])
            elif s.startswith("$upscope"):
                if scope:
                    scope.pop()
            elif s.startswith("$var"):
                m = var_re.match(s)
                if m:
                    ids[m.group(1)] = "/".join(scope + [m.group(2)])
            elif s.startswith("$enddefinitions"):
                break
    return ids


def find_tx_negedges(path, tx_id):
    """Timestamps (ps) where uart_tx_o falls 1 -> 0."""
    edges, now, prev = [], 0, None
    with open(path, "r", errors="replace") as f:
        header = True
        for line in f:
            if header:
                if line.startswith("$enddefinitions"):
                    header = False
                continue
            if line[0] == "#":
                now = int(line[1:])
            elif line[0] in "01" and line[1:].strip() == tx_id:
                v = line[0]
                if prev == "1" and v == "0":
                    edges.append(now)
                prev = v
    return edges


def count_windows(path, ids, windows):
    """windows: list of (lo_ps, hi_ps, label). Returns {label: {bucket: n}}."""
    out = {lab: defaultdict(int) for _, _, lab in windows}
    now = 0
    with open(path, "r", errors="replace") as f:
        header = True
        for line in f:
            if header:
                if line.startswith("$enddefinitions"):
                    header = False
                continue
            c = line[0]
            if c == "#":
                now = int(line[1:])
                continue
            if c in "01xXzZ":
                vid = line[1:].strip()
            elif c in "bB":
                p = line.split()
                vid = p[1] if len(p) > 1 else None
            else:
                continue
            if vid is None:
                continue
            for lo, hi, lab in windows:
                if lo <= now < hi:
                    out[lab][bucket_of(ids.get(vid, "other"))] += 1
    return out


def main():
    path = sys.argv[1]
    win_clocks = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    ids = read_ids(path)
    tx_id = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("$enddefinitions"):
                break
            m = re.match(r"^\$var\s+\S+\s+1\s+(\S+)\s+uart_tx_o\s+\$end", line.strip())
            if m:
                tx_id = m.group(1)
                break
    if tx_id is None:
        sys.exit("could not find uart_tx_o in VCD")

    edges = find_tx_negedges(path, tx_id)
    # First negedge of each response: any edge more than 2 frame times after the
    # previous one starts a new response burst.
    starts = [e for k, e in enumerate(edges) if k == 0 or e - edges[k - 1] > 2 * FRAME_PS]
    print(f"=== {path} ===")
    print(f"tx falling edges: {len(edges)}, response bursts: {len(starts)}")

    windows = []
    for k, t in enumerate(starts):
        windows.append((t - win_clocks * CLK_PS, t, f"inf{k}"))
    # a load window of 20k clocks well inside the last vector's RX phase
    if len(starts) >= 2:
        mid = starts[-1] - 150000 * CLK_PS
        windows.append((mid, mid + 20000 * CLK_PS, "load20k"))

    res = count_windows(path, ids, windows)

    inf = defaultdict(int)
    ninf = 0
    for lab, d in res.items():
        if lab.startswith("inf"):
            ninf += 1
            for k, v in d.items():
                inf[k] += v

    print(f"\ninference windows ({win_clocks} clocks each, n={ninf}) -- toggles per inference:")
    tot = sum(inf.values())
    for b, n in sorted(inf.items(), key=lambda kv: -kv[1]):
        print(f"  {b:10s} {n/ninf:10.1f}")
    print(f"  {'TOTAL':10s} {tot/ninf:10.1f}")

    if "load20k" in res:
        ld = res["load20k"]
        lt = sum(ld.values())
        print(f"\nload window (20,000 clocks) -- toggles per clock:")
        for b, n in sorted(ld.items(), key=lambda kv: -kv[1]):
            print(f"  {b:10s} {n/20000.0:10.4f}")
        print(f"  {'TOTAL':10s} {lt/20000.0:10.4f}")
        print(f"\ninference window rate / load rate: "
              f"{(tot/ninf/win_clocks) / (lt/20000.0):.1f}x")


if __name__ == "__main__":
    main()
