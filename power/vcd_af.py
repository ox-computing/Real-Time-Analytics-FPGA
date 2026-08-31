#!/usr/bin/env python3
"""Activity factor over the net population Radiant's power model actually counts.

Radiant's Logic AF applies to LUT and register counts (2413 + 1375 = 3788 for
the 200/400 build). The VCD's TOP-LEVEL nets -- those directly inside
dwn_uart_top, carrying RTL names like layer1_bits[150] -- number 3843, which is
the right population to average over. The tens of thousands of deeper nets are
SLICE-internal pins (D0/DI1/CE/CLK_dly/F0...) that the power model does not
enumerate separately; including them would divide the same switching over ~17x
as many nets and understate AF by the same factor.

AF convention, confirmed against Lattice's own documentation (the Power
Calculator's AF help text): Toggle Rate(MHz) = 1/2 * f * AF%, so AF=100% means
one transition per clock PERIOD -- which is exactly what this script computes:
    AF% = transitions / (n_nets * n_cycles) * 100
This holds for internal logic. It does NOT hold for a pin driven directly by an
external clock, which transitions on both edges (twice a period) -- see
radiant/calculate.tcl and RESULTS.md for how that case is handled.

Usage: vcd_af.py <cycles> <file.vcd> [file2.vcd]
"""
import re
import sys
from collections import defaultdict

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
CLK_RE = re.compile(r"^(clk|clk_c|clk_c_.*)$")


def bucket_of(name):
    for b, pat in BUCKETS:
        if pat.search(name):
            return b
    return "other"


def parse(path):
    """Return (toggles_by_bucket, n_toplevel_nets, clk_toggles)."""
    var_re = re.compile(r"^\$var\s+\S+\s+\d+\s+(\S+)\s+(.+?)\s*\$end")
    scope = []
    top_ids = {}          # id -> bare net name, only for depth-2 scope
    clk_ids = set()
    in_header = True
    toggles = defaultdict(int)

    with open(path, "r", errors="replace") as f:
        for line in f:
            if in_header:
                s = line.strip()
                if s.startswith("$scope"):
                    scope.append(s.split()[2])
                elif s.startswith("$upscope"):
                    if scope:
                        scope.pop()
                elif s.startswith("$var"):
                    # depth 2 == tb_gate_power / dwn_uart_top
                    if len(scope) == 2:
                        m = var_re.match(s)
                        if m:
                            vid = m.group(1)
                            nm = m.group(2).lstrip("\\")
                            base = re.sub(r"\[\d+\]$", "", nm)
                            if CLK_RE.match(base):
                                clk_ids.add(vid)
                            else:
                                top_ids[vid] = nm
                elif s.startswith("$enddefinitions"):
                    in_header = False
                continue

            c = line[0]
            if c in "01xXzZ":
                vid = line[1:].strip()
            elif c in "bB":
                p = line.split()
                vid = p[1] if len(p) > 1 else None
            else:
                continue
            if vid is not None:
                toggles[vid] += 1

    by_bucket = defaultdict(int)
    for vid, n in toggles.items():
        if vid in top_ids:
            by_bucket[bucket_of(top_ids[vid])] += n
    clk = sum(toggles[v] for v in clk_ids if v in toggles)
    return by_bucket, len(top_ids), clk


def main():
    cycles = int(sys.argv[1])
    res = {}
    for path in sys.argv[2:]:
        bb, nnets, clk = parse(path)
        res[path] = (bb, nnets, clk)
        tot = sum(bb.values())
        af = 100.0 * tot / (nnets * cycles)
        print(f"\n=== {path} ===")
        print(f"top-level nets: {nnets}   cycles: {cycles:,}   clk toggles: {clk:,}")
        print(f"total logic transitions: {tot:,}")
        print(f"MEAN ACTIVITY FACTOR: {af:.4f} %")
        print(f"{'block':10s} {'toggles':>14s} {'share':>8s} {'AF%':>10s}")
        for b, n in sorted(bb.items(), key=lambda kv: -kv[1]):
            print(f"  {b:10s} {n:14,} {100.0*n/tot:7.2f}% {100.0*n/(nnets*cycles):9.5f}")

    if len(res) == 2:
        (a, na, _), (b, nb, _) = res.values()
        ka, kb = list(res)
        ta, tb = sum(a.values()), sum(b.values())
        print(f"\n=== DIFFERENCE {ka} - {kb} ===")
        for k in sorted(set(a) | set(b), key=lambda k: -(a.get(k, 0) - b.get(k, 0))):
            print(f"  {k:10s} {a.get(k,0)-b.get(k,0):+12,}")
        print(f"  {'TOTAL':10s} {ta-tb:+12,}")
        print(f"  delta AF: {100.0*(ta-tb)/(na*cycles):+.5f} %")


if __name__ == "__main__":
    main()
