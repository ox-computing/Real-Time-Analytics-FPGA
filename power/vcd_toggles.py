#!/usr/bin/env python3
"""Per-block toggle counts from a gate-level VCD of dwn_uart_top.

Radiant's Power Calculator reports by resource type (Logic Block / Clock / IO),
not by design hierarchy, so it cannot say how much of "Logic Block" is the DWN
and how much is the UART. The post-PAR netlist keeps the RTL register names on
its flop instances (pixels_i1806, u_core/layer1_bits_i12, u_rx/cnt_i0, ...), so
the split can be recovered from the VCD directly and applied to Radiant's total.

The clock net is reported separately and never folded into a block: it toggles
tens of times more often than all compute combined, and bucketing it by owning
scope smears that single net across whatever block holds the port, producing a
bogus "the datapath is most of the switching" answer.

Usage: vcd_toggles.py <file.vcd> [file2.vcd]
Two files prints their difference as well -- the differential measurement.
"""
import re
import sys
from collections import defaultdict

# Ordered: first match wins, so more specific patterns come first.
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


def parse(path):
    ids = {}            # vcd id -> full hierarchical path
    scope = []
    toggles = defaultdict(int)
    clk_toggles = 0
    clk_ids = set()
    in_header = True

    var_re = re.compile(r"^\$var\s+\S+\s+\d+\s+(\S+)\s+(.+?)\s*\$end")

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
                    m = var_re.match(s)
                    if m:
                        vid, name = m.group(1), m.group(2)
                        full = "/".join(scope + [name])
                        ids[vid] = full
                        # the clock net, wherever it surfaces
                        if re.search(r"(^|/)(clk|CLK|CLK_dly)(\[|$)", full):
                            clk_ids.add(vid)
                elif s.startswith("$enddefinitions"):
                    in_header = False
                continue

            # value-change section: scalar "0!" / "1!" / "x!" , or "b1010 !"
            c = line[0]
            if c in "01xXzZ":
                vid = line[1:].strip()
            elif c == "b" or c == "B":
                parts = line.split()
                vid = parts[1] if len(parts) > 1 else None
            else:
                continue                      # #timestamp, $dumpvars, etc.
            if vid is None:
                continue
            if vid in clk_ids:
                clk_toggles += 1
            else:
                toggles[vid] += 1

    per_bucket = defaultdict(int)
    for vid, n in toggles.items():
        per_bucket[bucket_of(ids.get(vid, "other"))] += n
    return per_bucket, clk_toggles, len(ids)


def main():
    results = {}
    for path in sys.argv[1:]:
        per_bucket, clk, nvars = parse(path)
        results[path] = (per_bucket, clk, nvars)
        total = sum(per_bucket.values())
        print(f"\n=== {path} ===")
        print(f"vars={nvars}  clk-net toggles={clk:,}  non-clk toggles={total:,}")
        for name, n in sorted(per_bucket.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * n / total if total else 0.0
            print(f"  {name:10s} {n:12,}  {pct:5.1f}%")

    if len(results) == 2:
        (pa, ca, _), (pb, cb, _) = results.values()
        na, nb = list(results)
        print(f"\n=== DIFFERENCE  {na} - {nb} ===")
        keys = set(pa) | set(pb)
        ta, tb = sum(pa.values()), sum(pb.values())
        for name in sorted(keys, key=lambda k: -(pa.get(k, 0) - pb.get(k, 0))):
            d = pa.get(name, 0) - pb.get(name, 0)
            print(f"  {name:10s} {d:+12,}")
        print(f"  {'TOTAL':10s} {ta - tb:+12,}   (clk {ca - cb:+,})")


if __name__ == "__main__":
    main()
