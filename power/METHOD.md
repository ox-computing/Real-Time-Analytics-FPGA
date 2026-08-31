## What a power tool needs (Power Calculator on Radiant, the Lattice FPGA Tool)

Dynamic power in CMOS is `P = alpha * C * V^2 * f`. `C` and `V` are fixed by
the silicon, `f` by the clock; the only unknown is `alpha` — how often each
net actually switches. The power tool needs two inputs: what is on the
chip (LUT/register/routing counts, from place and route), and how much it
switches (activity). Every step below exists to produce it.

## Step 1 — get netlist

Radiant exports a post-place-and-route netlist,
`EUROP_MNIST_UP5K_impl_1_vo.vo` — 2,400 modules of `SLICE_nnn` instances wired to 
`LUT4`, `FA2`, `FD1P3XZ` primitives.

## Step 2 — simulate it at gate level

```
iverilog -g2005 -o tbg -I $LIB tb_gate_power.v $NETLIST -y $LIB -Y .v
```

`$LIB` is Lattice's `cae_library/simulation/verilog/iCE40UP` under the Radiant
install.

`tb_gate_power.v` drives MNIST vectors through the UART protocol
at the synthesised divider (104 clocks/bit), because the post-PAR top only
exposes `clk`, `uart_rx_i`, `uart_tx_o` 

**How the simulation's correctness was checked**: it reproduced the golden
PyTorch model exactly, predicted class and all ten GroupSum scores, byte for
byte.

## Step 3 — count every transition

`$dumpvars` on the design writes a VCD, a timestamped log of every net's every
state change. `vcd_toggles.py` parses it and buckets transitions by RTL module.
This works because the post-PAR netlist keeps the original signal names —
`layer1_bits[150]`, `u_groupsum/scores`, `u_rx/cnt` are all still visible in
the VCD.

The clock net is segregated and not folded into a block. It toggles tens of
times more often than everything else combined.

## Step 4 — convert transitions to an activity factor

```
AF% = transitions / (nets * cycles) * 100
```

`vcd_af.py` computes this over the **3,843 top-level nets** directly inside
`dwn_uart_top` , chosen because Radiant's power model counts 2,413 LUTs +
1,375 registers = 3,788 elements, a close match. Measured result: **AF = 0.381%**. 
Radiant's own default assumption is 10% —
the design is about 26x quieter than that default.

## Step 5 — turn Radiant into a calibrated coefficient

Radiant's Power Calculator has a Tcl API.

```tcl
pwc_open_project pop.pcf
pwc_set_freq clk_c 12
pwc_calculate
pwc_gen_report out.txt
```

To check linearity, it was run at AF = 1%, 10%, 20%
(`radiant/af_sweep.tcl`) and compared. The response is exactly linear:

```
AF  1% -> Logic 0.548 mW
AF 10% -> Logic 5.478 mW
AF 20% -> Logic 10.956 mW
```

`P_logic(mW) = 0.5478 * AF%`

`measured.pcf` was then produced by writing the measured 0.381% AF into the
`.pcf`'s Logic row directly, it is plain text, `Freq, AF` are the first two
fields of every resource row.

## Step 6 — isolate the DWN itself with a differential measurement

Total power describes a UART receiver, because that is what the chip spends
99.99% of its time doing. To see the DWN specifically, two gate-level
simulations were run differing **only** in whether the datapath's input
actually changed between loads:

- **cycle**: `v0,v1,v2,v0,v1,v2` — every load is new data, 6 inferences
- **paired**: `v0,v0,v1,v1,v2,v2` — alternate loads repeat, 3 inferences

Both arms send the **same multiset of bytes**, so UART line-transition density
is identical by construction, and in the repeated loads the `pixels` register
is rewritten with values it already holds — the write-enable fires but
`D == Q`, so no flop actually toggles, and the layers recompute identical
results and go quiet.


The clock and `uart_rx` differences between the
two arms are **+0**, exactly, and `uart_tx`/`fsm_glue` differ by a few dozen
out of millions of toggles.

The residual — 8,770 transitions over 3 extra inferences — converts through
the calibration coefficient 4.1 nJ of marginal compute energy
per inference.

## Step 7 — locate the switching in time

Splitting "loading bytes" from "computing" needs the exact moment of
inference. The **first falling edge of
`uart_tx_o`**. The FSM can only drive that pin once
`S_SETTLE -> S_GROUPSUM -> S_ARGMAX` have all completed, so a window ending
there brackets the inference exactly (`vcd_phase.py`).
That is what surfaced groupsum's 23,347 toggles/inference against argmax's
881.


## What each number rests on

| number | source | confidence |
|---|---|---|
| Toggle counts | direct measurement, gate-level sim | exact |
| AF 0.381% | arithmetic on those counts | exact, given the net-population choice |
| 3.456 mW | Radiant's model at measured + TR-corrected AF | vendor-model-limited |
| Per-block split | toggle share x logic power | assumes uniform capacitance per net |
| ~3.5-4.1 nJ/inference | differential x coefficient | marginal energy, not total-attributable |
