# Power and energy results — iCE40UP5K-SG48I

Model 200/400 @ tau1.6, post-P&R via Radiant/LSE, 12 MHz, 25 C ambient, Typical
process. 

---

## 1. Headline numbers

| quantity | value | basis |
|---|---|---|
| **Total power** | **3.456 mW** | measured AF 0.381%, 12 MHz, TR-corrected clock pin |
| — static | 0.410 mW | Radiant, device + temperature |
| — dynamic | 3.046 mW | |
| **System energy / inference** | **91.9 uJ** | 3.456 mW x 26.6 ms round trip |
| **Marginal compute energy / inference** | **4.1 nJ** | differential measurement, logic only |
| **DWN datapath power** | **0.295 uW** | 0.011% of chip power |
| Latency / inference | 26.6 ms | UART-bound; compute itself is ~1.75 us |
| Throughput | ~38 inferences/s | protocol-limited, not compute-limited |


## 2. Power distribution

| domain | power | share |
|---|---|---|
| I/O | 1422.0 uW | 41.2% |
| Clock tree | 1381.0 uW | 40.0% |
| Static leakage | 410.0 uW | 11.9% |
| Logic (all 3788 LUTs + registers) | 244.0 uW | 7.1% |
| **Total** | **3457 uW** | |


### Logic subdivided (by measured VCD toggle share)

| block | share of logic toggles | power |
|---|---|---|
| uart_rx | 50.043% | 122.104 uW |
| other (unclassified top-level nets) | 34.647% | 84.538 uW |
| fsm_glue | 13.113% | 31.995 uW |
| uart_tx | 2.077% | 5.067 uW |
| **groupsum** | 0.106% | **0.259 uW** |
| pixels | 0.006% | 0.015 uW |
| layers (layer1+layer2) | 0.004% | 0.011 uW |
| argmax | 0.004% | 0.011 uW |
| **DWN datapath total** | **0.121%** | **0.295 uW** |


## 3. Energy distribution per inference (91.9 uJ total)

| domain | energy |
|---|---|
| I/O (TR-corrected) | 37.8 uJ |
| Clock tree | 36.7 uJ |
| Static | 10.9 uJ |
| Logic | 6.5 uJ |
| — of which DWN datapath | 7.8 nJ |

Energy is dominated by holding the chip powered for 26.6 ms while 294 bytes
crawl in at 115200 baud. Compute is 0.007% of the wall time.

## 4. The differential measurement (A - B)

Validation that the two arms are otherwise comparable:

| block | A - B | expected |
|---|---|---|
| clk | **+0** | 0 |
| uart_rx | **+0** | 0 |
| uart_tx | +16 | ~0 |
| fsm_glue | +84 | ~0 |
| groupsum | +3,648 | positive |
| argmax | +572 | positive |
| layers (layer1+layer2) | +530 | positive |
| pixels | +624 | positive |

## 5. What each block costs during the inference window

From a 200-clock window ending at the `uart_tx_o` falling edge — the FSM can
only drive that pin once `S_SETTLE -> S_GROUPSUM -> S_ARGMAX` have completed,
so this window brackets the inference exactly, with no timestamp guessing:

| block | toggles/inference |
|---|---|
| **groupsum** | **23,347** |
| other | 4,263 |
| uart_rx | 3,766 |
| uart_tx | 1,328 |
| fsm_glue | 1,049 |
| argmax | 881 |
| pixels | 3.3 |
| **TOTAL** | **34,637** |

**groupsum dominates inference-phase switching by 27x over argmax.** This is
the time-multiplexed popcount: each of the 10 class clocks re-ripples a 40-bit
adder chain through a fresh mux selection, glitching through the carry chain
every time. Time-multiplexing groupsum was chosen to save area (~666 LC); this
is what it cost in switching energy — worth weighing if groupsum is ever
revisited.

Load-phase rates, for contrast (toggles/clock): uart_rx 24.76, other 19.47,
fsm_glue 6.00, groupsum 0.064, layer1 0.005, layer2 0.001.


## 6. Methodology and its limits

**Solid:**
- Netlist is post-P&R, timing-closed, bit-exact against the reference model.
- Toggle counts are exact, from gate-level simulation with real MNIST vectors.
- The differential's controls (clk +0, uart_rx +0) confirm the two arms are
  comparable.
- Radiant's power model reproduces the GUI's own number exactly
  (6.307 mW headless vs 6.31 mW in the GUI).

**Modelled, not measured:**
- Radiant's power model is flagged "Preliminary" for iCE40UP.
- AF averaged over the 3,843 top-level VCD nets as a proxy for Radiant's 3,788
  LUT + register elements. 
- Per-block power assumes power is proportional to toggle count within Logic,
  i.e. uniform capacitance per net. Real per-net capacitance varies with
  fanout and routing; groupsum's carry chain is likely above average, so its
  true share is probably understated here.
- The 4.1 nJ figure is **marginal** energy — the datapath's clock-tree
  share cancels in the subtraction by design. It is not a total-attributable
  energy-per-inference figure.
