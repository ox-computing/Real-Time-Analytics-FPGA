# Demonstrates that the Power Calculator's Logic power is exactly linear in
# AF% -- the basis for reading a calibration coefficient (mW per 1% AF) off two
# or three runs instead of trusting a single vectorless default. Run against
# pop.pcf (vectorless, clock pin not yet TR-corrected -- see calculate.tcl):
#
#   radiantc.exe af_sweep.tcl
#
# Expect Logic Block Total Dynamic Power to come out ~0.5478 mW per 1% AF at
# 12 MHz, i.e. af1 ~0.548 mW, af10 ~5.478 mW, af20 ~10.956 mW. That
# coefficient, not any single AF guess, is what turns a measured activity
# factor into a power number without needing Radiant to ingest a VCD at all.

pwc_open_project pop.pcf
pwc_set_freq clk_c 12

foreach af {1 10 20} {
    pwc_set_af $af -keepClocksAF
    pwc_calculate
    pwc_gen_report af${af}.txt
    puts "af=$af -> af${af}.txt"
}
