# Runs the Power Calculator headlessly against a .pcf project and writes a
# text + CSV report next to it. Works on any .pcf in this directory:
#
#   set target_pcf pop.pcf         ;# vectorless, default AF=10% -- 6.307 mW
#   set target_pcf measured.pcf    ;# measured activity + TR-corrected clock pin
#   radiantc.exe calculate.tcl
#
# pop.pcf is produced by populate.tcl from the post-PAR .udb; measured.pcf
# starts from pop.pcf with the Logic AF row set to the value vcd_af.py measured
# (0.381% on the 200/400 build) and the clock INPUT PIN's frequency field
# doubled (12 -> 24 MHz) rather than its AF raised past 100%.
#
# That doubling is not cosmetic. Per Lattice's own AF documentation,
# Toggle Rate(MHz) = 1/2 * f * AF%, so AF=100% at the design frequency means one
# transition per clock PERIOD -- correct for ordinary internal logic, which is
# what vcd_af.py measures. A clock INPUT PIN driven from outside the chip
# transitions on both edges, i.e. twice a period, so representing its true
# toggle rate at AF=100% requires DOUBLING THE FREQUENCY field instead. The
# Power Calculator's AF field clamps silently at 100 (entering 200 was written
# back to the default 10 in the CSV with no warning), so there is no direct way
# to enter AF=200; the frequency trick is the only way found to express it.
# Skipping this understates the clock pin's I/O contribution by 2x -- it is
# not a rounding difference, it moved the total power by roughly 700 uW here.
#
# pwc_set_freq/-clk and pwc_set_af/-iptype/-clk apply to a clock NAME across
# every section of the .pcf at once (Logic, Clocks, Input Output all share the
# row keyed "clk_c"), so there is no discovered Tcl call that overrides just
# the Input Output section's frequency while leaving Logic/Clocks at the design
# frequency. measured.pcf's split was written by editing the .pcf text
# directly -- it is plain text, [[Section]] headers with comma-separated rows,
# Freq and AF as the first two fields of each resource row.

if {![info exists target_pcf]} {
    set target_pcf measured.pcf
}
set base [file rootname $target_pcf]

pwc_open_project $target_pcf
pwc_calculate
pwc_gen_report ${base}_report.txt
pwc_gen_csvreport ${base}_report.csv
puts "wrote ${base}_report.txt / .csv"
