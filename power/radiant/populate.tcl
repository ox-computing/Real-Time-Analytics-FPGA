# Populates a Power Calculator project from the post-PAR design database, run
# headless from radiantc's Tcl console (Tools > Tcl Console in the GUI, or
# `radiantc.exe populate.tcl` from a shell). This is the step the GUI does
# silently the first time the Power Calculator tab is opened on an
# implementation -- doing it explicitly is what makes the flow scriptable.
#
# pwccalwrap.exe -pcf/-udb ALSO works standalone from bin/nt64, but only when
# it inherits Radiant's environment; called from a bare shell it exits 0 and
# writes nothing, with no error. `exec` from inside radiantc has that
# environment already set up, which is the only combination found to work.
#
# Usage (paths are placeholders -- point them at the actual implementation dir
# and a name for the new power project):
#   set impl_udb  {C:/path/to/impl_1/<design>_impl_1.udb}
#   set out_pcf   pop.pcf
#   radiantc.exe populate.tcl

if {![info exists impl_udb]} {
    set impl_udb {C:/path/to/impl_1/design_impl_1.udb}
}
if {![info exists out_pcf]} {
    set out_pcf pop.pcf
}

exec C:/lscc/radiant/2026.1/bin/nt64/pwccalwrap.exe -pcf $out_pcf -udb $impl_udb
puts "populated: $out_pcf (exists=[file exists $out_pcf])"
