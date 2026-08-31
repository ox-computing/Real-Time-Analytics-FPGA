// Gate-level activity capture on the post-PAR UP5K netlist, for Radiant's
// Power Calculator. Drives the real UART byte protocol (the post-PAR top has
// only clk/uart_rx_i/uart_tx_o -- the parallel datapath ports exist only in
// RTL), so DIV comes from the synthesised parameters: CLK_FREQ 12e6 / BAUD
// 115200 = 104 clocks per bit.
//
// Two arms, selected by +mode, for a differential measurement. Both send the
// SAME multiset of vectors, so the byte streams carry identical intra-frame bit
// transitions and the UART receiver's switching cancels by construction:
//   +mode=0  cycle   v0,v1,v2,v0,v1,v2  -- every load changes the data: 6 fresh
//   +mode=1  paired  v0,v0,v1,v1,v2,v2  -- alternate loads repeat:     3 fresh
// In the repeated loads `pixels` is rewritten with the values it already holds
// (CE fires, D == Q, no flop toggles) and the layers recompute identical
// results, so the datapath goes quiet while the UART works normally.
//
// A minus B is therefore 3 inferences of data-dependent datapath switching,
// with clock tree, UART and byte counter common-mode. An earlier design used
// N distinct vectors vs the same vector N times, which left the two arms with
// different byte content -- uart_rx then moved by ~6.6k toggles when it should
// have moved by zero, contaminating the total by ~28% of the datapath signal.
//
// Requires the post-PAR netlist and Lattice's iCE40UP simulation primitives
// (cae_library/simulation/verilog/iCE40UP under the Radiant install). Read
// rtl/generated/dwn_params.vh first on the source list -- same rule as every
// other testbench in the repo -- so DWN_IN_BITS/DWN_NUM_CLASSES are defined
// before this file is parsed:
//
//   iverilog -g2005 -o tbg -I $LIB rtl/generated/dwn_params.vh tb_gate_power.v $NETLIST -y $LIB -Y .v
//   vvp tbg +stim=stim.hex +n=6 +nbase=3 +mode=0 +vcd=cyc.vcd  +out=cyc.txt
//   vvp tbg +stim=stim.hex +n=6 +nbase=3 +mode=1 +vcd=pair.vcd +out=pair.txt
`timescale 1ns/1ps
module tb_gate_power;
    localparam IN_BITS     = `DWN_IN_BITS;
    localparam IN_BYTES    = IN_BITS / 8;
    localparam NUM_CLASSES = `DWN_NUM_CLASSES;
    localparam RESP_BYTES  = 1 + NUM_CLASSES;   // class byte + one byte/score, matches dwn_uart_top's protocol
    localparam MAX_VECS    = 200;
    localparam DIV         = 104;               // CLK_FREQ/BAUD as synthesised -- change if the UART params do

    reg clk = 0;
    always #41.667 clk = ~clk;          // 12 MHz

    reg  rx = 1'b1;
    wire tx;
    // Instance deliberately named for the module: Radiant's Power Calculator
    // matches the VCD scope against its own netlist, so the scope the design
    // sits in should read dwn_uart_top rather than a testbench-local alias.
    dwn_uart_top dwn_uart_top (.clk(clk), .uart_rx_i(rx), .uart_tx_o(tx));

    reg [IN_BITS-1:0] stim [0:MAX_VECS-1];
    reg [128*8-1:0] stim_path, vcd_path, out_path;
    integer num_vecs, rep, nbase, i, b, k, fd, dskip, dwin;
    reg [IN_BITS-1:0] vec;
    reg [7:0] resp [0:RESP_BYTES-1];
    reg timed_out;

    // Wall-clock progress, so a long gate-level run can be told apart from a
    // stalled one from outside the simulator.
    always #500000 $display("[t=%0t ns] running", $time);

    task send_byte(input [7:0] byte_val);
        begin
            rx = 1'b0;
            repeat (DIV) @(posedge clk);
            for (k = 0; k < 8; k = k + 1) begin
                rx = byte_val[k];
                repeat (DIV) @(posedge clk);
            end
            rx = 1'b1;
            repeat (DIV) @(posedge clk);
        end
    endtask

    // Free-running receive monitor. It must run concurrently with the send
    // loop, not be called after it: uart_rx raises `valid` at the MIDDLE of the
    // last byte's stop bit, ~DIV/2 clocks before send_byte returns, and the FSM
    // reaches S_TX about 21 clocks later -- so the response's start bit falls
    // roughly 31 clocks BEFORE the final send_byte finishes. A receiver that
    // only starts looking once sending is done has already missed that edge,
    // which silently costs exactly one byte per vector.
    //
    // Decoded bytes land in a queue that the main loop drains, decoupling
    // framing from the request/response sequencing entirely.
    reg [7:0] rxq [0:511];
    integer   rxq_wr = 0, rxq_rd = 0;
    integer   mbit;
    reg [7:0] macc;

    initial begin
        // The netlist powers up with every flop at 0 (FD1P3XZ forces Q=0 and
        // ignores the bitstream INIT behind shreg's 10'h3FF), so uart_tx_o
        // idles LOW until the first frame reloads it and the first start bit
        // has no falling edge at all. Real silicon powers up from INIT and does
        // not do this. Wait for the line to reach a genuine idle-high before
        // arming, which costs the first response of the run and nothing after.
        wait (tx === 1'b1);
        forever begin
            @(negedge tx);
            repeat (DIV / 2) @(posedge clk);
            if (tx === 1'b0) begin              // still low: real start bit
                for (mbit = 0; mbit < 8; mbit = mbit + 1) begin
                    repeat (DIV) @(posedge clk);
                    macc[mbit] = tx;
                end
                repeat (DIV) @(posedge clk);    // through the stop bit
                rxq[rxq_wr % 512] = macc;
                rxq_wr = rxq_wr + 1;
            end
        end
    end

    // Drain one byte from the monitor queue, with a bounded wait.
    task get_byte(output [7:0] byte_val);
        integer guard;
        begin
            guard     = 0;
            timed_out = 1'b0;
            while (rxq_rd == rxq_wr && guard < 40 * DIV) begin
                @(posedge clk);
                guard = guard + 1;
            end
            if (rxq_rd == rxq_wr) begin
                timed_out = 1'b1;
                byte_val  = 8'hxx;
            end else begin
                byte_val = rxq[rxq_rd % 512];
                rxq_rd   = rxq_rd + 1;
            end
        end
    endtask

    // VCD control lives in its own process. A full transaction dumps ~90 MB of
    // gate-level activity, more than Radiant's VCD importer will reliably take
    // (it accepted a 93 MB file, crashed silently on 268 MB), so +dumpskip
    // suppresses the dump for N clocks and +dumpwin ends it M clocks later,
    // capturing a representative slice instead of the whole run.
    initial begin
        if ($value$plusargs("vcd=%s", vcd_path)) begin
            $dumpfile(vcd_path);
            $dumpvars(0, dwn_uart_top);
            if ($value$plusargs("dumpskip=%d", dskip)) begin
                $dumpoff;
                repeat (dskip) @(posedge clk);
                $dumpon;
                if ($value$plusargs("dumpwin=%d", dwin)) begin
                    repeat (dwin) @(posedge clk);
                    $dumpoff;
                end
            end
        end
    end

    initial begin
        if (!$value$plusargs("stim=%s", stim_path)) stim_path = "stim.hex";
        if (!$value$plusargs("out=%s",  out_path))  out_path  = "gate_out.txt";
        if (!$value$plusargs("n=%d",    num_vecs))  num_vecs  = 6;
        if (!$value$plusargs("mode=%d", rep))       rep       = 0;
        if (!$value$plusargs("nbase=%d", nbase))    nbase     = 3;

        $readmemh(stim_path, stim);
        fd = $fopen(out_path, "w");

        // The design powers up with every flop at 0 (FD1P3XZ initialises Q=0),
        // which leaves uart_tx_o low -- a line break, not the idle mark. Let it
        // sit long enough for the transmitter to reach its idle state before
        // the first start bit, otherwise the first byte is framed against a
        // line that was never high.
        repeat (16 * DIV) @(posedge clk);

        for (i = 0; i < num_vecs; i = i + 1) begin
            // mode 0 cycles v0..v(nbase-1); mode 1 repeats each in pairs.
            vec = (rep != 0) ? stim[(i / 2) % nbase] : stim[i % nbase];
            $display("[t=%0t ns] vector %0d: sending %0d bytes", $time, i, IN_BYTES);
            for (b = 0; b < IN_BYTES; b = b + 1)
                send_byte(vec[b*8 +: 8]);
            $display("[t=%0t ns] vector %0d: sent, awaiting response", $time, i);
            for (b = 0; b < RESP_BYTES; b = b + 1) begin
                get_byte(resp[b]);
                if (timed_out) begin
                    $display("[t=%0t ns] vector %0d: TIMEOUT on response byte %0d (tx=%b)",
                             $time, i, b, tx);
                    b = RESP_BYTES;
                end
            end
            $fwrite(fd, "%0d", resp[0]);
            for (b = 1; b < RESP_BYTES; b = b + 1)
                $fwrite(fd, " %0d", resp[b]);
            $fwrite(fd, "\n");
            $display("[%0t] vector %0d -> class %0d", $time, i, resp[0]);
        end

        $fclose(fd);
        $display("done: %0d vectors, mode=%0d, nbase=%0d", num_vecs, rep, nbase);
        $finish;
    end
endmodule
