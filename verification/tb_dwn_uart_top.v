// Serial-level test of dwn_uart_top: pushes input vectors through the real
// UART byte protocol (294 bytes in, 11 bytes back) and writes each response
// as "class s0 ... s9" to uart_out.txt -- the same format as expected.txt
// from make_stim_real.py, so the check is a line diff.
//
// Simulating a full vector at 115200 baud would take ~300k+ bit-times, so the
// DUT is instantiated with CLK_FREQ=12/BAUD=1 (12 clocks per bit); the
// testbench uses the same divider, and the UART logic is identical at any
// ratio. NUM_VECS vectors from stim.hex are used (fewer than the file holds
// is fine -- $readmemh just leaves the rest unread).
//
//   python verification/make_stim_real.py
//   iverilog -g2005 -o tbu rtl/generated/dwn_params.vh verification/tb_dwn_uart_top.v rtl/*.v rtl/generated/*.v
//   vvp tbu && diff uart_out.txt <(head -n 5 expected.txt)
`timescale 1ns/1ps
module tb_dwn_uart_top;
    localparam NUM_VECS = 5;            // must match `head -n N` in the diff above
    localparam MAX_VECS = 200;          // stim array depth; >= make_stim_real.py's -n
    localparam IN_BITS  = `DWN_IN_BITS; // from rtl/generated/dwn_params.vh (compile it first)
    localparam IN_BYTES = IN_BITS / 8;
    localparam DIV      = 12;            // clocks per UART bit (matches params below)

    reg clk = 0;
    always #5 clk = ~clk;

    reg  rx = 1'b1;
    wire tx;
    dwn_uart_top #(.CLK_FREQ(12), .BAUD(1)) dut
        (.clk(clk), .uart_rx_i(rx), .uart_tx_o(tx));

    // Optional waveform dump: +vcd=<path> makes vvp write the DUT's activity
    // there (just the design, not the tb plumbing) -- toggle rates for power
    // estimation. Without the plusarg nothing is dumped, so the sim is
    // unaffected.
    reg [128*8-1:0] vcd_path;
    initial begin
        if ($value$plusargs("vcd=%s", vcd_path)) begin
            $dumpfile(vcd_path);
            $dumpvars(0, dut);
        end
    end

    reg [IN_BITS-1:0] stim [0:MAX_VECS-1];

    integer i, b, k, fd;
    reg [IN_BITS-1:0] vec;
    reg [7:0] resp [0:10];

    task send_byte(input [7:0] byte_val);
        begin
            rx = 1'b0;                              // start
            repeat (DIV) @(posedge clk);
            for (k = 0; k < 8; k = k + 1) begin     // data, LSB first
                rx = byte_val[k];
                repeat (DIV) @(posedge clk);
            end
            rx = 1'b1;                              // stop
            repeat (DIV) @(posedge clk);
        end
    endtask

    task recv_byte(output [7:0] byte_val);
        begin
            @(negedge tx);                          // start edge
            repeat (DIV / 2) @(posedge clk);        // mid start
            for (k = 0; k < 8; k = k + 1) begin
                repeat (DIV) @(posedge clk);        // mid data bit
                byte_val[k] = tx;
            end
            repeat (DIV) @(posedge clk);            // through stop
        end
    endtask

    initial begin
        $readmemh("stim.hex", stim);
        fd = $fopen("uart_out.txt", "w");
        for (i = 0; i < NUM_VECS; i = i + 1) begin
            vec = stim[i];
            for (b = 0; b < IN_BYTES; b = b + 1)    // byte k = bits [8k+7:8k]
                send_byte(vec[b*8 +: 8]);
            for (b = 0; b < 11; b = b + 1)
                recv_byte(resp[b]);
            $fwrite(fd, "%0d", resp[0]);
            for (b = 1; b < 11; b = b + 1)
                $fwrite(fd, " %0d", resp[b]);
            $fwrite(fd, "\n");
            $display("vector %0d: class %0d", i, resp[0]);
        end
        $fclose(fd);
        $finish;
    end
endmodule
