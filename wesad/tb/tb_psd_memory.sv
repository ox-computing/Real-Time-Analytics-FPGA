// Unit bench for the two memories in the PSD engine: the single 2048-point FFT
// scratch and the segment power accumulator. Checks the load path lands
// bit-reversed, that a butterfly writeback is visible to a later read of the
// same scratch, that the even/odd bank split routes both operands of a
// butterfly at every level, and that the accumulator sums, aligns exponents and
// clips instead of wrapping.
//
//   iverilog -g2012 -s tb_psd_memory wesad/tb/tb_psd_memory.sv \
//            wesad/rtl/feature_engine/psd_features.sv
`timescale 1ns / 1ps

module tb_psd_memory;

localparam int BW = 8;
localparam int ACC_W = 16;
localparam int WRITEBACK_DELAY = 4;
localparam int ACC_MAX = (1 << ACC_W) - 1;

logic clk = 0;
logic rst = 1;
always #5 clk = ~clk;

int errors = 0;

task automatic check (input int got, input int want, input string what);
    if (got !== want) begin
        $display("FAIL %-38s got %0d want %0d", what, got, want);
        errors++;
    end
endtask

function automatic logic [10 : 0] bitrev (input logic [10 : 0] a);
    for (int i = 0; i < 11; i++)
        bitrev[i] = a[10 - i];
endfunction


// --- the scratch ------------------------------------------------------------

logic load_valid = 0;
logic [10 : 0] load_address = 0;
logic signed [BW - 1 : 0] load_sample = 0;
logic butterfly_valid = 0;
logic [10 : 0] a_address = 0;
logic [10 : 0] b_address = 0;
logic butterfly_in_valid;
logic signed [BW - 1 : 0] a_real, a_imaginary, b_real, b_imaginary;
logic signed [BW - 1 : 0] a_out_real = 0, a_out_imaginary = 0;
logic signed [BW - 1 : 0] b_out_real = 0, b_out_imaginary = 0;
logic magnitude_exceeded;

fft_scratch_memory #(.BUTTERFLY_WIDTH(BW)) scratch (
    .clk (clk),
    .rst (rst),
    .load_valid (load_valid),
    .load_address (load_address),
    .load_sample (load_sample),
    .butterfly_valid (butterfly_valid),
    .a_address (a_address),
    .b_address (b_address),
    .butterfly_in_valid (butterfly_in_valid),
    .a_real (a_real),
    .a_imaginary (a_imaginary),
    .b_real (b_real),
    .b_imaginary (b_imaginary),
    .a_out_real (a_out_real),
    .a_out_imaginary (a_out_imaginary),
    .b_out_real (b_out_real),
    .b_out_imaginary (b_out_imaginary),
    .magnitude_exceeded (magnitude_exceeded)
);

// the load walks natural order and the module reverses the address for us
task automatic load_segment ();
    for (int i = 0; i < 2048; i++) begin
        @(posedge clk);
        load_valid <= 1;
        load_address <= i[10 : 0];
        load_sample <= sample_of(i);
    end
    @(posedge clk);
    load_valid <= 0;
endtask

function automatic logic signed [BW - 1 : 0] sample_of (input int i);
    sample_of = BW'((i * 7) - 128);
endfunction

// one butterfly: present the operand addresses, then hold the results on the
// cycle the writeback pipeline reaches the memory
task automatic butterfly (input logic [10 : 0] ia, input logic [10 : 0] ib,
                          input logic signed [BW - 1 : 0] ar,
                          input logic signed [BW - 1 : 0] ai,
                          input logic signed [BW - 1 : 0] br,
                          input logic signed [BW - 1 : 0] bi);
    @(posedge clk);
    a_address <= ia;
    b_address <= ib;
    butterfly_valid <= 1;
    @(posedge clk);
    butterfly_valid <= 0;
    repeat (WRITEBACK_DELAY - 1) @(posedge clk);
    a_out_real <= ar;
    a_out_imaginary <= ai;
    b_out_real <= br;
    b_out_imaginary <= bi;
    @(posedge clk);
endtask

task automatic read_pair (input logic [10 : 0] ia, input logic [10 : 0] ib);
    @(posedge clk);
    a_address <= ia;
    b_address <= ib;
    @(posedge clk);
    @(negedge clk);
endtask


// --- the accumulator --------------------------------------------------------

logic start_segment = 0;
logic first_segment = 0;
logic signed [5 : 0] segment_exponent = 0;
logic bin_valid = 0;
logic [10 : 0] bin_address = 0;
logic signed [BW - 1 : 0] bin_real = 0;
logic signed [BW - 1 : 0] bin_imaginary = 0;
logic read_enable = 0;
logic [10 : 0] read_address = 0;
logic [ACC_W - 1 : 0] read_data;
logic signed [5 : 0] power_exponent;
logic [2*BW : 0] bin_power;

assign bin_power = bin_real*bin_real + bin_imaginary*bin_imaginary;

psd_power_accumulator #(.BUTTERFLY_WIDTH(BW), .ACC_WIDTH(ACC_W)) accum (
    .clk (clk),
    .rst (rst),
    .start_segment (start_segment),
    .first_segment (first_segment),
    .segment_exponent (segment_exponent),
    .bin_valid (bin_valid),
    .bin_address (bin_address),
    .bin_power (bin_power),
    .read_enable (read_enable),
    .read_address (read_address),
    .read_data (read_data),
    .power_exponent (power_exponent)
);

// three bins per segment: a moderate one, a saturating one, and a zero
task automatic send_segment (input logic first, input logic signed [5 : 0] expo,
                             input logic signed [BW - 1 : 0] r0,
                             input logic signed [BW - 1 : 0] r1);
    @(posedge clk);
    start_segment <= 1;
    first_segment <= first;
    segment_exponent <= expo;
    @(posedge clk);
    start_segment <= 0;
    first_segment <= 0;
    for (int k = 0; k < 3; k++) begin
        @(posedge clk);
        bin_valid <= 1;
        bin_address <= k[10 : 0];
        bin_real <= (k == 0) ? r0 : (k == 1) ? r1 : 0;
        bin_imaginary <= (k == 1) ? r1 : 0;
    end
    @(posedge clk);
    bin_valid <= 0;
    repeat (2) @(posedge clk);
endtask

// read_data trails the address by two edges, and the cycle after that it is
// reloaded from whatever bin_address still points at, so sample it in between
task automatic read_bin (input int k);
    @(posedge clk);
    read_enable <= 1;
    read_address <= k[10 : 0];
    @(posedge clk);
    read_enable <= 0;
    @(posedge clk);
    @(negedge clk);
endtask


// --- sequence ---------------------------------------------------------------

int want;
int idx;

initial begin
    repeat (4) @(posedge clk);
    rst <= 0;
    @(posedge clk);

    // 1. the load lands bit-reversed and the imaginary half is cleared
    load_segment();
    for (int i = 0; i < 2048; i += 257) begin
        read_pair(bitrev(i[10 : 0]), bitrev((i + 1) % 2048));
        check(a_real, sample_of(i), $sformatf("load real @%0d", i));
        check(a_imaginary, 0, $sformatf("load imag @%0d", i));
    end

    // 2. a butterfly writeback is visible to a later read of the same scratch.
    //    Operand pairs differ in exactly one address bit, so this also walks the
    //    even/odd split across every FFT level.
    for (int level = 0; level < 11; level++) begin
        // the pair is (i, i + 2**level) only where bit `level` of i is clear
        idx = 3 & ~(1 << level);
        butterfly(idx[10 : 0], (idx + (1 << level)) % 2048,
                  8'sd11 + BW'(level), -8'sd5, 8'sd33, 8'sd7 + BW'(level));
        read_pair(idx[10 : 0], (idx + (1 << level)) % 2048);
        check(a_real, 11 + level, $sformatf("butterfly a_real level %0d", level));
        check(a_imaginary, -5, $sformatf("butterfly a_imag level %0d", level));
        check(b_real, 33, $sformatf("butterfly b_real level %0d", level));
        check(b_imaginary, 7 + level, $sformatf("butterfly b_imag level %0d", level));
    end

    // 3. accumulate four segments at one exponent: bin 0 sums, bin 1 clips
    send_segment(1, 0, 100, 127);
    send_segment(0, 0, 100, 127);
    send_segment(0, 0, 100, 127);
    send_segment(0, 0, 100, 127);
    read_bin(0);
    check(read_data, 4 * 10000, "accumulate bin0 over four segments");
    read_bin(1);
    check(read_data, ACC_MAX, "saturate bin1 instead of wrapping");
    read_bin(2);
    check(read_data, 0, "zero bin stays zero");

    // 4. a new window must start from this window's power, not inherit the last
    send_segment(1, 0, 100, 0);
    read_bin(0);
    check(read_data, 10000, "first segment of a new window replaces");

    // 5. exponent alignment: the larger exponent wins and the other side shifts
    send_segment(1, 0, 100, 0);
    send_segment(0, 2, 100, 0);
    read_bin(0);
    check(read_data, (10000 >> 2) + 10000, "align accumulator up to segment");
    check(power_exponent, 2, "exponent tracks the larger segment");
    send_segment(0, 0, 100, 0);
    read_bin(0);
    want = (10000 >> 2) + 10000 + (10000 >> 2);
    check(read_data, want, "align segment down to accumulator");
    check(power_exponent, 2, "exponent holds on a smaller segment");

    if (errors == 0)
        $display("PSD MEMORY PASS");
    else
        $display("PSD MEMORY FAIL (%0d)", errors);
    $finish;
end

endmodule
