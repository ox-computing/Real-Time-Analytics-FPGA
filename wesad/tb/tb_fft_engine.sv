// Runs one segment through fft_engine and checks every bin's power against the golden transform.
`timescale 1ns / 1ps

module tb_fft_engine;

logic clk = 0;
logic rst = 1;
always #5 clk = ~clk;

logic phase = 1;
logic stats_pass = 0;
logic pass_start = 0;
logic pass_done = 0;
logic stream_valid = 0;
logic signed [15:0] stream_data = 0;
logic [11:0] segment_length = 12'd2048;
logic [1:0] table_select = 2'd0;

logic busy;
logic signed [6:0] segment_exponent;
logic exponent_valid;
logic bin_valid;
logic [10:0] bin_address;
logic signed [7:0] bin_real, bin_imaginary;
logic spectrum_done;

logic [15:0] samples [0:2047];
logic [23:0] expected [0:1024];
logic [7:0] pe_file [0:0];
int errors = 0;
int seen = 0;

logic signed [7:0] mul_x, mul_y;
logic signed [15:0] mul_p;
assign mul_p = mul_x * mul_y;

fft_engine dut (
    .clk (clk),
    .rst (rst),
    .phase (phase),
    .stats_pass (stats_pass),
    .pass_start (pass_start),
    .pass_done (pass_done),
    .stream_valid (stream_valid),
    .stream_data (stream_data),
    .segment_length (segment_length),
    .table_select (table_select),
    .mul_x (mul_x),
    .mul_y (mul_y),
    .mul_p (mul_p),
    .busy (busy),
    .segment_exponent (segment_exponent),
    .exponent_valid (exponent_valid),
    .bin_valid (bin_valid),
    .bin_address (bin_address),
    .bin_real (bin_real),
    .bin_imaginary (bin_imaginary),
    .spectrum_done (spectrum_done)
);

task automatic one_pass (input logic is_stats);
    @(posedge clk);
    stats_pass <= is_stats;
    pass_start <= 1;
    @(posedge clk);
    pass_start <= 0;
    for (int i = 0; i < 2048; i++) begin
        @(posedge clk);
        stream_valid <= 1;
        stream_data <= samples[i];
    end
    @(posedge clk);
    stream_valid <= 0;
    pass_done <= 1;
    @(posedge clk);
    pass_done <= 0;
    wait (!busy);
    @(posedge clk);
endtask

int power;
always @(posedge clk)
    if (bin_valid) begin
        power = bin_real * bin_real + bin_imaginary * bin_imaginary;
        seen++;
        if (power !== expected[bin_address] && errors < 8) begin
            $display("FAIL bin %0d got %0d want %0d", bin_address, power, expected[bin_address]);
            errors++;
        end
        else if (power !== expected[bin_address]) errors++;
    end

initial begin
    $readmemh("wesad/sim_fixtures/fft_in.hex", samples);
    $readmemh("wesad/sim_fixtures/fft_power.hex", expected);
    $readmemh("wesad/sim_fixtures/fft_pe.hex", pe_file);

    repeat (4) @(posedge clk);
    rst <= 0;
    @(posedge clk);

    one_pass(1);
    one_pass(0);

    if (segment_exponent !== $signed(pe_file[0][6:0]))
        $display("FAIL exponent got %0d want %0d", segment_exponent, $signed(pe_file[0][6:0]));

    $display("bins checked %0d, mismatches %0d", seen, errors);
    if (errors == 0 && seen == 1025) $display("FFT ENGINE PASS");
    else $display("FFT ENGINE FAIL");
    $finish;
end

endmodule
