// Streams one ECG window through hrv_features and checks the eight feature words against the golden model.
`timescale 1ns / 1ps

module tb_hrv_features;

logic clk = 0;
logic rst = 1;
always #5 clk = ~clk;

logic sample_valid = 0;
logic signed [15:0] sample = 0;
logic window_start = 0;
logic window_end = 0;

logic div_start;
logic [39:0] div_numerator;
logic [23:0] div_denominator;
logic [3:0] div_frac;
logic [39:0] div_quotient;
logic div_done, div_busy;

logic sqrt_start;
logic [31:0] sqrt_data;
logic [15:0] sqrt_result;
logic sqrt_done, sqrt_busy;

logic busy;
logic ready;
logic feature_wr_en;
logic [2:0] feature_index;
logic [7:0] feature_word;

logic [15:0] codes [0:14999];
logic [7:0] expected [0:7];
logic [7:0] captured [0:7];
int errors = 0;

logic signed [31:0] mul_x;
logic signed [15:0] mul_y;
logic signed [47:0] mul_p;
assign mul_p = mul_x * mul_y;

hrv_features dut (
    .clk (clk), .rst (rst),
    .sample_valid (sample_valid), .sample (sample),
    .window_start (window_start), .window_end (window_end),
    .div_start (div_start), .div_numerator (div_numerator),
    .div_denominator (div_denominator), .div_frac (div_frac),
    .div_quotient (div_quotient), .div_done (div_done),
    .sqrt_start (sqrt_start), .sqrt_data (sqrt_data),
    .sqrt_result (sqrt_result), .sqrt_done (sqrt_done),
    .mul_x (mul_x), .mul_y (mul_y), .mul_p (mul_p),
    .busy (busy), .ready (ready), .feature_wr_en (feature_wr_en),
    .feature_index (feature_index), .feature_word (feature_word)
);

divider div (
    .clk (clk), .rst (rst), .start (div_start),
    .numerator (div_numerator), .denominator (div_denominator),
    .frac_bits (div_frac), .quotient (div_quotient),
    .done (div_done), .busy (div_busy)
);

isqrt32 root (
    .clk (clk), .rst (rst), .start (sqrt_start), .x (sqrt_data),
    .root (sqrt_result), .done (sqrt_done), .busy (sqrt_busy)
);

always_ff @(posedge clk)
    if (feature_wr_en) captured[feature_index] <= feature_word;

initial begin
    $readmemh("wesad/sim_fixtures/hrv_in.hex", codes);
    $readmemh("wesad/sim_fixtures/hrv_feat.hex", expected);

    repeat (4) @(posedge clk);
    rst <= 0;
    @(posedge clk);
    window_start <= 1;
    @(posedge clk);
    window_start <= 0;
    wait (busy);
    wait (!busy);

    for (int i = 0; i < 15000; i++) begin
        while (!ready) @(posedge clk);
        sample_valid <= 1;
        sample <= codes[i];
        @(posedge clk);
        sample_valid <= 0;
        @(posedge clk);
    end
    while (!ready) @(posedge clk);
    @(posedge clk);
    window_end <= 1;
    @(posedge clk);
    window_end <= 0;
    wait (busy);
    wait (!busy);
    repeat (4) @(posedge clk);

    for (int i = 0; i < 8; i++)
        if (captured[i] !== expected[i]) begin
            $display("FAIL feature %0d got %0d want %0d", i,
                     $signed(captured[i]), $signed(expected[i]));
            errors++;
        end
    if (errors == 0) $display("HRV PASS");
    else $display("HRV FAIL (%0d of 8)", errors);
    $finish;
end

endmodule
