// Drives one window's spectrum through psd_calc and checks all 13 feature values against the golden model.
`timescale 1ns / 1ps

module tb_psd_calc;

logic clk = 0;
logic rst = 1;
always #5 clk = ~clk;

logic start = 0;
logic [2:0] modality;
logic signed [6:0] power_exponent;

logic acc_read_enable;
logic [10:0] acc_read_address;
logic [15:0] acc_read_data;

logic div_start;
logic [39:0] div_numerator;
logic [23:0] div_denominator;
logic [3:0] div_frac;
logic [39:0] div_quotient;
logic div_done, div_busy;

logic [23:0] log_value;
logic [12:0] log_result;

logic sqrt_start;
logic [35:0] sqrt_data;
logic [17:0] sqrt_result;
logic sqrt_done, sqrt_busy;

logic busy;
logic feature_wr_en;
logic [6:0] feature_wr_addr;
logic [7:0] feature_wr_data;

logic [15:0] spectrum [0:1024];
logic [7:0] expected [0:12];
logic [7:0] captured [0:12];
logic [7:0] pe_file [0:0];
logic [7:0] mod_file [0:0];
int errors = 0;

// same two-cycle read latency psd_power_accumulator presents
logic [15:0] acc_q1;
always_ff @(posedge clk) begin
    acc_q1 <= spectrum[acc_read_address];
    acc_read_data <= acc_q1;
end

logic [20:0] mul_a;
logic [15:0] mul_b;
logic [36:0] product;
assign product = mul_a * mul_b;

psd_calc dut (
    .clk (clk),
    .rst (rst),
    .start (start),
    .modality (modality),
    .power_exponent (power_exponent),
    .acc_read_enable (acc_read_enable),
    .acc_read_address (acc_read_address),
    .acc_read_data (acc_read_data),
    .div_start (div_start),
    .div_numerator (div_numerator),
    .div_denominator (div_denominator),
    .div_frac (div_frac),
    .div_quotient (div_quotient),
    .div_done (div_done),
    .div_busy (div_busy),
    .log_value (log_value),
    .log_result (log_result),
    .sqrt_start (sqrt_start),
    .sqrt_data (sqrt_data),
    .sqrt_result (sqrt_result),
    .sqrt_done (sqrt_done),
    .sqrt_busy (sqrt_busy),
    .mul_a (mul_a),
    .mul_b (mul_b),
    .product (product),
    .busy (busy),
    .feature_wr_en (feature_wr_en),
    .feature_wr_addr (feature_wr_addr),
    .feature_wr_data (feature_wr_data)
);

divider div (
    .clk (clk),
    .rst (rst),
    .start (div_start),
    .numerator (div_numerator),
    .denominator (div_denominator),
    .frac_bits (div_frac),
    .quotient (div_quotient),
    .done (div_done),
    .busy (div_busy)
);

log2_unit lg (
    .value (log_value),
    .result (log_result)
);

isqrt32 #(.WIDTH(36)) root (
    .clk (clk),
    .rst (rst),
    .start (sqrt_start),
    .x (sqrt_data),
    .root (sqrt_result),
    .done (sqrt_done),
    .busy (sqrt_busy)
);

always_ff @(posedge clk)
    if (feature_wr_en) captured[feature_wr_addr - 7'd50 - 7'(modality * 13)] <= feature_wr_data;

initial begin
    $readmemh("wesad/sim_fixtures/psd_acc.hex", spectrum);
    $readmemh("wesad/sim_fixtures/psd_feat.hex", expected);
    $readmemh("wesad/sim_fixtures/psd_pe.hex", pe_file);
    $readmemh("wesad/sim_fixtures/psd_mod.hex", mod_file);
    power_exponent = pe_file[0][6:0];
    modality = mod_file[0][2:0];

    repeat (4) @(posedge clk);
    rst <= 0;
    @(posedge clk);
    start <= 1;
    @(posedge clk);
    start <= 0;
    @(posedge clk);
    wait (!busy);
    repeat (4) @(posedge clk);

    for (int i = 0; i < 13; i++)
        if (captured[i] !== expected[i]) begin
            $display("FAIL feature %0d got %0d want %0d", i, $signed(captured[i]), $signed(expected[i]));
            errors++;
        end

    if (errors == 0) $display("PSD CALC PASS");
    else $display("PSD CALC FAIL (%0d of 13)", errors);
    $finish;
end

endmodule
