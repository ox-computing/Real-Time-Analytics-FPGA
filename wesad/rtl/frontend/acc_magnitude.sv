module acc_magnitude (
    input logic clk,
    input logic rst,
    input logic burst_start,
    input logic word_done,
    input logic signed [15:0] sample,
    output logic [31:0] sumsq,
    input logic [15:0] root,
    input logic isqrt_done,
    output logic [15:0] mag,
    output logic mag_done
);

logic [31:0] sq;

logic signed [31:0] sq_signed;

assign sq_signed = sample * sample;
assign sq = $unsigned(sq_signed);

always @(posedge clk) begin
    if (rst || burst_start) sumsq <= '0;
    else if (word_done) sumsq <= sumsq + sq;
end

always @(posedge clk) begin
    if (rst) begin
        mag <= '0;
        mag_done <= 1'b0;
    end
    else begin
        mag_done <= isqrt_done;
        if (isqrt_done) mag <= root;
    end
end

endmodule
