// AFTER IMPLEMENTING OTHER FEATURE ENGINES, ADD A WRITE ARBITER

module feature_threshold_storage #(parameter N_FEATURE = 115,
                            parameter Z = 4,
                            parameter FEATURE_WIDTH = 16,
                            parameter THRESHOLD_WIDTH = 16,
                            parameter THRESHOLD_FILE = "wesad/rtl/thresholds.hex"
                            )(
    input logic clk,

    input logic wr_en,
    input logic [$clog2(N_FEATURE) - 1 : 0] wr_addr,
    input logic [FEATURE_WIDTH - 1 : 0] wr_data,

    input logic [$clog2(N_FEATURE) - 1 : 0] feature_addr,
    input logic [$clog2(Z) - 1 : 0] threshold_addr,
    output logic [FEATURE_WIDTH - 1 : 0] feature_data,
    output logic [THRESHOLD_WIDTH - 1 : 0] threshold_data
);

logic [FEATURE_WIDTH - 1 : 0] feature_mem [0 : N_FEATURE - 1];
logic [THRESHOLD_WIDTH - 1 : 0] threshold_mem [0 : Z*N_FEATURE - 1];

initial $readmemh(THRESHOLD_FILE, threshold_mem);

always_ff @(posedge clk) begin
    if (wr_en) feature_mem[wr_addr] <= wr_data;
    feature_data <= feature_mem[feature_addr];
    threshold_data <= threshold_mem[{feature_addr, threshold_addr}];
end

endmodule
