module i2c_result_tx (
    input logic clk,
    input logic rst,
    input logic result_valid,
    input logic [1:0] class_idx,
    input logic [14:0] scores,
    input logic tx_load,
    output logic [7:0] tx_data
);

logic [1:0] index;
logic [1:0] class_ff;
logic [14:0] scores_ff;

always_comb begin
    case (index)
        2'd0 : tx_data = {6'b0, class_ff};
        2'd1 : tx_data = {3'b0, scores_ff[4:0]};
        2'd2 : tx_data = {3'b0, scores_ff[9:5]};
        default : tx_data = {3'b0, scores_ff[14:10]};
    endcase
end

always_ff @(posedge clk) begin
    if (rst) begin
        index <= '0;
        class_ff <= '0;
        scores_ff <= '0;
    end
    else if (result_valid) begin
        class_ff <= class_idx;
        scores_ff <= scores;
        index <= '0;
    end
    else if (tx_load) index <= index + 1'b1;
end

endmodule
