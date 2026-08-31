// Single weightless-NN LUT node: an N-input truth table.
// INIT holds the truth table with little-endian addressing (address a reads
// INIT[a], and address bit k is driven by pin k with weight 2**
module lut_node #(
    parameter N = 4,
    parameter [(1 << N) - 1:0] INIT = {(1 << N){1'b0}}
) (
    input  wire [N-1:0] addr,
    output wire         out
);
    assign out = INIT[addr];
endmodule
