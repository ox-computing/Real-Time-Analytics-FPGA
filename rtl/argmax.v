// Argmax over the packed per-class scores from groupsum. Ties resolve to the
// lowest class index (strict > comparison)
//
// Split into two pipeline stages.
//
// One cycle of latency: `class_idx` is valid the cycle after `scores`. The
// caller's S_ARGMAX state covers it. The stage registers free-run, so they
// also load garbage while scores is mid-update
module argmax #(
    parameter NUM_CLASSES = 10,
    parameter SCORE_W     = 7,
    parameter IDX_W       = 4
) (
    input  wire                           clk,
    input  wire [NUM_CLASSES*SCORE_W-1:0] scores,
    output reg  [IDX_W-1:0]               class_idx
);
    // Lower half owns classes [0, LO_N), upper half [LO_N, NUM_CLASSES).
    // An odd NUM_CLASSES leaves the upper half one wider, which is fine --
    // the split only has to be near-even to halve the chain.
    localparam LO_N = NUM_CLASSES / 2;

    reg [SCORE_W-1:0] best_lo, best_hi;
    reg [IDX_W-1:0]   idx_lo,  idx_hi;

    integer g;
    reg [SCORE_W-1:0] b;
    reg [IDX_W-1:0]   i;

    always @(posedge clk) begin
        b = scores[0 +: SCORE_W];
        i = {IDX_W{1'b0}};
        for (g = 1; g < LO_N; g = g + 1)
            if (scores[g*SCORE_W +: SCORE_W] > b) begin
                b = scores[g*SCORE_W +: SCORE_W];
                i = g[IDX_W-1:0];
            end
        best_lo <= b;
        idx_lo  <= i;

        b = scores[LO_N*SCORE_W +: SCORE_W];
        i = LO_N[IDX_W-1:0];
        for (g = LO_N + 1; g < NUM_CLASSES; g = g + 1)
            if (scores[g*SCORE_W +: SCORE_W] > b) begin
                b = scores[g*SCORE_W +: SCORE_W];
                i = g[IDX_W-1:0];
            end
        best_hi <= b;
        idx_hi  <= i;
    end

    // Strict > keeps the lower-indexed half on a tie, matching the original
    // single-pass scan.
    always @* class_idx = (best_hi > best_lo) ? idx_hi : idx_lo;
endmodule
