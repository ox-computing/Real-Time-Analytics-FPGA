// GroupSum: per-class popcount of the final LUT layer's output. Class g owns
// bits[g*GROUP_SIZE +: GROUP_SIZE]; its score is how many of those bits are 1,
// landing in scores[g*SCORE_W +: SCORE_W]. This mirrors the (N, k, group_size)
// reshape in verification/hw_model.py. SCORE_W must satisfy 2**SCORE_W > GROUP_SIZE.
//
// Time-multiplexed: rather than NUM_CLASSES parallel popcount adders (each the
// full ~2*(GROUP_SIZE - SCORE_W) LUT4s, replicated), a single shared popcount
// evaluates one class per clock. A class-select mux feeds it the active slice;
// pulse `start` and the unit walks classes 0..NUM_CLASSES-1, raising `done` for
// one cycle when scores is complete. The extra NUM_CLASSES clocks are free here
// -- inference latency is dominated by the UART transfer, not the core.
module groupsum #(
    parameter NUM_CLASSES = 10,
    parameter GROUP_SIZE  = 75,
    parameter SCORE_W     = 7
) (
    input  wire                              clk,
    input  wire                              start,
    input  wire [NUM_CLASSES*GROUP_SIZE-1:0] bits,
    output reg  [NUM_CLASSES*SCORE_W-1:0]    scores,
    output reg                               done
);
    localparam IDX_W = $clog2(NUM_CLASSES);
    reg [IDX_W-1:0] idx;
    reg             busy;

    // Class-select mux + shared popcount of the active slice.
    wire [GROUP_SIZE-1:0] slice = bits[idx*GROUP_SIZE +: GROUP_SIZE];
    integer j;
    reg [SCORE_W-1:0] cnt;
    always @* begin
        cnt = {SCORE_W{1'b0}};
        for (j = 0; j < GROUP_SIZE; j = j + 1)
            cnt = cnt + slice[j];
    end

    always @(posedge clk) begin
        if (start) begin
            idx  <= 0;
            busy <= 1'b1;
            done <= 1'b0;
        end else if (busy) begin
            scores[idx*SCORE_W +: SCORE_W] <= cnt;
            if (idx == NUM_CLASSES - 1) begin
                busy <= 1'b0;
                done <= 1'b1;
            end else
                idx <= idx + 1'b1;
        end else begin
            done <= 1'b0;
        end
    end
endmodule
