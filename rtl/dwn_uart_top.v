// Protocol (strict request/response, 8N1):
//   host -> fpga : 294 bytes = one 2352-bit thermometer-encoded input vector,
//                  little-endian: byte k carries pixels_thermo[8k+7 : 8k]
//                  (python: vector_int.to_bytes(294, "little"))
//   fpga -> host : 11 bytes = {class_idx, score[0], ..., score[9]}
//                  (each 7-bit score zero-padded to a byte)
// The host must read the full 11-byte response before sending the next
// vector; bytes arriving mid-response are dropped (no FIFO).
//
// Each incoming byte is written straight to its own fixed slot
// (pixels[8k+7:8k] for byte k) rather than shifted through the register, so
// byte lanes with no downstream LUT reader carry no data and drop out in
// synthesis. The LUT layers live in the generated dwn_core, which registers each
// layer's output and is therefore a NUM_LAYERS-stage pipeline: the deepest
// combinational path is one layer, not pixels -> scores. After the last byte the
// FSM waits SETTLE_CYCLES clocks for those stages to fill, then pulses the
// groupsum unit, which walks the NUM_CLASSES scores one per clock (S_GROUPSUM)
// before the response is sent.
module dwn_uart_top #(
    parameter CLK_FREQ = 12_000_000,
    parameter BAUD     = 115_200
) (
    input  wire clk,
    input  wire uart_rx_i,
    output wire uart_tx_o
);
    localparam IN_BITS       = `DWN_IN_BITS;
    localparam IN_BYTES      = IN_BITS / 8;          // 294
    localparam NUM_CLASSES   = `DWN_NUM_CLASSES;
    localparam SCORE_W       = `DWN_SCORE_W;
    localparam NUM_LAYERS    = `DWN_NUM_LAYERS;
    localparam OUT_BITS      = `DWN_OUT_BITS;        // width of the core's last layer
    localparam GROUP_SIZE    = OUT_BITS / NUM_CLASSES;
    localparam RESP_BYTES    = 1 + NUM_CLASSES;      // class + scores
    // The core is a NUM_LAYERS-deep register pipeline, so it needs NUM_LAYERS
    // clocks to fill; the margin is slack, not requirement.
    localparam SETTLE_CYCLES = NUM_LAYERS + 5;
    localparam SETTLE_W      = $clog2(SETTLE_CYCLES + 1);
    localparam IDX_W         = $clog2(NUM_CLASSES);
    // Resync guard: if the host stalls partway through a vector, abort after
    // ~10 idle byte-frames (100 bit-times) of silence so a fresh vector
    // re-aligns instead of deadlocking mid-reception.
    localparam CLKS_PER_BIT  = CLK_FREQ / BAUD;
    localparam RX_TIMEOUT    = 100 * CLKS_PER_BIT;
    localparam TIMEOUT_W     = $clog2(RX_TIMEOUT + 1);

    // --- UART ---
    wire [7:0] rx_data;
    wire       rx_valid;
    uart_rx #(.CLK_FREQ(CLK_FREQ), .BAUD(BAUD)) u_rx (
        .clk(clk), .rx(uart_rx_i), .data(rx_data), .valid(rx_valid));

    reg  [7:0] tx_data  = 8'h00;
    reg        tx_start = 1'b0;
    wire       tx_busy;
    uart_tx #(.CLK_FREQ(CLK_FREQ), .BAUD(BAUD)) u_tx (
        .clk(clk), .data(tx_data), .start(tx_start), .tx(uart_tx_o), .busy(tx_busy));


    reg  [IN_BITS-1:0]  pixels = {IN_BITS{1'b0}};
    wire [OUT_BITS-1:0] core_bits;
    dwn_core u_core (.clk(clk), .in(pixels), .out(core_bits));

    reg                            gs_start = 1'b0;
    wire [NUM_CLASSES*SCORE_W-1:0] scores;
    wire                           gs_done;
    groupsum #(.NUM_CLASSES(NUM_CLASSES), .GROUP_SIZE(GROUP_SIZE), .SCORE_W(SCORE_W)) u_groupsum (
        .clk(clk), .start(gs_start), .bits(core_bits), .scores(scores), .done(gs_done));

    wire [IDX_W-1:0] class_idx;
    argmax #(.NUM_CLASSES(NUM_CLASSES), .SCORE_W(SCORE_W), .IDX_W(IDX_W)) u_argmax (
        .clk(clk), .scores(scores), .class_idx(class_idx));

    // --- control FSM ---
    localparam S_RX     = 3'd0, S_SETTLE = 3'd1, S_GROUPSUM = 3'd2,
               S_ARGMAX = 3'd3, S_TX     = 3'd4;
    reg [2:0]          state    = S_RX;
    reg [8:0]          byte_cnt = 0;
    reg [SETTLE_W-1:0] settle   = 0;
    reg [TIMEOUT_W-1:0] rx_timer = 0;
    reg [3:0]          resp_idx = 0;
    reg [7:0]          resp [0:RESP_BYTES-1];
    integer g;

    always @(posedge clk) begin
        gs_start <= 1'b0;                          // one-cycle strobe by default
        case (state)
            S_RX:
                if (rx_valid) begin
                    rx_timer <= 0;
                    pixels[byte_cnt*8 +: 8] <= rx_data;
                    if (byte_cnt == IN_BYTES - 1) begin
                        byte_cnt <= 0;
                        settle   <= 0;
                        state    <= S_SETTLE;
                    end else
                        byte_cnt <= byte_cnt + 1;
                end else if (byte_cnt != 0) begin
                    // mid-vector silence: count toward the resync timeout, then
                    // discard the partial vector so the next one re-aligns
                    if (rx_timer == RX_TIMEOUT) begin
                        byte_cnt <= 0;
                        rx_timer <= 0;
                    end else
                        rx_timer <= rx_timer + 1'b1;
                end
            S_SETTLE: begin
                settle <= settle + 1;
                if (settle == SETTLE_CYCLES) begin
                    gs_start <= 1'b1;              // kick off the sequential popcount
                    state    <= S_GROUPSUM;
                end
            end
            S_GROUPSUM: if (gs_done) begin
                // groupsum registers its last score on the same edge that
                // raises done, so `scores` only settles after this edge.
                for (g = 0; g < NUM_CLASSES; g = g + 1)
                    resp[g+1] <= scores[g*SCORE_W +: SCORE_W];
                state <= S_ARGMAX;
            end
            S_ARGMAX: begin
                resp[0]  <= class_idx;                // zero-extends to 8 bits
                resp_idx <= 0;
                state    <= S_TX;
            end
            S_TX: begin
                if (tx_start) begin
                    tx_start <= 1'b0;               // byte handed to uart_tx
                    if (resp_idx == RESP_BYTES - 1)
                        state <= S_RX;
                    else
                        resp_idx <= resp_idx + 1;
                end else if (!tx_busy) begin
                    tx_data  <= resp[resp_idx];
                    tx_start <= 1'b1;
                end
            end
            default: state <= S_RX;
        endcase
    end
endmodule
