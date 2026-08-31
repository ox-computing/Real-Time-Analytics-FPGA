// Minimal UART receiver, 8N1, LSB first. Samples mid-bit at CLK_FREQ/BAUD
// clocks per bit; rx is double-registered against metastability. `valid`
// pulses for one clock with the byte on `data`; bytes with a bad stop bit
// (framing error) are dropped silently. No parity, no FIFO -- the host-side
// protocol is strict request/response so nothing ever backs up.
module uart_rx #(
    parameter CLK_FREQ = 12_000_000,
    parameter BAUD     = 115_200
) (
    input  wire       clk,
    input  wire       rx,
    output reg  [7:0] data,
    output reg        valid
);
    localparam DIV = CLK_FREQ / BAUD;   // clocks per bit, must be >= 4

    reg rx_m = 1'b1, rx_s = 1'b1;
    always @(posedge clk) begin
        rx_m <= rx;
        rx_s <= rx_m;
    end

    localparam IDLE = 2'd0, START = 2'd1, DATA = 2'd2, STOP = 2'd3;
    reg [1:0]  state = IDLE;
    reg [15:0] cnt = 0;
    reg [2:0]  bit_idx = 0;
    reg [7:0]  shreg = 0;

    initial begin
        data  = 8'h00;
        valid = 1'b0;
    end

    always @(posedge clk) begin
        valid <= 1'b0;
        case (state)
            IDLE: if (!rx_s) begin          // start-bit edge
                state <= START;
                cnt   <= DIV / 2 - 1;       // wait to mid start bit
            end
            START: if (cnt == 0) begin
                if (!rx_s) begin            // still low: real start bit
                    state   <= DATA;
                    cnt     <= DIV - 1;
                    bit_idx <= 0;
                end else
                    state <= IDLE;          // glitch, ignore
            end else
                cnt <= cnt - 1;
            DATA: if (cnt == 0) begin
                shreg <= {rx_s, shreg[7:1]};
                cnt   <= DIV - 1;
                if (bit_idx == 7)
                    state <= STOP;
                else
                    bit_idx <= bit_idx + 1;
            end else
                cnt <= cnt - 1;
            STOP: if (cnt == 0) begin       // mid stop bit
                data  <= shreg;
                valid <= rx_s;              // stop must be high
                state <= IDLE;
            end else
                cnt <= cnt - 1;
        endcase
    end
endmodule
