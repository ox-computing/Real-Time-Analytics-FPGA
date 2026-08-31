// Minimal UART transmitter, 8N1, LSB first. Pulse `start` (with `data`
// held) while `busy` is low to send one byte; `busy` stays high for the
// full 10-bit frame. Line idles high.
module uart_tx #(
    parameter CLK_FREQ = 12_000_000,
    parameter BAUD     = 115_200
) (
    input  wire       clk,
    input  wire [7:0] data,
    input  wire       start,
    output wire       tx,
    output wire       busy
);
    localparam DIV = CLK_FREQ / BAUD;

    // {stop, data[7:0], start}; shifted out LSB first, ones shift in so the
    // line returns to idle-high.
    reg [9:0]  shreg = 10'h3FF;
    reg [3:0]  bits_left = 0;
    reg [15:0] cnt = 0;

    assign tx   = shreg[0];
    assign busy = (bits_left != 0);

    always @(posedge clk) begin
        if (busy) begin
            if (cnt == 0) begin
                shreg     <= {1'b1, shreg[9:1]};
                bits_left <= bits_left - 1;
                cnt       <= DIV - 1;
            end else
                cnt <= cnt - 1;
        end else if (start) begin
            shreg     <= {1'b1, data, 1'b0};
            bits_left <= 4'd10;
            cnt       <= DIV - 1;
        end
    end
endmodule
