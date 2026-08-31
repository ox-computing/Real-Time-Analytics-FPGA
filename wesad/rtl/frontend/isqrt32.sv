// Restoring integer square root, two bits of radicand a cycle. WIDTH must be even.
module isqrt32 #(parameter int WIDTH = 32) (
    input logic clk,
    input logic rst,
    input logic start,
    input logic [WIDTH - 1 : 0] x,
    output logic [WIDTH/2 - 1 : 0] root,
    output logic done,
    output logic busy
);

localparam int STEPS = WIDTH / 2;

logic [WIDTH/2 + 2 : 0] rem;
logic [WIDTH - 1 : 0] val;
logic [4 : 0] i;

logic [WIDTH/2 + 2 : 0] rem_shifted;
logic [WIDTH/2 + 1 : 0] trial;

assign rem_shifted = {rem[WIDTH/2 : 0], val[WIDTH - 1 : WIDTH - 2]};
assign trial = {root, 2'b01};

always @(posedge clk) begin
    if (rst) begin
        rem <= '0;
        val <= '0;
        root <= '0;
        i <= '0;
        busy <= 1'b0;
        done <= 1'b0;
    end
    else begin
        done <= 1'b0;

        if (start && !busy) begin
            rem <= '0;
            root <= '0;
            val <= x;
            i <= '0;
            busy <= 1'b1;
        end
        else if (busy) begin
            if (rem_shifted >= (WIDTH/2 + 3)'(trial)) begin
                rem <= rem_shifted - (WIDTH/2 + 3)'(trial);
                root <= {root[WIDTH/2 - 2 : 0], 1'b1};
            end
            else begin
                rem <= rem_shifted;
                root <= {root[WIDTH/2 - 2 : 0], 1'b0};
            end

            val <= {val[WIDTH - 3 : 0], 2'b00};

            if (i == 5'(STEPS - 1)) begin
                busy <= 1'b0;
                done <= 1'b1;
            end
            else begin
                i <= i + 1'b1;
            end
        end
    end
end

endmodule
