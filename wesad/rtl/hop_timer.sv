// A timer that creates a pulse for 1 clk cycle every HOP second(s)

module hop_timer #(parameter HOP = 1, parameter CLK_HZ = 12_000_000) (
    input logic clk,
    input logic rst,
    output logic hop_en
);

localparam max_count = HOP * CLK_HZ;
localparam count_width = $clog2(max_count);

logic [count_width-1:0] count = 0;
logic hop_en_ff = 0;

assign hop_en = hop_en_ff;

always @(posedge clk) begin
    if (rst) begin
        count <= 0;
        hop_en_ff <= 0;
    end

    else begin
        if (count == max_count - 2) begin
            hop_en_ff <= 1;
            count <= count + 1;
        end

        else if (count == max_count - 1) begin
            hop_en_ff <= 0;
            count <= 0;
        end
        else count <= count + 1;
    end
end

endmodule
