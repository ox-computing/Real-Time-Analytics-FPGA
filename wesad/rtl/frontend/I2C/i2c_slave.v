module i2c_slave #(
    parameter [6:0] ADDR = 7'h40
)(
    input wire clk,
    input wire rst,
    input wire scl_i,
    input wire sda_i,
    output reg sda_oe,
    output reg [7:0] rx_data,
    output reg rx_valid,
    input wire [7:0] tx_data,
    output reg tx_load
);

localparam S_IDLE = 3'd0;
localparam S_ADDR = 3'd1;
localparam S_ACK_A = 3'd2;
localparam S_WRITE = 3'd3;
localparam S_ACK_W = 3'd4;
localparam S_READ = 3'd5;
localparam S_ACK_R = 3'd6;

reg [2:0] scl_sync, sda_sync;
reg [2:0] state;
reg [3:0] bitcnt;
reg [7:0] shift;
reg [7:0] tx_shift;
reg rw;
reg nack;

wire scl_hi = scl_sync[1];
wire scl_rise = (scl_sync[2:1] == 2'b01);
wire scl_fall = (scl_sync[2:1] == 2'b10);
wire sda_bit = sda_sync[1];
wire start_cond = scl_hi && (sda_sync[2:1] == 2'b10);
wire stop_cond = scl_hi && (sda_sync[2:1] == 2'b01);

always @(posedge clk) begin
    scl_sync <= {scl_sync[1:0], scl_i};
    sda_sync <= {sda_sync[1:0], sda_i};
end

always @(posedge clk) begin
    if (rst) begin
        state <= S_IDLE;
        sda_oe <= 1'b0;
        rx_valid <= 1'b0;
        tx_load <= 1'b0;
        bitcnt <= 4'd0;
        shift <= 8'd0;
        tx_shift <= 8'd0;
        rw <= 1'b0;
        nack <= 1'b0;
    end
    else begin
        rx_valid <= 1'b0;
        tx_load <= 1'b0;

        if (start_cond) begin
            state <= S_ADDR;
            bitcnt <= 4'd0;
            sda_oe <= 1'b0;
        end
        else if (stop_cond) begin
            state <= S_IDLE;
            sda_oe <= 1'b0;
        end
        else case (state)
            S_ADDR : if (scl_rise) begin
                shift <= {shift[6:0], sda_bit};
                if (bitcnt == 4'd7) begin
                    bitcnt <= 4'd0;
                    state <= S_ACK_A;
                end
                else bitcnt <= bitcnt + 1'b1;
            end

            S_ACK_A : if (scl_fall) begin
                if (sda_oe) begin
                    if (rw) begin
                        sda_oe <= ~tx_data[7];
                        tx_shift <= {tx_data[6:0], 1'b0};
                        tx_load <= 1'b1;
                        bitcnt <= 4'd1;
                        state <= S_READ;
                    end
                    else begin
                        sda_oe <= 1'b0;
                        state <= S_WRITE;
                    end
                end
                else if (shift[7:1] == ADDR) begin
                    sda_oe <= 1'b1;
                    rw <= shift[0];
                end
                else state <= S_IDLE;
            end

            S_WRITE : if (scl_rise) begin
                shift <= {shift[6:0], sda_bit};
                if (bitcnt == 4'd7) begin
                    bitcnt <= 4'd0;
                    state <= S_ACK_W;
                end
                else bitcnt <= bitcnt + 1'b1;
            end

            S_ACK_W : if (scl_fall) begin
                if (sda_oe) begin
                    sda_oe <= 1'b0;
                    state <= S_WRITE;
                end
                else begin
                    sda_oe <= 1'b1;
                    rx_data <= shift;
                    rx_valid <= 1'b1;
                end
            end

            S_READ : if (scl_fall) begin
                if (bitcnt == 4'd8) begin
                    sda_oe <= 1'b0;
                    bitcnt <= 4'd0;
                    state <= S_ACK_R;
                end
                else begin
                    sda_oe <= ~tx_shift[7];
                    tx_shift <= {tx_shift[6:0], 1'b0};
                    bitcnt <= bitcnt + 1'b1;
                end
            end

            S_ACK_R : begin
                if (scl_rise) nack <= sda_bit;
                if (scl_fall) begin
                    if (nack) begin
                        sda_oe <= 1'b0;
                        state <= S_IDLE;
                    end
                    else begin
                        sda_oe <= ~tx_data[7];
                        tx_shift <= {tx_data[6:0], 1'b0};
                        tx_load <= 1'b1;
                        bitcnt <= 4'd1;
                        state <= S_READ;
                    end
                end
            end

            default : state <= S_IDLE;
        endcase
    end
end

endmodule
