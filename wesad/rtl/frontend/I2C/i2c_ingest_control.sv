module i2c_ingest_control #(
    parameter int N_ECG = 15000,
    parameter int N_ACC = 1920,
    parameter int N_RESP = 1500,
    parameter int N_EDA = 1500,
    parameter int N_EMG = 21000
)(
    input logic clk,
    input logic rst,

    input logic [7:0] rx_data,
    input logic rx_valid,

    input logic [15:0] wptr,
    input logic mag_done,

    output logic burst_start,
    output logic word_done,
    output logic mag_start,
    output logic [2:0] sensor,
    output logic [15:0] sample,
    output logic [15:0] addr,
    output logic wren,
    output logic [3:0] mask,
    output logic idle,
    output logic window_end
);

localparam logic [2:0] ACC_SEL = 3'd1;
localparam logic [2:0] S_IDLE = 3'd0;
localparam logic [2:0] S_BURST = 3'd1;
localparam logic [2:0] S_WORD = 3'd2;
localparam logic [2:0] S_MAG = 3'd3;
localparam logic [2:0] S_WAIT = 3'd4;
localparam logic [2:0] S_WRITE = 3'd5;

logic [2:0] state;
logic [2:0] modality;
logic [14:0] count;
logic [14:0] limit;
logic [1:0] axis;

logic byte_hi;
logic [7:0] byte_lo;
logic word_pending;

assign mask = '1;
assign sensor = modality;
assign addr = wptr;
assign burst_start = (state == S_BURST);
assign word_done = (state == S_WORD);
assign mag_start = (state == S_MAG);
assign wren = (state == S_WRITE);
assign idle = (state == S_IDLE) && !word_pending;
assign window_end = (state == S_WRITE) && (modality == 3'd4) && (count == limit - 1'b1);

always_comb begin
    case (modality)
        3'd0 : limit = 15'(N_ECG);
        3'd1 : limit = 15'(N_ACC);
        3'd2 : limit = 15'(N_RESP);
        3'd3 : limit = 15'(N_EDA);
        default : limit = 15'(N_EMG);
    endcase
end

always_ff @(posedge clk) begin
    if (rst) begin
        byte_hi <= 1'b0;
        word_pending <= 1'b0;
        sample <= '0;
    end
    else begin
        if (rx_valid) begin
            if (!byte_hi) begin
                byte_lo <= rx_data;
                byte_hi <= 1'b1;
            end
            else begin
                sample <= {rx_data, byte_lo};
                byte_hi <= 1'b0;
                word_pending <= 1'b1;
            end
        end
        if (state == S_IDLE && word_pending) word_pending <= 1'b0;
    end
end

always_ff @(posedge clk) begin
    if (rst) begin
        state <= S_IDLE;
        modality <= '0;
        count <= '0;
        axis <= '0;
    end
    else begin
        case (state)
            S_IDLE : if (word_pending) begin
                if (modality == ACC_SEL)
                    state <= (axis == 2'd0) ? S_BURST : S_WORD;
                else
                    state <= S_WRITE;
            end

            S_BURST : state <= S_WORD;

            S_WORD : if (axis == 2'd2) state <= S_MAG;
                     else begin
                         axis <= axis + 1'b1;
                         state <= S_IDLE;
                     end

            S_MAG : state <= S_WAIT;

            S_WAIT : if (mag_done) state <= S_WRITE;

            S_WRITE : begin
                axis <= '0;
                if (count == limit - 1'b1) begin
                    count <= '0;
                    modality <= (modality == 3'd4) ? 3'd0 : modality + 1'b1;
                end
                else count <= count + 1'b1;
                state <= S_IDLE;
            end

            default : state <= S_IDLE;
        endcase
    end
end

endmodule
