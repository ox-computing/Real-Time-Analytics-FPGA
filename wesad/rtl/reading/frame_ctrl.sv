module frame_ctrl (
    input logic clk, rst,
    input logic ingest_idle, frame_done, hop_en,
    input logic [4:0] window_full,
    input logic [10:0] offset_acc, offset_resp, offset_eda,
    input logic [13:0] offset_ecg,
    input logic [14:0] offset_emg,
    input logic [15:0] read_data,
    output logic armed, ingest_hold, head_request,
    output logic frame_start, frame_busy, overrun,
    output logic [10:0] off_acc_ff, off_resp_ff, off_eda_ff,
    output logic [13:0] off_ecg_ff,
    output logic [14:0] off_emg_ff,
    output logic [15:0] head_addr,
    output logic [15:0] head_ecg, head_acc, head_resp, head_eda, head_emg
);

localparam base_ecg = 0;
localparam base_acc = base_ecg + 15000;
localparam base_resp = base_acc + 1920;
localparam base_eda = base_resp + 1500;
localparam base_emg = base_eda + 1500;

logic [4:0] primed;
logic [2:0] cap;
logic tick_pending;
logic prime_now;

assign armed = &primed;
assign prime_now = ~armed & &(primed | window_full);
assign ingest_hold = tick_pending | (cap != 3'd6);
assign head_request = (cap < 3'd5);

assign head_addr = (cap == 3'd0) ? base_ecg + off_ecg_ff :
                   (cap == 3'd1) ? base_acc + off_acc_ff :
                   (cap == 3'd2) ? base_resp + off_resp_ff :
                   (cap == 3'd3) ? base_eda + off_eda_ff :
                                   base_emg + off_emg_ff;

always @(posedge clk) begin
    if (rst) begin
        primed <= '0;
        cap <= 3'd6;
        tick_pending <= 1'b0;
        frame_start <= 1'b0;
        frame_busy <= 1'b0;
        overrun <= 1'b0;
    end
    else begin
        primed <= primed | window_full;
        frame_start <= (cap == 3'd5);

        if (hop_en | prime_now) begin
            tick_pending <= 1'b1;
            overrun <= overrun | frame_busy | tick_pending;
        end

        if (tick_pending & ingest_idle & ~frame_busy) begin
            tick_pending <= 1'b0;
            off_ecg_ff <= offset_ecg;
            off_acc_ff <= offset_acc;
            off_resp_ff <= offset_resp;
            off_eda_ff <= offset_eda;
            off_emg_ff <= offset_emg;
            cap <= 3'd0;
            frame_busy <= 1'b1;
        end

        if (cap != 3'd6) begin
            cap <= cap + 1'b1;
            case (cap)
                3'd1: head_ecg <= read_data;
                3'd2: head_acc <= read_data;
                3'd3: head_resp <= read_data;
                3'd4: head_eda <= read_data;
                3'd5: head_emg <= read_data;
                default: ;
            endcase
        end

        if (frame_done) frame_busy <= 1'b0;
    end
end

endmodule
