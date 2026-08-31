// Data 0: ECG
// Data 1: Acc
// Data 2: Resp
// Data 3: EDA

module spram_addr_pointer (
    input logic clk,
    input logic rst,
    input logic [2:0] sensor, // spi_ingest_control.sensor
    input logic wren, // spi_ingest_control.wren
    output logic [15:0] wptr, // comb: address of the write
    output logic [13:0] offset_ecg, // 0 .. 16383
    output logic [10:0] offset_acc, // 0 .. 5759
    output logic [10:0] offset_resp, // 0 .. 1499
    output logic [10:0] offset_eda, // 0 .. 1499
    output logic [14:0] offset_emg,
    output logic [4:0] window_full // one-cycle pulse on wrap, per region
);

// Lengths in terms of samples per 60 s window for each modality
localparam len_ecg = 15000;

localparam len_acc = 1920;
localparam len_resp = 1500;
localparam len_eda = 1500;
localparam len_emg = 21000;


// Base addresses of modalities
localparam base_ecg = 0;
localparam base_acc = base_ecg + len_ecg;
localparam base_resp = base_acc + len_acc;
localparam base_eda = base_resp + len_resp;
localparam base_emg = base_eda + len_eda;

logic last_ecg, last_acc, last_resp, last_eda, last_emg;

assign last_ecg = (offset_ecg == len_ecg - 1);
assign last_acc = (offset_acc == len_acc - 1);
assign last_resp = (offset_resp == len_resp - 1);
assign last_eda = (offset_eda == len_eda - 1);
assign last_emg = (offset_emg == len_emg - 1);

initial begin
    offset_ecg = '0;
    offset_acc = '0;
    offset_resp = '0;
    offset_eda = '0;
    offset_emg = '0;
    window_full = '0;
end

always @(posedge clk) begin
    window_full <= '0;

    if (rst) begin
        offset_ecg <= '0;
        offset_acc <= '0;
        offset_resp <= '0;
        offset_eda <= '0;
        offset_emg <= '0;
    end
    else if (wren) begin
        unique case (sensor)
            3'd0: begin
                offset_ecg <= last_ecg ? '0 : offset_ecg + 1;
                window_full[0] <= last_ecg;
            end
            3'd1: begin
                offset_acc <= last_acc ? '0 : offset_acc + 1;
                window_full[1] <= last_acc;
            end
            3'd2: begin
                offset_resp <= last_resp ? '0 : offset_resp + 1;
                window_full[2] <= last_resp;
            end
            3'd3: begin
                offset_eda <= last_eda ? '0 : offset_eda + 1;
                window_full[3] <= last_eda;
            end
            3'd4: begin
                offset_emg <= last_emg ? '0 : offset_emg + 1;
                window_full[4] <= last_emg;
            end
        endcase
    end
end

logic [15:0] region_base;
logic [14:0] region_offset;

assign region_base = (sensor == 3'd0) ? base_ecg :
                     (sensor == 3'd1) ? base_acc :
                     (sensor == 3'd2) ? base_resp :
                     (sensor == 3'd3) ? base_eda : base_emg;

assign region_offset = (sensor == 3'd0) ? {1'b0, offset_ecg} :
                       (sensor == 3'd1) ? {4'b0, offset_acc} :
                       (sensor == 3'd2) ? {4'b0, offset_resp} :
                       (sensor == 3'd3) ? {4'b0, offset_eda} : offset_emg;

assign wptr = region_base + {1'b0, region_offset};

endmodule
