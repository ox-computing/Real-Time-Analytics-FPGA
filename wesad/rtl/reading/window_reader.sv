// Streams one modality's 60 s window in time order, oldest sample first.
//
// Sensor encoding matches spi_ingest_control: 0 ECG, 1 ACC, 2 Resp, 3 EDA, 4 EMG.
//
// The window is defined entirely by the snapshot offsets frame_ctrl latches on the
// frame tick; this module never reads a live write pointer. start_index is a window
// index (0 = oldest sample), Welch segment k passes 1024*k.
//
// head_* carries each region's window index 0, captured at the tick while ingest
// was held. Index 0 is the one sample the writer can reach before the reader does,
// so it is served from that register rather than from the SPRAM.

module window_reader (
    input logic clk, rst,
    // Frozen window description from frame_ctrl, held for the whole frame
    input logic [13:0] off_ecg_ff,
    input logic [10:0] off_acc_ff, off_resp_ff, off_eda_ff,
    input logic [14:0] off_emg_ff,
    input logic [15:0] head_ecg, head_acc, head_resp, head_eda, head_emg,
    // Request
    input logic start_reading,
    input logic [2:0] sensor_select,
    input logic [14:0] start_index, // window index, 0 = oldest
    input logic [14:0] sample_count,
    output logic busy,
    // SPRAM side, through the arbiter
    output logic read_request,
    output logic [15:0] addr,
    input logic grant, // arbiter: this address reached the memory
    input logic [15:0] read_data, // registered, one cycle behind its address
    // Stream
    output logic [15:0] stream_data,
    output logic stream_valid,
    output logic stream_last,
    input logic stream_ready // Backpressure from feature engine for starting new read cycle
);

localparam len_ecg = 15000;
localparam len_acc = 1920;
localparam len_resp = 1500;
localparam len_eda = 1500;
localparam len_emg = 21000;

localparam base_ecg = 0;
localparam base_acc = base_ecg + len_ecg;
localparam base_resp = base_acc + len_acc;
localparam base_eda = base_resp + len_resp;
localparam base_emg = base_eda + len_eda;

localparam last_ecg = base_ecg + len_ecg - 1; // 14999
localparam last_acc = base_acc + len_acc - 1; // 16919
localparam last_resp = base_resp + len_resp - 1; // 18419
localparam last_eda = base_eda + len_eda - 1; // 19919
localparam last_emg = base_emg + len_emg - 1; // 40919

logic [2:0] sensor_select_ff;
logic [14:0] samples_left;
logic first_sample;
logic serving_head_ff;

logic [2:0] active_sensor;
logic [14:0] window_len, oldest_offset;
logic [15:0] region_base, region_last_addr, head_sample;
logic [15:0] first_offset_unwrapped, first_addr;

// Live select during accept (busy still low), latched select for the rest of the pass
assign active_sensor = busy ? sensor_select_ff : sensor_select;

assign window_len = (active_sensor == 3'd0) ? len_ecg :
                    (active_sensor == 3'd1) ? len_acc :
                    (active_sensor == 3'd2) ? len_resp :
                    (active_sensor == 3'd3) ? len_eda : len_emg;

assign region_base = (active_sensor == 3'd0) ? base_ecg :
                     (active_sensor == 3'd1) ? base_acc :
                     (active_sensor == 3'd2) ? base_resp :
                     (active_sensor == 3'd3) ? base_eda : base_emg;

assign region_last_addr = region_base + {1'b0, window_len} - 1'b1;

assign oldest_offset = (active_sensor == 3'd0) ? off_ecg_ff :
                       (active_sensor == 3'd1) ? off_acc_ff :
                       (active_sensor == 3'd2) ? off_resp_ff :
                       (active_sensor == 3'd3) ? off_eda_ff : off_emg_ff;

assign head_sample = (active_sensor == 3'd0) ? head_ecg :
                     (active_sensor == 3'd1) ? head_acc :
                     (active_sensor == 3'd2) ? head_resp :
                     (active_sensor == 3'd3) ? head_eda : head_emg;


assign first_offset_unwrapped = oldest_offset + start_index;
assign first_addr = region_base + ((first_offset_unwrapped >= window_len)
                                   ? first_offset_unwrapped - window_len
                                   : first_offset_unwrapped);

assign read_request = busy & (samples_left != 0) & stream_ready;

assign stream_data = serving_head_ff ? head_sample : read_data;

always @(posedge clk) begin
    if (rst) begin
        busy <= 1'b0;
        stream_valid <= 1'b0;
        addr <= '0;
    end
    else begin
        if (start_reading & ~busy) begin
            addr <= first_addr;
            sensor_select_ff <= sensor_select;
            samples_left <= sample_count;
            first_sample <= (start_index == 0);
            busy <= 1'b1;
        end

        stream_valid <= read_request & grant;
        stream_last <= read_request & grant & (samples_left == 1);

        if (read_request & grant) begin
            addr <= (addr == region_last_addr) ? region_base : addr + 1'b1;
            samples_left <= samples_left - 1'b1;
            first_sample <= 1'b0;
            serving_head_ff <= first_sample;
        end

        if (busy & (samples_left == 0)) busy <= 1'b0;
    end
end

endmodule
