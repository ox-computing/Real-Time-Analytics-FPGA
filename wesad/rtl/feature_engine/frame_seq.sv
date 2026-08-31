module frame_seq (
    input logic clk, rst,

    // Handshake with frame_ctrl
    input logic frame_start,
    output logic frame_done,

    // Window_reader
    input logic stream_last,
    input logic stream_valid,
    output logic start_reading,
    output logic [2:0] sensor_select, // Labels the current sensor data being broadcast. Feature engines also self-select on it.
    output logic [14:0] start_index,
    output logic [14:0] sample_count,
    output logic stream_ready, // Per sample

    // Broadcast to various feature engines
    output logic pass_start, pass_done,  // Pass over window over (clear accumulators), latch results to feature storage registers
    input logic features_ready,// AND of all participating engines not busy flag.
    output logic phase,

    output logic busy,

    // To and from FFT engine (double buffer design, fft_buffer selects which buffer the reader will send the data to to be written while other buffer has an in place FFT)
    output logic stats_pass,
    input logic fft_busy
);

typedef enum logic [2:0] {IDLE, ISSUE, WAIT, ADVANCE, DONE} state_t;
state_t state, next_state;

logic [2:0] modality_idx;
logic [4:0] seg_idx;
logic [4:0] seg_last;
logic last_pass;

assign seg_last = (modality_idx == 3'd0) ? 5'd18 : (modality_idx == 3'd1) ? 5'd12 : 5'd0;
assign last_pass = phase & ~stats_pass & (modality_idx == 3'd4) & (seg_idx == seg_last);

always_comb begin
    case (modality_idx)
        3'd0: begin sensor_select = 3'd4; sample_count = phase ? 15'd2048 : 15'd21000; end
        3'd1: begin sensor_select = 3'd0; sample_count = phase ? 15'd2048 : 15'd15000; end
        3'd2: begin sensor_select = 3'd1; sample_count = 15'd1920; end
        3'd3: begin sensor_select = 3'd2; sample_count = 15'd1500; end
        default: begin sensor_select = 3'd3; sample_count = 15'd1500; end
    endcase
    start_index = phase ? {seg_idx, 10'd0} : 15'd0;
end

always_comb begin
    next_state = state;
    busy = (state != IDLE);
    start_reading = 0;
    stream_ready = 0;
    pass_start = 0;
    pass_done = 0;
    frame_done = 0;

    case (state)
        IDLE:
            if (frame_start) next_state = ISSUE;

        ISSUE:
            if (!(phase & fft_busy)) begin
                start_reading = 1;
                pass_start = 1;
                next_state = WAIT;
            end

        WAIT: begin
            stream_ready = 1;
            if (stream_valid & stream_last) next_state = ADVANCE;
        end

        ADVANCE: begin
            pass_done = 1;
            if (features_ready) begin
                if (last_pass) next_state = DONE;
                else next_state = ISSUE;
            end
        end

        DONE: begin
            frame_done = 1;
            next_state = IDLE;
        end

        default: next_state = IDLE;
    endcase
end

always_ff @(posedge clk) begin
    if (rst) begin
        state <= IDLE;
        modality_idx <= 0;
        phase <= 0;
        seg_idx <= 0;
        stats_pass <= 1;
    end
    else begin
        state <= next_state;

        if (state == IDLE && frame_start) begin
            modality_idx <= 0;
            phase <= 0;
            seg_idx <= 0;
            stats_pass <= 1;
        end

        if (state == ADVANCE && features_ready && !last_pass) begin
            if (!phase) begin
                if (modality_idx == 3'd4) begin
                    phase <= 1;
                    modality_idx <= 0;
                    seg_idx <= 0;
                end
                else modality_idx <= modality_idx + 1'b1;
            end
            // each segment is read twice: once for its statistics, once windowed
            else if (stats_pass) stats_pass <= 0;
            else begin
                stats_pass <= 1;
                if (seg_idx == seg_last) begin
                    seg_idx <= 0;
                    modality_idx <= modality_idx + 1'b1;
                end
                else seg_idx <= seg_idx + 1'b1;
            end
        end
    end
end

endmodule
