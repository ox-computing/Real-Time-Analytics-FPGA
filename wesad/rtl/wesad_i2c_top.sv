module wesad_i2c_top #(parameter CLK_FREQ = 12_000_000, parameter [6:0] I2C_ADDR = 7'h40)(
    input logic clk,
    input logic scl,
    inout wire sda
);

logic rst;
logic [3:0] por;

initial begin
    por = '0;
    rst = 1'b1;
end

always_ff @(posedge clk) begin
    if (por != 4'hF) begin
        por <= por + 1'b1;
        rst <= 1'b1;
    end
    else rst <= 1'b0;
end

localparam ACC_SEL = 3'd1;

logic [1:0] class_idx;
logic result_valid;

logic ingest_idle;
logic window_end;
logic [7:0] rx_byte;
logic rx_valid;
logic [7:0] tx_byte;
logic tx_load;
logic sda_oe;

logic burst_start;
logic word_done;
logic mag_start;
logic mag_done;
logic [15:0] mag_value;
logic [15:0] spram_din;
logic [15:0] sensor_write_data;

logic [15:0] sensor_write_addr;
logic wren_spram;
logic [15:0] read_data;

logic [15:0] spram_addr, spram_data;
logic spram_wren;
logic [3:0] spram_mask;

logic head_request, grant, read_request;
logic [15:0] head_addr, read_addr;

assign sda = sda_oe ? 1'b0 : 1'bz;

i2c_slave #(.ADDR(I2C_ADDR)) phy (
    .clk,
    .rst,
    .scl_i (scl),
    .sda_i (sda),
    .sda_oe,
    .rx_data (rx_byte),
    .rx_valid,
    .tx_data (tx_byte),
    .tx_load
);

spram_arbiter arb (
    .wr_en(wren_spram),
    .wr_addr(sensor_write_addr),
    .wr_data(spram_din),
    .head_request,
    .head_addr,
    .read_request,
    .read_addr,
    .spram_addr,
    .spram_data,
    .spram_wren,
    .spram_mask,
    .grant
);

input_buffer_spram spram (
    .clk,
    .rst,
    .addr(spram_addr),
    .data_in(spram_data),
    .wren(spram_wren),
    .mask(spram_mask),
    .data_out(read_data)
);

logic [2:0] sensor;
logic [15:0] wptr;
logic [13:0] offset_ecg;
logic [10:0] offset_acc, offset_resp, offset_eda;
logic [14:0] offset_emg;
logic [4:0] window_full;

spram_addr_pointer pointer (
    .clk,
    .rst,
    .sensor,
    .wren(wren_spram),
    .wptr,
    .offset_ecg,
    .offset_acc,
    .offset_resp,
    .offset_eda,
    .offset_emg,
    .window_full
);

assign spram_din = (sensor == ACC_SEL) ? mag_value : sensor_write_data;

logic [31:0] acc_sumsq;
logic [15:0] isqrt_root;
logic isqrt_done, isqrt_busy;

acc_magnitude acc_mag (
    .clk,
    .rst,
    .burst_start,
    .word_done,
    .sample(sensor_write_data),
    .sumsq(acc_sumsq),
    .root(isqrt_root),
    .isqrt_done,
    .mag(mag_value),
    .mag_done
);

isqrt32 root_unit (
    .clk,
    .rst,
    .start(mag_start),
    .x(acc_sumsq),
    .root(isqrt_root),
    .done(isqrt_done),
    .busy(isqrt_busy)
);

i2c_ingest_control #(.N_ECG(250), .N_ACC(32), .N_RESP(25), .N_EDA(25), .N_EMG(350)) ingest (
    .clk,
    .rst,
    .rx_data (rx_byte),
    .rx_valid,
    .wptr,
    .mag_done,
    .burst_start,
    .word_done,
    .mag_start,
    .sensor,
    .sample (sensor_write_data),
    .addr (sensor_write_addr),
    .wren (wren_spram),
    .mask (),
    .idle (ingest_idle),
    .window_end
);

logic armed, hop_en, ingest_hold;
logic frame_start, frame_busy, overrun;
logic [13:0] off_ecg_ff;
logic [10:0] off_acc_ff, off_resp_ff, off_eda_ff;
logic [14:0] off_emg_ff;
logic [15:0] head_ecg, head_acc, head_resp, head_eda, head_emg;

logic reader_busy, stream_valid, stream_last;
logic [15:0] stream_data;

logic frame_seq_busy;

logic stats_pass, fft_busy;

logic frame_done, start_reading, stream_ready;
logic reader_ready;
logic [2:0] sensor_select;
logic [14:0] start_index, sample_count;

logic time_stats_busy, pass_start, pass_done, phase, features_ready;
logic first_segment, last_segment;
logic psd_busy;

assign features_ready = ~time_stats_busy & ~fft_busy & ~psd_busy;
// segment position within the current modality's run, for the power accumulator
assign first_segment = (start_index == 15'd0);
assign last_segment = (sensor_select == 3'd4) ? (start_index == 15'd18432)
                    : (sensor_select == 3'd0) ? (start_index == 15'd12288)
                    : 1'b1;



assign hop_en = window_end;

frame_ctrl frame (
    .clk,
    .rst,
    .ingest_idle,
    .frame_done,
    .hop_en,
    .window_full,
    .offset_acc,
    .offset_resp,
    .offset_eda,
    .offset_ecg,
    .offset_emg,
    .read_data,
    .armed,
    .ingest_hold,
    .head_request,
    .frame_start,
    .frame_busy,
    .overrun,
    .off_acc_ff,
    .off_resp_ff,
    .off_eda_ff,
    .off_ecg_ff,
    .off_emg_ff,
    .head_addr,
    .head_ecg,
    .head_acc,
    .head_resp,
    .head_eda,
    .head_emg
);

window_reader reader (
    .clk,
    .rst,
    .off_ecg_ff,
    .off_acc_ff,
    .off_resp_ff,
    .off_eda_ff,
    .off_emg_ff,
    .head_ecg,
    .head_acc,
    .head_resp,
    .head_eda,
    .head_emg,
    .start_reading,
    .sensor_select,
    .start_index,
    .sample_count,
    .busy(reader_busy),
    .read_request,
    .addr(read_addr),
    .grant,
    .read_data,
    .stream_data,
    .stream_valid,
    .stream_last,
    .stream_ready (reader_ready)
);

assign reader_ready = stream_ready;

frame_seq frame_seq (
    .clk,
    .rst,
    .frame_start,
    .frame_done,
    .stream_last,
    .stream_valid,
    .start_reading,
    .sensor_select,
    .start_index,
    .sample_count,
    .stream_ready,
    .pass_start,
    .pass_done,
    .features_ready,
    .phase,
    .busy(frame_seq_busy),
    .stats_pass,
    .fft_busy
);

// the PSD bandwidth radicand needs 36 bits; the other callers zero-extend into it
logic [35:0] feat_x;
logic [17:0] feat_root;
logic feat_start, feat_done, feat_busy;

logic ts_wr_en, psd_wr_en;
logic [6:0] ts_wr_addr, psd_wr_addr;
logic [7:0] ts_wr_data, psd_wr_data;
logic feature_wr_en;
logic [6:0] feature_wr_addr;
logic [7:0] feature_wr_data;

// only one engine writes at a time: time in phase 0, spectral after
assign feature_wr_en = ts_wr_en | psd_wr_en;
assign feature_wr_addr = ts_wr_en ? ts_wr_addr : psd_wr_addr;
assign feature_wr_data = ts_wr_en ? ts_wr_data : psd_wr_data;

logic fft_exp_valid, fft_bin_valid, fft_spectrum_done;
logic [10:0] fft_bin_address;
logic signed [7:0] fft_bin_real, fft_bin_imaginary;
logic signed [6:0] fft_segment_exponent;
logic [11:0] segment_length;
logic [1:0] hann_select;
logic [15:0] acc_rd_data;
logic signed [5:0] acc_power_exponent;
logic acc_rd_en;
logic [10:0] acc_rd_addr;

logic ts_sqrt_start, psd_sqrt_start;
logic ts_sqrt_done, psd_sqrt_done;
logic ts_sqrt_busy, psd_sqrt_busy;
logic psd_div_done, psd_div_busy;
logic div_go;
logic [39:0] div_num;
logic [23:0] div_den;
logic [3:0] div_frac;
logic [31:0] ts_sqrt_data;
logic [35:0] psd_sqrt_data;
logic psd_div_start;
logic [39:0] psd_div_num;
logic [23:0] psd_div_den;
logic [3:0] psd_div_frac;
logic [39:0] div_quotient;
logic div_done, div_busy;
logic [23:0] log_value;
logic [12:0] log_result;

logic [6:0] feature_addr;
logic [1:0] threshold_addr;
logic [7:0] feature_data, threshold_data;

logic [459:0] binary_vector;
logic [50:0] core_bits;
logic [14:0] scores;
logic encode_done, gs_start, gs_done;
logic [2:0] settle;

logic signed [15:0] ts_mul_x, ts_mul_y;
logic signed [7:0] pre_mul_x, pre_mul_y;
logic [20:0] psd_mul_a;
logic [15:0] psd_mul_b;
logic [16:0] fft_bin_power;

logic signed [15:0] mul_t_x, mul_t_y;
logic signed [31:0] mul_t_p;
logic signed [21:0] mul_s_x;
logic signed [16:0] mul_s_y;
logic signed [38:0] mul_s_p;

assign mul_t_x = fft_bin_valid ? {{8{fft_bin_real[7]}}, fft_bin_real}
               : phase ? {{8{pre_mul_x[7]}}, pre_mul_x} : ts_mul_x;
assign mul_t_y = fft_bin_valid ? {{8{fft_bin_real[7]}}, fft_bin_real}
               : phase ? {{8{pre_mul_y[7]}}, pre_mul_y} : ts_mul_y;
assign mul_t_p = mul_t_x * mul_t_y;

assign mul_s_x = fft_bin_valid ? {{14{fft_bin_imaginary[7]}}, fft_bin_imaginary}
               : {1'b0, psd_mul_a};
assign mul_s_y = fft_bin_valid ? {{9{fft_bin_imaginary[7]}}, fft_bin_imaginary}
               : {1'b0, psd_mul_b};
assign mul_s_p = mul_s_x * mul_s_y;

assign fft_bin_power = mul_t_p[15:0] + mul_s_p[15:0];

time_stats stats (
    .clk,
    .rst,
    .sensor_select,
    .phase,
    .pass_start,
    .pass_done,
    .stream_data,
    .stream_last,
    .stream_valid,
    .busy(time_stats_busy),
    .sqrt_result(feat_root[15:0]),
    .sqrt_done(ts_sqrt_done),
    .sqrt_busy(ts_sqrt_busy),
    .sqrt_data(ts_sqrt_data),
    .sqrt_start(ts_sqrt_start),
    .mul_x (ts_mul_x),
    .mul_y (ts_mul_y),
    .mul_p (mul_t_p),
    .feature_wr_en (ts_wr_en),
    .feature_wr_addr (ts_wr_addr),
    .feature_wr_data (ts_wr_data)
);

assign segment_length = (sensor_select == 3'd1) ? 12'd1920
                      : (sensor_select == 3'd2 || sensor_select == 3'd3) ? 12'd1500
                      : 12'd2048;
assign hann_select = (segment_length == 12'd2048) ? 2'd0
                   : (segment_length == 12'd1920) ? 2'd1 : 2'd2;

fft_engine fft (
    .clk,
    .rst,
    .phase,
    .stats_pass,
    .pass_start,
    .pass_done,
    .stream_valid,
    .stream_data,
    .segment_length,
    .table_select (hann_select),
    .mul_x (pre_mul_x),
    .mul_y (pre_mul_y),
    .mul_p (mul_t_p[15:0]),
    .busy (fft_busy),
    .segment_exponent (fft_segment_exponent),
    .exponent_valid (fft_exp_valid),
    .bin_valid (fft_bin_valid),
    .bin_address (fft_bin_address),
    .bin_real (fft_bin_real),
    .bin_imaginary (fft_bin_imaginary),
    .spectrum_done (fft_spectrum_done)
);

psd_power_accumulator #(.BUTTERFLY_WIDTH(8), .ACC_WIDTH(16)) power (
    .clk,
    .rst,
    .start_segment (fft_exp_valid),
    .first_segment (first_segment),
    .segment_exponent (fft_segment_exponent[5:0]),
    .bin_valid (fft_bin_valid),
    .bin_address (fft_bin_address),
    .bin_power (fft_bin_power),
    .read_enable (acc_rd_en),
    .read_address (acc_rd_addr),
    .read_data (acc_rd_data),
    .power_exponent (acc_power_exponent)
);

psd_calc psd (
    .clk,
    .rst,
    .start (fft_spectrum_done & last_segment),
    .modality (sensor_select),
    .power_exponent ({acc_power_exponent[5], acc_power_exponent}),
    .acc_read_enable (acc_rd_en),
    .acc_read_address (acc_rd_addr),
    .acc_read_data (acc_rd_data),
    .div_start (psd_div_start),
    .div_numerator (psd_div_num),
    .div_denominator (psd_div_den),
    .div_frac (psd_div_frac),
    .div_quotient,
    .div_done (psd_div_done),
    .div_busy (psd_div_busy),
    .log_value,
    .log_result,
    .sqrt_start (psd_sqrt_start),
    .sqrt_data (psd_sqrt_data),
    .sqrt_result (feat_root),
    .sqrt_done (psd_sqrt_done),
    .sqrt_busy (psd_sqrt_busy),
    .mul_a (psd_mul_a),
    .mul_b (psd_mul_b),
    .product (mul_s_p[36:0]),
    .busy (psd_busy),
    .feature_wr_addr (psd_wr_addr),
    .feature_wr_en (psd_wr_en),
    .feature_wr_data (psd_wr_data)
);

divider shared_div (
    .clk,
    .rst,
    .start (div_go),
    .numerator (div_num),
    .denominator (div_den),
    .frac_bits (div_frac),
    .quotient (div_quotient),
    .done (div_done),
    .busy (div_busy)
);

log2_unit shared_log (
    .value (log_value),
    .result (log_result)
);

logic ts_sq_pend, psd_sq_pend;
logic [1:0] sq_owner;
logic [35:0] ts_sq_x, psd_sq_x;
logic sq_grant;

assign sq_grant = (sq_owner == 2'd0) & ~feat_busy & (ts_sq_pend | psd_sq_pend);
assign feat_start = sq_grant;
assign feat_x = ts_sq_pend ? ts_sq_x : psd_sq_x;

assign ts_sqrt_done = feat_done & (sq_owner == 2'd1);
assign psd_sqrt_done = feat_done & (sq_owner == 2'd2);
assign ts_sqrt_busy = (sq_owner != 2'd0) | ts_sq_pend;
assign psd_sqrt_busy = (sq_owner != 2'd0) | psd_sq_pend;

always_ff @(posedge clk) begin
    if (rst) begin
        ts_sq_pend <= 0;
        psd_sq_pend <= 0;
        sq_owner <= 0;
    end
    else begin
        if (ts_sqrt_start) begin
            ts_sq_pend <= 1;
            ts_sq_x <= {4'b0, ts_sqrt_data};
        end
        if (psd_sqrt_start) begin
            psd_sq_pend <= 1;
            psd_sq_x <= psd_sqrt_data;
        end
        if (sq_grant) begin
            if (ts_sq_pend) begin
                sq_owner <= 2'd1;
                ts_sq_pend <= 0;
            end
            else begin
                sq_owner <= 2'd2;
                psd_sq_pend <= 0;
            end
        end
        else if (feat_done) sq_owner <= 0;
    end
end

logic psd_dv_pend;
logic [1:0] dv_owner;
logic [39:0] psd_dv_num;
logic [23:0] psd_dv_den;
logic [3:0] psd_dv_frac;
logic dv_grant;

assign dv_grant = (dv_owner == 2'd0) & ~div_busy & psd_dv_pend;
assign div_go = dv_grant;
assign div_num = psd_dv_num;
assign div_den = psd_dv_den;
assign div_frac = psd_dv_frac;

assign psd_div_done = div_done & (dv_owner == 2'd1);
assign psd_div_busy = (dv_owner != 2'd0) | psd_dv_pend;

always_ff @(posedge clk) begin
    if (rst) begin
        psd_dv_pend <= 0;
        dv_owner <= 0;
    end
    else begin
        if (psd_div_start) begin
            psd_dv_pend <= 1;
            psd_dv_num <= psd_div_num;
            psd_dv_den <= psd_div_den;
            psd_dv_frac <= psd_div_frac;
        end
        if (dv_grant) begin
            dv_owner <= 2'd1;
            psd_dv_pend <= 0;
        end
        else if (div_done) dv_owner <= 0;
    end
end

isqrt32 #(.WIDTH(36)) feature_root_unit (
    .clk,
    .rst,
    .start(feat_start),
    .x(feat_x),
    .root(feat_root),
    .done(feat_done),
    .busy(feat_busy)
);

feature_threshold_storage #(.N_FEATURE(115), .FEATURE_WIDTH(8), .THRESHOLD_WIDTH(8)) storage (
    .clk,
    .wr_en(feature_wr_en),
    .wr_addr(feature_wr_addr),
    .wr_data(feature_wr_data),
    .feature_addr,
    .threshold_addr,
    .feature_data,
    .threshold_data
);

thermometer_encoder #(.N_FEATURE(115), .FEATURE_WIDTH(8), .THRESHOLD_WIDTH(8)) encoder (
    .clk,
    .rst,
    .frame_done,
    .feature_data,
    .feature_addr,
    .threshold_data,
    .threshold_addr,
    .binary_vector,
    .encode_done
);

dwn_core core (
    .clk,
    .in(binary_vector),
    .out(core_bits)
);

groupsum #(.NUM_CLASSES(3), .GROUP_SIZE(17), .SCORE_W(5)) gs (
    .clk,
    .start(gs_start),
    .bits(core_bits),
    .scores,
    .done(gs_done)
);

argmax #(.NUM_CLASSES(3), .SCORE_W(5), .IDX_W(2)) am (
    .clk,
    .scores,
    .class_idx
);

always_ff @(posedge clk) begin
    gs_start <= 0;
    result_valid <= 0;
    if (rst) settle <= 0;
    else begin
        result_valid <= gs_done;
        if (encode_done) settle <= 1;
        else if (settle != 0) begin
            settle <= settle + 1'b1;
            if (settle == 3'd7) begin
                settle <= 0;
                gs_start <= 1;
            end
        end
    end
end

i2c_result_tx result_sender (
    .clk,
    .rst,
    .result_valid,
    .class_idx,
    .scores,
    .tx_load,
    .tx_data (tx_byte)
);

endmodule
