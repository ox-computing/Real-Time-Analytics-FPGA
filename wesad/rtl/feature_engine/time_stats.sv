module time_stats (
    input logic clk, rst,

    // From frame_seq
    input logic [2:0] sensor_select,
    input logic phase,
    input logic pass_start, pass_done,

    // From window reader
    input logic signed [15:0] stream_data,
    input logic stream_last, stream_valid,

    // Back to frame_seq (contributes to features_ready)
    output logic busy,

    // Shared isqrt
    input logic [15:0] sqrt_result,
    input logic sqrt_done, sqrt_busy,
    output logic [31:0] sqrt_data,
    output logic sqrt_start,

    // Shared multiplier
    output logic signed [15:0] mul_x, mul_y,
    input logic signed [31:0] mul_p,

    // To thermometer
    output logic feature_wr_en,
    output logic [6:0] feature_wr_addr,
    output logic [7:0] feature_wr_data
);

logic signed [7:0] sample;
logic signed [23:0] mean_sum;
logic signed [29:0] sq_sum;
logic signed [37:0] pre_sum;
logic signed [7:0] max, min;
logic [2:0] sensor_select_ff;
logic [3:0] write_index;

logic [15:0] rms_result, std_result;
logic [43:0] variance;
logic signed [33:0] slope;

logic [14:0] window_length, window_length_minus_one;
logic [14:0] rank25, rank50, rank75;
logic signed [45:0] multiply_operand, multiply_accumulator;
logic [24:0] multiply_bits;

logic [23:0] absolute_mean;
logic signed [17:0] mean_for_slope;
logic signed [33:0] pre_for_slope;

logic signed [8:0] range_raw;
logic signed [7:0] median_raw;
logic signed [8:0] iqr_raw;
logic [23:0] mad_raw;

logic [3:0] table_index;
logic [5:0] narrow_shift;
logic signed [12:0] narrow_offset;
logic signed [47:0] narrow_source;
logic signed [47:0] shift_source;
logic signed [47:0] shift_register;
logic signed [47:0] narrow_shifted;
logic [5:0] shift_count;
logic [3:0] write_index_prev;
logic shift_load, shift_busy;
logic [3:0] state_prev;
logic [7:0] write_data;

// 256-bin tally over the sample alphabet: one streaming pass, then one sweep
logic [14:0] tally [0:255];
logic [7:0] tally_raddr;
logic [7:0] sweep_raddr;
logic [7:0] tally_waddr;
logic [14:0] tally_q;
logic [14:0] tally_din;
logic tally_we;
logic [7:0] bin_w, bin_w_prev, bin_w_prev2;
logic bin_pending, bin_pending_prev, bin_pending_prev2;
logic [14:0] tally_din_prev;
logic [7:0] sample_bin;
logic [14:0] tally_count;
logic [7:0] clear_addr;
logic initialised;

logic [14:0] cumulative;
logic [8:0] sweep_addr;
logic [7:0] bin_index;
logic sweep_active;
logic [7:0] p25, p50, p75;
logic found25, found50, found75;
logic [23:0] mad_accumulator;

typedef enum logic [3:0] {IDLE, CLEAR, ACC, MULT_SQ, MULT_MEANSQ, MULT_SLOPE, SWEEP, RMS, STD, WRITE} state_t;
state_t state, next_state;

assign sample = stream_data[15:8];
assign mul_x = sample;
assign mul_y = sample;
assign sample_bin = {~sample[7], sample[6:0]};
assign tally_raddr = (state == ACC) ? sample_bin : sweep_raddr;
assign absolute_mean = mean_sum[23] ? -mean_sum : mean_sum;
assign mean_for_slope = mean_sum >>> 14;
assign pre_for_slope = 34'(pre_sum >>> 13);
assign window_length_minus_one = window_length - 1'b1;
assign feature_wr_en = (state == WRITE) && !shift_busy;
assign shift_load = (state != state_prev) || (write_index != write_index_prev);
assign shift_busy = shift_load || (shift_count != 0);
assign feature_wr_addr = sensor_select_ff * 7'd10 + {3'b0, write_index};
assign feature_wr_data = write_data;
// the read misses the two writes still in flight, so both forward
assign tally_count = (bin_pending_prev && bin_w_prev == bin_w) ? tally_din
                   : (bin_pending_prev2 && bin_w_prev2 == bin_w) ? tally_din_prev
                   : tally_q;
assign table_index = (state == RMS) ? 4'd2 : (state == STD) ? 4'd1 : write_index;

always_ff @(posedge clk) begin
    state_prev <= state;
    write_index_prev <= write_index;
    if (shift_load) begin
        shift_register <= shift_source;
        shift_count <= narrow_shift;
    end
    else if (shift_count != 0) begin
        shift_register <= shift_register >>> 1;
        shift_count <= shift_count - 1'b1;
    end
end

always_ff @(posedge clk) begin
    tally_q <= tally[tally_raddr];
    if (tally_we) tally[tally_waddr] <= tally_din;
end

always_comb begin
    case (sensor_select_ff)
        3'd0: begin window_length = 15000; rank25 = 3750; rank50 = 7500; rank75 = 11250; end
        3'd1: begin window_length = 1920; rank25 = 480; rank50 = 960; rank75 = 1440; end
        3'd2, 3'd3: begin window_length = 1500; rank25 = 375; rank50 = 750; rank75 = 1125; end
        default: begin window_length = 21000; rank25 = 5250; rank50 = 10500; rank75 = 15750; end
    endcase

    case (write_index)
        4'd0: narrow_source = {{24{mean_sum[23]}}, mean_sum};
        4'd5: narrow_source = {{39{range_raw[8]}}, range_raw};
        4'd6: narrow_source = {{40{median_raw[7]}}, median_raw};
        4'd7: narrow_source = {{39{iqr_raw[8]}}, iqr_raw};
        4'd8: narrow_source = {24'b0, mad_raw};
        default: narrow_source = {{14{slope[33]}}, slope};
    endcase

    case (state)
        RMS : shift_source = {18'b0, $unsigned(sq_sum)};
        STD : shift_source = {4'b0, variance};
        default : shift_source = narrow_source;
    endcase

    narrow_shifted = shift_register;

    case (write_index)
        4'd1: write_data = std_result - narrow_offset;
        4'd2: write_data = rms_result - narrow_offset;
        4'd3: write_data = min;
        4'd4: write_data = max;
        default: write_data = narrow_shifted - narrow_offset;
    endcase
end

always_comb begin
    `include "narrowing.vh"
end

always_comb begin
    next_state = state;
    busy = (state != IDLE) || !initialised;
    sqrt_start = (state == RMS || state == STD) & ~sqrt_busy & ~sqrt_done & ~shift_busy;
    sqrt_data = narrow_shifted[31:0];

    case (state)
        IDLE: if (!initialised) next_state = CLEAR;
              else if (~phase & pass_start) next_state = ACC;
        CLEAR: if (clear_addr == 8'hFF) next_state = IDLE;
        ACC: if (pass_done) next_state = MULT_SQ;
        MULT_SQ: if (multiply_bits == 0) next_state = MULT_MEANSQ;
        MULT_MEANSQ: if (multiply_bits == 0) next_state = MULT_SLOPE;
        MULT_SLOPE: if (multiply_bits == 0) next_state = SWEEP;
        SWEEP: if (sweep_addr == 9'd256) next_state = RMS;
        RMS: if (sqrt_done) next_state = STD;
        STD: if (sqrt_done) next_state = WRITE;
        WRITE: if (write_index == 4'd9 && !shift_busy) next_state = IDLE;
        default: next_state = IDLE;
    endcase
end

always_ff @( posedge clk ) begin
    if (rst) begin
        state <= IDLE;
        write_index <= 0;
        tally_we <= 0;
        bin_pending <= 0;
        clear_addr <= 0;
        initialised <= 0;
    end
    else begin
        state <= next_state;
        tally_we <= 0;

        if (state == IDLE && ~phase && pass_start) begin
            sensor_select_ff <= sensor_select;
            mean_sum <= 0;
            sq_sum <= 0;
            pre_sum <= 0;
            max <= 8'sh80;
            min <= 8'sh7F;
            bin_pending <= 0;
            bin_pending_prev <= 0;
            bin_pending_prev2 <= 0;
        end

        // the sweep clears behind itself, so this only covers power-on
        if (state == CLEAR) begin
            tally_waddr <= clear_addr;
            tally_din <= 0;
            tally_we <= 1;
            clear_addr <= clear_addr + 1'b1;
            if (clear_addr == 8'hFF) initialised <= 1;
        end

        if (state == ACC) begin
            bin_w <= sample_bin;
            bin_pending <= stream_valid;
            bin_w_prev <= bin_w;
            bin_pending_prev <= bin_pending;
            bin_w_prev2 <= bin_w_prev;
            bin_pending_prev2 <= bin_pending_prev;
            tally_din_prev <= tally_din;
            if (bin_pending) begin
                tally_waddr <= bin_w;
                tally_din <= tally_count + 1'b1;
                tally_we <= 1;
            end
        end

        if (state == ACC && stream_valid) begin
            mean_sum <= mean_sum + sample;
            sq_sum <= sq_sum + mul_p;
            pre_sum <= pre_sum + mean_sum;
            max <= (sample > max) ? sample : max;
            min <= (sample < min) ? sample : min;
        end

        if (state == ACC && pass_done) begin
            range_raw <= {max[7], max} - {min[7], min};
            multiply_accumulator <= 0;
            multiply_operand <= {{16{sq_sum[29]}}, sq_sum};
            multiply_bits <= {10'b0, window_length};
        end

        if ((state == MULT_SQ || state == MULT_MEANSQ || state == MULT_SLOPE) && multiply_bits != 0) begin
            if (multiply_bits[0]) multiply_accumulator <= multiply_accumulator + multiply_operand;
            multiply_operand <= multiply_operand << 1;
            multiply_bits <= multiply_bits >> 1;
        end

        if (state == MULT_SQ && multiply_bits == 0) begin
            multiply_operand <= -$signed({22'b0, absolute_mean});
            multiply_bits <= {1'b0, absolute_mean};
        end

        if (state == MULT_MEANSQ && multiply_bits == 0) begin
            variance <= multiply_accumulator[43:0];
            multiply_accumulator <= -pre_for_slope;
            multiply_operand <= {{28{mean_for_slope[17]}}, mean_for_slope};
            multiply_bits <= {10'b0, window_length_minus_one};
        end

        if (state == MULT_SLOPE && multiply_bits == 0) begin
            slope <= multiply_accumulator[33:0];
            sweep_addr <= 0;
            sweep_raddr <= 0;
            sweep_active <= 0;
            cumulative <= 0;
            mad_accumulator <= 0;
            found25 <= 0;
            found50 <= 0;
            found75 <= 0;
        end

        // one sweep latches the three ranks and mad, and clears the tally behind it
        if (state == SWEEP) begin
            sweep_raddr <= sweep_raddr + 1'b1;
            bin_index <= sweep_raddr;
            tally_waddr <= sweep_raddr;
            tally_din <= 0;
            tally_we <= 1;
            sweep_active <= 1;
            if (sweep_active) begin
                sweep_addr <= sweep_addr + 1'b1;
                cumulative <= cumulative + tally_q;
                if (!found25 && (cumulative + tally_q) >= rank25) begin
                    p25 <= bin_index;
                    found25 <= 1;
                end
                if (!found50 && (cumulative + tally_q) >= rank50) begin
                    p50 <= bin_index;
                    found50 <= 1;
                end
                if (!found75 && (cumulative + tally_q) >= rank75) begin
                    p75 <= bin_index;
                    found75 <= 1;
                end
                // sum(cum) below the median bin, sum(n - cum) at and above it
                if ((cumulative + tally_q) < rank50)
                    mad_accumulator <= mad_accumulator + (cumulative + tally_q);
                else
                    mad_accumulator <= mad_accumulator + ({9'b0, window_length} - (cumulative + tally_q));
            end
        end

        if (state == SWEEP && sweep_addr == 9'd256) begin
            median_raw <= {~p50[7], p50[6:0]};
            iqr_raw <= {1'b0, p75} - {1'b0, p25};
            mad_raw <= mad_accumulator;
        end

        if (state == RMS && sqrt_done) rms_result <= sqrt_result;

        if (state == STD && sqrt_done) begin
            std_result <= sqrt_result;
            write_index <= 0;
        end

        if (state == WRITE && !shift_busy) write_index <= write_index + 1'b1;
    end
end

endmodule
