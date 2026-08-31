// ECG heart-rate variability: 5-15 Hz cascade, streaming R-peak detector, and the eight RR statistics.

module hrv_features (
    input logic clk, rst,
    input logic sample_valid,
    input logic signed [15 : 0] sample,
    input logic window_start,
    input logic window_end,

    output logic div_start,
    output logic [39 : 0] div_numerator,
    output logic [23 : 0] div_denominator,
    output logic [3 : 0] div_frac,
    input logic [39 : 0] div_quotient,
    input logic div_done,

    output logic sqrt_start,
    output logic [31 : 0] sqrt_data,
    input logic [15 : 0] sqrt_result,
    input logic sqrt_done,

    output logic signed [31 : 0] mul_x,
    output logic signed [15 : 0] mul_y,
    input logic signed [47 : 0] mul_p,

    output logic busy,
    output logic ready,
    output logic feature_wr_en,
    output logic [2 : 0] feature_index,
    output logic [7 : 0] feature_word
);

localparam int FS = 250;
localparam int REFRACTORY = 100;
localparam int PNN50_SAMPLES = 13;

// --- 5-15 Hz cascade: four biquads, b = +/-[1 2 1], a in Q14 ----------------
logic signed [23 : 0] w1 [0 : 3];
logic signed [23 : 0] w2 [0 : 3];
logic [1 : 0] section;
logic signed [23 : 0] stage_in;
logic signed [23 : 0] w0;
logic signed [26 : 0] stage_sum;
logic signed [23 : 0] stage_out;
logic signed [15 : 0] a1, a2;
logic [1 : 0] biquad_step;
logic biquad_run;
logic signed [23 : 0] filtered;
logic filtered_valid;
logic pending;
logic signed [15 : 0] pending_sample;
logic signed [47 : 0] feedback;

// one multiplier for the cascade, the RR squares and the variance products
always_comb begin
    case (section)
        2'd0 : begin a1 = -16'sd27486; a2 = 16'sd12232; end
        2'd1 : begin a1 = -16'sd29701; a2 = 16'sd13745; end
        2'd2 : begin a1 = -16'sd28590; a2 = 16'sd14238; end
        default : begin a1 = -16'sd31694; a2 = 16'sd15577; end
    endcase
    w0 = stage_in - 24'((feedback + mul_p) >>> 14);
    // the tap sum needs 26 bits: truncating to 24 before the shift clips real peaks
    stage_sum = (section < 2) ? (27'(w0) + (27'(w1[section]) <<< 1) + 27'(w2[section]))
                              : (27'(w0) - (27'(w1[section]) <<< 1) + 27'(w2[section]));
    stage_out = 24'(stage_sum >>> 3);
end

// --- amplitude gate and refractory ------------------------------------------
// the accumulator holds 256x the mean magnitude, so it needs 8 bits above the sample
logic [31 : 0] ewma;
logic [24 : 0] gate;
logic [23 : 0] magnitude;
logic signed [23 : 0] y1, y2;
logic [24 : 0] thr1;
logic [15 : 0] sample_index;
logic [15 : 0] idx1;
logic pend_valid;
logic [15 : 0] pend_idx;
logic signed [23 : 0] pend_val;
logic last_valid;
logic [15 : 0] last_idx;
logic is_candidate;
logic [23 : 0] ewma_mean;
logic signed [25 : 0] y1_wide;

// --- RR statistics ----------------------------------------------------------
logic [7 : 0] rr_count;
logic [15 : 0] rr_sum;
logic [27 : 0] rr_sqsum;
logic [15 : 0] rr_min, rr_max, rr_last;
logic [7 : 0] nn50;
logic [27 : 0] drr_sqsum;
logic [15 : 0] rr_new;
logic [15 : 0] drr_now;

assign drr_now = (rr_new > rr_last) ? rr_new - rr_last : rr_last - rr_new;
logic [7 : 0] drr_count;
logic rr_valid;
logic [39 : 0] n_times_15000;
logic [47 : 0] n_ss;

// squares of the current interval and its difference, taken on the shared multiplier
logic [1 : 0] sq_step;
logic [15 : 0] sq_a, sq_b;
logic sq_drr;

// 15000 = 2^13 + 2^12 + 2^11 + 2^9 + 2^7 + 2^4 + 2^3
assign n_times_15000 = {19'b0, rr_count, 13'b0} + {20'b0, rr_count, 12'b0}
                     + {21'b0, rr_count, 11'b0} + {23'b0, rr_count, 9'b0}
                     + {25'b0, rr_count, 7'b0} + {28'b0, rr_count, 4'b0}
                     + {29'b0, rr_count, 3'b0};

// --- median tally -----------------------------------------------------------
logic [7 : 0] tally [0 : 1023];
logic [9 : 0] tally_addr;
logic [7 : 0] tally_q;
logic tally_we;
logic [7 : 0] tally_din;
logic [1 : 0] tally_step;
logic [15 : 0] cumulative;
logic [9 : 0] median_bin;
logic [9 : 0] sweep_addr;
logic sweep_active;
logic median_found;
logic [9 : 0] bin_d;

always_ff @(posedge clk) begin
    tally_q <= tally[tally_addr];
    if (tally_we) tally[tally_addr] <= tally_din;
end

localparam logic [2 : 0] IDLE = 3'd0;
localparam logic [2 : 0] RUN = 3'd1;
localparam logic [2 : 0] CLEAR = 3'd2;
localparam logic [2 : 0] SWEEP = 3'd3;
localparam logic [2 : 0] FLUSH = 3'd6;
localparam logic [2 : 0] REDUCE = 3'd4;
localparam logic [2 : 0] WRITE = 3'd5;

logic [2 : 0] state;
logic [3 : 0] step;
logic [2 : 0] write_index;
logic [19 : 0] result [0 : 7];
logic [4 : 0] hrv_shift;
logic signed [19 : 0] hrv_offset;

assign busy = (state != IDLE) && (state != RUN);
// the reader registers its valid, so a granted sample lands a cycle after ready drops;
// one holding slot absorbs it and ready covers the slot, not the cascade
assign ready = (state == RUN) && !sample_valid && !pending;
assign magnitude = filtered[23] ? -filtered : filtered;
assign ewma_mean = ewma[31 : 8];
assign gate = {1'b0, ewma_mean} + {3'b0, ewma_mean[23 : 2]};
assign y1_wide = {{2{y1[23]}}, y1};
assign is_candidate = filtered_valid & (sample_index > 16'd256)
                    & (y1_wide >= $signed({1'b0, thr1}))
                    & ($signed(y1) > $signed(y2))
                    & ($signed(y1) >= $signed(filtered));
assign feature_wr_en = (state == WRITE);
assign feature_index = write_index;
assign feature_word = (result[write_index] >> hrv_shift) - hrv_offset;

always_comb begin
    `include "hrv_narrowing.vh"
end

always_ff @(posedge clk) begin
    if (rst) begin
        state <= IDLE;
        biquad_run <= 0;
        section <= 0;
        filtered_valid <= 0;
        div_start <= 0;
        sqrt_start <= 0;
        tally_we <= 0;
        sq_step <= 0;
        pending <= 0;
    end
    else begin
        div_start <= 0;
        sqrt_start <= 0;
        filtered_valid <= 0;
        tally_we <= 0;
        rr_valid <= 0;

        if (window_start) begin
            for (int i = 0; i < 4; i++) begin
                w1[i] <= 0;
                w2[i] <= 0;
            end
            ewma <= 0;
            sample_index <= 0;
            y1 <= 0;
            y2 <= 0;
            thr1 <= 0;
            pend_valid <= 0;
            last_valid <= 0;
            rr_count <= 0;
            rr_sum <= 0;
            rr_sqsum <= 0;
            drr_sqsum <= 0;
            nn50 <= 0;
            drr_count <= 0;
            rr_min <= 16'hFFFF;
            rr_max <= 0;
            rr_last <= 0;
            pending <= 0;
            tally_addr <= 0;
            tally_step <= 0;
            sq_step <= 0;
            state <= CLEAR;
            step <= 0;
        end

        case (state)
            CLEAR : begin
                tally_we <= 1;
                tally_din <= 0;
                tally_addr <= tally_addr + 1'b1;
                if (tally_addr == 10'h3FF) state <= RUN;
            end

            RUN : begin
                if (sample_valid) begin
                    pending_sample <= sample;
                    pending <= 1;
                end

                if (pending && !biquad_run && !filtered_valid
                    && (sq_step == 2'd0) && !rr_valid) begin
                    stage_in <= {{8{pending_sample[15]}}, pending_sample};
                    section <= 0;
                    biquad_step <= 0;
                    biquad_run <= 1;
                    pending <= 0;
                end
                else if (biquad_run) begin
                    case (biquad_step)
                        2'd0 : begin
                            mul_x <= 32'(w1[section]);
                            mul_y <= a1;
                            biquad_step <= 1;
                        end
                        2'd1 : begin
                            feedback <= mul_p;
                            mul_x <= 32'(w2[section]);
                            mul_y <= a2;
                            biquad_step <= 2;
                        end
                        default : begin
                            w1[section] <= w0;
                            w2[section] <= w1[section];
                            stage_in <= stage_out;
                            biquad_step <= 0;
                            if (section == 2'd3) begin
                                biquad_run <= 0;
                                filtered <= stage_out;
                                filtered_valid <= 1;
                            end
                            else section <= section + 1'b1;
                        end
                    endcase
                end

                if (filtered_valid) begin
                    ewma <= ewma + {8'b0, magnitude} - (ewma >> 8);
                    sample_index <= sample_index + 1'b1;
                    y1 <= filtered;
                    y2 <= y1;
                    thr1 <= gate;
                    idx1 <= sample_index;
                end

                if (is_candidate) begin
                    if (!pend_valid) begin
                        pend_valid <= 1;
                        pend_idx <= idx1;
                        pend_val <= y1;
                    end
                    else if ((idx1 - pend_idx) < REFRACTORY) begin
                        if ($signed(y1) > $signed(pend_val)) begin
                            pend_idx <= idx1;
                            pend_val <= y1;
                        end
                    end
                    else begin
                        last_valid <= 1;
                        last_idx <= pend_idx;
                        rr_new <= pend_idx - last_idx;
                        rr_valid <= last_valid;
                        pend_idx <= idx1;
                        pend_val <= y1;
                    end
                end

                if (window_end) begin
                    // the peak still pending closes the last interval
                    if (pend_valid && last_valid) begin
                        rr_new <= pend_idx - last_idx;
                        rr_valid <= 1;
                    end
                    step <= 0;
                    state <= FLUSH;
                end
            end

            // long enough for the last interval's tally read-modify-write to land
            FLUSH : begin
                step <= step + 1'b1;
                if (step == 3'd5) begin
                    tally_addr <= 0;
                    sweep_addr <= 0;
                    sweep_active <= 0;
                    cumulative <= 0;
                    median_bin <= 0;
                    median_found <= 0;
                    step <= 0;
                    state <= SWEEP;
                end
            end

            SWEEP : begin
                tally_addr <= tally_addr + 1'b1;
                bin_d <= tally_addr;
                sweep_active <= 1;
                if (sweep_active) begin
                    sweep_addr <= sweep_addr + 1'b1;
                    cumulative <= cumulative + {8'b0, tally_q};
                    if (!median_found
                        && (cumulative + {8'b0, tally_q}) >= ({8'b0, rr_count[7 : 1]} + 16'd1)) begin
                        median_bin <= bin_d;
                        median_found <= 1;
                    end
                end
                if (sweep_addr == 10'h3FF) begin
                    step <= 0;
                    state <= REDUCE;
                end
            end

            REDUCE : begin
                case (step)
                    4'd0 : begin
                        div_numerator <= n_times_15000;
                        div_denominator <= {8'b0, rr_sum};
                        div_frac <= 4'd8;
                        div_start <= 1;
                        step <= 1;
                    end
                    4'd1 : if (div_done) begin
                        result[0] <= div_quotient[19 : 0];
                        div_numerator <= {24'b0, rr_sum};
                        div_denominator <= {16'b0, rr_count};
                        div_frac <= 4'd8;
                        div_start <= 1;
                        step <= 2;
                    end
                    4'd2 : if (div_done) begin
                        result[1] <= div_quotient[19 : 0];
                        mul_x <= $signed({4'b0, rr_sqsum});
                        mul_y <= $signed({8'b0, rr_count});
                        step <= 3;
                    end
                    4'd3 : begin
                        n_ss <= mul_p;
                        mul_x <= $signed({16'b0, rr_sum});
                        mul_y <= $signed(rr_sum);
                        step <= 4;
                    end
                    4'd4 : begin
                        sqrt_data <= 32'(n_ss - mul_p);
                        sqrt_start <= 1;
                        step <= 5;
                    end
                    4'd5 : if (sqrt_done) begin
                        div_numerator <= {24'b0, sqrt_result};
                        div_denominator <= {16'b0, rr_count};
                        div_frac <= 4'd8;
                        div_start <= 1;
                        step <= 6;
                    end
                    4'd6 : if (div_done) begin
                        result[2] <= div_quotient[19 : 0];
                        div_numerator <= {drr_sqsum, 12'b0};
                        div_denominator <= {16'b0, drr_count};
                        div_frac <= 4'd0;
                        div_start <= 1;
                        step <= 7;
                    end
                    4'd7 : if (div_done) begin
                        sqrt_data <= div_quotient[31 : 0];
                        sqrt_start <= 1;
                        step <= 8;
                    end
                    4'd8 : if (sqrt_done) begin
                        result[3] <= {2'b0, sqrt_result, 2'b0};
                        div_numerator <= {24'b0, nn50, 8'b0};
                        div_denominator <= {16'b0, drr_count};
                        div_frac <= 4'd0;
                        div_start <= 1;
                        step <= 9;
                    end
                    default : if (div_done) begin
                        result[4] <= div_quotient[19 : 0];
                        result[5] <= {4'b0, rr_min};
                        result[6] <= {4'b0, rr_max};
                        result[7] <= {10'b0, median_bin};
                        write_index <= 0;
                        state <= WRITE;
                    end
                endcase
            end

            WRITE : begin
                write_index <= write_index + 1'b1;
                if (write_index == 3'd7) state <= IDLE;
            end

            default : ;
        endcase

        // outside the case: the flushed interval arrives once the state has left RUN
        if (rr_valid) begin
            rr_count <= rr_count + 1'b1;
            rr_sum <= rr_sum + {8'b0, rr_new};
            rr_last <= rr_new;
            if (rr_new < rr_min) rr_min <= rr_new;
            if (rr_new > rr_max) rr_max <= rr_new;
            if (rr_last != 0) begin
                drr_count <= drr_count + 1'b1;
                // |drr| > 0.05 s as the integer compare 2*|drr| > 0.1*fs
                if ({drr_now, 1'b0} > 16'd25) nn50 <= nn50 + 1'b1;
            end
            sq_a <= rr_new;
            sq_b <= drr_now;
            sq_drr <= (rr_last != 0);
            sq_step <= 2'd1;
            tally_addr <= rr_new[9 : 0];
            tally_step <= 2'd1;
        end

        if (sq_step == 2'd1) begin
            mul_x <= $signed({16'b0, sq_a});
            mul_y <= $signed(sq_a);
            sq_step <= 2'd2;
        end
        else if (sq_step == 2'd2) begin
            rr_sqsum <= rr_sqsum + 28'(mul_p);
            mul_x <= $signed({16'b0, sq_b});
            mul_y <= $signed(sq_b);
            sq_step <= 2'd3;
        end
        else if (sq_step == 2'd3) begin
            if (sq_drr) drr_sqsum <= drr_sqsum + 28'(mul_p);
            sq_step <= 0;
        end

        // one cycle to address the bin, one for the read, then the increment
        if (tally_step == 2'd1) tally_step <= 2'd2;
        else if (tally_step == 2'd2) begin
            tally_din <= tally_q + 1'b1;
            tally_we <= 1;
            tally_step <= 0;
        end
    end
end

endmodule
