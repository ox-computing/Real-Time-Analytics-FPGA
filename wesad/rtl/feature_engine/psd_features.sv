// Segment power accumulator and the 13 spectral features computed from one window's spectrum.

// Exponent-aligned power sum across the segments of one window
module psd_power_accumulator #(parameter BUTTERFLY_WIDTH = 8, parameter ACC_WIDTH = 16)(
    input logic clk,rst,
    input logic start_segment,
    input logic first_segment,
    input logic signed [5 : 0] segment_exponent,

    input logic bin_valid,
    input logic [10 : 0] bin_address,
    input logic [2*BUTTERFLY_WIDTH : 0] bin_power,

    input logic read_enable,
    input logic [10 : 0] read_address,
    output logic [ACC_WIDTH - 1 : 0] read_data,
    output logic signed [5 : 0] power_exponent
);

localparam int POWER_WIDTH = 2*BUTTERFLY_WIDTH + 1;
// the sum has to hold either operand plus a carry, and once ACC_WIDTH drops to
// the power width or below it is the power that sets the size
localparam int TOTAL_WIDTH = ((ACC_WIDTH > POWER_WIDTH) ? ACC_WIDTH : POWER_WIDTH) + 1;
localparam logic [ACC_WIDTH - 1 : 0] ACC_MAX = {ACC_WIDTH{1'b1}};

logic [ACC_WIDTH - 1 : 0] accumulator [0 : 1024];
logic [5 : 0] align_shift;
logic shift_power;

logic [POWER_WIDTH - 1 : 0] power_ff;
logic [ACC_WIDTH - 1 : 0] accumulator_read;
logic [TOTAL_WIDTH - 1 : 0] accumulator_term;
logic [TOTAL_WIDTH - 1 : 0] total;
// headroom for the round-half-up bias the alignment shift adds before shifting
localparam int ROUND_WIDTH = TOTAL_WIDTH + 6;
logic [ROUND_WIDTH - 1 : 0] biased;
logic [ROUND_WIDTH - 1 : 0] aligned;
logic replace_accumulator;
logic [10 : 0] write_address;
logic [10 : 0] memory_address;
logic write_enable;

// accumulate and readout are separate phases, so one read port serves both
assign memory_address = read_enable ? read_address : bin_address;

always_comb begin
    // nothing clears the array between windows, so the first segment overwrites
    // what the last window left behind rather than accumulating onto it
    biased = ROUND_WIDTH'(shift_power ? ROUND_WIDTH'(power_ff) : ROUND_WIDTH'(accumulator_read))
        + ((align_shift == 0) ? '0 : (ROUND_WIDTH'(1) << (align_shift - 1)));
    aligned = biased >> align_shift;
    accumulator_term = replace_accumulator ? '0
        : (shift_power ? TOTAL_WIDTH'(accumulator_read) : TOTAL_WIDTH'(aligned));
    total = accumulator_term + (shift_power ? TOTAL_WIDTH'(aligned) : TOTAL_WIDTH'(power_ff));
end

always_ff @(posedge clk) begin
    if (rst) begin
        power_exponent <= 0;
        align_shift <= 0;
        shift_power <= 0;
        write_enable <= 0;
        replace_accumulator <= 0;
    end
    else begin
        if (start_segment) begin
            replace_accumulator <= first_segment;
            // both operands are brought to the larger exponent before they meet
            if (first_segment || segment_exponent > power_exponent) begin
                align_shift <= first_segment ? 0 : segment_exponent - power_exponent;
                shift_power <= 0;
                power_exponent <= segment_exponent;
            end
            else begin
                align_shift <= power_exponent - segment_exponent;
                shift_power <= 1;
            end
        end

        accumulator_read <= accumulator[memory_address];
        power_ff <= bin_power;
        write_address <= bin_address;
        write_enable <= bin_valid;

        if (write_enable)
            accumulator[write_address] <= (total > TOTAL_WIDTH'(ACC_MAX)) ? ACC_MAX : total[ACC_WIDTH - 1 : 0];

        read_data <= accumulator_read;
    end
end

endmodule

module psd_calc (
    input logic clk, rst,
    input logic start,
    input logic [2 : 0] modality,
    input logic signed [6 : 0] power_exponent,

    output logic acc_read_enable,
    output logic [10 : 0] acc_read_address,
    input logic [15 : 0] acc_read_data,

    output logic div_start,
    output logic [39 : 0] div_numerator,
    output logic [23 : 0] div_denominator,
    output logic [3 : 0] div_frac,
    input logic [39 : 0] div_quotient,
    input logic div_done,
    input logic div_busy,

    output logic [23 : 0] log_value,
    input logic [12 : 0] log_result,

    output logic sqrt_start,
    output logic [35 : 0] sqrt_data,
    input logic [17 : 0] sqrt_result,
    input logic sqrt_done,
    input logic sqrt_busy,

    output logic [20 : 0] mul_a,
    output logic [15 : 0] mul_b,
    input logic [36 : 0] product,

    output logic busy,
    output logic feature_wr_en,
    output logic [6 : 0] feature_wr_addr,
    output logic [7 : 0] feature_wr_data
);

localparam logic [3 : 0] IDLE = 4'd0;
localparam logic [3 : 0] PASS1 = 4'd1;
localparam logic [3 : 0] PASS2 = 4'd2;
localparam logic [3 : 0] DIV_C = 4'd3;
localparam logic [3 : 0] PASS3 = 4'd4;
localparam logic [3 : 0] DIV_E = 4'd5;
localparam logic [3 : 0] SETUP4 = 4'd6;
localparam logic [3 : 0] PASS4 = 4'd7;
localparam logic [3 : 0] DIV_V = 4'd8;
localparam logic [3 : 0] SQRT_B = 4'd9;
localparam logic [3 : 0] WRITE = 4'd10;

logic [3 : 0] state;
logic [10 : 0] bin;
logic [10 : 0] bin_d1, bin_d2;
logic v1, v2, v3;
logic [10 : 0] bin_v;
logic in_pass;

logic [23 : 0] total;
logic [23 : 0] cum;
logic [15 : 0] peak_power;
logic [10 : 0] peak_freq;
logic [39 : 0] acc40;
logic [23 : 0] band [0 : 5];
logic [23 : 0] band_acc;
logic [2 : 0] band_ptr;
logic band_done;
logic [10 : 0] band_edge;

logic [18 : 0] centroid_q;
logic [10 : 0] centroid_i;
logic [7 : 0] centroid_d;
logic signed [13 : 0] entropy;
logic [17 : 0] bandwidth;
logic [10 : 0] rolloff;
logic rolloff_found;

logic [10 : 0] dk;
logic [20 : 0] dksq;
logic [3 : 0] write_index;
logic [23 : 0] centroid_sq;
logic [20 : 0] feature_value;
logic [4 : 0] psd_shift;
logic signed [20 : 0] psd_offset;
logic [28 : 0] psd_shift_register;
logic [4 : 0] psd_shift_count;
logic [3 : 0] state_prev;
logic [3 : 0] write_index_prev;
logic psd_shift_load, psd_shift_busy;

assign busy = (state != IDLE);
assign in_pass = (state == PASS1) || (state == PASS2) || (state == PASS3) || (state == PASS4);
assign acc_read_enable = in_pass && (bin <= 11'd1024);
assign acc_read_address = bin;
assign centroid_sq = {16'b0, product[15 : 8]};
assign bin_v = bin_d2;
always_comb begin
    if (state == PASS3) log_value = {8'b0, acc_read_data};
    else case (write_index)
        4'd2 : log_value = {8'b0, peak_power};
        4'd7 : log_value = band[0];
        4'd8 : log_value = band[1];
        4'd9 : log_value = band[2];
        4'd10 : log_value = band[3];
        4'd11 : log_value = band[4];
        4'd12 : log_value = band[5];
        default : log_value = total;
    endcase
end

always_comb begin
    case (band_ptr)
        3'd0 : case (modality)
                   3'd0 : band_edge = 11'd1;
                   3'd1 : band_edge = 11'd7;
                   3'd2, 3'd3 : band_edge = 11'd9;
                   default : band_edge = 11'd1;
               endcase
        3'd1 : case (modality)
                   3'd0 : band_edge = 11'd5;
                   3'd1 : band_edge = 11'd32;
                   3'd2, 3'd3 : band_edge = 11'd41;
                   default : band_edge = 11'd3;
               endcase
        3'd2 : case (modality)
                   3'd0 : band_edge = 11'd17;
                   3'd1 : band_edge = 11'd128;
                   3'd2, 3'd3 : band_edge = 11'd164;
                   default : band_edge = 11'd12;
               endcase
        3'd3 : case (modality)
                   3'd0 : band_edge = 11'd66;
                   3'd1 : band_edge = 11'd512;
                   3'd2, 3'd3 : band_edge = 11'd656;
                   default : band_edge = 11'd47;
               endcase
        3'd4 : case (modality)
                   3'd0 : band_edge = 11'd246;
                   3'd1, 3'd2, 3'd3 : band_edge = 11'd1025;
                   default : band_edge = 11'd176;
               endcase
        default : case (modality)
                      3'd0 : band_edge = 11'd820;
                      3'd1, 3'd2, 3'd3 : band_edge = 11'd1025;
                      default : band_edge = 11'd586;
                  endcase
    endcase

    case (write_index)
        4'd1 : feature_value = {10'b0, peak_freq};
        4'd3 : feature_value = {2'b0, centroid_q};
        4'd4 : feature_value = {3'b0, bandwidth};
        4'd5 : feature_value = {7'b0, entropy};
        4'd6 : feature_value = {10'b0, rolloff};
        // log2(acc * 2**pe) = pe + log2(acc), both carrying LOG2_FRAC = 8; an
        // empty band has no logarithm and reaches the word as zero
        default : feature_value = (log_value == 0) ? 21'd0
                  : {5'b0, power_exponent[5 : 0], 8'b0} + {8'b0, log_result};
    endcase
end

assign psd_shift_load = (state != state_prev) || (write_index != write_index_prev);
assign psd_shift_busy = psd_shift_load || (psd_shift_count != 0);
assign feature_wr_en = (state == WRITE) && !psd_shift_busy;
// the 115-column contract puts the 65 spectral words after the 50 time words
assign feature_wr_addr = 7'd50 + {3'b0, modality} * 7'd13 + {3'b0, write_index};
assign feature_wr_data = psd_shift_register - psd_offset;

always_ff @(posedge clk) begin
    state_prev <= state;
    write_index_prev <= write_index;
    if (psd_shift_load) begin
        psd_shift_register <= {8'b0, feature_value};
        psd_shift_count <= psd_shift;
    end
    else if (psd_shift_count != 0) begin
        psd_shift_register <= psd_shift_register >> 1;
        psd_shift_count <= psd_shift_count - 1'b1;
    end
end

always_comb begin
    `include "psd_narrowing.vh"
end

always_ff @(posedge clk) begin
    if (rst) begin
        state <= IDLE;
        div_start <= 0;
        sqrt_start <= 0;
        v1 <= 0;
        v2 <= 0;
        v3 <= 0;
    end
    else begin
        div_start <= 0;
        sqrt_start <= 0;

        // the accumulator answers two cycles after the address, and the product
        // one cycle after its operands, so validity walks three stages
        bin_d1 <= bin;
        bin_d2 <= bin_d1;
        v1 <= acc_read_enable;
        v2 <= v1;
        v3 <= v2;

        case (state)
            IDLE : if (start) begin
                bin <= 0;
                total <= 0;
                cum <= 0;
                peak_power <= 0;
                peak_freq <= 0;
                acc40 <= 0;
                band_ptr <= 0;
                band_done <= 0;
                rolloff <= 11'd1024;
                rolloff_found <= 0;
                for (int i = 0; i < 6; i++) band[i] <= 0;
                band_acc <= 0;
                state <= PASS1;
            end

            PASS1 : begin
                bin <= bin + 1'b1;
                if (v2) begin
                    total <= total + {8'b0, acc_read_data};
                    if (acc_read_data > peak_power) begin
                        peak_power <= acc_read_data;
                        peak_freq <= bin_v;
                    end
                    // the crossing bin opens the next band; past the last edge nothing counts
                    if (!band_done) begin
                        if (bin_v >= band_edge) begin
                            if (band_ptr == 3'd5) band_done <= 1;
                            else begin
                                band_ptr <= band_ptr + 1'b1;
                                band[band_ptr] <= band_acc;
                                band_acc <= {8'b0, acc_read_data};
                            end
                        end
                        else band_acc <= band_acc + {8'b0, acc_read_data};
                    end
                end
                if (bin == 11'd1027) band[band_ptr] <= band_acc;
                if (bin == 11'd1028) begin
                    bin <= 0;
                    acc40 <= 0;
                    state <= PASS2;
                end
            end

            PASS2 : begin
                bin <= bin + 1'b1;
                mul_a <= {10'b0, bin_v};
                mul_b <= acc_read_data;
                if (v3) acc40 <= acc40 + {3'b0, product};
                if (bin == 11'd1028) begin
                    div_numerator <= acc40;
                    div_denominator <= total;
                    div_frac <= 4'd8;
                    div_start <= 1;
                    state <= DIV_C;
                end
            end

            DIV_C : if (div_done) begin
                centroid_q <= div_quotient[18 : 0];
                centroid_i <= div_quotient[18 : 8];
                centroid_d <= div_quotient[7 : 0];
                bin <= 0;
                acc40 <= 0;
                state <= PASS3;
            end

            PASS3 : begin
                bin <= bin + 1'b1;
                mul_a <= {8'b0, log_result};
                mul_b <= acc_read_data;
                if (v3) acc40 <= acc40 + {3'b0, product};
                if (bin == 11'd1028) begin
                    div_numerator <= acc40;
                    div_denominator <= total;
                    div_frac <= 4'd0;
                    div_start <= 1;
                    state <= DIV_E;
                end
            end

            DIV_E : if (div_done) begin
                entropy <= $signed({1'b0, log_result}) - $signed(div_quotient[13 : 0]);
                mul_a <= {10'b0, centroid_i};
                mul_b <= {5'b0, centroid_i};
                state <= SETUP4;
            end

            // dk^2 is carried forward by (d+-1)^2 = d^2 +- 2d + 1, so the second
            // moment costs one multiply per bin rather than two
            SETUP4 : begin
                dk <= centroid_i;
                dksq <= product[20 : 0];
                bin <= 0;
                acc40 <= 0;
                cum <= 0;
                state <= PASS4;
            end

            PASS4 : begin
                bin <= bin + 1'b1;
                mul_a <= dksq;
                mul_b <= acc_read_data;
                if (v2) begin
                    dk <= (bin_v < centroid_i) ? dk - 1'b1 : dk + 1'b1;
                    dksq <= dksq + ((bin_v < centroid_i) ? ~{9'b0, dk, 1'b0} + 21'd2
                                                         : {9'b0, dk, 1'b0} + 21'd1);
                    cum <= cum + {8'b0, acc_read_data};
                    if (!rolloff_found
                        && ({4'b0, cum + {8'b0, acc_read_data}, 4'b0}
                            + {2'b0, cum + {8'b0, acc_read_data}, 2'b0})
                           >= ({4'b0, total, 4'b0} + {6'b0, total})) begin
                        rolloff <= bin_v;
                        rolloff_found <= 1;
                    end
                end
                if (v3) acc40 <= acc40 + {3'b0, product};
                if (bin == 11'd1028) begin
                    div_numerator <= acc40;
                    div_denominator <= total;
                    div_frac <= 4'd8;
                    div_start <= 1;
                    mul_a <= {13'b0, centroid_d};
                    mul_b <= {8'b0, centroid_d};
                    state <= DIV_V;
                end
            end

            DIV_V : if (div_done) begin
                // var_q reaches 25 bits on a broadband spectrum, so the
                // Q8 radicand needs the full 36
                sqrt_data <= (div_quotient[27 : 0] > centroid_sq)
                             ? {div_quotient[27 : 0] - centroid_sq, 8'b0}
                             : 36'd0;
                sqrt_start <= 1;
                state <= SQRT_B;
            end

            SQRT_B : if (sqrt_done) begin
                bandwidth <= sqrt_result;
                write_index <= 0;
                state <= WRITE;
            end

            WRITE : if (!psd_shift_busy) begin
                write_index <= write_index + 1'b1;
                if (write_index == 4'd12) state <= IDLE;
            end

            default : state <= IDLE;
        endcase
    end
end

endmodule
