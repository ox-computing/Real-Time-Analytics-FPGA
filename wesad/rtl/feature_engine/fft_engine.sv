// 2048-point block-floating-point FFT: segment statistics, windowed load, in-place transform, bin readout.

// Detrend and apply Hann window
module preprocessing #(parameter BUTTERFLY_WIDTH  = 8, parameter W_WIDTH = 8, parameter SAMPLE_WIDTH = 8)(
    input logic clk,rst,
    input logic start,
    input logic in_valid,
    input logic signed [SAMPLE_WIDTH - 1 : 0] sample,
    input logic signed [SAMPLE_WIDTH - 1 : 0] segment_mean,
    input logic signed [4 : 0] normalise_shift,

    output logic [10 : 0] hann_address,
    input logic signed [W_WIDTH - 1 : 0] hann_coefficient,

    output logic signed [BUTTERFLY_WIDTH - 1 : 0] mul_x,
    output logic signed [W_WIDTH - 1 : 0] mul_y,
    input logic signed [BUTTERFLY_WIDTH + W_WIDTH - 1 : 0] mul_p,

    output logic signed [BUTTERFLY_WIDTH - 1 : 0] sample_out,
    output logic out_valid
);

localparam int DETREND_WIDTH = SAMPLE_WIDTH + 1;
localparam int PRODUCT_WIDTH = BUTTERFLY_WIDTH + W_WIDTH;
localparam int HANN_FRACTION_BITS = W_WIDTH - 1;
localparam logic signed [PRODUCT_WIDTH : 0] PRODUCT_ROUND = 1 <<< (HANN_FRACTION_BITS - 1);
localparam logic signed [DETREND_WIDTH - 1 : 0] MAX_VALUE = (1 <<< (BUTTERFLY_WIDTH - 1)) - 1;

logic [10 : 0] sample_count;
logic signed [DETREND_WIDTH - 1 : 0] shift_round;
logic signed [DETREND_WIDTH - 1 : 0] shifted;
logic signed [DETREND_WIDTH - 1 : 0] saturated;

logic signed [DETREND_WIDTH - 1 : 0] detrended;
logic signed [BUTTERFLY_WIDTH - 1 : 0] narrowed;
logic signed [W_WIDTH - 1 : 0] hann_ff;
logic signed [PRODUCT_WIDTH - 1 : 0] product;

logic in_valid_ff;
logic in_valid_ff2;
logic in_valid_ff3;

assign hann_address = sample_count;
assign mul_x = narrowed;
assign mul_y = hann_ff;

always_comb begin
    // round-half-up on the way down, plain shift on the way up
    if (normalise_shift > 0) begin
        shift_round = 1 <<< (normalise_shift - 1);
        shifted = (detrended + shift_round) >>> normalise_shift;
    end
    else begin
        shift_round = 0;
        shifted = detrended <<< (-normalise_shift);
    end
    // the shift is sized so the sample fits, rounding up is the only way out of range
    saturated = (shifted > MAX_VALUE) ? MAX_VALUE : shifted;
end

// clk 0 = detrend and start the Hann read, clk 1 = normalise, clk 2 = multiply, clk 3 = round
always_ff @(posedge clk) begin
    if (rst) begin
        sample_count <= 0;
        detrended <= 0;
        narrowed <= 0;
        hann_ff <= 0;
        product <= 0;
        sample_out <= 0;
        in_valid_ff <= 0;
        in_valid_ff2 <= 0;
        in_valid_ff3 <= 0;
        out_valid <= 0;
    end
    else begin
        if (start)
            sample_count <= 0;
        else if (in_valid)
            sample_count <= sample_count + 1;

        detrended <= sample - segment_mean;
        narrowed <= saturated;
        hann_ff <= hann_coefficient;
        product <= mul_p;
        sample_out <= (product + PRODUCT_ROUND) >>> HANN_FRACTION_BITS;

        in_valid_ff <= in_valid;
        in_valid_ff2 <= in_valid_ff;
        in_valid_ff3 <= in_valid_ff2;
        out_valid <= in_valid_ff3;
    end
end


endmodule



module complex_butterfly #(parameter BUTTERFLY_WIDTH  = 8, parameter W_WIDTH = 8)(
    input logic clk,rst,
    input logic in_valid,
    input logic shift_enable,
    input logic signed [BUTTERFLY_WIDTH - 1 : 0] a_real,
    input logic signed [BUTTERFLY_WIDTH - 1 : 0] a_imaginary,
    input logic signed [BUTTERFLY_WIDTH - 1 : 0] b_real,
    input logic signed [BUTTERFLY_WIDTH - 1 : 0] b_imaginary,
    input logic signed [W_WIDTH - 1 : 0] w_real,
    input logic signed [W_WIDTH - 1 : 0] w_imaginary,

    output logic signed [BUTTERFLY_WIDTH - 1 : 0] a_out_real,
    output logic signed [BUTTERFLY_WIDTH - 1 : 0] a_out_imaginary,
    output logic signed [BUTTERFLY_WIDTH - 1 : 0] b_out_real,
    output logic signed [BUTTERFLY_WIDTH - 1 : 0] b_out_imaginary,

    output logic out_valid
);

localparam int PRODUCT_WIDTH = BUTTERFLY_WIDTH + W_WIDTH;
localparam int SUM_WIDTH = BUTTERFLY_WIDTH + 2;
localparam int TWIDDLE_FRACTION_BITS = W_WIDTH - 1;
localparam logic signed [PRODUCT_WIDTH : 0] PRODUCT_ROUND = 1 <<< (TWIDDLE_FRACTION_BITS - 1);
localparam logic signed [SUM_WIDTH - 1 : 0] MAX_VALUE = (1 <<< (BUTTERFLY_WIDTH - 1)) - 1;
localparam logic signed [SUM_WIDTH - 1 : 0] MIN_VALUE = -(1 <<< (BUTTERFLY_WIDTH - 1));

// These are supposed to map to the registers that come with the DSP blocks in UP5K (input side)
logic signed [BUTTERFLY_WIDTH - 1 : 0] a_real_ff;
logic signed [BUTTERFLY_WIDTH - 1 : 0] a_imaginary_ff;
logic signed [BUTTERFLY_WIDTH - 1 : 0] b_real_ff;
logic signed [BUTTERFLY_WIDTH - 1 : 0] b_imaginary_ff;
logic signed [W_WIDTH - 1 : 0] w_real_ff;
logic signed [W_WIDTH - 1 : 0] w_imaginary_ff;
// And the products to the output side registers of DSP
logic signed [BUTTERFLY_WIDTH + W_WIDTH - 1 : 0] p0,p1,p2,p3;

logic signed [BUTTERFLY_WIDTH - 1 : 0] a_real_ff2;
logic signed [BUTTERFLY_WIDTH - 1 : 0] a_imaginary_ff2;

logic in_valid_ff;
logic in_valid_ff2;
logic shift_enable_ff;
logic shift_enable_ff2;

logic signed [PRODUCT_WIDTH : 0] product_real_sum;
logic signed [PRODUCT_WIDTH : 0] product_imaginary_sum;
logic signed [SUM_WIDTH - 1 : 0] twiddled_real;
logic signed [SUM_WIDTH - 1 : 0] twiddled_imaginary;
logic signed [SUM_WIDTH - 1 : 0] a_sum_real;
logic signed [SUM_WIDTH - 1 : 0] a_sum_imaginary;
logic signed [SUM_WIDTH - 1 : 0] b_sum_real;
logic signed [SUM_WIDTH - 1 : 0] b_sum_imaginary;

function automatic logic signed [BUTTERFLY_WIDTH - 1 : 0] scale_and_saturate (input logic signed [SUM_WIDTH - 1 : 0] value, input logic shift);
    logic signed [SUM_WIDTH - 1 : 0] scaled;
    begin
        scaled = shift ? (value + 1) >>> 1 : value;
        if (scaled > MAX_VALUE)
            scale_and_saturate = MAX_VALUE;
        else if (scaled < MIN_VALUE)
            scale_and_saturate = MIN_VALUE;
        else
            scale_and_saturate = scaled;
    end
endfunction

always_comb begin
    product_real_sum = p0 - p1 + PRODUCT_ROUND;
    product_imaginary_sum = p2 + p3 + PRODUCT_ROUND;
    twiddled_real = product_real_sum >>> TWIDDLE_FRACTION_BITS;
    twiddled_imaginary = product_imaginary_sum >>> TWIDDLE_FRACTION_BITS;
    a_sum_real = a_real_ff2 + twiddled_real;
    a_sum_imaginary = a_imaginary_ff2 + twiddled_imaginary;
    b_sum_real = a_real_ff2 - twiddled_real;
    b_sum_imaginary = a_imaginary_ff2 - twiddled_imaginary;
end

// Reuse 4 DSP blocks, so clk 0 = read, clk 1 = wrbi, wibi, wrbr, wibr calculations, clk 2 = wrbr - wibi, wibr + wrbi, read ar ai, clk 3 = ar - ..., ai - ...,  etc
always_ff @(posedge clk) begin
    if(rst) begin
        a_real_ff <= 0;
        a_imaginary_ff <= 0;
        b_real_ff <= 0;
        b_imaginary_ff <= 0;
        w_real_ff <= 0;
        w_imaginary_ff <= 0;
        in_valid_ff <= 0;
        shift_enable_ff <= 0;
        p0 <= 0;
        p1 <= 0;
        p2 <= 0;
        p3 <= 0;
        a_real_ff2 <= 0;
        a_imaginary_ff2 <= 0;
        in_valid_ff2 <= 0;
        shift_enable_ff2 <= 0;
        a_out_real <= 0;
        a_out_imaginary <= 0;
        b_out_real <= 0;
        b_out_imaginary <= 0;
        out_valid <= 0;
    end
    else begin
        a_real_ff <= a_real;
        a_imaginary_ff <= a_imaginary;
        b_real_ff <= b_real;
        b_imaginary_ff <= b_imaginary;
        w_real_ff <= w_real;
        w_imaginary_ff <= w_imaginary;
        in_valid_ff <= in_valid;
        shift_enable_ff <= shift_enable;

        p0 <= w_real_ff * b_real_ff;
        p1 <= w_imaginary_ff * b_imaginary_ff;
        p2 <= w_real_ff * b_imaginary_ff;
        p3 <= w_imaginary_ff * b_real_ff;
        a_real_ff2 <= a_real_ff;
        a_imaginary_ff2 <= a_imaginary_ff;
        in_valid_ff2 <= in_valid_ff;
        shift_enable_ff2 <= shift_enable_ff;

        a_out_real <= scale_and_saturate(a_sum_real, shift_enable_ff2);
        a_out_imaginary <= scale_and_saturate(a_sum_imaginary, shift_enable_ff2);
        b_out_real <= scale_and_saturate(b_sum_real, shift_enable_ff2);
        b_out_imaginary <= scale_and_saturate(b_sum_imaginary, shift_enable_ff2);
        out_valid <= in_valid_ff2;
    end
end


endmodule

// Needs to generate per stage: address for butterfly pairs and needed W address
module address_generator (
    input logic clk,rst,
    input logic start,
    input logic initial_shift,
    input logic magnitude_exceeded,

    output logic [10:0] a_address,
    output logic [10:0] b_address,
    output logic [9:0] w_address,
    output logic butterfly_valid,
    output logic shift_enable,
    output logic [3:0] block_exponent,
    output logic done
);

localparam int LAST_LEVEL = 10;
localparam int LAST_BUTTERFLY = 1023;
// one read plus three butterfly stages puts the level's last writeback four
// cycles after its address; the drain runs one longer still so the magnitude
// flag riding with it is latched before the level advance clears it
localparam int PIPELINE_DEPTH = 5;

logic [3:0] level;
logic [9:0] butterfly_count;
logic [10:0] pair_offset;
logic [11:0] address_step;
logic [9:0] twiddle_mask;
logic [11:0] address_sum;
logic shift_enable_next;
logic [2:0] drain_count;

// a_address always has bit `level` clear, so the pair is an OR not an add
assign b_address = a_address | pair_offset;
assign w_address = butterfly_count & twiddle_mask;
assign address_sum = {1'b0, a_address} + address_step;

always_ff @(posedge clk) begin
    if (rst) begin
        a_address <= 0;
        butterfly_count <= 0;
        level <= 0;
        pair_offset <= 1;
        address_step <= 2;
        twiddle_mask <= 0;
        shift_enable <= 0;
        shift_enable_next <= 0;
        block_exponent <= 0;
        butterfly_valid <= 0;
        drain_count <= 0;
        done <= 0;
    end
    else if (start) begin
        a_address <= 0;
        butterfly_count <= 0;
        level <= 0;
        pair_offset <= 1;
        address_step <= 2;
        twiddle_mask <= 0;
        // the first level's shift comes from the loaded segment's own peak, and
        // the exponent has to carry it since the level loop only counts the rest
        shift_enable <= initial_shift;
        shift_enable_next <= 0;
        block_exponent <= {3'b0, initial_shift};
        butterfly_valid <= 1;
        drain_count <= 0;
        done <= 0;
    end
    else begin
        if (magnitude_exceeded)
            shift_enable_next <= 1;

        if (butterfly_valid) begin
            if (butterfly_count == LAST_BUTTERFLY) begin
                butterfly_valid <= 0;
                drain_count <= PIPELINE_DEPTH;
            end
            else begin
                // end-around carry: this is the rotate the level needs
                a_address <= address_sum[10:0] + address_sum[11];
                butterfly_count <= butterfly_count + 1;
            end
        end
        // let the butterfly pipeline empty before latching the level's shift flag
        else if (drain_count != 0) begin
            drain_count <= drain_count - 1;
            if (drain_count == 1) begin
                if (level == LAST_LEVEL) begin
                    done <= 1;
                end
                else begin
                    level <= level + 1;
                    pair_offset <= pair_offset << 1;
                    address_step <= address_step << 1;
                    twiddle_mask <= (twiddle_mask >> 1) | 10'h200;
                    shift_enable <= shift_enable_next;
                    block_exponent <= block_exponent + shift_enable_next;
                    shift_enable_next <= 0;
                    a_address <= 0;
                    butterfly_count <= 0;
                    butterfly_valid <= 1;
                end
            end
        end
    end
end

endmodule



// Single EBR bank: one write port, one registered read port
module fft_scratch_bank #(parameter WORD_WIDTH = 16)(
    input logic clk,
    input logic write_enable,
    input logic [9 : 0] write_address,
    input logic [WORD_WIDTH - 1 : 0] write_data,
    input logic [9 : 0] read_address,
    output logic [WORD_WIDTH - 1 : 0] read_data
);

logic [WORD_WIDTH - 1 : 0] memory [0 : 1023];

always_ff @(posedge clk) begin
    if (write_enable)
        memory[write_address] <= write_data;
    read_data <= memory[read_address];
end

endmodule



// One 2048-point scratch: even/odd banks split by address parity
module fft_scratch_buffer #(parameter WORD_WIDTH = 16)(
    input logic clk,
    input logic a_write_enable,
    input logic b_write_enable,
    input logic [10 : 0] a_write_address,
    input logic [10 : 0] b_write_address,
    input logic [WORD_WIDTH - 1 : 0] a_write_data,
    input logic [WORD_WIDTH - 1 : 0] b_write_data,
    input logic [10 : 0] a_read_address,
    input logic [10 : 0] b_read_address,
    output logic [WORD_WIDTH - 1 : 0] a_read_data,
    output logic [WORD_WIDTH - 1 : 0] b_read_data
);

logic a_write_is_odd;
logic b_write_is_odd;
logic a_read_is_odd;
logic a_read_is_odd_ff;
logic a_write_takes_even;
logic a_write_takes_odd;

logic even_write_enable;
logic odd_write_enable;
logic [9 : 0] even_write_address;
logic [9 : 0] odd_write_address;
logic [WORD_WIDTH - 1 : 0] even_write_data;
logic [WORD_WIDTH - 1 : 0] odd_write_data;
logic [9 : 0] even_read_address;
logic [9 : 0] odd_read_address;
logic [WORD_WIDTH - 1 : 0] even_read_data;
logic [WORD_WIDTH - 1 : 0] odd_read_data;

assign a_write_is_odd = ^a_write_address;
assign b_write_is_odd = ^b_write_address;
assign a_read_is_odd = ^a_read_address;

assign a_write_takes_even = a_write_enable & ~a_write_is_odd;
assign a_write_takes_odd = a_write_enable & a_write_is_odd;

assign even_write_enable = a_write_takes_even | (b_write_enable & ~b_write_is_odd);
assign odd_write_enable = a_write_takes_odd | (b_write_enable & b_write_is_odd);

assign even_write_address = a_write_takes_even ? a_write_address[10 : 1] : b_write_address[10 : 1];
assign odd_write_address = a_write_takes_odd ? a_write_address[10 : 1] : b_write_address[10 : 1];
assign even_write_data = a_write_takes_even ? a_write_data : b_write_data;
assign odd_write_data = a_write_takes_odd ? a_write_data : b_write_data;

assign even_read_address = a_read_is_odd ? b_read_address[10 : 1] : a_read_address[10 : 1];
assign odd_read_address = a_read_is_odd ? a_read_address[10 : 1] : b_read_address[10 : 1];

assign a_read_data = a_read_is_odd_ff ? odd_read_data : even_read_data;
assign b_read_data = a_read_is_odd_ff ? even_read_data : odd_read_data;

always_ff @(posedge clk) a_read_is_odd_ff <= a_read_is_odd;

fft_scratch_bank #(.WORD_WIDTH(WORD_WIDTH)) even_bank (
    .clk (clk),
    .write_enable (even_write_enable),
    .write_address (even_write_address),
    .write_data (even_write_data),
    .read_address (even_read_address),
    .read_data (even_read_data)
);

fft_scratch_bank #(.WORD_WIDTH(WORD_WIDTH)) odd_bank (
    .clk (clk),
    .write_enable (odd_write_enable),
    .write_address (odd_write_address),
    .write_data (odd_write_data),
    .read_address (odd_read_address),
    .read_data (odd_read_data)
);

endmodule



// One 2048-point scratch: a segment loads, then transforms in place
module fft_scratch_memory #(parameter BUTTERFLY_WIDTH  = 8)(
    input logic clk, rst,

    input logic load_valid,
    input logic [10 : 0] load_address,
    input logic signed [BUTTERFLY_WIDTH - 1 : 0] load_sample,

    input logic butterfly_valid,
    input logic [10 : 0] a_address,
    input logic [10 : 0] b_address,

    output logic butterfly_in_valid,
    output logic signed [BUTTERFLY_WIDTH - 1 : 0] a_real,
    output logic signed [BUTTERFLY_WIDTH - 1 : 0] a_imaginary,
    output logic signed [BUTTERFLY_WIDTH - 1 : 0] b_real,
    output logic signed [BUTTERFLY_WIDTH - 1 : 0] b_imaginary,

    input logic signed [BUTTERFLY_WIDTH - 1 : 0] a_out_real,
    input logic signed [BUTTERFLY_WIDTH - 1 : 0] a_out_imaginary,
    input logic signed [BUTTERFLY_WIDTH - 1 : 0] b_out_real,
    input logic signed [BUTTERFLY_WIDTH - 1 : 0] b_out_imaginary,

    output logic magnitude_exceeded
);

localparam int WORD_WIDTH = 2 * BUTTERFLY_WIDTH;
localparam int READ_LATENCY = 1;
localparam int BUTTERFLY_LATENCY = 3;
localparam int WRITEBACK_DELAY = READ_LATENCY + BUTTERFLY_LATENCY;
localparam int MAGNITUDE_LIMIT = (1 <<< (BUTTERFLY_WIDTH - 1)) - 1;
localparam int SHIFT_NUMERATOR = 5;
localparam int SHIFT_DENOMINATOR = 2;
localparam logic [BUTTERFLY_WIDTH : 0] MAGNITUDE_THRESHOLD =
    (BUTTERFLY_WIDTH + 1)'((SHIFT_DENOMINATOR * MAGNITUDE_LIMIT) / SHIFT_NUMERATOR);

logic [10 : 0] load_address_reversed;

logic [10 : 0] a_address_delay [0 : WRITEBACK_DELAY - 1];
logic [10 : 0] b_address_delay [0 : WRITEBACK_DELAY - 1];
logic butterfly_valid_delay [0 : WRITEBACK_DELAY - 1];

logic butterfly_write_valid;
logic [10 : 0] a_write_address;
logic [10 : 0] b_write_address;
logic [WORD_WIDTH - 1 : 0] a_write_data;
logic [WORD_WIDTH - 1 : 0] b_write_data;
logic [WORD_WIDTH - 1 : 0] load_write_data;

logic scratch_a_write_enable;
logic [10 : 0] scratch_a_write_address;
logic [WORD_WIDTH - 1 : 0] scratch_a_write_data;

logic [WORD_WIDTH - 1 : 0] a_read_data;
logic [WORD_WIDTH - 1 : 0] b_read_data;

function automatic logic over_threshold (input logic signed [BUTTERFLY_WIDTH - 1 : 0] value);
    logic signed [BUTTERFLY_WIDTH : 0] extended;
    logic [BUTTERFLY_WIDTH : 0] magnitude;
    begin
        extended = {value[BUTTERFLY_WIDTH - 1], value};
        magnitude = extended[BUTTERFLY_WIDTH] ? -extended : extended;
        over_threshold = magnitude > MAGNITUDE_THRESHOLD;
    end
endfunction

always_comb begin
    for (int bit_index = 0; bit_index < 11; bit_index++)
        load_address_reversed[bit_index] = load_address[10 - bit_index];
end

assign butterfly_in_valid = butterfly_valid_delay[0];
assign butterfly_write_valid = butterfly_valid_delay[WRITEBACK_DELAY - 1];
assign a_write_address = a_address_delay[WRITEBACK_DELAY - 1];
assign b_write_address = b_address_delay[WRITEBACK_DELAY - 1];
assign a_write_data = {a_out_imaginary, a_out_real};
assign b_write_data = {b_out_imaginary, b_out_real};
assign load_write_data = {{BUTTERFLY_WIDTH{1'b0}}, load_sample};

// load and transform are separate phases, so port A carries whichever is live
assign scratch_a_write_enable = butterfly_write_valid | load_valid;
assign scratch_a_write_address = butterfly_write_valid ? a_write_address : load_address_reversed;
assign scratch_a_write_data = butterfly_write_valid ? a_write_data : load_write_data;

assign a_real = a_read_data[BUTTERFLY_WIDTH - 1 : 0];
assign a_imaginary = a_read_data[WORD_WIDTH - 1 : BUTTERFLY_WIDTH];
assign b_real = b_read_data[BUTTERFLY_WIDTH - 1 : 0];
assign b_imaginary = b_read_data[WORD_WIDTH - 1 : BUTTERFLY_WIDTH];

assign magnitude_exceeded = butterfly_write_valid
    & (over_threshold(a_out_real) | over_threshold(a_out_imaginary)
       | over_threshold(b_out_real) | over_threshold(b_out_imaginary));

fft_scratch_buffer #(.WORD_WIDTH(WORD_WIDTH)) scratch (
    .clk (clk),
    .a_write_enable (scratch_a_write_enable),
    .b_write_enable (butterfly_write_valid),
    .a_write_address (scratch_a_write_address),
    .b_write_address (b_write_address),
    .a_write_data (scratch_a_write_data),
    .b_write_data (b_write_data),
    .a_read_address (a_address),
    .b_read_address (b_address),
    .a_read_data (a_read_data),
    .b_read_data (b_read_data)
);

always_ff @(posedge clk) begin
    if (rst) begin
        for (int stage = 0; stage < WRITEBACK_DELAY; stage++) begin
            a_address_delay[stage] <= 0;
            b_address_delay[stage] <= 0;
            butterfly_valid_delay[stage] <= 0;
        end
    end
    else begin
        a_address_delay[0] <= a_address;
        b_address_delay[0] <= b_address;
        butterfly_valid_delay[0] <= butterfly_valid;
        for (int stage = 1; stage < WRITEBACK_DELAY; stage++) begin
            a_address_delay[stage] <= a_address_delay[stage - 1];
            b_address_delay[stage] <= b_address_delay[stage - 1];
            butterfly_valid_delay[stage] <= butterfly_valid_delay[stage - 1];
        end
    end
end

endmodule


// Twiddles for the 2048-point engine, one word per address
module fft_twiddle_rom #(parameter W_WIDTH = 8,
                         parameter TWIDDLE_FILE = "wesad/rtl/twiddle.hex")(
    input logic clk,
    input logic [9 : 0] w_address,
    output logic signed [W_WIDTH - 1 : 0] w_real,
    output logic signed [W_WIDTH - 1 : 0] w_imaginary
);

logic [2*W_WIDTH - 1 : 0] twiddle_memory [0 : 1023];

initial $readmemh(TWIDDLE_FILE, twiddle_memory);

always_ff @(posedge clk) begin
    w_real <= twiddle_memory[w_address][W_WIDTH - 1 : 0];
    w_imaginary <= twiddle_memory[w_address][2*W_WIDTH - 1 : W_WIDTH];
end

endmodule



// Hann windows for the three segment lengths, stored half
module hann_rom #(parameter W_WIDTH = 8,
                  parameter HANN_FILE = "wesad/rtl/hann.hex")(
    input logic clk,
    input logic [1 : 0] table_select,
    input logic [10 : 0] hann_address,
    output logic signed [W_WIDTH - 1 : 0] hann_coefficient
);

logic [W_WIDTH - 1 : 0] hann_memory [0 : 2733];
logic [11 : 0] table_base;
logic [10 : 0] half_length;
logic [10 : 0] last_index;
logic [10 : 0] folded_address;
logic [11 : 0] memory_address;

initial $readmemh(HANN_FILE, hann_memory);

always_comb begin
    case (table_select)
        2'd0 : begin table_base = 0; half_length = 1024; last_index = 2047; end
        2'd1 : begin table_base = 1024; half_length = 960; last_index = 1919; end
        default : begin table_base = 1984; half_length = 750; last_index = 1499; end
    endcase
    // the window is symmetric, so the upper half folds onto the stored half
    folded_address = (hann_address < half_length) ? hann_address : last_index - hann_address;
    memory_address = table_base + {1'b0, folded_address};
end

always_ff @(posedge clk) hann_coefficient <= hann_memory[memory_address];

endmodule



// One pass over a segment gives its mean and block-float input shift
module segment_stats #(parameter SAMPLE_WIDTH = 8, parameter BUTTERFLY_WIDTH = 8)(
    input logic clk,rst,
    input logic start,
    input logic in_valid,
    input logic signed [SAMPLE_WIDTH - 1 : 0] sample,
    input logic [11 : 0] segment_length,

    output logic signed [SAMPLE_WIDTH - 1 : 0] segment_mean,
    output logic signed [4 : 0] normalise_shift,
    output logic done
);

localparam int SUM_WIDTH = SAMPLE_WIDTH + 12;
localparam int REMAINDER_WIDTH = 13;
localparam int PEAK_WIDTH = SAMPLE_WIDTH + 1;
localparam logic [1 : 0] IDLE = 2'd0;
localparam logic [1 : 0] ACCUMULATE = 2'd1;
localparam logic [1 : 0] DIVIDE = 2'd2;
localparam logic [1 : 0] FINISH = 2'd3;
localparam logic [5 : 0] STEP_INIT = SUM_WIDTH[5 : 0];
localparam logic [5 : 0] TARGET_BITS = BUTTERFLY_WIDTH[5 : 0];

logic [1 : 0] state;
logic signed [SUM_WIDTH - 1 : 0] sum;
logic signed [SUM_WIDTH - 1 : 0] sample_extended;
logic signed [SUM_WIDTH - 1 : 0] next_sum;
logic signed [SAMPLE_WIDTH - 1 : 0] minimum;
logic signed [SAMPLE_WIDTH - 1 : 0] maximum;
logic [11 : 0] count;

logic [REMAINDER_WIDTH - 1 : 0] divisor;
logic [SUM_WIDTH - 1 : 0] dividend;
logic [REMAINDER_WIDTH - 1 : 0] remainder;
logic [SUM_WIDTH - 1 : 0] quotient;
logic [REMAINDER_WIDTH - 1 : 0] shifted;
logic [SUM_WIDTH - 1 : 0] rounded;
logic [REMAINDER_WIDTH : 0] twice_remainder;
logic [5 : 0] step;
logic negative;

logic signed [PEAK_WIDTH - 1 : 0] high_gap;
logic signed [PEAK_WIDTH - 1 : 0] low_gap;
logic [PEAK_WIDTH - 1 : 0] peak;
logic [5 : 0] peak_bits;
logic signed [5 : 0] shift_next;
logic signed [SAMPLE_WIDTH - 1 : 0] mean_magnitude;

assign sample_extended = {{(SUM_WIDTH - SAMPLE_WIDTH){sample[SAMPLE_WIDTH - 1]}}, sample};
assign next_sum = sum + sample_extended;
assign divisor = {{(REMAINDER_WIDTH - 12){1'b0}}, segment_length};
assign mean_magnitude = rounded[SAMPLE_WIDTH - 1 : 0];

always_comb begin
    shifted = {remainder[REMAINDER_WIDTH - 2 : 0], dividend[SUM_WIDTH - 1]};
    twice_remainder = {remainder, 1'b0};
    // np.rint is half to even, so an exact half takes the low bit of the quotient
    if (twice_remainder > {1'b0, divisor})
        rounded = quotient + 1;
    else if (twice_remainder == {1'b0, divisor})
        rounded = quotient + {{(SUM_WIDTH - 1){1'b0}}, quotient[0]};
    else
        rounded = quotient;

    high_gap = {maximum[SAMPLE_WIDTH - 1], maximum} - {segment_mean[SAMPLE_WIDTH - 1], segment_mean};
    low_gap = {segment_mean[SAMPLE_WIDTH - 1], segment_mean} - {minimum[SAMPLE_WIDTH - 1], minimum};
    peak = (high_gap > low_gap) ? high_gap : low_gap;
    peak_bits = 0;
    for (int i = 0; i < PEAK_WIDTH; i++)
        if (peak[i]) peak_bits = i[5 : 0] + 6'd1;
    shift_next = ((peak_bits == 0) ? 6'd1 : peak_bits + 6'd1) - TARGET_BITS;
end

always_ff @(posedge clk) begin
    if (rst) begin
        state <= IDLE;
        sum <= 0;
        minimum <= 0;
        maximum <= 0;
        count <= 0;
        done <= 0;
        segment_mean <= 0;
        normalise_shift <= 0;
    end
    else begin
        done <= 0;
        case (state)
            IDLE : begin
                if (start) begin
                    sum <= 0;
                    count <= 0;
                    minimum <= {1'b0, {(SAMPLE_WIDTH - 1){1'b1}}};
                    maximum <= {1'b1, {(SAMPLE_WIDTH - 1){1'b0}}};
                    state <= ACCUMULATE;
                end
            end
            ACCUMULATE : begin
                if (in_valid) begin
                    sum <= next_sum;
                    if (sample < minimum) minimum <= sample;
                    if (sample > maximum) maximum <= sample;
                    count <= count + 1;
                    if (count == segment_length - 1) begin
                        negative <= next_sum < 0;
                        dividend <= (next_sum < 0) ? -next_sum : next_sum;
                        remainder <= 0;
                        quotient <= 0;
                        step <= STEP_INIT;
                        state <= DIVIDE;
                    end
                end
            end
            DIVIDE : begin
                if (step != 0) begin
                    if (shifted >= divisor) begin
                        remainder <= shifted - divisor;
                        quotient <= {quotient[SUM_WIDTH - 2 : 0], 1'b1};
                    end
                    else begin
                        remainder <= shifted;
                        quotient <= {quotient[SUM_WIDTH - 2 : 0], 1'b0};
                    end
                    dividend <= {dividend[SUM_WIDTH - 2 : 0], 1'b0};
                    step <= step - 1;
                end
                else begin
                    segment_mean <= negative ? -mean_magnitude : mean_magnitude;
                    state <= FINISH;
                end
            end
            FINISH : begin
                normalise_shift <= shift_next[4 : 0];
                done <= 1;
                state <= IDLE;
            end
        endcase
    end
end

endmodule

module fft_engine (
    input logic clk, rst,
    input logic phase,
    input logic stats_pass,
    input logic pass_start,
    input logic pass_done,
    input logic stream_valid,
    input logic signed [15 : 0] stream_data,
    input logic [11 : 0] segment_length,
    input logic [1 : 0] table_select,

    output logic signed [7 : 0] mul_x,
    output logic signed [7 : 0] mul_y,
    input logic signed [15 : 0] mul_p,

    output logic busy,
    output logic signed [6 : 0] segment_exponent,
    output logic exponent_valid,
    output logic bin_valid,
    output logic [10 : 0] bin_address,
    output logic signed [7 : 0] bin_real,
    output logic signed [7 : 0] bin_imaginary,
    output logic spectrum_done
);

localparam logic [2 : 0] IDLE = 3'd0;
localparam logic [2 : 0] STATS = 3'd1;
localparam logic [2 : 0] LOAD = 3'd2;
localparam logic [2 : 0] PAD = 3'd3;
localparam logic [2 : 0] XFORM = 3'd4;
localparam logic [2 : 0] READ = 3'd5;

logic [2 : 0] state;
logic [11 : 0] write_count;
logic [11 : 0] pad_count;
logic [10 : 0] read_count;
logic stream_ended;

logic stats_start, stats_done;
logic signed [15 : 0] segment_mean;
logic signed [4 : 0] normalise_shift;

logic pre_start;
logic [10 : 0] pre_hann_address;
logic signed [7 : 0] pre_sample;
logic pre_valid;

logic [1 : 0] hann_select;
logic [10 : 0] hann_address;
logic signed [7 : 0] hann_coefficient;

logic ag_start, ag_done;
logic [10 : 0] ag_a_address, ag_b_address;
logic [9 : 0] w_address;
logic ag_valid, shift_enable;
logic [3 : 0] block_exponent;

logic signed [7 : 0] w_real, w_imaginary;
logic [10 : 0] scratch_a_address;
logic butterfly_in_valid;
logic signed [7 : 0] a_real, a_imaginary, b_real, b_imaginary;
logic signed [7 : 0] a_out_real, a_out_imaginary, b_out_real, b_out_imaginary;
logic butterfly_out_valid;
logic magnitude_exceeded;

logic load_valid;
logic [10 : 0] load_address;
logic signed [7 : 0] load_sample;
logic [7 : 0] load_peak;
logic [7 : 0] load_magnitude;
logic initial_shift;

assign busy = (state != IDLE);
assign hann_select = table_select;
assign hann_address = pre_hann_address;
assign scratch_a_address = (state == READ) ? read_count : ag_a_address;
assign stats_start = pass_start & phase & stats_pass;
assign pre_start = pass_start & phase & ~stats_pass;

assign load_valid = (state == LOAD) ? pre_valid : (state == PAD);
assign load_address = (state == LOAD) ? write_count[10 : 0] : pad_count[10 : 0];
assign load_sample = (state == LOAD) ? pre_sample : 8'sd0;

assign ag_start = (state == PAD) & (pad_count == 12'd2048);
assign load_magnitude = pre_sample[7] ? (~pre_sample + 8'd1) : pre_sample;
// golden rule: shift while 5*peak > 2*lim, i.e. peak > 50 at an 8-bit datapath
assign initial_shift = (load_peak > 8'd50);

assign spectrum_done = (state == READ) & (read_count == 11'd1026);

segment_stats #(.SAMPLE_WIDTH(16), .BUTTERFLY_WIDTH(8)) stats (
    .clk (clk),
    .rst (rst),
    .start (stats_start),
    .in_valid (stream_valid & (state == STATS)),
    .sample (stream_data),
    .segment_length (segment_length),
    .segment_mean (segment_mean),
    .normalise_shift (normalise_shift),
    .done (stats_done)
);

preprocessing #(.BUTTERFLY_WIDTH(8), .W_WIDTH(8), .SAMPLE_WIDTH(16)) pre (
    .clk (clk),
    .rst (rst),
    .start (pre_start),
    .in_valid (stream_valid & (state == LOAD)),
    .sample (stream_data),
    .segment_mean (segment_mean),
    .normalise_shift (normalise_shift),
    .hann_address (pre_hann_address),
    .hann_coefficient (hann_coefficient),
    .mul_x (mul_x),
    .mul_y (mul_y),
    .mul_p (mul_p),
    .sample_out (pre_sample),
    .out_valid (pre_valid)
);

hann_rom #(.W_WIDTH(8)) hann (
    .clk (clk),
    .table_select (hann_select),
    .hann_address (hann_address),
    .hann_coefficient (hann_coefficient)
);

address_generator ag (
    .clk (clk),
    .rst (rst),
    .start (ag_start),
    .initial_shift (initial_shift),
    .magnitude_exceeded (magnitude_exceeded),
    .a_address (ag_a_address),
    .b_address (ag_b_address),
    .w_address (w_address),
    .butterfly_valid (ag_valid),
    .shift_enable (shift_enable),
    .block_exponent (block_exponent),
    .done (ag_done)
);

fft_twiddle_rom #(.W_WIDTH(8)) twiddle (
    .clk (clk),
    .w_address (w_address),
    .w_real (w_real),
    .w_imaginary (w_imaginary)
);

fft_scratch_memory #(.BUTTERFLY_WIDTH(8)) scratch (
    .clk (clk),
    .rst (rst),
    .load_valid (load_valid),
    .load_address (load_address),
    .load_sample (load_sample),
    .butterfly_valid (ag_valid),
    .a_address (scratch_a_address),
    .b_address (ag_b_address),
    .butterfly_in_valid (butterfly_in_valid),
    .a_real (a_real),
    .a_imaginary (a_imaginary),
    .b_real (b_real),
    .b_imaginary (b_imaginary),
    .a_out_real (a_out_real),
    .a_out_imaginary (a_out_imaginary),
    .b_out_real (b_out_real),
    .b_out_imaginary (b_out_imaginary),
    .magnitude_exceeded (magnitude_exceeded)
);

complex_butterfly #(.BUTTERFLY_WIDTH(8), .W_WIDTH(8)) butterfly (
    .clk (clk),
    .rst (rst),
    .in_valid (butterfly_in_valid),
    .shift_enable (shift_enable),
    .a_real (a_real),
    .a_imaginary (a_imaginary),
    .b_real (b_real),
    .b_imaginary (b_imaginary),
    .w_real (w_real),
    .w_imaginary (w_imaginary),
    .a_out_real (a_out_real),
    .a_out_imaginary (a_out_imaginary),
    .b_out_real (b_out_real),
    .b_out_imaginary (b_out_imaginary),
    .out_valid (butterfly_out_valid)
);

assign bin_real = a_real;
assign bin_imaginary = a_imaginary;

always_ff @(posedge clk) begin
    if (rst) begin
        state <= IDLE;
        write_count <= 0;
        pad_count <= 0;
        read_count <= 0;
        stream_ended <= 0;
        bin_valid <= 0;
        exponent_valid <= 0;
        segment_exponent <= 0;
    end
    else begin
        bin_valid <= 0;
        exponent_valid <= 0;

        case (state)
            IDLE : begin
                if (stats_start) state <= STATS;
                else if (pre_start) begin
                    write_count <= 0;
                    stream_ended <= 0;
                    load_peak <= 0;
                    state <= LOAD;
                end
            end

            STATS : if (stats_done) state <= IDLE;

            LOAD : begin
                if (pre_valid) begin
                    write_count <= write_count + 1'b1;
                    if (load_magnitude > load_peak) load_peak <= load_magnitude;
                end
                if (pass_done) stream_ended <= 1;
                // the windowed samples trail the stream by the preprocessing pipeline
                if (stream_ended && write_count == segment_length) begin
                    pad_count <= segment_length;
                    state <= PAD;
                end
            end

            PAD : begin
                if (pad_count != 12'd2048) pad_count <= pad_count + 1'b1;
                else state <= XFORM;
            end

            XFORM : if (ag_done) begin
                segment_exponent <= 7'($signed({1'b0, block_exponent}) + normalise_shift) <<< 1;
                exponent_valid <= 1;
                read_count <= 0;
                state <= READ;
            end

            READ : begin
                if (read_count != 11'd1026) read_count <= read_count + 1'b1;
                // the scratch answers a cycle after the address, so both travel together
                bin_valid <= (read_count <= 11'd1024);
                bin_address <= read_count;
                if (read_count == 11'd1026) state <= IDLE;
            end

            default : state <= IDLE;
        endcase
    end
end

endmodule
