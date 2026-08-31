// Data 0: ECG
// Data 1: Acc
// Data 2: Resp
// Data 3: EDA


module spi_ingest_control (
    input logic clk,
    input logic rst,
    input logic [4:0] data_ready,
    input logic [15:0] wptr, // Write address for the currently-selected sensor, from the pointer module
    input logic mag_done,
    output logic burst_start,
    output logic word_done,
    output logic mag_start,
    output logic [2:0] sensor,
    output logic [4:0] cs,
    output logic [15:0] addr,
    output logic wren,
    output logic [3:0] mask,
    output logic sample_en,
    output logic sclk,
    output logic mosi
);

typedef enum logic [2:0] {IDLE, CS_SETUP, SHIFT, ACCUM, MAG, WRITE} state_e;

localparam DIV = 6;
localparam WORD_BITS = 16;
localparam ACC = 3'd1;
localparam ACC_WORDS = 2'd3;

state_e state = IDLE;
state_e next_state = IDLE;
logic [4:0] pending = '0;
logic [4:0] serviced;

// Default write and read everything in the SPRAM
assign mask = '1;

// Read-only bus, nothing is ever driven to the sensors
assign mosi = 1'b0;


// In case multiple sensors fire DR at the same time
logic [2:0] winner;
assign winner = pending[0] ? 3'd0 :
                pending[1] ? 3'd1 :
                pending[2] ? 3'd2 :
                pending[3] ? 3'd3 : 3'd4;

// Protect against metastability
logic [4:0] dr_meta = '0;
logic [4:0] dr_sync = '0;
logic [4:0] dr_sync_q = '0;

// Counter values for generating SCLK
logic [4:0] sclk_div = '0;
logic sclk_tick;
assign sclk_tick = (sclk_div == DIV - 1);

logic [4:0] tick_cnt = '0;
assign sclk = tick_cnt[0];

// SCLK toggles only inside SHIFT, so CS_SETUP is a quiet half-period with sclk parked low
always @(posedge clk)
    if (rst)
        sclk_div <= '0;
    else if ((state == CS_SETUP) || (state == SHIFT))
        sclk_div <= sclk_tick ? '0 : sclk_div + 1;

// One SPI bit per rising edge of sclk
always @(posedge clk) begin
    if (rst) tick_cnt <= '0;
    else if ((state == CS_SETUP) || (state == ACCUM) || (state == WRITE)) tick_cnt <= '0;
    else if ((state == SHIFT) && sclk_tick) tick_cnt <= tick_cnt + 1;
end

logic [2:0] active = '0;
assign sensor = active;

always @(posedge clk)
    if (rst) active <= '0;
    else if (state == IDLE && |pending) active <= winner;

// ACC returns x/y/z, so one data-ready is three words inside a single CS assertion
logic [1:0] word_cnt = '0;
logic [1:0] last_word;
assign last_word = (active == ACC) ? ACC_WORDS - 1'b1 : '0;

always @(posedge clk) begin
    if (rst) word_cnt <= '0;
    else if (state == CS_SETUP) word_cnt <= '0;
    else if (state == ACCUM) word_cnt <= word_cnt + 1;
end

always @(posedge clk)
    if (rst) mag_start <= 1'b0;
    else mag_start <= (state == ACCUM) && (word_cnt == last_word);

always @(posedge clk) begin
    dr_meta <= rst ? '0 : data_ready;
    dr_sync <= rst ? '0 : dr_meta;
    dr_sync_q <= rst ? '0 : dr_sync;
end

// High for exactly one cycle per DR assertion, however long DR itself stays up
logic [4:0] dr_pulse;
assign dr_pulse = dr_sync & ~dr_sync_q;

// Clears the pending bit of whichever sensor just got written
assign serviced = (state == WRITE) ? (5'b00001 << active) : '0;

// Precaution in case data ready flag is raised while FSM is not in IDLE,
// each data ready leaves its trace in the 4-bit register pending
for (genvar i = 0; i < 5; i = i + 1) begin : pending_bit
    always @(posedge clk) begin
        if (rst) pending[i] <= 1'b0;
        else if (dr_pulse[i]) pending[i] <= 1'b1;
        else if (serviced[i]) pending[i] <= 1'b0;
    end
end

// Update state every clock cycle
always @(posedge clk) if (rst) state <= IDLE; else state <= next_state;

always @* begin
    // defaults:
    cs = '1;
    addr = '0;
    wren = 1'b0;
    sample_en = 1'b0;
    next_state = state;
    burst_start = (state == CS_SETUP);
    word_done = (state == ACCUM);

    // CS is held low for the whole transaction: setup, shift and the write cycle
    if (state != IDLE) cs = ~(5'b00001 << active);

    unique case (state)
        IDLE: begin
            if (pending != '0) next_state = CS_SETUP;
        end

        CS_SETUP: begin
            if (sclk_tick) next_state = SHIFT;
        end

        SHIFT: begin
            // capture on the rising edge; leave on the falling edge that follows bit 16
            if (sclk_tick && ~tick_cnt[0]) sample_en = 1'b1;
            if (sclk_tick && (tick_cnt == 2*WORD_BITS - 1))
                next_state = state_e'((active == ACC) ? ACCUM : WRITE);
        end

        ACCUM: begin
            next_state = state_e'((word_cnt == last_word) ? MAG : SHIFT);
        end

        MAG: begin
            if (mag_done) next_state = WRITE;
        end

        WRITE: begin
            addr = wptr;
            wren = 1'b1;
            next_state = IDLE;
        end
    endcase

end

endmodule
