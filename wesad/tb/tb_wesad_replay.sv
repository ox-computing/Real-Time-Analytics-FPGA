`timescale 1ns/1ps

// Replays one decimated WESAD window through the SPI sensors at the real
// per-modality rate ratios, then streams every region back out through the read
// path and checks it sample-for-sample against the source.
//
// Hex stimulus comes from wesad/scripts/rtl_fixtures/make_replay_hex.py (not in the repo -- it is
// dataset content). Compile with the iCE40 cell models:
//   iverilog -g2012 -s tb_wesad_replay wesad/tb/tb_wesad_replay.sv \
//            $(find wesad/rtl -name '*.sv') $(yosys-config --datdir)/ice40/cells_sim.v
//   vvp a.out +hex=<dir>/

module tb_wesad_replay;

localparam N_ECG = 15000, N_ACC = 1920, N_RESP = 1500, N_EDA = 1500, N_EMG = 21000;

// EMG (350 Hz) fires every 600 cycles; the rest scale by rate ratio, so all five
// regions fill in the same span. Bus utilisation lands near 0.76.
localparam P_ECG = 840, P_ACC = 6562, P_RESP = 8400, P_EDA = 8400, P_EMG = 600;

logic clk = 0, rst = 1;
always #5 clk = ~clk;

logic [4:0] dr = 0;
logic miso;
logic sclk, mosi;
logic [4:0] cs;

wesad_dwn_top dut (
    .clk(clk), .rst(rst), .dr(dr), .miso(miso),
    .sclk(sclk), .mosi(mosi), .cs(cs)
);

logic [15:0] g_ecg [0:N_ECG-1];
logic [15:0] g_acc [0:N_ACC-1];
logic [15:0] g_resp[0:N_RESP-1];
logic [15:0] g_eda [0:N_EDA-1];
logic [15:0] g_emg [0:N_EMG-1];

int idx [0:4];
int fired [0:4];
int cnt [0:4];
int per [0:4];
int num [0:4];
int errors = 0;

// DEAD marks a read past the end of the replay: the one sample written after the
// frame snapshot, which must never reach the stream.
function [15:0] golden(input int s, input int i);
    case (s)
        0: golden = (i < N_ECG)  ? g_ecg[i]  : 16'hDEAD;
        1: golden = (i < N_ACC)  ? g_acc[i]  : 16'hDEAD;
        2: golden = (i < N_RESP) ? g_resp[i] : 16'hDEAD;
        3: golden = (i < N_EDA)  ? g_eda[i]  : 16'hDEAD;
        default: golden = (i < N_EMG) ? g_emg[i] : 16'hDEAD;
    endcase
endfunction

// ---------------- sensor model -------------------------------------------

wire cs_active = ~&cs;
logic cs_active_q = 0, sclk_q = 0;
always @(posedge clk) begin
    cs_active_q <= cs_active;
    sclk_q <= sclk;
end
wire sclk_fall = sclk_q & ~sclk;

logic [2:0] sensor_sel;
always @* begin
    case (cs)
        5'b11110: sensor_sel = 3'd0;
        5'b11101: sensor_sel = 3'd1;
        5'b11011: sensor_sel = 3'd2;
        5'b10111: sensor_sel = 3'd3;
        default:  sensor_sel = 3'd4;
    endcase
end

logic [2:0] sensor_active = 0;
always @(posedge clk) if (cs_active) sensor_active <= sensor_sel;

// ACC is cached as magnitude already, so the three-word burst is (mag, 0, 0);
// acc_magnitude returns |mag| and ACC codes are non-negative, so it round-trips.
function [15:0] word_for(input int s, input int w);
    word_for = (s == 3'd1) ? ((w == 0) ? golden(1, idx[1]) : 16'd0)
                           : golden(s, idx[s]);
endfunction

logic [1:0] word_idx = 0;
logic [4:0] bit_cnt = 0;
logic [15:0] tx_shift = 0;
assign miso = tx_shift[15];

always @(posedge clk) begin
    if (!cs_active) begin
        bit_cnt <= 0;
        word_idx <= 0;
    end
    else if (!cs_active_q) begin
        bit_cnt <= 0;
        word_idx <= 0;
        tx_shift <= word_for(sensor_sel, 0);
    end
    else if (sclk_fall) begin
        if (bit_cnt == 15) begin
            bit_cnt <= 0;
            word_idx <= word_idx + 1;
            tx_shift <= word_for(sensor_sel, word_idx + 1);
        end
        else begin
            bit_cnt <= bit_cnt + 1;
            tx_shift <= {tx_shift[14:0], 1'b0};
        end
    end
end

always @(posedge clk)
    if (cs_active_q && !cs_active) idx[sensor_active] <= idx[sensor_active] + 1;

// ---------------- data-ready at scaled sensor rates -----------------------

// A data-ready raised while the sensor's pending bit is still set is absorbed by
// the ingest FSM and that sample is lost, so the period is held until the bit
// clears rather than fired blind. Every pulse issued therefore becomes exactly
// one write, and each region takes exactly one window.
integer k;
always @(posedge clk) begin
    if (rst) begin
        dr <= 0;
        for (k = 0; k < 5; k = k + 1) begin
            cnt[k] <= 0;
            fired[k] <= 0;
        end
    end
    else begin
        dr <= 0;
        for (k = 0; k < 5; k = k + 1) begin
            if (fired[k] < num[k]) begin
                if (cnt[k] == per[k] - 1) begin
                    if (!dut.control.pending[k]) begin
                        cnt[k] <= 0;
                        dr[k] <= 1'b1;
                        fired[k] <= fired[k] + 1;
                    end
                end
                else cnt[k] <= cnt[k] + 1;
            end
        end
    end
end

// ---------------- read-path driver ----------------------------------------

logic [2:0] rq_sel = 0;
logic [14:0] rq_si = 0, rq_n = 0;
logic rq_start = 0, rq_ready = 1;

initial begin
    force dut.sensor_select = rq_sel;
    force dut.start_index = rq_si;
    force dut.sample_count = rq_n;
    force dut.start_reading = rq_start;
    force dut.stream_ready = rq_ready;
end

task automatic read_pass(input int sel, input int si, input int n);
    int beat;
    begin
        @(negedge clk);
        rq_sel = sel[2:0];
        rq_si = si[14:0];
        rq_n = n[14:0];
        rq_start = 1'b1;
        @(negedge clk);
        rq_start = 1'b0;
        beat = 0;
        while (dut.reader_busy || dut.stream_valid) begin
            @(posedge clk);
            #1;
            if (dut.stream_valid) begin
                if (dut.stream_data !== golden(sel, si + beat)) begin
                    if (errors < 10)
                        $display("FAIL  sel=%0d window_idx=%0d got %h expected %h",
                                 sel, si + beat, dut.stream_data, golden(sel, si + beat));
                    errors = errors + 1;
                end
                if (dut.stream_last !== (beat == n - 1)) begin
                    $display("FAIL  sel=%0d beat %0d stream_last %b", sel, beat, dut.stream_last);
                    errors = errors + 1;
                end
                beat = beat + 1;
            end
            @(negedge clk);
        end
        if (beat != n) begin
            $display("FAIL  sel=%0d got %0d beats, expected %0d", sel, beat, n);
            errors = errors + 1;
        end
        else $display("PASS  sel=%0d start_index=%0d count=%0d", sel, si, n);
    end
endtask

task check_int(input string nm, input int got, input int exp);
    if (got !== exp) begin
        $display("FAIL  %s = %0d, expected %0d", nm, got, exp);
        errors = errors + 1;
    end
    else $display("PASS  %s = %0d", nm, got);
endtask

// ---------------- run ------------------------------------------------------

string hexdir;

initial begin
    per[0] = P_ECG;  per[1] = P_ACC;  per[2] = P_RESP;  per[3] = P_EDA;  per[4] = P_EMG;
    num[0] = N_ECG;  num[1] = N_ACC;  num[2] = N_RESP;  num[3] = N_EDA;  num[4] = N_EMG;
    for (k = 0; k < 5; k = k + 1) idx[k] = 0;

    if (!$value$plusargs("hex=%s", hexdir)) hexdir = "";
    $readmemh({hexdir, "ecg.hex"},  g_ecg);
    $readmemh({hexdir, "acc.hex"},  g_acc);
    $readmemh({hexdir, "resp.hex"}, g_resp);
    $readmemh({hexdir, "eda.hex"},  g_eda);
    $readmemh({hexdir, "emg.hex"},  g_emg);

    repeat (5) @(negedge clk);
    rst = 0;

    // The replay stops after exactly one window per region, so every write
    // pointer lands back on zero and window index i is source sample i.
    wait (dut.frame_start == 1'b1);
    @(negedge clk);
    $display("frame captured at %0t", $time);

    check_int("off_ecg_ff",  dut.off_ecg_ff,  0);
    check_int("off_acc_ff",  dut.off_acc_ff,  0);
    check_int("off_resp_ff", dut.off_resp_ff, 0);
    check_int("off_eda_ff",  dut.off_eda_ff,  0);
    check_int("off_emg_ff",  dut.off_emg_ff,  0);
    check_int("head_ecg",    dut.head_ecg,    g_ecg[0]);
    check_int("head_emg",    dut.head_emg,    g_emg[0]);
    check_int("overrun",     dut.overrun,     0);

    // One EMG sample lands after the snapshot and overwrites window index 0 in
    // the SPRAM. The head register must still win, or beat 0 reads DEAD.
    force dr = 5'b10000;
    repeat (2) @(negedge clk);
    force dr = 5'b00000;
    wait (~&cs);
    wait (&cs);
    repeat (4) @(negedge clk);
    release dr;

    read_pass(4, 0, N_EMG);
    read_pass(0, 0, N_ECG);
    read_pass(1, 0, N_ACC);
    read_pass(2, 0, N_RESP);
    read_pass(3, 0, N_EDA);

    // Welch segments: k passes 1024*k
    read_pass(4, 1024, 2048);
    read_pass(0, 12288, 2048);

    if (errors == 0) $display("\nTB PASS");
    else             $display("\nTB FAIL (%0d errors)", errors);
    $finish;
end

initial begin
    #400_000_000;
    $display("\nTB TIMEOUT");
    $finish;
end

endmodule
