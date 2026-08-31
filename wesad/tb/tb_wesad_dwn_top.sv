`timescale 1ns/1ps

module tb_wesad_dwn_top;

logic clk = 0;
logic rst = 1;
logic [4:0] dr = 0;
logic miso;
logic sclk, mosi;
logic [4:0] cs;

integer errors = 0;

localparam CLK_PERIOD = 10;

always #(CLK_PERIOD/2) clk = ~clk;

wesad_dwn_top dut (
    .clk  (clk),
    .rst  (rst),
    .dr   (dr),
    .miso (miso),
    .sclk (sclk),
    .mosi (mosi),
    .cs   (cs)
);

// ---------------- sensor model: MSB-first, sampled on rising sclk ----------

wire cs_active = ~&cs;

logic cs_active_q = 0;
logic sclk_q = 0;

always @(posedge clk) begin
    cs_active_q <= cs_active;
    sclk_q      <= sclk;
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

logic [1:0]  word_idx = 0;
logic [4:0]  bit_cnt  = 0;
logic [15:0] tx_shift = 0;

assign miso = tx_shift[15];

function [15:0] pattern(input [2:0] s, input [1:0] w);
    pattern = 16'hA000 | (s << 8) | w;
endfunction

// 16 falling edges per word: 15 shifts, then reload for the next word of a burst
always @(posedge clk) begin
    if (!cs_active) begin
        bit_cnt  <= 0;
        word_idx <= 0;
    end
    else if (!cs_active_q) begin
        bit_cnt  <= 0;
        word_idx <= 0;
        tx_shift <= pattern(sensor_sel, 2'd0);
    end
    else if (sclk_fall) begin
        if (bit_cnt == 15) begin
            bit_cnt  <= 0;
            word_idx <= word_idx + 1;
            tx_shift <= pattern(sensor_sel, word_idx + 1);
        end
        else begin
            bit_cnt  <= bit_cnt + 1;
            tx_shift <= {tx_shift[14:0], 1'b0};
        end
    end
end

// ---------------- helpers -------------------------------------------------

task fire_dr(input [4:0] m);
    begin
        @(posedge clk);
        dr <= m;
        repeat (3) @(posedge clk);
        dr <= 5'b00000;
    end
endtask

// asserts off the clock grid, so the synchronizer sees an unaligned edge
task fire_dr_async(input [4:0] m, input integer phase_ns);
    begin
        @(posedge clk);
        #(phase_ns);
        dr = m;
        #(2*CLK_PERIOD);
        dr = 5'b00000;
    end
endtask

task wait_done;
    begin
        wait (~&cs);
        wait (&cs);
        repeat (4) @(posedge clk);
    end
endtask

task check(input string nm, input [15:0] got, input [15:0] exp);
    begin
        if (got === exp)
            $display("PASS  %-20s = %h", nm, got);
        else begin
            $display("FAIL  %-20s = %h, expected %h", nm, got, exp);
            errors = errors + 1;
        end
    end
endtask

// window_full is a one-cycle pulse, so latch it
logic wf_eda = 0;
always @(posedge clk) if (dut.window_full[3]) wf_eda <= 1;
logic wf_emg = 0;
always @(posedge clk) if (dut.window_full[4]) wf_emg <= 1;

// ---------------- tests ---------------------------------------------------

integer i;

initial begin
    if ($test$plusargs("vcd")) begin
        $dumpfile("tb_wesad_dwn_top.vcd");
        $dumpvars(0, tb_wesad_dwn_top);
    end

    repeat (5) @(posedge clk);
    rst <= 0;
    repeat (5) @(posedge clk);

    // 1. single ECG word -> buffer 0, address 0
    fire_dr(5'b00001);
    wait_done;
    check("ecg[0]",     dut.spram.spram_16k_0.mem[0], 16'hA000);
    check("offset_ecg", dut.offset_ecg,             16'd1);

    // 2. ACC burst -> three consecutive words in buffer 1
    fire_dr(5'b00010);
    wait_done;
    check("acc_mag[0]", dut.spram.spram_16k_0.mem[15000], 16'hA489);
    check("acc_mag_unwritten", dut.spram.spram_16k_0.mem[15001], 16'hxxxx);
    check("offset_acc", dut.offset_acc,             16'd1);

    // 3. ECG and Resp asserted together: both must be serviced
    fire_dr(5'b00101);
    wait_done;
    wait_done;
    check("ecg[1]",  dut.spram.spram_16k_0.mem[1],    16'hA000);
    check("resp[0]", dut.spram.spram_16k_1.mem[536], 16'hA200);

    // 4. asynchronous data-ready at every phase of the clock: one write each
    for (i = 0; i < CLK_PERIOD; i = i + 1) begin
        fire_dr_async(5'b00100, i);
        wait_done;
    end
    for (i = 0; i < CLK_PERIOD; i = i + 1)
        check($sformatf("resp_phase%0d", i),
              dut.spram.spram_16k_1.mem[537+i], 16'hA200);
    check("offset_resp", dut.offset_resp, 16'd11);

    // 5. data-ready held high across a whole transaction: still one write
    @(posedge clk);
    #3 dr = 5'b00100;
    wait (~&cs);
    wait (&cs);
    repeat (50) @(posedge clk);
    dr = 5'b00000;
    repeat (400) @(posedge clk);
    check("resp_held", dut.spram.spram_16k_1.mem[547], 16'hA200);
    check("offset_resp_held", dut.offset_resp, 16'd12);

    // 6. EDA wrap after 1500 words
    for (i = 0; i < 1500; i = i + 1) begin
        fire_dr(5'b01000);
        wait_done;
    end
    check("eda[0]",          dut.spram.spram_16k_1.mem[2036],      16'hA300);
    check("eda[1499]",       dut.spram.spram_16k_1.mem[2036+1499], 16'hA300);
    check("offset_eda",      dut.offset_eda,                     16'd0);
    check("window_full_eda", {15'd0, wf_eda},                      16'd1);

    for (i = 0; i < 21000; i = i + 1) begin
        fire_dr(5'b10000);
        wait_done;
    end
    check("emg[0]",          dut.spram.spram_16k_1.mem[3536],       16'hA400);
    check("emg[12847]",      dut.spram.spram_16k_1.mem[16383],      16'hA400);
    check("emg[12848]",      dut.spram.spram_16k_2.mem[0],          16'hA400);
    check("emg[20999]",      dut.spram.spram_16k_2.mem[8151],       16'hA400);
    check("offset_emg",      dut.offset_emg,                      16'd0);
    check("window_full_emg", {15'd0, wf_emg},                       16'd1);

    if (errors == 0) $display("\nTB PASS");
    else             $display("\nTB FAIL (%0d errors)", errors);
    $finish;
end

initial begin
    #90_000_000;
    $display("\nTB TIMEOUT");
    $finish;
end

endmodule
