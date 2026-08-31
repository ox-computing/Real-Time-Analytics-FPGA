`timescale 1ns/1ps

module tb_wesad_i2c;

localparam CLK_FREQ = 12_000_000;
localparam HALF = 5;
localparam [6:0] ADDR = 7'h40;

localparam N_ECG = 15000, N_ACC = 1920, N_RESP = 1500, N_EDA = 1500, N_EMG = 21000;

logic clk = 0;
always #5 clk = ~clk;

logic scl = 1'b1;
logic m_oe = 1'b0;
wire sda;
assign sda = m_oe ? 1'b0 : 1'bz;
pullup (sda);

wesad_i2c_top #(.CLK_FREQ(CLK_FREQ), .I2C_ADDR(ADDR)) dut (
    .clk (clk),
    .scl (scl),
    .sda (sda)
);

logic [15:0] g_ecg [0:N_ECG-1];
logic [15:0] g_accx [0:N_ACC-1];
logic [15:0] g_accy [0:N_ACC-1];
logic [15:0] g_accz [0:N_ACC-1];
logic [15:0] g_resp [0:N_RESP-1];
logic [15:0] g_eda [0:N_EDA-1];
logic [15:0] g_emg [0:N_EMG-1];
logic [7:0] exp_feat [0:114];
logic exp_bits [0:459];
logic [15:0] exp_class [0:0];

integer errors = 0;
integer i;
string hexdir;
logic [7:0] resp_byte [0:3];
logic ack;

task automatic send_bit (input logic b);
    begin
        m_oe = ~b;
        repeat (HALF) @(posedge clk);
        scl = 1'b1;
        repeat (HALF) @(posedge clk);
        scl = 1'b0;
    end
endtask

task automatic recv_bit (output logic b);
    begin
        m_oe = 1'b0;
        repeat (HALF) @(posedge clk);
        scl = 1'b1;
        repeat (HALF - 1) @(posedge clk);
        b = sda;
        @(posedge clk);
        scl = 1'b0;
    end
endtask

task automatic i2c_start;
    begin
        m_oe = 1'b0;
        scl = 1'b1;
        repeat (HALF) @(posedge clk);
        m_oe = 1'b1;
        repeat (HALF) @(posedge clk);
        scl = 1'b0;
    end
endtask

task automatic i2c_stop;
    begin
        m_oe = 1'b1;
        repeat (HALF) @(posedge clk);
        scl = 1'b1;
        repeat (HALF) @(posedge clk);
        m_oe = 1'b0;
        repeat (HALF) @(posedge clk);
    end
endtask

task automatic write_byte (input logic [7:0] d);
    integer k;
    begin
        for (k = 7; k >= 0; k = k - 1) send_bit(d[k]);
        recv_bit(ack);
        if (ack !== 1'b0) begin
            $display("FAIL  no ack for byte %02x at %0t", d, $time);
            errors = errors + 1;
        end
    end
endtask

task automatic write_word (input logic [15:0] w);
    begin
        write_byte(w[7:0]);
        write_byte(w[15:8]);
    end
endtask

task automatic read_byte (input logic last, output logic [7:0] d);
    integer k;
    begin
        for (k = 7; k >= 0; k = k - 1) recv_bit(d[k]);
        send_bit(last);
    end
endtask

initial begin
    if (!$value$plusargs("hex=%s", hexdir)) hexdir = "wesad/sim_fixtures/";
    $readmemh({hexdir, "ecg.hex"}, g_ecg);
    $readmemh({hexdir, "acc_x.hex"}, g_accx);
    $readmemh({hexdir, "acc_y.hex"}, g_accy);
    $readmemh({hexdir, "acc_z.hex"}, g_accz);
    $readmemh({hexdir, "resp.hex"}, g_resp);
    $readmemh({hexdir, "eda.hex"}, g_eda);
    $readmemh({hexdir, "emg.hex"}, g_emg);
    $readmemh({hexdir, "feat.hex"}, exp_feat);
    $readmemh({hexdir, "bits.hex"}, exp_bits);
    $readmemh({hexdir, "class.hex"}, exp_class);

    repeat (64) @(posedge clk);

    i2c_start;
    write_byte({ADDR, 1'b0});

    for (i = 0; i < N_ECG; i = i + 1) write_word(g_ecg[i]);
    for (i = 0; i < N_ACC; i = i + 1) begin
        write_word(g_accx[i]);
        write_word(g_accy[i]);
        write_word(g_accz[i]);
    end
    for (i = 0; i < N_RESP; i = i + 1) write_word(g_resp[i]);
    for (i = 0; i < N_EDA; i = i + 1) write_word(g_eda[i]);
    for (i = 0; i < N_EMG; i = i + 1) write_word(g_emg[i]);

    i2c_stop;

    $display("window uploaded at %0t", $time);

    wait (dut.encode_done);
    @(negedge clk);

    for (i = 0; i < 115; i = i + 1)
        if (dut.storage.feature_mem[i] !== exp_feat[i]) begin
            $display("FAIL  feature %0d got %0d expected %0d",
                     i, $signed(dut.storage.feature_mem[i]), $signed(exp_feat[i]));
            errors = errors + 1;
        end

    for (i = 0; i < 460; i = i + 1)
        if (dut.binary_vector[i] !== exp_bits[i]) begin
            $display("FAIL  bit %0d got %b expected %b", i, dut.binary_vector[i], exp_bits[i]);
            errors = errors + 1;
        end

    i2c_start;
    write_byte({ADDR, 1'b1});
    for (i = 0; i < 4; i = i + 1) read_byte(i == 3, resp_byte[i]);
    i2c_stop;

    if (resp_byte[0] !== exp_class[0][7:0]) begin
        $display("FAIL  class got %0d expected %0d", resp_byte[0], exp_class[0]);
        errors = errors + 1;
    end

    $display("i2c response: class %0d scores [%0d %0d %0d]",
             resp_byte[0], resp_byte[1], resp_byte[2], resp_byte[3]);

    if (errors == 0) $display("\nI2C TB PASS  class %0d", resp_byte[0]);
    else $display("\nI2C TB FAIL (%0d errors)", errors);
    $finish;
end

initial begin
    #4_000_000_000;
    $display("\nI2C TB TIMEOUT");
    $finish;
end

endmodule
