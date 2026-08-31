`timescale 1ns/1ps

// Replays one window through the SPI sensors and checks the 115 stored features, the 460 thermometer bits and the class against the hex files from make_infer_fixture.py.

module tb_wesad_infer;

localparam N_ECG = 15000, N_ACC = 1920, N_RESP = 1500, N_EDA = 1500, N_EMG = 21000;
localparam P_ECG = 840, P_ACC = 6562, P_RESP = 8400, P_EDA = 8400, P_EMG = 600;

logic clk = 0, rst = 1;
always #5 clk = ~clk;

logic [4:0] dr = 0;
logic miso;
logic sclk, mosi;
logic [4:0] cs;
logic [1:0] class_idx;
logic result_valid;

wesad_dwn_top dut (
    .clk(clk), .rst(rst), .dr(dr), .miso(miso),
    .sclk(sclk), .mosi(mosi), .cs(cs),
    .class_idx(class_idx), .result_valid(result_valid)
);

logic [15:0] g_ecg [0:N_ECG-1];
logic [15:0] g_acc [0:N_ACC-1];
logic [15:0] g_accx[0:N_ACC-1];
logic [15:0] g_accy[0:N_ACC-1];
logic [15:0] g_accz[0:N_ACC-1];
logic [15:0] g_resp[0:N_RESP-1];
logic [15:0] g_eda [0:N_EDA-1];
logic [15:0] g_emg [0:N_EMG-1];

logic [7:0] exp_feat [0:114];
logic exp_bits [0:459];
logic [15:0] exp_class [0:0];

int idx [0:4];
int fired [0:4];
int cnt [0:4];
int per [0:4];
int num [0:4];
int errors = 0;
int i;

function [15:0] golden(input int s, input int j);
    case (s)
        0: golden = g_ecg[j];
        1: golden = g_acc[j];
        2: golden = g_resp[j];
        3: golden = g_eda[j];
        default: golden = g_emg[j];
    endcase
endfunction

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

function [15:0] word_for(input int s, input int w);
    word_for = (s != 3'd1) ? golden(s, idx[s])
             : (w == 0) ? g_accx[idx[1]]
             : (w == 1) ? g_accy[idx[1]] : g_accz[idx[1]];
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

string hexdir;

initial begin
    per[0] = P_ECG;  per[1] = P_ACC;  per[2] = P_RESP;  per[3] = P_EDA;  per[4] = P_EMG;
    num[0] = N_ECG;  num[1] = N_ACC;  num[2] = N_RESP;  num[3] = N_EDA;  num[4] = N_EMG;
    for (i = 0; i < 5; i = i + 1) idx[i] = 0;

    if (!$value$plusargs("hex=%s", hexdir)) hexdir = "wesad/sim_fixtures/";
    $readmemh({hexdir, "ecg.hex"},   g_ecg);
    $readmemh({hexdir, "acc.hex"},   g_acc);
    $readmemh({hexdir, "acc_x.hex"}, g_accx);
    $readmemh({hexdir, "acc_y.hex"}, g_accy);
    $readmemh({hexdir, "acc_z.hex"}, g_accz);
    $readmemh({hexdir, "resp.hex"},  g_resp);
    $readmemh({hexdir, "eda.hex"},   g_eda);
    $readmemh({hexdir, "emg.hex"},   g_emg);
    $readmemh({hexdir, "feat.hex"},  exp_feat);
    $readmemh({hexdir, "bits.hex"},  exp_bits);
    $readmemh({hexdir, "class.hex"}, exp_class);

    repeat (5) @(negedge clk);
    rst = 0;

    wait (dut.encode_done);
    @(negedge clk);
    $display("frame encoded at %0t", $time);

    for (i = 0; i < 115; i = i + 1)
        if (dut.storage.feature_mem[i] !== exp_feat[i]) begin
            $display("FAIL  feature %0d got %0d expected %0d",
                     i, $signed(dut.storage.feature_mem[i]), $signed(exp_feat[i]));
            errors = errors + 1;
        end

    for (i = 0; i < 460; i = i + 1)
        if (dut.binary_vector[i] !== exp_bits[i]) begin
            $display("FAIL  bit %0d got %b expected %b",
                     i, dut.binary_vector[i], exp_bits[i]);
            errors = errors + 1;
        end

    wait (result_valid);
    @(negedge clk);
    if (class_idx !== exp_class[0][1:0]) begin
        $display("FAIL  class got %0d expected %0d", class_idx, exp_class[0]);
        errors = errors + 1;
    end

    if (errors == 0) $display("\nTB PASS  class %0d", class_idx);
    else             $display("\nTB FAIL (%0d errors)", errors);
    $finish;
end

initial begin
    #400_000_000;
    $display("\nTB TIMEOUT");
    $finish;
end

endmodule
