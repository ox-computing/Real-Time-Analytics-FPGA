// Streams one window per sensor through time_stats and checks all 50 feature words against the golden model.
`timescale 1ns / 1ps

module tb_time_stats;

logic clk = 0;
logic rst = 1;
always #5 clk = ~clk;

logic [2:0] sensor_select;
logic phase = 0;
logic pass_start = 0;
logic pass_done = 0;
logic signed [15:0] stream_data = 0;
logic stream_last = 0;
logic stream_valid = 0;
logic busy;

logic [15:0] sqrt_result;
logic sqrt_done, sqrt_busy;
logic [31:0] sqrt_data;
logic sqrt_start;

logic feature_wr_en;
logic [6:0] feature_wr_addr;
logic [7:0] feature_wr_data;

logic signed [15:0] mul_x, mul_y;
logic signed [31:0] mul_p;
assign mul_p = mul_x * mul_y;

int errors = 0;

time_stats dut (
    .clk (clk),
    .rst (rst),
    .sensor_select (sensor_select),
    .phase (phase),
    .pass_start (pass_start),
    .pass_done (pass_done),
    .stream_data (stream_data),
    .stream_last (stream_last),
    .stream_valid (stream_valid),
    .busy (busy),
    .sqrt_result (sqrt_result),
    .sqrt_done (sqrt_done),
    .sqrt_busy (sqrt_busy),
    .sqrt_data (sqrt_data),
    .sqrt_start (sqrt_start),
    .mul_x (mul_x),
    .mul_y (mul_y),
    .mul_p (mul_p),
    .feature_wr_en (feature_wr_en),
    .feature_wr_addr (feature_wr_addr),
    .feature_wr_data (feature_wr_data)
);

isqrt32 root (
    .clk (clk),
    .rst (rst),
    .start (sqrt_start),
    .x (sqrt_data),
    .root (sqrt_result),
    .done (sqrt_done),
    .busy (sqrt_busy)
);

logic [15:0] ecg [0:14999];
logic [15:0] acc [0:1919];
logic [15:0] resp [0:1499];
logic [15:0] eda [0:1499];
logic [15:0] emg [0:20999];
logic [7:0] expected [0:49];
logic [7:0] captured [0:49];

int lengths [0:4];

task automatic run_sensor (input int s, input int n);
    @(posedge clk);
    sensor_select <= s[2:0];
    pass_start <= 1;
    @(posedge clk);
    pass_start <= 0;
    for (int i = 0; i < n; i++) begin
        @(posedge clk);
        stream_valid <= 1;
        stream_last <= (i == n - 1);
        case (s)
            0 : stream_data <= ecg[i];
            1 : stream_data <= acc[i];
            2 : stream_data <= resp[i];
            3 : stream_data <= eda[i];
            default : stream_data <= emg[i];
        endcase
    end
    @(posedge clk);
    stream_valid <= 0;
    stream_last <= 0;
    pass_done <= 1;
    @(posedge clk);
    pass_done <= 0;
    wait (!busy);
    @(posedge clk);
endtask

always_ff @(posedge clk)
    if (feature_wr_en) captured[feature_wr_addr] <= feature_wr_data;

initial begin
    $readmemh("wesad/sim_fixtures/ts_ecg.hex", ecg);
    $readmemh("wesad/sim_fixtures/ts_acc.hex", acc);
    $readmemh("wesad/sim_fixtures/ts_resp.hex", resp);
    $readmemh("wesad/sim_fixtures/ts_eda.hex", eda);
    $readmemh("wesad/sim_fixtures/ts_emg.hex", emg);
    $readmemh("wesad/sim_fixtures/ts_feat.hex", expected);
    lengths[0] = 15000;
    lengths[1] = 1920;
    lengths[2] = 1500;
    lengths[3] = 1500;
    lengths[4] = 21000;

    repeat (4) @(posedge clk);
    rst <= 0;
    @(posedge clk);
    wait (!busy);

    for (int s = 0; s < 5; s++) run_sensor(s, lengths[s]);

    for (int i = 0; i < 50; i++)
        if (captured[i] !== expected[i]) begin
            $display("FAIL feature %0d (sensor %0d index %0d) got %0d want %0d",
                     i, i / 10, i % 10, $signed(captured[i]), $signed(expected[i]));
            errors++;
        end

    if (errors == 0) $display("TIME STATS PASS");
    else $display("TIME STATS FAIL (%0d of 50)", errors);
    $finish;
end

endmodule
