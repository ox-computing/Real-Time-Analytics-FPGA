// Input buffers: 3 16K x 16 banks. The ECP5 has no SPRAM, so the storage is
// described behaviourally and inferred as block RAM instead of instantiated.

module input_buffer_spram (
    input logic clk,
    input logic rst,
    input logic [15:0] addr,
    input logic [15:0] data_in,
    input logic wren,
    input logic [3:0] mask,
    output logic [15:0] data_out
);

// 1 bit buffer select: 0 for buffer 0, 1 for buffer 1
logic [1:0] buf_select;
assign buf_select = addr[15:14];

// The bank output is registered, so addr input and output data are 1 cycle
// apart. This keeps buffer select for 1 cycle so that the banks' outputs are
// aligned with buf_select to multiplex 3:1 as data_out
logic [1:0] buf_select_q;
always @(posedge clk) buf_select_q <= rst ? 2'b0 : buf_select;

logic [15:0] bank0 [0:16383];
logic [15:0] bank1 [0:16383];
logic [15:0] bank2 [0:16383];
logic [15:0] dout0, dout1, dout2;
logic [13:0] word_addr;

assign word_addr = addr[13:0];

// the nibble guards stand in for MASKWREN; the arbiter drives mask all ones,
// so they fold away rather than costing per-nibble write ports
always @(posedge clk) begin
    if (wren && buf_select == 2'd0) begin
        if (mask[0]) bank0[word_addr][3:0] <= data_in[3:0];
        if (mask[1]) bank0[word_addr][7:4] <= data_in[7:4];
        if (mask[2]) bank0[word_addr][11:8] <= data_in[11:8];
        if (mask[3]) bank0[word_addr][15:12] <= data_in[15:12];
    end
    dout0 <= bank0[word_addr];
end

always @(posedge clk) begin
    if (wren && buf_select == 2'd1) begin
        if (mask[0]) bank1[word_addr][3:0] <= data_in[3:0];
        if (mask[1]) bank1[word_addr][7:4] <= data_in[7:4];
        if (mask[2]) bank1[word_addr][11:8] <= data_in[11:8];
        if (mask[3]) bank1[word_addr][15:12] <= data_in[15:12];
    end
    dout1 <= bank1[word_addr];
end

always @(posedge clk) begin
    if (wren && buf_select == 2'd2) begin
        if (mask[0]) bank2[word_addr][3:0] <= data_in[3:0];
        if (mask[1]) bank2[word_addr][7:4] <= data_in[7:4];
        if (mask[2]) bank2[word_addr][11:8] <= data_in[11:8];
        if (mask[3]) bank2[word_addr][15:12] <= data_in[15:12];
    end
    dout2 <= bank2[word_addr];
end

assign data_out = (buf_select_q == 2'd0) ? dout0 :
                  (buf_select_q == 2'd1) ? dout1 : dout2;

endmodule
