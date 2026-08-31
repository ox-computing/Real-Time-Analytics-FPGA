// Input buffers: 2 16K x 16 SPRAMs.

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

// Output of SPRAM is registered, so addr input and output data are 1 cycle apart.
// This keeps buffer select for 1 cycle so that SPRAMs' outputs are aligned with
// buf_select to multiplex 2:1 as data_out
logic [1:0] buf_select_q;
always @(posedge clk) buf_select_q <= rst ? 2'b0 : buf_select;

logic [15:0] dout0, dout1, dout2;

    SB_SPRAM256KA spram_16k_0 (
        .ADDRESS (addr[13:0]),
        .DATAIN (data_in),
        .MASKWREN (mask),
        .WREN (wren & (buf_select == 2'd0)),
        .CHIPSELECT(buf_select == 2'd0),
        .CLOCK (clk),
        .STANDBY (1'b0),
        .SLEEP (1'b0),
        .POWEROFF (1'b1),
        .DATAOUT (dout0)
    );

    SB_SPRAM256KA spram_16k_1 (
        .ADDRESS (addr[13:0]),
        .DATAIN (data_in),
        .MASKWREN (mask),
        .WREN (wren & (buf_select == 2'd1)),
        .CHIPSELECT(buf_select == 2'd1),
        .CLOCK (clk),
        .STANDBY (1'b0),
        .SLEEP (1'b0),
        .POWEROFF (1'b1),
        .DATAOUT (dout1)
    );

    SB_SPRAM256KA spram_16k_2 (
        .ADDRESS (addr[13:0]),
        .DATAIN (data_in),
        .MASKWREN (mask),
        .WREN (wren & (buf_select == 2'd2)),
        .CHIPSELECT(buf_select == 2'd2),
        .CLOCK (clk),
        .STANDBY (1'b0),
        .SLEEP (1'b0),
        .POWEROFF (1'b1),
        .DATAOUT (dout2)
    );

assign data_out = (buf_select_q == 2'd0) ? dout0 :
                  (buf_select_q == 2'd1) ? dout1 : dout2;

endmodule
