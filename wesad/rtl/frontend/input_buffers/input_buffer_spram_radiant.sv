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

    SP256K spram_16k_0 (
        .AD (addr[13:0]),
        .DI (data_in),
        .MASKWE (mask),
        .WE (wren & (buf_select == 2'd0)),
        .CS(buf_select == 2'd0),
        .CK (clk),
        .STDBY (1'b0),
        .SLEEP (1'b0),
        .PWROFF_N (1'b1),
        .DO (dout0)
    );

    SP256K spram_16k_1 (
        .AD (addr[13:0]),
        .DI (data_in),
        .MASKWE (mask),
        .WE (wren & (buf_select == 2'd1)),
        .CS(buf_select == 2'd1),
        .CK (clk),
        .STDBY (1'b0),
        .SLEEP (1'b0),
        .PWROFF_N (1'b1),
        .DO (dout1)
    );

    SP256K spram_16k_2 (
        .AD (addr[13:0]),
        .DI (data_in),
        .MASKWE (mask),
        .WE (wren & (buf_select == 2'd2)),
        .CS(buf_select == 2'd2),
        .CK (clk),
        .STDBY (1'b0),
        .SLEEP (1'b0),
        .PWROFF_N (1'b1),
        .DO (dout2)
    );

assign data_out = (buf_select_q == 2'd0) ? dout0 :
                  (buf_select_q == 2'd1) ? dout1 : dout2;

endmodule
