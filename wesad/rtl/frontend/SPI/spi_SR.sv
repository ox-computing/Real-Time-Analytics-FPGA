// Shift register that holds the 16 bit word coming from the serial SPI protocol,

module spi_SR (
    input logic clk,
    input logic rst,
    input logic data_in,
    input logic sample_en, // Made 1 for every time the SPI port sends over 1 bit
    output logic [15:0] data_out
);

logic [15:0] shift_register = '0;

assign data_out = shift_register;

always @(posedge clk)
    if (rst)
        shift_register <= '0;
    else if (sample_en)
        shift_register <= {shift_register[14:0], data_in}; // MSB first

endmodule
