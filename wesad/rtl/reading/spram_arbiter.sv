module spram_arbiter (
    input logic wr_en,
    input logic [15:0] wr_addr,
    input logic [15:0] wr_data,
    input logic head_request,
    input logic [15:0] head_addr,
    input logic read_request,
    input logic [15:0] read_addr,
    output logic [15:0] spram_addr,
    output logic [15:0] spram_data,
    output logic spram_wren,
    output logic [3:0] spram_mask,
    output logic grant
);

assign spram_addr = wr_en ? wr_addr :
                    head_request ? head_addr :
                    read_request ? read_addr : '0;

assign spram_data = wr_data;
assign spram_wren = wr_en;
assign spram_mask = '1;
assign grant = read_request & ~wr_en & ~head_request;

endmodule
