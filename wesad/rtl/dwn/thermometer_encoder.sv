module thermometer_encoder #(parameter N_FEATURE = 115, 
                            parameter Z = 4, 
                            parameter FEATURE_WIDTH = 16,
                            parameter THRESHOLD_WIDTH = 16
                            )(
    input logic clk, rst,

    input logic frame_done, // From frame seq, 1 for 1 clk cycle

    // Feature 
    input logic [FEATURE_WIDTH - 1 : 0] feature_data, // 16 bit features
    output logic [$clog2(N_FEATURE) - 1: 0] feature_addr, // log2(115) = 7 bit address space
    
    // Comparison thresholds
    input logic [THRESHOLD_WIDTH -1 : 0] threshold_data, // 16 bit thresholds
    output logic [$clog2(Z) - 1: 0] threshold_addr, // 3 bit address space. 0-3
    
    output logic [Z*N_FEATURE - 1 : 0] binary_vector, // 4 * 115 = 460
    output logic encode_done

);

logic [$clog2(Z*N_FEATURE) - 1 : 0] count, count_d;
logic [Z*N_FEATURE - 1: 0] binary_vector_storage; // 4 X 115 DWN input array
logic binary_encoding;
logic busy;

assign feature_addr = count[$clog2(Z*N_FEATURE) - 1 : $clog2(Z)];
assign threshold_addr = count[$clog2(Z) - 1 : 0];

// 1 counter: LSB is 0,1,2,3,0, ..., MSB is 0,1,2,3,...,122
always_ff @(posedge clk) begin
    encode_done <= 0;
    if (rst | frame_done) begin
        count <= 0;
        count_d <= 0;
        busy <= frame_done;
    end else if (busy) begin
        count <= count + 1;
        count_d <= count; // Counter is 1 cycle ahead of data, so register it for 1 cycle.
        if (count != 0) binary_vector_storage <= {binary_encoding, binary_vector_storage[Z*N_FEATURE - 1 : 1]};
        if (count_d == Z*N_FEATURE - 1) begin
            busy <= 0;
            encode_done <= 1;
        end
    end
end

assign binary_encoding = $signed(feature_data) > $signed(threshold_data);
assign binary_vector = binary_vector_storage;


endmodule