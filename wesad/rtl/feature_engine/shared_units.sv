// Shared sequential units for the feature engines: restoring divider and fixed-point log2.

module divider (
    input logic clk, rst,
    input logic start,
    input logic [39 : 0] numerator,
    input logic [23 : 0] denominator,
    input logic [3 : 0] frac_bits,
    output logic [39 : 0] quotient,
    output logic done,
    output logic busy
);

logic [39 : 0] dividend;
logic [24 : 0] remainder;
logic [24 : 0] shifted;
logic [23 : 0] divisor;
logic [5 : 0] step;
logic denominator_zero;

assign shifted = {remainder[23 : 0], dividend[39]};

always_ff @(posedge clk) begin
    if (rst) begin
        step <= 0;
        busy <= 0;
        done <= 0;
        quotient <= 0;
    end
    else begin
        done <= 0;
        if (start && !busy) begin
            dividend <= numerator;
            divisor <= denominator;
            denominator_zero <= (denominator == 0);
            remainder <= 0;
            quotient <= 0;
            step <= 6'd40 + {2'b0, frac_bits};
            busy <= 1;
        end
        else if (busy) begin
            if (step != 0) begin
                if (denominator_zero) begin
                    quotient <= 0;
                    step <= 0;
                end
                else if (shifted >= {1'b0, divisor}) begin
                    remainder <= shifted - {1'b0, divisor};
                    quotient <= {quotient[38 : 0], 1'b1};
                    dividend <= {dividend[38 : 0], 1'b0};
                    step <= step - 1'b1;
                end
                else begin
                    remainder <= shifted;
                    quotient <= {quotient[38 : 0], 1'b0};
                    dividend <= {dividend[38 : 0], 1'b0};
                    step <= step - 1'b1;
                end
            end
            else begin
                busy <= 0;
                done <= 1;
            end
        end
    end
end

endmodule



module log2_unit (
    input logic [23 : 0] value,
    output logic [12 : 0] result
);

logic [4 : 0] exponent;
logic [4 : 0] mantissa;
logic [7 : 0] log2_frac;
logic [28 : 0] aligned;

always_comb begin
    exponent = 0;
    for (int i = 0; i < 24; i++)
        if (value[i]) exponent = i[4 : 0];

    aligned = {5'b0, value} << (5 - exponent[2 : 0]);
    mantissa = (exponent >= 5) ? value[(exponent - 5'd1) -: 5] : aligned[4 : 0];

    case (mantissa)
        5'd0 : log2_frac = 8'd0;
        5'd1 : log2_frac = 8'd11;
        5'd2 : log2_frac = 8'd22;
        5'd3 : log2_frac = 8'd33;
        5'd4 : log2_frac = 8'd44;
        5'd5 : log2_frac = 8'd54;
        5'd6 : log2_frac = 8'd63;
        5'd7 : log2_frac = 8'd73;
        5'd8 : log2_frac = 8'd82;
        5'd9 : log2_frac = 8'd92;
        5'd10 : log2_frac = 8'd100;
        5'd11 : log2_frac = 8'd109;
        5'd12 : log2_frac = 8'd118;
        5'd13 : log2_frac = 8'd126;
        5'd14 : log2_frac = 8'd134;
        5'd15 : log2_frac = 8'd142;
        5'd16 : log2_frac = 8'd150;
        5'd17 : log2_frac = 8'd157;
        5'd18 : log2_frac = 8'd165;
        5'd19 : log2_frac = 8'd172;
        5'd20 : log2_frac = 8'd179;
        5'd21 : log2_frac = 8'd186;
        5'd22 : log2_frac = 8'd193;
        5'd23 : log2_frac = 8'd200;
        5'd24 : log2_frac = 8'd207;
        5'd25 : log2_frac = 8'd213;
        5'd26 : log2_frac = 8'd220;
        5'd27 : log2_frac = 8'd226;
        5'd28 : log2_frac = 8'd232;
        5'd29 : log2_frac = 8'd238;
        5'd30 : log2_frac = 8'd244;
        5'd31 : log2_frac = 8'd250;
    endcase

    result = (value == 0) ? 13'd0 : {exponent, 8'd0} + {5'b0, log2_frac};
end

endmodule
