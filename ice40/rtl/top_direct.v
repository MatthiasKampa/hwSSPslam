// 12 MHz direct-clock top (no PLL): enc_top at 3 Mbaud (12/4).
module top (
    input  wire CLK,               // 12 MHz
    input  wire RX,
    output wire TX,
    output wire LEDR_N,
    output wire LEDG_N
);
    enc_top #(.UDIV(4)) core (.clk(CLK), .RX(RX), .TX(TX),
                              .LEDR_N(LEDR_N), .LEDG_N(LEDG_N));
endmodule
