// PLL top for the v6 build (cis-address fold + argmax batches).
// The v5 limiter (SPRAM->negate->mux->DSP stage-B cone) is dissolved
// by the fold — first attempt at 36 MHz; this file currently carries
// the 24 MHz settings and is switched by the build step if 36 closes.
module top (
    input  wire CLK,               // 12 MHz crystal (pad consumed by PLL)
    input  wire RX,
    output wire TX,
    output wire LEDR_N,
    output wire LEDG_N
);
    wire clk24;
    SB_PLL40_PAD #(
        .FEEDBACK_PATH("SIMPLE"),
        .DIVR(4'd0),
        .DIVF(7'd63),
        .DIVQ(3'd5),
        .FILTER_RANGE(3'd1)
    ) pll (
        .PACKAGEPIN(CLK),
        .PLLOUTGLOBAL(clk24),
        .RESETB(1'b1),
        .BYPASS(1'b0)
    );
    enc_top_m3 #(.UDIV(8)) core (.clk(clk24), .RX(RX), .TX(TX),
                                 .LEDR_N(LEDR_N), .LEDG_N(LEDG_N));
endmodule
