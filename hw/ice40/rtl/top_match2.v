// 24 MHz PLL top for the v5 encoder+matcher+batch build. The v5 fixes
// moved the limiter from the accumulate path (35.75 ns in v4) to the
// SPRAM-out -> negate -> rot90-mux stage-B cone (~30.6 MHz placed) —
// 36 MHz needs the stage-B split or the mc-prefetch cis-address fold
// (filed). 24 MHz here = 1.27x margin; 3 Mbaud = 24/8 exact.
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
    enc_top_m2 #(.UDIV(8)) core (.clk(clk24), .RX(RX), .TX(TX),
                                 .LEDR_N(LEDR_N), .LEDG_N(LEDG_N));
endmodule
