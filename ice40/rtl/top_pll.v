// 36 MHz PLL top (corner T): 12 MHz crystal -> SB_PLL40_PAD -> 36 MHz
// fabric clock; 3 Mbaud = 36/12 exact. icepll -i 12 -o 36:
//   DIVR=0 DIVF=47 DIVQ=4 (VCO 576 MHz), FILTER_RANGE=1.
// Margin at build time: icetime 39.68 MHz conservative / nextpnr 39.75
// achieved vs the 36 constraint (1.10x) — the empirical gate is the
// full hw-replay sweep, per the track's hw-in-the-loop method.
module top (
    input  wire CLK,               // 12 MHz crystal (pad consumed by PLL)
    input  wire RX,
    output wire TX,
    output wire LEDR_N,
    output wire LEDG_N
);
    wire clk36;
    SB_PLL40_PAD #(
        .FEEDBACK_PATH("SIMPLE"),
        .DIVR(4'd0),
        .DIVF(7'd47),
        .DIVQ(3'd4),
        .FILTER_RANGE(3'd1)
    ) pll (
        .PACKAGEPIN(CLK),
        .PLLOUTGLOBAL(clk36),
        .RESETB(1'b1),
        .BYPASS(1'b0)
    );
    enc_top #(.UDIV(12)) core (.clk(clk36), .RX(RX), .TX(TX),
                               .LEDR_N(LEDR_N), .LEDG_N(LEDG_N));
endmodule
