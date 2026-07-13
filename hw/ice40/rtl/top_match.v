// 24 MHz PLL top for the encoder+matcher build (corner F): the match
// datapath places at ~30.4 MHz (icetime/nextpnr; limiters: exec-FSM
// decode + score accumulate mux — follow-up filed), so this build runs
// 24 MHz with 1.27x margin. 3 Mbaud = 24/8 exact. The encoder-only
// build (top_pll.v) keeps the 36 MHz corner. icepll -i 12 -o 24:
//   DIVR=0 DIVF=63 DIVQ=5 (VCO 768 MHz), FILTER_RANGE=1.
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
    enc_top_m #(.UDIV(8)) core (.clk(clk24), .RX(RX), .TX(TX),
                                .LEDR_N(LEDR_N), .LEDG_N(LEDG_N));
endmodule
