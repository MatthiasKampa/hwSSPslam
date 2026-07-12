// PLL top for the v7 SOLO build (lean core + tracker + top_solo).
// 12 MHz crystal -> 24 MHz fabric clock; UART at 3 Mbaud (UDIV=8).
// Corner configs (build session 2026-07-12):
//   S/F: REFINE=0 (divider + tot RAM dropped; classroom A/B banked —
//        localization p90 +2 mm, selfmap actually better)
//   T:   the v6 deploy build (top_pll.v + enc_top_m3 + encoder_match3)
//        remains the throughput corner: 488 cyc/cand parallel-ring core.
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
    top_solo #(.UDIV(8), .REFINE(0)) core (.clk(clk24), .RX(RX), .TX(TX),
                                           .LEDR_N(LEDR_N),
                                           .LEDG_N(LEDG_N));
endmodule
