// top_solo_ecp5_v2.v — the v8 "maxed" STANDALONE SLAM top on the Icepi
// Zero (ECP5-25F). Same board contract as top_solo_ecp5.v (50 MHz osc,
// /2 clock domain, 1 Mbaud on usb_rx/usb_tx), swapping in the v8 stack:
//   top_solo2.v      novelty-gated folding (map budget buys AREA)
//   solo_tracker2.v  +-9 deg fine rho window, rho edge retry (x2),
//                    staged/global lost re-search, novel output
// Golden: hw/ecp5/host/solo2.py; bench: hunter crispness 9.29 -> 10.45+
// pts/cell, spot no-regression (ATE p90 4.48 -> 2.37).
//
//   make roms-solo
//   make TOP=top_solo_ecp5_v2 RTL="rtl/top_solo_ecp5_v2.v \
//        rtl/sb_spram_compat.v rtl/top_solo2.v rtl/solo_tracker2.v \
//        ../ice40/rtl/encoder_lean.v ../ice40/rtl/uart.v" build timing
`default_nettype none

module top (
    input  wire       clk,        // 50 MHz board osc
    input  wire       usb_rx,
    output wire       usb_tx,
    output wire [4:0] led
);
    reg clk2 = 1'b0;
    always @(posedge clk) clk2 <= ~clk2;

    wire ledr_n, ledg_n;

    top_solo2 #(
        .UDIV(25),                // 25 MHz / 25 = 1 Mbaud
        .NSEG(64),
        .SEG_KF(5),
        .NOV_D2(670761),          // 0.8 m novelty radius
        .REFINE(0)                // lean build (no divider)
    ) solo (
        .clk(clk2),
        .RX(usb_rx),
        .TX(usb_tx),
        .LEDR_N(ledr_n),
        .LEDG_N(ledg_n)
    );

    assign led = {3'b0, ~ledg_n, ~ledr_n};
endmodule
