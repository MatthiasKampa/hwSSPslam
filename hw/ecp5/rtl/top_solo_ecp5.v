// top_solo_ecp5.v — the ice40 v7 STANDALONE SLAM top on the Icepi Zero
// (ECP5-25F). AUTONOMY deploy ("dumb cable" contract): points + odom
// deltas in (0x30 keyframes), pose frames out (0xB0), map dump (0x28).
// The chip runs the FULL loop alone — encode -> track -> fold -> freeze
// -> resident map; the Linux box is vis + data transport only.
//
// Rule-1 port: top_solo.v / solo_tracker.v / encoder_lean.v build
// VERBATIM from hw/ice40/rtl (golden = ice40/host/solo.py, sim gates
// 220/220 bit-exact); the only substitutions are
//   - SB_SPRAM256KA -> sb_spram_compat.v (inferred EBR, same contract)
//   - clock/pins/baud: 50 MHz osc, UDIV=25 -> 2 Mbaud on the same
//     usb_rx/usb_tx pins as top_stream_enc (icepi-zero.lpf)
//
//   make roms-solo    # az_t/ang_c/ang_s/cis/cs_t.hex into build/
//   make TOP=top_solo_ecp5 RTL="rtl/top_solo_ecp5.v \
//        rtl/sb_spram_compat.v ../ice40/rtl/top_solo.v \
//        ../ice40/rtl/solo_tracker.v ../ice40/rtl/encoder_lean.v \
//        ../ice40/rtl/uart.v" build timing
`default_nettype none

module top (
    input  wire       clk,        // 50 MHz board osc
    input  wire       usb_rx,
    output wire       usb_tx,
    output wire [4:0] led
);
    // /2 domain: the solo core places at ~44 MHz on the 25F -> run it
    // at 25 MHz (10 Hz step budget ~46 ms, fits) with UDIV=25 ->
    // 1 Mbaud (keyframes 46 KB/s at 10 Hz = 2x headroom).
    reg clk2 = 1'b0;
    always @(posedge clk) clk2 <= ~clk2;

    wire ledr_n, ledg_n;

    top_solo #(
        .UDIV(25),                // 25 MHz / 25 = 1 Mbaud
        .NSEG(64),
        .SEG_KF(5),
        .REFINE(0)                // the lean/banked build
    ) solo (
        .clk(clk2),
        .RX(usb_rx),
        .TX(usb_tx),
        .LEDR_N(ledr_n),
        .LEDG_N(ledg_n)
    );

    assign led = {3'b0, ~ledg_n, ~ledr_n};
endmodule
