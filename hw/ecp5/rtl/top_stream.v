// top_stream.v — virtual-sensor USB streaming ingest top (STREAM.md v0).
// Robot PC streams lidar + camera packets over the FT231X UART @ 2 Mbaud;
// stream_ingest parses/digests; ECHO_DIGEST/STATUS return on the same UART.
// Gate: python3 host/hw_stream.py gate   (G1 — silicon loopback, digests
// bit-exact, zero drops).
//
//   make TOP=top_stream RTL="rtl/stream_ingest.v ../ice40/rtl/uart.v \
//        rtl/top_stream.v" build prog
`default_nettype none

module top (
    input  wire       clk,        // 50 MHz
    input  wire       usb_rx,
    output wire       usb_tx,
    output wire [4:0] led
);
    localparam DIV = 25;          // 2 Mbaud exact. (2.4M tried 2026-07-16:
                                  // the macOS FTDI VCP silently kept ~2M ->
                                  // +19% mismatch, zero packets; Linux
                                  // ftdi_sio may honor it — retest on the
                                  // robot box before bumping DIV to 21.)

    wire [7:0] rxd, txd;
    wire       rxv, tsend, tbusy;
    wire [3:0] sled;

    uart_rx #(.DIV(DIV)) urx (.clk(clk), .rx(usb_rx), .data(rxd),
                              .valid(rxv));
    uart_tx #(.DIV(DIV)) utx (.clk(clk), .data(txd), .send(tsend),
                              .tx(usb_tx), .busy(tbusy));

    stream_ingest ing (.clk(clk), .rx_data(rxd), .rx_valid(rxv),
                       .tx_data(txd), .tx_send(tsend), .tx_busy(tbusy),
                       .led(sled));

    // heartbeat on led[4] so a flashed-but-idle board is visible
    reg [24:0] hb = 0;
    always @(posedge clk) hb <= hb + 1;
    assign led = {hb[24], sled};
endmodule
