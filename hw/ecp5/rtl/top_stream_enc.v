// top_stream_enc.v — STREAM ingest + ON-CHIP VSA ENCODER + on-chip map
// (STREAM.md v1.1). Single-ring lidar frames are encoded on the device
// (bit-exact to hw/ice40/golden.encode_int); the full int32 vector
// streams back (0x92, golden crosscheck on the laptop) and its 2-bit
// QPSK codes land in the on-chip 64-segment map bank, fetchable by
// MAP_READ (0x0F -> 0x93) — the compressed representation the laptop
// decodes. ROMs: make roms (host/gen_enc_roms.py).
//
//   make roms
//   make TOP=top_stream_enc RTL="rtl/stream_ingest.v rtl/enc_feeder.v \
//        ../ice40/rtl/encoder.v ../ice40/rtl/uart.v rtl/top_stream_enc.v" \
//        build prog
`default_nettype none

module top (
    input  wire       clk,        // 50 MHz
    input  wire       usb_rx,
    output wire       usb_tx,
    output wire [4:0] led
);
    localparam DIV = 25;          // 2 Mbaud

    wire [7:0] rxd, txd;
    wire       rxv, tsend, tbusy;
    wire [3:0] sled;

    uart_rx #(.DIV(DIV)) urx (.clk(clk), .rx(usb_rx), .data(rxd),
                              .valid(rxv));
    uart_tx #(.DIV(DIV)) utx (.clk(clk), .data(txd), .send(tsend),
                              .tx(usb_tx), .busy(tbusy));

    wire [7:0]  pl_type, pl_byte;
    wire [15:0] pl_idx;
    wire        pl_valid, pkt_commit, pkt_abort, lid_done;
    wire [31:0] lid_fid;
    wire        vec_req, vec_ack;
    wire [31:0] vec_fid;
    wire [7:0]  vec_rd;
    wire signed [31:0] vec_re, vec_im;
    wire        enc_clear, enc_start, enc_busy;
    wire [9:0]  enc_az;
    wire [15:0] enc_r;
    wire [7:0]  enc_w;

    stream_ingest ing (.clk(clk), .rx_data(rxd), .rx_valid(rxv),
                       .tx_data(txd), .tx_send(tsend), .tx_busy(tbusy),
                       .led(sled),
                       .pl_type(pl_type), .pl_idx(pl_idx),
                       .pl_byte(pl_byte), .pl_valid(pl_valid),
                       .pkt_commit(pkt_commit), .pkt_abort(pkt_abort),
                       .lid_done(lid_done), .lid_fid(lid_fid),
                       .vec_req(vec_req), .vec_fid(vec_fid),
                       .vec_ack(vec_ack), .vec_rd(vec_rd),
                       .vec_re(vec_re), .vec_im(vec_im));

    enc_feeder fdr (.clk(clk), .pl_type(pl_type), .pl_idx(pl_idx),
                    .pl_byte(pl_byte), .pl_valid(pl_valid),
                    .pkt_commit(pkt_commit), .lid_done(lid_done),
                    .lid_fid(lid_fid),
                    .enc_clear(enc_clear), .enc_start(enc_start),
                    .enc_az(enc_az), .enc_r(enc_r), .enc_w(enc_w),
                    .enc_busy(enc_busy),
                    .vec_req(vec_req), .vec_fid(vec_fid),
                    .vec_ack(vec_ack));

    encoder enc (.clk(clk), .clear(enc_clear), .start(enc_start),
                 .az(enc_az), .r_mm(enc_r), .w(enc_w), .busy(enc_busy),
                 .rd_idx(vec_rd), .rd_re(vec_re), .rd_im(vec_im));

    reg [24:0] hb = 0;
    always @(posedge clk) hb <= hb + 1;
    assign led = {hb[24], sled[2:0], enc_busy};
endmodule
