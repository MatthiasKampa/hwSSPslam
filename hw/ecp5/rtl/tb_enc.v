// tb_enc.v — sim gate for the stream+encode+map chain (STREAM.md v1.1).
// Sends LIDAR_HDR (n_rings=1) + one LIDAR_COL (8 columns incl. a miss and
// an out-of-mask range) through the FULL packet path; expects the ECHO,
// then the 0x92 VEC (240 x int32 pairs == golden encode_int, from
// build/tb_enc_exp.hex), then requests MAP_READ and checks the 60-byte
// 2-bit QPSK codes (build/tb_enc_mcode.hex).
//   make sim-enc
`timescale 1ns/1ps
`default_nettype none

module tb;
    reg clk = 0;
    always #10 clk = ~clk;                       // 50 MHz

    localparam DIV = 25;
    reg  rx = 1'b1;
    wire tx;
    wire [7:0] rxd, txd;
    wire       rxv, tsend, tbusy;
    wire [3:0] sled;
    uart_rx #(.DIV(DIV)) urx (.clk(clk), .rx(rx), .data(rxd), .valid(rxv));
    uart_tx #(.DIV(DIV)) utx (.clk(clk), .data(txd), .send(tsend),
                              .tx(tx), .busy(tbusy));

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

    // TX monitor
    wire [7:0] mon_d;
    wire       mon_v;
    uart_rx #(.DIV(DIV)) mon (.clk(clk), .rx(tx), .data(mon_d),
                              .valid(mon_v));
    reg [7:0] rxbuf [0:4095];
    integer   rxn = 0;
    always @(posedge clk) if (mon_v) begin
        rxbuf[rxn] = mon_d; rxn = rxn + 1;
    end

    // ---- helpers ----
    function [15:0] crc16_up(input [15:0] c, input [7:0] d);
        integer i;
        reg [15:0] x;
        begin
            x = c ^ {d, 8'h00};
            for (i = 0; i < 8; i = i + 1)
                x = x[15] ? ((x << 1) ^ 16'h1021) : (x << 1);
            crc16_up = x;
        end
    endfunction
    task send_byte(input [7:0] b);
        integer i;
        begin
            rx = 0; #(DIV * 20);
            for (i = 0; i < 8; i = i + 1) begin rx = b[i]; #(DIV * 20); end
            rx = 1; #(DIV * 20 * 2);
        end
    endtask
    reg [15:0] pcrc;
    task pkt_begin(input [7:0] typ, input [15:0] len, input [15:0] seq);
        begin
            send_byte(8'hA5); send_byte(8'h5A);
            pcrc = 16'hFFFF;
            send_byte(typ);        pcrc = crc16_up(pcrc, typ);
            send_byte(8'h00);      pcrc = crc16_up(pcrc, 8'h00);
            send_byte(len[7:0]);   pcrc = crc16_up(pcrc, len[7:0]);
            send_byte(len[15:8]);  pcrc = crc16_up(pcrc, len[15:8]);
            send_byte(seq[7:0]);   pcrc = crc16_up(pcrc, seq[7:0]);
            send_byte(seq[15:8]);  pcrc = crc16_up(pcrc, seq[15:8]);
        end
    endtask
    task put_p(input [7:0] b);
        begin send_byte(b); pcrc = crc16_up(pcrc, b); end
    endtask
    task pkt_end;
        begin send_byte(pcrc[7:0]); send_byte(pcrc[15:8]); end
    endtask

    reg [15:0] scan [0:7];
    reg [31:0] expv [0:479];
    reg [7:0]  expm [0:59];
    integer j, vi, mi_, errs;

    initial begin
        #60_000_000;                       // 60 ms watchdog
        $display("FAIL: watchdog (rxn=%0d; vec never completed?)", rxn);
        $display("  probe: fdr.fst=%0d fdr.nr_ok=%b fdr.pend_done=%b "
                 , fdr.fst, fdr.nr_ok, fdr.pend_done);
        $display("  probe: vec_req=%b ing.tst=%0d enc_busy=%b",
                 vec_req, ing.tst, enc_busy);
        $finish;
    end

    // event probes (debug visibility; cheap)
    always @(posedge clk) begin
        if (pkt_commit) $display("t=%0t commit type=%02x", $time, pl_type);
        if (lid_done)   $display("t=%0t lid_done fid=%0d", $time, lid_fid);
        if (vec_req && !vec_ack && ing.tst == 0)
            ;                              // (quiet; state visible on WD)
        if (vec_ack)    $display("t=%0t vec_ack (TX starts)", $time);
    end

    initial begin
        $readmemh("build/tb_enc_scan.hex", scan);
        $readmemh("build/tb_enc_exp.hex", expv);
        $readmemh("build/tb_enc_mcode.hex", expm);
        #2000;

        // LIDAR_HDR fid=5, n_rings=1, ring 33, n_az=8, fmt=1
        pkt_begin(8'h01, 16'd17, 16'd0);
        put_p(8'd5); put_p(8'd0); put_p(8'd0); put_p(8'd0);   // fid
        for (j = 0; j < 8; j = j + 1) put_p(8'h00);           // t_us
        put_p(8'd1);                                          // n_rings
        put_p(8'd33);                                         // ring_ids
        put_p(8'd8); put_p(8'd0);                             // n_az
        put_p(8'd1);                                          // fmt
        pkt_end;

        // LIDAR_COL az0=100, ncols=8
        pkt_begin(8'h02, 16'd7 + 16'd16, 16'd1);
        put_p(8'd5); put_p(8'd0); put_p(8'd0); put_p(8'd0);
        put_p(8'd100); put_p(8'd0); put_p(8'd8);
        for (j = 0; j < 8; j = j + 1) begin
            put_p(scan[j][7:0]); put_p(scan[j][15:8]);
        end
        pkt_end;

        // wait: echo (21 B) + vec (1934 B) at 2 Mbaud ~ 10 ms
        wait (rxn >= 21 + 1934);
        #200_000;

        // parse: find 0x92 packet
        vi = -1;
        for (j = 0; j < rxn - 8; j = j + 1)
            if (rxbuf[j] == 8'hA5 && rxbuf[j+1] == 8'h5A &&
                rxbuf[j+2] == 8'h92) vi = j;
        if (vi < 0) begin $display("FAIL: no VEC packet"); $finish; end
        if (rxbuf[vi+8] !== 8'd5) begin
            $display("FAIL: vec fid %0d != 5", rxbuf[vi+8]); $finish;
        end
        errs = 0;
        for (j = 0; j < 480; j = j + 1) begin
            if ({rxbuf[vi+12+4*j+3], rxbuf[vi+12+4*j+2],
                 rxbuf[vi+12+4*j+1], rxbuf[vi+12+4*j]} !== expv[j])
                errs = errs + 1;
        end
        if (errs != 0) begin
            $display("FAIL: %0d/480 vector words mismatch golden", errs);
            $display("  first: got %02x%02x%02x%02x exp %08x",
                     rxbuf[vi+15], rxbuf[vi+14], rxbuf[vi+13], rxbuf[vi+12],
                     expv[0]);
            $finish;
        end

        // MAP_READ seg 5
        pkt_begin(8'h0F, 16'd1, 16'd2);
        put_p(8'd5);
        pkt_end;
        wait (rxn >= 21 + 1934 + 71);
        #200_000;
        mi_ = -1;
        for (j = 0; j < rxn - 8; j = j + 1)
            if (rxbuf[j] == 8'hA5 && rxbuf[j+1] == 8'h5A &&
                rxbuf[j+2] == 8'h93) mi_ = j;
        if (mi_ < 0) begin $display("FAIL: no MAP_SEG packet"); $finish; end
        if (rxbuf[mi_+8] !== 8'd5) begin
            $display("FAIL: map seg %0d != 5", rxbuf[mi_+8]); $finish;
        end
        errs = 0;
        for (j = 0; j < 60; j = j + 1)
            if (rxbuf[mi_+9+j] !== expm[j]) errs = errs + 1;
        if (errs != 0) begin
            $display("FAIL: %0d/60 mcode bytes mismatch (got %02x exp %02x)",
                     errs, rxbuf[mi_+9], expm[0]);
            $finish;
        end
        $display("PASS: on-chip encode == golden (480/480 words), on-chip");
        $display("      map codes == golden mcodes (60/60 bytes), fid ok");
        $finish;
    end
endmodule
