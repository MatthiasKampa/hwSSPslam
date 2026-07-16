// tb_enc2.v — repro of the silicon two-packet divergence: n_az=32, two
// 16-col packets; dumps every (az, r) the feeder hands the encoder.
`timescale 1ns/1ps
`default_nettype none
module tb;
    reg clk = 0;
    always #10 clk = ~clk;
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
                       .led(sled), .pl_type(pl_type), .pl_idx(pl_idx),
                       .pl_byte(pl_byte), .pl_valid(pl_valid),
                       .pkt_commit(pkt_commit), .pkt_abort(pkt_abort),
                       .lid_done(lid_done), .lid_fid(lid_fid),
                       .vec_req(vec_req), .vec_fid(vec_fid),
                       .vec_ack(vec_ack), .vec_rd(vec_rd),
                       .vec_re(vec_re), .vec_im(vec_im));
    enc_feeder fdr (.clk(clk), .pl_type(pl_type), .pl_idx(pl_idx),
                    .pl_byte(pl_byte), .pl_valid(pl_valid),
                    .pkt_commit(pkt_commit), .lid_done(lid_done),
                    .lid_fid(lid_fid), .enc_clear(enc_clear),
                    .enc_start(enc_start), .enc_az(enc_az), .enc_r(enc_r),
                    .enc_w(enc_w), .enc_busy(enc_busy),
                    .vec_req(vec_req), .vec_fid(vec_fid),
                    .vec_ack(vec_ack));
    encoder enc (.clk(clk), .clear(enc_clear), .start(enc_start),
                 .az(enc_az), .r_mm(enc_r), .w(enc_w), .busy(enc_busy),
                 .rd_idx(vec_rd), .rd_re(vec_re), .rd_im(vec_im));


    always @(posedge clk) if (pkt_commit)
        $display("t=%0t COMMIT type=%02x fst=%0d nr_ok=%b pend_clear=%b enc_busy=%b",
                 $time, pl_type, fdr.fst, fdr.nr_ok, fdr.pend_clear, enc_busy);
    always @(posedge clk) if (enc_start)
        $display("PT az=%0d r=%0d", enc_az, enc_r);

    function [15:0] crc16_up(input [15:0] c, input [7:0] d);
        integer i; reg [15:0] x;
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
            send_byte(typ);       pcrc = crc16_up(pcrc, typ);
            send_byte(8'h00);     pcrc = crc16_up(pcrc, 8'h00);
            send_byte(len[7:0]);  pcrc = crc16_up(pcrc, len[7:0]);
            send_byte(len[15:8]); pcrc = crc16_up(pcrc, len[15:8]);
            send_byte(seq[7:0]);  pcrc = crc16_up(pcrc, seq[7:0]);
            send_byte(seq[15:8]); pcrc = crc16_up(pcrc, seq[15:8]);
        end
    endtask
    task put_p(input [7:0] b);
        begin send_byte(b); pcrc = crc16_up(pcrc, b); end
    endtask
    task pkt_end;
        begin send_byte(pcrc[7:0]); send_byte(pcrc[15:8]); end
    endtask

    integer j;
    reg [15:0] rr;
    initial begin
        #2000;
        pkt_begin(8'h01, 16'd17, 16'd0);
        put_p(8'd7); put_p(0); put_p(0); put_p(0);
        for (j = 0; j < 8; j = j + 1) put_p(8'h00);
        put_p(8'd1); put_p(8'd33); put_p(8'd32); put_p(8'd0); put_p(8'd1);
        pkt_end;
        pkt_begin(8'h02, 16'd7 + 16'd32, 16'd1);      // az0=0, 16 cols
        put_p(8'd7); put_p(0); put_p(0); put_p(0);
        put_p(8'd0); put_p(8'd0); put_p(8'd16);
        for (j = 0; j < 16; j = j + 1) begin
            rr = 2000 + 11 * j;
            put_p(rr[7:0]); put_p(rr[15:8]);
        end
        pkt_end;
        pkt_begin(8'h02, 16'd7 + 16'd32, 16'd2);      // az0=16, 16 cols
        put_p(8'd7); put_p(0); put_p(0); put_p(0);
        put_p(8'd16); put_p(8'd0); put_p(8'd16);
        for (j = 0; j < 16; j = j + 1) begin
            rr = 3000 + 7 * j;
            put_p(rr[7:0]); put_p(rr[15:8]);
        end
        pkt_end;
        #3_000_000;
        $display("done (expected az0-15 r=2000+11i, az16-31 r=3000+7i)");
        $finish;
    end
endmodule
// probe appended
module probe_hack; endmodule
