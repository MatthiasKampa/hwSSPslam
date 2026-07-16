// tb_stream.v — G0 sim gate for stream_ingest (STREAM.md v0).
// Exercises: magic resync on garbage, cam frame (2 row packets), lidar
// frame (1 col packet), a CORRUPT mid-frame packet (CRC fail -> drop +
// digest ROLLBACK), clean resend completing the frame with the clean
// digest, and STATUS readback (crc_drops==1, seq_gaps==1, frames 2/1).
//   make sim-stream
`timescale 1ns/1ps
`default_nettype none

module tb;
    reg clk = 0;
    always #10 clk = ~clk;                     // 50 MHz

    localparam DIV = 25;                       // 2 Mbaud
    reg  rx = 1'b1;
    wire tx;

    wire [7:0] rxd, txd;
    wire       rxv, tsend, tbusy;
    wire [3:0] led;
    uart_rx #(.DIV(DIV)) urx (.clk(clk), .rx(rx), .data(rxd), .valid(rxv));
    uart_tx #(.DIV(DIV)) utx (.clk(clk), .data(txd), .send(tsend),
                              .tx(tx), .busy(tbusy));
    stream_ingest dut (.clk(clk), .rx_data(rxd), .rx_valid(rxv),
                       .tx_data(txd), .tx_send(tsend), .tx_busy(tbusy),
                       .led(led));

    // monitor: second uart_rx on the DUT's tx line
    wire [7:0] mon_d;
    wire       mon_v;
    uart_rx #(.DIV(DIV)) mon (.clk(clk), .rx(tx), .data(mon_d),
                              .valid(mon_v));
    reg [7:0] rxbuf [0:255];
    integer   rxn = 0;
    always @(posedge clk) if (mon_v) begin
        rxbuf[rxn] = mon_d; rxn = rxn + 1;
    end

    // ---- helpers -------------------------------------------------------
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
            rx = 0; #(DIV * 20);                       // start
            for (i = 0; i < 8; i = i + 1) begin
                rx = b[i]; #(DIV * 20);
            end
            rx = 1; #(DIV * 20 * 2);                   // stop + gap
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
    task pkt_end(input corrupt);
        begin
            send_byte(corrupt ? (pcrc[7:0] ^ 8'hFF) : pcrc[7:0]);
            send_byte(pcrc[15:8]);
        end
    endtask
    task put_u32(input [31:0] v);
        begin put_p(v[7:0]); put_p(v[15:8]); put_p(v[23:16]);
              put_p(v[31:24]); end
    endtask
    task put_u64z;                                // zero timestamp
        integer i;
        begin for (i = 0; i < 8; i = i + 1) put_p(8'h00); end
    endtask

    // reference digests
    reg [15:0] ref_cam7, ref_cam8, ref_lid9;
    integer j;
    reg [7:0] px;

    task cam_row(input [31:0] id, input [15:0] row0, input [7:0] nrows,
                 input [15:0] w, input [15:0] seq, input corrupt,
                 input track_ref, input [1:0] which);
        begin
            pkt_begin(8'h04, 16'd7 + {8'd0, nrows} * w, seq);
            put_u32(id);
            put_p(row0[7:0]); put_p(row0[15:8]); put_p(nrows);
            for (j = 0; j < nrows * w; j = j + 1) begin
                px = id[7:0] + row0[7:0] * 8'd16 + j[7:0];   // deterministic
                put_p(px);
                if (track_ref && which == 0)
                    ref_cam7 = crc16_up(ref_cam7, px);
                if (track_ref && which == 1)
                    ref_cam8 = crc16_up(ref_cam8, px);
            end
            pkt_end(corrupt);
        end
    endtask

    integer expect_i;
    task expect_echo(input [7:0] stream, input [31:0] id,
                     input [15:0] dg, input [31:0] cnt);
        begin
            // rxbuf[expect_i]: A5 5A 90 00 0B 00 sq sq | s id4 dg2 cnt4 | crc2
            if (rxbuf[expect_i + 2] !== 8'h90 ||
                rxbuf[expect_i + 8] !== stream ||
                rxbuf[expect_i + 9]  !== id[7:0] ||
                rxbuf[expect_i + 13] !== dg[7:0] ||
                rxbuf[expect_i + 14] !== dg[15:8] ||
                rxbuf[expect_i + 15] !== cnt[7:0]) begin
                $display("FAIL: echo mismatch at %0d (want stream %02x id %0d dg %04x)",
                         expect_i, stream, id, dg);
                $display("  got type %02x stream %02x id %02x dg %02x%02x cnt %02x",
                         rxbuf[expect_i+2], rxbuf[expect_i+8],
                         rxbuf[expect_i+9], rxbuf[expect_i+14],
                         rxbuf[expect_i+13], rxbuf[expect_i+15]);
                $finish;
            end
            expect_i = expect_i + 21;              // 19 + crc2
        end
    endtask

    initial begin
        ref_cam7 = 16'hFFFF; ref_cam8 = 16'hFFFF; ref_lid9 = 16'hFFFF;
        #2000;

        // garbage (resync test) — includes a lone magic byte
        send_byte(8'h11); send_byte(8'hA5); send_byte(8'h77);

        // ---- cam frame id=7 (8x4), two row packets ----
        pkt_begin(8'h03, 16'd17, 16'd0);           // CAM_HDR
        put_u32(32'd7); put_u64z;
        put_p(8'd8); put_p(8'd0);                  // w=8
        put_p(8'd4); put_p(8'd0);                  // h=4
        put_p(8'd0);                               // fmt Y8
        pkt_end(0);
        cam_row(32'd7, 16'd0, 8'd2, 16'd8, 16'd1, 0, 1, 0);
        cam_row(32'd7, 16'd2, 8'd2, 16'd8, 16'd2, 0, 1, 0);

        // ---- lidar frame id=9 (3 rings x 4 az, range16) ----
        pkt_begin(8'h01, 16'd19, 16'd3);           // LIDAR_HDR n_rings=3
        put_u32(32'd9); put_u64z;
        put_p(8'd3);                               // n_rings
        put_p(8'd16); put_p(8'd33); put_p(8'd50);  // ring_ids
        put_p(8'd4); put_p(8'd0);                  // n_az=4
        put_p(8'd1);                               // fmt range16
        pkt_end(0);
        pkt_begin(8'h02, 16'd7 + 16'd24, 16'd4);   // LIDAR_COL 4 cols
        put_u32(32'd9); put_p(8'd0); put_p(8'd0); put_p(8'd4);
        for (j = 0; j < 24; j = j + 1) begin
            px = 8'h30 + j[7:0];
            put_p(px); ref_lid9 = crc16_up(ref_lid9, px);
        end
        pkt_end(0);

        // ---- cam frame id=8 (8x2): corrupt first attempt, then clean --
        pkt_begin(8'h03, 16'd17, 16'd5);
        put_u32(32'd8); put_u64z;
        put_p(8'd8); put_p(8'd0); put_p(8'd2); put_p(8'd0); put_p(8'd0);
        pkt_end(0);
        cam_row(32'd8, 16'd0, 8'd2, 16'd8, 16'd6, 1, 0, 1);  // CORRUPT
        cam_row(32'd8, 16'd0, 8'd2, 16'd8, 16'd7, 0, 1, 1);  // clean resend

        // ---- STATUS request ----
        pkt_begin(8'h10, 16'd5, 16'd8);
        put_p(8'd1); put_u32(32'd0);
        pkt_end(0);

        #3_000_000;                                // drain TX

        // ---- checks ----
        expect_i = 0;
        expect_echo(8'd3, 32'd7, ref_cam7, 32'd32);
        expect_echo(8'd1, 32'd9, ref_lid9, 32'd24);
        expect_echo(8'd3, 32'd8, ref_cam8, 32'd16);
        // STATUS: A5 5A 80 00 0C 00 sq sq | ok4 drops2 gaps2 cam2 lid2 |crc
        if (rxbuf[expect_i + 2] !== 8'h80) begin
            $display("FAIL: no status packet at %0d (got %02x)", expect_i,
                     rxbuf[expect_i + 2]);
            $finish;
        end
        if (rxbuf[expect_i + 12] !== 8'd1) begin
            $display("FAIL: crc_drops %0d != 1", rxbuf[expect_i + 12]);
            $finish;
        end
        if (rxbuf[expect_i + 14] !== 8'd1) begin
            $display("FAIL: seq_gaps %0d != 1", rxbuf[expect_i + 14]);
            $finish;
        end
        if (rxbuf[expect_i + 16] !== 8'd2 ||
            rxbuf[expect_i + 18] !== 8'd1) begin
            $display("FAIL: frames cam %0d lid %0d != 2/1",
                     rxbuf[expect_i + 16], rxbuf[expect_i + 18]);
            $finish;
        end
        $display("PASS: resync + 3 frame digests bit-exact + rollback + status (drops=1 gaps=1 cam=2 lid=1)");
        $finish;
    end
endmodule
