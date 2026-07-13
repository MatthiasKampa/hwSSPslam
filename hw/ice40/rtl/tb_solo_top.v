// tb_solo_top.v — v7 TOP-LEVEL gate: full UART protocol replay against
// top_solo (encoder_lean + solo_tracker REFINE=0). Reads the paced word
// stream build/solo_topin.hex (0x00XX = send byte XX over RX at line
// rate; 0x01NN = wait until NN more TX bytes arrive — the pose frame /
// ack / dump is the flow control, mirroring the host contract), captures
// every TX byte via a local uart_rx and prints "TXB xx" — solo_top.py
// diffs pose frames + the dump against the golden byte-for-byte.
`timescale 1ns/1ps
module tb;
    reg clk = 0;
    always #20.8 clk = ~clk;               // ~24 MHz

    localparam UDIV = 4;                   // sim baud: 4 clks/bit

    wire rx_line, tx_line;
    reg  [7:0] send_b;
    reg        send_go = 0;
    wire       send_busy;
    uart_tx #(.DIV(UDIV)) hosttx (.clk(clk), .data(send_b), .send(send_go),
                                  .tx(rx_line), .busy(send_busy));
    wire [7:0] cap_b;
    wire       cap_v;
    uart_rx #(.DIV(UDIV)) hostrx (.clk(clk), .rx(tx_line), .data(cap_b),
                                  .valid(cap_v));
    integer tx_cnt = 0;
    integer n, i, target;
    always @(posedge clk) if (cap_v) begin
        $display("TXB %02x", cap_b);
        tx_cnt = tx_cnt + 1;
    end
    // watchdog: no TX progress for 20M cycles = a stuck wait -> dump state
    integer wd = 0, tx_last = 0;
    always @(posedge clk) begin
        if (tx_cnt != tx_last) begin tx_last = tx_cnt; wd = 0; end
        else wd = wd + 1;
        if (wd > 20000000) begin
            $display("WATCHDOG: tx_cnt=%0d i=%0d kst=%0d dst=%0d prx=%0d enc_busy=%b trk_tst=%0d",
                     tx_cnt, i, dut.kst, dut.dst, dut.prx,
                     dut.enc_busy, dut.trk.tst);
            $finish;
        end
    end

    top_solo #(.UDIV(UDIV), .REFINE(0)) dut (.clk(clk), .RX(rx_line),
                                             .TX(tx_line),
                                             .LEDR_N(), .LEDG_N());

    reg [15:0] stream [0:6000000-1];
    initial begin
        $readmemh("build/solo_topin.hex", stream);
        @(posedge clk);
        for (i = 0; stream[i] !== 16'hxxxx && i < 6000000; i = i + 1) begin
            if (stream[i][8]) begin        // wait word
                target = tx_cnt + stream[i][7:0];
                while (tx_cnt < target) @(posedge clk);
            end else begin                 // send byte
                while (send_busy) @(posedge clk);
                send_b <= stream[i][7:0];
                send_go <= 1;
                @(posedge clk);
                send_go <= 0;
                @(posedge clk);
            end
        end
        // drain any tail bytes, then finish
        repeat (20000) @(posedge clk);
        // post-run memory peeks (bug isolation: writes vs read path)
        for (i = 0; i < 8; i = i + 1)
            $display("MCPEEK %0d %04x", i, dut.enc.mcram.mem[i]);
        for (i = 0; i < 8; i = i + 1)
            $display("S2ACC %0d %04x", i, dut.s2ram.mem[i]);
        for (i = 0; i < 8; i = i + 1)
            $display("S2ANC %0d %04x", i, dut.s2ram.mem[14'h0400 + i]);
        for (i = 0; i < 4; i = i + 1)
            $display("S2LIV %0d %04x", i, dut.s2ram.mem[14'h0600 + i]);
        $finish;
    end
endmodule
