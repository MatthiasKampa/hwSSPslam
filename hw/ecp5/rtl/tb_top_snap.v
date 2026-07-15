// tb_top_snap.v — FULL-SYSTEM deploy sim of top_ov5640_snap: uart BFM
// host + bit-level SCCB slave (ACKs the init ROM, serves the chip-ID)
// + synthetic DVP source. Exercises every UART command end-to-end:
//   boot -> auto init (all 172 writes ACKed over the real SCCB engine)
//   'R' poll until init_done, nerr == 0
//   'r' chip-ID 0x300A -> expect {0x56, err 0}
//   'W' register write -> expect {err 0}
//   'S' snapshot -> W*H bytes, bit-exact vs the per-frame-seeded LFSR
//       pattern of exactly ONE source frame
//   'R' again -> n_bytes == W*H, lines == H, bytes/line == W
// Shrunk geometry (64x48) + fast boot/SCCB/delay parameters keep the
// whole run ~10 ms of sim time.
`timescale 1ns/1ps

module tb;
    localparam W = 64, H = 48;
    localparam NPIX = W * H;

    reg clk = 0;
    always #10 clk = ~clk;                     // 50 MHz

    // ---- DUT ------------------------------------------------------------
    wire rx_line, tx_line;
    tri1 siod;                                 // module pullup
    wire sioc;
    reg  pclk = 0;
    always #40 pclk = ~pclk;                   // 12.5 MHz
    reg        href = 0, vsync = 0;
    reg [7:0]  d = 0;
    wire       resetb, pwdn, xclk;

    top #(.W(W), .H(H), .BOOT_BIT(10), .DIV_SCCB(8), .MS_CYC(10),
          .ROMFILE("ov5640_init.hex")) dut (
        .clk(clk), .usb_rx(rx_line), .usb_tx(tx_line), .led(),
        .cam_xclk(xclk), .cam_resetb(resetb), .cam_pwdn(pwdn),
        .cam_siod(siod), .cam_sioc(sioc),
        .cam_pclk(pclk), .cam_href(href), .cam_vsync(vsync), .cam_d(d));

    // ---- host uart BFM ----------------------------------------------------
    reg  [7:0] hd = 0;
    reg        hsend = 0;
    wire       hbusy;
    uart_tx #(.DIV(25)) host_tx (.clk(clk), .data(hd), .send(hsend),
                                 .tx(rx_line), .busy(hbusy));
    wire [7:0] rd;
    wire       rv;
    uart_rx #(.DIV(25)) host_rx (.clk(clk), .rx(tx_line), .data(rd),
                                 .valid(rv));
    reg [7:0] rxbuf [0:NPIX+63];
    integer   nrx = 0;
    always @(posedge clk)
        if (rv) begin
            rxbuf[nrx] = rd;
            nrx = nrx + 1;
        end

    task send(input [7:0] b);
        begin
            @(negedge clk);
            hd = b; hsend = 1;
            @(negedge clk);
            hsend = 0;
            wait (hbusy);
            wait (!hbusy);
            repeat (50) @(posedge clk);
        end
    endtask

    task waitrx(input integer n);
        integer t;
        begin
            t = 0;
            while (nrx < n && t < 4_000_000) begin
                @(posedge clk);
                t = t + 1;
            end
            if (nrx < n) begin
                $display("TOP-SNAP FAIL: rx timeout (%0d/%0d)", nrx, n);
                $finish;
            end
        end
    endtask

    // ---- SCCB slave (tb_sccb's bit-level model: ACK + read-serve) --------
    reg sda_drive = 1'b1;
    assign siod = sda_drive ? 1'bz : 1'b0;
    integer bitcnt = 0;
    reg [7:0] serve = 8'h56;                   // chip-ID high byte
    reg [7:0] sh;
    reg prev_scl = 1, prev_sda = 1;
    reg started = 0, post_sr = 0;
    integer n_ack = 0;
    wire scl_now = (sioc !== 1'b0);
    wire sda_now = (siod !== 1'b0);
    always @(posedge clk) begin
        if (prev_scl && scl_now && prev_sda && !sda_now) begin  // START
            bitcnt <= 0;
            sda_drive <= 1'b1;
            if (started) post_sr <= 1;
            started <= 1;
        end
        if (prev_scl && scl_now && !prev_sda && sda_now) begin  // STOP
            started <= 0;
            post_sr <= 0;
        end
        if (!prev_scl && scl_now)
            bitcnt <= bitcnt + 1;
        if (prev_scl && !scl_now) begin
            if ((bitcnt % 9) == 8 && !(post_sr && bitcnt == 17)) begin
                sda_drive <= 1'b0;
                n_ack <= n_ack + 1;
                if (bitcnt == 8 && post_sr)
                    sh <= serve;
            end else if (post_sr && bitcnt >= 9 && bitcnt < 17) begin
                sda_drive <= sh[7] ? 1'b1 : 1'b0;
                sh <= {sh[6:0], 1'b0};
            end else
                sda_drive <= 1'b1;
        end
        prev_scl <= scl_now;
        prev_sda <= sda_now;
    end

    // ---- DVP source: endless frames, per-frame LFSR seed ------------------
    function [15:0] lnext(input [15:0] s);
        lnext = {s[14:0], s[15] ^ s[13] ^ s[12] ^ s[10]};
    endfunction
    reg [15:0] lf;
    integer fidx = 0;
    integer fy, fx;
    initial begin : dvp
        wait (resetb === 1'b1);
        forever begin
            @(negedge pclk); vsync = 1;
            repeat (20) @(negedge pclk);
            vsync = 0;
            repeat (12) @(negedge pclk);
            lf = 16'h1000 + fidx;
            for (fy = 0; fy < H; fy = fy + 1) begin
                for (fx = 0; fx < W; fx = fx + 1) begin
                    @(negedge pclk);
                    href = 1;
                    d = lf[7:0];
                    lf = lnext(lf);
                end
                @(negedge pclk);
                href = 0;
                repeat (7) @(negedge pclk);
            end
            fidx = fidx + 1;
        end
    end

    // ---- checks -----------------------------------------------------------
    integer errs = 0;
    integer i, tries, nmatch, mf;
    reg [15:0] ck;
    reg all, initdone;
    initial begin
        // 1. poll 'R' until init done (13-byte report)
        tries = 0;
        initdone = 0;
        while (!initdone) begin
            nrx = 0;
            send("R");
            waitrx(13);
            if (rxbuf[12][0]) begin
                if (rxbuf[11] !== 8'd0) begin
                    $display("TOP-SNAP FAIL: init nerr=%0d", rxbuf[11]);
                    $finish;
                end
                $display("  init done, nerr 0 (slave ACKs %0d)", n_ack);
                initdone = 1;
            end else begin
                tries = tries + 1;
                if (tries > 400) begin
                    $display("TOP-SNAP FAIL: init never finished");
                    $finish;
                end
                repeat (20_000) @(posedge clk);
            end
        end
        // 2. chip-ID read via passthrough
        nrx = 0;
        serve = 8'h56;
        send("r"); send(8'h30); send(8'h0A);
        waitrx(2);
        if (rxbuf[0] !== 8'h56 || rxbuf[1] !== 8'h00) begin
            $display("TOP-SNAP FAIL: chip-ID got %02x err %02x",
                     rxbuf[0], rxbuf[1]);
            errs = errs + 1;
        end else
            $display("  chip-ID read 0x56, err 0");
        // 3. register write via passthrough
        nrx = 0;
        send("W"); send(8'h30); send(8'h36); send(8'h2C);
        waitrx(1);
        if (rxbuf[0] !== 8'h00) begin
            $display("TOP-SNAP FAIL: write err %02x", rxbuf[0]);
            errs = errs + 1;
        end else
            $display("  passthrough write ACKed");
        // 4. snapshot: arm mid-frame, receive NPIX bytes
        nrx = 0;
        send("S");
        waitrx(NPIX);
        nmatch = 0;
        for (mf = 0; mf < fidx + 2; mf = mf + 1) begin
            ck = 16'h1000 + mf;
            all = 1;
            for (i = 0; i < NPIX; i = i + 1) begin
                if (rxbuf[i] !== ck[7:0])
                    all = 0;
                ck = lnext(ck);
            end
            if (all) begin
                nmatch = nmatch + 1;
                $display("  snapshot bit-exact = source frame %0d", mf);
            end
        end
        if (nmatch !== 1) begin
            $display("TOP-SNAP FAIL: %0d frame nmatch", nmatch);
            errs = errs + 1;
        end
        // 5. final report: geometry + n_bytes
        nrx = 0;
        send("R");
        waitrx(13);
        if ({rxbuf[8][0], rxbuf[9], rxbuf[10]} != NPIX
            || {rxbuf[4], rxbuf[5]} != H
            || {rxbuf[6], rxbuf[7]} != W) begin
            $display("TOP-SNAP FAIL: report n=%0d lines=%0d bpl=%0d",
                     {rxbuf[8][0], rxbuf[9], rxbuf[10]},
                     {rxbuf[4], rxbuf[5]}, {rxbuf[6], rxbuf[7]});
            errs = errs + 1;
        end else
            $display("  report: %0d bytes, %0dx%0d geometry",
                     NPIX, H, W);
        if (errs == 0)
            $display("TOP-SNAP PASS: init + chip-ID + write + snapshot + report end-to-end");
        else
            $display("TOP-SNAP FAIL: %0d errors", errs);
        $finish;
    end

    initial begin
        #60_000_000;
        $display("TOP-SNAP FAIL: global timeout (init_done=%b st=%0d)",
                 dut.init_done, dut.st);
        $finish;
    end
endmodule
