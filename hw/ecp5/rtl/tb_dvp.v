// tb_dvp.v — gate dvp_capture.v against a synthetic DVP source.
// PCLK = 12.5 MHz (the predicted OV5640 operating point, 4x oversample
// at 50 MHz), transitions on sysclk fall edges; data/HREF/VSYNC change
// on PCLK fall, valid over PCLK rise (OV5640 polarity per 0x4740=0x21).
// Frames carry a per-frame-seeded LFSR pattern so the checker PROVES
// which frame landed in the snapshot: arm mid-frame-1 must capture
// frame 2 (Y8), re-arm mid-frame-3 in YUYV mode must capture frame 4's
// even (luma) bytes only. Also checks n_bytes and the free-running
// geometry counters (lines/frame, bytes/line, frame count).
`timescale 1ns/1ps

module tb;
    localparam W = 32, H = 8;

    reg clk = 0;
    always #10 clk = ~clk;                 // 50 MHz
    reg pclk = 0;
    always #40 pclk = ~pclk;               // 12.5 MHz, edges on clk falls

    reg        href = 0, vsync = 0, rst = 0;
    reg [7:0]  d = 0;
    reg        mode_y8 = 1, arm = 0;
    wire       busy, done;
    wire [16:0] n_bytes;
    reg  [16:0] rd_addr = 0;
    wire [7:0]  rd_data;
    wire [31:0] cnt_frames;
    wire [15:0] cnt_lines, cnt_bytes_line;

    dvp_capture #(.W(W), .H(H), .AW(17)) dut (
        .clk(clk), .rst(rst),
        .dvp_pclk(pclk), .dvp_href(href), .dvp_vsync(vsync), .dvp_d(d),
        .mode_y8(mode_y8), .arm(arm), .busy(busy), .done(done),
        .n_bytes(n_bytes), .rd_addr(rd_addr), .rd_data(rd_data),
        .cnt_frames(cnt_frames), .cnt_lines(cnt_lines),
        .cnt_bytes_line(cnt_bytes_line));

    function [15:0] lnext(input [15:0] s);
        lnext = {s[14:0], s[15] ^ s[13] ^ s[12] ^ s[10]};
    endfunction

    // one frame: VSYNC pulse, blank, H lines (W luma bytes; YUYV mode
    // interleaves 0xAA chroma), line blanks
    reg [15:0] lf;
    integer fy, fx, nb;
    task frame(input [15:0] seed, input y8);
        begin
            @(negedge pclk); vsync = 1;
            repeat (30) @(negedge pclk);
            vsync = 0;
            repeat (20) @(negedge pclk);
            lf = seed;
            nb = y8 ? W : 2 * W;
            for (fy = 0; fy < H; fy = fy + 1) begin
                for (fx = 0; fx < nb; fx = fx + 1) begin
                    @(negedge pclk);
                    href = 1;
                    if (y8 || !(fx & 1)) begin
                        d = lf[7:0];
                        lf = lnext(lf);
                    end else
                        d = 8'hAA;
                end
                @(negedge pclk);
                href = 0;
                d = 0;
                repeat (9) @(negedge pclk);
            end
            repeat (10) @(negedge pclk);
        end
    endtask

    integer errs = 0;
    reg [15:0] ck;
    integer i;
    task verify(input [15:0] seed, input [15:0] exp_bpl);
        begin
            if (n_bytes !== W * H) begin
                $display("FAIL: n_bytes %0d != %0d", n_bytes, W * H);
                errs = errs + 1;
            end
            ck = seed;
            for (i = 0; i < W * H; i = i + 1) begin
                @(negedge clk) rd_addr = i;
                @(negedge clk);
                if (rd_data !== ck[7:0]) begin
                    if (errs < 10)
                        $display("MISMATCH byte %0d: got %02x want %02x",
                                 i, rd_data, ck[7:0]);
                    errs = errs + 1;
                end
                ck = lnext(ck);
            end
            if (cnt_lines !== H) begin
                $display("FAIL: cnt_lines %0d != %0d", cnt_lines, H);
                errs = errs + 1;
            end
            if (cnt_bytes_line !== exp_bpl) begin
                $display("FAIL: bytes/line %0d != %0d",
                         cnt_bytes_line, exp_bpl);
                errs = errs + 1;
            end
        end
    endtask

    task pulse_arm;
        begin
            @(negedge clk) arm = 1;
            @(negedge clk) arm = 0;
        end
    endtask

    integer n_done = 0;
    always @(posedge clk) if (done) n_done = n_done + 1;

    // phase-sequential: stream a 2-frame pair with a mid-frame-1 arm
    // (proves partial frames are skipped), THEN verify while the source
    // is quiet — verification is slower than a frame, so concurrent
    // streaming would slip the vsync-edge synchronization by a frame.
    initial begin : ctrl
        rst = 1;
        repeat (4) @(negedge clk);
        rst = 0;
        fork
            begin
                frame(16'h1111, 1);
                frame(16'h2222, 1);        // <- captured (armed in f1)
            end
            begin
                @(negedge vsync);          // frame 1 body begins
                pulse_arm;                 // -> capture starts at frame 2
            end
        join
        if (n_done !== 1 || busy) begin
            $display("FAIL: phase1 n_done=%0d busy=%b", n_done, busy);
            errs = errs + 1;
        end
        verify(16'h2222, W);
        if (cnt_frames < 2) begin
            $display("FAIL: cnt_frames %0d < 2", cnt_frames);
            errs = errs + 1;
        end
        mode_y8 = 0;                       // YUYV: luma = even bytes
        fork
            begin
                frame(16'h3333, 0);
                frame(16'h4444, 0);        // <- captured (armed in f3)
            end
            begin
                @(negedge vsync);          // frame 3 body begins
                pulse_arm;                 // -> capture starts at frame 4
            end
        join
        if (n_done !== 2 || busy) begin
            $display("FAIL: phase2 n_done=%0d busy=%b", n_done, busy);
            errs = errs + 1;
        end
        verify(16'h4444, 2 * W);
        if (errs == 0)
            $display("DVP PASS: Y8 + YUYV snapshots bit-exact, counters OK");
        else
            $display("DVP FAIL: %0d errors", errs);
        $finish;
    end

    initial begin
        #40_000_000;
        $display("DVP FAIL: timeout (st=%0d busy=%b frames=%0d)",
                 dut.st, busy, cnt_frames);
        $finish;
    end
endmodule
