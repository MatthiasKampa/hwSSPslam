// tb_solo_bridge.v — v7 stage-1 gate: RESIDENT-MAP addressing.
// Encode build/solo_pts.hex (one fixture scan), preload TWO segments'
// codes (build/solo_seg_a.hex at base segA, build/solo_seg_b.hex at
// base segB), then run the candidates in build/solo_cands.hex
// (5 lines each: seg, dx, dy, rho, sh) and print "SC cand ring re im"
// — python (solo_bridge.py) diffs against ssp_ice40.match_int.
`timescale 1ns/1ps
module tb;
    reg clk = 0;
    always #20.8 clk = ~clk;               // ~24 MHz

    reg clear = 0, start = 0;
    reg [9:0] az;
    reg [15:0] r_mm;
    reg [7:0] w;
    wire busy;
    reg [7:0] rd_idx = 0;
    wire signed [31:0] rd_re, rd_im;
    reg        mc_we = 0;
    reg [7:0]  mc_addr;
    reg [1:0]  mc_code;
    reg [5:0]  mc_seg = 0;
    reg        m_start = 0;
    reg signed [15:0] m_dx, m_dy;
    reg [5:0]  m_rho;
    reg [3:0]  m_sh;
    wire       m_done;
    reg [1:0]  s_sel = 0;
    wire signed [31:0] s_re, s_im;
    encoder enc (.clk(clk), .clear(clear), .start(start), .az(az),
                 .r_mm(r_mm), .w(w), .busy(busy), .rd_idx(rd_idx),
                 .rd_re(rd_re), .rd_im(rd_im),
                 .mc_we(mc_we), .mc_addr(mc_addr), .mc_code(mc_code),
                 .mc_seg(mc_seg),
                 .m_start(m_start), .m_dx(m_dx), .m_dy(m_dy),
                 .m_rho(m_rho), .m_sh(m_sh), .m_done(m_done),
`ifdef LEAN
                 .f_en(1'b0), .f_cq(16'sd0), .f_sq(16'sd0),
                 .f_tx(16'sd0), .f_ty(16'sd0),
`endif
                 .s_sel(s_sel), .s_re(s_re), .s_im(s_im));

    reg [15:0] pts [0:3*4096-1];
    reg [15:0] sega [0:239];
    reg [15:0] segb [0:239];
    reg [15:0] cmd [0:5*64-1];
    integer n, i, c, k, nc;
    initial begin
        $readmemh("build/solo_pts.hex", pts);
        $readmemh("build/solo_seg_a.hex", sega);
        $readmemh("build/solo_seg_b.hex", segb);
        $readmemh("build/solo_cands.hex", cmd);
        n = pts[0]; nc = cmd[0];
        @(posedge clk); clear <= 1; @(posedge clk); clear <= 0;
        @(posedge clk);
        while (busy) @(posedge clk);   // S_CLR runs 60 cycles; start is
                                       // only honored in S_IDLE
        // encode the scan
        for (i = 0; i < n; i = i + 1) begin
            @(posedge clk);
            az   <= pts[1 + 3*i][9:0];
            r_mm <= pts[2 + 3*i];
            w    <= pts[3 + 3*i][7:0];
            start <= 1;
            @(posedge clk);
            start <= 0;
            @(posedge clk);
            while (busy) @(posedge clk);
        end
        // encode readback: diff the accumulators against golden Q first
        for (i = 0; i < 240; i = i + 1) begin
            rd_idx <= {2'((i / 60)), 6'((i % 60))};
            @(posedge clk); @(posedge clk); @(posedge clk);
            $display("RD %0d %0d %0d", i, rd_re, rd_im);
        end
        // preload segment A at base 8, segment B at base 9 (mc_we path).
        // Address layout is {ring[1:0], j[5:0]} — ring-major with 64-slot
        // stride (the deploy M-load format), NOT linear 0..239.
        for (i = 0; i < 240; i = i + 1) begin
            @(posedge clk);
            mc_seg <= 6'd8;
            mc_addr <= {2'((i / 60)), 6'((i % 60))};
            mc_code <= sega[i][1:0]; mc_we <= 1;
            @(posedge clk); mc_we <= 0;
        end
        for (i = 0; i < 240; i = i + 1) begin
            @(posedge clk);
            mc_seg <= 6'd9;
            mc_addr <= {2'((i / 60)), 6'((i % 60))};
            mc_code <= segb[i][1:0]; mc_we <= 1;
            @(posedge clk); mc_we <= 0;
        end
        // candidates: seg, dx, dy, rho, sh. NOTE: ring 3's final
        // accumulate lands ON the m_done edge via the S_MD drain, so a
        // single run + post-done reads is sufficient (the historical x
        // here was the {ring,j} preload-addressing bug, not timing).
        // Each candidate runs TWICE back-to-back anyway — behaviour-
        // neutral (identical totals) and it exercises the hot-batch
        // regime the deploy always uses.
        for (c = 0; c < nc; c = c + 1) begin
            for (i = 0; i < 2; i = i + 1) begin
                @(posedge clk);
                mc_seg <= cmd[1 + 5*c][5:0];
                m_dx  <= cmd[2 + 5*c];
                m_dy  <= cmd[3 + 5*c];
                m_rho <= cmd[4 + 5*c][5:0];
                m_sh  <= cmd[5 + 5*c][3:0];
                m_start <= 1;
                @(posedge clk);
                m_start <= 0;
                @(posedge clk);
                while (!m_done) @(posedge clk);
            end
            for (k = 0; k < 4; k = k + 1) begin
                s_sel <= k[1:0];
                @(posedge clk); @(posedge clk);
                $display("SC %0d %0d %0d %0d", c, k, s_re, s_im);
            end
        end
        $finish;
    end
endmodule
