// Matcher testbench (encoder_match.v): encode build/points.hex, load
// build/mcodes.hex (240 x 2b QPSK codes, {ring,j} order), then run the
// candidates in build/mcmds.hex (4 lines each: dx, dy, rho, sh as 16-bit
// hex, dx/dy signed) and print "SC cand ring re im" for every ring.
`timescale 1ns/1ps
module tb;
    reg clk = 0;
    always #13.8 clk = ~clk;               // ~36 MHz

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
                 .m_start(m_start), .m_dx(m_dx), .m_dy(m_dy),
                 .m_rho(m_rho), .m_sh(m_sh), .m_done(m_done),
                 .s_sel(s_sel), .s_re(s_re), .s_im(s_im));

    reg [15:0] pts [0:3*4096-1];
    reg [15:0] mcs [0:239];
    reg [15:0] cmd [0:4*256-1];
    integer n, i, c, k, nc;
    initial begin
        $readmemh("build/points.hex", pts);
        $readmemh("build/mcodes.hex", mcs);
        $readmemh("build/mcmds.hex", cmd);
        n = 0;
        while (pts[3 * n] !== 16'hxxxx && n < 4096) n = n + 1;
        nc = 0;
        while (cmd[4 * nc] !== 16'hxxxx && nc < 256) nc = nc + 1;
        @(posedge clk); #1;
        clear = 1; @(posedge clk); #1; clear = 0;
        @(posedge clk);
        while (busy) @(posedge clk);
        for (i = 0; i < n; i = i + 1) begin
            #1;
            az = pts[3 * i][9:0];
            r_mm = pts[3 * i + 1];
            w = pts[3 * i + 2][7:0];
            start = 1; @(posedge clk); #1; start = 0;
            @(posedge clk);
            while (busy) @(posedge clk);
        end
        for (i = 0; i < 240; i = i + 1) begin  // M codes: {ring, j} address
            #1;
            mc_addr = (i / 60) * 64 + (i % 60);
            mc_code = mcs[i][1:0];
            mc_we = 1; @(posedge clk); #1; mc_we = 0;
        end
        for (c = 0; c < nc; c = c + 1) begin
            #1;
            m_dx  = cmd[4 * c];
            m_dy  = cmd[4 * c + 1];
            m_rho = cmd[4 * c + 2][5:0];
            m_sh  = cmd[4 * c + 3][3:0];
            m_start = 1; @(posedge clk); #1; m_start = 0;
            @(posedge clk);
            while (!m_done) @(posedge clk);
            for (k = 0; k < 4; k = k + 1) begin
                s_sel = k[1:0]; #1;
                $display("SC %0d %0d %0d %0d", c, k, s_re, s_im);
            end
        end
        $display("TB DONE n=%0d nc=%0d", n, nc);
        $finish;
    end
endmodule
