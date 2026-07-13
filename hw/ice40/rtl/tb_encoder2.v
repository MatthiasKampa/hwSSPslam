// Simulation testbench for encoder_pipe.v (BIT-SLICED readback addressing
// {ring[1:0], j[5:0]} — see encoder_pipe.v). Same vectors/printout contract
// as tb_encoder.v: drive build/points.hex, print "ACC idx re im" with idx
// the PACKED component index k*60+j (so the golden compare is unchanged).
`timescale 1ns/1ps
module tb;
    reg clk = 0;
    always #41.6 clk = ~clk;               // ~12 MHz

    reg clear = 0, start = 0;
    reg [9:0] az;
    reg [15:0] r_mm;
    reg [7:0] w;
    wire busy;
    reg [7:0] rd_idx = 0;
    wire signed [31:0] rd_re, rd_im;
    encoder enc (.clk(clk), .clear(clear), .start(start), .az(az),
                 .r_mm(r_mm), .w(w), .busy(busy), .rd_idx(rd_idx),
                 .rd_re(rd_re), .rd_im(rd_im));

    reg [15:0] pts [0:3*4096-1];           // az, r, w per point
    integer n, i, k, j;
    initial begin
        $readmemh("build/points.hex", pts);
        n = 0;
        while (pts[3 * n] !== 16'hxxxx && n < 4096) n = n + 1;
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
        for (i = 0; i < 240; i = i + 1) begin
            k = i / 60;
            j = i % 60;
            rd_idx = {k[1:0], j[5:0]};
            @(posedge clk); @(posedge clk); #1;
            $display("ACC %0d %0d %0d", i, rd_re, rd_im);
        end
        $display("TB DONE n=%0d", n);
        $finish;
    end
endmodule
