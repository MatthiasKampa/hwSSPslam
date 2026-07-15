// tb_fast9.v — gate fast9.v bit-exact against golden_cam.py vectors.
// Fixtures (written by `python3 -m runners.spot_cam vectors <run>` or
// golden_cam.write_vectors): fast9_img.hex (W*H bytes, raster) and
// fast9_exp.hex (W*H 16-bit words: {corner, score[11:0]}). The DUT emits
// centres (y-3, x-3) for y,x >= 6 — exactly the golden interior.
`timescale 1ns/1ps

module tb;
    localparam W = 64, H = 64;
    localparam THRESH = `THRESH;     // -DTHRESH=<t> must match the fixture

    reg  clk = 0, rst = 1;
    reg  in_valid = 0;
    reg  [7:0] pix = 0;
    wire out_valid, out_corner;
    wire [11:0] out_score;

    fast9 #(.W(W)) dut (
        .clk(clk), .rst(rst), .t(THRESH[7:0]),
        .in_valid(in_valid), .in_pix(pix),
        .out_valid(out_valid), .out_corner(out_corner),
        .out_score(out_score));

    reg [7:0]  img [0:W*H-1];
    reg [15:0] exp [0:W*H-1];

    integer i, oy, ox, errs, nout;
    reg [15:0] want;

    always #5 clk = ~clk;

    task check;
        begin
            if (out_valid) begin
                want = exp[(oy - 3) * W + (ox - 3)];
                if ({3'b000, out_corner, out_score} !== want) begin
                    if (errs < 10)
                        $display("MISMATCH at centre (%0d,%0d): dut {%b,%0d} exp %04x",
                                 oy - 3, ox - 3, out_corner, out_score, want);
                    errs = errs + 1;
                end
                nout = nout + 1;
                if (ox == W - 1) begin ox = 6; oy = oy + 1; end
                else ox = ox + 1;
            end
        end
    endtask

    initial begin
        $readmemh("fast9_img.hex", img);
        $readmemh("fast9_exp.hex", exp);
        @(posedge clk); @(negedge clk); rst = 0;   // release off-edge
        oy = 6; ox = 6; errs = 0; nout = 0;
        for (i = 0; i < W * H; i = i + 1) begin
            @(negedge clk);
            in_valid = 1;
            pix = img[i];
            @(posedge clk);
            #1;
            check;
        end
        // drain: the last centre's verdict lands one edge after the stream
        @(negedge clk); in_valid = 0;
        repeat (3) begin
            @(posedge clk); #1; check;
        end
        if (errs == 0 && nout == (W - 6) * (H - 6))
            $display("PASS: %0d centres bit-exact vs golden (t=%0d)",
                     nout, THRESH);
        else
            $display("FAIL: %0d mismatches, %0d/%0d centres",
                     errs, nout, (W - 6) * (H - 6));
        $finish;
    end
endmodule
