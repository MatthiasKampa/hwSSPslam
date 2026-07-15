// fast9.v — streaming integer FAST-9 corner detector (ECP5 track v0).
// Contract: bit-exact vs hw/ecp5/host/golden_cam.py fast9() (pre-NMS
// corner mask + 12-bit score). Raster-streamed 8-bit pixels, one/clk when
// in_valid; emits the (y-3, x-3) centre verdict once the 7x7 window is
// full. Comparators and adders only — no multipliers (OV5640 DVP pixel
// clock streams straight through; NMS is host-side in v0).
// Line buffers are async-read register arrays in v0 (distributed RAM);
// EBR-registered reads are the v1 follow-up.
`default_nettype none

module fast9 #(
    parameter W = 320
) (
    input  wire        clk,
    input  wire        rst,
    input  wire [7:0]  t,          // FAST threshold (host servo / feedback)
    input  wire        in_valid,
    input  wire [7:0]  in_pix,
    output reg         out_valid  /* verilator lint_off UNDRIVEN */,
    output reg         out_corner,
    output reg  [11:0] out_score
);
    initial begin
        out_valid = 1'b0; out_corner = 1'b0; out_score = 12'd0;
    end
    // ---- raster counters (declaration initials: ECP5 regs power up to
    // these, and it kills the classic tb reset-release race) ---------------
    reg [$clog2(W)-1:0] x = 0;
    reg [15:0] y = 0;
    wire x_last = (x == W - 1);

    // ---- 6 line buffers + 7x7 window --------------------------------------
    reg [7:0] lb0 [0:W-1]; reg [7:0] lb1 [0:W-1]; reg [7:0] lb2 [0:W-1];
    reg [7:0] lb3 [0:W-1]; reg [7:0] lb4 [0:W-1]; reg [7:0] lb5 [0:W-1];
    // column vector at x: row 0 = oldest (y-6) ... row 6 = current (y)
    wire [7:0] col [0:6];
    assign col[0] = lb5[x]; assign col[1] = lb4[x]; assign col[2] = lb3[x];
    assign col[3] = lb2[x]; assign col[4] = lb1[x]; assign col[5] = lb0[x];
    assign col[6] = in_pix;

    reg [7:0] win [0:6][0:6];      // win[r][6] = newest column
    integer r, c;

    always @(posedge clk) begin
        if (rst) begin
            x <= 0; y <= 0;
        end else if (in_valid) begin
            lb0[x] <= in_pix;  lb1[x] <= lb0[x];  lb2[x] <= lb1[x];
            lb3[x] <= lb2[x];  lb4[x] <= lb3[x];  lb5[x] <= lb4[x];
            for (r = 0; r < 7; r = r + 1) begin
                for (c = 0; c < 6; c = c + 1)
                    win[r][c] <= win[r][c+1];
                win[r][6] <= col[r];
            end
            if (x_last) begin
                x <= 0; y <= y + 1;
            end else
                x <= x + 1;
        end
    end

    // ---- circle taps (radius-3 Bresenham, order == golden CIRCLE) --------
    // centre = win[3][3]; tap i at win[3+dy][3+dx]
    wire [7:0] cen = win[3][3];
    wire [7:0] tap [0:15];
    assign tap[0]  = win[0][3];  assign tap[1]  = win[0][4];
    assign tap[2]  = win[1][5];  assign tap[3]  = win[2][6];
    assign tap[4]  = win[3][6];  assign tap[5]  = win[4][6];
    assign tap[6]  = win[5][5];  assign tap[7]  = win[6][4];
    assign tap[8]  = win[6][3];  assign tap[9]  = win[6][2];
    assign tap[10] = win[5][1];  assign tap[11] = win[4][0];
    assign tap[12] = win[3][0];  assign tap[13] = win[2][0];
    assign tap[14] = win[1][1];  assign tap[15] = win[0][2];

    // ---- per-tap diffs, arms, relu terms (explicit widths; the iCE40
    // lesson: no $signed() on part-selects) --------------------------------
    wire signed [9:0] ts = {2'b00, t};
    wire [15:0] bright, dark;
    wire [8:0] rb [0:15];          // relu(d - t)  <= 251
    wire [8:0] rd [0:15];          // relu(-d - t)
    genvar gi;
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : arms
            wire signed [9:0] d = $signed({2'b00, tap[gi]})
                                - $signed({2'b00, cen});
            assign bright[gi] = (d > ts);
            assign dark[gi]   = (d < -ts);
            wire signed [9:0] db = d - ts;
            wire signed [9:0] dd = -d - ts;
            assign rb[gi] = db > 0 ? db[8:0] : 9'd0;
            assign rd[gi] = dd > 0 ? dd[8:0] : 9'd0;
        end
    endgenerate

    // ---- stage 1: register per-tap arms + relu terms (timing: the full
    // diff->contig->sum cone missed 50 MHz single-cycle; iCE40-v6 pd-stage
    // pattern) --------------------------------------------------------------
    // Valid-path bookkeeping: hv0 marks "pixel ingested at T had a full
    // window" (set AT T); the arms computed from the post-T window belong
    // to that same pixel but latch one edge later (at T+1), so the valid
    // bit rides one register ahead of the data at every stage.
    wire have = (y >= 6) && (x >= 6);
    reg         hv0 = 1'b0;
    reg         s1_have = 1'b0;
    reg  [15:0] s1_bright = 0, s1_dark = 0;
    reg  [8:0]  s1_rb [0:15];
    reg  [8:0]  s1_rd [0:15];
    always @(posedge clk) begin
        if (rst) begin
            hv0 <= 1'b0; s1_have <= 1'b0; s1_bright <= 0; s1_dark <= 0;
        end else begin
            hv0       <= in_valid && have;
            s1_have   <= hv0;
            s1_bright <= bright;
            s1_dark   <= dark;
            for (r = 0; r < 16; r = r + 1) begin
                s1_rb[r] <= rb[r];
                s1_rd[r] <= rd[r];
            end
        end
    end

    // ---- stage 2 comb: >=9 contiguous (wraparound) + arm sums --------------
    wire [31:0] b2 = {s1_bright, s1_bright};
    wire [31:0] d2 = {s1_dark, s1_dark};
    wire [15:0] cb, cd;
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : contig
            assign cb[gi] = &b2[gi +: 9];
            assign cd[gi] = &d2[gi +: 9];
        end
    endgenerate
    wire corner = (|cb) | (|cd);

    wire [12:0] bs =
        (s1_rb[0] + s1_rb[1]) + (s1_rb[2] + s1_rb[3])
      + (s1_rb[4] + s1_rb[5]) + (s1_rb[6] + s1_rb[7])
      + (s1_rb[8] + s1_rb[9]) + (s1_rb[10] + s1_rb[11])
      + (s1_rb[12] + s1_rb[13]) + (s1_rb[14] + s1_rb[15]);
    wire [12:0] ds =
        (s1_rd[0] + s1_rd[1]) + (s1_rd[2] + s1_rd[3])
      + (s1_rd[4] + s1_rd[5]) + (s1_rd[6] + s1_rd[7])
      + (s1_rd[8] + s1_rd[9]) + (s1_rd[10] + s1_rd[11])
      + (s1_rd[12] + s1_rd[13]) + (s1_rd[14] + s1_rd[15]);
    wire [12:0] mx = (bs > ds) ? bs : ds;

    // ---- output stage: centre (y-3, x-3); total latency 2 edges ------------
    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0; out_corner <= 1'b0; out_score <= 12'd0;
        end else begin
            out_valid  <= s1_have;
            out_corner <= corner;
            out_score  <= corner ? mx[11:0] : 12'd0;
        end
    end
endmodule
`default_nettype wire
