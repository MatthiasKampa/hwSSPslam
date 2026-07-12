// tb_solo_step.v — v7 stage-2 gate: the FULL tracker step loop.
// Preloads the resident map (build/solo_mapw.hex, flat words of 2b codes,
// seg-major {ring,j} order) and the anchor table (build/solo_ancw.hex,
// 8 words/entry), then replays build/solo_feedw.hex:
//   per kf: n_pts, {az, r_mm, w}*n, sh, pred_xlo, pred_xhi, pred_ylo,
//           pred_yhi, pred_h
// encodes the scan into encoder_solo, pulses step_start on solo_tracker,
// prints "KF n x y h ai score state" — python diffs vs solo_expect.
`timescale 1ns/1ps
module tb;
    reg clk = 0;
    always #20.8 clk = ~clk;

    // encoder core
    reg clear = 0, start = 0;
    reg [9:0] az;
    reg [15:0] r_mm;
    reg [7:0] w;
    wire busy;
    reg        pre_we = 0;             // tb map preload
    reg [7:0]  pre_addr;
    reg [1:0]  pre_code;
    reg [5:0]  pre_seg;
    wire       m_done;
    wire [1:0] s_sel;
    wire signed [31:0] s_re, s_im;
    // tracker
    reg  anc_we = 0;
    reg  [8:0]  anc_wa;
    reg  [15:0] anc_wd;
    reg  [5:0]  n_seg;
    reg         step_start = 0;
    reg  signed [31:0] pred_x, pred_y;
    reg  [9:0]  pred_h;
    wire        step_done;
    wire signed [31:0] out_x, out_y, out_score;
    wire [9:0]  out_h;
    wire [5:0]  out_ai;
    wire [1:0]  out_state;
    wire [5:0]  trk_seg;
    wire        trk_mstart;
    wire signed [15:0] trk_dx, trk_dy;
    wire [5:0]  trk_rho;
    wire [3:0]  trk_sh;
    reg  [3:0]  sh_in;

    wire [7:0] rd_idx;
    wire signed [31:0] rd_re, rd_im;
    encoder enc (.clk(clk), .clear(clear), .start(start), .az(az),
                 .r_mm(r_mm), .w(w), .busy(busy), .rd_idx(rd_idx),
                 .rd_re(rd_re), .rd_im(rd_im),
                 .mc_we(pre_we),
                 .mc_addr(pre_addr),
                 .mc_code(pre_code),
                 .mc_seg(pre_we ? pre_seg : trk_seg),
                 .m_start(trk_mstart), .m_dx(trk_dx), .m_dy(trk_dy),
                 .m_rho(trk_rho), .m_sh(trk_sh), .m_done(m_done),
`ifdef LEAN
                 .f_en(1'b0), .f_cq(16'sd0), .f_sq(16'sd0),
                 .f_tx(16'sd0), .f_ty(16'sd0),
`endif
                 .s_sel(s_sel), .s_re(s_re), .s_im(s_im));

`ifdef REFINE0
    localparam TRK_REFINE = 0;
`else
    localparam TRK_REFINE = 1;
`endif
    solo_tracker #(.REFINE(TRK_REFINE)) trk (.clk(clk),
                 .anc_we(anc_we), .anc_wa(anc_wa), .anc_wd(anc_wd),
                 .n_seg(n_seg),
                 .step_start(step_start), .pred_x(pred_x),
                 .pred_y(pred_y), .pred_h(pred_h),
                 .step_done(step_done), .out_x(out_x), .out_y(out_y),
                 .out_h(out_h), .out_ai(out_ai), .out_score(out_score),
                 .out_state(out_state),
                 .mc_seg(trk_seg), .m_start(trk_mstart),
                 .m_dx(trk_dx), .m_dy(trk_dy), .m_rho(trk_rho),
                 .m_sh(trk_sh), .m_busy(busy), .m_done(m_done),
                 .s_sel(s_sel), .s_re(s_re), .sh_in(sh_in),
                 .rd_idx(rd_idx), .rd_re(rd_re), .rd_im(rd_im),
                 .svc_start(1'b0), .svc_op(2'd0), .svc_x(16'sd0),
                 .svc_y(16'sd0), .svc_h(10'd0),
                 .svc_done(), .o_cq(), .o_sq(), .o_fx(), .o_fy());

    reg [15:0] mapw [0:64*240-1];
    reg [15:0] ancw [0:64*8-1];
    reg [15:0] feed [0:1000000-1];
    integer fp, nkf, kf, i, n, sg;
    initial begin
        $readmemh("build/solo_mapw.hex", mapw);
        $readmemh("build/solo_ancw.hex", ancw);
        $readmemh("build/solo_feedw.hex", feed);
        n_seg = feed[0][5:0];
        nkf   = feed[1];
        fp = 2;
        // preload map + anchors
        for (sg = 0; sg < n_seg; sg = sg + 1)
            for (i = 0; i < 240; i = i + 1) begin
                @(posedge clk);
                pre_seg <= sg[5:0];
                pre_addr <= {2'((i / 60)), 6'((i % 60))};
                pre_code <= mapw[sg*240 + i][1:0];
                pre_we <= 1;
                @(posedge clk); pre_we <= 0;
            end
        for (i = 0; i < 8*n_seg; i = i + 1) begin
            @(posedge clk);
            anc_wa <= i[8:0]; anc_wd <= ancw[i]; anc_we <= 1;
            @(posedge clk); anc_we <= 0;
        end
        // keyframe loop
        for (kf = 0; kf < nkf; kf = kf + 1) begin
            n = feed[fp]; fp = fp + 1;
            @(posedge clk); clear <= 1; @(posedge clk); clear <= 0;
            @(posedge clk);
            while (busy) @(posedge clk);
            for (i = 0; i < n; i = i + 1) begin
                @(posedge clk);
                az   <= feed[fp][9:0];
                r_mm <= feed[fp+1];
                w    <= feed[fp+2][7:0];
                fp = fp + 3;
                start <= 1;
                @(posedge clk);
                start <= 0;
                @(posedge clk);
                while (busy) @(posedge clk);
            end
            sh_in  <= feed[fp][3:0];
            pred_x <= {feed[fp+2], feed[fp+1]};
            pred_y <= {feed[fp+4], feed[fp+3]};
            pred_h <= feed[fp+5][9:0];
            fp = fp + 6;
            @(posedge clk);
            step_start <= 1;
            @(posedge clk);
            step_start <= 0;
            @(posedge clk);
            while (!step_done) @(posedge clk);
            $display("KF %0d %0d %0d %0d %0d %0d %0d", kf,
                     out_x, out_y, out_h, out_ai, out_score, out_state);
        end
        $finish;
    end
endmodule
