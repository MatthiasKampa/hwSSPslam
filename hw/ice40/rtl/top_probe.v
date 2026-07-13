// top_probe.v — BASELINE SYNTH PROBE for the v7 budget (not a deploy top).
// Instantiates solo_tracker + encoder (encoder_solo.v) wired exactly as
// tb_solo_step, with control inputs fed from an RX shift chain and all
// outputs parity-folded into TX so nothing optimizes away. Purpose:
// LC/EBR/SPRAM/DSP counts + a first fmax read before top_solo is written.
module top (
    input  wire CLK,
    input  wire RX,
    output wire TX,
    output wire LEDR_N,
    output wire LEDG_N
);
    // input shift chain: RX sampled every cycle (synth probe only)
    reg [120:0] sh = 0;
    always @(posedge CLK) sh <= {sh[119:0], RX};

    wire        clear      = sh[0];
    wire        start      = sh[1];
    wire [9:0]  az         = sh[11:2];
    wire [15:0] r_mm       = sh[27:12];
    wire [7:0]  w          = sh[35:28];
    wire        pre_we     = sh[36];
    wire [7:0]  pre_addr   = sh[44:37];
    wire [1:0]  pre_code   = sh[46:45];
    wire [5:0]  pre_seg    = sh[52:47];
    wire        anc_we     = sh[53];
    wire [8:0]  anc_wa     = sh[62:54];
    wire [15:0] anc_wd     = sh[78:63];
    wire [5:0]  n_seg      = sh[84:79];
    wire        step_start = sh[85];
    wire signed [31:0] pred_x = {sh[100:86], 17'b0};
    wire signed [31:0] pred_y = {sh[115:101], 17'b0};
    wire [9:0]  pred_h     = sh[110:101];
    wire [3:0]  sh_in      = sh[120:117];

    wire busy, m_done, step_done;
    wire [1:0] s_sel;
    wire signed [31:0] s_re, s_im;
    wire signed [31:0] out_x, out_y, out_score;
    wire [9:0]  out_h;
    wire [5:0]  out_ai;
    wire [1:0]  out_state;
    wire [5:0]  trk_seg;
    wire        trk_mstart;
    wire signed [15:0] trk_dx, trk_dy;
    wire [5:0]  trk_rho;
    wire [3:0]  trk_sh;
    wire [7:0]  rd_idx;
    wire signed [31:0] rd_re, rd_im;

    encoder enc (.clk(CLK), .clear(clear), .start(start), .az(az),
                 .r_mm(r_mm), .w(w), .busy(busy), .rd_idx(rd_idx),
                 .rd_re(rd_re), .rd_im(rd_im),
                 .mc_we(pre_we),
                 .mc_addr(pre_addr),
                 .mc_code(pre_code),
                 .mc_seg(pre_we ? pre_seg : trk_seg),
                 .m_start(trk_mstart), .m_dx(trk_dx), .m_dy(trk_dy),
                 .m_rho(trk_rho), .m_sh(trk_sh), .m_done(m_done),
                 .s_sel(s_sel), .s_re(s_re), .s_im(s_im));

    solo_tracker trk (.clk(CLK),
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
                 .rd_idx(rd_idx), .rd_re(rd_re), .rd_im(rd_im));

    assign TX = ^{out_x, out_y, out_h, out_ai, out_score, out_state,
                  step_done, busy, m_done, s_im, rd_im};
    assign LEDR_N = ~step_done;
    assign LEDG_N = ~busy;
endmodule
