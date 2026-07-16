// v8_shims.v — GATE-ONLY name shims so the UNTOUCHED ice40 testbenches
// (tb_solo_step.v / tb_solo_top.v) drive the v8 modules. Never part of
// a bitstream build (top_solo_ecp5_v2.v instantiates top_solo2
// directly).
`timescale 1ns/1ps

`ifdef V8_STEP_SHIM
// tb_solo_step.v instantiates `solo_tracker` — route it to v8. The
// tracking semantics are BIT-IDENTICAL, so the historical lean
// fixtures must pass verbatim; `novel` dangles.
module solo_tracker #(
    parameter NSEG = 64,
    parameter REFINE = 1
) (
    input  wire        clk,
    input  wire        anc_we,
    input  wire [8:0]  anc_wa,
    input  wire [15:0] anc_wd,
    input  wire [6:0]  n_seg,
    input  wire        step_start,
    input  wire signed [31:0] pred_x,
    input  wire signed [31:0] pred_y,
    input  wire [9:0]  pred_h,
    output wire        step_done,
    output wire signed [31:0] out_x,
    output wire signed [31:0] out_y,
    output wire [9:0]  out_h,
    output wire [5:0]  out_ai,
    output wire signed [31:0] out_score,
    output wire [1:0]  out_state,
    output wire [5:0]  mc_seg,
    output wire        m_start,
    output wire signed [15:0] m_dx,
    output wire signed [15:0] m_dy,
    output wire [5:0]  m_rho,
    output wire [3:0]  m_sh,
    input  wire        m_busy,
    input  wire        m_done,
    output wire [1:0]  s_sel,
    input  wire signed [31:0] s_re,
    input  wire [3:0]  sh_in,
    output wire [7:0]  rd_idx,
    input  wire signed [31:0] rd_re,
    input  wire signed [31:0] rd_im,
    input  wire        svc_start,
    input  wire [1:0]  svc_op,
    input  wire signed [15:0] svc_x,
    input  wire signed [15:0] svc_y,
    input  wire [9:0]  svc_h,
    output wire        svc_done,
    output wire signed [15:0] o_cq,
    output wire signed [15:0] o_sq,
    output wire signed [31:0] o_fx,
    output wire signed [31:0] o_fy
);
    solo_tracker2 #(.NSEG(NSEG), .REFINE(REFINE)) i (
        .clk(clk), .anc_we(anc_we), .anc_wa(anc_wa), .anc_wd(anc_wd),
        .n_seg(n_seg), .step_start(step_start), .pred_x(pred_x),
        .pred_y(pred_y), .pred_h(pred_h), .step_done(step_done),
        .out_x(out_x), .out_y(out_y), .out_h(out_h), .out_ai(out_ai),
        .out_score(out_score), .out_state(out_state), .novel(),
        .mc_seg(mc_seg), .m_start(m_start), .m_dx(m_dx), .m_dy(m_dy),
        .m_rho(m_rho), .m_sh(m_sh), .m_busy(m_busy), .m_done(m_done),
        .s_sel(s_sel), .s_re(s_re), .sh_in(sh_in), .rd_idx(rd_idx),
        .rd_re(rd_re), .rd_im(rd_im), .svc_start(svc_start),
        .svc_op(svc_op), .svc_x(svc_x), .svc_y(svc_y), .svc_h(svc_h),
        .svc_done(svc_done), .o_cq(o_cq), .o_sq(o_sq), .o_fx(o_fx),
        .o_fy(o_fy));
endmodule
`endif

`ifdef V8_TOP_SHIM
// tb_solo_top.v instantiates `top_solo #(.UDIV(), .REFINE())` — route
// to top_solo2 (novelty folding ON, its golden is solo2_gates.py).
module top_solo #(
    parameter UDIV = 12,
    parameter NSEG = 64,
    parameter SEG_KF = 5,
    parameter REFINE = 1
) (
    input  wire clk,
    input  wire RX,
    output wire TX,
    output wire LEDR_N,
    output wire LEDG_N
);
    top_solo2 #(.UDIV(UDIV), .NSEG(NSEG), .SEG_KF(SEG_KF),
                .REFINE(REFINE)) i (
        .clk(clk), .RX(RX), .TX(TX), .LEDR_N(LEDR_N), .LEDG_N(LEDG_N));
endmodule
`endif
