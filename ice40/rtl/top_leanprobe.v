// top_leanprobe.v — SYNTH/TIMING PROBE for the v7 lean core alone (fmax
// of the serial-ring datapath; not a deploy top). RX shift chain feeds
// inputs; outputs parity-fold into TX.
module top (
    input  wire CLK,
    input  wire RX,
    output wire TX,
    output wire LEDR_N,
    output wire LEDG_N
);
    reg [63:0] sh = 0;
    always @(posedge CLK) sh <= {sh[62:0], RX};

    wire busy, m_done;
    wire signed [31:0] rd_re, rd_im, s_re, s_im;
    wire [15:0] mc_q;
    encoder enc (.clk(CLK), .clear(sh[0]), .start(sh[1]),
                 .az(sh[11:2]), .r_mm(sh[27:12]), .w(sh[35:28]),
                 .busy(busy),
                 .rd_idx(sh[43:36]), .rd_re(rd_re), .rd_im(rd_im),
                 .mc_we(sh[44]), .mc_addr(sh[52:45]), .mc_code(sh[54:53]),
                 .mc_seg(sh[60:55]),
                 .mc_rsel(sh[61]), .mc_ra(sh[13:0]), .mc_q(mc_q),
                 .m_start(sh[62]), .m_dx(sh[15:0]), .m_dy(sh[31:16]),
                 .m_rho(sh[37:32]), .m_sh(sh[41:38]), .m_done(m_done),
                 .f_en(sh[63]), .f_cq(sh[15:0]), .f_sq(sh[31:16]),
                 .f_tx(sh[47:32]), .f_ty(sh[63:48]),
                 .s_sel(sh[1:0]), .s_re(s_re), .s_im(s_im));
    assign TX = ^{busy, m_done, rd_re, rd_im, s_re, s_im, mc_q};
    assign LEDR_N = ~busy;
    assign LEDG_N = ~m_done;
endmodule
