// solo_tracker.v — v7 on-fabric tracker FSM (golden contract =
// ice40/host/solo.py SoloTracker.step, K_TRY=1). SIM-GATED BIT-EXACT:
// 220/220 kf (classroom, incl. holds; with the on-chip shift scan) and
// 160/160 kf on the kidnap fixture (incl. both RELOCKS) — see RESULTS
// 2026-07-12 v7 stages 2/3a/3b + the build-session regate (lean core,
// REFINE=0 fixtures).
//
// Per step: on-chip shift_for (T_SH: OR-of-|Q| bitlen == max bitlen) ->
// NEAREST anchor (d² scan, SATURATING +-32767 u deltas == golden clip) ->
// REL (Q15 irot) -> candidate grid through the matcher core (parametric:
// fine 7x7x5/T=12 with one edge-recenter retry + optional integer
// parabolic refine (parameter REFINE; the lean build drops the divider —
// classroom A/B: localization p90 +2 mm, selfmap med 0.073->0.055), or
// wide re-search 15x15x7/T=48/rho-step-2, argmax-only) -> EMA commit gate
// (29/64, shift-add; every 12th consecutive hold triggers the wide
// re-search, relock gate 35/64) -> pose compose (Q15 rot) -> outputs.
// Arithmetic mirrors solo.py bit-for-bit: floor shifts (>>>), trunc-
// toward-zero division, first-max argmax in rho-major/dx/dy candidate
// order, wrap960 heading with branch select (wide-aware in relock).
//
// Build-session resource contract: ONE 16x16 MAC16 (was a 17x17 multi-DSP
// mult; operands saturate to +-32767 == golden), grid-offset constants by
// shift-add (12v = (v<<3)+(v<<2); 48v = 12v<<2), divider numerator *12
// likewise. EBR: anchor table 2, totram 2 (REFINE=1 only), cs ROM 1.
//
// SE2 SERVICE (v7 top_solo): svc_start with svc_op 0 CS / 1 ROT / 2 IROT
// computes cs_of(svc_h) on the quarter-wave cs ROM (cs_t.hex — the
// TABLE-DEFINED golden cs_of) and optionally R(h)/R(-h) @ (svc_x, svc_y)
// with round-half-up Q15 (rot_q15/irot_q15 contract). The top sequences
// pred-compose, fold-prep and segment-open through this port; the step
// path never touches it.
//
// Anchor table: internal memory, 8x16-bit words per segment
// {x_lo, x_hi, y_lo, y_hi, h, c, s, pad}, loaded via anc_we port.
`timescale 1ns/1ps
module solo_tracker #(
    parameter NSEG = 64,
    parameter REFINE = 1
) (
    input  wire        clk,
    // anchor table load
    input  wire        anc_we,
    input  wire [8:0]  anc_wa,           // {seg[5:0], word[2:0]}
    input  wire [15:0] anc_wd,
    input  wire [6:0]  n_seg,            // live segment count (<= NSEG=64)
    // step
    input  wire        step_start,
    input  wire signed [31:0] pred_x,
    input  wire signed [31:0] pred_y,
    input  wire [9:0]  pred_h,           // 0..959
    output reg         step_done = 0,
    output reg  signed [31:0] out_x,
    output reg  signed [31:0] out_y,
    output reg  [9:0]  out_h,
    output reg  [5:0]  out_ai,
    output reg  signed [31:0] out_score,
    output reg  [1:0]  out_state,        // 0 tracking, 1 hold, 2 relock
    // encoder_solo matcher interface (scan already encoded by the top)
    output reg  [5:0]  mc_seg,
    output reg         m_start = 0,
    output reg  signed [15:0] m_dx,
    output reg  signed [15:0] m_dy,
    output reg  [5:0]  m_rho,
    output reg  [3:0]  m_sh,
    input  wire        m_busy,
    input  wire        m_done,
    output reg  [1:0]  s_sel,
    input  wire signed [31:0] s_re,
    input  wire [3:0]  sh_in,            // debug override (SH_ONCHIP=0)
    // encode accumulator readback (on-chip shift_for scan, stage 3)
    output reg  [7:0]  rd_idx,
    input  wire signed [31:0] rd_re,
    input  wire signed [31:0] rd_im,
    // SE2 service (top_solo): op 0 CS, 1 ROT, 2 IROT
    input  wire        svc_start,
    input  wire [1:0]  svc_op,
    input  wire signed [15:0] svc_x,
    input  wire signed [15:0] svc_y,
    input  wire [9:0]  svc_h,
    output reg         svc_done = 0,
    output reg  signed [15:0] o_cq,
    output reg  signed [15:0] o_sq,
    output reg  signed [31:0] o_fx,
    output reg  signed [31:0] o_fy
);
    parameter SH_ONCHIP = 1;
    // ---------------- anchor memory (EBR): 64 x 8 x 16 ----------------
    reg [15:0] anc [0:8*NSEG-1];
    reg [15:0] anc_q;
    reg [8:0]  anc_ra;
    always @(posedge clk) begin
        if (anc_we) anc[anc_wa] <= anc_wd;
        anc_q <= anc[anc_ra];
    end

    // ---------------- signed multiplier 16x16 -> 32 (ONE MAC16) --------
    // Registered combinational multiply: exact semantics for the stage-2
    // gate. Operands SATURATE at +-32767 at the load sites (sat16), the
    // golden clips identically; in-envelope (all fixtures) the values are
    // untouched. (A serial LC-lean unit was tried and banked as a defect
    // class — OR'd overlapping partials; any replacement must pass the
    // standalone MUL unit test first.)
    reg         mul_go = 0;
    reg  signed [15:0] mul_a, mul_b;
    reg  signed [31:0] mul_p;
    wire        mul_busy = 1'b0;
    always @(posedge clk)
        if (mul_go) mul_p <= mul_a * mul_b;
    function signed [15:0] sat16(input signed [31:0] v);
        sat16 = (v > 32'sd32767) ? 16'sd32767 :
                (v < -32'sd32767) ? -16'sd32767 : v[15:0];
    endfunction

    // ---------------- serial trunc divider (|num|/|den|) ---------------
    reg         div_go = 0;
    reg  signed [39:0] div_n, div_d;
    reg  signed [39:0] div_q;
    reg  [5:0]  div_c = 0;
    reg  [39:0] dnum, dquo;
    reg  [79:0] drem;
    reg         dsign;
    wire        div_busy = (div_c != 0);
    always @(posedge clk) begin
        if (div_go) begin
            dnum  <= div_n[39] ? (~div_n + 1'b1) : div_n;
            drem  <= 0;
            dquo  <= 0;
            dsign <= div_n[39] ^ div_d[39];
            div_c <= 6'd40;
        end else if (div_c != 0) begin : dv
            reg [79:0] r2;
            reg [39:0] dd;
            dd = div_d[39] ? (~div_d + 1'b1) : div_d;
            r2 = {drem[78:0], dnum[39]};
            dnum <= dnum << 1;
            if (r2[39:0] >= dd && r2[79:40] == 0) begin
                drem <= {r2[79:40], r2[39:0] - dd};
                dquo <= {dquo[38:0], 1'b1};
                if (div_c == 1)
                    div_q <= dsign ? (~{dquo[38:0], 1'b1} + 1'b1)
                                   : {dquo[38:0], 1'b1};
            end else begin
                drem <= r2;
                dquo <= {dquo[38:0], 1'b0};
                if (div_c == 1)
                    div_q <= dsign ? (~{dquo[38:0], 1'b0} + 1'b1)
                                   : {dquo[38:0], 1'b0};
            end
            div_c <= div_c - 1;
        end
    end

    // ---------------- tot RAM: 245 x 32 (REFINE=1 builds only) ---------
    reg signed [31:0] totram [0:255];
    reg signed [31:0] tot_q;
    reg [7:0] tot_ra;
    reg       tot_we;
    reg [7:0] tot_wa;
    reg signed [31:0] tot_wd;
    always @(posedge clk) begin
        if (tot_we) totram[tot_wa] <= tot_wd;
        tot_q <= totram[tot_ra];
    end

    // ---------------- cs ROM: quarter-wave heading table ---------------
    // cs_t.hex = round(32767*cos(o*2pi/960)), o in [0,240); Tc[240]=0 is
    // the o==0-on-odd-quadrant special case. Fold == solo.cs_of EXACTLY
    // (the golden is table-defined; verified exhaustively in gen_luts).
    reg signed [15:0] rom_csT [0:255];
    initial $readmemh("build/cs_t.hex", rom_csT);
    reg  [9:0] cs_hh;                     // fold input (registered)
    wire [1:0] cs_qd = (cs_hh >= 10'd720) ? 2'd3 :
                       (cs_hh >= 10'd480) ? 2'd2 :
                       (cs_hh >= 10'd240) ? 2'd1 : 2'd0;
    wire [9:0] cs_o10 = cs_hh - (cs_qd == 2'd3 ? 10'd720 :
                                 cs_qd == 2'd2 ? 10'd480 :
                                 cs_qd == 2'd1 ? 10'd240 : 10'd0);
    wire [7:0] cs_o   = cs_o10[7:0];
    wire [7:0] cs_a   = cs_qd[0] ? (8'd240 - cs_o) : cs_o;
    wire       cs_zero = cs_qd[0] && (cs_o == 8'd0);
    wire       cs_neg  = (cs_qd == 2'd1) || (cs_qd == 2'd2);
    reg signed [15:0] cst_q;
    reg        cs_zero_q, cs_neg_q;
    always @(posedge clk) begin
        cst_q     <= rom_csT[cs_a];
        cs_zero_q <= cs_zero;
        cs_neg_q  <= cs_neg;
    end
    wire signed [15:0] cs_val = cs_zero_q ? 16'sd0
                              : (cs_neg_q ? -cst_q : cst_q);
    wire [9:0] cs_sin_h = (svc_h >= 10'd240) ? (svc_h - 10'd240)
                                             : (svc_h + 10'd720);

    // ---------------- helpers ------------------------------------------
    function [9:0] wrap960(input signed [11:0] v);
        // v in (-960, 960*2): reduce to [0, 960)
        reg signed [11:0] t;
        begin
            t = v;
            if (t < 0) t = t + 12'sd960;
            if (t >= 12'sd960) t = t - 12'sd960;
            wrap960 = t[9:0];
        end
    endfunction
    function signed [10:0] wdist(input signed [11:0] v);
        // wrapped distance to (-480, 480]
        reg signed [11:0] t;
        begin
            t = v;
            if (t < -12'sd480) t = t + 12'sd960;
            if (t > 12'sd480)  t = t - 12'sd960;
            wdist = t[10:0];
        end
    endfunction
    function [5:0] rho60(input signed [7:0] v);
        reg signed [7:0] t;
        begin
            t = v;
            if (t < 0) t = t + 8'sd60;
            if (t >= 8'sd60) t = t - 8'sd60;
            rho60 = t[5:0];
        end
    endfunction

    // ---------------- state --------------------------------------------
    localparam T_IDLE = 0, T_NR0 = 1, T_NR1 = 2, T_NR2 = 3, T_NR3 = 4,
               T_NR4 = 5, T_AR0 = 6, T_AR1 = 7, T_RELA = 8, T_RELB = 9,
               T_RELC = 10, T_GSET = 11, T_GISS = 12, T_GWT = 13,
               T_GTOT = 14, T_GAX = 15, T_EDGE = 16, T_PN0 = 17,
               T_PN1 = 18, T_PX = 19, T_PY = 20, T_EMA = 21, T_PC0 = 22,
               T_PC1 = 23, T_HSEL = 24, T_OUT = 25, T_SH = 26, T_RSF = 27,
               T_SV0 = 28, T_SV1 = 29, T_SV2 = 30, T_SV3 = 31, T_SV4 = 32;
    reg [5:0] tst = T_IDLE;
    // on-chip shift_for scan state
    reg [31:0] shmax;
    reg [8:0]  shi;
    reg [1:0]  shk;
    reg [5:0]  shj;
    wire [3:0] shenc =
        shmax[30] ? 4'd15 : shmax[29] ? 4'd15 :
        shmax[28] ? 4'd14 : shmax[27] ? 4'd13 : shmax[26] ? 4'd12 :
        shmax[25] ? 4'd11 : shmax[24] ? 4'd10 : shmax[23] ? 4'd9  :
        shmax[22] ? 4'd8  : shmax[21] ? 4'd7  : shmax[20] ? 4'd6  :
        shmax[19] ? 4'd5  : shmax[18] ? 4'd4  : shmax[17] ? 4'd3  :
        shmax[16] ? 4'd2  : shmax[15] ? 4'd1  : 4'd0;

    reg signed [31:0] px_q, py_q;
    reg [9:0]  ph_q;
    reg [5:0]  si;                     // nearest scan index
    reg [5:0]  ai;                     // chosen segment
    reg signed [31:0] ax, ay;          // anchor fields
    reg [9:0]  ah;
    reg signed [15:0] acq, asq;
    reg signed [31:0] dx32, dy32;
    reg signed [33:0] d2, best_d2;
    reg [1:0]  nph;                    // nearest sub-phase
    reg signed [31:0] relx, rely;      // rel accum
    reg signed [15:0] tx_u, ty_u;
    reg signed [15:0] d0x, d0y;
    reg [5:0]  rho0, rho0_sv;          // rho0_sv: pre-retry (re-search seed)
    reg [3:0]  gi, gj;                 // dx, dy idx (0..6 fine, 0..14 wide)
    reg [2:0]  gr;                     // rho idx (0..4 fine, 0..6 wide)
    reg [10:0] cand;
    reg        retry;
    // parametric grid (fine sweep vs wide re-search)
    reg        wide;                   // 1: re-search (argmax only)
    wire [3:0]  g_NN  = wide ? 4'd14 : 4'd6;    // 2N
    wire [2:0]  g_RR  = wide ? 3'd6 : 3'd4;     // rho idx max
    wire [3:0]  g_N   = wide ? 4'd7 : 4'd3;
    wire signed [7:0] g_Roff = wide ? 8'sd6 : 8'sd2;
    reg [1:0]  tph;
    reg signed [31:0] best_tot;
    reg [7:0]  best_c;
    reg [2:0]  bir;
    reg [3:0]  bix, biy;   // wide grid indexes to 14
    reg        have_best;
    reg signed [31:0] pa, pc;
    reg [2:0]  pph;
    reg signed [15:0] pxo, pyo;        // parabolic offsets (post-div)
    reg signed [15:0] fpx, fpy;        // final peak d (grid + refine)
    reg        ema_init = 1;
    reg signed [31:0] ema;
    reg [3:0]  lost12;                 // consecutive-hold counter mod 12
    reg        rlk;                    // relock pose in flight
    reg signed [31:0] posx, posy;
    reg [1:0]  mph;
    reg signed [11:0] hbase;
    reg [2:0]  hbi;
    reg signed [10:0] hbest_d;
    reg signed [11:0] hbest_v;

    wire signed [31:0] ema_val = ema_init ? best_tot : ema;
    wire signed [31:0] ema6 = ema_val >>> 6;
    wire signed [31:0] th29 = (ema6 <<< 5) - (ema6 <<< 1) - ema6;
    wire signed [31:0] th35 = (ema6 <<< 5) + (ema6 <<< 1) + ema6;

    // grid offset by shift-add (no DSP): 12*(idx-N) fine / 48*(idx-N) wide
    function signed [15:0] gmul(input [3:0] idx);
        reg signed [5:0] v;
        reg signed [15:0] t12;
        begin
            v = $signed({2'b0, idx}) - $signed({2'b0, g_N});
            t12 = (v <<< 3) + (v <<< 2);
            gmul = wide ? (t12 <<< 2) : t12;
        end
    endfunction
    // SHARED instances (hand-CSE: function calls inline a fresh compare/
    // mux tree per site — one net per operand instead)
    wire signed [15:0] gmx = gmul((tst == T_GISS) ? gi : bix);
    wire signed [15:0] gmy = gmul((tst == T_GISS) ? gj : biy);
    wire signed [15:0] dxs16 = sat16(dx32);
    wire signed [15:0] dys16 = sat16(dy32);
    // rho index scaled by g_rs (1 fine / 2 wide) — shift, no DSP
    wire [3:0] grs  = wide ? {gr, 1'b0}  : {1'b0, gr};
    wire [3:0] birs = wide ? {bir, 1'b0} : {1'b0, bir};
    // parabolic divider numerator: (pa - pc) * 12 by shift-add (shared)
    wire signed [39:0] d40s = {{6{pa[31]}}, pa} - {{6{pc[31]}}, pc};
    wire signed [39:0] div12 = (d40s <<< 3) + (d40s <<< 2);
    // SE2 service datapath regs
    reg signed [31:0] sv_rx, sv_ry;
    reg [1:0]  svph;

    always @(posedge clk) begin
        m_start <= 1'b0;
        mul_go <= 1'b0;
        div_go <= 1'b0;
        tot_we <= 1'b0;
        step_done <= 1'b0;
        svc_done <= 1'b0;
        case (tst)
        T_IDLE: if (step_start) begin
            px_q <= pred_x; py_q <= pred_y; ph_q <= pred_h;
            si <= 0; best_d2 <= 34'sh1FFFFFFFF;
            anc_ra <= 0; nph <= 0;
            m_sh <= sh_in;               // overwritten by T_SH when ONCHIP
            retry <= 0; wide <= 0; rlk <= 0;
            shmax <= 32'd0; shi <= 0; shk <= 0; shj <= 0; rd_idx <= 8'd0;
            tst <= (SH_ONCHIP != 0) ? T_SH : T_NR0;
        end else if (svc_start) begin
            cs_hh <= svc_h;              // cos fold; ROM read lands T_SV0
            tst <= T_SV0;
        end
        // ---- SE2 service: CS lookups (quarter-wave fold), then Q15
        // rot/irot via the shared multiplier (round-half-up +2^14 >>> 15)
        T_SV0: begin
            cs_hh <= cs_sin_h;           // sin fold; cos T lands this edge
            tst <= T_SV1;
        end
        T_SV1: begin
            o_cq <= cs_val;              // cos value (flags from T_IDLE reg)
            tst <= T_SV2;
        end
        T_SV2: begin
            o_sq <= cs_val;              // sin value
            mph <= 0;
            tst <= (svc_op == 2'd0) ? T_SV4 : T_SV3;
        end
        T_SV3: begin                     // 4 mults: cq*x, sq*y, sq*x, cq*y
            if (mph == 0) begin
                mul_a <= o_cq; mul_b <= svc_x;
                mul_go <= 1'b1; mph <= 1;
            end else if (mph == 1 && !mul_go && !mul_busy) begin
                sv_rx <= mul_p;
                mul_a <= o_sq; mul_b <= svc_y;
                mul_go <= 1'b1; mph <= 2;
            end else if (mph == 2 && !mul_go && !mul_busy) begin
                // ROT: rx = cq*x - sq*y ; IROT: rx = cq*x + sq*y
                sv_rx <= (svc_op == 2'd1) ? (sv_rx - mul_p)
                                          : (sv_rx + mul_p);
                mul_a <= o_sq; mul_b <= svc_x;
                mul_go <= 1'b1; mph <= 3;
            end else if (mph == 3 && !mul_go && !mul_busy) begin
                // ROT: ry = sq*x + cq*y ; IROT: ry = -sq*x + cq*y
                sv_ry <= (svc_op == 2'd1) ? mul_p : -mul_p;
                mul_a <= o_cq; mul_b <= svc_y;
                mul_go <= 1'b1; mph <= 0;
                tst <= T_SV4;
            end
        end
        T_SV4: if (!mul_go && !mul_busy) begin
            if (svc_op != 2'd0) begin
                o_fx <= (sv_rx + 32'sd16384) >>> 15;
                o_fy <= ((sv_ry + mul_p) + 32'sd16384) >>> 15;
            end
            svc_done <= 1'b1;
            tst <= T_IDLE;
        end
        // ---- on-chip shift_for: OR of |Q| has the same bit-length as
        // max|Q|, so sh = max(0, bitlen(OR)-15) exactly. Readback is a
        // registered 2-stage path: idx issued at n is readable at n+2.
        T_SH: begin
            rd_idx <= {shk, shj};
            if (shj == 6'd59) begin shj <= 0; shk <= shk + 1; end
            else shj <= shj + 1;
            if (shi >= 9'd2 && shi <= 9'd241)
                shmax <= shmax
                         | (rd_re[31] ? (~rd_re + 1'b1) : rd_re)
                         | (rd_im[31] ? (~rd_im + 1'b1) : rd_im);
            shi <= shi + 1;
            if (shi == 9'd242) begin
                m_sh <= shenc;
                tst <= T_NR0;
            end
        end
        // ---- nearest: per segment read x,y (words 0..3), 2 serial mults
        T_NR0: begin anc_ra <= {si, 3'd0}; tst <= T_NR1; nph <= 0; end
        T_NR1: begin                     // pipeline the 4 reads
            anc_ra <= anc_ra + 1;
            nph <= nph + 1;
            case (nph)
            2'd1: ax[15:0]  <= anc_q;
            2'd2: ax[31:16] <= anc_q;
            2'd3: ay[15:0]  <= anc_q;
            endcase
            if (nph == 2'd3) tst <= T_NR2;
        end
        T_NR2: begin
            ay[31:16] <= anc_q;
            tst <= T_NR3;
        end
        T_NR3: begin                     // dx² via serial mult
            dx32 <= px_q - ax;
            dy32 <= py_q - ay;
            tst <= T_NR4; nph <= 0;
        end
        T_NR4: begin
            if (nph == 0) begin
                mul_a <= dxs16; mul_b <= dxs16;
                mul_go <= 1'b1; nph <= 1;
            end else if (nph == 1 && !mul_go && !mul_busy) begin
                d2 <= mul_p;
                mul_a <= dys16; mul_b <= dys16;
                mul_go <= 1'b1; nph <= 2;
            end else if (nph == 2 && !mul_go && !mul_busy) begin
                if (d2 + mul_p < best_d2) begin
                    best_d2 <= d2 + mul_p;
                    ai <= si;
                end
                if ({1'b0, si} == n_seg - 7'd1) begin
                    tst <= T_AR0;
                end else begin
                    si <= si + 1;
                    tst <= T_NR0;
                end
            end
        end
        // ---- load chosen anchor record fully
        T_AR0: begin anc_ra <= {ai, 3'd0}; nph <= 0; tst <= T_AR1; end
        T_AR1: begin
            anc_ra <= anc_ra + 1;
            nph <= nph + 1;
            case (nph)
            2'd1: ax[15:0]  <= anc_q;
            2'd2: ax[31:16] <= anc_q;
            2'd3: ay[15:0]  <= anc_q;
            endcase
            if (nph == 2'd3) begin nph <= 0; tst <= T_RELA; end
        end
        T_RELA: begin                    // continue record: y_hi, h, c, s
            anc_ra <= anc_ra + 1;
            nph <= nph + 1;
            case (nph)
            2'd0: ay[31:16] <= anc_q;
            2'd1: ah  <= anc_q[9:0];
            2'd2: acq <= $signed(anc_q);
            2'd3: asq <= $signed(anc_q);
            endcase
            if (nph == 2'd3) begin
                dx32 <= px_q - ax;
                dy32 <= py_q - ay;
                nph <= 0; mph <= 0;
                tst <= T_RELB;
            end
        end
        T_RELB: begin                    // irot: tx=(c*dx+s*dy+2^14)>>>15
            if (mph == 0) begin
                mul_a <= acq; mul_b <= dxs16;
                mul_go <= 1'b1; mph <= 1;
            end else if (mph == 1 && !mul_go && !mul_busy) begin
                relx <= mul_p;
                mul_a <= asq; mul_b <= dys16;
                mul_go <= 1'b1; mph <= 2;
            end else if (mph == 2 && !mul_go && !mul_busy) begin
                relx <= relx + mul_p;
                mul_a <= asq; mul_b <= dxs16;
                mul_go <= 1'b1; mph <= 3;
            end else if (mph == 3 && !mul_go && !mul_busy) begin
                rely <= -mul_p;
                mul_a <= acq; mul_b <= dys16;
                mul_go <= 1'b1; mph <= 0;
                tst <= T_RELC;
            end
        end
        T_RELC: if (!mul_go && !mul_busy) begin
            tx_u <= (relx + 32'sd16384) >>> 15;
            ty_u <= ((rely + mul_p) + 32'sd16384) >>> 15;
            tst <= T_GSET;
        end
        T_GSET: begin : gset
            reg signed [11:0] dh;
            reg signed [7:0]  r6;
            d0x <= -tx_u;
            d0y <= -ty_u;
            dh = wdist({2'b0, ph_q} - {2'b0, ah});
            r6 = (dh + 12'sd4) >>> 3;
            rho0 <= rho60(r6);
            rho0_sv <= rho60(r6);        // re-search seed (pre-retry)
            gi <= 0; gj <= 0; gr <= 0; cand <= 0;
            have_best <= 0; best_tot <= 32'sh80000000;
            tst <= T_GISS;
        end
        // ---- grid: rho-major, dx, dy (matches solo.py cands order);
        // parametric: fine sweep (T=12, 7x7x5) or wide re-search
        // (T=48, 15x15x7, rho step 2, argmax only)
        T_GISS: begin
            mc_seg <= ai;
            m_dx  <= d0x + gmx;
            m_dy  <= d0y + gmy;
            m_rho <= rho60($signed({2'b0, rho0})
                           + $signed({4'b0, grs}) - g_Roff);
            m_start <= 1'b1;
            tst <= T_GWT;
        end
        T_GWT: if (!m_start && !m_busy && m_done) begin
            s_sel <= 0; tph <= 0;
            tot_wd <= 32'sd0;
            tst <= T_GTOT;
        end
        T_GTOT: begin                    // sum 4 ring Re parts
            tot_wd <= tot_wd + s_re;
            s_sel <= s_sel + 1;
            tph <= tph + 1;
            if (tph == 2'd3) begin
                tot_we <= !wide && (REFINE != 0);  // refine neighbors only
                tot_wa <= cand[7:0];
                tst <= T_GAX;
            end
        end
        T_GAX: begin
            if (!have_best || tot_wd > best_tot) begin
                best_tot <= tot_wd;
                best_c <= cand[7:0];
                bir <= gr; bix <= gi; biy <= gj;
                have_best <= 1;
            end
            cand <= cand + 1;
            if (gj != g_NN) begin gj <= gj + 1; tst <= T_GISS; end
            else if (gi != g_NN) begin gj <= 0; gi <= gi + 1; tst <= T_GISS; end
            else if (gr != g_RR) begin gj <= 0; gi <= 0; gr <= gr + 1; tst <= T_GISS; end
            else tst <= T_EDGE;
        end
        T_EDGE: begin
            if (wide) begin              // re-search: grid-commit, no
                fpx <= d0x + gmx;        // refine
                fpy <= d0y + gmy;
                tst <= T_RSF;
            end else
            if (!retry && (bix == 0 || bix == g_NN
                           || biy == 0 || biy == g_NN)) begin
                retry <= 1;
                d0x <= d0x + gmx;
                d0y <= d0y + gmy;
                rho0 <= rho60($signed({2'b0, rho0})
                              + $signed({1'b0, bir}) - 8'sd2);
                gi <= 0; gj <= 0; gr <= 0; cand <= 0;
                have_best <= 0; best_tot <= 32'sh80000000;
                tst <= T_GISS;
            end else begin
                fpx <= d0x + gmx;
                fpy <= d0y + gmy;
                pxo <= 0; pyo <= 0; pph <= 0;
                tst <= (REFINE == 0) ? T_EMA
                       : ((bix > 0 && bix < g_NN) ? T_PN0 : T_PN1);
            end
        end
        T_RSF: begin                     // relock gate: 35/64 of EMA
            if (best_tot > th35) begin
                rlk <= 1'b1;
                lost12 <= 0;
                dx32 <= -{{16{fpx[15]}}, fpx};
                dy32 <= -{{16{fpy[15]}}, fpy};
                mph <= 0;
                tst <= T_PC0;            // pose compose, state 2 at HSEL
            end else begin
                tst <= T_OUT;            // hold outputs already set
            end
        end
        // ---- parabolic X: a = tot[c-7], c = tot[c+7]
        // (tot_q is a REGISTERED ram read: address at phase n is readable
        // at phase n+2 — the pc capture needs its own wait phase)
        T_PN0: begin
            case (pph)
            3'd0: begin tot_ra <= best_c - 8'd7; pph <= 1; end
            3'd1: pph <= 2;
            3'd2: begin pa <= tot_q; tot_ra <= best_c + 8'd7; pph <= 3; end
            3'd3: pph <= 4;
            3'd4: begin
                pc <= tot_q;
                pph <= 0;
                tst <= T_PX;
            end
            endcase
        end
        T_PX: begin : ppx
            reg signed [33:0] den;
            den = {pa[31], pa[31], pa} - {best_tot[31], best_tot[31],
                   best_tot} - {best_tot[31], best_tot[31], best_tot}
                  + {pc[31], pc[31], pc};
            if (den < 0 && pph == 0) begin
                div_n <= div12;                          // *12, no DSP
                div_d <= {den[33], den[33], den[33], den[33], den[33],
                          den[33], den} <<< 1;
                div_go <= 1'b1; pph <= 1;
            end else if (pph == 1 && !div_go && !div_busy) begin
                pxo <= (div_q > 40'sd6) ? 16'sd6 :
                       ((div_q < -40'sd6) ? -16'sd6 : div_q[15:0]);
                pph <= 0;
                tst <= T_PN1;
            end else if (pph == 0) begin
                tst <= T_PN1;             // den >= 0: no refine
            end
        end
        T_PN1: begin
            if (biy > 0 && biy < 3'd6) begin
                case (pph)
                3'd0: begin tot_ra <= best_c - 8'd1; pph <= 1; end
                3'd1: pph <= 2;
                3'd2: begin pa <= tot_q; tot_ra <= best_c + 8'd1; pph <= 3; end
                3'd3: pph <= 4;
                3'd4: begin pc <= tot_q; pph <= 0; tst <= T_PY; end
                endcase
            end else tst <= T_EMA;
        end
        T_PY: begin : ppy
            reg signed [33:0] den;
            den = {pa[31], pa[31], pa} - {best_tot[31], best_tot[31],
                   best_tot} - {best_tot[31], best_tot[31], best_tot}
                  + {pc[31], pc[31], pc};
            if (den < 0 && pph == 0) begin
                div_n <= div12;                          // *12, no DSP
                div_d <= {den[33], den[33], den[33], den[33], den[33],
                          den[33], den} <<< 1;
                div_go <= 1'b1; pph <= 1;
            end else if (pph == 1 && !div_go && !div_busy) begin
                pyo <= (div_q > 40'sd6) ? 16'sd6 :
                       ((div_q < -40'sd6) ? -16'sd6 : div_q[15:0]);
                pph <= 0;
                tst <= T_EMA;
            end else if (pph == 0) begin
                tst <= T_EMA;
            end
        end
        T_EMA: begin
            rlk <= 1'b0;
            if (best_tot > th29) begin
                ema <= ema_val + ((best_tot - ema_val) >>> 6);
                ema_init <= 0;
                lost12 <= 0;
                out_state <= 2'd0;
                dx32 <= -(fpx + pxo);     // t_est for pose compose
                dy32 <= -(fpy + pyo);
                mph <= 0;
                tst <= T_PC0;
            end else begin
                if (ema_init) begin ema <= best_tot; ema_init <= 0; end
                out_state <= 2'd1;
                out_x <= px_q; out_y <= py_q; out_h <= ph_q;
                out_ai <= ai; out_score <= best_tot;
                if (lost12 == 4'd11) begin        // every 12th hold:
                    lost12 <= 0;                  // wide re-search
                    wide <= 1'b1;
                    d0x <= -tx_u; d0y <= -ty_u;   // pred-seeded (python
                    rho0 <= rho0_sv;              // _research semantics)
                    gi <= 0; gj <= 0; gr <= 0; cand <= 0;
                    have_best <= 0; best_tot <= 32'sh80000000;
                    tst <= T_GISS;
                end else begin
                    lost12 <= lost12 + 1;
                    tst <= T_OUT;
                end
            end
        end
        // ---- pose compose: R(ah) @ t_est + anchor
        T_PC0: begin
            if (mph == 0) begin
                mul_a <= acq; mul_b <= dxs16;
                mul_go <= 1'b1; mph <= 1;
            end else if (mph == 1 && !mul_go && !mul_busy) begin
                relx <= mul_p;
                mul_a <= asq; mul_b <= dys16;
                mul_go <= 1'b1; mph <= 2;
            end else if (mph == 2 && !mul_go && !mul_busy) begin
                relx <= relx - mul_p;
                mul_a <= asq; mul_b <= dxs16;
                mul_go <= 1'b1; mph <= 3;
            end else if (mph == 3 && !mul_go && !mul_busy) begin
                rely <= mul_p;
                mul_a <= acq; mul_b <= dys16;
                mul_go <= 1'b1; mph <= 0;
                tst <= T_PC1;
            end
        end
        T_PC1: if (!mul_go && !mul_busy) begin
            posx <= ax + ((relx + 32'sd16384) >>> 15);
            posy <= ay + (((rely + mul_p) + 32'sd16384) >>> 15);
            hbase <= {2'b0, ah}
                     + ($signed({3'b0,
                        rho60($signed({2'b0, rho0})
                              + $signed({4'b0, birs})
                              - g_Roff)}) <<< 3);   // wide-aware (relock)
            hbi <= 0; hbest_d <= 11'sd511;
            tst <= T_HSEL;
        end
        T_HSEL: begin : hsel              // branch select: min |wrap(h-pred)|
            reg signed [12:0] hv;
            reg signed [10:0] hd;
            case (hbi)
            3'd0: hv = {hbase[11], hbase} - 13'sd960;
            3'd1: hv = {hbase[11], hbase} - 13'sd480;
            3'd2: hv = {hbase[11], hbase};
            3'd3: hv = {hbase[11], hbase} + 13'sd480;
            default: hv = {hbase[11], hbase} + 13'sd960;
            endcase
            hd = wdist(hv[11:0] - {2'b0, ph_q});
            if ((hd < 0 ? -hd : hd) < hbest_d) begin
                hbest_d <= (hd < 0 ? -hd : hd);
                hbest_v <= hv[11:0];
            end
            hbi <= hbi + 1;
            if (hbi == 3'd4) begin
                out_x <= posx; out_y <= posy;
                out_h <= wrap960(hbest_v);
                out_ai <= ai; out_score <= best_tot;
                out_state <= rlk ? 2'd2 : 2'd0;
                tst <= T_OUT;
            end
        end
        T_OUT: begin
            step_done <= 1'b1;
            tst <= T_IDLE;
        end
        default: tst <= T_IDLE;
        endcase
    end
endmodule
