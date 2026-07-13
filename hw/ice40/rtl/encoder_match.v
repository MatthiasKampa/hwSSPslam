// v4: encoder_pipe.v + on-fabric MATCHER (corner F). Encode path is
// UNCHANGED (same golden contract and schedule as v3; the match-mode
// gates are enable-muxes that free-run in encode mode — re-proven by
// sim + hw-replay). New capability, golden: ssp_ice40.match_int:
//   load 240 2-bit QPSK map codes (the deploy store), then per candidate
//   (dx, dy, rho, sh) compute 4 per-ring complex partial sums
//     s[k] = sum_j conj(rot(Q>>sh, rho))[k,j] * i^mc[k,j] * cis(u_kj(dx,dy))
//   where Q is the LAST ENCODED SCAN sitting in the accumulator banks
//   (banks are read-only in match mode: one encode serves many candidates).
// The translation phases reuse the encoder's exact proj->cis datapath
// (x,y regs carry dx,dy; same ang ROMs, same cis EBRs); rotation is index
// arithmetic with conjugate wrap; H = conj(Q)*i^mc needs NO multiplier
// (sign/swap); the 4 real products/ring go to the 4 free MAC16s.
//
// Match schedule: 8 cycles/angle, sequential (no cross-angle overlap):
//   ph0: ang_a <= j; mr_a <= j' = wrap(j+rho); wflag <= (j+rho >= 60)
//   ph1: (ang ROM read in flight)
//   ph2: u_q <= proj(j) gate; are_q <= bank[j'] gate
//   ph3..6: stage ring (ph+1)&3: qre_s/qim_s <= conj?(Q>>sh), mc_q
//   ph4..7: consume ring ph&3 (4 MAC16s on staged operands) -> p*_q
//   ph5..7,0': accumulate p*_q into sacc[ring]  (+1 drain state)
// ~484 cycles/candidate = 13.5 us @ 36 MHz (~74k candidate poses/s).
module encoder #(
    parameter NA = 60,
    parameter NR = 4
) (
    input  wire        clk,
    input  wire        clear,
    input  wire        start,
    input  wire [9:0]  az,
    input  wire [15:0] r_mm,
    input  wire [7:0]  w,
    output wire        busy,
    input  wire [7:0]  rd_idx,             // readback addr {ring[1:0], j[5:0]}
    output reg  signed [31:0] rd_re,
    output reg  signed [31:0] rd_im,
    // ---- matcher ----
    input  wire        mc_we,              // M-code store write port
    input  wire [7:0]  mc_addr,            //   {ring[1:0], j[5:0]}
    input  wire [1:0]  mc_code,
    input  wire        m_start,            // latch candidate, run match
    input  wire signed [15:0] m_dx,        // translation in U units
    input  wire signed [15:0] m_dy,
    input  wire [5:0]  m_rho,              // grid rotation steps 0..NA-1
    input  wire [3:0]  m_sh,               // per-scan Q pre-shift
    output reg         m_done = 0,         // sticky until next cmd
    input  wire [1:0]  s_sel,              // score readback ring select
    output wire signed [31:0] s_re,
    output wire signed [31:0] s_im
);
    localparam signed HALF = 1 << 13;      // F_AZ = F_ANG = 14

    // ---- ROMs (images from gen_luts.py == golden model) ----
    reg signed [15:0] rom_azc [0:1023];
    reg signed [15:0] rom_azs [0:1023];
    reg signed [15:0] rom_angc[0:63];
    reg signed [15:0] rom_angs[0:63];
    (* ram_style = "block" *)
    reg        [15:0] rom_cis [0:255];      // {im[7:0], re[7:0]} packed
    initial begin
        $readmemh("build/az_c.hex",  rom_azc);
        $readmemh("build/az_s.hex",  rom_azs);
        $readmemh("build/ang_c.hex", rom_angc);
        $readmemh("build/ang_s.hex", rom_angs);
        $readmemh("build/cis.hex",   rom_cis);
    end

    // FSM state (early: gates below reference it)
    localparam S_IDLE = 0, S_CLR = 1, S_AZ = 2, S_XY = 3, S_L = 4,
               S_M = 5, S_MD = 6;
    reg [2:0] st = S_IDLE;
    wire m_run = (st == S_M) || (st == S_MD);
    reg [2:0] ph = 0;                       // match phase within an angle
    reg [5:0] mj = 0;                       // match angle j

    reg signed [15:0] azc_q, azs_q, angc_q, angs_q;
    reg [5:0] ang_a;
    always @(posedge clk) begin
        azc_q  <= rom_azc[az];
        azs_q  <= rom_azs[az];
        angc_q <= rom_angc[ang_a];
        angs_q <= rom_angs[ang_a];
    end

    // projection register (pipe stage 1); in match mode x,y carry dx,dy
    // and u_q updates only at ph2 (operands stay stable per angle)
    reg signed [15:0] x, y;
    wire signed [32:0] um_c = x * angc_q + y * angs_q;
    reg signed [18:0] u_q;
    always @(posedge clk)
        if (!m_run || ph == 3'd2) u_q <= (um_c + HALF) >>> 14;

    // per-ring cis lookups; match mode latches all 4 rings at ph3 and
    // holds them for the rest of the angle
    reg signed [7:0] cre_q [0:NR-1];
    reg signed [7:0] cim_q [0:NR-1];
    genvar gk;
    generate for (gk = 0; gk < NR; gk = gk + 1) begin : g_cis
        wire signed [18:0] ush = u_q >>> gk;
        always @(posedge clk) if (!m_run || ph == 3'd3) begin
            cre_q[gk] <= $signed(rom_cis[ush[7:0]][7:0]);
            cim_q[gk] <= $signed(rom_cis[ush[7:0]][15:8]);
        end
    end endgenerate

    // ---- banked accumulators: per ring, 64 x 32 (re, im) ----
    wire [1:0] rd_k = rd_idx[7:6];
    wire [5:0] rd_j = rd_idx[5:0];
    reg [5:0] acc_a;
    reg [5:0] wr_a;
    reg       acc_we;
    reg       clr_we;
    reg [5:0] clr_a;
    reg signed [7:0] w_q;
    wire signed [31:0] rb_re_w [0:NR-1];
    wire signed [31:0] rb_im_w [0:NR-1];
    reg [5:0] mr_a;                         // match read addr j' (wrapped)
    wire [5:0] r_addr = m_run ? mr_a : ((st != S_IDLE) ? acc_a : rd_j);
    wire [5:0] w_addr = wr_a;
    wire       w_en   = acc_we | clr_we;
    generate for (gk = 0; gk < NR; gk = gk + 1) begin : g_acc
        reg signed [31:0] bank_re [0:63];
        reg signed [31:0] bank_im [0:63];
        reg signed [31:0] are_q, aim_q;
        wire signed [15:0] preA =
            (w_q[0] ? {{8{cre_q[gk][7]}}, cre_q[gk]}        : 16'sd0) +
            (w_q[1] ? {{7{cre_q[gk][7]}}, cre_q[gk], 1'b0}  : 16'sd0) +
            (w_q[2] ? {{6{cre_q[gk][7]}}, cre_q[gk], 2'b0}  : 16'sd0) +
            (w_q[3] ? {{5{cre_q[gk][7]}}, cre_q[gk], 3'b0}  : 16'sd0);
        wire signed [15:0] preB =
            (w_q[4] ? {{4{cre_q[gk][7]}}, cre_q[gk], 4'b0}  : 16'sd0) +
            (w_q[5] ? {{3{cre_q[gk][7]}}, cre_q[gk], 5'b0}  : 16'sd0) +
            (w_q[6] ? {{2{cre_q[gk][7]}}, cre_q[gk], 6'b0}  : 16'sd0);
        wire signed [15:0] pimA =
            (w_q[0] ? {{8{cim_q[gk][7]}}, cim_q[gk]}        : 16'sd0) +
            (w_q[1] ? {{7{cim_q[gk][7]}}, cim_q[gk], 1'b0}  : 16'sd0) +
            (w_q[2] ? {{6{cim_q[gk][7]}}, cim_q[gk], 2'b0}  : 16'sd0) +
            (w_q[3] ? {{5{cim_q[gk][7]}}, cim_q[gk], 3'b0}  : 16'sd0);
        wire signed [15:0] pimB =
            (w_q[4] ? {{4{cim_q[gk][7]}}, cim_q[gk], 4'b0}  : 16'sd0) +
            (w_q[5] ? {{3{cim_q[gk][7]}}, cim_q[gk], 5'b0}  : 16'sd0) +
            (w_q[6] ? {{2{cim_q[gk][7]}}, cim_q[gk], 6'b0}  : 16'sd0);
        reg signed [15:0] pAre_q, pBre_q, pAim_q, pBim_q;
        reg signed [15:0] pre_q, pim_q;
        always @(posedge clk) begin
            pAre_q <= preA;
            pBre_q <= preB;
            pAim_q <= pimA;
            pBim_q <= pimB;
            pre_q <= pAre_q + pBre_q;
            pim_q <= pAim_q + pBim_q;
            if (!m_run || ph == 3'd2) begin
                are_q <= bank_re[r_addr];
                aim_q <= bank_im[r_addr];
            end
            if (w_en) begin
                bank_re[w_addr] <= clr_we ? 32'sd0 : are_q + pre_q;
                bank_im[w_addr] <= clr_we ? 32'sd0 : aim_q + pim_q;
            end
        end
        assign rb_re_w[gk] = are_q;
        assign rb_im_w[gk] = aim_q;
    end endgenerate
    reg [1:0] rd_k_q;
    always @(posedge clk) begin
        rd_k_q <= rd_k;
        rd_re  <= rb_re_w[rd_k_q];
        rd_im  <= rb_im_w[rd_k_q];
    end

    // ---- matcher datapath (combinational, consumed by the FSM) ----
    // The staging is a 3-stage pipeline (the single-cycle version was
    // the fmax limiter at 28.9 MHz: ring-mux + barrel shift + negate):
    //   stage A (ph3..6, ring = ph+1): qm  <= (bank[ring] >>> sh)[15:0]
    //   stage B (ph4..7, ring = ph):   q/n <= +-conj?(qm)  (4 copies,
    //                                  negations precomputed here)
    //   consume (ph5..7,0', ring = ph+3): rot90 = PURE 4:1 mux -> MAC16s
    //   accumulate one cycle after each consume (con_q/rk_q pipeline).
    reg [5:0]  rho_q;
    reg [3:0]  sh_q;
    reg        wflag = 0;                   // j' wrapped -> conj cancels
    reg signed [15:0] qmre_q = 0, qmim_q = 0;         // stage A
    reg signed [15:0] qre_s = 0, qim_s = 0;           // stage B (+conj)
    reg signed [15:0] nre_s = 0, nim_s = 0;           //   negated copies
    reg        c0 = 0;                      // ring-3 consume carries to ph0
    reg [1:0]  rk_q = 0;                    // ring whose products are in p*_q
    reg        con_q = 0;                   // p*_q valid -> accumulate
    reg [3:0]  first_acc = 0;               // ring's next accumulate LOADS
                                            // (avoids a 256-bit clear-enable
                                            // cone from m_start — fmax path)
    reg signed [23:0] p0_q = 0, p1_q = 0, p2_q = 0, p3_q = 0;
    reg signed [31:0] sacc_re [0:NR-1];
    reg signed [31:0] sacc_im [0:NR-1];
    assign s_re = sacc_re[s_sel];
    assign s_im = sacc_im[s_sel];

    wire [6:0] wsum = {1'b0, mj} - {1'b0, rho_q};   // src = j - rho
    wire       wgt  = wsum[6];                      // borrow -> wrapped
    wire [1:0] rstg = ph[1:0] + 2'd1;       // ring staged (A) at ph3..6
    wire [1:0] rstb = ph[1:0];              // ring staged (B) at ph4..7
    wire [1:0] rcs  = ph[1:0] + 2'd3;       // ring consumed at ph5..7,0'
    wire signed [31:0] qs_re = rb_re_w[rstg] >>> sh_q;
    wire signed [31:0] qs_im = rb_im_w[rstg] >>> sh_q;
    // M-code store lives in a SPRAM (all 30 EBRs are taken; a register
    // file here costs ~900 LC). Single-port is safe: writes happen only
    // while idle (0x06), reads only during match beats. The SPRAM's
    // registered read IS the mc staging register: address {ring, mj} is
    // sampled at the ph4..7 edges, so DATAOUT holds ring rcs's code
    // during ph5..7,0' — exactly when the rot90 muxes consume it.
    wire [13:0] mc_a  = mc_we ? {6'b0, mc_addr} : {6'b0, rstb, mj};
    wire [15:0] mc_do;
    SB_SPRAM256KA mcram (
        .ADDRESS(mc_a), .DATAIN({14'b0, mc_code}), .MASKWREN(4'b1111),
        .WREN(mc_we), .CHIPSELECT(1'b1), .CLOCK(clk),
        .STANDBY(1'b0), .SLEEP(1'b0), .POWEROFF(1'b1), .DATAOUT(mc_do));
    wire [1:0] mc_q = mc_do[1:0];
    // rot90 by mc_q: negations were precomputed at stage B, so this is
    // a pure 4:1 mux in front of the MAC16s
    wire signed [15:0] hre = (mc_q == 2'd0) ? qre_s :
                             (mc_q == 2'd1) ? nim_s :
                             (mc_q == 2'd2) ? nre_s : qim_s;
    wire signed [15:0] him = (mc_q == 2'd0) ? qim_s :
                             (mc_q == 2'd1) ? qre_s :
                             (mc_q == 2'd2) ? nim_s : nre_s;
    wire signed [7:0]  mcre = cre_q[rcs];
    wire signed [7:0]  mcim = cim_q[rcs];
    wire signed [23:0] p0 = hre * mcre;     // 16x8 -> MAC16
    wire signed [23:0] p1 = him * mcim;
    wire signed [23:0] p2 = hre * mcim;
    wire signed [23:0] p3 = him * mcre;

    // ---- FSM ----
    reg [15:0] r_q;
    reg signed [31:0] xm, ym;
    reg [6:0] n;
    assign busy = (st != S_IDLE);

    always @(posedge clk) begin
        acc_we <= 1'b0;
        clr_we <= 1'b0;
        con_q  <= 1'b0;
        if (con_q) begin                    // accumulate ring rk_q products
            sacc_re[rk_q] <= (first_acc[rk_q] ? 32'sd0 : sacc_re[rk_q])
                             + (p0_q - p1_q);
            sacc_im[rk_q] <= (first_acc[rk_q] ? 32'sd0 : sacc_im[rk_q])
                             + (p2_q + p3_q);
            first_acc[rk_q] <= 1'b0;
        end
        case (st)
        S_IDLE: begin
            if (clear) begin clr_a <= 0; st <= S_CLR; m_done <= 1'b0; end
            else if (start) begin
                r_q <= r_mm; w_q <= {1'b0, w[6:0]};
                m_done <= 1'b0;
                st <= S_AZ;
            end
            else if (m_start) begin
                x <= m_dx; y <= m_dy;
                rho_q <= m_rho; sh_q <= m_sh;
                first_acc <= 4'b1111;       // lazy per-ring score clear
                m_done <= 1'b0;
                mj <= 0; ph <= 0;
                st <= S_M;
            end
        end
        S_CLR: begin
            clr_we <= 1'b1;
            wr_a <= clr_a;
            clr_a <= clr_a + 1;
            if (clr_a == NA - 1) st <= S_IDLE;
        end
        S_AZ: begin
            ang_a <= 0;
            st <= S_XY;
        end
        S_XY: begin
            xm = $signed(r_q) * azc_q;
            ym = $signed(r_q) * azs_q;
            x <= (xm + HALF) >>> 14;
            y <= (ym + HALF) >>> 14;
            ang_a <= 1;
            n <= 1;
            st <= S_L;
        end
        S_L: begin
            if (n >= 4 && n <= NA + 3) begin
                acc_we <= 1'b1;
                wr_a <= n - 4;
            end
            acc_a <= n - 3;
            ang_a <= n + 1;
            n <= n + 1;
            if (n == NA + 4) st <= S_IDLE;
        end
        S_M: begin
            ph <= ph + 1;                   // 3b, wraps 7 -> 0
            if (ph == 3'd0) begin
                ang_a <= mj;
                mr_a  <= wgt ? (wsum[5:0] + 6'd60) : wsum[5:0];
                wflag <= wgt;
                c0    <= 1'b0;
            end
            // ph1: ang ROM lands; ph2: u_q + are_q gates fire (see above)
            if (ph >= 3'd3 && ph <= 3'd6) begin   // stage A: ring rstg
                qmre_q <= qs_re[15:0];
                qmim_q <= qs_im[15:0];
            end
            if (ph >= 3'd4) begin           // stage B: ring rstb (+conj/neg)
                qre_s <=  qmre_q;
                nre_s <= -qmre_q;
                qim_s <= wflag ? qmim_q : -qmim_q;
                nim_s <= wflag ? -qmim_q : qmim_q;
            end
            if (ph >= 3'd5 || c0) begin     // consume ring rcs
                p0_q <= p0; p1_q <= p1; p2_q <= p2; p3_q <= p3;
                rk_q <= rcs;
                con_q <= 1'b1;
            end
            if (ph == 3'd7) begin
                c0 <= 1'b1;                 // ring 3 consumes at next ph0
                if (mj == NA - 1) st <= S_MD;
                else mj <= mj + 1;
            end
        end
        S_MD: begin                         // 2-cycle drain:
            ph <= ph + 1;                   // ph0: consume ring 3 (via c0),
            if (c0) begin                   //      accumulate ring 2
                p0_q <= p0; p1_q <= p1; p2_q <= p2; p3_q <= p3;
                rk_q <= rcs;
                con_q <= 1'b1;
                c0 <= 1'b0;
            end
            if (ph == 3'd1) begin           // ph1: ring 3 accumulated now
                m_done <= 1'b1;
                st <= S_IDLE;
            end
        end
        default: st <= S_IDLE;
        endcase
    end
endmodule
