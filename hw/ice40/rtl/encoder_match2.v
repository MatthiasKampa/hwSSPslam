// v5: encoder_match.v + the three measured timing fixes (same golden
// contracts: ssp_ice40.encode_int / match_int — outputs bit-identical):
//   - the score-accumulate path (icetime #1, 35.75 ns: first_acc select
//     cone + 3-operand add) gains a pd = p0-p1 pipeline stage and
//     PRE-REGISTERED ONE-HOT write enables — the accumulate is now a
//     1-LUT zero-mux + one 32-bit add behind 1-bit enables;
//   - the SPRAM->rot90-mux->DSP cone (icetime #2) dissolves: mc codes
//     are sampled ONE PHASE EARLY ({rstg,mj} at stage-A time) and rot90
//     (+ conjugate wflag) folds INTO stage-B, so the MAC16 operands
//     (hre_q, him_q, mcre_q, mcim_q) are all plain registers;
//   - xm/ym are continuous wire mults (the FSM-embedded blocking mults
//     LUT-mapped in a v4 rebuild — DSP inference is now structural, and
//     the build flow asserts the MAC16 count).
// Match = 8 cyc/angle, 60 angles + 3-cycle drain: 484 cyc/candidate.
//
// Per-angle schedule (angle j, phases ph0..7):
//   ph0: ang_a <= j; mr_a <= wrap(j - rho); wflag
//   ph2: u_q gate; are_q <= bank[j'] gate
//   ph3..6 stage-A: qm <= (bank[rstg] >>> sh)[15:0]; mc sampled {rstg,j}
//   ph4..7 stage-B: hre_q/him_q <= rot90(mc_do, conj?(qm)); mc/cis regs
//   ph5..7,0'   consume: p*_q <= MAC16 products (all-register operands)
//   ph6..7,0',1'  pd:    pd_re/im_q <= p0-p1 / p2+p3; one-hot acc_en
//   ph7..0',1',2' acc:   sacc[k] += pd (enable = 1 bit, no decode)
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
    input  wire        mc_we,
    input  wire [7:0]  mc_addr,            //   {ring[1:0], j[5:0]}
    input  wire [1:0]  mc_code,
    input  wire        m_start,
    input  wire signed [15:0] m_dx,
    input  wire signed [15:0] m_dy,
    input  wire [5:0]  m_rho,
    input  wire [3:0]  m_sh,
    output reg         m_done = 0,
    input  wire [1:0]  s_sel,
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

    localparam S_IDLE = 0, S_CLR = 1, S_AZ = 2, S_XY = 3, S_L = 4,
               S_M = 5, S_MD = 6, S_XY2 = 7;
    reg [2:0] st = S_IDLE;
    wire m_run = (st == S_M) || (st == S_MD);
    reg [2:0] ph = 0;
    reg [5:0] mj = 0;

    reg signed [15:0] azc_q, azs_q, angc_q, angs_q;
    reg [5:0] ang_a;
    always @(posedge clk) begin
        azc_q  <= rom_azc[az];
        azs_q  <= rom_azs[az];
        angc_q <= rom_angc[ang_a];
        angs_q <= rom_angs[ang_a];
    end

    reg signed [15:0] x, y;
    wire signed [32:0] um_c = x * angc_q + y * angs_q;
    reg signed [18:0] u_q;
    always @(posedge clk)
        if (!m_run || ph == 3'd2) u_q <= (um_c + HALF) >>> 14;

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
    reg [5:0] mr_a;
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

    // ---- matcher datapath ----
    reg [5:0]  rho_q;
    reg [3:0]  sh_q;
    reg        wflag = 0;
    reg signed [15:0] qmre_q = 0, qmim_q = 0;         // stage A
    reg signed [15:0] hre_q = 0, him_q = 0;           // stage B (rot90 done)
    reg signed [7:0]  mcre_q = 0, mcim_q = 0;         //   cis regs
    reg [3:0]  first_acc = 0;
    reg [3:0]  acc_en = 0;                  // one-hot: accumulate pd now
    reg        acc_fa = 0;                  // captured first_acc bit
    reg        con_q = 0;                   // p*_q valid -> pd stage
    reg        c0 = 0;                      // ring-3 consume carries to ph0
    reg [1:0]  rk_q = 0;
    reg signed [23:0] p0_q = 0, p1_q = 0, p2_q = 0, p3_q = 0;
    reg signed [24:0] pd_re_q = 0, pd_im_q = 0;
    reg signed [31:0] sacc_re [0:NR-1];
    reg signed [31:0] sacc_im [0:NR-1];
    assign s_re = sacc_re[s_sel];
    assign s_im = sacc_im[s_sel];

    wire [6:0] wsum = {1'b0, mj} - {1'b0, rho_q};   // src = j - rho
    wire       wgt  = wsum[6];                      // borrow -> wrapped
    wire [1:0] rstg = ph[1:0] + 2'd1;       // ring staged (A) at ph3..6
    wire [1:0] rstb = ph[1:0];              // ring staged (B) at ph4..7
    wire signed [31:0] qs_re = rb_re_w[rstg] >>> sh_q;
    wire signed [31:0] qs_im = rb_im_w[rstg] >>> sh_q;
    // mc SPRAM sampled at stage-A time ({rstg, mj}) so DATAOUT holds the
    // staged ring's code DURING stage-B — rot90 happens there, and the
    // MAC16s below see only registers.
    wire [13:0] mc_a  = mc_we ? {6'b0, mc_addr} : {6'b0, rstg, mj};
    wire [15:0] mc_do;
    SB_SPRAM256KA mcram (
        .ADDRESS(mc_a), .DATAIN({14'b0, mc_code}), .MASKWREN(4'b1111),
        .WREN(mc_we), .CHIPSELECT(1'b1), .CLOCK(clk),
        .STANDBY(1'b0), .SLEEP(1'b0), .POWEROFF(1'b1), .DATAOUT(mc_do));
    // stage-B combinational: conj (unless wrapped) then * i^mc
    wire signed [15:0] bre =  qmre_q;
    wire signed [15:0] bim =  wflag ? qmim_q : -qmim_q;
    wire [1:0] mcs = mc_do[1:0];
    wire signed [15:0] hre_n = (mcs == 2'd0) ?  bre :
                               (mcs == 2'd1) ? -bim :
                               (mcs == 2'd2) ? -bre : bim;
    wire signed [15:0] him_n = (mcs == 2'd0) ?  bim :
                               (mcs == 2'd1) ?  bre :
                               (mcs == 2'd2) ? -bim : -bre;
    // MAC16 products: all-register operands
    wire signed [23:0] p0 = hre_q * mcre_q;
    wire signed [23:0] p1 = him_q * mcim_q;
    wire signed [23:0] p2 = hre_q * mcim_q;
    wire signed [23:0] p3 = him_q * mcre_q;

    // ---- FSM ----
    reg signed [15:0] r_sq;
    // REGISTERED full-width products — the canonical DSP pattern (both
    // the FSM-embedded blocking form and bare truncated wires LUT-mapped
    // in rebuilds: yosys narrows a truncated product and loses the DSP
    // match). Costs one extra encode state (S_XY2): 68 cyc/pt.
    reg signed [31:0] xm_q, ym_q;
    always @(posedge clk) begin
        xm_q <= r_sq * azc_q;
        ym_q <= r_sq * azs_q;
    end
    reg [6:0] n;
    assign busy = (st != S_IDLE);

    always @(posedge clk) begin
        acc_we <= 1'b0;
        clr_we <= 1'b0;
        con_q  <= 1'b0;
        acc_en <= 4'b0;
        if (con_q) begin                    // pd stage + one-hot arm
            pd_re_q <= {p0_q[23], p0_q} - {p1_q[23], p1_q};
            pd_im_q <= {p2_q[23], p2_q} + {p3_q[23], p3_q};
            acc_en  <= 4'b0001 << rk_q;
            acc_fa  <= first_acc[rk_q];
            first_acc[rk_q] <= 1'b0;
        end
        // accumulate: per-ring 1-bit registered enables, 2-operand adds
        // (pd_re/im_q and acc_en register at the SAME edge — no extra
        // delay stage, or pd gets overwritten by back-to-back consumes)
        if (acc_en[0]) begin
            sacc_re[0] <= (acc_fa ? 32'sd0 : sacc_re[0])
                          + {{7{pd_re_q[24]}}, pd_re_q};
            sacc_im[0] <= (acc_fa ? 32'sd0 : sacc_im[0])
                          + {{7{pd_im_q[24]}}, pd_im_q};
        end
        if (acc_en[1]) begin
            sacc_re[1] <= (acc_fa ? 32'sd0 : sacc_re[1])
                          + {{7{pd_re_q[24]}}, pd_re_q};
            sacc_im[1] <= (acc_fa ? 32'sd0 : sacc_im[1])
                          + {{7{pd_im_q[24]}}, pd_im_q};
        end
        if (acc_en[2]) begin
            sacc_re[2] <= (acc_fa ? 32'sd0 : sacc_re[2])
                          + {{7{pd_re_q[24]}}, pd_re_q};
            sacc_im[2] <= (acc_fa ? 32'sd0 : sacc_im[2])
                          + {{7{pd_im_q[24]}}, pd_im_q};
        end
        if (acc_en[3]) begin
            sacc_re[3] <= (acc_fa ? 32'sd0 : sacc_re[3])
                          + {{7{pd_re_q[24]}}, pd_re_q};
            sacc_im[3] <= (acc_fa ? 32'sd0 : sacc_im[3])
                          + {{7{pd_im_q[24]}}, pd_im_q};
        end
        case (st)
        S_IDLE: begin
            if (clear) begin clr_a <= 0; st <= S_CLR; m_done <= 1'b0; end
            else if (start) begin
                r_sq <= r_mm; w_q <= {1'b0, w[6:0]};
                m_done <= 1'b0;
                st <= S_AZ;
            end
            else if (m_start) begin
                x <= m_dx; y <= m_dy;
                rho_q <= m_rho; sh_q <= m_sh;
                first_acc <= 4'b1111;
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
        S_XY: st <= S_XY2;              // xm_q/ym_q register this edge
        S_XY2: begin
            x <= (xm_q + HALF) >>> 14;
            y <= (ym_q + HALF) >>> 14;
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
            ph <= ph + 1;
            if (ph == 3'd0) begin
                ang_a <= mj;
                mr_a  <= wgt ? (wsum[5:0] + 6'd60) : wsum[5:0];
                wflag <= wgt;
                c0    <= 1'b0;
            end
            if (ph >= 3'd3 && ph <= 3'd6) begin   // stage A: ring rstg
                qmre_q <= qs_re[15:0];
                qmim_q <= qs_im[15:0];
            end
            if (ph >= 3'd4) begin           // stage B: rot90 + operand regs
                hre_q  <= hre_n;
                him_q  <= him_n;
                mcre_q <= cre_q[rstb];
                mcim_q <= cim_q[rstb];
            end
            if (ph >= 3'd5 || c0) begin     // consume (ph5..7, 0' via c0)
                p0_q <= p0; p1_q <= p1; p2_q <= p2; p3_q <= p3;
                rk_q <= ph[1:0] + 2'd3;
                con_q <= 1'b1;
            end
            if (ph == 3'd7) begin
                c0 <= 1'b1;                 // ring 3 consumes at next ph0
                if (mj == NA - 1) st <= S_MD;
                else mj <= mj + 1;
            end
        end
        S_MD: begin                         // 3-cycle drain:
            ph <= ph + 1;                   // ph0: consume ring3 (via c0)
            if (c0) begin                   // ph1: pd + enable
                p0_q <= p0; p1_q <= p1; p2_q <= p2; p3_q <= p3;
                rk_q <= 2'd3;               // ph2 edge: sacc commit + done
                con_q <= 1'b1;
                c0 <= 1'b0;
            end
            if (ph == 3'd2) begin
                m_done <= 1'b1;
                st <= S_IDLE;
            end
        end
        default: st <= S_IDLE;
        endcase
    end
endmodule
