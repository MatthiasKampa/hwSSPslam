// encoder_lean.v — v7 S/F-corner core: SERIAL-RING encode/match + resident
// map + fold-at-pose. Same golden contracts as the accepted v6 datapath
// (ssp_ice40 encode_int / match_int / solo.py encode_int_at) — every value
// bit-exact; only the SCHEDULE changes (the v6 4-ring parallel pipeline is
// a T-corner design: 488 cyc/cand where the solo task needs ~250 cand/kf
// at 5 Hz, i.e. ~500x less throughput than the fabric provides).
//
// Resource contract vs v6 core (measured at synth):
//   - ONE accumulator memory pair (256 x 32 re/im, 4 EBR) instead of the
//     8 per-ring banks (16 EBR); addr = {ring[1:0], j[5:0]}.
//   - ONE weight tree pair (comb, 24 MHz single-cycle) instead of 8
//     pipelined trees.
//   - TWO MAC16 (shared: r*az products, fold rotate, u projection, match
//     products) instead of 8.
//   - az ROMs quarter-wave folded: 1 EBR (az_t) instead of 8 (az_c+az_s);
//     fold verified EXHAUSTIVELY vs the flat ROMs in gen_luts.py.
// Throughput (cycle-counted in sim): encode ~11 cyc/angle -> ~668 cyc/pt;
// match ~23 cyc/angle -> ~1385 cyc/cand. Budget at 24 MHz / 5 Hz kf:
// 4.8 M cyc/kf >> scan encode (~680 k) + sweep (~360 k) + fold (~680 k).
//
// Fold-at-pose (encode_int_at contract, solo.py): when f_en is high at
// start, the SE2 pre-transform runs between the az->xy stage and the
// u-projection stage:
//   xr = (f_cq*x - f_sq*y + 2^14) >>> 15;  x2 = xr + f_tx   (y likewise)
// f_* must be held stable through the point; envelope (|x2|,|y2| < 2^15)
// is asserted by the golden model at fixture generation.
//
// Match per (ring k, angle j) — v6 semantics verbatim:
//   jsrc  = j - rho (mod 60), wrapped -> CONJUGATE NOT APPLIED (wflag)
//   qs    = acc[{k, jsrc}] >>> sh;  qre = qs_re[15:0];
//   qim   = wflag ? qs_im[15:0] : -qs_im[15:0]
//   cadr  = ((u_j >>> k) & 255) + {mc[{seg,k,j}], 6'b0}   (i^mc EXACT)
//   sacc_re[k] += qre*cre - qim*cim;  sacc_im[k] += qre*cim + qim*cre
// Readback contract (tracker T_SH): rd_idx issued at n readable at n+2.
`timescale 1ns/1ps
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
    input  wire [7:0]  rd_idx,             // readback {ring[1:0], j[5:0]}
    output reg  signed [31:0] rd_re,
    output reg  signed [31:0] rd_im,
    // ---- resident map ----
    input  wire        mc_we,
    input  wire [7:0]  mc_addr,            // {ring[1:0], j[5:0]}
    input  wire [1:0]  mc_code,
    input  wire [5:0]  mc_seg,
    // resident-map READ service (map dump; valid while idle only):
    // address {seg, ring, j}; data on mc_q one cycle later
    input  wire        mc_rsel,
    input  wire [13:0] mc_ra,
    output wire [15:0] mc_q,
    // ---- matcher ----
    input  wire        m_start,
    input  wire signed [15:0] m_dx,
    input  wire signed [15:0] m_dy,
    input  wire [5:0]  m_rho,
    input  wire [3:0]  m_sh,
    output reg         m_done = 0,
    input  wire [1:0]  s_sel,
    output wire signed [31:0] s_re,
    output wire signed [31:0] s_im,
    // ---- fold-at-pose ----
    input  wire        f_en,
    input  wire signed [15:0] f_cq,
    input  wire signed [15:0] f_sq,
    input  wire signed [15:0] f_tx,
    input  wire signed [15:0] f_ty
);
    localparam signed HALF14 = 1 << 13;    // F_AZ = F_ANG = 14 round const
    localparam signed HALF15 = 1 << 14;    // Q15 round const

    // ---- ROMs (images from gen_luts.py == golden model) ----
    reg signed [15:0] rom_azT [0:255];     // quarter-wave az table
    reg signed [15:0] rom_angc[0:63];
    reg signed [15:0] rom_angs[0:63];
    (* ram_style = "block" *)
    reg        [15:0] rom_cis [0:255];     // {im[7:0], re[7:0]} packed
    initial begin
        $readmemh("build/az_t.hex",  rom_azT);
        $readmemh("build/ang_c.hex", rom_angc);
        $readmemh("build/ang_s.hex", rom_angs);
        $readmemh("build/cis.hex",   rom_cis);
    end

    // ---- FSM ----
    localparam S_IDLE = 0, S_CLR = 1, S_AZA = 2, S_AZB = 3, S_AZC = 4,
               S_XYA = 5, S_XYB = 6, S_F1 = 7, S_F2 = 8, S_F3 = 9,
               S_UI = 10, S_UM2 = 11, S_UQ = 12,
               S_EA = 13, S_EB = 14,
               S_MA = 15, S_MB = 16, S_MC = 17, S_MD = 18, S_ME = 19,
               S_MDONE = 20;
    reg [4:0] st = S_IDLE;
    assign busy = (st != S_IDLE);

    reg        mmode = 0;                  // 0 encode, 1 match
    reg        fold = 0;                   // fold-at-pose armed (this point)
    reg [5:0]  j = 0;                      // angle
    reg [1:0]  k = 0;                      // ring
    reg [7:0]  clr_a = 0;
    reg signed [15:0] r_sq;
    reg [6:0]  w_q;
    reg signed [15:0] x, y;
    reg signed [15:0] azc, azs;            // folded az LUT values
    reg signed [31:0] xr32;
    reg signed [18:0] u_q;
    reg [5:0]  rho_q;
    reg [3:0]  sh_q;
    reg [5:0]  jsrc;
    reg        wflag;
    reg signed [15:0] qre_q, qim_q;

    // ---- az quarter-wave fold (address/sign comb; EXACT — see gen_luts)
    // cosfold(i): q=i[9:8], o=i[7:0] -> q0:+T[o] q1:-T[256-o] q2:-T[o]
    // q3:+T[256-o]; o==0 on odd q -> 0. az_c[i] = -cosfold(i);
    // az_s[i] = -cosfold((i-256)&1023). Two serial lookups (AZA/AZB).
    wire [9:0] az_s_i = az - 10'd256;      // sin index (mod 1024 by width)
    wire [9:0] azi    = (st == S_AZA) ? az : az_s_i;
    wire [1:0] az_qd  = azi[9:8];
    wire [7:0] az_o   = azi[7:0];
    wire [7:0] azt_a  = az_qd[0] ? (8'd0 - az_o) : az_o;   // 256-o mod 256
    wire       az_zero = az_qd[0] && (az_o == 8'd0);
    // -cosfold sign: q0,q3 -> negate T; q1,q2 -> pass T
    wire       az_neg  = (az_qd == 2'd0) || (az_qd == 2'd3);
    reg signed [15:0] azt_q;
    reg        az_zero_q, az_neg_q;
    always @(posedge clk) begin
        azt_q     <= rom_azT[azt_a];
        az_zero_q <= az_zero;
        az_neg_q  <= az_neg;
    end
    wire signed [15:0] az_val = az_zero_q ? 16'sd0
                              : (az_neg_q ? -azt_q : azt_q);

    // ---- angle ROM (registered read; addr = j comb) ----
    reg signed [15:0] angc_q, angs_q;
    always @(posedge clk) begin
        angc_q <= rom_angc[j];
        angs_q <= rom_angs[j];
    end

    // ---- accumulator memory (256 x 32 re/im; {k, j} addressing) ----
    reg signed [31:0] accre [0:255];
    reg signed [31:0] accim [0:255];
    reg signed [31:0] accre_q, accim_q;
    wire [7:0] acc_ra = (st == S_MA) ? {k, jsrc}
                      : (st == S_EA) ? {k, j}
                      : rd_idx;
    reg        acc_we;
    reg [7:0]  acc_wa;
    reg signed [31:0] acc_wdre, acc_wdim;
    always @(posedge clk) begin
        accre_q <= accre[acc_ra];
        accim_q <= accim[acc_ra];
        if (acc_we) begin
            accre[acc_wa] <= acc_wdre;
            accim[acc_wa] <= acc_wdim;
        end
    end
    always @(posedge clk) begin            // readback stage (n -> n+2)
        rd_re <= accre_q;
        rd_im <= accim_q;
    end

    // ---- mc SPRAM (resident map: {seg, ring, j}, one 2b code/word) ----
    wire [13:0] mc_a = mc_we ? {mc_seg, mc_addr}
                             : (mc_rsel && st == S_IDLE) ? mc_ra
                             : {mc_seg, k, j};
    wire [15:0] mc_do;
    assign mc_q = mc_do;
    SB_SPRAM256KA mcram (
        .ADDRESS(mc_a), .DATAIN({14'b0, mc_code}), .MASKWREN(4'b1111),
        .WREN(mc_we), .CHIPSELECT(1'b1), .CLOCK(clk),
        .STANDBY(1'b0), .SLEEP(1'b0), .POWEROFF(1'b1), .DATAOUT(mc_do));

    // ---- cis ROM (registered read; addr comb by mode) ----
    // encode: (u >>> k)[7:0]; match: + {mc, 6'b0} (i^mc address fold)
    wire signed [18:0] ush = u_q >>> k;
    wire [7:0] cis_a = ush[7:0] + (mmode ? {mc_do[1:0], 6'd0} : 8'd0);
    reg [15:0] cis_q;
    always @(posedge clk) cis_q <= rom_cis[cis_a];
    wire signed [15:0] cre16 = {{8{cis_q[7]}},  cis_q[7:0]};
    wire signed [15:0] cim16 = {{8{cis_q[15]}}, cis_q[15:8]};

    // ---- weight trees (comb; w constant per point) ----
    wire signed [7:0] cre8 = cis_q[7:0];
    wire signed [7:0] cim8 = cis_q[15:8];
    wire signed [15:0] wre =
        (w_q[0] ? {{8{cre8[7]}}, cre8}       : 16'sd0) +
        (w_q[1] ? {{7{cre8[7]}}, cre8, 1'b0} : 16'sd0) +
        (w_q[2] ? {{6{cre8[7]}}, cre8, 2'b0} : 16'sd0) +
        (w_q[3] ? {{5{cre8[7]}}, cre8, 3'b0} : 16'sd0) +
        (w_q[4] ? {{4{cre8[7]}}, cre8, 4'b0} : 16'sd0) +
        (w_q[5] ? {{3{cre8[7]}}, cre8, 5'b0} : 16'sd0) +
        (w_q[6] ? {{2{cre8[7]}}, cre8, 6'b0} : 16'sd0);
    wire signed [15:0] wim =
        (w_q[0] ? {{8{cim8[7]}}, cim8}       : 16'sd0) +
        (w_q[1] ? {{7{cim8[7]}}, cim8, 1'b0} : 16'sd0) +
        (w_q[2] ? {{6{cim8[7]}}, cim8, 2'b0} : 16'sd0) +
        (w_q[3] ? {{5{cim8[7]}}, cim8, 3'b0} : 16'sd0) +
        (w_q[4] ? {{4{cim8[7]}}, cim8, 4'b0} : 16'sd0) +
        (w_q[5] ? {{3{cim8[7]}}, cim8, 5'b0} : 16'sd0) +
        (w_q[6] ? {{2{cim8[7]}}, cim8, 6'b0} : 16'sd0);

    // ---- shared MAC16 pair (registered products, comb-muxed operands)
    reg signed [15:0] pA_a, pA_b, pB_a, pB_b;
    always @(*) begin
        case (st)
        S_XYA:   begin pA_a = r_sq;  pA_b = azc;    pB_a = r_sq;  pB_b = azs;    end
        S_F1:    begin pA_a = f_cq;  pA_b = x;      pB_a = f_sq;  pB_b = y;      end
        S_F2:    begin pA_a = f_sq;  pA_b = x;      pB_a = f_cq;  pB_b = y;      end
        S_UM2:   begin pA_a = x;     pA_b = angc_q; pB_a = y;     pB_b = angs_q; end
        S_MC:    begin pA_a = qre_q; pA_b = cre16;  pB_a = qim_q; pB_b = cim16;  end
        default: begin pA_a = qre_q; pA_b = cim16;  pB_a = qim_q; pB_b = cre16;  end
        endcase
    end
    reg signed [31:0] pa_q, pb_q;
    always @(posedge clk) begin
        pa_q <= pA_a * pA_b;
        pb_q <= pB_a * pB_b;
    end

    // ---- match score accumulators ----
    reg signed [31:0] sacc_re [0:NR-1];
    reg signed [31:0] sacc_im [0:NR-1];
    assign s_re = sacc_re[s_sel];
    assign s_im = sacc_im[s_sel];

    // ---- conj/wrap per angle: src = j - rho, borrow -> wrapped ----
    wire [6:0] wsum = {1'b0, j} - {1'b0, rho_q};
    wire       wgt  = wsum[6];

    always @(posedge clk) begin
        acc_we <= 1'b0;
        case (st)
        S_IDLE: begin
            if (clear) begin
                clr_a <= 0;
                m_done <= 1'b0;
                st <= S_CLR;
            end else if (start) begin
                r_sq <= $signed(r_mm);
                w_q  <= w[6:0];
                fold <= f_en;
                mmode <= 1'b0;
                m_done <= 1'b0;
                st <= S_AZA;
            end else if (m_start) begin
                x <= m_dx; y <= m_dy;
                rho_q <= m_rho; sh_q <= m_sh;
                mmode <= 1'b1;
                m_done <= 1'b0;
                sacc_re[0] <= 0; sacc_im[0] <= 0;
                sacc_re[1] <= 0; sacc_im[1] <= 0;
                sacc_re[2] <= 0; sacc_im[2] <= 0;
                sacc_re[3] <= 0; sacc_im[3] <= 0;
                j <= 0;
                st <= S_UI;
            end
        end
        S_CLR: begin
            acc_we <= 1'b1;
            acc_wa <= clr_a;
            acc_wdre <= 32'sd0;
            acc_wdim <= 32'sd0;
            clr_a <= clr_a + 1;
            if (clr_a == 8'd255) st <= S_IDLE;
        end
        // ---- az stage: 2 serial quarter-wave lookups ----
        S_AZA: st <= S_AZB;                // T[cos addr] lands this edge
        S_AZB: begin
            azc <= az_val;                 // sin lookup issued (azi mux)
            st <= S_AZC;
        end
        S_AZC: begin
            azs <= az_val;
            st <= S_XYA;
        end
        S_XYA: st <= S_XYB;                // r*azc, r*azs land this edge
        S_XYB: begin
            x <= (pa_q + HALF14) >>> 14;
            y <= (pb_q + HALF14) >>> 14;
            j <= 0;
            st <= fold ? S_F1 : S_UI;
        end
        // ---- fold-at-pose: Q15 rotate + translate (encode_int_at) ----
        S_F1: st <= S_F2;                  // cq*x, sq*y land this edge
        S_F2: begin
            xr32 <= pa_q - pb_q;           // sq*x, cq*y land next edge
            st <= S_F3;
        end
        S_F3: begin
            x <= ((xr32 + HALF15) >>> 15) + f_tx;
            y <= (((pa_q + pb_q) + HALF15) >>> 15) + f_ty;
            j <= 0;
            st <= S_UI;
        end
        // ---- angle head (shared encode/match) ----
        S_UI: begin                        // ang ROM read lands this edge
            jsrc  <= wgt ? (wsum[5:0] + 6'd60) : wsum[5:0];
            wflag <= wgt;
            st <= S_UM2;
        end
        S_UM2: st <= S_UQ;                 // x*C, y*S land this edge
        S_UQ: begin
            u_q <= (pa_q + pb_q + HALF14) >>> 14;
            k <= 0;
            st <= mmode ? S_MA : S_EA;
        end
        // ---- encode ring RMW (2 cyc/ring) ----
        S_EA: st <= S_EB;                  // acc + cis reads land this edge
        S_EB: begin
            acc_we <= 1'b1;
            acc_wa <= {k, j};
            acc_wdre <= accre_q + {{16{wre[15]}}, wre};
            acc_wdim <= accim_q + {{16{wim[15]}}, wim};
            k <= k + 1;
            if (k == 2'd3) begin
                if (j == NA - 1) st <= S_IDLE;      // point done
                else begin j <= j + 1; st <= S_UI; end
            end else st <= S_EA;
        end
        // ---- match ring (5 cyc/ring) ----
        S_MA: st <= S_MB;                  // acc read lands; mc_do valid MB
        S_MB: begin : mb
            reg signed [31:0] qsr, qsi;
            qsr = accre_q >>> sh_q;
            qsi = accim_q >>> sh_q;
            qre_q <= qsr[15:0];
            qim_q <= wflag ? qsi[15:0] : -qsi[15:0];
            st <= S_MC;                    // cis (with +64*mc) lands MB edge
        end
        S_MC: st <= S_MD;                  // qre*cre, qim*cim land this edge
        S_MD: begin
            sacc_re[k] <= sacc_re[k] + (pa_q - pb_q);
            st <= S_ME;                    // qre*cim, qim*cre land this edge
        end
        S_ME: begin
            sacc_im[k] <= sacc_im[k] + (pa_q + pb_q);
            k <= k + 1;
            if (k == 2'd3) begin
                if (j == NA - 1) st <= S_MDONE;
                else begin j <= j + 1; st <= S_UI; end
            end else st <= S_MA;
        end
        S_MDONE: begin
            m_done <= 1'b1;
            st <= S_IDLE;
        end
        default: st <= S_IDLE;
        endcase
    end
endmodule
