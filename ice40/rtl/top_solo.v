// top_solo.v — v7 STANDALONE SLAM top ("dumb cable" contract): points +
// odometry deltas in, pose frames out, map dump on demand. The chip runs
// the full loop alone: encode -> track (solo_tracker, REFINE per build) ->
// fold-at-pose into the open segment -> freeze every SEG_KF keyframes
// (mcode + 3-bit dead-zero LIVENESS plane + per-ring SCALES + anchor) ->
// resident map grows. Golden contract: ice40/host/solo.py (SoloTracker +
// SoloMapper + liveness_int); fixture gate: tb_solo_top.v via solo_top.py.
//
// Freeze store contract (BANKED 2026-07-12 scratch_liveness, both
// fixtures x both lattices):
//   liveness theta = 1/2:  alive iff 2*M >= Mmax_ring,
//     M = max(|I|,|Q|) + (min(|I|,|Q|) >> 1)      (Chebyshev |A| proxy)
//   per-ring scale = Mmax_ring as 5-bit exp + 8-bit mantissa (serial
//     normalize m >>= 1 until < 256). liveness+scales beats the FLOAT
//     store on extraction (school .6138 vs .5984, stata .6714 vs .6677);
//   caveat: mask/scales are DUMP-side only — the on-chip matcher reads
//   the raw 2b codes (masked local reads measurably worse).
//
// Memory map:
//   SPRAM (encoder)   resident map codes {seg, ring, j}
//   SPRAM2 (here)     0x0000 seg-acc 240 x {re_lo,re_hi,im_lo,im_hi} at
//                            ({ring,j} << 2) — i32 5-kf sums (envelope
//                            |acc| < 2^26 measured; golden asserts 2^31)
//                     0x0400 anchor mirror 64 x 8 words
//                     0x0600 liveness 64 x 15 words (bit index k*60+j,
//                            LSB-first; preload marks all-alive)
//                     0x0A00 ring scales 64 x 4 words {3'b0, e[4:0], m[7:0]}
//                            (preload marks unit: e=0, m=128)
//   SPRAM3 (here)     point buffer 3 words/point (az, r_mm, w): written at
//                     line rate by the parser, drained by the encoder for
//                     the tracking pass, re-drained for the fold pass
//
// UART protocol (8N1, DIV=UDIV; multi-byte LE; one command in flight —
// the reply is the flow control):
//   0x20                    -> 0xA5 (ping)
//   0x22 seg + 60 B codes   -> preload segment codes (4/byte LSB-first,
//                              component order k*60+j); liveness all-alive
//                              + unit scales; ack 0x2C
//   0x23 seg + 14 B anchor  -> x i32, y i32, h i16, cq i16, sq i16 ->
//                              tracker table + mirror; ack 0x2D
//   0x24 n                  -> live segment count; ack 0x2E
//   0x25 x i32 y i32 h u16  -> set pose; ack 0x2F
//   0x26 mode               -> bit0 mapping_en, bit1 stream_en; ack 0xA6
//   0x28                    -> map dump: 0xD0, n_seg, then per segment
//                              60 B codes + 30 B liveness + 8 B scales +
//                              16 B anchor
//   0x30 n u16 + n x (az u16, r u16, w u8) + dx i16 dy i16 dh i16
//                           -> KEYFRAME: encode-at-identity during ingest,
//                              pred = pose (+) R(pose_h) d, step (n_seg>0
//                              and n>=5), fold at the committed pose
//                              (mapping_en, n>=5, n_seg<NSEG), freeze on
//                              the SEG_KF'th fold; pose frame (stream_en):
//                              0xB0|{frozen<<3, state}, x i32, y i32, h u16
module top_solo #(
    parameter UDIV = 8,                   // 24 MHz / 8 = 3 Mbaud
    parameter NSEG = 64,
    parameter SEG_KF = 5,
    parameter REFINE = 0
) (
    input  wire clk,
    input  wire RX,
    output wire TX,
    output wire LEDR_N,
    output wire LEDG_N
);
    // ---------------- declarations (single-owner discipline) -----------
    // parser-owned
    reg [3:0]  prx = 0;
    reg        req_ping = 0, req_dump = 0, req_nseg = 0, req_pose = 0,
               req_mode = 0;
    reg [3:0]  ack_code = 0;              // 1 -> 0x2C, 2 -> 0x2D
    reg [6:0]  n_seg_req;
    reg [1:0]  mode_req;
    reg [7:0]  pb [0:13];
    reg [3:0]  pbi;
    reg [5:0]  mld_seg;
    reg [5:0]  mld_j;
    reg [1:0]  mld_k;
    reg [7:0]  mld_sh;
    reg [2:0]  mld_cnt = 0;
    reg [4:0]  liv_cnt = 0;               // preload liveness+scales marker
    reg [3:0]  anc_burst = 0;
    reg [15:0] npts_total;
    reg [15:0] npts = 0;
    reg [2:0]  ptb;
    reg [7:0]  pt_b0;
    reg        kf_hdr = 0, kf_dq = 0;
    reg signed [15:0] d_dx, d_dy, d_dh;
    reg        mld_we = 0;
    reg [7:0]  mld_addr;
    reg [1:0]  mld_code;
    reg        anc_we_p = 0;
    reg [8:0]  anc_wa_p;
    reg [15:0] anc_wd_p;
    reg        p2_we = 0;
    reg [13:0] p2_wa;
    reg [15:0] p2_wd;
    reg        s3_we = 0;
    reg [13:0] s3_wa;
    reg [15:0] s3_wd;
    // main-FSM-owned
    reg [4:0]  kst = 0;
    reg        req_done = 0;              // parser clears pending reqs
    reg        kf_ack = 0;
    reg [7:0]  ack_byte;
    reg signed [31:0] pose_x = 0, pose_y = 0;
    reg [9:0]  pose_h = 0;
    reg [6:0]  n_seg = 0;
    reg        mapping_en = 0, stream_en = 1;
    reg        seg_open = 0;
    reg [2:0]  folded = 0;
    reg signed [31:0] oa_x, oa_y;
    reg [9:0]  oa_h;
    reg signed [15:0] oa_cq, oa_sq;
    reg        frozen_kf = 0;
    reg [1:0]  kstate = 0;
    reg signed [31:0] pred_x, pred_y;
    reg [9:0]  pred_h;
    reg        enc_clear = 0;
    reg        fold_mode = 0;
    reg signed [15:0] f_cq, f_sq, f_tx, f_ty;
    reg        frz_we = 0;
    reg [7:0]  frz_addr;
    reg [1:0]  frz_code;
    reg        mc_rsel = 0;
    reg [13:0] mc_ra;
    reg        anc_we_f = 0;
    reg [8:0]  anc_wa_f;
    reg [15:0] anc_wd_f;
    reg        step_start = 0;
    reg        svc_start = 0;
    reg [1:0]  svc_op;
    reg signed [15:0] svc_x, svc_y;
    reg [9:0]  svc_h;
    reg [7:0]  top_rd_idx;
    reg        top_rd_own = 0;
    reg [13:0] s2_ra;
    reg        s2_we = 0;
    reg [13:0] s2_wa;
    reg [15:0] s2_wd;
    reg        drain_en = 0;
    reg        drain_rst = 0;
    reg [1:0]  k2;
    reg [5:0]  j6;
    reg [4:0]  ph;
    reg signed [31:0] w_re, w_im, b_re, b_im;
    reg signed [31:0] nsum;
    reg [32:0] mmax [0:3];
    reg [32:0] snorm;
    reg [4:0]  sexp;
    reg [15:0] livew;
    reg [5:0]  dseg;
    reg [7:0]  dcnt;
    reg [7:0]  packb;
    reg [1:0]  pk;
    reg [3:0]  txi = 0;
    reg [87:0] txsh;
    reg        txp_go = 0;
    // drain-owned
    reg [2:0]  dst = 0;
    reg [15:0] rd_pt = 0;
    reg        enc_start = 0;
    reg [9:0]  e_az;
    reg [15:0] e_r;
    reg [7:0]  e_w;
    reg        s3_owned = 0;

    // ---------------- UART ----------------
    wire [7:0] rxd;
    wire       rxv;
    uart_rx #(.DIV(UDIV)) u_rx (.clk(clk), .rx(RX), .data(rxd), .valid(rxv));
    reg  [7:0] txd;
    reg        send = 0;
    wire       tbusy;
    uart_tx #(.DIV(UDIV)) u_tx (.clk(clk), .data(txd), .send(send), .tx(TX),
                                .busy(tbusy));
    reg tx_free = 0;
    always @(posedge clk) tx_free <= !tbusy && !send;

    // ---------------- helpers ----------------
    function [9:0] wrap960(input signed [11:0] v);
        reg signed [11:0] t;
        begin
            t = v;
            if (t < 0) t = t + 12'sd960;
            if (t >= 12'sd960) t = t - 12'sd960;
            wrap960 = t[9:0];
        end
    endfunction
    function signed [15:0] sat16t(input signed [31:0] v);
        sat16t = (v > 32'sd32767) ? 16'sd32767 :
                 (v < -32'sd32767) ? -16'sd32767 : v[15:0];
    endfunction
    function [15:0] anc_word(input [3:0] wi);
        case (wi)
        4'd0: anc_word = {pb[1], pb[0]};
        4'd1: anc_word = {pb[3], pb[2]};
        4'd2: anc_word = {pb[5], pb[4]};
        4'd3: anc_word = {pb[7], pb[6]};
        4'd4: anc_word = {pb[9], pb[8]};
        4'd5: anc_word = {pb[11], pb[10]};
        4'd6: anc_word = {pb[13], pb[12]};
        default: anc_word = 16'd0;
        endcase
    endfunction
    localparam A_ACC = 14'h0000, A_ANC = 14'h0400, A_LIVE = 14'h0600,
               A_SCL = 14'h0A00;
    function [13:0] mul3(input [15:0] n);  // 3n = 2n + n (point stride)
        mul3 = {n[12:0], 1'b0} + n[13:0];
    endfunction
    // seg*15 / seg*8 / seg*4 region address helpers
    function [13:0] live_base(input [5:0] sg);
        // 15s = 16s - s. (The first form here was 16s - 4s = 12s: segment
        // liveness regions overlapped by 3 words and each freeze clobbered
        // its predecessor's last words — caught by the 220-kf dump diff's
        // fingerprint: words 12..14 wrong in every segment, codes/scales/
        // anchors clean, poses 220/220. Single-segment smokes CANNOT see
        // stride bugs — the gate needs >= 2 segments.)
        live_base = A_LIVE + {4'b0, sg, 4'b0} - {8'b0, sg};
    endfunction
    function [13:0] anc_base(input [5:0] sg);
        anc_base = A_ANC + {5'b0, sg, 3'b0};
    endfunction
    function [13:0] scl_base(input [5:0] sg);
        scl_base = A_SCL + {6'b0, sg, 2'b0};
    endfunction

    // ---------------- encoder + tracker ----------------
    wire       enc_busy;
    wire       m_done;
    wire [1:0] s_sel;
    wire signed [31:0] s_re, s_im;
    wire [5:0] trk_seg;
    wire       trk_mstart;
    wire signed [15:0] trk_dx, trk_dy;
    wire [5:0] trk_rho;
    wire [3:0] trk_sh;
    wire [7:0] trk_rd_idx;
    wire signed [31:0] rd_re, rd_im;
    wire [15:0] mc_q;
    wire       mc_we   = mld_we | frz_we;
    wire [5:0] mc_seg_w = mld_we ? mld_seg : n_seg[5:0];
    wire [7:0] mc_addr = mld_we ? mld_addr : frz_addr;
    wire [1:0] mc_code = mld_we ? mld_code : frz_code;
    wire [7:0] rd_idx_e = top_rd_own ? top_rd_idx : trk_rd_idx;

    encoder enc (.clk(clk), .clear(enc_clear), .start(enc_start),
                 .az(e_az), .r_mm(e_r), .w(e_w), .busy(enc_busy),
                 .rd_idx(rd_idx_e), .rd_re(rd_re), .rd_im(rd_im),
                 .mc_we(mc_we), .mc_addr(mc_addr), .mc_code(mc_code),
                 .mc_seg(mc_we ? mc_seg_w : trk_seg),
                 .mc_rsel(mc_rsel), .mc_ra(mc_ra), .mc_q(mc_q),
                 .m_start(trk_mstart), .m_dx(trk_dx), .m_dy(trk_dy),
                 .m_rho(trk_rho), .m_sh(trk_sh), .m_done(m_done),
                 .f_en(fold_mode), .f_cq(f_cq), .f_sq(f_sq),
                 .f_tx(f_tx), .f_ty(f_ty),
                 .s_sel(s_sel), .s_re(s_re), .s_im(s_im));

    wire       step_done;
    wire signed [31:0] out_x, out_y, out_score;
    wire [9:0] out_h;
    wire [5:0] out_ai;
    wire [1:0] out_state;
    wire       svc_done;
    wire signed [15:0] o_cq, o_sq;
    wire signed [31:0] o_fx, o_fy;

    solo_tracker #(.NSEG(NSEG), .REFINE(REFINE)) trk (.clk(clk),
                 .anc_we(anc_we_p | anc_we_f),
                 .anc_wa(anc_we_p ? anc_wa_p : anc_wa_f),
                 .anc_wd(anc_we_p ? anc_wd_p : anc_wd_f),
                 .n_seg(n_seg),
                 .step_start(step_start), .pred_x(pred_x),
                 .pred_y(pred_y), .pred_h(pred_h),
                 .step_done(step_done), .out_x(out_x), .out_y(out_y),
                 .out_h(out_h), .out_ai(out_ai), .out_score(out_score),
                 .out_state(out_state),
                 .mc_seg(trk_seg), .m_start(trk_mstart),
                 .m_dx(trk_dx), .m_dy(trk_dy), .m_rho(trk_rho),
                 .m_sh(trk_sh), .m_busy(enc_busy), .m_done(m_done),
                 .s_sel(s_sel), .s_re(s_re), .sh_in(4'd0),
                 .rd_idx(trk_rd_idx), .rd_re(rd_re), .rd_im(rd_im),
                 .svc_start(svc_start), .svc_op(svc_op),
                 .svc_x(svc_x), .svc_y(svc_y), .svc_h(svc_h),
                 .svc_done(svc_done), .o_cq(o_cq), .o_sq(o_sq),
                 .o_fx(o_fx), .o_fy(o_fy));

    // ---------------- SPRAM2: seg-acc / anchors / liveness / scales ----
    wire        s2_wen = s2_we | p2_we;
    wire [13:0] s2_a   = s2_we ? s2_wa : (p2_we ? p2_wa : s2_ra);
    wire [15:0] s2_din = s2_we ? s2_wd : p2_wd;
    wire [15:0] s2_q;
    SB_SPRAM256KA s2ram (
        .ADDRESS(s2_a), .DATAIN(s2_din), .MASKWREN(4'b1111),
        .WREN(s2_wen), .CHIPSELECT(1'b1), .CLOCK(clk),
        .STANDBY(1'b0), .SLEEP(1'b0), .POWEROFF(1'b1), .DATAOUT(s2_q));

    // ---------------- SPRAM3: point buffer ----------------
    reg  [13:0] s3_ra;
    wire [13:0] s3_a = s3_we ? s3_wa : s3_ra;
    wire [15:0] s3_q;
    SB_SPRAM256KA s3ram (
        .ADDRESS(s3_a), .DATAIN(s3_wd), .MASKWREN(4'b1111),
        .WREN(s3_we), .CHIPSELECT(1'b1), .CLOCK(clk),
        .STANDBY(1'b0), .SLEEP(1'b0), .POWEROFF(1'b1), .DATAOUT(s3_q));
    always @(posedge clk) s3_owned <= !s3_we;

    // ---------------- RX parser ----------------
    localparam P_IDLE = 0, P22S = 1, P22B = 2, P23S = 3, P23B = 4,
               P24 = 5, P25 = 6, P26 = 7, P30A = 8, P30B = 9, P30P = 10,
               P30D = 11;
    always @(posedge clk) begin
        s3_we <= 1'b0;
        mld_we <= 1'b0;
        anc_we_p <= 1'b0;
        p2_we <= 1'b0;
        if (req_done) begin
            req_ping <= 0; req_dump <= 0; req_nseg <= 0;
            req_pose <= 0; req_mode <= 0; ack_code <= 0;
        end
        if (kf_ack) begin kf_hdr <= 0; kf_dq <= 0; end
        // 0x22 code-byte unpacker: 4 mc writes per byte
        if (mld_cnt != 0) begin
            mld_we   <= 1'b1;
            mld_addr <= {mld_k, mld_j};
            mld_code <= mld_sh[1:0];
            mld_sh   <= mld_sh >> 2;
            mld_cnt  <= mld_cnt - 1;
            if (mld_j == 6'd59) begin
                mld_j <= 0;
                mld_k <= mld_k + 1;
                if (mld_k == 2'd3) begin
                    liv_cnt <= 5'd19;      // 15 liveness + 4 scale words
                    prx <= P_IDLE;
                end
            end else
                mld_j <= mld_j + 1;
        end
        // preload liveness all-alive + unit scales
        if (liv_cnt != 0) begin : plv
            reg [4:0] idx;
            idx = 5'd19 - liv_cnt;
            p2_we <= 1'b1;
            if (idx < 5'd15) begin
                p2_wa <= live_base(mld_seg) + {10'b0, idx[3:0]};
                p2_wd <= 16'hFFFF;
            end else begin
                p2_wa <= scl_base(mld_seg) + {12'b0, idx[1:0]};
                p2_wd <= 16'h0080;         // e=0, m=128 (unit weight)
            end
            liv_cnt <= liv_cnt - 1;
            if (liv_cnt == 5'd1) ack_code <= 4'd1;
        end
        // 0x23 anchor burst: table + mirror after 14 bytes
        if (anc_burst != 0) begin : ab
            reg [3:0] wi;
            reg [15:0] aw;
            wi = 4'd8 - anc_burst;         // word 0..7
            aw = anc_word(wi);             // ONE mux tree, two sinks
            anc_we_p <= 1'b1;
            anc_wa_p <= {mld_seg, wi[2:0]};
            anc_wd_p <= aw;
            p2_we <= 1'b1;
            p2_wa <= anc_base(mld_seg) + {11'b0, wi[2:0]};
            p2_wd <= aw;
            anc_burst <= anc_burst - 1;
            if (anc_burst == 4'd1) ack_code <= 4'd2;
        end
        if (rxv) begin
            case (prx)
            P_IDLE: case (rxd)
                8'h20: req_ping <= 1'b1;
                8'h22: prx <= P22S;
                8'h23: begin prx <= P23S; pbi <= 0; end
                8'h24: prx <= P24;
                8'h25: begin prx <= P25; pbi <= 0; end
                8'h26: prx <= P26;
                8'h28: req_dump <= 1'b1;
                8'h30: prx <= P30A;
                default: ;
                endcase
            P22S: begin
                mld_seg <= rxd[5:0]; mld_j <= 0; mld_k <= 0;
                prx <= P22B;
            end
            P22B: begin                    // 60 bytes; exits via unpacker
                mld_sh <= rxd; mld_cnt <= 3'd4;
            end
            P23S: begin mld_seg <= rxd[5:0]; prx <= P23B; end
            P23B: begin
                pb[pbi] <= rxd;
                pbi <= pbi + 1;
                if (pbi == 4'd13) begin
                    anc_burst <= 4'd8;
                    prx <= P_IDLE;
                end
            end
            P24: begin
                n_seg_req <= rxd[6:0]; req_nseg <= 1'b1; prx <= P_IDLE;
            end
            P25: begin
                pb[pbi] <= rxd;
                pbi <= pbi + 1;
                if (pbi == 4'd9) begin req_pose <= 1'b1; prx <= P_IDLE; end
            end
            P26: begin
                mode_req <= rxd[1:0]; req_mode <= 1'b1; prx <= P_IDLE;
            end
            P30A: begin npts_total[7:0] <= rxd; prx <= P30B; end
            P30B: begin
                npts_total[15:8] <= rxd;
                npts <= 0; ptb <= 0; pbi <= 0;
                kf_hdr <= 1'b1;
                prx <= ({rxd, npts_total[7:0]} == 16'd0) ? P30D : P30P;
            end
            P30P: begin
                ptb <= ptb + 1;
                case (ptb)
                3'd0: pt_b0 <= rxd;
                3'd1: begin                // az word at 3n
                    s3_we <= 1'b1;
                    s3_wa <= mul3(npts);
                    s3_wd <= {rxd, pt_b0};
                end
                3'd2: pt_b0 <= rxd;
                3'd3: begin                // r word at 3n+1
                    s3_we <= 1'b1;
                    s3_wa <= mul3(npts) + 14'd1;
                    s3_wd <= {rxd, pt_b0};
                end
                default: begin             // w word at 3n+2 completes
                    s3_we <= 1'b1;
                    s3_wa <= mul3(npts) + 14'd2;
                    s3_wd <= {8'b0, rxd};
                    npts <= npts + 1;
                    ptb <= 0;
                    if (npts + 16'd1 == npts_total) prx <= P30D;
                end
                endcase
            end
            P30D: begin
                pbi <= pbi + 1;
                case (pbi[2:0])
                3'd0: d_dx[7:0]  <= rxd;
                3'd1: d_dx[15:8] <= rxd;
                3'd2: d_dy[7:0]  <= rxd;
                3'd3: d_dy[15:8] <= rxd;
                3'd4: d_dh[7:0]  <= rxd;
                default: begin
                    d_dh[15:8] <= rxd;
                    kf_dq <= 1'b1;
                    pbi <= 0;
                    prx <= P_IDLE;
                end
                endcase
            end
            default: prx <= P_IDLE;
            endcase
        end
    end
    // ---------------- encode drain (SPRAM3 consumer) ----------------
    localparam D_IDLE = 0, D_A0 = 1, D_A1 = 2, D_A2 = 3, D_GO = 4;
    wire [15:0] rd_limit = fold_mode ? npts_total : npts;
    wire [13:0] rp0 = mul3(rd_pt);         // shared point base address
    // !drain_rst: while the reset pulse is in flight rd_pt still holds the
    // PREVIOUS drain's count — without the guard the fold replay "finishes"
    // instantly (rd_pt == npts_total from the tracking pass) and every
    // segment freezes an all-zero accumulator (the code-0 DC-map bug)
    wire drain_done = drain_en && !drain_rst && (dst == D_IDLE)
                      && (rd_pt == rd_limit)
                      && !enc_busy && !enc_start && kf_dq;
    always @(posedge clk) begin
        enc_start <= 1'b0;
        if (drain_rst) begin
            rd_pt <= 0;
            dst <= D_IDLE;
        end else if (!drain_en) dst <= D_IDLE;
        else case (dst)
        D_IDLE: if (rd_pt < rd_limit) begin    // < not !=: a shrunk limit
            s3_ra <= rp0;                      // must halt, never wrap
            dst <= D_A0;
        end
        D_A0: if (s3_owned) begin          // az addr cycle owned
            s3_ra <= rp0 + 14'd1;
            dst <= D_A1;
        end
        D_A1: begin
            if (s3_owned) begin            // az word valid now
                e_az <= s3_q[9:0];
                s3_ra <= rp0 + 14'd2;
                dst <= D_A2;
            end else begin                 // stolen: restart the point
                s3_ra <= rp0;
                dst <= D_A0;
            end
        end
        D_A2: begin
            if (s3_owned) begin
                e_r <= s3_q;
                dst <= D_GO;
            end else begin
                s3_ra <= rp0;
                dst <= D_A0;
            end
        end
        D_GO: begin
            if (!s3_owned) begin
                s3_ra <= rp0;
                dst <= D_A0;
            end else if (!enc_busy && !enc_start) begin
                e_w <= s3_q[7:0];
                enc_start <= 1'b1;
                rd_pt <= rd_pt + 1;
                dst <= D_IDLE;
            end
        end
        default: dst <= D_IDLE;
        endcase
    end

    // ---------------- main FSM ----------------
    localparam K_IDLE = 0, K_ACK = 1, K_CLR = 2, K_CLRW = 3, K_ING = 4,
               K_PRED = 5, K_PREDW = 6, K_STEP = 7, K_STEPW = 8,
               K_FOLDQ = 9, K_CSW = 10, K_IRW = 11, K_FCLR = 12,
               K_FCLRW = 13, K_FRPL = 14, K_FACC = 15, K_FRZA = 16,
               K_FRZB = 17, K_FRZC = 18, K_FRZS = 19, K_TXP = 20,
               K_DMP0 = 21, K_DMP1 = 22, K_DMPC = 23, K_DMPW = 24,
               K_DMPA = 25, K_KFEND = 26;
    wire [9:0] dh10 = wrap960({2'b0, pose_h} - {2'b0, oa_h});
    wire [31:0] aI = b_re[31] ? (~b_re + 1'b1) : b_re;
    wire [31:0] aQ = b_im[31] ? (~b_im + 1'b1) : b_im;
    wire [32:0] mM = (aI >= aQ) ? ({1'b0, aI} + {2'b0, aQ[31:1]})
                                : ({1'b0, aQ} + {2'b0, aI[31:1]});
    wire        alive = ({mM, 1'b0} >= {1'b0, mmax[k2]});  // theta = 1/2
    wire [7:0]  c60 = {k2, 6'd0} - {2'b0, k2, 2'd0} + {2'b0, j6};
    wire [13:0] acc_base = A_ACC + {4'b0, k2, j6, 2'b00};
    reg  [2:0]  dmode;                     // dump sub-block: 0 live 1 scl
                                           // 2 anchor
    reg  [7:0]  dwords;

    always @(posedge clk) begin
        enc_clear <= 1'b0;
        step_start <= 1'b0;
        svc_start <= 1'b0;
        send <= 1'b0;
        s2_we <= 1'b0;
        frz_we <= 1'b0;
        anc_we_f <= 1'b0;
        drain_rst <= 1'b0;
        req_done <= 1'b0;
        kf_ack <= 1'b0;
        case (kst)
        K_IDLE: begin
            top_rd_own <= 1'b0;
            mc_rsel <= 1'b0;
            txi <= 0;
            if (req_nseg) begin
                n_seg <= n_seg_req;
                ack_byte <= 8'h2E; req_done <= 1'b1; kst <= K_ACK;
            end else if (req_pose) begin
                pose_x <= {pb[3], pb[2], pb[1], pb[0]};
                pose_y <= {pb[7], pb[6], pb[5], pb[4]};
                pose_h <= {pb[9][1:0], pb[8]};
                ack_byte <= 8'h2F; req_done <= 1'b1; kst <= K_ACK;
            end else if (req_mode) begin
                mapping_en <= mode_req[0]; stream_en <= mode_req[1];
                ack_byte <= 8'hA6; req_done <= 1'b1; kst <= K_ACK;
            end else if (req_ping) begin
                ack_byte <= 8'hA5; req_done <= 1'b1; kst <= K_ACK;
            end else if (ack_code != 0) begin
                ack_byte <= (ack_code == 4'd1) ? 8'h2C : 8'h2D;
                req_done <= 1'b1;
                kst <= K_ACK;
            end else if (kf_hdr) begin
                frozen_kf <= 1'b0;
                kst <= K_CLR;
            end else if (req_dump) begin
                req_done <= 1'b1;
                dseg <= 0;
                kst <= K_DMP0;
            end
        end
        K_ACK: if (tx_free && !send) begin
            txd <= ack_byte; send <= 1'b1; kst <= K_IDLE;
        end
        // ---- keyframe: clear, ingest+encode, pred, step ----
        K_CLR: begin enc_clear <= 1'b1; kst <= K_CLRW; end
        K_CLRW: if (!enc_clear && !enc_busy) begin
            fold_mode <= 1'b0;
            drain_rst <= 1'b1;
            drain_en <= 1'b1;
            kst <= K_ING;
        end
        K_ING: if (drain_done) begin
            drain_en <= 1'b0;
            kst <= K_PRED;
        end
        K_PRED: begin                      // pred = pose (+) R(pose_h) d
            svc_op <= 2'd1;
            svc_x <= d_dx; svc_y <= d_dy; svc_h <= pose_h;
            svc_start <= 1'b1;
            kst <= K_PREDW;
        end
        K_PREDW: if (svc_done) begin
            pred_x <= pose_x + o_fx;
            pred_y <= pose_y + o_fy;
            pred_h <= wrap960($signed({2'b0, pose_h}) + d_dh[11:0]);
            kst <= K_STEP;
        end
        K_STEP: begin
            if (npts_total >= 16'd5 && n_seg != 0) begin
                step_start <= 1'b1;
                kst <= K_STEPW;
            end else begin
                pose_x <= pred_x; pose_y <= pred_y; pose_h <= pred_h;
                kstate <= 2'd3;            // odometry
                kst <= K_FOLDQ;
            end
        end
        K_STEPW: if (step_done) begin
            pose_x <= out_x; pose_y <= out_y; pose_h <= out_h;
            kstate <= out_state;
            kst <= K_FOLDQ;
        end
        // ---- fold: segment open / fold-prep via SE2 service ----
        K_FOLDQ: begin
            if (mapping_en && npts_total >= 16'd5
                && n_seg < NSEG[6:0]) begin
                if (!seg_open) begin
                    oa_x <= pose_x; oa_y <= pose_y; oa_h <= pose_h;
                    svc_op <= 2'd0; svc_h <= pose_h;
                    svc_start <= 1'b1;
                    f_cq <= 16'sd32767; f_sq <= 16'sd0;
                    f_tx <= 16'sd0; f_ty <= 16'sd0;
                    seg_open <= 1'b1;
                    kst <= K_CSW;
                end else begin
                    svc_op <= 2'd0; svc_h <= dh10;
                    svc_start <= 1'b1;
                    kst <= K_CSW;
                end
            end else kst <= K_TXP;
        end
        K_CSW: if (svc_done) begin
            if (folded == 0) begin         // segment open: identity fold
                oa_cq <= o_cq; oa_sq <= o_sq;
                kst <= K_FCLR;
            end else begin
                f_cq <= o_cq; f_sq <= o_sq;
                svc_op <= 2'd2;            // IROT at the anchor heading
                svc_x <= sat16t(pose_x - oa_x);
                svc_y <= sat16t(pose_y - oa_y);
                svc_h <= oa_h;
                svc_start <= 1'b1;
                kst <= K_IRW;
            end
        end
        K_IRW: if (svc_done) begin
            f_tx <= o_fx[15:0];
            f_ty <= o_fy[15:0];
            kst <= K_FCLR;
        end
        K_FCLR: begin enc_clear <= 1'b1; kst <= K_FCLRW; end
        K_FCLRW: if (!enc_clear && !enc_busy) begin
            fold_mode <= 1'b1;
            drain_rst <= 1'b1;
            drain_en <= 1'b1;
            kst <= K_FRPL;
        end
        K_FRPL: if (drain_done) begin
            drain_en <= 1'b0;
            fold_mode <= 1'b0;
            k2 <= 0; j6 <= 0; ph <= 0;
            top_rd_own <= 1'b1;
            kst <= K_FACC;
        end
        // ---- seg-acc RMW: bank component + 4 SPRAM2 words -------------
        K_FACC: begin
            ph <= ph + 1;
            case (ph)
            5'd0: begin top_rd_idx <= {k2, j6}; s2_ra <= acc_base; end
            5'd1: s2_ra <= acc_base + 14'd1;
            5'd2: begin w_re[15:0]  <= s2_q; s2_ra <= acc_base + 14'd2; end
            5'd3: begin w_re[31:16] <= s2_q; s2_ra <= acc_base + 14'd3;
                        b_re <= rd_re; end
            5'd4: begin w_im[15:0]  <= s2_q; b_im <= rd_im; end
            5'd5: begin
                w_im[31:16] <= s2_q;
                nsum <= (folded == 0) ? b_re : (w_re + b_re);
            end
            5'd6: begin
                s2_we <= 1'b1; s2_wa <= acc_base;
                s2_wd <= nsum[15:0];
            end
            5'd7: begin
                s2_we <= 1'b1; s2_wa <= acc_base + 14'd1;
                s2_wd <= nsum[31:16];
                nsum <= (folded == 0) ? b_im : (w_im + b_im);
            end
            5'd8: begin
                s2_we <= 1'b1; s2_wa <= acc_base + 14'd2;
                s2_wd <= nsum[15:0];
            end
            default: begin
                s2_we <= 1'b1; s2_wa <= acc_base + 14'd3;
                s2_wd <= nsum[31:16];
                ph <= 0;
                if (j6 == 6'd59) begin
                    j6 <= 0;
                    if (k2 == 2'd3) begin
                        k2 <= 0;
                        top_rd_own <= 1'b0;
                        if (folded + 3'd1 >= SEG_KF[2:0]) begin
                            mmax[0] <= 0; mmax[1] <= 0;
                            mmax[2] <= 0; mmax[3] <= 0;
                            kst <= K_FRZA;
                        end else begin
                            folded <= folded + 3'd1;
                            kst <= K_TXP;
                        end
                    end else k2 <= k2 + 1;
                end else j6 <= j6 + 1;
            end
            endcase
        end
        // ---- freeze pass A: mcode writes + per-ring Mmax ---------------
        K_FRZA: begin
            ph <= ph + 1;
            case (ph)
            5'd0: s2_ra <= acc_base;
            5'd1: s2_ra <= acc_base + 14'd1;
            5'd2: begin b_re[15:0]  <= s2_q; s2_ra <= acc_base + 14'd2; end
            5'd3: begin b_re[31:16] <= s2_q; s2_ra <= acc_base + 14'd3; end
            5'd4: b_im[15:0] <= s2_q;
            5'd5: b_im[31:16] <= s2_q;
            default: begin
                frz_we <= 1'b1;
                frz_addr <= {k2, j6};
                frz_code <= (aI >= aQ) ? (b_re[31] ? 2'd2 : 2'd0)
                                       : (b_im[31] ? 2'd3 : 2'd1);
                if (mM > mmax[k2]) mmax[k2] <= mM;
                ph <= 0;
                if (j6 == 6'd59) begin
                    j6 <= 0;
                    if (k2 == 2'd3) begin
                        k2 <= 0; livew <= 0;
                        kst <= K_FRZB;
                    end else k2 <= k2 + 1;
                end else j6 <= j6 + 1;
            end
            endcase
        end
        // ---- freeze pass B: liveness bits (theta = 1/2) ----------------
        K_FRZB: begin
            ph <= ph + 1;
            case (ph)
            5'd0: s2_ra <= acc_base;
            5'd1: s2_ra <= acc_base + 14'd1;
            5'd2: begin b_re[15:0]  <= s2_q; s2_ra <= acc_base + 14'd2; end
            5'd3: begin b_re[31:16] <= s2_q; s2_ra <= acc_base + 14'd3; end
            5'd4: b_im[15:0] <= s2_q;
            5'd5: b_im[31:16] <= s2_q;
            default: begin
                livew[c60[3:0]] <= alive;
                if (c60[3:0] == 4'd15) begin
                    s2_we <= 1'b1;
                    s2_wa <= live_base(n_seg[5:0]) + {10'b0, c60[7:4]};
                    s2_wd <= livew | ({15'b0, alive} << 4'd15);
                    livew <= 0;
                end
                ph <= 0;
                if (j6 == 6'd59) begin
                    j6 <= 0;
                    if (k2 == 2'd3) begin
                        k2 <= 0;
                        snorm <= mmax[0]; sexp <= 0;
                        kst <= K_FRZS;
                    end else k2 <= k2 + 1;
                end else j6 <= j6 + 1;
            end
            endcase
        end
        // ---- freeze pass S: ring scales (serial exp/mant normalize) ----
        K_FRZS: begin
            if (snorm >= 33'd256) begin
                snorm <= snorm >> 1;
                sexp <= sexp + 1;
            end else begin
                s2_we <= 1'b1;
                s2_wa <= scl_base(n_seg[5:0]) + {12'b0, k2};
                s2_wd <= {3'b0, sexp, snorm[7:0]};
                if (k2 == 2'd3) begin
                    k2 <= 0; ph <= 0;
                    kst <= K_FRZC;
                end else begin
                    k2 <= k2 + 1;
                    snorm <= mmax[{1'b0, k2} + 2'd1]; // next ring
                    sexp <= 0;
                end
            end
        end
        // ---- freeze pass C: anchor table + mirror, close segment -------
        K_FRZC: begin : frzc
            reg [15:0] aw;
            if (ph <= 5'd7) begin
                aw = (ph[2:0] == 3'd0) ? oa_x[15:0]
                   : (ph[2:0] == 3'd1) ? oa_x[31:16]
                   : (ph[2:0] == 3'd2) ? oa_y[15:0]
                   : (ph[2:0] == 3'd3) ? oa_y[31:16]
                   : (ph[2:0] == 3'd4) ? {6'b0, oa_h}
                   : (ph[2:0] == 3'd5) ? oa_cq[15:0]
                   : (ph[2:0] == 3'd6) ? oa_sq[15:0] : 16'd0;
                anc_we_f <= 1'b1;
                anc_wa_f <= {n_seg[5:0], ph[2:0]};
                anc_wd_f <= aw;
                s2_we <= 1'b1;
                s2_wa <= anc_base(n_seg[5:0]) + {11'b0, ph[2:0]};
                s2_wd <= aw;
                ph <= ph + 1;
            end else begin
                n_seg <= n_seg + 7'd1;
                seg_open <= 1'b0;
                folded <= 0;
                frozen_kf <= 1'b1;
                ph <= 0;
                kst <= K_TXP;
            end
        end
        // ---- pose frame ------------------------------------------------
        K_TXP: begin
            if (!txp_go) begin
                if (!stream_en) begin
                    kf_ack <= 1'b1;
                    kst <= K_KFEND;
                end else begin
                    txsh <= {6'b0, pose_h, pose_y, pose_x,
                             8'hB0 | {4'b0, frozen_kf, 1'b0, kstate}};
                    txi <= 4'd11;
                    txp_go <= 1'b1;
                end
            end else if (tx_free && !send) begin
                txd <= txsh[7:0];
                send <= 1'b1;
                txsh <= txsh >> 8;
                txi <= txi - 1;
                if (txi == 4'd1) begin
                    txp_go <= 1'b0;
                    kf_ack <= 1'b1;
                    kst <= K_KFEND;
                end
            end
        end
        // handshake completion: K_IDLE must not see the STALE kf_hdr (the
        // one-cycle ack race spawned a PHANTOM keyframe whose drain
        // re-encoded stale buffer points into the next kf's tracking Q —
        // the smoke-gate kf5+ drift AND the dump hang, one root cause)
        K_KFEND: if (!kf_hdr && !kf_dq) kst <= K_IDLE;
        // ---- map dump ----------------------------------------------------
        K_DMP0: if (tx_free && !send) begin
            txd <= 8'hD0; send <= 1'b1;
            kst <= K_DMP1;
        end
        K_DMP1: if (tx_free && !send) begin
            txd <= {1'b0, n_seg}; send <= 1'b1;
            dseg <= 0; dcnt <= 0; pk <= 0; k2 <= 0; j6 <= 0;
            ph <= 5'd0;                    // ph carries freeze residue
            mc_rsel <= 1'b1;
            kst <= (n_seg == 0) ? K_IDLE : K_DMPC;
        end
        K_DMPC: begin                      // codes: 4 x 2b -> 1 byte
            case (ph)
            5'd0: begin mc_ra <= {dseg, k2, j6}; ph <= 5'd1; end
            5'd1: ph <= 5'd2;              // mc_q lands
            5'd2: begin
                packb <= {mc_q[1:0], packb[7:2]};
                if (j6 == 6'd59) begin j6 <= 0; k2 <= k2 + 1; end
                else j6 <= j6 + 1;
                pk <= pk + 1;
                ph <= (pk == 2'd3) ? 5'd3 : 5'd0;
            end
            default: if (tx_free && !send) begin
                txd <= packb; send <= 1'b1;
                dcnt <= dcnt + 1;
                ph <= 5'd0;
                if (dcnt == 8'd59) begin
                    dcnt <= 0;
                    dmode <= 3'd0; dwords <= 8'd15;
                    s2_ra <= live_base(dseg);
                    ph <= 5'd0;
                    kst <= K_DMPW;
                end
            end
            endcase
        end
        K_DMPW: begin                      // stream dwords s2 words (LE)
            case (ph)
            5'd0: ph <= 5'd1;              // s2_q lands
            5'd1: begin livew <= s2_q; ph <= 5'd2; end
            5'd2: if (tx_free && !send) begin
                txd <= livew[7:0]; send <= 1'b1; ph <= 5'd3;
            end
            5'd3: if (tx_free && !send) begin
                txd <= livew[15:8]; send <= 1'b1;
                dcnt <= dcnt + 1;
                if (dcnt + 8'd1 == dwords) begin
                    dcnt <= 0; ph <= 5'd0;
                    case (dmode)
                    3'd0: begin            // -> scales
                        dmode <= 3'd1; dwords <= 8'd4;
                        s2_ra <= scl_base(dseg);
                    end
                    3'd1: begin            // -> anchor
                        dmode <= 3'd2; dwords <= 8'd8;
                        s2_ra <= anc_base(dseg);
                    end
                    default: begin         // segment done
                        if ({1'b0, dseg} + 7'd1 == n_seg) begin
                            mc_rsel <= 1'b0;
                            kst <= K_IDLE;
                        end else begin
                            dseg <= dseg + 1;
                            pk <= 0; k2 <= 0; j6 <= 0;
                            kst <= K_DMPC;
                        end
                    end
                    endcase
                end else begin
                    s2_ra <= s2_ra + 14'd1;
                    ph <= 5'd0;
                end
            end
            default: ph <= 5'd0;
            endcase
        end
        default: kst <= K_IDLE;
        endcase
    end

    reg [23:0] hb = 0;
    always @(posedge clk) hb <= hb + 1;
    assign LEDR_N = ~hb[23];
    assign LEDG_N = ~seg_open;
endmodule
