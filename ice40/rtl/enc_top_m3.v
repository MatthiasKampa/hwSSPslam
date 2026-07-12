// Protocol wrapper v4 (module enc_top_m2): enc_top_m.v + the BATCHED
// sweep command (pairs with encoder_match2.v). Protocol additions:
//   0x08 mode cnt_lo cnt_hi + cnt x (dxl dxh dyl dyh rho sh)
//     -> candidates BUFFERED through the shared 256-word FIFO (2 words
//        each, <=120 per batch); replies stream back-to-back per
//        candidate: mode bit0=1 -> 4-byte total (sum of the 4 ring Re
//        partials — the deployed unit-weight criterion), bit0=0 -> the
//        full 32-byte per-ring reply. NO host pacing invariant: the
//        FIFO absorbs write bursts (the paced-write overwrite crash
//        class dies here); reply consumption is the flow control.
//   Points and batches are phase-exclusive (host contract; the point
//   drain arm is hard-gated while a batch is pending).
// Base protocol:
//   0x01                          -> clear accumulators, ack 0x2B
//   0x02 az_lo az_hi r_lo r_hi w  -> encode one point, ack 0x2A on completion
//   0x03                          -> stream 240 x (re i32 LE, im i32 LE)
//   0x04                          -> point count (lo, hi)
//   0x05 (as 0x02)                -> point WITHOUT completion ack (bulk)
//   0x06 + 60 bytes               -> load 240 2b M codes (4 per byte,
//                                    LSB-first, component order k*60+j);
//                                    ack 0x2C when stored
//   0x07 dxl dxh dyl dyh rho sh   -> match one candidate against the
//                                    LAST ENCODED SCAN; reply 32 bytes =
//                                    4 rings x (re i32 LE, im i32 LE)
// Contracts: 0x06/0x07 only while no points are in flight; one match in
// flight at a time (the 32-byte reply is the flow control). Point FIFO
// and mode-noack semantics unchanged from enc_top.v.
module enc_top_m3 #(
    parameter UDIV = 4                 // clk / UDIV = baud (3 Mbaud target)
) (
    input  wire clk,
    input  wire RX,
    output wire TX,
    output wire LEDR_N,
    output wire LEDG_N
);
    wire [7:0] rxd;
    wire       rxv;
    uart_rx #(.DIV(UDIV)) u_rx (.clk(clk), .rx(RX), .data(rxd), .valid(rxv));
    reg  [7:0] txd;
    reg        send = 0;
    wire       tbusy;
    uart_tx #(.DIV(UDIV)) u_tx (.clk(clk), .data(txd), .send(send), .tx(TX),
                                .busy(tbusy));

    reg        clear = 0, start = 0;
    reg [9:0]  az;
    reg [15:0] r_mm;
    reg [7:0]  w;
    wire       ebusy;
    reg [7:0]  rd_idx;
    wire signed [31:0] rd_re, rd_im;
    reg        mc_we = 0;
    reg [7:0]  mc_addr;
    reg [1:0]  mc_code;
    reg        m_go = 0;
    reg signed [15:0] m_dx, m_dy;
    reg [5:0]  m_rho;
    reg [3:0]  m_sh;
    wire       m_done;
    reg [1:0]  s_sel;
    wire signed [31:0] s_re, s_im;
    wire signed [15:0] e_dx = bsel ? $signed(b_dx) : m_dx;
    wire signed [15:0] e_dy = bsel ? $signed(b_dy) : m_dy;
    wire [5:0] e_rho = bsel ? b_rho : m_rho;
    wire [3:0] e_sh  = bsel ? b_sh : m_sh;
    encoder enc (.clk(clk), .clear(clear), .start(start), .az(az),
                 .r_mm(r_mm), .w(w), .busy(ebusy), .rd_idx(rd_idx),
                 .rd_re(rd_re), .rd_im(rd_im),
                 .mc_we(mc_we), .mc_addr(mc_addr), .mc_code(mc_code),
                 .m_start(m_go), .m_dx(e_dx), .m_dy(e_dy),
                 .m_rho(e_rho), .m_sh(e_sh), .m_done(m_done),
                 .s_sel(s_sel), .s_re(s_re), .s_im(s_im));

    // ---- RX parser: always consumes, never blocks ----
    reg [3:0]  prx = 0;
    reg [9:0]  p_az;
    reg [14:0] p_r;
    reg        mode_noack = 0;         // 0x02 -> acked, 0x05 -> bulk
    reg        req_clear = 0, req_rb = 0, req_st = 0;
    reg        bhdr = 0;               // parsing batch candidates
    reg        bmode = 0;              // 1 = totals-only replies
    reg        bmode2 = 0;             // 1 = argmax batch (one reply)
    reg [8:0]  bpar = 0;               // candidates left to parse
    reg [1:0]  bpsh = 0;               // 2-word FIFO pusher
    reg        bset = 0;               // pulse: load bcnt from btot
    reg [8:0]  btot;
    reg        mld_ack = 0;            // M-load complete, owe 0x2C
    reg        push = 0;
    reg [31:0] pdata;
    // M-load unpacker: 4 mc writes per received byte (bytes arrive
    // >= 120 clocks apart at 3 Mbaud, the burst always drains first)
    reg [7:0]  mld_sh;
    reg [2:0]  mld_cnt = 0;
    reg [5:0]  mld_j = 0;
    reg [1:0]  mld_k = 0;
    always @(posedge clk) begin
        push <= 1'b0;
        mc_we <= 1'b0;
        if (mld_cnt != 0) begin
            mc_we   <= 1'b1;
            mc_addr <= {mld_k, mld_j};
            mc_code <= mld_sh[1:0];
            mld_sh  <= mld_sh >> 2;
            mld_cnt <= mld_cnt - 1;
            if (mld_j == 6'd59) begin
                mld_j <= 0;
                mld_k <= mld_k + 1;
                if (mld_k == 2'd3) begin mld_ack <= 1'b1; prx <= 0; end
            end else
                mld_j <= mld_j + 1;
        end
        bset <= 1'b0;
        if (bpsh != 0) begin              // batch candidate -> 2 words
            push  <= 1'b1;
            pdata <= (bpsh == 2'd2) ? {m_dx, m_dy}
                                    : {m_rho, m_sh, 22'b0};
            bpsh  <= bpsh - 1;
            if (bpsh == 2'd1) begin
                bpar <= bpar - 1;
                prx  <= (bpar == 9'd1) ? 4'd0 : 4'd7;
            end
        end
        if (rxv) begin
            case (prx)
            4'd0: case (rxd)
                  8'h02,                       // deploy build: 0x02 == 0x05
                  8'h05: begin mode_noack <= 1'b1; prx <= 1; end
                  8'h01: req_clear <= 1'b1;
                  8'h03: req_rb <= 1'b1;
                  8'h04: req_st <= 1'b1;
                  8'h06: begin mld_j <= 0; mld_k <= 0; prx <= 6; end
                  8'h07: begin bhdr <= 1'b0; prx <= 7; end
                  8'h08: prx <= 13;
                  default: ;
                  endcase
            4'd1: begin p_az[7:0]  <= rxd;      prx <= 2; end
            4'd2: begin p_az[9:8]  <= rxd[1:0]; prx <= 3; end
            4'd3: begin p_r[7:0]   <= rxd;      prx <= 4; end
            4'd4: begin p_r[14:8]  <= rxd[6:0]; prx <= 5; end
            4'd5: begin
                pdata <= {p_az, p_r, rxd[6:0]};
                push  <= 1'b1;
                prx   <= 0;
            end
            4'd6: begin                // M-load byte -> 4-write burst
                mld_sh  <= rxd;        // prx exits via the unpacker after
                mld_cnt <= 3'd4;       // the 240th component (mld_k wrap)
            end
            4'd7:  begin m_dx[7:0]  <= rxd;      prx <= 8; end
            4'd8:  begin m_dx[15:8] <= rxd;      prx <= 9; end
            4'd9:  begin m_dy[7:0]  <= rxd;      prx <= 10; end
            4'd10: begin m_dy[15:8] <= rxd;      prx <= 11; end
            4'd11: begin m_rho      <= rxd[5:0]; prx <= 12; end
            4'd12: begin
                m_sh <= rxd[3:0];
                if (bhdr) bpsh <= 2'd2;    // batch-only in the deploy
                else prx <= 0;             // build: bare 0x07 is DISCARDED
            end
            4'd13: begin bmode <= rxd[0]; bmode2 <= rxd[1]; prx <= 14; end
            4'd14: begin bpar[7:0] <= rxd; prx <= 15; end
            4'd15: begin
                bpar[8] <= rxd[0];
                btot <= {rxd[0], bpar[7:0]};
                bset <= 1'b1;
                bhdr <= 1'b1;
                prx <= 7;                  // candidates reuse states 7..12
            end
            default: prx <= 0;
            endcase
        end
        if (ack_clear) req_clear <= 1'b0;
        if (ack_rb)    req_rb <= 1'b0;
        if (ack_st)    req_st <= 1'b0;
        if (mldack_tx) mld_ack <= 1'b0;
    end

    // ---- point FIFO (256 deep x 32 -> exactly 2 EBRs) ----
    reg [31:0] fmem [0:255];
    reg [7:0]  fwr = 0, frd = 0;
    reg [8:0]  fcnt = 0;
    // registered flag: the 9-bit NOR fed the E_IDLE priority cone (fmax
    // path). Safe to lag 1 cycle — after any pop the FSM is away from
    // E_IDLE for >=3 cycles, so the flag always settles before it is
    // read again; a late-seen push only delays the pop by a cycle.
    reg fifo_empty = 1;
    always @(posedge clk) fifo_empty <= (fcnt == 0) && !push;
    reg fifo_ge2 = 0;
    always @(posedge clk) fifo_ge2 <= (fcnt >= 2);
    reg        pop = 0;
    reg [31:0] fq;
    always @(posedge clk) begin
        if (push) begin fmem[fwr] <= pdata; fwr <= fwr + 1; end
        if (pop)  begin fq <= fmem[frd];    frd <= frd + 1; end
        case ({push, pop})
        2'b10: fcnt <= fcnt + 1;
        2'b01: fcnt <= fcnt - 1;
        default: ;
        endcase
    end

    // ---- exec FSM: drain fifo, run encoder/matcher, arbitrate TX ----
    localparam E_IDLE = 0, E_POP = 1, E_GO = 2, E_RUN = 3, E_CLR = 4,
               E_RB = 5, E_ST = 6, E_MRB = 7,
               E_BP1 = 8, E_BP2 = 9, E_BGO = 10, E_BWT = 11,
               E_BT = 12, E_BTS = 13, E_BAX = 14, E_BXS = 15;
    reg [3:0]  est = E_IDLE;
    reg [8:0]  bcnt = 0;               // candidates left to dispatch
    reg        bsel = 0;               // encoder inputs from batch regs
    reg [15:0] b_dx, b_dy;
    reg [5:0]  b_rho;
    reg [3:0]  b_sh;
    reg signed [31:0] tot_q;
    reg [1:0]  tsel;
    reg [8:0]  bx_i;                   // candidate index within the batch
    reg [8:0]  bx_bidx;
    reg signed [31:0] bx_best;
    reg        ack_clear = 0, ack_rb = 0, ack_st = 0;
    reg        mldack_tx = 0;
    // registered TX readiness: the raw (bits==0) compare from uart_tx
    // threaded the whole 10-state priority cone (fmax path). The !send
    // term in each condition covers the pulse cycle; tx_free covers the
    // cycle after (send kills it) through the busy frame (tbusy holds it).
    reg        tx_free = 0;
    reg        clr_ack_due = 0;
    reg [15:0] npts = 0;
    reg [3:0]  rb_b;
    reg [5:0]  rb_j;               // angle counter 0..NA-1 (inner)
    reg [1:0]  rb_k;               // ring counter (outer) — rd_idx is
                                   // {rb_k, rb_j}, so the wire byte order
                                   // stays the packed ring-major stream
    reg [1:0]  rb_wait;
    reg [63:0] rb_sh;
    reg        st_hi = 0;

    always @(posedge clk) tx_free <= !tbusy && !send;

    always @(posedge clk) begin
        if (bset) begin
            bcnt <= btot;
            bx_i <= 9'd0;
            bx_bidx <= 9'd0;
            bx_best <= 32'sh80000000;
        end
        send <= 1'b0;
        start <= 1'b0;
        clear <= 1'b0;
        pop <= 1'b0;
        ack_clear <= 1'b0;
        ack_rb <= 1'b0;
        ack_st <= 1'b0;
        m_go <= 1'b0;
        mldack_tx <= 1'b0;
        case (est)
        E_IDLE: begin
            if (clr_ack_due && tx_free && !send) begin
                txd <= 8'h2B; send <= 1'b1; clr_ack_due <= 1'b0;
            end else if (mld_ack && tx_free && !send) begin
                txd <= 8'h2C; send <= 1'b1; mldack_tx <= 1'b1;
            end else if (req_clear && fifo_empty && !ebusy) begin
                clear <= 1'b1; ack_clear <= 1'b1; npts <= 0; est <= E_CLR;
            end else if (bcnt != 0 && fifo_ge2 && !ebusy) begin
                pop <= 1'b1; est <= E_BP1;
            end else if (!fifo_empty && !ebusy && bcnt == 0) begin
                pop <= 1'b1; est <= E_POP;
            end else if (req_rb && fifo_empty && !ebusy 
                         && !clr_ack_due) begin
                ack_rb <= 1'b1; rb_j <= 0; rb_k <= 0; rd_idx <= 0;
                rb_wait <= 0; rb_b <= 0; est <= E_RB;
            end else if (req_st && fifo_empty && !ebusy 
                         && !clr_ack_due) begin
                ack_st <= 1'b1; st_hi <= 0; est <= E_ST;
            end
        end
        E_POP: est <= E_GO;                     // fq valid next cycle
        E_GO: begin
            az   <= fq[31:22];
            r_mm <= {1'b0, fq[21:7]};
            w    <= {1'b0, fq[6:0]};
            start <= 1'b1;
            est <= E_RUN;
        end
        E_RUN: if (!start && !ebusy) begin      // point done
            npts <= npts + 1;
            est <= E_IDLE;
        end
        E_CLR: if (!clear && !ebusy) begin
            clr_ack_due <= 1'b1;
            est <= E_IDLE;
        end
        E_RB: begin
            if (rb_wait != 2) rb_wait <= rb_wait + 1;
            else if (rb_b == 0) begin
                rb_sh <= {rd_im, rd_re};
                rb_b  <= 1;
            end else if (tx_free && !send) begin
                txd  <= rb_sh[7:0];
                send <= 1'b1;
                rb_sh <= rb_sh >> 8;
                if (rb_b == 8) begin
                    rb_b <= 0; rb_wait <= 0;
                    if (rb_j == 59) begin
                        rb_j <= 0; rb_k <= rb_k + 1;
                        rd_idx <= {rb_k + 2'd1, 6'd0};
                        if (rb_k == 3) est <= E_IDLE;
                    end else begin
                        rb_j <= rb_j + 1;
                        rd_idx <= {rb_k, rb_j + 6'd1};
                    end
                end else rb_b <= rb_b + 1;
            end
        end
        E_ST: if (tx_free && !send) begin
            txd <= st_hi ? npts[15:8] : npts[7:0];
            send <= 1'b1;
            if (st_hi) est <= E_IDLE;
            st_hi <= ~st_hi;
        end
        E_MRB: begin                    // stream 4 rings x (re, im) i32 LE
            if (rb_b == 0) begin
                rb_sh <= {s_im, s_re};
                rb_b  <= 1;
            end else if (tx_free && !send) begin
                txd  <= rb_sh[7:0];
                send <= 1'b1;
                rb_sh <= rb_sh >> 8;
                if (rb_b == 8) begin
                    rb_b <= 0;
                    s_sel <= s_sel + 1;
                    if (s_sel == 2'd3) est <= E_IDLE;
                end else rb_b <= rb_b + 1;
            end
        end
        E_BP1: begin                    // w0 lands in fq at this edge
            pop <= 1'b1;                // request w1
            est <= E_BP2;
        end
        E_BP2: begin                    // fq = w0 now (w1 lands this edge)
            b_dx <= fq[31:16];
            b_dy <= fq[15:0];
            est <= E_BGO;
        end
        E_BGO: begin
            b_rho <= fq[31:26];
            b_sh  <= fq[25:22];
            bsel  <= 1'b1;
            m_go  <= 1'b1;              // encoder samples e_* next edge
            bcnt  <= bcnt - 1;
            est   <= E_BWT;
        end
        E_BWT: if (!m_go && !ebusy) begin
            if (bmode || bmode2) begin
                tot_q <= 32'sd0; tsel <= 2'd0; s_sel <= 2'd0;
                est <= E_BT;
            end else begin
                s_sel <= 2'd0; rb_b <= 0;
                est <= E_MRB;           // full replies; loop via E_IDLE
            end
        end
        E_BT: begin                     // totals: sum the 4 ring Re parts
            tot_q <= tot_q + s_re;
            s_sel <= s_sel + 1;
            tsel  <= tsel + 1;
            if (tsel == 2'd3) begin
                rb_b <= 0;
                est <= bmode2 ? E_BAX : E_BTS;
            end
        end
        E_BAX: begin                    // argmax update (strict >:
            if (tot_q > bx_best) begin  //  first max wins == np.argmax)
                bx_best <= tot_q;
                bx_bidx <= bx_i;
            end
            bx_i <= bx_i + 1;
            if (bcnt == 0) begin rb_b <= 0; est <= E_BXS; end
            else est <= E_IDLE;
        end
        E_BXS: begin                    // one 6-byte reply per batch:
            if (rb_b == 0) begin        // idx u16 LE + best i32 LE
                rb_sh <= {32'b0, bx_best};
                rb_b  <= 1;
            end else if (tx_free && !send) begin
                txd  <= (rb_b == 1) ? bx_bidx[7:0] :
                        (rb_b == 2) ? {7'b0, bx_bidx[8]} : rb_sh[7:0];
                send <= 1'b1;
                if (rb_b >= 3) rb_sh <= rb_sh >> 8;
                if (rb_b == 6) est <= E_IDLE;
                else rb_b <= rb_b + 1;
            end
        end
        E_BTS: begin                    // stream 4-byte total
            if (rb_b == 0) begin
                rb_sh <= {32'b0, tot_q};
                rb_b  <= 1;
            end else if (tx_free && !send) begin
                txd  <= rb_sh[7:0];
                send <= 1'b1;
                rb_sh <= rb_sh >> 8;
                if (rb_b == 4) est <= E_IDLE;
                else rb_b <= rb_b + 1;
            end
        end
        default: est <= E_IDLE;
        endcase
    end

    reg [23:0] hb = 0;
    always @(posedge clk) hb <= hb + 1;
    assign LEDR_N = ~hb[23];
    assign LEDG_N = ~ebusy;
endmodule
