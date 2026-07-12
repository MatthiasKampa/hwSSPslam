// Protocol wrapper v3 (module enc_top_m): enc_top.v + the MATCHER
// commands (pairs with encoder_match.v). Protocol:
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
module enc_top_m #(
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
    encoder enc (.clk(clk), .clear(clear), .start(start), .az(az),
                 .r_mm(r_mm), .w(w), .busy(ebusy), .rd_idx(rd_idx),
                 .rd_re(rd_re), .rd_im(rd_im),
                 .mc_we(mc_we), .mc_addr(mc_addr), .mc_code(mc_code),
                 .m_start(m_go), .m_dx(m_dx), .m_dy(m_dy),
                 .m_rho(m_rho), .m_sh(m_sh), .m_done(m_done),
                 .s_sel(s_sel), .s_re(s_re), .s_im(s_im));

    // ---- RX parser: always consumes, never blocks ----
    reg [3:0]  prx = 0;
    reg [9:0]  p_az;
    reg [14:0] p_r;
    reg        mode_noack = 0;         // 0x02 -> acked, 0x05 -> bulk
    reg        req_clear = 0, req_rb = 0, req_st = 0, req_m = 0;
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
        if (rxv) begin
            case (prx)
            4'd0: case (rxd)
                  8'h02: begin mode_noack <= 1'b0; prx <= 1; end
                  8'h05: begin mode_noack <= 1'b1; prx <= 1; end
                  8'h01: req_clear <= 1'b1;
                  8'h03: req_rb <= 1'b1;
                  8'h04: req_st <= 1'b1;
                  8'h06: begin mld_j <= 0; mld_k <= 0; prx <= 6; end
                  8'h07: prx <= 7;
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
            4'd12: begin m_sh       <= rxd[3:0]; req_m <= 1'b1; prx <= 0; end
            default: prx <= 0;
            endcase
        end
        if (ack_clear) req_clear <= 1'b0;
        if (ack_rb)    req_rb <= 1'b0;
        if (ack_st)    req_st <= 1'b0;
        if (ack_m)     req_m <= 1'b0;
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
               E_RB = 5, E_ST = 6, E_MGO = 7, E_MWAIT = 8, E_MRB = 9;
    reg [3:0]  est = E_IDLE;
    reg        ack_clear = 0, ack_rb = 0, ack_st = 0;
    reg        ack_m = 0, mldack_tx = 0;
    // registered TX readiness: the raw (bits==0) compare from uart_tx
    // threaded the whole 10-state priority cone (fmax path). The !send
    // term in each condition covers the pulse cycle; tx_free covers the
    // cycle after (send kills it) through the busy frame (tbusy holds it).
    reg        tx_free = 0;
    reg        clr_ack_due = 0;
    reg [8:0]  ack_due = 0;        // completed points awaiting 0x2A
    // registered (ack_due != 0): settles during the 10-bit-time TX gap
    reg        ack_pend = 0;
    always @(posedge clk) ack_pend <= (ack_due != 0);
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
        send <= 1'b0;
        start <= 1'b0;
        clear <= 1'b0;
        pop <= 1'b0;
        ack_clear <= 1'b0;
        ack_rb <= 1'b0;
        ack_st <= 1'b0;
        ack_m <= 1'b0;
        m_go <= 1'b0;
        mldack_tx <= 1'b0;
        case (est)
        E_IDLE: begin
            if (clr_ack_due && tx_free && !send) begin
                txd <= 8'h2B; send <= 1'b1; clr_ack_due <= 1'b0;
            end else if (mld_ack && tx_free && !send) begin
                txd <= 8'h2C; send <= 1'b1; mldack_tx <= 1'b1;
            end else if (ack_pend && tx_free && !send) begin
                txd <= 8'h2A; send <= 1'b1; ack_due <= ack_due - 1;
            end else if (req_clear && fifo_empty && !ebusy) begin
                clear <= 1'b1; ack_clear <= 1'b1; npts <= 0; est <= E_CLR;
            end else if (!fifo_empty && !ebusy) begin
                pop <= 1'b1; est <= E_POP;
            end else if (req_rb && fifo_empty && !ebusy && !ack_pend
                         && !clr_ack_due) begin
                ack_rb <= 1'b1; rb_j <= 0; rb_k <= 0; rd_idx <= 0;
                rb_wait <= 0; rb_b <= 0; est <= E_RB;
            end else if (req_st && fifo_empty && !ebusy && !ack_pend
                         && !clr_ack_due) begin
                ack_st <= 1'b1; st_hi <= 0; est <= E_ST;
            end else if (req_m && fifo_empty && !ebusy && !ack_pend
                         && !clr_ack_due && mld_cnt == 0) begin
                ack_m <= 1'b1; est <= E_MGO;
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
            if (!mode_noack) ack_due <= ack_due + 1;
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
        E_MGO: begin
            m_go <= 1'b1;               // encoder is idle (guarded above)
            est <= E_MWAIT;
        end
        E_MWAIT: if (!m_go && !ebusy) begin
            // busy-based wait: the encoder raises busy the cycle after
            // the m_go pulse, so !m_go gates the pulse cycle. (The sticky
            // m_done from the PREVIOUS match raced here: ring 0 streamed
            // stale scores while rings 1-3 were saved only by the UART
            // being slower than the new match — caught by hw-match.)
            s_sel <= 2'd0; rb_b <= 0;
            est <= E_MRB;
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
        default: est <= E_IDLE;
        endcase
    end

    reg [23:0] hb = 0;
    always @(posedge clk) hb <= hb + 1;
    assign LEDR_N = ~hb[23];
    assign LEDG_N = ~ebusy;
endmodule
