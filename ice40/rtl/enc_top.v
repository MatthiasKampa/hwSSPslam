// Protocol wrapper v2 (module enc_top, clock/UDIV parameterized so thin
// tops can run it at 12 MHz direct or behind a PLL). Same UART protocol
// as top_encoder.v:
//   0x01                          -> clear accumulators, ack 0x2B
//   0x02 az_lo az_hi r_lo r_hi w  -> encode one point, ack 0x2A on completion
//   0x03                          -> stream 240 x (re i32 LE, im i32 LE)
//   0x04                          -> point count (lo, hi)
//   0x05 (as 0x02)                -> point WITHOUT completion ack (bulk)
// vs top_encoder.v: the point FIFO is 32 bits wide (az10 | r15 | w7 —
// r_mm <= 31200 fits 15 bits, the encode mask guarantees it), which
// packs it into 2 EBRs instead of 3; the ack/noack flag no longer rides
// the FIFO — it is a MODE REGISTER set by the command byte. Contract:
// hosts must not interleave 0x02 and 0x05 while points are in flight
// (both vectors.py paths are phased chunks of one kind, so this holds).
module enc_top #(
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
    encoder enc (.clk(clk), .clear(clear), .start(start), .az(az),
                 .r_mm(r_mm), .w(w), .busy(ebusy), .rd_idx(rd_idx),
                 .rd_re(rd_re), .rd_im(rd_im));

    // ---- RX parser: always consumes, never blocks ----
    reg [2:0]  prx = 0;
    reg [9:0]  p_az;
    reg [14:0] p_r;
    reg        mode_noack = 0;         // 0x02 -> acked, 0x05 -> bulk
    reg        req_clear = 0, req_rb = 0, req_st = 0;
    reg        push = 0;
    reg [31:0] pdata;
    always @(posedge clk) begin
        push <= 1'b0;
        if (rxv) begin
            case (prx)
            3'd0: case (rxd)
                  8'h02: begin mode_noack <= 1'b0; prx <= 1; end
                  8'h05: begin mode_noack <= 1'b1; prx <= 1; end
                  8'h01: req_clear <= 1'b1;
                  8'h03: req_rb <= 1'b1;
                  8'h04: req_st <= 1'b1;
                  default: ;
                  endcase
            3'd1: begin p_az[7:0]  <= rxd;      prx <= 2; end
            3'd2: begin p_az[9:8]  <= rxd[1:0]; prx <= 3; end
            3'd3: begin p_r[7:0]   <= rxd;      prx <= 4; end
            3'd4: begin p_r[14:8]  <= rxd[6:0]; prx <= 5; end
            3'd5: begin
                pdata <= {p_az, p_r, rxd[6:0]};
                push  <= 1'b1;
                prx   <= 0;
            end
            default: prx <= 0;
            endcase
        end
        if (ack_clear) req_clear <= 1'b0;
        if (ack_rb)    req_rb <= 1'b0;
        if (ack_st)    req_st <= 1'b0;
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

    // ---- exec FSM: drain fifo, run encoder, arbitrate TX ----
    localparam E_IDLE = 0, E_POP = 1, E_GO = 2, E_RUN = 3, E_CLR = 4,
               E_RB = 5, E_ST = 6;
    reg [2:0]  est = E_IDLE;
    reg        ack_clear = 0, ack_rb = 0, ack_st = 0;
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

    always @(posedge clk) begin
        send <= 1'b0;
        start <= 1'b0;
        clear <= 1'b0;
        pop <= 1'b0;
        ack_clear <= 1'b0;
        ack_rb <= 1'b0;
        ack_st <= 1'b0;
        case (est)
        E_IDLE: begin
            if (clr_ack_due && !tbusy && !send) begin
                txd <= 8'h2B; send <= 1'b1; clr_ack_due <= 1'b0;
            end else if (ack_pend && !tbusy && !send) begin
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
            end else if (!tbusy && !send) begin
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
        E_ST: if (!tbusy && !send) begin
            txd <= st_hi ? npts[15:8] : npts[7:0];
            send <= 1'b1;
            if (st_hi) est <= E_IDLE;
            st_hi <= ~st_hi;
        end
        default: est <= E_IDLE;
        endcase
    end

    reg [23:0] hb = 0;
    always @(posedge clk) hb <= hb + 1;
    assign LEDR_N = ~hb[23];
    assign LEDG_N = ~ebusy;
endmodule
