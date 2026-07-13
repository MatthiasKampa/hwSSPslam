// Hw-in-loop encoder: UART protocol wrapping encoder.v (1 Mbaud).
//   0x01                          -> clear accumulators, ack 0x2B
//   0x02 az_lo az_hi r_lo r_hi w  -> encode one point, ack 0x2A on completion
//   0x03                          -> stream 240 x (re i32 LE, im i32 LE)
//   0x04                          -> point count (lo, hi)
// RX parser runs ALWAYS (no byte is ever dropped); points land in a
// 256-deep FIFO the encoder drains. Host keeps chunks <= 128 points
// outstanding. 0x03/0x04 are honored once the FIFO is drained and all
// acks are out (host protocol is phased, so this is safe).
module top (
    input  wire CLK,               // 12 MHz
    input  wire RX,
    output wire TX,
    output wire LEDR_N,
    output wire LEDG_N
);
    localparam UDIV = 4;               // 12 MHz / 4 = 3 Mbaud (FT2232H native)
    wire [7:0] rxd;
    wire       rxv;
    uart_rx #(.DIV(UDIV)) u_rx (.clk(CLK), .rx(RX), .data(rxd), .valid(rxv));
    reg  [7:0] txd;
    reg        send = 0;
    wire       tbusy;
    uart_tx #(.DIV(UDIV)) u_tx (.clk(CLK), .data(txd), .send(send), .tx(TX),
                                .busy(tbusy));

    reg        clear = 0, start = 0;
    reg [9:0]  az;
    reg [15:0] r_mm;
    reg [7:0]  w;
    wire       ebusy;
    reg [7:0]  rd_idx;
    wire signed [31:0] rd_re, rd_im;
    encoder enc (.clk(CLK), .clear(clear), .start(start), .az(az),
                 .r_mm(r_mm), .w(w), .busy(ebusy), .rd_idx(rd_idx),
                 .rd_re(rd_re), .rd_im(rd_im));

    // ---- RX parser: always consumes, never blocks ----
    // 0x05 = point WITHOUT completion ack (bulk replay mode: the host
    // streams the whole scan in one write and polls 0x04 instead — at
    // 3 Mbaud the encoder outruns the wire, so the FIFO never fills)
    reg [2:0]  prx = 0;
    reg [9:0]  p_az;
    reg [15:0] p_r;
    reg        p_noack;
    reg        req_clear = 0, req_rb = 0, req_st = 0;
    reg        push = 0;
    reg [33:0] pdata;
    always @(posedge CLK) begin
        push <= 1'b0;
        if (rxv) begin
            case (prx)
            3'd0: case (rxd)
                  8'h02: begin p_noack <= 1'b0; prx <= 1; end
                  8'h05: begin p_noack <= 1'b1; prx <= 1; end
                  8'h01: req_clear <= 1'b1;
                  8'h03: req_rb <= 1'b1;
                  8'h04: req_st <= 1'b1;
                  default: ;
                  endcase
            3'd1: begin p_az[7:0]  <= rxd;      prx <= 2; end
            3'd2: begin p_az[9:8]  <= rxd[1:0]; prx <= 3; end
            3'd3: begin p_r[7:0]   <= rxd;      prx <= 4; end
            3'd4: begin p_r[15:8]  <= rxd;      prx <= 5; end
            3'd5: begin
                pdata <= {p_noack, p_az, p_r, rxd[6:0]};
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

    // ---- point FIFO (256 deep) ----
    reg [33:0] fmem [0:255];
    reg [7:0]  fwr = 0, frd = 0;
    reg [8:0]  fcnt = 0;
    wire fifo_empty = (fcnt == 0);
    reg        pop = 0;
    reg [33:0] fq;
    always @(posedge CLK) begin
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
    reg        noack_q = 0;
    reg [15:0] npts = 0;
    reg [3:0]  rb_b;
    reg [7:0]  rb_i;
    reg [1:0]  rb_wait;
    reg [63:0] rb_sh;
    reg        st_hi = 0;

    always @(posedge CLK) begin
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
            end else if (ack_due != 0 && !tbusy && !send) begin
                txd <= 8'h2A; send <= 1'b1; ack_due <= ack_due - 1;
            end else if (req_clear && fifo_empty && !ebusy) begin
                clear <= 1'b1; ack_clear <= 1'b1; npts <= 0; est <= E_CLR;
            end else if (!fifo_empty && !ebusy) begin
                pop <= 1'b1; est <= E_POP;
            end else if (req_rb && fifo_empty && !ebusy && ack_due == 0
                         && !clr_ack_due) begin
                ack_rb <= 1'b1; rb_i <= 0; rd_idx <= 0; rb_wait <= 0;
                rb_b <= 0; est <= E_RB;
            end else if (req_st && fifo_empty && !ebusy && ack_due == 0
                         && !clr_ack_due) begin
                ack_st <= 1'b1; st_hi <= 0; est <= E_ST;
            end
        end
        E_POP: est <= E_GO;                     // fq valid next cycle
        E_GO: begin
            az   <= fq[32:23];
            r_mm <= fq[22:7];
            w    <= {1'b0, fq[6:0]};
            noack_q <= fq[33];
            start <= 1'b1;
            est <= E_RUN;
        end
        E_RUN: if (!start && !ebusy) begin      // point done
            npts <= npts + 1;
            if (!noack_q) ack_due <= ack_due + 1;
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
                    rd_idx <= rb_i + 1;
                    rb_i   <= rb_i + 1;
                    if (rb_i == 239) est <= E_IDLE;
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
    always @(posedge CLK) hb <= hb + 1;
    assign LEDR_N = ~hb[23];
    assign LEDG_N = ~ebusy;
endmodule
