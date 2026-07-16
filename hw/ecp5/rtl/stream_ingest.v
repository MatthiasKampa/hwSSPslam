// stream_ingest.v — virtual-sensor packet ingest (STREAM.md, v0).
// Byte-at-a-time parser: magic hunt -> header -> payload dispatched on the
// fly -> CRC16 check. SHADOW-COMMIT: per-stream digest/meta registers are
// snapshotted at header time and rolled back on CRC failure, so a corrupt
// packet leaves no trace (no payload buffering, zero EBR).
// TX: ECHO_DIGEST on frame completion, STATUS on request, CREDIT at start.
`default_nettype none

module stream_ingest (
    input  wire       clk,           // 50 MHz
    input  wire [7:0] rx_data,
    input  wire       rx_valid,
    output reg  [7:0] tx_data,
    output reg        tx_send,
    input  wire       tx_busy,
    output wire [3:0] led,
    // ---- payload event interface (v0 tops leave these dangling) ----
    output wire [7:0]  pl_type,      // packet type during PAY/commit
    output reg  [15:0] pl_idx,       // payload byte index (regd w/ pl_byte)
    output reg  [7:0]  pl_byte,
    output reg         pl_valid,     // one pulse per payload byte
    output reg         pkt_commit,   // CRC pass (pl_type valid)
    output reg         pkt_abort,    // CRC fail
    output reg         lid_done,     // lidar frame completed (commit cyc)
    output wire [31:0] lid_fid,
    // ---- VEC readback TX (type 0x92): silicon-encoded VSA vector ----
    input  wire        vec_req,      // level; hold until vec_ack
    input  wire [31:0] vec_fid,
    output reg         vec_ack,      // pulse: request latched
    output reg  [7:0]  vec_rd,       // encoder rd_idx
    input  wire signed [31:0] vec_re,
    input  wire signed [31:0] vec_im
);
    // ---- CRC16-CCITT (poly 0x1021, init 0xFFFF), one byte per step ----
    function [15:0] crc16_up(input [15:0] c, input [7:0] d);
        integer i;
        reg [15:0] x;
        begin
            x = c ^ {d, 8'h00};
            for (i = 0; i < 8; i = i + 1)
                x = x[15] ? ((x << 1) ^ 16'h1021) : (x << 1);
            crc16_up = x;
        end
    endfunction

    // ---- RX parser -----------------------------------------------------
    localparam HUNT_A5 = 0, HUNT_5A = 1, H_TYPE = 2, H_FLAGS = 3,
               H_LEN0 = 4, H_LEN1 = 5, H_SEQ0 = 6, H_SEQ1 = 7,
               PAY = 8, CRC0 = 9, CRC1 = 10;
    reg [3:0]  st = HUNT_A5;
    reg [7:0]  ptype;
    reg [15:0] plen, pseq, pidx;
    reg [15:0] crc, rxcrc;
    reg [15:0] last_seq = 16'hFFFF;
    reg        seq_seen = 1'b0;

    // counters (STATUS)
    reg [31:0] pkts_ok = 0;
    reg [15:0] crc_drops = 0, seq_gaps = 0, cam_frames = 0, lidar_frames = 0;

    // per-stream digest/meta state (+ shadow copies for rollback)
    reg [15:0] dg_cam = 16'hFFFF,  dg_lid = 16'hFFFF;
    reg [31:0] cnt_cam = 0,        cnt_lid = 0;
    reg [31:0] id_cam = 0,         id_lid = 0;
    reg [31:0] tot_cam = 0,        tot_lid = 0;   // expected data bytes
    reg [15:0] sh_dg_cam, sh_dg_lid;
    reg [31:0] sh_cnt_cam, sh_cnt_lid, sh_id_cam, sh_id_lid,
               sh_tot_cam, sh_tot_lid;

    // in-payload field decode scratch
    reg [31:0] f_id;                 // frame_id from payload head
    reg [7:0]  f_nrings = 3, f_fmt = 0;
    reg [15:0] f_w = 0, f_naz = 0, f_h = 0;
    reg [7:0]  ctrl_cmd, ctrl_arg0;
    reg        echo_en = 1'b1;

    // completion events -> TX
    reg        ev_cam = 0, ev_lid = 0, ev_status = 0, ev_credit = 0;
    reg [31:0] ev_cam_id, ev_lid_id, ev_cam_cnt, ev_lid_cnt;
    reg [15:0] ev_cam_dg, ev_lid_dg;

    assign pl_type = ptype;
    assign lid_fid = id_lid;
    // MAP_READ (type 0x0F): fetch a stored segment's 2-bit codes
    reg        ev_mr = 0;                  // RX-side pulse -> TX latches
    reg [7:0]  mr_seg, mr_tmp;

    always @(posedge clk) begin
        ev_cam <= 1'b0; ev_lid <= 1'b0; ev_status <= 1'b0; ev_mr <= 1'b0;
        pl_valid <= 1'b0; pkt_commit <= 1'b0; pkt_abort <= 1'b0;
        lid_done <= 1'b0;
        if (rx_valid) begin
            case (st)
              HUNT_A5: if (rx_data == 8'hA5) st <= HUNT_5A;
              HUNT_5A: st <= (rx_data == 8'h5A) ? H_TYPE :
                             (rx_data == 8'hA5) ? HUNT_5A : HUNT_A5;
              H_TYPE: begin
                  ptype <= rx_data;
                  crc   <= crc16_up(16'hFFFF, rx_data);
                  // snapshot for rollback
                  sh_dg_cam <= dg_cam;  sh_dg_lid <= dg_lid;
                  sh_cnt_cam <= cnt_cam; sh_cnt_lid <= cnt_lid;
                  sh_id_cam <= id_cam;  sh_id_lid <= id_lid;
                  sh_tot_cam <= tot_cam; sh_tot_lid <= tot_lid;
                  st <= H_FLAGS;
              end
              H_FLAGS: begin crc <= crc16_up(crc, rx_data); st <= H_LEN0; end
              H_LEN0: begin
                  plen[7:0] <= rx_data; crc <= crc16_up(crc, rx_data);
                  st <= H_LEN1;
              end
              H_LEN1: begin
                  plen[15:8] <= rx_data; crc <= crc16_up(crc, rx_data);
                  st <= H_SEQ0;
              end
              H_SEQ0: begin
                  pseq[7:0] <= rx_data; crc <= crc16_up(crc, rx_data);
                  st <= H_SEQ1;
              end
              H_SEQ1: begin
                  pseq[15:8] <= rx_data; crc <= crc16_up(crc, rx_data);
                  pidx <= 0;
                  if (plen > 16'd4096) st <= HUNT_A5;   // framing error
                  else st <= (plen != 0) ? PAY : CRC0;
              end
              PAY: begin
                  crc <= crc16_up(crc, rx_data);
                  pl_byte <= rx_data;
                  pl_idx <= pidx;          // registered WITH the byte
                  pl_valid <= 1'b1;
                  // ---- on-the-fly dispatch by (ptype, pidx) ----------
                  case (ptype)
                    8'h01: begin                       // LIDAR_HDR
                        if (pidx == 0)  f_id[7:0]    <= rx_data;
                        if (pidx == 1)  f_id[15:8]   <= rx_data;
                        if (pidx == 2)  f_id[23:16]  <= rx_data;
                        if (pidx == 3)  f_id[31:24]  <= rx_data;
                        if (pidx == 12) f_nrings     <= rx_data;
                        // ring_ids consumed pidx 13..12+n (recorded only)
                        if (pidx == 13 + {8'd0, f_nrings})
                            f_naz[7:0]  <= rx_data;
                        if (pidx == 14 + {8'd0, f_nrings})
                            f_naz[15:8] <= rx_data;
                        if (pidx == 15 + {8'd0, f_nrings}) begin
                            f_fmt <= rx_data;
                            id_lid  <= f_id;
                            dg_lid  <= 16'hFFFF;
                            cnt_lid <= 0;
                            tot_lid <= {16'd0, f_naz} * {24'd0, f_nrings}
                                       * (rx_data[1] ? 32'd3 : 32'd2);
                        end
                    end
                    8'h02: begin                       // LIDAR_COL
                        if (pidx >= 7) begin           // data bytes
                            dg_lid  <= crc16_up(dg_lid, rx_data);
                            cnt_lid <= cnt_lid + 1;
                        end
                    end
                    8'h03: begin                       // CAM_HDR
                        if (pidx == 0)  f_id[7:0]   <= rx_data;
                        if (pidx == 1)  f_id[15:8]  <= rx_data;
                        if (pidx == 2)  f_id[23:16] <= rx_data;
                        if (pidx == 3)  f_id[31:24] <= rx_data;
                        if (pidx == 12) f_w[7:0]    <= rx_data;
                        if (pidx == 13) f_w[15:8]   <= rx_data;
                        if (pidx == 14) f_h[7:0]    <= rx_data;
                        if (pidx == 15) f_h[15:8]   <= rx_data;
                        if (pidx == 16) begin          // fmt byte = last
                            id_cam  <= f_id;
                            dg_cam  <= 16'hFFFF;
                            cnt_cam <= 0;
                            tot_cam <= {16'd0, f_w} * {16'd0, f_h};
                        end
                    end
                    8'h04: begin                       // CAM_ROW
                        if (pidx >= 7) begin
                            dg_cam  <= crc16_up(dg_cam, rx_data);
                            cnt_cam <= cnt_cam + 1;
                        end
                    end
                    8'h10: begin                       // CTRL
                        if (pidx == 0) ctrl_cmd <= rx_data;
                        if (pidx == 1) ctrl_arg0 <= rx_data;
                    end
                    8'h0F: begin                       // MAP_READ {seg u8}
                        if (pidx == 0) mr_tmp <= rx_data;
                    end
                    default: ;
                  endcase
                  pidx <= pidx + 1;
                  if (pidx == plen - 1) st <= CRC0;
              end
              CRC0: begin rxcrc[7:0] <= rx_data; st <= CRC1; end
              CRC1: begin
                  st <= HUNT_A5;
                  if ({rx_data, rxcrc[7:0]} == crc) begin
                      // -------- COMMIT --------
                      pkt_commit <= 1'b1;
                      pkts_ok <= pkts_ok + 1;
                      if (seq_seen && pseq != last_seq + 16'd1)
                          seq_gaps <= seq_gaps + 1;
                      last_seq <= pseq; seq_seen <= 1'b1;
                      if (ptype == 8'h04 && cnt_cam != 0 &&
                          cnt_cam == tot_cam) begin
                          ev_cam <= echo_en; ev_cam_id <= id_cam;
                          ev_cam_dg <= dg_cam; ev_cam_cnt <= cnt_cam;
                          cam_frames <= cam_frames + 1;
                      end
                      if (ptype == 8'h02 && cnt_lid != 0 &&
                          cnt_lid == tot_lid) begin
                          ev_lid <= echo_en; ev_lid_id <= id_lid;
                          ev_lid_dg <= dg_lid; ev_lid_cnt <= cnt_lid;
                          lidar_frames <= lidar_frames + 1;
                          lid_done <= 1'b1;
                      end
                      if (ptype == 8'h0F) begin
                          ev_mr <= 1'b1; mr_seg <= mr_tmp;
                      end
                      if (ptype == 8'h10) begin
                          if (ctrl_cmd == 8'd0) begin
                              crc_drops <= 0; seq_gaps <= 0;
                              cam_frames <= 0; lidar_frames <= 0;
                              pkts_ok <= 0; seq_seen <= 1'b0;
                          end
                          if (ctrl_cmd == 8'd1) ev_status <= 1'b1;
                          if (ctrl_cmd == 8'd2) echo_en <= ctrl_arg0[0];
                      end
                  end else begin
                      // -------- ROLLBACK --------
                      pkt_abort <= 1'b1;
                      crc_drops <= crc_drops + 1;
                      dg_cam <= sh_dg_cam;  dg_lid <= sh_dg_lid;
                      cnt_cam <= sh_cnt_cam; cnt_lid <= sh_cnt_lid;
                      id_cam <= sh_id_cam;  id_lid <= sh_id_lid;
                      tot_cam <= sh_tot_cam; tot_lid <= sh_tot_lid;
                  end
              end
              default: st <= HUNT_A5;
            endcase
        end
    end

    // ---- TX packetizer (ECHO/STATUS buffered; VEC + MAP_SEG streamed) --
    // fixed priority: buffered msgs > VEC (0x92) > MAP_SEG (0x93)
    reg        p_cam = 0, p_lid = 0, p_stat = 0;
    reg [31:0] q_cam_id, q_lid_id, q_cam_cnt, q_lid_cnt;
    reg [15:0] q_cam_dg, q_lid_dg;
    reg [7:0]  txbuf [0:22];
    reg [4:0]  txn = 0, txi = 0;
    reg [15:0] txcrc;
    reg [15:0] txseq = 0;
    reg        sending = 0;

    // on-chip MAP: 64 segments x 240 components x 2-bit QPSK codes,
    // written as the VEC streams out (quantize-on-readback), read back
    // by MAP_READ. Segment slot = frame_id % 64.
    reg [1:0]  mcode [0:15359];
    reg [1:0]  mc_q;
    reg        mc_we;
    reg [13:0] mc_wa, mc_ra;
    reg [1:0]  mc_wd;
    always @(posedge clk) begin
        mc_q <= mcode[mc_ra];
        if (mc_we) mcode[mc_wa] <= mc_wd;
    end

    localparam V_HDR = 1, V_SET = 2, V_W1 = 3, V_W2 = 4, V_B = 5,
               V_CRC = 6, M_HDR = 7, M_SEG = 8, M_RD = 9, M_B = 10,
               M_CRC = 11;
    reg [3:0]  tst = 0;                    // 0 = idle (buffered path free)
    reg [7:0]  comp;
    reg [63:0] sh64;
    reg [13:0] vbase, mbase;
    reg [31:0] vfid_q;
    reg [3:0]  hi;                         // header/byte sub-index
    reg [1:0]  mk;
    reg [7:0]  mbyte;
    reg [5:0]  mi;
    reg [1:0]  mph = 0;
    reg        pm_req = 0;
    reg [7:0]  pm_seg;
    wire [1:0] qcode = (( (vec_im[31] ? -vec_im : vec_im)
                        > (vec_re[31] ? -vec_re : vec_re))
                        ? (vec_im[31] ? 2'd3 : 2'd1)
                        : (vec_re[31] ? 2'd2 : 2'd0));

    integer k;
    always @(posedge clk) begin
        tx_send <= 1'b0;
        vec_ack <= 1'b0;
        mc_we <= 1'b0;
        if (ev_cam)  begin p_cam <= 1; q_cam_id <= ev_cam_id;
                           q_cam_dg <= ev_cam_dg; q_cam_cnt <= ev_cam_cnt; end
        if (ev_lid)  begin p_lid <= 1; q_lid_id <= ev_lid_id;
                           q_lid_dg <= ev_lid_dg; q_lid_cnt <= ev_lid_cnt; end
        if (ev_status) p_stat <= 1;

        if (!sending && tst == 0 && (p_cam || p_lid || p_stat)) begin
            txbuf[0] <= 8'hA5; txbuf[1] <= 8'h5A;
            txbuf[4] <= (p_cam || p_lid) ? 8'd11 : 8'd12;   // payload len
            txbuf[5] <= 8'd0;
            txbuf[6] <= txseq[7:0]; txbuf[7] <= txseq[15:8];
            txseq <= txseq + 1;
            if (p_cam || p_lid) begin
                txbuf[2] <= 8'h90; txbuf[3] <= 8'h00;
                txbuf[8] <= p_cam ? 8'd3 : 8'd1;         // stream tag
                if (p_cam) begin
                    txbuf[9]  <= q_cam_id[7:0];   txbuf[10] <= q_cam_id[15:8];
                    txbuf[11] <= q_cam_id[23:16]; txbuf[12] <= q_cam_id[31:24];
                    txbuf[13] <= q_cam_dg[7:0];   txbuf[14] <= q_cam_dg[15:8];
                    txbuf[15] <= q_cam_cnt[7:0];  txbuf[16] <= q_cam_cnt[15:8];
                    txbuf[17] <= q_cam_cnt[23:16];
                    txbuf[18] <= q_cam_cnt[31:24];
                    p_cam <= 0;
                end else begin
                    txbuf[9]  <= q_lid_id[7:0];   txbuf[10] <= q_lid_id[15:8];
                    txbuf[11] <= q_lid_id[23:16]; txbuf[12] <= q_lid_id[31:24];
                    txbuf[13] <= q_lid_dg[7:0];   txbuf[14] <= q_lid_dg[15:8];
                    txbuf[15] <= q_lid_cnt[7:0];  txbuf[16] <= q_lid_cnt[15:8];
                    txbuf[17] <= q_lid_cnt[23:16];
                    txbuf[18] <= q_lid_cnt[31:24];
                    p_lid <= 0;
                end
            end else begin
                txbuf[2] <= 8'h80; txbuf[3] <= 8'h00;
                txbuf[8]  <= pkts_ok[7:0];    txbuf[9]  <= pkts_ok[15:8];
                txbuf[10] <= pkts_ok[23:16];  txbuf[11] <= pkts_ok[31:24];
                txbuf[12] <= crc_drops[7:0];  txbuf[13] <= crc_drops[15:8];
                txbuf[14] <= seq_gaps[7:0];   txbuf[15] <= seq_gaps[15:8];
                txbuf[16] <= cam_frames[7:0]; txbuf[17] <= cam_frames[15:8];
                txbuf[18] <= lidar_frames[7:0];
                txbuf[19] <= lidar_frames[15:8];
                p_stat <= 0;
            end
            txn <= (p_cam || p_lid) ? 5'd19 : 5'd20;   // bytes before CRC
            txi <= 0;
            txcrc <= 16'hFFFF;
            sending <= 1;
        end else if (sending && !tx_busy && !tx_send) begin
            if (txi < txn) begin
                tx_data <= txbuf[txi];
                tx_send <= 1'b1;
                if (txi >= 2)                 // CRC over type..payload
                    txcrc <= crc16_up(txcrc, txbuf[txi]);
                txi <= txi + 1;
            end else if (txi == txn) begin
                tx_data <= txcrc[7:0]; tx_send <= 1'b1; txi <= txi + 1;
            end else begin
                tx_data <= txcrc[15:8]; tx_send <= 1'b1;
                sending <= 0;
            end
        end

        // ---- streamed TX: VEC (0x92) then MAP_SEG (0x93) ---------------
        if (ev_mr) begin pm_req <= 1'b1; pm_seg <= mr_seg; end
        if (!sending && tst == 0 && !(p_cam || p_lid || p_stat)) begin
            if (vec_req) begin
                vfid_q <= vec_fid;
                vbase <= {vec_fid[5:0], 8'd0} - {8'd0, vec_fid[5:0], 4'd0};
                vec_ack <= 1'b1;
                txcrc <= 16'hFFFF; hi <= 0; comp <= 0;
                tst <= V_HDR;
            end else if (pm_req) begin
                pm_req <= 1'b0;
                mbase <= {pm_seg[5:0], 8'd0} - {8'd0, pm_seg[5:0], 4'd0};
                txcrc <= 16'hFFFF; hi <= 0; mi <= 0; mk <= 0; mph <= 0;
                mbyte <= 0;
                tst <= M_HDR;
            end
        end
        case (tst)
          V_HDR: if (!tx_busy && !tx_send) begin
              case (hi)
                4'd0: tx_data <= 8'hA5;
                4'd1: tx_data <= 8'h5A;
                4'd2: tx_data <= 8'h92;
                4'd3: tx_data <= 8'h00;
                4'd4: tx_data <= 8'h84;      // len 1924 = 0x0784 LE
                4'd5: tx_data <= 8'h07;
                4'd6: tx_data <= txseq[7:0];
                4'd7: tx_data <= txseq[15:8];
                4'd8: tx_data <= vfid_q[7:0];
                4'd9: tx_data <= vfid_q[15:8];
                4'd10: tx_data <= vfid_q[23:16];
                default: tx_data <= vfid_q[31:24];
              endcase
              tx_send <= 1'b1;
              if (hi >= 2)
                  txcrc <= crc16_up(txcrc,
                      hi == 2 ? 8'h92 : hi == 3 ? 8'h00 :
                      hi == 4 ? 8'h84 : hi == 5 ? 8'h07 :
                      hi == 6 ? txseq[7:0] : hi == 7 ? txseq[15:8] :
                      hi == 8 ? vfid_q[7:0] : hi == 9 ? vfid_q[15:8] :
                      hi == 10 ? vfid_q[23:16] : vfid_q[31:24]);
              if (hi == 4'd11) begin
                  txseq <= txseq + 1;
                  tst <= V_SET;
              end
              hi <= hi + 1;
          end
          V_SET: begin vec_rd <= comp; tst <= V_W1; end
          V_W1: tst <= V_W2;                 // rd_re/rd_im settle
          V_W2: begin
              sh64 <= {vec_im, vec_re};
              mc_we <= 1'b1;
              mc_wa <= vbase + {6'd0, comp};
              mc_wd <= qcode;
              hi <= 0;
              tst <= V_B;
          end
          V_B: if (!tx_busy && !tx_send) begin
              tx_data <= sh64[8 * hi[2:0] +: 8];
              tx_send <= 1'b1;
              txcrc <= crc16_up(txcrc, sh64[8 * hi[2:0] +: 8]);
              if (hi[2:0] == 3'd7) begin
                  if (comp == 8'd239) begin hi <= 0; tst <= V_CRC; end
                  else begin comp <= comp + 1; tst <= V_SET; end
              end else
                  hi <= hi + 1;
          end
          V_CRC: if (!tx_busy && !tx_send) begin
              tx_data <= (hi == 0) ? txcrc[7:0] : txcrc[15:8];
              tx_send <= 1'b1;
              if (hi != 0) tst <= 0;
              hi <= hi + 1;
          end
          M_HDR: if (!tx_busy && !tx_send) begin
              case (hi)
                4'd0: tx_data <= 8'hA5;
                4'd1: tx_data <= 8'h5A;
                4'd2: tx_data <= 8'h93;
                4'd3: tx_data <= 8'h00;
                4'd4: tx_data <= 8'h3D;      // len 61
                4'd5: tx_data <= 8'h00;
                4'd6: tx_data <= txseq[7:0];
                4'd7: tx_data <= txseq[15:8];
                default: tx_data <= pm_seg;  // payload byte 0
              endcase
              tx_send <= 1'b1;
              if (hi >= 2)
                  txcrc <= crc16_up(txcrc,
                      hi == 2 ? 8'h93 : hi == 3 ? 8'h00 :
                      hi == 4 ? 8'h3D : hi == 5 ? 8'h00 :
                      hi == 6 ? txseq[7:0] : hi == 7 ? txseq[15:8] :
                      pm_seg);
              if (hi == 4'd8) begin
                  txseq <= txseq + 1;
                  mc_ra <= mbase;
                  tst <= M_RD;
              end
              hi <= hi + 1;
          end
          M_RD: begin                        // 4 sub-reads -> one byte
              // issue -> wait -> consume (EBR q lags rd_a by 2 edges)
              if (mph == 2'd0) begin
                  mc_ra <= mbase + {6'd0, mi, mk};
                  mph <= 2'd1;
              end else if (mph == 2'd1) begin
                  mph <= 2'd2;
              end else begin
                  mbyte <= mbyte | ({6'd0, mc_q} << {mk, 1'b0});
                  mph <= 2'd0;
                  if (mk == 2'd3) tst <= M_B;
                  mk <= mk + 1;
              end
          end
          M_B: if (!tx_busy && !tx_send) begin
              tx_data <= mbyte;
              tx_send <= 1'b1;
              txcrc <= crc16_up(txcrc, mbyte);
              mbyte <= 0; mk <= 0; mph <= 0;
              if (mi == 6'd59) begin hi <= 0; tst <= M_CRC; end
              else begin mi <= mi + 1; tst <= M_RD; end
          end
          M_CRC: if (!tx_busy && !tx_send) begin
              tx_data <= (hi == 0) ? txcrc[7:0] : txcrc[15:8];
              tx_send <= 1'b1;
              if (hi != 0) tst <= 0;
              hi <= hi + 1;
          end
          default: ;
        endcase
    end

    assign led = {st != HUNT_A5, crc_drops != 0,
                  cam_frames[0], lidar_frames[0]};
endmodule
