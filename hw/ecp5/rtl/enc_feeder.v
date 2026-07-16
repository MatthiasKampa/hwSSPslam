// enc_feeder.v — bridges stream_ingest's payload events to the encoder
// core (STREAM.md 0x92/0x93 extension). Single-ring (n_rings==1) LIDAR
// frames are ENCODED ON CHIP: HDR commit clears the accumulators; each
// committed LIDAR_COL packet replays its columns as (az, r_mm, w=127)
// points (skipping misses r==0 and out-of-mask r>31200 — the golden
// scan_to_ints law); when the frame completes, vec_req hands the
// accumulator readback to stream_ingest's VEC TX (which also quantizes
// the 2-bit QPSK codes into the on-chip map bank). Corrupt packets never
// reach the encoder: replay starts only on pkt_commit (store-and-forward
// per packet in a ping-pong buffer; digests in stream_ingest roll back
// independently).
`default_nettype none

module enc_feeder (
    input  wire        clk,
    // payload events from stream_ingest
    input  wire [7:0]  pl_type,
    input  wire [15:0] pl_idx,
    input  wire [7:0]  pl_byte,
    input  wire        pl_valid,
    input  wire        pkt_commit,
    input  wire        lid_done,
    input  wire [31:0] lid_fid,
    // encoder core
    output reg         enc_clear,
    output reg         enc_start,
    output reg  [9:0]  enc_az,
    output reg  [15:0] enc_r,
    output wire [7:0]  enc_w,
    input  wire        enc_busy,
    // VEC handoff to stream_ingest TX
    output reg         vec_req,
    output reg  [31:0] vec_fid,
    input  wire        vec_ack
);
    assign enc_w = 8'd127;                 // unit weights (golden w_unit)
    localparam [15:0] R_MASK = 16'd31200;  // golden R_MASK_MM
    initial vec_req = 1'b0;                // (x here deadlocks the !vec_req
                                           //  guard — found by sim probe)

    // ---- ping-pong packet buffers (2 x 2048 B) -------------------------
    reg [7:0] buf0 [0:2047];
    reg [7:0] buf1 [0:2047];
    reg       wr_bank = 0;
    reg [7:0] b0_q, b1_q;
    reg [10:0] rd_a;
    always @(posedge clk) begin
        b0_q <= buf0[rd_a];
        b1_q <= buf1[rd_a];
        if (pl_valid && pl_type == 8'h02 && pl_idx < 16'd2048) begin
            if (wr_bank) buf1[pl_idx[10:0]] <= pl_byte;
            else         buf0[pl_idx[10:0]] <= pl_byte;
        end
    end
    reg        rd_bank = 0;
    wire [7:0] rd_q = rd_bank ? b1_q : b0_q;

    // ---- HDR capture (n_rings gate) ------------------------------------
    reg [7:0] nr_tmp = 0;
    reg       nr_ok = 0;                   // current frame is single-ring
    always @(posedge clk)
        if (pl_valid && pl_type == 8'h01 && pl_idx == 16'd12)
            nr_tmp <= pl_byte;

    // ---- replay FSM -----------------------------------------------------
    // EBR read discipline: an address ISSUED at edge k is CONSUMED at
    // edge k+2 (rd_a -> q -> reg). Every state below issues two edges
    // ahead of its consumer.
    localparam IDLE = 0, CLR = 1, R1 = 2, R2 = 3, R3 = 4, R4 = 5,
               R5 = 6, R6 = 7, WGO = 8, WBUSY = 9, NEXT = 10, N2 = 11;
    reg [3:0]  fst = IDLE;
    reg [15:0] az0;
    reg [7:0]  ncols, ci;
    reg [7:0]  rlo;
    reg [15:0] r_cur;
    reg        pend_done = 0, pend_clear = 0;
    reg        pend_col = 0, pend_bank = 0;
    reg [7:0]  ovr_drop = 0;
    reg [31:0] done_fid;

    always @(posedge clk) begin
        enc_start <= 1'b0;
        enc_clear <= 1'b0;
        if (vec_ack) vec_req <= 1'b0;
        if (lid_done) begin pend_done <= 1'b1; done_fid <= lid_fid; end

        // commits are latched HERE (any fst state — the serial encoder
        // core takes ~840 cyc/point, so a multi-col replay can outlast
        // the next packet's arrival; missing its commit dropped half the
        // real-scan points, found on silicon 2026-07-16). One-deep queue
        // + ping-pong banks: senders keep n_cols <= 8 at 2 Mbaud so the
        // queue never overflows; overflows are counted, never silent.
        if (pkt_commit && pl_type == 8'h01) begin
            nr_ok <= (nr_tmp == 8'd1);
            if (nr_tmp == 8'd1) pend_clear <= 1'b1;
        end
        if (pkt_commit && pl_type == 8'h02 && nr_ok) begin
            if (!pend_col) begin
                pend_col <= 1'b1;
                pend_bank <= wr_bank;
                wr_bank <= ~wr_bank;     // next packet captures elsewhere
            end else
                ovr_drop <= ovr_drop + 1;
        end

        case (fst)
          IDLE: begin
              if (pend_col) begin          // queued packet replays first
                  rd_bank <= pend_bank;
                  pend_col <= 1'b0;
                  rd_a <= 11'd4;           // issue az0 lo
                  fst <= R1;
              end else if (pend_clear && !enc_busy) begin
                  enc_clear <= 1'b1;       // 240-cycle accumulator clear
                  pend_clear <= 1'b0;
                  fst <= CLR;
              end else if (pend_done && !vec_req && !enc_busy
                           && !pend_clear) begin
                  vec_fid <= done_fid;
                  vec_req <= 1'b1;
                  pend_done <= 1'b0;
              end
          end
          CLR: if (!enc_clear && !enc_busy) fst <= IDLE;
          R1: begin rd_a <= 11'd5; fst <= R2; end
          R2: begin az0[7:0] <= rd_q; rd_a <= 11'd6; fst <= R3; end
          R3: begin az0[15:8] <= rd_q; rd_a <= 11'd7; fst <= R4; end
          R4: begin ncols <= rd_q; rd_a <= 11'd8; ci <= 0; fst <= R5; end
          R5: begin rlo <= rd_q; fst <= R6; end
          R6: begin r_cur <= {rd_q, rlo}; fst <= WGO; end
          WGO: begin
              if (r_cur != 16'd0 && r_cur <= R_MASK) begin
                  if (!enc_busy) begin
                      enc_az <= az0[9:0] + {2'd0, ci};
                      enc_r <= r_cur;
                      enc_start <= 1'b1;
                      fst <= WBUSY;
                  end
              end else
                  fst <= NEXT;             // skipped point
          end
          WBUSY: if (!enc_start && !enc_busy) fst <= NEXT;
          NEXT: begin
              if (ci + 8'd1 == ncols) fst <= IDLE;
              else begin
                  ci <= ci + 1;
                  rd_a <= 11'd7 + {ci + 8'd1, 1'b0};   // issue next lo
                  fst <= N2;
              end
          end
          N2: begin
              rd_a <= 11'd8 + {ci, 1'b0};              // issue next hi
              fst <= R5;
          end
          default: fst <= IDLE;
        endcase
    end
endmodule
