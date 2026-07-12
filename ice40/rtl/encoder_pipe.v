// v3: encoder_par.v + three pipeline registers (corner T). Same interface,
// same bit-exact contract (golden: ssp_ice40.encode_int). vs v2:
//   - u_q registers the projection (DSP MACs -> round) before the cis
//     ROM address slicing (breaks the proj->cis path),
//   - the 7-tap w*cis shift-add is split into TWO registered partial
//     sums (taps 0-3 / 4-6) then one 16-bit add into pre_q/pim_q —
//     the single-cycle tree was 31 ns of LUT+routing on its own
//     (breaks v2's 38 ns w*cis->RMW critical path twice over).
// Still 1 cycle/angle; the software pipeline is 3 stages deeper
// (writes land at n+5 instead of n+2): ~67 cycles/point.
//
// Per-angle schedule (cycle(c) = S_L with counter n==c; angle j):
//   end cycle(j-1): ang_a <= j            (ROM addr set)
//   end cycle(j):   angc_q <= rom[j]
//   end cycle(j+1): u_q    <= u(j)        (proj computed from angc_q)
//   end cycle(j+2): cre_q  <= cis(j)      (addr = u_q bit-slices)
//   end cycle(j+3): pA/pB  <= tap partials(j)
//   end cycle(j+4): pre_q  <= pA+pB;     are_q <= bank[j]  (acc_a<=n-3)
//   end cycle(j+5): bank[j] <= are_q + pre_q               (wr_a<=n-4)
// Last write: j=NA-1 at end of cycle(NA+4) == the S_L exit edge.
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
                                           // (BIT-SLICED — not the packed
                                           // k*NA+j index of encoder.v/_par.v;
                                           // pairs with enc_top.v's counters)
    output reg  signed [31:0] rd_re,
    output reg  signed [31:0] rd_im
);
    localparam signed HALF = 1 << 13;      // F_AZ = F_ANG = 14

    // ---- ROMs (images from gen_luts.py == golden model) ----
    reg signed [15:0] rom_azc [0:1023];
    reg signed [15:0] rom_azs [0:1023];
    reg signed [15:0] rom_angc[0:63];
    reg signed [15:0] rom_angs[0:63];
    // ram_style: yosys's cost model LUT-maps this 4-read-port ROM (its
    // replication cost estimate), putting a 9-LUT-level cone on the
    // u_q -> cre_q path — force 4 EBR replicas (one per ring port)
    (* ram_style = "block" *)
    reg        [15:0] rom_cis [0:255];      // {im[7:0], re[7:0]} packed
    initial begin
        $readmemh("build/az_c.hex",  rom_azc);
        $readmemh("build/az_s.hex",  rom_azs);
        $readmemh("build/ang_c.hex", rom_angc);
        $readmemh("build/ang_s.hex", rom_angs);
        $readmemh("build/cis.hex",   rom_cis);
    end
    reg signed [15:0] azc_q, azs_q, angc_q, angs_q;
    reg [5:0] ang_a;
    always @(posedge clk) begin
        azc_q  <= rom_azc[az];
        azs_q  <= rom_azs[az];
        angc_q <= rom_angc[ang_a];
        angs_q <= rom_angs[ang_a];
    end

    // projection for the current angle, REGISTERED (pipe stage 1)
    reg signed [15:0] x, y;
    wire signed [32:0] um_c = x * angc_q + y * angs_q;
    reg signed [18:0] u_q;
    always @(posedge clk) u_q <= (um_c + HALF) >>> 14;

    // per-ring cis lookups (bit-slice addresses, sync read, packed ROM)
    reg signed [7:0] cre_q [0:NR-1];
    reg signed [7:0] cim_q [0:NR-1];
    genvar gk;
    generate for (gk = 0; gk < NR; gk = gk + 1) begin : g_cis
        wire signed [18:0] ush = u_q >>> gk;
        always @(posedge clk) begin
            cre_q[gk] <= $signed(rom_cis[ush[7:0]][7:0]);
            cim_q[gk] <= $signed(rom_cis[ush[7:0]][15:8]);
        end
    end endgenerate

    // ---- banked accumulators: per ring, 64 x 32 (re, im) ----
    // readback decode is pure bit-slicing (the v2 packed-index compare/
    // subtract cone was a 14-level fmax limiter through the r_addr mux)
    wire [1:0] rd_k = rd_idx[7:6];
    wire [5:0] rd_j = rd_idx[5:0];
    reg [5:0] acc_a;                        // read addr (angle j)
    reg [5:0] wr_a;                         // write addr (angle j, lagging)
    reg       acc_we;
    reg       clr_we;
    reg [5:0] clr_a;
    reg signed [7:0] w_q;
    wire signed [31:0] rb_re_w [0:NR-1];
    wire signed [31:0] rb_im_w [0:NR-1];
    wire [5:0] r_addr = busy ? acc_a : rd_j;
    wire [5:0] w_addr = wr_a;
    wire       w_en   = acc_we | clr_we;
    generate for (gk = 0; gk < NR; gk = gk + 1) begin : g_acc
        reg signed [31:0] bank_re [0:63];
        reg signed [31:0] bank_im [0:63];
        reg signed [31:0] are_q, aim_q;
        // 8x8 products as explicit 7-tap shift-adds (structural LUT logic;
        // 4 DSPs stay free for the matcher), TWO-STAGE: registered
        // partials (taps 0-3 / 4-6), then one 16-bit add into pre_q
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
            are_q <= bank_re[r_addr];
            aim_q <= bank_im[r_addr];
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

    // ---- FSM ----
    localparam S_IDLE = 0, S_CLR = 1, S_AZ = 2, S_XY = 3, S_L = 4;
    reg [2:0] st = S_IDLE;
    reg [15:0] r_q;
    reg signed [31:0] xm, ym;
    reg [6:0] n;                            // pipeline cycle 1..NA+3
    assign busy = (st != S_IDLE);

    always @(posedge clk) begin
        acc_we <= 1'b0;
        clr_we <= 1'b0;
        case (st)
        S_IDLE: begin
            if (clear) begin clr_a <= 0; st <= S_CLR; end
            else if (start) begin
                r_q <= r_mm; w_q <= {1'b0, w[6:0]};
                st <= S_AZ;                 // az ROM read in flight
            end
        end
        S_CLR: begin
            clr_we <= 1'b1;
            wr_a <= clr_a;
            clr_a <= clr_a + 1;
            if (clr_a == NA - 1) st <= S_IDLE;
        end
        S_AZ: begin                         // azc_q/azs_q valid next cycle
            ang_a <= 0;                     // angle 0 ROM read in flight
            st <= S_XY;
        end
        S_XY: begin
            xm = $signed(r_q) * azc_q;
            ym = $signed(r_q) * azs_q;
            x <= (xm + HALF) >>> 14;
            y <= (ym + HALF) >>> 14;
            ang_a <= 1;                     // angle 1 ROM read in flight
            n <= 1;
            st <= S_L;
        end
        S_L: begin
            // deeper pipeline: reads run 3 cycles ahead of v2's schedule
            if (n >= 4 && n <= NA + 3) begin
                acc_we <= 1'b1;
                wr_a <= n - 4;              // write angle n-4's arm
            end
            acc_a <= n - 3;                 // bank read for angle n-3
            ang_a <= n + 1;                 // ROM read for angle n+1
            n <= n + 1;
            if (n == NA + 4) st <= S_IDLE;
        end
        default: st <= S_IDLE;
        endcase
    end
endmodule
