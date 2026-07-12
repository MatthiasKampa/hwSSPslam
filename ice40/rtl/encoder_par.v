// Ring-parallel, angle-pipelined SSP encoder — same interface and bit-exact
// contract as encoder.v (golden model: ssp_ice40.encode_int), but:
//   - the 4 octave rings accumulate SIMULTANEOUSLY (banked RAMs per ring),
//   - the angle loop is software-pipelined at 1 cycle/angle:
//       during L(n): write j=n-1 arms; roms for j=n load; addrs for j=n+1 set.
// ~64 cycles/point vs ~845 (v1): a 903-pt scan encodes in ~4.8 ms @ 12 MHz.
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
    input  wire [7:0]  rd_idx,             // component 0..NA*NR-1
    output reg  signed [31:0] rd_re,
    output reg  signed [31:0] rd_im
);
    localparam signed HALF = 1 << 13;      // F_AZ = F_ANG = 14

    // ---- ROMs (images from gen_luts.py == golden model) ----
    reg signed [15:0] rom_azc [0:1023];
    reg signed [15:0] rom_azs [0:1023];
    reg signed [15:0] rom_angc[0:63];
    reg signed [15:0] rom_angs[0:63];
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

    // combinational projection for the CURRENT angle -> cis addresses
    reg signed [15:0] x, y;
    wire signed [32:0] um_c = x * angc_q + y * angs_q;
    wire signed [18:0] u_c  = (um_c + HALF) >>> 14;

    // per-ring cis lookups (bit-slice addresses, sync read, packed ROM)
    reg signed [7:0] cre_q [0:NR-1];
    reg signed [7:0] cim_q [0:NR-1];
    genvar gk;
    generate for (gk = 0; gk < NR; gk = gk + 1) begin : g_cis
        wire signed [18:0] ush = u_c >>> gk;
        always @(posedge clk) begin
            cre_q[gk] <= $signed(rom_cis[ush[7:0]][7:0]);
            cim_q[gk] <= $signed(rom_cis[ush[7:0]][15:8]);
        end
    end endgenerate

    // ---- banked accumulators: per ring, 64 x 32 (re, im) ----
    // readback decode (rd_idx = k*NA + j), constant-NA compares
    wire [1:0] rd_k = (rd_idx >= 3 * NA) ? 2'd3 :
                      (rd_idx >= 2 * NA) ? 2'd2 :
                      (rd_idx >= NA)     ? 2'd1 : 2'd0;
    wire [5:0] rd_j = rd_idx - rd_k * NA;
    reg [5:0] acc_a;                        // read addr (angle j)
    reg [5:0] wr_a;                         // write addr (angle j, lagging)
    reg       acc_we;
    reg       clr_we;
    reg [5:0] clr_a;
    reg signed [7:0] w_q;
    wire signed [31:0] rb_re_w [0:NR-1];
    wire signed [31:0] rb_im_w [0:NR-1];
    // one read + one write port per bank (EBR dual-port shape):
    // read addr muxes to the readback index while the FSM is idle
    // wr_a is latched TOGETHER with its enable flag (acc_we or clr_we),
    // so address and enable always travel aligned through the pipeline
    wire [5:0] r_addr = busy ? acc_a : rd_j;
    wire [5:0] w_addr = wr_a;
    wire       w_en   = acc_we | clr_we;
    generate for (gk = 0; gk < NR; gk = gk + 1) begin : g_acc
        reg signed [31:0] bank_re [0:63];
        reg signed [31:0] bank_im [0:63];
        reg signed [31:0] are_q, aim_q;
        // 8x8 products as explicit 7-tap shift-adds: w_q is a NON-NEGATIVE
        // 7-bit weight, so w*cis = sum of conditionally-shifted cis terms.
        // Structural LUT logic by construction — the DSPs stay reserved
        // for the four 16x16 mults (yosys maps this to ~fabric adders).
        wire signed [15:0] pre =
            (w_q[0] ? {{8{cre_q[gk][7]}}, cre_q[gk]}        : 16'sd0) +
            (w_q[1] ? {{7{cre_q[gk][7]}}, cre_q[gk], 1'b0}  : 16'sd0) +
            (w_q[2] ? {{6{cre_q[gk][7]}}, cre_q[gk], 2'b0}  : 16'sd0) +
            (w_q[3] ? {{5{cre_q[gk][7]}}, cre_q[gk], 3'b0}  : 16'sd0) +
            (w_q[4] ? {{4{cre_q[gk][7]}}, cre_q[gk], 4'b0}  : 16'sd0) +
            (w_q[5] ? {{3{cre_q[gk][7]}}, cre_q[gk], 5'b0}  : 16'sd0) +
            (w_q[6] ? {{2{cre_q[gk][7]}}, cre_q[gk], 6'b0}  : 16'sd0);
        wire signed [15:0] pim =
            (w_q[0] ? {{8{cim_q[gk][7]}}, cim_q[gk]}        : 16'sd0) +
            (w_q[1] ? {{7{cim_q[gk][7]}}, cim_q[gk], 1'b0}  : 16'sd0) +
            (w_q[2] ? {{6{cim_q[gk][7]}}, cim_q[gk], 2'b0}  : 16'sd0) +
            (w_q[3] ? {{5{cim_q[gk][7]}}, cim_q[gk], 3'b0}  : 16'sd0) +
            (w_q[4] ? {{4{cim_q[gk][7]}}, cim_q[gk], 4'b0}  : 16'sd0) +
            (w_q[5] ? {{3{cim_q[gk][7]}}, cim_q[gk], 5'b0}  : 16'sd0) +
            (w_q[6] ? {{2{cim_q[gk][7]}}, cim_q[gk], 6'b0}  : 16'sd0);
        always @(posedge clk) begin
            are_q <= bank_re[r_addr];
            aim_q <= bank_im[r_addr];
            if (w_en) begin
                bank_re[w_addr] <= clr_we ? 32'sd0 : are_q + pre;
                bank_im[w_addr] <= clr_we ? 32'sd0 : aim_q + pim;
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
    reg [6:0] n;                            // pipeline cycle 1..NA+1
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
            acc_a <= 0;                     // bank read j=0 in flight
            ang_a <= 1;                     // angle 1 ROM read in flight
            n <= 1;
            st <= S_L;
        end
        S_L: begin
            // during L(n): angc_q holds angle n-1... pipeline invariant:
            //   roms/bank loaded at THIS cycle's end serve angle n;
            //   write armed here (wr_a = n-1) executes at end of L(n+1).
            if (n <= NA) begin
                acc_we <= 1'b1;
                wr_a <= n - 1;
            end
            acc_a <= n;                     // are(n) loads at end of L(n+1)
            ang_a <= n + 1;                 // angc(n+1) loads at end of L(n+1)
            n <= n + 1;
            if (n == NA + 1) st <= S_IDLE;
        end
        default: st <= S_IDLE;
        endcase
    end
endmodule
