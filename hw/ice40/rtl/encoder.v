// SSP encoder core — bit-exact to ssp_ice40.encode_int (the golden model).
// Per point (az10, r16 mm, w8): x,y from az LUT; NA projections u_j via
// (cos,sin) ROM; ring cis addresses as bit-slices of u (octave ladder);
// int32 accumulate of w*cis. Serial micro-sequenced datapath, one point
// at a time (start/busy handshake).
module encoder #(
    parameter NA = 60,                     // angles
    parameter NR = 4                       // octave rings
) (
    input  wire        clk,
    input  wire        clear,              // zero accumulators
    input  wire        start,              // latch az/r/w, encode one point
    input  wire [9:0]  az,
    input  wire [15:0] r_mm,
    input  wire [7:0]  w,
    output wire        busy,
    // readback: set rd_idx, acc_* valid 2 cycles later
    input  wire [7:0]  rd_idx,             // component 0..NA*NR-1
    output reg  signed [31:0] rd_re,
    output reg  signed [31:0] rd_im
);
    localparam NC = NA * NR;               // components (240)
    localparam signed HALF_AZ  = 1 << 13;  // F_AZ=14 rounding
    localparam signed HALF_ANG = 1 << 13;  // F_ANG=14 rounding

    // ---- ROMs (sync read; images from gen_luts.py == golden model) ----
    reg signed [15:0] rom_azc [0:1023];
    reg signed [15:0] rom_azs [0:1023];
    reg signed [15:0] rom_angc[0:63];
    reg signed [15:0] rom_angs[0:63];
    reg signed [7:0]  rom_cre [0:255];
    reg signed [7:0]  rom_cim [0:255];
    initial begin
        $readmemh("build/az_c.hex",  rom_azc);
        $readmemh("build/az_s.hex",  rom_azs);
        $readmemh("build/ang_c.hex", rom_angc);
        $readmemh("build/ang_s.hex", rom_angs);
        $readmemh("build/cis_re.hex", rom_cre);
        $readmemh("build/cis_im.hex", rom_cim);
    end
    reg signed [15:0] azc_q, azs_q, angc_q, angs_q;
    reg signed [7:0]  cre_q, cim_q;
    reg [5:0] ang_a;
    reg [7:0] cis_a;
    always @(posedge clk) begin
        azc_q  <= rom_azc[az];
        azs_q  <= rom_azs[az];
        angc_q <= rom_angc[ang_a];
        angs_q <= rom_angs[ang_a];
        cre_q  <= rom_cre[cis_a];
        cim_q  <= rom_cim[cis_a];
    end

    // ---- accumulators: two 240 x 32 RAMs ----
    reg signed [31:0] acc_re[0:NC-1];
    reg signed [31:0] acc_im[0:NC-1];
    reg [7:0] acc_a;
    reg signed [31:0] are_q, aim_q;
    reg               acc_we;
    reg signed [31:0] acc_wre, acc_wim;
    reg [7:0]         acc_wa;
    always @(posedge clk) begin
        are_q <= acc_re[acc_a];
        aim_q <= acc_im[acc_a];
        if (acc_we) begin
            acc_re[acc_wa] <= acc_wre;
            acc_im[acc_wa] <= acc_wim;
        end
        rd_re <= acc_re[rd_idx];
        rd_im <= acc_im[rd_idx];
    end

    // ---- FSM ----
    localparam S_IDLE = 0, S_CLR = 1, S_AZ = 2, S_XY = 3, S_ANG = 4,
               S_PROJ = 5, S_KRD = 6, S_KMUL = 7, S_KWR = 8;
    reg [3:0]  st = S_IDLE;
    reg [15:0] r_q;
    reg signed [7:0] w_q;
    reg signed [31:0] xm, ym;
    reg signed [15:0] x, y;
    reg signed [32:0] um;
    reg signed [18:0] u;
    reg [5:0] j;
    reg [1:0] k;
    reg [7:0] idx;                          // k*NA + j
    reg [7:0] clr_i;
    assign busy = (st != S_IDLE);

    wire signed [18:0] u_sh = u >>> k;      // ring bit-slice
    always @(posedge clk) begin
        acc_we <= 1'b0;
        case (st)
        S_IDLE: begin
            if (clear) begin clr_i <= 0; st <= S_CLR; end
            else if (start) begin
                r_q <= r_mm; w_q <= {1'b0, w[6:0]};
                st <= S_AZ;                 // az ROM read issued (addr = az)
            end
        end
        S_CLR: begin
            acc_we <= 1'b1; acc_wa <= clr_i; acc_wre <= 0; acc_wim <= 0;
            clr_i <= clr_i + 1;
            if (clr_i == NC - 1) st <= S_IDLE;
        end
        S_AZ: begin                         // azc_q/azs_q valid next edge
            st <= S_XY;
        end
        S_XY: begin
            xm = $signed(r_q) * azc_q;      // blocking: use immediately
            ym = $signed(r_q) * azs_q;
            x <= (xm + HALF_AZ) >>> 14;
            y <= (ym + HALF_AZ) >>> 14;
            j <= 0; ang_a <= 0;
            st <= S_ANG;
        end
        S_ANG: begin                        // angc_q/angs_q valid next edge
            st <= S_PROJ;
        end
        S_PROJ: begin
            um = x * angc_q + y * angs_q;
            u <= (um + HALF_ANG) >>> 14;
            k <= 0; idx <= {2'b00, j};      // j (k=0 base)
            st <= S_KRD;
        end
        S_KRD: begin                        // issue acc + cis reads
            acc_a <= idx;
            cis_a <= u_sh[7:0];
            st <= S_KMUL;
        end
        S_KMUL: begin                       // cis_q + acc_q valid at edge
            st <= S_KWR;
        end
        S_KWR: begin
            acc_we  <= 1'b1;
            acc_wa  <= idx;
            acc_wre <= are_q + w_q * cre_q;
            acc_wim <= aim_q + w_q * cim_q;
            if (k == NR - 1) begin
                if (j == NA - 1) st <= S_IDLE;
                else begin
                    j <= j + 1; ang_a <= j + 1;
                    st <= S_ANG;
                end
            end else begin
                k <= k + 1; idx <= idx + NA;
                st <= S_KRD;
            end
        end
        default: st <= S_IDLE;
        endcase
    end
endmodule
