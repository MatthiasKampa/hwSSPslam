// dvp_capture.v — OV5640 DVP snapshot front (ECP5 bring-up step 2).
// All-sysclk oversampling capture: PCLK/HREF/VSYNC/D go through 2FF
// synchronizers and bytes are taken on the synchronized PCLK rising
// edge — no CDC clocking, safe while PCLK <= ~12.5 MHz (4x oversample
// at 50 MHz; the 120 fps point lifts the capture domain via PLL, step
// 4). Polarities per the vendored init (0x4740 = 0x21): data latched on
// PCLK rise, VSYNC pulses HIGH between frames, HREF high during valid
// pixels.
//
// arm=pulse -> wait for a clean frame boundary (VSYNC rise then fall)
// -> write Y bytes to the snapshot RAM (EBR) -> done pulse at frame end
// or when full. mode_y8: 1 = every HREF byte is luma (Y8 / grayscale
// format, the ROM default); 0 = YUYV, take even-index bytes.
// Free-running debug counters (no arming needed): frames since reset,
// HREF lines in the last frame, bytes in the last line — the live
// frame-rate/geometry evidence the SCCB rate servo reads.
`default_nettype none

module dvp_capture #(
    parameter W = 320,
    parameter H = 240,
    parameter AW = 17               // >= clog2(W*H)
) (
    input  wire          clk,
    input  wire          rst,
    // DVP pins (async)
    input  wire          dvp_pclk,
    input  wire          dvp_href,
    input  wire          dvp_vsync,
    input  wire [7:0]    dvp_d,
    // control
    input  wire          mode_y8,
    input  wire          arm,        // pulse: capture next full frame
    output reg           busy = 1'b0,
    output reg           done = 1'b0,   // 1-clk pulse: snapshot ready
    output reg  [AW-1:0] n_bytes = 0,   // bytes written to the snapshot
    // snapshot read port (sysclk, 1-cycle latency)
    input  wire [AW-1:0] rd_addr,
    output reg  [7:0]    rd_data = 0,
    // free-running counters
    output reg  [31:0]   cnt_frames = 0,
    output reg  [15:0]   cnt_lines = 0,      // lines in last frame
    output reg  [15:0]   cnt_bytes_line = 0  // bytes in last line
);
    // ---- 2FF sync; data rides the same stages as its strobe ------------
    reg [2:0] ps = 0;
    reg [1:0] hs = 0, vs = 0;
    reg [7:0] d0 = 0, d1 = 0;
    always @(posedge clk) begin
        ps <= {ps[1:0], dvp_pclk};
        hs <= {hs[0], dvp_href};
        vs <= {vs[0], dvp_vsync};
        d0 <= dvp_d;
        d1 <= d0;
    end
    wire pclk_rise = ps[1] && !ps[2];
    wire href_q  = hs[1];
    wire vs_q    = vs[1];
    wire [7:0] dq = d1;

    reg vprev = 1'b0, hprev = 1'b0;
    wire vs_rise = vs_q && !vprev;
    wire vs_fall = !vs_q && vprev;
    wire hr_rise = href_q && !hprev;
    wire hr_fall = !href_q && hprev;

    // ---- free-running geometry counters (pclk-edge granularity) --------
    reg [15:0] line_n = 0, byte_n = 0;
    always @(posedge clk) begin
        if (rst) begin
            vprev <= 1'b0; hprev <= 1'b0;
            line_n <= 0; byte_n <= 0;
            cnt_frames <= 0; cnt_lines <= 0; cnt_bytes_line <= 0;
        end else if (pclk_rise) begin
            vprev <= vs_q;
            hprev <= href_q;
            if (vs_rise) begin
                cnt_frames <= cnt_frames + 1;
                cnt_lines <= line_n;
                line_n <= 0;
            end
            if (hr_rise) begin
                line_n <= line_n + 1;
                byte_n <= 16'd1;               // this edge carries byte 0
            end else if (href_q)
                byte_n <= byte_n + 1;
            if (hr_fall)
                cnt_bytes_line <= byte_n;
        end
    end

    // ---- snapshot FSM ---------------------------------------------------
    reg [7:0] mem [0:W*H-1];
    reg [AW-1:0] waddr = 0;
    reg [1:0] st = 0;                  // 0 idle, 1 wait vs rise, 2 wait
                                       // vs fall, 3 capture
    reg par = 1'b0;                    // parity of the NEXT in-line byte
    // current byte parity: 0 at line start (hr_rise edge), else par —
    // YUYV luma sits at even indices; Y8 takes every byte
    wire ysel = mode_y8 || (hr_rise ? 1'b1 : !par);
    wire full = (waddr == W * H);
    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            st <= 0; busy <= 1'b0; waddr <= 0; n_bytes <= 0;
        end else begin
            case (st)
                0: if (arm) begin
                    busy <= 1'b1;
                    waddr <= 0;
                    st <= 1;
                end
                1: if (pclk_rise && vs_rise) st <= 2;
                2: if (pclk_rise && vs_fall) st <= 3;
                3: begin
                    if (pclk_rise && href_q) begin
                        if (ysel && !full) begin
                            mem[waddr] <= dq;
                            waddr <= waddr + 1;
                        end
                        par <= hr_rise ? 1'b1 : ~par;
                    end
                    if (full || (pclk_rise && vs_rise)) begin
                        n_bytes <= waddr;
                        busy <= 1'b0;
                        done <= 1'b1;
                        st <= 0;
                    end
                end
            endcase
        end
    end

    always @(posedge clk)
        rd_data <= mem[rd_addr];
endmodule
`default_nettype wire
