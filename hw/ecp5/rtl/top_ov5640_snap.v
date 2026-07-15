// top_ov5640_snap.v — camera bring-up step 2 top (flash-ready): power
// sequence, auto-run the init ROM (QVGA Y8, ~25 fps predicted), then
// serve UART commands at 2 Mbaud:
//   'S' 0x53                 arm snapshot; when the frame lands, stream
//                            W*H = 76800 bytes (raster) back
//   'R' 0x52                 13-byte report: frames[4] lines[2]
//                            bytes/line[2] n_bytes[3] init_nerr[1]
//                            status[1] = {6'd0, init_running, init_done}
//   'W' 0x57 + ah + al + d   SCCB register write, reply {7'd0, ack_err}
//   'r' 0x72 + ah + al       SCCB register read, reply {data, err}
//   'I' 0x49                 re-run the init ROM (poll 'R' for done)
// The report + passthrough are the frame-rate servo: PLL/VTS registers
// are stepped live against MEASURED geometry, no reflash (the ROM's
// rate section is predicted-not-measured).
`default_nettype none

module top #(
    // sim overrides (tb_top_snap shrinks all of these; synthesis uses
    // the defaults)
    parameter W = 320,
    parameter H = 240,
    parameter BOOT_BIT = 20,        // cam_ready at 2^20 clk ~ 21 ms
    parameter DIV_SCCB = 64,        // ~195 kHz SCL
    parameter MS_CYC = 50_000,      // init-ROM delay cycles per ms
    parameter ROMFILE = "build/ov5640_init.hex"
) (
    input  wire       clk,          // 50 MHz
    input  wire       usb_rx,
    output wire       usb_tx,
    output wire [4:0] led,
    output reg        cam_xclk = 1'b0,
    output wire       cam_resetb,
    output wire       cam_pwdn,
    inout  wire       cam_siod,
    output wire       cam_sioc,
    input  wire       cam_pclk,
    input  wire       cam_href,
    input  wire       cam_vsync,
    input  wire [7:0] cam_d
);
    localparam DIV_UART = 25;       // 2 Mbaud
    localparam NPIX = W * H;

    always @(posedge clk) cam_xclk <= ~cam_xclk;   // 25 MHz

    // power-up: RESET# low for the first quarter of the boot window;
    // SCCB/init no earlier than cam_ready (~21 ms at BOOT_BIT 20)
    reg [BOOT_BIT:0] boot = 0;
    always @(posedge clk) if (!boot[BOOT_BIT]) boot <= boot + 1;
    assign cam_resetb = boot[BOOT_BIT] || boot[BOOT_BIT-1]
                        || boot[BOOT_BIT-2];
    assign cam_pwdn   = 1'b0;
    wire cam_ready    = boot[BOOT_BIT];

    // ---- UART -----------------------------------------------------------
    wire [7:0] rxd;
    wire       rxv;
    uart_rx #(.DIV(DIV_UART)) urx (.clk(clk), .rx(usb_rx), .data(rxd),
                                   .valid(rxv));
    reg  [7:0] txd = 0;
    reg        txsend = 1'b0;
    wire       txbusy;
    uart_tx #(.DIV(DIV_UART)) utx (.clk(clk), .data(txd), .send(txsend),
                                   .tx(usb_tx), .busy(txbusy));
    wire tx_free = !txbusy && !txsend;

    // ---- SCCB master, shared init-walker / command mux -------------------
    wire        i_start;
    wire [15:0] i_addr;
    wire [7:0]  i_wdata;
    reg         c_start = 1'b0, c_wr = 1'b1;
    reg  [15:0] c_addr = 0;
    reg  [7:0]  c_wdata = 0;
    wire        init_running, init_done;
    wire [7:0]  init_nerr;
    wire        s_busy, s_ack_err;
    wire [7:0]  s_rdata;
    sccb #(.DIV(DIV_SCCB)) bus (
        .clk(clk),
        .start(init_running ? i_start : c_start),
        .wr(init_running ? 1'b1 : c_wr),
        .addr(init_running ? i_addr : c_addr),
        .wdata(init_running ? i_wdata : c_wdata),
        .rdata(s_rdata), .busy(s_busy), .ack_err(s_ack_err),
        .siod(cam_siod), .sioc(cam_sioc));

    // auto-start init once the camera is powered and settled
    reg  init_go = 1'b0;
    reg  init_started = 1'b0;
    ov5640_init #(.ROMFILE(ROMFILE), .MS_CYCLES(MS_CYC)) rom (
        .clk(clk), .go(init_go), .running(init_running),
        .done(init_done), .nerr(init_nerr),
        .s_start(i_start), .s_addr(i_addr), .s_wdata(i_wdata),
        .s_busy(s_busy), .s_ack_err(s_ack_err));

    // ---- DVP snapshot -----------------------------------------------------
    reg         arm = 1'b0;
    wire        cap_busy, cap_done;
    wire [16:0] n_bytes;
    reg  [16:0] rd_addr = 0;
    wire [7:0]  rd_data;
    wire [31:0] cnt_frames;
    wire [15:0] cnt_lines, cnt_bytes_line;
    dvp_capture #(.W(W), .H(H), .AW(17)) cap (
        .clk(clk), .rst(1'b0),
        .dvp_pclk(cam_pclk), .dvp_href(cam_href),
        .dvp_vsync(cam_vsync), .dvp_d(cam_d),
        .mode_y8(1'b1), .arm(arm), .busy(cap_busy), .done(cap_done),
        .n_bytes(n_bytes), .rd_addr(rd_addr), .rd_data(rd_data),
        .cnt_frames(cnt_frames), .cnt_lines(cnt_lines),
        .cnt_bytes_line(cnt_bytes_line));

    // ---- command FSM ------------------------------------------------------
    localparam ST_IDLE = 0, ST_WA1 = 1, ST_WA2 = 2, ST_WD = 3,
               ST_XACT = 4, ST_XWAIT = 5, ST_WREP = 6, ST_RREP = 7,
               ST_CAP = 8, ST_DSET = 9, ST_DLAT = 10, ST_DSEND = 11,
               ST_REP = 12;
    reg [3:0]  st = ST_IDLE;
    reg [3:0]  rep_i = 0;
    reg [16:0] di = 0;
    reg [7:0]  rep [0:12];

    always @(posedge clk) begin
        arm <= 1'b0;
        txsend <= 1'b0;
        c_start <= 1'b0;
        init_go <= 1'b0;
        if (cam_ready && !init_started) begin      // one-shot auto init
            init_go <= 1'b1;
            init_started <= 1'b1;
        end
        case (st)
            ST_IDLE: if (rxv) case (rxd)
                "S": begin arm <= 1'b1; st <= ST_CAP; end
                "R": begin
                    rep[0]  <= cnt_frames[31:24];
                    rep[1]  <= cnt_frames[23:16];
                    rep[2]  <= cnt_frames[15:8];
                    rep[3]  <= cnt_frames[7:0];
                    rep[4]  <= cnt_lines[15:8];
                    rep[5]  <= cnt_lines[7:0];
                    rep[6]  <= cnt_bytes_line[15:8];
                    rep[7]  <= cnt_bytes_line[7:0];
                    rep[8]  <= {7'd0, n_bytes[16]};
                    rep[9]  <= n_bytes[15:8];
                    rep[10] <= n_bytes[7:0];
                    rep[11] <= init_nerr;
                    rep[12] <= {6'd0, init_running, init_done};
                    rep_i <= 0;
                    st <= ST_REP;
                end
                "W": begin c_wr <= 1'b1; st <= ST_WA1; end
                "r": begin c_wr <= 1'b0; st <= ST_WA1; end
                "I": init_go <= 1'b1;
                default: ;
            endcase
            // -- SCCB passthrough ----------------------------------------
            ST_WA1: if (rxv) begin c_addr[15:8] <= rxd; st <= ST_WA2; end
            ST_WA2: if (rxv) begin
                c_addr[7:0] <= rxd;
                st <= c_wr ? ST_WD : ST_XACT;
            end
            ST_WD: if (rxv) begin c_wdata <= rxd; st <= ST_XACT; end
            ST_XACT: if (!init_running && !s_busy) begin
                c_start <= 1'b1;
                st <= ST_XWAIT;
            end
            // busy rises one cycle after the start pulse; !c_start masks
            // that window so the wait cannot fall through early
            ST_XWAIT: if (!s_busy && !c_start) begin
                st <= c_wr ? ST_WREP : ST_RREP;
            end
            ST_WREP: if (tx_free) begin
                txd <= {7'd0, s_ack_err};
                txsend <= 1'b1;
                st <= ST_IDLE;
            end
            ST_RREP: if (tx_free) begin
                txd <= s_rdata;
                txsend <= 1'b1;
                st <= ST_WREP;      // then the err byte
            end
            // -- snapshot dump ---------------------------------------------
            ST_CAP: if (cap_done) begin
                di <= 0;
                st <= ST_DSET;
            end
            ST_DSET: begin rd_addr <= di; st <= ST_DLAT; end
            ST_DLAT: st <= ST_DSEND;
            ST_DSEND: if (tx_free) begin
                txd <= rd_data;
                txsend <= 1'b1;
                if (di == NPIX - 1)
                    st <= ST_IDLE;
                else begin
                    di <= di + 1;
                    st <= ST_DSET;
                end
            end
            // -- counter report ---------------------------------------------
            ST_REP: if (tx_free) begin
                txd <= rep[rep_i];
                txsend <= 1'b1;
                if (rep_i == 12)
                    st <= ST_IDLE;
                else
                    rep_i <= rep_i + 1;
            end
            default: st <= ST_IDLE;
        endcase
    end

    reg [25:0] beat = 0;
    always @(posedge clk) beat <= beat + 1;
    assign led = {beat[25], init_done, cap_busy, init_nerr != 0,
                  st != ST_IDLE};
endmodule
`default_nettype wire
