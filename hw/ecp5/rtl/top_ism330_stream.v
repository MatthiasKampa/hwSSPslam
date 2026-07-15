// top_ism330_stream.v — ISM330DHCX bring-up rung 3 (flash-ready before
// the IMU arrives, camera-eve pattern): boot-configure the part
// (417 Hz XL+G, BDU+IF_INC, gyro DRDY -> INT1), then stream one frame
// per INT1 rise, stamped with the free-running 50 MHz counter (the
// SHARED cross-sensor time base — lidar/cam alignment happens on this
// axis, ts unit = 20 ns):
//
//   0xAA 0x55 | ts[47:0] LSB-first (6 B) | OUT 0x22..0x2D (12 B:
//   gyro XYZ then accel XYZ, int16 LE) | XOR of the 18 payload bytes
//
// 21 B/frame @ 2 Mbaud = 105 us; @417 Hz = 23% UART duty. UART cmds:
//   'G'/'g'  stream on/off (OFF at boot — host arms it)
//   'I'      re-run the config sequence
//   'R'      reply {WHO_AM_I, {6'd0, int2, int1}}   (rung-1 compatible)
//   'W' a d  raw register write    'r' a  raw read (1-byte reply)
// No-slave-safe: SPI cannot hang; 'R' then reads 0xFF/0x00 -> host
// detects absence (same signature discipline as the OV5640 tops).
`default_nettype none

module top #(
    parameter BOOT_BIT = 21,         // ready at 2^21 clk ~ 42 ms
    parameter DIV_SPI  = 8,          // 50/(2*8)  = 3.125 MHz SCLK
    parameter DIV_UART = 25          // 50/25     = 2 Mbaud
) (
    input  wire       clk,           // 50 MHz
    input  wire       usb_rx,
    output wire       usb_tx,
    output wire [4:0] led,
    output wire       imu_sclk,
    output wire       imu_mosi,
    input  wire       imu_miso,
    output wire       imu_cs,
    input  wire       imu_int1,
    input  wire       imu_int2
);
    // ---- boot guard + shared time base ------------------------------------
    reg [BOOT_BIT:0] boot = 0;
    always @(posedge clk) if (!boot[BOOT_BIT]) boot <= boot + 1;
    wire ready = boot[BOOT_BIT];

    reg [47:0] tstamp = 0;
    always @(posedge clk) tstamp <= tstamp + 1;

    // ---- UART --------------------------------------------------------------
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

    // ---- SPI master ---------------------------------------------------------
    reg        s_start = 1'b0, s_rw = 1'b0;
    reg  [6:0] s_addr = 0;
    reg  [7:0] s_wdata = 0;
    wire [7:0] s_rdata;
    wire       s_busy;
    spi_reg #(.DIV(DIV_SPI)) bus (
        .clk(clk), .start(s_start), .rw(s_rw), .addr(s_addr),
        .wdata(s_wdata), .rdata(s_rdata), .busy(s_busy),
        .sclk(imu_sclk), .mosi(imu_mosi), .miso(imu_miso), .cs(imu_cs));

    // ---- INT sync + one-deep pend ------------------------------------------
    reg [2:0] i1s = 0, i2s = 0;
    always @(posedge clk) begin
        i1s <= {i1s[1:0], imu_int1};
        i2s <= {i2s[1:0], imu_int2};
    end
    wire i1rise = i1s[1] & ~i1s[2];
    reg  pend = 1'b0;

    // ---- config ROM (order matters: iface mode first, DRDY route last;
    // a function, not always@*, so it also evaluates at time zero) ------
    reg [1:0] cfgi = 0;
    function [14:0] cfg_of(input [1:0] i);
        case (i)
            2'd0: cfg_of = {7'h12, 8'h44}; // CTRL3_C:  BDU | IF_INC
            2'd1: cfg_of = {7'h10, 8'h60}; // CTRL1_XL: 417 Hz, +-2 g
            2'd2: cfg_of = {7'h11, 8'h60}; // CTRL2_G:  417 Hz, +-250 dps
            2'd3: cfg_of = {7'h0D, 8'h02}; // INT1_CTRL: DRDY_G -> INT1
        endcase
    endfunction

    // ---- frame store --------------------------------------------------------
    reg [47:0] ts_lat = 0;
    reg [7:0]  dat [0:11];
    reg [7:0]  xacc = 0;
    reg [3:0]  bidx = 0;
    reg [4:0]  txi = 0;
    reg [7:0]  fb;
    always @* begin
        case (txi)
            5'd0:    fb = 8'hAA;
            5'd1:    fb = 8'h55;
            5'd2:    fb = ts_lat[7:0];
            5'd3:    fb = ts_lat[15:8];
            5'd4:    fb = ts_lat[23:16];
            5'd5:    fb = ts_lat[31:24];
            5'd6:    fb = ts_lat[39:32];
            5'd7:    fb = ts_lat[47:40];
            5'd20:   fb = xacc;
            default: fb = dat[txi - 5'd8];
        endcase
    end

    // ---- main FSM ------------------------------------------------------------
    localparam S_BOOT  = 0,  S_CFGGO = 1,  S_CFGB = 2,  S_CFGE = 3,
               S_MAIN  = 4,  S_RDGO  = 5,  S_RDB  = 6,  S_RDE  = 7,
               S_TX    = 8,  S_WA    = 9,  S_WD   = 10, S_XGO  = 11,
               S_XB    = 12, S_XE    = 13, S_XR1  = 14, S_XR2  = 15,
               S_RA    = 16;
    reg [4:0] st = S_BOOT;
    reg       stream_en = 1'b0;
    reg [1:0] xrep = 0;                    // 0 none, 1 rdata, 2 id+ints
    reg [1:0] ints_lat = 0;

    always @(posedge clk) begin
        s_start <= 1'b0;
        txsend  <= 1'b0;
        if (i1rise) pend <= 1'b1;
        case (st)
            S_BOOT: if (ready) begin
                cfgi <= 0;
                st <= S_CFGGO;
            end
            // -- boot / 'I' config sequence ---------------------------------
            S_CFGGO: begin
                s_rw    <= 1'b0;
                {s_addr, s_wdata} <= cfg_of(cfgi);
                s_start <= 1'b1;
                st <= S_CFGB;
            end
            S_CFGB: if (s_busy) st <= S_CFGE;
            S_CFGE: if (!s_busy) begin
                cfgi <= cfgi + 1;
                st <= (cfgi == 2'd3) ? S_MAIN : S_CFGGO;
            end
            // -- dispatch: commands first (rare) over frames (frequent) -----
            S_MAIN: if (rxv) begin
                case (rxd)
                    "G": stream_en <= 1'b1;
                    "g": stream_en <= 1'b0;
                    "I": begin cfgi <= 0; st <= S_CFGGO; end
                    "R": begin
                        s_rw <= 1'b1; s_addr <= 7'h0F; xrep <= 2'd2;
                        ints_lat <= {i2s[1], i1s[1]};
                        st <= S_XGO;
                    end
                    "W": st <= S_WA;
                    "r": st <= S_RA;
                    default: ;
                endcase
            end else if (pend && stream_en) begin
                pend   <= i1rise;
                ts_lat <= tstamp;
                xacc   <= tstamp[7:0] ^ tstamp[15:8] ^ tstamp[23:16]
                          ^ tstamp[31:24] ^ tstamp[39:32] ^ tstamp[47:40];
                bidx <= 0;
                st <= S_RDGO;
            end
            // -- frame: 12 register reads then 21-byte dump -----------------
            S_RDGO: begin
                s_rw    <= 1'b1;
                s_addr  <= 7'h22 + {3'd0, bidx};
                s_start <= 1'b1;
                st <= S_RDB;
            end
            S_RDB: if (s_busy) st <= S_RDE;
            S_RDE: if (!s_busy) begin
                dat[bidx] <= s_rdata;
                xacc <= xacc ^ s_rdata;
                bidx <= bidx + 1;
                if (bidx == 4'd11) begin
                    txi <= 0;
                    st <= S_TX;
                end else
                    st <= S_RDGO;
            end
            S_TX: if (tx_free) begin
                txd    <= fb;
                txsend <= 1'b1;
                txi    <= txi + 1;
                if (txi == 5'd20) st <= S_MAIN;
            end
            // -- 'W' aa dd / 'r' aa passthrough ------------------------------
            S_WA: if (rxv) begin s_addr <= rxd[6:0]; st <= S_WD; end
            S_WD: if (rxv) begin
                s_rw <= 1'b0; s_wdata <= rxd; xrep <= 2'd0;
                st <= S_XGO;
            end
            S_RA: if (rxv) begin
                s_rw <= 1'b1; s_addr <= rxd[6:0]; xrep <= 2'd1;
                st <= S_XGO;
            end
            S_XGO: begin s_start <= 1'b1; st <= S_XB; end
            S_XB:  if (s_busy) st <= S_XE;
            S_XE:  if (!s_busy) st <= (xrep == 2'd0) ? S_MAIN : S_XR1;
            S_XR1: if (tx_free) begin
                txd <= s_rdata; txsend <= 1'b1;
                st <= (xrep == 2'd2) ? S_XR2 : S_MAIN;
            end
            S_XR2: if (tx_free) begin
                txd <= {6'd0, ints_lat}; txsend <= 1'b1;
                st <= S_MAIN;
            end
            default: st <= S_MAIN;
        endcase
    end

    reg [25:0] beat = 0;
    always @(posedge clk) beat <= beat + 1;
    assign led = {beat[25], ready, stream_en, i1s[1], st != S_MAIN};
endmodule
`default_nettype wire
