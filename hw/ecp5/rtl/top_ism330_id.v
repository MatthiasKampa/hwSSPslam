// top_ism330_id.v — ISM330DHCX bring-up rung 1 (flash-ready before the
// IMU arrives, mirroring top_ov5640_id): on ANY UART byte, read
// WHO_AM_I (0x0F) over SPI and reply {id, {6'd0, int2, int1}}.
// Expect 0x6B 0x0x. No reset pin on the part — 50 ms power-on guard.
`default_nettype none

module top #(
    parameter BOOT_BIT = 21          // ready at 2^21 clk ~ 42 ms
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
    localparam DIV_UART = 25;        // 2 Mbaud

    reg [BOOT_BIT:0] boot = 0;
    always @(posedge clk) if (!boot[BOOT_BIT]) boot <= boot + 1;
    wire ready = boot[BOOT_BIT];

    wire [7:0] rxd;
    wire       rxv;
    uart_rx #(.DIV(DIV_UART)) urx (.clk(clk), .rx(usb_rx), .data(rxd),
                                   .valid(rxv));
    reg  [7:0] txd = 0;
    reg        txsend = 1'b0;
    wire       txbusy;
    uart_tx #(.DIV(DIV_UART)) utx (.clk(clk), .data(txd), .send(txsend),
                                   .tx(usb_tx), .busy(txbusy));

    reg        s_start = 1'b0;
    wire [7:0] s_rdata;
    wire       s_busy;
    spi_reg #(.DIV(8)) bus (                   // 3.125 MHz SCLK
        .clk(clk), .start(s_start), .rw(1'b1), .addr(7'h0F),
        .wdata(8'h00), .rdata(s_rdata), .busy(s_busy),
        .sclk(imu_sclk), .mosi(imu_mosi), .miso(imu_miso),
        .cs(imu_cs));

    reg [2:0] st = 0;
    reg [1:0] ints = 0;
    always @(posedge clk) begin
        s_start <= 1'b0;
        txsend  <= 1'b0;
        case (st)
            0: if (rxv && ready) begin
                s_start <= 1'b1;
                ints <= {imu_int2, imu_int1};
                st <= 1;
            end
            1: if (s_busy) st <= 2;
            2: if (!s_busy) st <= 3;
            3: if (!txbusy) begin
                txd <= s_rdata;
                txsend <= 1'b1;
                st <= 4;
            end
            4: if (!txbusy && !txsend) begin
                txd <= {6'd0, ints};
                txsend <= 1'b1;
                st <= 0;
            end
        endcase
    end

    reg [25:0] beat = 0;
    always @(posedge clk) beat <= beat + 1;
    assign led = {beat[25], ready, st != 0, imu_int2, imu_int1};
endmodule
`default_nettype wire
