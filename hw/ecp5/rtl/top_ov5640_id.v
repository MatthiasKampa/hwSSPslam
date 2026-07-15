// top_ov5640_id.v — camera bring-up step 1 (flash-ready before the cam
// arrives): power-sequence the OV5640, then on ANY UART byte read the
// chip ID (0x300A/0x300B) over SCCB and reply {id_hi, id_lo, ack_flags}.
// Expect 0x56 0x40 0x00. XCLK = 25 MHz (sysclk/2, within the 6-27 MHz
// spec, no PLL). Power-up: PWDN low, RESET# held low 5 ms, then high,
// SCCB no earlier than 20 ms after (datasheet timing, generous margins).
`default_nettype none

module top (
    input  wire       clk,          // 50 MHz
    input  wire       usb_rx,
    output wire       usb_tx,
    output wire [4:0] led,
    output reg        cam_xclk = 1'b0,
    output wire       cam_resetb,
    output wire       cam_pwdn,
    inout  wire       cam_siod,
    output wire       cam_sioc
);
    localparam DIV_UART = 25;       // 2 Mbaud
    localparam DIV_SCCB = 64;       // ~195 kHz SCL

    always @(posedge clk) cam_xclk <= ~cam_xclk;   // 25 MHz

    // power-up sequencer: RESET# low for 5 ms, ready after 25 ms
    reg [20:0] boot = 0;
    always @(posedge clk) if (!boot[20]) boot <= boot + 1;
    assign cam_resetb = (boot > 21'd250_000);      // 5 ms @ 50 MHz
    assign cam_pwdn   = 1'b0;
    wire cam_ready    = boot[20];                  // ~21 ms

    wire [7:0] rxd;
    wire       rxv;
    uart_rx #(.DIV(DIV_UART)) urx (.clk(clk), .rx(usb_rx), .data(rxd),
                                   .valid(rxv));
    reg  [7:0] txd = 0;
    reg        txsend = 1'b0;
    wire       txbusy;
    uart_tx #(.DIV(DIV_UART)) utx (.clk(clk), .data(txd), .send(txsend),
                                   .tx(usb_tx), .busy(txbusy));

    reg         s_start = 1'b0;
    reg  [15:0] s_addr = 0;
    wire [7:0]  s_rdata;
    wire        s_busy, s_ack_err;
    sccb #(.DIV(DIV_SCCB)) bus (
        .clk(clk), .start(s_start), .wr(1'b0), .addr(s_addr),
        .wdata(8'h00), .rdata(s_rdata), .busy(s_busy),
        .ack_err(s_ack_err), .siod(cam_siod), .sioc(cam_sioc));

    reg [3:0] st = 0;
    reg [7:0] id_hi = 0, id_lo = 0;
    reg [1:0] errs = 0;
    always @(posedge clk) begin
        s_start <= 1'b0;
        txsend  <= 1'b0;
        case (st)
            0: if (rxv && cam_ready) begin
                s_addr <= 16'h300A;
                s_start <= 1'b1;
                errs <= 0;
                st <= 1;
            end
            1: if (s_busy) st <= 2;
            2: if (!s_busy) begin
                id_hi <= s_rdata;
                errs[0] <= s_ack_err;
                s_addr <= 16'h300B;
                s_start <= 1'b1;
                st <= 3;
            end
            3: if (s_busy) st <= 4;
            4: if (!s_busy) begin
                id_lo <= s_rdata;
                errs[1] <= s_ack_err;
                st <= 5;
            end
            5: if (!txbusy) begin txd <= id_hi; txsend <= 1'b1; st <= 6; end
            6: if (!txbusy && !txsend) begin
                txd <= id_lo; txsend <= 1'b1; st <= 7;
            end
            7: if (!txbusy && !txsend) begin
                txd <= {6'd0, errs}; txsend <= 1'b1; st <= 0;
            end
        endcase
    end

    reg [25:0] beat = 0;
    always @(posedge clk) beat <= beat + 1;
    assign led = {beat[25], cam_ready, st != 0, errs};
endmodule
`default_nettype wire
