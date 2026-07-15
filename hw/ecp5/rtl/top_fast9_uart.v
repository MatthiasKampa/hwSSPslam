// top_fast9_uart.v — the Icepi Zero fast9 HW GATE top.
// Protocol (2 Mbaud 8N1 over the FT231X, DIV=25 @ 50 MHz):
//   host sends: 1 byte threshold, then W*H=4096 pixel bytes (raster).
//   fabric replies, per emitted centre (y>=6, x>=6 => 3364 of them, in
//   raster order): 2 bytes {3'b000, corner, score[11:8]}, score[7:0].
//   The threshold byte pulses the core reset, so frames re-run without
//   reflash. Always-listening RX (the iCE40 dropped-byte lesson); TX
//   through a 2 KB FIFO (reply is 2 bytes per input byte — host paces).
`default_nettype none

module top (
    input  wire       clk,        // 50 MHz
    input  wire       usb_rx,
    output wire       usb_tx,
    output wire [4:0] led
);
    localparam DIV = 25;          // 2 Mbaud exact
    localparam NPIX = 64 * 64;

    wire [7:0] rxd;
    wire       rxv;
    uart_rx #(.DIV(DIV)) urx (.clk(clk), .rx(usb_rx), .data(rxd),
                              .valid(rxv));

    // frame FSM with a MAGIC arm byte: port-open glitches on the FT231X
    // corrupted the raw first byte (hw signature: t=garbage -> zero
    // corners, perfect framing); garbage rarely equals 0xA5.
    localparam [7:0] MAGIC = 8'hA5;
    reg [1:0]  st = 2'd0;          // 0 wait-magic, 1 wait-t, 2 stream
    reg [12:0] npix = 0;
    reg [7:0]  t_reg = 8'd12;
    reg        core_rst = 1'b0;
    reg        pv = 1'b0;
    reg [7:0]  pix = 0;

    always @(posedge clk) begin
        core_rst <= 1'b0;
        pv <= 1'b0;
        if (rxv) begin
            case (st)
                2'd0: if (rxd == MAGIC) st <= 2'd1;
                2'd1: begin
                    t_reg <= rxd;
                    core_rst <= 1'b1;
                    npix <= 0;
                    st <= 2'd2;
                end
                default: begin
                    pix <= rxd;
                    pv <= 1'b1;
                    if (npix == NPIX - 1)
                        st <= 2'd0;
                    npix <= npix + 1;
                end
            endcase
        end
    end
    wire streaming = (st == 2'd2);

    wire ov, oc;
    wire [11:0] os;
    fast9 #(.W(64)) core (
        .clk(clk), .rst(core_rst), .t(t_reg),
        .in_valid(pv), .in_pix(pix),
        .out_valid(ov), .out_corner(oc), .out_score(os));

    // ---- TX FIFO (2 KB) ----------------------------------------------------
    reg [7:0] fifo [0:2047];
    reg [10:0] wp = 0, rp = 0;
    wire [10:0] used = wp - rp;
    always @(posedge clk) begin
        if (ov) begin
            fifo[wp]     <= {3'b000, oc, os[11:8]};
            fifo[wp + 1] <= os[7:0];
            wp <= wp + 2;
        end
    end
    wire       tbusy;
    reg        tsend = 1'b0;
    reg [7:0]  tdat = 0;
    always @(posedge clk) begin
        tsend <= 1'b0;
        if (!tbusy && !tsend && used != 0) begin
            tdat  <= fifo[rp];
            rp    <= rp + 1;
            tsend <= 1'b1;
        end
    end
    uart_tx #(.DIV(DIV)) utx (.clk(clk), .data(tdat), .send(tsend),
                              .tx(usb_tx), .busy(tbusy));

    reg [25:0] beat = 0;
    always @(posedge clk) beat <= beat + 1;
    assign led = {beat[25], streaming, rxv, ov, used != 0};
endmodule
`default_nettype wire
