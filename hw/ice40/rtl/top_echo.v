// Bring-up: UART echo-plus-one at 1 Mbaud. Each received byte b is answered
// with (b+1)&0xFF -- proves clock, RX, TX and fabric compute (a pure echo
// could be a wire loopback). Green LED toggles per byte; red heartbeats.
module top (
    input  wire CLK,        // 12 MHz
    input  wire RX,
    output wire TX,
    output wire LEDR_N,
    output wire LEDG_N
);
    wire [7:0] rxd;
    wire       rxv;
    uart_rx #(.DIV(12)) u_rx (.clk(CLK), .rx(RX), .data(rxd), .valid(rxv));

    reg  [7:0] txd;
    reg        send = 0;
    wire       busy;
    uart_tx #(.DIV(12)) u_tx (.clk(CLK), .data(txd), .send(send), .tx(TX), .busy(busy));

    reg [7:0] pend;
    reg       have = 0;
    reg       act  = 0;
    always @(posedge CLK) begin
        send <= 1'b0;
        if (rxv) begin
            pend <= rxd + 8'd1;
            have <= 1'b1;
            act  <= ~act;
        end
        if (have && !busy && !send) begin
            txd  <= pend;
            send <= 1'b1;
            have <= 1'b0;
        end
    end

    reg [23:0] hb = 0;
    always @(posedge CLK) hb <= hb + 1;
    assign LEDR_N = ~hb[23];
    assign LEDG_N = ~act;
endmodule
