// 8N1 UART, DIV clocks per bit (12 MHz / DIV=12 -> 1 Mbaud).
module uart_rx #(parameter DIV = 12) (
    input  wire       clk,
    input  wire       rx,
    output reg  [7:0] data,
    output reg        valid
);
    reg [1:0] sync = 2'b11;
    always @(posedge clk) sync <= {sync[0], rx};
    wire r = sync[1];

    reg [3:0] state = 0;            // 0 idle, 1..8 data bits, 9 stop
    reg [7:0] cnt = 0;
    reg [7:0] sh;
    always @(posedge clk) begin
        valid <= 1'b0;
        if (state == 0) begin
            if (!r) begin                       // start edge
                state <= 1;
                cnt   <= DIV + DIV/2 - 2;       // middle of data bit 0
            end
        end else if (cnt != 0) begin
            cnt <= cnt - 1;
        end else if (state != 9) begin
            sh    <= {r, sh[7:1]};              // LSB first
            state <= state + 1;
            cnt   <= DIV - 1;
        end else begin
            data  <= sh;
            valid <= r;                         // stop bit must be high
            state <= 0;
        end
    end
endmodule

module uart_tx #(parameter DIV = 12) (
    input  wire       clk,
    input  wire [7:0] data,
    input  wire       send,
    output wire       tx,
    output wire       busy
);
    reg [9:0] sh   = 10'h3FF;                   // idle high
    reg [3:0] bits = 0;
    reg [7:0] cnt  = 0;
    assign tx   = sh[0];
    assign busy = (bits != 0);
    always @(posedge clk) begin
        if (bits == 0) begin
            if (send) begin
                sh   <= {1'b1, data, 1'b0};     // stop, data, start
                bits <= 10;
                cnt  <= DIV - 1;
            end
        end else if (cnt != 0) begin
            cnt <= cnt - 1;
        end else begin
            sh   <= {1'b1, sh[9:1]};
            bits <= bits - 1;
            cnt  <= DIV - 1;
        end
    end
endmodule
