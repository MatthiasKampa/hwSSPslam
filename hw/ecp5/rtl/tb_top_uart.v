// tb_top_uart.v — full-system sim of top_fast9_uart: drives usb_rx with
// serialized {threshold, 4096 pixels}, captures usb_tx replies, compares
// against the golden fixture exactly like the host script. The uart
// cores themselves serve as the host BFM.
`timescale 1ns/1ps

module tb;
    localparam W = 64, H = 64, NOUT = (W - 6) * (H - 6);
    localparam THRESH = `THRESH;

    reg clk = 0;
    always #10 clk = ~clk;                 // 50 MHz

    wire rx_line, tx_line;
    top dut (.clk(clk), .usb_rx(rx_line), .usb_tx(tx_line), .led());

    // host-side serializer / deserializer (same DIV)
    reg  [7:0] hd = 0;
    reg        hsend = 0;
    wire       hbusy;
    uart_tx #(.DIV(25)) host_tx (.clk(clk), .data(hd), .send(hsend),
                                 .tx(rx_line), .busy(hbusy));
    wire [7:0] rd;
    wire       rv;
    uart_rx #(.DIV(25)) host_rx (.clk(clk), .rx(tx_line), .data(rd),
                                 .valid(rv));

    reg [7:0]  img [0:W*H-1];
    reg [15:0] exp [0:W*H-1];
    reg [7:0]  got [0:2*NOUT-1];
    integer    ngot = 0;
    always @(posedge clk)
        if (rv && ngot < 2 * NOUT) begin
            got[ngot] = rd;
            ngot = ngot + 1;
        end

    task send(input [7:0] b);
        begin
            @(negedge clk);
            hd = b; hsend = 1;
            @(negedge clk);
            hsend = 0;
            wait (hbusy);
            wait (!hbusy);
            repeat (300) @(posedge clk);   // pace 2.2x (reply = 2 bytes
        end                                // per pixel; matches the host)
    endtask

    integer i, y, x, k, errs;
    reg [15:0] want, g16;
    initial begin
        $readmemh("fast9_img.hex", img);
        $readmemh("fast9_exp.hex", exp);
        repeat (10) @(posedge clk);
        send(8'hA5);
        send(THRESH[7:0]);
        for (i = 0; i < W * H; i = i + 1)
            send(img[i]);
        // drain: wait for the reply to finish
        i = 0;
        while (ngot < 2 * NOUT && i < 2_000_000) begin
            @(posedge clk);
            i = i + 1;
        end
        errs = 0;
        k = 0;
        for (y = 6; y < H; y = y + 1)
            for (x = 6; x < W; x = x + 1) begin
                want = exp[(y - 3) * W + (x - 3)];
                g16 = {got[2 * k], got[2 * k + 1]};
                if (g16 !== want) begin
                    if (errs < 10)
                        $display("MISMATCH (%0d,%0d): got %04x want %04x",
                                 y - 3, x - 3, g16, want);
                    errs = errs + 1;
                end
                k = k + 1;
            end
        if (errs == 0 && ngot == 2 * NOUT)
            $display("TOP-SIM PASS: %0d centres bit-exact via UART top (t=%0d)",
                     NOUT, THRESH);
        else
            $display("TOP-SIM FAIL: %0d mismatches, %0d/%0d bytes",
                     errs, ngot, 2 * NOUT);
        $finish;
    end
endmodule
