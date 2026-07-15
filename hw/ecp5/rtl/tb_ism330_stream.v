// tb_ism330_stream.v — full-system gate for the rung-3 streaming top:
// bit-level ST-style SPI slave (register file: WHO_AM_I + captured
// config writes + pattern-serving OUT regs) + DRDY INT1 generator +
// UART BFMs both directions. Gates: (1) boot config sequence lands the
// 4 expected writes, (2) 'R' hello, (3) 3 stream frames sync/checksum/
// timestamp-spacing/data-pattern exact, (4) 'g' silences the stream,
// (5) 'W'/'r' passthrough. PASS line: "IMU-STREAM PASS".
`timescale 1ns/1ps

module tb;
    localparam DIVU = 4;                    // UART clocks/bit in sim
    localparam PERIOD = 5000;               // INT1 period in clk (100 us)

    reg clk = 0;
    always #10 clk = ~clk;                  // 50 MHz

    reg  usb_rx_r = 1'b1;
    wire usb_tx;
    wire sclk, mosi, cs;
    reg  miso_r = 1'b0;
    reg  int1 = 1'b0;
    wire [4:0] led;

    top #(.BOOT_BIT(10), .DIV_SPI(4), .DIV_UART(DIVU)) dut (
        .clk(clk), .usb_rx(usb_rx_r), .usb_tx(usb_tx), .led(led),
        .imu_sclk(sclk), .imu_mosi(mosi), .imu_miso(miso_r),
        .imu_cs(cs), .imu_int1(int1), .imu_int2(1'b0));

    // ---- behavioral ISM330 slave (mode 3) ----------------------------------
    integer scount = 0;                     // DRDY sample counter
    function [7:0] rom(input [6:0] a);
        rom = (a == 7'h0F) ? 8'h6B :
              (a >= 7'h22 && a <= 7'h2D)
                  ? ((((a - 7'h22) + 8'd1) << 4) | scount[3:0])
                  : 8'h00;
    endfunction

    reg [7:0]   cmd = 0, wreg = 0;
    reg [7:0]   wcap [0:127];
    reg [127:0] wvalid = 0;
    integer nbit = 0;
    reg prev_sclk = 1, prev_cs = 1;
    always @(posedge clk) begin
        if (prev_cs && !cs)
            nbit <= 0;
        if (!cs && !prev_sclk && sclk) begin        // rising: sample MOSI
            if (nbit < 8)
                cmd <= {cmd[6:0], mosi};
            else
                wreg <= {wreg[6:0], mosi};
            nbit <= nbit + 1;
            if (nbit == 15 && !cmd[7]) begin        // write completed
                wcap[cmd[6:0]] <= {wreg[6:0], mosi};
                wvalid[cmd[6:0]] <= 1'b1;
            end
        end
        if (!cs && prev_sclk && !sclk) begin        // falling: drive SDO
            if (nbit >= 8 && cmd[7])
                miso_r <= rom(cmd[6:0]) >> (7 - (nbit - 8)) & 1'b1;
        end
        prev_sclk <= sclk;
        prev_cs <= cs;
    end

    // ---- DRDY generator after INT1_CTRL routes gyro DRDY -------------------
    initial begin
        wait (wvalid[7'h0D] && wcap[7'h0D][1]);
        repeat (20) @(posedge clk);
        forever begin
            repeat (PERIOD) @(posedge clk);
            scount = scount + 1;
            int1 = 1'b1;
            repeat (50) @(posedge clk);
            int1 = 1'b0;
        end
    end

    // ---- UART BFMs ----------------------------------------------------------
    task send_byte(input [7:0] b);
        integer k;
        begin
            @(negedge clk);
            usb_rx_r = 1'b0;                        // start
            repeat (DIVU) @(posedge clk);
            for (k = 0; k < 8; k = k + 1) begin
                usb_rx_r = b[k];
                repeat (DIVU) @(posedge clk);
            end
            usb_rx_r = 1'b1;                        // stop + gap
            repeat (2 * DIVU) @(posedge clk);
        end
    endtask

    task rx_byte(output [7:0] b);
        integer k;
        begin
            @(negedge usb_tx);
            repeat (DIVU + DIVU / 2) @(posedge clk);
            for (k = 0; k < 8; k = k + 1) begin
                b[k] = usb_tx;
                repeat (DIVU) @(posedge clk);
            end
            if (usb_tx !== 1'b1) begin
                $display("IMU-STREAM FAIL: bad stop bit");
                $finish;
            end
        end
    endtask

    // TX activity counter ('g' silence gate)
    integer txedges = 0;
    always @(negedge usb_tx) txedges = txedges + 1;

    // ---- frame reader --------------------------------------------------------
    reg [7:0]  fby;
    reg [47:0] fts;
    reg [7:0]  fdat [0:11];
    reg [7:0]  fxor;
    task rx_frame;
        integer k;
        begin
            fby = 0;
            while (fby !== 8'hAA) rx_byte(fby);     // sync hunt
            rx_byte(fby);
            if (fby !== 8'h55) begin
                $display("IMU-STREAM FAIL: sync %02x after AA", fby);
                $finish;
            end
            fxor = 0;
            for (k = 0; k < 6; k = k + 1) begin
                rx_byte(fby);
                fts[8*k +: 8] = fby;
                fxor = fxor ^ fby;
            end
            for (k = 0; k < 12; k = k + 1) begin
                rx_byte(fby);
                fdat[k] = fby;
                fxor = fxor ^ fby;
            end
            rx_byte(fby);
            if (fby !== fxor) begin
                $display("IMU-STREAM FAIL: xor %02x want %02x", fby, fxor);
                $finish;
            end
        end
    endtask

    // ---- sequence -------------------------------------------------------------
    reg [7:0]  b0, b1;
    reg [47:0] ts_prev;
    integer f, k, delta, edges0;
    initial begin
        // (1) boot config lands the 4 writes
        wait (wvalid[7'h12] & wvalid[7'h10] & wvalid[7'h11]
              & wvalid[7'h0D]);
        repeat (100) @(posedge clk);
        if (wcap[7'h12] !== 8'h44 || wcap[7'h10] !== 8'h60 ||
            wcap[7'h11] !== 8'h60 || wcap[7'h0D] !== 8'h02) begin
            $display("IMU-STREAM FAIL: cfg %02x %02x %02x %02x",
                     wcap[7'h12], wcap[7'h10], wcap[7'h11], wcap[7'h0D]);
            $finish;
        end
        // (2) 'R' hello
        send_byte("R");
        rx_byte(b0);
        rx_byte(b1);
        if (b0 !== 8'h6B || b1 !== 8'h00) begin
            $display("IMU-STREAM FAIL: R -> %02x %02x", b0, b1);
            $finish;
        end
        // (3) stream 3 frames
        send_byte("G");
        for (f = 0; f < 3; f = f + 1) begin
            rx_frame;
            for (k = 0; k < 12; k = k + 1)
                if (fdat[k] !== ((((k + 1) << 4) | (f + 1)) & 8'hFF)) begin
                    $display("IMU-STREAM FAIL: frame %0d dat[%0d] %02x",
                             f, k, fdat[k]);
                    $finish;
                end
            if (f > 0) begin
                delta = fts - ts_prev;
                if (delta < PERIOD + 50 - 100 ||
                    delta > PERIOD + 50 + 100) begin
                    $display("IMU-STREAM FAIL: ts delta %0d", delta);
                    $finish;
                end
            end
            ts_prev = fts;
        end
        // (4) stream off: allow one in-flight frame, then silence
        send_byte("g");
        repeat (2 * PERIOD) @(posedge clk);
        edges0 = txedges;
        repeat (3 * PERIOD) @(posedge clk);
        if (txedges !== edges0) begin
            $display("IMU-STREAM FAIL: tx active after 'g'");
            $finish;
        end
        // (5) passthrough write + read
        send_byte("W");
        send_byte(8'h10);
        send_byte(8'hA0);
        repeat (400) @(posedge clk);
        if (wcap[7'h10] !== 8'hA0) begin
            $display("IMU-STREAM FAIL: W 10 A0 captured %02x",
                     wcap[7'h10]);
            $finish;
        end
        send_byte("r");
        send_byte(8'h0F);
        rx_byte(b0);
        if (b0 !== 8'h6B) begin
            $display("IMU-STREAM FAIL: r 0F -> %02x", b0);
            $finish;
        end
        $display(
            "IMU-STREAM PASS: cfg 4/4, R hello 6B, 3 frames sync+xor+ts+data exact, 'g' silent, W/r passthrough OK");
        $finish;
    end

    initial begin
        #10_000_000;                        // 10 ms
        $display("IMU-STREAM FAIL: timeout");
        $finish;
    end
endmodule
