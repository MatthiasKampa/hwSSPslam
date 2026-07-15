// tb_spi.v — gate spi_reg.v against a bit-level ST-style SPI slave
// (mode 3): 8 command bits {RW, addr} sampled on SCLK rising; the data
// byte is driven by the slave on falling edges for reads (WHO_AM_I
// 0x0F -> 0x6B, the ISM330DHCX id) and recorded for writes.
`timescale 1ns/1ps

module tb;
    reg clk = 0;
    always #10 clk = ~clk;                 // 50 MHz

    reg        start = 0, rw = 0;
    reg  [6:0] addr = 0;
    reg  [7:0] wdata = 0;
    wire [7:0] rdata;
    wire busy, sclk, mosi, cs;
    reg  miso_r = 1'b0;

    spi_reg #(.DIV(4)) dut (
        .clk(clk), .start(start), .rw(rw), .addr(addr), .wdata(wdata),
        .rdata(rdata), .busy(busy), .sclk(sclk), .mosi(mosi),
        .miso(miso_r), .cs(cs));

    // ---- behavioral slave --------------------------------------------------
    reg [7:0] cmd = 0, wreg = 0, serve = 8'h6B;
    reg [7:0] wcap_addr = 0, wcap_data = 0;
    integer nbit = 0;
    reg prev_sclk = 1, prev_cs = 1;
    always @(posedge clk) begin
        if (prev_cs && !cs)
            nbit <= 0;
        if (!cs && !prev_sclk && sclk) begin       // rising: sample MOSI
            if (nbit < 8)
                cmd <= {cmd[6:0], mosi};
            else
                wreg <= {wreg[6:0], mosi};
            nbit <= nbit + 1;
            if (nbit == 15 && !cmd[7]) begin       // write completed
                wcap_addr <= cmd & 8'h7F;
                wcap_data <= {wreg[6:0], mosi};
            end
        end
        if (!cs && prev_sclk && !sclk) begin       // falling: drive SDO
            if (nbit >= 8 && cmd[7])
                miso_r <= serve[7 - (nbit - 8)];
        end
        prev_sclk <= sclk;
        prev_cs <= cs;
    end

    task xact(input r, input [6:0] a, input [7:0] d);
        begin
            @(negedge clk);
            rw = r; addr = a; wdata = d; start = 1;
            @(negedge clk);
            start = 0;
            wait (busy);
            wait (!busy);
            repeat (4) @(posedge clk);
        end
    endtask

    initial begin
        repeat (10) @(posedge clk);
        // WHO_AM_I read
        xact(1, 7'h0F, 8'h00);
        if (rdata !== 8'h6B) begin
            $display("SPI FAIL: WHO_AM_I got %02x want 6b", rdata);
            $finish;
        end
        // register write (CTRL1_XL = 0xA0)
        xact(0, 7'h10, 8'hA0);
        if (wcap_addr !== 8'h10 || wcap_data !== 8'hA0) begin
            $display("SPI FAIL: write captured %02x=%02x",
                     wcap_addr, wcap_data);
            $finish;
        end
        // second read after the write (bus reusable)
        serve = 8'hA5;
        xact(1, 7'h22, 8'h00);
        if (rdata !== 8'hA5) begin
            $display("SPI FAIL: reread got %02x", rdata);
            $finish;
        end
        $display("SPI PASS: WHO_AM_I 0x6B read, write captured, reread OK");
        $finish;
    end

    initial begin
        #2_000_000;
        $display("SPI FAIL: timeout");
        $finish;
    end
endmodule
