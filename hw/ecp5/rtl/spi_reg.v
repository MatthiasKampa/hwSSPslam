// spi_reg.v — SPI mode-3 register master for the ISM330DHCX (ST 6-axis
// IMU). Transaction = 16 bits: {RW, addr[6:0], data[7:0]} — RW=1 read
// (slave drives the data byte on SDO), RW=0 write. SCLK idles HIGH;
// MOSI updates on the falling edge, both sides sample on the rising
// edge. Single-byte v0 (FIFO burst reads = the streaming follow-up,
// same engine with a byte counter + IF_INC auto-increment).
`default_nettype none

module spi_reg #(
    parameter DIV = 8              // half-period in clk cycles: 50 MHz
) (                                // /(2*8) = 3.125 MHz SCLK (max 10)
    input  wire       clk,
    input  wire       start,       // pulse; latches rw/addr/wdata
    input  wire       rw,          // 1 = read
    input  wire [6:0] addr,
    input  wire [7:0] wdata,
    output reg  [7:0] rdata = 0,
    output reg        busy = 1'b0,
    output reg        sclk = 1'b1,
    output reg        mosi = 1'b0,
    input  wire       miso,
    output reg        cs = 1'b1
);
    reg [15:0] sh = 0;
    reg [4:0]  nbit = 0;
    reg [7:0]  div = 0;
    reg [1:0]  st = 0;             // 0 idle, 1 shift, 2 end
    wire tick = (div == DIV - 1);

    always @(posedge clk) begin
        div <= tick ? 8'd0 : div + 8'd1;
        case (st)
            0: if (start) begin
                sh   <= {rw, addr, rw ? 8'h00 : wdata};
                cs   <= 1'b0;
                busy <= 1'b1;
                sclk <= 1'b1;
                nbit <= 0;
                div  <= 0;
                st   <= 1;
            end
            1: if (tick) begin
                if (sclk) begin            // falling edge: present bit
                    sclk <= 1'b0;
                    mosi <= sh[15];
                end else begin             // rising edge: sample
                    sclk <= 1'b1;
                    rdata <= {rdata[6:0], miso};
                    sh <= {sh[14:0], 1'b0};
                    nbit <= nbit + 1;
                    if (nbit == 5'd15)
                        st <= 2;
                end
            end
            2: if (tick) begin             // CS release half-period later
                cs <= 1'b1;
                busy <= 1'b0;
                st <= 0;
            end
        endcase
    end
endmodule
`default_nettype wire
