// sccb.v — SCCB (I2C-subset) master for the OV5640, write + read.
// 16-bit register address, 8-bit data; ~200 kHz SCL from 50 MHz
// (DIV=250 -> 4 phases of 62.5 kHz*4). Open-drain emulation: drive 0 or
// release (tristate via output enable); SIOD needs a board/module pullup
// (OV5640 modules carry one).
// Transactions: wr=1 -> {ID_W, addr_hi, addr_lo, data}
//               rd=1 -> {ID_W, addr_hi, addr_lo} Sr {ID_R, data(NACK)}
// ID = 0x78/0x79 (OV5640 SCCB address).
`default_nettype none

module sccb #(
    parameter DIV = 64            // quarter-bit period in clk cycles
) (
    input  wire        clk,
    input  wire        start,     // pulse; latches wr/addr/wdata
    input  wire        wr,        // 1 = write, 0 = read
    input  wire [15:0] addr,
    input  wire [7:0]  wdata,
    output reg  [7:0]  rdata = 0,
    output reg         busy = 1'b0,
    output reg         ack_err = 1'b0,
    inout  wire        siod,
    output wire        sioc
);
    // open-drain: drive low or release
    reg scl = 1'b1, sda_o = 1'b1;
    reg sda_oe = 1'b0;                    // 1 = drive sda_o (only ever 0)
    assign sioc = scl ? 1'bz : 1'b0;
    assign siod = (sda_oe && !sda_o) ? 1'b0 : 1'bz;
    wire sda_in = siod;

    localparam ID_W = 8'h78, ID_R = 8'h79;

    reg [7:0]  div = 0;
    reg [1:0]  ph = 0;                    // quarter-bit phase
    wire       tick = (div == DIV - 1);
    always @(posedge clk) div <= tick ? 8'd0 : div + 8'd1;

    // byte-level engine
    reg [3:0]  bit_n = 0;
    reg [7:0]  sh = 0;
    reg [5:0]  st = 0;                    // transaction micro-state
    reg        r_wr = 1'b1;
    reg [15:0] r_addr = 0;
    reg [7:0]  r_wd = 0;

    // states: 0 idle; 1 start-cond; byte states send 8 bits + ack;
    // sequence controlled by seq index
    reg [2:0]  seq = 0;                   // which byte of the transaction
    reg        reading = 1'b0;

    always @(posedge clk) begin
        if (st == 0) begin
            scl <= 1'b1; sda_oe <= 1'b0; sda_o <= 1'b1;
            if (start) begin
                r_wr <= wr; r_addr <= addr; r_wd <= wdata;
                busy <= 1'b1; ack_err <= 1'b0;
                seq <= 0; reading <= 1'b0;
                st <= 1; ph <= 0;
            end
        end else if (tick) begin
            ph <= ph + 2'd1;
            case (st)
                1: begin                          // START: SDA falls, SCL high
                    case (ph)
                        0: begin sda_oe <= 1'b1; sda_o <= 1'b1; scl <= 1'b1; end
                        1: sda_o <= 1'b0;
                        3: begin
                            scl <= 1'b0;
                            sh <= (seq == 0) ? ID_W
                                : (reading ? ID_R : 8'h00);
                            // seq==0 handled here; later bytes loaded in st2
                            bit_n <= 0;
                            st <= 2;
                        end
                    endcase
                end
                2: begin                          // shift 8 bits
                    case (ph)
                        0: begin sda_oe <= 1'b1; sda_o <= sh[7]; end
                        1: scl <= 1'b1;
                        3: begin
                            scl <= 1'b0;
                            sh <= {sh[6:0], 1'b0};
                            if (bit_n == 7) st <= 3;
                            bit_n <= bit_n + 4'd1;
                        end
                    endcase
                end
                3: begin                          // ACK slot (slave drives)
                    case (ph)
                        0: sda_oe <= 1'b0;
                        1: scl <= 1'b1;
                        2: if (sda_in) ack_err <= 1'b1;
                        3: begin
                            scl <= 1'b0;
                            seq <= seq + 3'd1;
                            // next byte / phase decisions
                            if (!reading) begin
                                case (seq)
                                    0: begin sh <= r_addr[15:8]; bit_n <= 0; st <= 2; end
                                    1: begin sh <= r_addr[7:0];  bit_n <= 0; st <= 2; end
                                    2: begin
                                        if (r_wr) begin
                                            sh <= r_wd; bit_n <= 0; st <= 2;
                                        end else begin
                                            reading <= 1'b1; seq <= 0;
                                            st <= 5;   // repeated start
                                        end
                                    end
                                    default: st <= 4;     // stop after data
                                endcase
                            end else begin
                                bit_n <= 0;
                                st <= 6;                  // read data byte
                            end
                        end
                    endcase
                end
                4: begin                          // STOP: SDA rises, SCL high
                    case (ph)
                        0: begin sda_oe <= 1'b1; sda_o <= 1'b0; end
                        1: scl <= 1'b1;
                        2: sda_o <= 1'b1;
                        3: begin sda_oe <= 1'b0; busy <= 1'b0; st <= 0; end
                    endcase
                end
                5: begin                          // repeated START
                    case (ph)
                        0: begin sda_oe <= 1'b1; sda_o <= 1'b1; end
                        1: scl <= 1'b1;
                        2: sda_o <= 1'b0;
                        3: begin
                            scl <= 1'b0;
                            sh <= ID_R; bit_n <= 0; st <= 2;
                        end
                    endcase
                end
                6: begin                          // read 8 bits
                    case (ph)
                        0: sda_oe <= 1'b0;
                        1: scl <= 1'b1;
                        2: rdata <= {rdata[6:0], sda_in};
                        3: begin
                            scl <= 1'b0;
                            if (bit_n == 7) st <= 7;
                            bit_n <= bit_n + 4'd1;
                        end
                    endcase
                end
                7: begin                          // NACK the single read byte
                    case (ph)
                        0: begin sda_oe <= 1'b1; sda_o <= 1'b1; end
                        1: scl <= 1'b1;
                        3: begin scl <= 1'b0; st <= 4; end
                    endcase
                end
            endcase
        end
    end
endmodule
`default_nettype wire
