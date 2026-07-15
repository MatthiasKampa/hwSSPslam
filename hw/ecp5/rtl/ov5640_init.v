// ov5640_init.v — walk the generated init ROM (build/ov5640_init.hex,
// from host/gen_ov5640_rom.py) through the sccb master. ROM word =
// {addr[15:0], data[7:0]}; addr 0xFFFF = delay (data = ms), addr 0x0000
// = end. nerr counts NACKed writes (saturating) — the bring-up gate
// expects 0. Delay = nested ms/cycle counters (no multiplier).
`default_nettype none

module ov5640_init #(
    parameter DEPTH = 256,
    parameter ROMFILE = "ov5640_init.hex",
    parameter MS_CYCLES = 50_000       // cycles per ms (shrunk in sim)
) (
    input  wire        clk,
    input  wire        go,             // pulse: cam powered + settled
    output reg         running = 1'b0,
    output reg         done = 1'b0,
    output reg  [7:0]  nerr = 0,
    // sccb master hookup (wr-only)
    output reg         s_start = 1'b0,
    output reg  [15:0] s_addr = 0,
    output reg  [7:0]  s_wdata = 0,
    input  wire        s_busy,
    input  wire        s_ack_err
);
    reg [23:0] rom [0:DEPTH-1];
    initial $readmemh(ROMFILE, rom);

    reg [$clog2(DEPTH)-1:0] ip = 0;
    reg [23:0] cur = 0;
    reg [2:0]  st = 0;
    reg [7:0]  dly_ms = 0;
    reg [16:0] cyc = 0;

    always @(posedge clk) begin
        s_start <= 1'b0;
        case (st)
            0: if (go) begin
                ip <= 0; nerr <= 0;
                running <= 1'b1; done <= 1'b0;
                st <= 1;
            end
            1: begin cur <= rom[ip]; st <= 2; end
            2: begin
                if (cur[23:8] == 16'h0000) begin
                    running <= 1'b0; done <= 1'b1; st <= 0;
                end else if (cur[23:8] == 16'hFFFF) begin
                    dly_ms <= cur[7:0];
                    cyc <= 0;
                    st <= 3;
                end else begin
                    s_addr <= cur[23:8];
                    s_wdata <= cur[7:0];
                    s_start <= 1'b1;
                    st <= 4;
                end
            end
            3: begin                       // delay: dly_ms x MS_CYCLES
                if (cyc != 0)
                    cyc <= cyc - 1;
                else if (dly_ms != 0) begin
                    dly_ms <= dly_ms - 1;
                    cyc <= MS_CYCLES[16:0] - 1;
                end else begin
                    ip <= ip + 1;
                    st <= 1;
                end
            end
            4: if (s_busy) st <= 5;
            5: if (!s_busy) begin
                if (s_ack_err && nerr != 8'hFF) nerr <= nerr + 1;
                ip <= ip + 1;
                st <= 1;
            end
        endcase
    end
endmodule
`default_nettype wire
