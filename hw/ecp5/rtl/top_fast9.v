// top_fast9.v — Icepi Zero synthesis/timing smoke for the fast9 core.
// LFSR pixel stream at the full 50 MHz pixel clock, W=320 line length
// (the binned OV5640 geometry); outputs fold into LEDs so nothing
// optimises away. led[4] heartbeat.
`default_nettype none

module top (
    input  wire       clk,        // 50 MHz osc (LPF: M1)
    output wire [4:0] led
);
    reg [15:0] lfsr = 16'hACE1;
    always @(posedge clk)
        lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};

    reg [25:0] beat = 0;
    always @(posedge clk) beat <= beat + 1;
    // slow-wandering threshold exercises the comparator paths
    wire [7:0] t = {2'b00, beat[25:20]} + 8'd8;

    wire ov, oc;
    wire [11:0] os;
    fast9 #(.W(320)) core (
        .clk(clk), .rst(1'b0), .t(t),
        .in_valid(1'b1), .in_pix(lfsr[7:0]),
        .out_valid(ov), .out_corner(oc), .out_score(os));

    reg [11:0] fold = 0;
    always @(posedge clk)
        if (ov) fold <= fold ^ os ^ {11'd0, oc};

    assign led = {beat[25], fold[3:0]};
endmodule
`default_nettype wire
