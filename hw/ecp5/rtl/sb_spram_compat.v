// sb_spram_compat.v — SB_SPRAM256KA behavioral compatibility for ECP5
// (the ice40 SOLO port, rule-1 style: top_solo.v / encoder_lean.v build
// VERBATIM; only the storage primitive is swapped). 16K x 16, sync
// 1-cycle registered read (the SPRAM contract the SOLO FSMs assume),
// nibble write masks (all SOLO instances drive MASKWREN=4'b1111, which
// yosys constant-folds to a plain write). STANDBY/SLEEP/POWEROFF are
// power pins — ignored. Maps to inferred EBR (~15 DP16KD per instance).
`default_nettype none

module SB_SPRAM256KA (
    input  wire [13:0] ADDRESS,
    input  wire [15:0] DATAIN,
    input  wire [3:0]  MASKWREN,
    input  wire        WREN,
    input  wire        CHIPSELECT,
    input  wire        CLOCK,
    input  wire        STANDBY,
    input  wire        SLEEP,
    input  wire        POWEROFF,
    output reg  [15:0] DATAOUT
);
    reg [15:0] mem [0:16383];

    always @(posedge CLOCK) begin
        if (CHIPSELECT) begin
            if (WREN) begin
                if (MASKWREN[0]) mem[ADDRESS][3:0]   <= DATAIN[3:0];
                if (MASKWREN[1]) mem[ADDRESS][7:4]   <= DATAIN[7:4];
                if (MASKWREN[2]) mem[ADDRESS][11:8]  <= DATAIN[11:8];
                if (MASKWREN[3]) mem[ADDRESS][15:12] <= DATAIN[15:12];
            end
            DATAOUT <= mem[ADDRESS];
        end
    end
endmodule
