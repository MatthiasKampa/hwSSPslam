// tb_ov5640_init.v — gate the init ROM walker against the GENERATED hex
// (build/ov5640_init.hex): every non-delay/non-tail entry must arrive at
// the sccb master exactly once, in order, and be ACKed by a behavioral
// write-slave (tb_sccb's slave, writes only). Checks: per-write
// addr/data vs the hex, total count, nerr==0, done.
`timescale 1ns/1ps

module tb;
    reg clk = 0;
    always #10 clk = ~clk;                     // 50 MHz

    tri1 siod;
    wire sioc;
    wire        s_busy, s_ack_err, s_start;
    wire [15:0] s_addr;
    wire [7:0]  s_wdata;
    wire        running, done;
    wire [7:0]  nerr;

    sccb #(.DIV(4)) bus (
        .clk(clk), .start(s_start), .wr(1'b1), .addr(s_addr),
        .wdata(s_wdata), .rdata(), .busy(s_busy), .ack_err(s_ack_err),
        .siod(siod), .sioc(sioc));

    reg go = 0;
    ov5640_init #(.ROMFILE("ov5640_init.hex"), .MS_CYCLES(10)) dut (
        .clk(clk), .go(go), .running(running), .done(done), .nerr(nerr),
        .s_start(s_start), .s_addr(s_addr), .s_wdata(s_wdata),
        .s_busy(s_busy), .s_ack_err(s_ack_err));

    // ---- write-only behavioral slave (ACK every 9th SCL rise) ----------
    reg sda_drive = 1'b1;
    assign siod = sda_drive ? 1'bz : 1'b0;
    integer bitcnt = 0;
    reg prev_scl = 1, prev_sda = 1;
    wire scl_now = (sioc !== 1'b0);
    wire sda_now = (siod !== 1'b0);
    always @(posedge clk) begin
        if (prev_scl && scl_now && prev_sda && !sda_now)   // START
            bitcnt <= 0;
        if (!prev_scl && scl_now)
            bitcnt <= bitcnt + 1;
        if (prev_scl && !scl_now) begin
            if ((bitcnt % 9) == 8)
                sda_drive <= 1'b0;                          // ACK slot
            else
                sda_drive <= 1'b1;
        end
        prev_scl <= scl_now;
        prev_sda <= sda_now;
    end

    // ---- reference: same hex, delays/tail skipped ----------------------
    reg [23:0] rom [0:255];
    integer nw = 0, refi = 0, errs = 0;
    always @(posedge clk)
        if (s_start) begin
            while (rom[refi][23:8] == 16'hFFFF)
                refi = refi + 1;
            if ({s_addr, s_wdata} !== rom[refi]) begin
                if (errs < 10)
                    $display("MISMATCH write %0d: got %04x=%02x want %06x",
                             nw, s_addr, s_wdata, rom[refi]);
                errs = errs + 1;
            end
            refi = refi + 1;
            nw = nw + 1;
        end

    integer i, nexp;
    initial begin
        $readmemh("ov5640_init.hex", rom);
        nexp = 0;
        for (i = 0; rom[i][23:8] != 16'h0000; i = i + 1)
            if (rom[i][23:8] != 16'hFFFF)
                nexp = nexp + 1;
        repeat (10) @(posedge clk);
        @(negedge clk);                 // drive stimulus off-edge
        go = 1;
        @(negedge clk);
        go = 0;
        wait (done);
        repeat (10) @(posedge clk);
        if (errs == 0 && nw == nexp && nerr == 0)
            $display("INIT-ROM PASS: %0d writes in order, all ACKed", nw);
        else
            $display("INIT-ROM FAIL: %0d/%0d writes, %0d mismatches, nerr=%0d",
                     nw, nexp, errs, nerr);
        $finish;
    end

    initial begin
        #20_000_000;
        $display("INIT-ROM FAIL: timeout (walker st=%0d ip=%0d nw=%0d | sccb st=%0d seq=%0d busy=%b | slave bitcnt=%0d)",
                 dut.st, dut.ip, nw, bus.st, bus.seq, s_busy, bitcnt);
        $finish;
    end
endmodule
