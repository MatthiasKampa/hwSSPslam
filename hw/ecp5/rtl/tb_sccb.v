// tb_sccb.v — SCCB master gate against a bit-level behavioral slave.
// The slave ACKs every 9th SCL rise and serves 0x56 / 0x40 on the two
// read transactions (the OV5640 chip-ID bytes at 0x300A/0x300B).
`timescale 1ns/1ps

module tb;
    reg clk = 0;
    always #10 clk = ~clk;                     // 50 MHz

    tri1 siod;                                 // pulled-up bus
    wire sioc;
    reg        start = 0, wr = 0;
    reg [15:0] addr = 0;
    reg [7:0]  wdata = 0;
    wire [7:0] rdata;
    wire busy, ack_err;

    sccb #(.DIV(8)) dut (                      // fast DIV for sim
        .clk(clk), .start(start), .wr(wr), .addr(addr), .wdata(wdata),
        .rdata(rdata), .busy(busy), .ack_err(ack_err),
        .siod(siod), .sioc(sioc));

    // ---- behavioral slave ---------------------------------------------------
    tri1 scl_b = sioc;                         // pulled-up clock line
    reg  sda_drive = 1'b1;                     // 1 = release
    assign siod = sda_drive ? 1'bz : 1'b0;

    integer bitcnt = 0;
    reg reading = 0;
    reg [7:0] serve = 8'h56;
    reg [7:0] sh;
    reg prev_scl = 1, prev_sda = 1;
    reg started = 0, post_sr = 0;      // repeated-start tracking
    wire scl_now = (scl_b !== 1'b0);
    wire sda_now = (siod !== 1'b0);

    always @(posedge clk) begin
        // detect START (SDA fall with SCL high)
        if (prev_scl && scl_now && prev_sda && !sda_now) begin
            bitcnt <= 0;
            sda_drive <= 1'b1;
            if (started) post_sr <= 1;
            started <= 1;
        end
        // detect STOP (SDA rise with SCL high)
        if (prev_scl && scl_now && !prev_sda && sda_now) begin
            started <= 0;
            post_sr <= 0;
        end
        // on each SCL rising edge count bits
        if (!prev_scl && scl_now) begin
            bitcnt <= bitcnt + 1;
        end
        // after each SCL falling edge decide drive for next bit
        if (prev_scl && !scl_now) begin
            if ((bitcnt % 9) == 8 && !(post_sr && bitcnt == 17)) begin
                sda_drive <= 1'b0;             // ACK slot (not master NACK)
                if (bitcnt == 8 && post_sr)
                    sh <= serve;               // load read byte after ID_R
            end else if (post_sr && bitcnt >= 9 && bitcnt < 17) begin
                sda_drive <= sh[7] ? 1'b1 : 1'b0;
                sh <= {sh[6:0], 1'b0};
            end else
                sda_drive <= 1'b1;
        end
        prev_scl <= scl_now;
        prev_sda <= sda_now;
    end

    task xact(input w, input [15:0] a, input [7:0] d);
        begin
            @(negedge clk);
            wr = w; addr = a; wdata = d; start = 1;
            @(negedge clk);
            start = 0;
            wait (busy);
            wait (!busy);
        end
    endtask

    initial begin
        repeat (10) @(posedge clk);
        // write transaction (register write shape)
        xact(1, 16'h3008, 8'h82);
        if (ack_err) begin $display("FAIL: write NACK"); $finish; end
        // read chip-ID high
        reading = 1; serve = 8'h56;
        xact(0, 16'h300A, 8'h00);
        if (rdata !== 8'h56 || ack_err) begin
            $display("FAIL: id-high got %02x ack_err %b", rdata, ack_err);
            $finish;
        end
        serve = 8'h40;
        xact(0, 16'h300B, 8'h00);
        if (rdata !== 8'h40 || ack_err) begin
            $display("FAIL: id-low got %02x", rdata);
            $finish;
        end
        $display("SCCB PASS: write ACKed, chip-ID reads 0x5640");
        $finish;
    end
endmodule
