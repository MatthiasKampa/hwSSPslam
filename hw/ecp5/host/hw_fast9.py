"""fast9 SILICON gate: stream the golden fixture through the Icepi Zero
and compare bit-exact (the house hw-replay discipline).

Needs top_fast9_uart flashed (make TOP=top_fast9_uart RTL="rtl/fast9.v
../ice40/rtl/uart.v rtl/top_fast9_uart.v" build prog) and the fixtures
in ../build (python3 -m runners.spot_cam vectors school_run2).

python3 hw/ecp5/host/hw_fast9.py [port]
"""
import sys
import time
from pathlib import Path

import serial

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent / "build"
W = H = 64
N_OUT = (W - 6) * (H - 6)
PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/cu.usbserial-DK0GEIG0"


def main():
    img = [int(x, 16) for x in
           (BUILD / "fast9_img.hex").read_text().split()]
    exp = [int(x, 16) for x in
           (BUILD / "fast9_exp.hex").read_text().split()]
    t = int((BUILD / "fast9_thresh.txt").read_text().strip())
    assert len(img) == W * H
    s = serial.Serial(PORT, 2_000_000, timeout=2)
    time.sleep(0.3)                    # let post-open line glitches settle
    s.reset_input_buffer()
    s.reset_output_buffer()
    s.write(bytes([0xA5, t]))          # magic arm byte + threshold
    reply = bytearray()
    # paced sends: the reply is ~1.64x the send volume at equal baud
    CHUNK = 32
    for i in range(0, W * H, CHUNK):
        s.write(bytes(img[i:i + CHUNK]))
        time.sleep(CHUNK * 5e-6 * 2.2)
        reply += s.read(s.in_waiting or 0)
    t0 = time.time()
    while len(reply) < 2 * N_OUT and time.time() - t0 < 5:
        reply += s.read(2 * N_OUT - len(reply))
    s.close()
    if len(reply) < 2 * N_OUT:
        print(f"FAIL: short reply {len(reply)}/{2 * N_OUT} bytes")
        return 1
    errs = 0
    k = 0
    for y in range(6, H):
        for x in range(6, W):
            want = exp[(y - 3) * W + (x - 3)]
            got = (reply[2 * k] << 8) | reply[2 * k + 1]
            if got != want:
                if errs < 10:
                    print(f"MISMATCH centre ({y-3},{x-3}): got "
                          f"{got:04x} want {want:04x}")
                errs += 1
            k += 1
    if errs == 0:
        print(f"HW PASS: {N_OUT} centres bit-exact on silicon (t={t})")
        return 0
    print(f"HW FAIL: {errs}/{N_OUT} mismatches")
    return 1


if __name__ == "__main__":
    sys.exit(main())
