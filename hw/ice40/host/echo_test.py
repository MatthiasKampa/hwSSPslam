#!/usr/bin/env python3
"""Bring-up check: send bytes, expect (b+1)&0xFF back (top_echo.v)."""
import sys
import glob
import serial

def port():
    cands = sorted(glob.glob("/dev/tty.usbserial-*1"))  # FT2232H ch B = UART
    if not cands:
        cands = sorted(glob.glob("/dev/tty.usbserial-*"))
    return sys.argv[1] if len(sys.argv) > 1 else cands[-1]

def main():
    p = port()
    with serial.Serial(p, 1_000_000, timeout=1.0) as s:
        s.reset_input_buffer()
        tx = bytes(range(0, 250, 7)) + bytes([0xFF, 0x00])
        s.write(tx)
        rx = s.read(len(tx))
        want = bytes((b + 1) & 0xFF for b in tx)
        ok = rx == want
        print(f"port {p}: sent {len(tx)}, got {len(rx)}, "
              f"{'OK (fabric compute verified)' if ok else 'MISMATCH'}")
        if not ok:
            print(" want", want.hex())
            print(" got ", rx.hex())
        sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
