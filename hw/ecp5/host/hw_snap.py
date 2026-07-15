"""OV5640 snapshot HW gate (bring-up step 2, needs top_ov5640_snap.bit
flashed and the camera wired per the README pin map).

Sequence (each line is a machine check, mirroring hw_fast9.py):
  1. poll 'R' until the init ROM reports done, nerr == 0
  2. chip-ID via the SCCB passthrough -> expect 0x5640
  3. frame geometry: lines/frame == 240, bytes/line == 320 (Y8), and an
     fps estimate from the frame counter over ~2 s (ROM rate section
     predicts ~25 fps; the live PLL/VTS servo steps from there)
  4. 'S' snapshot -> 76800 bytes -> build/snap.{npz,pgm}; sanity stats
     (std floor vs dead-bus constants) + fast9 servo smoke on the frame

python3 hw/ecp5/host/hw_snap.py [port]
"""
import sys
import time
from pathlib import Path

import numpy as np
import serial

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent / "build"
W, H = 320, 240
PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/cu.usbserial-DK0GEIG0"


def report(s):
    s.write(b"R")
    r = s.read(13)
    if len(r) != 13:
        raise SystemExit(f"FAIL: short report ({len(r)}/13 bytes) — "
                         f"is top_ov5640_snap.bit flashed?")
    return dict(frames=int.from_bytes(r[0:4], "big"),
                lines=int.from_bytes(r[4:6], "big"),
                bpl=int.from_bytes(r[6:8], "big"),
                nbytes=int.from_bytes(r[8:11], "big"),
                nerr=r[11],
                init_done=bool(r[12] & 1),
                init_running=bool(r[12] & 2))


def sccb_read(s, addr):
    s.write(bytes([ord("r"), addr >> 8, addr & 0xFF]))
    r = s.read(2)
    if len(r) != 2:
        raise SystemExit("FAIL: SCCB read reply short")
    return r[0], r[1]


def sccb_write(s, addr, data):
    s.write(bytes([ord("W"), addr >> 8, addr & 0xFF, data]))
    r = s.read(1)
    return r[0] if r else None


def main():
    s = serial.Serial(PORT, 2_000_000, timeout=3)
    time.sleep(0.3)                     # FT231X open-glitch settle
    s.reset_input_buffer()
    s.reset_output_buffer()

    r = report(s)
    for _ in range(40):
        if r["init_done"] and not r["init_running"]:
            break
        time.sleep(0.1)
        r = report(s)
    ok_init = r["init_done"] and r["nerr"] == 0
    print(f"init: done={r['init_done']} nerr={r['nerr']} "
          f"{'OK' if ok_init else 'FAIL'}")

    hi, e1 = sccb_read(s, 0x300A)
    lo, e2 = sccb_read(s, 0x300B)
    ok_id = (hi, lo, e1 | e2) == (0x56, 0x40, 0)
    print(f"chip-ID: 0x{hi:02x}{lo:02x} errs {e1}/{e2} "
          f"{'OK' if ok_id else 'FAIL (expect 0x5640)'}")

    r0 = report(s)
    t0 = time.time()
    time.sleep(2.0)
    r1 = report(s)
    fps = (r1["frames"] - r0["frames"]) / (time.time() - t0)
    ok_geo = r1["lines"] == H and r1["bpl"] == W
    print(f"geometry: {r1['lines']} lines x {r1['bpl']} bytes/line @ "
          f"{fps:.2f} fps {'OK' if ok_geo else 'FAIL (want 240x320, Y8)'}"
          f"  (ROM rate section predicts ~25 fps)")

    s.write(b"S")
    buf = bytearray()
    t0 = time.time()
    while len(buf) < W * H and time.time() - t0 < 10:
        buf += s.read(W * H - len(buf))
    s.close()
    if len(buf) < W * H:
        print(f"FAIL: snapshot short ({len(buf)}/{W * H} bytes)")
        return 1
    a = np.frombuffer(bytes(buf), np.uint8).reshape(H, W)
    np.savez_compressed(BUILD / "snap.npz", gray=a)
    with open(BUILD / "snap.pgm", "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (W, H))
        f.write(a.tobytes())
    ok_img = float(a.std()) > 5.0       # dead bus = constant bytes
    print(f"snapshot: mean {a.mean():.1f} std {a.std():.1f} "
          f"range [{a.min()}, {a.max()}] "
          f"{'OK' if ok_img else 'FAIL (flat — bus dead?)'} "
          f"-> build/snap.pgm")

    sys.path.insert(0, str(HERE))
    import golden_cam as GC
    t, (ys, xs, sc) = GC.detect_target(a, 900)
    print(f"fast9 servo smoke: t={t} n={len(ys)} (target 900)")

    if ok_init and ok_id and ok_geo and ok_img:
        print("SNAP GATE PASS")
        return 0
    print("SNAP GATE FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
