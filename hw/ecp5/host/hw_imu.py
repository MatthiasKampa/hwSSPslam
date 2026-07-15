"""ISM330DHCX streaming HW gate (bring-up rung 3, needs
top_ism330_stream.bit flashed and the IMU wired per the README pin map).

Sequence (each line is a machine check, mirroring hw_snap.py):
  1. 'R' hello -> WHO_AM_I == 0x6B (0xFF/0x00 = not wired / no power)
  2. 'r' passthrough spot-check of the boot config (CTRL1_XL/CTRL2_G
     0x60, CTRL3_C 0x44, INT1_CTRL 0x02)
  3. 'G' stream for N seconds: sync-hunt 21-byte frames
     (AA 55 | ts48 LE | 12 B gyro+accel LE | XOR); gate checksum-drop
     rate < 1%, rate ~417 Hz from the FPGA timestamps (20 ns units),
     ts monotone
  4. physics sanity at rest: |accel| in 0.9..1.1 g (FS +-2 g,
     0.061 mg/LSB), gyro |bias| < 3 dps (FS +-250, 8.75 mdps/LSB)
  5. write scratch/imu_log.npz (ts_s, gyro_dps[3], accel_g[3]) for the
     rung-4 fusion baselines (gravity anchor / gyro-vs-1.08deg service)

python3 hw/ecp5/host/hw_imu.py [port] [seconds]
"""
import struct
import sys
import time
from pathlib import Path

import numpy as np
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/cu.usbserial-DK0GEIG0"
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
ROOT = Path(__file__).resolve().parents[3]
G_LSB = 0.061e-3          # g/LSB at +-2 g
DPS_LSB = 8.75e-3         # dps/LSB at +-250 dps
TS_HZ = 50e6


def rreg(s, addr):
    s.write(bytes([ord("r"), addr]))
    r = s.read(1)
    return r[0] if r else None


def main():
    s = serial.Serial(PORT, 2_000_000, timeout=2.0)
    time.sleep(0.2)
    s.reset_input_buffer()

    # 1. hello
    s.write(b"R")
    r = s.read(2)
    if len(r) != 2:
        raise SystemExit("FAIL: no 'R' reply — is top_ism330_stream.bit "
                         "flashed?")
    if r[0] != 0x6B:
        raise SystemExit(f"FAIL: WHO_AM_I {r[0]:02x} != 6b "
                         f"(ff/00 = IMU not wired/powered)")
    print(f"WHO_AM_I 0x6B  ints={r[1]:02x}  OK")

    # 2. config spot-check
    want = {0x10: 0x60, 0x11: 0x60, 0x12: 0x44, 0x0D: 0x02}
    got = {a: rreg(s, a) for a in want}
    bad = {a: v for a, v in got.items() if v != want[a]}
    if bad:
        raise SystemExit(f"FAIL: config mismatch {bad} want {want}")
    print(f"config CTRL1_XL/CTRL2_G/CTRL3_C/INT1_CTRL "
          f"{[hex(got[a]) for a in (0x10, 0x11, 0x12, 0x0D)]}  OK")

    # 3. stream
    s.write(b"G")
    t_end = time.time() + SECS
    buf = bytearray()
    frames, drops = [], 0
    while time.time() < t_end:
        buf += s.read(4096)
        while True:
            k = buf.find(b"\xaa\x55")
            if k < 0 or len(buf) - k < 21:
                break
            fr = bytes(buf[k:k + 21])
            buf = buf[k + 21:]
            pay = fr[2:20]
            if (np.bitwise_xor.reduce(bytearray(pay)) != fr[20]):
                drops += 1
                continue
            ts = int.from_bytes(pay[0:6], "little")
            vals = struct.unpack("<6h", pay[6:18])
            frames.append((ts, *vals))
    s.write(b"g")
    time.sleep(0.1)
    s.reset_input_buffer()
    if len(frames) < 50:
        raise SystemExit(f"FAIL: only {len(frames)} frames in {SECS}s "
                         f"(drops {drops}) — INT1 wired?")
    a = np.array(frames, dtype=np.float64)
    ts = a[:, 0] / TS_HZ
    dts = np.diff(a[:, 0])
    if (dts <= 0).any():
        raise SystemExit("FAIL: non-monotone timestamps")
    rate = 1.0 / np.median(dts / TS_HZ)
    drop_pct = 100.0 * drops / max(len(frames) + drops, 1)
    if drop_pct > 1.0:
        raise SystemExit(f"FAIL: checksum drop rate {drop_pct:.2f}%")
    if not (300 <= rate <= 500):
        raise SystemExit(f"FAIL: rate {rate:.1f} Hz not ~417")
    print(f"stream {len(frames)} frames  rate {rate:.1f} Hz  "
          f"drops {drop_pct:.3f}%  ts monotone  OK")

    # 4. physics at rest
    gyro = a[:, 1:4] * DPS_LSB
    acc = a[:, 4:7] * G_LSB
    amag = np.linalg.norm(acc, axis=1)
    bias = np.abs(np.median(gyro, axis=0))
    print(f"|accel| med {np.median(amag):.3f} g  "
          f"gyro bias {bias.round(2).tolist()} dps")
    if not (0.9 <= np.median(amag) <= 1.1):
        raise SystemExit("FAIL: gravity magnitude off (rest assumed)")
    if (bias > 3.0).any():
        raise SystemExit("FAIL: gyro bias > 3 dps at rest")

    # 5. log for rung-4 fusion work
    out = ROOT / "scratch" / "imu_log.npz"
    np.savez_compressed(out, ts_s=ts, gyro_dps=gyro, accel_g=acc)
    print(f"wrote {out} ({len(frames)} samples)")
    print("IMU STREAM GATE PASS")


if __name__ == "__main__":
    main()
