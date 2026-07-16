#!/usr/bin/env python3
"""solo_host — host client for the ECP5 SOLO standalone-SLAM top
(top_solo_ecp5.bit, 1 Mbaud). The chip runs the FULL loop (encode ->
track -> fold -> freeze -> resident map); this client is data transport
only: keyframes (points + odom deltas) in, pose frames out, map dump on
demand. Byte contract mirrors hw/ice40/host/solo_top.py golden()
byte-for-byte (unit-tested below).

  python3 hw/ecp5/host/solo_host.py selftest     # byte-layer vs golden
  python3 hw/ecp5/host/solo_host.py ping [port]  # board check
"""
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import hw.ice40.golden as G                                    # noqa: E402

U = 0.25 / 256.0                # position unit, meters
TAU_Q = 960                     # heading units per revolution
PORT = "/dev/ttyUSB0"


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def kf_bytes(az, r_mm, w, d_q):
    kb = bytearray([0x30]) + struct.pack("<H", len(az))
    for a, r, ww in zip(az, r_mm, w):
        kb += struct.pack("<HHB", int(a), int(r), int(ww))
    kb += struct.pack("<hhh", int(d_q[0]), int(d_q[1]), int(d_q[2]))
    return bytes(kb)


class SoloChip:
    """One serial owner. All methods block on the chip's reply (the
    reply IS the flow control, per the protocol)."""

    def __init__(self, port=PORT, baud=1_000_000):
        import serial
        self.s = serial.Serial(port, baud, timeout=0.05)
        time.sleep(0.2)
        self.s.reset_input_buffer()
        self.odom_prev = None
        self.n_kf = 0
        self.last = (0.0, 0.0, 0.0, 3, 0, 0)   # x y yaw state frozen nseg
        self.n_seg = 0

    def _rd(self, n, deadline_s=1.0):
        out = bytearray()
        t0 = time.time()
        while len(out) < n and time.time() - t0 < deadline_s:
            out += self.s.read(n - len(out))
        return bytes(out)

    def ping(self):
        self.s.write(bytes([0x20]))
        return self._rd(1) == b"\xa5"

    def set_pose(self, x_m=0.0, y_m=0.0, yaw=0.0):
        q = (int(round(x_m / U)), int(round(y_m / U)),
             int(round(yaw * TAU_Q / (2 * np.pi))) % TAU_Q)
        self.s.write(bytes([0x25]) + struct.pack("<iiH", *q))
        return self._rd(1) == b"\x2f"

    def set_mode(self, mapping=True, stream=True):
        self.s.write(bytes([0x26, (mapping << 0) | (stream << 1)]))
        return self._rd(1) == b"\xa6"

    def reset_map(self):
        self.s.write(bytes([0x29]))
        self.odom_prev = None
        self.n_seg = 0
        return self._rd(1) == b"\xa7"

    def keyframe(self, r_scan, odom_pose=None):
        """r_scan: 1024 ranges (m, inf=miss). odom_pose: world (x,y,yaw)
        from the platform (delta computed here, body frame); None ->
        zero delta (chip predicts from its own pose). -> (x, y, yaw,
        state, frozen) in meters/rad; state 0 track / 1 hold / 2 relock
        / 3 open-loop (n<5 or empty map)."""
        az, r_mm, w = G.scan_to_ints(np.asarray(r_scan, float))
        if odom_pose is None or self.odom_prev is None:
            d_q = (0, 0, 0)
        else:
            p, c = np.asarray(self.odom_prev), np.asarray(odom_pose)
            dw = c[:2] - p[:2]
            cy, sy = np.cos(-p[2]), np.sin(-p[2])
            d = (cy * dw[0] - sy * dw[1], sy * dw[0] + cy * dw[1],
                 wrap(c[2] - p[2]))
            d_q = (int(round(d[0] / U)), int(round(d[1] / U)),
                   int(round(d[2] * TAU_Q / (2 * np.pi))))
        if odom_pose is not None:
            self.odom_prev = np.asarray(odom_pose, float).copy()
        self.s.write(kf_bytes(az, r_mm, w, d_q))
        fr = self._rd(11)
        self.n_kf += 1
        if len(fr) != 11 or (fr[0] & 0xF0) != 0xB0:
            return None                        # frame lost — caller logs
        x, y = struct.unpack_from("<ii", fr, 1)
        h = struct.unpack_from("<H", fr, 9)[0]
        state, frozen = fr[0] & 3, (fr[0] >> 3) & 1
        self.n_seg += frozen
        self.last = (x * U, y * U, h * 2 * np.pi / TAU_Q, state,
                     frozen, self.n_seg)
        return self.last

    def dump(self):
        """-> [(anchor_pose(3) m/rad, codes(240) uint2, alive(240) bool,
        scales(4)), ...] — the chip-built resident map."""
        self.s.write(bytes([0x28]))
        hd = self._rd(2, 2.0)
        if len(hd) != 2 or hd[0] != 0xD0:
            return None
        n_seg = hd[1]
        segs = []
        for _ in range(n_seg):
            b = self._rd(60 + 30 + 8 + 16, 3.0)
            if len(b) != 114:
                return None
            codes = np.empty(240, np.uint8)
            for i in range(60):
                for j in range(4):
                    codes[i * 4 + j] = (b[i] >> (2 * j)) & 3
            bits = int.from_bytes(b[60:90], "little")
            alive = np.array([(bits >> i) & 1 for i in range(240)], bool)
            scales = [struct.unpack_from("<H", b, 90 + 2 * i)[0]
                      for i in range(4)]
            ax, ay, ah = struct.unpack_from("<iih", b, 98)
            segs.append((np.array([ax * U, ay * U,
                                   ah * 2 * np.pi / TAU_Q]),
                         codes, alive, scales))
        return segs


def selftest():
    """Byte-layer parity vs solo_top.py's golden() keyframe encoding."""
    rng = np.random.default_rng(0)
    az = rng.integers(0, 1024, 50).astype(np.int32)
    r = rng.integers(300, 30000, 50).astype(np.int32)
    w = np.full(50, 127, np.int32)
    d_q = (-305, 203, 12)
    kb = kf_bytes(az, r, w, d_q)
    ref = bytearray([0x30]) + struct.pack("<H", 50)
    for a, rr, ww in zip(az, r, w):
        ref += struct.pack("<HHB", int(a), int(rr), int(ww))
    ref += struct.pack("<hhh", *d_q)
    assert kb == bytes(ref)
    # unit laws
    assert abs(U - 0.0009765625) < 1e-12
    assert int(round(0.30 / U)) == 307
    print("solo_host selftest ok (keyframe bytes == golden encoding; "
          "U/TAU_Q pinned)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        selftest()
    elif cmd == "ping":
        c = SoloChip(sys.argv[2] if len(sys.argv) > 2 else PORT)
        print("ping:", "OK" if c.ping() else "NO REPLY")
