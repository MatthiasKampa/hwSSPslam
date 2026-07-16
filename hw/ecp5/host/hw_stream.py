"""Virtual-sensor stream: host reference sender + G1 silicon loopback gate
(STREAM.md). This file IS the robot-side reference implementation — pure
pyserial, runs unchanged on the robot's Linux box (swap the port).

Needs top_stream flashed:
  make TOP=top_stream RTL="rtl/stream_ingest.v ../ice40/rtl/uart.v \
       rtl/top_stream.v" build prog

  python3 hw/ecp5/host/hw_stream.py gate [port] [baud]   # G1: synthetic
  python3 hw/ecp5/host/hw_stream.py soak [port] [secs]   # continuous rate
"""
import struct
import sys
import time

import numpy as np
import serial

PORT = "/dev/cu.usbserial-DK0GEIG0"
BAUD = 2_000_000
MAGIC = b"\xa5\x5a"


def crc16(data, crc=0xFFFF):
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021 if crc & 0x8000 else crc << 1) & 0xFFFF
    return crc


class Sender:
    def __init__(self, s):
        self.s, self.seq = s, 0

    def pkt(self, typ, payload):
        hdr = struct.pack("<BBHH", typ, 0, len(payload), self.seq)
        self.seq = (self.seq + 1) & 0xFFFF
        body = hdr + payload
        self.s.write(MAGIC + body + struct.pack("<H", crc16(body)))

    def cam_frame(self, fid, img, t_us=0, rows_per_pkt=8):
        h, w = img.shape
        self.pkt(0x03, struct.pack("<IQHHB", fid, t_us, w, h, 0))
        for r0 in range(0, h, rows_per_pkt):
            n = min(rows_per_pkt, h - r0)
            self.pkt(0x04, struct.pack("<IHB", fid, r0, n)
                     + img[r0:r0 + n].tobytes())

    def lidar_frame(self, fid, rng_mm, ring_ids, t_us=0, cols_per_pkt=32):
        n_rings, n_az = rng_mm.shape          # (rings, az) uint16 mm
        self.pkt(0x01, struct.pack("<IQB", fid, t_us, n_rings)
                 + bytes(ring_ids) + struct.pack("<HB", n_az, 1))
        by_col = np.ascontiguousarray(rng_mm.T)        # (az, rings) LE
        for c0 in range(0, n_az, cols_per_pkt):
            n = min(cols_per_pkt, n_az - c0)
            self.pkt(0x02, struct.pack("<IHB", fid, c0, n)
                     + by_col[c0:c0 + n].tobytes())

    def ctrl(self, cmd, arg=0):
        self.pkt(0x10, struct.pack("<BI", cmd, arg))


def read_pkts(s, want, timeout=5.0, buf=None):
    """Parse framed packets. Pass a CALLER-OWNED bytearray as `buf` when
    calling repeatedly on one stream — a return mid-packet keeps the
    unparsed tail there (a fresh local buffer would drop it and desync
    on the next call; cost us the first silicon VEC packets)."""
    if buf is None:
        buf = bytearray()
    out, t0 = [], time.time()
    while len(out) < want and time.time() - t0 < timeout:
        buf += s.read(256)
        while True:
            i = buf.find(MAGIC)
            if i < 0 or len(buf) < i + 8:
                break
            typ, _, ln, seq = struct.unpack_from("<BBHH", buf, i + 2)
            if len(buf) < i + 8 + ln + 2:
                break
            body = bytes(buf[i + 2:i + 8 + ln])
            rxc, = struct.unpack_from("<H", buf, i + 8 + ln)
            if crc16(body) == rxc:
                out.append((typ, body[6:]))
            del buf[:i + 8 + ln + 2]
    return out


def digest_ref(*chunks):
    c = 0xFFFF
    for ch in chunks:
        c = crc16(ch, c)
    return c


def gate(port=PORT, baud=BAUD, n_cam=6, n_lid=12):
    rng = np.random.default_rng(0)
    s = serial.Serial(port, baud, timeout=0.5)
    time.sleep(0.3)
    s.reset_input_buffer()
    tx = Sender(s)
    tx.ctrl(0)                                   # reset counters

    cams = [rng.integers(0, 256, (240, 320), dtype=np.uint8).astype(np.uint8)
            for _ in range(n_cam)]
    lids = [(rng.uniform(200, 60000, (3, 1024))).astype(np.uint16)
            for _ in range(n_lid)]
    t0 = time.time()
    nbytes = 0
    for i in range(max(n_cam, n_lid)):           # interleaved, lidar-heavy
        if i < n_lid:
            tx.lidar_frame(100 + i, lids[i], [16, 33, 50])
            nbytes += lids[i].nbytes
        if i < n_cam:
            tx.cam_frame(i, cams[i])
            nbytes += cams[i].nbytes
    dt = time.time() - t0
    tx.ctrl(1)                                   # status request

    pkts = read_pkts(s, n_cam + n_lid + 1)
    ok = fail = 0
    stat = None
    for typ, pl in pkts:
        if typ == 0x90:
            stream, fid, dg, cnt = struct.unpack("<BIHI", pl)
            if stream == 3:
                ref = digest_ref(cams[fid].tobytes())
                exp_cnt = cams[fid].size
            else:
                ref = digest_ref(
                    np.ascontiguousarray(lids[fid - 100].T).tobytes())
                exp_cnt = lids[fid - 100].nbytes
            good = dg == ref and cnt == exp_cnt
            ok += good
            fail += not good
            if not good:
                print(f"  MISMATCH {'cam' if stream == 3 else 'lidar'} "
                      f"{fid}: dg {dg:04x} ref {ref:04x} cnt {cnt}")
        elif typ == 0x80:
            stat = struct.unpack("<IHHHH", pl)
    print(f"streamed {n_cam} cam QVGA + {n_lid} lidar 3x1024 frames "
          f"({nbytes/1024:.0f} KB payload) in {dt:.2f}s "
          f"= {nbytes/dt/1024:.0f} KB/s goodput @ {baud/1e6:g} Mbaud")
    if stat:
        print(f"  fpga status: pkts_ok {stat[0]} crc_drops {stat[1]} "
              f"seq_gaps {stat[2]} frames cam {stat[3]} lidar {stat[4]}")
    print(f"  digests: {ok} bit-exact, {fail} mismatched, "
          f"{n_cam + n_lid - ok - fail} missing")
    if fail == 0 and ok == n_cam + n_lid and stat and stat[1] == 0 \
            and stat[2] == 0:
        print("HW STREAM GATE PASS: all frame digests bit-exact on silicon, "
              "zero drops/gaps")
        return True
    print("HW STREAM GATE FAIL")
    return False


def soak(port=PORT, secs=10, baud=BAUD):
    rng = np.random.default_rng(1)
    s = serial.Serial(port, baud, timeout=0.5)
    time.sleep(0.3)
    s.reset_input_buffer()
    tx = Sender(s)
    tx.ctrl(0)
    tx.ctrl(2, 0)                                # echo off: pure ingest rate
    lid = (rng.uniform(200, 60000, (3, 1024))).astype(np.uint16)
    t0, nb, nf = time.time(), 0, 0
    while time.time() - t0 < secs:
        tx.lidar_frame(nf, lid, [16, 33, 50])
        nb += lid.nbytes
        nf += 1
    dt = time.time() - t0
    tx.ctrl(2, 1)
    tx.ctrl(1)
    pkts = read_pkts(s, 1, timeout=3.0)
    print(f"soak: {nf} lidar frames, {nb/dt/1024:.0f} KB/s payload "
          f"({nf/dt:.1f} fps vs 20 Hz live target)")
    for typ, pl in pkts:
        if typ == 0x80:
            st = struct.unpack("<IHHHH", pl)
            print(f"  fpga: pkts_ok {st[0]} crc_drops {st[1]} "
                  f"seq_gaps {st[2]} lidar frames {st[4]}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gate"
    if cmd == "gate":
        p = sys.argv[2] if len(sys.argv) > 2 else PORT
        b = int(sys.argv[3]) if len(sys.argv) > 3 else BAUD
        sys.exit(0 if gate(p, b) else 1)
    elif cmd == "soak":
        p = sys.argv[2] if len(sys.argv) > 2 else PORT
        t = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        soak(p, t)
