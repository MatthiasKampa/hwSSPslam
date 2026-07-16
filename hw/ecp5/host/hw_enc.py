"""ON-CHIP ENCODE silicon gate: stream REAL spot scans through the Icepi
Zero (top_stream_enc) and verify, per scan, (a) the transport digest,
(b) the on-chip-encoded VSA VECTOR bit-exact vs hw.ice40.golden.encode_int,
(c) the on-chip MAP bank's 2-bit QPSK codes vs golden mcode_from_vec.

Needs top_stream_enc flashed (see rtl/top_stream_enc.v header).

  python3 hw/ecp5/host/hw_enc.py [n_scans] [port]
"""
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import serial                                                  # noqa: E402

import hw_stream as HS                                         # noqa: E402
import hw.ice40.golden as G                                    # noqa: E402
import runners.spot as SP                                      # noqa: E402

PORT = "/dev/cu.usbserial-DK0GEIG0"


def collect(s, deadline, want, buf):
    """Read framed packets until a packet of type `want` arrives (or
    deadline); return all packets seen. `buf` = caller-owned stream
    buffer (survives across calls)."""
    out = []
    while time.time() < deadline:
        out += HS.read_pkts(s, 1, timeout=0.05, buf=buf)
        if any(t == want for t, _ in out):
            break
    return out


def main(n=25, port=PORT):
    luts = G.make_luts()
    b = SP.make_bundle()
    s = serial.Serial(port, 2_000_000, timeout=0.02)
    time.sleep(0.3)
    s.reset_input_buffer()
    tx = HS.Sender(s)
    tx.ctrl(0)
    rxbuf = bytearray()
    ok_dig = ok_vec = ok_map = 0
    t_enc = []
    for k in range(n):
        r = np.asarray(b["keys"][k][0], float)
        mm = np.where(np.isfinite(r), np.clip(np.round(r * 1000), 0, 65535),
                      0.0).astype(np.uint16)[None, :]
        t0 = time.time()
        tx.lidar_frame(k, mm, [33], cols_per_pkt=8)  # <=8: encoder keeps up
        pk = collect(s, t0 + 1.5, 0x92, rxbuf)       # echo + VEC first
        tx.pkt(0x0F, struct.pack("<B", k % 64))      # THEN read the slot
        pk += collect(s, time.time() + 0.5, 0x93, rxbuf)
        # golden on the transmitted integers (kept-point law on chip)
        keep = (mm[0] > 0) & (mm[0] <= G.R_MASK_MM)
        az = np.flatnonzero(keep).astype(np.int32)
        exp = G.encode_int(az, mm[0][keep].astype(np.int32),
                           np.full(len(az), 127, np.int32), luts)
        expm = G.mcode_from_vec(exp[:, 0] + 1j * exp[:, 1])
        ref = HS.digest_ref(np.ascontiguousarray(mm.T).tobytes())
        for typ, pl in pk:
            if typ == 0x90:
                _, fid, dg, cnt = struct.unpack("<BIHI", pl)
                ok_dig += (fid == k and dg == ref and cnt == mm.nbytes)
            elif typ == 0x92:
                fid, = struct.unpack_from("<I", pl, 0)
                acc = np.frombuffer(pl[4:], "<i4").reshape(240, 2)
                if fid == k and np.array_equal(acc, exp):
                    ok_vec += 1
                    t_enc.append(time.time() - t0)
                elif fid == k:
                    d = np.abs(acc.astype(np.int64) - exp).max()
                    print(f"  kf {k}: VEC mismatch (max |d| = {d})")
            elif typ == 0x93:
                seg = pl[0]
                codes = np.frombuffer(pl[1:], np.uint8)
                got = np.empty(240, np.int64)
                for j in range(4):
                    got[j::4] = (codes >> (2 * j)) & 3
                ok_map += (seg == k % 64 and np.array_equal(got, expm))
    print(f"HW ENCODE GATE: {n} real spot scans through silicon")
    print(f"  transport digests : {ok_dig}/{n}")
    print(f"  on-chip VSA vector: {ok_vec}/{n} bit-exact vs golden "
          f"(med roundtrip {np.median(t_enc) * 1e3:.0f} ms)"
          if t_enc else "  on-chip VSA vector: 0 received")
    print(f"  on-chip map codes : {ok_map}/{n} == golden mcodes")
    if ok_dig == ok_vec == ok_map == n:
        print("HW ENCODE GATE PASS: the VSA encoder + compressed map store "
              "are bit-exact ON SILICON")
        return True
    print("HW ENCODE GATE FAIL")
    return False


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    p = sys.argv[2] if len(sys.argv) > 2 else PORT
    sys.exit(0 if main(n, p) else 1)
