#!/usr/bin/env python3
"""cbits wrapper — the C desc-bit consumer (cbits.c) behind a numpy-shaped
API, with self-compile and a parity/speed gate vs sspax.headio.

  CBits(head_npz).bits(gray)   gray (240,320) or (120,160) uint8
                               -> (15,20,32) bool, same contract as
                               headio.cell_bits(head, gray, source="desc")

The .so builds on first use (cc -O3 -march=native, fallback plain -O3) —
works unchanged on the robot's x86 Linux and the dev arm64 mac. Float
summation order differs from numpy einsum, so cross-impl bits can flip
where a pooled activation is within float noise of 0; the gate measures
that on REAL robot frames and enforces >= 99.9% agreement. Within one
implementation bits are deterministic, and a live system uses cbits for
EVERY grid (map + queries), so it is self-consistent.

  python3 hw/ecp5/host/cbits.py gate [capture.npz]   # parity + ms/frame
"""
import ctypes
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = HERE / "cbits.c"
LIB = HERE / "libcbits.so"

TRUNK_GEOM = [  # (kind, k, s, cin, cout) — must match the head exactly
    ("conv", 3, 2, 1, 16),
    ("conv", 3, 2, 32, 16),
    ("conv", 3, 1, 32, 32),
    ("conv", 1, 1, 64, 32),
]


def _build():
    if LIB.exists() and LIB.stat().st_mtime >= SRC.stat().st_mtime:
        return
    for flags in (["-O3", "-march=native"], ["-O3", "-mcpu=native"],
                  ["-O3"]):
        r = subprocess.run(["cc", *flags, "-shared", "-fPIC",
                            "-o", str(LIB), str(SRC)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return
    raise RuntimeError(f"cbits build failed: {r.stderr[:400]}")


def _reorder(stack):
    """headio stack [(L, wq(cout,cin,k,k) float32, b)] -> C layouts."""
    out = []
    for L, w, b in stack:
        if L["k"] == 3:
            w = np.ascontiguousarray(w.transpose(2, 3, 1, 0),
                                     np.float32)     # (ky,kx,cin,cout)
        else:
            w = np.ascontiguousarray(w[:, :, 0, 0].T, np.float32)
        out.append((w, np.ascontiguousarray(b, np.float32)))
    return out


class CBits:
    def __init__(self, head_npz):
        sys.path.insert(0, str(ROOT))
        from sspax import headio as HIO
        head = HIO.load_head(str(head_npz))
        m = head["meta"]
        geom = [(L["kind"], L["k"], L["s"], L["cin"], L["cout"])
                for L in m["trunk"]]
        assert (m["modality"], m["in_h"], m["in_w"], m["act"]) == \
            ("vision", 120, 160, "crelu") and geom == TRUNK_GEOM \
            and m["desc"] == [dict(kind="conv", k=1, s=1, cin=64,
                                   cout=32)], \
            f"head geometry not the cbits build: {geom}"
        _build()
        self.lib = ctypes.CDLL(str(LIB))
        ws = _reorder(head["trunk"]) + _reorder(head["desc"])
        self._keep = ws
        fp = ctypes.POINTER(ctypes.c_float)
        args = []
        for w, b in ws:
            args += [w.ctypes.data_as(fp), b.ctypes.data_as(fp)]
        assert self.lib.cbits_setup(*args) == 0
        self._bits = np.empty(15 * 20 * 32, np.uint8)

    def bits(self, gray):
        g = np.asarray(gray, np.uint8)
        if g.shape == (240, 320):
            g = g[::2, ::2]                  # the trained nets' resize
        assert g.shape == (120, 160), g.shape
        g = np.ascontiguousarray(g)
        u8 = ctypes.POINTER(ctypes.c_ubyte)
        self.lib.cbits_forward(
            g.ctypes.data_as(u8), self._bits.ctypes.data_as(u8), None)
        return self._bits.reshape(15, 20, 32).astype(bool)


def gate(capture=None):
    sys.path.insert(0, str(ROOT))
    from sspax import headio as HIO
    head_npz = ROOT / "sspax" / "artifacts" / "vision_head.npz"
    head = HIO.load_head(str(head_npz))
    cb = CBits(head_npz)

    # 1) parity on random images, real trained weights
    rng = np.random.default_rng(0)
    agree, n = 0, 0
    for _ in range(20):
        g = rng.integers(0, 256, (240, 320), np.uint8)
        ref = HIO.cell_bits(head, g, source="desc")
        got = cb.bits(g)
        agree += (ref == got).sum()
        n += ref.size
    print(f"random frames : {agree}/{n} bits agree "
          f"({agree / n:.6f})")
    r_rand = agree / n

    # 2) parity + timing on REAL robot frames (capture jpegs)
    r_real = None
    if capture and Path(capture).exists():
        from PIL import Image
        import io
        z = np.load(capture, allow_pickle=True)
        jbs = list(z["jpegs"])[:120]
        agree = n = 0
        grays = []
        for jb in jbs:
            im = Image.open(io.BytesIO(bytes(jb))).convert("L") \
                .resize((320, 240))
            grays.append(np.asarray(im, np.uint8))
        for g in grays:
            ref = HIO.cell_bits(head, g, source="desc")
            got = cb.bits(g)
            agree += (ref == got).sum()
            n += ref.size
        r_real = agree / n
        print(f"robot frames  : {agree}/{n} bits agree "
              f"({r_real:.6f}) over {len(grays)} jpegs")
        g = grays[0]
    else:
        g = rng.integers(0, 256, (240, 320), np.uint8)

    # 3) speed: C kernel vs numpy reference
    for _ in range(3):
        cb.bits(g)                            # warm
    t0 = time.time()
    reps = 200
    for _ in range(reps):
        cb.bits(g)
    ms_c = (time.time() - t0) / reps * 1e3
    t0 = time.time()
    for _ in range(5):
        HIO.cell_bits(head, g, source="desc")
    ms_py = (time.time() - t0) / 5 * 1e3
    print(f"speed         : C {ms_c:.2f} ms/frame ({1e3 / ms_c:.0f} Hz) "
          f"vs numpy {ms_py:.1f} ms ({ms_py / ms_c:.0f}x)")
    ok = r_rand >= 0.999 and (r_real is None or r_real >= 0.999) \
        and ms_c <= 8.0
    print(f"CBITS GATE {'PASS' if ok else 'FAIL'} "
          f"(>=99.9% parity, <=8 ms = 120 Hz headroom)")
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gate":
        cap = sys.argv[2] if len(sys.argv) > 2 else None
        sys.exit(0 if gate(cap) else 1)
    print(__doc__)
