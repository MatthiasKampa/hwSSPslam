#!/usr/bin/env python3
"""solo_bridge.py — v7 stage-1 sim gate: resident-map segment addressing.

Generates fixtures from the VALIDATED solo vector set (segments 8 & 9 of
the classroom map + one feed scan), runs tb_solo_bridge through iverilog,
and diffs every per-ring partial against ssp_ice40.match_int with the
respective segment's codes. Candidates interleave segments A/B to prove
the base switch (the 2-line diff vs the accepted v6 core).
"""
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ssp_ice40 as G                                      # noqa: E402

ICE = Path(__file__).resolve().parents[1]
BUILD = ICE / "build"
TOOLS = Path.home() / "tools" / "oss-cad-suite" / "bin"
CELLS = TOOLS.parent / "share" / "yosys" / "ice40" / "cells_sim.v"

SEG_A, SEG_B = 8, 9


def load_fixture():
    by = bytes(int(l, 16) for l in open(BUILD / "solo_map.hex").read().split())
    codes = []
    for s in range(len(by) // 60):
        c = []
        for b in by[s * 60:(s + 1) * 60]:
            for j in range(4):
                c.append((b >> (2 * j)) & 3)
        codes.append(np.array(c, np.int64))
    fb = (BUILD / "solo_feed_k1.bin").read_bytes()
    (npts,) = struct.unpack_from("<H", fb, 0)
    off = 2
    az = np.empty(npts, np.int64)
    r = np.empty(npts, np.int64)
    w = np.empty(npts, np.int64)
    for i in range(npts):
        a_, r_, w_ = struct.unpack_from("<HHB", fb, off)
        off += 5
        az[i], r[i], w[i] = a_, r_, w_
    return codes, (az, r, w)


def main():
    codes, (az, r, w) = load_fixture()
    luts = G.make_luts()
    Q = G.encode_int(az, r, w, luts)
    sh = G.shift_for(Q)
    cands = [(SEG_A, 12, -24, 3, sh), (SEG_B, 12, -24, 3, sh),
             (SEG_A, -36, 0, 58, sh), (SEG_B, -36, 0, 58, sh),
             (SEG_A, 0, 0, 0, sh), (SEG_B, 0, 0, 0, sh),
             (SEG_B, 48, 48, 30, sh), (SEG_A, 48, 48, 30, sh)]
    with open(BUILD / "solo_pts.hex", "w") as f:
        f.write(f"{len(az):04x}\n")
        for a, rr, ww in zip(az, r, w):
            f.write(f"{int(a):04x}\n{int(rr):04x}\n{int(ww):04x}\n")
    for name, seg in (("solo_seg_a.hex", SEG_A), ("solo_seg_b.hex", SEG_B)):
        with open(BUILD / name, "w") as f:
            for c in codes[seg]:
                f.write(f"{int(c):04x}\n")
    with open(BUILD / "solo_cands.hex", "w") as f:
        f.write(f"{len(cands):04x}\n")
        for (sg, dx, dy, rho, s) in cands:
            f.write(f"{sg:04x}\n{dx & 0xffff:04x}\n{dy & 0xffff:04x}\n"
                    f"{rho:04x}\n{s:04x}\n")
    exp = []
    for (sg, dx, dy, rho, s) in cands:
        exp.append(G.match_int(Q, codes[sg], dx, dy, rho, s, luts))
    enc = sys.argv[1] if len(sys.argv) > 1 else "solo"
    enc_args = (["-DLEAN", "rtl/encoder_lean.v"] if enc == "lean"
                else ["rtl/encoder_solo.v"])
    subprocess.run([str(TOOLS / "iverilog"), "-g2012", "-o",
                    str(BUILD / "tbs.vvp")] + enc_args +
                   ["rtl/tb_solo_bridge.v", str(CELLS)], cwd=ICE, check=True)
    out = subprocess.run([str(TOOLS / "vvp"), str(BUILD / "tbs.vvp")],
                         cwd=ICE, capture_output=True, text=True, check=True)
    got = {}
    enc_bad = 0
    for line in out.stdout.splitlines():
        if line.startswith("RD "):
            _, i, re_, im_ = line.split()
            i = int(i)
            if (int(re_), int(im_)) != (int(Q[i, 0]), int(Q[i, 1])):
                if enc_bad < 5:
                    print(f"ENC MISMATCH comp {i}: rtl ({re_},{im_}) golden "
                          f"({Q[i,0]},{Q[i,1]})")
                enc_bad += 1
        if line.startswith("SC "):
            _, c, k, re_, im_ = line.split()
            got[(int(c), int(k))] = (int(re_), int(im_))
    print(f"encode readback: {240 - enc_bad}/240 bit-exact")
    bad = 0
    for c, e in enumerate(exp):
        for k in range(4):
            g = got.get((c, k))
            if g != (int(e[k, 0]), int(e[k, 1])):
                bad += 1
                print(f"MISMATCH cand {c} ring {k}: rtl {g} golden "
                      f"{(int(e[k, 0]), int(e[k, 1]))}")
    n = len(exp) * 4
    print(f"solo bridge gate: {n - bad}/{n} ring partials bit-exact "
          f"across segment-base switches "
          f"{'-> PASS' if bad == 0 else '-> FAIL'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
