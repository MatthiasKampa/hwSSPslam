#!/usr/bin/env python3
"""solo_step.py — v7 stage-2 sim gate: the FULL tracker step loop.

Converts the validated solo fixture (map/anchors/feed/expect for K_TRY=1)
into tb-friendly word-hex images (incl. the per-kf golden shift_for — the
on-chip max-abs scan is stage 3), runs tb_solo_step through iverilog, and
diffs every keyframe's (x, y, h, ai, score, state) against the expect
stream. Usage: solo_step.py [n_kf] [suffix] [enc] — suffix picks the
fixture pair solo_feed_<suffix>.bin / solo_expect_<suffix>.bin ("k1"
classroom default; "rl" = the kidnap/relock fixture; "l1"/"lr" = the
v7-lean REFINE=0 fixtures). enc = "solo" (default, v6-parallel core) or
"lean" (serial-ring core; also sets tracker REFINE=0 to match the lean
fixtures). Prefix n_kf for iteration.
"""
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import hw.ice40.golden as G                                      # noqa: E402

ICE = Path(__file__).resolve().parents[1]
BUILD = ICE / "build"
TOOLS = Path.home() / "tools" / "oss-cad-suite" / "bin"
CELLS = TOOLS.parent / "share" / "yosys" / "ice40" / "cells_sim.v"
N_KF = int(sys.argv[1]) if len(sys.argv) > 1 else 220
SUF = sys.argv[2] if len(sys.argv) > 2 else "k1"
ENC = sys.argv[3] if len(sys.argv) > 3 else "solo"


def main():
    fam = "_l" if SUF.startswith("l") else ""      # lean fixture family
    by = bytes(int(l, 16)
               for l in open(BUILD / f"solo_map{fam}.hex").read().split())
    n_seg = len(by) // 60
    with open(BUILD / "solo_mapw.hex", "w") as f:
        for s in range(n_seg):
            for b in by[s * 60:(s + 1) * 60]:
                for j in range(4):
                    f.write(f"{(b >> (2 * j)) & 3:04x}\n")
    ab = bytes(int(l, 16)
               for l in open(BUILD / f"solo_anchors{fam}.hex").read().split())
    with open(BUILD / "solo_ancw.hex", "w") as f:
        for i in range(0, len(ab), 2):
            f.write(f"{ab[i] | (ab[i + 1] << 8):04x}\n")
    fb = (BUILD / f"solo_feed_{SUF}.bin").read_bytes()
    eb = (BUILD / f"solo_expect_{SUF}.bin").read_bytes()
    REC = struct.calcsize("<iihHiB")
    luts = G.make_luts()
    words = []
    off = 0
    kf = 0
    exp = []
    while off < len(fb) and kf < N_KF:
        (npts,) = struct.unpack_from("<H", fb, off)
        off += 2
        words.append(npts)
        az = np.empty(npts, np.int64)
        r = np.empty(npts, np.int64)
        w = np.empty(npts, np.int64)
        for i in range(npts):
            a_, r_, w_ = struct.unpack_from("<HHB", fb, off)
            off += 5
            az[i], r[i], w[i] = a_, r_, w_
            words += [a_, r_, w_]
        px, py, ph = struct.unpack_from("<iih", fb, off)
        off += 10
        Q = G.encode_int(az, r, w, luts)
        sh = G.shift_for(Q)
        words += [sh, px & 0xffff, (px >> 16) & 0xffff,
                  py & 0xffff, (py >> 16) & 0xffff, ph & 0xffff]
        exp.append(struct.unpack_from("<iihHiB", eb, kf * REC))
        kf += 1
    hdr = [n_seg, kf]
    with open(BUILD / "solo_feedw.hex", "w") as f:
        for v in hdr + words:
            f.write(f"{v & 0xffff:04x}\n")
    print(f"fixture: {n_seg} segs, {kf} kf, {len(words) + 2} feed words",
          flush=True)
    enc_args = (["-DLEAN", "-DREFINE0", "rtl/encoder_lean.v"]
                if ENC == "lean" else ["rtl/encoder_solo.v"])
    subprocess.run([str(TOOLS / "iverilog"), "-g2012", "-o",
                    str(BUILD / "tbstep.vvp")] + enc_args +
                   ["rtl/solo_tracker.v", "rtl/tb_solo_step.v",
                    str(CELLS)], cwd=ICE, check=True)
    out = subprocess.run([str(TOOLS / "vvp"), str(BUILD / "tbstep.vvp")],
                         cwd=ICE, capture_output=True, text=True, check=True)
    bad = 0
    seen = 0
    for line in out.stdout.splitlines():
        if not line.startswith("KF "):
            continue
        p = line.split()
        n = int(p[1])
        got = (int(p[2]), int(p[3]), int(p[4]), int(p[5]), int(p[6]),
               int(p[7]))
        e = exp[n]
        want = (e[0], e[1], e[2] % 960, e[3], e[4], e[5])
        seen += 1
        if got != want:
            if bad < 8:
                print(f"MISMATCH kf {n}: rtl {got} golden {want}")
            bad += 1
    print(f"solo step gate: {seen - bad}/{seen} keyframes bit-exact "
          f"{'-> PASS' if bad == 0 and seen == kf else '-> FAIL'}")
    return 0 if bad == 0 and seen == kf else 1


if __name__ == "__main__":
    sys.exit(main())
