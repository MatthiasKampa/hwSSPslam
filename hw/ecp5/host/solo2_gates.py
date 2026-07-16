#!/usr/bin/env python3
"""solo2_gates.py — v8 sim gates (ECP5 track).

GATE 1 (step parity): solo_tracker2.v must reproduce the UNTOUCHED
historical v7-lean fixtures (l1 classroom 220 kf + lr kidnap/relock)
verbatim — v8 changes no tracking semantics. Reuses the ice40
solo_step.py flow with the RTL list swapped via the V8_STEP_SHIM.

GATE 2 (top, novelty): top_solo2.v vs the v8 golden (Solo2Tracker +
fold_gate) over the classroom self-mapping UART protocol — pose frames
+ full dump byte-exact, plus a RESET-MAP arm. Adapted from ice40
solo_top.py (imported for its fixture writers where reusable).

  python3 hw/ecp5/host/solo2_gates.py            # both gates
  python3 hw/ecp5/host/solo2_gates.py step|top
"""
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ECP5 = HERE.parent
ICE = ROOT / "hw" / "ice40"
BUILD = ICE / "build"
TOOLS = Path.home() / "tools" / "oss-cad-suite" / "bin"
CELLS = TOOLS.parent / "share" / "yosys" / "ice40" / "cells_sim.v"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ICE / "host"))
sys.path.insert(0, str(HERE))

import hw.ice40.golden as G                                    # noqa: E402
import solo                                                    # noqa: E402
import solo2                                                   # noqa: E402


def gate_step(suffix="l1", n_kf=220):
    """Historical lean fixture -> solo_tracker2 via the step shim."""
    fam = "_l"
    by = bytes(int(l, 16)
               for l in open(BUILD / f"solo_map{fam}.hex").read().split())
    n_seg = len(by) // 60
    with open(BUILD / "solo_mapw.hex", "w") as f:
        for s in range(n_seg):
            for b in by[s * 60:(s + 1) * 60]:
                for j in range(4):
                    f.write(f"{(b >> (2 * j)) & 3:04x}\n")
    ab = bytes(int(l, 16) for l in
               open(BUILD / f"solo_anchors{fam}.hex").read().split())
    with open(BUILD / "solo_ancw.hex", "w") as f:
        for i in range(0, len(ab), 2):
            f.write(f"{ab[i] | (ab[i + 1] << 8):04x}\n")
    fb = (BUILD / f"solo_feed_{suffix}.bin").read_bytes()
    eb = (BUILD / f"solo_expect_{suffix}.bin").read_bytes()
    REC = struct.calcsize("<iihHiB")
    luts = G.make_luts()
    words = []
    off = 0
    kf = 0
    exp = []
    while off < len(fb) and kf < n_kf:
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
    with open(BUILD / "solo_feedw.hex", "w") as f:
        for v in [n_seg, kf] + words:
            f.write(f"{v & 0xffff:04x}\n")
    subprocess.run(
        [str(TOOLS / "iverilog"), "-g2012", "-o",
         str(BUILD / "tbstep2.vvp"),
         "-DLEAN", "-DREFINE0", "-DV8_STEP_SHIM",
         "rtl/encoder_lean.v",
         str(ECP5 / "rtl" / "solo_tracker2.v"),
         str(ECP5 / "rtl" / "v8_shims.v"),
         "rtl/tb_solo_step.v", str(CELLS)],
        cwd=ICE, check=True)
    out = subprocess.run([str(TOOLS / "vvp"), str(BUILD / "tbstep2.vvp")],
                         cwd=ICE, capture_output=True, text=True,
                         check=True)
    bad = seen = 0
    for line in out.stdout.splitlines():
        if not line.startswith("KF "):
            continue
        p = line.split()
        n = int(p[1])
        got = tuple(int(x) for x in p[2:8])
        e = exp[n]
        want = (e[0], e[1], e[2] % 960, e[3], e[4], e[5])
        seen += 1
        if got != want:
            if bad < 8:
                print(f"MISMATCH kf {n}: rtl {got} golden {want}")
            bad += 1
    ok = bad == 0 and seen == kf
    print(f"[gate step:{suffix}] {seen - bad}/{seen} keyframes "
          f"bit-exact vs the UNTOUCHED v7 fixture "
          f"{'-> PASS' if ok else '-> FAIL'}")
    return ok


def golden_top(n_kf, reset_at=None):
    """v8 golden of the full top sequencing: Solo2Tracker + NOVELTY
    fold gate (everything else verbatim from solo_top.golden)."""
    import live as LV
    import sspslam.lattice as L

    solo.REFINE = False
    solo.K_TRY = 1
    feed = LV.Feed(seed=11, laps=2, env="classroom", traj="orbit")
    luts = G.make_luts()
    trk = solo2.Solo2Tracker(solo.FakeFab(luts), [], [])
    mapper = solo.SoloMapper(luts)
    it = iter(feed)
    st_code = {"tracking": 0, "hold": 1, "relock": 2}
    words = []

    def send(bs):
        words.extend(0x000 | b for b in bs)

    def wait(n):
        while n > 0:
            words.append(0x100 | min(n, 255))
            n -= min(n, 255)
    exp = bytearray()
    pose_q = None
    odom_prev = None
    n_seg = 0
    for kk in range(n_kf):
        item = next(it)
        if reset_at is not None and kk == reset_at:
            send(bytes([0x29]))
            wait(1)
            trk = solo2.Solo2Tracker(solo.FakeFab(luts), [], [])
            mapper = solo.SoloMapper(luts)
            n_seg = 0
        az, r_mm, w = G.scan_to_ints(item["r"])
        n = len(az)
        if odom_prev is None:
            d_q = (0, 0, 0)
            pose_q = (int(round(item["odom"][0] / solo.U)),
                      int(round(item["odom"][1] / solo.U)),
                      int(round(item["odom"][2] * solo.TAU_Q
                                / (2 * np.pi))) % solo.TAU_Q)
            send(bytes([0x25]) + struct.pack("<iiH", *pose_q))
            wait(1)
            send(bytes([0x26, 0x03]))
            wait(1)
        else:
            d = L.se2_mul(L.se2_inv(odom_prev), item["odom"])
            d_q = (int(round(d[0] / solo.U)), int(round(d[1] / solo.U)),
                   int(round(d[2] * solo.TAU_Q / (2 * np.pi))))
        odom_prev = item["odom"].copy()
        cq, sq = solo.cs_of(pose_q[2])
        dxw, dyw = solo.rot_q15(cq, sq, d_q[0], d_q[1])
        pred_q = (pose_q[0] + dxw, pose_q[1] + dyw,
                  (pose_q[2] + d_q[2]) % solo.TAU_Q)
        frozen = 0
        if n >= 5 and n_seg > 0:
            pose_q, ai, score, st = trk.step((az, r_mm, w), pred_q)
            state = st_code[st]
        else:
            pose_q = pred_q
            state = 3
        # v8: novelty-gated fold (the ONE semantic change)
        if n >= 5 and n_seg < solo2.NSEG \
                and solo2.fold_gate(trk, mapper, n_seg):
            n_before = len(mapper.frozen)
            mapper.fold((az, r_mm, w), pose_q)
            if len(mapper.frozen) > n_before:
                anc, codes = mapper.frozen[-1][0], mapper.frozen[-1][1]
                trk.add_segment(anc, codes)
                n_seg += 1
                frozen = 1
        kb = bytearray([0x30]) + struct.pack("<H", n)
        for a, r, ww in zip(az, r_mm, w):
            kb += struct.pack("<HHB", int(a), int(r), int(ww))
        kb += struct.pack("<hhh", d_q[0], d_q[1], d_q[2])
        send(kb)
        wait(11)
        exp += struct.pack("<iiHBB", pose_q[0], pose_q[1], pose_q[2],
                           state, frozen)
    dump = bytearray([0xD0, n_seg])
    for (anc, codes, _d, alive, scales) in mapper.frozen:
        by = 0
        for i, c in enumerate(codes):
            by |= (int(c) & 3) << (2 * (i % 4))
            if i % 4 == 3:
                dump.append(by)
                by = 0
        bits = 0
        for i, a in enumerate(alive):
            bits |= int(a) << i
        dump += int(bits).to_bytes(30, "little")
        for (e, m) in scales:
            dump += struct.pack("<H", (e << 8) | m)
        acq, asq = solo.cs_of(anc[2])
        dump += struct.pack("<iihhh", anc[0], anc[1], anc[2], acq, asq)
        dump += b"\x00\x00"
    send(bytes([0x28]))
    wait(len(dump))
    solo.REFINE = True
    return words, exp, bytes(dump), n_seg


def gate_top(n_kf=120, reset_at=None):
    words, exp, dump, n_seg = golden_top(n_kf, reset_at)
    with open(BUILD / "solo_topin.hex", "w") as f:
        for wv in words:
            f.write(f"{wv:04x}\n")
    print(f"[gate top] fixture: {n_kf} kf, {n_seg} segments (novelty-"
          f"gated), golden dump {len(dump)} B"
          + (f", RESET at {reset_at}" if reset_at is not None else ""),
          flush=True)
    subprocess.run(
        [str(TOOLS / "iverilog"), "-g2012", "-o",
         str(BUILD / "tbtop2.vvp"), "-DLEAN", "-DV8_TOP_SHIM",
         "rtl/uart.v", "rtl/encoder_lean.v",
         str(ECP5 / "rtl" / "solo_tracker2.v"),
         str(ECP5 / "rtl" / "top_solo2.v"),
         str(ECP5 / "rtl" / "v8_shims.v"),
         "rtl/tb_solo_top.v", str(CELLS)],
        cwd=ICE, check=True)
    out = subprocess.run([str(TOOLS / "vvp"), str(BUILD / "tbtop2.vvp")],
                         cwd=ICE, capture_output=True, text=True,
                         check=True)
    tx = bytearray()
    for line in out.stdout.splitlines():
        if line.startswith("TXB "):
            tx.append(int(line.split()[1], 16))
    off = 0
    assert tx[off] == 0x2F, f"set-pose ack: {tx[:4].hex()}"
    off += 1
    assert tx[off] == 0xA6, "mode ack"
    off += 1
    bad = 0
    REC = struct.calcsize("<iiHBB")
    for k in range(n_kf):
        if reset_at is not None and k == reset_at:
            assert tx[off] == 0xA7, f"reset-map ack: {tx[off]:02x}"
            off += 1
        hdr = tx[off]
        x, y = struct.unpack_from("<ii", tx, off + 1)
        h = struct.unpack_from("<H", tx, off + 9)[0]
        off += 11
        gx, gy, gh, gst, gfz = struct.unpack_from("<iiHBB", exp, k * REC)
        got = (x, y, h, hdr & 3, (hdr >> 3) & 1)
        want = (gx, gy, gh % 960, gst & 3, gfz)
        if got != want:
            if bad < 8:
                print(f"POSE MISMATCH kf {k}: rtl {got} hdr {hdr:02x} "
                      f"golden {want}")
            bad += 1
    dgot = bytes(tx[off:])
    dbad = 0
    if dgot != dump:
        n = min(len(dgot), len(dump))
        for i in range(n):
            if dgot[i] != dump[i]:
                if dbad < 8:
                    print(f"DUMP MISMATCH byte {i}: rtl {dgot[i]:02x} "
                          f"golden {dump[i]:02x}")
                dbad += 1
        dbad += abs(len(dgot) - len(dump))
        print(f"dump lengths: rtl {len(dgot)} golden {len(dump)}")
    ok = bad == 0 and dbad == 0
    print(f"[gate top] {n_kf - bad}/{n_kf} pose frames bit-exact, dump "
          f"{'bit-exact' if dbad == 0 else f'{dbad} byte diffs'} "
          f"{'-> PASS' if ok else '-> FAIL'}")
    return ok


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    ok = True
    if which in ("all", "step"):
        ok &= gate_step("l1", 220)
        ok &= gate_step("lr", 160)
    if which in ("all", "top"):
        ok &= gate_top(120)
        ok &= gate_top(90, reset_at=45)
    print("[solo2_gates] " + ("ALL PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)
