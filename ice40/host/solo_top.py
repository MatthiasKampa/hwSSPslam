#!/usr/bin/env python3
"""solo_top.py — v7 TOP-LEVEL sim gate: the full top_solo UART protocol on
the classroom SELF-MAPPING sequence (the standalone demo in sim).

Golden mirror of the top's sequencing (REFINE=0, K_TRY=1, mapping on):
per kf: pred = pose (+) R(pose_h) d_q; step iff n>=5 and n_seg>0 else
pose=pred/state=odo; fold at the committed pose iff mapping and n>=5 and
n_seg<NSEG; freeze on the SEG_KF'th fold (mcode + liveness theta=1/2 +
ring scales + anchor with table-cs_of). Emits:

  build/solo_topin.hex    RX WORD stream (host pacing encoded): 0x00XX =
                          send byte XX; 0x01NN = wait for NN more TX bytes
                          (the pose frame / ack / dump is the flow
                          control, mirroring the host contract). Commands:
                          0x25 set-pose, 0x26 mode=3, per kf 0x30 n +
                          points + d_q, final 0x28 dump.
  build/solo_topexp.bin   per-kf golden (x i32, y i32, h u16, state u8,
                          frozen u8)  [12 B records]
  build/solo_topdump.bin  golden dump reply (0xD0, n_seg, per seg 60 B
                          codes + 30 B liveness + 8 B scales + 16 B anchor)

Then compiles tb_solo_top.v (encoder_lean + solo_tracker + top_solo) and
diffs every pose frame and every dump byte. Usage:
  solo_top.py [n_kf] [reset_at]
(default 60 kf; 220 = the full classroom tour. reset_at inserts the
0x29 RESET-MAP command before that keyframe — the env-switch gate: the
golden restarts with a fresh tracker+mapper, n_seg=0, the open segment
discarded; the dump must contain only post-reset segments.)
"""
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import solo                                                # noqa: E402
import ssp_ice40 as G                                      # noqa: E402

ICE = Path(__file__).resolve().parents[1]
BUILD = ICE / "build"
TOOLS = Path.home() / "tools" / "oss-cad-suite" / "bin"
CELLS = TOOLS.parent / "share" / "yosys" / "ice40" / "cells_sim.v"
N_KF = int(sys.argv[1]) if len(sys.argv) > 1 else 60
RESET_AT = int(sys.argv[2]) if len(sys.argv) > 2 else None
NSEG = 64


def golden(n_kf, reset_at=None):
    import live as LV
    import ssp_slam_loop as L

    solo.REFINE = False
    solo.K_TRY = 1
    feed = LV.Feed(seed=11, laps=2, env="classroom", traj="orbit")
    luts = G.make_luts()
    trk = solo.SoloTracker(solo.FakeFab(luts), [], [])
    mapper = solo.SoloMapper(luts)
    it = iter(feed)
    st_code = {"tracking": 0, "hold": 1, "relock": 2}
    words = []                                 # 0x00XX send / 0x01NN wait

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
            # env switch: RESET MAP. Chip: n_seg=0, open segment
            # discarded, pose kept. Golden: fresh tracker + mapper.
            send(bytes([0x29]))
            wait(1)
            trk = solo.SoloTracker(solo.FakeFab(luts), [], [])
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
            # host: set-pose before the first keyframe
            send(bytes([0x25]) + struct.pack("<iiH", *pose_q))
            wait(1)
            send(bytes([0x26, 0x03]))          # mapping + stream
            wait(1)
        else:
            d = L.se2_mul(L.se2_inv(odom_prev), item["odom"])
            d_q = (int(round(d[0] / solo.U)), int(round(d[1] / solo.U)),
                   int(round(d[2] * solo.TAU_Q / (2 * np.pi))))
        odom_prev = item["odom"].copy()
        # pred compose == top K_PRED (svc ROT + wrap960)
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
        if n >= 5 and n_seg < NSEG:
            n_before = len(mapper.frozen)
            mapper.fold((az, r_mm, w), pose_q)
            if len(mapper.frozen) > n_before:
                anc, codes = mapper.frozen[-1][0], mapper.frozen[-1][1]
                trk.add_segment(anc, codes)
                n_seg += 1
                frozen = 1
        # keyframe bytes (pose frame = flow control)
        kb = bytearray([0x30]) + struct.pack("<H", n)
        for a, r, ww in zip(az, r_mm, w):
            kb += struct.pack("<HHB", int(a), int(r), int(ww))
        kb += struct.pack("<hhh", d_q[0], d_q[1], d_q[2])
        send(kb)
        wait(11)
        exp += struct.pack("<iiHBB", pose_q[0], pose_q[1], pose_q[2],
                           state, frozen)
    # golden dump image
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
        dump += b"\x00\x00"                    # pad word
    send(bytes([0x28]))                        # map dump command
    wait(len(dump))
    solo.REFINE = True
    return words, exp, bytes(dump), n_seg


def main():
    words, exp, dump, n_seg = golden(N_KF, RESET_AT)
    with open(BUILD / "solo_topin.hex", "w") as f:
        for wv in words:
            f.write(f"{wv:04x}\n")
    (BUILD / "solo_topexp.bin").write_bytes(exp)
    (BUILD / "solo_topdump.bin").write_bytes(dump)
    print(f"top fixture: {N_KF} kf, {len(words)} stream words, {n_seg} "
          f"segments, golden dump {len(dump)} B"
          + (f", RESET at kf {RESET_AT}" if RESET_AT is not None else ""),
          flush=True)
    subprocess.run([str(TOOLS / "iverilog"), "-g2012", "-o",
                    str(BUILD / "tbtop.vvp"), "-DLEAN",
                    "rtl/uart.v", "rtl/encoder_lean.v",
                    "rtl/solo_tracker.v", "rtl/top_solo.v",
                    "rtl/tb_solo_top.v", str(CELLS)], cwd=ICE, check=True)
    out = subprocess.run([str(TOOLS / "vvp"), str(BUILD / "tbtop.vvp")],
                         cwd=ICE, capture_output=True, text=True,
                         check=True)
    tx = bytearray()
    for line in out.stdout.splitlines():
        if line.startswith("TXB "):
            tx.append(int(line.split()[1], 16))
    # parse the TX stream: 0x2F (pose ack), 0xA6 (mode ack), pose frames,
    # then the dump
    off = 0
    assert tx[off] == 0x2F, f"set-pose ack: {tx[:4].hex()}"
    off += 1
    assert tx[off] == 0xA6, "mode ack"
    off += 1
    bad = 0
    REC = struct.calcsize("<iiHBB")
    for k in range(N_KF):
        if RESET_AT is not None and k == RESET_AT:
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
                    print(f"DUMP MISMATCH byte {i}: rtl "
                          f"{dgot[i]:02x} golden {dump[i]:02x}")
                dbad += 1
        dbad += abs(len(dgot) - len(dump))
        print(f"dump lengths: rtl {len(dgot)} golden {len(dump)}")
    print(f"solo top gate: {N_KF - bad}/{N_KF} pose frames bit-exact, "
          f"dump {'bit-exact' if dbad == 0 else f'{dbad} byte diffs'} "
          f"{'-> PASS' if bad == 0 and dbad == 0 else '-> FAIL'}")
    # write the dump in decode.py-consumable form
    if dbad == 0 and bad == 0:
        write_decode_files(dgot)
    return 0 if bad == 0 and dbad == 0 else 1


def write_decode_files(dump):
    """Split the chip dump into decode.py inputs (map/anchors/liveness/
    scales hex) — the laptop side of the demo."""
    n_seg = dump[1]
    off = 2
    mh = open(BUILD / "solo_map_top.hex", "w")
    lh = open(BUILD / "solo_live_top.hex", "w")
    sh = open(BUILD / "solo_scl_top.hex", "w")
    ah = open(BUILD / "solo_anchors_top.hex", "w")
    for s in range(n_seg):
        for b in dump[off:off + 60]:
            mh.write(f"{b:02x}\n")
        off += 60
        for b in dump[off:off + 30]:
            lh.write(f"{b:02x}\n")
        off += 30
        for b in dump[off:off + 8]:
            sh.write(f"{b:02x}\n")
        off += 8
        for b in dump[off:off + 16]:
            ah.write(f"{b:02x}\n")
        off += 16
    for f in (mh, lh, sh, ah):
        f.close()
    print(f"decode files written: solo_map_top/live_top/scl_top/"
          f"anchors_top.hex ({n_seg} segments)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
