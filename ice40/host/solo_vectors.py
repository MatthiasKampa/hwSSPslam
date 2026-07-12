#!/usr/bin/env python3
"""solo_vectors.py — SIM-GATE VECTORS for the v7 top_solo RTL.

Generates, from the golden solo.py on a deterministic classroom feed:
  build/solo_map.hex      resident map: per-segment 2b codes (seg-major,
                          4 codes/byte, LSB-first — SPRAM image)
  build/solo_anchors.hex  anchor table: per segment x_u i32, y_u i32,
                          ah_q i16, cq i16, sq i16 (LE, 16 B/entry padded)
  build/solo_feed.bin     per-kf records: n_pts u16, then (az u16, r_mm
                          u16, w u8)*n, then pred (x_u i32, y_u i32,
                          h_q i16)
  build/solo_expect.bin   per-kf expected: pose (x_u i32, y_u i32,
                          h_q i16), ai u16, score i32, state u8
                          (0=tracking 1=hold 2=relock)
  Written for K_TRY in {1, 3} (suffix _k1/_k3).

The RTL testbench replays solo_feed and must reproduce solo_expect
BIT-EXACTLY (the same acceptance shape as v3-v6: golden -> sim -> hw).
Map source: pass-1 python SLAM on Feed(classroom/orbit) exactly as
solo.bench (the validated parity protocol).
"""
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solo                                                # noqa: E402
import ssp_ice40 as G                                      # noqa: E402

BUILD = Path(__file__).resolve().parents[1] / "build"


def build_map(env="classroom", traj="orbit", seed=11):
    import live as LV
    import ssp_fpga as F
    import ssp_dynenv as DE
    import ssp_slam as S
    import ssp_slam_loop as L

    feed = LV.Feed(seed=seed, laps=2, env=env, traj=traj)
    slam = F.BandSLAM(robust=True, attempt_every=4, relax_every=25,
                      gap_kf=300, recent_aids=12, spec=None, nph=0)
    slam.store_dtype = np.complex64
    it = iter(feed)
    items = [next(it) for _ in range(feed.n_loop)]
    est_prev = odom_prev = None
    for item in items:
        rr = np.where((item["r"] > 0.05) & (item["r"] < S.MAX_RANGE - 0.05),
                      item["r"], np.inf)
        pts, w = DE._bridge2(rr, feed.beam)
        if est_prev is None:
            guess = item["odom"].copy()
        else:
            guess = L.se2_mul(est_prev, L.se2_mul(
                L.se2_inv(odom_prev), item["odom"]))
        est = slam.add_keyframe(pts, w, guess)
        est_prev, odom_prev = est.copy(), item["odom"].copy()
    fmap = LV.FrozenMap(slam)
    anchors_q = [(int(round(fmap.anchor[a][0] / solo.U)),
                  int(round(fmap.anchor[a][1] / solo.U)),
                  int(round(fmap.anchor[a][2] * solo.TAU_Q / (2 * np.pi)))
                  % solo.TAU_Q)
                 for a in fmap.aids]
    codes = [np.asarray(fmap.codes[a], np.int64) for a in fmap.aids]
    return feed, it, anchors_q, codes


def write_map_hex(codes, path):
    with open(path, "w") as f:
        for c in codes:
            by = bytearray()
            for i in range(0, len(c), 4):
                b = 0
                for j in range(4):
                    if i + j < len(c):
                        b |= (int(c[i + j]) & 3) << (2 * j)
                by.append(b)
            f.write("".join(f"{x:02x}\n" for x in by))


def write_anchor_hex(anchors_q, path):
    with open(path, "w") as f:
        for (ax, ay, ah) in anchors_q:
            cq, sq = solo.cs_of(ah)
            rec = struct.pack("<iihhh", ax, ay, ah, cq, sq)
            rec += b"\x00" * (16 - len(rec))
            f.write("".join(f"{x:02x}\n" for x in rec))


def gen(k_try, feed, it, anchors_q, codes, n_kf=220, seed=11, kick_kf=()):
    """kick_kf: keyframes at which a PERSISTENT +320,+320 u pose kick is
    injected (kidnap: the odometry chain teleports, the scans do not) —
    drives holds -> every-12th wide re-search -> relock. The v7-lean
    kidnap fixture (lr) kicks at kf 60 and 110."""
    import ssp_slam_loop as L
    solo.K_TRY = k_try
    luts = G.make_luts()
    trk = solo.SoloTracker(solo.FakeFab(luts), anchors_q, codes)
    st_code = {"tracking": 0, "hold": 1, "relock": 2}
    feed_b, exp_b = bytearray(), bytearray()
    pose_q = None
    odom_prev = None
    n_written = 0
    for kk in range(n_kf):
        item = next(it)
        if kk in kick_kf and pose_q is not None:
            pose_q = (pose_q[0] + 320, pose_q[1] + 320, pose_q[2])
        q = G.scan_to_ints(item["r"])
        if odom_prev is None:
            d_q = (0, 0, 0)
            pose_q = (int(round(item["odom"][0] / solo.U)),
                      int(round(item["odom"][1] / solo.U)),
                      int(round(item["odom"][2] * solo.TAU_Q / (2 * np.pi)))
                      % solo.TAU_Q)
        else:
            d = L.se2_mul(L.se2_inv(odom_prev), item["odom"])
            d_q = (int(round(d[0] / solo.U)), int(round(d[1] / solo.U)),
                   int(round(d[2] * solo.TAU_Q / (2 * np.pi))))
        odom_prev = item["odom"].copy()
        cq, sq = solo.cs_of(pose_q[2])
        dxw, dyw = solo.rot_q15(cq, sq, d_q[0], d_q[1])
        pred_q = (pose_q[0] + dxw, pose_q[1] + dyw,
                  (pose_q[2] + d_q[2]) % solo.TAU_Q)
        if len(q[0]) < 5:
            pose_q = pred_q
            continue
        pose_q, ai, score, st = trk.step(q, pred_q)
        az, r_mm, w = q
        feed_b += struct.pack("<H", len(az))
        for a, r, ww in zip(az, r_mm, w):
            feed_b += struct.pack("<HHB", int(a), int(r), int(ww))
        feed_b += struct.pack("<iih", pred_q[0], pred_q[1], pred_q[2])
        exp_b += struct.pack("<iihHiB", pose_q[0], pose_q[1], pose_q[2],
                             int(ai), int(score), st_code[st])
        n_written += 1
    return feed_b, exp_b, n_written


def main():
    BUILD.mkdir(exist_ok=True)
    feed, it, anchors_q, codes = build_map()
    write_map_hex(codes, BUILD / "solo_map.hex")
    write_anchor_hex(anchors_q, BUILD / "solo_anchors.hex")
    print(f"map: {len(codes)} segments -> solo_map.hex "
          f"({len(codes) * len(codes[0]) // 4} B), solo_anchors.hex", flush=True)
    for k in (1, 3):
        fb, eb, n = gen(k, feed, it, anchors_q, codes)
        (BUILD / f"solo_feed_k{k}.bin").write_bytes(fb)
        (BUILD / f"solo_expect_k{k}.bin").write_bytes(eb)
        print(f"K_TRY={k}: {n} kf -> solo_feed_k{k}.bin "
              f"({len(fb)} B), solo_expect_k{k}.bin ({len(eb)} B)", flush=True)
    solo.K_TRY = 1


def main_lean():
    """v7-lean fixture family (build session): REFINE=0 (divider dropped;
    classroom A/B banked), table-defined cs_of. Own map/anchor hex names
    so the historical k1/rl gates stay byte-reproducible."""
    BUILD.mkdir(exist_ok=True)
    solo.REFINE = False
    feed, it, anchors_q, codes = build_map()
    write_map_hex(codes, BUILD / "solo_map_l.hex")
    write_anchor_hex(anchors_q, BUILD / "solo_anchors_l.hex")
    print(f"lean map: {len(codes)} segments -> solo_map_l.hex, "
          f"solo_anchors_l.hex", flush=True)
    fb, eb, n = gen(1, feed, it, anchors_q, codes, n_kf=220)
    (BUILD / "solo_feed_l1.bin").write_bytes(fb)
    (BUILD / "solo_expect_l1.bin").write_bytes(eb)
    print(f"l1 (classroom, REFINE=0): {n} kf", flush=True)
    fb, eb, n = gen(1, feed, it, anchors_q, codes, n_kf=160,
                    kick_kf=(60, 110))
    (BUILD / "solo_feed_lr.bin").write_bytes(fb)
    (BUILD / "solo_expect_lr.bin").write_bytes(eb)
    import struct as _s
    states = [(_s.unpack_from("<iihHiB", eb, i * 17)[5]) for i in range(n)]
    print(f"lr (kidnap kicks kf 60/110, REFINE=0): {n} kf | states "
          f"tracking {states.count(0)} hold {states.count(1)} relock "
          f"{states.count(2)}", flush=True)
    solo.REFINE = True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "lean":
        main_lean()
    else:
        main()
