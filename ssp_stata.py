"""MIT Stata Center adapter — the suite's first independent-class reference.

Every ATE in this repo is scored against GMapping's own output (FINDINGS §9
caveat 2). The Stata Center dataset provides per-scan ground truth anchored
to as-built FLOORPLANS (batches of 160 scans aligned to the floorplan at both
ends, iSAM-relaxed in between; ~2-3 cm claimed) — a reference class
independent of any particle-filter SLAM family. This adapter runs the
shipped/E2/lean pipelines on a bag's base laser + wheel odometry and scores
against the `.gt.laser.poses` derivative files.

Data (fetched via gdown from the project's Drive folder — see
data/README.md): data/stata/2012-01-27-07-37-01.bag (7.1 GB, 6:48 min,
PR2, Hokuyo UTM-30LX base laser: 1040 beams x 0.25 deg = 260 deg FOV) +
2012-01-27_part{1,3}_floor2.gt.poses (dense 20 Hz GT for two segments).

The 260-deg FOV needs no shipped edits: the encoder/matcher are FOV-agnostic
(they take points); the 180-deg assumption lives only in the CARMEN harness's
beam construction, and this loader supplies the true beam array from the
LaserScan message. The base->base_laser static offset is read from /tf and
composed onto the odometry so the estimated trajectory lives in the LASER
frame the GT is expressed in.

Usage: python3 ssp_stata.py [bag] [configs: shipped e2 lean]
"""
import sys
from pathlib import Path

import numpy as np

import ssp_slam as S
import ssp_slam_loop as L
import ssp_fpga as F

BAG = "data/stata/2012-01-27-07-37-01.bag"
GT_FILES = ("data/stata/2012-01-27_part1_floor2.gt.poses",
            "data/stata/2012-01-27_part3_floor2.gt.poses")
KEY_TRANS, KEY_ROT = 0.10, np.deg2rad(5.0)
RANGE_MIN = 0.05


def _yaw(q):
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def load_bag(path):
    """-> (scans [(ts_us, ranges)], beam angles, odom poses at scan times in
    the LASER frame, range_max)."""
    from rosbags.highlevel import AnyReader
    scans, odom_t, odom_p = [], [], []
    T_bl = None
    beam = None
    range_max = 30.0
    with AnyReader([Path(path)]) as reader:
        conns = {c.topic: c for c in reader.connections}
        want = [conns[t] for t in ("/base_scan", "/base_odometry/odom", "/tf")
                if t in conns]
        for con, ts, raw in reader.messages(connections=want):
            if con.topic == "/base_scan":
                msg = reader.deserialize(raw, con.msgtype)
                if beam is None:
                    n = len(msg.ranges)
                    beam = msg.angle_min + np.arange(n) * msg.angle_increment
                    range_max = float(msg.range_max)
                    print(f"  base_scan: {n} beams, "
                          f"FOV {np.degrees(beam[-1] - beam[0]):.0f} deg, "
                          f"range_max {range_max:.1f}")
                st = msg.header.stamp
                scans.append((st.sec * 10**6 + st.nanosec // 1000,
                              np.asarray(msg.ranges, np.float32)))
            elif con.topic == "/base_odometry/odom":
                msg = reader.deserialize(raw, con.msgtype)
                st = msg.header.stamp
                p = msg.pose.pose
                odom_t.append(st.sec * 10**6 + st.nanosec // 1000)
                odom_p.append((p.position.x, p.position.y, _yaw(p.orientation)))
            elif con.topic == "/tf" and T_bl is None:
                msg = reader.deserialize(raw, con.msgtype)
                for tr in msg.transforms:
                    if "base_laser" in tr.child_frame_id:
                        t = tr.transform.translation
                        T_bl = np.array([t.x, t.y, _yaw(tr.transform.rotation)])
                        print(f"  base->laser offset: {T_bl.round(3)}")
                        break
    if T_bl is None:
        T_bl = np.array([0.275, 0.0, 0.0])   # PR2 nominal
        print("  base->laser offset not in /tf; using PR2 nominal 0.275 m")
    odom_t = np.asarray(odom_t)
    odom_p = np.asarray(odom_p)
    o = np.argsort(odom_t)
    odom_t, odom_p = odom_t[o], odom_p[o]
    # laser-frame odometry pose at each scan time (nearest odom sample)
    sts = np.array([s[0] for s in scans])
    j = np.clip(np.searchsorted(odom_t, sts), 0, len(odom_t) - 1)
    poses = np.array([L.se2_mul(odom_p[i], T_bl) for i in j])
    return scans, beam, poses, range_max


def keyframes(scans, poses):
    keys, last = [], None
    for (ts, r), p in zip(scans, poses):
        if last is None or np.linalg.norm(p[:2] - last[:2]) > KEY_TRANS \
                or abs(S.wrap(p[2] - last[2])) > KEY_ROT:
            keys.append((r, p, ts))
            last = p
    return keys


def load_gt():
    rows = []
    for f in GT_FILES:
        a = np.loadtxt(f, delimiter=",")
        rows.append(a)
    gt = np.concatenate(rows)
    return gt[:, 0].astype(np.int64), gt[:, 1:3]


def run(cls, keys, beam, range_max, sample="seg", **kw):
    import time as _t
    n = len(keys)
    odom = np.stack([k[1] for k in keys])
    slam = cls(robust=True, attempt_every=4, relax_every=25, gap_kf=300,
               recent_aids=12, **kw)
    slam.store_dtype = np.complex64
    est = np.zeros((n, 3))
    t0 = _t.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where((r > RANGE_MIN) & (r < range_max - 0.1), r, np.inf
                      ).astype(float)
        if sample == "point":
            pts, w = F.points_from_scan(rr, beam)
        else:
            pts, w, _ = S.scan_to_samples(rr, beam)
        guess = odom[0] if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
    if slam.dirty:
        slam.relax()
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    nloop = sum(1 for e in slam.edges if e[5] == "loop")
    return fin, nloop, (_t.time() - t0) / n * 1e3


def evaluate(fin, keys, gts, gxy, tag):
    import ssp_slam_carmen as C
    kts = np.array([t for _, _, t in keys])
    j = np.abs(gts[:, None] - kts[None, :]).argmin(1)
    good = np.abs(gts - kts[j]) < 30_000        # 30 ms
    al = C.align_se2(fin[j[good], :2], gxy[good])
    e = np.linalg.norm(al - gxy[good], axis=1)
    print(f"  {tag:<22} ATE {np.sqrt((e ** 2).mean()):6.3f} m  "
          f"med {np.median(e):6.3f}  max {e.max():5.2f}  "
          f"({int(good.sum())} GT poses)", flush=True)


def main():
    bag = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".bag") \
        else BAG
    print(f"loading {bag} ...", flush=True)
    cache = Path(bag + ".npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        scans = list(zip(z["sts"].tolist(), z["ranges"]))
        beam, poses, range_max = z["beam"], z["poses"], float(z["range_max"])
        print(f"  (from cache {cache})", flush=True)
    else:
        scans, beam, poses, range_max = load_bag(bag)
        np.savez_compressed(cache,
                            sts=np.array([s[0] for s in scans], np.int64),
                            ranges=np.stack([s[1] for s in scans]),
                            beam=beam, poses=poses, range_max=range_max)
    keys = keyframes(scans, poses)
    print(f"  {len(scans)} scans -> {len(keys)} keyframes", flush=True)
    gts, gxy = load_gt()
    print(f"  GT: {len(gts)} floorplan-anchored poses "
          f"(independent-class reference)", flush=True)
    odom = np.stack([k[1] for k in keys])
    evaluate(odom, keys, gts, gxy, "raw odometry")
    cfgs = sys.argv[2:] or ["shipped", "e2", "lean"]
    if "shipped" in cfgs:
        fin, nl, ms = run(F.BandSLAM, keys, beam, range_max,
                          spec=None, nph=0)
        evaluate(fin, keys, gts, gxy, f"shipped ({nl} loops, {ms:.0f}ms/kf)")
    if "e2" in cfgs:
        fin, nl, ms = run(F.BandSLAM, keys, beam, range_max,
                          sample="point", spec=None, nph=0)
        evaluate(fin, keys, gts, gxy, f"E2 point ({nl} loops)")
    if "lean" in cfgs:
        fin, nl, ms = run(F.BandSLAM, keys, beam, range_max, sample="point",
                          spec=F.IntSpec(8, 7, 7), nph=4, nmag=1,
                          ring_scales=True)
        evaluate(fin, keys, gts, gxy, f"FPGA-lean ({nl} loops)")


if __name__ == "__main__":
    main()
