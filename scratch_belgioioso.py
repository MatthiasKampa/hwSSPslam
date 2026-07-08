"""Belgioioso Castle held-out run + odometry/frontend/loops breakdown.

Belgioioso's gfs timestamps are corrupt (low-precision), so ATE is evaluated
by the MIT range-array-identity convention (reused from ssp_scancontext) rather
than timestamp matching. Non-Manhattan structure (irregular stone walls,
courtyards) probes the encoder's one documented fragility.
"""
import sys
import time

import numpy as np

import ssp_slam as S
import ssp_slam_carmen as C
import ssp_slam_loop as L
import ssp_bounded as B

VALID_MAX = 40.0
PATH = "data/belgioioso.log"


def ref_xy_by_identity(keys):
    """Per-keyframe corrected position via range-array identity (belgioioso gfs
    timestamps are corrupt, same as MIT)."""
    kts = np.array([t for _, _, t in keys])
    raw = C.parse_flaser(PATH)
    gfs = C.parse_flaser(PATH.replace(".log", ".gfs.log"))
    gxy = np.stack([p[:2] for _, p, _ in gfs])
    idx = {}
    for i, (r, _, _) in enumerate(raw):
        idx.setdefault(r.tobytes(), []).append(i)
    gts, keep = [], []
    for m, (r, _, _) in enumerate(gfs):
        b = r.tobytes()
        if b in idx:
            gts.append(raw[idx[b][0]][2])
            keep.append(m)
    gts = np.array(gts)
    gxy = gxy[keep]
    print(f"  range-identity matched {len(gts)}/{len(gfs)} gfs poses "
          f"({100*len(gts)/len(gfs):.0f}%)")
    o = np.argsort(gts)
    gts, gxy = gts[o], gxy[o]
    uniq = np.concatenate([[True], np.diff(gts) > 0])
    gts, gxy = gts[uniq], gxy[uniq]
    rx = np.interp(kts, gts, gxy[:, 0])
    ry = np.interp(kts, gts, gxy[:, 1])
    valid = (kts >= gts[0]) & (kts <= gts[-1])
    return np.stack([rx, ry], 1), valid


def ate(fin_xy, ref, valid):
    al = C.align_se2(fin_xy[valid], ref[valid])
    e = np.linalg.norm(al - ref[valid], axis=1)
    return np.sqrt((e ** 2).mean()), np.median(e), e.max()


def run_cfg(tag, keys, odom, ref, valid, **kw):
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    slam = B.BoundedSLAM(robust=True, attempt_every=4, relax_every=25,
                         gap_kf=300, recent_aids=12)
    slam.store_dtype = np.complex64
    for k, v in kw.items():
        setattr(slam, k, v)
    est = np.zeros((n, 3))
    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < VALID_MAX, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = opose if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
    if slam.dirty:
        slam.relax()
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    rmse, med, mx = ate(fin[:, :2], ref, valid)
    nloop = sum(1 for e in slam.edges if e[5] == "loop")
    print(f"  {tag:<22} ATE {rmse:5.3f} m (med {med:.3f}, max {mx:5.2f})  "
          f"loops {nloop:3d}  veto {slam.n_veto:3d}  innov {slam.n_innov_rej:3d}  "
          f"inflate {slam.n_inflate:3d}  mem {slam.memory_kb():.0f} KB  "
          f"{time.time()-t0:.0f}s", flush=True)


def main():
    scans = C.parse_flaser(PATH)
    keys = C.keyframes(scans)
    odom = np.stack([k[1] for k in keys])
    ref, valid = ref_xy_by_identity(keys)
    r, m, x = ate(odom[:, :2], ref, valid)
    print(f"== {PATH}: {len(scans)} scans -> {len(keys)} keyframes  "
          f"({len(keys[0][0])} beams)")
    print(f"  {'raw odometry':<22} ATE {r:5.3f} m (med {m:.3f}, max {x:5.2f})")
    run_cfg("frontend-only", keys, odom, ref, valid, attempt_every=10 ** 9)
    run_cfg("shipped soft@0.55", keys, odom, ref, valid)


if __name__ == "__main__":
    main()
