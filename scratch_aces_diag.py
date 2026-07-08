"""ACES do-no-harm diagnosis: on the one log where odometry (5.41 m) already
beats our shipped result (6.21 m), do loop closures help or hurt, and which
mechanism admits the harmful edges?

Runs the identical BoundedSLAM core / keyframing / eval as
ssp_bounded_carmen.py, but sweeps a few backend configs and reports the
mechanism fire-counts so we can see WHERE the regression enters.
"""
import sys
import time

import numpy as np

import ssp_slam as S
import ssp_slam_carmen as C
import ssp_slam_loop as L
import ssp_bounded as B

VALID_MAX = 40.0
PATH = sys.argv[1] if len(sys.argv) > 1 else "data/aces.log"


def build(**kw):
    slam = B.BoundedSLAM(robust=True, attempt_every=4, relax_every=25,
                         gap_kf=300, recent_aids=12)
    slam.store_dtype = np.complex64
    for k, v in kw.items():
        setattr(slam, k, v)
    return slam


def evaluate(fin, kts):
    ref = C.parse_flaser(PATH.replace(".log", ".gfs.log"))
    rts = np.array([t for _, _, t in ref])
    rxy = np.stack([p[:2] for _, p, _ in ref])
    j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
    good = np.abs(rts - kts[j]) < 0.3
    al = C.align_se2(fin[j[good], :2], rxy[good])
    e = np.linalg.norm(al - rxy[good], axis=1)
    return np.sqrt((e ** 2).mean()), np.median(e), e.max()


def run_cfg(tag, keys, odom, kts, **kw):
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    slam = build(**kw)
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
    rmse, med, mx = evaluate(fin, kts)
    nloop = sum(1 for e in slam.edges if e[5] == "loop")
    dt = time.time() - t0
    print(f"  {tag:<22} ATE {rmse:5.3f} m (med {med:.3f}, max {mx:5.2f})  "
          f"loops {nloop:3d}  veto {slam.n_veto:3d}  innov {slam.n_innov_rej:3d}  "
          f"inflate {slam.n_inflate:3d}  pruned {slam.n_pruned:2d}  "
          f"relax {slam.n_relax}  {dt:.0f}s", flush=True)
    return rmse


def main():
    scans = C.parse_flaser(PATH)
    keys = C.keyframes(scans)
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    # odometry-only ATE (raw guess chain, no SLAM)
    ref = C.parse_flaser(PATH.replace(".log", ".gfs.log"))
    rts = np.array([t for _, _, t in ref])
    rxy = np.stack([p[:2] for _, p, _ in ref])
    j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
    good = np.abs(rts - kts[j]) < 0.3
    al = C.align_se2(odom[j[good], :2], rxy[good])
    e = np.linalg.norm(al - rxy[good], axis=1)
    print(f"== {PATH}: {len(scans)} scans -> {len(keys)} keyframes")
    print(f"  {'raw odometry':<22} ATE {np.sqrt((e**2).mean()):5.3f} m "
          f"(med {np.median(e):.3f}, max {e.max():5.2f})")
    run_cfg("frontend-only", keys, odom, kts, attempt_every=10 ** 9)
    run_cfg("shipped soft@0.55", keys, odom, kts)
    run_cfg("hard veto", keys, odom, kts, coh_soft=False)
    run_cfg("no-loop-frontend(der)", keys, odom, kts, attempt_every=10 ** 9)


if __name__ == "__main__":
    main()
