"""Synthetic 360-deg bench — the SPOT-proxy environment (exact GT).

Joined the acceptance suite 2026-07-10 (user call, with the intel removal):
the deployment target is a SPOT carrying a 360-deg lidar, so this bench
simulates exactly that sensor class over the validated `sspslam/worlds.py`
geometries — full-circle beams (default 360 x 1 deg), range noise / angle
jitter / dropout from the original encoder-study sim, odometry as composed
noisy GT deltas, EXACT ground truth for scoring.

It emits bundles with the `ssp_datasets` interface, so the ONE shared
runner works unchanged on synthetic data:

    import runners.synth as ssp_synth, runners.datasets as DS
    r = DS.run(ssp_synth.make("mixed", seed=11), spec=None, nph=0)

Determinism: every noise draw comes from a LOCAL np.random.default_rng(seed)
(the module never touches ssp_slam.RNG), so bundles are reproducible and
seed-parameterized per PROTOCOL 5.

Usage:
  python3 -m runners.synth bench            # shipped vs E2 on 3 worlds x 2 seeds
  python3 -m runners.synth fov              # 180 vs 260 vs 360 FOV, same worlds
"""
import sys

import numpy as np

import sspslam.encoder as S
import sspslam.lattice as L
from sspslam.worlds import WORLDS

STEP = 0.10                  # ~keyframe stride along the trajectory [m]
ODO_T, ODO_R = 0.010, np.deg2rad(0.25)   # per-frame odometry noise (bench_loop)


def _traj(laps, rng, n):
    """Multiloop orbit with continuously varying scale (bench_loop recipe)."""
    u = np.linspace(0, laps * 2 * np.pi, n, endpoint=False)
    s = 1.0 + 0.12 * np.sin(0.9 * u)
    a, b = 4.8 * s, 1.9 * s
    x, y = 8 + a * np.cos(u), 5 + b * np.sin(u)
    dx, dy = np.gradient(x), np.gradient(y)
    th = np.arctan2(dy, dx) + rng.normal(0, np.deg2rad(0.3), n)
    return np.stack([x, y, th], 1)


def make(world="mixed", laps=3, seed=11, n_beams=360, fov_deg=360.0):
    """-> ssp_datasets-shaped bundle with exact GT (eval='exact')."""
    rng = np.random.default_rng(seed)
    segs = WORLDS[world]()
    # frame count from path length so the stride matches shipped keyframing
    peri = 2 * np.pi * (4.8 + 1.9) / 2 * laps
    n = int(peri / STEP)
    gt = _traj(laps, rng, n)
    # full-circle (or fov_deg) beam grid; endpoint=False keeps 360 deg exact
    fov = np.deg2rad(fov_deg)
    if fov_deg >= 360.0:
        beam = -np.pi + np.arange(n_beams) * (2 * np.pi / n_beams)
    else:
        beam = -fov / 2 + np.arange(n_beams) * (fov / n_beams)
    odom = gt.copy()
    keys = []
    for k in range(n):
        ang = gt[k, 2] + beam + rng.normal(0, S.ANGLE_JITTER, n_beams)
        r = S.raycast(segs, gt[k, :2], ang)
        r = np.where(r <= S.MAX_RANGE,
                     r + rng.normal(0, S.RANGE_NOISE, n_beams), np.inf)
        r[rng.random(n_beams) < S.DROPOUT] = np.inf
        if k:
            d = L.se2_mul(L.se2_inv(gt[k - 1]), gt[k])
            d[:2] += (rng.normal(0, ODO_T, 2)
                      + 0.02 * np.abs(d[:2]) * rng.normal(0, 1, 2))
            d[2] += rng.normal(0, ODO_R)
            odom[k] = L.se2_mul(odom[k - 1], d)
        keys.append((r.astype(np.float32), odom[k].copy(), 0.1 * k))
    return dict(name=f"synth-{world}-s{seed}-fov{int(fov_deg)}",
                kind="synth", eval="exact", path=None, keys=keys, beam=beam,
                odom=np.stack([k[1] for k in keys]),
                kts=np.array([t for _, _, t in keys]),
                rmin=0.05, rmax=S.MAX_RANGE - 0.05, gt=gt)


def bench(worlds=("mixed", "office", "corridor"), seeds=(11, 12)):
    import runners.datasets as DS
    import sspslam.quantized as F
    import sspslam.frontend as C
    for world in worlds:
        for seed in seeds:
            b = make(world, seed=seed)
            odo = b["odom"]
            e = np.linalg.norm(
                C.align_se2(
                    odo[:, :2], b["gt"][:, :2]) - b["gt"][:, :2], axis=1)
            print(f"== {b['name']} ({len(b['keys'])} kf): raw odom ATE "
                  f"{np.sqrt((e ** 2).mean()):.3f}", flush=True)
            for tag, sm in (("shipped seg", "seg"), ("E2 point  ", "point")):
                r = DS.run(dict(b), F.BandSLAM, sample=sm, spec=None, nph=0)
                print(f"  {tag}  ATE {r['ate']:6.3f}  med {r['med']:6.3f}  "
                      f"loops {r['loops']}", flush=True)


def fov_study(world="mixed", seed=11):
    import runners.datasets as DS
    import sspslam.quantized as F
    for fov, nb in ((180.0, 180), (260.0, 260), (360.0, 360)):
        b = make(world, seed=seed, n_beams=nb, fov_deg=fov)
        for tag, sm in (("shipped seg", "seg"), ("E2 point  ", "point")):
            r = DS.run(dict(b), F.BandSLAM, sample=sm, spec=None, nph=0)
            print(f"  {b['name']} {tag}  ATE {r['ate']:6.3f}  "
                  f"med {r['med']:6.3f}  loops {r['loops']}", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if what == "bench":
        bench()
    elif what == "fov":
        fov_study(*sys.argv[2:])
