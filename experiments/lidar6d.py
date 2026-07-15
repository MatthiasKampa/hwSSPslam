"""LIDAR 6-DoF refinement (user 2026-07-14: "refine lidar 6DoF ... then
cohere into the deploy variant").

Venue: TUM DEPTH CLOUDS as the 6-DoF range-sensor surrogate (dense 3D
geometry, no camera features, mocap truth, real pitch/yaw/roll motion —
SPOT walks flat floors). Transfer check: synthetic SE(3) on real school
clouds (label-free). Decode stack, mirroring the vision-6DoF findings:
  coarse   separable joint SE(3): rotation grid (5 deg) x one-matmul
           translation correlation (0.1 m)
  refine   local grid (1.67 deg / 0.033 m)
  GN       iterated 6x6 linearized decode ON POINTS (both frames fresh —
           an ego-motion op, not a frozen-map op): analytic rotation +
           translation derivative vectors, 2-3 iterations
  gyro     adjacent-frame 6x6 linear decode (the cloud gyro @ sensor rate)

Usage: python3 -m experiments.lidar6d se3|gyro [seq]
       python3 -m experiments.lidar6d se3syn      # school clouds
"""
import sys
from pathlib import Path

import numpy as np

import experiments.lattice3d as L3
import experiments.vision6d as V6

BASE = Path(__file__).resolve().parents[1] / "data" / "tum"
SEQ = "rgbd_dataset_freiburg3_long_office_household"
SUB = 4


def depth_cloud(depth_mm, K):
    d = depth_mm[::SUB, ::SUB].astype(np.float64) / 1000.0
    ys, xs = np.mgrid[0:depth_mm.shape[0]:SUB, 0:depth_mm.shape[1]:SUB]
    ok = (d > 0.2) & (d < 8.0)
    Ki = np.linalg.inv(K)
    uv = np.stack([xs[ok].astype(float), ys[ok].astype(float),
                   np.ones(int(ok.sum()))])
    b = (Ki @ uv).T
    return b * d[ok][:, None], np.ones(int(ok.sum()))


def _gn_iterate(W, P0, w0, v1, R0, t0, iters=3):
    """Gauss-Newton on points: relinearize the 6x6 model at the current
    (R, t); v1 fixed. Update R by left-multiplied axis rotations."""
    R, t = R0.copy(), t0.copy()
    for _ in range(iters):
        Pc = P0 @ R.T + t
        v0c = V6._enc_raw(W, Pc, w0)
        Dm = np.concatenate([V6._deriv_axes(W, Pc, w0),
                             V6._deriv_transl(W, Pc, w0)], 1)
        A2 = np.concatenate([Dm.real, Dm.imag])
        b2 = np.concatenate([(v1 - v0c).real, (v1 - v0c).imag])
        th, *_ = np.linalg.lstsq(A2, b2, rcond=None)
        dR = L3._axis_rot("yaw", np.degrees(th[0])) \
            @ L3._axis_rot("pitch", np.degrees(th[1])) \
            @ L3._axis_rot("roll", np.degrees(th[2]))
        R = dR @ R
        t = t + th[3:6]
    return R, t


def _report(tag, er, et):
    print(f"  {tag}: rot med {np.median(er):.2f} deg p90 "
          f"{np.percentile(er, 90):.2f} | transl med "
          f"{np.median(et):.3f} m p90 {np.percentile(et, 90):.3f}",
          flush=True)


def bench_se3(seq=SEQ):
    gray, depth, gt, K = V6.load(seq)
    pairs = V6._se3_pairs(gt, max_pairs=25)
    W = L3.make_lattices()["azel3d"]
    rg = V6._rot_grid()
    tg = V6._t_grid()
    Et = np.exp(1j * (tg @ W.T))
    clouds = [depth_cloud(d, K) for d in depth]
    print(f"lidar-surrogate SE(3) ({seq.split('_freiburg')[-1]}, "
          f"{len(pairs)} pairs, pts med "
          f"{int(np.median([len(c[0]) for c in clouds]))}):")
    E = {k: ([], []) for k in ("coarse", "refine", "GNx3")}
    for i, j in pairs:
        Rt, tt = V6._rel_pose(gt[i], gt[j])
        sc, ypr, Rh, th = V6._decode_se3(W, clouds[i], clouds[j], rg, tg,
                                         Et)
        _ang = lambda A, B: np.degrees(np.arccos(np.clip(
            (np.trace(A @ B.T) - 1) / 2, -1, 1)))
        E["coarse"][0].append(_ang(Rh, Rt))
        E["coarse"][1].append(np.linalg.norm(th - tt))
        rg2 = V6._rot_grid(center=ypr, half=3.33, step=1.67)
        tg2 = th[None] + V6._t_grid(half=0.1, step=0.033)
        Et2 = np.exp(1j * (tg2 @ W.T))
        sc2, ypr2, Rh2, th2 = V6._decode_se3(W, clouds[i], clouds[j],
                                             rg2, tg2, Et2)
        E["refine"][0].append(_ang(Rh2, Rt))
        E["refine"][1].append(np.linalg.norm(th2 - tt))
        v1 = V6._enc_raw(W, *clouds[j])
        Rg, tgn = _gn_iterate(W, clouds[i][0], clouds[i][1], v1, Rh2, th2)
        E["GNx3"][0].append(_ang(Rg, Rt))
        E["GNx3"][1].append(np.linalg.norm(tgn - tt))
    for k, (er, et) in E.items():
        _report(k, er, et)


def bench_se3syn(run="school_run2", n=16, seed=11):
    """Transfer check on REAL school lidar clouds, synthetic SE(3)."""
    _, kts, _, _ = L3.sample_kf(run, n, need_ref=False)
    clouds, _ = L3.load_clouds(run, kts)
    clouds = [(c, np.ones(len(c))) for c in clouds
              if c is not None and len(c) > 500]
    W = L3.make_lattices()["azel3d"]
    rg = V6._rot_grid()
    tg = V6._t_grid()
    Et = np.exp(1j * (tg @ W.T))
    rng = np.random.default_rng(seed)
    print(f"school synthetic SE(3) ({len(clouds)} clouds):")
    E = {k: ([], []) for k in ("coarse", "GNx3")}
    for P, w in clouds:
        y, p, r = rng.uniform(-12, 12, 3)
        Rt = L3._axis_rot("yaw", y) @ L3._axis_rot("pitch", p) \
            @ L3._axis_rot("roll", r)
        tt = rng.uniform(-0.5, 0.5, 3)
        P1 = P @ Rt.T + tt
        sc, ypr, Rh, th = V6._decode_se3(W, (P, w), (P1, w), rg, tg, Et)
        _ang = lambda A, B: np.degrees(np.arccos(np.clip(
            (np.trace(A @ B.T) - 1) / 2, -1, 1)))
        E["coarse"][0].append(_ang(Rh, Rt))
        E["coarse"][1].append(np.linalg.norm(th - tt))
        v1 = V6._enc_raw(W, P1, w)
        Rg, tgn = _gn_iterate(W, P, w, v1, Rh, th)
        E["GNx3"][0].append(_ang(Rg, Rt))
        E["GNx3"][1].append(np.linalg.norm(tgn - tt))
    for k, (er, et) in E.items():
        _report(k, er, et)


def bench_gyro(seq=SEQ, max_pairs=100):
    gray, depth, gt, K = V6.load(seq)
    W = L3.make_lattices()["azel3d"]
    errs, mags = [], []
    n = 0
    for i in range(len(gray) - 1):
        if n >= max_pairs:
            break
        Rt, tt = V6._rel_pose(gt[i], gt[i + 1])
        ang = np.degrees(np.arccos(np.clip((np.trace(Rt) - 1) / 2, -1, 1)))
        if np.linalg.norm(tt) > 0.06 or not 0.3 <= ang <= 6.0:
            continue
        P0, w0 = depth_cloud(depth[i], K)
        P1, w1 = depth_cloud(depth[i + 1], K)
        v1 = V6._enc_raw(W, P1, w1)
        R, t = _gn_iterate(W, P0, w0, v1, np.eye(3), np.zeros(3), iters=2)
        w_true = np.array([Rt[2, 1] - Rt[1, 2], Rt[0, 2] - Rt[2, 0],
                           Rt[1, 0] - Rt[0, 1]])
        angr = np.arccos(np.clip((np.trace(Rt) - 1) / 2, -1, 1))
        w_true = w_true / max(2 * np.sin(angr), 1e-9) * angr
        wa = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                       R[1, 0] - R[0, 1]])
        ar = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        wa = wa / max(2 * np.sin(ar), 1e-9) * ar
        errs.append(np.degrees(np.linalg.norm(wa - w_true)))
        mags.append(ang)
        n += 1
    print(f"cloud gyro GNx2 ({seq.split('_freiburg')[-1]}, {n} pairs, "
          f"true rot med {np.median(mags):.1f} deg):")
    print(f"  rotation-vector err med {np.median(errs):.2f} deg p90 "
          f"{np.percentile(errs, 90):.2f}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "se3"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd == "se3":
        bench_se3(arg or SEQ)
    elif cmd == "se3syn":
        bench_se3syn(arg or "school_run2")
    elif cmd == "gyro":
        bench_gyro(arg or SEQ)