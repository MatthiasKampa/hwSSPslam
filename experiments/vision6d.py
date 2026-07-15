"""First 6-DoF vision probe on TUM RGB-D (mocap GT; user: "get a 3D/6D
SLAM vision data set" + "cover 2D+heading vs 3D+pitch/yaw/roll").

TUM gives what SPOT couldn't: honest 6-DoF labels with real pitch/roll
motion, and DEPTH REGISTERED TO RGB — the depth-augmented-landmark
experiment (banked NEGATIVE on SPOT under guessed extrinsics) gets its
clean venue: p3 = depth * K^-1 [u, v, 1], no extrinsics at all.

Benches (python3 -m experiments.vision6d <bench> <seq>):
  place   same-place AUC (mocap labels; forward/reverse heading split):
          gridint dense vs fast9 sparse, bearings (vis-fib) vs
          depth-3D landmarks (azel3d)
  rot     REAL-motion rotation decode: pairs with |dt| < 0.15 m and
          5..20 deg relative rotation (mocap-selected — capability
          DIAGNOSTIC); azel3d/fib3d permutation search on frozen
          3D-landmark vectors vs GT rotation; az2d = heading-only
          control
"""
import sys
from pathlib import Path

import numpy as np

import experiments.lattice3d as L3
import experiments.twomap as TM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hw" / "ecp5"
                       / "host"))
import golden_cam as GC          # noqa: E402

BASE = Path(__file__).resolve().parents[1] / "data" / "tum"
SEQ = "rgbd_dataset_freiburg3_long_office_household"


def quat_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load(seq):
    z = np.load(BASE / f"{seq}.npz")
    return z["gray"], z["depth_mm"], z["gt"], z["K"]


def feats(gray, K, mode, depth=None):
    if mode == "gridint":
        ys, xs, w = TM.grid_int(gray)
    else:
        import experiments.detzoo as DZ
        _, (ys, xs, sc) = DZ.detect_target(DZ.det_fast9, gray, 900)
        w = sc.astype(float) + 1.0
    Ki = np.linalg.inv(K)
    uv = np.stack([xs + 0.0, ys + 0.0, np.ones(len(xs))])
    b = (Ki @ uv).T
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    if depth is None:
        return b, w
    d = depth[np.asarray(ys, int), np.asarray(xs, int)] / 1000.0
    ok = (d > 0.2) & (d < 8.0)
    return (b[ok] * d[ok, None]), w[ok]


def bench_place(seq=SEQ):
    gray, depth, gt, K = load(seq)
    pos, quat = gt[:, :3], gt[:, 3:7]
    yaw = np.array([np.arctan2(quat_R(q)[1, 0], quat_R(q)[0, 0])
                    for q in quat])
    pose = np.stack([pos[:, 0], pos[:, 1], yaw], 1)
    n = len(gray)
    d3 = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    ii, jj = np.triu_indices(n, 1)
    gap = np.abs(ii - jj) >= 30
    same = (d3[ii, jj] < 0.5) & gap
    diff = (d3[ii, jj] > 2.0) & (d3[ii, jj] < 6.0)
    si, sj, di, dj = ii[same], jj[same], ii[diff], jj[diff]
    dyaw = np.abs(np.arctan2(np.sin(yaw[si] - yaw[sj]),
                             np.cos(yaw[si] - yaw[sj])))
    fwd = dyaw < np.deg2rad(60)
    print(f"TUM place ({seq.split('_freiburg')[-1]}, {n} kf, mocap "
          f"labels; {len(si)} same [{int(fwd.sum())} fwd] / "
          f"{len(di)} diff):")
    Wfib = TM.W_vis_fib()
    Waz = L3.make_lattices()["azel3d"]
    arms = {
        "gridint-bear": [L3.encode(Wfib, *feats(g, K, "gridint"))
                         for g in gray],
        "fast9-bear ": [L3.encode(Wfib, *feats(g, K, "fast9"))
                        for g in gray],
        "gridint-3D ": [L3.encode(Waz, *feats(g, K, "gridint", d))
                        for g, d in zip(gray, depth)],
        "fast9-3D   ": [L3.encode(Waz, *feats(g, K, "fast9", d))
                        for g, d in zip(gray, depth)],
    }
    for name, V in arms.items():
        V = np.stack(V)
        S = np.abs(V @ V.conj().T)
        auc = L3._auc(S[si, sj], S[di, dj])
        af = L3._auc(S[si, sj][fwd], S[di, dj]) if fwd.any() else np.nan
        ar = L3._auc(S[si, sj][~fwd], S[di, dj]) if (~fwd).any() else np.nan
        print(f"  {name} AUC {auc:.3f} (fwd {af:.3f} / rev {ar:.3f})",
              flush=True)


def bench_rot(seq=SEQ, max_pairs=40):
    gray, depth, gt, K = load(seq)
    pos, quat = gt[:, :3], gt[:, 3:7]
    Rs = [quat_R(q) for q in quat]
    n = len(gray)
    pairs = []
    for i in range(n):
        for j in range(i + 1, min(i + 12, n)):
            dt = np.linalg.norm(pos[i] - pos[j])
            Rrel = Rs[j].T @ Rs[i]
            ang = np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2,
                                               -1, 1)))
            if dt < 0.15 and 5 <= ang <= 20:
                pairs.append((i, j, Rrel, ang))
    pairs = pairs[:: max(1, len(pairs) // max_pairs)][:max_pairs]
    print(f"TUM real-motion rotation decode ({len(pairs)} pairs, "
          f"|dt|<0.15 m, 5-20 deg):")
    degs = np.arange(-20, 21, 5)
    grid = [(y, p, r) for y in degs for p in degs for r in degs]
    Rg = [L3._axis_rot("yaw", y) @ L3._axis_rot("pitch", p)
          @ L3._axis_rot("roll", r) for y, p, r in grid]
    F3 = [feats(g, K, "gridint", d) for g, d in zip(gray, depth)]
    for name in ("az2d", "fib3d", "azel3d"):
        W = L3.make_lattices()[name]
        perms = [L3._perm_of(W, R)[:2] for R in Rg]
        errs = []
        for i, j, Rt, ang in pairs:
            v0 = L3.encode(W, *F3[i])
            v1 = L3.encode(W, *F3[j])
            bs, best = -1, 0
            for gi, (perm, sgn) in enumerate(perms):
                s = np.abs(np.vdot(v1, L3._apply_perm(v0, perm, sgn)))
                if s > bs:
                    bs, best = s, gi
            err = np.degrees(np.arccos(np.clip(
                (np.trace(Rg[best] @ Rt.T) - 1) / 2, -1, 1)))
            errs.append(err)
        print(f"  {name:7s} rot err med {np.median(errs):.1f} deg "
              f"p90 {np.percentile(errs, 90):.1f}  (grid floor ~3.5; "
              f"true rot 5-20)", flush=True)


def _rel_pose(gt_i, gt_j):
    """Camera-frame relative pose j<-i: p_j = R p_i + t."""
    Ri, Rj = quat_R(gt_i[3:7]), quat_R(gt_j[3:7])
    R = Rj.T @ Ri
    t = Rj.T @ (gt_i[:3] - gt_j[:3])
    return R, t


def _se3_pairs(gt, dt_lo=0.10, dt_hi=0.50, rot_lo=5, rot_hi=20,
               max_pairs=30):
    pos = gt[:, :3]
    Rs = [quat_R(q) for q in gt[:, 3:7]]
    out = []
    for i in range(len(gt)):
        for j in range(i + 1, min(i + 15, len(gt))):
            d = np.linalg.norm(pos[i] - pos[j])
            Rrel = Rs[j].T @ Rs[i]
            ang = np.degrees(np.arccos(np.clip(
                (np.trace(Rrel) - 1) / 2, -1, 1)))
            if dt_lo <= d <= dt_hi and rot_lo <= ang <= rot_hi:
                out.append((i, j))
    return out[:: max(1, len(out) // max_pairs)][:max_pairs]


def _rot_grid(center=(0, 0, 0), half=15, step=5):
    ys = np.arange(center[0] - half, center[0] + half + 1e-9, step)
    ps = np.arange(center[1] - half, center[1] + half + 1e-9, step)
    rs = np.arange(center[2] - half, center[2] + half + 1e-9, step)
    out = []
    for y in ys:
        for p in ps:
            for r in rs:
                out.append(((y, p, r),
                            L3._axis_rot("yaw", y) @ L3._axis_rot("pitch", p)
                            @ L3._axis_rot("roll", r)))
    return out


def _t_grid(half=0.6, step=0.1):
    s = np.arange(-half, half + 1e-9, step)
    gx, gy, gz = np.meshgrid(s, s, s, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)


def _decode_se3(W, F0, F1, rgrid, tgrid, Et):
    """Separable joint decode: perm per rotation cand, translation by one
    correlation matmul; -> (ypr, R, t, score)."""
    v1 = L3.encode(W, *F1)
    best = None
    for ypr, R in rgrid:
        v0r = L3.encode(W, (F0[0] @ R.T), F0[1])
        g = np.conj(v1) * v0r
        sc = np.real(Et @ g)
        k = int(np.argmax(sc))
        if best is None or sc[k] > best[0]:
            best = (sc[k], ypr, R, tgrid[k])
    return best


def bench_se3(seq=SEQ):
    """Joint SE(3) decode on real pairs (|dt| 0.1-0.5 m, rot 5-20 deg):
    coarse (5 deg x 0.1 m) + local refine (1.67 deg x 0.033 m)."""
    gray, depth, gt, K = load(seq)
    pairs = _se3_pairs(gt)
    F = [feats(g, K, "gridint", d) for g, d in zip(gray, depth)]
    W = L3.make_lattices()["azel3d"]
    rg = _rot_grid()
    tg = _t_grid()
    Et = np.exp(1j * (tg @ W.T))
    print(f"SE(3) joint decode ({seq.split('_freiburg')[-1]}, "
          f"{len(pairs)} pairs, coarse 5deg/0.1m + refine):")
    er_c, et_c, er_f, et_f = [], [], [], []
    for i, j in pairs:
        Rt, tt = _rel_pose(gt[i], gt[j])
        sc, ypr, Rh, th = _decode_se3(W, F[i], F[j], rg, tg, Et)
        er_c.append(np.degrees(np.arccos(np.clip(
            (np.trace(Rh @ Rt.T) - 1) / 2, -1, 1))))
        et_c.append(np.linalg.norm(th - tt))
        # local refine around the coarse peak
        rg2 = _rot_grid(center=ypr, half=3.33, step=1.67)
        tg2 = th[None] + _t_grid(half=0.1, step=0.033)
        Et2 = np.exp(1j * (tg2 @ W.T))
        sc2, ypr2, Rh2, th2 = _decode_se3(W, F[i], F[j], rg2, tg2, Et2)
        er_f.append(np.degrees(np.arccos(np.clip(
            (np.trace(Rh2 @ Rt.T) - 1) / 2, -1, 1))))
        et_f.append(np.linalg.norm(th2 - tt))
    print(f"  coarse: rot med {np.median(er_c):.1f} deg p90 "
          f"{np.percentile(er_c, 90):.1f} | transl med "
          f"{np.median(et_c):.3f} m p90 {np.percentile(et_c, 90):.3f}")
    print(f"  refine: rot med {np.median(er_f):.1f} deg p90 "
          f"{np.percentile(er_f, 90):.1f} | transl med "
          f"{np.median(et_f):.3f} m p90 {np.percentile(et_f, 90):.3f}",
          flush=True)


def _enc_raw(W, pts, w):
    v = np.exp(1j * (pts @ W.T))
    return (v.T @ w)                      # UNNORMALIZED (gyro linear model)


def _deriv_axes(W, pts, w):
    """Analytic dV/dtheta_axis at theta=0 (the segder/Cx pattern in SO(3)):
    d/dth exp(iW.(R p)) = i (W.(A p)) exp(iW.p), A = so(3) generator."""
    A = {"yaw": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0.0]]),
         "pitch": np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0.0]]),
         "roll": np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0.0]])}
    ph = np.exp(1j * (pts @ W.T))
    D = []
    for ax in ("yaw", "pitch", "roll"):
        Ap = pts @ A[ax].T
        D.append(((1j * (Ap @ W.T)) * ph).T @ w)
    return np.stack(D, 1)                 # (D, 3)


def _deriv_transl(W, pts, w):
    """Analytic dV/dt_k = i W[:,k] . V-terms — translation derivatives
    (three more cis-MAC accumulations; completes the 6x6 linear model)."""
    ph = np.exp(1j * (pts @ W.T))
    return np.stack([(1j * W[:, k][None, :] * ph).T @ w
                     for k in range(3)], 1)   # (D, 3)


def bench_gyro(seq=SEQ, max_pairs=120):
    """VSA visual gyro: adjacent keyframes (~0.2 s), rotation vector by
    3x3 least squares on the analytic derivative vectors — sub-grid
    small-rotation decode, inner products only (RTL: 3 extra cis-MAC
    accumulations at encode)."""
    gray, depth, gt, K = load(seq)
    W = L3.make_lattices()["azel3d"]
    errs, mags = [], []
    n = 0
    for i in range(0, len(gray) - 1):
        if n >= max_pairs:
            break
        Rt, tt = _rel_pose(gt[i], gt[i + 1])
        ang = np.degrees(np.arccos(np.clip((np.trace(Rt) - 1) / 2, -1, 1)))
        if np.linalg.norm(tt) > 0.06 or not 0.3 <= ang <= 6.0:
            continue
        F0 = feats(gray[i], K, "gridint", depth[i])
        F1 = feats(gray[i + 1], K, "gridint", depth[i + 1])
        v0, v1 = _enc_raw(W, *F0), _enc_raw(W, *F1)
        # 6x6 JOINT linear model: rotation + translation derivatives
        # (translation-induced change no longer aliases into rotation).
        # Full rings: coarse-only was tried and REGRESSED (med 2.31,
        # p90 12.7 — too few constraints); fine-ring linearization error
        # is the accuracy limiter -> iterated GN on points filed for the
        # frame-to-frame service (both frames fresh; not a frozen-map op).
        Dm = np.concatenate([_deriv_axes(W, *F0),
                             _deriv_transl(W, *F0)], 1)
        A2 = np.concatenate([Dm.real, Dm.imag])
        b2 = np.concatenate([(v1 - v0).real, (v1 - v0).imag])
        th, *_ = np.linalg.lstsq(A2, b2, rcond=None)
        # truth rotation vector (log map)
        w_true = np.array([Rt[2, 1] - Rt[1, 2], Rt[0, 2] - Rt[2, 0],
                           Rt[1, 0] - Rt[0, 1]])
        angr = np.arccos(np.clip((np.trace(Rt) - 1) / 2, -1, 1))
        w_true = w_true / max(2 * np.sin(angr), 1e-9) * angr
        # decoded generators are body-axis: (yaw, pitch, roll) -> (z, y, x)
        w_hat = np.array([th[2], th[1], th[0]])
        errs.append(np.degrees(np.linalg.norm(w_hat - w_true)))
        mags.append(ang)
        n += 1
    print(f"visual gyro ({seq.split('_freiburg')[-1]}, {n} adjacent pairs,"
          f" |dt|<0.06 m, true rot med {np.median(mags):.1f} deg):")
    print(f"  rotation-vector err med {np.median(errs):.2f} deg  p90 "
          f"{np.percentile(errs, 90):.2f}  (grid-search floor was 3.5)",
          flush=True)


def bench_depthrob(seq=SEQ, seed=11):
    """SE(3) decode vs depth COVERAGE (lidar-projected-depth readiness):
    randomly keep a fraction of landmark cells."""
    gray, depth, gt, K = load(seq)
    pairs = _se3_pairs(gt, max_pairs=15)
    W = L3.make_lattices()["azel3d"]
    rg = _rot_grid()
    tg = _t_grid()
    Et = np.exp(1j * (tg @ W.T))
    rng = np.random.default_rng(seed)
    F = [feats(g, K, "gridint", d) for g, d in zip(gray, depth)]
    print(f"depth-coverage ladder ({seq.split('_freiburg')[-1]}, "
          f"{len(pairs)} pairs):")
    for frac in (1.0, 0.5, 0.25, 0.10):
        er, et = [], []
        for i, j in pairs:
            def sub(Fk):
                m = rng.random(len(Fk[0])) < frac
                return (Fk[0][m], Fk[1][m])
            Rt, tt = _rel_pose(gt[i], gt[j])
            sc, ypr, Rh, th = _decode_se3(W, sub(F[i]), sub(F[j]), rg,
                                          tg, Et)
            er.append(np.degrees(np.arccos(np.clip(
                (np.trace(Rh @ Rt.T) - 1) / 2, -1, 1))))
            et.append(np.linalg.norm(th - tt))
        print(f"  cover {int(frac*100):3d}%: rot med {np.median(er):.1f} "
              f"deg | transl med {np.median(et):.3f} m", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "place"
    seq = sys.argv[2] if len(sys.argv) > 2 else SEQ
    dict(place=bench_place, rot=bench_rot, se3=bench_se3,
         gyro=bench_gyro, depthrob=bench_depthrob)[cmd](seq)
