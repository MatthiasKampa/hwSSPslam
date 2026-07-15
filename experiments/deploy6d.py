"""Deploy-readiness experiments for the lidar+cam 6-DoF stack (user
2026-07-15: "continue experimenting for deploy of lidar + cam 6DoF").

Four deploy-blocking questions, each measured on TUM (mocap DIAGNOSTIC
labels select pairs and score errors; nothing enters an encoder):

  verify    the per-anchor 6-DoF VERIFY op on QUANTIZED stored vectors
            (the deploy stores 2b/3b snapshots + 6 derivative vectors;
            every banked SE(3)/gyro number was float). Two regimes:
            small-motion 6x6 derivative solve (stored {v0, D} quantized,
            fresh query v1) and rotation-search via frozen permutations
            (5-20 deg pairs). Channels: lidar depth-cloud + vision
            gridint-3D.
  rowdepth  depth-landmark machinery under LIDAR-STRUCTURED depth
            (elevation-band + N-beam row masks) instead of the banked
            random dropout — the lidar-projected-depth readiness curve
            in its real geometry (extrinsics = the standing unlock).
  gyro      ego-motion service refinements: GN-iterated VISUAL gyro
            (the filed fine-ring-linearization fix; both frames fresh,
            points available) + cloud/visual gyro FUSION
            (fixed-precision weights from the banked medians).
  hopf      the quaternion question, measured: uniform axis-angle-ball
            rotation grid vs the Euler box at equal candidate count
            (coarse SO(3) decode stage).

Usage: python3 -m experiments.deploy6d selftest|verify|rowdepth|gyro|hopf [seq]
"""
import sys

import numpy as np

import experiments.lattice3d as L3
import experiments.vision6d as V6
import experiments.lidar6d as L6

SEQ = "rgbd_dataset_freiburg3_long_office_household"
RNG = 11
VIS_LAMS = [0.35, 0.7, 1.4, 2.8]     # camera-range ladder (depth 0.2-8 m)


def W_vis3d():
    """The VISION channel's OWN vector space (two-map law: two distinct
    spaces, fusion at the estimate level only): azel-style directions
    (exact-yaw permutations kept) x a camera-range ladder."""
    return np.concatenate([(2 * np.pi / lam)
                           * L3.dirs_azel(12, [-40, -20, 0, 20, 40])
                           for lam in VIS_LAMS])


# --------------------------------------------------------------------------
#  quantization: house phase-only store, global energy preserved
# --------------------------------------------------------------------------
def qvec(v, nph):
    """Phase-only nph quantization, flat per-element magnitude, the
    vector's global norm preserved (the 2b/3b store model: relative
    phases carry the signal, one scale per vector)."""
    if nph == 0:
        return v
    ph = np.round(np.angle(v) * nph / (2 * np.pi)) * (2 * np.pi / nph)
    q = np.exp(1j * ph)
    return q * (np.linalg.norm(v) / max(np.linalg.norm(q), 1e-12))


def qstack(M, nph):
    return np.stack([qvec(M[:, k], nph) for k in range(M.shape[1])], 1)


# --------------------------------------------------------------------------
#  pair machinery
# --------------------------------------------------------------------------
def small_pairs(gt, n_max=80):
    """adjacent small-motion pairs (the gyro/near-anchor verify regime)."""
    out = []
    for i in range(len(gt) - 1):
        Rt, tt = V6._rel_pose(gt[i], gt[i + 1])
        ang = np.degrees(np.arccos(np.clip((np.trace(Rt) - 1) / 2, -1, 1)))
        if np.linalg.norm(tt) <= 0.06 and 0.3 <= ang <= 6.0:
            out.append((i, i + 1, Rt, tt))
        if len(out) == n_max:
            break
    return out


def rotvec(R):
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]])
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    return w / max(2 * np.sin(ang), 1e-9) * ang


def solve66(v0, Dm, v1):
    """the stored-anchor verify op: v1 ~ v0 + Dm . theta (6-vector)."""
    A2 = np.concatenate([Dm.real, Dm.imag])
    b2 = np.concatenate([(v1 - v0).real, (v1 - v0).imag])
    th, *_ = np.linalg.lstsq(A2, b2, rcond=None)
    return th


def gyro_err(th, Rt):
    w_true = rotvec(Rt)
    w_hat = np.array([th[2], th[1], th[0]])   # (yaw,pitch,roll)->(z,y,x)
    return np.degrees(np.linalg.norm(w_hat - w_true))


# --------------------------------------------------------------------------
#  verify: quantized stored anchors
# --------------------------------------------------------------------------
def bench_verify(seq=SEQ):
    gray, depth, gt, K = V6.load(seq)
    pairs = small_pairs(gt)
    chans = {
        "lidar-cloud": (lambda i: L6.depth_cloud(depth[i], K),
                        L3.make_lattices()["azel3d"]),
        "vis-grid3D ": (lambda i: V6.feats(gray[i], K, "gridint",
                                           depth[i]), W_vis3d()),
    }
    print(f"quantized-anchor small-motion verify ({seq.split('_freiburg')[-1]},"
          f" {len(pairs)} pairs, stored {{v0, 6 D-vectors}} quantized, "
          f"query fresh; TWO vector spaces, lidar/vision ladders):")
    for cname, (feat, W) in chans.items():
        F = {}
        for nph, tag in ((0, "float"), (8, "3b"), (4, "2b")):
            errs = []
            for i, j, Rt, tt in pairs:
                if i not in F:
                    P0, w0 = feat(i)
                    P1, w1 = feat(j)
                    v0 = V6._enc_raw(W, P0, w0)
                    Dm = np.concatenate([V6._deriv_axes(W, P0, w0),
                                         V6._deriv_transl(W, P0, w0)], 1)
                    v1 = V6._enc_raw(W, P1, w1)
                    F[i] = (v0, Dm, v1)
                v0, Dm, v1 = F[i]
                th = solve66(qvec(v0, nph), qstack(Dm, nph), v1)
                errs.append(gyro_err(th, Rt))
            print(f"  {cname} {tag:5s}: rot-vec err med "
                  f"{np.median(errs):.2f} deg  p90 "
                  f"{np.percentile(errs, 90):.2f}", flush=True)
    # rotation-search regime: 5-20 deg pairs, frozen permutations on the
    # quantized stored vector
    rp = []
    for i in range(len(gt)):
        for j in range(i + 1, min(i + 12, len(gt))):
            Rt, tt = V6._rel_pose(gt[i], gt[j])
            ang = np.degrees(np.arccos(np.clip((np.trace(Rt) - 1) / 2,
                                               -1, 1)))
            if np.linalg.norm(tt) < 0.15 and 5 <= ang <= 20:
                rp.append((i, j, Rt))
    rp = rp[:: max(1, len(rp) // 30)][:30]
    W = W_vis3d()                      # stored VISION snapshot space
    degs = np.arange(-20, 21, 5)
    grid = [(y, p, r) for y in degs for p in degs for r in degs]
    Rg = [L3._axis_rot("yaw", y) @ L3._axis_rot("pitch", p)
          @ L3._axis_rot("roll", r) for y, p, r in grid]
    perms = [L3._perm_of(W, R)[:2] for R in Rg]
    F3 = {}
    print(f"rotation-search verify ({len(rp)} pairs, 5-20 deg, frozen "
          f"permutations on the stored vector):")
    for nph, tag in ((0, "float"), (8, "3b"), (4, "2b")):
        errs = []
        for i, j, Rt in rp:
            if i not in F3:
                F3[i] = L3.encode(W, *V6.feats(gray[i], K, "gridint",
                                               depth[i]))
            if j not in F3:
                F3[j] = L3.encode(W, *V6.feats(gray[j], K, "gridint",
                                               depth[j]))
            v0 = qvec(F3[i], nph)
            v1 = F3[j]
            bs, best = -1, 0
            for gi, (perm, sgn) in enumerate(perms):
                s = np.abs(np.vdot(v1, L3._apply_perm(v0, perm, sgn)))
                if s > bs:
                    bs, best = s, gi
            err = np.degrees(np.arccos(np.clip(
                (np.trace(Rg[best] @ Rt.T) - 1) / 2, -1, 1)))
            errs.append(err)
        print(f"  {tag:5s}: rot err med {np.median(errs):.1f} deg  p90 "
              f"{np.percentile(errs, 90):.1f}", flush=True)


# --------------------------------------------------------------------------
#  rowdepth: lidar-structured depth masks
# --------------------------------------------------------------------------
def bench_rowdepth(seq=SEQ):
    """SE(3) decode + place AUC with depth restricted to an elevation
    band + N beam rows (the lidar-projection geometry) — vs the banked
    RANDOM-dropout grace curve."""
    gray, depth, gt, K = V6.load(seq)
    H = depth[0].shape[0]
    band = (int(H * 0.15), int(H * 0.85))       # lidar elevation band

    def mask_depth(d, beams):
        m = np.zeros_like(d)
        if beams == 0:
            return d                             # full registered depth
        rows = np.linspace(band[0], band[1] - 1, beams).astype(int)
        # each beam illuminates its row (cell granularity is 16 px, so
        # spread each beam over +-1 px to model spot size)
        for r in rows:
            m[max(r - 1, 0):r + 2] = d[max(r - 1, 0):r + 2]
        return m

    pairs = V6._se3_pairs(gt, max_pairs=15)
    W = W_vis3d()                      # vision channel, own space
    rg = V6._rot_grid()
    tg = V6._t_grid()
    Et = np.exp(1j * (tg @ W.T))
    print(f"lidar-structured depth ({seq.split('_freiburg')[-1]}, "
          f"{len(pairs)} SE(3) pairs; band rows {band[0]}-{band[1]}):")
    for beams, tag in ((0, "full-reg"), (64, "64-beam"), (16, "16-beam"),
                       (8, " 8-beam"), (4, " 4-beam")):
        F = {}
        er, et, ncell = [], [], []
        for i, j in pairs:
            for k in (i, j):
                if k not in F:
                    F[k] = V6.feats(gray[k], K, "gridint",
                                    mask_depth(depth[k], beams))
            if len(F[i][0]) < 20 or len(F[j][0]) < 20:
                continue
            Rt, tt = V6._rel_pose(gt[i], gt[j])
            sc, ypr, Rh, th = V6._decode_se3(W, F[i], F[j], rg, tg, Et)
            er.append(np.degrees(np.arccos(np.clip(
                (np.trace(Rh @ Rt.T) - 1) / 2, -1, 1))))
            et.append(np.linalg.norm(th - tt))
            ncell.append(len(F[i][0]))
        print(f"  {tag}: rot med {np.median(er):5.1f} deg | transl med "
              f"{np.median(et):.3f} m | cells med {int(np.median(ncell))}",
              flush=True)


# --------------------------------------------------------------------------
#  gyro: GN-iterated visual gyro + cloud/visual fusion
# --------------------------------------------------------------------------
def bench_gyro(seq=SEQ, max_pairs=80):
    gray, depth, gt, K = V6.load(seq)
    Wl = L3.make_lattices()["azel3d"]  # lidar space
    Wv = W_vis3d()                     # vision space (two-map law)
    pairs = small_pairs(gt, max_pairs)
    e_vis, e_gn, e_cld, e_fuse, e_sel = [], [], [], [], []
    for i, j, Rt, tt in pairs:
        Fv0 = V6.feats(gray[i], K, "gridint", depth[i])
        Fv1 = V6.feats(gray[j], K, "gridint", depth[j])
        Pc0, wc0 = L6.depth_cloud(depth[i], K)
        Pc1, wc1 = L6.depth_cloud(depth[j], K)
        w_true = rotvec(Rt)
        # linear visual gyro (banked service) — VISION space
        v1 = V6._enc_raw(Wv, *Fv1)
        Dm = np.concatenate([V6._deriv_axes(Wv, *Fv0),
                             V6._deriv_transl(Wv, *Fv0)], 1)
        th = solve66(V6._enc_raw(Wv, *Fv0), Dm, v1)
        wv = np.array([th[2], th[1], th[0]])
        e_vis.append(np.degrees(np.linalg.norm(wv - w_true)))
        # GN-iterated visual gyro (points available frame-to-frame)
        Rg_, tg_ = L6._gn_iterate(Wv, Fv0[0], Fv0[1], v1, np.eye(3),
                                  np.zeros(3), iters=2)
        wg = rotvec(Rg_)
        e_gn.append(np.degrees(np.linalg.norm(wg - w_true)))
        # cloud gyro (banked service) — LIDAR space
        vc1 = V6._enc_raw(Wl, Pc1, wc1)
        Rc, tc = L6._gn_iterate(Wl, Pc0, wc0, vc1, np.eye(3),
                                np.zeros(3), iters=2)
        wc = rotvec(Rc)
        e_cld.append(np.degrees(np.linalg.norm(wc - w_true)))
        # fusion: fixed precision weights from the banked medians
        # (visual 1.82 deg, cloud 1.34 deg)
        pv, pc = 1 / 1.82 ** 2, 1 / 1.34 ** 2
        wf = (pv * wg + pc * wc) / (pv + pc)
        e_fuse.append(np.degrees(np.linalg.norm(wf - w_true)))
        # selection by encode residual (label-free)
        rv = np.linalg.norm(v1 - V6._enc_raw(
            Wv, Fv0[0] @ Rg_.T + tg_, Fv0[1]))
        rc = np.linalg.norm(vc1 - V6._enc_raw(
            Wl, Pc0 @ Rc.T + tc, wc0))
        e_sel.append(e_gn[-1] if rv / max(np.linalg.norm(v1), 1e-9)
                     < rc / max(np.linalg.norm(vc1), 1e-9) else e_cld[-1])
    print(f"gyro services ({seq.split('_freiburg')[-1]}, {len(pairs)} "
          f"adjacent pairs; vision/lidar in their OWN spaces, fusion "
          f"at the rotation-vector level):")
    for tag, e in (("visual linear (banked)", e_vis),
                   ("visual GNx2 (filed fix)", e_gn),
                   ("cloud GNx2 (banked)", e_cld),
                   ("fused (fixed weights)", e_fuse),
                   ("selected (residual)", e_sel)):
        print(f"  {tag:24s}: med {np.median(e):.2f} deg  p90 "
              f"{np.percentile(e, 90):.2f}", flush=True)


# --------------------------------------------------------------------------
#  hopf: uniform rotation-ball grid vs Euler box (the quaternion question)
# --------------------------------------------------------------------------
def _ball_grid(n, half_deg, seed=RNG):
    """~uniform rotation-vector ball: deterministic low-discrepancy fill
    (uniform quaternion-ball equivalent for small angles)."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        c = rng.uniform(-1, 1, 3)
        if np.linalg.norm(c) <= 1.0:
            pts.append(c * np.deg2rad(half_deg))
    return np.array(pts)


def _R_of_w(w):
    ang = np.linalg.norm(w)
    if ang < 1e-12:
        return np.eye(3)
    k = w / ang
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]],
                   [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx


def bench_hopf(seq=SEQ):
    gray, depth, gt, K = V6.load(seq)
    rp = []
    for i in range(len(gt)):
        for j in range(i + 1, min(i + 12, len(gt))):
            Rt, tt = V6._rel_pose(gt[i], gt[j])
            ang = np.degrees(np.arccos(np.clip((np.trace(Rt) - 1) / 2,
                                               -1, 1)))
            if np.linalg.norm(tt) < 0.15 and 5 <= ang <= 18:
                rp.append((i, j, Rt))
    rp = rp[:: max(1, len(rp) // 25)][:25]
    W = W_vis3d()
    F3 = {}
    for i, j, _ in rp:
        for k in (i, j):
            if k not in F3:
                F3[k] = L3.encode(W, *V6.feats(gray[k], K, "gridint",
                                               depth[k]))
    degs = np.arange(-20, 21, 5)
    euler = [L3._axis_rot("yaw", y) @ L3._axis_rot("pitch", p)
             @ L3._axis_rot("roll", r)
             for y in degs for p in degs for r in degs]
    arms = {
        f"euler box 5deg (n={len(euler)})": euler,
        "ball n=729 (equal count)": [_R_of_w(w) for w in
                                     _ball_grid(729, 22)],
        "ball n=364 (half count) ": [_R_of_w(w) for w in
                                     _ball_grid(364, 22)],
        "ball n=182 (quarter)    ": [_R_of_w(w) for w in
                                     _ball_grid(182, 22)],
    }
    print(f"rotation-grid geometry ({seq.split('_freiburg')[-1]}, "
          f"{len(rp)} pairs, 5-18 deg, frozen-permutation decode):")
    for tag, Rs in arms.items():
        perms = [L3._perm_of(W, R)[:2] for R in Rs]
        errs = []
        for i, j, Rt in rp:
            v0, v1 = F3[i], F3[j]
            bs, best = -1, 0
            for gi, (perm, sgn) in enumerate(perms):
                s = np.abs(np.vdot(v1, L3._apply_perm(v0, perm, sgn)))
                if s > bs:
                    bs, best = s, gi
            errs.append(np.degrees(np.arccos(np.clip(
                (np.trace(Rs[best] @ Rt.T) - 1) / 2, -1, 1))))
        print(f"  {tag}: med {np.median(errs):.1f} deg  p90 "
              f"{np.percentile(errs, 90):.1f}", flush=True)


def selftest():
    rng = np.random.default_rng(RNG)
    v = rng.normal(size=240) + 1j * rng.normal(size=240)
    for nph in (4, 8):
        q = qvec(v, nph)
        assert abs(np.linalg.norm(q) - np.linalg.norm(v)) < 1e-9
        assert np.abs(np.abs(q) - np.abs(q[0])).max() < 1e-9
    # small-motion solve on synthetic points survives 2b storage
    W = L3.make_lattices()["azel3d"]
    P = rng.uniform(-3, 3, (400, 3))
    w = np.ones(400)
    Rt = L3._axis_rot("yaw", 1.5) @ L3._axis_rot("pitch", -0.8)
    tt = np.array([0.02, -0.01, 0.015])
    v0 = V6._enc_raw(W, P, w)
    v1 = V6._enc_raw(W, P @ Rt.T + tt, w)
    Dm = np.concatenate([V6._deriv_axes(W, P, w),
                         V6._deriv_transl(W, P, w)], 1)
    e_f = gyro_err(solve66(v0, Dm, v1), Rt)
    e_q = gyro_err(solve66(qvec(v0, 4), qstack(Dm, 4), v1), Rt)
    # gate the QUANT DELTA, not an absolute — the linear model's own
    # floor on a ~1.7 deg motion is a few tenths (banked: 1.8 on 2.6)
    assert e_f < 1.0, e_f
    assert e_q < e_f + 1.0, (e_f, e_q)
    print(f"selftest ok: quant preserves norm/flat-mag; synthetic "
          f"small-motion solve float {e_f:.3f} deg / 2b-stored "
          f"{e_q:.3f} deg")
    # rotation ball grid: inside the ball, right count
    B = _ball_grid(100, 20)
    assert len(B) == 100 and np.degrees(
        np.linalg.norm(B, axis=1)).max() <= 20.001


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    seq = sys.argv[2] if len(sys.argv) > 2 else SEQ
    dict(selftest=lambda s: selftest(), verify=bench_verify,
         rowdepth=bench_rowdepth, gyro=bench_gyro,
         hopf=bench_hopf)[cmd](seq)
