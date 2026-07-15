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

TUNING round 2 (same session, additive benches):
  dtune   vision 6-DoF space tuning: camera-range ladder x cell weights
          x store model (flat-mag 2b/3b vs PER-RING magnitude scales —
          the filed fix for the +1 deg flattening cost); verify + gyro
          + TRANSLATION errors (never reported before)
  fuse2   two-space fusion IN THE SOLVER: block-stacked GN residuals
          (spaces stay distinct; one 6x6 solve) vs the banked post-hoc
          rotation-vector averaging
  chain   ego-motion at deploy rate: chain the fused gyro over 15 Hz
          steps (fr3 stride-2 parse) across one 5 Hz keyframe interval
          vs the single-shot 5 Hz decode
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


# --------------------------------------------------------------------------
#  tuning round 2: store model / ladder / weights, solver fusion, rate
# --------------------------------------------------------------------------
def qvec_rs(v, nph, nlam):
    """Ring-scaled store: phase-only nph + ONE magnitude scalar per
    wavelength ring (the filed fix: tonight's flat-mag store cost ~+1
    deg and 2b ~= 3b showed magnitude, not phase resolution, is what
    the derivative solve misses). Cost: nlam scalars per vector."""
    if nph == 0:
        return v
    D = len(v)
    rows = D // nlam
    ph = np.round(np.angle(v) * nph / (2 * np.pi)) * (2 * np.pi / nph)
    q = np.exp(1j * ph)
    out = np.empty_like(q)
    for r in range(nlam):
        s = slice(r * rows, (r + 1) * rows)
        out[s] = q[s] * (np.linalg.norm(v[s]) / max(np.sqrt(rows), 1e-12))
    return out


def qstack_rs(M, nph, nlam):
    return np.stack([qvec_rs(M[:, k], nph, nlam)
                     for k in range(M.shape[1])], 1)


def feats_w(gray, depth, K, wmode):
    """gridint-3D landmark cells with selectable weights (banked law:
    intensity = place weights, gradmag = ego-motion weights — the gyro
    and verify ran on intensity so far)."""
    import experiments.detzoo as DZ_
    ys, xs, w_int = __import__("experiments.twomap", fromlist=["x"]
                               ).grid_int(gray)
    if wmode == "gradmag":
        dx, dy = DZ_._sobel(gray)
        gm = np.zeros(gray.shape)
        gm[1:-1, 1:-1] = np.abs(dx) + np.abs(dy)
        cs = 8
        h, wd = gray.shape[0] // cs * cs, gray.shape[1] // cs * cs
        cw = gm[:h, :wd].reshape(h // cs, cs, wd // cs, cs).mean((1, 3))
        w = cw.ravel() + 1.0
    else:
        w = w_int
    Ki = np.linalg.inv(K)
    uv = np.stack([xs + 0.0, ys + 0.0, np.ones(len(xs))])
    b = (Ki @ uv).T
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    d = depth[np.asarray(ys, int), np.asarray(xs, int)] / 1000.0
    ok = (d > 0.2) & (d < 8.0)
    return (b[ok] * d[ok, None]), w[ok]


LADDERS_V = {
    "lam.35-2.8 (v1)": [0.35, 0.7, 1.4, 2.8],
    "lam.25-2.0     ": [0.25, 0.5, 1.0, 2.0],
    "lam.5-4.0      ": [0.5, 1.0, 2.0, 4.0],
    "lam.35-5.6(x6) ": [0.35, 0.7, 1.4, 2.8, 5.6, 11.2],
}


def _W_of_lams(lams):
    return np.concatenate([(2 * np.pi / lam)
                           * L3.dirs_azel(12, [-40, -20, 0, 20, 40])
                           for lam in lams])


def bench_dtune(seq=SEQ):
    gray, depth, gt, K = V6.load(seq)
    pairs = small_pairs(gt)
    print(f"vision 6-DoF space tuning ({seq.split('_freiburg')[-1]}, "
          f"{len(pairs)} small-motion pairs; rot deg / transl m):")
    print("  -- ladder x weights (float store, linear solve) --")
    best = None
    for lname, lams in LADDERS_V.items():
        W = _W_of_lams(lams)
        for wmode in ("int", "gradmag"):
            er, et = [], []
            for i, j, Rt, tt in pairs:
                P0, w0 = feats_w(gray[i], depth[i], K, wmode)
                P1, w1 = feats_w(gray[j], depth[j], K, wmode)
                v0 = V6._enc_raw(W, P0, w0)
                v1 = V6._enc_raw(W, P1, w1)
                Dm = np.concatenate([V6._deriv_axes(W, P0, w0),
                                     V6._deriv_transl(W, P0, w0)], 1)
                th = solve66(v0, Dm, v1)
                er.append(gyro_err(th, Rt))
                et.append(np.linalg.norm(th[3:6] - tt))
            key = np.median(er)
            print(f"  {lname} {wmode:7s}: rot med {key:.2f} p90 "
                  f"{np.percentile(er, 90):.2f} | transl med "
                  f"{np.median(et)*1000:.1f} mm", flush=True)
            if best is None or key < best[0]:
                best = (key, lname, lams, wmode)
    _, lname, lams, wmode = best
    print(f"  -- store model at the winner ({lname.strip()}, {wmode}) --")
    W = _W_of_lams(lams)
    nlam = len(lams)
    F = {}
    for i, j, Rt, tt in pairs:
        P0, w0 = feats_w(gray[i], depth[i], K, wmode)
        P1, w1 = feats_w(gray[j], depth[j], K, wmode)
        F[i] = (V6._enc_raw(W, P0, w0),
                np.concatenate([V6._deriv_axes(W, P0, w0),
                                V6._deriv_transl(W, P0, w0)], 1),
                V6._enc_raw(W, P1, w1))
    for tag, qv, qm in (
            ("float      ", lambda v: v, lambda M: M),
            ("2b flat-mag", lambda v: qvec(v, 4), lambda M: qstack(M, 4)),
            ("2b ring-mag", lambda v: qvec_rs(v, 4, nlam),
             lambda M: qstack_rs(M, 4, nlam)),
            ("3b ring-mag", lambda v: qvec_rs(v, 8, nlam),
             lambda M: qstack_rs(M, 8, nlam))):
        er, et = [], []
        for i, j, Rt, tt in pairs:
            v0, Dm, v1 = F[i]
            th = solve66(qv(v0), qm(Dm), v1)
            er.append(gyro_err(th, Rt))
            et.append(np.linalg.norm(th[3:6] - tt))
        print(f"  {tag}: rot med {np.median(er):.2f} p90 "
              f"{np.percentile(er, 90):.2f} | transl med "
              f"{np.median(et)*1000:.1f} mm", flush=True)


def _gn_stacked(chans, v1s, R0, t0, weights, iters=2):
    """Two-space GN: block-stacked residuals, ONE 6x6 solve per iter.
    chans = [(W, P0, w0)], v1s = fresh queries per channel; spaces stay
    distinct (no vector mixing) — fusion happens in the solver."""
    R, tt = R0.copy(), t0.copy()
    for _ in range(iters):
        As, bs = [], []
        for (W, P0, w0), v1, wt in zip(chans, v1s, weights):
            Pc = P0 @ R.T + tt
            v0c = V6._enc_raw(W, Pc, w0)
            Dm = np.concatenate([V6._deriv_axes(W, Pc, w0),
                                 V6._deriv_transl(W, Pc, w0)], 1)
            s = wt / max(np.linalg.norm(v1), 1e-9)
            As.append(np.concatenate([Dm.real, Dm.imag]) * s)
            bs.append(np.concatenate([(v1 - v0c).real,
                                      (v1 - v0c).imag]) * s)
        th, *_ = np.linalg.lstsq(np.concatenate(As),
                                 np.concatenate(bs), rcond=None)
        dR = L3._axis_rot("yaw", np.degrees(th[0]))             @ L3._axis_rot("pitch", np.degrees(th[1]))             @ L3._axis_rot("roll", np.degrees(th[2]))
        R = dR @ R
        tt = tt + th[3:6]
    return R, tt


def bench_fuse2(seq=SEQ, max_pairs=80):
    gray, depth, gt, K = V6.load(seq)
    Wl = L3.make_lattices()["azel3d"]
    Wv = W_vis3d()
    pairs = small_pairs(gt, max_pairs)
    e_post, e_stack, t_post, t_stack = [], [], [], []
    for i, j, Rt, tt in pairs:
        Fv0 = V6.feats(gray[i], K, "gridint", depth[i])
        Fv1 = V6.feats(gray[j], K, "gridint", depth[j])
        Pc0, wc0 = L6.depth_cloud(depth[i], K)
        Pc1, wc1 = L6.depth_cloud(depth[j], K)
        w_true = rotvec(Rt)
        v1v = V6._enc_raw(Wv, *Fv1)
        v1c = V6._enc_raw(Wl, Pc1, wc1)
        # post-hoc (the banked 1.08): per-channel GN then omega average
        Rv, tv = L6._gn_iterate(Wv, Fv0[0], Fv0[1], v1v, np.eye(3),
                                np.zeros(3), iters=2)
        Rc, tc = L6._gn_iterate(Wl, Pc0, wc0, v1c, np.eye(3),
                                np.zeros(3), iters=2)
        pv, pc = 1 / 1.82 ** 2, 1 / 1.34 ** 2
        wf = (pv * rotvec(Rv) + pc * rotvec(Rc)) / (pv + pc)
        tf = (pv * tv + pc * tc) / (pv + pc)
        e_post.append(np.degrees(np.linalg.norm(wf - w_true)))
        t_post.append(np.linalg.norm(tf - tt))
        # stacked: one solve over both spaces
        Rs, ts = _gn_stacked(
            [(Wv, Fv0[0], Fv0[1]), (Wl, Pc0, wc0)], [v1v, v1c],
            np.eye(3), np.zeros(3), [1 / 1.82, 1 / 1.34])
        e_stack.append(np.degrees(np.linalg.norm(rotvec(Rs) - w_true)))
        t_stack.append(np.linalg.norm(ts - tt))
    print(f"two-space fusion in the solver ({seq.split('_freiburg')[-1]},"
          f" {len(pairs)} pairs):")
    for tag, e, tm in (("post-hoc omega avg (banked)", e_post, t_post),
                       ("block-stacked GN (one solve)", e_stack,
                        t_stack)):
        print(f"  {tag}: rot med {np.median(e):.2f} p90 "
              f"{np.percentile(e, 90):.2f} | transl med "
              f"{np.median(tm)*1000:.1f} mm p90 "
              f"{np.percentile(tm, 90)*1000:.1f}", flush=True)


def bench_chain(seq=SEQ, max_windows=60):
    """Deploy-rate check: chain the stacked-GN gyro over 15 Hz steps
    (stride-2 parse) across one 5 Hz keyframe interval (3 steps) and
    compare against the single-shot 5 Hz decode over the same window."""
    import numpy as _np
    from pathlib import Path as _P
    f = _P(__file__).resolve().parents[1] / "data" / "tum"         / f"{seq}_s2.npz"
    z = _np.load(f)
    gray, depth, gt = z["gray"], z["depth_mm"], z["gt"]
    K = z["K"]
    Wl = L3.make_lattices()["azel3d"]
    Wv = W_vis3d()

    def step(i, j, R0, t0):
        Fv0 = V6.feats(gray[i], K, "gridint", depth[i])
        Fv1 = V6.feats(gray[j], K, "gridint", depth[j])
        Pc0, wc0 = L6.depth_cloud(depth[i], K)
        Pc1, wc1 = L6.depth_cloud(depth[j], K)
        return _gn_stacked(
            [(Wv, Fv0[0], Fv0[1]), (Wl, Pc0, wc0)],
            [V6._enc_raw(Wv, *Fv1), V6._enc_raw(Wl, Pc1, wc1)],
            R0, t0, [1 / 1.82, 1 / 1.34])

    e_chain, e_shot, t_chain, t_shot = [], [], [], []
    n = 0
    for k0 in range(0, len(gray) - 3, 5):
        if n >= max_windows:
            break
        Rt, tt = V6._rel_pose(gt[k0], gt[k0 + 3])
        ang = np.degrees(np.arccos(np.clip((np.trace(Rt) - 1) / 2,
                                           -1, 1)))
        if not (0.5 <= ang <= 9.0) or np.linalg.norm(tt) > 0.18:
            continue
        # chained 15 Hz: compose three per-step estimates
        Rc_, tc_ = np.eye(3), np.zeros(3)
        for s in range(3):
            Rs, ts = step(k0 + s, k0 + s + 1, np.eye(3), np.zeros(3))
            Rc_, tc_ = Rs @ Rc_, Rs @ tc_ + ts
        # single-shot 5 Hz over the whole window
        Rs1, ts1 = step(k0, k0 + 3, np.eye(3), np.zeros(3))
        e_chain.append(np.degrees(np.linalg.norm(rotvec(Rc_ @ Rt.T))))
        e_shot.append(np.degrees(np.linalg.norm(rotvec(Rs1 @ Rt.T))))
        t_chain.append(np.linalg.norm(tc_ - tt))
        t_shot.append(np.linalg.norm(ts1 - tt))
        n += 1
    print(f"deploy-rate chaining ({seq.split('_freiburg')[-1]} @15 Hz, "
          f"{n} keyframe windows of 3 steps):")
    for tag, e, tm in (("chained 3x15 Hz", e_chain, t_chain),
                       ("single-shot 5 Hz", e_shot, t_shot)):
        print(f"  {tag}: rot med {np.median(e):.2f} p90 "
              f"{np.percentile(e, 90):.2f} | transl med "
              f"{np.median(tm)*1000:.1f} mm", flush=True)


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
         hopf=bench_hopf, dtune=bench_dtune, fuse2=bench_fuse2,
         chain=bench_chain)[cmd](seq)
