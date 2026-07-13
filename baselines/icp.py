"""Baseline 1/3: classic point-to-line ICP scan matching (PLICP flavor, Censi
2008) with scan-to-scan odometry refinement + keyframe pose graph with ICP
loop closures.

Usage: python3 -m baselines.icp [data/intel.log] [n_keyframes]

Shared harness (identical across the three baselines, reused from
ssp_slam_carmen): parse_flaser, keyframing (0.10 m / 5 deg), compose_guess,
align_se2, VALID_MAX=40, beam angles deg2rad(-90 + arange(n)*(180/n)),
ATE vs <log>.gfs.log with a 0.3 s timestamp gate.

Design:
- Front end: per keyframe, point-to-line ICP (scipy cKDTree NN, <=20
  Gauss-Newton iterations, robust trim at 2.5 sigma via MAD) against the
  PREVIOUS keyframe's scan, initialized from the odometry-composed relative
  guess -> sequential pose-graph edges. Poor fits fall back to raw odometry
  (weaker edge weight).
- Loop closures: for keyframes within 5 m (current estimate) and > 300
  keyframes apart, ICP against the candidate keyframe's scan seeded with the
  estimated relative pose; accepted on convergence + inlier ratio + residual
  thresholds (plus a basin gate on the correction size).
- Back end: full pose graph over all keyframes, scipy least_squares (TRF,
  analytic sparse Jacobian, soft_l1) with SE(2) residuals in the style of
  ssp_bounded / ssp_slam_loop; re-solved (throttled) on accepted closures and
  once at the end; solution applied to all poses.

This baseline KEEPS every keyframe scan (points + normals) — that is its
nature; the stored-state memory is reported honestly at the end. KD-trees are
built on demand per registration (transient, not stored).

Deterministic: no randomness anywhere in this pipeline.
"""

import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

import sspslam.encoder as S
import sspslam.frontend as C

VALID_MAX = 40.0

# --- ICP parameters ---------------------------------------------------------
ICP_ITERS = 20
D_MAX = 1.0                      # correspondence distance gate [m]
TRIM_NSIG = 2.5                  # robust trim (MAD sigma units)
NORMAL_MAX_GAP = 0.5             # max chord to a beam neighbor for a valid normal [m]
CONV_EPS = 1e-4                  # GN step norm considered converged

# sequential-edge acceptance (deviation from odometry guess) — like the
# innovation gate in ssp_slam_carmen
SEQ_GATE_T, SEQ_GATE_R = 0.45, np.deg2rad(11.0)
SEQ_MIN_INLIER = 0.4

# --- loop-closure parameters -------------------------------------------------
LC_RADIUS = 5.0                  # candidate distance in current estimate [m]
LC_MIN_SEP = 300                 # min keyframe separation
LC_MIN_INLIER = 0.65             # inlier ratio (matched+kept / valid points)
LC_MAX_RESID = 0.08              # mean |point-to-line residual| over inliers [m]
LC_GATE_T, LC_GATE_R = 4.0, np.deg2rad(30.0)   # basin gate on ICP correction
SOLVE_EVERY = 30                 # min keyframes between graph re-solves

# --- edge weights (1/sigma, whitened residuals) ------------------------------
W_SEQ_T, W_SEQ_R = 1 / 0.03, 1 / np.deg2rad(0.5)
W_ODO_T, W_ODO_R = 1 / 0.15, 1 / np.deg2rad(3.0)
W_LC_T, W_LC_R = 1 / 0.05, 1 / np.deg2rad(1.0)


# ---------------------------------------------------------------------------
# SE(2) helpers
# ---------------------------------------------------------------------------

def se2_mul(A, B):
    return np.array([*(A[:2] + S._rot(A[2]) @ B[:2]), S.wrap(A[2] + B[2])])


def se2_inv(A):
    return np.array([*(-(S._rot(-A[2]) @ A[:2])), -S.wrap(A[2])])


# ---------------------------------------------------------------------------
# Scan -> points + normals (robot frame, beam order preserved)
# ---------------------------------------------------------------------------

def scan_points_normals(r, beam):
    """Valid points and per-point unit normals from a raw scan.

    Normals come from the chord between the two beam-adjacent valid points
    (PLICP's local line); points at depth discontinuities (either neighbor
    chord > NORMAL_MAX_GAP) or scan ends get an invalid (NaN) normal and are
    excluded from point-to-line correspondences.
    """
    valid = (r > 0.05) & (r < VALID_MAX)
    pts = np.stack([r[valid] * np.cos(beam[valid]),
                    r[valid] * np.sin(beam[valid])], 1)
    n = len(pts)
    normals = np.full((n, 2), np.nan)
    if n >= 3:
        fwd = np.linalg.norm(pts[1:] - pts[:-1], axis=1)      # chord i -> i+1
        ok = (fwd[:-1] <= NORMAL_MAX_GAP) & (fwd[1:] <= NORMAL_MAX_GAP)
        tang = pts[2:] - pts[:-2]
        nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
        ln = np.linalg.norm(nrm, axis=1)
        ok &= ln > 1e-9
        normals[1:-1][ok] = nrm[ok] / ln[ok, None]
    return pts, normals


# ---------------------------------------------------------------------------
# Point-to-line ICP (Gauss-Newton, robust trim)
# ---------------------------------------------------------------------------

def icp_p2l(ref_pts, ref_normals, ref_tree, cur_pts, guess, iters=ICP_ITERS):
    """Register cur_pts onto the reference scan; T maps cur frame -> ref frame.

    Returns (T, inlier_ratio, mean_abs_resid, converged)."""
    T = np.array(guess, float)
    inl, resid, converged = 0.0, np.inf, False
    if len(cur_pts) < 20 or len(ref_pts) < 20:
        return T, 0.0, np.inf, False
    for _ in range(iters):
        q = cur_pts @ S._rot(T[2]).T + T[:2]
        dist, idx = ref_tree.query(q)
        nrm = ref_normals[idx]
        keep = (dist < D_MAX) & np.isfinite(nrm[:, 0])
        if keep.sum() < 10:
            return T, 0.0, np.inf, False
        e = np.einsum('ij,ij->i', nrm[keep], q[keep] - ref_pts[idx[keep]])
        med = np.median(e)
        sig = max(1.4826 * np.median(np.abs(e - med)), 0.02)
        inlier = np.abs(e - med) < TRIM_NSIG * sig
        if inlier.sum() < 10:
            return T, 0.0, np.inf, False
        ki = np.flatnonzero(keep)[inlier]
        nn, ee, qq = nrm[ki], np.einsum(
            'ij,ij->i', nrm[ki], q[ki] - ref_pts[idx[ki]]), q[ki] - T[:2]
        # rows: [n_x, n_y, n . d(R p)/dth];  d(R p)/dth = perp(R p)
        A = np.stack([nn[:, 0], nn[:, 1],
                      nn[:, 0] * (-qq[:, 1]) + nn[:, 1] * qq[:, 0]], 1)
        delta, *_ = np.linalg.lstsq(A, -ee, rcond=None)  # min-norm: degenerate
        T[:2] += delta[:2]                               # directions stay at guess
        T[2] = S.wrap(T[2] + delta[2])
        inl = len(ki) / len(cur_pts)
        resid = np.abs(ee).mean()
        if np.linalg.norm(delta[:2]) < CONV_EPS and abs(delta[2]) < CONV_EPS:
            converged = True
            break
    return T, inl, resid, converged


# ---------------------------------------------------------------------------
# Pose graph: SE(2) residuals, analytic sparse Jacobian, node 0 anchored
# ---------------------------------------------------------------------------

class PoseGraph:
    def __init__(self):
        self.edges = []                       # (a, b, Z, wt, wr)

    def add(self, a, b, Z, wt, wr):
        self.edges.append((a, b, np.asarray(Z, float), wt, wr))

    def optimize(self, P0):
        n = len(P0)
        E = [e for e in self.edges if e[0] < n and e[1] < n]
        m = len(E)
        a_idx = np.array([e[0] for e in E])
        b_idx = np.array([e[1] for e in E])
        Z = np.stack([e[2] for e in E])
        wt = np.array([e[3] for e in E])
        wr = np.array([e[4] for e in E])

        def unpack(x):
            return np.vstack([P0[0], x.reshape(-1, 3)])

        def resid(x):
            P = unpack(x)
            Pa, Pb = P[a_idx], P[b_idx]
            ca, sa = np.cos(Pa[:, 2]), np.sin(Pa[:, 2])
            v = Pb[:, :2] - Pa[:, :2]
            dx = ca * v[:, 0] + sa * v[:, 1]
            dy = -sa * v[:, 0] + ca * v[:, 1]
            out = np.empty(3 * m)
            out[0::3] = (dx - Z[:, 0]) * wt
            out[1::3] = (dy - Z[:, 1]) * wt
            out[2::3] = S.wrap(Pb[:, 2] - Pa[:, 2] - Z[:, 2]) * wr
            return out

        def jac(x):
            P = unpack(x)
            Pa, Pb = P[a_idx], P[b_idx]
            ca, sa = np.cos(Pa[:, 2]), np.sin(Pa[:, 2])
            v = Pb[:, :2] - Pa[:, :2]
            rows, cols, vals = [], [], []

            def put(r, c, val):
                rows.append(r), cols.append(c), vals.append(val)

            for e in range(m):
                a, b = a_idx[e], b_idx[e]
                # d(dx,dy)/d(tb) = R(-tha); d/d(ta) = -R(-tha)
                Rm = np.array([[ca[e], sa[e]], [-sa[e], ca[e]]])
                for i in range(2):
                    for j in range(2):
                        if b > 0:
                            put(3 * e + i, 3 * (b - 1) + j, Rm[i, j] * wt[e])
                        if a > 0:
                            put(3 * e + i, 3 * (a - 1) + j, -Rm[i, j] * wt[e])
                if a > 0:
                    # d(dx)/dtha = -sa*vx + ca*vy = dy ; d(dy)/dtha = -dx
                    dxe = ca[e] * v[e, 0] + sa[e] * v[e, 1]
                    dye = -sa[e] * v[e, 0] + ca[e] * v[e, 1]
                    put(3 * e, 3 * (a - 1) + 2, dye * wt[e])
                    put(3 * e + 1, 3 * (a - 1) + 2, -dxe * wt[e])
                    put(3 * e + 2, 3 * (a - 1) + 2, -wr[e])
                if b > 0:
                    put(3 * e + 2, 3 * (b - 1) + 2, wr[e])
            return csr_matrix((vals, (rows, cols)), shape=(3 * m, 3 * (n - 1)))

        sol = least_squares(resid, P0[1:].ravel(), jac=jac, method="trf",
                            loss="soft_l1", f_scale=2.0, max_nfev=60)
        P = unpack(sol.x)
        P[:, 2] = S.wrap(P[:, 2])
        return P


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/intel.log"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9

    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)[:limit]
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    print(f"{len(scans)} scans -> {n} keyframes, {n_beams} beams")

    # stored per-keyframe state (this baseline keeps ALL keyframe scans)
    kf_pts, kf_normals = [], []
    est = np.zeros((n, 3))
    graph = PoseGraph()
    n_fallback = n_lc_try = n_lc_acc = 0
    last_solve, dirty = -10 ** 9, False

    t0 = time.time()
    for k in range(n):
        pts, normals = scan_points_normals(keys[k][0], beam)
        kf_pts.append(pts)
        kf_normals.append(normals)
        if k == 0:
            est[0] = odom[0]
            continue

        # --- sequential edge: ICP against previous keyframe ------------------
        Z_odo = se2_mul(se2_inv(odom[k - 1]), odom[k])
        ref_tree = cKDTree(kf_pts[k - 1])
        Z, inl, resid, conv = icp_p2l(kf_pts[k - 1], kf_normals[k - 1],
                                      ref_tree, pts, Z_odo)
        ok = (inl > SEQ_MIN_INLIER
              and np.linalg.norm(Z[:2] - Z_odo[:2]) < SEQ_GATE_T
              and abs(S.wrap(Z[2] - Z_odo[2])) < SEQ_GATE_R)
        if ok:
            graph.add(k - 1, k, Z, W_SEQ_T, W_SEQ_R)
        else:
            Z = Z_odo
            graph.add(k - 1, k, Z, W_ODO_T, W_ODO_R)
            n_fallback += 1
        est[k] = se2_mul(est[k - 1], Z)

        # --- loop closure attempt --------------------------------------------
        if k >= LC_MIN_SEP:
            d = np.linalg.norm(est[:k - LC_MIN_SEP + 1, :2] - est[k, :2], axis=1)
            j = int(np.argmin(d))
            if d[j] < LC_RADIUS:
                n_lc_try += 1
                Zg = se2_mul(se2_inv(est[j]), est[k])
                cand_tree = cKDTree(kf_pts[j])
                Zl, inl, resid, conv = icp_p2l(kf_pts[j], kf_normals[j],
                                               cand_tree, pts, Zg)
                if (conv and inl > LC_MIN_INLIER and resid < LC_MAX_RESID
                        and np.linalg.norm(Zl[:2] - Zg[:2]) < LC_GATE_T
                        and abs(S.wrap(Zl[2] - Zg[2])) < LC_GATE_R):
                    graph.add(j, k, Zl, W_LC_T, W_LC_R)
                    n_lc_acc += 1
                    dirty = True
        if dirty and k - last_solve >= SOLVE_EVERY:
            est[:k + 1] = graph.optimize(est[:k + 1])
            last_solve, dirty = k, False

        if k % 500 == 0:
            print(f"  kf {k}/{n}  closures {n_lc_acc}/{n_lc_try}  "
                  f"t={time.time() - t0:.0f}s")

    if dirty:
        est = graph.optimize(est)
    dt = time.time() - t0
    ms_kf = dt / n * 1e3
    print(f"done: {dt:.1f}s ({ms_kf:.1f} ms/kf, CPU shared)  "
          f"odometry fallbacks: {n_fallback} ({100 * n_fallback / n:.1f}%)  "
          f"loop closures: {n_lc_acc} accepted / {n_lc_try} attempted")

    # honest stored-state memory: all keyframe scan arrays + trajectory state
    scan_bytes = sum(p.nbytes for p in kf_pts) + sum(m.nbytes for m in kf_normals)
    state_bytes = scan_bytes + est.nbytes + odom.nbytes + kts.nbytes
    print(f"stored state: {state_bytes / 1e6:.1f} MB "
          f"({scan_bytes / 1e6:.1f} MB keyframe scans+normals across {n} kf; "
          f"KD-trees built on demand, transient)")

    stem = os.path.basename(path).replace(".log", "")
    np.savez(f"{stem}_traj_icp.npz", est=est, odom=odom, kts=kts)
    print(f"wrote {stem}_traj_icp.npz")

    # --- eval: ATE vs corrected log (identical to ssp_slam_carmen) -----------
    try:
        ref = C.parse_flaser(path.replace(".log", ".gfs.log"))
        rts = np.array([t for _, _, t in ref])
        rxy = np.stack([p[:2] for _, p, _ in ref])
        j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
        good = np.abs(rts - kts[j]) < 0.3
        if good.sum() < 50:
            print(f"eval skipped: only {good.sum()} timestamp matches "
                  "(corrupt/mismatched reference timestamps, MIT-style)")
            return
        aligned = C.align_se2(est[j[good], :2], rxy[good])
        err = np.linalg.norm(aligned - rxy[good], axis=1)
        o_al = C.align_se2(odom[j[good], :2], rxy[good])
        o_err = np.linalg.norm(o_al - rxy[good], axis=1)
        print(f"ATE vs corrected log over {good.sum()} matched poses: "
              f"rmse {np.sqrt((err ** 2).mean()):.3f} m   "
              f"median {np.median(err):.3f} m   max {err.max():.3f} m")
        print(f"raw odometry ATE (same poses): "
              f"rmse {np.sqrt((o_err ** 2).mean()):.3f} m   "
              f"median {np.median(o_err):.3f} m")
    except FileNotFoundError:
        print("no corrected log found for reference")


if __name__ == "__main__":
    main()
