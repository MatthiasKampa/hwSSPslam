"""Baseline 2/3: correlative scan matching (CSM) on occupancy grids.

Olson (ICRA'09) style coarse-to-fine correlative matching against a rolling
local likelihood grid, plus Cartographer-style submaps + wide-window
correlative loop closure + sparse SE(2) pose-graph relaxation.

Fairness rules shared with the other baselines (see ssp_slam_carmen.py):
- parse_flaser / keyframes (0.10 m / 5 deg) / compose_guess / align_se2 reused
- VALID_MAX = 40 m no-return threshold
- beams = deg2rad(-90 + arange(n) * (180 / n))
- eval = ATE vs .gfs.log, timestamp match gated at 0.3 s (skipped gracefully
  if the reference timestamps are corrupt, e.g. the MIT log)
- deterministic (no RNG anywhere), saves <log>_traj_csm.npz

Design:
- Rolling local map: binary hit grid at 5 cm from the endpoints of the last
  100 keyframes (within 10 m of the guess), Gaussian-blurred (sigma 1.5
  cells), normalized so a point on a blurred wall line scores ~1, stored
  uint8. A 2x max-pooled 10 cm level sits on top (pooled = upper bound).
- Per keyframe: two-level correlative search around the odometry-composed
  guess: 10 cm cells x 1 deg over +-0.5 m x +-10 deg, then 5 cm x 0.25 deg
  over +-0.15 m x +-1.25 deg. All scoring is vectorized numpy gather-sums.
- Submaps: every 50 keyframes the block's endpoints are frozen into a grid
  in the frame of the block's first keyframe (levels 5/10/40 cm). On
  relaxation only the origin keyframe pose moves; grids stay fixed
  (Cartographer-style rigid submap move).
- Loop closures: for keyframes > 300 kf past a submap and within 5 m of it,
  a wide correlative search (+-6 m x +-30 deg at 40 cm/2 deg, top-5 NMS
  candidates refined at 10 cm then 5 cm). Accept on absolute fine score
  (max-pool upper bound prunes hopeless candidates early). Each accepted
  closure adds a loop edge and triggers a full sparse pose-graph relax
  (scipy least_squares TRF, analytic sparse Jacobian, residual form as in
  ssp_bounded.py, Huber-robustified).

Usage: python3 baseline_csm.py [data/intel.log] [n_keyframes]
"""

import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

import ssp_slam as S
from ssp_slam_carmen import (parse_flaser, keyframes, compose_guess,
                             align_se2, VALID_MAX)
from ssp_slam_loop import se2_mul, se2_inv

# ---- grid ------------------------------------------------------------------
RES = 0.05                       # fine cell size [m]
BLUR_SIG = 1.5                   # blur sigma [cells]
NORM = 1.0 / (np.sqrt(2 * np.pi) * BLUR_SIG)  # blurred straight-wall peak
LOCAL_N = 100                    # keyframes in the rolling local map
LOCAL_R = 10.0                   # ignore keyframes farther than this [m]
LOCAL_EXT = 25.0                 # hard cap on local grid half-extent [m]

# ---- local (frame-to-map) search -------------------------------------------
WIN_T, WIN_R = 0.5, 10.0         # coarse window: +-0.5 m x +-10 deg
COARSE_STEP_R = 1.0              # deg, on the 10 cm level
FINE_CELLS, FINE_R, FINE_STEP_R = 3, 1.25, 0.25  # +-0.15 m x +-1.25 deg @5 cm
MIN_PTS = 20
MIN_SCORE = 0.15                 # below this, fall back to odometry
GATE_T, GATE_R = 0.40, np.deg2rad(7.0)  # innovation gate vs guess (safety net)
N_SCORE_MAX = 200                # decimate scans beyond this for scoring
# Scans on rotation-triggered keyframes (in-place pivots; keyframing fires at
# 5 deg) are rotationally motion-distorted (SICK sweep ~180 ms vs spins up to
# ~100 deg/s on intel). Inserting them poisons the map and starts a
# self-reinforcing rotational slide, so they are matched but never inserted
# into local maps or submaps (standard no-IMU distortion mitigation).
INSERT_ROT_MAX = np.deg2rad(4.5)

# ---- submaps + loop closure --------------------------------------------------
SUBMAP_LEN = 50                  # keyframes per submap
LC_GAP = 300                     # min keyframe separation
LC_RADIUS = 5.0                  # max distance to submap center [m]
LC_WIN_T, LC_WIN_R = 6.0, 30.0   # wide window: +-6 m x +-30 deg
LC_COARSE_STEP_R = 2.0           # deg, on the 40 cm level
LC_SCORE = 0.65                  # accept threshold on the fine (5 cm) score
LC_RATIO = 0.92                  # reject if a far runner-up peak scores this
LC_STRIDE = 3                    # attempt every 3rd keyframe
LC_COOLDOWN = 3                  # keyframes to wait after an acceptance
LC_SAME_SM = 40                  # min kf between commits to the same submap
LC_TOPK = 5                      # NMS candidates refined per attempt
LC_PAIR_KF = 24                  # pairwise verification: max kf between hits
LC_PAIR_T, LC_PAIR_R = 0.30, np.deg2rad(2.0)  # chain-consistency tolerance

# ---- pose graph edge sigmas ---------------------------------------------------
SIG_SEQ = (0.05, np.deg2rad(1.0))    # matched sequential edge
SIG_ODO = (0.15, np.deg2rad(3.0))    # odometry-fallback sequential edge
SIG_LOOP = (0.10, np.deg2rad(1.5))   # accepted loop edge
PRUNE_SIG = 5.0                      # drop loop edges > this whitened residual


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

def rasterize(pts, lo, shape):
    """Binary hit cells -> Gaussian blur -> clip at wall-line peak -> uint8."""
    ij = np.floor((pts - lo) / RES).astype(np.int64)
    ok = ((ij[:, 0] >= 0) & (ij[:, 0] < shape[0])
          & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1]))
    ij = ij[ok]
    g = np.zeros(shape[0] * shape[1], np.float32)
    g[np.unique(ij[:, 0] * shape[1] + ij[:, 1])] = 1.0
    g = gaussian_filter(g.reshape(shape), BLUR_SIG)
    np.clip(g / NORM, 0.0, 1.0, out=g)
    return (g * 255.0 + 0.5).astype(np.uint8)


def maxpool(g, f):
    """f x f max-pool (same origin; pooled cell res = f * RES; upper bound)."""
    nx, ny = g.shape
    g = np.pad(g, ((0, (-nx) % f), (0, (-ny) % f)))
    return g.reshape(g.shape[0] // f, f, g.shape[1] // f, f).max(axis=(1, 3))


def score_shifts(grid, pg, ox, oy):
    """Sum of grid values at floor(pg) + (ox, oy) per shift (out-of-grid = 0).

    grid: (nx, ny) uint8; pg: (N, 2) point coords in cell units;
    ox, oy: (Sh,) integer cell shifts. Returns (Sh,) integer scores."""
    c = np.floor(pg).astype(np.int64)
    IX = c[:, 0][None, :] + ox[:, None]
    IY = c[:, 1][None, :] + oy[:, None]
    inb = ((IX >= 0) & (IX < grid.shape[0]) & (IY >= 0) & (IY < grid.shape[1]))
    v = grid[np.where(inb, IX, 0), np.where(inb, IY, 0)]
    return (v * inb).sum(1)


def search(grid, lo, res, pts, center, angles, ncell):
    """Correlative search: for each absolute angle, all integer-cell shifts of
    the guess translation within +-ncell. Returns (best_pose, best_score01).

    Integer cell shifts are exact: floor((p + k*res - lo)/res) =
    floor((p - lo)/res) + k."""
    n = np.arange(-ncell, ncell + 1)
    ox, oy = np.repeat(n, len(n)), np.tile(n, len(n))
    best_s, best_p = -1, None
    for th in angles:
        pg = (pts @ S._rot(th).T + center[:2] - lo) / res
        sc = score_shifts(grid, pg, ox, oy)
        i = int(sc.argmax())
        if sc[i] > best_s:
            best_s = sc[i]
            best_p = np.array([center[0] + ox[i] * res,
                               center[1] + oy[i] * res, th])
    return best_p, best_s / (255.0 * len(pts))


def search_topk(grid, lo, res, pts, center, angles, ncell, k, min_score):
    """Wide search returning up to k NMS-separated candidates (score01, pose).

    Candidates below min_score are pruned: the max-pooled grid upper-bounds
    the fine grid, so they cannot pass the fine acceptance threshold."""
    n = np.arange(-ncell, ncell + 1)
    ox, oy = np.repeat(n, len(n)), np.tile(n, len(n))
    denom = 255.0 * len(pts)
    all_s, all_p = [], []
    for th in angles:
        pg = (pts @ S._rot(th).T + center[:2] - lo) / res
        sc = score_shifts(grid, pg, ox, oy) / denom
        keep = sc >= min_score
        if keep.any():
            all_s.append(sc[keep])
            po = np.empty((int(keep.sum()), 3))
            po[:, 0] = center[0] + ox[keep] * res
            po[:, 1] = center[1] + oy[keep] * res
            po[:, 2] = th
            all_p.append(po)
    if not all_s:
        return []
    sc = np.concatenate(all_s)
    po = np.concatenate(all_p)
    order = np.argsort(-sc)
    out = []
    for i in order:
        if any(np.linalg.norm(po[i, :2] - q[1][:2]) < 3 * res
               and abs(S.wrap(po[i, 2] - q[1][2])) < np.deg2rad(6.0)
               for q in out):
            continue
        out.append((sc[i], po[i]))
        if len(out) == k:
            break
    return out


# ---------------------------------------------------------------------------
# Pose graph relax (residual form as in ssp_bounded._gn, Huber-robustified)
# ---------------------------------------------------------------------------

def _solve(P0, edges, max_nfev=50):
    """Sparse SE(2) pose-graph solve, node 0 fixed. edges: (a, b, Z, wt, wr, kind).
    Residual per edge: [R(tha)^T (pb - pa) - Zt] * wt, wrap(thb-tha-Zth) * wr.
    Analytic sparse Jacobian (12 nonzeros/edge); TRF, linear loss (robustness
    comes from explicit loop-edge pruning in relax_graph, not from a robust
    loss — Huber/Cauchy also downweight TRUE closures with large drift)."""
    N, E = len(P0), len(edges)
    free = np.arange(1, N)
    if E == 0 or len(free) == 0:
        return P0.copy(), np.zeros((E, 3))
    aa = np.array([e[0] for e in edges])
    bb = np.array([e[1] for e in edges])
    Zt = np.stack([e[2] for e in edges])
    wts = np.array([e[3] for e in edges])
    wrs = np.array([e[4] for e in edges])

    def resid(x):
        P = P0.copy()
        P[free] = x.reshape(-1, 3)
        ca, sa = np.cos(P[aa, 2]), np.sin(P[aa, 2])
        dx, dy = P[bb, 0] - P[aa, 0], P[bb, 1] - P[aa, 1]
        out = np.empty((E, 3))
        out[:, 0] = (ca * dx + sa * dy - Zt[:, 0]) * wts
        out[:, 1] = (-sa * dx + ca * dy - Zt[:, 1]) * wts
        out[:, 2] = S.wrap(P[bb, 2] - P[aa, 2] - Zt[:, 2]) * wrs
        return out.ravel()

    # sparsity: rows r0,r1 x cols (xa,ya,tha,xb,yb); row r2 x cols (tha,thb)
    ro = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
    comp = np.array([0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 2])
    isb = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1], bool)
    col_of = np.full(N, -1)
    col_of[free] = np.arange(len(free))
    node = np.where(isb, bb[:, None], aa[:, None])
    nc = col_of[node]
    keep = (nc >= 0).ravel()
    rows = (3 * np.arange(E)[:, None] + ro).ravel()[keep]
    cols = (3 * nc + comp).ravel()[keep]

    def jac(x):
        P = P0.copy()
        P[free] = x.reshape(-1, 3)
        ca, sa = np.cos(P[aa, 2]), np.sin(P[aa, 2])
        dx, dy = P[bb, 0] - P[aa, 0], P[bb, 1] - P[aa, 1]
        V = np.empty((E, 12))
        V[:, 0], V[:, 1], V[:, 2] = -ca, -sa, -sa * dx + ca * dy
        V[:, 3], V[:, 4] = ca, sa
        V[:, 5], V[:, 6], V[:, 7] = sa, -ca, -(ca * dx + sa * dy)
        V[:, 8], V[:, 9] = -sa, ca
        V[:, :10] *= wts[:, None]
        V[:, 10], V[:, 11] = -wrs, wrs
        return csr_matrix((V.ravel()[keep], (rows, cols)),
                          shape=(3 * E, 3 * len(free)))

    sol = least_squares(resid, P0[free].ravel(), jac=jac, method="trf",
                        x_scale="jac", max_nfev=max_nfev)
    P = P0.copy()
    P[free] = sol.x.reshape(-1, 3)
    P[:, 2] = S.wrap(P[:, 2])
    return P, sol.fun.reshape(E, 3)


def relax_graph(P0, edges, max_nfev=50):
    """Solve; then prune loop edges whose whitened residual norm exceeds
    PRUNE_SIG (false closures the solve could not absorb) and re-solve.
    Up to 3 rounds. Returns (P, surviving_edges, n_pruned)."""
    n_pruned = 0
    for _ in range(3):
        P, R = _solve(P0, edges, max_nfev)
        rn = np.linalg.norm(R, axis=1)
        bad = [i for i, e in enumerate(edges)
               if e[5] == "loop" and rn[i] > PRUNE_SIG]
        if not bad:
            return P, edges, n_pruned
        # drop only the worst offender per round: one false closure can
        # inflate residuals of nearby true ones until it is removed
        worst = max(bad, key=lambda i: rn[i])
        edges = [e for i, e in enumerate(edges) if i != worst]
        n_pruned += 1
    P, _ = _solve(P0, edges, max_nfev)
    return P, edges, n_pruned


# ---------------------------------------------------------------------------
# Submaps
# ---------------------------------------------------------------------------

class Submap:
    """Frozen grid in the frame of keyframe k0 (levels 5 / 10 / 40 cm).
    Global placement at query time = current est[k0] (rigid submap move)."""

    __slots__ = ("k0", "kend", "lo", "g0", "g1", "g2", "last_lc")

    def __init__(self, k0, kend, est, pts_list, insert_ok):
        self.k0, self.kend, self.last_lc = k0, kend, -10 ** 9
        inv0 = se2_inv(est[k0])
        P = []
        for j in range(k0, kend):
            if not len(pts_list[j]) or not insert_ok[j]:
                continue
            rel = se2_mul(inv0, est[j])
            P.append(pts_list[j] @ S._rot(rel[2]).T + rel[:2])
        P = np.concatenate(P) if P else np.zeros((0, 2))
        margin = 4 * BLUR_SIG * RES + 2 * RES
        self.lo = P.min(0) - margin if len(P) else np.zeros(2)
        hi = (P.max(0) + margin) if len(P) else np.ones(2)
        shape = np.ceil((hi - self.lo) / RES).astype(int) + 1
        self.g0 = rasterize(P, self.lo, tuple(shape))
        self.g1 = maxpool(self.g0, 2)
        self.g2 = maxpool(self.g0, 8)

    @property
    def nbytes(self):
        return self.g0.nbytes + self.g1.nbytes + self.g2.nbytes


def try_loop_close(sm, pts, est_k0, est_k):
    """Wide correlative search of scan pts (robot frame) against submap sm.
    Returns (score01, Z rel pose k0->k) or None."""
    Zg = se2_mul(se2_inv(est_k0), est_k)
    a2 = Zg[2] + np.deg2rad(np.arange(-LC_WIN_R, LC_WIN_R + 1e-9,
                                      LC_COARSE_STEP_R))
    cands = search_topk(sm.g2, sm.lo, 8 * RES, pts, Zg, a2,
                        int(round(LC_WIN_T / (8 * RES))), LC_TOPK, LC_SCORE)
    refined = []
    for _, cp in cands:
        a1 = cp[2] + np.deg2rad(np.arange(-2.0, 2.001, 0.5))
        p1, _ = search(sm.g1, sm.lo, 2 * RES, pts, cp, a1, 3)
        a0 = p1[2] + np.deg2rad(np.arange(-0.5, 0.501, 0.25))
        p0, s0 = search(sm.g0, sm.lo, RES, pts, p1, a0, 2)
        refined.append((s0, p0))
    if not refined:
        return None
    refined.sort(key=lambda q: -q[0])
    s0, p0 = refined[0]
    if s0 < LC_SCORE:
        return None
    # peak-ambiguity test: a distinct far peak nearly as good => reject
    for s1, p1 in refined[1:]:
        far = (np.linalg.norm(p1[:2] - p0[:2]) > 1.0
               or abs(S.wrap(p1[2] - p0[2])) > np.deg2rad(8.0))
        if far and s1 >= LC_RATIO * s0:
            return None
    return s0, p0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/intel.log"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    stem = os.path.splitext(os.path.basename(path))[0]

    scans = parse_flaser(path)
    keys = keyframes(scans)[:limit]
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    print(f"{len(scans)} scans -> {len(keys)} keyframes, {n_beams} beams")
    print(f"CSM: {RES * 100:.0f} cm grid, blur {BLUR_SIG} cells, "
          f"local map {LOCAL_N} kf; window +-{WIN_T} m x +-{WIN_R} deg; "
          f"loop: +-{LC_WIN_T} m x +-{LC_WIN_R} deg vs {SUBMAP_LEN}-kf "
          f"submaps, gap>{LC_GAP} kf, r<{LC_RADIUS} m, accept>={LC_SCORE}")

    K = len(keys)
    est = np.zeros((K, 3))
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    pts_list = []                    # per-kf robot-frame endpoints (float32)
    insert_ok = []                   # False = rotation-distorted, match-only
    edges = []                       # (a, b, Z, wt, wr)
    submaps = []
    n_fallback = n_lc_try = n_lc_hit = n_lc_acc = n_lc_pruned = n_relax = 0
    pend = {}                        # submap idx -> (k, Z) unconfirmed hit
    t_match = t_lc = t_relax = 0.0
    local_grid_bytes_max = 0
    last_lc_accept = -10 ** 9

    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.asarray(r, float)
        valid = (rr > 0.05) & (rr < VALID_MAX)
        pts = np.stack([rr[valid] * np.cos(beam[valid]),
                        rr[valid] * np.sin(beam[valid])], 1).astype(np.float32)
        pts_list.append(pts)
        insert_ok.append(k == 0 or abs(
            S.wrap(odom[k, 2] - odom[k - 1, 2])) <= INSERT_ROT_MAX)
        spts = pts[::max(1, int(np.ceil(len(pts) / N_SCORE_MAX)))].astype(float)

        if k == 0:
            est[0] = opose
        else:
            guess = compose_guess(est[k - 1], odom[k - 1], odom[k])
            guess[2] = S.wrap(guess[2])
            est[k] = guess
            sig = SIG_ODO
            tm = time.time()
            # rolling local likelihood grid from recent keyframes
            j0 = max(0, k - LOCAL_N)
            sel = [j for j in range(j0, k) if len(pts_list[j]) and
                   insert_ok[j] and
                   np.linalg.norm(est[j, :2] - guess[:2]) < LOCAL_R]
            if sel and len(spts) >= MIN_PTS:
                P = np.concatenate(
                    [pts_list[j] @ S._rot(est[j, 2]).T + est[j, :2]
                     for j in sel])
                margin = WIN_T + 4 * BLUR_SIG * RES + 2 * RES
                lo = np.maximum(P.min(0), guess[:2] - LOCAL_EXT) - margin
                hi = np.minimum(P.max(0), guess[:2] + LOCAL_EXT) + margin
                shape = np.ceil((hi - lo) / RES).astype(int) + 1
                g0 = rasterize(P, lo, tuple(shape))
                g1 = maxpool(g0, 2)
                local_grid_bytes_max = max(local_grid_bytes_max,
                                           g0.nbytes + g1.nbytes)
                ac = guess[2] + np.deg2rad(
                    np.arange(-WIN_R, WIN_R + 1e-9, COARSE_STEP_R))
                pc, _ = search(g1, lo, 2 * RES, spts, guess, ac,
                               int(round(WIN_T / (2 * RES))))
                af = pc[2] + np.deg2rad(
                    np.arange(-FINE_R, FINE_R + 1e-9, FINE_STEP_R))
                pf, sf = search(g0, lo, RES, spts, pc, af, FINE_CELLS)
                pf[2] = S.wrap(pf[2])
                if (sf >= MIN_SCORE
                        and np.linalg.norm(pf[:2] - guess[:2]) < GATE_T
                        and abs(S.wrap(pf[2] - guess[2])) < GATE_R):
                    est[k] = pf
                    sig = SIG_SEQ
                else:
                    n_fallback += 1
            else:
                n_fallback += 1
            t_match += time.time() - tm
            Z = se2_mul(se2_inv(est[k - 1]), est[k])
            edges.append((k - 1, k, Z, 1 / sig[0], 1 / sig[1], "seq"))

        # freeze a submap when its block completes
        if k + 1 >= SUBMAP_LEN and (k + 1) % SUBMAP_LEN == 0:
            submaps.append(Submap(k + 1 - SUBMAP_LEN, k + 1, est, pts_list,
                                  insert_ok))

        # wide-window loop closure against old submaps. A single hit is only
        # PENDING; it is committed when a second hit to the same submap from
        # a nearby keyframe agrees with the first through the local est
        # chain (Olson-style pairwise consistency — single-scan correlative
        # peaks in self-similar clutter are not trustworthy on their own).
        if (k % LC_STRIDE == 0 and k - last_lc_accept > LC_COOLDOWN
                and len(spts) >= MIN_PTS):
            tm = time.time()
            cands = [(i, sm) for i, sm in enumerate(submaps)
                     if k - sm.kend > LC_GAP and k - sm.last_lc > LC_SAME_SM]
            if cands:
                d = np.linalg.norm(
                    np.stack([est[(sm.k0 + sm.kend) // 2, :2]
                              for _, sm in cands]) - est[k, :2], axis=1)
                for i in np.argsort(d)[:2]:
                    if d[i] > LC_RADIUS:
                        break
                    smi, sm = cands[i]
                    n_lc_try += 1
                    hit = try_loop_close(sm, spts, est[sm.k0], est[k])
                    if hit is None:
                        continue
                    s0, Z = hit
                    n_lc_hit += 1
                    prev = pend.get(smi)
                    if prev is not None and k - prev[0] <= LC_PAIR_KF:
                        k1, Z1 = prev
                        D = se2_mul(se2_inv(est[k1]), est[k])
                        e = se2_mul(se2_inv(se2_mul(Z1, D)), Z)
                        if (np.linalg.norm(e[:2]) < LC_PAIR_T
                                and abs(S.wrap(e[2])) < LC_PAIR_R):
                            edges.append((sm.k0, k1, Z1, 1 / SIG_LOOP[0],
                                          1 / SIG_LOOP[1], "loop"))
                            edges.append((sm.k0, k, Z, 1 / SIG_LOOP[0],
                                          1 / SIG_LOOP[1], "loop"))
                            sm.last_lc = k
                            n_lc_acc += 2
                            last_lc_accept = k
                            del pend[smi]
                            t_lc += time.time() - tm
                            tm = time.time()
                            est[:k + 1], edges, npr = relax_graph(
                                est[:k + 1], edges)
                            n_lc_pruned += npr
                            n_relax += 1
                            t_relax += time.time() - tm
                            tm = time.time()
                            break
                    pend[smi] = (k, Z)
            t_lc += time.time() - tm

        if k % 500 == 0:
            print(f"  frame {k}/{K}  t={time.time() - t0:.0f}s  "
                  f"loops {n_lc_acc}/{n_lc_try}")

    # final relax
    tm = time.time()
    if n_lc_acc:
        est, edges, npr = relax_graph(est, edges)
        n_lc_pruned += npr
        n_relax += 1
    t_relax += time.time() - tm
    dt = time.time() - t0
    print(f"done: {dt:.0f}s ({dt / K * 1e3:.1f} ms/keyframe)   "
          f"match {t_match / K * 1e3:.1f} + loop-search {t_lc / K * 1e3:.1f} "
          f"+ relax {t_relax / K * 1e3:.1f} ms/kf")
    print(f"odometry fallbacks: {n_fallback} ({100 * n_fallback / K:.1f}%)   "
          f"loop edges: {n_lc_acc} committed ({n_lc_hit} hits / {n_lc_try} "
          f"attempts, pair-verified), {n_lc_pruned} pruned in relax   "
          f"relaxes: {n_relax}")

    # honest memory report: everything the algorithm keeps around
    sm_bytes = sum(sm.nbytes for sm in submaps)
    ep_bytes = sum(p.nbytes for p in pts_list)
    pose_bytes = est.nbytes + odom.nbytes + len(edges) * (3 * 8 + 2 * 8 + 16)
    total = sm_bytes + ep_bytes + pose_bytes + local_grid_bytes_max
    print(f"memory: submap grids {sm_bytes / 1e6:.1f} MB ({len(submaps)} "
          f"submaps) + endpoints {ep_bytes / 1e6:.1f} MB + local grid (peak) "
          f"{local_grid_bytes_max / 1e6:.1f} MB + poses/edges "
          f"{pose_bytes / 1e6:.1f} MB = {total / 1e6:.1f} MB")

    np.savez(f"{stem}_traj_csm.npz", est=est, odom=odom, kts=kts)
    print(f"wrote {stem}_traj_csm.npz")

    # reference comparison (corrected log, matched by timestamp)
    ref_err, good = None, None
    try:
        ref = parse_flaser(path.replace(".log", ".gfs.log"))
        rts = np.array([t for _, _, t in ref])
        rxy = np.stack([p[:2] for _, p, _ in ref])
        j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
        good = np.abs(rts - kts[j]) < 0.3
        if good.sum() < 10:
            print(f"reference timestamps unusable ({good.sum()} matches "
                  f"within 0.3 s) — skipping ATE")
            ref_err = None
        else:
            aligned = align_se2(est[j[good], :2], rxy[good])
            ref_err = np.linalg.norm(aligned - rxy[good], axis=1)
            print(f"ATE vs corrected log over {good.sum()} matched poses: "
                  f"rmse {np.sqrt((ref_err ** 2).mean()):.3f} m   "
                  f"median {np.median(ref_err):.3f} m   "
                  f"max {ref_err.max():.3f} m")
    except FileNotFoundError:
        print("no corrected log found for reference")
    except Exception as e:
        print(f"reference eval skipped ({e})")

    # map + trajectory figure
    cl = np.concatenate([pts_list[k] @ S._rot(est[k, 2]).T + est[k, :2]
                         for k in range(K) if len(pts_list[k])])
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    axes[0].plot(odom[:, 0], odom[:, 1], "0.6", lw=0.8)
    axes[0].set_title("odometry dead reckoning (drift)")
    axes[0].set_aspect("equal")
    axes[1].scatter(cl[::4, 0], cl[::4, 1], s=0.2, c="k", alpha=0.25)
    axes[1].plot(est[:, 0], est[:, 1], "r-", lw=0.6, alpha=0.8)
    axes[1].set_title(f"CSM map + trajectory ({n_lc_acc} loop closures)")
    axes[1].set_aspect("equal")
    if ref_err is not None:
        axes[2].plot(np.flatnonzero(good), ref_err)
        axes[2].set_xlabel("matched reference pose #")
        axes[2].set_ylabel("position error vs corrected log [m]")
        axes[2].grid(alpha=0.3)
        axes[2].set_title("ATE vs RBPF-corrected reference")
    fig.suptitle(f"Correlative scan matching baseline on {path}")
    fig.tight_layout()
    fig.savefig(f"{stem}_csm.png", dpi=110)
    print(f"wrote {stem}_csm.png")


if __name__ == "__main__":
    main()
