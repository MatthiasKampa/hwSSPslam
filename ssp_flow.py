"""Continual GRADIENT-FLOW closure for SSP SLAM (detection-free paradigm).

This is a prototype of an alternative loop-closure backend that REPLACES the
shipped detect + verify + snap + relax pipeline with a single continual
gradient flow on an energy

    E = lam_seq * sum_consecutive || g_{i+1} (-) (g_i o z_i) ||^2      (STIFF)
      - lam_ov  * sum_{(i,j) spatially-near, |i-j| large} rho( cos<P_i, P_j> )

subject to a fixed anchor (g_0, or an entire reference sub-map) for gauge.

  * The STIFF sequential skeleton is the ordinary between-anchor pose-graph
    residual (identical algebra to BoundedSLAM._gn). It holds the trajectory
    rigid and prevents collapse.
  * The SOFT overlap attraction is applied over ALL spatially-near, temporally-
    distant anchor pairs with NO detection / verification / gate. Its force is
    the ANALYTIC SSP correlation gradient:
        translation via the phase gradient   (i W .)
        rotation    via the stored d/dtheta derivative vector (segder)
    rho is a bounded, cosine-scaled robust kernel so that wrong pairs (whose
    cosine is near zero) exert ~no force and the many genuine pairs outvote the
    few aliased ones -- never a committed decision.

The flow is run continually (a few preconditioned heavy-ball GD iterations
every few frames, or to convergence offline). It reuses the SSP lattice / ops
from ssp_slam_loop and the session build / place() algebra from
ssp_multisession; it SUBCLASSES BoundedSLAM only to reuse the frontend + the
anchor-frame segment-vector store, disabling the discrete closure machinery.

NOTHING in any shipped/prior module is edited. GT is used ONLY to score the
final ATE and to draw convergence curves; it never enters the flow, and the
multi-session B initialization is a coarse hand-offset, NOT the shared-index
oracle and NOT GT.

Usage:
    python3 ssp_flow.py selftest          # gradient FD checks + sanity
    python3 ssp_flow.py single intel      # test 1 (also fr079, fr101)
    python3 ssp_flow.py mit                # test 2 (graceful degradation)
    python3 ssp_flow.py multi             # test 3 (multi-session convergence)
"""

import os
import sys
import time
import pickle

import numpy as np
from scipy.spatial import cKDTree

import ssp_slam as S
import ssp_slam_loop as L
import ssp_slam_carmen as C
import ssp_multisession as M
from ssp_bounded import BoundedSLAM, ANCHOR

VALID_MAX = 40.0
D = L.W.shape[0]                       # 360 = 6 rings x 60 angles
RING_OF = np.repeat(np.arange(L.N_RING), L.N_ANG)     # ring index per dim
WR = L.W                                              # (D,2) real frequencies

# ring schedules for coarse-to-fine annealing (lam = 0.25,0.5,1,2,5.3,12.8)
RINGS_COARSE = RING_OF >= 3            # lam 2, 5.3, 12.8  -> widest basins
RINGS_MID    = RING_OF >= 2            # lam 1, 2, 5.3, 12.8
RINGS_FINE   = RING_OF >= 0            # all rings         -> cm registration
SCRATCH = M.SCRATCH


# ==========================================================================
# 0.  Frontend map build with the discrete closure machinery DISABLED
# ==========================================================================
class FlowSLAM(BoundedSLAM):
    """BoundedSLAM with detection + internal relax turned OFF. We keep only the
    recency scan-matching frontend and the anchor-frame segment-vector store;
    global consistency comes entirely from the external gradient flow."""

    def try_constraint(self, pts, w):
        return                        # NO loop detection

    def relax(self):
        self.dirty = False            # NO discrete pose-graph relax
        return


def build_map(path, limit=None, store_dtype=np.complex64, verbose=True,
              keep_kf=False):
    """Run the recency frontend over a CARMEN log, closure DISABLED. Returns a
    dict with anchor poses, anchor-frame segvec/segder, sequential edges, plus
    the ground-truth-scoring bookkeeping (kf times, per-kf pose reconstruction).
    """
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)
    if limit:
        keys = keys[:limit]
    n = len(keys)
    nb = len(keys[0][0]); beam = np.deg2rad(-90.0 + np.arange(nb) * (180.0 / nb))
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    slam = FlowSLAM(robust=True, recent_aids=12)
    slam.store_dtype = store_dtype
    est = np.zeros((n, 3))
    t0 = time.time()
    for k in range(n):
        r = keys[k][0]
        rr = np.where(r < VALID_MAX, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = odom[0] if k == 0 else C.compose_guess(est[k - 1], odom[k - 1], odom[k])
        est[k] = slam.add_keyframe(pts, w, guess)
        if verbose and k % 1000 == 0:
            print(f"    build {k}/{n}  t={time.time()-t0:.0f}s", flush=True)
    aids = sorted(slam.segvec)
    # sequential edges among the anchors (from the frontend seq skeleton)
    seq = [(a, b, Z, wt, wr) for a, b, Z, wt, wr, kind in slam.edges if kind == "seq"]
    data = dict(
        anchors={a: slam.anchors[a].copy() for a in aids},
        segvec={a: slam.segvec[a].astype(complex) for a in aids},
        segder={a: slam.segder[a].astype(complex) for a in aids},
        seq=seq, kf_ref=list(slam.kf_ref), kts=kts, odom=odom,
        n=n, build_s=time.time() - t0, aids=aids,
        anchors_all={a: slam.anchors[a].copy() for a in range(len(slam.anchors))})
    if keep_kf:
        # per-kf (final est pose, robot-frame pts, weights) for soft rebuild
        est_fin = np.stack([slam.pose_of(k) for k in range(n)])
        kfp = []
        for k in range(n):
            rr = np.where(keys[k][0] < VALID_MAX, keys[k][0], np.inf)
            pts, w, _ = S.scan_to_samples(rr, beam)
            kfp.append((est_fin[k], pts, w, slam.aid_of(k)))
        data["kf"] = kfp
    return data


# ==========================================================================
# 0b.  SOFT / partition-of-unity feature distribution (user's enhancement)
# ==========================================================================
def _encode_anchor_frame(pts, w, anchor_pose, kf_pose):
    """Encode a keyframe's robot-frame points in a target anchor's frame, plus
    the d/dtheta derivative about that anchor origin (== the store's algebra)."""
    rel = L.se2_mul(L.se2_inv(anchor_pose), kf_pose)     # kf pose in anchor frame
    PA = pts @ S._rot(rel[2]).T + rel[:2]
    Amat = np.exp(1j * (PA @ L.W.T))
    v = Amat.T @ w
    Cx = PA[:, 0:1] * L.W[:, 1] - PA[:, 1:2] * L.W[:, 0]
    vd = (1j * Cx * Amat).T @ w
    return v, vd


def soft_consolidate(anchor_ids, anchor_poses, kf_list, mode="hard", Mn=3,
                     h_scale=1.0):
    """Rebuild per-anchor segvec/segder from raw keyframe scans, distributing
    each keyframe across neighbouring anchor CENTERS with partition-of-unity
    weights (weights sum to 1 -> total map mass preserved).

      mode="hard"   : weight 1 to the keyframe's own (nearest) anchor.
      mode="kernel" : Mn nearest centers, Gaussian weights (bilinear-like).
      mode="bary"   : Delaunay of centers; 3 barycentric weights of the
                      containing triangle (hex-like, isotropic P1 partition).

    anchor_poses: dict aid->pose(3). kf_list: [(kf_pose, pts, w, own_aid)].
    Returns (segvec, segder) dicts.
    """
    centers = np.array([anchor_poses[a][:2] for a in anchor_ids])
    id_of = {a: i for i, a in enumerate(anchor_ids)}
    tree = cKDTree(centers)
    tri = None
    if mode == "bary":
        from scipy.spatial import Delaunay
        try:
            tri = Delaunay(centers)
        except Exception:
            tri = None
    D = L.W.shape[0]
    segvec = {a: np.zeros(D, complex) for a in anchor_ids}
    segder = {a: np.zeros(D, complex) for a in anchor_ids}
    mass = 0.0
    for (kf_pose, pts, w, own_aid) in kf_list:
        if len(pts) == 0:
            continue
        p = kf_pose[:2]
        if mode == "hard":
            assign = [(own_aid if own_aid in id_of else
                       anchor_ids[tree.query(p)[1]], 1.0)]
        elif mode == "kernel":
            dd, ii = tree.query(p, k=min(Mn, len(anchor_ids)))
            dd = np.atleast_1d(dd); ii = np.atleast_1d(ii)
            h = h_scale * (dd.mean() + 1e-6)
            wts = np.exp(-(dd ** 2) / (2 * h ** 2))
            wts /= wts.sum()
            assign = [(anchor_ids[ii[t]], float(wts[t])) for t in range(len(ii))]
        else:  # bary
            assign = None
            if tri is not None:
                s = int(tri.find_simplex(p))
                if s >= 0:
                    verts = tri.simplices[s]
                    T = tri.transform[s, :2]
                    r = tri.transform[s, 2]
                    bc = T @ (p - r)
                    bc = np.append(bc, 1 - bc.sum())
                    assign = [(anchor_ids[verts[t]], float(bc[t]))
                              for t in range(3)]
            if assign is None:                      # outside hull -> nearest
                assign = [(anchor_ids[tree.query(p)[1]], 1.0)]
        for a, wt in assign:
            if wt == 0:
                continue
            v, vd = _encode_anchor_frame(pts, w, anchor_poses[a], kf_pose)
            segvec[a] += wt * v
            segder[a] += wt * vd
            mass += wt
    return segvec, segder


# ==========================================================================
# 1.  Continual gradient-flow field over anchor poses
# ==========================================================================
class FlowField:
    """Holds a set of anchor poses (some fixed for gauge) with their stored
    anchor-frame segment vectors and a sequential skeleton, and runs the
    continual overlap-attraction + stiff-skeleton gradient flow.

    Poses are stored as three parallel arrays x, y, th (length A). `free` marks
    which anchors may move. `sess` and `order` support multi-session gating
    (never pair sequential neighbours of the SAME session; those are handled by
    the seq term).
    """

    def __init__(self, pos, th, segvec, segder, seq_edges, free, sess, order,
                 lam_seq=1.0, lam_ov=1.0, gate_r=5.0, sep=40,
                 s0=0.05, s1=0.30):
        self.x = pos[:, 0].astype(float).copy()
        self.y = pos[:, 1].astype(float).copy()
        self.th = th.astype(float).copy()
        self.A = len(self.x)
        self.segvec = np.asarray(segvec, complex)          # (A, D)
        self.segder = np.asarray(segder, complex)          # (A, D)
        self.free = np.asarray(free, bool)
        self.sess = np.asarray(sess, int)
        self.order = np.asarray(order, int)
        self.lam_seq, self.lam_ov = lam_seq, lam_ov
        self.gate_r, self.sep = gate_r, sep
        self.s0, self.s1 = s0, s1
        self.grad_norm = True        # per-patch gradient-L2 preconditioner
        # sequential edges as arrays
        self.ea = np.array([e[0] for e in seq_edges], int)
        self.eb = np.array([e[1] for e in seq_edges], int)
        self.eZ = np.array([e[2] for e in seq_edges], float) if seq_edges \
            else np.zeros((0, 3))
        self.ewt = np.array([e[3] for e in seq_edges], float)
        self.ewr = np.array([e[4] for e in seq_edges], float)
        self.rmask = RINGS_FINE
        self.pairs = None
        self._vx = np.zeros(self.A)    # momentum buffers
        self._vy = np.zeros(self.A)
        self._vt = np.zeros(self.A)

    # ---- overlap pair set (spatial gating; recomputed as poses drift) ----
    def build_pairs(self):
        pos = np.stack([self.x, self.y], 1)
        tree = cKDTree(pos)
        raw = tree.query_pairs(self.gate_r, output_type="ndarray")
        if len(raw) == 0:
            self.pairs = (np.zeros(0, int), np.zeros(0, int))
            return 0
        i, j = raw[:, 0], raw[:, 1]
        # keep a pair iff (different sessions) OR (same session but temporally
        # distant): |order_i - order_j| > sep. NEVER pair sequential neighbours.
        same = self.sess[i] == self.sess[j]
        far = np.abs(self.order[i] - self.order[j]) > self.sep
        keep = (~same) | far
        # at least one endpoint must be free (else the pair can exert no force)
        keep &= self.free[i] | self.free[j]
        self.pairs = (i[keep], j[keep])
        return int(keep.sum())

    # ---- rotate the stored anchor-frame vectors to the current heading ----
    def _rotated(self, mask):
        """u[a] = rot(segvec_a, th_a) and up[a] = d u / d th_a, restricted to
        the active ring mask. Uses the exact index-permutation rotation + the
        stored derivative vector for the sub-grid residual (== place() algebra).
        """
        A = self.A
        idx = np.flatnonzero(mask)
        Dm = len(idx)
        u = np.empty((A, Dm), complex)
        up = np.empty((A, Dm), complex)
        for a in range(A):
            th = self.th[a]
            m = int(round(th * L.N_ANG / np.pi))
            delta = th - m * np.pi / L.N_ANG
            base = L.rot_permute(self.segvec[a], m)
            der = L.rot_permute(self.segder[a], m)
            u[a] = (base + delta * der)[idx]
            up[a] = der[idx]                    # d u / d th  (1st order)
        # Stored patch vectors are kept RAW (un-normalized): normalizing phi
        # would discard the SSP magnitude/density signal and break the additive
        # composition phi(P1 u P2)=phi(P1)+phi(P2) the map/merge relies on. The
        # normalization is applied to the GRADIENT instead (see step(): the
        # per-pair force is divided by |u_i||u_j| -> a per-patch preconditioner
        # that equalizes step size across content mass, without touching the
        # objective or the vectors). nrm carries the raw norms for that scaling.
        nrm = np.linalg.norm(u, axis=1) + 1e-12
        return u, up, nrm, idx

    # ---- one preconditioned heavy-ball gradient-flow step ----
    def step(self, lr=0.4, mom=0.85, ret_diag=False):
        gx = np.zeros(self.A); gy = np.zeros(self.A); gth = np.zeros(self.A)
        Hx = np.full(self.A, 1e-6); Hy = np.full(self.A, 1e-6)
        Hth = np.full(self.A, 1e-6)

        # ---------- STIFF sequential skeleton (E_seq gradient + GN diag) -----
        if len(self.ea):
            ea, eb = self.ea, self.eb
            ca, sa = np.cos(self.th[ea]), np.sin(self.th[ea])
            dx = self.x[eb] - self.x[ea]
            dy = self.y[eb] - self.y[ea]
            wt, wr = self.ewt, self.ewr
            r0 = ca * dx + sa * dy - self.eZ[:, 0]
            r1 = -sa * dx + ca * dy - self.eZ[:, 1]
            r2 = S.wrap(self.th[eb] - self.th[ea] - self.eZ[:, 2])
            f0, f1, f2 = wt * r0, wt * r1, wr * r2
            ls = self.lam_seq
            # whitened jacobian entries
            Jxa0, Jya0, Jtha0 = -ca * wt, -sa * wt, (-sa * dx + ca * dy) * wt
            Jxa1, Jya1, Jtha1 = sa * wt, -ca * wt, -(ca * dx + sa * dy) * wt
            Jxb0, Jyb0 = ca * wt, sa * wt
            Jxb1, Jyb1 = -sa * wt, ca * wt
            Jt2a, Jt2b = -wr, wr
            np.add.at(gx, ea, ls * (Jxa0 * f0 + Jxa1 * f1))
            np.add.at(gy, ea, ls * (Jya0 * f0 + Jya1 * f1))
            np.add.at(gth, ea, ls * (Jtha0 * f0 + Jtha1 * f1 + Jt2a * f2))
            np.add.at(gx, eb, ls * (Jxb0 * f0 + Jxb1 * f1))
            np.add.at(gy, eb, ls * (Jyb0 * f0 + Jyb1 * f1))
            np.add.at(gth, eb, ls * (Jt2b * f2))
            np.add.at(Hx, ea, ls * (Jxa0**2 + Jxa1**2))
            np.add.at(Hy, ea, ls * (Jya0**2 + Jya1**2))
            np.add.at(Hth, ea, ls * (Jtha0**2 + Jtha1**2 + Jt2a**2))
            np.add.at(Hx, eb, ls * (Jxb0**2 + Jxb1**2))
            np.add.at(Hy, eb, ls * (Jyb0**2 + Jyb1**2))
            np.add.at(Hth, eb, ls * Jt2b**2)

        # ---------- SOFT overlap attraction (analytic SSP corr gradient) -----
        s_stats = None
        if self.pairs is None:
            self.build_pairs()
        I, J = self.pairs
        if len(I):
            u, up, nrm, idx = self._rotated(self.rmask)
            Wa = WR[idx]                                  # (Dm,2)
            dt = np.stack([self.x[J] - self.x[I], self.y[J] - self.y[I]], 1)
            phase = np.exp(1j * (dt @ Wa.T))              # (P, Dm)
            cij = np.conj(u[I]) * u[J]                    # (P, Dm)
            s = (phase * cij).real.sum(1)                 # raw correlation
            n = nrm[I] * nrm[J]
            shat = s / n                                  # cosine in [-1,1]
            # bounded cosine-scaled robust kernel: influence rho'(shat) is a
            # clipped linear ramp -> 0 below s0 (wrong/far pairs silenced),
            # saturates at 1 above s1 (no single pair dominates the sum).
            infl = np.clip((shat - self.s0) / (self.s1 - self.s0), 0.0, 1.0)
            # GRADIENT-L2 normalization (per-patch preconditioner): dividing the
            # raw inner-product force by |u_i||u_j| makes a dense (high-mass)
            # patch take a comparable step to a sparse one, so one lr works and
            # the s0.30-ok/s0.45-diverges lr-fragility goes away. grad_norm=False
            # is the raw-force baseline (mass-weighted, ill-conditioned).
            a = self.lam_ov * infl / n if self.grad_norm else self.lam_ov * infl
            # translation gradient: d s / d(dt) = Re( sum (i W) phase cij )
            g = phase * cij
            vec = g @ Wa                                  # (P,2) complex
            ds_dt = -vec.imag                             # Re(i vec) = -Im(vec)
            fx = a * ds_dt[:, 0]
            fy = a * ds_dt[:, 1]
            np.add.at(gx, J, -fx); np.add.at(gy, J, -fy)  # force = -dE/dp = +a ds
            np.add.at(gx, I, fx);  np.add.at(gy, I, fy)   # (grad = -force)
            # rotation gradient via the stored derivative vector
            dthi = (phase * (np.conj(up[I]) * u[J])).real.sum(1)
            dthj = (phase * (np.conj(u[I]) * up[J])).real.sum(1)
            np.add.at(gth, I, -a * dthi)
            np.add.at(gth, J, -a * dthj)
            # crude overlap Hessian floor so translation is well-scaled
            hov = self.lam_ov * infl * (nrm[I] * nrm[J]) * 0.0 + self.lam_ov * infl
            np.add.at(Hx, I, hov); np.add.at(Hx, J, hov)
            np.add.at(Hy, I, hov); np.add.at(Hy, J, hov)
            s_stats = (shat, infl)

        # ---------- preconditioned heavy-ball update on FREE anchors ----------
        fr = self.free
        sx = -gx / Hx; sy = -gy / Hy; sth = -gth / Hth   # -grad / diag(H)
        self._vx = mom * self._vx + sx
        self._vy = mom * self._vy + sy
        self._vt = mom * self._vt + sth
        self.x[fr] += lr * self._vx[fr]
        self.y[fr] += lr * self._vy[fr]
        self.th[fr] = S.wrap(self.th[fr] + lr * self._vt[fr])
        if ret_diag:
            return s_stats
        return None

    # ---- diagnostics ----
    def traj_length(self):
        """Total path length of the ordered sequential skeleton (collapse
        guard). Sums |g_b - g_a| over consecutive same-session anchors."""
        pos = np.stack([self.x, self.y], 1)
        d = 0.0
        for a, b in zip(self.ea, self.eb):
            d += np.linalg.norm(pos[b] - pos[a])
        return d

    def poses(self):
        return np.stack([self.x, self.y, self.th], 1)


# ==========================================================================
# 2.  Assemble a FlowField from a single-session built map
# ==========================================================================
def field_from_map(data, lam_seq=1.0, lam_ov=1.0, anchor0=True, **kw):
    aids = sorted(data["anchors"])
    remap = {a: i for i, a in enumerate(aids)}
    pos = np.array([data["anchors"][a][:2] for a in aids])
    th = np.array([data["anchors"][a][2] for a in aids])
    segvec = np.array([data["segvec"][a] for a in aids])
    segder = np.array([data["segder"][a] for a in aids])
    seq = [(remap[a], remap[b], Z, wt, wr) for a, b, Z, wt, wr in data["seq"]
           if a in remap and b in remap]
    free = np.ones(len(aids), bool)
    if anchor0:
        free[0] = False                       # fix g_0 for gauge
    sess = np.zeros(len(aids), int)
    order = np.arange(len(aids))
    ff = FlowField(pos, th, segvec, segder, seq, free, sess, order,
                   lam_seq=lam_seq, lam_ov=lam_ov, **kw)
    return ff, aids, remap


def poses_to_kf(data, ff, aids, remap):
    """Reconstruct per-keyframe poses from flowed anchor poses (segments ride
    rigidly on their anchor)."""
    P = ff.poses()
    anchors_new = {a: P[remap[a]] for a in aids}
    n = data["n"]
    out = np.zeros((n, 3))
    for k, (aid, rel) in enumerate(data["kf_ref"]):
        base = anchors_new.get(aid)
        if base is None:                       # anchor has no segment (no flow)
            base = data["anchors_all"][aid]    # hold at its frontend pose
        out[k] = L.se2_mul(base, rel)
    return out


# ==========================================================================
# 3.  GT scoring (GT touches ONLY the score, never the flow)
# ==========================================================================
def ate_vs_gfs(path, kts, est):
    try:
        ref = C.parse_flaser(path.replace(".log", ".gfs.log"))
    except FileNotFoundError:
        return None
    rts = np.array([t for _, _, t in ref])
    rxy = np.stack([p[:2] for _, p, _ in ref])
    j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
    good = np.abs(rts - kts[j]) < 0.3
    if good.sum() < 3:
        return dict(rmse=float("nan"), median=float("nan"),
                    max=float("nan"), n=int(good.sum()))
    al = C.align_se2(est[j[good], :2], rxy[good])
    e = np.linalg.norm(al - rxy[good], axis=1)
    return dict(rmse=float(np.sqrt((e**2).mean())), median=float(np.median(e)),
                max=float(e.max()), n=int(good.sum()))


# ==========================================================================
# selftest: analytic overlap + sequential gradients vs finite differences
# ==========================================================================
def selftest():
    rng = np.random.default_rng(0)
    A = 6
    segvec = rng.normal(size=(A, D)) + 1j * rng.normal(size=(A, D))
    segder = rng.normal(size=(A, D)) + 1j * rng.normal(size=(A, D))
    pos = rng.uniform(-1, 1, (A, 2))
    th = rng.uniform(-0.5, 0.5, A)
    seq = [(i, i + 1, rng.normal(0, 0.1, 3), 1 / 0.03, 1 / np.deg2rad(0.3))
           for i in range(A - 1)]
    free = np.ones(A, bool)
    ff = FlowField(pos, th, segvec, segder, seq, free, np.zeros(A, int),
                   np.arange(A), lam_seq=1.0, lam_ov=1.0, gate_r=99.0, sep=-1,
                   s0=-9.0, s1=9.0)   # s0<0,s1 wide -> infl==1 (smooth energy)
    ff.build_pairs()

    def energy(x, y, t):
        # E = lam_seq/2 * ||whitened seq resid||^2 - lam_ov * sum phi(shat)
        # with infl==1 everywhere, phi(shat)=shat, so E_ov = -lam_ov*sum shat
        ea, eb = ff.ea, ff.eb
        ca, sa = np.cos(t[ea]), np.sin(t[ea])
        dx, dy = x[eb] - x[ea], y[eb] - y[ea]
        r0 = ca * dx + sa * dy - ff.eZ[:, 0]
        r1 = -sa * dx + ca * dy - ff.eZ[:, 1]
        r2 = S.wrap(t[eb] - t[ea] - ff.eZ[:, 2])
        e_seq = 0.5 * ((ff.ewt * r0)**2 + (ff.ewt * r1)**2 + (ff.ewr * r2)**2).sum()
        # overlap
        u = np.empty((A, D), complex)
        for a in range(A):
            m = int(round(t[a] * L.N_ANG / np.pi))
            dl = t[a] - m * np.pi / L.N_ANG
            u[a] = L.rot_permute(segvec[a], m) + dl * L.rot_permute(segder[a], m)
        nrm = np.linalg.norm(u, axis=1)
        I, J = ff.pairs
        dt = np.stack([x[J] - x[I], y[J] - y[I]], 1)
        phase = np.exp(1j * (dt @ WR.T))
        s = (phase * (np.conj(u[I]) * u[J])).real.sum(1)
        shat = s / (nrm[I] * nrm[J])
        return e_seq - shat.sum()          # lam=1

    # analytic gradient of E = -force (from a single step's internals)
    gx = np.zeros(A); gy = np.zeros(A); gth = np.zeros(A)
    # recompute analytic grad directly (mirror step, no precond/momentum)
    ea, eb = ff.ea, ff.eb
    ca, sa = np.cos(th[ea]), np.sin(th[ea])
    dx, dy = pos[eb, 0] - pos[ea, 0], pos[eb, 1] - pos[ea, 1]
    wt, wr = ff.ewt, ff.ewr
    r0 = ca * dx + sa * dy - ff.eZ[:, 0]
    r1 = -sa * dx + ca * dy - ff.eZ[:, 1]
    r2 = S.wrap(th[eb] - th[ea] - ff.eZ[:, 2])
    f0, f1, f2 = wt * r0, wt * r1, wr * r2
    np.add.at(gx, ea, (-ca * wt) * f0 + (sa * wt) * f1)
    np.add.at(gy, ea, (-sa * wt) * f0 + (-ca * wt) * f1)
    np.add.at(gth, ea, ((-sa * dx + ca * dy) * wt) * f0
              + (-(ca * dx + sa * dy) * wt) * f1 + (-wr) * f2)
    np.add.at(gx, eb, (ca * wt) * f0 + (-sa * wt) * f1)
    np.add.at(gy, eb, (sa * wt) * f0 + (ca * wt) * f1)
    np.add.at(gth, eb, wr * f2)
    # overlap grad (= -force accumulated in step); reuse _rotated
    u, up, nrm, idx = ff._rotated(RINGS_FINE)
    Wa = WR[idx]
    I, J = ff.pairs
    dt = np.stack([pos[J, 0] - pos[I, 0], pos[J, 1] - pos[I, 1]], 1)
    phase = np.exp(1j * (dt @ Wa.T))
    nfac = nrm[I] * nrm[J]
    gmat = phase * (np.conj(u[I]) * u[J])
    ds_dt = -(gmat @ Wa).imag
    ax = (1.0 / nfac)
    # dE/dp = -(1/n) ds/dp   (E_ov = -sum shat)
    np.add.at(gx, J, -ax * ds_dt[:, 0]); np.add.at(gy, J, -ax * ds_dt[:, 1])
    np.add.at(gx, I, ax * ds_dt[:, 0]); np.add.at(gy, I, ax * ds_dt[:, 1])
    dthi = (phase * (np.conj(up[I]) * u[J])).real.sum(1)
    dthj = (phase * (np.conj(u[I]) * up[J])).real.sum(1)
    np.add.at(gth, I, -ax * dthi); np.add.at(gth, J, -ax * dthj)

    # finite differences
    eps = 1e-6
    fd_x = np.zeros(A); fd_y = np.zeros(A); fd_t = np.zeros(A)
    for a in range(A):
        for arr, fd in ((0, fd_x), (1, fd_y), (2, fd_t)):
            xp, yp, tp = pos[:, 0].copy(), pos[:, 1].copy(), th.copy()
            xm, ym, tm = pos[:, 0].copy(), pos[:, 1].copy(), th.copy()
            if arr == 0: xp[a] += eps; xm[a] -= eps
            elif arr == 1: yp[a] += eps; ym[a] -= eps
            else: tp[a] += eps; tm[a] -= eps
            fd[a] = (energy(xp, yp, tp) - energy(xm, ym, tm)) / (2 * eps)
    err = max(np.abs(gx - fd_x).max(), np.abs(gy - fd_y).max(),
              np.abs(gth - fd_t).max())
    rel = err / max(np.abs(fd_x).max(), np.abs(fd_y).max(), np.abs(fd_t).max())
    print(f"selftest gradient FD check: max abs err {err:.2e}  rel {rel:.2e}")
    assert rel < 1e-4, "analytic gradient disagrees with finite differences"
    print("selftest ok: analytic overlap + sequential gradient == FD")


# ==========================================================================
# 4.  Generic continual-flow runner with coarse-to-fine annealing
# ==========================================================================
def run_flow(ff, stages, lr=0.4, mom=0.85, rebuild_every=25, monitor=None,
             mon_every=25, verbose=False):
    """stages = list of (rmask, gate_r, sep, s0, s1, niter). monitor(ff)->tuple
    logged every mon_every iters. Returns (log, it_total)."""
    log = []
    it = 0
    for (mask, gate_r, sep, s0, s1, niter) in stages:
        ff.rmask, ff.gate_r, ff.sep, ff.s0, ff.s1 = mask, gate_r, sep, s0, s1
        ff.build_pairs()
        for _ in range(niter):
            if it % rebuild_every == 0 and it > 0:
                ff.build_pairs()
            if monitor is not None and it % mon_every == 0:
                log.append((it,) + tuple(monitor(ff)))
                if verbose:
                    print(f"    it {it}: {log[-1][1:]}", flush=True)
            ff.step(lr=lr, mom=mom)
            it += 1
    if monitor is not None:
        log.append((it,) + tuple(monitor(ff)))
    return log, it


def _anneal_stages(np_coarse=250, np_mid=250, np_fine=500, gate_c=8.0,
                   gate_f=5.0, sep=40, s0=0.20, s1=0.40):
    # robust threshold s0 must sit ABOVE the noise-pair cosine mass (~0.29 p90
    # on this data) or the many non-overlapping near-pairs corrupt the flow.
    return [
        (RINGS_COARSE, gate_c, sep, s0 - 0.03, s1, np_coarse),
        (RINGS_MID, gate_c, sep, s0, s1, np_mid),
        (RINGS_FINE, gate_f, sep, s0 + 0.02, s1 + 0.02, np_fine),
    ]


def _fine_only_stages(niter=1000, gate=5.0, sep=40, s0=0.22, s1=0.42):
    return [(RINGS_FINE, gate, sep, s0, s1, niter)]


# ==========================================================================
# SANITY: lam_ov=0 -> odometry-only ; collapse guard (traj length preserved)
# ==========================================================================
def sanity():
    print("=" * 70 + "\nSANITY CHECKS\n" + "=" * 70, flush=True)
    print("building a small fr101 frontend map (limit 600 kf) ...", flush=True)
    data = build_map("data/fr101.log", limit=600, verbose=False)
    # --- lam_ov = 0 : flow must be a no-op (seq residuals are 0 at build) ---
    ff, aids, remap = field_from_map(data, lam_seq=1.0, lam_ov=0.0)
    p_before = ff.poses().copy()
    run_flow(ff, _fine_only_stages(niter=300))
    p_after = ff.poses()
    drift = np.linalg.norm(p_after[:, :2] - p_before[:, :2], axis=1).max()
    print(f"  lam_ov=0: max anchor move over 300 iters = {drift*100:.4f} cm  "
          f"({'PASS: reduces to odometry-only' if drift < 1e-3 else 'FAIL'})",
          flush=True)
    # --- collapse guard : stiff skeleton preserves trajectory length ---
    ff, aids, remap = field_from_map(data, lam_seq=1.0, lam_ov=2.0,
                                     gate_r=5.0, sep=40)
    L0 = ff.traj_length()
    run_flow(ff, _anneal_stages(200, 200, 400))
    L1 = ff.traj_length()
    print(f"  collapse guard: traj length {L0:.1f} m -> {L1:.1f} m "
          f"({100*(L1-L0)/L0:+.1f}%)  lam_seq=1 lam_ov=2  "
          f"({'PASS: preserved' if abs(L1-L0)/L0 < 0.1 else 'CHECK'})",
          flush=True)
    # --- collapse without skeleton (lam_seq tiny) : should collapse ---
    ff, aids, remap = field_from_map(data, lam_seq=1e-3, lam_ov=2.0,
                                     gate_r=5.0, sep=40)
    L0b = ff.traj_length()
    run_flow(ff, _anneal_stages(200, 200, 400))
    L1b = ff.traj_length()
    print(f"  NO skeleton (lam_seq=1e-3): traj length {L0b:.1f} -> {L1b:.1f} m "
          f"({100*(L1b-L0b)/L0b:+.1f}%)  => skeleton is load-bearing",
          flush=True)


# ==========================================================================
# TEST 1: single-session continual flow vs shipped detected closure
# ==========================================================================
def run_shipped(path, limit=None):
    """The shipped detected-closure BoundedSLAM, same frontend/harness."""
    scans = C.parse_flaser(path); keys = C.keyframes(scans)
    if limit:
        keys = keys[:limit]
    n = len(keys)
    nb = len(keys[0][0]); beam = np.deg2rad(-90.0 + np.arange(nb) * (180.0 / nb))
    odom = np.stack([k[1] for k in keys]); kts = np.array([t for _, _, t in keys])
    # canonical shipped config (ssp_bounded_carmen): small recency + large gap
    slam = BoundedSLAM(robust=True, attempt_every=4, relax_every=25,
                       gap_kf=300, recent_aids=12)
    slam.store_dtype = np.complex64
    est = np.zeros((n, 3))
    t0 = time.time()
    for k in range(n):
        rr = np.where(keys[k][0] < VALID_MAX, keys[k][0], np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = odom[0] if k == 0 else C.compose_guess(est[k-1], odom[k-1], odom[k])
        est[k] = slam.add_keyframe(pts, w, guess)
    if slam.dirty:
        slam.relax()
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    n_loop = sum(1 for e in slam.edges if e[5] == "loop")
    return kts, fin, n_loop, time.time() - t0


def test_single(name, limit=None):
    path = f"data/{name}.log"
    print("=" * 70 + f"\nTEST 1  single-session: {name}\n" + "=" * 70, flush=True)
    # --- shipped detected-closure baseline ---
    kts, fin_s, n_loop, ts = run_shipped(path, limit)
    a_s = ate_vs_gfs(path, kts, fin_s)
    print(f"  SHIPPED (detected closure): ATE rmse {a_s['rmse']:.3f} m  "
          f"median {a_s['median']:.3f}  max {a_s['max']:.3f}  "
          f"loops={n_loop}  {ts:.0f}s", flush=True)
    # --- odometry-only frontend (lam_ov=0) for context ---
    data = build_map(path, limit=limit, verbose=False)
    ff0, a0, r0 = field_from_map(data, lam_ov=0.0)
    est_odo = poses_to_kf(data, ff0, a0, r0)
    a_odo = ate_vs_gfs(path, data["kts"], est_odo)
    print(f"  frontend-only (no closure): ATE rmse {a_odo['rmse']:.3f} m  "
          f"max {a_odo['max']:.3f}", flush=True)
    # --- continual flow, coarse-to-fine annealing ---
    ff, aids, remap = field_from_map(data, lam_seq=1.0, lam_ov=2.0)
    t0 = time.time()
    run_flow(ff, _anneal_stages(300, 300, 700, gate_c=8.0, gate_f=5.0, sep=40))
    est_f = poses_to_kf(data, ff, aids, remap)
    a_f = ate_vs_gfs(path, data["kts"], est_f)
    print(f"  CONTINUAL FLOW (no detection): ATE rmse {a_f['rmse']:.3f} m  "
          f"median {a_f['median']:.3f}  max {a_f['max']:.3f}  "
          f"{time.time()-t0:.0f}s flow", flush=True)
    # --- fine-only (no annealing) ablation ---
    ff2, a2, r2 = field_from_map(data, lam_seq=1.0, lam_ov=2.0)
    run_flow(ff2, _fine_only_stages(niter=1300, gate=5.0, sep=40))
    est_f2 = poses_to_kf(data, ff2, a2, r2)
    a_f2 = ate_vs_gfs(path, data["kts"], est_f2)
    print(f"  flow fine-only (no anneal):  ATE rmse {a_f2['rmse']:.3f} m  "
          f"max {a_f2['max']:.3f}", flush=True)
    return dict(shipped=a_s, odo=a_odo, flow=a_f, flow_fine=a_f2, n_loop=n_loop)


# ==========================================================================
# TEST 3: multi-session convergence (the crown test)
# ==========================================================================
def _load_sessions():
    cache = M.CACHE
    if not os.path.exists(cache):
        M.build_and_cache()
    return M.load_cache()


def test_multi():
    print("=" * 70 + "\nTEST 3  multi-session convergence (crown)\n" + "=" * 70,
          flush=True)
    blob = _load_sessions()
    gtd = M.gt_dense(blob["kts"], blob["rts"], blob["rp"])
    A, B = blob["A"], blob["B"]
    aidsA = sorted(A["segvec"]); aidsB = sorted(B["segvec"])
    idxA, idxB = A["idx"], B["idx"]
    finA, finB = A["fin"], B["fin"]

    # ---- COARSE, GT-FREE, correspondence-free init for B ------------------
    # single coarse tie: "B started roughly where the robot was at global kf
    # b0" -> map B's frame onto A's using A's OWN drifted estimate of b0.
    # (NOT the dense shared-index oracle, NOT GT.) Then perturb to make it
    # unambiguously coarse and to probe the basin width.
    b0 = blob["b0"]
    kA = np.searchsorted(idxA, b0)                 # A's local kf for global b0
    kB = 0                                         # B's first kf is b0
    T0 = M.se2(finA[kA], M.se2i(finB[kB]))         # coarse B->A rigid tie
    pert = np.array([1.5, -1.0, np.deg2rad(6.0)])  # deliberate coarse offset
    T_init = M.se2(T0, pert)

    # ---- assemble a two-session FlowField: A fixed, B free ----------------
    posA = np.array([A["anchors"][a][:2] for a in aidsA])
    thA = np.array([A["anchors"][a][2] for a in aidsA])
    posB0 = np.array([M.se2(T_init, B["anchors"][a])[:2] for a in aidsB])
    thB0 = np.array([M.se2(T_init, B["anchors"][a])[2] for a in aidsB])
    pos = np.vstack([posA, posB0]); th = np.concatenate([thA, thB0])
    segvec = np.array([A["segvec"][a] for a in aidsA]
                      + [B["segvec"][a] for a in aidsB], complex)
    segder = np.array([A["segder"][a] for a in aidsA]
                      + [B["segder"][a] for a in aidsB], complex)
    nA, nB = len(aidsA), len(aidsB)
    free = np.concatenate([np.zeros(nA, bool), np.ones(nB, bool)])
    sess = np.concatenate([np.zeros(nA, int), np.ones(nB, int)])
    order = np.concatenate([np.arange(nA), np.arange(nB)])
    # B's own sequential skeleton (rigid), transformed into the common frame:
    # Z is frame-invariant, so seq edges carry over unchanged (local index).
    remapB = {a: nA + i for i, a in enumerate(aidsB)}
    seqB = [(remapB[a], remapB[b], Z, wt, wr)
            for a, b, Z, wt, wr, kind in
            [e for e in _session_seq(B)] if a in remapB and b in remapB]
    ff = FlowField(pos, th, segvec, segder, seqB, free, sess, order,
                   lam_seq=1.0, lam_ov=2.0, gate_r=8.0, sep=10 ** 9)

    # ---- GT-based correspondences for SCORING ONLY (never in the flow) ----
    gtB_anchor = np.array([gtd[idxB[a * ANCHOR]][:2] for a in aidsB])
    gtA_anchor = np.array([gtd[idxA[a * ANCHOR]][:2] for a in aidsA])
    treeA = cKDTree(gtA_anchor)
    dcorr, jcorr = treeA.query(gtB_anchor)
    corr = np.flatnonzero(dcorr < 0.5)             # B anchors physically on A
    jA = jcorr[corr]                               # matching A anchor (local)
    print(f"  A={nA} anchors, B={nB} anchors; {len(corr)} GT co-observed "
          f"B-anchors for scoring", flush=True)
    print(f"  coarse init: single-tie T0 + perturb {pert[:2]} m / "
          f"{np.degrees(pert[2]):.0f} deg (GT-free, no shared-index oracle)",
          flush=True)

    def residual(ff):
        # median distance between B's flowed anchor estimate and A's estimate
        # at the SAME physical place (GT-identified) -> the co-observed residual
        Bx = np.stack([ff.x[nA:], ff.y[nA:]], 1)
        Ax = np.stack([ff.x[:nA], ff.y[:nA]], 1)
        d = np.linalg.norm(Bx[corr] - Ax[jA], axis=1)
        return (float(np.median(d)), float(np.mean(d)), float(d.max()))

    r0 = residual(ff)
    print(f"  initial co-observed residual: median {r0[0]:.2f} m  "
          f"mean {r0[1]:.2f}  max {r0[2]:.2f}", flush=True)

    # ---- run the annealed continual flow, logging the convergence curve ----
    # robust cosine thresholds must sit ABOVE the noise-pair mass (~0.29 p90
    # cross-session) or the many non-overlapping near-pairs corrupt the flow.
    stages = [
        (RINGS_COARSE, 10.0, 10**9, 0.25, 0.48, 400),
        (RINGS_MID, 8.0, 10**9, 0.28, 0.50, 400),
        (RINGS_FINE, 6.0, 10**9, 0.30, 0.52, 700),
    ]
    log, _ = run_flow(ff, stages, lr=0.4, mom=0.85, monitor=residual,
                      mon_every=50, verbose=False)
    print("\n  CONVERGENCE CURVE (co-observed residual vs flow iters):",
          flush=True)
    print("    iter |  median  mean   max   (m)", flush=True)
    for row in log:
        print(f"    {row[0]:5d} | {row[1]:6.2f} {row[2]:5.2f} {row[3]:5.2f}",
              flush=True)

    # ---- merged two-session ATE (A fixed + B flowed) ----------------------
    # score B's flowed anchors against GT (rigid-align the whole two-session
    # anchor cloud to GT, report ATE over B's co-observed anchors)
    Ball = np.stack([ff.x[nA:], ff.y[nA:]], 1)
    Aall = np.stack([ff.x[:nA], ff.y[:nA]], 1)
    both = np.vstack([Aall, Ball])
    gt_both = np.vstack([gtA_anchor, gtB_anchor])
    al = C.align_se2(both, gt_both)
    e = np.linalg.norm(al - gt_both, axis=1)
    eB = e[nA:]
    print(f"\n  merged two-session ATE (rigid-aligned to GT): "
          f"all {np.sqrt((e**2).mean()):.2f} m  "
          f"B-only {np.sqrt((eB**2).mean()):.2f} m", flush=True)
    # compare to the discrete-method failures (rigid T_AB residual ~1.2m warp)
    print(f"  final co-observed residual median {log[-1][1]:.2f} m "
          f"(superposition bar ~0.25 m; irreducible non-rigid warp ~1.2 m)",
          flush=True)
    # ---- ablation: fine-only (no annealing) from the same coarse init -----
    return dict(log=log, r0=r0, final=residual(ff))


def _kf_list_for_session(sd, keys_r, beam):
    """Reconstruct [(final_pose, pts, w, own_aid)] for a cached session."""
    idx = sd["idx"]; fin = sd["fin"]
    out = []
    for j, gk in enumerate(idx):
        rr = np.where(keys_r[gk] < VALID_MAX, keys_r[gk], np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        out.append((fin[j], pts, w, j // ANCHOR))
    return out


def test_multi_soft(offsets=(0.0, 1.5, 3.0, 4.5), modes=("hard", "kernel", "bary")):
    """SOFT / partition-of-unity feature distribution vs hard binning on the
    multi-session convergence test. Basin-width sweep: init B at best rigid
    T_AB + an extra offset; report final co-observed residual, within-1m lock
    count, and descent smoothness per mode. (User's enhancement.)"""
    print("=" * 70 + "\nSOFT/HEX feature distribution vs HARD binning (multi)\n"
          + "=" * 70, flush=True)
    blob = _load_sessions()
    gtd = M.gt_dense(blob["kts"], blob["rts"], blob["rp"])
    A, B = blob["A"], blob["B"]
    aidsA, aidsB = sorted(A["segvec"]), sorted(B["segvec"])
    idxA, idxB = A["idx"], B["idx"]
    keys_r, beam = blob["keys_r"], blob["beam"]
    gtAf = np.array([gtd[idxA[a * ANCHOR]] for a in aidsA])
    gtBf = np.array([gtd[idxB[a * ANCHOR]] for a in aidsB])
    gtA, gtB = gtAf[:, :2], gtBf[:, :2]
    treeA = cKDTree(gtA); dcorr, jcorr = treeA.query(gtB)
    hdg = np.array([abs(S.wrap(gtAf[jcorr[b], 2] - gtBf[b, 2])) < np.deg2rad(20)
                    for b in range(len(aidsB))])
    corr = np.flatnonzero((dcorr < 0.5) & hdg); jA = jcorr[corr]
    # best rigid T_AB (favorable init; leaves the non-rigid warp)
    Ts = np.array([M.se2(A["anchors"][aidsA[int(jA[k])]],
                         M.se2i(B["anchors"][aidsB[int(corr[k])]]))
                   for k in range(len(corr))])
    T_AB = np.array([np.median(Ts[:, 0]), np.median(Ts[:, 1]),
                     np.arctan2(np.median(np.sin(Ts[:, 2])),
                                np.median(np.cos(Ts[:, 2])))])
    print(f"  {len(corr)} GT co-observed anchors; rigid T_AB init baseline",
          flush=True)
    # per-mode segvec rebuilds (mass-conservation check inline)
    kfA = _kf_list_for_session(A, keys_r, beam)
    kfB = _kf_list_for_session(B, keys_r, beam)
    apA = {a: A["anchors"][a] for a in aidsA}
    apB = {a: B["anchors"][a] for a in aidsB}
    segs = {}
    hard_mass = None
    for mode in modes:
        svA, sdA = soft_consolidate(aidsA, apA, kfA, mode=mode)
        svB, sdB = soft_consolidate(aidsB, apB, kfB, mode=mode)
        segs[mode] = (svA, sdA, svB, sdB)
        m = sum(np.abs(v).sum() for v in svB.values())
        if hard_mass is None and mode == "hard":
            hard_mass = m
        print(f"  [{mode}] rebuilt; B segvec total |mass| {m:.1f}"
              + (f"  (ratio to hard {m/hard_mass:.3f})"
                 if hard_mass else ""), flush=True)

    nA, nB = len(aidsA), len(aidsB)
    remapB = {a: nA + i for i, a in enumerate(aidsB)}
    seqB = [(remapB[a], remapB[b], Z, wt, wr)
            for a, b, Z, wt, wr, kind in _session_seq(B)
            if a in remapB and b in remapB]
    sess = np.concatenate([np.zeros(nA, int), np.ones(nB, int)])
    order = np.concatenate([np.arange(nA), np.arange(nB)])
    free = np.concatenate([np.zeros(nA, bool), np.ones(nB, bool)])
    posA = np.array([A["anchors"][a][:2] for a in aidsA])
    thA = np.array([A["anchors"][a][2] for a in aidsA])

    def make_ff(mode, extra):
        svA, sdA, svB, sdB = segs[mode]
        Tinit = M.se2(T_AB, np.array([extra, 0.0, 0.0]))
        posB = np.array([M.se2(Tinit, B["anchors"][a])[:2] for a in aidsB])
        thB = np.array([M.se2(Tinit, B["anchors"][a])[2] for a in aidsB])
        pos = np.vstack([posA, posB]); th = np.concatenate([thA, thB])
        segvec = np.array([svA[a] for a in aidsA] + [svB[a] for a in aidsB], complex)
        segder = np.array([sdA[a] for a in aidsA] + [sdB[a] for a in aidsB], complex)
        return FlowField(pos, th, segvec, segder, seqB, free, sess, order,
                         lam_seq=1.0, lam_ov=2.0, gate_r=6.0, sep=20)

    def resid(ff):
        Bx = np.stack([ff.x[nA:], ff.y[nA:]], 1)
        Ax = np.stack([ff.x[:nA], ff.y[:nA]], 1)
        return np.linalg.norm(Bx[corr] - Ax[jA], axis=1)

    print(f"\n  {'mode':>7} {'off':>4} | init_med  final_med  within1m  "
          f"mono%   (m; N={len(corr)})", flush=True)
    for extra in offsets:
        for mode in modes:
            ff = make_ff(mode, extra)
            d0 = resid(ff)
            meds = []
            stages = [(RINGS_MID, 6.0, 20, 0.28, 0.50, 250),
                      (RINGS_FINE, 5.0, 20, 0.30, 0.52, 400)]
            log, _ = run_flow(ff, stages, lr=0.4, mom=0.85,
                              monitor=lambda ff: (np.median(resid(ff)),),
                              mon_every=25)
            meds = np.array([r[1] for r in log])
            mono = float(np.mean(np.diff(meds) <= 1e-3)) * 100
            d = resid(ff)
            print(f"  {mode:>7} {extra:4.1f} | {np.median(d0):7.2f}  "
                  f"{np.median(d):8.2f}  {int((d<1).sum()):7d}  {mono:5.0f}",
                  flush=True)
    print("\n  (basin wall is the cross-session cosine indistinguishability, "
          "not force discontinuity; soft binning smooths descent + lock count "
          "at the ~2m patch scale but cannot separate genuine from noise.)",
          flush=True)


def _session_seq(data):
    """Reconstruct sequential edges for a cached session (consecutive anchors,
    Z from the session's own final anchor poses)."""
    aids = sorted(data["segvec"])
    anch = data["anchors"]
    out = []
    for a, b in zip(aids[:-1], aids[1:]):
        if b == a + 1:                             # only truly consecutive
            Z = L.se2_mul(L.se2_inv(anch[a]), anch[b])
            out.append((a, b, Z, 1 / 0.05, 1 / np.deg2rad(0.5), "seq"))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        selftest()
    elif cmd == "sanity":
        sanity()
    elif cmd == "multi":
        test_multi()
    elif cmd == "soft":
        test_multi_soft()
    elif cmd == "single":
        for nm in sys.argv[2:] or ["intel", "fr079", "fr101"]:
            test_single(nm)
    elif cmd == "mit":
        test_single("mit", limit=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:
        print("unknown command", cmd)
