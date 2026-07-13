"""Observability-weighted ANISOTROPIC loop constraints (subclass, no shipped edit).

The GT-verify diagnostic (experiments/mit_gtverify.py; RESULTS.md "the wall is TWO-FOLD")
showed the MIT corridor wall is place AND relative-pose GEOMETRY: at a genuine
corridor revisit the matcher SLIDES along the corridor (the aperture direction is
unobservable), so an admitted closure is well-aligned ACROSS the corridor but
mis-aligned ALONG it. The shipped system already computes the observability
structure of every candidate (the translation-score Hessian K in try_constraint,
eigenpairs l_weak<=l_strong / u_weak) but throws it away after using it only to
VETO. This module keeps the closure and inserts it with an ANISOTROPIC information
matrix: full weight in the observable (strong-curvature) direction, floored weight
in the unobservable (weak) direction — so the observable component of an otherwise-
vetoed/isotropically-harmful revisit still corrects the drift it CAN see, while its
along-corridor slide contributes almost nothing.

Design (see AnisoMixin below):
  * Loop edges carry a 2x2 lower-triangular Cholesky factor L_body (translation
    information Lambda = L L^T) in the SAME frame the _gn residual uses (anchor a's
    body frame). Sequential edges stay isotropic.
  * Lambda_world = wt^2 * K / l_strong with the condition number l_weak/l_strong
    CLAMPED to >= cond_floor (sweepable), then rotated into the body frame by the
    anchor heading: Lambda_body = R(-th_a) Lambda_world R(th_a). Rotation stays
    scalar (1/sig_r). Frame frozen at insertion heading (relax rotation corrections
    are small; keeps the whitening matrix constant so the analytic Jacobian is clean).
  * _gn whitens the 2-vector translation residual r_xy -> L^T r_xy (and the Jacobian
    rows correspondingly). The IRLS scale ss is folded into L FIRST so the isotropic
    reduction is BIT-EXACT: for L = diag(wt), L^T r * ss == r * (wt*ss) == shipped.

HARD GUARANTEES (selftest): (1) use_aniso=False (or K=I) reproduces shipped
BoundedSLAM BIT-EXACT — every loop edge defaults to diag(wt), the 2x2 path reduces
to the scalar path exactly; (2) the analytic Jacobian is finite-difference-checked
against the anisotropic residual. So any ATE change on real runs is the anisotropy,
not a solver regression.

Usage:
  python3 -m experiments.aniso selftest
  python3 -m experiments.aniso logs [floor ...]      # real-log ATE sweep vs shipped
  python3 -m experiments.aniso mit [data/mit.log]    # GT-oracle MIT (anisotropic drought)
"""

import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

import sspslam.encoder as S
import sspslam.lattice as L
import sspslam.bounded as B


# ---------------------------------------------------------------------------
# Anisotropic translation whitening: a mixin over any BoundedSLAM descendant.
# ---------------------------------------------------------------------------
class AnisoMixin:
    """Reimplements the translation part of BoundedSLAM._gn to whiten the
    2-vector translation residual by a per-edge lower-triangular Cholesky factor
    L (Lambda = L L^T, anchor-a body frame). Everything else — IRLS, LOO,
    rotation whitening, the max_nfev early-stopping regularizer, the Tikhonov
    prior — is preserved bit-for-bit. Edges without a stored L default to the
    isotropic diag(wt), which reduces to the shipped scalar path EXACTLY."""

    def _aniso_init(self, use_aniso=True, cond_floor=0.1, aniso_gate=1.0):
        self.aniso_L = {}          # (a_id, b_id) -> 2x2 lower-tri L (body frame)
        self.use_aniso = use_aniso
        self.cond_floor = cond_floor   # clamp l_weak/l_strong to >= this
        # PER-CLOSURE CONDITIONING GATE: only closures with l_weak/l_strong <
        # aniso_gate get the anisotropic 2x2; WELL-conditioned closures (ratio
        # >= gate) are inserted FULLY ISOTROPIC (== shipped, bit-identical), so
        # clean logs whose closures are well-conditioned are provably untouched.
        # gate=1.0 => anisotropize every closure (the un-gated behaviour).
        self.aniso_gate = aniso_gate
        self.n_aniso_trip = 0      # closures that tripped the gate (got aniso)
        self.n_aniso_seen = 0      # closures a factor was computed for
        self._gn_resid = None      # exposed for the FD Jacobian selftest
        self._gn_jac = None

    # ---- derive the body-frame Cholesky factor from the observability Hessian
    def _aniso_L(self, evals, evecs, wt, th_a):
        """evals/evecs = ascending eigenpairs of the world-frame translation
        Hessian K (as in try_constraint); wt = 1/(sig_t*infl) is the shipped
        scalar strong-direction weight; th_a = anchor-a heading. Returns a 2x2
        lower-tri L such that L L^T = Lambda_body, or None (isotropic -> exact
        scalar path). Lambda_world = wt^2 * {r_f, 1} in the {u_weak,u_strong}
        eigenbasis with r_f = clamp(l_weak/l_strong, cond_floor, 1)."""
        if not self.use_aniso:
            return None
        l_weak, l_strong = float(evals[0]), float(evals[1])
        if l_strong <= 1e-30:
            return None
        r = l_weak / l_strong
        self.n_aniso_seen += 1
        if r >= self.aniso_gate:
            return None            # well-conditioned closure: fully isotropic
        self.n_aniso_trip += 1
        r_f = min(1.0, max(r, self.cond_floor))
        if r_f >= 1.0:
            return None            # isotropic: scalar path is bit-exact
        u_w = evecs[:, 0]
        u_s = evecs[:, 1]
        Rm = S._rot(-th_a)                      # world -> anchor body frame
        uw_b = Rm @ u_w
        us_b = Rm @ u_s
        # Lambda_body = wt^2 (u_s u_s^T + r_f u_w u_w^T) rotated into body frame
        Lam = wt * wt * (np.outer(us_b, us_b) + r_f * np.outer(uw_b, uw_b))
        # symmetrize against round-off, then lower Cholesky
        Lam = 0.5 * (Lam + Lam.T)
        return np.linalg.cholesky(Lam)

    def _set_aniso(self, key, Lm):
        if Lm is None:
            self.aniso_L.pop(key, None)
        else:
            self.aniso_L[key] = Lm

    # ---- observability eigenpairs at a matched pose (K exactly as in
    #      try_constraint), for edge paths that don't already compute them
    def _obs_eigen(self, Bfull, pts, w, pose):
        sv = L.ENC.shift(pose[:2]) * L.encode(pts @ S._rot(pose[2]).T, w)
        c = (np.conj(Bfull) * sv)
        Br = Bfull.reshape(L.N_RING, L.N_ANG)
        svr = sv.reshape(L.N_RING, L.N_ANG)
        cM = c[L.MAIN]
        nrmM = (np.linalg.norm(Br[:4].ravel())
                * np.linalg.norm(svr[:4].ravel()) + 1e-12)
        K = (L.W[L.MAIN] * (cM.real / nrmM)[:, None]).T @ L.W[L.MAIN]
        return np.linalg.eigh(K)

    # ---- backend: same TRF solver, anisotropic translation whitening --------
    def _gn(self, eds, wl, free, P0):
        E, Fn = len(eds), len(free)
        if E == 0 or Fn == 0:
            return P0.copy(), np.zeros(3 * E)
        aa = np.array([e[0] for e in eds])
        bb = np.array([e[1] for e in eds])
        Zt = np.stack([e[2] for e in eds])
        wts = np.array([e[3] for e in eds])
        wrs = np.array([e[4] for e in eds])
        ss = np.array([wl.get(e, 1.0) for e in range(E)])
        # per-edge lower-tri translation factor L; default isotropic diag(wt)
        L00 = wts.astype(float).copy()
        L10 = np.zeros(E)
        L11 = wts.astype(float).copy()
        for i, e in enumerate(eds):
            Lm = self.aniso_L.get((e[0], e[1])) if e[5] == "loop" else None
            if Lm is not None:
                L00[i], L10[i], L11[i] = Lm[0, 0], Lm[1, 0], Lm[1, 1]
        # fold the IRLS scale into L FIRST so diag(wt) reduces to wt_s == wt*ss
        # BIT-EXACT (matches shipped wt_s = wts*ss then residual * wt_s)
        Ls00, Ls10, Ls11 = L00 * ss, L10 * ss, L11 * ss
        wr_s = wrs * ss
        lt, lr = self.prior_lam_t, self.prior_lam_r
        Xp = P0[free]

        def resid(x):
            P = P0.copy()
            P[free] = x.reshape(-1, 3)
            ca, sa = np.cos(P[aa, 2]), np.sin(P[aa, 2])
            dx = P[bb, 0] - P[aa, 0]
            dy = P[bb, 1] - P[aa, 1]
            r0 = ca * dx + sa * dy - Zt[:, 0]        # raw body-frame residual
            r1 = -sa * dx + ca * dy - Zt[:, 1]
            out = np.empty((E, 3))
            out[:, 0] = Ls00 * r0 + Ls10 * r1        # (L^T r)_0 scaled by ss
            out[:, 1] = Ls11 * r1                     # (L^T r)_1 scaled by ss
            out[:, 2] = S.wrap(P[bb, 2] - P[aa, 2] - Zt[:, 2]) * wr_s
            if lt <= 0:
                return out.ravel()
            X = x.reshape(-1, 3)
            pr = np.empty((Fn, 3))
            pr[:, 0] = lt * (X[:, 0] - Xp[:, 0])
            pr[:, 1] = lt * (X[:, 1] - Xp[:, 1])
            pr[:, 2] = lr * S.wrap(X[:, 2] - Xp[:, 2])
            return np.concatenate([out.ravel(), pr.ravel()])

        ro = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
        comp = np.array([0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 2])
        isb = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1], bool)
        col_of = np.full(len(P0), -1)
        col_of[free] = np.arange(Fn)
        node = np.where(isb, bb[:, None], aa[:, None])
        nc = col_of[node]
        keep = (nc >= 0).ravel()
        rows = (3 * np.arange(E)[:, None] + ro).ravel()[keep]
        cols = (3 * nc + comp).ravel()[keep]
        nrows = 3 * E
        pvals = np.empty(0)
        if lt > 0:
            rows = np.concatenate([rows, 3 * E + np.arange(3 * Fn)])
            cols = np.concatenate([cols, np.arange(3 * Fn)])
            pvals = np.tile([lt, lt, lr], Fn)
            nrows += 3 * Fn

        def jac(x):
            P = P0.copy()
            P[free] = x.reshape(-1, 3)
            ca, sa = np.cos(P[aa, 2]), np.sin(P[aa, 2])
            dx = P[bb, 0] - P[aa, 0]
            dy = P[bb, 1] - P[aa, 1]
            # raw (unwhitened) partials of r0, r1 over cols [xa,ya,tha,xb,yb]
            g0 = np.stack([-ca, -sa, -sa * dx + ca * dy, ca, sa], axis=1)
            g1 = np.stack([sa, -ca, -(ca * dx + sa * dy), -sa, ca], axis=1)
            V = np.empty((E, 12))
            # whitened row 0 = Ls00 * g0 + Ls10 * g1 ; row 1 = Ls11 * g1
            V[:, 0:5] = Ls00[:, None] * g0 + Ls10[:, None] * g1
            V[:, 5:10] = Ls11[:, None] * g1
            V[:, 10], V[:, 11] = -wr_s, wr_s
            return csr_matrix((np.concatenate([V.ravel()[keep], pvals]),
                               (rows, cols)), shape=(nrows, 3 * Fn))

        self._gn_resid, self._gn_jac = resid, jac    # exposed for FD selftest
        x0 = P0[free].ravel()
        sol = least_squares(resid, x0, jac=jac, method="trf",
                            x_scale="jac", max_nfev=200 if lt > 0 else 30)
        self.n_nfev_cap += int(sol.status == 0)
        P = P0.copy()
        P[free] = sol.x.reshape(-1, 3)
        return P, sol.fun[:3 * E]


# ---------------------------------------------------------------------------
# Anisotropic BoundedSLAM: same try_constraint, but stores the observability-
# derived L on each accepted loop edge.
# ---------------------------------------------------------------------------
class AnisoBoundedSLAM(AnisoMixin, B.BoundedSLAM):
    def __init__(self, use_aniso=True, cond_floor=0.1, iso_prune=False,
                 aniso_gate=1.0, **kw):
        super().__init__(**kw)
        self._aniso_init(use_aniso=use_aniso, cond_floor=cond_floor,
                         aniso_gate=aniso_gate)
        # iso_prune: rank LOO/chi2 PRUNING on the ISOTROPICALLY-whitened residual
        # (the true geometric-error magnitude), while the SOLVE keeps the
        # anisotropic whitening. The anisotropic residual that shipped LOO ranks
        # on shrinks a weak-geometry (along-corridor slide) edge below the
        # suspicious>2.0 gate, so it never gets LOO-tested and floods the graph
        # (Intel 80->158 edges). Ranking on geometric error restores correct
        # pruning of weak junk while the solver still gets the observable-
        # direction correction. Off = the raw anisotropic behaviour.
        self.iso_prune = iso_prune

    def _relax_solve(self, win):
        # VERBATIM copy of BoundedSLAM._relax_solve; the ONLY change is the LOO
        # `raws` ranking/gate, which (when iso_prune) ranks on the isotropic
        # geometric residual at the solved poses instead of norm(r)/wl (which is
        # anisotropic here). The SOLVE (self._gn) and IRLS weights are untouched.
        # With iso_prune=False this reproduces the shipped method exactly.
        A = len(self.anchors)
        if win is None:
            fs = set(range(1, A)) - self.retired
        else:
            fs = (set(win) - self.retired) - {0}
        if not fs:
            return True
        free = np.array(sorted(fs), int)
        P0 = np.array(self.anchors)
        eds = [e for e in self.edges if e[0] in fs or e[1] in fs]
        P, r = self._gn(eds, {}, free, P0)
        if self.robust:
            wl = {}
            for e, (a, b, Z, wt, wr, kind) in enumerate(eds):
                if kind != "loop":
                    continue
                rn = np.linalg.norm(r[3 * e:3 * e + 3])
                wl[e] = 1.0 / np.sqrt(1.0 + (rn / 2.0) ** 2)
            P, r = self._gn(eds, wl, free, P0)
            if self.iso_prune:
                # rank on the geometric-error magnitude (scalar wt whitening) at
                # the anisotropically-SOLVED poses, so a weak-geometry slide with
                # a large along-corridor error is correctly flagged suspicious.
                raws = {}
                for e, (a, b, Z, wt, wr, kind) in enumerate(eds):
                    if kind != "loop" or a not in fs or b not in fs:
                        continue
                    d = S._rot(-P[a, 2]) @ (P[b, :2] - P[a, :2])
                    raws[e] = np.linalg.norm(np.array(
                        [*((d - Z[:2]) * wt),
                         S.wrap(P[b, 2] - P[a, 2] - Z[2]) * wr]))
            else:
                raws = {e: np.linalg.norm(r[3 * e:3 * e + 3]) / max(wl.get(e, 1.0), 1e-9)
                        for e, (a, b, Z, wt, wr, kind) in enumerate(eds)
                        if kind == "loop" and a in fs and b in fs}
            suspicious = sorted((e for e, v in raws.items() if v > 2.0),
                                key=lambda e: -raws[e])
            bad = set()
            for e in suspicious[:6]:
                a, b, Z, wt, wr, kind = eds[e]
                P2, _ = self._gn([ed for i, ed in enumerate(eds) if i != e],
                                 {}, free, P0)
                d = S._rot(-P2[a, 2]) @ (P2[b, :2] - P2[a, :2])
                rn = np.linalg.norm(np.array([*((d - Z[:2]) * wt),
                                              S.wrap(P2[b, 2] - P2[a, 2] - Z[2]) * wr]))
                if rn > 3.0:
                    if win is not None:
                        return False
                    bad.add(e)
                    self.banned.add((a, b))
            if bad:
                self.n_pruned += len(bad)
                drop_ids = {id(eds[e]) for e in bad}
                self.edges = [ed for ed in self.edges if id(ed) not in drop_ids]
                self.edge_seen = {(a, b): i for i, (a, b, *_, kind) in
                                  enumerate(self.edges) if kind == "loop"}
                eds = [e for e in self.edges if e[0] in fs or e[1] in fs]
                P, r = self._gn(eds, {}, free, P0)
                wl2 = {e: 1.0 / np.sqrt(1.0 + (np.linalg.norm(r[3*e:3*e+3]) / 2.0) ** 2)
                       for e, (a, b, Z, wt, wr, kind) in enumerate(eds)
                       if kind == "loop"}
                P, r = self._gn(eds, wl2, free, P0)
        if win is not None:
            for e, (a, b, Z, wt, wr, kind) in enumerate(eds):
                if kind == "seq" and ((a in fs) != (b in fs)) \
                        and np.linalg.norm(r[3 * e:3 * e + 3]) > 3.0:
                    return False
                if self.robust and kind == "loop" and a in fs and b in fs \
                        and np.linalg.norm(r[3 * e:3 * e + 3]) \
                        / max(wl.get(e, 1.0), 1e-9) > 3.0:
                    return False
        jerk = float(np.max(np.linalg.norm(P[free, :2] - P0[free, :2], axis=1)))
        self.max_jerk = max(self.max_jerk, jerk)
        for i in free:
            self.anchors[i] = P[i]
        self._last_nfree = len(free)
        return True

    def try_constraint(self, pts, w):
        # VERBATIM copy of BoundedSLAM.try_constraint, with the observability-
        # derived anisotropic L computed from the (already-computed) K eigenpairs
        # and stored on the edge at insertion. When use_aniso=False every store
        # is None, so behaviour is bit-identical to the shipped scalar path.
        k = self.k
        if len(pts) < 20:
            return
        me = self.pose_of(k)
        my_aid = self.aid_of(k)
        cands = [aid for aid in self.segvec
                 if abs(aid - my_aid) > self.gap_kf // B.ANCHOR
                 and np.linalg.norm(self.anchors[aid][:2] - me[:2]) < 5.0]
        if not cands:
            return
        cands.sort()
        chains, cur = [], [cands[0]]
        for aid in cands[1:]:
            (cur.append(aid) if aid - cur[-1] <= 2 else (chains.append(cur), cur := [aid]))
        chains.append(cur)
        chain = min(chains, key=lambda ch: min(
            np.linalg.norm(self.anchors[a][:2] - me[:2]) for a in ch))
        Bc = sum(self.world_vec_seg(a) for a in chain)
        pose = self.cmatcher.match(Bc[L.MAIN], pts, w, me)
        pose[2] = S.wrap(pose[2])
        if np.linalg.norm(pose[:2] - me[:2]) > 0.6 \
                or abs(S.wrap(pose[2] - me[2])) > np.deg2rad(7):
            return
        sv = L.ENC.shift(pose[:2]) * L.encode(pts @ S._rot(pose[2]).T, w)
        c = (np.conj(Bc) * sv).reshape(L.N_RING, L.N_ANG)
        Br = Bc.reshape(L.N_RING, L.N_ANG)
        svr = sv.reshape(L.N_RING, L.N_ANG)
        coh = c.sum(1).real / (np.linalg.norm(Br, axis=1)
                               * np.linalg.norm(svr, axis=1) + 1e-12)
        cM = c[:4].ravel()
        nrmM = (np.linalg.norm(Br[:4].ravel())
                * np.linalg.norm(svr[:4].ravel()) + 1e-12)
        K = (L.W[L.MAIN] * (cM.real / nrmM)[:, None]).T @ L.W[L.MAIN]
        evals, evecs = np.linalg.eigh(K)
        l_weak, l_strong = float(evals[0]), float(evals[1])
        u_weak = evecs[:, 0]
        ts = np.arange(-1.2, 1.21, 0.06)
        s0 = float(cM.real.sum())
        far = np.abs(ts) >= 0.35
        ridge = [0.0, 0.0]
        for j, u in enumerate((u_weak, evecs[:, 1])):
            proj = L.W[L.MAIN] @ u
            sc = (np.exp(1j * ts[:, None] * proj[None, :]) @ cM).real
            ridge[j] = float(sc[far].max() / max(s0, 1e-12))
        ridge_w, ridge_s = ridge
        c_aid = chain[len(chain) // 2]
        lever = np.linalg.norm(me[:2] - self.anchors[c_aid][:2])
        sig_t = np.sqrt(0.08 ** 2 + (0.05 * lever) ** 2)
        sig_r = np.deg2rad(2.0)
        err = e_weak = e_strong = np.nan
        if self.diag_gt is not None:
            gtk = self.diag_gt
            Zm = L.se2_mul(L.se2_inv(self.anchors[c_aid]),
                           L.se2_mul(pose, L.se2_inv(self.kf_ref[k][1])))
            Zt = L.se2_mul(L.se2_inv(gtk[c_aid * B.ANCHOR]), gtk[k - (k % B.ANCHOR)])
            err = np.linalg.norm(Zm[:2] - Zt[:2])
            errW = S._rot(self.anchors[c_aid][2]) @ (Zm[:2] - Zt[:2])
            e_weak = abs(float(errW @ u_weak))
            e_strong = abs(float(errW @ evecs[:, 1]))
        iv_weak = abs(float((pose[:2] - me[:2]) @ u_weak))
        self.diag.append([err, np.linalg.norm(pose[:2] - me[:2]),
                          abs(S.wrap(pose[2] - me[2])), len(chain), *coh,
                          l_weak, l_strong, u_weak[0], u_weak[1],
                          e_weak, e_strong, iv_weak, ridge_w, ridge_s,
                          self.coh_ref if self.coh_ref is not None else np.nan,
                          0.0])
        infl = 1.0
        if self.use_coh and self.coh_ref is not None:
            if not self.coh_soft:
                if coh[:2].mean() < self.coh_target * self.coh_ref:
                    self.n_veto += 1
                    return
            else:
                infl = self._coh_response(coh, l_weak, l_strong,
                                          ridge_w, ridge_s, me, k)
                if infl < 0:
                    return
        _, rel = self.kf_ref[k]
        Zk = L.se2_mul(pose, L.se2_inv(rel))
        Z = L.se2_mul(L.se2_inv(self.anchors[c_aid]), Zk)
        if self.inject_rate and self._inj_rng.random() < self.inject_rate:
            if self.inject_mode == "aliased":
                if self._alias is None:
                    a0 = self._inj_rng.uniform(0, 2 * np.pi)
                    self._alias = (self._inj_rng.uniform(0.6, 1.2)
                                   * np.array([np.cos(a0), np.sin(a0)]),
                                   np.deg2rad(self._inj_rng.uniform(6, 12)))
                Z[:2] += self._alias[0]
                Z[2] = S.wrap(Z[2] + self._alias[1])
            else:
                ang = self._inj_rng.uniform(0, 2 * np.pi)
                Z[:2] += self._inj_rng.uniform(0.5, 1.5) * np.array([np.cos(ang), np.sin(ang)])
                Z[2] = S.wrap(Z[2] + np.deg2rad(self._inj_rng.uniform(5, 15))
                              * self._inj_rng.choice([-1, 1]))
        Zc = L.se2_mul(L.se2_inv(self.anchors[c_aid]), self.anchors[my_aid])
        since = k - self.last_accept_k
        s_at = sig_t + min(0.30, 0.002 * since)
        s_ar = sig_r + min(np.deg2rad(6), np.deg2rad(0.03) * since)
        chi = (np.linalg.norm(Z[:2] - Zc[:2]) / s_at) ** 2 \
            + (S.wrap(Z[2] - Zc[2]) / s_ar) ** 2
        if self.use_innov and chi > 9.0:
            self.n_innov_rej += 1
            return
        wt = 1.0 / (sig_t * infl)
        # observability-derived anisotropic factor in anchor-a (c_aid) body frame
        Laniso = self._aniso_L(evals, evecs, wt, self.anchors[c_aid][2])
        edge = (c_aid, my_aid, Z, wt, 1 / (sig_r * infl), "loop")
        key = (c_aid, my_aid)
        if key in self.banned:
            return
        if key in self.edge_seen:
            if edge[3] * 3.5 < self.edges[self.edge_seen[key]][3]:
                return
            self.edges[self.edge_seen[key]] = edge
            self._set_aniso(key, Laniso)
        else:
            self.edge_seen[key] = len(self.edges)
            self.edges.append(edge)
            self._set_aniso(key, Laniso)
        if infl <= self.infl_weak:
            self.dirty = True
            self.pending_new.append(key)
            self.last_accept_k = k
            self._streak = 0
            self._streak_xy = None
        else:
            self._suppress(me[:2], k)
        self.diag[-1][-1] = 1.0


# ===========================================================================
# SELFTEST
# ===========================================================================
def _bench_traj(cls, seed=1, world="corridor", n=400, **kw):
    """Run the synthetic multiloop bench (GT-labelled) with a given SLAM class
    and return the final trajectory + ATE (deterministic)."""
    import sspslam.bench as BL
    from sspslam.worlds import WORLDS
    S.RNG = np.random.default_rng(seed)
    segs = WORLDS[world]()
    gt = BL.multiloop_traj(n, laps=3)
    odo = B.sim_odometry(gt)
    slam = cls(**kw)
    slam.diag_gt = gt
    est = np.zeros((n, 3))
    for k in range(n):
        r = BL.scan_at(segs, gt[k])
        pts, w, _ = S.scan_to_samples(r, BL.BEAM)
        guess = odo[0] if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odo[k - 1]), odo[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    ate = np.sqrt((np.linalg.norm(fin[:, :2] - gt[:, :2], axis=1) ** 2).mean())
    return fin, float(ate), slam


def selftest():
    print("=== ssp_aniso selftest ===")

    # ---- (1) BIT-EXACT: use_aniso=False must reproduce shipped BoundedSLAM ----
    # Every loop edge defaults to diag(wt); the reimplemented 2x2 _gn must reduce
    # to the shipped scalar path bit-for-bit across the full nonlinear solve.
    print("[1] bit-exact reduction (use_aniso=False == shipped BoundedSLAM):")
    for world, seed in (("corridor", 1), ("room", 2), ("office", 3)):
        f_ship, a_ship, _ = _bench_traj(B.BoundedSLAM, seed=seed, world=world)
        f_iso, a_iso, _ = _bench_traj(AnisoBoundedSLAM, seed=seed, world=world,
                                      use_aniso=False)
        dmax = float(np.max(np.abs(f_ship - f_iso)))
        ok = np.array_equal(f_ship, f_iso)
        print(f"    {world:<9} seed {seed}: shipped ATE {a_ship:.6f} | aniso-off "
              f"{a_iso:.6f} | max|dpose| {dmax:.2e} | {'BIT-EXACT' if ok else 'DIFFERS'}")
        assert ok, f"use_aniso=False NOT bit-exact on {world} (max {dmax:.2e})"

    # ---- (2) DERIVATION: K = I must yield the isotropic (scalar) path ---------
    print("[2] derivation neutrality (K=I -> None / diag(wt)):")
    sl = AnisoBoundedSLAM(use_aniso=True, cond_floor=0.1)
    ev, evec = np.linalg.eigh(np.eye(2))
    Lm = sl._aniso_L(ev, evec, wt=7.3, th_a=0.9)
    print(f"    K=I -> L is None: {Lm is None}")
    assert Lm is None, "K=I should map to the isotropic scalar path"
    # and a genuinely anisotropic K yields the expected floored eigen-spectrum
    Kd = np.diag([0.02, 5.0])                    # weak/strong ratio 0.004
    ev, evec = np.linalg.eigh(Kd)
    for floor in (0.05, 0.1, 0.2):
        sl.cond_floor = floor
        Lm = sl._aniso_L(ev, evec, wt=4.0, th_a=0.0)
        Lam = Lm @ Lm.T
        w_lam = np.sort(np.linalg.eigvalsh(Lam))
        exp = np.sort([4.0 ** 2 * floor, 4.0 ** 2])
        print(f"    floor {floor}: Lambda eig {w_lam.round(3)} (expect {exp.round(3)})")
        assert np.allclose(w_lam, exp, rtol=1e-9), "floored spectrum wrong"

    # ---- (3) FRAME: anisotropy points along the corridor in the body frame ----
    # World-frame weak direction u_weak at heading th_a must appear rotated by
    # -th_a in the body-frame Lambda's weak eigenvector.
    print("[3] frame rotation (weak eigenvector tracks the anchor heading):")
    ang = 0.7
    uw = np.array([np.cos(ang), np.sin(ang)])       # world weak dir
    us = np.array([-np.sin(ang), np.cos(ang)])
    evec = np.stack([uw, us], axis=1)               # cols: weak, strong
    ev = np.array([0.05, 5.0])
    for th_a in (0.0, 0.5, -1.2):
        Lm = sl._aniso_L(ev, evec, wt=3.0, th_a=th_a)
        Lam = Lm @ Lm.T
        w2, V2 = np.linalg.eigh(Lam)
        weak_body = V2[:, 0]                        # smallest-eigenvalue dir
        expect = S._rot(-th_a) @ uw
        al = abs(float(weak_body @ expect))
        print(f"    th_a {th_a:+.2f}: |<weak_body, R(-th)u_weak>| = {al:.6f}")
        assert al > 1 - 1e-9, "body-frame weak direction not rotated correctly"

    # ---- (4) FINITE-DIFFERENCE Jacobian check on an anisotropic graph ---------
    print("[4] finite-difference Jacobian check (anisotropic edges):")
    rng = np.random.default_rng(0)
    sl2 = AnisoBoundedSLAM(use_aniso=True)
    A = 6
    sl2.anchors = [rng.normal(0, 1, 3) for _ in range(A)]
    P0 = np.array(sl2.anchors)
    eds = []
    for i in range(A - 1):                          # seq chain (isotropic)
        Z = L.se2_mul(L.se2_inv(P0[i]), P0[i + 1]) + rng.normal(0, 0.05, 3)
        eds.append((i, i + 1, Z, 1 / 0.03, 1 / np.deg2rad(0.3), "seq"))
    for (a, b) in [(0, 3), (1, 4), (0, 5)]:         # loop edges (anisotropic)
        Z = L.se2_mul(L.se2_inv(P0[a]), P0[b]) + rng.normal(0, 0.1, 3)
        eds.append((a, b, Z, 1 / 0.2, 1 / np.deg2rad(2.0), "loop"))
        M = rng.normal(0, 1, (2, 2))
        Lam = M @ M.T + 0.5 * np.eye(2)             # random SPD
        sl2.aniso_L[(a, b)] = np.linalg.cholesky(Lam)
    free = np.arange(1, A)
    wl = {e: 0.3 + 0.6 * rng.random() for e in range(len(eds))}  # IRLS scales
    sl2._gn(eds, wl, free, P0)                       # populates resid/jac
    resid, jac = sl2._gn_resid, sl2._gn_jac
    x = P0[free].ravel() + rng.normal(0, 0.1, 3 * len(free))
    Ja = jac(x).toarray()
    Jn = np.zeros_like(Ja)
    eps = 1e-6
    for j in range(len(x)):
        xp = x.copy(); xp[j] += eps
        xm = x.copy(); xm[j] -= eps
        Jn[:, j] = (resid(xp) - resid(xm)) / (2 * eps)
    err = float(np.max(np.abs(Ja - Jn)))
    print(f"    max|analytic - finite-diff| = {err:.2e}")
    assert err < 1e-5, f"Jacobian mismatch {err:.2e}"

    # ---- (5) sanity: anisotropy actually changes the solve when engaged -------
    _, a_iso, _ = _bench_traj(AnisoBoundedSLAM, seed=1, world="corridor",
                              use_aniso=False)
    _, a_an, _ = _bench_traj(AnisoBoundedSLAM, seed=1, world="corridor",
                             use_aniso=True, cond_floor=0.1)
    print(f"[5] engaged-vs-off on corridor bench: off {a_iso:.4f} | "
          f"aniso {a_an:.4f} | delta {a_an - a_iso:+.4f} m")
    print("selftest PASSED")


# ===========================================================================
# REAL-LOG HARNESS (CARMEN driver, same params as runners/carmen.py)
# ===========================================================================
def run_log(path, use_aniso=True, cond_floor=0.1, iso_prune=False,
            aniso_gate=1.0, n_max=None, verbose=False):
    import sspslam.frontend as C
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)
    if n_max:
        keys = keys[:n_max]
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    slam = AnisoBoundedSLAM(use_aniso=use_aniso, cond_floor=cond_floor,
                            iso_prune=iso_prune, aniso_gate=aniso_gate,
                            robust=True, attempt_every=4, relax_every=25,
                            gap_kf=300, recent_aids=12)
    slam.store_dtype = np.complex64
    est = np.zeros((n, 3))
    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < 40.0, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = opose if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
    if slam.dirty:
        slam.relax()
    dt = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    n_loop = sum(1 for e in slam.edges if e[5] == "loop")
    n_an = len(slam.aniso_L)
    # ATE vs corrected reference (gfs), same eval as the shipped driver
    ref = C.parse_flaser(path.replace(".log", ".gfs.log"))
    rts = np.array([t for _, _, t in ref])
    rxy = np.stack([p[:2] for _, p, _ in ref])
    j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
    good = np.abs(rts - kts[j]) < 0.3
    al = C.align_se2(fin[j[good], :2], rxy[good])
    e = np.linalg.norm(al - rxy[good], axis=1)
    return dict(ate=float(np.sqrt((e ** 2).mean())), med=float(np.median(e)),
                emax=float(e.max()), loops=n_loop, aniso=n_an,
                trip=slam.n_aniso_trip, seen=slam.n_aniso_seen,
                pruned=slam.n_pruned, secs=dt, n=n)


LOGS = {
    "intel": "data/intel.log",
    "fr079": "data/fr079.log",
    "aces": "data/aces.log",
    "fr101": "data/fr101.log",
    "mit_real": "data/mit.log",
}


def logs_sweep(floors, which=("intel", "fr079", "aces", "fr101")):
    print("=== REAL-LOG ATE: shipped-isotropic vs anisotropic (condition-floor sweep) ===")
    print(f"{'log':<8} {'shipped(iso)':>13} {'loops':>6} | " +
          "  ".join(f"floor={f}" for f in floors))
    for name in which:
        path = LOGS[name]
        base = run_log(path, use_aniso=False)
        row = (f"{name:<8} {base['ate']:>10.3f} m  {base['loops']:>6} | ")
        cells = []
        for f in floors:
            r = run_log(path, use_aniso=True, cond_floor=f)
            cells.append(f"{r['ate']:.3f}({r['aniso']},{r['loops']})")
        print(row + "  ".join(cells), flush=True)
        sys.stdout.flush()
    print("\ncell = aniso_ATE(#anisotropic_edges, #loop_edges); "
          "floor=1.0 would be fully isotropic == shipped.")


def prunefix_sweep(floors, which=("intel", "fr079", "aces", "fr101")):
    """Coordinator's fix: rank LOO/chi2 PRUNING on the isotropic geometric
    residual (iso_prune=True) while keeping the anisotropic SOLVE, to stop the
    weak-geometry edge flood that regresses Intel. Table vs shipped and vs the
    un-fixed anisotropic."""
    print("=== PRUNING-FIX: anisotropic SOLVE + isotropic PRUNING (iso_prune) ===")
    print(f"{'log':<8} {'shipped':>8} | " +
          "  ".join(f"aniso/fix floor={f}" for f in floors))
    for name in which:
        path = LOGS[name]
        base = run_log(path, use_aniso=False)
        cells = []
        for f in floors:
            r0 = run_log(path, use_aniso=True, cond_floor=f, iso_prune=False)
            r1 = run_log(path, use_aniso=True, cond_floor=f, iso_prune=True)
            cells.append(f"{r0['ate']:.3f}({r0['loops']})/{r1['ate']:.3f}({r1['loops']})")
        print(f"{name:<8} {base['ate']:>6.3f}({base['loops']}) | "
              + "  ".join(cells), flush=True)
    print("\ncell = anisoNOfix_ATE(loops) / anisoFIX_ATE(loops); "
          "shipped shows ATE(loops). Target: FIX recovers Intel toward shipped "
          "(loops ~80) while fr079/fr101 keep their win.")


def gated_sweep(gates, cond_floor=0.1, which=("intel", "fr079", "aces", "fr101")):
    """Per-closure CONDITIONING GATE: only closures with l_weak/l_strong < gate
    get the anisotropic 2x2; well-conditioned closures stay fully isotropic
    (== shipped). Table vs shipped and vs the un-gated anisotropic (gate=1.0)."""
    print(f"=== CONDITIONING-GATED anisotropy (cond_floor={cond_floor}) ===")
    print(f"{'log':<8} {'shipped':>10}  {'ungated(g=1)':>13} | " +
          "  ".join(f"gate={g}" for g in gates))
    for name in which:
        path = LOGS[name]
        base = run_log(path, use_aniso=False)
        ung = run_log(path, use_aniso=True, cond_floor=cond_floor, aniso_gate=1.0)
        cells = []
        for g in gates:
            r = run_log(path, use_aniso=True, cond_floor=cond_floor, aniso_gate=g)
            trp = f"{r['trip']}/{r['seen']}"
            cells.append(f"{r['ate']:.3f}({r['loops']},trip {trp})")
        print(f"{name:<8} {base['ate']:>7.3f}({base['loops']})  "
              f"{ung['ate']:>7.3f}({ung['loops']}) | " + "  ".join(cells), flush=True)
    print("\ncell = ATE(loops, tripped/seen closures); shipped/ungated show ATE(loops). "
          "Target: gated Intel ~2.44 (few trips) while fr079/fr101 keep the win.")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        selftest()
    elif cmd == "logs":
        floors = [float(x) for x in sys.argv[2:]] or [0.05, 0.1, 0.15, 0.2]
        logs_sweep(floors)
    elif cmd == "prunefix":
        floors = [float(x) for x in sys.argv[2:]] or [0.1, 0.2]
        prunefix_sweep(floors)
    elif cmd == "gated":
        gates = [float(x) for x in sys.argv[2:]] or [0.3, 0.4, 0.5]
        gated_sweep(gates)
    elif cmd == "mit":
        import experiments.aniso_mit as M     # heavy oracle harness kept separate
        M.report(sys.argv[2] if len(sys.argv) > 2 else "data/mit.log")
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main()
