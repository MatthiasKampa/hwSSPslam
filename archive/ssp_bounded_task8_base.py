"""Bounded-memory continual SSP SLAM.

No sensor history: raw scan points exist only for the current frame. Each
keyframe is folded IMMEDIATELY (exact rotation, points still in hand) into its
5-keyframe segment's vector, encoded RIGIDLY in the anchor's frame, together
with the vector's d/dtheta (rotation derivative about the anchor origin).
Query-time anchor rotation = nearest-3-deg permutation + delta * derivative,
so the sub-grid residual is corrected analytically (O(delta^2) error), not
absorbed by matcher slack. Segments live in a coarse spatial grid capped per
cell, so MAP memory is O(area), not O(time). (Trajectory bookkeeping —
anchor poses, per-keyframe relative poses, one seq edge per anchor — still
grows O(time), as any trajectory output must.)

The frontend matches only against RECENT segments; old passes influence poses
exclusively through gated, robustified, prunable loop constraints.

Graph corrections move anchors only; segments ride along rigidly (nothing is
ever re-encoded), so between relaxations the representation cannot drift.
Constraints are pass-segregated (one contiguous segment chain at a time),
sequential edges are quadratic, loop edges get IRLS reweighting + chi-square
pruning. Runs continually from frame 0: attempts and relaxations on a fixed
schedule; a relaxation with no new edges is a no-op.

Usage: python3 ssp_bounded.py [quick|full]   (synthetic bench, GT labels)
"""

import sys
import time

import numpy as np
from scipy.sparse import csr_matrix, eye as speye
from scipy.sparse.linalg import spsolve

import ssp_slam as S
import ssp_slam_loop as L
from worlds import WORLDS
from bench_loop import multiloop_traj, FOV_BEAMS, BEAM, scan_at

ANCHOR = 5
GAP = 60
CELL, CELL_CAP = 2.0, 6


def sim_odometry(gt, sig_t=0.02, sig_r=np.deg2rad(0.5),
                 scale_bias=0.0, head_bias=0.0):
    """Noisy composed deltas; optional systematic scale / per-frame heading
    bias (real odometry is biased, not zero-mean)."""
    odo = gt.copy()
    for k in range(1, len(gt)):
        d = L.se2_mul(L.se2_inv(gt[k - 1]), gt[k])
        d[:2] = d[:2] * (1 + scale_bias) + S.RNG.normal(0, sig_t, 2)
        d[2] += head_bias + S.RNG.normal(0, sig_r)
        odo[k] = L.se2_mul(odo[k - 1], d)
    return odo


class BoundedSLAM:
    def __init__(self, robust=True, attempt_every=3, relax_every=5, gap_kf=GAP,
                 recent_aids=12):
        self.gap_kf = gap_kf
        # Band configuration. Abutting bands (recent_aids = gap_kf/ANCHOR) leave
        # no unmatched age band, BUT Intel A/B showed a small recency window
        # beats abutment even at the cost of a dead band: wide recency exposes
        # quadratic seq edges to drifted-geometry snaps (4.48 m with dead band
        # vs 7.10 m wide-recency vs 4.89 m abutting-small). Callers choose.
        self.recent_aids = recent_aids
        self.last_accept_k = 0
        # ENC_MAIN.W = W[:240] is the ring-major 4 x N_ANG polar lattice, so
        # coarse rotations are exact index permutations (one encode per match)
        self.matcher = S.Matcher(L.ENC_MAIN, t_half=0.48, rot_half_deg=9,
                                 rot_step_deg=1.5, perm=(4, L.N_ANG))
        self.cmatcher = S.Matcher(L.ENC_MAIN, t_half=0.72, rot_half_deg=9,
                                  rot_step_deg=1.5, perm=(4, L.N_ANG))
        self.robust, self.attempt_every, self.relax_every = robust, attempt_every, relax_every
        self.use_der = self.use_coh = self.use_innov = True   # ablation switches
        self.anchors = []            # anchor poses, list of np3 (graph nodes)
        self.kf_ref = []             # per kf: (anchor_id, rel pose in anchor frame)
        self.segvec = {}             # anchor_id -> consolidated vector (anchor frame)
        self.segder = {}             # anchor_id -> d/dtheta of that vector
        self.banned = set()          # anchor pairs whose edges were LOO-pruned
        self.diag = []               # per-candidate diagnostics (bench analysis)
        self.diag_gt = None
        self.cells = {}              # cell -> [anchor_id]
        self.edges = []              # (a, b, Z, wt, wr, kind)
        self.edge_seen = {}          # (a,b) -> index into edges, for re-measurement
        self.dirty = False
        self.k = -1
        self.max_jerk = 0.0
        self.n_relax = self.n_pruned = 0
        self.n_veto = self.n_innov_rej = 0   # mechanism fire counts
        self.inject_rate = 0.0               # bench: corrupt this fraction of Z
        self.inject_mode = "iid"             # "iid" | "aliased" (correlated)
        self._inj_rng = np.random.default_rng(777)
        self._alias = None
        self.n_evict = 0                     # cell-cap evictions (memory bound)
        self._sp_cache = (None, None)        # jac sparsity cache

    # ---- pose bookkeeping -------------------------------------------------
    def pose_of(self, k):
        aid, rel = self.kf_ref[k]
        return L.se2_mul(self.anchors[aid], rel)

    def aid_of(self, k):
        return self.kf_ref[k][0]

    # ---- map bundles ------------------------------------------------------
    def world_vec_seg(self, aid):
        a = self.anchors[aid]
        m = int(round(a[2] * L.N_ANG / np.pi))
        delta = a[2] - m * np.pi / L.N_ANG
        v = L.rot_permute(self.segvec[aid], m)
        if self.use_der and aid in self.segder:  # 1st-order rotation correction
            v = v + delta * L.rot_permute(self.segder[aid], m)
        return L.ENC.shift(a[:2]) * v

    def local_bundle(self, center, radius=8.0):
        """Frontend local map: RECENT segments only. Old passes must reach the
        pose only through gated loop edges, never through the frontend (else
        frontend snaps onto drifted old geometry and the resulting jump is
        baked into a quadratic, unprunable sequential edge)."""
        lo = len(self.anchors) - 1 - self.recent_aids
        B = np.zeros(L.W.shape[0], complex)
        for aid in self.segvec:
            if aid >= lo and np.linalg.norm(self.anchors[aid][:2] - center) < radius:
                B += self.world_vec_seg(aid)
        return B

    # ---- lifecycle --------------------------------------------------------
    def add_keyframe(self, pts, w, guess):
        self.k += 1
        k = self.k
        if k % ANCHOR == 0:
            self.anchors.append(guess.copy())
        aid = len(self.anchors) - 1
        est = guess
        fell_back = True
        if k > 0 and len(pts) >= 20:
            B = self.local_bundle(guess[:2])[L.MAIN]
            if np.abs(B).sum() > 0:
                cand = self.matcher.match(B, pts, w, guess)
                cand[2] = S.wrap(cand[2])
                if np.linalg.norm(cand[:2] - guess[:2]) < 0.45 \
                        and abs(S.wrap(cand[2] - guess[2])) < np.deg2rad(11):
                    est = cand
                    fell_back = False
        self._span_fallback = getattr(self, "_span_fallback", False) or fell_back
        if k % ANCHOR == 0:
            self.anchors[aid] = est.copy()
        rel = L.se2_mul(L.se2_inv(self.anchors[aid]), est)
        self.kf_ref.append((aid, rel))

        # consolidate IMMEDIATELY with exact rotation (points still in hand):
        # the segment vector never sees per-keyframe quantization
        if len(pts):
            PA = pts @ S._rot(rel[2]).T + rel[:2]      # anchor-frame points
            A = np.exp(1j * (PA @ L.W.T))
            v = A.T @ w
            # d/dtheta about the anchor origin, exact at frame time
            Cx = PA[:, 0:1] * L.W[:, 1] - PA[:, 1:2] * L.W[:, 0]
            vd = (1j * Cx * A).T @ w
            if aid not in self.segvec:
                self.segvec[aid] = np.zeros(L.W.shape[0], complex)
                self.segder[aid] = np.zeros(L.W.shape[0], complex)
                cell = tuple((self.anchors[aid][:2] // CELL).astype(int))
                self.cells.setdefault(cell, []).append(aid)
                if len(self.cells[cell]) > CELL_CAP:      # bounded by area
                    drop = self.cells[cell].pop(0)
                    self.segvec.pop(drop, None)
                    self.segder.pop(drop, None)
                    self.n_evict += 1
            self.segvec[aid] += v
            self.segder[aid] += vd

        if aid > 0 and k % ANCHOR == 0:   # sequential edge between anchors
            # frontend 5-frame relative accuracy is ~2-3 cm / 0.2-0.3 deg —
            # UNLESS any frame in the span fell back to raw odometry, which is
            # ~10x worse; an information-honest edge must reflect that
            Z = L.se2_mul(L.se2_inv(self.anchors[aid - 1]), self.anchors[aid])
            if getattr(self, "_span_fallback", False):
                st, sr = 0.10, np.deg2rad(1.5)
            else:
                st, sr = 0.03, np.deg2rad(0.3)
            self._span_fallback = False
            self.edges.append((aid - 1, aid, Z, 1 / st, 1 / sr, "seq"))

        if k % self.attempt_every == 0:
            self.try_constraint(pts, w)
        if k % self.relax_every == 0 and self.dirty:
            self.relax()
            return self.pose_of(k)   # post-relax pose, keeps chaining consistent
        return est

    # ---- pass-segregated constraints ---------------------------------------
    def try_constraint(self, pts, w):
        k = self.k
        if len(pts) < 20:
            return
        me = self.pose_of(k)
        my_aid = self.aid_of(k)
        cands = [aid for aid in self.segvec
                 if abs(aid - my_aid) > self.gap_kf // ANCHOR
                 and np.linalg.norm(self.anchors[aid][:2] - me[:2]) < 5.0]
        if not cands:
            return
        # group into contiguous chains (one pass each); score by proximity
        cands.sort()
        chains, cur = [], [cands[0]]
        for aid in cands[1:]:
            (cur.append(aid) if aid - cur[-1] <= 2 else (chains.append(cur), cur := [aid]))
        chains.append(cur)
        chain = min(chains, key=lambda ch: min(
            np.linalg.norm(self.anchors[a][:2] - me[:2]) for a in ch))
        B = sum(self.world_vec_seg(a) for a in chain)
        pose = self.cmatcher.match(B[L.MAIN], pts, w, me)
        pose[2] = S.wrap(pose[2])
        # gates well inside the matcher's reach (0.81 m / 10.5 deg), so
        # boundary-saturated (failed) matches cannot slip through
        if np.linalg.norm(pose[:2] - me[:2]) > 0.6 \
                or abs(S.wrap(pose[2] - me[2])) > np.deg2rad(7):
            return
        # per-ring coherence at the matched pose (fine rings corroborate)
        sv = L.ENC.shift(pose[:2]) * L.encode(pts @ S._rot(pose[2]).T, w)
        c = (np.conj(B) * sv).reshape(L.N_RING, L.N_ANG)
        Br = B.reshape(L.N_RING, L.N_ANG)
        svr = sv.reshape(L.N_RING, L.N_ANG)
        coh = c.sum(1).real / (np.linalg.norm(Br, axis=1)
                               * np.linalg.norm(svr, axis=1) + 1e-12)
        c_aid = chain[len(chain) // 2]     # attribute to chain center
        lever = np.linalg.norm(me[:2] - self.anchors[c_aid][:2])
        sig_t = np.sqrt(0.08 ** 2 + (0.05 * lever) ** 2)
        sig_r = np.deg2rad(2.0)
        if self.diag_gt is not None:       # bench diagnostics with GT label
            gtk = self.diag_gt
            Zm = L.se2_mul(L.se2_inv(self.anchors[c_aid]),
                           L.se2_mul(pose, L.se2_inv(self.kf_ref[k][1])))
            Zt = L.se2_mul(L.se2_inv(gtk[c_aid * ANCHOR]), gtk[k - (k % ANCHOR)])
            err = np.linalg.norm(Zm[:2] - Zt[:2])
            self.diag.append([err, np.linalg.norm(pose[:2] - me[:2]),
                              abs(S.wrap(pose[2] - me[2])), len(chain), *coh])
        if self.use_coh and coh[:2].mean() < 0.02:  # fine-ring corroboration veto
            self.n_veto += 1
            return
        _, rel = self.kf_ref[k]
        Zk = L.se2_mul(pose, L.se2_inv(rel))          # implied anchor pose of k
        Z = L.se2_mul(L.se2_inv(self.anchors[c_aid]), Zk)
        if self.inject_rate and self._inj_rng.random() < self.inject_rate:
            # separate RNG keeps sim streams paired across configs
            if self.inject_mode == "aliased":
                # correlated aliasing: ONE fixed wrong transform, repeatedly
                # proposed (the outlier model that actually breaks SLAM)
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
        # pre-insertion innovation gate: a wrong Z is many sigma from the
        # current arrangement BEFORE the optimizer can bend to hide it.
        # The allowance reflects drift ACCUMULATED SINCE THE LAST ACCEPTED
        # CLOSURE (raw keyframe-gap allowances saturate above the match-gate
        # reach and make the test vacuous).
        Zc = L.se2_mul(L.se2_inv(self.anchors[c_aid]), self.anchors[my_aid])
        since = k - self.last_accept_k
        s_at = sig_t + min(0.30, 0.002 * since)
        s_ar = sig_r + min(np.deg2rad(6), np.deg2rad(0.03) * since)
        chi = (np.linalg.norm(Z[:2] - Zc[:2]) / s_at) ** 2 \
            + (S.wrap(Z[2] - Zc[2]) / s_ar) ** 2
        if self.use_innov and chi > 9.0:
            self.n_innov_rej += 1
            return
        edge = (c_aid, my_aid, Z, 1 / sig_t, 1 / sig_r, "loop")
        key = (c_aid, my_aid)
        if key in self.banned:
            return
        if key in self.edge_seen:          # re-measure: replace, don't keep stale
            self.edges[self.edge_seen[key]] = edge
        else:
            self.edge_seen[key] = len(self.edges)
            self.edges.append(edge)
        self.dirty = True
        self.last_accept_k = k

    # ---- backend: quadratic seq, IRLS + chi2-pruned loops -------------------
    def relax(self):
        self.dirty = False
        self.n_relax += 1
        A = len(self.anchors)
        P0 = np.array(self.anchors)
        edges = self.edges

        def solve(eds, wl):
            E = len(eds)
            if E == 0 or A < 2:
                return P0.copy(), np.zeros(0)
            aa = np.array([e[0] for e in eds])
            bb = np.array([e[1] for e in eds])
            Zt = np.stack([e[2] for e in eds])
            wts = np.array([e[3] for e in eds])
            wrs = np.array([e[4] for e in eds])
            ss = np.array([wl.get(e, 1.0) for e in range(E)])
            wt_s, wr_s = wts * ss, wrs * ss

            def resid(x):
                P = np.vstack([P0[0], x.reshape(-1, 3)])
                ca, sa = np.cos(P[aa, 2]), np.sin(P[aa, 2])
                dx = P[bb, 0] - P[aa, 0]
                dy = P[bb, 1] - P[aa, 1]
                out = np.empty((E, 3))
                out[:, 0] = (ca * dx + sa * dy - Zt[:, 0]) * wt_s
                out[:, 1] = (-sa * dx + ca * dy - Zt[:, 1]) * wt_s
                out[:, 2] = S.wrap(P[bb, 2] - P[aa, 2] - Zt[:, 2]) * wr_s
                return out.ravel()

            # analytic sparse Jacobian: 12 potential nonzeros per edge
            # (rows r0,r1 x cols xa,ya,tha,xb,yb; row r2 x cols tha,thb);
            # gauge fix = node 0 excluded. Structure depends on edges only.
            ck = (E, A, int(aa.sum()), int(bb.sum()))
            if self._sp_cache[0] != ck:
                ro = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
                comp = np.array([0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 2])
                isb = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1], bool)
                node = np.where(isb, bb[:, None], aa[:, None])      # E x 12
                keep = (node > 0).ravel()
                rows = (3 * np.arange(E)[:, None] + ro).ravel()[keep]
                cols = ((3 * (node - 1) + comp)).ravel()[keep]
                self._sp_cache = (ck, (rows, cols, keep))
            rows, cols, keep = self._sp_cache[1]

            def jac(x):
                P = np.vstack([P0[0], x.reshape(-1, 3)])
                ca, sa = np.cos(P[aa, 2]), np.sin(P[aa, 2])
                dx = P[bb, 0] - P[aa, 0]
                dy = P[bb, 1] - P[aa, 1]
                V = np.empty((E, 12))
                V[:, 0], V[:, 1], V[:, 2] = -ca, -sa, -sa * dx + ca * dy
                V[:, 3], V[:, 4] = ca, sa
                V[:, 5], V[:, 6], V[:, 7] = sa, -ca, -(ca * dx + sa * dy)
                V[:, 8], V[:, 9] = -sa, ca
                V[:, :10] *= wt_s[:, None]
                V[:, 10], V[:, 11] = -wr_s, wr_s
                return csr_matrix((V.ravel()[keep], (rows, cols)),
                                  shape=(3 * E, 3 * (A - 1)))

            # damped Gauss-Newton on the normal equations J^T J + lambda I
            x = P0[1:].ravel().copy()
            r = resid(x)
            cost = float(r @ r)
            lam = 1e-6
            I = speye(3 * (A - 1), format="csr")
            for _ in range(6):
                J = jac(x)
                g = J.T @ r
                if np.max(np.abs(g)) < 1e-12:
                    break
                step = spsolve((J.T @ J + lam * I).tocsc(), -g)
                rn = resid(x + step)
                cn = float(rn @ rn)
                if cn <= cost:
                    x, r, cost = x + step, rn, cn
                    lam = max(1e-9, lam * 0.1)
                    if np.max(np.abs(step)) < 1e-10:
                        break
                else:
                    lam *= 100.0
            return np.vstack([P0[0], x.reshape(-1, 3)]), r

        P, r = solve(edges, {})
        if self.robust:
            wl = {}
            for e, (a, b, Z, wt, wr, kind) in enumerate(edges):
                if kind != "loop":
                    continue
                rn = np.linalg.norm(r[3 * e:3 * e + 3])
                wl[e] = 1.0 / np.sqrt(1.0 + (rn / 2.0) ** 2)   # Cauchy-ish IRLS
            P, r = solve(edges, wl)
            # leave-one-out pruning: the graph bends to hide outliers, so test
            # each suspicious edge against the solution computed WITHOUT it.
            # Rank by UNWEIGHTED residual (the IRLS-weighted statistic is
            # capped near 2.0 and cannot rank) and take the worst few.
            raws = {e: np.linalg.norm(r[3 * e:3 * e + 3]) / max(wl.get(e, 1.0), 1e-9)
                    for e, (a, b, Z, wt, wr, kind) in enumerate(edges)
                    if kind == "loop"}
            suspicious = sorted((e for e, v in raws.items() if v > 2.0),
                                key=lambda e: -raws[e])
            bad = set()
            for e in suspicious[:6]:      # bounded work per relaxation
                a, b, Z, wt, wr, kind = edges[e]
                P2, _ = solve([ed for i, ed in enumerate(edges) if i != e], {})
                d = S._rot(-P2[a, 2]) @ (P2[b, :2] - P2[a, :2])
                rn = np.linalg.norm(np.array([*((d - Z[:2]) * wt),
                                              S.wrap(P2[b, 2] - P2[a, 2] - Z[2]) * wr]))
                if rn > 3.0:
                    bad.add(e)
                    self.banned.add((a, b))
            if bad:
                self.n_pruned += len(bad)
                self.edges = [ed for e, ed in enumerate(edges) if e not in bad]
                edges = self.edges
                self.edge_seen = {(a, b): i for i, (a, b, *_, kind) in
                                  enumerate(edges) if kind == "loop"}
                P, r = solve(edges, {})
                wl2 = {e: 1.0 / np.sqrt(1.0 + (np.linalg.norm(r[3*e:3*e+3]) / 2.0) ** 2)
                       for e, (a, b, Z, wt, wr, kind) in enumerate(edges)
                       if kind == "loop"}
                P, r = solve(edges, wl2)   # end on a robust solve, not quadratic
        jerk = float(np.max(np.linalg.norm(P[:, :2] - P0[:, :2], axis=1)))
        self.max_jerk = max(self.max_jerk, jerk)
        for i in range(A):
            self.anchors[i] = P[i]

    def memory_kb(self):
        # segvec + segder, both complex128 D-vectors per segment
        return len(self.segvec) * L.W.shape[0] * 16 * 2 / 1024


# --------------------------------------------------------------------------
# Bench
# --------------------------------------------------------------------------

def run(world="room", n=750, seed=1, use_graph=True, robust=True, laps=3,
        ablate=None, inject=0.0, dropout=0.0, scale_bias=0.0, head_bias=0.0):
    S.RNG = np.random.default_rng(seed)
    segs = WORLDS[world]()
    gt = multiloop_traj(n, laps=laps)
    odo = sim_odometry(gt, scale_bias=scale_bias, head_bias=head_bias)
    slam = BoundedSLAM(robust=robust)
    slam.diag_gt = gt
    slam.inject_rate = inject
    if ablate:
        setattr(slam, ablate, False)
    if not use_graph:
        slam.attempt_every = 10 ** 9
    # frontend dropout: contiguous 10-frame windows where scan matching is
    # unavailable and the pose falls back to raw odometry composition
    drop = np.zeros(n, bool)
    if dropout > 0:
        for s0 in S.RNG.choice(n - 10, int(dropout * n / 10), replace=False):
            drop[s0:s0 + 10] = True
    est = np.zeros((n, 3))
    t0 = time.time()
    for k in range(n):
        r = scan_at(segs, gt[k])
        pts, w, _ = S.scan_to_samples(r, BEAM)
        if drop[k]:
            pts, w = pts[:0], w[:0]   # <20 pts -> frontend skips matching
        guess = odo[0] if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odo[k - 1]), odo[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
    # final poses after all corrections (derived from anchors)
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    ate = np.linalg.norm(fin[:, :2] - gt[:, :2], axis=1)
    n_loop = sum(1 for e in slam.edges if e[5] == "loop")
    # GT-label surviving loop edges
    false = 0
    for a, b, Z, wt, wr, kind in slam.edges:
        if kind != "loop":
            continue
        # approximate GT anchor poses by GT at anchor keyframes
        Zt = L.se2_mul(L.se2_inv(gt[a * ANCHOR]), gt[b * ANCHOR])
        if np.linalg.norm(Z[:2] - Zt[:2]) > 0.30 or abs(S.wrap(Z[2] - Zt[2])) > np.deg2rad(3):
            false += 1
    odo_ate = np.linalg.norm(odo[:, :2] - gt[:, :2], axis=1)
    return dict(ate=np.sqrt((ate ** 2).mean()), ate_max=ate.max(),
                odo=np.sqrt((odo_ate ** 2).mean()), edges=n_loop, false=false,
                pruned=slam.n_pruned, relax=slam.n_relax, jerk=slam.max_jerk,
                veto=slam.n_veto, innov=slam.n_innov_rej,
                mem=slam.memory_kb(), secs=time.time() - t0)


if __name__ == "__main__":
    quick = (sys.argv[1] if len(sys.argv) > 1 else "quick") == "quick"
    seeds = (1, 2, 3) if quick else tuple(range(1, 9))
    worlds = ["room"] if quick else ["room", "office", "corridor", "sparse"]
    CONFIGS = [("frontend-only", dict(use_graph=False)),
               ("baseline", dict()),
               ("no robust", dict(robust=False)),
               ("ablate deriv corr", dict(ablate="use_der")),
               ("inject 10% outliers", dict(inject=0.10)),
               ("inject, no protections", dict(inject=0.10, ablate="use_innov")),
               ("dropout 10% + odo bias", dict(dropout=0.10, scale_bias=0.01,
                                               head_bias=np.deg2rad(0.03)))]
    for world in worlds:
        print(f"== {world}")
        base = {}
        for tag, kw in CONFIGS:
            rs = [run(world, seed=s, **kw) for s in seeds]
            ates = np.array([r["ate"] for r in rs])
            if tag == "frontend-only":
                base = ates
            d = ates - base
            print(f"{tag:<24} ATE {100 * ates.mean():6.1f} cm "
                  f"(paired vs frontend {100 * d.mean():+6.1f}, "
                  f"{np.sum(d < 0)}/{len(d)} better)  "
                  f"edges {np.mean([r['edges'] for r in rs]):5.1f}  "
                  f"false {np.mean([r['false'] for r in rs]):4.1f}  "
                  f"pruned {np.mean([r['pruned'] for r in rs]):4.1f}  "
                  f"veto {np.mean([r['veto'] for r in rs]):4.1f}  "
                  f"innov-rej {np.mean([r['innov'] for r in rs]):4.1f}  "
                  f"jerk {np.mean([r['jerk'] for r in rs]):4.2f}", flush=True)
