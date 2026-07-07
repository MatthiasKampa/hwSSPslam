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
from collections import deque

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

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
        self.diag = []               # per-candidate diagnostics: [gt_err,
        #   innov_t, innov_r, chain_len, coh_0..5, l_weak, l_strong, u_weak_xy,
        #   err_weak, err_strong, innov_weak, ridge_w, ridge_s, coh_ref,
        #   accepted]; GT fields are NaN unless diag_gt is set
        self.diag_gt = None
        self.coh_ref = None          # EMA of frontend-match fine-ring coherence
        # Session-relative coherence response. ratio = coh01 / coh_ref.
        # coh_soft=False: hard veto below coh_target * coh_ref (legacy
        # behavior). coh_soft=True (DEFAULT): the SOFT response — low-
        # coherence matches are kept but sigma-inflated
        # (coh_target/ratio)^infl_pow (x ill_mult when the degeneracy
        # indicators corroborate an aperture failure), capped at
        # coh_infl_cap; hard vetoes remain only below coh_floor / for
        # ill-conditioned matches below ill_floor. The soft response is
        # SMOOTH in the coh_target knob (Intel sweep 0.45/0.55/0.65 within
        # ~15%, where the hard veto swings 2.9 -> 7.3 between 0.55 and 0.60).
        # SHIPPED DEFAULT soft @ 0.55, chosen WITH THE TRF SOLVER by
        # minimizing the worst-of-three-logs ratio to each log's best-known
        # result (2026-07 acceptance matrix, deterministic, ATE rmse m):
        #   Intel  hard 4.78 | soft@0.55 2.44 | soft@0.65 5.97  (best 2.44)
        #   fr079  hard 7.26 | soft@0.55 5.52 | soft@0.65 10.36 (best 2.80)
        #   ACES   hard 5.94 | soft@0.55 6.21 | soft@0.65 6.11  (odo 5.41)
        # worst-of-three ratios: hard 2.59, soft@0.55 1.97, soft@0.65 3.70.
        # The earlier "soft costs accuracy on Intel" finding (3.2-5.5 vs
        # 2.8-2.9 m) was a damped-GN-era artifact; under TRF step control
        # the pairing reverses (see the note in try_constraint).
        self.coh_target = 0.55       # veto threshold / soft-response knob
        self.coh_floor = 0.20        # hard veto below this ratio
        self.ill_floor = 0.35        # hard veto below this ratio IF the match
        #   also carries the aperture-failure signature (see ill_* below);
        #   fixed thresholds, independent of the coh_target knob
        self.coh_infl_cap = 40.0     # max sigma inflation factor
        self.infl_pow = 2.0          # ramp steepness: infl=(target/ratio)^pow
        self.coh_soft = True         # soft sigma-inflation response (default;
        #   False = legacy hard veto — see shipped-default note above)
        # Degeneracy corroboration: mid-band coherence on a sharp isotropic
        # peak is the genuine-closure signature (Intel closures against
        # internally-drifted old bundles), so skip inflation when the
        # already-computed Hessian/ridge indicators say well-conditioned.
        self.coh_degen_ok = True
        self.aniso_min = 0.40        # l_weak/l_strong above = isotropic peak
        self.ridge_max = 0.60        # weak-dir off-peak score below = no plateau
        self.ridge_s_max = 0.50      # strong dir too: corridor junk that fakes
        #   isotropy still shows a high strong-dir ridge (measured: all 7 such
        #   impostors in 16 GT runs had ridge_s >= 0.50; Intel genuine
        #   accepted population sits at ridge_s p95 = 0.48)
        self.coh_mid = 0.35          # exemption only above this ratio
        # ILL tier: sub-target candidates carrying the aperture-failure
        # signature (weak-dir plateau / collapsed curvature / strong-dir
        # ridge) get ill_mult-times-heavier inflation. Mid-weight wrong edges
        # are the worst case: they evade the innovation gate, IRLS and LOO
        # pruning (all keyed to whitened residuals) yet still pull — measured
        # on Intel, plain **2 inflation scored 4.43-5.49 vs 3.32 with
        # full-weight junk that the backend could SEE and prune. Calibration
        # (16 GT runs): rule catches 38/40 corridor mid-band junk, 42/44
        # all-world mid-band junk; thresholds are target-independent so the
        # knob response stays smooth.
        self.ill_aniso = 0.30
        self.ill_ridge_w = 0.50
        self.ill_ridge_s = 0.50
        self.ill_mult = 8.0
        # Feedback cap, STREAK-GATED: only when candidates are being vetoed /
        # heavily inflated repeatedly in the same region (>= streak_n
        # consecutive suppressions within streak_r of each other) is
        # last_accept_k advanced to k - since_cap, so a veto storm cannot
        # ride the full innovation allowance indefinitely. Streak-gating
        # follows the failure geometry: isolated vetoes between genuine
        # closures (Intel at target 0.55) must not trim the 0.30 m drift
        # allowance those closures need, while sustained storms (corridors;
        # Intel at target 0.60, the old 7.3 m blowup) are exactly runs of
        # same-region suppressions.
        self.since_cap = 100
        self.streak_n = 3
        self.streak_r = 3.0
        self._streak = 0             # consecutive suppressed candidates
        self._streak_xy = None       # position of the last suppressed one
        # only an UNinflated accept resets the drift clock: an edge at k-x
        # sigma corrects ~1/k of the drift, and letting it reset the clock
        # starves later genuine closures at the innovation gate (measured on
        # Intel: full reset for infl<=4 drove innov-rej 3x up, ATE 2.9->6.6)
        self.infl_weak = 1.0
        self.n_inflate = 0
        self.cells = {}              # cell -> [anchor_id]
        self.edges = []              # (a, b, Z, wt, wr, kind)
        self.edge_seen = {}          # (a,b) -> index into edges, for re-measurement
        self.dirty = False
        self.k = -1
        self.max_jerk = 0.0
        # Explicit Tikhonov prior toward the PRE-RELAX anchor poses (P0 of
        # each _gn call): whitened rows lam_t*(x-x0), lam_t*(y-y0),
        # lam_r*wrap(th-th0) per free anchor. lam_t == 0 disables the prior
        # and falls back to the audited max_nfev=30 early-stop regularizer.
        # DEFAULT-OFF after a multi-log sweep (2026-07-07, RESULTS.md
        # "Tikhonov prior sweep"): lam_t in {0.5,1,2,4} m^-1 with
        # lam_r=0.5*lam_t all CONVERGE (nfev cap never hit at 200) yet lose
        # to early stopping on fr079 (best 9.48 m at lam_t=1 vs 5.52
        # shipped; worst-of-three ratio 3.39 vs 1.97). Early stopping is not
        # equivalent to a quadratic prior toward P0: it truncates the
        # correction along ALL directions path-dependently (well-conditioned
        # components move first), and the partially-converged states feed the
        # IRLS/LOO/veto cascade, changing which loop edges survive.
        self.prior_lam_t = 0.0       # 1/m
        self.prior_lam_r = 0.0       # 1/rad
        self.n_nfev_cap = 0          # solves that exhausted max_nfev (status 0)
        self.n_relax = self.n_pruned = 0
        self.n_veto = self.n_innov_rej = 0   # mechanism fire counts
        self.inject_rate = 0.0               # bench: corrupt this fraction of Z
        self.inject_mode = "iid"             # "iid" | "aliased" (correlated)
        self._inj_rng = np.random.default_rng(777)
        self._alias = None
        self.n_evict = 0                     # cell-cap evictions (memory bound)
        # ---- O(1)-per-frame backend state ----
        self.windowed = False                # opt-in windowed relax (see relax())
        self.retired = set()                 # marginalized anchors: frozen, no edges
        self.pending_new = []                # loop pairs accepted since last relax
        self.full_every = 20                 # global bleed: full solve cadence
        self.n_win = self.n_grow = self.n_full = 0
        self.solve_log = []                  # (frame, n_free, level) per solve
        self._last_nfree = 0

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
                    # fine-ring coherence of the ACCEPTED frontend match: the
                    # session's own baseline for how coherent a genuine
                    # alignment looks in this domain (crisp synthetic walls
                    # ~0.8, real cluttered lidar ~0.25). The loop-constraint
                    # veto is calibrated RELATIVE to this (EMA), not to an
                    # absolute scale that cannot transfer across domains.
                    W01 = L.W[:2 * L.N_ANG]
                    PW = pts @ S._rot(est[2]).T + est[:2]
                    sv01 = np.exp(1j * (PW @ W01.T)).T @ w
                    B01 = B[:2 * L.N_ANG].reshape(2, L.N_ANG)
                    s01 = sv01.reshape(2, L.N_ANG)
                    c01 = float(np.mean(
                        (np.conj(B01) * s01).sum(1).real
                        / (np.linalg.norm(B01, axis=1)
                           * np.linalg.norm(s01, axis=1) + 1e-12)))
                    self.coh_ref = c01 if self.coh_ref is None \
                        else 0.95 * self.coh_ref + 0.05 * c01
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
                    if self.windowed:   # marginalization serves the unbounded
                        self._maybe_retire(drop)   # regime; full-solve mode
                                                   # keeps all anchors free
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
    def _suppress(self, xy, k):
        """Record a coherence-suppressed candidate (vetoed or inflated past
        infl_weak). Once streak_n consecutive suppressions land within
        streak_r of each other, cap the innovation-allowance clock: a region
        that keeps producing junk must not accumulate an ever-looser gate."""
        if self._streak_xy is not None \
                and np.linalg.norm(xy - self._streak_xy) < self.streak_r:
            self._streak += 1
        else:
            self._streak = 1
        self._streak_xy = xy.copy()
        if self._streak >= self.streak_n:
            self.last_accept_k = max(self.last_accept_k, k - self.since_cap)

    def _coh_response(self, coh, l_weak, l_strong, ridge_w, ridge_s, me, k):
        """Soft (coh_soft=True) response to the coherence ratio: returns a
        sigma-inflation factor >= 1, or -1 for a hard veto. Three tiers below
        coh_target, split by the translation-Hessian / ridge indicators:
        WELL  — sharp isotropic peak with mid-band coherence (the genuine-
                closure signature on Intel): no inflation;
        ILL   — aperture-failure signature (weak-dir plateau, collapsed
                curvature or strong-dir ridge): vetoed below ill_floor,
                ill_mult-times-heavier inflation above it;
        else  — the gentle ramp (coh_target/ratio)**infl_pow.
        All geometry cutoffs are independent of the coh_target knob, so the
        knob's response stays smooth."""
        ratio = coh[:2].mean() / max(self.coh_ref, 1e-12)
        ill = (self.coh_degen_ok
               and (ridge_w > self.ill_ridge_w
                    or ridge_s > self.ill_ridge_s
                    or l_weak < self.ill_aniso * max(l_strong, 1e-12)))
        if ratio < self.coh_floor or (ill and ratio < self.ill_floor):
            # junk zone: far below any genuine population, or low-ish
            # coherence WITH the aperture-failure signature (corridor junk
            # reaches ratio ~0.46, but 38/40 of it carries the signature)
            self.n_veto += 1
            self._suppress(me[:2], k)
            return -1.0
        if ratio >= self.coh_target:
            return 1.0
        well = (self.coh_degen_ok and ratio >= self.coh_mid
                and l_weak > self.aniso_min * max(l_strong, 1e-12)
                and ridge_w < self.ridge_max
                and ridge_s < self.ridge_s_max)
        if well:
            return 1.0
        infl = (self.coh_target / max(ratio, 0.05)) ** self.infl_pow
        if ill:
            infl *= self.ill_mult
        self.n_inflate += 1
        return min(self.coh_infl_cap, infl)

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
        # translation-score curvature at the matched pose (aperture/degeneracy
        # detector): score(d) = Re sum_k c_k exp(i w_k . d), so the negative
        # Hessian is K = sum_k Re(c_k) w_k w_k^T — analytic, no probe grid.
        # Encodings of wall points concentrate energy in frequencies
        # PERPENDICULAR to the wall, so along-corridor curvature collapses
        # exactly when translation along the corridor is unobservable.
        # MAIN rings (lam 0.25..2) = what the matcher actually optimizes.
        cM = c[:4].ravel()
        nrmM = (np.linalg.norm(Br[:4].ravel())
                * np.linalg.norm(svr[:4].ravel()) + 1e-12)
        K = (L.W[L.MAIN] * (cM.real / nrmM)[:, None]).T @ L.W[L.MAIN]
        evals, evecs = np.linalg.eigh(K)          # ascending: [weak, strong]
        l_weak, l_strong = float(evals[0]), float(evals[1])
        u_weak = evecs[:, 0]                      # world-frame weak direction
        # ridge probe: translation ambiguity is a PLATEAU at scales beyond the
        # peak curvature (corridor: the score stays high sliding along the
        # walls). Sample the matcher's own objective along each eigendirection
        # out to +-1.2 m; ridge = best off-peak score (>=0.35 m out, i.e.
        # beyond a genuine peak's support) relative to the peak.
        ts = np.arange(-1.2, 1.21, 0.06)
        s0 = float(cM.real.sum())                 # score at the matched pose
        far = np.abs(ts) >= 0.35
        ridge = [0.0, 0.0]
        for j, u in enumerate((u_weak, evecs[:, 1])):
            proj = L.W[L.MAIN] @ u
            sc = (np.exp(1j * ts[:, None] * proj[None, :]) @ cM).real
            ridge[j] = float(sc[far].max() / max(s0, 1e-12))
        ridge_w, ridge_s = ridge
        c_aid = chain[len(chain) // 2]     # attribute to chain center
        lever = np.linalg.norm(me[:2] - self.anchors[c_aid][:2])
        sig_t = np.sqrt(0.08 ** 2 + (0.05 * lever) ** 2)
        sig_r = np.deg2rad(2.0)
        err = e_weak = e_strong = np.nan   # GT fields (bench only)
        if self.diag_gt is not None:       # GT labels available (bench)
            gtk = self.diag_gt
            Zm = L.se2_mul(L.se2_inv(self.anchors[c_aid]),
                           L.se2_mul(pose, L.se2_inv(self.kf_ref[k][1])))
            Zt = L.se2_mul(L.se2_inv(gtk[c_aid * ANCHOR]), gtk[k - (k % ANCHOR)])
            err = np.linalg.norm(Zm[:2] - Zt[:2])
            # decompose the GT error into weak/strong eigendirections (world
            # frame; anchor-frame error rotated by the anchor heading)
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
        # Fine-ring corroboration, calibrated RELATIVE to the session's
        # frontend baseline (coh_ref). ROC evidence (12 runs, 4 worlds, GT
        # labels): corridor aperture-failure candidates sit at ratio <= 0.46
        # (p95) while genuine candidates sit at >= 0.73 (p5) in every world;
        # absolute coherence does NOT transfer (crisp synthetic walls ~0.8 vs
        # real cluttered lidar ~0.24 for equally genuine matches). The 0.55
        # factor is pinned from above by Intel, whose genuine closures match
        # against internally-drifted old bundles (accepted-ratio median 0.49).
        # coh_soft=True (DEFAULT, see the shipped-default note in __init__)
        # is the sigma-inflation response, smooth in the coh_target knob.
        # The hard veto's old Intel edge (2.8-2.9 vs 3.2-5.5 m) was measured
        # under the damped-GN solver; with TRF step control the soft
        # response wins on Intel AND fr079 (2.44/5.52 vs 4.78/7.26) — the
        # hard veto starves the floppy graphs of the closures TRF needs.
        infl = 1.0
        if self.use_coh and self.coh_ref is not None:
            if not self.coh_soft:                  # legacy hard veto
                if coh[:2].mean() < self.coh_target * self.coh_ref:
                    self.n_veto += 1
                    return
            else:
                infl = self._coh_response(coh, l_weak, l_strong,
                                          ridge_w, ridge_s, me, k)
                if infl < 0:
                    return                         # soft-path veto
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
        # inflation applies to the EDGE weights only, not to the innovation
        # allowance above: the gate stays exactly as strict as before, while
        # a low-coherence match enters as a nearly-uninformative constraint
        # instead of triggering the veto-starvation feedback
        edge = (c_aid, my_aid, Z, 1 / (sig_t * infl), 1 / (sig_r * infl), "loop")
        key = (c_aid, my_aid)
        if key in self.banned:
            return
        if key in self.edge_seen:          # re-measure: replace, don't keep stale
            # ... unless the replacement is MATERIALLY weaker: an inflated
            # low-coherence re-measurement must not overwrite a full-
            # confidence edge. The 3.5x margin exceeds the max lever-arm
            # weight spread (0.263/0.08 = 3.3x), so uninflated re-
            # measurements always replace, exactly as before.
            if edge[3] * 3.5 < self.edges[self.edge_seen[key]][3]:
                return
            self.edges[self.edge_seen[key]] = edge
        else:
            self.edge_seen[key] = len(self.edges)
            self.edges.append(edge)
        if infl <= self.infl_weak:
            self.dirty = True
            self.pending_new.append(key)   # seeds the next relax window
            self.last_accept_k = k         # full-confidence closure
            self._streak = 0
            self._streak_xy = None
        else:
            # PASSIVE insertion: an inflated edge carries (1/infl)^2 of the
            # information, so it must not trigger backend work of its own —
            # it participates whenever a full-confidence edge causes a relax.
            # Measured on Intel: letting inflated accepts set `dirty` drove
            # ~10x more relaxations whose anchor jitter feeds back through
            # the frontend (ATE 4.3-5.5 vs 2.8 hard veto EVEN at 400x
            # inflation, where the edges' pull is provably negligible).
            # It also must not reset the drift clock (it barely corrects
            # drift); repeated suppression in one region is streak-clamped
            # exactly like the veto path.
            self._suppress(me[:2], k)
        self.diag[-1][-1] = 1.0            # mark this candidate accepted

    # ---- seq-edge marginalization on cell-cap eviction ----------------------
    def _maybe_retire(self, drop):
        """When a segment is evicted AND its anchor carries no live loop edge
        AND it is outside the recency band, marginalize it: compose the two
        adjacent seq edges Z1 (p->drop), Z2 (drop->q) into a bypass p->q and
        freeze the anchor. Anchor indices stay stable: `retired` anchors are
        excluded from future solve unknowns; pose_of stays valid via the
        frozen anchor pose."""
        if drop >= len(self.anchors) - 1 - self.recent_aids:
            return
        e_in = e_out = -1
        for i, (a, b, Z, wt, wr, kind) in enumerate(self.edges):
            if kind == "loop":
                if a == drop or b == drop:
                    return                # live loop edge: keep in the graph
            elif b == drop:
                e_in = i
            elif a == drop:
                e_out = i
        if e_in < 0 or e_out < 0:         # chain end / node 0: nothing to bypass
            return
        p, _, Z1, wt1, wr1, _ = self.edges[e_in]
        _, q, Z2, wt2, wr2, _ = self.edges[e_out]
        Z = L.se2_mul(Z1, Z2)
        st1, sr1, st2, sr2 = 1 / wt1, 1 / wr1, 1 / wt2, 1 / wr2
        # first-order covariance composition: rotation noise of the first leg
        # levers the translation of the second
        st = np.sqrt(st1 ** 2 + st2 ** 2 + (np.linalg.norm(Z2[:2]) * sr1) ** 2)
        sr = np.sqrt(sr1 ** 2 + sr2 ** 2)
        self.edges = [e for i, e in enumerate(self.edges) if i not in (e_in, e_out)]
        self.edges.append((p, q, Z, 1 / st, 1 / sr, "seq"))
        self.edge_seen = {(a, b): i for i, (a, b, *_, kind) in
                          enumerate(self.edges) if kind == "loop"}
        self.retired.add(drop)

    # ---- backend: windowed relax, quadratic seq, IRLS + chi2-pruned loops ---
    def _gn(self, eds, wl, free, P0):
        """Sparse nonlinear least squares over the anchor subset `free` (all
        other anchors fixed at P0 — boundary anchors act through crossing
        edges as priors on the interior endpoint). Returns (full pose array
        with free rows updated, whitened residual vector over eds; the
        Tikhonov prior rows, if enabled, are appended after the edge rows in
        the objective but stripped from the returned residual).

        Step control is scipy's trust-region reflective (TRF); the Jacobian
        stays ANALYTIC and sparse (that is the 9x relax speedup vs finite
        differences). A hand-rolled damped Gauss-Newton on the normal
        equations was tried first and over-converges along flat-valley
        directions on floppy graphs (fr079: 10.4 m vs 2.8 m with TRF step
        control; well-conditioned graphs agree to ~1e-8, see the bisect
        notes), so plain GN steps are gone for good."""
        E, Fn = len(eds), len(free)
        if E == 0 or Fn == 0:
            return P0.copy(), np.zeros(3 * E)
        aa = np.array([e[0] for e in eds])
        bb = np.array([e[1] for e in eds])
        Zt = np.stack([e[2] for e in eds])
        wts = np.array([e[3] for e in eds])
        wrs = np.array([e[4] for e in eds])
        ss = np.array([wl.get(e, 1.0) for e in range(E)])
        wt_s, wr_s = wts * ss, wrs * ss
        # Explicit Tikhonov prior toward the pre-relax state: P0 here IS the
        # pre-relax anchor array (x0 below is P0[free]), so the prior anchors
        # flat-valley directions at the odometry-consistent state while
        # data-constrained directions move freely. Rows are appended AFTER
        # the 3E edge rows so callers can keep indexing r[3e:3e+3].
        lt, lr = self.prior_lam_t, self.prior_lam_r
        Xp = P0[free]                                   # prior reference poses

        def resid(x):
            P = P0.copy()
            P[free] = x.reshape(-1, 3)
            ca, sa = np.cos(P[aa, 2]), np.sin(P[aa, 2])
            dx = P[bb, 0] - P[aa, 0]
            dy = P[bb, 1] - P[aa, 1]
            out = np.empty((E, 3))
            out[:, 0] = (ca * dx + sa * dy - Zt[:, 0]) * wt_s
            out[:, 1] = (-sa * dx + ca * dy - Zt[:, 1]) * wt_s
            out[:, 2] = S.wrap(P[bb, 2] - P[aa, 2] - Zt[:, 2]) * wr_s
            if lt <= 0:
                return out.ravel()
            X = x.reshape(-1, 3)
            pr = np.empty((Fn, 3))
            pr[:, 0] = lt * (X[:, 0] - Xp[:, 0])
            pr[:, 1] = lt * (X[:, 1] - Xp[:, 1])
            pr[:, 2] = lr * S.wrap(X[:, 2] - Xp[:, 2])
            return np.concatenate([out.ravel(), pr.ravel()])

        # analytic sparse Jacobian: 12 potential nonzeros per edge
        # (rows r0,r1 x cols xa,ya,tha,xb,yb; row r2 x cols tha,thb);
        # columns exist only for free anchors (fixed anchors = priors)
        ro = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
        comp = np.array([0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 2])
        isb = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1], bool)
        col_of = np.full(len(P0), -1)
        col_of[free] = np.arange(Fn)
        node = np.where(isb, bb[:, None], aa[:, None])      # E x 12
        nc = col_of[node]
        keep = (nc >= 0).ravel()
        rows = (3 * np.arange(E)[:, None] + ro).ravel()[keep]
        cols = (3 * nc + comp).ravel()[keep]
        nrows = 3 * E
        pvals = np.empty(0)
        if lt > 0:                     # prior block: constant diagonal rows
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
            V = np.empty((E, 12))
            V[:, 0], V[:, 1], V[:, 2] = -ca, -sa, -sa * dx + ca * dy
            V[:, 3], V[:, 4] = ca, sa
            V[:, 5], V[:, 6], V[:, 7] = sa, -ca, -(ca * dx + sa * dy)
            V[:, 8], V[:, 9] = -sa, ca
            V[:, :10] *= wt_s[:, None]
            V[:, 10], V[:, 11] = -wr_s, wr_s
            return csr_matrix((np.concatenate([V.ravel()[keep], pvals]),
                               (rows, cols)), shape=(nrows, 3 * Fn))

        x0 = P0[free].ravel()
        # With the explicit prior active, max_nfev is a true CONVERGENCE
        # budget (verify via n_nfev_cap). Without it (lam_t == 0), the 30-eval
        # cap is the audited LOAD-BEARING early-stop regularizer of
        # flat-valley directions on floppy graphs (fr079 degrades 2.3x at
        # max_nfev=300) — do not raise one without enabling the other.
        sol = least_squares(resid, x0, jac=jac, method="trf",
                            x_scale="jac", max_nfev=200 if lt > 0 else 30)
        self.n_nfev_cap += int(sol.status == 0)
        P = P0.copy()
        P[free] = sol.x.reshape(-1, 3)
        return P, sol.fun[:3 * E]

    @staticmethod
    def _bfs_path(adj, a, b):
        """Shortest hop path a->b (inclusive); [] if disconnected."""
        if a == b:
            return [a]
        prev = {a: a}
        q = deque([a])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    if v == b:
                        path = [b]
                        while path[-1] != a:
                            path.append(prev[path[-1]])
                        return path
                    q.append(v)
        return []

    def _graph_adj(self, skip_pairs=()):
        A = len(self.anchors)
        adj = [[] for _ in range(A)]
        seq_adj = [[] for _ in range(A)]
        skip = set(skip_pairs)
        for a, b, Z, wt, wr, kind in self.edges:
            if kind == "loop" and (a, b) in skip:
                continue
            adj[a].append(b)
            adj[b].append(a)
            if kind == "seq":
                seq_adj[a].append(b)
                seq_adj[b].append(a)
        return adj, seq_adj

    def _margin(self, Sw, seq_adj, hops=5):
        frontier = set(Sw)
        for _ in range(hops):
            frontier = {n for s in frontier for n in seq_adj[s]} - Sw
            if not frontier:
                break
            Sw |= frontier
        return Sw

    def _build_window(self, new_pairs):
        """Anchors to relax: for each new-since-last-relax loop edge, the
        chain path between its endpoints short-circuited by existing loop
        edges (the cycle the new edge closes), plus MAP-LIVE anchors (those
        still holding a segment; O(area) by the cell cap) within ~6 m of the
        endpoints, plus a 5-anchor margin past each boundary. Restricting
        proximity to map-live anchors is what keeps the window O(area)
        rather than O(time): evicted-but-loop-holding anchors of old passes
        pile up at every location, but they carry no map content — they are
        reached via the cycle path / act as boundary priors / get the
        periodic full solve instead."""
        adj, seq_adj = self._graph_adj(skip_pairs=new_pairs)
        near = np.array(sorted(self.segvec), int)
        P0 = np.array(self.anchors)
        Sw = set()
        for a, b in new_pairs:
            Sw.update((a, b))
            Sw.update(self._bfs_path(adj, a, b))
            for c in (a, b):
                d = np.linalg.norm(P0[near, :2] - P0[c, :2], axis=1)
                Sw.update(near[d < 6.0].tolist())
        Sw -= self.retired                 # frozen anchors are never unknowns
        return self._margin(Sw, seq_adj)

    def _grow_window(self, Sw):
        """Escalation: close the window under loop-edge connectivity (union
        of loop-connected components, connecting chain paths included), then
        re-apply the margin."""
        adj, seq_adj = self._graph_adj()
        Sw = set(Sw)
        while True:
            grow = [(a, b) for a, b, *_, kind in self.edges
                    if kind == "loop" and ((a in Sw) != (b in Sw))]
            if not grow:
                break
            for a, b in grow:
                Sw.update((a, b))
                Sw.update(self._bfs_path(adj, a, b))
        Sw -= self.retired
        return self._margin(Sw, seq_adj)

    def _relax_solve(self, win):
        """One full relaxation pass (solve + IRLS + LOO prune) over free
        anchors = win (None = all live anchors; gauge node 0 and retired
        anchors are always fixed). Returns False iff a boundary-crossing seq
        edge is left with whitened residual > 3 (escalation trigger)."""
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
                wl[e] = 1.0 / np.sqrt(1.0 + (rn / 2.0) ** 2)   # Cauchy-ish IRLS
            P, r = self._gn(eds, wl, free, P0)
            # leave-one-out pruning: the graph bends to hide outliers, so test
            # each suspicious edge against the solution computed WITHOUT it.
            # Rank by UNWEIGHTED residual (the IRLS-weighted statistic is
            # capped near 2.0 and cannot rank) and take the worst few.
            # Restricted to loop edges fully inside the window: a
            # boundary-crossing loop edge is judged against frozen drift.
            raws = {e: np.linalg.norm(r[3 * e:3 * e + 3]) / max(wl.get(e, 1.0), 1e-9)
                    for e, (a, b, Z, wt, wr, kind) in enumerate(eds)
                    if kind == "loop" and a in fs and b in fs}
            suspicious = sorted((e for e, v in raws.items() if v > 2.0),
                                key=lambda e: -raws[e])
            bad = set()
            for e in suspicious[:6]:      # bounded work per relaxation
                a, b, Z, wt, wr, kind = eds[e]
                P2, _ = self._gn([ed for i, ed in enumerate(eds) if i != e],
                                 {}, free, P0)
                d = S._rot(-P2[a, 2]) @ (P2[b, :2] - P2[a, :2])
                rn = np.linalg.norm(np.array([*((d - Z[:2]) * wt),
                                              S.wrap(P2[b, 2] - P2[a, 2] - Z[2]) * wr]))
                if rn > 3.0:
                    if win is not None:
                        # D1 fix: inside a window this may be a GENUINE closure
                        # the frozen boundary cannot absorb — escalate rather
                        # than permanently ban it
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
                P, r = self._gn(eds, wl2, free, P0)  # end on a robust solve
        if win is not None:
            # escalation tests BEFORE applying (D2: a failed window must not
            # leave a half-applied state behind): (a) boundary-crossing seq
            # residual > 3; (b) any in-window loop edge whose post-IRLS RAW
            # whitened residual > 3 — Cauchy IRLS parks unabsorbable tension
            # in the downweighted loop edge, so the boundary-seq statistic
            # alone is provably blind to it
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

    def relax(self):
        self.dirty = False
        self.n_relax += 1
        new = [p for p in dict.fromkeys(self.pending_new)
               if p in self.edge_seen]     # dedupe; drop already-pruned pairs
        self.pending_new = []
        if len(self.anchors) < 2 or not self.edges:
            return
        # Windowed mode is OPT-IN: on drift-heavy real logs the frozen window
        # boundaries under-propagate corrections (Intel: 6.5 m windowed vs
        # 3.3 m full at the SAME 15 ms/kf — analytic GN made full solves cheap
        # at this scale). Enable for genuinely unbounded runs where O(t)
        # relax cost eventually dominates.
        if not self.windowed or self.n_relax % self.full_every == 0 or not new:
            self._relax_solve(None)        # periodic global bleed
            self.n_full += 1
            self.solve_log.append((self.k, self._last_nfree, "full"))
            return
        Sw = self._build_window(new)
        ok = self._relax_solve(Sw)
        self.n_win += 1
        self.solve_log.append((self.k, self._last_nfree, "win"))
        if ok:
            return
        Sg = self._grow_window(Sw)
        if len(Sg - self.retired - {0}) > self._last_nfree:
            ok = self._relax_solve(Sg)
            self.n_grow += 1
            self.solve_log.append((self.k, self._last_nfree, "grow"))
            if ok:
                return
        self._relax_solve(None)            # last resort: full solve
        self.n_full += 1
        self.solve_log.append((self.k, self._last_nfree, "full-esc"))

    def memory_kb(self):
        # segvec + segder, both complex128 D-vectors per segment
        return len(self.segvec) * L.W.shape[0] * 16 * 2 / 1024


# --------------------------------------------------------------------------
# Bench
# --------------------------------------------------------------------------

def run(world="room", n=750, seed=1, use_graph=True, robust=True, laps=3,
        ablate=None, inject=0.0, dropout=0.0, scale_bias=0.0, head_bias=0.0,
        coh_target=None):
    S.RNG = np.random.default_rng(seed)
    segs = WORLDS[world]()
    gt = multiloop_traj(n, laps=laps)
    odo = sim_odometry(gt, scale_bias=scale_bias, head_bias=head_bias)
    slam = BoundedSLAM(robust=robust)
    slam.diag_gt = gt
    if coh_target is not None:
        slam.coh_target = coh_target
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
                inflate=slam.n_inflate,
                mem=slam.memory_kb(), secs=time.time() - t0,
                evict=slam.n_evict, retired=len(slam.retired),
                n_win=slam.n_win, n_grow=slam.n_grow, n_full=slam.n_full,
                sizes=slam.solve_log, diag=np.array(slam.diag))


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
