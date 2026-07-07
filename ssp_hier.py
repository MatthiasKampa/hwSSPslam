"""Hierarchical multi-scale VSA mapping (experiment).

Replaces BoundedSLAM's rigid per-segment anchor-frame vectors with TIERED
WORLD-FRAME region maps, one tier per band of lattice rings (fine -> coarse):

1. RANGE GATING at encode: a scan sample at robot-frame range r contributes
   ring lam only if lam >= beta * r * dtheta_beam (per-point ring mask on the
   N x D phase matrix before the weighted sum). Far samples cannot support
   sub-beam-spacing wavelengths; blanking them keeps fine rings crisp.
2. TIERED WORLD-FRAME MAPS: the 6 rings are partitioned into tiers. Each tier
   stores region vectors on TWO staggered square cell grids (cell size
   s_t = cellk * max_lam(tier), second grid offset s_t/2 in both axes; a write
   adds 0.5x to both covering cells, a query sums both covering cells ->
   ~50% overlap continuity). Writes are EMA-decayed per write:
   cell <- (1-eps)*cell + contribution, so fine tiers self-enforce recency.
   The coarsest tier(s) can be GLOBAL-SINGLE: one vector for everything
   (wide-relocalization store; decays with eps_global per keyframe since it
   is written every frame, not only when its cell is occupied).
3. PIPELINE mirrors BoundedSLAM (poses, anchors every 5 kf, seq edges,
   innovation-gated loop constraints, IRLS + LOO, TRF with max_nfev=30, same
   gates/sigmas). Frontend matches against the fine+mid ring query bundle at
   the current position (world-frame: no anchor rotation needed). Loop
   constraints match against the MID rings (lam 1, 2) only; the coarse global
   vector serves the relocalize_global path. The map is NOT retro-corrected
   after graph updates (world-frame sums are unmixable) -- decay is the
   corrective mechanism; the pose estimate still benefits from the graph.
4. GRANULARITY SWEEP: G1 = BoundedSLAM itself (rigid segments, control);
   G2 = fine {0.25,0.5} cells + {1,2,5.3,12.8} global-single;
   G3 = {0.25,0.5} cells, {1,2} cells, {5.3,12.8} global-single;
   G6 = one tier per ring, coarsest two global-single.
   Plus beta in {2,4} and beta=inf (gating off) at the best granularity.

R2 adds the HYBRID SPLIT-BAND design (HybridSLAM): FINE band (lam 0.25,0.5)
keeps the SHIPPED rigid anchor-relative per-segment vectors + d/dtheta
derivative vectors (recency-windowed frontend, pass-segregated loop
constraints, graph corrections move anchors) but stores ONLY the fine-band
sub-vectors (120 of 360 dims + derivative -> 1/3 segment memory); MID band
(lam 1,2) goes to slow-decay world-frame region cells (eps=0.005) used as
additional frontend context and loop-bundle support; COARSE band (5.3,12.8)
is one global vector (eps=0.002) for wide relocalization. Loop constraints
match fine-band segment chains (slow rigid memory, per the R1 law) with mid
region support mixed into the bundle; range gating stays on (beta=3).
Plus two CSM-transferred ablations: SPIN EXCLUSION (rotation-triggered
keyframes matched but not folded into segments/tiers) and IRLS on/off with
LOO + explicit pruning kept in both arms. seg_nring=4 ("HY4") moves the
split to the MAIN/coarse boundary: segments keep the whole matched band
(2/3 memory), only the coarse relocalization band is tiered. R2 VERDICT
(RESULTS.md): HY fails its 1.15x gate (sparse 2.0x -- the matched band
cannot live in world-frame sums where it carries the signal); HY4 is
G1-identical at 2/3 map memory (Intel 2.440 m at 5.24 vs 8.06 MB, fr079
5.523 m at 3.48 MB); both CSM transfers are NEGATIVE on Intel (spin
8.72 m, linear loss 4.32 m vs 2.44 m).

R3 adds DROUGHT RELOCALIZATION to HybridSLAM (armed by default,
drought_kf=500): when no loop constraint has been accepted for drought_kf
keyframes, the coarse global vector is searched MAP-ANCHORED (candidate
cells near old pre-drought segment anchors — MIT measurement: at true
revisits the drifted estimate is 90-236 m from where the old pass was
mapped, so any estimate-centered window is blind) over the full 360 deg
heading lattice with two-stage translation refinement; a hypothesis
passing the z-gate is verified through the normal fine-band machinery
(nearest old chain, cmatcher seed grid, session-relative coherence
clearing coh_target outright), passed through a drought-scaled innovation
allowance (0.30 m Intel cap -> 0.5 x distance-travelled-since-accept;
measured MIT revisit separation is ~0.4x path), and TWO consistent
independently-spaced verified closures are required (25-kf base cadence,
single pending — every faster/looser pairing variant measured WORSE on
MIT, 59.6-86.9 vs 38.0 m); the dangling chain is then pre-aligned
(distributed unbend — a building-scale snap cannot survive max_nfev=30
TRF + IRLS + LOO otherwise) and both edges enter the normal backend.
eps_global default moved 0.002 -> 1e-4 (store must remember for the
revisit horizon; feeds relocalization only). R3 VERDICT (RESULTS.md):
bench droughts 3.3-3.5 m -> 0.05 m (4/4 seeds); MIT 45.3 -> 38.0 m with
final-third closures 15 -> 27; Intel/fr079 bit-identical, 0 snaps; the
coarse band's residual limit is corridor self-similarity (capacity and
drift pollution both ruled out by measurement).

R4 implements the aliasing/capacity study's recommendations (RESULTS.md
"Aliasing and capacity in unbounded environments") in the drought
relocalizer: (1) PER-RING WHITENING of the coarse correlation (raw
queries are red-dominated — 74% of energy in the 12.8 m ring — and
whitening removed every >=0.5x alias to +-100 m and sharpened the relo
lobe 3.75 -> 2.25 m); (2) NMS at 2.5 m on the stage-1 shortlist before
the top-40 (raw top-40 collapses into one ~5 m blob; measured recall
0.12 -> 0.62 at N=400); (4) SHARDED WRITE-ONCE coarse vectors keyed by
~250 m of accumulated travel (each keyframe's coarse content goes to its
era's shard, no decay; drought search scores candidate cells against the
shard(s) of the old anchors that generated them; the run-global EMA gvec
stays only for the legacy relocalize_global recovery path); (5) the
matched band stays OUT of drought scoring (asserted: shard store and
query projection live on WC). Rec (3) — widening drought_seed to
(3.0, 0.75) — is REJECTED BY MEASUREMENT: the wide grid reaches the
matched band's ~2 m scene aliases and multi-meter corridor slides, which
verify at ratio 0.9-1.2 and poison the pairing (bench seeds 1/4 went
5 cm -> 3.3-3.5 m; no arbiter survived). The controlled bench forced the
final shape instead: TWO-REGIME stage-1 placement (when raw and whitened
winners agree at basin level, R3-exact raw placement/z; when they
disagree — the blob/alias-corrupted regime whitening provably fixes —
the whitened winner, re-centered on its basin's raw local peak),
verification SUB-FLOOR TIEBREAK (raw score decides basins outright above
a measured 2% discrimination floor; within it the coarse hypothesis
picks by proximity — margins measured: decided >= 2.5%, alias flips
+-0.5%), and pair_tol_t 3.0 -> 1.2 m, BELOW the 2 m alias spacing, so a
truth fire and an alias fire can never pair.

Usage:
  python3 ssp_hier.py selftest
  python3 ssp_hier.py cell <gran> <world> <seed> [beta]          # one json row
      gran: G1|G2|G3|G6 | HY|HY-spin|HY-noirls | HY4|HY4-spin|HY4-noirls
  python3 ssp_hier.py bench [quick]                              # R1 sweep
  python3 ssp_hier.py intel [data/intel.log] [gran] [beta]       # R1 CARMEN
  python3 ssp_hier.py hyintel [log] [spin] [noirls] [s2] [nodrought]  # R2/R3
  python3 ssp_hier.py drought [seeds...]                         # R3 bench
"""

import json
import sys
import time

import numpy as np

import ssp_slam as S
import ssp_slam_loop as L
import ssp_bounded as B
from worlds import WORLDS
from bench_loop import multiloop_traj, FOV_BEAMS, BEAM, scan_at

ANCHOR = B.ANCHOR                                # 5 kf per anchor, as shipped
NA = L.N_ANG
RING_IDX = [np.arange(r * NA, (r + 1) * NA) for r in range(L.N_RING)]
FINE_RINGS = (0, 1)                              # lam 0.25, 0.5
MID_RINGS = (2, 3)                               # lam 1, 2  (loop band)
FRONT_RINGS = (0, 1, 2, 3)                       # frontend = fine + mid
FINE = slice(0, 2 * NA)
MID = slice(2 * NA, 4 * NA)
COARSE = slice(4 * NA, 6 * NA)
WF, WM, WC = L.W[FINE], L.W[MID], L.W[COARSE]

ENC_MID = S.SSPEncoder.__new__(S.SSPEncoder)
ENC_MID.W = L.W[MID]

# gran -> (ring tuples fine->coarse, indices of global-single tiers)
TIER_CONFIGS = {
    "G2": (((0, 1), (2, 3, 4, 5)), frozenset({1})),
    "G3": (((0, 1), (2, 3), (4, 5)), frozenset({2})),
    "G6": (((0,), (1,), (2,), (3,), (4,), (5,)), frozenset({4, 5})),
}


def _sub_rot_permute(vec, m, n_ring):
    """rot_permute on a ring-major SUBSET lattice (n_ring x N_ANG)."""
    A = vec.reshape(n_ring, NA)
    ext = np.concatenate([A, np.conj(A)], axis=1)
    return ext[:, (np.arange(NA) - m) % (2 * NA)].reshape(-1)


class HierSLAM(B.BoundedSLAM):
    """BoundedSLAM with the segment store swapped for tiered world-frame maps.

    Inherits the whole backend (seq edges, innovation gate, soft coherence
    response, IRLS + LOO, TRF relax with max_nfev=30) and the frontend gates;
    overrides map writes/queries and the loop-constraint frontend."""

    def __init__(self, tiers, global_tiers, beta=3.0, cellk=10.0, eps=0.02,
                 eps_global=0.002, robust=True, attempt_every=3, relax_every=5,
                 gap_kf=B.GAP):
        super().__init__(robust=robust, attempt_every=attempt_every,
                         relax_every=relax_every, gap_kf=gap_kf)
        # ---- knobs ----
        self.beta = beta                  # range gate: lam >= beta*r*dtheta
        self.cellk = cellk                # cell size = cellk * max_lam(tier)
        self.eps = eps                    # EMA decay per write (cell tiers)
        # Global-single tiers are written EVERY keyframe (cells only while
        # occupied), so the literal per-write eps=0.02 would forget a lap in
        # ~35 kf. eps_global rescales to a relocalization-useful half-life
        # (~350 kf here); it is a knob, not tuned per world.
        self.eps_global = eps_global
        self.dtheta_beam = np.pi / FOV_BEAMS   # bench default; CARMEN: pi/n_beams
        # ---- tier structure ----
        self.tiers = [tuple(t) for t in tiers]
        self.global_tiers = set(global_tiers)
        self.tier_idx = [np.concatenate([RING_IDX[r] for r in t])
                         for t in self.tiers]
        self.tier_s = [self.cellk * max(L.LAMS[r] for r in t)
                       for t in self.tiers]
        self.grid = [dict() for _ in self.tiers]  # (g,ix,iy)->vec; global: None
        self.write_snap = []              # per anchor: pose at write time
        # loop matcher on the MID rings only (fine content at old regions is
        # stale-decayed); ring-major subset lattice -> exact permutations
        self.cmatcher = S.Matcher(ENC_MID, t_half=0.72, rot_half_deg=9,
                                  rot_step_deg=1.5, perm=(2, NA))
        # mid-band ridge probe scale: the shipped +-1.2 m / >=0.35 m probe was
        # calibrated with lam_min=0.25 in the matched band; the mid band's
        # finest lam is 1.0, so a genuine mid peak's support is ~4x wider.
        # Scale the probe accordingly (Hessian anisotropy is scale-free).
        self.ridge_scale = 4.0

    # ---- encode with per-point ring mask (range gating) --------------------
    def _encode_gated(self, PW, w, rng):
        """World-frame points PW, weights w, robot-frame ranges rng ->
        D-vector; ring lam blanked per point where lam < beta*r*dtheta."""
        A = np.exp(1j * (PW @ L.W.T))               # N x D phase matrix
        if not np.isfinite(self.beta):
            return A.T @ w
        ok = L.LAMS[None, :] >= self.beta * rng[:, None] * self.dtheta_beam
        return (A * np.repeat(ok, NA, axis=1)).T @ w

    # ---- tiered map writes / queries ---------------------------------------
    def _cell_keys(self, t, p):
        s = self.tier_s[t]
        return [(g, int(np.floor((p[0] - 0.5 * s * g) / s)),
                 int(np.floor((p[1] - 0.5 * s * g) / s))) for g in (0, 1)]

    def write_map(self, pos, v_full):
        for t, idx in enumerate(self.tier_idx):
            vt = v_full[idx]
            G = self.grid[t]
            if t in self.global_tiers:
                cur = G.get(None)
                G[None] = vt.copy() if cur is None \
                    else (1 - self.eps_global) * cur + vt
            else:
                for key in self._cell_keys(t, pos):
                    cur = G.get(key)
                    G[key] = 0.5 * vt if cur is None \
                        else (1 - self.eps) * cur + 0.5 * vt

    def query_bundle(self, p, rings):
        """Full-lattice D-vector with the requested rings filled from their
        holding tiers (both staggered grids summed); other rings zero."""
        Bf = np.zeros(L.W.shape[0], complex)
        want = set(rings)
        for t, tr in enumerate(self.tiers):
            if not want.intersection(tr):
                continue
            G = self.grid[t]
            if t in self.global_tiers:
                vt = G.get(None)
            else:
                vt = None
                for key in self._cell_keys(t, p):
                    c = G.get(key)
                    if c is not None:
                        vt = c.copy() if vt is None else vt + c
            if vt is None:
                continue
            for j, r in enumerate(tr):
                if r in want:
                    Bf[RING_IDX[r]] = vt[j * NA:(j + 1) * NA]
        return Bf

    def map_memory_kb(self):
        """Sum over tiers: touched cells (both grids; global counts 1) x
        D_tier x 16 bytes."""
        tot = 0
        for t, tr in enumerate(self.tiers):
            tot += len(self.grid[t]) * len(tr) * NA * 16
        return tot / 1024

    # ---- lifecycle ----------------------------------------------------------
    def add_keyframe(self, pts, w, guess):
        self.k += 1
        k = self.k
        if k % ANCHOR == 0:
            self.anchors.append(guess.copy())
            self.write_snap.append(guess.copy())
        aid = len(self.anchors) - 1
        est = guess
        fell_back = True
        if k > 0 and len(pts) >= 20:
            Bq = self.query_bundle(guess[:2], FRONT_RINGS)
            if np.abs(Bq[L.MAIN]).sum() > 0:
                cand = self.matcher.match(Bq[L.MAIN], pts, w, guess)
                cand[2] = S.wrap(cand[2])
                if np.linalg.norm(cand[:2] - guess[:2]) < 0.45 \
                        and abs(S.wrap(cand[2] - guess[2])) < np.deg2rad(11):
                    est = cand
                    fell_back = False
                    # session baseline for the loop veto: MID-ring coherence
                    # of the accepted frontend match (the loop matcher runs on
                    # mid rings, so the reference must be the same band)
                    if np.abs(Bq[MID]).sum() > 0:
                        PW = pts @ S._rot(est[2]).T + est[:2]
                        sv = np.exp(1j * (PW @ L.W[MID].T)).T @ w
                        B2 = Bq[MID].reshape(2, NA)
                        s2 = sv.reshape(2, NA)
                        c2 = float(np.mean(
                            (np.conj(B2) * s2).sum(1).real
                            / (np.linalg.norm(B2, axis=1)
                               * np.linalg.norm(s2, axis=1) + 1e-12)))
                        self.coh_ref = c2 if self.coh_ref is None \
                            else 0.95 * self.coh_ref + 0.05 * c2
        self._span_fallback = getattr(self, "_span_fallback", False) or fell_back
        if k % ANCHOR == 0:
            self.anchors[aid] = est.copy()
            self.write_snap[aid] = est.copy()
        rel = L.se2_mul(L.se2_inv(self.anchors[aid]), est)
        self.kf_ref.append((aid, rel))

        # world-frame, range-gated write into every tier (both grids, EMA)
        if len(pts):
            rng = np.linalg.norm(pts, axis=1)     # robot-frame ranges
            PW = pts @ S._rot(est[2]).T + est[:2]
            self.write_map(est[:2], self._encode_gated(PW, w, rng))

        if aid > 0 and k % ANCHOR == 0:           # seq edge, shipped sigmas
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
            return self.pose_of(k)
        return est

    # ---- loop constraints against the MID tiers -----------------------------
    def try_constraint(self, pts, w):
        k = self.k
        if len(pts) < 20:
            return
        me = self.pose_of(k)
        my_aid = self.aid_of(k)
        cands = [aid for aid in range(len(self.anchors))
                 if abs(aid - my_aid) > self.gap_kf // ANCHOR
                 and np.linalg.norm(self.anchors[aid][:2] - me[:2]) < 5.0]
        if not cands:
            return
        cands.sort()
        chains, cur = [], [cands[0]]
        for aid in cands[1:]:
            (cur.append(aid) if aid - cur[-1] <= 2
             else (chains.append(cur), cur := [aid]))
        chains.append(cur)
        chain = min(chains, key=lambda ch: min(
            np.linalg.norm(self.anchors[a][:2] - me[:2]) for a in ch))
        c_aid = chain[len(chain) // 2]
        Bm = self.query_bundle(me[:2], MID_RINGS)[MID]
        if np.abs(Bm).sum() == 0:
            return
        pose = self.cmatcher.match(Bm, pts, w, me)
        pose[2] = S.wrap(pose[2])
        if np.linalg.norm(pose[:2] - me[:2]) > 0.6 \
                or abs(S.wrap(pose[2] - me[2])) > np.deg2rad(7):
            return
        # mid-ring coherence + degeneracy indicators at the matched pose
        Wm = L.W[MID]
        PW = pts @ S._rot(pose[2]).T + pose[:2]
        sv = np.exp(1j * (PW @ Wm.T)).T @ w
        c = np.conj(Bm) * sv
        Br = Bm.reshape(2, NA)
        svr = sv.reshape(2, NA)
        coh = c.reshape(2, NA).sum(1).real / (
            np.linalg.norm(Br, axis=1) * np.linalg.norm(svr, axis=1) + 1e-12)
        nrm = np.linalg.norm(Bm) * np.linalg.norm(sv) + 1e-12
        K = (Wm * (c.real / nrm)[:, None]).T @ Wm
        evals, evecs = np.linalg.eigh(K)
        l_weak, l_strong = float(evals[0]), float(evals[1])
        u_weak = evecs[:, 0]
        ts = np.arange(-1.2, 1.21, 0.06) * self.ridge_scale
        s0 = float(c.real.sum())
        far = np.abs(ts) >= 0.35 * self.ridge_scale
        ridge = [0.0, 0.0]
        for j, u in enumerate((u_weak, evecs[:, 1])):
            proj = Wm @ u
            sc = (np.exp(1j * ts[:, None] * proj[None, :]) @ c).real
            ridge[j] = float(sc[far].max() / max(s0, 1e-12))
        ridge_w, ridge_s = ridge
        lever = np.linalg.norm(me[:2] - self.anchors[c_aid][:2])
        sig_t = np.sqrt(0.08 ** 2 + (0.05 * lever) ** 2)
        sig_r = np.deg2rad(2.0)
        self.diag.append([np.linalg.norm(pose[:2] - me[:2]),
                          abs(S.wrap(pose[2] - me[2])), len(chain), *coh,
                          l_weak, l_strong, ridge_w, ridge_s,
                          self.coh_ref if self.coh_ref is not None else np.nan,
                          0.0])
        infl = 1.0
        if self.use_coh and self.coh_ref is not None:
            if not self.coh_soft:
                if coh.mean() < self.coh_target * self.coh_ref:
                    self.n_veto += 1
                    return
            else:
                infl = self._coh_response(coh, l_weak, l_strong,
                                          ridge_w, ridge_s, me, k)
                if infl < 0:
                    return
        _, rel = self.kf_ref[k]
        Zk = L.se2_mul(pose, L.se2_inv(rel))      # implied anchor pose of k
        # measurement frame: the map content near the candidate was written at
        # the old pass's WRITE-TIME poses (world-frame sums never move with
        # the graph), so Z is taken against the write-time snapshot; applying
        # it as an edge to the CURRENT anchor propagates graph corrections to
        # the new pose even though the map itself is stale
        Z = L.se2_mul(L.se2_inv(self.write_snap[c_aid]), Zk)
        Zc = L.se2_mul(L.se2_inv(self.anchors[c_aid]), self.anchors[my_aid])
        since = k - self.last_accept_k
        s_at = sig_t + min(0.30, 0.002 * since)
        s_ar = sig_r + min(np.deg2rad(6), np.deg2rad(0.03) * since)
        chi = (np.linalg.norm(Z[:2] - Zc[:2]) / s_at) ** 2 \
            + (S.wrap(Z[2] - Zc[2]) / s_ar) ** 2
        if self.use_innov and chi > 9.0:
            self.n_innov_rej += 1
            return
        edge = (c_aid, my_aid, Z, 1 / (sig_t * infl), 1 / (sig_r * infl), "loop")
        key = (c_aid, my_aid)
        if key in self.banned:
            return
        if key in self.edge_seen:
            if edge[3] * 3.5 < self.edges[self.edge_seen[key]][3]:
                return
            self.edges[self.edge_seen[key]] = edge
        else:
            self.edge_seen[key] = len(self.edges)
            self.edges.append(edge)
        if infl <= self.infl_weak:
            self.dirty = True
            self.pending_new.append(key)
            self.last_accept_k = k
            self._streak = 0
            self._streak_xy = None
        else:
            self._suppress(me[:2], k)
        self.diag[-1][-1] = 1.0

    # ---- wide relocalization against the global-single vector ---------------
    def relocalize_global(self, pts, w, guess, half=12.0, step=0.7,
                          rot_half=14, z_min=3.0):
        """Prior-windowed basin search against the coarse global vector(s)
        (+-rot_half lattice steps x +-half m). Returns (coarse pose, z) or
        None. Recovery path only; not exercised on the bench. R4: per-ring
        whitened (see HybridSLAM.relocalize_global)."""
        g_rings = sorted(r for t in self.global_tiers for r in self.tiers[t])
        if not g_rings or not len(pts):
            return None
        Bfull = self.query_bundle(guess[:2], g_rings)
        Bg = np.concatenate([Bfull[RING_IDX[r]] for r in g_rings])
        if np.abs(Bg).sum() == 0:
            return None
        e = (np.abs(Bg.reshape(len(g_rings), NA)) ** 2).sum(1)
        Bg = Bg * np.repeat(1.0 / np.maximum(e, 1e-12), NA)
        Wg = np.concatenate([L.W[RING_IDX[r]] for r in g_rings])
        S0 = np.exp(1j * (pts @ Wg.T)).T @ w
        v = np.arange(-half, half + step / 2, step)
        gx, gy = np.meshgrid(v, v)
        off = np.stack([gx.ravel(), gy.ravel()], 1)
        E = np.exp(1j * (off @ Wg.T))
        ph = np.exp(1j * (Wg @ guess[:2]))
        m0 = int(round(guess[2] * NA / np.pi))
        ms = m0 + np.arange(-rot_half, rot_half + 1)
        sc = np.empty((len(ms), len(off)))
        Bc = np.conj(Bg)
        for i, m in enumerate(ms):
            sc[i] = (E @ (Bc * _sub_rot_permute(S0, m, len(g_rings)) * ph)).real
        z = (sc.max() - sc.mean()) / (sc.std() + 1e-12)
        if z < z_min:
            return None
        i_pk, g_pk = np.unravel_index(int(sc.argmax()), sc.shape)
        return np.array([*(guess[:2] + off[g_pk]),
                         S.wrap(ms[i_pk] * np.pi / NA)]), float(z)


# ---------------------------------------------------------------------------
# R2: HYBRID SPLIT-BAND — shipped fine-band segments + tiered mid/coarse
# ---------------------------------------------------------------------------

class HybridSLAM(B.BoundedSLAM):
    """Split-band bounded SLAM.

    FINE band (rings 0-1, lam 0.25/0.5): the SHIPPED BoundedSLAM design,
    verbatim in structure — rigid anchor-relative per-segment vectors plus
    d/dtheta derivative vectors, folded immediately with exact rotation,
    recency-windowed frontend, pass-segregated loop constraints, graph
    corrections move anchors (map stays fully correctable) — except segments
    store only the 2-ring fine sub-vectors (120 of 360 dims + derivative:
    exactly 1/3 of G1's per-segment memory).

    MID band (rings 2-3, lam 1/2): G2-style world-frame region cells on two
    staggered grids (cell = cellk * 2 m), SLOW EMA decay eps_mid=0.005 per
    write. Used as additional frontend context (rings 2-3 of the matcher
    bundle) and as loop-bundle support; never the primary loop reference —
    per the R1 law the loop-reference band lives in the slow rigid segments.

    COARSE band (rings 4-5, lam 5.3/12.8): one global vector, eps_global
    per keyframe, for wide relocalization only (relocalize_global).

    Range gating (beta) applies to every write, fine fold included.
    CSM-transferred switches: spin_excl (rotation-triggered keyframes are
    matched but NOT folded into segments/tiers) and use_irls (False = linear
    loss; LOO + explicit chi2 pruning kept either way)."""

    def __init__(self, beta=3.0, cellk=10.0, eps_mid=0.005, eps_global=1e-4,
                 robust=True, attempt_every=3, relax_every=5, gap_kf=B.GAP,
                 recent_aids=12, spin_excl=False, spin_deg=4.5, use_irls=True,
                 gate_fine=False, mid_front=True, mid_loop=True, seg_nring=2,
                 drought_kf=500):
        super().__init__(robust=robust, attempt_every=attempt_every,
                         relax_every=relax_every, gap_kf=gap_kf,
                         recent_aids=recent_aids)
        self.beta = beta
        # seg_nring: rings (fine->mid) kept in the rigid segments. 2 = the
        # split-band main event (fine segments + mid world cells, 1/3 segment
        # memory); 4 = split at the MAIN/coarse boundary (segments keep the
        # whole matched band, 2/3 memory; the mid tier is then redundant and
        # unused -- only the coarse global vector remains tiered).
        self.snr = seg_nring
        self.SEG = slice(0, seg_nring * NA)
        self.WSEG = L.W[self.SEG]
        # gate_fine=False: the fine fold stays SHIPPED-EXACT (ungated) — the
        # R1 gating law was established for world-frame tier sums, where far
        # samples alias into fine rings ACROSS drifted passes; a rigid anchor-
        # frame segment has no such cross-pass mixing (measured: gating the
        # fine fold costs ~2x room ATE, see RESULTS.md R2). Mid/coarse tier
        # writes stay gated (beta) regardless.
        self.gate_fine = gate_fine
        self.mid_front = mid_front          # mid region cells in the frontend
        self.mid_loop = mid_loop            # mid region cells in loop bundles
        self.mid_s = cellk * 2.0            # cell = cellk * max_lam(mid band)
        self.eps_mid = eps_mid
        self.eps_global = eps_global
        self.spin_excl = spin_excl
        self.spin_rad = np.deg2rad(spin_deg)
        self.use_irls = use_irls
        self.dtheta_beam = np.pi / FOV_BEAMS  # bench default; CARMEN: pi/n
        self.midgrid = {}                   # (g,ix,iy) -> 120-dim mid vector
        self.gvec = None                    # 120-dim coarse global vector
        # R4: sharded WRITE-ONCE coarse store. One 120-dim vector per
        # shard_m of accumulated odometry travel; a keyframe's coarse
        # content is summed into its era's shard and NEVER decayed (the
        # relocalization store must remember for the revisit horizon —
        # write-once is the limit of eps->0 with the capacity question handled
        # by the sharding itself: N per shard stays under the measured
        # N*~400 75%-recall capacity at MIT corridor density). gvec (EMA)
        # is kept only for the legacy relocalize_global recovery path.
        self.shard_m = 250.0                # m of travel per coarse shard
        self.gshards = {}                   # sid -> write-once coarse sum
        self.anchor_shards = {}             # aid -> set of sids written
        self.n_spin_skip = 0
        self._last_est = None               # previous returned pose (spin det)
        # ---- R3: drought relocalization (coarse global vector) -------------
        # When no loop constraint has been ACCEPTED for drought_kf keyframes,
        # run a wide relocalization against the coarse global vector and, if
        # a hypothesis verifies through the NORMAL fine-band machinery twice
        # consistently, snap the dangling chain and insert both edges.
        # drought_kf = 0/None disables. eps_global default moved 0.002 ->
        # 1e-4 (R2's 0.002 half-life is ~350 kf; MIT droughts span 4500+ kf
        # and pair with passes ~10,000 kf old, where 0.002 leaves 2e-9 of the
        # content — the store must remember for the REVISIT horizon, not
        # minutes. eps_global feeds relocalization only, so this is invisible
        # to any run where the drought path never accepts).
        self.drought_kf = drought_kf or 0
        self.drought_every = 25             # min kf between reloc attempts
        self.drought_z = 3.0                # refined-peak z-score gate
        self.drought_step = 0.8             # candidate-cell grid (m)
        self.drought_dilate = 4.8           # cells within this of an old anchor
        # R4: the study's rec 3 (widen to (3.0, 0.75), after its measured
        # 1-3 m coarse-peak displacement) is REJECTED BY MEASUREMENT: on
        # the bench the wide grid reaches the matched band's ~2 m scene
        # aliases and multi-meter corridor slides, which VERIFY (ratio
        # 0.9-1.2) and poison the pairing — seeds 1/4 went 5 cm -> 3.3-3.5 m
        # under every arbiter tried (raw score, 4-ring and fine-ring
        # coherence ranks, hyp-distance tiebreak, near-tie rejection).
        # Instead the HYPOTHESIS is re-centered on the raw local peak
        # (see _coarse_hyp), which restores R3's seed-placement statistics;
        # the seed grid keeps the R3-validated reach (1.5 + 0.6 cmatcher
        # capture = 2.1 m, deliberately BELOW the 2 m+capture alias reach).
        self.drought_seed = (1.5, 0.75)     # verify seed grid: half-width, step
        self.drought_nms = 2.5              # R4: stage-1 NMS radius (m)
        self.drought_slope = 0.5            # innovation allowance per m travelled
        self.pair_window = 200              # kf: two fires must pair within
        #   this; the base cadence (drought_every) doubles as the MINIMUM
        #   spacing — two fires 25 kf (~3 m) apart are independent
        #   viewpoints. (A 3-kf pair from an accelerated variant snapped a
        #   bench seed to 74 cm residual vs 5 cm from spaced pairs.)
        # R4: pair_tol_t 3.0 -> 1.2 m. Verified fires are fine-band poses
        # (cm-precise when right); genuine pairs agree to well under a
        # meter. 3.0 m let a correct fire pair with a matched-band 2 m
        # scene-alias fire (measured, bench seed 4 post-widening) — the
        # tolerance must sit BELOW the 2 m alias spacing.
        self.pair_tol_t = 1.2               # m: implied-correction agreement
        self.pair_tol_r = np.deg2rad(8.0)
        # R4 DEEP-SEARCH ESCALATION. In-pipeline stage-1 recall at MIT's
        # true corridor revisits is 0.00 (R3) -> 0.03 (whitened+NMS+
        # sharded): corridor retrieval is information-limited in this
        # pipeline (the study's 0.62-0.75 was measured under its own
        # flagged caveats — heading known, independent segment content).
        # Sharpened retrieval therefore does not find corridor revisits;
        # it produces MORE confident wrong fires there, and one pair of
        # them was CONSISTENT (0.54 m / 0.6 deg — tighter than a genuine
        # junction snap's 0.72 m / 2.6 deg, so no tolerance separates it)
        # and snapped an MIT run from 38 -> 54 m. What separates that
        # pair is SEARCH BREADTH, the study's own crowding law (junk
        # extreme values grow with searched area): it formed while
        # sweeping 10672 cells over 4+ era shards, vs 1948/2989 at the
        # two genuine junction snaps and <=3300 on every bench fire. When
        # the sweep exceeded deep_noff cells, a consistent pair is HELD
        # and must be re-confirmed by a third consecutive consistent fire
        # before the graph accepts it; the fallback (no snap) is exactly
        # R3's measured-best behavior in the deep corridor drought.
        self.deep_noff = 6000
        self._drought_buf = []              # recent verified closures (dicts)
        self._last_reloc_k = -10 ** 9
        self._force_relax = False
        # drought clock: last GENUINE accept only. last_accept_k cannot serve
        # — the _suppress streak cap advances it to k - since_cap (its
        # innovation-allowance job), which would silently re-arm-delay the
        # drought forever wherever veto storms occur (exactly the drifted
        # regions the mechanism exists for).
        self.last_true_accept_k = 0
        self.dist_trav = 0.0                # odometry distance travelled (m)
        self.dist_at_accept = 0.0           # dist_trav at last accepted closure
        self.n_drought_try = self.n_drought_hyp = 0
        self.n_drought_verify = self.n_drought_snap = 0
        self.drought_log = []               # per-attempt diagnostics (dicts)

    # ---- fine-band segment map (shipped design, 2 rings) -------------------
    def world_vec_fine(self, aid):
        a = self.anchors[aid]
        m = int(round(a[2] * NA / np.pi))
        delta = a[2] - m * np.pi / NA
        v = _sub_rot_permute(self.segvec[aid], m, self.snr)
        if self.use_der and aid in self.segder:
            v = v + delta * _sub_rot_permute(self.segder[aid], m, self.snr)
        return np.exp(1j * (self.WSEG @ a[:2])) * v

    def _ring_mask(self, rng, ring_lo, ring_hi):
        """N x n_rings gate mask expanded to N x (n_rings*NA); None if off."""
        if not np.isfinite(self.beta):
            return None
        ok = L.LAMS[None, ring_lo:ring_hi] \
            >= self.beta * rng[:, None] * self.dtheta_beam
        return np.repeat(ok, NA, axis=1)

    def _mid_keys(self, p):
        s = self.mid_s
        return [(g, int(np.floor((p[0] - 0.5 * s * g) / s)),
                 int(np.floor((p[1] - 0.5 * s * g) / s))) for g in (0, 1)]

    def _mid_query(self, p):
        vt = None
        for key in self._mid_keys(p):
            c = self.midgrid.get(key)
            if c is not None:
                vt = c.copy() if vt is None else vt + c
        return vt

    def local_bundle(self, center, radius=8.0):
        """Frontend bundle: RECENT fine segments (shipped recency law) plus
        mid-band region cells as additional world-frame context."""
        lo = len(self.anchors) - 1 - self.recent_aids
        Bf = np.zeros(L.W.shape[0], complex)
        for aid in self.segvec:
            if aid >= lo and np.linalg.norm(
                    self.anchors[aid][:2] - center) < radius:
                Bf[self.SEG] += self.world_vec_fine(aid)
        if self.snr == 2 and self.mid_front:
            vm = self._mid_query(center)
            if vm is not None:
                Bf[MID] = vm
        return Bf

    def map_memory_kb(self):
        seg = len(self.segvec) * self.snr * NA * 16 * 2   # vec + derivative
        mid = len(self.midgrid) * 2 * NA * 16
        gl = (2 * NA * 16) if self.gvec is not None else 0
        gl += len(self.gshards) * 2 * NA * 16          # R4 era shards
        return (seg + mid + gl) / 1024

    # ---- lifecycle (mirrors BoundedSLAM.add_keyframe) -----------------------
    def add_keyframe(self, pts, w, guess):
        self.k += 1
        k = self.k
        # spin detector: odometry dtheta over the keyframe interval (guess is
        # last returned pose composed with the odom delta, so the difference
        # of headings IS the odom rotation increment)
        spin = False
        if self._last_est is not None:
            # odometry distance travelled (guess = last est (+) odom delta)
            self.dist_trav += np.linalg.norm(guess[:2] - self._last_est[:2])
        if self.spin_excl and self._last_est is not None:
            spin = abs(S.wrap(guess[2] - self._last_est[2])) > self.spin_rad
        if k % ANCHOR == 0:
            self.anchors.append(guess.copy())
        aid = len(self.anchors) - 1
        est = guess
        fell_back = True
        if k > 0 and len(pts) >= 20:
            Bq = self.local_bundle(guess[:2])[L.MAIN]
            if np.abs(Bq).sum() > 0:
                cand = self.matcher.match(Bq, pts, w, guess)
                cand[2] = S.wrap(cand[2])
                if np.linalg.norm(cand[:2] - guess[:2]) < 0.45 \
                        and abs(S.wrap(cand[2] - guess[2])) < np.deg2rad(11):
                    est = cand
                    fell_back = False
                    # session coherence baseline: fine rings, as shipped
                    B01 = Bq[FINE].reshape(2, NA)
                    PW = pts @ S._rot(est[2]).T + est[:2]
                    sv01 = np.exp(1j * (PW @ WF.T)).T @ w
                    s01 = sv01.reshape(2, NA)
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

        if len(pts) and not spin:
            rng = np.linalg.norm(pts, axis=1)          # robot-frame ranges
            # fine fold: anchor-frame, exact rotation, range-gated
            PA = pts @ S._rot(rel[2]).T + rel[:2]
            A = np.exp(1j * (PA @ self.WSEG.T))
            if self.gate_fine:
                mk = self._ring_mask(rng, 0, self.snr)
                if mk is not None:
                    A = A * mk
            v = A.T @ w
            Cx = PA[:, 0:1] * self.WSEG[:, 1] - PA[:, 1:2] * self.WSEG[:, 0]
            vd = (1j * Cx * A).T @ w
            if aid not in self.segvec:
                self.segvec[aid] = np.zeros(self.snr * NA, complex)
                self.segder[aid] = np.zeros(self.snr * NA, complex)
                cell = tuple((self.anchors[aid][:2] // B.CELL).astype(int))
                self.cells.setdefault(cell, []).append(aid)
                if len(self.cells[cell]) > B.CELL_CAP:  # bounded by area
                    drop = self.cells[cell].pop(0)
                    self.segvec.pop(drop, None)
                    self.segder.pop(drop, None)
                    self.n_evict += 1
                    if self.windowed:
                        self._maybe_retire(drop)
            self.segvec[aid] += v
            self.segder[aid] += vd
            # mid + coarse world-frame writes (EMA decay)
            PW = pts @ S._rot(est[2]).T + est[:2]
            if self.snr == 2:
                Am = np.exp(1j * (PW @ WM.T))
                mk = self._ring_mask(rng, 2, 4)
                if mk is not None:
                    Am = Am * mk
                vm = Am.T @ w
                for key in self._mid_keys(est[:2]):
                    cur = self.midgrid.get(key)
                    self.midgrid[key] = 0.5 * vm if cur is None \
                        else (1 - self.eps_mid) * cur + 0.5 * vm
            Ac = np.exp(1j * (PW @ WC.T))
            mk = self._ring_mask(rng, 4, 6)
            if mk is not None:
                Ac = Ac * mk
            vc = Ac.T @ w
            self.gvec = vc if self.gvec is None \
                else (1 - self.eps_global) * self.gvec + vc
            # R4: write-once era shard (no decay); remember which era(s)
            # wrote while this anchor was current, so the drought search
            # can score candidate cells against the right shard(s)
            sid = int(self.dist_trav // self.shard_m)
            cur = self.gshards.get(sid)
            self.gshards[sid] = vc if cur is None else cur + vc
            self.anchor_shards.setdefault(aid, set()).add(sid)
        elif spin:
            self.n_spin_skip += 1

        if aid > 0 and k % ANCHOR == 0:               # seq edge, shipped sigmas
            Z = L.se2_mul(L.se2_inv(self.anchors[aid - 1]), self.anchors[aid])
            if getattr(self, "_span_fallback", False):
                st, sr = 0.10, np.deg2rad(1.5)
            else:
                st, sr = 0.03, np.deg2rad(0.3)
            self._span_fallback = False
            self.edges.append((aid - 1, aid, Z, 1 / st, 1 / sr, "seq"))

        if k % self.attempt_every == 0:
            self.try_constraint(pts, w)
            # FIXED base cadence, deliberately: a "hot window" that retried
            # at the attempt cadence while a verified fire awaited its pair
            # was built and REVERTED (R3) — it multiplies snap opportunities
            # in exactly the ambiguous corridor stretches where verification
            # is least trustworthy, and every accelerated variant measured
            # WORSE on MIT (59.6 / 86.9 / 70.3 m vs 38.0 conservative, 45.3
            # no-mechanism). The 25-kf spacing doubles as the independence
            # guarantee between paired fires (~3 m of travel).
            if (self.drought_kf and len(pts) >= 20
                    and k - self.last_true_accept_k >= self.drought_kf
                    and k - self._last_reloc_k >= self.drought_every):
                self._last_reloc_k = k
                self._try_drought(pts, w)
        if (k % self.relax_every == 0 and self.dirty) or self._force_relax:
            self._force_relax = False
            self.relax()
            out = self.pose_of(k)
        else:
            out = est
        self._last_est = out.copy()
        return out

    # ---- loop constraints: fine segment chains + mid region support --------
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
        cands.sort()
        chains, cur = [], [cands[0]]
        for aid in cands[1:]:
            (cur.append(aid) if aid - cur[-1] <= 2
             else (chains.append(cur), cur := [aid]))
        chains.append(cur)
        chain = min(chains, key=lambda ch: min(
            np.linalg.norm(self.anchors[a][:2] - me[:2]) for a in ch))
        Bb = np.zeros(L.W.shape[0], complex)
        for a in chain:                     # slow rigid loop reference (fine)
            Bb[self.SEG] += self.world_vec_fine(a)
        if self.snr == 2 and self.mid_loop:
            vm = self._mid_query(me[:2])
            if vm is not None:              # mid-band region support
                Bb[MID] = vm
        pose = self.cmatcher.match(Bb[L.MAIN], pts, w, me)
        pose[2] = S.wrap(pose[2])
        if np.linalg.norm(pose[:2] - me[:2]) > 0.6 \
                or abs(S.wrap(pose[2] - me[2])) > np.deg2rad(7):
            return
        # per-ring coherence at the matched pose (rings 4-5 empty -> 0)
        sv = L.ENC.shift(pose[:2]) * L.encode(pts @ S._rot(pose[2]).T, w)
        c = (np.conj(Bb) * sv).reshape(L.N_RING, NA)
        Br = Bb.reshape(L.N_RING, NA)
        svr = sv.reshape(L.N_RING, NA)
        coh = c.sum(1).real / (np.linalg.norm(Br, axis=1)
                               * np.linalg.norm(svr, axis=1) + 1e-12)
        # translation-Hessian + ridge probes on the matched MAIN band, exactly
        # as shipped (fine band present -> shipped probe scale is calibrated)
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
        self.diag.append([np.linalg.norm(pose[:2] - me[:2]),
                          abs(S.wrap(pose[2] - me[2])), len(chain), *coh,
                          l_weak, l_strong, ridge_w, ridge_s,
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
        Zk = L.se2_mul(pose, L.se2_inv(rel))      # implied anchor pose of k
        # fine band rides with its anchors (fully correctable), so Z is taken
        # against the CURRENT candidate anchor pose, exactly as shipped — no
        # write-time snapshot needed (that was the world-frame tiers' burden)
        Z = L.se2_mul(L.se2_inv(self.anchors[c_aid]), Zk)
        Zc = L.se2_mul(L.se2_inv(self.anchors[c_aid]), self.anchors[my_aid])
        since = k - self.last_accept_k
        s_at = sig_t + min(0.30, 0.002 * since)
        s_ar = sig_r + min(np.deg2rad(6), np.deg2rad(0.03) * since)
        chi = (np.linalg.norm(Z[:2] - Zc[:2]) / s_at) ** 2 \
            + (S.wrap(Z[2] - Zc[2]) / s_ar) ** 2
        if self.use_innov and chi > 9.0:
            self.n_innov_rej += 1
            return
        edge = (c_aid, my_aid, Z, 1 / (sig_t * infl), 1 / (sig_r * infl), "loop")
        key = (c_aid, my_aid)
        if key in self.banned:
            return
        if key in self.edge_seen:
            if edge[3] * 3.5 < self.edges[self.edge_seen[key]][3]:
                return
            self.edges[self.edge_seen[key]] = edge
        else:
            self.edge_seen[key] = len(self.edges)
            self.edges.append(edge)
        if infl <= self.infl_weak:
            self.dirty = True
            self.pending_new.append(key)
            self.last_accept_k = k
            self.last_true_accept_k = k
            self.dist_at_accept = self.dist_trav
            self._streak = 0
            self._streak_xy = None
        else:
            self._suppress(me[:2], k)
        self.diag[-1][-1] = 1.0

    # ---- IRLS ablation: linear loss, LOO + explicit pruning kept ------------
    def _irls_w(self, rn):
        return 1.0 / np.sqrt(1.0 + (rn / 2.0) ** 2) if self.use_irls else 1.0

    def _relax_solve(self, win):
        """Copy of BoundedSLAM._relax_solve with the IRLS weight factored
        through _irls_w so use_irls=False gives LINEAR loss while keeping the
        LOO + explicit chi2 pruning path intact (the CSM baseline's winning
        combination). Structure otherwise identical to the shipped method."""
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
                wl[e] = self._irls_w(rn)
            P, r = self._gn(eds, wl, free, P0)
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
                wl2 = {e: self._irls_w(np.linalg.norm(r[3 * e:3 * e + 3]))
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

    # ---- wide relocalization against the coarse global vector ---------------
    def relocalize_global(self, pts, w, guess, half=12.0, step=0.7,
                          rot_half=14, z_min=3.0):
        """Prior-windowed basin search over the coarse global vector (rings
        4-5). Recovery path only; not exercised on the bench. R4: PER-RING
        WHITENED — raw queries are red-dominated (74% of scene energy in
        the 12.8 m ring) and behave like a lone 12.8 m ring with 13 m
        aliases; dividing each ring's correlation by its bundle energy
        removes every measured >=0.5x alias to +-100 m."""
        if self.gvec is None or not len(pts):
            return None
        Bg = self.gvec
        e = (np.abs(Bg.reshape(2, NA)) ** 2).sum(1)
        wht = np.repeat(1.0 / np.maximum(e, 1e-12), NA)
        S0 = np.exp(1j * (pts @ WC.T)).T @ w
        v = np.arange(-half, half + step / 2, step)
        gx, gy = np.meshgrid(v, v)
        off = np.stack([gx.ravel(), gy.ravel()], 1)
        E = np.exp(1j * (off @ WC.T))
        ph = np.exp(1j * (WC @ guess[:2]))
        m0 = int(round(guess[2] * NA / np.pi))
        ms = m0 + np.arange(-rot_half, rot_half + 1)
        sc = np.empty((len(ms), len(off)))
        Bc = np.conj(Bg) * wht
        for i, m in enumerate(ms):
            sc[i] = (E @ (Bc * _sub_rot_permute(S0, m, 2) * ph)).real
        z = (sc.max() - sc.mean()) / (sc.std() + 1e-12)
        if z < z_min:
            return None
        i_pk, g_pk = np.unravel_index(int(sc.argmax()), sc.shape)
        return np.array([*(guess[:2] + off[g_pk]),
                         S.wrap(ms[i_pk] * np.pi / NA)]), float(z)

    # ---- R3: drought-breaking wide relocalization ---------------------------
    # No accepted closure for drought_kf keyframes means drift has outrun the
    # matcher gates; the normal 5 m candidate radius is then centered on a
    # position that can be arbitrarily far (MIT: 90-236 m, measured) from
    # where the old pass was MAPPED. So the coarse search is MAP-ANCHORED:
    # candidate positions are grid cells near OLD segment anchors (the only
    # places verification is possible at all), swept over the full 360 deg
    # heading lattice against the coarse global vector; a passing hypothesis
    # is then verified through the normal fine-band machinery (nearest old
    # segment chain, cmatcher from a small seed grid, coherence + Hessian +
    # ridge indicators at FULL confidence only), and two consistent verified
    # closures are required before the graph accepts anything.

    def _drought_cut(self):
        """Newest anchor a drought closure may target: the anchor of the last
        genuine accept, i.e. the frontier of the CONSTRAINED map. Content
        mapped DURING the current drought lives in the same dangling frame —
        'relocalizing' against it cannot break the drought, only reset the
        clock and (measured, MIT dr3) inject bent-frame junk snaps at
        strong sigma. Before any accept exists there is no constrained
        region and every pass-segregated anchor is fair game (bootstrap:
        such closures are ordinary loop closures found by wide search)."""
        if self.last_true_accept_k <= 0:
            return None
        return self.kf_ref[self.last_true_accept_k][0]

    def _drought_offsets(self, me, my_aid, dist_since):
        """Candidate translation offsets: centers of drought_step cells within
        drought_dilate of any old (pass-segregated, pre-drought) live anchor,
        excluding a drift-scaled ball around the current estimate (fresh
        self-content in the global vector always correlates at offset ~0).
        R4: grouped BY ERA SHARD — each group's cells come from the old
        anchors written during that shard's travel window and are scored
        against that shard's write-once coarse vector (an area revisited in
        two eras appears in both groups). Returns [(sid, offsets), ...] or
        None."""
        gap_a = self.gap_kf // ANCHOR
        cut = self._drought_cut()
        old = [a for a in self.segvec if my_aid - a > gap_a
               and (cut is None or a <= cut)]
        if not old:
            return None
        st = self.drought_step
        rr = int(np.ceil(self.drought_dilate / st))
        disk = np.array([(i, j) for i in range(-rr, rr + 1)
                         for j in range(-rr, rr + 1)
                         if (i * i + j * j) * st * st
                         <= self.drought_dilate ** 2], np.int64)
        # self-exclusion radius ~ 8% of distance travelled since last accept
        # (drift scale), floored at 2 m (the coarse self-peak's support)
        r_excl = np.clip(0.08 * dist_since, 2.0, 12.0)
        by_sid = {}
        for a in old:
            for sid in self.anchor_shards.get(a, {0}):
                by_sid.setdefault(sid, []).append(a)
        groups, ntot = [], 0
        for sid in sorted(by_sid):
            P = np.stack([self.anchors[a][:2] for a in by_sid[sid]])
            base = np.floor(P / st).astype(np.int64)
            ii = base[:, None, 0] + disk[None, :, 0]
            jj = base[:, None, 1] + disk[None, :, 1]
            key = np.unique(ii.ravel() * np.int64(1 << 21)
                            + (jj.ravel() + (1 << 20)))
            ij = np.stack([key >> 21, (key & ((1 << 21) - 1)) - (1 << 20)], 1)
            off = (ij + 0.5) * st - me[:2]
            off = off[np.linalg.norm(off, axis=1) >= r_excl]
            if len(off):
                groups.append((sid, off))
                ntot += len(off)
        return groups if ntot >= 20 else None

    def _coarse_hyp(self, pts, w, me, groups):
        """Full-360 coarse correlation of the scan against the era shard
        vectors at the candidate offsets (groups = [(sid, offsets)] from
        _drought_offsets), two-stage: drought_step sweep, then a 0.2 m
        local refinement over the shortlist (the coarse rings' phase turns
        ~0.5 rad over half a 0.8 m cell, so the true peak can score below
        junk on the raw grid).

        R4, per the aliasing/capacity study: (a) PER-RING WHITENING — each
        ring's correlation is divided by that ring's bundle energy;
        without it the red-dominated store behaves like a lone 12.8 m ring
        (13 m aliases, centroid blob outscoring truth at any N); (b) the
        shortlist is the top-40 NMS'd cells (best-heading score, greedy
        suppression at drought_nms=2.5 m) per shard, not raw (cell,
        heading) pairs — raw top-40 collapses into one ~5 m blob (measured
        recall 0.12 -> 0.62 at N=400); (c) per-shard surface statistics
        (whitened scales differ across shards), best candidate by z.

        TWO-REGIME PLACEMENT (bench-forced): whitening re-weights the main
        lobe too, displacing the peak ~1 m from the raw local peak — and
        the whole downstream verification (seed reach 1.5 + 0.6 capture,
        score pick, coherence gates) is R3-calibrated to RAW placement.
        Seeding it from whitened peaks measurably re-aims it into the
        matched band's ~2 m scene aliases (bench seeds 1/4 broke at
        3.3-3.5 m; three basin arbiters and two coherence ranks all failed
        to fix it). So BOTH surfaces are computed: when the raw winner
        lies in the whitened winner's basin (<= 1.2 m), the raw refined
        peak and raw z are returned — the R3-validated regime, verbatim
        placement statistics; when they disagree, the raw surface is
        blob/alias-corrupted (the regime whitening provably fixes — on
        MIT true revisits the raw peak is 26 m off, 0/25) and the whitened
        winner is returned, re-centered on the raw LOCAL maximum (+-1 m)
        of its own basin. Returns (pose hypothesis, z) or None; the
        drought_z gate lives in the caller so sub-gate peaks stay
        observable in drought_log."""
        S0 = np.exp(1j * (pts @ WC.T)).T @ w
        ph = np.exp(1j * (WC @ me[:2]))
        m0 = int(round(me[2] * NA / np.pi))
        ms = m0 + np.arange(-NA, NA)              # full circle, 3 deg lattice
        R0 = np.stack([_sub_rot_permute(S0, m, 2) for m in ms], 1)
        d = np.arange(-0.4, 0.41, 0.2)
        dg = np.stack(np.meshgrid(d, d), -1).reshape(-1, 2)
        best_w = (-np.inf, None, None, None)      # whitened: z, off, m, vec
        best_r = (-np.inf, None, None, -np.inf)   # raw: score-z, off, m, raw z
        for sid, off in groups:
            vec = self.gshards.get(sid)
            if vec is None or not len(off):
                continue
            # rec 5: drought scoring is COARSE-BAND ONLY (the matched band
            # is dead under drift: peak attenuation 0.03-0.09 of clean)
            assert vec.shape == (2 * NA,)
            e = (np.abs(vec.reshape(2, NA)) ** 2).sum(1)
            wht = np.repeat(1.0 / np.maximum(e, 1e-12), NA)
            Br = np.conj(vec) * ph
            M = Br[:, None] * R0
            E = np.exp(1j * (off @ WC.T))
            sc_r = (E @ M).real                   # raw surface (R3's)
            sc = (E @ (wht[:, None] * M)).real    # whitened surface
            mu, sd = float(sc.mean()), float(sc.std()) + 1e-12
            mu_r, sd_r = float(sc_r.mean()), float(sc_r.std()) + 1e-12
            # whitened winner: NMS shortlist (per-cell best over headings,
            # greedy suppression), 0.2 m refinement, best by whitened z
            cell = sc.max(1)
            order = np.argsort(cell)[::-1]
            kept = []
            for g in order:
                if kept and np.min(np.linalg.norm(
                        off[g] - off[kept], axis=1)) < self.drought_nms:
                    continue
                kept.append(int(g))
                if len(kept) >= 40:
                    break
            Bw = np.conj(vec) * wht * ph
            cache = {}
            for g in kept:
                i = int(sc[g].argmax())
                for dm in (-1, 0, 1):
                    m = int(ms[i]) + dm
                    Cv = cache.get(m)
                    if Cv is None:
                        Cv = cache[m] = Bw * _sub_rot_permute(S0, m, 2)
                    s2 = (np.exp(1j * ((off[g] + dg) @ WC.T)) @ Cv).real
                    j = int(s2.argmax())
                    z = (float(s2[j]) - mu) / sd
                    if z > best_w[0]:
                        best_w = (z, off[g] + dg[j], m, vec)
            # raw winner: R3's exact two-stage (top-40 raw (cell, heading)
            # pairs, 0.2 m refinement, best by raw score)
            flat = sc_r.ravel()
            K = min(40, flat.size)
            top = np.argpartition(flat, -K)[-K:]
            cache = {}
            sb = (-np.inf, None, None)
            for t in top:
                g, i = divmod(int(t), sc_r.shape[1])
                for dm in (-1, 0, 1):
                    m = int(ms[i]) + dm
                    Cv = cache.get(m)
                    if Cv is None:
                        Cv = cache[m] = Br * _sub_rot_permute(S0, m, 2)
                    s2 = (np.exp(1j * ((off[g] + dg) @ WC.T)) @ Cv).real
                    j = int(s2.argmax())
                    if s2[j] > sb[0]:
                        sb = (float(s2[j]), off[g] + dg[j], m)
            zr = (sb[0] - mu_r) / sd_r
            if zr > best_r[3]:
                best_r = (sb[0], sb[1], sb[2], zr)
        if best_w[1] is None:
            return None
        z, p0, m, vec = best_w
        if best_r[1] is not None \
                and np.linalg.norm(best_r[1] - p0) <= 1.2:
            # same basin: R3-exact placement and confidence
            return np.array([*(me[:2] + best_r[1]),
                             S.wrap(best_r[2] * np.pi / NA)]), float(best_r[3])
        # corrupted-raw regime: whitened winner, raw local re-center
        Cr = np.conj(vec) * ph * _sub_rot_permute(S0, m, 2)
        d2 = np.arange(-1.0, 1.01, 0.2)
        dg2 = np.stack(np.meshgrid(d2, d2), -1).reshape(-1, 2)
        loc = p0 + dg2
        s2 = (np.exp(1j * (loc @ WC.T)) @ Cr).real
        return np.array([*(me[:2] + loc[int(s2.argmax())]),
                         S.wrap(m * np.pi / NA)]), float(z)

    def _drought_verify(self, pts, w, hyp, my_aid):
        """Fine/mid verification of a coarse hypothesis against the nearest
        old segment chain: cmatcher from a small seed grid around hyp (the
        coarse peak is only good to ~a cell), then the standard coherence /
        Hessian / ridge indicators. FULL-confidence only: the candidate must
        clear coh_target or the well-conditioned exemption; anything that
        would be inflated or vetoed is rejected outright (a wrong building-
        scale snap is catastrophic). Returns (pose, c_aid, chain, coh_ratio)
        or None."""
        gap_a = self.gap_kf // ANCHOR
        cut = self._drought_cut()
        cands = [aid for aid in self.segvec if my_aid - aid > gap_a
                 and (cut is None or aid <= cut)
                 and np.linalg.norm(self.anchors[aid][:2] - hyp[:2]) < 5.0]
        if not cands or self.coh_ref is None:
            return None
        cands.sort()
        chains, cur = [], [cands[0]]
        for aid in cands[1:]:
            (cur.append(aid) if aid - cur[-1] <= 2
             else (chains.append(cur), cur := [aid]))
        chains.append(cur)
        chain = min(chains, key=lambda ch: min(
            np.linalg.norm(self.anchors[a][:2] - hyp[:2]) for a in ch))
        Bb = np.zeros(L.W.shape[0], complex)
        for a in chain:
            Bb[self.SEG] += self.world_vec_fine(a)
        if self.snr == 2 and self.mid_loop:
            vm = self._mid_query(hyp[:2])
            if vm is not None:
                Bb[MID] = vm
        BbM = Bb[L.MAIN]
        half, step = self.drought_seed
        v = np.arange(-half, half + step / 2, step)
        basins = []                        # [raw score, pose], 0.7 m clusters
        for dx in v:
            for dy in v:
                seed = np.array([hyp[0] + dx, hyp[1] + dy, hyp[2]])
                pose = self.cmatcher.match(BbM, pts, w, seed)
                pose[2] = S.wrap(pose[2])
                if np.linalg.norm(pose[:2] - seed[:2]) > 0.6 \
                        or abs(S.wrap(pose[2] - seed[2])) > np.deg2rad(7):
                    continue                       # boundary-saturated
                sv = np.exp(1j * ((pts @ S._rot(pose[2]).T + pose[:2])
                                  @ L.W[L.MAIN].T)).T @ w
                s = float((np.conj(BbM) * sv).real.sum())
                for b in basins:
                    if np.linalg.norm(pose[:2] - b[1][:2]) < 0.7:
                        if s > b[0]:
                            b[0], b[1] = s, pose
                        break
                else:
                    basins.append([s, pose])
        if not basins:
            return None
        # R4 SUB-FLOOR TIEBREAK. The corridor scene aliases at the matched
        # band's 2 m period form rival basins; MEASURED margins (bench,
        # first drought attempts, seeds 1-4): decided basins differ by
        # >= 2.5% of the top raw score, while alias/truth flips live
        # within +-0.5% — below the band's discrimination floor. Score
        # decides outright above the floor (a nearest-to-hyp rule breaks
        # seed 2, whose alias sits NEARER the hyp at a 5.8% deficit);
        # within the 2% floor the raw sum is statistically blind and the
        # coarse hypothesis (alias-free band) breaks the tie by proximity
        # (measured: correct on all 8 truth/alias cases, and it is what
        # rescues seed 4's only pre-fault-2 pairing slot).
        basins.sort(key=lambda b: -b[0])
        smax = basins[0][0]
        if smax > 0:
            tie = [b for b in basins if b[0] >= 0.98 * smax]
            pose = min(tie,
                       key=lambda b: np.linalg.norm(b[1][:2] - hyp[:2]))[1]
        else:
            pose = basins[0][1]
        # fine-ring coherence at the matched pose, session-relative
        sv = L.ENC.shift(pose[:2]) * L.encode(pts @ S._rot(pose[2]).T, w)
        c = (np.conj(Bb) * sv).reshape(L.N_RING, NA)
        Br = Bb.reshape(L.N_RING, NA)
        svr = sv.reshape(L.N_RING, NA)
        coh = c.sum(1).real / (np.linalg.norm(Br, axis=1)
                               * np.linalg.norm(svr, axis=1) + 1e-12)
        ratio = coh[:2].mean() / max(self.coh_ref, 1e-12)
        if ratio < self.coh_target:
            return None
        return pose, chain[len(chain) // 2], chain, float(ratio)

    def _distribute_correction(self, b2, T2):
        """Pre-align the dangling chain before the solver sees the snap: a
        building-scale correction cannot be absorbed by max_nfev=30 TRF (the
        first solve leaves a huge residual, IRLS then downweights the genuine
        edge to nothing and LOO bans it). Distribute the heading correction
        linearly along the anchors since the last accepted closure (rotation
        about the pivot, the standard bent-chain unbend), then spread the
        residual translation; the drought edges then enter relax with small
        residuals and the normal machinery polishes."""
        a0 = self.kf_ref[min(self.last_accept_k, self.k)][0]
        a0 = max(0, min(a0, b2 - 1))
        m = b2 - a0
        dth = S.wrap(T2[2] - self.anchors[b2][2])
        p0 = self.anchors[a0][:2].copy()
        for i in range(a0 + 1, b2 + 1):
            f = (i - a0) / m
            R = S._rot(f * dth)
            self.anchors[i] = np.array(
                [*(p0 + R @ (self.anchors[i][:2] - p0)),
                 S.wrap(self.anchors[i][2] + f * dth)])
        dt = T2[:2] - self.anchors[b2][:2]
        for i in range(a0 + 1, b2 + 1):
            self.anchors[i][:2] += ((i - a0) / m) * dt

    def _try_drought(self, pts, w):
        k = self.k
        self.n_drought_try += 1
        me = self.pose_of(k)
        my_aid = self.aid_of(k)
        dist_since = self.dist_trav - self.dist_at_accept
        log = dict(k=k, phase="cells", z=np.nan, chi=np.nan, ratio=np.nan)
        self.drought_log.append(log)
        groups = self._drought_offsets(me, my_aid, dist_since)
        if groups is None:
            return
        log.update(nsh=len(groups), noff=int(sum(len(o) for _, o in groups)))
        got = self._coarse_hyp(pts, w, me, groups)
        log["phase"] = "coarse"
        if got is None:
            return
        hyp, z = got
        log.update(z=z, hyp=hyp.copy())        # sub-gate peaks stay visible
        if z < self.drought_z:
            return
        log["phase"] = "hyp"
        self.n_drought_hyp += 1
        res = self._drought_verify(pts, w, hyp, my_aid)
        if res is None:
            return
        pose, c_aid, chain, ratio = res
        self.n_drought_verify += 1
        log.update(phase="verified", ratio=ratio, pose=pose.copy(),
                   c_aid=c_aid)
        _, rel = self.kf_ref[k]
        Zk = L.se2_mul(pose, L.se2_inv(rel))
        Z = L.se2_mul(L.se2_inv(self.anchors[c_aid]), Zk)
        Zc = L.se2_mul(L.se2_inv(self.anchors[c_aid]), self.anchors[my_aid])
        lever = np.linalg.norm(pose[:2] - self.anchors[c_aid][:2])
        sig_t = np.sqrt(0.08 ** 2 + (0.05 * lever) ** 2)
        sig_r = np.deg2rad(2.0)
        # drought innovation allowance: the 0.30 m cap models Intel-scale
        # droughts (saturates at since=150 kf); at MIT scale the est-frame
        # separation at TRUE revisits measures 0.38x the path travelled since
        # the last accept (p90 213 m at ~554 m), i.e. drift is rotation-
        # levered and distance-proportional, not sqrt-like (sqrt(since) caps
        # ~6x low). Allowance = drought_slope (0.5) x distance travelled;
        # the gate keeps only a sanity role (snaps implying > 1.5x-path
        # drift are physically impossible); precision gating is delegated
        # to fine verification + the two-closure consistency pairing.
        s_at = sig_t + max(0.30, self.drought_slope * dist_since)
        s_ar = sig_r + np.deg2rad(20.0)
        chi = (np.linalg.norm(Z[:2] - Zc[:2]) / s_at) ** 2 \
            + (S.wrap(Z[2] - Zc[2]) / s_ar) ** 2
        log["chi"] = float(chi)
        if self.use_innov and chi > 9.0:
            self.n_innov_rej += 1
            return
        T = L.se2_mul(self.anchors[c_aid], Z)     # implied CURRENT my_aid pose
        entry = dict(k=k, my_aid=my_aid, c_aid=c_aid, Z=Z, sig_t=sig_t,
                     sig_r=sig_r, T=T, b_pos=self.anchors[my_aid][:2].copy(),
                     dr=S.wrap(T[2] - self.anchors[my_aid][2]))
        log["phase"] = "pending"
        # Consistency: each fire implies a rigid world correction
        # new(x) = T + R(dr) (x - b_pos); compare the pending and the new
        # fire at x* = the current anchor position (short lever between the
        # fire points, so small heading disagreements are not amplified by
        # absolute coordinates).
        # SINGLE pending + two-way consistency + replace-on-conflict, at the
        # base cadence. This exact shape is the measured optimum of five MIT
        # variants (38.0 m vs 59.6 hot-window / 86.9 hot+12-kf-floor / 70.3
        # buffered / bench-degrading triplet); the aggressive variants pair
        # faster but pair CORRELATED or ambiguous fires in self-similar
        # corridors, and a wrong pre-aligned snap is unrecoverable.
        buf = [e for e in self._drought_buf
               if k - e["k"] <= self.pair_window]
        P = buf[-1] if buf else None
        if P is not None and P["my_aid"] != my_aid:
            y1 = P["T"][:2] + S._rot(P["dr"]) \
                @ (self.anchors[my_aid][:2] - P["b_pos"])
            log["pair_dt"] = float(np.linalg.norm(y1 - T[:2]))
            log["pair_drh"] = float(np.rad2deg(
                abs(S.wrap(P["dr"] - entry["dr"]))))
            log["pair_gap"] = k - P["k"]
            if np.linalg.norm(y1 - T[:2]) < self.pair_tol_t \
                    and abs(S.wrap(P["dr"] - entry["dr"])) < self.pair_tol_r:
                if len(buf) == 1 and log.get("noff", 0) > self.deep_noff:
                    # deep-search escalation: hold the pair, demand a
                    # third consecutive consistent fire
                    log["phase"] = "pair-held"
                    self._drought_buf = [P, entry]
                    return
                log["phase"] = "snap"
                self._distribute_correction(my_aid, T)
                for e in (P, entry):
                    key = (e["c_aid"], e["my_aid"])
                    if key in self.banned:
                        continue
                    edge = (e["c_aid"], e["my_aid"], e["Z"], 1 / e["sig_t"],
                            1 / e["sig_r"], "loop")
                    if key in self.edge_seen:
                        self.edges[self.edge_seen[key]] = edge
                    else:
                        self.edge_seen[key] = len(self.edges)
                        self.edges.append(edge)
                    self.pending_new.append(key)
                self.dirty = True
                self._force_relax = True
                self.last_accept_k = k
                self.last_true_accept_k = k
                self.dist_at_accept = self.dist_trav
                self._streak = 0
                self._streak_xy = None
                self._drought_buf = []
                self.n_drought_snap += 1
                return
            log["phase"] = "pair-conflict"
        self._drought_buf = [entry]


# ---------------------------------------------------------------------------
# Bench (mirrors ssp_bounded.run: same RNG protocol -> paired seeds)
# ---------------------------------------------------------------------------

def run_hier(world="room", n=750, seed=1, gran="G3", beta=3.0, laps=3):
    S.RNG = np.random.default_rng(seed)
    segs = WORLDS[world]()
    gt = multiloop_traj(n, laps=laps)
    odo = B.sim_odometry(gt)
    tiers, gl = TIER_CONFIGS[gran]
    slam = HierSLAM(tiers, gl, beta=beta)
    est = np.zeros((n, 3))
    t0 = time.time()
    for k in range(n):
        r = scan_at(segs, gt[k])
        pts, w, _ = S.scan_to_samples(r, BEAM)
        guess = odo[0] if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odo[k - 1]), odo[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
    secs = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    ate = np.linalg.norm(fin[:, :2] - gt[:, :2], axis=1)
    n_loop = false = 0
    for a, b, Z, wt, wr, kind in slam.edges:
        if kind != "loop":
            continue
        n_loop += 1
        Zt = L.se2_mul(L.se2_inv(gt[a * ANCHOR]), gt[b * ANCHOR])
        if np.linalg.norm(Z[:2] - Zt[:2]) > 0.30 \
                or abs(S.wrap(Z[2] - Zt[2])) > np.deg2rad(3):
            false += 1
    odo_ate = np.linalg.norm(odo[:, :2] - gt[:, :2], axis=1)
    return dict(ate=float(np.sqrt((ate ** 2).mean())),
                ate_max=float(ate.max()),
                odo=float(np.sqrt((odo_ate ** 2).mean())),
                edges=n_loop, false=false, pruned=slam.n_pruned,
                veto=slam.n_veto, innov=slam.n_innov_rej,
                inflate=slam.n_inflate, relax=slam.n_relax,
                mem=float(slam.map_memory_kb()), ms=1000 * secs / n)


def run_hybrid(world="room", n=750, seed=1, beta=3.0, laps=3, **hy_kw):
    """R2 hybrid bench cell (same RNG protocol as run_hier/B.run -> paired)."""
    S.RNG = np.random.default_rng(seed)
    segs = WORLDS[world]()
    gt = multiloop_traj(n, laps=laps)
    odo = B.sim_odometry(gt)
    slam = HybridSLAM(beta=beta, **hy_kw)
    est = np.zeros((n, 3))
    t0 = time.time()
    for k in range(n):
        r = scan_at(segs, gt[k])
        pts, w, _ = S.scan_to_samples(r, BEAM)
        guess = odo[0] if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odo[k - 1]), odo[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
    secs = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    ate = np.linalg.norm(fin[:, :2] - gt[:, :2], axis=1)
    n_loop = false = 0
    for a, b, Z, wt, wr, kind in slam.edges:
        if kind != "loop":
            continue
        n_loop += 1
        Zt = L.se2_mul(L.se2_inv(gt[a * ANCHOR]), gt[b * ANCHOR])
        if np.linalg.norm(Z[:2] - Zt[:2]) > 0.30 \
                or abs(S.wrap(Z[2] - Zt[2])) > np.deg2rad(3):
            false += 1
    return dict(ate=float(np.sqrt((ate ** 2).mean())),
                ate_max=float(ate.max()), edges=n_loop, false=false,
                pruned=slam.n_pruned, veto=slam.n_veto,
                innov=slam.n_innov_rej, inflate=slam.n_inflate,
                spin_skip=slam.n_spin_skip, nseg=len(slam.segvec),
                ncell=len(slam.midgrid),
                mem=float(slam.map_memory_kb()), ms=1000 * secs / n)


def run_drought_bench(world="sparse", n=2000, seed=1, armed=True,
                      drought_kf=350, fault0=800, ev_per=400, ev_len=70,
                      bias=np.deg2rad(0.4), n_events=2):
    """R3 drought scenario: 5 laps; a heading-biased sensor DROPOUT event
    (ev_len kf at bias rad/kf -> ~28 deg per event) hits once per lap from
    fault0, n_events times — the MIT failure mode in miniature: each event
    steps the frame past every matcher gate, closures stop, and every pass
    after an event lives in its own rotated frame. Baseline HY4
    (armed=False) never recovers; the drought mechanism must fire, verify
    twice, snap, and let normal closures resume (one snap per event).

    Scenario design is MEASURED, not assumed. Three shapes that do NOT
    produce an MIT-like drought on this bench: (1) a single concentrated
    fault — all post-fault laps share one displaced frame and close against
    each other; (2) persistent odometry heading bias with scans up — the
    frontend erases it per keyframe; (3) short (12 kf) dropout events — the
    recency window (12 anchors = 60 kf) still holds pre-event segments at
    relock time and the frontend walks the frame back at up to ~9 deg per
    keyframe. Hence: events longer than the recency window, repeated per
    lap, cross-lap-only candidates (gap 350), and drought_kf below the
    same-frame revisit horizon (gap_kf + lap fraction; 500 stays right for
    CARMEN where revisit periods are thousands of kf). Same RNG protocol as
    run_hybrid -> armed/baseline runs are paired per seed."""
    S.RNG = np.random.default_rng(seed)
    segs = WORLDS[world]()
    gt = multiloop_traj(n, laps=5)

    def in_fault(k):
        return (k >= fault0 and (k - fault0) % ev_per < ev_len
                and (k - fault0) // ev_per < n_events)

    odo = gt.copy()                     # bench-noise odometry + fault events
    for k in range(1, n):
        d = L.se2_mul(L.se2_inv(gt[k - 1]), gt[k])
        d[:2] = d[:2] + S.RNG.normal(0, 0.02, 2)
        d[2] += S.RNG.normal(0, np.deg2rad(0.5)) + (bias if in_fault(k) else 0.0)
        odo[k] = L.se2_mul(odo[k - 1], d)
    # gap_kf=350: candidates are CROSS-LAP only (lap period 400; the bench
    # default 60 lets nearby arcs of the SAME drifted lap close on each other
    # and reset the drought clock — the MIT setting, gap 300 on corridors
    # that never self-cross within the gap, has no such shortcut).
    slam = HybridSLAM(beta=3.0, seg_nring=4, gap_kf=350,
                      drought_kf=(drought_kf if armed else 0))
    est = np.zeros((n, 3))
    t0 = time.time()
    for k in range(n):
        r = scan_at(segs, gt[k])
        pts, w, _ = S.scan_to_samples(r, BEAM)
        if in_fault(k):
            pts, w = pts[:0], w[:0]     # dropout: odometry fallback
        guess = odo[0] if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odo[k - 1]), odo[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
    secs = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    ate = np.linalg.norm(fin[:, :2] - gt[:, :2], axis=1)
    post = slice(fault0, n)
    n_loop = false = n_post = 0
    for a, b, Z, wt, wr, kind in slam.edges:
        if kind != "loop":
            continue
        n_loop += 1
        if b * ANCHOR >= fault0:
            n_post += 1
        Zt = L.se2_mul(L.se2_inv(gt[a * ANCHOR]), gt[b * ANCHOR])
        if np.linalg.norm(Z[:2] - Zt[:2]) > 0.30 \
                or abs(S.wrap(Z[2] - Zt[2])) > np.deg2rad(3):
            false += 1
    return dict(ate=float(np.sqrt((ate ** 2).mean())),
                ate_post=float(np.sqrt((ate[post] ** 2).mean())),
                ate_max=float(ate.max()), edges=n_loop, edges_post=n_post,
                false=false, pruned=slam.n_pruned,
                d_try=slam.n_drought_try, d_hyp=slam.n_drought_hyp,
                d_verify=slam.n_drought_verify, d_snap=slam.n_drought_snap,
                snap_k=[g["k"] for g in slam.drought_log
                        if g["phase"] == "snap"],
                ms=1000 * secs / n)


def drought_bench(seeds=(1, 2, 3, 4), world="sparse"):
    print(f"R3 drought bench: {world}, n=2000, 5 laps, two 70-kf dropout+"
          f"heading-bias events (~28 deg each) at kf 800/1200, seeds {seeds}")
    hdr = (f"{'seed':>4} {'arm':>4} {'ATE cm':>8} {'post cm':>8} {'max cm':>8}"
           f" {'f/e':>7} {'post-e':>6} {'try':>4} {'hyp':>4} {'ver':>4}"
           f" {'snap':>4}  snap@kf")
    print(hdr)
    for s in seeds:
        for armed in (False, True):
            r = run_drought_bench(world, seed=s, armed=armed)
            print(f"{s:>4} {'ON' if armed else 'off':>4}"
                  f" {100 * r['ate']:>8.1f} {100 * r['ate_post']:>8.1f}"
                  f" {100 * r['ate_max']:>8.1f}"
                  f" {r['false']:>3}/{r['edges']:<3} {r['edges_post']:>6}"
                  f" {r['d_try']:>4} {r['d_hyp']:>4} {r['d_verify']:>4}"
                  f" {r['d_snap']:>4}  {r['snap_k']}", flush=True)


HY_VARIANTS = {          # gran code -> HybridSLAM kwargs
    "HY": {},
    "HY-spin": dict(spin_excl=True),
    "HY-noirls": dict(use_irls=False),
    "HY4": dict(seg_nring=4),            # split at the MAIN/coarse boundary
    "HY4-spin": dict(seg_nring=4, spin_excl=True),
    "HY4-noirls": dict(seg_nring=4, use_irls=False),
}


def run_cell(gran, world, seed, beta=3.0, n=750):
    """One bench cell -> flat dict. gran 'G1' = BoundedSLAM control;
    'HY*' = R2 hybrid variants."""
    if gran == "G1":
        r = B.run(world, n=n, seed=seed)
        out = dict(ate=float(r["ate"]), edges=int(r["edges"]),
                   false=int(r["false"]), mem=float(r["mem"]),
                   ms=1000 * r["secs"] / n)
    elif gran in HY_VARIANTS:
        r = run_hybrid(world, n=n, seed=seed, beta=beta, **HY_VARIANTS[gran])
        out = {k: r[k] for k in ("ate", "edges", "false", "mem", "ms",
                                 "spin_skip", "pruned")}
    else:
        r = run_hier(world, n=n, seed=seed, gran=gran, beta=beta)
        out = {k: r[k] for k in ("ate", "edges", "false", "mem", "ms")}
    out.update(gran=gran, world=world, seed=seed, beta=beta)
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-4, 4, (200, 2))
    w = np.full(200, 0.1)
    sl = HierSLAM(*TIER_CONFIGS["G3"], beta=np.inf)
    # beta=inf gated encode == plain encode
    v = sl._encode_gated(pts, w, np.linalg.norm(pts, axis=1))
    assert np.allclose(v, L.encode(pts, w), atol=1e-9)
    # beta finite: gated rings are exactly the plain per-ring sums over the
    # points that pass the gate
    sl.beta = 3.0
    r = np.linalg.norm(pts, axis=1)
    v = sl._encode_gated(pts, w, r)
    for ring in range(L.N_RING):
        keep = L.LAMS[ring] >= 3.0 * r * sl.dtheta_beam
        ref = np.exp(1j * (pts[keep] @ L.W[RING_IDX[ring]].T)).T @ w[keep]
        assert np.allclose(v[RING_IDX[ring]], ref, atol=1e-9)
    # write/query roundtrip: one write, query at same point returns the full
    # contribution (0.5 from each staggered grid) on every requested ring
    vec = L.encode(pts, w)
    sl.write_map(np.array([3.3, 4.4]), vec)
    Bq = sl.query_bundle(np.array([3.3, 4.4]), range(L.N_RING))
    assert np.allclose(Bq, vec, atol=1e-9)
    # overlap continuity: query s/4 away still sees >= half the mass
    s0 = sl.tier_s[0]
    Bq2 = sl.query_bundle(np.array([3.3 + s0 / 4, 4.4]), FINE_RINGS)
    assert np.abs(Bq2[RING_IDX[0]]).sum() >= 0.5 * np.abs(vec[RING_IDX[0]]).sum()
    # relocalize_global finds a translated copy of the scan
    sl2 = HierSLAM(*TIER_CONFIGS["G3"], beta=np.inf)
    sl2.write_map(np.zeros(2), L.encode(pts + np.array([2.0, -1.5]), w))
    got = sl2.relocalize_global(pts, w, np.array([0.0, 0.0, 0.0]))
    assert got is not None and np.linalg.norm(got[0][:2] - [2.0, -1.5]) < 0.5, got
    print("self-test ok: range gate, staggered write/query, relocalize_global")
    hybrid_self_test()


def hybrid_self_test():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-4, 4, (200, 2))
    w = np.full(200, 0.1)
    r = np.linalg.norm(pts, axis=1)
    hy = HybridSLAM(beta=np.inf)
    hy.k = 0
    hy.anchors = [np.array([1.0, -2.0, 4 * np.pi / NA])]   # lattice-aligned
    hy.kf_ref = [(0, np.zeros(3))]
    # fine fold at a lattice-aligned anchor: world_vec_fine must equal the
    # direct fine-band encoding of the world points (exact permutation path)
    a = hy.anchors[0]
    PW = pts @ S._rot(a[2]).T + a[:2]                      # world points
    PA = pts                                               # anchor frame
    A = np.exp(1j * (PA @ WF.T))
    hy.segvec[0] = A.T @ w
    Cx = PA[:, 0:1] * WF[:, 1] - PA[:, 1:2] * WF[:, 0]
    hy.segder[0] = (1j * Cx * A).T @ w
    ref = np.exp(1j * (PW @ WF.T)).T @ w
    assert np.allclose(hy.world_vec_fine(0), ref, atol=1e-9)
    # gated fine fold == plain per-ring sums over gate survivors
    hy.beta = 3.0
    A2 = np.exp(1j * (PA @ WF.T)) * hy._ring_mask(r, 0, 2)
    for ring in range(2):
        keep = L.LAMS[ring] >= 3.0 * r * hy.dtheta_beam
        refr = np.exp(1j * (PA[keep] @ L.W[RING_IDX[ring]].T)).T @ w[keep]
        assert np.allclose((A2.T @ w)[ring * NA:(ring + 1) * NA], refr,
                           atol=1e-9)
    # mid write/query roundtrip (both staggered grids -> full contribution)
    vm = np.exp(1j * (PW @ WM.T)).T @ w
    for key in hy._mid_keys(np.array([3.3, 4.4])):
        hy.midgrid[key] = 0.5 * vm
    assert np.allclose(hy._mid_query(np.array([3.3, 4.4])), vm, atol=1e-9)
    # coarse relocalization finds a translated copy of the scan
    hy2 = HybridSLAM(beta=np.inf)
    P2 = pts + np.array([2.0, -1.5])
    hy2.gvec = np.exp(1j * (P2 @ WC.T)).T @ w
    got = hy2.relocalize_global(pts, w, np.zeros(3))
    assert got is not None and np.linalg.norm(got[0][:2] - [2.0, -1.5]) < 0.5, got
    # spin exclusion: large odom dtheta -> matched but not folded
    hy3 = HybridSLAM(spin_excl=True)
    hy3.add_keyframe(pts, w, np.zeros(3))
    nrm0 = np.linalg.norm(hy3.segvec[0])
    g2 = hy3._last_est.copy()
    g2[2] = S.wrap(g2[2] + np.deg2rad(10))                 # rotation-triggered
    hy3.add_keyframe(pts, w, g2)
    assert hy3.n_spin_skip == 1
    assert np.linalg.norm(hy3.segvec[0]) == nrm0           # not folded
    g3 = hy3._last_est.copy()
    g3[2] = S.wrap(g3[2] + np.deg2rad(0.5))                # gentle rotation
    hy3.add_keyframe(pts, w, g3)
    assert hy3.n_spin_skip == 1                            # folded again
    assert np.linalg.norm(hy3.segvec[0]) != nrm0
    # IRLS ablation: linear weights are exactly 1
    hy4 = HybridSLAM(use_irls=False)
    assert hy4._irls_w(5.0) == 1.0 and abs(
        HybridSLAM()._irls_w(2.0) - 1 / np.sqrt(2)) < 1e-12
    # R3/R4: coarse hypothesis over an explicit shard group recovers a
    # rotated AND translated copy (full-360 sweep, lattice heading; whitened
    # scoring + NMS shortlist; z-gate lives in the caller)
    hy5 = HybridSLAM(beta=np.inf)
    m_true, t_true = 9, np.array([6.0, -3.5])          # 27 deg, off-window
    P5 = pts @ S._rot(m_true * np.pi / NA).T + t_true
    hy5.gshards[0] = np.exp(1j * (P5 @ WC.T)).T @ w
    vv = np.arange(-8, 8.01, 0.8)
    gx, gy = np.meshgrid(vv, vv)
    got = hy5._coarse_hyp(pts, w, np.zeros(3),
                          [(0, np.stack([gx.ravel(), gy.ravel()], 1))])
    assert got is not None
    hyp5, z5 = got
    assert np.linalg.norm(hyp5[:2] - t_true) < 0.35, (hyp5, z5)
    assert abs(S.wrap(hyp5[2] - m_true * np.pi / NA)) < 1e-9, hyp5
    # R4 rec 5: drought scoring is coarse-band only — the shard store and
    # the query projection both live on WC (rings 4-5, the relo band); the
    # matched band enters only at fine verification. Structural asserts:
    assert np.allclose(WC, L.W[COARSE])
    assert all(v.shape == (2 * NA,) for v in hy5.gshards.values())
    # and a second shard group with junk content must not displace the
    # true peak (per-shard z statistics pick the right era)
    rngj = np.random.default_rng(7)
    hy5.gshards[1] = np.exp(1j * (rngj.uniform(-30, 30, (200, 2))
                                  @ WC.T)).T @ w
    got2 = hy5._coarse_hyp(pts, w, np.zeros(3),
                           [(0, np.stack([gx.ravel(), gy.ravel()], 1)),
                            (1, np.stack([gx.ravel(), gy.ravel()], 1))])
    assert got2 is not None and np.linalg.norm(got2[0][:2] - t_true) < 0.35
    # R3: chain pre-alignment hits the target pose exactly and interpolates
    hy6 = HybridSLAM()
    hy6.k = 50
    hy6.anchors = [np.array([float(i), 0.0, 0.0]) for i in range(11)]
    hy6.kf_ref = [(0, np.zeros(3))] * 51
    hy6.last_accept_k = 0
    T2 = np.array([9.0, 3.0, np.deg2rad(30)])
    hy6._distribute_correction(10, T2)
    assert np.allclose(hy6.anchors[10], T2, atol=1e-9)
    assert np.allclose(hy6.anchors[0], [0, 0, 0])       # pivot fixed
    assert abs(hy6.anchors[5][2] - np.deg2rad(15)) < 1e-9   # linear heading
    # R3/R4: drought offset raster covers old anchors, excludes the self
    # ball, and groups by era shard (unwritten anchors default to shard 0)
    hy7 = HybridSLAM()
    hy7.anchors = [np.array([10.0 * i, 0.0, 0.0]) for i in range(3)]
    hy7.segvec = {0: None, 1: None, 2: None}
    hy7.anchor_shards = {0: {0}, 1: {0, 1}, 2: {1}}
    g7 = hy7._drought_offsets(np.array([0.0, 0.0, 0.0]), 100, 100.0)
    assert [sid for sid, _ in g7] == [0, 1]
    off7 = np.concatenate([o for _, o in g7])
    d0 = np.linalg.norm(off7, axis=1)
    assert d0.min() >= 8.0 - 1e-9                       # r_excl = 8 m here
    assert np.abs(off7 - np.array([20.0, 0.0])).sum(1).min() < 1.2  # covered
    o1 = dict(g7)[1]                                    # shard-1 group only
    assert np.abs(o1 - np.array([20.0, 0.0])).sum(1).min() < 1.2
    assert np.abs(o1 - np.array([0.0, 0.0])).sum(1).min() > 5.0  # aid0 not in 1
    print("hybrid self-test ok: fine fold/rotation, gating, mid roundtrip, "
          "coarse relocalization (whitened), spin exclusion, IRLS switch, "
          "drought coarse-hyp (whitened+NMS, sharded)/pre-align/raster, "
          "coarse-band-only scoring")


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

BENCH_WORLDS = ("room", "sparse", "corridor")


def _table(rows, results, seeds):
    hdr = f"{'config':<16}" + "".join(
        f"{wl + ' ATE cm':>14}{'f/e':>8}{'mem KB':>9}{'ms':>6}"
        for wl in BENCH_WORLDS)
    print(hdr)
    for tag, gran, beta in rows:
        line = f"{tag:<16}"
        for wl in BENCH_WORLDS:
            cell = [results[(gran, wl, s, beta)] for s in seeds
                    if (gran, wl, s, beta) in results]
            if not cell:
                line += f"{'-':>14}{'-':>8}{'-':>9}{'-':>6}"
                continue
            line += (f"{100 * np.mean([c['ate'] for c in cell]):>14.1f}"
                     f"{np.mean([c['false'] for c in cell]):>4.1f}/"
                     f"{np.mean([c['edges'] for c in cell]):<3.0f}"
                     f"{np.mean([c['mem'] for c in cell]):>9.0f}"
                     f"{np.mean([c['ms'] for c in cell]):>6.0f}")
        print(line, flush=True)


def bench(quick=False):
    self_test()
    seeds = (1, 2) if quick else (1, 2, 3, 4)
    results = {}

    def do(gran, beta=3.0):
        for wl in BENCH_WORLDS:
            for s in seeds:
                r = run_cell(gran, wl, s, beta)
                results[(gran, wl, s, beta)] = r
                print(json.dumps(r), flush=True)

    for gran in ("G1", "G2", "G3", "G6"):
        do(gran)
    rows = [("G1 (control)", "G1", 3.0), ("G2 b3", "G2", 3.0),
            ("G3 b3", "G3", 3.0), ("G6 b3", "G6", 3.0)]
    _table(rows, results, seeds)
    # best hierarchical granularity by mean ATE across worlds
    best = min(("G2", "G3", "G6"), key=lambda g: np.mean(
        [results[(g, wl, s, 3.0)]["ate"] for wl in BENCH_WORLDS for s in seeds]))
    print(f"best hierarchical granularity: {best}")
    for beta in (2.0, 4.0, np.inf):
        do(best, beta)
        rows.append((f"{best} b{beta:g}", best, beta))
    _table(rows, results, seeds)
    return results


# ---------------------------------------------------------------------------
# CARMEN driver (mirrors ssp_bounded_carmen.py; zero new tuning)
# ---------------------------------------------------------------------------

def main_carmen(argv):
    import ssp_slam_carmen as C
    path = argv[0] if argv else "data/intel.log"
    gran = argv[1] if len(argv) > 1 else "G3"
    beta = float(argv[2]) if len(argv) > 2 else 3.0
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    print(f"{len(scans)} scans -> {n} keyframes   gran={gran} beta={beta:g}")
    tiers, gl = TIER_CONFIGS[gran]
    slam = HierSLAM(tiers, gl, beta=beta, attempt_every=4, relax_every=25,
                    gap_kf=300)
    slam.dtheta_beam = np.pi / n_beams
    est = np.zeros((n, 3))
    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < 40.0, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = opose if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
        if k % 1000 == 0:
            print(f"  kf {k}/{n}  t={time.time() - t0:.0f}s  "
                  f"loops={sum(1 for e in slam.edges if e[5] == 'loop')}  "
                  f"mem={slam.map_memory_kb():.0f} KB", flush=True)
    if slam.dirty:
        slam.relax()
    dt = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    n_loop = sum(1 for e in slam.edges if e[5] == "loop")
    print(f"done: {dt:.0f}s ({dt / n * 1e3:.0f} ms/kf)  loop edges={n_loop}  "
          f"pruned={slam.n_pruned}  relax={slam.n_relax}  "
          f"map memory={slam.map_memory_kb():.0f} KB")
    ref = C.parse_flaser(path.replace(".log", ".gfs.log"))
    rts = np.array([t for _, _, t in ref])
    rxy = np.stack([p[:2] for _, p, _ in ref])
    j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
    good = np.abs(rts - kts[j]) < 0.3
    al = C.align_se2(fin[j[good], :2], rxy[good])
    e = np.linalg.norm(al - rxy[good], axis=1)
    print(f"ATE vs corrected log: rmse {np.sqrt((e ** 2).mean()):.3f} m   "
          f"median {np.median(e):.3f} m   max {e.max():.3f} m")


def main_carmen_hybrid(argv):
    """R2 CARMEN entry: mirrors main_carmen / ssp_bounded_carmen.py exactly
    (attempt_every=4, relax_every=25, gap 300, recency 12; zero new tuning).
    argv: [path] [spin] [noirls] [s2] [nodrought]."""
    import ssp_slam_carmen as C
    path = argv[0] if argv and not argv[0].startswith("-") \
        and argv[0] not in ("spin", "noirls", "s2", "nodrought") \
        else "data/intel.log"
    spin = "spin" in argv
    use_irls = "noirls" not in argv
    snr = 2 if "s2" in argv else 4          # default: gate-passing HY4 split
    drought = 0 if "nodrought" in argv else 500
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    print(f"{len(scans)} scans -> {n} keyframes   hybrid seg_nring={snr} "
          f"spin_excl={spin} use_irls={use_irls}")
    slam = HybridSLAM(beta=3.0, seg_nring=snr, spin_excl=spin,
                      use_irls=use_irls, attempt_every=4, relax_every=25,
                      gap_kf=300, recent_aids=12, drought_kf=drought)
    slam.dtheta_beam = np.pi / n_beams
    est = np.zeros((n, 3))
    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < 40.0, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = opose if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
        if k % 1000 == 0:
            print(f"  kf {k}/{n}  t={time.time() - t0:.0f}s  "
                  f"loops={sum(1 for e in slam.edges if e[5] == 'loop')}  "
                  f"spin_skip={slam.n_spin_skip}  "
                  f"mem={slam.map_memory_kb():.0f} KB", flush=True)
    if slam.dirty:
        slam.relax()
    dt = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    n_loop = sum(1 for e in slam.edges if e[5] == "loop")
    print(f"done: {dt:.0f}s ({dt / n * 1e3:.0f} ms/kf)  loop edges={n_loop}  "
          f"pruned={slam.n_pruned}  relax={slam.n_relax}  "
          f"spin_skip={slam.n_spin_skip}  "
          f"map memory={slam.map_memory_kb():.0f} KB "
          f"({len(slam.segvec)} segments x {slam.snr} rings, "
          f"{len(slam.midgrid)} mid cells, "
          f"{len(slam.gshards)} coarse shards)")
    print(f"drought: try={slam.n_drought_try} hyp={slam.n_drought_hyp} "
          f"verified={slam.n_drought_verify} snaps={slam.n_drought_snap}")
    ref = C.parse_flaser(path.replace(".log", ".gfs.log"))
    rts = np.array([t for _, _, t in ref])
    rxy = np.stack([p[:2] for _, p, _ in ref])
    j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
    good = np.abs(rts - kts[j]) < 0.3
    al = C.align_se2(fin[j[good], :2], rxy[good])
    e = np.linalg.norm(al - rxy[good], axis=1)
    print(f"ATE vs corrected log: rmse {np.sqrt((e ** 2).mean()):.3f} m   "
          f"median {np.median(e):.3f} m   max {e.max():.3f} m")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if cmd == "selftest":
        self_test()
    elif cmd == "cell":
        gran, world, seed = sys.argv[2], sys.argv[3], int(sys.argv[4])
        beta = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
        print(json.dumps(run_cell(gran, world, seed, beta)))
    elif cmd == "intel":
        main_carmen(sys.argv[2:])
    elif cmd == "hyintel":
        main_carmen_hybrid(sys.argv[2:])
    elif cmd == "drought":
        seeds = tuple(int(a) for a in sys.argv[2:]) or (1, 2, 3, 4)
        drought_bench(seeds=seeds)
    else:
        bench(quick=(len(sys.argv) > 2 and sys.argv[2] == "quick")
              or cmd == "quick")
