"""Belief-carrying (harmonic-Bayes) frontend for bounded-memory SSP SLAM.

CONCEPT AS BUILT (the cheap Rao-Blackwell translation of RBPF into VSA form —
the graph already owns poses, the map is deterministic given them, so the ONLY
thing we add is a small pose BELIEF the frontend refuses to collapse until it
is unimodal-tight).

The shipped `BoundedSLAM` frontend COMMITS one pose per keyframe (matcher
argmax) and only inflates the sequential-edge sigma when it hard-falls-back to
raw odometry. It has no way to notice that a match it ACCEPTED (inside the
innovation gate) is ambiguous — a corridor aperture ridge, or a perceptual
alias with two plausible along-corridor positions. It bakes that confident
wrong pose into a tight, unprunable quadratic seq edge; drift accumulates and
loop closures are then needed to undo it.

`BeliefSLAM` carries a translation BELIEF on the matcher's OWN coarse
correlation surface (the 17x17 +-0.48 m / 6 cm grid it already scans — no new
heavy compute, no phasor CF):

  - LIKELIHOOD: the matcher's coarse translation score surface at the winning
    heading, standardized and tempered:  lik(d) = softmax(beta * zscore(s(d))).
    A sharp peak -> concentrated; an aperture ridge / alias -> spread / bimodal.
  - MOTION PREDICT: the previous posterior, re-sampled onto the new grid (its
    center rides the constant-velocity / odometry guess) and Gaussian-blurred by
    the per-step motion noise (ndimage.shift + gaussian_filter — O(cells)).
  - POSTERIOR: prediction * likelihood, renormalized on the grid.

COMMIT RULE (the point): if the posterior is unimodal AND tight (one dominant
mode, positional spread < sig_cap) -> COMMIT exactly as the shipped frontend
does (matcher argmax pose, fold into the map, base 0.03 m seq sigma). If
MULTIMODAL or DIFFUSE -> do NOT trust the argmax: (a) navigate on the belief
mean, (b) INFLATE that span's seq-edge sigma to the belief's own spread (the
ambiguity becomes honest uncertainty instead of a confident wrong pose), and
(c) OPTIONALLY suppress folding the ambiguous frame into the map
(suppress_ambiguous, default off). On a later distinctive frame the carried
belief collapses to one mode and the accumulated uncertainty is consistent.

Backend (anchors, pass-segregated seq/loop edges, innovation gate, coherence
response, IRLS + LOO, TRF relax) is inherited from BoundedSLAM UNCHANGED.

HYPOTHESIS UNDER TEST: carrying belief reduces drift in ambiguous stretches
(corridors/apertures, Intel's ambiguous segments) -> lower ATE and/or lower
seq-edge-implied drift there, WITHOUT hurting the well-conditioned room case.
Honest-failure clause is live: the shipped stack already inflates seq sigma on
fallback and innovation-gates bad matches, so belief may add nothing.

Usage:
  python3 -m experiments.belief selftest
  python3 -m experiments.belief bench [quick]          (corridor/sparse/room, paired)
  python3 -m experiments.belief carmen <log> [n] [--base] [--suppress]
"""

import sys
import time

import numpy as np
from scipy import ndimage

import sspslam.encoder as S
import sspslam.lattice as L
import sspslam.bounded as B
from sspslam.worlds import WORLDS
from sspslam.bench import multiloop_traj, BEAM, scan_at

ANCHOR = B.ANCHOR

# The matcher's coarse translation grid, reproduced here so the belief lives on
# EXACTLY the surface the matcher already scans (S.Matcher.coarse_off with
# t_half=0.48, step 0.06 -> 17x17). Cell [i,j] holds offset (x=GV[j], y=GV[i]),
# matching S._grid_offsets' meshgrid('xy')+ravel ordering.
STEP = 0.06
HALF = 0.48
GV = np.arange(-HALF, HALF + STEP / 2, STEP)
NS = len(GV)                                    # 17
_XX, _YY = np.meshgrid(GV, GV)                  # (NS,NS): _XX[i,j]=GV[j]
OFF = np.stack([_XX.ravel(), _YY.ravel()], 1)   # == S._grid_offsets(HALF, STEP)


def _modes(P, rel=0.35, min_sep=0.15):
    """Local maxima of the (NS,NS) posterior above rel*peak, greedily thinned
    to a minimum separation. Returns [(value, (x,y)), ...] peak-sorted."""
    thr = rel * P.max()
    peaks = []
    for i in range(NS):
        for j in range(NS):
            v = P[i, j]
            if v < thr:
                continue
            nb = P[max(0, i - 1):i + 2, max(0, j - 1):j + 2]
            if v >= nb.max():
                peaks.append((v, np.array([GV[j], GV[i]])))
    peaks.sort(key=lambda t: -t[0])
    kept = []
    for v, p in peaks:
        if all(np.linalg.norm(p - q) >= min_sep for _, q in kept):
            kept.append((v, p))
    return kept


class TransBelief:
    """A carried translation posterior on the matcher's 17x17 coarse grid.

    State: P (NS,NS) normalized density and c (world xy) = the grid center
    (always the current motion guess). All ops are O(cells)."""

    def __init__(self, beta=4.0, sig_cap=0.12, mode_rel=0.35, mode_sep=0.15,
                 sig_t0=0.02, sig_t1=0.10):
        self.beta = beta
        self.sig_cap = sig_cap
        self.mode_rel, self.mode_sep = mode_rel, mode_sep
        self.sig_t0, self.sig_t1 = sig_t0, sig_t1
        self.P = None
        self.c = None

    def _motion_prior(self, sig):
        g = np.exp(-0.5 * (_XX ** 2 + _YY ** 2) / max(sig, STEP) ** 2)
        return g / g.sum()

    def predict(self, new_c, motion, carry=True):
        """Shift the carried posterior onto the grid centered at new_c and blur
        by the per-step motion noise. carry=False re-seeds a fresh motion prior
        each frame (the no-carrying ablation)."""
        new_c = np.asarray(new_c, float)
        sig = self.sig_t0 + self.sig_t1 * motion
        sig_cells = max(sig / STEP, 0.6)
        if self.P is None or not carry or self.c is None:
            self.P = self._motion_prior(sig)
        else:
            # world point W = c_old + off_old = new_c + off_new  =>
            # off_old = off_new + (new_c - c_old); ndimage.shift resamples so
            # pred[j_new] = P[j_old], j_old = j_new + dcell.
            dcell = (new_c - self.c) / STEP            # (dx, dy) in cells
            sh = ndimage.shift(self.P, (-dcell[1], -dcell[0]), order=1,
                               mode="constant", cval=0.0)
            sh = ndimage.gaussian_filter(sh, sig_cells, mode="constant")
            s = sh.sum()
            self.P = sh / s if s > 1e-12 else self._motion_prior(sig)
        self.c = new_c.copy()

    def update(self, scores):
        """Measurement update: multiply the prediction by the tempered,
        standardized coarse likelihood surface and renormalize."""
        s = scores.reshape(NS, NS).astype(float)
        z = (s - s.mean()) / (s.std() + 1e-12)
        lik = np.exp(self.beta * (z - z.max()))
        post = self.P * lik
        t = post.sum()
        self.P = post / t if t > 1e-300 else self.P
        return self.stats()

    def shift_center(self, dxy):
        """Ride a backend relaxation jump: the whole map (and this local
        belief) translated by dxy in world coords."""
        if self.c is not None:
            self.c = self.c + np.asarray(dxy, float)

    def stats(self):
        P = self.P
        mean_off = np.array([(P * _XX).sum(), (P * _YY).sum()])
        dx, dy = _XX - mean_off[0], _YY - mean_off[1]
        cxx = (P * dx * dx).sum()
        cyy = (P * dy * dy).sum()
        cxy = (P * dx * dy).sum()
        ev = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]]))
        sig_max = float(np.sqrt(max(ev[-1], 0.0)))
        sig_min = float(np.sqrt(max(ev[0], 0.0)))
        g0 = int(np.argmax(P))
        modes = _modes(P, self.mode_rel, self.mode_sep)
        multimodal = len(modes) >= 2
        diffuse = sig_max > self.sig_cap
        return dict(mean_off=mean_off, map_off=OFF[g0],
                    sig_max=sig_max, sig_min=sig_min, n_modes=len(modes),
                    multimodal=multimodal, diffuse=diffuse,
                    tight=not (multimodal or diffuse), P=P)


class BeliefSLAM(B.BoundedSLAM):
    """BoundedSLAM with the frontend gate-and-commit replaced by a carried
    translation belief. Everything else (backend, loop machinery) inherited."""

    def __init__(self, robust=True, attempt_every=3, relax_every=5,
                 gap_kf=B.GAP, recent_aids=12, belief=True, carry=True,
                 suppress_ambiguous=False, beta=4.0, sig_cap=0.12):
        super().__init__(robust=robust, attempt_every=attempt_every,
                         relax_every=relax_every, gap_kf=gap_kf,
                         recent_aids=recent_aids)
        self.belief = belief                 # off -> reproduces BoundedSLAM
        self.carry = carry                   # off -> per-frame belief (ablation)
        self.suppress_ambiguous = suppress_ambiguous
        self.sig_cap = sig_cap
        self.bel = TransBelief(beta=beta, sig_cap=sig_cap)
        self.max_suppress = 12               # forced map write on long amb spans
        # per-span info-honest seq-sigma accumulators (reset each anchor)
        self._span_st = 0.03
        self._span_sr = np.deg2rad(0.3)
        self._prev_c = None
        self._supp_run = 0
        # stats
        self.n_amb = self.n_multi = self.n_diffuse = 0
        self.n_commit = self.n_forced = 0
        self.n_belief_meas = 0
        self.amb_sig = []                    # sig_max on ambiguous frames
        self.bel_log = []                    # (k, sig_max, n_modes, ambiguous)

    # ---- the matcher's coarse correlation surface at the winning heading -----
    def _coarse_surface(self, Bv, pts, w, guess):
        """Reproduce the perm-matcher's coarse translation scan and return the
        FULL 289-cell score surface at the best heading, plus that heading and
        its score margin over the runner-up heading. No re-encoding beyond the
        two the matcher itself uses (lattice-aligned + half-step)."""
        mt = self.matcher
        n_ang = mt.perm[1]
        lat = np.pi / n_ang
        S0 = mt.enc.encode(pts, w)
        S0h = mt.enc.encode(pts @ S._rot(lat / 2).T, w)
        m0 = int(round(guess[2] / lat))
        h = int(np.ceil(mt.rot_half / lat - 1e-9))
        best = (-np.inf, None, 0.0)
        second = -np.inf
        for m in range(m0 - h, m0 + h + 1):
            for Sbase, off in ((S0, 0.0), (S0h, lat / 2)):
                Sv = mt._rot_permute(Sbase, m)
                _, sc, scores, _ = mt._best_translation(
                    Bv, Sv, guess[:2], mt.coarse_off, mt.E_coarse)
                if sc > best[0]:
                    second = best[0]
                    best = (sc, scores, m * lat + off)
                elif sc > second:
                    second = sc
        sc0, scores, th = best
        marg = (sc0 - second) / (abs(sc0) + 1e-9)
        return scores, th, marg

    # ---- lifecycle (mirrors BoundedSLAM.add_keyframe; frontend replaced) -----
    def add_keyframe(self, pts, w, guess):
        self.k += 1
        k = self.k
        if k % ANCHOR == 0:
            self.anchors.append(guess.copy())
        aid = len(self.anchors) - 1
        est = guess
        fell_back = True
        ambiguous = False
        st = None

        # belief motion predict onto this frame's guess center
        motion = 0.0 if self._prev_c is None \
            else float(np.linalg.norm(guess[:2] - self._prev_c))
        if self.belief and k > 0:
            self.bel.predict(guess[:2], motion, carry=self.carry)

        if k > 0 and len(pts) >= 20:
            Bv = self.local_bundle(guess[:2])[L.MAIN]
            if np.abs(Bv).sum() > 0:
                cand = self.matcher.match(Bv, pts, w, guess)
                cand[2] = S.wrap(cand[2])
                if np.linalg.norm(cand[:2] - guess[:2]) < 0.45 \
                        and abs(S.wrap(cand[2] - guess[2])) < np.deg2rad(11):
                    est = cand
                    fell_back = False
                    # session-relative coherence baseline (verbatim from base)
                    W01 = L.W[:2 * L.N_ANG]
                    PW = pts @ S._rot(est[2]).T + est[:2]
                    sv01 = np.exp(1j * (PW @ W01.T)).T @ w
                    B01 = Bv[:2 * L.N_ANG].reshape(2, L.N_ANG)
                    s01 = sv01.reshape(2, L.N_ANG)
                    c01 = float(np.mean(
                        (np.conj(B01) * s01).sum(1).real
                        / (np.linalg.norm(B01, axis=1)
                           * np.linalg.norm(s01, axis=1) + 1e-12)))
                    self.coh_ref = c01 if self.coh_ref is None \
                        else 0.95 * self.coh_ref + 0.05 * c01
                    # belief measurement update on the coarse surface
                    if self.belief:
                        surf, _, _ = self._coarse_surface(Bv, pts, w, guess)
                        st = self.bel.update(surf)
                        self.n_belief_meas += 1
                        ambiguous = not st["tight"]
                        if ambiguous:
                            # NAVIGATE on the belief mean (keep the matcher's
                            # heading — the belief only carries translation)
                            bmean = self.bel.c + st["mean_off"]
                            est = np.array([bmean[0], bmean[1], cand[2]])

        # ambiguity accounting + info-honest seq-sigma inflation for the span
        if ambiguous and st is not None:
            self.n_amb += 1
            self.n_multi += int(st["multimodal"])
            self.n_diffuse += int(st["diffuse"] and not st["multimodal"])
            self.amb_sig.append(st["sig_max"])
            self._span_st = max(self._span_st, 0.10, min(st["sig_max"], 0.30))
            self._span_sr = max(self._span_sr, np.deg2rad(1.5))
        elif not fell_back and st is not None:
            self.n_commit += 1
        if self.belief and st is not None:
            self.bel_log.append((k, st["sig_max"], st["n_modes"], ambiguous))
        self._span_fallback = getattr(self, "_span_fallback", False) or fell_back

        if k % ANCHOR == 0:
            self.anchors[aid] = est.copy()
        rel = L.se2_mul(L.se2_inv(self.anchors[aid]), est)
        self.kf_ref.append((aid, rel))

        # optionally suppress folding highly-ambiguous frames (forced write on
        # long spans keeps the local bundle alive); default off -> fold all
        write = len(pts) > 0
        if self.suppress_ambiguous and ambiguous and len(pts) > 0:
            self._supp_run += 1
            if self._supp_run < self.max_suppress:
                write = False
            else:
                self._supp_run = 0
                self.n_forced += 1
        else:
            self._supp_run = 0

        if write:
            PA = pts @ S._rot(rel[2]).T + rel[:2]
            A = np.exp(1j * (PA @ L.W.T))
            v = A.T @ w
            Cx = PA[:, 0:1] * L.W[:, 1] - PA[:, 1:2] * L.W[:, 0]
            vd = (1j * Cx * A).T @ w
            if aid not in self.segvec:
                self.segvec[aid] = np.zeros(L.W.shape[0], self.store_dtype)
                self.segder[aid] = np.zeros(L.W.shape[0], self.store_dtype)
                cell = tuple((self.anchors[aid][:2] // B.CELL).astype(int))
                self.cells.setdefault(cell, []).append(aid)
                if len(self.cells[cell]) > B.CELL_CAP:
                    drop = self.cells[cell].pop(0)
                    self.segvec.pop(drop, None)
                    self.segder.pop(drop, None)
                    self.n_evict += 1
                    if self.windowed:
                        self._maybe_retire(drop)
            self.segvec[aid] += v
            self.segder[aid] += vd

        if aid > 0 and k % ANCHOR == 0:   # sequential edge between anchors
            Z = L.se2_mul(L.se2_inv(self.anchors[aid - 1]), self.anchors[aid])
            st_v, sr_v = 0.03, np.deg2rad(0.3)
            if getattr(self, "_span_fallback", False):
                st_v, sr_v = 0.10, np.deg2rad(1.5)
            st_v = max(st_v, self._span_st)
            sr_v = max(sr_v, self._span_sr)
            self._span_fallback = False
            self._span_st, self._span_sr = 0.03, np.deg2rad(0.3)
            self.edges.append((aid - 1, aid, Z, 1 / st_v, 1 / sr_v, "seq"))

        if k % self.attempt_every == 0:
            self.try_constraint(pts, w)
        ret = est
        if k % self.relax_every == 0 and self.dirty:
            self.relax()
            newp = self.pose_of(k)
            if self.belief:
                self.bel.shift_center(newp[:2] - est[:2])
            ret = newp
        self._prev_c = guess[:2].copy()
        return ret


# ---------------------------------------------------------------------------
# Selftest: corridor-fork toy — belief stays bimodal through the ambiguous
# stretch and collapses at the fork; a single-commit frontend picks a wrong
# mode there.
# ---------------------------------------------------------------------------

def _blob(mu, sig):
    d0 = _XX - mu[0]
    d1 = _YY - mu[1]
    return np.exp(-0.5 * ((d0 / sig[0]) ** 2 + (d1 / sig[1]) ** 2))


def _pseudo_scores(P):
    """Turn a synthetic density on the grid into a matcher-like SCORE surface
    (the belief's update standardizes + tempers it, so only the shape matters).
    log gives a broad, matcher-plausible score field."""
    return np.log(P.ravel() + 1e-3)


def selftest():
    rng = np.random.default_rng(0)

    # --- 1. mode detector + moments on the grid ---------------------------
    bel = TransBelief()
    bel.c = np.array([2.0, 1.0])
    bel.P = _blob((-0.18, 0.0), (0.05, 0.03)) + _blob((0.18, 0.0), (0.05, 0.03))
    bel.P /= bel.P.sum()
    st = bel.stats()
    assert st["n_modes"] == 2, f"bimodal not detected: {st['n_modes']}"
    assert not st["tight"] and st["multimodal"]
    print(f"ST1 modes: {st['n_modes']} at +-0.18 m, sig_max "
          f"{st['sig_max'] * 100:.1f} cm (multimodal, not tight)")
    bel.P = _blob((0.02, -0.01), (0.045, 0.045))
    bel.P /= bel.P.sum()
    st = bel.stats()
    assert st["tight"] and st["n_modes"] == 1
    print(f"ST1 tight: 1 mode, sig_max {st['sig_max'] * 100:.1f} cm (commits)")

    # --- 2. motion predict re-centers onto the moving guess ----------------
    bel = TransBelief()
    bel.predict(np.array([0.0, 0.0]), 0.05, carry=True)   # seed
    bel.P = _blob((0.10, 0.0), (0.05, 0.05))
    bel.P /= bel.P.sum()
    # the true peak is at world (0.10, 0). Move the grid center to (0.06, 0);
    # after re-centering the peak must sit at offset ~(0.04, 0).
    bel.predict(np.array([0.06, 0.0]), 0.02, carry=True)
    st = bel.stats()
    peak = OFF[int(np.argmax(bel.P))]
    assert abs(peak[0] - 0.04) < 0.03 and abs(peak[1]) < 0.03, peak
    print(f"ST2 predict re-center: peak rode to offset "
          f"({peak[0]:+.2f}, {peak[1]:+.2f}) m (expected ~+0.04, 0)")

    # --- 3. carry-and-collapse corridor fork -------------------------------
    # True position sits at mode B (+0.18). Through 8 ambiguous frames the
    # likelihood is bimodal (A at -0.18, B at +0.18) with a small per-frame
    # asymmetry that flips the ARGMAX between the two — a single-commit
    # frontend has no way to stay consistent. The carried belief integrates
    # both. At the fork the likelihood becomes unimodal at B and the belief
    # collapses.
    bel = TransBelief()
    single_x = []            # per-frame argmax x (the single-commit frontend)
    for t in range(8):
        c = np.array([2.0 + 0.05 * t, 1.0])
        bel.predict(c, 0.05, carry=True)
        bias = 0.12 * np.sin(1.7 * t)           # deterministic mode flip
        sA = (1.0 + max(bias, 0)) * _blob((-0.18, 0.0), (0.05, 0.03))
        sB = (1.0 + max(-bias, 0)) * _blob((0.18, 0.0), (0.05, 0.03))
        dens = 0.02 + sA + sB
        st = bel.update(_pseudo_scores(dens))
        single_x.append(OFF[int(np.argmax(dens.ravel()))][0])
    assert st["n_modes"] == 2, f"belief lost bimodality: {st['n_modes']}"
    assert not st["tight"]
    # the single-commit argmax flipped to the WRONG mode (A, -0.18) at least once
    n_wrong = sum(1 for x in single_x if x < -0.09)
    assert n_wrong >= 1, f"single-commit never picked wrong mode: {single_x}"
    print(f"ST3a carry: belief held {st['n_modes']} modes over 8 frames "
          f"(sig_max {st['sig_max'] * 100:.1f} cm); single-commit argmax "
          f"picked the WRONG fork on {n_wrong}/8 frames")

    # fork: distinctive frame, likelihood keeps only mode B (the true one)
    for t in range(8, 11):
        c = np.array([2.0 + 0.05 * t, 1.0])
        bel.predict(c, 0.05, carry=True)
        dens = 0.02 + _blob((0.18, 0.0), (0.05, 0.03))
        st = bel.update(_pseudo_scores(dens))
    assert st["tight"], "belief must collapse and commit at the fork"
    true_off = np.array([0.18, 0.0])
    err = np.linalg.norm(st["mean_off"] - true_off)
    assert err < 0.05, f"collapsed to wrong place: {st['mean_off']} err {err}"
    print(f"ST3b collapse: tight={st['tight']} sig_max {st['sig_max'] * 100:.1f} "
          f"cm, belief mean at +{st['mean_off'][0]:.2f} m (true +0.18, err "
          f"{err * 100:.1f} cm). Single-commit argmax path over the ambiguous "
          f"stretch: [{' '.join(f'{x:+.2f}' for x in single_x)}] — it jumped "
          f"between forks ({n_wrong} wrong), never carried the ambiguity, and "
          f"baked those jumps into its trajectory; the belief held both and "
          f"snapped to the true fork only when the distinctive frame arrived.")
    print("selftest ok")


# ---------------------------------------------------------------------------
# Bench: paired vs the shipped single-commit BoundedSLAM frontend
# ---------------------------------------------------------------------------

def run(world, n=750, seed=1, cls=B.BoundedSLAM, laps=3, **kw):
    S.RNG = np.random.default_rng(seed)
    segs = WORLDS[world]()
    gt = multiloop_traj(n, laps=laps)
    odo = B.sim_odometry(gt)
    slam = cls(robust=True, **kw) if cls is BeliefSLAM else cls(robust=True)
    slam.diag_gt = gt
    est = np.zeros((n, 3))
    t0 = time.time()
    for k in range(n):
        r = scan_at(segs, gt[k])
        pts, w, _ = S.scan_to_samples(r, BEAM)
        guess = odo[0] if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odo[k - 1]), odo[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
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
    seqs = [e for e in slam.edges if e[5] == "seq"]
    # seq-edge-implied drift: mean sigma_t and inflated fraction
    seq_st = np.array([1 / e[3] for e in seqs]) if seqs else np.zeros(0)
    fb = float(np.mean(seq_st > 0.09)) if len(seq_st) else 0.0
    out = dict(ate=np.sqrt((ate ** 2).mean()), ate_max=ate.max(),
               edges=n_loop, false=false, pruned=slam.n_pruned,
               fb=fb, seq_st=float(seq_st.mean()) if len(seq_st) else 0.0,
               secs=time.time() - t0)
    if isinstance(slam, BeliefSLAM):
        tot = max(slam.n_belief_meas, 1)
        out.update(amb=slam.n_amb / tot, multi=slam.n_multi,
                   diffuse=slam.n_diffuse, forced=slam.n_forced,
                   amb_sig=float(np.mean(slam.amb_sig)) if slam.amb_sig else 0.0)
    return out


def bench(worlds=("corridor", "sparse", "room"), seeds=(1, 2, 3, 4),
          extra_belief=()):
    """Paired BoundedSLAM (shipped single-commit) vs BeliefSLAM variants."""
    variants = [("BoundedSLAM", B.BoundedSLAM, {}),
                ("Belief", BeliefSLAM, {}),
                *extra_belief]
    for world in worlds:
        print(f"== {world}")
        base = None
        for tag, cls, kw in variants:
            rs = [run(world, seed=s, cls=cls, **kw) for s in seeds]
            ates = np.array([r["ate"] for r in rs])
            if base is None:
                base = ates
            d = ates - base
            extra = ""
            if "amb" in rs[0]:
                extra = (f"  amb {100 * np.mean([r['amb'] for r in rs]):4.1f}%"
                         f" (multi {np.mean([r['multi'] for r in rs]):.0f}"
                         f"/diff {np.mean([r['diffuse'] for r in rs]):.0f})"
                         f"  amb-sig {100 * np.mean([r['amb_sig'] for r in rs]):.1f}cm"
                         f"  forced {np.mean([r['forced'] for r in rs]):.0f}")
            print(f"{tag:<20} ATE {100 * ates.mean():7.1f} cm "
                  f"[{' '.join(f'{100 * a:.0f}' for a in ates)}] "
                  f"(paired {100 * d.mean():+7.1f}, {int(np.sum(d < 0))}/{len(d)} better)  "
                  f"loops {np.mean([r['edges'] for r in rs]):4.1f}  "
                  f"false {np.mean([r['false'] for r in rs]):4.1f}  "
                  f"seq-st {100 * np.mean([r['seq_st'] for r in rs]):4.1f}cm  "
                  f"infl-seq {100 * np.mean([r['fb'] for r in rs]):4.1f}%  "
                  f"{np.mean([r['secs'] for r in rs]):5.1f}s{extra}", flush=True)


# ---------------------------------------------------------------------------
# CARMEN driver (mirrors runners/carmen.py; odometry is the belief prior)
# ---------------------------------------------------------------------------

def carmen(path, limit=10 ** 9, base=False, suppress=False):
    import sspslam.frontend as C
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)[:limit]
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    tag = "BoundedSLAM" if base else "BeliefSLAM"
    print(f"[{path}] {len(scans)} scans -> {n} keyframes ({tag})")
    if base:
        slam = B.BoundedSLAM(robust=True, attempt_every=4, relax_every=25,
                             gap_kf=300, recent_aids=12)
    else:
        slam = BeliefSLAM(robust=True, attempt_every=4, relax_every=25,
                          gap_kf=300, recent_aids=12,
                          suppress_ambiguous=suppress)
        # real odom keyframed at 0.10 m: match the driver's motion scale
        slam.bel.sig_t0, slam.bel.sig_t1 = 0.02, 0.10
    slam.store_dtype = np.complex64
    est = np.zeros((n, 3))
    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < VALID_MAX, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = opose if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
        if k % 1000 == 0:
            print(f"  kf {k}/{n} t={time.time() - t0:.0f}s "
                  f"loops={sum(1 for e in slam.edges if e[5] == 'loop')}",
                  flush=True)
    if slam.dirty:
        slam.relax()
    dt = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    seqs = [e for e in slam.edges if e[5] == "seq"]
    seq_st = np.array([1 / e[3] for e in seqs]) if seqs else np.zeros(0)
    fb = float(np.mean(seq_st > 0.05)) if len(seq_st) else 0.0
    n_loop = sum(1 for e in slam.edges if e[5] == "loop")
    msg = (f"done: {dt:.0f}s ({dt / n * 1e3:.0f} ms/kf) loops={n_loop} "
           f"pruned={slam.n_pruned} infl-seq={100 * fb:.1f}%")
    if not base:
        tot = max(slam.n_belief_meas, 1)
        msg += (f" amb={100 * slam.n_amb / tot:.1f}% "
                f"(multi={slam.n_multi} diff={slam.n_diffuse}) "
                f"forced={slam.n_forced}")
    print(msg)
    np.savez(path.rsplit("/", 1)[-1].replace(".log", "") + "_traj_belief.npz",
             est=fin, odom=odom, kts=kts)
    try:
        ref = C.parse_flaser(path.replace(".log", ".gfs.log"))
    except FileNotFoundError:
        print("no corrected reference log; skipping ATE")
        return
    rts = np.array([t for _, _, t in ref])
    rxy = np.stack([p[:2] for _, p, _ in ref])
    j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
    good = np.abs(rts - kts[j]) < 0.3
    al = C.align_se2(fin[j[good], :2], rxy[good])
    e = np.linalg.norm(al - rxy[good], axis=1)
    print(f"ATE vs corrected log: rmse {np.sqrt((e ** 2).mean()):.3f} m  "
          f"median {np.median(e):.3f} m  max {e.max():.3f} m")


VALID_MAX = 40.0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if mode == "selftest":
        selftest()
    elif mode == "bench":
        if "quick" in sys.argv:
            bench(worlds=("corridor",), seeds=(1, 2))
        else:
            bench()
    elif mode == "carmen":
        path = sys.argv[2]
        lim = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() \
            else 10 ** 9
        carmen(path, lim, base="--base" in sys.argv,
               suppress="--suppress" in sys.argv)
    else:
        raise SystemExit(f"unknown mode {mode}")
