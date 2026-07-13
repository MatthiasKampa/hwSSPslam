"""SEQUENCE-consistency (SeqSLAM, Milford & Wyeth 2012) vs the MIT corridor
twin-disambiguation wall that per-frame ring-key retrieval + PCM pairwise
GEOMETRIC consensus cannot break.

Motivation (RESULTS.md "Ring-key shortlister: MIT corridor RETRIEVAL solved, but
the wall moves to consensus"): the Scan-Context ring-key SOLVED MIT retrieval
(true-revisit recall@40 0.317 -> 0.808) but verified drought fires rose 70 -> 252
while PCM admitted ZERO consistent cliques (baseline 2) and ATE went 42.66 ->
45.24. Mechanism: the corridor TWINS each pass fine geometric verification (they
align) but point to DIFFERENT wrong places, so PCM's pairwise GEOMETRIC
consistency finds no clique, and genuine revisits are diluted by twin scatter.
PCM checks geometric consistency; it does NOT check TEMPORAL/sequential
consistency. That is the gap this module tests.

Hypothesis: a genuine revisit produces a temporally-consistent RUN of place
matches — if query keyframe k matches old keyframe j then k+1 matches ~j+1 (same
traverse), k+2 ~ j+2, ... along a velocity-swept diagonal of the sequence
similarity matrix. A corridor twin is an ISOLATED coincidence whose neighbours do
NOT extend the match along a consistent diagonal. So a SeqSLAM sequence score
should confirm genuine revisits and suppress twins where single-frame ring-key +
geometric PCM cannot.

HONEST CAVEAT tested head-on: MIT's twins may be the SAME infinite-corridor
geometry re-traversed, in which case the DESCRIPTOR SEQUENCE ITSELF aliases — a
genuine revisit and a false twin both produce a consistent diagonal because the
corridor looks identical over the whole window. If sequence consistency ALSO
cannot separate them, the environment is sequence-ambiguous and lidar-appearance
localization there is fundamentally impossible (needs an independent absolute
cue). This module reports which way it falls: the genuine-vs-twin sequence-score
separation is the crux measurement, not just the ATE.

Design (subclass RingKeySLAM; NO edit to any shipped/experiment module — imports
only, same pattern as ssp_ringkey / ssp_scancontext):
  * per-KEYFRAME ring-key descriptor store (SeqSLAM needs per-frame, not
    per-segment). Bounded/pruned on anchor eviction exactly like the ring-key
    per-segment store. Descriptors are unit-normalised ring keys (yaw-invariant),
    reusing ssp_ringkey.grid_from_pts + SC.ring_key.
  * at a drought (reuse the EXISTING trigger + pool constraints — pass
    segregation, _drought_cut, gap): kNN the query frame over the pooled OLD
    keyframes for top-K candidate endpoints j (NMS in kf-index space so the
    shortlist spans DISTINCT places, including twins). For each candidate build
    the SeqSLAM velocity-swept diagonal score between the last L query keyframe
    descriptors and the L old descriptors around j; sweep a small range of signed
    trajectory speeds (both traversal directions) to handle pace/direction.
  * admit a drought closure ONLY when the sequence score clears a margin AND the
    per-frame geometry still verifies: the sequence-confirmed candidate anchor +
    column-shift yaw seed is fed into the EXISTING _drought_verify + innovation
    gate UNCHANGED. SeqSLAM is an ADDITIONAL gate that REPLACES (admit="seq") or
    AUGMENTS (admit="pcm") the PCM pairwise gate for admission; both are compared.

Usage:
  python3 -m experiments.seqslam selftest
  python3 -m experiments.seqslam mit   [data/mit.log]     # separation + ATE, all systems
  python3 -m experiments.seqslam intel [data/intel.log]   # sanity: must not regress
"""

import sys
import time

import numpy as np

import sspslam.encoder as S
import sspslam.lattice as L
import sspslam.frontend as C
import experiments.hier as H
import baselines.scancontext as SC
import experiments.ringkey as RK

N_RING = RK.N_RING
N_SECTOR = RK.N_SECTOR
VALID_MAX = RK.VALID_MAX
MAX_R = RK.MAX_R
TOL_REV = SC.TOL_REV

# ---- SeqSLAM sequence parameters -------------------------------------------
# keyframe spacing ~0.10 m (KEY_TRANS) so L * STRIDE * 0.1 m is the travel window
# the diagonal spans. L=15, STRIDE=6 => ~9 m — long enough that a genuine corridor
# revisit should cross a door/alcove/junction that a twin coincidence would not.
L_SEQ = 15
STRIDE = 6
# signed velocity sweep (fraction of query stride): both traversal DIRECTIONS
# (a corridor is walked both ways) and a small pace range. 0 excluded.
VELS = tuple(v for s in (1.0, -1.0)
             for v in (s * 0.7, s * 0.85, s * 1.0, s * 1.15, s * 1.3))
ENDPOINT_TOL = 3          # +-kf slack on the DB endpoint (mis-centering)
CAND_K = 20               # sequence candidates scored per drought
CAND_NMS = 60             # kf: min index separation between shortlist endpoints
MISS_PENALTY = 1.0        # per-frame cosine-distance for an unavailable DB frame

# admission thresholds (sequence gate). A genuine corridor revisit's diagonal is
# a run of near-identical frames => low mean cosine distance; the margin demands
# the winning place beat the runner-up place (SeqSLAM uniqueness).
SEQ_SCORE_MAX = 0.06      # max mean per-frame cosine distance along best diagonal
SEQ_MARGIN = 1.10         # 2nd-best / best sequence score (place uniqueness)


def _unit(v):
    n = float(np.linalg.norm(v))
    return None if n <= 0 else v / n


class SeqSLAM(RK.RingKeySLAM):
    """RingKeySLAM with the drought candidate generator + admission gate driven
    by a SeqSLAM velocity-swept sequence score over a per-keyframe ring-key
    store. Everything else — matcher, coherence, Hessian/ridge, _drought_verify,
    innovation gate, backend — inherited unchanged."""

    def __init__(self, admit="seq", l_seq=L_SEQ, stride=STRIDE,
                 seq_score_max=SEQ_SCORE_MAX, seq_margin=SEQ_MARGIN, **kw):
        super().__init__(**kw)
        self.admit = admit                 # "seq" (single seq fire) | "pcm"
        self.l_seq = l_seq
        self.stride = stride
        self.seq_score_max = seq_score_max
        self.seq_margin = seq_margin
        self.kf_rk = {}                    # kf -> unit ring key (per-KEYFRAME)
        self._aid_kfs = {}                 # aid -> [kf,...] (for eviction prune)
        self.seq_log = []                  # per drought-attempt diagnostics
        self.n_seq_snap = 0
        # REF-aware diagnostic (offline measurement, NOT used by admission): set
        # self._diag = dict(ref=..., valid=...) before the run and the drought
        # records the sequence score AT the REF-genuine location vs the best
        # twin — the honest "is the diagonal real / does the corridor alias in
        # sequence" test, independent of whether single-frame retrieval surfaces
        # the genuine frame.
        self._diag = None
        self._diag_Ls = [self.l_seq]        # window lengths measured in the diag
        self.diag_log = []

    # ---- per-keyframe ring-key store (bounded like the per-segment store) ---
    def add_keyframe(self, pts, w, guess):
        grid = RK.grid_from_pts(pts) if len(pts) else None
        nev = self.n_evict
        out = super().add_keyframe(pts, w, guess)     # RingKeySLAM store + fold
        if self.n_evict != nev:                       # anchor(s) evicted: prune
            for aid in [a for a in self._aid_kfs if a not in self.segvec]:
                for kf in self._aid_kfs.pop(aid):
                    self.kf_rk.pop(kf, None)
        if grid is not None and grid.sum() > 0:
            uk = _unit(SC.ring_key(grid))
            if uk is not None:
                self.kf_rk[self.k] = uk
                self._aid_kfs.setdefault(self.aid_of(self.k), []).append(self.k)
        return out

    # ---- SeqSLAM velocity-swept diagonal score -----------------------------
    def _seq_score(self, qd, j):
        """Best (lowest) mean per-frame cosine distance along a velocity-swept
        diagonal ending near old keyframe j, matched to the L query descriptors
        qd (qd[-1] is the current frame). Sweeps signed velocities (both
        traversal directions) and a +-ENDPOINT_TOL slack on the DB endpoint.
        Window length is len(qd), so the diagnostic can sweep L in one pass."""
        Ln = len(qd)
        rk = self.kf_rk
        best = np.inf
        for s in range(j - ENDPOINT_TOL, j + ENDPOINT_TOL + 1):
            for v in VELS:
                acc = 0.0
                for i in range(Ln):
                    db = int(round(s + v * self.stride * (i - (Ln - 1))))
                    d = rk.get(db)
                    acc += MISS_PENALTY if d is None else 1.0 - float(qd[i] @ d)
                m = acc / Ln
                if m < best:
                    best = m
        return best

    # ---- REF-aware diagnostic: genuine-vs-twin sequence-score separation ----
    def _diag_record(self, k, qd, qcur, pool, sd, retrieved):
        """Offline (REF-aware) measurement — NOT used by admission. For a drought
        attempt at query keyframe k, locate the GENUINE old keyframes (pool kf
        within TOL_REV of k in the REF frame) and the TWINS (pool kf > TOL_REV
        away), and compute the SeqSLAM diagonal score at both. This answers 'do
        genuine revisits score better, and does the corridor alias in sequence?'
        directly, regardless of whether single-frame retrieval surfaced the
        genuine frame."""
        ref, valid = self._diag["ref"], self._diag["valid"]
        if not valid[k]:
            return
        qxy = ref[k]
        pv = valid[pool]
        dref = np.full(pool.size, np.inf)
        dref[pv] = np.linalg.norm(ref[pool[pv]] - qxy, axis=1)
        gen = pool[dref <= TOL_REV]
        if gen.size == 0:
            return                                   # not a REF true revisit
        twin = pool[(dref > TOL_REV) & np.isfinite(dref)]
        gsel = gen[np.argsort(dref[dref <= TOL_REV])[:5]]     # REF-nearest genuine
        tpick = []                                            # best distinct twins
        if twin.size:
            ts = 1.0 - np.stack([self.kf_rk[int(j)] for j in twin]) @ qcur
            for j in twin[np.argsort(ts, kind="stable")]:
                if all(abs(int(j) - c) >= CAND_NMS for c in tpick):
                    tpick.append(int(j))
                    if len(tpick) >= 15:
                        break
        g_single = float(min(1.0 - self.kf_rk[int(j)] @ qcur for j in gsel))
        t_single = float(min((1.0 - self.kf_rk[j] @ qcur for j in tpick),
                             default=np.inf))
        gen_set = set(int(j) for j in gen)
        rec = dict(k=int(k), n_gen=int(gen.size), n_twin=int(twin.size),
                   g_single=g_single, t_single=t_single,
                   gen_retrieved=bool(gen_set & set(int(j) for j in retrieved)))
        # sequence scores at one or more WINDOW LENGTHS (the aliasing-vs-window
        # honesty test): for each L build the query window from qcur + the L-1
        # older stored frames, score genuine & twin diagonals.
        for Ln in self._diag_Ls:
            q = self._build_query(k, qcur, Ln)
            if q is None:
                rec[f"g_seq_{Ln}"] = rec[f"t_seq_{Ln}"] = np.nan
                continue
            rec[f"g_seq_{Ln}"] = float(min(self._seq_score(q, int(j))
                                           for j in gsel))
            rec[f"t_seq_{Ln}"] = float(min((self._seq_score(q, j) for j in tpick),
                                           default=np.inf))
        # primary-L aliases (the run's own l_seq) for diag_report
        rec["g_seq"] = rec.get(f"g_seq_{self.l_seq}", np.nan)
        rec["t_seq"] = rec.get(f"t_seq_{self.l_seq}", np.nan)
        self.diag_log.append(rec)

    def _build_query(self, k, qcur, Ln):
        """Query window of length Ln ending at the current frame (qcur), older
        frames drawn from the per-keyframe store. None if any older frame is
        missing (rare — dense keyframes)."""
        q_old = [k - (Ln - 1 - i) * self.stride for i in range(Ln - 1)]
        if q_old and (q_old[0] < 0 or any(kf not in self.kf_rk for kf in q_old)):
            return None
        return [self.kf_rk[kf] for kf in q_old] + [qcur]

    # ---- drought: sequence shortlist + gate -> UNCHANGED verifier ----------
    def _try_drought(self, pts, w):
        k = self.k
        self.n_drought_try += 1
        me = self.pose_of(k)
        my_aid = self.aid_of(k)
        dist_since = self.dist_trav - self.dist_at_accept
        log = dict(k=k, phase="cells", z=np.nan, chi=np.nan, ratio=np.nan)
        self.drought_log.append(log)

        # query sequence: last L keyframes ending at the CURRENT frame k (stride
        # ST), ascending. LIFECYCLE: _try_drought runs INSIDE HybridSLAM.
        # add_keyframe (experiments/hier.py:828), BEFORE our add_keyframe wrapper stores
        # kf_rk[k] — so kf_rk holds only 0..k-1 here. The current frame's
        # descriptor is therefore computed DIRECTLY from pts (available), and
        # only the L-1 OLDER query frames are drawn from the stored kf_rk.
        qgrid = RK.grid_from_pts(pts)
        qcur = _unit(SC.ring_key(qgrid))
        if qcur is None:
            return
        q_old = [k - (self.l_seq - 1 - i) * self.stride
                 for i in range(self.l_seq - 1)]
        if q_old[0] < 0 or any(kf not in self.kf_rk for kf in q_old):
            return
        qd = [self.kf_rk[kf] for kf in q_old] + [qcur]

        # candidate pool: SAME pass-segregation / _drought_cut / gap as shipped,
        # but over KEYFRAMES (SeqSLAM is per-frame). Need room for the diagonal.
        gap_a = self.gap_kf // ANCHOR
        cut = self._drought_cut()
        span = int(round(max(v for v in VELS) * self.stride * (self.l_seq - 1)))
        pool = [j for j in self.kf_rk
                if my_aid - self.aid_of(j) > gap_a
                and (cut is None or self.aid_of(j) <= cut)
                and j - span >= 0]
        if not pool:
            return
        pool = np.array(sorted(pool), dtype=int)

        # single-frame kNN of the current query frame -> distinct-place shortlist
        PK = np.stack([self.kf_rk[j] for j in pool])
        sd = 1.0 - PK @ qcur                       # cosine distance
        order = pool[np.argsort(sd, kind="stable")]
        cand = []
        for j in order:
            if all(abs(int(j) - c) >= CAND_NMS for c in cand):
                cand.append(int(j))
                if len(cand) >= CAND_K:
                    break

        # SeqSLAM diagonal score per candidate endpoint
        sfd = {int(j): float(sd[np.searchsorted(pool, j)]) for j in cand}
        scored = sorted(((self._seq_score(qd, j), j) for j in cand),
                        key=lambda t: (t[0], t[1]))
        cands_log = [dict(j=int(j), aid=int(self.aid_of(j)), seq=float(sc),
                          single=sfd[int(j)]) for sc, j in scored]
        best_seq, best_j = scored[0]
        # place-uniqueness margin: best runner-up from a DIFFERENT place
        runner = next((sc for sc, j in scored[1:]
                       if abs(j - best_j) >= CAND_NMS), np.inf)
        margin = runner / best_seq if best_seq > 0 else np.inf
        entry = dict(k=k, my_aid=int(my_aid), pool=int(pool.size),
                     cands=cands_log, best_j=int(best_j), best_seq=float(best_seq),
                     runner=float(runner), margin=float(margin), admitted=False)
        self.seq_log.append(entry)

        if self._diag is not None:
            self._diag_record(k, qd, qcur, pool, sd, [c["j"] for c in cands_log])

        log["phase"] = "coarse"
        log["nsh"] = 1
        noff = int(pool.size)
        log["noff"] = noff
        self.n_drought_hyp += 1

        # sequence GATE: absolute score + place-uniqueness margin
        if best_seq > self.seq_score_max or margin < self.seq_margin:
            return

        # feed the sequence-confirmed candidate to the UNCHANGED verifier
        aid = int(self.aid_of(best_j))
        if aid not in self.sc_grid:
            return
        yaw = self._yaw_seed(qgrid, aid)          # reuse the query grid above
        hyp = np.array([self.anchors[aid][0], self.anchors[aid][1], yaw])
        res = self._drought_verify(pts, w, hyp, my_aid)
        if res is None:
            return
        pose, c_aid, chain, ratio = res
        self.n_drought_verify += 1
        log.update(phase="verified", ratio=ratio, z=best_seq, pose=pose.copy(),
                   c_aid=c_aid, hyp=pose.copy())

        # ---- innovation gate (UNCHANGED, verbatim from HybridSLAM) ----------
        _, rel = self.kf_ref[k]
        Zk = L.se2_mul(pose, L.se2_inv(rel))
        Z = L.se2_mul(L.se2_inv(self.anchors[c_aid]), Zk)
        Zc = L.se2_mul(L.se2_inv(self.anchors[c_aid]), self.anchors[my_aid])
        lever = np.linalg.norm(pose[:2] - self.anchors[c_aid][:2])
        sig_t = np.sqrt(0.08 ** 2 + (0.05 * lever) ** 2)
        sig_r = np.deg2rad(2.0)
        s_at = sig_t + max(0.30, self.drought_slope * dist_since)
        s_ar = sig_r + np.deg2rad(20.0)
        chi = (np.linalg.norm(Z[:2] - Zc[:2]) / s_at) ** 2 \
            + (S.wrap(Z[2] - Zc[2]) / s_ar) ** 2
        log["chi"] = float(chi)
        if self.use_innov and chi > 9.0:
            self.n_innov_rej += 1
            return
        T = L.se2_mul(self.anchors[c_aid], Z)
        cand_entry = dict(k=k, my_aid=my_aid, c_aid=c_aid, Z=Z, sig_t=sig_t,
                          sig_r=sig_r, T=T,
                          b_pos=self.anchors[my_aid][:2].copy(),
                          dr=S.wrap(T[2] - self.anchors[my_aid][2]),
                          noff=noff, lever=float(lever), seq=float(best_seq),
                          margin=float(margin), best_j=int(best_j))

        if self.admit == "pcm":
            self._admit_pcm(cand_entry, log)
        else:
            self._admit_seq(cand_entry, log)
        entry["admitted"] = (log["phase"] == "snap")

    # ---- admission tail: sequence-only (single confirmed fire snaps) -------
    def _admit_seq(self, e, log):
        """SeqSLAM REPLACES the PCM pairwise-geometric consensus: a single
        sequence-confirmed + geometry-verified + innovation-gated fire snaps.
        This is the head-on test — does temporal consistency alone admit
        genuine revisits that pairwise-PCM could not?"""
        log["phase"] = "snap"
        log["clique"] = 1
        log["my_aid"] = int(e["my_aid"])
        log["clique_pairs"] = [(int(e["c_aid"]), int(e["my_aid"]))]
        log["seq"] = float(e["seq"])
        log["margin"] = float(e["margin"])
        log["pcm_deep"] = False
        self._insert_edges([e])
        self.n_seq_snap += 1
        self.n_drought_snap += 1

    # ---- admission tail: sequence gate AND PCM clique (verbatim PCM) -------
    def _admit_pcm(self, e, log):
        log["phase"] = "pending"
        self._drought_cand = [c for c in self._drought_cand
                              if e["k"] - c["k"] <= self.cand_window]
        self._drought_cand.append(e)
        self._pcm_best = 1
        self._pcm_deep = False
        clique = self._pcm_admit(self._drought_cand)
        log["cand"] = len(self._drought_cand)
        log["clique_max"] = int(self._pcm_best)
        if clique is None:
            return
        log["phase"] = "snap"
        log["clique"] = len(clique)
        log["my_aid"] = int(e["my_aid"])
        log["pcm_deep"] = bool(self._pcm_deep)
        log["clique_pairs"] = [(int(c["c_aid"]), int(c["my_aid"])) for c in clique]
        log["seq"] = float(e["seq"])
        log["margin"] = float(e["margin"])
        self._insert_edges(clique)
        self._drought_cand = []
        self.n_seq_snap += 1
        self.n_drought_snap += 1

    def _insert_edges(self, clique):
        """Distribute-correction + edge insertion tail, verbatim from
        HybridSLAM._try_drought (experiments/hier.py ~1454-1476)."""
        tail = clique[-1]
        self._distribute_correction(tail["my_aid"], tail["T"])
        for e in clique:
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
        self.last_accept_k = self.k
        self.last_true_accept_k = self.k
        self.dist_at_accept = self.dist_trav
        self._streak = 0
        self._streak_xy = None


ANCHOR = H.ANCHOR


# ---------------------------------------------------------------------------
# CARMEN harness (MIT / Intel), range-identity REF ATE convention.
# ---------------------------------------------------------------------------

def run_carmen(name, path, system, drought=500, verbose=True, diag=False, **skw):
    """system: 'base' (shipped HybridSLAM), 'seq' (SeqSLAM admit='seq'),
    'seqpcm' (SeqSLAM admit='pcm'). Returns ATE (range-identity REF) + drought
    stats (+ the SeqSLAM per-attempt sequence log). diag=True attaches the
    REF-aware genuine-vs-twin sequence-score diagnostic."""
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    odom = np.stack([k[1] for k in keys])
    if system == "base":
        slam = H.HybridSLAM(beta=3.0, seg_nring=4, attempt_every=4,
                            relax_every=25, gap_kf=300, recent_aids=12,
                            drought_kf=drought)
    else:
        slam = SeqSLAM(admit=("pcm" if system == "seqpcm" else "seq"),
                       beta=3.0, seg_nring=4, attempt_every=4, relax_every=25,
                       gap_kf=300, recent_aids=12, drought_kf=drought, **skw)
    if diag and system != "base":
        _, dref, dvalid = SC.ref_positions(name, keys)
        slam._diag = dict(ref=dref, valid=dvalid)
    slam.dtheta_beam = np.pi / n_beams
    est = np.zeros((n, 3))
    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < 40.0, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = opose if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
        if verbose and k % 2000 == 0:
            print(f"    [{name} {system}] kf {k}/{n} t={time.time()-t0:.0f}s "
                  f"loops={sum(1 for e in slam.edges if e[5]=='loop')} "
                  f"snaps={slam.n_drought_snap}", flush=True)
    if slam.dirty:
        slam.relax()
    dt = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    kts2, ref, valid = SC.ref_positions(name, keys)
    al = C.align_se2(fin[valid, :2], ref[valid])
    e = np.linalg.norm(al - ref[valid], axis=1)
    n_loop = sum(1 for ed in slam.edges if ed[5] == "loop")
    return dict(name=name, system=system, n=n, secs=dt,
                ate=float(np.sqrt((e ** 2).mean())), med=float(np.median(e)),
                mx=float(e.max()), loops=n_loop,
                d_try=slam.n_drought_try, d_hyp=slam.n_drought_hyp,
                d_verify=slam.n_drought_verify, d_snap=slam.n_drought_snap,
                snaps=[g for g in slam.drought_log if g["phase"] == "snap"],
                seq_log=getattr(slam, "seq_log", []),
                diag_log=getattr(slam, "diag_log", []),
                kf_ref=list(slam.kf_ref), ref=ref, valid=valid)


def window_sweep(name, path, Ls=(8, 15, 25, 40), stride=STRIDE):
    """ONE MIT pass that, at every REF true-revisit drought, scores the genuine
    and best-twin diagonals at SEVERAL window lengths L. Tests the honest caveat
    head-on: does a LONGER sequence window (more travel, more chance to cross a
    junction that breaks corridor symmetry) EVER separate genuine from twin, or
    does the corridor alias at every length?"""
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    odom = np.stack([k[1] for k in keys])
    # large l_seq only sizes the live query window (unused for admission here —
    # admit stays 'seq' but we read only diag); stride shared with the sweep.
    slam = SeqSLAM(admit="seq", beta=3.0, seg_nring=4, attempt_every=4,
                   relax_every=25, gap_kf=300, recent_aids=12, drought_kf=500,
                   l_seq=max(Ls), stride=stride)
    _, dref, dvalid = SC.ref_positions(name, keys)
    slam._diag = dict(ref=dref, valid=dvalid)
    slam._diag_Ls = list(Ls)
    slam.dtheta_beam = np.pi / n_beams
    est = np.zeros((n, 3))
    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < 40.0, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = opose if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
        if k % 2000 == 0:
            print(f"    [{name} sweep] kf {k}/{n} t={time.time()-t0:.0f}s",
                  flush=True)
    dl = slam.diag_log
    print(f"\n  window-length sweep over {len(dl)} REF true-revisit droughts "
          f"(stride {stride}, ~{stride*0.1:.1f} m/step):")
    print(f"  {'L':>4} {'travel_m':>9}  {'gen_med':>8} {'twin_med':>8}  "
          f"{'gen<twin':>9}  {'frac_ratio>1.1':>14}")
    for Ln in Ls:
        g = np.array([d[f"g_seq_{Ln}"] for d in dl], float)
        t = np.array([d[f"t_seq_{Ln}"] for d in dl], float)
        m = np.isfinite(g) & np.isfinite(t)
        if not m.any():
            continue
        sep = np.mean(g[m] < t[m])
        ratio = t[m] / np.maximum(g[m], 1e-9)
        print(f"  {Ln:>4} {Ln*stride*0.1:>9.1f}  {np.median(g[m]):>8.4f} "
              f"{np.median(t[m]):>8.4f}  {sep:>9.3f}  {np.mean(ratio>1.1):>14.3f}")
    print("  (gen<twin = fraction of true revisits where the genuine diagonal "
          "beats the best twin; 0.5 = no separation, corridor aliases)")
    return dl


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _ref_tools(res):
    ref, valid, kf_ref = res["ref"], res["valid"], res["kf_ref"]
    aid_kf = {}
    for kf, (aid, _) in enumerate(kf_ref):
        aid_kf.setdefault(aid, kf)

    def ref_of_kf(kf):
        return ref[kf] if (0 <= kf < len(valid) and valid[kf]) else None

    def ref_of_aid(aid):
        kf = aid_kf.get(aid)
        return ref_of_kf(kf) if kf is not None else None
    return ref_of_kf, ref_of_aid


def separation_report(res):
    """THE CRUX. For every drought attempt classify each scored candidate
    endpoint as GENUINE (query kf & candidate kf within TOL_REV in the REF frame)
    or TWIN, and compare their SeqSLAM sequence scores. Reports:
      (1) at attempts that ARE true revisits, does the best GENUINE candidate
          outscore the best TWIN candidate (is the diagonal real / separable)?
      (2) pooled genuine-vs-twin sequence-score distributions.
      (3) single-frame vs sequence: does the diagonal add discrimination?"""
    ref_of_kf, _ = _ref_tools(res)
    seq_log = res["seq_log"]
    tol = TOL_REV
    gen_scores, twin_scores = [], []
    gen_single, twin_single = [], []
    n_rev = win_gen = tie = win_twin = 0            # per true-revisit attempt
    n_rev_single = 0                                # single-frame would-win
    for a in seq_log:
        q = ref_of_kf(a["k"])
        if q is None:
            continue
        gj, tj = [], []                             # (seq, single) per class
        for c in a["cands"]:
            cp = ref_of_kf(c["j"])
            if cp is None:
                continue
            genuine = np.linalg.norm(cp - q) <= tol
            (gj if genuine else tj).append((c["seq"], c["single"]))
            (gen_scores if genuine else twin_scores).append(c["seq"])
            (gen_single if genuine else twin_single).append(c["single"])
        if not gj:                                  # not a true revisit here
            continue
        n_rev += 1
        bg = min(s for s, _ in gj)
        bt = min((s for s, _ in tj), default=np.inf)
        win_gen += bg < bt
        win_twin += bt < bg
        tie += bg == bt
        # would a single-frame ring-key have picked the genuine place?
        bg1 = min(s for _, s in gj)
        bt1 = min((s for _, s in tj), default=np.inf)
        n_rev_single += bg1 < bt1

    def stats(x):
        x = np.array(x, float)
        if x.size == 0:
            return "  (none)"
        return (f"n={x.size:5d}  mean={x.mean():.4f}  med={np.median(x):.4f}  "
                f"p10={np.percentile(x,10):.4f}  min={x.min():.4f}")

    print("  --- genuine-vs-twin SEQUENCE-score separation (lower = better) ---")
    print(f"  GENUINE candidate seq scores: {stats(gen_scores)}")
    print(f"  TWIN    candidate seq scores: {stats(twin_scores)}")
    print(f"  GENUINE candidate single    : {stats(gen_single)}")
    print(f"  TWIN    candidate single    : {stats(twin_single)}")
    if n_rev:
        print(f"  true-revisit attempts (query has a GENUINE candidate): {n_rev}")
        print(f"    SEQUENCE picks GENUINE over best twin: {win_gen}/{n_rev} "
              f"({win_gen/n_rev:.3f})   twin wins: {win_twin}   ties: {tie}")
        print(f"    SINGLE-frame picks GENUINE over twin : {n_rev_single}/{n_rev} "
              f"({n_rev_single/n_rev:.3f})   (does the diagonal add over 1 frame?)")
    else:
        print("  (no drought attempt had a GENUINE REF-frame candidate)")
    return dict(n_rev=n_rev, win_gen=win_gen, win_twin=win_twin, tie=tie,
                n_rev_single=n_rev_single,
                gen=np.array(gen_scores), twin=np.array(twin_scores))


def diag_report(res):
    """THE CRUX (REF-aware, retrieval-independent). Over every drought attempt
    that IS a true REF revisit, compare the SeqSLAM diagonal score AT the genuine
    location vs the best twin — measured directly from ground truth, so it does
    NOT depend on single-frame retrieval surfacing the genuine frame. Answers:
    do genuine revisits score better, and does the corridor ALIAS in sequence?"""
    dl = res.get("diag_log", [])
    print(f"  REF true-revisit drought attempts measured: {len(dl)}")
    if not dl:
        print("  (diagnostic not attached / no REF true-revisit droughts)")
        return dict(n=0)
    g_seq = np.array([d["g_seq"] for d in dl])
    t_seq = np.array([d["t_seq"] for d in dl])
    g_sin = np.array([d["g_single"] for d in dl])
    t_sin = np.array([d["t_single"] for d in dl])
    fin = np.isfinite(t_seq)
    retr = np.mean([d["gen_retrieved"] for d in dl])

    def stat(x):
        x = x[np.isfinite(x)]
        return (f"n={x.size:4d} mean={x.mean():.4f} med={np.median(x):.4f} "
                f"p10={np.percentile(x,10):.4f} p90={np.percentile(x,90):.4f}")

    print(f"  GENUINE sequence score: {stat(g_seq)}")
    print(f"  TWIN    sequence score: {stat(t_seq)}")
    print(f"  GENUINE single-frame  : {stat(g_sin)}")
    print(f"  TWIN    single-frame  : {stat(t_sin)}")
    sep_seq = np.mean(g_seq[fin] < t_seq[fin])
    sep_sin = np.mean(g_sin[fin] < t_sin[fin])
    print(f"  genuine BEATS best twin (lower score):  "
          f"SEQUENCE {sep_seq:.3f}   SINGLE-frame {sep_sin:.3f}   "
          f"(n={int(fin.sum())})")
    print(f"  genuine frame present in retrieved top-{CAND_K} shortlist: "
          f"{retr:.3f}")
    # threshold-sensitivity: at each SEQ_SCORE_MAX, how many GENUINE vs TWIN
    # attempts have their (best-twin) score below it => would clear the abs gate
    print("  --- gate sensitivity: attempts with score < thresh (gen | twin) ---")
    for thr in (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20):
        gp = int(np.sum(g_seq <= thr))
        tp = int(np.sum(t_seq[fin] <= thr))
        print(f"    thresh {thr:.2f}:  genuine {gp:4d}/{len(dl)}   "
              f"twin {tp:4d}/{int(fin.sum())}")
    # margin view: genuine attempts where genuine outscores twin by the margin
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = t_seq / np.maximum(g_seq, 1e-9)      # >1 => genuine better
    print("  --- margin (best-twin / genuine): >1 means genuine wins ---")
    r = ratio[np.isfinite(ratio)]
    if r.size:
        print(f"    mean={r.mean():.3f} med={np.median(r):.3f} "
              f"p10={np.percentile(r,10):.3f} frac>1.10={np.mean(r>1.10):.3f} "
              f"frac>1.0={np.mean(r>1.0):.3f}")
    return dict(n=len(dl), sep_seq=float(sep_seq), sep_sin=float(sep_sin),
                g_seq=g_seq, t_seq=t_seq, retr=float(retr))


def snap_report(res):
    """Per-snap validation: every sequence-confirmed snap, GENUINE vs TWIN by
    REF (both anchors' REF positions within TOL_REV), with the seq score."""
    _, ref_of_aid = _ref_tools(res)
    tol = TOL_REV
    snaps = res["snaps"]
    print(f"  sequence-confirmed snaps: {len(snaps)}")
    if not snaps:
        print("  (no snaps)")
        return dict(n_snap=0, n_genuine=0, n_twin=0)
    print(f"  {'k':>6} {'clique':>6} {'seq':>7} {'margin':>7}  "
          f"{'pair(c->b)':>14} {'REFdist':>8}  verdict")
    n_gen = n_twin = 0
    for g in snaps:
        pairs = g.get("clique_pairs", [(g.get("c_aid"), g.get("my_aid"))])
        for i, (c_aid, my_aid) in enumerate(pairs):
            rc, rb = ref_of_aid(c_aid), ref_of_aid(my_aid)
            if rc is None or rb is None:
                dist, verdict = float("nan"), "REF-invalid"
            else:
                dist = float(np.linalg.norm(rc - rb))
                genuine = dist <= tol
                verdict = "GENUINE" if genuine else "TWIN/false"
                n_gen += genuine
                n_twin += (not genuine)
            head = (f"  {g['k']:>6} {g.get('clique',1):>6} "
                    f"{g.get('seq',float('nan')):>7.4f} "
                    f"{g.get('margin',float('nan')):>7.3f}") if i == 0 \
                else " " * 31
            print(f"{head}  {c_aid:>5}->{my_aid:<7} {dist:>8.2f}  {verdict}")
    print(f"  clique edges: {n_gen} GENUINE, {n_twin} TWIN/false")
    return dict(n_snap=len(snaps), n_genuine=n_gen, n_twin=n_twin)


# ---------------------------------------------------------------------------

def selftest():
    print("selftest: per-keyframe ring-key store + SeqSLAM diagonal")
    rng = np.random.default_rng(0)
    n_beams = 180
    beam = np.deg2rad(np.arange(n_beams) - 90.0)

    # (1) per-keyframe store populates and stays bounded to live anchors
    slam = SeqSLAM(seg_nring=4, drought_kf=0)
    for k in range(40):
        rr = rng.uniform(1.0, 30.0, n_beams)
        p, ww, _ = S.scan_to_samples(np.where(rr < VALID_MAX, rr, np.inf), beam)
        slam.add_keyframe(p, ww, np.array([0.1 * k, 0.0, 0.0]))
    assert slam.kf_rk, "no per-keyframe ring keys stored"
    for kf in slam.kf_rk:
        assert slam.aid_of(kf) in slam.segvec, "kf ring key for a dead anchor"
        assert abs(np.linalg.norm(slam.kf_rk[kf]) - 1.0) < 1e-9, "not unit"
    print(f"  stored {len(slam.kf_rk)} per-keyframe unit ring keys, "
          f"all backed by a live anchor OK")

    # (2) the diagonal scores a genuine sequence LOWER than a shuffled one.
    #     Build a synthetic descriptor track; a query window that is a noisy
    #     copy of an old run should score ~0 at the right endpoint and high at a
    #     random endpoint / against a scrambled DB.
    slam2 = SeqSLAM(seg_nring=4, drought_kf=0, l_seq=10, stride=1)
    track = [_unit(rng.uniform(0.1, 1.0, N_RING)) for _ in range(300)]
    slam2.kf_rk = {i: track[i] for i in range(300)}
    Ln = slam2.l_seq
    j0 = 200                                        # true old endpoint
    qd = [track[j0 - (Ln - 1) + i] for i in range(Ln)]   # exact replay (v=+1)
    s_true = slam2._seq_score(qd, j0)
    s_wrong = min(slam2._seq_score(qd, jr)
                  for jr in (60, 100, 140, 260))    # unrelated endpoints
    assert s_true < 1e-6, f"true-diagonal score should be ~0, got {s_true}"
    assert s_wrong > 10 * max(s_true, 1e-9) and s_wrong > 0.05, \
        f"unrelated endpoint too low ({s_wrong}) vs true ({s_true})"
    # reversed traversal must also be found (signed velocity): the robot walks
    # the old run backwards, so its NEWEST frame is track[j0-(Ln-1)] and the DB
    # endpoint it lands on is j0-(Ln-1); older query frames climb back up to j0.
    j1 = j0 - (Ln - 1)
    qd_rev = [track[j0 - i] for i in range(Ln)]      # newest (i=Ln-1) = track[j1]
    s_rev = slam2._seq_score(qd_rev, j1)
    assert s_rev < 1e-6, f"reverse-traversal diagonal should be ~0, got {s_rev}"
    print(f"  diagonal: true={s_true:.2e}  reverse={s_rev:.2e}  "
          f"unrelated={s_wrong:.3f} OK (velocity sweep finds both directions)")

    # (3) LIFECYCLE: _try_drought runs INSIDE super().add_keyframe, before the
    #     wrapper stores kf_rk[k]. A regression that draws the current query
    #     frame from kf_rk (instead of from pts) would early-return on EVERY
    #     drought and leave seq_log empty. Drive a real drought and assert the
    #     path actually executed (seq_log populated, current frame scored).
    slam3 = SeqSLAM(seg_nring=4, drought_kf=40, gap_kf=50, l_seq=5, stride=2,
                    attempt_every=4)
    slam3.dtheta_beam = np.pi / n_beams
    base = [_unit(rng.uniform(0.2, 1.0, N_RING)) for _ in range(30)]
    for k in range(240):                              # loop a 30-frame world
        rr = rng.uniform(1.0, 30.0, n_beams)
        p, ww, _ = S.scan_to_samples(np.where(rr < VALID_MAX, rr, np.inf), beam)
        slam3.coh_ref = 0.5                           # allow the verifier to run
        slam3.add_keyframe(p, ww, np.array([0.05 * (k % 30), 0.0, 0.0]))
    assert slam3.n_drought_try > 0, "no drought fired (test setup)"
    assert slam3.seq_log, ("LIFECYCLE BUG: drought fired but seq_log EMPTY — "
                           "the current query frame is not available at drought "
                           "time (kf_rk[k] not yet stored)")
    assert all(e["k"] not in slam3.kf_rk for e in slam3.seq_log), \
        "current frame k should NOT be in kf_rk at its own drought"
    print(f"  lifecycle: {slam3.n_drought_try} droughts fired, "
          f"{len(slam3.seq_log)} scored (current frame from pts, not kf_rk) OK")
    print("selftest PASSED")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        selftest()
        return
    if cmd == "sweep":
        name = sys.argv[2] if len(sys.argv) > 2 else "mit"
        path = sys.argv[3] if len(sys.argv) > 3 else f"data/{name}.log"
        print(f"=== {name.upper()}: SeqSLAM window-length sweep (aliasing test) ===")
        window_sweep(name, path)
        return
    if cmd not in ("mit", "intel"):
        raise SystemExit(
            f"unknown command {cmd!r} (selftest | mit | intel | sweep)")
    path = sys.argv[2] if len(sys.argv) > 2 else f"data/{cmd}.log"
    print(f"=== {cmd.upper()}: SeqSLAM sequence consistency vs the corridor wall ===")

    def show(tag, r):
        print(f"[{tag:16}] ATE {r['ate']:.3f} m  med {r['med']:.3f}  "
              f"max {r['mx']:.2f}  loops {r['loops']}  "
              f"try/hyp/ver/snap {r['d_try']}/{r['d_hyp']}/"
              f"{r['d_verify']}/{r['d_snap']}  {r['secs']:.0f}s", flush=True)

    # honest baselines: drought ON (shipped HY4) and drought DISABLED — so a
    # seq change cannot be confused with merely turning the twin-scattering
    # drought relocalizer off.
    show("baseline HY4", run_carmen(cmd, path, "base", drought=500))
    show("HY4 drought-OFF", run_carmen(cmd, path, "base", drought=0))
    for system in ("seq", "seqpcm"):
        r = run_carmen(cmd, path, system, diag=True)
        show(f"SeqSLAM {system}", r)
        print(f"--- REF-aware genuine-vs-twin DIAGONAL separation ({system}) ---")
        diag_report(r)
        print(f"--- retrieved-candidate separation ({system}) ---")
        separation_report(r)
        print(f"--- SNAP validation ({system}) ---")
        snap_report(r)


if __name__ == "__main__":
    main()
