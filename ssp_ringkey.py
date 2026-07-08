"""Scan-Context RING-KEY shortlister inside the DROUGHT relocalization path.

Motivation (RESULTS.md R3/R4 + ssp_scancontext.py): the MIT Infinite-Corridor
drought relocalizer scores 0/106 stage-1 hits at deep true revisits — the
coarse SSP relocalization ring (lam 5.3/12.8, world-frame summed) carries NO
place-recognition information there. ssp_scancontext.py then proved the info is
NOT gone from the raw scan: a rotation-invariant Scan-Context RING KEY (per-ring
mean occupancy of a 20-ring x 60-sector polar point-density matrix) recovers the
same MIT revisits at recall@1=0.272 (19x chance) / recall@40=0.674. So the
corridor limit is SSP-LOSSY (representation), not environmental.

This module tests whether feeding the fine drought VERIFIER the ring-key's
top-40 candidates — the correct old anchor the coarse SSP band never surfaced —
breaks the MIT corridor limit, WITHOUT weakening the verifier.

Design (subclass; NO edit to any shipped module — same pattern as ssp_angles.py
and ssp_scancontext.py):
  * ONE ring-key per SEGMENT (anchor), computed from that segment's raw scan(s)
    at fold time as the mean Scan-Context grid over its keyframes. It is STATIC
    appearance (never moves with the graph); the candidate POSE it points to is
    the anchor's LIVE graph pose (anchors[aid]). This sidesteps the R1/R2
    world-frame-summing law: the descriptor is a body-frame appearance vector,
    the geometry comes from the correctable graph.
  * At a drought (trigger / _drought_cut / gap / pass-segregation UNCHANGED):
    kNN the query ring-key against pre-drought segment ring-keys -> top-40
    candidate ANCHORS. Each anchor's graph position + a column-shift yaw seed
    becomes a hypothesis handed to the EXISTING _drought_verify UNCHANGED
    (cmatcher seed grid + coherence + Hessian/ridge + PCM / deep-search
    escalation). We keep the single best-coherence verified fire per attempt and
    run it through the IDENTICAL chi-gate + PCM admission tail. The coarse-SSP
    hypothesis generator (_drought_offsets / _coarse_hyp) is bypassed; the
    verifier is untouched — the whole point is to give it the right candidate.

The ring-key descriptor + column-shift are REUSED BY IMPORT from
ssp_scancontext (SC.scan_context / SC.ring_key / SC.sc_distance) — not
reimplemented.

Honesty bound (ssp_scancontext R4 verdict: retrieval is NECESSARY-NOT-SUFFICIENT
— corridor TWINS align almost perfectly and defeat the fine verifier). Two
separate numbers, both reported:
  (a) RETRIEVAL: at each MIT drought attempt, is a correct old anchor (within
      ~5 m in the REF frame of the query) in the top-40 ring-key shortlist?
      vs the coarse-band whitened-correlation shortlist on the IDENTICAL
      attempts and the IDENTICAL candidate pool.
  (b) ATE: shipped HybridSLAM (coarse-band drought) vs ring-key HybridSLAM on
      the full MIT log, plus drought stats. MIT gfs timestamps are corrupt, so
      ATE uses the range-array-IDENTITY REF convention (SC.ref_positions).

Usage:
  python3 ssp_ringkey.py selftest
  python3 ssp_ringkey.py mit [data/mit.log]     # retrieval + ATE, both systems
  python3 ssp_ringkey.py intel [data/intel.log] # sanity: ring-key must not regress
"""

import sys
import time

import numpy as np

import ssp_slam as S
import ssp_slam_loop as L
import ssp_slam_carmen as C
import ssp_hier as H
import ssp_scancontext as SC

NA = H.NA
ANCHOR = H.ANCHOR
WC = H.WC
_sub_rot_permute = H._sub_rot_permute

# Scan-Context grid extent (matches ssp_scancontext MIT config: 20x60, R=50 m)
N_RING = SC.N_RING
N_SECTOR = SC.N_SECTOR
MAX_R = 50.0
VALID_MAX = 50.0
TOP_K = 40                 # shortlist size fed to the verifier

# Yaw-seed sign convention (relative body-frame column shift -> world heading of
# the query given the candidate anchor heading). Pinned empirically in selftest.
YAW_SIGN = 1.0


def grid_from_pts(pts):
    """Scan-Context grid (20x60 point density) from body-frame samples, reusing
    SC.scan_context by feeding it per-point ranges and bearings."""
    if len(pts) == 0:
        return np.zeros((N_RING, N_SECTOR), float)
    r = np.linalg.norm(pts, axis=1)
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    return SC.scan_context(r, ang, VALID_MAX, MAX_R, N_RING, N_SECTOR)


class RingKeySLAM(H.HybridSLAM):
    """HybridSLAM with the drought candidate generator replaced by a
    Scan-Context ring-key shortlister. Everything else — matcher, coherence,
    Hessian/ridge, PCM/deep-search admission, backend — inherited unchanged."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.ringkey = {}          # aid -> 20-float yaw-invariant ring key
        self.sc_grid = {}          # aid -> 20x60 mean SC grid (for the yaw seed)
        self._grid_sum = {}        # aid -> running grid sum
        self._grid_cnt = {}        # aid -> #scans folded
        # per-drought-attempt retrieval diagnostics (filled by _try_drought)
        self.retr_log = []
        # DEEP-CONSENSUS ESCALATION for the ring-key source. The shipped
        # deep_noff=6000 counts 0.8 m cells swept by the coarse search; a
        # ring-key attempt does not sweep cells, it RANKS a pool of anchors, so
        # 6000 is unreachable (MIT pool <= ~2839) and clique>=2 would admit
        # every corridor snap — exactly the twin-prone case R4/R5 built the
        # clique>=3 tier for. The comparable crowding unit here is ANCHORS (~1
        # per place); the shipped 6000-cell corridor sweep (0.8 m cells, 4.8 m
        # dilation) spans ~a few hundred anchors of overlapping disks, so a
        # pool of >~500 anchors is the ring-key analog of a "deep/crowded"
        # search. noff is set to the pool size in _try_drought; this threshold
        # makes a fire drawn from a corridor-scale pool demand a 3-way
        # consistent family, on the same footing as the shipped path (and
        # strictly stricter, since ring-key top-40 is maximally twin-prone).
        self.deep_noff = 500

    # ---- fold-time ring-key store ------------------------------------------
    def add_keyframe(self, pts, w, guess):
        grid = grid_from_pts(pts) if len(pts) else None
        nev = self.n_evict
        out = super().add_keyframe(pts, w, guess)
        if self.n_evict != nev:        # HybridSLAM evicted anchor(s): prune ours
            for d in (self.ringkey, self.sc_grid,
                      self._grid_sum, self._grid_cnt):
                for aid in [a for a in d if a not in self.segvec]:
                    del d[aid]
        if grid is not None and grid.sum() > 0:
            # RECALL CAVEAT: the DB entry is a running MEAN grid over the
            # segment's keyframes while the query is a SINGLE keyframe grid.
            # This asymmetry is NOT the single-vs-single config that produced
            # ssp_scancontext's 0.272/0.674 recall table, so the retrieval
            # numbers to trust here are this module's own retrieval_report (mean
            # grids are divided by the fold count so the ring-key magnitude
            # stays on the single-scan scale; ring_key is a per-sector mean, so
            # ring_key(mean grid) ~ mean of per-keyframe ring keys).
            aid = self.aid_of(self.k)
            if aid in self.segvec:                 # fold actually happened
                s = self._grid_sum.get(aid)
                self._grid_sum[aid] = grid if s is None else s + grid
                self._grid_cnt[aid] = self._grid_cnt.get(aid, 0) + 1
                mean = self._grid_sum[aid] / self._grid_cnt[aid]
                self.sc_grid[aid] = mean
                self.ringkey[aid] = SC.ring_key(mean)
        return out

    # ---- ring-key retrieval + coarse-band comparison -----------------------
    def _yaw_seed(self, qgrid, aid):
        _, s = SC.sc_distance(qgrid, self.sc_grid[aid])
        return S.wrap(self.anchors[aid][2] + YAW_SIGN * s * (2 * np.pi / N_SECTOR))

    def _coarse_score_anchors(self, pts, w, me, old):
        """Rank the SAME candidate anchor pool by the shipped coarse-SSP
        whitened correlation evaluated AT each anchor's own live cell (max over
        the full 360 deg heading lattice, whitened by the anchor's write-once
        era shard). This is the coarse-band retrieval signal (_coarse_hyp's
        per-cell score) restricted to the anchor pool, for an apples-to-apples
        top-K comparison against the ring-key ranking. Returns top-K aids."""
        S0 = np.exp(1j * (pts @ WC.T)).T @ w
        ph = np.exp(1j * (WC @ me[:2]))
        m0 = int(round(me[2] * NA / np.pi))
        ms = m0 + np.arange(-NA, NA)
        R0 = np.stack([_sub_rot_permute(S0, m, 2) for m in ms], 1)   # (2NA, 2NA)
        score = np.full(len(old), -np.inf)
        for i, aid in enumerate(old):
            off = self.anchors[aid][:2] - me[:2]
            E = np.exp(1j * (off @ WC.T))                            # (2NA,)
            best = -np.inf
            for sid in self.anchor_shards.get(aid, ()):  # write-once era shards
                vec = self.gshards.get(sid)
                if vec is None:
                    continue
                e = (np.abs(vec.reshape(2, NA)) ** 2).sum(1)
                wht = np.repeat(1.0 / np.maximum(e, 1e-12), NA)
                M = (np.conj(vec) * wht * ph)[:, None] * R0
                best = max(best, float((E @ M).real.max()))
            score[i] = best
        order = np.argsort(score)[::-1]
        return [int(old[j]) for j in order[:TOP_K]]

    # ---- drought: ring-key shortlist -> UNCHANGED verifier + PCM admission ---
    def _try_drought(self, pts, w):
        k = self.k
        self.n_drought_try += 1
        me = self.pose_of(k)
        my_aid = self.aid_of(k)
        dist_since = self.dist_trav - self.dist_at_accept
        log = dict(k=k, phase="cells", z=np.nan, chi=np.nan, ratio=np.nan)
        self.drought_log.append(log)

        # candidate pool: SAME pass-segregation / _drought_cut / gap as shipped
        gap_a = self.gap_kf // ANCHOR
        cut = self._drought_cut()
        old = np.array(sorted(a for a in self.segvec
                              if my_aid - a > gap_a
                              and (cut is None or a <= cut)
                              and a in self.ringkey), dtype=int)
        if old.size == 0:
            return

        # ring-key kNN
        qgrid = grid_from_pts(pts)
        qk = SC.ring_key(qgrid)
        RK = np.stack([self.ringkey[a] for a in old])
        d = np.linalg.norm(RK - qk, axis=1)
        rk_order = old[np.argsort(d, kind="stable")]
        rk_short = [int(a) for a in rk_order[:TOP_K]]

        # coarse-band shortlist over the IDENTICAL pool, for the retrieval A/B
        coarse_short = self._coarse_score_anchors(pts, w, me, old)
        self.retr_log.append(dict(k=k, my_aid=my_aid, pool=int(old.size),
                                  pool_aids=[int(a) for a in old],
                                  rk=rk_short, coarse=coarse_short))

        log["phase"] = "coarse"
        log["nsh"] = 1
        # search breadth = size of the pool the ring-key ranked (deep-search
        # escalation input; keeps the PCM 'evidence scales with breadth' law
        # meaningful for the ring-key source).
        noff = int(old.size)
        log["noff"] = noff
        self.n_drought_hyp += 1

        # verify each ring-key candidate through the UNCHANGED verifier; keep the
        # single best-coherence verified fire (one fire/attempt = shipped PCM
        # semantics). The column-shift yaw is the free cmatcher heading seed.
        best = None
        for aid in rk_short:
            yaw = self._yaw_seed(qgrid, aid)
            hyp = np.array([self.anchors[aid][0], self.anchors[aid][1], yaw])
            res = self._drought_verify(pts, w, hyp, my_aid)
            if res is None:
                continue
            if best is None or res[3] > best[3]:      # res = (pose,c_aid,chain,ratio)
                best = res
        if best is None:
            return
        pose, c_aid, chain, ratio = best
        self.n_drought_verify += 1
        log.update(phase="verified", ratio=ratio, z=ratio, pose=pose.copy(),
                   c_aid=c_aid, hyp=pose.copy())

        # ---- IDENTICAL admission tail (chi-gate + PCM), verbatim from
        #      HybridSLAM._try_drought (ssp_hier.py ~1394-1476) ---------------
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
        entry = dict(k=k, my_aid=my_aid, c_aid=c_aid, Z=Z, sig_t=sig_t,
                     sig_r=sig_r, T=T, b_pos=self.anchors[my_aid][:2].copy(),
                     dr=S.wrap(T[2] - self.anchors[my_aid][2]),
                     noff=noff, lever=float(lever))
        log["phase"] = "pending"
        self._drought_cand = [e for e in self._drought_cand
                              if k - e["k"] <= self.cand_window]
        self._drought_cand.append(entry)
        self._pcm_best = 1
        self._pcm_deep = False
        clique = self._pcm_admit(self._drought_cand)
        log["cand"] = len(self._drought_cand)
        log["clique_max"] = int(self._pcm_best)
        if clique is None:
            return
        if len(clique) >= 2:
            te, re_ = self._pcm_cycle(clique[-2], clique[-1])
            log["pair_dt"], log["pair_drh"] = float(te), float(np.rad2deg(re_))
            log["pair_gap"] = clique[-1]["k"] - clique[-2]["k"]
        log["phase"] = "snap"
        log["clique"] = len(clique)
        log["clique_noff"] = [int(e["noff"]) for e in clique]
        # ---- item-2 validation fields (checked against REF in the harness) ---
        log["my_aid"] = int(my_aid)
        log["pcm_deep"] = bool(self._pcm_deep)
        log["clique_pairs"] = [(int(e["c_aid"]), int(e["my_aid"]))
                               for e in clique]
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
        self.last_accept_k = k
        self.last_true_accept_k = k
        self.dist_at_accept = self.dist_trav
        self._streak = 0
        self._streak_xy = None
        self._drought_cand = []
        self.n_drought_snap += 1


# ---------------------------------------------------------------------------
# CARMEN harness (MIT / Intel) with the range-identity REF ATE convention.
# ---------------------------------------------------------------------------

def run_carmen(name, path, ringkey, drought=500, verbose=True):
    """Run the FULL log through either RingKeySLAM (ringkey=True) or the shipped
    HybridSLAM baseline (ringkey=False), same shipped CARMEN config. Returns a
    dict with ATE (range-identity REF) + drought stats (+ the RingKeySLAM's
    per-attempt retrieval log)."""
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    cls = RingKeySLAM if ringkey else H.HybridSLAM
    slam = cls(beta=3.0, seg_nring=4, attempt_every=4, relax_every=25,
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
        if verbose and k % 2000 == 0:
            print(f"    [{name} {'RK' if ringkey else 'base'}] kf {k}/{n} "
                  f"t={time.time()-t0:.0f}s loops="
                  f"{sum(1 for e in slam.edges if e[5]=='loop')} "
                  f"snaps={slam.n_drought_snap}", flush=True)
    if slam.dirty:
        slam.relax()
    dt = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])

    # range-identity REF ATE (gfs timestamps corrupt on MIT)
    kts2, ref, valid = SC.ref_positions(name, keys)
    al = C.align_se2(fin[valid, :2], ref[valid])
    e = np.linalg.norm(al - ref[valid], axis=1)
    n_loop = sum(1 for ed in slam.edges if ed[5] == "loop")
    return dict(name=name, ringkey=ringkey, n=n, secs=dt,
                ate=float(np.sqrt((e ** 2).mean())), med=float(np.median(e)),
                mx=float(e.max()), loops=n_loop,
                d_try=slam.n_drought_try, d_hyp=slam.n_drought_hyp,
                d_verify=slam.n_drought_verify, d_snap=slam.n_drought_snap,
                snap_k=[g["k"] for g in slam.drought_log if g["phase"] == "snap"],
                snaps=[g for g in slam.drought_log if g["phase"] == "snap"],
                retr=getattr(slam, "retr_log", []),
                kf_ref=list(slam.kf_ref), ref=ref, valid=valid)


def retrieval_report(res):
    """Per-attempt retrieval hit-rate: is a correct old anchor (within TOL_REV m
    in the REF frame of the query kf) present in the top-40 ring-key shortlist,
    vs the coarse-band shortlist, on the IDENTICAL drought attempts + pool."""
    ref, valid, kf_ref = res["ref"], res["valid"], res["kf_ref"]
    # anchor -> its reference keyframe (first kf mapped to that aid)
    aid_kf = {}
    for kf, (aid, _) in enumerate(kf_ref):
        aid_kf.setdefault(aid, kf)

    def ref_of(aid):
        kf = aid_kf.get(aid)
        if kf is None or not valid[kf]:
            return None
        return ref[kf]

    tol = SC.TOL_REV
    n_kf = len(kf_ref)

    def correct(aids, qxy):
        for aid in aids:
            rp = ref_of(aid)
            if rp is not None and np.linalg.norm(rp - qxy) <= tol:
                return True
        return False

    # buckets: all REF-valid attempts, and the subset that are TRUE REVISITS
    # (some pool anchor within tol m in the REF frame — the ssp_scancontext
    # denominator). Stratify true-revisit hit-rate by run-third to expose the
    # deep-corridor drought region.
    def third(kf):
        return ("early" if kf < n_kf / 3 else
                "mid" if kf < 2 * n_kf / 3 else "late/deep-corridor")

    n_att = rk_all = co_all = 0
    strat = {t: dict(rev=0, rk=0, co=0) for t in
             ("early", "mid", "late/deep-corridor")}
    tot = dict(rev=0, rk=0, co=0)
    for a in res["retr"]:
        kf = a["k"]
        if not valid[kf]:
            continue
        qxy = ref[kf]
        rk_c = correct(a["rk"], qxy)
        co_c = correct(a["coarse"], qxy)
        n_att += 1
        rk_all += rk_c
        co_all += co_c
        # true revisit? scan the full candidate pool for any correct anchor
        is_rev = correct(a.get("pool_aids", []), qxy)
        if is_rev:
            b = strat[third(kf)]
            b["rev"] += 1
            b["rk"] += rk_c
            b["co"] += co_c
            tot["rev"] += 1
            tot["rk"] += rk_c
            tot["co"] += co_c
    print(f"  drought attempts (REF-valid): {n_att}")
    if n_att:
        print(f"  hit-rate@{TOP_K} over ALL attempts:      "
              f"ring-key {rk_all/n_att:.3f} ({rk_all}/{n_att})   "
              f"coarse-band {co_all/n_att:.3f} ({co_all}/{n_att})")
    if tot["rev"]:
        print(f"  hit-rate@{TOP_K} over TRUE REVISITS:     "
              f"ring-key {tot['rk']/tot['rev']:.3f} ({tot['rk']}/{tot['rev']})   "
              f"coarse-band {tot['co']/tot['rev']:.3f} ({tot['co']}/{tot['rev']})")
        for t in ("early", "mid", "late/deep-corridor"):
            b = strat[t]
            if b["rev"]:
                print(f"    {t:20} ({b['rev']:3d} revisits): "
                      f"ring-key {b['rk']/b['rev']:.3f}   "
                      f"coarse-band {b['co']/b['rev']:.3f}")
    else:
        print("  (no drought attempt landed at a REF-frame true revisit)")
    return dict(n_att=n_att, tot=tot, strat=strat,
                rk_all=rk_all, co_all=co_all)


def snap_validation(res):
    """Per-snap validation table (item-2 honesty gate). For every ring-key snap,
    print the snap keyframe, admitted clique SIZE, _pcm_deep, and — for each
    (c_aid, my_aid) closure in the clique — whether it is a GENUINE REF-frame
    revisit: both anchors' REF (gfs range-identity) positions within TOL_REV.
    A clique-2, not-deep snap from a large pool that is NOT a true revisit is a
    twin false positive; any ATE 'win' resting on such snaps is not trusted."""
    ref, valid, kf_ref = res["ref"], res["valid"], res["kf_ref"]
    aid_kf = {}
    for kf, (aid, _) in enumerate(kf_ref):
        aid_kf.setdefault(aid, kf)

    def ref_of(aid):
        kf = aid_kf.get(aid)
        return ref[kf] if (kf is not None and valid[kf]) else None

    tol = SC.TOL_REV
    snaps = res.get("snaps", [])
    print(f"  ring-key snaps: {len(snaps)}")
    if not snaps:
        print("  (no snaps — nothing to validate)")
        return dict(n_snap=0, n_genuine=0, n_twin=0)
    print(f"  {'k':>6} {'clique':>6} {'deep':>5}  {'pair(c->b)':>14} "
          f"{'REF dist':>9}  verdict")
    n_gen = n_twin = 0
    for g in snaps:
        pairs = g.get("clique_pairs", [(g.get("c_aid"), g.get("my_aid"))])
        for i, (c_aid, my_aid) in enumerate(pairs):
            rc, rb = ref_of(c_aid), ref_of(my_aid)
            if rc is None or rb is None:
                dist, verdict = float("nan"), "REF-invalid"
            else:
                dist = float(np.linalg.norm(rc - rb))
                genuine = dist <= tol
                verdict = "GENUINE" if genuine else "TWIN/false"
                n_gen += genuine
                n_twin += (not genuine)
            head = (f"  {g['k']:>6} {g.get('clique', 1):>6} "
                    f"{str(g.get('pcm_deep', False)):>5}") if i == 0 \
                else " " * 21
            print(f"{head}  {c_aid:>5}->{my_aid:<7} {dist:>9.2f}  {verdict}")
    print(f"  clique edges: {n_gen} GENUINE revisits, {n_twin} TWIN/false")
    return dict(n_snap=len(snaps), n_genuine=n_gen, n_twin=n_twin)


# ---------------------------------------------------------------------------

def selftest():
    print("selftest: ring-key store + column-shift yaw seed")
    rng = np.random.default_rng(0)
    # (1) grid_from_pts reproduces SC.scan_context on the same returns
    n_beams = 180
    beam = np.deg2rad(np.arange(n_beams) - 90.0)
    r = rng.uniform(1.0, 30.0, n_beams)
    pts, w, _ = S.scan_to_samples(np.where(r < VALID_MAX, r, np.inf), beam)
    g_ref = SC.scan_context(np.linalg.norm(pts, axis=1),
                            np.arctan2(pts[:, 1], pts[:, 0]),
                            VALID_MAX, MAX_R)
    assert np.array_equal(grid_from_pts(pts), g_ref), "grid_from_pts mismatch"
    # (2) column-shift yaw seed: rotate the robot heading by a known dtheta and
    #     confirm the recovered world heading (YAW_SIGN convention) is right.
    slam = RingKeySLAM(seg_nring=4, drought_kf=0)
    theta_c = 0.3
    body_c = pts                                    # candidate body scan
    slam.anchors = [np.array([2.0, -1.0, theta_c])]
    slam.sc_grid[0] = grid_from_pts(body_c)
    slam.ringkey[0] = SC.ring_key(slam.sc_grid[0])
    max_err = 0.0
    for dtheta in (np.deg2rad(12.0), np.deg2rad(-30.0), np.deg2rad(48.0)):
        body_q = body_c @ S._rot(-dtheta).T          # robot rotated +dtheta
        theta_q = S.wrap(theta_c + dtheta)
        seed = slam._yaw_seed(grid_from_pts(body_q), 0)
        max_err = max(max_err, abs(S.wrap(seed - theta_q)))
    assert max_err <= np.deg2rad(N_SECTOR and 360 / N_SECTOR), \
        f"yaw seed off by {np.rad2deg(max_err):.1f} deg (> one 6 deg bin)"
    print(f"  grid_from_pts == SC.scan_context OK; yaw-seed max err "
          f"{np.rad2deg(max_err):.2f} deg (<= 6 deg bin) OK")
    # (3) fold-time store: a short synthetic run populates ring keys per segment
    slam2 = RingKeySLAM(seg_nring=4, drought_kf=0)
    for k in range(12):
        rr = rng.uniform(1.0, 30.0, n_beams)
        p, ww, _ = S.scan_to_samples(np.where(rr < VALID_MAX, rr, np.inf), beam)
        slam2.add_keyframe(p, ww, np.array([0.1 * k, 0.0, 0.0]))
    assert slam2.ringkey, "no ring keys stored"
    for aid, rk in slam2.ringkey.items():
        assert rk.shape == (N_RING,) and aid in slam2.segvec
    print(f"  fold-time store OK: {len(slam2.ringkey)} segment ring keys "
          f"({N_RING} floats each), all backed by a live segment")
    print("selftest PASSED")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        selftest()
    elif cmd in ("mit", "intel"):
        path = sys.argv[2] if len(sys.argv) > 2 else f"data/{cmd}.log"
        print(f"=== {cmd.upper()}: ring-key vs baseline HybridSLAM ===")
        base = run_carmen(cmd, path, ringkey=False)
        print(f"[baseline] ATE rmse {base['ate']:.3f} m  med {base['med']:.3f}  "
              f"max {base['mx']:.2f}  loops {base['loops']}  "
              f"drought try/hyp/ver/snap "
              f"{base['d_try']}/{base['d_hyp']}/{base['d_verify']}/{base['d_snap']}"
              f"  snaps@{base['snap_k']}  {base['secs']:.0f}s", flush=True)
        rk = run_carmen(cmd, path, ringkey=True)
        print(f"[ring-key] ATE rmse {rk['ate']:.3f} m  med {rk['med']:.3f}  "
              f"max {rk['mx']:.2f}  loops {rk['loops']}  "
              f"drought try/hyp/ver/snap "
              f"{rk['d_try']}/{rk['d_hyp']}/{rk['d_verify']}/{rk['d_snap']}"
              f"  snaps@{rk['snap_k']}  {rk['secs']:.0f}s", flush=True)
        print("--- RETRIEVAL (ring-key run, identical attempts + pool) ---")
        retrieval_report(rk)
        print("--- SNAP VALIDATION (ring-key run; every snap vs REF) ---")
        snap_validation(rk)
    else:
        raise SystemExit(f"unknown command {cmd!r} (selftest | mit | intel)")


if __name__ == "__main__":
    main()
