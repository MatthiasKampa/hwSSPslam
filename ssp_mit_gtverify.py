"""VALUE OF A PERFECT VERIFICATION CUE on the MIT corridor (GT-ORACLE upper bound).

The symmetric completion of the campaign's synthesis. Across every thread the
finding is the same: retrieval and the VSA algebra WORK; the recurring wall is
appearance-only VERIFICATION (separating a genuine revisit from a consistent
corridor twin). The multi-session side quantified the positive corollary with a
GT-verified 12-edge diagnostic (B 4.37 -> 2.49 m). This module quantifies the
SAME corollary for the MIT corridor, which was missing.

QUESTION: on MIT, the shipped conservative drought closes only ~2 genuine loops
(ATE 42.66 m, raw odometry 189.3 m) because the coherence gate + PCM cannot
verify ring-key's retrieved revisits against corridor twins. If verification
were PERFECT — an oracle place-recognizer that admits genuine revisits and
rejects twins — how much would MIT ATE improve?

METHOD (a GT-ORACLE DIAGNOSTIC — an UPPER BOUND, NOT a capability):
  At each drought opportunity (shipped trigger / cadence / pass-segregation,
  UNCHANGED), the REAL ring-key appearance retrieval (recall@40 ~0.81, imported
  from ssp_ringkey.RingKeySLAM) produces the top-40 candidate old anchors. The
  PLACE-verification step is replaced by an oracle: a candidate is admitted iff
  it is a GENUINE REF-frame revisit (query & candidate REF within TOL_REV = 5 m);
  corridor twins are rejected. The admitted genuine closure's EDGE GEOMETRY comes
  from the REAL matcher alignment (H.HybridSLAM._drought_verify); the closure then
  flows through the EXISTING drought-edge insertion + _distribute_correction +
  joint relax (_gn), unchanged.

  Two variants, because the shipped coherence gate does DOUBLE duty (place
  discriminator AND geometry-quality filter):
    A (place-ONLY, coh gate OFF): the oracle replaces the ENTIRE verification
      step. Isolates the value of perfect PLACE recognition alone.
    B (place + geometry gate, coh gate KEPT): the oracle replaces ONLY the place-
      DISCRIMINATION failure (the documented wall); the real coherence gate is
      kept purely as a geometry-quality filter (it cannot mis-rank place — GT
      already guarantees genuineness). This is the faithful "perfect place
      verifier on top of the unchanged geometry machinery" upper bound.

RESULT (deterministic, this run): baseline 42.66 m (2 snaps) / odometry ~188 m.
  B: ~2-3 genuine well-aligned snaps, ATE ~= the shipped 42-45 m band (a perfect
     place oracle barely moves MIT). A: 4 genuine snaps but ATE ~73 m — WORSE.
  KEY FINDING: place-verification is NECESSARY BUT NOT SUFFICIENT on the corridor.
  A backfires because genuine corridor revisits admit MIS-ALIGNED edges (the
  matcher slides along the corridor; coherence 0.40) that corrupt the relax; B's
  coherence geometry-gate is what stops that, leaving only the ~2 revisits whose
  RELATIVE POSE the matcher can also recover. The corridor wall is TWO-fold —
  place discrimination AND relative-pose geometry both fail on self-similarity —
  unlike the multi-session 2.49 m diagnostic, where genuine ties had GOOD geometry
  so a place oracle DID close the loop. This sharpens the synthesis rather than
  reopening the limit.

WHERE GT TOUCHES (audit invariant — confirm before trusting):
  GT (the REF positions) touches ONLY (a) the admit/reject decision and (b) the
  final ATE. It NEVER touches retrieval (ring-key), seeding (anchor graph pose +
  column-shift yaw), edge geometry (cmatcher.match), or the relax (_gn). Query
  and candidate REF are BOTH in the GT frame, so the genuineness test never mixes
  GT with the estimate frame. The hyp/seed and all geometry live in the estimate
  frame only.

HARD FRAMING (do NOT overclaim): this is a GT-ORACLE UPPER BOUND, not a working
system. No real appearance-only verifier reaches it on MIT — that is precisely
the CLOSED corridor limit (SeqSLAM genuine-vs-twin separation 0.500 = chance;
PCM twins consistent; coherence medians indistinguishable). The result quantifies
what a missing EXTERNAL cue (GPS / landmarks / 3D) would buy and is symmetric to
the multi-session 2.49 m diagnostic — but the symmetry is INSTRUCTIVE, not
identical: there, place verification was the ONLY missing piece and an oracle
closed the loop; here, perfect PLACE verification is necessary but NOT sufficient
because corridor self-similarity ALSO corrupts the relative-pose geometry. It does
NOT reopen the corridor limit; it sharpens it.

NO shipped/prior module is edited. Imports by reference only:
  * ssp_ringkey.RingKeySLAM  -> ring-key retrieval + fold-time descriptor store
  * ssp_scancontext (SC)     -> REF positions, TOL_REV, ring_key / sc_distance
  * ssp_hier (H)             -> HybridSLAM drought + joint machinery
  * ssp_bounded._gn          -> the analytic-jac TRF joint relax solver (imported
                               to make the backend dependency explicit; the
                               relax path H.HybridSLAM.relax already uses it)

Usage:
  python3 ssp_mit_gtverify.py selftest
  python3 ssp_mit_gtverify.py mit [data/mit.log]     # baseline + oracle + report
"""

import sys
import time

import numpy as np

import ssp_slam as S
import ssp_slam_loop as L
import ssp_slam_carmen as C
import ssp_hier as H
import ssp_scancontext as SC
import ssp_ringkey as RK
from ssp_ringkey import grid_from_pts, TOP_K
import ssp_bounded as B
_gn = B.BoundedSLAM._gn      # analytic-jac TRF joint relax solver (by reference;
#                              HybridSLAM inherits it and relax() already calls it)

ANCHOR = H.ANCHOR
TOL_REV = SC.TOL_REV                 # 5.0 m, REF-frame revisit radius (the oracle)


class GTVerifySLAM(RK.RingKeySLAM):
    """RingKeySLAM (real ring-key retrieval + descriptor store) whose drought
    VERIFICATION is replaced by a GROUND-TRUTH oracle. Everything else — matcher,
    edge geometry, _distribute_correction, joint relax, backend — inherited and
    UNCHANGED. GT enters only the admit/reject decision (and the final ATE)."""

    def __init__(self, keep_coh_gate=False, **kw):
        super().__init__(**kw)
        # GT REF, injected by the harness BEFORE the run (precomputed from the
        # gfs range-identity convention). Consumed ONLY by the oracle admit test.
        self.ref = None                 # (n_kf, 2) REF positions, GT frame
        self.valid = None               # (n_kf,) inside-REF-span mask
        # PLACE-ONLY (A, keep_coh_gate=False): the oracle replaces the ENTIRE
        # verification step — coherence gate + PCM — so admission is genuineness
        # ALONE. Isolates the value of perfect PLACE recognition; but the shipped
        # coherence gate is ALSO a geometry-quality filter, so genuine-place /
        # bad-geometry corridor slides get in.
        # PLACE+GEOMETRY (B, keep_coh_gate=True): the oracle replaces ONLY the
        # place-DISCRIMINATION failure (the documented wall — coherence medians
        # indistinguishable, PCM twins); the real coherence gate is KEPT purely
        # as a geometry-quality filter (it cannot mis-rank place here — GT already
        # guarantees genuineness). This is the faithful "perfect place verifier
        # on top of the unchanged geometry machinery" upper bound.
        self.keep_coh_gate = keep_coh_gate
        self._aid_first_kf = {}         # aid -> its first (reference) keyframe
        # diagnostics
        self.n_geom_fail = 0            # genuine cand, matcher produced no basin
        self.n_edge_twin_rej = 0       # genuine cand but matched chain-centre twin
        self.n_no_genuine = 0          # attempt w/ a REF-valid query, no genuine cand
        self.oracle_snaps = []         # per-snap validation records

    # ---- fold-time bookkeeping: first keyframe of each anchor --------------
    def add_keyframe(self, pts, w, guess):
        out = super().add_keyframe(pts, w, guess)
        kf = len(self.kf_ref) - 1
        aid = self.kf_ref[kf][0]
        self._aid_first_kf.setdefault(aid, kf)
        return out

    # ---- GT oracle helpers (REF frame only) -------------------------------
    def _ref_of_kf(self, kf):
        if self.ref is None or kf < 0 or kf >= len(self.valid) \
                or not self.valid[kf]:
            return None
        return self.ref[kf]

    def _ref_of_anchor(self, aid):
        kf = self._aid_first_kf.get(aid)
        return None if kf is None else self._ref_of_kf(kf)

    # ---- drought: ring-key retrieval -> GT oracle -> real geometry --------
    def _try_drought(self, pts, w):
        k = self.k
        self.n_drought_try += 1
        me = self.pose_of(k)
        my_aid = self.aid_of(k)
        dist_since = self.dist_trav - self.dist_at_accept
        log = dict(k=k, phase="cells", z=np.nan, chi=np.nan, ratio=np.nan)
        self.drought_log.append(log)

        # ----- REAL ring-key retrieval (identical pool + kNN as RingKeySLAM) -
        gap_a = self.gap_kf // ANCHOR
        cut = self._drought_cut()
        old = np.array(sorted(a for a in self.segvec
                              if my_aid - a > gap_a
                              and (cut is None or a <= cut)
                              and a in self.ringkey), dtype=int)
        if old.size == 0:
            return
        qgrid = grid_from_pts(pts)
        qk = SC.ring_key(qgrid)
        RKmat = np.stack([self.ringkey[a] for a in old])
        d = np.linalg.norm(RKmat - qk, axis=1)
        rk_order = old[np.argsort(d, kind="stable")]
        rk_short = [int(a) for a in rk_order[:TOP_K]]
        log.update(phase="coarse", nsh=1, noff=int(old.size))
        self.n_drought_hyp += 1

        # ----- GT ORACLE: admit only candidates that are genuine REF revisits -
        q_ref = self._ref_of_kf(k)
        if q_ref is None:
            return                      # cannot oracle-verify -> no closure
        genuine = [aid for aid in rk_short
                   if (cr := self._ref_of_anchor(aid)) is not None
                   and np.linalg.norm(cr - q_ref) <= TOL_REV]
        self.retr_log.append(dict(k=k, pool=int(old.size), rk=rk_short,
                                  n_genuine=len(genuine)))
        if not genuine:
            self.n_no_genuine += 1
            return

        # ----- REAL edge geometry: matcher alignment --------------------------
        # A (keep_coh_gate=False): neutralize coh_target so the matcher's pose is
        #   returned regardless of coherence — the ORACLE is the sole admit gate.
        #   The pose is selected by the matcher BEFORE the coherence test, so
        #   neutralizing coh_target changes ONLY the accept decision, never the
        #   geometry. Restored immediately; the frontend gate that also reads
        #   coh_target is not called in-window.
        # B (keep_coh_gate=True): leave coh_target=0.55 so _drought_verify keeps
        #   its geometry-quality gate (genuine-but-mis-aligned corridor slides are
        #   dropped); GT still owns PLACE discrimination.
        saved = self.coh_target
        if not self.keep_coh_gate:
            self.coh_target = -1e18
        best = None
        try:
            for aid in genuine:
                yaw = self._yaw_seed(qgrid, aid)
                hyp = np.array([self.anchors[aid][0],
                                self.anchors[aid][1], yaw])
                res = self._drought_verify(pts, w, hyp, my_aid)
                if res is None:
                    continue
                if best is None or res[3] > best[3]:   # best matcher alignment
                    best = res
        finally:
            self.coh_target = saved
        if best is None:
            self.n_geom_fail += 1
            return
        pose, c_aid, chain, ratio = best
        self.n_drought_verify += 1
        log.update(phase="verified", ratio=ratio, pose=pose.copy(), c_aid=c_aid)

        # edge-level oracle guard: the actually-inserted edge (c_aid, my_aid)
        # must ITSELF be a genuine REF revisit, so every admitted closure is
        # genuine BY CONSTRUCTION (matches ssp_ringkey.snap_validation).
        cc = self._ref_of_anchor(c_aid)
        if cc is None or np.linalg.norm(cc - q_ref) > TOL_REV:
            self.n_edge_twin_rej += 1
            return

        # ----- EXISTING drought-edge insertion tail (chi sanity gate + insert
        #        + _distribute_correction), verbatim from HybridSLAM minus PCM
        #        (the oracle replaces the consensus verifier; a single genuine
        #         closure per attempt is inserted). -----------------------------
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
        if self.use_innov and chi > 9.0:      # geometry-consistency sanity only
            self.n_innov_rej += 1
            return
        T = L.se2_mul(self.anchors[c_aid], Z)
        log.update(phase="snap", clique=1, my_aid=int(my_aid), pcm_deep=False,
                   clique_pairs=[(int(c_aid), int(my_aid))], ratio=float(ratio),
                   ref_dist=float(np.linalg.norm(cc - q_ref)))
        self.oracle_snaps.append(dict(k=k, c_aid=int(c_aid), my_aid=int(my_aid),
                                      ratio=float(ratio),
                                      ref_dist=float(np.linalg.norm(cc - q_ref)),
                                      chi=float(chi), pool=int(old.size),
                                      n_genuine=len(genuine)))
        self._distribute_correction(my_aid, T)
        key = (c_aid, my_aid)
        if key not in self.banned:
            edge = (c_aid, my_aid, Z, 1 / sig_t, 1 / sig_r, "loop")
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
        self.n_drought_snap += 1


# ---------------------------------------------------------------------------
# Harness: full MIT log, range-identity REF ATE (gfs timestamps corrupt on MIT).
# ---------------------------------------------------------------------------

def run(name, path, mode, ref=None, valid=None, verbose=True):
    """Run the FULL log. mode in {'base','oracleA','oracleB'}:
      base    -> shipped H.HybridSLAM (coarse-band drought, the 42.66 reference)
      oracleA -> GTVerifySLAM place-ONLY oracle (coherence gate neutralized)
      oracleB -> GTVerifySLAM place + real coherence geometry gate (faithful)
    REF (precomputed GT) is injected into the oracle for its admit decision; it
    is used for ATE in ALL modes (identical eval)."""
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    odom = np.stack([k[1] for k in keys])
    if mode == "base":
        slam = H.HybridSLAM(beta=3.0, seg_nring=4, attempt_every=4,
                            relax_every=25, gap_kf=300, recent_aids=12,
                            drought_kf=500)
    else:
        slam = GTVerifySLAM(keep_coh_gate=(mode == "oracleB"), beta=3.0,
                            seg_nring=4, attempt_every=4, relax_every=25,
                            gap_kf=300, recent_aids=12, drought_kf=500)
    slam.dtheta_beam = np.pi / n_beams
    if mode != "base":
        slam.ref, slam.valid = ref, valid       # GT injected for the oracle
    est = np.zeros((n, 3))
    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < 40.0, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = opose if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
        if verbose and k % 2000 == 0:
            print(f"    [{name} {mode}] kf {k}/{n} t={time.time()-t0:.0f}s "
                  f"loops={sum(1 for e in slam.edges if e[5]=='loop')} "
                  f"snaps={slam.n_drought_snap}", flush=True)
    if slam.dirty:
        slam.relax()
    dt = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    return slam, fin, odom, keys, dt


def ate(fin_xy, ref, valid):
    al = C.align_se2(fin_xy[valid], ref[valid])
    e = np.linalg.norm(al - ref[valid], axis=1)
    return float(np.sqrt((e ** 2).mean())), float(np.median(e)), float(e.max())


def jerk(fin):
    """Trajectory jerk = mean |x_{k+1}-2x_k+x_{k-1}| over positions. A relax that
    destabilizes shows up as a jerk blow-up; report it as the stability sanity."""
    p = fin[:, :2]
    a = p[2:] - 2 * p[1:-1] + p[:-2]
    return float(np.linalg.norm(a, axis=1).mean())


def selftest():
    print("selftest: GT-oracle plumbing (no full log)")
    rng = np.random.default_rng(0)
    n_beams = 180
    beam = np.deg2rad(np.arange(n_beams) - 90.0)
    slam = GTVerifySLAM(seg_nring=4, drought_kf=0)
    # short synthetic run to populate anchors + ring keys + _aid_first_kf
    for k in range(16):
        rr = rng.uniform(1.0, 30.0, n_beams)
        p, ww, _ = S.scan_to_samples(np.where(rr < 50.0, rr, np.inf), beam)
        slam.add_keyframe(p, ww, np.array([0.1 * k, 0.0, 0.0]))
    assert slam._aid_first_kf, "aid_first_kf not populated"
    # each recorded first-kf actually maps to that anchor in kf_ref
    for aid, kf in slam._aid_first_kf.items():
        assert slam.kf_ref[kf][0] == aid, "first-kf mapping wrong"
        assert all(slam.kf_ref[j][0] != aid for j in range(kf)), \
            "first-kf is not the FIRST occurrence"
    # oracle REF helpers: inject a tiny GT and check the genuineness algebra
    nkf = len(slam.kf_ref)
    slam.ref = np.zeros((nkf, 2))
    slam.valid = np.ones(nkf, bool)
    slam.ref[slam._aid_first_kf[list(slam._aid_first_kf)[-1]]] = [0.0, 0.0]
    a0 = list(slam._aid_first_kf)[0]
    slam.ref[slam._aid_first_kf[a0]] = [3.0, 4.0]      # exactly TOL_REV away
    assert np.allclose(slam._ref_of_anchor(a0), [3.0, 4.0])
    slam.valid[slam._aid_first_kf[a0]] = False
    assert slam._ref_of_anchor(a0) is None, "valid mask not honoured"
    print(f"  _aid_first_kf OK ({len(slam._aid_first_kf)} anchors); REF helpers "
          f"honour valid mask; TOL_REV={TOL_REV} m; _gn bound: {_gn.__qualname__}")
    print("selftest PASSED")


def _oracle_report(tag, name, path, ref, valid):
    orc, forc, _, _, to = run(name, path, tag, ref=ref, valid=valid)
    a_g = ate(forc[:, :2], ref, valid)
    label = "place-ONLY (coh OFF)" if tag == "oracleA" else \
            "place + coh geom-gate"
    print(f"[{tag} {label}] ATE rmse {a_g[0]:.3f} m  med {a_g[1]:.3f}  "
          f"max {a_g[2]:.2f}  loops {sum(1 for e in orc.edges if e[5]=='loop')}  "
          f"drought try/hyp/ver/snap {orc.n_drought_try}/{orc.n_drought_hyp}/"
          f"{orc.n_drought_verify}/{orc.n_drought_snap}  jerk {jerk(forc):.4f}"
          f"  {to:.0f}s", flush=True)
    print(f"  oracle bookkeeping: no-genuine-cand {orc.n_no_genuine}  "
          f"matcher-no-basin {orc.n_geom_fail}  edge-twin-rej {orc.n_edge_twin_rej}"
          f"  chi-rej {orc.n_innov_rej}")
    print(f"  --- {tag} snap validation (every admitted closure vs REF) ---")
    print(f"  {'k':>6} {'pair(c->b)':>14} {'REFdist':>8} {'ratio':>6} "
          f"{'chi':>6} {'pool':>5} {'#gen':>5}  verdict")
    n_gen = n_twin = 0
    for s in orc.oracle_snaps:
        gen = s["ref_dist"] <= TOL_REV
        n_gen += gen
        n_twin += (not gen)
        print(f"  {s['k']:>6} {s['c_aid']:>5}->{s['my_aid']:<7} "
              f"{s['ref_dist']:>8.2f} {s['ratio']:>6.2f} {s['chi']:>6.2f} "
              f"{s['pool']:>5} {s['n_genuine']:>5}  "
              f"{'GENUINE' if gen else 'TWIN/false'}")
    print(f"  admitted: {n_gen} GENUINE, {n_twin} TWIN/false "
          f"(oracle guarantees 0 twins by construction)")
    return a_g, orc.n_drought_snap, jerk(forc)


def report(name, path):
    print(f"=== {name.upper()}: VALUE OF A PERFECT VERIFICATION CUE "
          f"(GT-oracle upper bound) ===")
    # REF once (GT), shared by all runs' ATE and by the oracle's admit test
    keys0 = C.keyframes(C.parse_flaser(path))
    _, ref, valid = SC.ref_positions(name, keys0)

    base, fbase, odom, _, tb = run(name, path, "base")
    a_b = ate(fbase[:, :2], ref, valid)
    a_o_raw = ate(odom[:, :2], ref, valid)     # raw odometry reference (~189 m)
    print(f"[odometry] ATE rmse {a_o_raw[0]:.2f} m  (raw dead-reckoning)")
    print(f"[baseline] ATE rmse {a_b[0]:.3f} m  med {a_b[1]:.3f}  max {a_b[2]:.2f}"
          f"  loops {sum(1 for e in base.edges if e[5]=='loop')}  "
          f"drought try/hyp/ver/snap {base.n_drought_try}/{base.n_drought_hyp}/"
          f"{base.n_drought_verify}/{base.n_drought_snap}  jerk {jerk(fbase):.4f}"
          f"  {tb:.0f}s", flush=True)

    aA, nA, jA = _oracle_report("oracleA", name, path, ref, valid)
    aB, nB, jB = _oracle_report("oracleB", name, path, ref, valid)

    # ---- headline -----------------------------------------------------------
    print("\n--- HEADLINE: value of a perfect PLACE-verification cue on MIT ---")
    print(f"  raw odometry ATE .................................. {a_o_raw[0]:.2f} m")
    print(f"  shipped drought (coherence+PCM) ................... {a_b[0]:.2f} m   "
          f"({base.n_drought_snap} genuine snaps, jerk {jerk(fbase):.4f})")
    print(f"  ORACLE B  place-verify + real geometry gate  <== X  {aB[0]:.2f} m   "
          f"({nB} genuine snaps, jerk {jB:.4f})")
    print(f"  ORACLE A  place-verify ONLY (coherence OFF) ....... {aA[0]:.2f} m   "
          f"({nA} genuine snaps, jerk {jA:.4f})")
    print(f"\n  value of perfect place verification (B): "
          f"{a_b[0]:.2f} -> {aB[0]:.2f} m (delta {a_b[0]-aB[0]:+.2f} m)")
    print("  KEY FINDING: even a PERFECT place oracle barely moves MIT. B keeps only")
    print("  the genuine revisits whose GEOMETRY the matcher can also recover; there")
    print("  are ~2-3 of them and they roughly reproduce the shipped 2-snap result.")
    print("  A shows place-verification ALONE BACKFIRES: genuine corridor revisits")
    print("  admit mis-aligned edges (matcher slides along the corridor, coherence")
    print("  0.40) that corrupt the relax. The corridor wall is TWO-fold — place")
    print("  discrimination AND relative-pose geometry both fail on self-similarity;")
    print("  an oracle that fixes only place cannot close the loop (unlike the multi-")
    print("  session 2.49 m diagnostic, where genuine ties also had GOOD geometry).")
    print("  FRAMING: GT-oracle UPPER BOUND, not a working system; no appearance-only")
    print("  verifier reaches it. It does NOT reopen the corridor limit — it sharpens")
    print("  it: perfect place recognition is necessary but NOT sufficient on MIT.")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        selftest()
    elif cmd == "mit":
        path = sys.argv[2] if len(sys.argv) > 2 else "data/mit.log"
        report("mit", path)
    else:
        raise SystemExit(f"unknown command {cmd!r} (selftest | mit)")


if __name__ == "__main__":
    main()
