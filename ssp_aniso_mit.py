"""GT-oracle MIT corridor with ANISOTROPIC drought closures (subclass; no edit).

Direct test of the geometry half of the two-fold corridor wall (RESULTS.md
"the wall is TWO-FOLD"). The isotropic GT-oracle (ssp_mit_gtverify.py) found:
  ORACLE B (place-verify + coherence geometry gate) = 41.04 m (3 snaps)
  ORACLE A (place-verify ONLY, coherence off)       = 73.47 m (4 snaps, WORSE)
A backfires because 2 of its 4 genuine revisits are along-corridor SLIDES
(coherence ~0.40): geometrically-partial closures whose along-corridor component
is wrong. Inserted ISOTROPICALLY (and rigidly pre-aligned), the slide drags the
graph along the corridor -> 73 m.

This harness inserts the SAME oracle-admitted genuine revisits ANISOTROPICALLY:
  * the drought edge carries the observability-derived 2x2 L (AnisoMixin._gn),
    so the along-corridor (weak) direction contributes ~cond_floor of the weight;
  * the rigid pre-alignment (_distribute_correction) is PROJECTED onto the
    observable (strong) direction, so the slide is not baked in before relax.
QUESTION: does the cross-corridor component of the geometrically-partial genuine
revisits now CONTRIBUTE (beating oracle-B 41.04), or does the slide contaminate
the observable direction too / the weak eigenvalue is too near-zero to help?

GT-leak boundary is inherited UNCHANGED from ssp_mit_gtverify: REF touches only
the admit/reject decision and the final ATE. The anisotropy is derived purely
from the estimate-frame observability Hessian (no GT).

Usage: python3 ssp_aniso_mit.py [data/mit.log] [oracleA|oracleB|both]
"""

import sys
import time

import numpy as np

import ssp_slam as S
import ssp_slam_loop as L
import ssp_slam_carmen as C
import ssp_scancontext as SC
import ssp_mit_gtverify as GTV
import ssp_aniso as A

ANCHOR = GTV.ANCHOR
TOL_REV = GTV.TOL_REV


class AnisoGTVerifySLAM(A.AnisoMixin, GTV.GTVerifySLAM):
    """GTVerifySLAM (real ring-key retrieval + GT place oracle) whose admitted
    drought closures are inserted with observability-weighted ANISOTROPIC
    information and observable-only rigid pre-alignment."""

    def __init__(self, use_aniso=True, cond_floor=0.1, aniso_prealign=True,
                 keep_coh_gate=False, **kw):
        super().__init__(keep_coh_gate=keep_coh_gate, **kw)
        self._aniso_init(use_aniso=use_aniso, cond_floor=cond_floor)
        self.aniso_prealign = aniso_prealign
        self.aniso_strong = {}       # (a,b) -> world-frame strong (observable) dir
        self._cur_strong = None      # observable dir for the in-flight pre-align

    # ---- pre-align only the OBSERVABLE translation component -----------------
    def _distribute_correction(self, b2, T2):
        us = self._cur_strong
        if us is None or not self.aniso_prealign:
            return super()._distribute_correction(b2, T2)
        # verbatim structure of HybridSLAM._distribute_correction, but the
        # residual translation is projected onto the observable (strong) dir:
        # the along-corridor slide is NOT rigidly applied.
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
        dt = float(dt @ us) * us            # observable component only
        for i in range(a0 + 1, b2 + 1):
            self.anchors[i][:2] += ((i - a0) / m) * dt

    # ---- drought: same oracle admission, anisotropic insertion --------------
    def _try_drought(self, pts, w):
        k = self.k
        self.n_drought_try += 1
        me = self.pose_of(k)
        my_aid = self.aid_of(k)
        dist_since = self.dist_trav - self.dist_at_accept
        log = dict(k=k, phase="cells", z=np.nan, chi=np.nan, ratio=np.nan)
        self.drought_log.append(log)

        gap_a = self.gap_kf // ANCHOR
        cut = self._drought_cut()
        old = np.array(sorted(a for a in self.segvec
                              if my_aid - a > gap_a
                              and (cut is None or a <= cut)
                              and a in self.ringkey), dtype=int)
        if old.size == 0:
            return
        from ssp_ringkey import grid_from_pts, TOP_K
        qgrid = grid_from_pts(pts)
        qk = SC.ring_key(qgrid)
        RKmat = np.stack([self.ringkey[a] for a in old])
        d = np.linalg.norm(RKmat - qk, axis=1)
        rk_order = old[np.argsort(d, kind="stable")]
        rk_short = [int(a) for a in rk_order[:TOP_K]]
        log.update(phase="coarse", nsh=1, noff=int(old.size))
        self.n_drought_hyp += 1

        q_ref = self._ref_of_kf(k)
        if q_ref is None:
            return
        genuine = [aid for aid in rk_short
                   if (cr := self._ref_of_anchor(aid)) is not None
                   and np.linalg.norm(cr - q_ref) <= TOL_REV]
        self.retr_log.append(dict(k=k, pool=int(old.size), rk=rk_short,
                                  n_genuine=len(genuine)))
        if not genuine:
            self.n_no_genuine += 1
            return

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
                if best is None or res[3] > best[3]:
                    best = res
        finally:
            self.coh_target = saved
        if best is None:
            self.n_geom_fail += 1
            return
        pose, c_aid, chain, ratio = best
        self.n_drought_verify += 1
        log.update(phase="verified", ratio=ratio, pose=pose.copy(), c_aid=c_aid)

        cc = self._ref_of_anchor(c_aid)
        if cc is None or np.linalg.norm(cc - q_ref) > TOL_REV:
            self.n_edge_twin_rej += 1
            return

        # ---- OBSERVABILITY HESSIAN at the matched pose (estimate frame only) --
        Bb = np.zeros(L.W.shape[0], complex)
        for a in chain:
            Bb[self.SEG] += self.world_vec_fine(a)
        evals, evecs = self._obs_eigen(Bb, pts, w, pose)
        l_weak, l_strong = float(evals[0]), float(evals[1])

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

        wt = 1.0 / sig_t
        th_a = self.anchors[c_aid][2]
        Laniso = self._aniso_L(evals, evecs, wt, th_a)
        # observable (strong) direction in the WORLD frame for pre-align proj
        self._cur_strong = evecs[:, 1] if (self.use_aniso and Laniso is not None) \
            else None
        r_ratio = l_weak / max(l_strong, 1e-30)

        T = L.se2_mul(self.anchors[c_aid], Z)
        log.update(phase="snap", clique=1, my_aid=int(my_aid), pcm_deep=False,
                   clique_pairs=[(int(c_aid), int(my_aid))], ratio=float(ratio),
                   ref_dist=float(np.linalg.norm(cc - q_ref)))
        self.oracle_snaps.append(dict(k=k, c_aid=int(c_aid), my_aid=int(my_aid),
                                      ratio=float(ratio),
                                      ref_dist=float(np.linalg.norm(cc - q_ref)),
                                      chi=float(chi), pool=int(old.size),
                                      n_genuine=len(genuine),
                                      cond=float(r_ratio),
                                      aniso=bool(Laniso is not None)))
        self._distribute_correction(my_aid, T)
        self._cur_strong = None
        key = (c_aid, my_aid)
        if key not in self.banned:
            edge = (c_aid, my_aid, Z, 1 / sig_t, 1 / sig_r, "loop")
            if key in self.edge_seen:
                self.edges[self.edge_seen[key]] = edge
            else:
                self.edge_seen[key] = len(self.edges)
                self.edges.append(edge)
            self._set_aniso(key, Laniso)
            if Laniso is not None:
                self.aniso_strong[key] = evecs[:, 1]
            self.pending_new.append(key)
        self.dirty = True
        self._force_relax = True
        self.last_accept_k = k
        self.last_true_accept_k = k
        self.dist_at_accept = self.dist_trav
        self._streak = 0
        self._streak_xy = None
        self.n_drought_snap += 1


def run_aniso(path, mode, ref, valid, cond_floor=0.1, aniso_prealign=True,
              use_aniso=True, verbose=True):
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    odom = np.stack([k[1] for k in keys])
    slam = AnisoGTVerifySLAM(use_aniso=use_aniso, cond_floor=cond_floor,
                             aniso_prealign=aniso_prealign,
                             keep_coh_gate=(mode == "oracleB"), beta=3.0,
                             seg_nring=4, attempt_every=4, relax_every=25,
                             gap_kf=300, recent_aids=12, drought_kf=500)
    slam.dtheta_beam = np.pi / n_beams
    slam.ref, slam.valid = ref, valid
    est = np.zeros((n, 3))
    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < 40.0, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = opose if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
        if verbose and k % 3000 == 0:
            print(f"    [{mode} f{cond_floor} pa{int(aniso_prealign)}] kf {k}/{n} "
                  f"t={time.time()-t0:.0f}s snaps={slam.n_drought_snap}", flush=True)
    if slam.dirty:
        slam.relax()
    dt = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    return slam, fin, dt


def report(path="data/mit.log", modes=("oracleA", "oracleB"),
           floors=(0.05, 0.1, 0.2)):
    print("=== MIT GT-ORACLE: ANISOTROPIC drought closures ===")
    print("known isotropic bounds (RESULTS.md): baseline 42.66 | oracleB 41.04 | "
          "oracleA 73.47 (WORSE)")
    keys0 = C.keyframes(C.parse_flaser(path))
    _, ref, valid = SC.ref_positions("mit", keys0)
    for mode in modes:
        print(f"\n--- {mode} ---")
        for f in floors:
            slam, fin, dt = run_aniso(path, mode, ref, valid, cond_floor=f,
                                      aniso_prealign=True, verbose=True)
            a = GTV.ate(fin[:, :2], ref, valid)
            n_gen = sum(1 for s in slam.oracle_snaps if s["ref_dist"] <= TOL_REV)
            n_an = sum(1 for s in slam.oracle_snaps if s.get("aniso"))
            conds = [round(s["cond"], 4) for s in slam.oracle_snaps]
            print(f"  [aniso {mode} floor={f}] ATE {a[0]:.2f} m  med {a[1]:.2f}  "
                  f"max {a[2]:.1f}  snaps {slam.n_drought_snap} ({n_gen} genuine, "
                  f"{n_an} aniso)  jerk {GTV.jerk(fin):.4f}  {dt:.0f}s", flush=True)
            print(f"     snap cond-ratios (l_weak/l_strong): {conds}")
    print("\nVerdict: compare aniso oracleA/B ATE vs isotropic oracleB 41.04 m.")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "data/mit.log"
    md = sys.argv[2] if len(sys.argv) > 2 else "both"
    modes = ("oracleA", "oracleB") if md == "both" else (md,)
    report(p, modes=modes)
