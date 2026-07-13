"""Decision-stability (dither-consensus) closure admission.

The 2026-07-10 perturbation-band study localized the pipeline's fragility to
FLIPPED DISCRETE DECISIONS in the closure cascade: >=1e-4 relative map noise
re-rolls which candidates are admitted (audit: the continuous channel predicts
~1 cm at eps=1e-3; observed ~1.7 m with shifted decision counters). Hypothesis
tested here: candidates whose MATCHED POSE is unstable under exactly that
noise channel (small relative jitter of the old-map bundle) are the flip-prone
ambiguous ones; rejecting them DETERMINISTICALLY should (a) narrow the
perturbation band, (b) cost little on the point value, because a genuine
closure's peak should be jitter-stable.

JitterMixin.try_constraint = the shipped body (copied verbatim from
ssp_bounded.BoundedSLAM.try_constraint — subclass override per PROTOCOL 1,
parent untouched) + one inserted consensus block after the match gate:
re-match against jit_k bundles B + eps*mean|B|*N(0,1) (deterministic seed per
anchor pair) and suppress the candidate unless every re-match lands within
(jit_tol_t, jit_tol_r) of the unjittered pose.

Neutralisation: jit_k=0 -> bit-exact parent (asserted in selftest).
Usage:
  python3 -m experiments.stablegate selftest
  python3 -m experiments.stablegate band intel      # jitter-gate perturbation bands
"""
import sys

import numpy as np

import sspslam.encoder as S
import sspslam.lattice as L
import sspslam.quantized as F

LOGS = F.LOGS


class JitterMixin:
    jit_k = 2                    # consensus re-matches (0 = neutral)
    jit_eps = 1e-3               # relative bundle jitter (the band's channel)
    jit_tol_t = 0.07             # pose-stability tolerance [m]
    jit_tol_r = np.deg2rad(0.7)  # [rad]
    n_jit_rej = 0

    def try_constraint(self, pts, w):
        # ---- verbatim from ssp_bounded.BoundedSLAM.try_constraint, with the
        # jitter-consensus block inserted after the match gate ----
        k = self.k
        if len(pts) < 20:
            return
        me = self.pose_of(k)
        my_aid = self.aid_of(k)
        cands = [aid for aid in self.segvec
                 if abs(aid - my_aid) > self.gap_kf // 5
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
        B = sum(self.world_vec_seg(a) for a in chain)
        pose = self.cmatcher.match(B[L.MAIN], pts, w, me)
        pose[2] = S.wrap(pose[2])
        if np.linalg.norm(pose[:2] - me[:2]) > 0.6 \
                or abs(S.wrap(pose[2] - me[2])) > np.deg2rad(7):
            return
        # ---- INSERTED: decision-stability consensus --------------------
        if self.jit_k:
            c_mid = int(chain[len(chain) // 2])
            rng = np.random.default_rng((c_mid << 20) ^ int(my_aid))
            s = float(np.abs(B).mean())
            for _ in range(self.jit_k):
                Bj = B + self.jit_eps * s * (
                    rng.standard_normal(len(B))
                    + 1j * rng.standard_normal(len(B)))
                pj = self.cmatcher.match(Bj[L.MAIN], pts, w, me)
                if np.linalg.norm(pj[:2] - pose[:2]) > self.jit_tol_t \
                        or abs(S.wrap(pj[2] - pose[2])) > self.jit_tol_r:
                    self.n_jit_rej += 1
                    self._suppress(me[:2], k)
                    return
        # ---- end insert; parent body continues unchanged ---------------
        sv = L.ENC.shift(pose[:2]) * L.encode(pts @ S._rot(pose[2]).T, w)
        c = (np.conj(B) * sv).reshape(L.N_RING, L.N_ANG)
        Br = B.reshape(L.N_RING, L.N_ANG)
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
        if getattr(self, "_nearest_attrib", False):
            # STABLE attribution (divergence-trace fix candidate): the chain
            # anchor nearest the matched pose is invariant to membership
            # flips at the candidate-radius boundary, unlike the chain median
            # (first flip of the perturbation band, intel kf 3424).
            c_aid = int(chain[int(np.argmin(
                [np.linalg.norm(self.anchors[a][:2] - pose[:2])
                 for a in chain]))])
        else:
            c_aid = chain[len(chain) // 2]
        lever = np.linalg.norm(me[:2] - self.anchors[c_aid][:2])
        sig_t = np.sqrt(0.08 ** 2 + (0.05 * lever) ** 2)
        sig_r = np.deg2rad(2.0)
        err = e_weak = e_strong = np.nan
        if self.diag_gt is not None:
            gtk = self.diag_gt
            Zm = L.se2_mul(L.se2_inv(self.anchors[c_aid]),
                           L.se2_mul(pose, L.se2_inv(self.kf_ref[k][1])))
            Zt = L.se2_mul(L.se2_inv(gtk[c_aid * 5]), gtk[k - (k % 5)])
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
                Z[:2] += self._inj_rng.uniform(0.5, 1.5) \
                    * np.array([np.cos(ang), np.sin(ang)])
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
        edge = (c_aid, my_aid, Z, 1 / (sig_t * infl), 1 / (sig_r * infl),
                "loop")
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


class JitterBand(JitterMixin, F.BandSLAM):
    """Jitter-consensus gate + the freeze-noise band harness."""

    def __init__(self, jit_k=2, jit_eps=1e-3, jit_tol_t=0.07,
                 jit_tol_deg=0.7, **kw):
        super().__init__(**kw)
        self.jit_k = jit_k
        self.jit_eps = jit_eps
        self.jit_tol_t = jit_tol_t
        self.jit_tol_r = np.deg2rad(jit_tol_deg)


class ConsensusMatcher(S.Matcher):
    """FRONTEND jitter-consensus (v2): the channel-localization showed the
    perturbation band enters entirely through the frontend's local-bundle
    reads (loop-channel noise = exactly shipped), and pinning coh_ref does
    not collapse it — so the amplifier is the frontend's own discrete layer
    (accept-gate / coarse-argmax flips integrating through the est chain).
    This matcher re-matches against fc_k jittered copies of the local map
    and, if the matched pose is jitter-UNSTABLE, returns a far pose so the
    caller's accept gate fails -> clean odometry fallback for that keyframe
    (ambiguous frontend evidence is declined, deterministically)."""

    def __init__(self, *a, fc_k=2, fc_eps=1e-3, fc_tol_t=0.05,
                 fc_tol_deg=0.5, **kw):
        super().__init__(*a, **kw)
        self.fc_k, self.fc_eps = fc_k, fc_eps
        self.fc_tol_t = fc_tol_t
        self.fc_tol_r = np.deg2rad(fc_tol_deg)
        self._fc_rng = np.random.default_rng(4242)
        self.n_fc_rej = 0

    def match(self, M, pts, weights, guess):
        pose = super().match(M, pts, weights, guess)
        if not self.fc_k:
            return pose
        s = float(np.abs(M).mean())
        for _ in range(self.fc_k):
            Mj = M + self.fc_eps * s * (
                self._fc_rng.standard_normal(len(M))
                + 1j * self._fc_rng.standard_normal(len(M)))
            pj = super().match(Mj, pts, weights, guess)
            if np.linalg.norm(pj[:2] - pose[:2]) > self.fc_tol_t \
                    or abs(S.wrap(pj[2] - pose[2])) > self.fc_tol_r:
                self.n_fc_rej += 1
                far = np.array(guess, float)
                far[0] += 99.0          # fails the caller's accept gate ->
                return far              # odometry fallback for this keyframe
        return pose


class FrontConsensusSLAM(F.BandSLAM):
    def __init__(self, fc_k=2, fc_eps=1e-3, **kw):
        super().__init__(**kw)
        self.matcher = ConsensusMatcher(L.ENC_MAIN, fc_k=fc_k, fc_eps=fc_eps,
                                        t_half=0.48, rot_half_deg=9,
                                        rot_step_deg=1.5, perm=(4, L.N_ANG))


def selftest():
    import sspslam.bounded as B
    cap = 1200
    a = F.run_log(LOGS["fr101"], B.BoundedSLAM, cap=cap)
    b = F.run_log(LOGS["fr101"], JitterBand, cap=cap, jit_k=0,
                  spec=None, nph=0)
    d = float(np.abs(a["fin"] - b["fin"]).max())
    print(f"selftest fr101[:{cap}]  parent {a['ate']:.4f}  "
          f"jit_k=0 {b['ate']:.4f}  max|dpose| {d:.2e}")
    assert d == 0.0, "jit_k=0 is NOT bit-exact to parent"
    print("selftest ok: neutralised == parent bit-exact")


def band(logs):
    for lg in logs:
        vals = []
        for eps, seed in F.BAND_EPS:
            r = F.run_log(LOGS[lg], JitterBand, eps=eps, seed=seed,
                          spec=None, nph=0)
            vals.append(r["ate"])
            print(f"  {lg} jitter-gate eps{eps:7.0e}/s{seed}  "
                  f"ATE {r['ate']:.3f}  med {r['med']:.3f}  "
                  f"loops {r['loops']}", flush=True)
        v = np.array(vals)
        print(f"  {lg} jitter-gate BAND [{v.min():.2f} .. {v.max():.2f}] "
              f"median {np.median(v):.2f}", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if what == "selftest":
        selftest()
    elif what == "band":
        band(sys.argv[2:] or ["intel"])
    elif what == "front":
        for lg in sys.argv[2:] or ["intel"]:
            vals = []
            for eps, seed in F.BAND_EPS:
                r = F.run_log(LOGS[lg], FrontConsensusSLAM, eps=eps,
                              seed=seed, spec=None, nph=0)
                vals.append(r["ate"])
                print(f"  {lg} front-consensus eps{eps:7.0e}/s{seed}  "
                      f"ATE {r['ate']:.3f}  med {r['med']:.3f}  "
                      f"loops {r['loops']}", flush=True)
            v = np.array(vals)
            print(f"  {lg} front-consensus BAND [{v.min():.2f} .. "
                  f"{v.max():.2f}] median {np.median(v):.2f}", flush=True)
