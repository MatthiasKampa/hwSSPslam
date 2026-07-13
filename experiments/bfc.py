"""Bundle-frame smooth closure factors (BFC) — de-discretizing loop admission.

The divergence trace localized the perturbation band's first flip to the
closure edge's single-anchor ATTRIBUTION (chain median; a 0.4 mm difference
across the 5 m candidate radius re-anchored the constraint and re-rolled the
run), and the channel study showed the whole band is decision noise (given a
fixed decision sequence, the continuous response to 1e-3 map noise is ~1 cm).

Reshape the FACTOR to match the physics: the matcher matched the BUNDLE, so
the constraint is "pose relative to the weighted configuration of anchors
whose content explains the correlation peak" —
  - smooth membership: anchor weight tapers 1->0 over [R0, R1] m instead of a
    hard 5.0 m cut (the measured flip class vanishes by construction);
  - bundle-frame factor: after the match, per-anchor peak contributions
    w_a (proportional to m_a * Re<v_a, scan>) split the SAME matched pose
    into k weighted edges with total information equal to the parent's
    single edge (sum w_a / sigma_a^2);
  - smooth innovation gate: chi^2 = sum_a w_a chi_a^2 (no reference anchor).

use_bfc=False routes to the verbatim parent body (bit-exact; selftest).
Judged per PROTOCOL by bands and medians, multi-log (the aniso lesson: edge-
structure changes wobble admission; no point-value claims).
"""
import sys

import numpy as np

import sspslam.encoder as S
import sspslam.lattice as L
import sspslam.quantized as F
import experiments.stablegate as G

LOGS = F.LOGS


class BFCMixin:
    use_bfc = True
    bfc_r0 = 4.0        # taper starts [m]
    bfc_r1 = 5.5        # membership zero beyond [m]
    bfc_wmin = 0.05     # drop members below this normalized weight
    bfc_kmax = 5        # cap edges per closure (top weights, renormalized)

    def _taper(self, d):
        if d <= self.bfc_r0:
            return 1.0
        if d >= self.bfc_r1:
            return 0.0
        t = (self.bfc_r1 - d) / (self.bfc_r1 - self.bfc_r0)
        return t * t * (3 - 2 * t)          # smoothstep

    def try_constraint(self, pts, w):
        if not self.use_bfc:
            return super().try_constraint(pts, w)   # verbatim parent body
        k = self.k
        if len(pts) < 20:
            return
        me = self.pose_of(k)
        my_aid = self.aid_of(k)
        # collect out to r1 (membership handles the boundary smoothly)
        cands = [aid for aid in self.segvec
                 if abs(aid - my_aid) > self.gap_kf // 5
                 and np.linalg.norm(self.anchors[aid][:2] - me[:2])
                 < self.bfc_r1]
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
        mvec, vlist = [], []
        for a in chain:
            m_a = self._taper(np.linalg.norm(self.anchors[a][:2] - me[:2]))
            mvec.append(m_a)
            vlist.append(self.world_vec_seg(a) if m_a > 0 else None)
        if not any(m > 0 for m in mvec):
            return
        B = sum(m * v for m, v in zip(mvec, vlist) if m > 0)
        pose = self.cmatcher.match(B[L.MAIN], pts, w, me)
        pose[2] = S.wrap(pose[2])
        if np.linalg.norm(pose[:2] - me[:2]) > 0.6 \
                or abs(S.wrap(pose[2] - me[2])) > np.deg2rad(7):
            return
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
        # ---- BFC: per-anchor peak contributions -> normalized weights ----
        contrib = []
        svM = sv[L.MAIN]
        for m_a, v in zip(mvec, vlist):
            if m_a <= 0:
                contrib.append(0.0)
                continue
            contrib.append(max(0.0, m_a * float(
                np.real(np.vdot(v[L.MAIN], svM)))))
        tot = sum(contrib)
        if tot <= 0:
            j = int(np.argmin([np.linalg.norm(self.anchors[a][:2] - me[:2])
                               for a in chain]))
            weights = {chain[j]: 1.0}
        else:
            weights = {a: c_ / tot for a, c_ in zip(chain, contrib)
                       if c_ / tot >= self.bfc_wmin}
            if not weights:
                j = int(np.argmax(contrib))
                weights = {chain[j]: 1.0}
            elif len(weights) > self.bfc_kmax:
                top = sorted(weights, key=weights.get)[-self.bfc_kmax:]
                weights = {a: weights[a] for a in top}
            z = sum(weights.values())
            weights = {a: v_ / z for a, v_ in weights.items()}
        c_top = max(weights, key=weights.get)     # diagnostics / pending key
        # session-relative soft coherence response (parent logic, on B/sv)
        self.diag.append([np.nan, np.linalg.norm(pose[:2] - me[:2]),
                          abs(S.wrap(pose[2] - me[2])), len(chain), *coh,
                          l_weak, l_strong, u_weak[0], u_weak[1],
                          np.nan, np.nan,
                          abs(float((pose[:2] - me[:2]) @ u_weak)),
                          ridge_w, ridge_s,
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
        # ---- smooth innovation gate: weighted chi^2 over members ----------
        since = k - self.last_accept_k
        chi = 0.0
        sig_cache = {}
        for a, w_a in weights.items():
            Z_a = L.se2_mul(L.se2_inv(self.anchors[a]), Zk)
            Zc_a = L.se2_mul(L.se2_inv(self.anchors[a]),
                             self.anchors[my_aid])
            lever = np.linalg.norm(me[:2] - self.anchors[a][:2])
            sig_t = np.sqrt(0.08 ** 2 + (0.05 * lever) ** 2)
            sig_r = np.deg2rad(2.0)
            sig_cache[a] = (sig_t, sig_r, Z_a)
            s_at = sig_t + min(0.30, 0.002 * since)
            s_ar = sig_r + min(np.deg2rad(6), np.deg2rad(0.03) * since)
            chi += w_a * (
                (np.linalg.norm(Z_a[:2] - Zc_a[:2]) / s_at) ** 2
                + (S.wrap(Z_a[2] - Zc_a[2]) / s_ar) ** 2)
        if self.use_innov and chi > 9.0:
            self.n_innov_rej += 1
            return
        # ---- emit the weighted factor group -------------------------------
        accepted_any = False
        for a, w_a in weights.items():
            if (a, my_aid) in self.banned:
                continue
            sig_t, sig_r, Z_a = sig_cache[a]
            sw = np.sqrt(w_a)
            edge = (a, my_aid, Z_a, sw / (sig_t * infl),
                    sw / (sig_r * infl), "loop")
            key = (a, my_aid)
            if key in self.edge_seen:
                if edge[3] * 3.5 < self.edges[self.edge_seen[key]][3]:
                    continue
                self.edges[self.edge_seen[key]] = edge
            else:
                self.edge_seen[key] = len(self.edges)
                self.edges.append(edge)
            accepted_any = True
        if not accepted_any:
            return
        if infl <= self.infl_weak:
            self.dirty = True
            self.pending_new.append((c_top, my_aid))
            self.last_accept_k = k
            self._streak = 0
            self._streak_xy = None
        else:
            self._suppress(me[:2], k)
        self.diag[-1][-1] = 1.0


class BFCSLAM(BFCMixin, G.JitterMixin, F.BandSLAM):
    """BFC over the band harness; jit_k=0 (no jitter gate); use_bfc=False
    routes to JitterMixin's verbatim parent body (bit-exact)."""
    jit_k = 0

    def __init__(self, use_bfc=True, **kw):
        super().__init__(**kw)
        self.use_bfc = use_bfc


def selftest():
    import sspslam.bounded as B
    cap = 1200
    a = F.run_log(LOGS["fr101"], B.BoundedSLAM, cap=cap)
    b = F.run_log(LOGS["fr101"], BFCSLAM, cap=cap, use_bfc=False,
                  spec=None, nph=0)
    d = float(np.abs(a["fin"] - b["fin"]).max())
    print(f"selftest fr101[:{cap}] parent {a['ate']:.4f} "
          f"bfc-off {b['ate']:.4f} max|dpose| {d:.2e}")
    assert d == 0.0
    print("selftest ok: use_bfc=False is bit-exact to parent")


def eps0(logs):
    for lg in logs:
        r = F.run_log(LOGS[lg], BFCSLAM, spec=None, nph=0)
        print(f"  {lg} BFC eps0: ATE {r['ate']:.3f} med {r['med']:.3f} "
              f"loops {r['loops']} [veto {r['veto']} innov {r['innov']}]",
              flush=True)


def band(logs):
    for lg in logs:
        vals = []
        for eps, seed in F.BAND_EPS:
            r = F.run_log(LOGS[lg], BFCSLAM, eps=eps, seed=seed,
                          spec=None, nph=0)
            vals.append(r["ate"])
            print(f"  {lg} BFC eps{eps:7.0e}/s{seed}: ATE {r['ate']:.3f} "
                  f"med {r['med']:.3f} loops {r['loops']}", flush=True)
        v = np.array(vals)
        print(f"  {lg} BFC BAND [{v.min():.2f} .. {v.max():.2f}] "
              f"median {np.median(v):.2f}", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if what == "selftest":
        selftest()
    elif what == "eps0":
        eps0(sys.argv[2:] or ["fr101", "fr079", "aces", "intel"])
    elif what == "band":
        band(sys.argv[2:] or ["intel"])
