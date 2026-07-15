"""FROZEN raw-sample anchor bank (user 2026-07-14: "frozen samples (e.g.
the first, some metric for other especially salient samples) that won't
get overwritten, to enhance loop closure and remove drift through
anchoring").

Design: alongside the rotating reservoir, a FROZEN tier of raw scans that
are never overwritten. Scan #0 is definitionally true (it IS the gauge
origin) — re-registering against it removes ALL drift at start revisits.
Later frozen samples pin drift accumulated SINCE their freeze (their own
freeze-time gauge is the commit-policy-wall floor). Selection is bounded
per spatial cell (O(area), the map's own law) and optionally
DISTINCTIVENESS-gated: only freeze scans whose place-vector is unlike the
existing bank — corridor twins are excluded from anchor duty by
construction (anti-aliasing at selection time).

Closure: every att_every kf, the nearest sufficiently-old frozen sample
within rad (est-based candidacy, GT-free) is re-encoded at its LIVE pose
(anchors[owner] o rel_f) and the current scan is matched against it
raw-vs-raw; a gated match appends a standard loop edge
Z = inv(anchors[owner]) o cand o inv(rel_cur) (the try_constraint form),
solved by the existing IRLS relax.

Policies: off | first (scan #0 only) | cell (first-in-3m-cell) |
celld (cell + distinctiveness tau).

Usage: python3 -m experiments.frozenbank school_run2
"""
import sys

import numpy as np

import sspslam.encoder as S
import sspslam.lattice as L
import sspslam.quantized as F
import runners.datasets as DS

CELL_M = 3.0
CAP = 64
ATT_EVERY = 4
MIN_AGE = 300                    # kf; anti-trivial gate (shipped gap_kf)
RAD = 5.0                        # m, est-based candidate radius
DIST_TAU = 0.75                  # freeze only if max place-sim < tau
GATE_T, GATE_R = 1.5, np.deg2rad(20)
SIG_T, SIG_R = 0.08, 0.03
COH_FLOOR = 0.45                 # v1.5: raw-raw match coherence gate +
INFL_POW = 2.0                   # weight inflation by 1/coh^p (belg fix)


class FrozenBankSLAM(F.BandSLAM):
    def __init__(self, policy="cell", strict_frozen=True, **kw):
        super().__init__(**kw)
        self.policy = policy
        self.strict_frozen = strict_frozen
        self.frozen = []             # dict(pts, w, owner, rel, k, vec)
        self.fcells = set()
        self.n_frozen = self.n_fatt = self.n_fclose = 0

    def _placevec(self, pts, w):
        v = np.exp(1j * (pts @ L.W[L.MAIN].T)).T @ w
        return v / max(np.linalg.norm(v), 1e-12)

    def _try_frozen(self, pts, w, est):
        best, bd = None, RAD
        for f in self.frozen:
            if self.k - f["k"] < MIN_AGE:
                continue
            pf = L.se2_mul(self.anchors[f["owner"]], f["rel"])
            d = np.hypot(pf[0] - est[0], pf[1] - est[1])
            if d < bd:
                best, bd = (f, pf), d
        if best is None:
            return
        f, pf = best
        self.n_fatt += 1
        PW = f["pts"] @ S._rot(pf[2]).T + pf[:2]
        B = np.exp(1j * (PW @ L.W[L.MAIN].T)).T @ f["w"]
        cand = self.matcher.match(B, pts, w, est.copy())
        cand[2] = S.wrap(cand[2])
        if (np.hypot(cand[0] - est[0], cand[1] - est[1]) > GATE_T
                or abs(S.wrap(cand[2] - est[2])) > GATE_R):
            return
        # v2: the SHIPPED admission, replicated verbatim on the raw-raw
        # match (v1.0 fixed-weight crashed belg; v1.5's 2-constant gate
        # fixed belg but lost fr101/stata — per-regime, as always). Per-
        # ring coherence + analytic translation-Hessian + ridge probes ->
        # self._coh_response (all session constants + the coh_ref EMA).
        if self.coh_ref is None:
            return
        PWf = f["pts"] @ S._rot(pf[2]).T + pf[:2]
        Bfull = np.exp(1j * (PWf @ L.W.T)).T @ f["w"]
        sv = L.ENC.shift(cand[:2]) * L.encode(pts @ S._rot(cand[2]).T, w)
        c = (np.conj(Bfull) * sv).reshape(L.N_RING, L.N_ANG)
        Br = Bfull.reshape(L.N_RING, L.N_ANG)
        svr = sv.reshape(L.N_RING, L.N_ANG)
        coh = c.sum(1).real / (np.linalg.norm(Br, axis=1)
                               * np.linalg.norm(svr, axis=1) + 1e-12)
        cM = c[:4].ravel()
        nrmM = (np.linalg.norm(Br[:4].ravel())
                * np.linalg.norm(svr[:4].ravel()) + 1e-12)
        K = (L.W[L.MAIN] * (cM.real / nrmM)[:, None]).T @ L.W[L.MAIN]
        evals, evecs = np.linalg.eigh(K)
        l_weak, l_strong = float(evals[0]), float(evals[1])
        ts = np.arange(-1.2, 1.21, 0.06)
        s0 = float(cM.real.sum())
        far = np.abs(ts) >= 0.35
        ridge = [0.0, 0.0]
        for j, u in enumerate((evecs[:, 0], evecs[:, 1])):
            proj = L.W[L.MAIN] @ u
            sc = (np.exp(1j * ts[:, None] * proj[None, :]) @ cM).real
            ridge[j] = float(sc[far].max() / max(s0, 1e-12))
        infl = self._coh_response(coh, l_weak, l_strong, ridge[0],
                                  ridge[1], est, self.k)
        if infl < 0:
            return
        # v2.1 (STRICT mode): accept-or-reject — an inflatable (weak)
        # match is rejected outright; accepted ones enter at full weight.
        # v2's soft inflation regressed fr101 (good raw matches weakened);
        # v1.0 full-weight crashed belg (junk at full strength). Strict =
        # the shipped gates decide membership, weights stay strong.
        if self.strict_frozen and infl > 1.0:
            return
        if self.strict_frozen:
            infl = 1.0
        aid_c, rel_c = self.kf_ref[-1]
        if aid_c == f["owner"] or aid_c not in self.segvec:
            return
        Z = L.se2_mul(L.se2_inv(self.anchors[f["owner"]]),
                      L.se2_mul(cand, L.se2_inv(rel_c)))
        key = (f["owner"], aid_c)
        if key in getattr(self, "banned", set()):
            return
        lever = np.hypot(est[0] - self.anchors[f["owner"]][0],
                         est[1] - self.anchors[f["owner"]][1])
        sig_t = np.sqrt(0.08 ** 2 + (0.05 * lever) ** 2)
        sig_r = np.deg2rad(2.0)
        edge = (f["owner"], aid_c, Z, 1 / (sig_t * infl),
                1 / (sig_r * infl), "loop")
        if key in self.edge_seen:
            self.edges[self.edge_seen[key]] = edge
        else:
            self.edge_seen[key] = len(self.edges)
            self.edges.append(edge)
        self.dirty = True
        self.n_fclose += 1

    def _maybe_freeze(self, pts, w, est):
        if len(pts) < 20 or len(self.frozen) >= CAP:
            return
        cell = (int(np.floor(est[0] / CELL_M)),
                int(np.floor(est[1] / CELL_M)))
        first = self.k == 0
        if self.policy == "first" and not first:
            return
        if not first and cell in self.fcells:
            return
        vec = self._placevec(pts, w)
        if self.policy == "celld" and self.frozen and not first:
            sims = [abs(np.vdot(vec, f["vec"])) for f in self.frozen]
            if max(sims) > DIST_TAU:
                return
        aid = len(self.anchors) - 1
        self.frozen.append(dict(
            pts=pts.copy(), w=w.copy(), owner=aid,
            rel=L.se2_mul(L.se2_inv(self.anchors[aid]), est),
            k=self.k, vec=vec))
        self.fcells.add(cell)
        self.n_frozen += 1

    def add_keyframe(self, pts, w, guess):
        est = super().add_keyframe(pts, w, guess)
        if self.policy != "off":
            if self.k % ATT_EVERY == 0 and self.k > 0:
                self._try_frozen(pts, w, est)
            self._maybe_freeze(pts, w, est)
        return est


def main(run="school_run2"):
    print(f"frozen-bank anchoring ({run}; OFF gate must reproduce "
          f"acceptance):")
    for policy in ("off", "first", "cell", "celld"):
        r = DS.run(run, FrozenBankSLAM, spec=None, nph=0, policy=policy)
        sl = r["slam"]
        print(f"  {policy:5s}: ATE {r['ate']:.3f} med {r['med']:.3f} "
              f"loops {r['loops']:3d}  frozen {sl.n_frozen:3d} "
              f"att {sl.n_fatt:4d} closed {sl.n_fclose:3d}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "school_run2")
