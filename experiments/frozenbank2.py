"""Frozen-bank v2.2 — the FILED weight-semantics reconciliation
(docs/RESULTS.md "frozen-bank v2/v2.1": "suite-default filed pending a
weight-semantics reconciliation (e.g. WELL-tier -> full weight, else
soft — v2.2)").

The banked variant table's axis is WEIGHT SEMANTICS on the marginal
(inflatable) tier of raw-raw frozen-anchor matches:
  v2   (soft):   accept marginal at the shipped gentle-ramp inflation
                 -> wins {belg 2.042, stata-celld 0.193, fhw 0.610}
                 but REGRESSES fr101 2.132 (soft edges perturb the
                 point-stable log);
  v2.1 (strict): reject marginal outright, full weight for the rest
                 -> wins {fr101 1.827, fhw 0.242} but belg falls back
                 to 2.729 (its inflated-weak edges were the helpful
                 ones).
v2.2 bridges them with ONE knob: WELL/target-tier matches
(_coh_response == 1.0) enter at FULL weight (as in every variant);
marginal matches are accepted SOFT at weak_mult x the shipped
inflation. weak_mult=1 IS v2 (reproduction gate); weak_mult -> inf
approaches v2.1 (edges present at negligible weight). The suite-default
question: is there one weak_mult where every venue is >= its OFF
acceptance number and near its per-venue best?

_try_frozen is replicated verbatim from experiments/frozenbank.py
(frozen post-verdict) with only the weight branch changed — the same
replicate-with-cite discipline frozenbank itself used on the shipped
admission.

Usage:
  python3 -m experiments.frozenbank2 sweep            # fr101+belg x mults
  python3 -m experiments.frozenbank2 confirm <mult>   # fhw cell + stata celld
  python3 -m experiments.frozenbank2 venue <name> <mult> [policy]
"""
import sys

import numpy as np

import sspslam.encoder as S
import sspslam.lattice as L
import runners.datasets as DS
import experiments.frozenbank as FB

MULTS = (1.0, 2.0, 4.0, 16.0)
BANKED = ("banked: OFF fr101 1.881 / fhw 0.981 / stata 0.202 / belg 2.64"
          " | v2 fr101 2.132, belg 2.042, fhw 0.610, stata-celld 0.193"
          " | v2.1 fr101 1.827, fhw 0.242, stata 0.209, belg 2.729")


class FrozenBank22(FB.FrozenBankSLAM):
    """v2.2 weight semantics: WELL tier full, marginal soft at
    weak_mult x the shipped inflation."""

    def __init__(self, weak_mult=4.0, **kw):
        kw.setdefault("strict_frozen", False)
        super().__init__(**kw)
        self.weak_mult = float(weak_mult)

    def _try_frozen(self, pts, w, est):
        # --- replicated from frozenbank.FrozenBankSLAM._try_frozen ---
        best, bd = None, FB.RAD
        for f in self.frozen:
            if self.k - f["k"] < FB.MIN_AGE:
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
        if (np.hypot(cand[0] - est[0], cand[1] - est[1]) > FB.GATE_T
                or abs(S.wrap(cand[2] - est[2])) > FB.GATE_R):
            return
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
        # --- the ONE changed branch: v2.2 weight semantics ------------
        if infl > 1.0:                    # marginal tier: soft, heavier
            infl *= self.weak_mult
        # WELL/target tier (infl == 1.0): full weight, unchanged
        # ---------------------------------------------------------------
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


def venue(name, mult, policy="cell"):
    r = DS.run(name, FrozenBank22, spec=None, nph=0,
               policy=policy, weak_mult=mult)
    sl = r["slam"]
    print(f"  {name:6s} {policy:5s} mult {mult:5.1f}: ATE {r['ate']:.3f} "
          f"med {r['med']:.3f} loops {r['loops']:3d}  frozen "
          f"{sl.n_frozen:3d} att {sl.n_fatt:4d} closed {sl.n_fclose:3d}",
          flush=True)
    return r["ate"]


def sweep():
    print(f"frozen-bank v2.2 weak_mult sweep (conflict pair; mult=1 must "
          f"REPRODUCE v2)\n{BANKED}", flush=True)
    for name in ("fr101", "belg"):
        for m in MULTS:
            venue(name, m, "cell")


def confirm(mult):
    print(f"frozen-bank v2.2 confirmation at weak_mult {mult}\n{BANKED}",
          flush=True)
    venue("fhw", mult, "cell")
    venue("stata", mult, "celld")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if cmd == "sweep":
        sweep()
    elif cmd == "confirm":
        confirm(float(sys.argv[2]))
    else:
        venue(sys.argv[2], float(sys.argv[3]),
              sys.argv[4] if len(sys.argv) > 4 else "cell")
