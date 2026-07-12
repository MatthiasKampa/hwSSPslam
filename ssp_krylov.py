"""Truncated-Krylov Gauss-Newton backend: an exactly-specifiable
semiconvergent early stop to replace the scipy-TRF-path-dependent max_nfev=30.

Background (ssp_bounded._gn docstring; RESULTS.md "Tikhonov prior sweep"):
the shipped solver's max_nfev=30 TRF truncation is the audited LOAD-BEARING
regularizer of flat-valley directions on floppy graphs — fully-converged
solvers slide the valley (fr079 degrades 2.3x at max_nfev=300), and the
principled replacement (static Tikhonov prior toward P0) was built and
REJECTED (worst-of-3 ratio 3.39 vs 1.97). Deployment problem: "port scipy
TRF exactly, stopped at nfev 30" is not a portable spec. Hypothesis tested
here: a FIXED-BUDGET truncated-Krylov solve of each Gauss-Newton step (CG on
the normal equations, started at 0, HARD iteration cap, no tolerance-based
exit) is the same MECHANISM class — semiconvergent iterative regularization
(Hanke 1997): the Krylov space applies the dominant data-consistent
components of the correction first, and (Steihaug) the CG iterate norm grows
monotonically from 0 toward the full GN step, so truncation both orders and
limits the applied correction path-dependently — unlike the rejected static
prior, which stiffens every direction independently.

Implementation: KrylovMixin._gn keeps the shipped residual/Jacobian
construction VERBATIM (copied-body override per PROTOCOL 1, pattern of
ssp_stablegate — the parent module is untouched) and replaces ONLY the
scipy.optimize.least_squares call with the exactly-specifiable loop

    x = x0
    repeat n_outer times:
        r = resid(x);  J = jac(x)                    # shipped analytic sparse
        s_j = 1 / ||J[:, j]||_2   (zero column -> 1) # x_scale="jac" analogue
        y = CG_k((JS)^T (JS), -(JS)^T r), y0 = 0, k = n_inner HARD cap
        x += S y
    return x and resid(x) edge rows

No damping, no trust region, no convergence test: the only early exits are
exact-zero residual and nonpositive/nonfinite curvature (both deterministic
degeneracies, not tolerances). The solver is ~25 lines of csr matvecs — no
scipy.optimize / scipy.sparse.linalg in the solve path.

use_krylov=False falls through to super()._gn (the parent scipy path) and is
asserted BIT-EXACT to the parent over a full fr101[:1200] run (selftest).

Anti-oracle: .gfs references score only (harness = ssp_datasets.run).
Deterministic: no RNG in the solver; band probes use the standard
ssp_fpga.BandSLAM freeze-noise ladder with fixed seeds.

Usage:
  python3 ssp_krylov.py selftest                # CG check + bit-exact proof
  python3 ssp_krylov.py screen                  # fr101[:1200] grid sweep
  python3 ssp_krylov.py run fr101 2 10          # full log at (n_outer,n_inner)
  python3 ssp_krylov.py band fr079 2 10         # eps-ladder perturbation band
"""
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix, diags

import ssp_slam as S
import ssp_fpga as F
import ssp_datasets as DS


def cg_normal(J, r, n_inner):
    """k-step CG on the column-scaled Gauss-Newton normal equations.

    Solves (JS)^T (JS) y = -(JS)^T r from y0 = 0 with S = diag(s),
    s_j = 1/||J[:,j]||_2 (zero columns -> 1), HARD cap n_inner iterations,
    tolerance-free (only exact-zero / nonfinite degeneracy exits).
    Returns dx = S y in the UNSCALED variables."""
    cn = np.sqrt(np.asarray(J.multiply(J).sum(axis=0)).ravel())
    cn[cn == 0.0] = 1.0
    sc = 1.0 / cn
    Js = (J @ diags(sc)).tocsr()
    JsT = Js.T.tocsr()
    b = -(JsT @ r)
    y = np.zeros_like(b)
    rk = b.copy()
    p = rk.copy()
    rs = float(rk @ rk)
    for _ in range(n_inner):
        if rs == 0.0:
            break
        q = Js @ p
        pAp = float(q @ q)                 # p^T (Js^T Js) p >= 0 exactly
        if pAp <= 0.0 or not np.isfinite(pAp):
            break
        alpha = rs / pAp
        y += alpha * p
        rk -= alpha * (JsT @ q)
        rs_new = float(rk @ rk)
        if not np.isfinite(rs_new):
            break
        p = rk + (rs_new / rs) * p
        rs = rs_new
    return sc * y


class KrylovMixin:
    """_gn override: shipped resid/jac verbatim, solver = truncated-Krylov GN.
    use_krylov=False -> super()._gn (bit-exact parent path)."""

    use_krylov = True
    n_outer = 2
    n_inner = 10
    n_kry_solve = 0

    def _gn(self, eds, wl, free, P0):
        if not self.use_krylov:
            return super()._gn(eds, wl, free, P0)
        # ---- residual/Jacobian construction copied VERBATIM from
        # ssp_bounded.BoundedSLAM._gn (parent untouched, PROTOCOL 1) --------
        E, Fn = len(eds), len(free)
        if E == 0 or Fn == 0:
            return P0.copy(), np.zeros(3 * E)
        aa = np.array([e[0] for e in eds])
        bb = np.array([e[1] for e in eds])
        Zt = np.stack([e[2] for e in eds])
        wts = np.array([e[3] for e in eds])
        wrs = np.array([e[4] for e in eds])
        ss = np.array([wl.get(e, 1.0) for e in range(E)])
        wt_s, wr_s = wts * ss, wrs * ss
        lt, lr = self.prior_lam_t, self.prior_lam_r
        Xp = P0[free]                                   # prior reference poses

        def resid(x):
            P = P0.copy()
            P[free] = x.reshape(-1, 3)
            ca, sa = np.cos(P[aa, 2]), np.sin(P[aa, 2])
            dx = P[bb, 0] - P[aa, 0]
            dy = P[bb, 1] - P[aa, 1]
            out = np.empty((E, 3))
            out[:, 0] = (ca * dx + sa * dy - Zt[:, 0]) * wt_s
            out[:, 1] = (-sa * dx + ca * dy - Zt[:, 1]) * wt_s
            out[:, 2] = S.wrap(P[bb, 2] - P[aa, 2] - Zt[:, 2]) * wr_s
            if lt <= 0:
                return out.ravel()
            X = x.reshape(-1, 3)
            pr = np.empty((Fn, 3))
            pr[:, 0] = lt * (X[:, 0] - Xp[:, 0])
            pr[:, 1] = lt * (X[:, 1] - Xp[:, 1])
            pr[:, 2] = lr * S.wrap(X[:, 2] - Xp[:, 2])
            return np.concatenate([out.ravel(), pr.ravel()])

        ro = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
        comp = np.array([0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 2])
        isb = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1], bool)
        col_of = np.full(len(P0), -1)
        col_of[free] = np.arange(Fn)
        node = np.where(isb, bb[:, None], aa[:, None])      # E x 12
        nc = col_of[node]
        keep = (nc >= 0).ravel()
        rows = (3 * np.arange(E)[:, None] + ro).ravel()[keep]
        cols = (3 * nc + comp).ravel()[keep]
        nrows = 3 * E
        pvals = np.empty(0)
        if lt > 0:                     # prior block: constant diagonal rows
            rows = np.concatenate([rows, 3 * E + np.arange(3 * Fn)])
            cols = np.concatenate([cols, np.arange(3 * Fn)])
            pvals = np.tile([lt, lt, lr], Fn)
            nrows += 3 * Fn

        def jac(x):
            P = P0.copy()
            P[free] = x.reshape(-1, 3)
            ca, sa = np.cos(P[aa, 2]), np.sin(P[aa, 2])
            dx = P[bb, 0] - P[aa, 0]
            dy = P[bb, 1] - P[aa, 1]
            V = np.empty((E, 12))
            V[:, 0], V[:, 1], V[:, 2] = -ca, -sa, -sa * dx + ca * dy
            V[:, 3], V[:, 4] = ca, sa
            V[:, 5], V[:, 6], V[:, 7] = sa, -ca, -(ca * dx + sa * dy)
            V[:, 8], V[:, 9] = -sa, ca
            V[:, :10] *= wt_s[:, None]
            V[:, 10], V[:, 11] = -wr_s, wr_s
            return csr_matrix((np.concatenate([V.ravel()[keep], pvals]),
                               (rows, cols)), shape=(nrows, 3 * Fn))

        x0 = P0[free].ravel()
        # ---- end verbatim copy; REPLACED solver: truncated-Krylov GN ------
        x = x0.copy()
        for _ in range(self.n_outer):
            x = x + cg_normal(jac(x), resid(x), self.n_inner)
        self.n_kry_solve += 1
        P = P0.copy()
        P[free] = x.reshape(-1, 3)
        return P, resid(x)[:3 * E]


class KrylovBand(KrylovMixin, F.BandSLAM):
    """Truncated-Krylov backend + the standard freeze-noise band harness."""

    def __init__(self, use_krylov=True, n_outer=2, n_inner=10, **kw):
        super().__init__(**kw)
        self.use_krylov = use_krylov
        self.n_outer = n_outer
        self.n_inner = n_inner


# ---------------------------------------------------------------------------
# harness wrappers (all runs via ssp_datasets.run; shipped config only:
# sample="seg", spec=None, nph=0 — this experiment varies the SOLVER, nothing
# else, per PROTOCOL 6)
# ---------------------------------------------------------------------------

def _run(name, cap=None, eps=0.0, seed=1, **kry):
    t0 = time.time()
    r = DS.run(name, KrylovBand, cap=cap, sample="seg", spec=None, nph=0,
               eps=eps, seed=seed, **kry)
    r["wall"] = time.time() - t0
    return r


def _fmt(tag, r):
    return (f"  {tag}  ATE {r['ate']:.4f}  med {r['med']:.3f}  "
            f"loops {r['loops']}  [veto {r['veto']} infl {r['infl']} "
            f"innov {r['innov']} pruned {r['pruned']}]  "
            f"{r['ms']:.1f} ms/kf  wall {r['wall']:.0f}s")


def selftest():
    # (a) the CG solver itself: full-budget CG must match the dense
    # lstsq solution of a fixed random overdetermined system.
    rng = np.random.default_rng(7)             # test fixture only, not logic
    A = csr_matrix(rng.standard_normal((60, 24)))
    b = rng.standard_normal(60)
    dx = cg_normal(A, b, 500)
    ref = np.linalg.lstsq(A.toarray(), -b, rcond=None)[0]
    err = float(np.abs(dx - ref).max())
    print(f"selftest CG: full-budget vs dense lstsq max|d| {err:.2e}")
    assert err < 1e-9, "cg_normal does not solve the normal equations"
    d5 = cg_normal(A, b, 5)
    print(f"selftest CG: 5-step iterate differs from solution by "
          f"{float(np.abs(d5 - ref).max()):.2e} (cap binds)")
    # (b) neutralised path bit-exact to parent over a full pipeline run.
    cap = 1200
    t0 = time.time()
    a = DS.run("fr101", F.BandSLAM, cap=cap, spec=None, nph=0)
    bres = DS.run("fr101", KrylovBand, cap=cap, spec=None, nph=0,
                  use_krylov=False)
    d = float(np.abs(a["fin"] - bres["fin"]).max())
    print(f"selftest fr101[:{cap}]  parent ATE {a['ate']:.4f}  "
          f"use_krylov=False ATE {bres['ate']:.4f}  max|dpose| {d:.2e}  "
          f"({time.time() - t0:.0f}s)")
    assert d == 0.0, "neutralised KrylovBand is NOT bit-exact to parent"
    print("selftest ok: neutralised == parent bit-exact")


def screen(cap=1200):
    print(f"SCREEN fr101[:{cap}] — truncated-Krylov GN grid "
          f"(shipped reference at this cap: 1.9284)\n")
    r0 = _run("fr101", cap=cap, use_krylov=False)
    print(_fmt("shipped (TRF nfev30)   ", r0), flush=True)
    for n_outer in (1, 2, 3):
        for n_inner in (5, 10, 20, 40):
            r = _run("fr101", cap=cap, n_outer=n_outer, n_inner=n_inner)
            print(_fmt(f"krylov o{n_outer} i{n_inner:<2d}          ", r),
                  flush=True)


def run_one(name, n_outer, n_inner, cap=None):
    r = _run(name, cap=cap, n_outer=n_outer, n_inner=n_inner)
    print(_fmt(f"{name} krylov o{n_outer} i{n_inner}", r), flush=True)


def band(name, n_outer, n_inner):
    print(f"BAND {name} krylov o{n_outer} i{n_inner} — eps ladder "
          f"{[e for e, s in F.BAND_EPS]}")
    vals = []
    for eps, seed in F.BAND_EPS:
        r = _run(name, eps=eps, seed=seed, n_outer=n_outer, n_inner=n_inner)
        vals.append(r["ate"])
        print(_fmt(f"{name} o{n_outer} i{n_inner} eps{eps:7.0e}/s{seed}", r),
              flush=True)
    v = np.array(vals)
    print(f"  {name} krylov o{n_outer} i{n_inner} BAND "
          f"[{v.min():.2f} .. {v.max():.2f}] median {np.median(v):.2f}",
          flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if what == "selftest":
        selftest()
    elif what == "screen":
        screen(int(sys.argv[2]) if len(sys.argv) > 2 else 1200)
    elif what == "run":
        run_one(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                cap=int(sys.argv[5]) if len(sys.argv) > 5 else None)
    elif what == "band":
        band(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
