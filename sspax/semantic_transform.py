"""The bounded QUERYABLE semantic map obeys the SSP TRANSFORM ALGEBRA — so a
graph correction / loop closure MOVES the whole map (every object AND its class
binding) with one O(D) phase op, never re-encoding any object. This unifies the
project's two contributions: the map is simultaneously QUERYABLE (by class) and
ALGEBRAICALLY TRANSFORMABLE (translate/rotate on frozen content).

map = Σ_o roles[bits_o] ⊙ pos(x_o),  pos(x) = exp(i W·x),  ⊙ = elementwise mult.
- TRANSLATION by t:  map ⊙ exp(i W·t)  ==  rebuild at {x_o + t}, BIT-EXACT (any t)
  because pos(x)⊙pos(t) = pos(x+t) and t is global.
- ROTATION by R:     map[perm_R]        ==  rebuild at {R·x_o}, exact at LATTICE
  angles (index permutation), approximate off-lattice (the ring-stagger sphere's
  approximate-permutability; residual shrinks with D / the d/dθ correction).
The class query still localises objects in the NEW frame after either transform.

Anti-oracle: deterministic synthetic scene; positions score localisation only.

  python3 -m sspax.semantic_transform
"""
import numpy as np

from sspax import semantic as SM
from sspax import sphere as SPH

W = SM.W_POS
D = SM.D
RINGKW = dict(spacing="arc", apportion="const", stagger=0.5, n_rings=6)


def _scene(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return [(SM.CLASSES[rng.integers(len(SM.CLASSES))],
             np.array([rng.uniform(-4, 4), rng.uniform(-4, 4), 0.5]))
            for _ in range(n)]


def _query_localize(mp, code, truth_xy, roles_perm=None):
    grid, shape = SM._grid()
    if roles_perm is None:
        dens = SM.query(mp, code, grid)
    else:
        g = np.concatenate([grid, np.full((len(grid), 1), 0.5)], 1)
        unb = np.conj(SM.ROLES[code][:, roles_perm]).sum(0) * mp
        dens = np.real(np.exp(1j * (g @ W.T)) @ np.conj(unb)) / D
    tp, fp, errs = SM._match(SM.detect(dens, grid, shape), truth_xy)
    return tp, len(truth_xy), (np.median(errs) if errs else float("nan"))


def robustness(seeds=16, k=12, n_obj=6):
    """How much CORRECTION ERROR does the queryable map tolerate? A real loop
    closure applies an APPROXIMATE transform (residual translation + off-lattice
    rotation). We translate the map by the TRUE t but with an error e added, then
    query and measure chair recall vs |e| — the deploy tolerance ("how accurate
    must a graph correction be to keep the map queryable")."""
    print("\n  QUERY ROBUSTNESS to correction error (translate map by t, query "
          "at the TRUE shifted positions; chair recall vs added error):",
          flush=True)
    print(f"    {'trans-err(m)':>12} {'recall':>8}", flush=True)
    for te in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
        rs = []
        for s in range(seeds):
            codes = SM.class_codes(SM.CLASSES, k=k, m=SM.M_ROLES)
            objs = _scene(n_obj, s)
            chairs = [x[:2] for n, x in objs if n == "chair"]
            if not chairs:
                continue
            mp = SM.build_map(objs, codes)
            t = np.array([2.0, -1.0, 0.0])
            err = np.zeros(3)
            err[:2] = np.random.default_rng(s * 7).normal(0, te / np.sqrt(2), 2)
            mp_t = mp * np.exp(1j * (W @ (t + err)))       # imperfect correction
            truth = [c + t[:2] for c in chairs]            # query at TRUE shift
            grid, shape = SM._grid()
            dens = SM.query(mp_t, codes["chair"], grid)
            tp, fp, e = SM._match(SM.detect(dens, grid, shape), truth)
            rs.append(tp / len(chairs))
        print(f"    {te:>12.2f} {np.mean(rs):>8.2f}", flush=True)
    print("    => NOTE (rule-4 corrected): translation is LOSSLESS — the SSP "
          "peak is exact at ANY correction magnitude. This curve measures the "
          "QUERY's match window (semantic._match tol=0.8 m), not representational "
          "robustness: recall = fraction with |e| < 0.8 m. Honest bound: a loop "
          "correction must land within the ~0.8 m query match window; the "
          "representation adds no error at any correction size.", flush=True)


def run(seed=0):
    codes = SM.class_codes(SM.CLASSES, k=12)
    objs = _scene(seed=seed)
    mp = SM.build_map(objs, codes)
    c = objs[0][0]
    print(f"semantic-map TRANSFORM algebra (D={D}, {len(objs)} objects):\n")

    # TRANSLATION — bit-exact for any t
    t = np.array([2.0, -1.0, 0.0])
    mp_t = mp * np.exp(1j * (W @ t))
    err = np.abs(mp_t - SM.build_map([(n, x + t) for n, x in objs], codes)).max()
    tp, k, pe = _query_localize(mp_t, codes[c], [x[:2] + t[:2] for n, x in objs
                                                 if n == c])
    print(f"  TRANSLATION t={t.tolist()}: map⊙exp(iW·t) vs rebuild "
          f"max|err| {err:.1e} ({'EXACT' if err < 1e-9 else 'MISMATCH'}); "
          f"query '{c}' localises {tp}/{k} @ {pe:.2f} m in the shifted frame")

    # ROTATION — exact at lattice angles, approximate off-lattice
    for deg in (36.0, 30.0):
        th = np.deg2rad(deg)
        perm = SPH.yaw_perm(60, th, n_lams=6, **RINGKW)[0]
        Rz = np.array([[np.cos(th), -np.sin(th), 0],
                       [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        mp_r = mp[perm]
        reb = np.zeros(D, complex)
        for n, x in objs:
            reb += SM.ROLES[codes[n]][:, perm].sum(0) * np.exp(1j * ((Rz @ x) @ W.T))
        err = np.abs(mp_r - reb).max()
        tag = "EXACT (lattice angle)" if err < 1e-6 else \
              f"approx (off-lattice residual {err:.1f})"
        tp, k, pe = _query_localize(mp_r, codes[c], [(Rz @ x)[:2] for n, x in objs
                                                     if n == c], roles_perm=perm)
        print(f"  ROTATION {deg:.0f}°: map[perm_R] vs rebuild max|err| "
              f"{err:.1e} -> {tag}; query '{c}' localises {tp}/{k} @ {pe:.2f} m")
    print("\n  => a graph correction (translate + on-lattice rotate) moves the "
          "ENTIRE queryable map with one O(D) op; nothing is re-encoded.")


if __name__ == "__main__":
    run()
    robustness()
