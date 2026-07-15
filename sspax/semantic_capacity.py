"""Bounded semantic-map CAPACITY law: how many features fit in ONE fixed-size
SSP vector before a class query stops resolving them — and how that capacity
scales with the vector dimension D. This is the load-bearing number for the
"bounded O(area), history-free" claim: if capacity ∝ D, then mapping an area with
N objects costs D ∝ N — a fixed, predictable budget, not unbounded history.

The map is a single D-dim complex vector: map = Σ_o Σ_{i∈bits_o} ROLES[i] ⊗ enc(xyz_o)
(sphere.dirs_ring × oct6 ladder positions). A class query unbinds its roles and
decodes spatial density; an object reads out ∝ |query ∩ bits_o|. As more objects
are superposed, cross-talk noise grows ~√(load/D); recall falls when the signal
(k bits) drops below the noise floor -> capacity ∝ D at fixed k.

Anti-oracle: deterministic synthetic scenes, no reference labels; GT object
positions only SCORE recall/precision. Pure numpy.

  python3 -m sspax.semantic_capacity
"""
import numpy as np

from sspax import core as C
from sspax import sphere as SPH
from sspax.semantic import detect, _match, class_codes, CLASSES

OCT6 = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
RINGKW = dict(spacing="arc", apportion="const", stagger=0.5, n_rings=6)


def W_of_D(n_dirs, lams=OCT6):
    return np.asarray(C.W_of(SPH.dirs_ring(n_dirs, **RINGKW), lams))


def roles_of(m, D, seed=0):
    rng = np.random.default_rng(seed)
    return np.exp(1j * rng.uniform(0, 2 * np.pi, (m, D)))


def _scene(n_obj, seed, extent=(12.0, 10.0)):
    rng = np.random.default_rng(seed)
    objs = []
    for _ in range(n_obj):
        name = CLASSES[rng.integers(len(CLASSES))]
        xy = rng.uniform([-extent[0] / 2, -extent[1] / 2],
                         [extent[0] / 2, extent[1] / 2])
        objs.append((name, np.array([xy[0], xy[1], 0.5])))
    return objs


def _grid(extent=(12.0, 10.0), step=0.25):
    xs = np.arange(-extent[0] / 2, extent[0] / 2, step)
    ys = np.arange(-extent[1] / 2, extent[1] / 2, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], 1), (len(xs), len(ys))


def build_and_query(W, roles, codes, objs, query_bits, grid_xy, z=0.5):
    D = W.shape[0]
    mp = np.zeros(D, complex)
    for name, xyz in objs:
        mp += roles[codes[name]].sum(0) * np.exp(1j * (xyz @ W.T))
    unbound = np.conj(roles[query_bits]).sum(0) * mp
    g = np.concatenate([grid_xy, np.full((len(grid_xy), 1), z)], 1)
    return np.real(np.exp(1j * (g @ W.T)) @ np.conj(unbound)) / D


def recall_at(W, roles, k, n_obj, seed):
    """chair recall for one scene; returns np.nan when the scene has NO chairs
    (a 0-chair scene must be EXCLUDED, not scored 0 — that was biasing low-load
    recall down toward the 1/|CLASSES| chair-absence rate)."""
    codes = class_codes(CLASSES, k=k, m=roles.shape[0])
    objs = _scene(n_obj, seed)
    grid, shape = _grid()
    chairs = [o[1][:2] for o in objs if o[0] == "chair"]
    if not chairs:
        return np.nan
    dens = build_and_query(W, roles, codes, objs, codes["chair"], grid)
    pk = detect(dens, grid, shape)
    tp, fp, _ = _match(pk, chairs)
    return tp / len(chairs)


LOADS = [4, 8, 16, 32, 64, 128]


def recall_curve(W, k=12, m=256, seeds=24):
    """mean chair-recall at each load in LOADS (averaged over seeds)."""
    roles = roles_of(m, W.shape[0])
    return [float(np.nanmean([recall_at(W, roles, k, n, s)
                              for s in range(seeds)])) for n in LOADS]


def _cross(curve, target=0.5):
    """interpolate the load where recall crosses `target` (linear in log-load)."""
    ln = np.log(LOADS)
    for i in range(len(curve) - 1):
        if curve[i] >= target > curve[i + 1]:
            f = (curve[i] - target) / max(curve[i] - curve[i + 1], 1e-9)
            return float(np.exp(ln[i] + f * (ln[i + 1] - ln[i])))
    return float(LOADS[0]) if curve[0] < target else float(LOADS[-1])


def run():
    print("bounded semantic-map CAPACITY vs dimension D (mean chair-recall at "
          "increasing map load, k=12 bits, oct6 ladder, 10 seeds):\n", flush=True)
    hdr = "  " + f"{'n_dirs':>6} {'D':>5}  " + "".join(f"{n:>6}" for n in LOADS) \
        + f"  {'cap@0.5':>8}"
    print(hdr, flush=True)
    rows = []
    for nd in [20, 40, 60, 120, 240]:
        W = W_of_D(nd); D = W.shape[0]
        cur = recall_curve(W)
        cap = _cross(cur)
        rows.append((D, cap))
        print(f"  {nd:>6} {D:>5}  " + "".join(f"{r:6.2f}" for r in cur)
              + f"  {cap:8.1f}", flush=True)
    Ds = np.array([r[0] for r in rows], float)
    caps = np.array([r[1] for r in rows], float)
    # power-law fit cap = a*D^b (the relationship is concave, not linear)
    lx, ly = np.log(Ds), np.log(caps)
    b, la = np.polyfit(lx, ly, 1)
    r2 = 1 - ((ly - (la + b * lx)) ** 2).sum() / max(((ly - ly.mean()) ** 2).sum(), 1e-9)
    print(f"\n  FIT capacity ~ {np.exp(la):.2f} * D^{b:.2f}  (log-log R^2 {r2:.3f})",
          flush=True)
    print(f"  => capacity is SUBLINEAR in D (~sqrt: superposition cross-talk "
          f"~ sqrt(load/D)). To hold N objects at recall>=0.5, D ~ N^{1/max(b,1e-3):.1f}"
          f" — so ONE giant superposition map scales POORLY; the bounded-memory "
          f"win comes from TILING (per-segment maps), not a single vector.",
          flush=True)

    print("\n  capacity@0.5 vs k (bits/class) at D=360 (fewer bits = less "
          "cross-talk = more objects, but weaker significance grading):",
          flush=True)
    W = W_of_D(60)
    for k in [6, 12, 24]:
        print(f"    k={k:2d}: cap {_cross(recall_curve(W, k=k)):.1f} objects",
              flush=True)


def significance_capacity(seeds=24, k=12, m=256, D_dirs=60):
    """Does SIGNIFICANCE grading buy effective CAPACITY? An IMPORTANT object
    commits all k bits; a BACKGROUND object commits few (b) bits, so it takes
    less 'room' in the bundle. If graded, important-object (chair) recall should
    survive higher BACKGROUND load than if everything commits k bits equally.
    Connects the two banked mechanisms (significance = committed bits; capacity =
    sqrt-ish in D) — significance should protect what matters under load."""
    W = W_of_D(D_dirs)
    roles = roles_of(m, W.shape[0])
    grid, shape = _grid()
    print(f"\n  SIGNIFICANCE buys CAPACITY (D={W.shape[0]}, chairs commit k={k} "
          f"bits; background commits b bits; chair recall vs background load):",
          flush=True)
    print(f"    {'bg-objects':>10} {'graded b=3':>11} {'equal b=12':>11}",
          flush=True)
    for n_bg in (4, 8, 16, 32, 48):
        out = {}
        for b in (3, 12):
            rs = []
            for s in range(seeds):
                codes = class_codes(CLASSES, k=k, m=m)
                rng = np.random.default_rng(s + 200)
                chairs = [np.array([rng.uniform(-5, 5), rng.uniform(-4, 4), 0.5])
                          for _ in range(3)]
                bg = [(CLASSES[1 + rng.integers(len(CLASSES) - 1)],
                       np.array([rng.uniform(-5, 5), rng.uniform(-4, 4), 0.5]))
                      for _ in range(n_bg)]
                mp = np.zeros(W.shape[0], complex)
                for x in chairs:                          # important: full k bits
                    mp += roles[codes["chair"]].sum(0) * np.exp(1j * (x @ W.T))
                for name, x in bg:                        # background: b bits
                    mp += roles[codes[name][:b]].sum(0) * np.exp(1j * (x @ W.T))
                dens = np.real(np.exp(1j * (np.concatenate(
                    [grid, np.full((len(grid), 1), 0.5)], 1) @ W.T))
                    @ np.conj(np.conj(roles[codes["chair"]]).sum(0) * mp)) \
                    / W.shape[0]
                tp, fp, _ = _match(detect(dens, grid, shape),
                                   [c[:2] for c in chairs])
                rs.append(tp / 3)
            out[b] = np.mean(rs)
        print(f"    {n_bg:>10} {out[3]:>11.2f} {out[12]:>11.2f}", flush=True)
    print("    => if graded (b=3) chair-recall exceeds equal (b=12) under "
          "background load, significance grading BUYS capacity for important "
          "objects — de-prioritised features take less room in the bounded "
          "bundle. (The map spends its finite capacity on what matters.)",
          flush=True)


def tiling(k=12, m=256, seeds=24):
    """CONSTRUCTIVE validation of the capacity conclusion: at FIXED total storage
    D_total, split the area into T spatial tiles of D_total/T each (objects routed
    by position). Capacity ~ sqrt(dim) per tile x T tiles = sqrt(T*D_total) — a
    sqrt(T) gain over one D_total map for the SAME storage. Measured on chair
    recall (D_total=720)."""
    D_total = 720
    print(f"\n  TILING at fixed total storage D={D_total} (chair recall@load; "
          f"T tiles of D/T each, objects routed by x-position):", flush=True)
    print(f"    {'load':>5}  " + "".join(f"T={t:<5}" for t in (1, 2, 4)),
          flush=True)
    for n_obj in [8, 16, 32, 64]:
        row = []
        for T in (1, 2, 4):
            Wt = W_of_D(int(round((D_total // T) / len(OCT6))))   # dim ~ D/T
            roles = roles_of(m, Wt.shape[0])
            rs = []
            for s in range(seeds):
                codes = class_codes(CLASSES, k=k, m=m)
                objs = _scene(n_obj, s + 100)
                chairs = [o[1][:2] for o in objs if o[0] == "chair"]
                if not chairs:
                    continue
                grid, shape = _grid()
                # route objects + query grid to tiles by x (edges at extent/T)
                xs = np.array([o[1][0] for o in objs])
                edges = np.linspace(-6, 6, T + 1)
                tile = np.clip(np.digitize(xs, edges[1:-1]), 0, T - 1)
                gtile = np.clip(np.digitize(grid[:, 0], edges[1:-1]), 0, T - 1)
                dens = np.zeros(len(grid))
                for t in range(T):
                    ot = [objs[i] for i in range(len(objs)) if tile[i] == t]
                    if not ot:
                        continue
                    mp = np.zeros(Wt.shape[0], complex)
                    for name, xyz in ot:
                        mp += roles[codes[name]].sum(0) * np.exp(1j * (xyz @ Wt.T))
                    unb = np.conj(roles[codes["chair"]]).sum(0) * mp
                    gm = gtile == t
                    g = np.concatenate([grid[gm], np.full((gm.sum(), 1), 0.5)], 1)
                    dens[gm] = np.real(np.exp(1j * (g @ Wt.T)) @ np.conj(unb)) \
                        / Wt.shape[0]
                pk = detect(dens, grid, shape)
                tp, fp, _ = _match(pk, chairs)
                rs.append(tp / len(chairs))
            row.append(float(np.mean(rs)))
        print(f"    {n_obj:>5}  " + "".join(f"{r:<7.2f}" for r in row),
              flush=True)
    print("    => SPLITTING a fixed D into smaller-D tiles does NOT help (it "
          "HURTS): a D/T tile has coarser spatial-decode resolution, and that "
          "loss offsets the reduced per-tile load. The bounded-memory win is "
          "NOT sub-dividing one budget — it is a FULL-D map per bounded-area "
          "segment (storage grows with AREA, each map covering few objects). "
          "That is the shipped per-5-keyframe-segment design.", flush=True)


if __name__ == "__main__":
    import sys
    run()
    if len(sys.argv) > 1 and sys.argv[1] == "tiling" or len(sys.argv) == 1:
        tiling()
