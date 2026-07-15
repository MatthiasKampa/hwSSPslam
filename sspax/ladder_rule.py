"""P5 (msg.txt round 2): turn the ladder_extent MECHANISM into a fitted DESIGN
RULE. ladder_extent.py showed the best coarsest wavelength lam_max tracks venue
extent on 3 points (8->8, 30->22.6, 80->90.5). Here we densify to 6 extents x a
finer lam_max grid, pick the argmax per extent, and FIT lam_max*(extent) so the
deploy side can install a bbox -> ladder preset (the static cousin of the learned
thermometer head). We also add an n_rings=8 row to check ring-count invariance of
the rule (the fit should not depend on how many rungs discretise the ladder).

Anti-oracle: deterministic synthetic, no reference labels (same-place = two
jittered views of one layout; different-place = distinct layouts at same extent).
Fixed seeds; the fit is reported with its R^2 so a weak rule cannot masquerade as
a strong one.

  python3 -m sspax.ladder_rule
"""
import numpy as np
import jax

from sspax import core as C
from sspax import sphere as SPH
from sspax import fusion as F
from sspax import worlds as Wd

# 6 extents room -> building; finer lam_max grid to localise the argmax.
EXTENTS = [8.0, 16.0, 30.0, 50.0, 80.0, 120.0]
LAM_GRID = list(np.geomspace(4.0, 200.0, 9))          # candidate coarsest rungs
K_PLACES = 32
N_PTS = 4000
BASE_KW = dict(spacing="arc", apportion="const", stagger=0.5)


def ladder(lam_max, n=6, lam_min=0.25):
    return list(np.geomspace(lam_min, lam_max, n))


def _places(extent, seed0, k=K_PLACES, n_pts=N_PTS):
    """k distinct box layouts at horizontal footprint `extent`; 2 jittered
    views each (yaw +-40 deg, small translation, 2 cm noise)."""
    places = []
    for i in range(k):
        base = Wd.room(n_pts * 2, seed=1000 + i,
                       dims=(extent, 0.75 * extent, 3.0), n_boxes=10)
        views = []
        for v in range(2):
            rr = np.random.default_rng(seed0 * 7 + 3 * i + v)
            R = C._axis_rot("yaw", rr.uniform(-40, 40))
            t = rr.uniform(-0.3, 0.3, 3) * np.array([1, 1, 0.3])
            idx = rr.permutation(len(base))[:n_pts]
            pts = base[idx] @ R.T + t + rr.normal(0, 0.02, (n_pts, 3))
            views.append((pts, None, None))
        places.append(views)
    return places


def _fit_through_origin(x, y):
    """least-squares slope c for y ~ c*x (no intercept) + R^2."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    c = float((x @ y) / (x @ x))
    yhat = c * x
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return c, r2


def _sweep(n_rings, n_dir=60):
    """best lam_max per extent for a given ring count; returns (best[], grid)."""
    dirs = SPH.dirs_ring(n_dir, n_rings=n_rings, **BASE_KW)
    Ws = {lm: C.W_of(dirs, ladder(lm, n=n_rings)) for lm in LAM_GRID}
    best, grid = [], {}
    header = "  extent \\ lam_max " + "".join(f"{lm:>8.0f}" for lm in LAM_GRID)
    print(header, flush=True)
    for E in EXTENTS:
        places = _places(E, seed0=int(E) + n_rings)
        row = []
        for lm in LAM_GRID:
            auc, _, _ = F.place_auc(Ws[lm], places, mode="lidar", n_yaw=12)
            row.append(auc)
            grid[(E, lm)] = auc
        blm = LAM_GRID[int(np.argmax(row))]
        best.append(blm)
        star = ["*" if abs(lm - blm) < 1e-9 else " " for lm in LAM_GRID]
        print(f"  {E:6.0f} m        "
              + "".join(f"{a:7.3f}{s}" for a, s in zip(row, star))
              + f"  -> {blm:.0f} m", flush=True)
    return best, grid


def run():
    print(f"jax devices: {jax.devices()}", flush=True)
    print(f"ladder DESIGN RULE: {len(EXTENTS)} extents x {len(LAM_GRID)} "
          f"lam_max, ring_arc_const, {K_PLACES} places/extent, "
          f"rotation-searched lidar place AUC\n", flush=True)

    print("=== n_rings=6 ===", flush=True)
    best6, _ = _sweep(6)
    c6, r6 = _fit_through_origin(EXTENTS, best6)
    print(f"\n  FIT (n_rings=6): lam_max ~ {c6:.3f} * extent   R^2={r6:.3f}",
          flush=True)

    print("\n=== n_rings=8 (ring-count invariance check) ===", flush=True)
    best8, _ = _sweep(8)
    c8, r8 = _fit_through_origin(EXTENTS, best8)
    print(f"\n  FIT (n_rings=8): lam_max ~ {c8:.3f} * extent   R^2={r8:.3f}",
          flush=True)

    print("\n=== DESIGN RULE SUMMARY ===", flush=True)
    print(f"  n_rings=6: lam_max = {c6:.2f} * bbox_extent   (R^2 {r6:.3f})",
          flush=True)
    print(f"  n_rings=8: lam_max = {c8:.2f} * bbox_extent   (R^2 {r8:.3f})",
          flush=True)
    print(f"  ring-count invariance: |c6-c8|/c6 = "
          f"{abs(c6 - c8) / c6 * 100:.1f}%", flush=True)
    print(f"  => deploy: ladder = geomspace(0.25, {c6:.2f}*extent, n_rings)",
          flush=True)


if __name__ == "__main__":
    run()
