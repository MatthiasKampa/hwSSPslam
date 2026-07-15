"""P1 (msg.txt): WHY oct6 won the synthetic rooms but coarse16 wins buildings.

The deploy transfer gate found the sweep's "oct6 (0.25-8 m) is the best place
ladder" is VENUE-SCALED: on building-scale school_run2 the coarse (0.5-90.5 m)
rings win. This bench isolates the mechanism on the synthetic surface: sweep the
world EXTENT against the ladder's coarsest wavelength lam_max, fixed geometry
(ring_arc_const_s0.5_r6), rotation-searched lidar place AUC. Prediction: the
best lam_max tracks the venue extent (coarse rings must reach scene scale to
resolve global layout without the fine rings aliasing across a big room).

Deterministic synthetic (anti-oracle: no reference labels; same-place = two
jittered views of one layout, different-place = distinct layouts).

  python3 -m sspax.ladder_extent
"""
import numpy as np
import jax

from sspax import core as C
from sspax import sphere as SPH
from sspax import fusion as F
from sspax import worlds as Wd

RINGKW = dict(spacing="arc", apportion="const", stagger=0.5, n_rings=6)
EXTENTS = [8.0, 30.0, 80.0]           # room -> hall -> building
LAM_MAX = [8.0, 22.6, 90.5]           # oct6 tail -> mid -> coarse16 tail
K_PLACES = 24


def ladder(lam_max, n=6, lam_min=0.25):
    return list(np.geomspace(lam_min, lam_max, n))


def _places(extent, seed0, k=K_PLACES, n_pts=4000):
    """k distinct box layouts in a footprint of horizontal size `extent`
    (ceiling fixed ~3 m); 2 jittered views each (yaw +-40, small translation)."""
    rng = np.random.default_rng(seed0)
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


def run():
    print(f"jax devices: {jax.devices()}", flush=True)
    print("ladder-vs-world-extent: rotation-searched lidar place AUC, "
          f"ring_arc_const D=360, {K_PLACES} places/extent\n")
    header = "  extent \\ lam_max " + "".join(f"{lm:>10.1f}m" for lm in LAM_MAX)
    print(header)
    Ws = {lm: C.W_of(SPH.dirs_ring(60, **RINGKW), ladder(lm)) for lm in LAM_MAX}
    grid = {}
    for E in EXTENTS:
        places = _places(E, seed0=int(E))
        row = []
        for lm in LAM_MAX:
            auc, _, _ = F.place_auc(Ws[lm], places, mode="lidar", n_yaw=12)
            row.append(auc)
            grid[(E, lm)] = auc
        best = LAM_MAX[int(np.argmax(row))]
        print(f"  {E:6.0f} m        " + "".join(f"{a:11.3f}" for a in row)
              + f"   -> best lam_max {best:.1f} m", flush=True)
    print("\ndiagonal test (best lam_max should rise with extent):")
    for E in EXTENTS:
        row = [grid[(E, lm)] for lm in LAM_MAX]
        print(f"  extent {E:5.0f} m: argmax at lam_max="
              f"{LAM_MAX[int(np.argmax(row))]:.1f} m", flush=True)


if __name__ == "__main__":
    run()
