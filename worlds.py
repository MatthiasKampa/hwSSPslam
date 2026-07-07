"""Environment collection for ssp_slam experiments.

All worlds keep the standard elliptical trajectory (make_trajectory), so only
make_environment is swapped. Each world is a function returning (S, 2, 2) wall
segments; check_world() asserts the path keeps clearance and scans get hits.
"""

import numpy as np

import ssp_slam as S

_ORIG_ENV = S.make_environment  # captured before any monkeypatching


def _rect(x0, y0, x1, y1, rot=0.0):
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    p = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], float)
    if rot:
        p = (p - c) @ S._rot(np.deg2rad(rot)).T + c
    return [[p[i], p[(i + 1) % 4]] for i in range(4)]


def _ellipse(cx, cy, a, b, n):
    t = np.linspace(0, 2 * np.pi, n + 1)
    p = np.stack([cx + a * np.cos(t), cy + b * np.sin(t)], 1)
    return [[p[i], p[i + 1]] for i in range(n)]


def _ring(cx, cy, r, n, ph=0.0):
    t = np.linspace(ph, ph + 2 * np.pi, n + 1)
    p = np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)
    return [[p[i], p[i + 1]] for i in range(n)]


def world_room():
    """The original flat-wall room."""
    return _ORIG_ENV()


def world_corridor():
    """Closed ring corridor around the path: curved walls ~parallel to motion,
    classic aperture-problem geometry."""
    segs = _ellipse(8, 5, 6.4, 3.4, 48) + _ellipse(8, 5, 3.4, 0.8, 40)
    segs += _rect(14.0, 4.6, 14.25, 5.4)     # wall notches breaking the aperture
    segs += _rect(1.75, 4.4, 2.0, 5.2)
    segs += _rect(7.6, 8.15, 8.4, 8.4)
    return np.array(segs, float)


def world_office():
    """Flat outer room + scattered 'furniture' boxes at mixed rotations."""
    segs = _rect(0, 0, 16, 10)
    for box in [(6.9, 4.4, 7.7, 5.2, 0), (8.6, 4.6, 9.6, 5.4, 37),
                (1.0, 1.0, 2.2, 2.0, 0), (13.5, 8.0, 14.7, 9.2, 15),
                (1.2, 8.2, 2.4, 9.0, 60), (14.0, 0.8, 15.2, 1.8, 75),
                (5.0, 8.8, 6.2, 9.6, 20), (10.5, 0.3, 11.7, 1.1, 50)]:
        segs += _rect(*box[:4], rot=box[4])
    segs.append([(0.0, 5.5), (1.6, 5.5)])    # short wall stubs
    segs.append([(16.0, 4.2), (14.6, 4.2)])
    return np.array(segs, float)


def world_sparse():
    """Big mostly-empty hall: many beams exceed max range, few features."""
    segs = _rect(-6, -5, 22, 15)
    segs += _rect(12.4, 1.4, 13.2, 2.2, 30)
    segs += _ring(2.6, 8.6, 0.7, 10)
    segs.append([(1.8, 1.2), (3.4, 0.6)])
    return np.array(segs, float)


def world_blob():
    """Irregular curved boundary + faceted pillars (no long straight walls)."""
    a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    r = 7.6 + 0.9 * np.sin(3 * a + 1.0) + 0.5 * np.cos(5 * a) + 0.35 * np.sin(7 * a + 2.0)
    p = np.stack([8.3 + r * np.cos(a), 5.2 + r * np.sin(a)], 1)
    segs = [[p[i], p[(i + 1) % 48]] for i in range(48)]
    segs += _ring(7.2, 5.1, 0.9, 12, 0.3) + _ring(3.5, 1.6, 0.9, 12, 1.1)
    segs += _ring(12.6, 8.2, 0.8, 12, 2.0)
    segs.append([(1.5, 7.0), (3.0, 8.2)])
    segs.append([(13.0, 2.0), (14.5, 1.2)])
    return np.array(segs, float)


def world_mixed():
    """Original flat room plus curved clutter: both spectral regimes at once."""
    segs = list(_ORIG_ENV())
    segs += _ring(12.9, 2.9, 0.7, 12, 0.4)
    segs += _rect(1.3, 1.2, 2.5, 2.2, 37)
    return np.array(segs, float)


WORLDS = {
    "room": world_room, "corridor": world_corridor, "office": world_office,
    "sparse": world_sparse, "blob": world_blob, "mixed": world_mixed,
}


def check_world(env_fn, min_clear=0.5):
    """Min path-to-wall distance (segment-aware) and mean scan hit fraction."""
    segs, gt = env_fn(), S.make_trajectory(140)
    a, b = segs[:, 0], segs[:, 1]
    e = b - a
    ee = (e * e).sum(1)
    dmin, hits = np.inf, []
    rng_save = S.RNG
    S.RNG = np.random.default_rng(0)
    for p in gt:
        t = np.clip(((p[:2] - a) * e).sum(1) / np.maximum(ee, 1e-12), 0, 1)
        d = np.linalg.norm(a + t[:, None] * e - p[:2], axis=1).min()
        dmin = min(dmin, d)
        r, _ = S.simulate_scan(segs, p)
        hits.append(np.isfinite(r).mean())
    S.RNG = rng_save
    assert dmin >= min_clear, f"clearance {dmin:.2f} < {min_clear}"
    return dmin, float(np.mean(hits))


if __name__ == "__main__":
    for name, fn in WORLDS.items():
        d, h = check_world(fn)
        print(f"{name:<9} clearance {d:.2f} m   hit fraction {h:.2f}")
