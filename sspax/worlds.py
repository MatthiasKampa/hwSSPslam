"""Deterministic synthetic 3D scenes for the lattice-algebra benches.

The 6-DoF studies here are about the *representation's algebra* (permutation-vs-
re-encode fidelity, decode error), which is data-agnostic — so we generate
fixed-size structured clouds rather than depend on the (un-fetched) school /
spot 3D parquets. Fixed size => the whole batch encodes in one vmapped call.
No wall-clock, no global RNG: every scene is seeded.
"""
import numpy as np


def room(n_pts=3000, seed=0, dims=(8.0, 6.0, 3.0), n_boxes=5):
    """A box room (6 walls) with a few internal box obstacles, sampled to
    exactly n_pts points centred near the origin. Returns (n_pts, 3)."""
    rng = np.random.default_rng(seed)
    lx, ly, lz = dims
    pts = []

    def wall(fixed_axis, val, a_rng, b_rng, m):
        a = rng.uniform(*a_rng, m)
        b = rng.uniform(*b_rng, m)
        col = np.full(m, val)
        if fixed_axis == 0:
            return np.stack([col, a, b], 1)
        if fixed_axis == 1:
            return np.stack([a, col, b], 1)
        return np.stack([a, b, col], 1)

    m = n_pts // 6
    ax, ay, az = (-lx / 2, lx / 2), (-ly / 2, ly / 2), (0.0, lz)
    pts += [wall(0, -lx / 2, ay, az, m), wall(0, lx / 2, ay, az, m)]
    pts += [wall(1, -ly / 2, ax, az, m), wall(1, ly / 2, ax, az, m)]
    pts += [wall(2, 0.0, ax, ay, m), wall(2, lz, ax, ay, m)]
    # internal box clutter
    for _ in range(n_boxes):
        c = rng.uniform([-lx / 3, -ly / 3, 0.3], [lx / 3, ly / 3, lz - 0.5])
        s = rng.uniform(0.3, 0.9, 3)
        q = c + rng.uniform(-s / 2, s / 2, (m // 2, 3))
        pts.append(q)
    P = np.concatenate(pts)
    # centre so rotations are about the scene, subsample to exact n_pts
    P = P - P.mean(0)
    idx = rng.permutation(len(P))[:n_pts]
    return P[idx].astype(np.float64)


def rooms(n_scenes=12, n_pts=3000, seed0=0, **kw):
    """A deterministic batch of distinct rooms, (n_scenes, n_pts, 3)."""
    return np.stack([room(n_pts, seed=seed0 + i,
                          dims=(6 + (i % 4), 5 + (i % 3), 2.5 + 0.5 * (i % 3)),
                          **kw)
                     for i in range(n_scenes)])


# --------------------------------------------------------------------------
#  synthetic pinhole camera -> bearing vectors on S^2
# --------------------------------------------------------------------------
def camera_bearings(cloud, n_out=400, fov_deg=(90.0, 60.0), cam_pos=None,
                    look=None, seed=0):
    """Unit bearing vectors (world frame) of the cloud points visible to a
    pinhole camera at cam_pos looking along `look`, within the horizontal x
    vertical FOV. Returns (n_out, 3) bearings + (n_out,) weights, resampled to
    EXACTLY n_out (vmap-friendly). Camera rotation R acts on bearings exactly
    as on lidar points (B @ R.T), so the rotation algebra is shared."""
    rng = np.random.default_rng(seed)
    cloud = np.asarray(cloud, float)
    if cam_pos is None:
        cam_pos = rng.uniform(-0.5, 0.5, 3)
    if look is None:
        a = rng.uniform(0, 2 * np.pi)
        look = np.array([np.cos(a), np.sin(a), 0.0])
    look = look / np.linalg.norm(look)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(look, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, look)
    v = cloud - cam_pos
    rng_all = np.linalg.norm(v, axis=1) + 1e-9
    omni = fov_deg[0] >= 359                   # spherical camera: keep all rays
    if omni:
        elev = np.arcsin(np.clip((v @ true_up) / rng_all, -1, 1))
        keep = np.abs(elev) < np.deg2rad(fov_deg[1] / 2) if fov_deg[1] < 179 \
            else np.ones(len(v), bool)
    else:
        depth = v @ look
        front = depth > 0.3
        ah = np.abs(np.arctan2(v @ right, np.maximum(depth, 1e-9)))
        av = np.abs(np.arctan2(v @ true_up, np.maximum(depth, 1e-9)))
        keep = front & (ah < np.deg2rad(fov_deg[0] / 2)) \
            & (av < np.deg2rad(fov_deg[1] / 2))
    v = v[keep]
    b = v / np.linalg.norm(v, axis=1, keepdims=True)
    w = 1.0 / np.sqrt(rng_all[keep])          # nearer features weigh more
    if len(b) == 0:                           # degenerate view: look at centroid
        return camera_bearings(cloud, n_out, fov_deg, cam_pos,
                               cloud.mean(0) - cam_pos, seed + 1)
    idx = rng.integers(0, len(b), n_out)      # resample to fixed size
    return b[idx], w[idx]


def bearing_batch(n_scenes=16, n_out=400, seed0=0, **kw):
    """Camera bearings for a batch of rooms, (n_scenes, n_out, 3) + weights."""
    B, Wt = [], []
    for i in range(n_scenes):
        r = room(3000, seed=seed0 + i,
                 dims=(6 + (i % 4), 5 + (i % 3), 2.5 + 0.5 * (i % 3)))
        b, w = camera_bearings(r, n_out, seed=seed0 + i, **kw)
        B.append(b)
        Wt.append(w)
    return np.stack(B), np.stack(Wt)
