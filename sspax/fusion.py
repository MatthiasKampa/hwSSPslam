"""Camera + lidar + combined encodings and the place-separability metric.

Fusion is a VSA bundle on a SHARED lattice: v = a*phi_cam + (1-a)*phi_lidar,
each channel L2-normalized first (else lidar's point count dominates). Same D as
one channel => fusion is memory-free (vs concat, 2D). One rotation permutation
rotates the bundle. The place metric is the SLAM-relevant one: can two views of
the SAME place be told from DIFFERENT places, under a yaw-rotation search?
"""
import numpy as np
import jax.numpy as jnp

from sspax import core as C
from sspax import worlds as Wd


# --------------------------------------------------------------------------
#  encodings
# --------------------------------------------------------------------------
def enc_lidar(W, pts, quant=None):
    return C.encode(W, pts, quant=quant)


def enc_cam(W, bearings, w, quant=None):
    return C.encode(W, bearings, w=w, quant=quant)


def enc_fused(W, pts, bearings, wb, alpha=0.5, quant=None, concat=False):
    """alpha in [0,1]: weight on the camera channel. concat=True stacks the two
    channels (2D) instead of bundling (D)."""
    vl = C.encode(W, pts, quant=quant)
    vc = C.encode(W, bearings, w=wb, quant=quant)
    if concat:
        return jnp.concatenate([np.sqrt(1 - alpha) * vl,
                                np.sqrt(alpha) * vc])
    v = (1 - alpha) * vl + alpha * vc
    return v / jnp.maximum(jnp.linalg.norm(v), 1e-12)


# --------------------------------------------------------------------------
#  place-recognition data: K places x V jittered views (lidar + camera)
# --------------------------------------------------------------------------
def gen_places(n_places=24, n_views=2, n_pts=3000, n_bear=400, seed0=0,
               yaw_jit=60.0, t_jit=0.3, noise=0.02, fov=(360.0, 120.0)):
    """For each place (a distinct room), n_views observations differing by a
    yaw (up to +-yaw_jit deg), a small translation, an independent point
    subsample and sensor noise — 'same place, different viewpoint'. Different
    places are different rooms. fov=(360,120) is an omni camera (rotation-
    equivariant, favourable to the sphere lattice); a narrow fov is the hard
    realistic case. Camera look is fixed (+x) so the yaw jitter IS the heading
    change the rotation search must undo. Returns places[k][v]=(pts,b,wb)."""
    rng = np.random.default_rng(seed0)
    places = []
    for k in range(n_places):
        base = Wd.room(n_pts * 2, seed=1000 + k,
                       dims=(6 + (k % 4), 5 + (k % 3), 2.5 + 0.5 * (k % 3)))
        views = []
        for v in range(n_views):
            yaw = rng.uniform(-yaw_jit, yaw_jit)
            R = C._axis_rot("yaw", yaw)
            t = rng.uniform(-t_jit, t_jit, 3) * np.array([1, 1, 0.3])
            idx = rng.permutation(len(base))[:n_pts]
            pts = base[idx] @ R.T + t + rng.normal(0, noise, (n_pts, 3))
            b, wb = Wd.camera_bearings(pts, n_bear, fov_deg=fov, cam_pos=t,
                                       look=np.array([1.0, 0.0, 0.0]),
                                       seed=seed0 + 7 * k + v)
            views.append((pts, b, wb))
        places.append(views)
    return places


def _yaw_perm_set(W, n=12):
    """Rotation-search permutations for n yaw angles over [0, pi)."""
    perms, sgns = [], []
    for k in range(n):
        R = C._axis_rot("yaw", 180.0 * k / n)
        p, s, _, _ = C.perm_of(np.asarray(W), R)
        perms.append(jnp.asarray(p))
        sgns.append(jnp.asarray(s))
    return perms, sgns


def _auc(pos, neg):
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x)
    r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1)
    return (r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) \
        / max(len(pos) * len(neg), 1)


def place_auc(W, places, mode="fused", alpha=0.5, quant=None, concat=False,
              rot_search=True, n_yaw=12):
    """Same-place-vs-different-place AUC of the (rotation-searched) cosine
    similarity. mode in {lidar, cam, fused}."""
    K, V = len(places), len(places[0])
    Wj = jnp.asarray(W)

    def enc(view):
        pts, b, wb = view
        if mode == "lidar":
            return enc_lidar(Wj, pts, quant)
        if mode == "cam":
            return enc_cam(Wj, b, wb, quant)
        return enc_fused(Wj, pts, b, wb, alpha, quant, concat)

    Vs = [[enc(v) for v in place] for place in places]
    Wrot = W if not concat else np.concatenate([W, W])
    perms, sgns = _yaw_perm_set(Wrot, n_yaw) if rot_search else (None, None)

    def sim(a, b):
        if not rot_search:
            return float(jnp.abs(jnp.vdot(a, b)))
        best = 0.0
        for p, s in zip(perms, sgns):
            best = max(best, float(jnp.abs(jnp.vdot(C.apply_perm(a, p, s), b))))
        return best

    pos, neg = [], []
    for k in range(K):
        pos.append(sim(Vs[k][0], Vs[k][1]))          # two views, same place
        for j in range(K):
            if j != k:
                neg.append(sim(Vs[k][0], Vs[j][0]))   # different places
    return _auc(np.array(pos), np.array(neg)), len(pos), len(neg)
