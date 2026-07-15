"""Learned, FPGA-fittable lidar pre-processing — a differentiable saliency
detector trained END-TO-END for SSP place-recognition / registration.

Pipeline (user directive 2026-07-15): rasterize the scan to a small BEV grid ->
a tiny CNN scores per-cell saliency on a 2x-downsampled map -> threshold to a
SPARSE set of salient cells -> encode those (weighted) into the SSP on the swept
winner lattice. Because the SSP encode `sum_i w_i exp(i W.p_i)` is differentiable
in the point positions AND weights, a contrastive place-rec loss backprops
through the encoder into the CNN: the net learns to keep the repeatable,
discriminative structure and drop the rest.

FPGA fit: the net is 3 small convs (int8-quantizable); saliency threshold makes
the downstream encode CHEAPER (fewer, sparser features). Trained on synthetic
rooms (anti-oracle: GT relative poses only align the contrastive targets — they
never seed the encoder). Compared against uniform random subsampling at equal
feature budget.

  PYTHONPATH=. python3 -m sspax.learn_lidar train      # train + eval vs baseline
"""
import sys
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from sspax.nnconv import Conv
import optax

from sspax import core as C
from sspax import sphere as SPH
from sspax import worlds as Wd

# swept winner lattice: permutable const-ring + wide oct6 ladder
OCT6 = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
W_LAT = jnp.asarray(C.W_of(SPH.dirs_ring(60, spacing="arc", apportion="const",
                                         stagger=0.5, n_rings=6), OCT6))
R_EXT = 6.0          # BEV half-extent (m)
G = 48               # BEV grid resolution
GD = G // 2          # saliency map is 2x downsampled


# --------------------------------------------------------------------------
#  BEV rasterization (fixed, given a cloud) + cell geometry
# --------------------------------------------------------------------------
def rasterize(cloud, r=R_EXT, g=G):
    """cloud (M,3) -> (g,g,3) grid: [log-occupancy, max-height, mean-height]."""
    xy = cloud[:, :2]
    ix = np.clip(((xy[:, 0] + r) / (2 * r) * g).astype(int), 0, g - 1)
    iy = np.clip(((xy[:, 1] + r) / (2 * r) * g).astype(int), 0, g - 1)
    occ = np.zeros((g, g))
    hmax = np.full((g, g), -1e3)
    hsum = np.zeros((g, g))
    np.add.at(occ, (ix, iy), 1.0)
    np.maximum.at(hmax, (ix, iy), cloud[:, 2])
    np.add.at(hsum, (ix, iy), cloud[:, 2])
    hmax[hmax < -1e2] = 0.0
    hmean = np.where(occ > 0, hsum / np.maximum(occ, 1), 0.0)
    return np.stack([np.log1p(occ), hmax, hmean], -1).astype(np.float64)


def _cell_centers(gd=GD, r=R_EXT):
    c = (np.arange(gd) + 0.5) / gd * (2 * r) - r
    cx, cy = np.meshgrid(c, c, indexing="ij")
    return np.stack([cx.ravel(), cy.ravel()], -1)     # (gd*gd, 2)


CELLS = jnp.asarray(_cell_centers())


# --------------------------------------------------------------------------
#  tiny saliency CNN (FPGA-sized) + differentiable SSP encode
# --------------------------------------------------------------------------
class SaliencyNet(nn.Module):
    ch: int = 8

    @nn.compact
    def __call__(self, grid):                          # grid (B,G,G,3)
        x = nn.relu(Conv(self.ch, (3, 3))(grid))
        x = nn.relu(Conv(self.ch, (3, 3), strides=(2, 2))(x))  # 2x down
        sal = Conv(1, (1, 1))(x)[..., 0]            # (B,GD,GD) logits
        zc = Conv(1, (1, 1))(x)[..., 0]             # learned cell height
        return sal, zc


def encode_learned(params, model, grid, temp=1.0):
    """Differentiable SSP vector from the learned saliency (soft-weighted cells)."""
    sal, zc = model.apply(params, grid)                # (B,GD,GD)
    B = grid.shape[0]
    w = jax.nn.sigmoid(sal / temp).reshape(B, -1)      # (B, GD*GD) soft weights
    z = zc.reshape(B, -1)
    pts = jnp.concatenate([jnp.broadcast_to(CELLS, (B,) + CELLS.shape),
                           z[..., None]], -1)          # (B, N, 3)
    ph = jnp.exp(1j * jnp.einsum("bnd,kd->bnk", pts, W_LAT))   # (B,N,D)
    v = jnp.einsum("bnk,bn->bk", ph, w.astype(ph.dtype))
    return v / jnp.maximum(jnp.linalg.norm(v, axis=1, keepdims=True), 1e-12)


# --------------------------------------------------------------------------
#  training data: same-place view pairs (aligned by known pose) + negatives
# --------------------------------------------------------------------------
# fixed room footprint so places differ only by their INTERNAL structure
# (object layout) -> place-rec must use discriminative features, not gross size.
DIMS = (8.0, 6.0, 3.0)
N_CLUTTER = 700          # non-repeatable interior junk each view must learn to drop


def _view(base, vseed, clutter=True):
    rr = np.random.default_rng(vseed)
    yaw = rr.uniform(-40, 40)
    R = C._axis_rot("yaw", yaw)
    t = rr.uniform(-0.3, 0.3, 3) * np.array([1, 1, 0.3])
    idx = rr.permutation(len(base))[:2600]
    pts = base[idx] @ R.T + t + rr.normal(0, 0.02, (len(idx), 3))
    if clutter:                                   # random INTERIOR points (moving
        cl = rr.uniform([-3.5, -2.5, 0.2], [3.5, 2.5, 2.6], (N_CLUTTER, 3))
        cl = cl @ R.T + t
        pts = np.concatenate([pts, cl])
    aligned = (pts - t) @ R                        # GT pose only aligns the target
    return rasterize(aligned), aligned


def _scene(seed):
    # distinct object layout per place; identical footprint
    return Wd.room(5200, seed=1000 + seed, dims=DIMS, n_boxes=8)


def make_batch(n_scenes, seed0):
    A, Bv = [], []
    for i in range(n_scenes):
        base = _scene(seed0 + i)
        A.append(_view(base, (seed0 + i) * 2 + 1)[0])
        Bv.append(_view(base, (seed0 + i) * 2 + 2)[0])
    return jnp.asarray(np.stack(A)), jnp.asarray(np.stack(Bv))


def info_nce(va, vb, tau=0.1):
    """Same-index pairs positive, all others negative (rotation-aligned)."""
    S = jnp.abs(va @ vb.conj().T)                      # (B,B) cosine
    logits = S / tau
    labels = jnp.arange(len(va))
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


# --------------------------------------------------------------------------
#  train + eval
# --------------------------------------------------------------------------
def _auc(pos, neg):
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x)
    r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1)
    return (r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) \
        / max(len(pos) * len(neg), 1)


def _yaw_perms(nyaw=12):
    ps, ss = [], []
    for j in range(nyaw):
        p, s, _, _ = C.perm_of(np.asarray(W_LAT), C._axis_rot("yaw", 180.0 * j / nyaw))
        ps.append(jnp.asarray(p)); ss.append(jnp.asarray(s))
    return ps, ss


def _sparse_encode_learned(params, model, grid, k):
    """Encode ONLY the top-k salient cells (the deployed sparse path)."""
    sal, zc = model.apply(params, grid[None])
    sal, zc = np.asarray(sal[0]).ravel(), np.asarray(zc[0]).ravel()
    top = np.argsort(-sal)[:k]
    pts = np.concatenate([np.asarray(CELLS)[top], zc[top, None]], 1)
    w = jax.nn.sigmoid(jnp.asarray(sal[top]))
    return C.encode(W_LAT, pts, w=w)


def evaluate(params, model, seed0=9000, n=24, k=32):
    """Place-rec AUC at a TIGHT k-feature budget, rotation-invariant matching:
    learned top-k salient cells vs uniform random k raw points."""
    vL, vU = [], []
    for grp in (0, 1):
        for i in range(n):
            base = _scene(seed0 + i)
            grid, cloud = _view(base, seed0 * 5 + i + grp * 99)
            vL.append(_sparse_encode_learned(params, model, jnp.asarray(grid), k))
            sub = cloud[np.random.default_rng(seed0 + i + grp).permutation(
                len(cloud))[:k]]
            vU.append(C.encode(W_LAT, sub))
    vL = jnp.stack(vL); vU = jnp.stack(vU)
    ps, ss = _yaw_perms()

    def auc(V):
        A_, B_ = V[:n], V[n:]
        S = np.zeros((n, n))
        for p, s in zip(ps, ss):
            Ap = C.apply_perm(A_, p, s)
            S = np.maximum(S, np.abs(np.asarray(Ap @ B_.conj().T)))
        pos = np.diag(S); neg = S[~np.eye(n, dtype=bool)]
        return _auc(pos, neg)
    salA, _ = model.apply(params, jnp.asarray(np.stack(
        [_view(_scene(seed0 + i), seed0 * 5 + i)[0] for i in range(n)])))
    frac = float((jax.nn.sigmoid(salA) > 0.5).mean())
    return auc(vL), auc(vU), k, frac


def train(steps=400, bs=16, lr=3e-3, seed=0):
    model = SaliencyNet()
    key = jax.random.PRNGKey(seed)
    dummy = jnp.zeros((1, G, G, 3))
    params = model.init(key, dummy)
    n_par = sum(x.size for x in jax.tree.leaves(params))
    print(f"SaliencyNet params: {n_par}  (int8 ~{n_par} B; lattice D={W_LAT.shape[0]})",
          flush=True)
    opt = optax.adam(lr)
    st = opt.init(params)

    @jax.jit
    def step(params, st, A, Bv):
        def loss_fn(p):
            va = encode_learned(p, model, A)
            vb = encode_learned(p, model, Bv)
            return info_nce(va, vb)
        loss, g = jax.value_and_grad(loss_fn)(params)
        upd, st2 = opt.update(g, st, params)
        return optax.apply_updates(params, upd), st2, loss

    t0 = time.time()
    a0, u0, k, fr0 = evaluate(params, model)
    print(f"pre-train  AUC learned {a0:.3f}  uniform {u0:.3f}  "
          f"(k={k} features, {fr0*100:.0f}% cells fire)", flush=True)
    for s in range(steps):
        A, Bv = make_batch(bs, seed0=1 + s * bs)       # fresh scenes each step
        params, st, loss = step(params, st, A, Bv)
        if s % 50 == 0 or s == steps - 1:
            print(f"  step {s:4d}  loss {float(loss):.4f}  "
                  f"t={time.time()-t0:.0f}s", flush=True)
    aL, uL, k, frL = evaluate(params, model)
    print(f"post-train AUC learned {aL:.3f}  uniform {uL:.3f}  at k={k} "
          f"features ({frL*100:.0f}% cells fire, vs ~3300 raw pts)", flush=True)
    print(f"delta over uniform baseline: {aL-uL:+.3f} AUC", flush=True)
    import pickle
    from pathlib import Path
    out = Path(__file__).resolve().parents[1] / "scratch" / "learned_lidar.pkl"
    with open(out, "wb") as f:
        pickle.dump(jax.tree.map(np.asarray, params), f)
    print(f"saved params -> {out}", flush=True)
    return params, model


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    print(f"jax devices: {jax.devices()}", flush=True)
    if cmd == "train":
        train()
