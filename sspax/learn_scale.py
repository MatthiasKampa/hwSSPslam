"""Learned scale-modulation (TRAINING_PROGRAM.md P4 centerpiece): a THERMOMETER
blanking head that LEARNS the ladder allocation the recipes hand-tuned.

Motivation (proven in sspax/ladder_extent.py): a fixed ladder is venue-
suboptimal — the best coarsest wavelength tracks venue extent, so no single
ladder wins all venues. Here a tiny CNN emits, per BEV cell, one scalar = the
FINEST USEFUL scale (a thermometer cutoff over the 6-ring ladder); rings finer
than the cutoff are BLANKED. On-fabric this is a per-ring comparator
(ring_idx ≥ cutoff → accumulate), zero multipliers. Trained end-to-end through
the differentiable SSP encode with a contrastive place loss, across MIXED
extents; it must beat the best FIXED ladder and approach the per-extent oracle
at equal D.

Straight-through: soft sigmoid ring mask in training (β annealed), hard blank at
eval. QAT-friendly. Anti-oracle: no reference labels (same-place = jittered view
pairs; negatives = distinct layouts at the same extent).

STATUS (2026-07-15): mechanism built + budget-fit (1778 params, zero-multiplier
per-ring comparator). HONEST NEGATIVE on the synthetic surface: box-rooms have
no perceptual-aliasing regime where blanking helps — aligned views make the full
ladder trivially perfect (learned head correctly converges to cutoff≈0 = keep
all rings), and full-rotation views drop every ladder to chance. So the learned
BLANKING advantage needs the real building-scale aliasing (school_run2) — this
module is the transfer-gate candidate; ladder_extent.py is the transferable
synthetic result that motivates it. Gate on the deploy box before adoption.

  python3 -m sspax.learn_scale train
"""
import sys
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from sspax.nnconv import Conv
from sspax import core as C
from sspax import sphere as SPH
from sspax import worlds as Wd

# FULL ladder: all 6 rings available for the thermometer to select among.
FULL_LAMS = list(np.geomspace(0.25, 90.5, 6))            # 0.25 .. 90.5 m
RINGKW = dict(spacing="arc", apportion="const", stagger=0.5, n_rings=6)
DIRS = SPH.dirs_ring(60, **RINGKW)
N_DIR = 60
N_RING = 6
W_LAT = jnp.asarray(C.W_of(DIRS, FULL_LAMS))             # D = 360, ring-major
EXTENTS = [8.0, 30.0, 80.0]
G = 48
GD = G // 2


# --------------------------------------------------------------------------
#  rasterize (extent-aware) + per-scene cell centres
# --------------------------------------------------------------------------
def rasterize(cloud, extent):
    r = extent / 2
    g = G
    xy = cloud[:, :2]
    ix = np.clip(((xy[:, 0] + r) / (2 * r) * g).astype(int), 0, g - 1)
    iy = np.clip(((xy[:, 1] + r) / (2 * r) * g).astype(int), 0, g - 1)
    occ = np.zeros((g, g)); hmax = np.full((g, g), -1e3); hsum = np.zeros((g, g))
    np.add.at(occ, (ix, iy), 1.0)
    np.maximum.at(hmax, (ix, iy), cloud[:, 2])
    np.add.at(hsum, (ix, iy), cloud[:, 2])
    hmax[hmax < -1e2] = 0.0
    hmean = np.where(occ > 0, hsum / np.maximum(occ, 1), 0.0)
    cell_m = np.log(extent / GD)                          # metric scale channel
    return np.stack([np.log1p(occ), hmax, hmean,
                     np.full((g, g), cell_m)], -1).astype(np.float64)


def cells_of(extent):
    r = extent / 2
    c = (np.arange(GD) + 0.5) / GD * (2 * r) - r
    cx, cy = np.meshgrid(c, c, indexing="ij")
    return np.stack([cx.ravel(), cy.ravel(), np.zeros(GD * GD)], -1)


CELLS = {E: jnp.asarray(cells_of(E)) for E in EXTENTS}


# --------------------------------------------------------------------------
#  net: per-cell weight + per-cell thermometer cutoff
# --------------------------------------------------------------------------
class ScaleNet(nn.Module):
    ch: int = 12

    @nn.compact
    def __call__(self, grid):
        x = nn.relu(Conv(self.ch, (3, 3))(grid))
        x = nn.relu(Conv(self.ch, (3, 3), strides=(2, 2))(x))   # 2x down
        w = Conv(1, (1, 1))(x)[..., 0]                          # saliency logit
        cut = Conv(1, (1, 1))(x)[..., 0]                        # cutoff logit
        return w, cut


def encode_mod(params, model, grid, cells, beta, hard=False):
    """SSP with per-cell thermometer ring-blanking. mask_r = keep rings with
    index >= cutoff (coarser); finer rings blanked."""
    wlog, cutlog = model.apply(params, grid)                    # (B,GD,GD)
    B = grid.shape[0]
    w = jax.nn.softplus(wlog).reshape(B, -1)                    # (B,N) >=0
    cutoff = (N_RING * jax.nn.sigmoid(cutlog)).reshape(B, -1)   # (B,N) in [0,6]
    r = jnp.arange(N_RING)                                      # 0..5 fine->coarse
    if hard:
        mask = (r[None, None, :] >= jnp.round(cutoff)[..., None]).astype(grid.dtype)
    else:
        mask = jax.nn.sigmoid(beta * (r[None, None, :] - cutoff[..., None]))
    pts = jnp.broadcast_to(cells, (B,) + cells.shape)          # (B,N,3)
    ph = jnp.exp(1j * jnp.einsum("bnd,kd->bnk", pts, W_LAT))   # (B,N,360)
    ph = ph.reshape(B, -1, N_RING, N_DIR) * mask[..., None]    # blank rings
    v = jnp.einsum("bnrd,bn->brd", ph, w.astype(ph.dtype)).reshape(B, -1)
    return v / jnp.maximum(jnp.linalg.norm(v, axis=1, keepdims=True), 1e-12)


# --------------------------------------------------------------------------
#  data
# --------------------------------------------------------------------------
def _scene(extent, seed, n_pts=4000):
    base = Wd.room(n_pts * 2, seed=1000 + seed,
                   dims=(extent, 0.75 * extent, 3.0), n_boxes=10)
    return base


def _view(base, extent, vseed, n_pts=4000, align=True):
    """Sensor-frame BEV of a jittered view. align=False keeps the yaw (a revisit
    from a different heading) -> the encode must be rotation-searched, and fine
    rings alias under rotation at large extent (the regime coarse rings win)."""
    rr = np.random.default_rng(vseed)
    R = C._axis_rot("yaw", rr.uniform(-180, 180))
    t = rr.uniform(-0.3, 0.3, 3) * np.array([1, 1, 0.3])
    idx = rr.permutation(len(base))[:n_pts]
    pts = base[idx] @ R.T + t + rr.normal(0, 0.02, (n_pts, 3))
    if align:
        pts = (pts - t) @ R
    return rasterize(pts, extent)


# rotation-search permutation set (per-ring snap over the full circle)
_YAWP = [SPH.yaw_perm(60, k * 2 * np.pi / 24, n_lams=N_RING, **RINGKW)[0]
         for k in range(24)]
_YAWP = [jnp.asarray(p) for p in _YAWP]


def batch(extent, bs, seed0):
    A, Bv = [], []
    for i in range(bs):
        base = _scene(extent, seed0 + i)
        A.append(_view(base, extent, (seed0 + i) * 2 + 1))
        Bv.append(_view(base, extent, (seed0 + i) * 2 + 2))
    return jnp.asarray(np.stack(A)), jnp.asarray(np.stack(Bv))


def rot_nce(va, vb, tau=0.1):
    """Rotation-searched InfoNCE: S[i,j]=max_p |<perm_p(va_i), vb_j>|. The max
    over yaw perms makes the task rotation-invariant, so fine rings that alias
    under rotation inflate the negatives — the pressure that teaches blanking."""
    S = jnp.max(jnp.stack([jnp.abs(va[:, p] @ vb.conj().T) for p in _YAWP]), 0)
    return optax.softmax_cross_entropy_with_integer_labels(
        S / tau, jnp.arange(len(va))).mean()


# --------------------------------------------------------------------------
#  fixed-ladder baselines (equal D) + eval
# --------------------------------------------------------------------------
def _auc(pos, neg):
    x = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
    return (r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / max(len(pos) * len(neg), 1)


def _fixed_mask(cut):
    r = np.arange(N_RING)
    return jnp.asarray((r >= cut).astype(float))          # global thermometer


def _encode_fixed(grid, cells, cut):
    """All cells weighted by occupancy, fixed global ring cutoff (baseline)."""
    B = grid.shape[0]
    occ = grid[..., 0].reshape(B, GD, 2, GD, 2).mean((2, 4))   # pool 48->24
    w = jax.nn.softplus(occ).reshape(B, -1)               # occupancy as weight
    pts = jnp.broadcast_to(cells, (B,) + cells.shape)
    ph = jnp.exp(1j * jnp.einsum("bnd,kd->bnk", pts, W_LAT)).reshape(B, -1, N_RING, N_DIR)
    ph = ph * _fixed_mask(cut)[None, None, :, None]
    v = jnp.einsum("bnrd,bn->brd", ph, w.astype(ph.dtype)).reshape(B, -1)
    return v / jnp.maximum(jnp.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def _place_auc(V, n, rot=True):
    """Rotation-searched (max over 24 yaw perms) same-vs-different place AUC."""
    A_, B_ = V[:n], V[n:]
    if not rot:
        S = np.abs(np.asarray(A_ @ B_.conj().T))
    else:
        S = np.zeros((n, n))
        for p in _YAWP:
            S = np.maximum(S, np.abs(np.asarray(A_[:, p] @ B_.conj().T)))
    return _auc(np.diag(S), S[~np.eye(n, dtype=bool)])


def evaluate(params, model, n=24, seed0=7000):
    """Place AUC per extent: learned modulation vs fixed global cutoffs."""
    print(f"  {'extent':>8s} " + "".join(f"{'fix c'+str(c):>9s}" for c in (0, 2, 4))
          + f"{'LEARNED':>10s}  learned-cutoffs(mean)")
    res = {}
    for E in EXTENTS:
        A = jnp.asarray(np.stack([_view(_scene(E, seed0 + i), E, seed0 * 3 + i)
                                  for i in range(n)]))
        Bv = jnp.asarray(np.stack([_view(_scene(E, seed0 + i), E, seed0 * 3 + i + 999)
                                   for i in range(n)]))
        gg = jnp.concatenate([A, Bv]); cells = CELLS[E]
        fixed = {c: _place_auc(_encode_fixed(gg, cells, c), n) for c in (0, 2, 4)}
        vlear = encode_mod(params, model, gg, cells, beta=8.0, hard=True)
        alear = _place_auc(vlear, n)
        _, cutlog = model.apply(params, A)
        cmean = float((N_RING * jax.nn.sigmoid(cutlog)).mean())
        res[E] = (fixed, alear, cmean)
        print(f"  {E:6.0f} m " + "".join(f"{fixed[c]:9.3f}" for c in (0, 2, 4))
              + f"{alear:10.3f}   {cmean:.2f}", flush=True)
    # summary: best-single-fixed (avg over extents) vs learned (avg)
    bestfix = max((0, 2, 4), key=lambda c: np.mean([res[E][0][c] for E in EXTENTS]))
    fix_avg = np.mean([res[E][0][bestfix] for E in EXTENTS])
    lear_avg = np.mean([res[E][1] for E in EXTENTS])
    oracle = np.mean([max(res[E][0].values()) for E in EXTENTS])
    print(f"  MEAN over extents: best-fixed(c{bestfix}) {fix_avg:.3f} | "
          f"LEARNED {lear_avg:.3f} | per-extent-oracle {oracle:.3f}", flush=True)
    return lear_avg, fix_avg


def train(steps=600, bs=20, lr=3e-3, seed=0):
    model = ScaleNet()
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, G, G, 4)))
    n_par = sum(x.size for x in jax.tree.leaves(params))
    opt = optax.adam(lr); st = opt.init(params)
    print(f"ScaleNet params {n_par} (~{n_par}B int8); ladder 0.25-90.5 m x6 "
          f"D={W_LAT.shape[0]}; extents {EXTENTS}", flush=True)

    @partial(jax.jit, static_argnums=())
    def step(params, st, A, Bv, beta):
        def loss(p):
            va = encode_mod(p, model, A, cellsA, beta)
            vb = encode_mod(p, model, Bv, cellsB, beta)
            return rot_nce(va, vb)
        l, g = jax.value_and_grad(loss)(params)
        u, st2 = opt.update(g, st, params)
        return optax.apply_updates(params, u), st2, l

    print("pre-train:"); evaluate(params, model)
    t0 = time.time()
    for s in range(steps):
        E = EXTENTS[s % len(EXTENTS)]                     # cycle extents
        cellsA = cellsB = CELLS[E]
        A, Bv = batch(E, bs, seed0=1 + s * bs)
        beta = 2.0 + 8.0 * min(1.0, s / (steps * 0.6))    # anneal to hard
        params, st, l = step(params, st, A, Bv, beta)
        if s % 100 == 0 or s == steps - 1:
            print(f"  step {s:4d} E={E:.0f} loss {float(l):.3f} beta {beta:.1f} "
                  f"t={time.time()-t0:.0f}s", flush=True)
    print("post-train:"); lear, fix = evaluate(params, model)
    print(f"delta learned - best-fixed: {lear-fix:+.3f} AUC (mean over extents)",
          flush=True)
    import pickle
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "scratch" / "learned_scale.pkl"
    pickle.dump(jax.tree.map(np.asarray, params), open(p, "wb"))
    print(f"saved -> {p}", flush=True)
    return params, model


if __name__ == "__main__":
    print(f"jax devices: {jax.devices()}", flush=True)
    if (sys.argv[1] if len(sys.argv) > 1 else "train") == "train":
        train()
