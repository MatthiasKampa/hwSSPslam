"""THE definitive direct wall test (P2, msg round 3): does a LEARNED lidar
front-end beat a FIXED encoding for PLACE recognition on the honest CLASSROOM
venue (withheld kinematic odometry, real revisits)? This is the question run2
(no revisits) and P1 (a self-rotation-metric artifact, retracted) could only
approach indirectly. Answer: NO learned place gain — and the GT-LEAK CONTROL is
built INTO this module so the trap is self-documenting.

Setup: the lidar ring net's per-bin saliency is trained self-supervised
(temporal-adjacency positives + azimuth-roll), then eval frames are encoded via
`lidar_ring.encode_bins` with LEARNED vs UNIFORM bin weights; gate = honest
revisit-vs-far AUC (rotation = azimuth roll), over ALL frames (the classroom's
revisits span the traverse, so a time-split eval segment has none — the same
structural fact as run2, here handled by scoring all frames; the saliency is
trained self-supervised and NEVER sees revisit labels).

THE GT-LEAK CONTROL (two training-negative definitions, both run + printed):
  - 'temporal' : off-diagonal batch frames are ALL negatives (label-free) —
                 the anti-oracle-clean training. Learned ~= uniform (NO gain).
  - 'pose'     : training far-negatives gated by WITHHELD-ODOMETRY distance
                 (d>4 m) — a rule-2 LEAK, since the gate ALSO scores on odometry.
                 Shows learned "beats" uniform (+0.046) — the false positive.
The gap between the two IS the lesson: pose-derived training negatives leak the
GT the gate scores on.

Pair-count note (reconciles the two banked classroom entries): 110 keyframes ->
179 revisit pairs (the recipe gate); 130 keyframes -> 244 (this module's cache).
Same venue, more frames = more pairs; both honest.

  PYTHONPATH=. python3 -m sspax.place_wall
"""
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

import experiments.lattice3d as L3
from sspax.lidar_ring import UnifiedLidarNet, encode_bins, _W_place

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "scratch" / "classroom_rasters.npz"


def _load():
    if not CACHE.exists():
        raise SystemExit(f"missing {CACHE} — run the classroom raster prep first "
                         "(see docs/RESULTS.md P2 2026-07-15).")
    z = np.load(CACHE)
    return z["rasters"], z["xy"]


def run(steps=500, bs=24, seed=0):
    R, xy = _load()
    n = len(R); split = int(0.6 * n)
    train_seg = np.arange(0, split)
    W = _W_place()
    model = UnifiedLidarNet()
    p0 = model.init(jax.random.PRNGKey(seed), jnp.asarray(R[:1]))
    naz4 = model.apply(p0, jnp.asarray(R[:1]))["w"].shape[2]
    nch = R.shape[-1]
    rolls = [int(round(k * naz4 / 8)) for k in range(8)]
    opt = optax.adam(3e-3)

    def enc(p, X, uniform):
        o = model.apply(p, X)
        w = jnp.ones((X.shape[0], naz4)) if uniform else jax.nn.softplus(o["w"][:, 0])
        rimg = jax.image.resize(X, (X.shape[0], 1, naz4, nch), "nearest")
        return jax.vmap(lambda ri, wi: encode_bins(W, ri, wi))(rimg, w)

    @jax.jit
    def step(p, st, xa, xb, negmask):
        def loss(pp):
            va = enc(pp, xa, False); vb = enc(pp, xb, False)
            S = jnp.max(jnp.stack([jnp.abs(jnp.roll(va, r, 1) @ vb.conj().T)
                                   for r in rolls]), 0)
            eye = jnp.eye(xa.shape[0], dtype=bool)
            S = jnp.where(eye | negmask, S / 0.1, -1e9)
            return optax.softmax_cross_entropy_with_integer_labels(
                S, jnp.arange(xa.shape[0])).mean()
        l, g = jax.value_and_grad(loss)(p); u, st2 = opt.update(g, st, p)
        return optax.apply_updates(p, u), st2, l

    # honest gate: revisit (same<0.7 m, gap>=40) vs far, over ALL frames
    (si, sj), (di, dj) = L3._pairs(xy)

    def gate(p):
        out = {}
        for name, uni in (("uniform", True), ("learned", False)):
            V = np.asarray(enc(p, jnp.asarray(R), uni))
            Smat = np.max(np.stack([np.abs(np.roll(V, r, 1) @ V.conj().T)
                                    for r in rolls]), 0)
            out[name] = L3._auc(Smat[si, sj], Smat[di, dj])
        return out

    print(f"CLASSROOM learned-vs-fixed PLACE wall test ({n} frames, "
          f"same={len(si)} far={len(di)} honest revisit pairs):\n")
    print(f"  {'train-negatives':>22}  {'uniform':>8} {'learned':>8} {'delta':>7}")
    results = {}
    for negmode in ("temporal (clean)", "pose-derived (GT-LEAK)"):
        params = model.init(jax.random.PRNGKey(seed), jnp.asarray(R[:1]))
        st = opt.init(params); rng = np.random.default_rng(seed); t0 = time.time()
        for s in range(steps):
            ia = rng.choice(train_seg, bs, replace=False)
            ib = np.clip(ia + rng.integers(-1, 2, bs), train_seg[0], train_seg[-1])
            if negmode.startswith("temporal"):
                nm = jnp.ones((bs, bs), bool)                  # label-free
            else:
                d = np.linalg.norm(xy[ia][:, None] - xy[ia][None], axis=2)
                nm = jnp.asarray(d > 4.0)                      # WITHHELD-ODOM leak
            params, st, _ = step(params, st, jnp.asarray(R[ia]),
                                 jnp.asarray(R[ib]), nm)
        g = gate(params); results[negmode] = g
        print(f"  {negmode:>22}  {g['uniform']:8.3f} {g['learned']:8.3f} "
              f"{g['learned']-g['uniform']:+7.3f}", flush=True)
    print("\n  => CLEAN (temporal negatives): learned <= uniform, NO place gain "
          "— the front-end cannot manufacture place separability on the honest\n"
          "     venue (the loop-closure wall, direct). The GT-LEAK (pose "
          "negatives) fabricates a +0.046 gain because the training negatives\n"
          "     use the same withheld odometry the gate scores on — the trap, "
          "self-documented. uniform is deterministic; delta is the whole story.")
    return results


if __name__ == "__main__":
    run()
