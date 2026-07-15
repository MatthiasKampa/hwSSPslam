"""Tiny, FPGA-fittable vision CNN pretrained on CIFAR-100, with a BINARY
descriptor head that plugs straight into the semantic SSP map.

Architecture (unifying the two modalities, user directive 2026-07-15): image ->
tiny CNN -> (a) 100-way class head for pretraining, (b) a BINARY embedding head
(sign of a learned projection) = the descriptor that binds with position. The
binarization is both the FPGA substrate (BNN XNOR-popcount) and the VSA binding
key; the number of shared bits with a class prototype grades significance.

CIFAR-100 has fine labels chair / couch / table / bed / wardrobe, so a furniture
detector falls out directly -> "highlight the chairs in the map".

Runs on GPU via the cuDNN-free im2col convs in sspax/nnconv.py (this box's cuDNN
mismatches jaxlib); the net is tiny so a few epochs are quick.

  python3 -m sspax.vision.tinycnn train
"""
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from sspax.nnconv import Conv
import optax

ROOT = Path(__file__).resolve().parents[2]
CIFAR = ROOT / "scratch" / "datasets" / "cifar100.npz"
FURNITURE = ["chair", "couch", "table", "bed", "wardrobe"]
DESC_BITS = 64


def load(n_train=None):
    z = np.load(CIFAR, allow_pickle=True)
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    mean, std = Xtr.mean((0, 1, 2)), Xtr.std((0, 1, 2)) + 1e-6
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    if n_train:
        Xtr, ytr = Xtr[:n_train], z["ytr"][:n_train]
    else:
        ytr = z["ytr"]
    return Xtr, ytr, Xte, z["yte"], list(z["classes"])


class TinyCNN(nn.Module):
    """~3 conv blocks + a small head. Params kept tiny for the fabric."""
    ch: int = 16
    n_class: int = 100

    @nn.compact
    def __call__(self, x, train=False):
        for c in (self.ch, self.ch * 2, self.ch * 4):
            x = nn.relu(Conv(c, (3, 3), padding="SAME")(x))
            x = nn.max_pool(x, (2, 2), (2, 2))
        x = x.mean((1, 2))                          # global average pool
        emb = nn.Dense(DESC_BITS)(x)               # descriptor pre-activation
        logits = nn.Dense(self.n_class)(nn.relu(emb))
        return logits, emb


def binary_desc(emb):
    """Sign binarization -> {0,1}^DESC_BITS descriptor (the VSA binding key)."""
    return (np.asarray(emb) > 0).astype(np.int8)


def train(epochs=8, bs=256, lr=1e-3, n_train=20000, seed=0):
    if not CIFAR.exists():
        print(f"CIFAR npz not found at {CIFAR} — download still pending.")
        return
    Xtr, ytr, Xte, yte, classes = load(n_train)
    model = TinyCNN()
    key = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.zeros((1, 32, 32, 3)))
    n_par = sum(x.size for x in jax.tree.leaves(params))
    opt = optax.adamw(lr)
    st = opt.init(params)
    print(f"TinyCNN params {n_par} (~{n_par*1/1024:.0f} KB int8); "
          f"train {len(Xtr)} imgs, {epochs} epochs on {jax.devices()[0]}",
          flush=True)

    @jax.jit
    def step(params, st, xb, yb):
        def loss_fn(p):
            logits, _ = model.apply(p, xb)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, yb).mean()
        loss, g = jax.value_and_grad(loss_fn)(params)
        upd, st2 = opt.update(g, st, params)
        return optax.apply_updates(params, upd), st2, loss

    @jax.jit
    def acc(params, xb, yb):
        logits, _ = model.apply(params, xb)
        return (logits.argmax(1) == yb).mean()

    rng = np.random.default_rng(seed)
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr) - bs, bs):
            b = perm[i:i + bs]
            params, st, loss = step(params, st, jnp.asarray(Xtr[b]),
                                    jnp.asarray(ytr[b]))
        a = np.mean([float(acc(params, jnp.asarray(Xte[j:j + 500]),
                                jnp.asarray(yte[j:j + 500])))
                     for j in range(0, 5000, 500)])
        print(f"  epoch {ep} loss {float(loss):.3f} test-acc {a:.3f} "
              f"t={time.time()-t0:.0f}s", flush=True)

    # furniture-subset accuracy + per-class recall for chair
    fidx = [classes.index(c) for c in FURNITURE]
    logits = np.concatenate([np.asarray(model.apply(
        params, jnp.asarray(Xte[j:j + 500]))[0]) for j in range(0, len(Xte), 500)])
    pred = logits.argmax(1)
    for c, ci in zip(FURNITURE, fidx):
        m = yte == ci
        print(f"  {c:9s} recall {np.mean(pred[m] == ci):.3f} "
              f"(n={int(m.sum())})", flush=True)
    import pickle
    out = ROOT / "scratch" / "vision_cnn.pkl"
    with open(out, "wb") as f:
        pickle.dump({"params": jax.tree.map(np.asarray, params),
                     "classes": classes}, f)
    print(f"saved -> {out}", flush=True)
    return params, model, classes


def fpga_cost():
    """Static resource accounting for the two learned front-ends."""
    import sspax.learn_lidar as LL
    key = jax.random.PRNGKey(0)
    lidar = LL.SaliencyNet().init(key, jnp.zeros((1, LL.G, LL.G, 3)))
    n_lidar = sum(x.size for x in jax.tree.leaves(lidar))
    vis = TinyCNN().init(key, jnp.zeros((1, 32, 32, 3)))
    n_vis = sum(x.size for x in jax.tree.leaves(vis))
    print("FPGA front-end cost (int8 weights unless noted):")
    print(f"  lidar SaliencyNet : {n_lidar:6d} params  ~{n_lidar/1024:.1f} KB "
          f"(BNN 1-bit: {n_lidar/8:.0f} B)  -> per-cell saliency on a "
          f"{LL.GD}x{LL.GD} 2x-down map")
    print(f"  vision TinyCNN    : {n_vis:6d} params  ~{n_vis/1024:.0f} KB "
          f"(BNN 1-bit: {n_vis/8/1024:.1f} KB)  -> {DESC_BITS}-bit descriptor")
    print(f"  descriptor bind   : O(D) phase mult, D=360; map = one D-vector "
          f"(bounded, history-free)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cost"
    print(f"jax devices: {jax.devices()}", flush=True)
    if cmd == "train":
        train()
    elif cmd == "cost":
        fpga_cost()
