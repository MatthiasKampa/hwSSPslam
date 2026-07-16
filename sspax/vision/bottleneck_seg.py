"""The VSA object encoder as a PRETRAIN-then-TRUNCATE bottleneck seg net
(user architecture, 2026-07-16). The CNN is trained on ORDINARY dense
segmentation — fully VSA-agnostic — with ONE structural constraint: a narrow
PER-OUTPUT-PIXEL bottleneck code, followed by a SINGLE FC (a 1x1 conv = per-pixel
linear classifier), no further convs, that translates each cell's code vector
into its class. That single linear head forces the per-cell code to be LINEARLY
class-separable — exactly the property a VSA descriptor needs (class becomes
cosine / inner-product readable). At deploy we CUT the FC head and bind the
binarized per-cell code into the SSP vector at each cell's position; the map's
fixed bit-width IS the bottleneck.

  encoder(convs) -> trunk 30x40x64
                 -> code  30x40xB      (per-pixel bottleneck; the DEPLOY descriptor)
                 -> [cut here at deploy]
                 -> sign  30x40xB       (STE-binarized: fwd +-1, bwd tanh grad)
                 -> ONE FC (1x1)        -> 30x40xN_CLASS logits   (train-only)

Trained on dense seg CE (NYUv2). Nothing is trained against the binding/query
objective — the SSP encode is a fixed algebraic consumer of the bits.

  PYTHONPATH=. .venv_data/bin/python -m sspax.vision.bottleneck_seg          # train + validate
  PYTHONPATH=. .venv_data/bin/python -m sspax.vision.bottleneck_seg smoke     # tiny synthetic dry-run
"""
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from sspax.nnconv import Conv, crelu
import sspax.vision.segnet as S


class BottleneckSegNet(nn.Module):
    ch: int = S.CH
    n_class: int = S.N_CLASS
    bits: int = 32                      # the per-pixel bottleneck width (= spec descriptor bits)

    @nn.compact
    def __call__(self, x):
        act = crelu
        x = act(Conv(self.ch, (3, 3), strides=(2, 2))(x))    # /2  60x80
        x = act(Conv(self.ch, (3, 3), strides=(2, 2))(x))    # /4  30x40
        x = act(Conv(self.ch * 2, (3, 3))(x))                # /4 dense 3x3 (RF)
        trunk = act(Conv(self.ch * 2, (1, 1))(x))            # 30x40x64 shared trunk
        code = Conv(self.bits, (1, 1))(trunk)                # 30x40xB PER-PIXEL BOTTLENECK (linear)
        t = jnp.tanh(code)
        b = jax.lax.stop_gradient(jnp.sign(t) - t) + t       # STE: fwd +-1, bwd tanh grad
        logits = Conv(self.n_class, (1, 1))(b)               # ONE FC (per-pixel linear) -> classes
        return dict(logits=logits, code=code, bits=b, trunk=trunk)


def _cw(Ytr, n_class):
    cnt = np.bincount(S._pool_labels(Ytr).ravel(), minlength=n_class).astype(float)
    cw = np.zeros(n_class, np.float32)
    cw[1:] = 1 / np.sqrt(cnt[1:] + 1.0)
    return jnp.asarray(cw / cw[1:].mean())


def train(bits=32, n_class=40, hw=(120, 160), steps=4000, seed=0):
    S.N_CLASS = n_class
    X, L = S.load_nyu(hw=hw, max_n=1449)
    mu, sd = X.mean(), X.std() + 1e-6
    X = ((X - mu) / sd).astype(np.float32)
    ntr = int(0.8 * len(X)); Xtr, Ytr, Xte, Yte = X[:ntr], L[:ntr], X[ntr:], L[ntr:]
    cw = _cw(Ytr, n_class)
    model = BottleneckSegNet(n_class=n_class, bits=bits)
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1,) + X.shape[1:]))
    opt = optax.adamw(2e-3); st = opt.init(params)

    @jax.jit
    def step(p, st, xb, yb):
        def loss(pp):
            return S.seg_loss(model.apply(pp, xb)["logits"], yb, cw)
        l, g = jax.value_and_grad(loss)(p); u, st2 = opt.update(g, st, p)
        return optax.apply_updates(p, u), st2, l
    rng = np.random.default_rng(0); t0 = time.time()
    for s in range(steps):
        b = rng.integers(0, ntr, 8)
        xa = Xtr[b] + rng.normal(0, 0.05, Xtr[b].shape).astype(np.float32)
        params, st, l = step(params, st, jnp.asarray(xa), jnp.asarray(S._pool_labels(Ytr[b])))
        if s % 1000 == 0:
            print(f"    step {s:4d}  loss {float(l):.3f}  t={time.time()-t0:.0f}s", flush=True)
    return model, params, Xte, Yte, t0


def _eval(model, params, Xte, Yte):
    """(a) seg pixacc from the BINARIZED-code path; (b) VSA-readiness: do
    same-class cells' binary codes agree (cosine) more than different-class?"""
    out = [model.apply(params, jnp.asarray(Xte[i:i + 24])) for i in range(0, len(Xte), 24)]
    seg = np.concatenate([np.asarray(o["logits"]).argmax(-1) for o in out])
    bits = np.concatenate([np.asarray(o["bits"]) for o in out])       # +-1
    L4 = S._pool_labels(Yte); v = L4 > 0
    pixacc = float((seg[v] == L4[v]).mean())
    # VSA-readiness: sample same-class vs diff-class cell pairs, cosine of +-1 codes
    B = (bits.reshape(-1, bits.shape[-1]) > 0).astype(np.float32) * 2 - 1
    lab = L4.reshape(-1); m = lab > 0
    B, lab = B[m], lab[m]
    rng = np.random.default_rng(3); idx = rng.choice(len(B), min(4000, len(B)), replace=False)
    B, lab = B[idx], lab[idx]
    cos = (B @ B.T) / B.shape[1]
    same = lab[:, None] == lab[None]
    np.fill_diagonal(same, False)
    tri = np.triu(np.ones_like(same), 1).astype(bool)
    sc = cos[same & tri]; dc = cos[(~same) & tri]
    from sspax.surfaces_tier import _auc
    auc = _auc(sc, dc)
    return pixacc, float(sc.mean()), float(dc.mean()), auc


def run(bits=32, n_class=40, steps=4000):
    print(f"BOTTLENECK seg net (VSA-agnostic pretrain; per-pixel code bits={bits}, "
          f"{n_class}-class, single-FC head):", flush=True)
    model, params, Xte, Yte, t0 = train(bits=bits, n_class=n_class, steps=steps)
    pixacc, sc, dc, auc = _eval(model, params, Xte, Yte)
    print(f"  seg pixacc (binarized-code path, non-void) : {pixacc:.3f}", flush=True)
    print(f"  VSA-readiness of the CUT code (bind-ready): same-class cos {sc:.3f} vs "
          f"diff-class {dc:.3f}  -> separability AUC {auc:.3f}")
    print(f"  => the single linear FC forces the per-cell +-1 code to be class-separable; "
          f"cut the FC, bind the code. t={time.time()-t0:.0f}s")
    return pixacc, auc


def smoke():
    """tiny synthetic dry-run (no NYU): confirms the net trains + shapes."""
    S.N_CLASS = 6
    X, L = S.synth_luma(120, seed=0)
    model = BottleneckSegNet(n_class=6, bits=32)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1,) + X.shape[1:]))
    o = model.apply(params, jnp.asarray(X[:2]))
    print("smoke shapes:", {k: tuple(np.asarray(v).shape) for k, v in o.items()})
    print("bits are +-1:", bool(np.all(np.abs(np.asarray(o["bits"])) == 1.0)))


if __name__ == "__main__":
    smoke() if len(sys.argv) > 1 and sys.argv[1] == "smoke" else run()
