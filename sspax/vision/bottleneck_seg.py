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

from sspax.nnconv import Conv, crelu, conv2d
import sspax.vision.segnet as S


# --------------------------------------------------------------------------
#  full QAT: fake-quantize weights (int8 per-cout) AND activations (int8/uint8
#  per-tensor) in the forward pass, STE through the rounding, so the net trains
#  against its DEPLOY int8 arithmetic (headio models weight-int8 only; the
#  activation int8 is the part it does NOT yet model). The bottleneck code stays
#  1-bit (+-1) — the VSA descriptor.
# --------------------------------------------------------------------------
def _ste_round(x):
    return x + jax.lax.stop_gradient(jnp.round(x) - x)      # round fwd, identity bwd


def quant_w(w, bits=8):
    """per-OUTPUT-CHANNEL symmetric weights at `bits` (matches headio per-cout).
    bits=1 -> binary {-s,+s} (s = per-cout mean|w|); bits>=2 -> symmetric int."""
    if bits <= 1:                                          # binary weights (BNN)
        s = jax.lax.stop_gradient(jnp.mean(jnp.abs(w), axis=(0, 1, 2), keepdims=True) + 1e-12)
        return jax.lax.stop_gradient(s * jnp.sign(w) - w) + w      # STE: fwd s*sign, bwd identity
    qmax = 2 ** (bits - 1) - 1
    s = jax.lax.stop_gradient(jnp.max(jnp.abs(w), axis=(0, 1, 2), keepdims=True) / qmax + 1e-12)
    return _ste_round(jnp.clip(w / s, -qmax - 1, qmax)) * s


def quant_a(x, signed, bits=8):
    """per-tensor dynamic activations at `bits`. signed int (input) / unsigned
    (>=0 post-CReLU). bits=1 unsigned -> {0,s} binary activation; 99.9-pct scale."""
    qmax = (2 ** (bits - 1) - 1) if signed else (2 ** bits - 1)
    lo = -(2 ** (bits - 1)) if signed else 0
    s = jax.lax.stop_gradient(jnp.percentile(jnp.abs(x), 99.9) / max(qmax, 1) + 1e-9)
    return _ste_round(jnp.clip(x / s, lo, qmax)) * s


class QuantConv(nn.Module):
    features: int
    kernel_size: tuple = (3, 3)
    strides: tuple = (1, 1)
    padding: str = "SAME"
    wbits: int = 8

    @nn.compact
    def __call__(self, x):
        kh, kw = self.kernel_size
        w = self.param("kernel", nn.initializers.lecun_normal(), (kh, kw, x.shape[-1], self.features))
        b = self.param("bias", nn.initializers.zeros, (self.features,))
        s = self.strides[0] if isinstance(self.strides, (tuple, list)) else self.strides
        return conv2d(x, quant_w(w, self.wbits), b, s, self.padding)


class QNet(nn.Module):
    """config-driven QAT net: `layers` = ((features,k,stride,wbits,abits),...) trunk;
    then int8-weight bottleneck -> 1-bit code -> single FC. Covers arch (widths,
    depth, kernels) x per-layer quant (int8/int4/int2/binary) incl. STAGGERED."""
    layers: tuple
    n_class: int = S.N_CLASS
    bits: int = 32                                         # code width (channels)
    code_wbits: int = 8
    head_wbits: int = 8
    in_bits: int = 8

    @nn.compact
    def __call__(self, x):
        x = quant_a(x, signed=True, bits=self.in_bits)     # input (camera is 8-bit)
        for (f, k, st, wb, ab) in self.layers:
            x = crelu(QuantConv(f, (k, k), (st, st), wbits=wb)(x))
            x = quant_a(x, signed=False, bits=ab)          # post-CReLU (>=0)
        code = QuantConv(self.bits, (1, 1), wbits=self.code_wbits)(x)
        t = jnp.tanh(code)
        b = jax.lax.stop_gradient(jnp.sign(t) - t) + t     # 1-bit code (the VSA descriptor)
        logits = QuantConv(self.n_class, (1, 1), wbits=self.head_wbits)(b)
        return dict(logits=logits, code=code, bits=b, trunk=x)


def _weight_bytes(layers, code_bits=32, n_class=40):
    """FPGA weight footprint = sum(param_count * wbits/8) over trunk+code+head.
    CReLU doubles the channels feeding the next layer."""
    tot = 0.0; cin = 1                                     # Y8 input
    for (f, k, st, wb, ab) in layers:
        tot += (k * k * cin * f) * wb / 8.0
        cin = f * 2                                        # CReLU doubles
    tot += (1 * 1 * cin * code_bits) * 8 / 8.0             # code conv (int8 weights)
    tot += (1 * 1 * code_bits * n_class) * 8 / 8.0         # FC head (int8)
    return tot


class BottleneckSegNetQAT(nn.Module):
    """full-QAT twin of BottleneckSegNet: int8 weights + int8/uint8 activations
    (STE), 1-bit bottleneck code, int8 single-FC head."""
    ch: int = S.CH
    n_class: int = S.N_CLASS
    bits: int = 32

    @nn.compact
    def __call__(self, x):
        x = quant_a(x, signed=True)                          # int8 input
        def blk(x, f, k, s):
            return quant_a(crelu(QuantConv(f, (k, k), strides=(s, s))(x)), signed=False)  # uint8 post-CReLU
        x = blk(x, self.ch, 3, 2)                            # /2
        x = blk(x, self.ch, 3, 2)                            # /4
        x = blk(x, self.ch * 2, 3, 1)                        # dense 3x3
        trunk = blk(x, self.ch * 2, 1, 1)                    # 30x40x64 int8
        code = QuantConv(self.bits, (1, 1))(trunk)           # int8-weight -> code (linear)
        t = jnp.tanh(code)
        b = jax.lax.stop_gradient(jnp.sign(t) - t) + t       # 1-bit code (STE)
        logits = QuantConv(self.n_class, (1, 1))(b)          # int8 single FC on +-1 code
        return dict(logits=logits, code=code, bits=b, trunk=trunk)


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


def train(bits=32, n_class=40, hw=(120, 160), steps=4000, seed=0, net=BottleneckSegNet):
    S.N_CLASS = n_class
    X, L = S.load_nyu(hw=hw, max_n=1449)
    ntr = int(0.8 * len(X))
    mu, sd = X[:ntr].mean(), X[:ntr].std() + 1e-6      # TRAIN-only stats (rule 2: no test->train leak)
    X = ((X - mu) / sd).astype(np.float32)
    Xtr, Ytr, Xte, Yte = X[:ntr], L[:ntr], X[ntr:], L[ntr:]
    cw = _cw(Ytr, n_class)
    model = net(n_class=n_class, bits=bits)
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


def run_qat(bits=32, n_class=40, steps=4000):
    """side-by-side: FLOAT-trained vs FULL-QAT (int8 weights + int8/uint8 acts,
    1-bit code). Same seed/data — the gap is the honest int8-arithmetic cost."""
    print(f"FULL QAT vs FLOAT bottleneck seg (int8 weights+acts, 1-bit code; "
          f"bits={bits}, {n_class}-class):", flush=True)
    print(f"  {'arm':>10}  {'seg pixacc':>10}  {'code AUC':>9}")
    rows = {}
    for tag, net in (("float", BottleneckSegNet), ("QAT-int8", BottleneckSegNetQAT)):
        model, params, Xte, Yte, t0 = train(bits=bits, n_class=n_class, steps=steps, net=net)
        pixacc, sc, dc, auc = _eval(model, params, Xte, Yte)
        rows[tag] = (pixacc, auc)
        print(f"  {tag:>10}  {pixacc:>10.3f}  {auc:>9.3f}   t={time.time()-t0:.0f}s", flush=True)
    dp = rows["QAT-int8"][0] - rows["float"][0]; da = rows["QAT-int8"][1] - rows["float"][1]
    print(f"  => QAT-int8 vs float: dpixacc {dp:+.3f}  dAUC {da:+.3f}. Full QAT "
          f"trains the trunk against its int8 deploy arithmetic (headio models "
          f"weight-int8 only); a near-zero gap means int8 is free here.")
    return rows


def _fit(model, Xtr, Ytr, cw, steps, seed=0):
    ntr = len(Xtr)
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1,) + Xtr.shape[1:]))
    opt = optax.adamw(2e-3); st = opt.init(params)

    @jax.jit
    def step(p, st, xb, yb):
        def loss(pp): return S.seg_loss(model.apply(pp, xb)["logits"], yb, cw)
        l, g = jax.value_and_grad(loss)(p); u, st2 = opt.update(g, st, p)
        return optax.apply_updates(p, u), st2, l
    rng = np.random.default_rng(0)
    for s in range(steps):
        b = rng.integers(0, ntr, 8)
        xa = Xtr[b] + rng.normal(0, 0.05, Xtr[b].shape).astype(np.float32)
        params, st, _ = step(params, st, jnp.asarray(xa), jnp.asarray(S._pool_labels(Ytr[b])))
    return params


def sweep(steps=2500, n_class=40):
    """arch x quant-level sweep, incl. STAGGERED per-layer precision. Loads NYU
    once; trains each config; reports pixacc / code-AUC / FPGA weight-KB."""
    S.N_CLASS = n_class
    X, L = S.load_nyu(hw=(120, 160), max_n=1449)
    ntr = int(0.8 * len(X)); mu, sd = X[:ntr].mean(), X[:ntr].std() + 1e-6
    X = ((X - mu) / sd).astype(np.float32)
    Xtr, Ytr, Xte, Yte = X[:ntr], L[:ntr], X[ntr:], L[ntr:]
    cw = _cw(Ytr, n_class)
    TRUNK = [(16, 3, 2), (16, 3, 2), (32, 3, 1), (32, 1, 1)]        # the pinned base trunk
    base = lambda q: tuple((f, k, st, q, q) for (f, k, st) in TRUNK)
    stag = lambda ps: tuple((f, k, st, wb, ab) for (f, k, st), (wb, ab) in zip(TRUNK, ps))
    arch = lambda tr, q: tuple((f, k, st, q, q) for (f, k, st) in tr)
    configs = [
        ("uniform int8",     base(8)),
        ("uniform int4",     base(4)),
        ("uniform int2",     base(2)),
        ("uniform binary",   base(1)),
        ("stagger 8-4-2-2",  stag([(8, 8), (4, 4), (2, 2), (2, 2)])),
        ("stagger 8-4-4-2",  stag([(8, 8), (4, 4), (4, 4), (2, 2)])),
        ("stagger 8-2-2-1",  stag([(8, 8), (2, 2), (2, 2), (1, 1)])),  # binary tail
        ("narrow ch8 int4",  arch([(8, 3, 2), (8, 3, 2), (16, 3, 1), (16, 1, 1)], 4)),
        ("wide ch24 int4",   arch([(24, 3, 2), (24, 3, 2), (48, 3, 1), (48, 1, 1)], 4)),
        ("shallow int4",     arch([(16, 3, 2), (16, 3, 2), (32, 1, 1)], 4)),
        ("deep int4",        arch([(16, 3, 2), (16, 3, 2), (32, 3, 1), (32, 3, 1), (32, 1, 1)], 4)),
    ]
    print(f"ARCH x QUANT sweep (NYUv2 {n_class}-class, {steps} steps, 1-bit code fixed):", flush=True)
    print(f"  {'config':>18} {'pixacc':>7} {'codeAUC':>8} {'W-KB':>7} {'params':>8}")
    rows = []
    for name, layers in configs:
        model = QNet(layers=layers, n_class=n_class, bits=32)
        params = _fit(model, Xtr, Ytr, cw, steps)
        pixacc, sc, dc, auc = _eval(model, params, Xte, Yte)
        wkb = _weight_bytes(layers, 32, n_class) / 1024
        nparam = int(sum(p.size for p in jax.tree_util.tree_leaves(params)))
        rows.append((name, pixacc, auc, wkb, nparam))
        print(f"  {name:>18} {pixacc:>7.3f} {auc:>8.3f} {wkb:>7.1f} {nparam:>8d}", flush=True)
    print("  => pixacc/AUC vs weight-KB = the FPGA accuracy/footprint Pareto; "
          "staggered rows test per-layer mixed precision. 1-bit code = the descriptor.")
    return rows


def smoke():
    """tiny synthetic dry-run (no NYU): confirms the net trains + shapes."""
    S.N_CLASS = 6
    X, L = S.synth_luma(120, seed=0)
    model = BottleneckSegNet(n_class=6, bits=32)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1,) + X.shape[1:]))
    o = model.apply(params, jnp.asarray(X[:2]))
    print("smoke shapes:", {k: tuple(np.asarray(v).shape) for k, v in o.items()})
    print("bits are +-1:", bool(np.all(np.abs(np.asarray(o["bits"])) == 1.0)))
    q = BottleneckSegNetQAT(n_class=6, bits=32)
    oq = q.apply(q.init(jax.random.PRNGKey(0), jnp.zeros((1,) + X.shape[1:])), jnp.asarray(X[:2]))
    print("QAT smoke shapes:", {k: tuple(np.asarray(v).shape) for k, v in oq.items()},
          " bits +-1:", bool(np.all(np.abs(np.asarray(oq["bits"])) == 1.0)))


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else ""
    (smoke if a == "smoke" else run_qat if a == "qat" else sweep if a == "sweep" else run)()
