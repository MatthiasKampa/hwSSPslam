"""Unified dual-objective VISION net (user directive 2026-07-15b + deploy-agent
pinned geometry, commit 34989b1 / TRAINING_PROGRAM.md "Network geometry"): ONE
shared-trunk net at DEPLOY resolution with a tracking head AND a per-cell
segmented-classification head, so per-cell features are spatially bindable into
the SSP map.

Pinned geometry (this file conforms):
  input   Y8 mono, 160x120 (half) or 320x240 (full)   [depth is the LIDAR net's]
  trunk   /4  ->  30x40x64  (shared)
  heads @ 30x40 (per-cell, load-bearing — whole-image labels can't be bound):
    TRACKING  (Regime A): per-cell saliency WEIGHT + thermometer CUTOFF
              + binary DESCRIPTOR (the appearance/tracking key bound with pos)
    SEG       (Regime B): per-cell class logits -> argmax -> k-bit label code
              (bits double as encode STRENGTH / significance in the map)

Trunk + tracking head run @frame rate; the seg head re-uses the latest trunk
features @keyframe rate. CReLU trunk for param efficiency. numpy-only forward
(`forward_np`) exported for the deploy box; `fpga_cost` reports the budget line
for the trained variant.

Trained joint: per-pixel seg CE (labels pooled to 30x40) + cell-InfoNCE on the
descriptor (self-supervised, two colour-jittered views -> appearance-invariant,
cell-discriminative bits). NYUv2 (resized to the pinned res, RGB->luma) is the
seg source; a synthetic luma fallback validates the architecture with no
download.

  PYTHONPATH=. python3 -m sspax.vision.segnet demo      # synthetic sanity
  PYTHONPATH=. python3 -m sspax.vision.segnet train      # NYUv2 if present
"""
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from sspax.nnconv import Conv, crelu

ROOT = Path(__file__).resolve().parents[2]
NYU = ROOT / "scratch" / "datasets" / "nyu_depth_v2_labeled.mat"
DESC_BITS = 32
LABEL_BITS = 16             # k-bit per-cell label latent (Regime B)
N_CLASS = 40               # NYUv2 40-class benchmark (0 = void/other)
IN_HW = (120, 160)         # HALF res (160x120 W x H); Y8 mono -> 1 channel
CH = 16                    # trunk base width -> 4*CH = 64-ch trunk (pinned)


# --------------------------------------------------------------------------
#  net: shared trunk (/4 -> 30x40x64) + tracking head + seg head, all per-cell
# --------------------------------------------------------------------------
class UnifiedVisionNet(nn.Module):
    ch: int = CH
    n_class: int = N_CLASS
    desc_bits: int = DESC_BITS
    use_crelu: bool = True

    @nn.compact
    def __call__(self, x):                              # x (B,120,160,1)
        act = crelu if self.use_crelu else nn.relu
        # stride EARLY (most work at /4), then 1x1 channel-mixes — keeps the
        # dense 3x3 cost off the big feature maps (the pinned-budget lever).
        x = act(Conv(self.ch, (3, 3), strides=(2, 2))(x))       # /2  60x80
        x = act(Conv(self.ch, (3, 3), strides=(2, 2))(x))       # /4  30x40
        x = act(Conv(self.ch * 2, (3, 3))(x))           # /4 dense 3x3 (RF+cap)
        trunk = act(Conv(self.ch * 2, (1, 1))(x))       # 30x40x64 SHARED trunk
        # TRACKING head (Regime A) @frame rate
        w = Conv(1, (1, 1))(trunk)[..., 0]              # per-cell saliency logit
        cut = Conv(1, (1, 1))(trunk)[..., 0]           # thermometer cutoff logit
        desc = Conv(self.desc_bits, (1, 1))(trunk)     # tracking descriptor
        # SEG head (Regime B) @keyframe rate
        seg = Conv(self.n_class, (1, 1))(trunk)        # per-cell class logits
        return dict(seg=seg, w=w, cut=cut, desc=desc, trunk=trunk)


def binarize(a):
    return (np.asarray(a) > 0).astype(np.int8)


# a fixed near-orthogonal class -> LABEL_BITS codebook (the map-binding code).
def label_codebook(n_class=N_CLASS, bits=LABEL_BITS, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n_class, bits)) > 0.5).astype(np.int8)


# --------------------------------------------------------------------------
#  data — NYUv2 (resized, RGB->luma) if present, else synthetic luma
# --------------------------------------------------------------------------
def _resize_nn(img, hw):
    H, W = hw
    yi = (np.arange(H) * img.shape[0] / H).astype(int)
    xi = (np.arange(W) * img.shape[1] / W).astype(int)
    return img[yi][:, xi]


_LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def load_nyu(hw=IN_HW, max_n=None, rgb=False):
    """NYUv2 labeled .mat -> (img (N,H,W,C) [0,1], labels40 (N,H,W)).
    Deploy-faithful default: RGB collapsed to Y8 luma (1 ch; depth belongs to the
    lidar net). rgb=True keeps 3 channels (input-lever probe) — NOTE: rgb training
    is currently UNSTABLE (diverges to NaN: 3x input energy overflows the
    un-normalised CReLU stack, and luma_jitter assumes [0,1]); needs coordinated
    input standardisation + jitter rework before use. See docs/RESULTS.md
    2026-07-15 "seg bottleneck is NOT capacity".
    Remaps 894 raw ids -> N_CLASS-1 most-frequent + 0=other (deterministic).
    ANTI-ORACLE NOTE (rule 2): the top-K frequency TAXONOMY is chosen over the
    FULL set here, before any train/test split — it touches test labels for
    label-SPACE selection (not predictions). The frequency ranking is
    split-invariant, so ZERO numeric impact on reported accuracies (scoring-space
    setup, not a per-sample leak); fit the taxonomy on the train split if used
    beyond these diagnostics."""
    import h5py
    with h5py.File(NYU, "r") as f:
        imgs, labs = f["images"], f["labels"]
        N = imgs.shape[0] if max_n is None else min(max_n, imgs.shape[0])
        Y, L = [], []
        for i in range(N):
            im = np.array(imgs[i]).transpose(2, 1, 0).astype(np.float32) / 255.0
            px = _resize_nn(im, hw) if rgb else _resize_nn(im @ _LUMA, hw)[..., None]
            Y.append(px.astype(np.float32))
            L.append(_resize_nn(np.array(labs[i]).T, hw).astype(np.int32))
    Y = np.stack(Y); L = np.stack(L)
    ids, cnt = np.unique(L, return_counts=True)
    m = ids > 0
    top = ids[m][np.argsort(-cnt[m])][:N_CLASS - 1]
    remap = np.zeros(int(L.max()) + 1, np.int32)
    for k, tid in enumerate(top):
        remap[int(tid)] = k + 1
    return Y.astype(np.float32), remap[L].astype(np.int32)


def synth_luma(n, hw=IN_HW, seed=0):
    """luma scenes: wall/floor split + furniture boxes; per-pixel class labels
    0..5. 1-channel (Y8). Deterministic."""
    H, W = hw
    rng = np.random.default_rng(seed)
    tone = np.array([0.30, 0.72, 0.45, 0.55, 0.62, 0.38], np.float32)
    X, Yl = [], []
    for _ in range(n):
        lab = np.zeros((H, W), np.int32)
        img = np.ones((H, W), np.float32) * tone[0]
        lab[: H // 2] = 1; img[: H // 2] = tone[1]
        for _ in range(rng.integers(2, 5)):
            c = int(rng.integers(2, 6))
            h0, w0 = rng.integers(H // 3, H - 8), rng.integers(0, W - 24)
            hh, ww = rng.integers(8, 20), rng.integers(12, 28)
            lab[h0:h0 + hh, w0:w0 + ww] = c
            img[h0:h0 + hh, w0:w0 + ww] = tone[c] + rng.normal(0, .02)
        img = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)
        X.append(img[..., None].astype(np.float32)); Yl.append(lab)
    return np.stack(X), np.stack(Yl)


def luma_jitter(x, rng):
    """appearance-only (geometry preserved -> cell correspondence holds):
    brightness/contrast/gamma + noise on the single luma channel."""
    g = rng.uniform(0.8, 1.2, (x.shape[0], 1, 1, 1))
    b = rng.uniform(-0.1, 0.1, (x.shape[0], 1, 1, 1))
    gamma = rng.uniform(0.8, 1.25, (x.shape[0], 1, 1, 1))
    y = np.clip(x, 1e-4, 1.0) ** gamma
    return np.clip(y * g + b + rng.normal(0, 0.02, x.shape), 0, 1).astype(np.float32)


# --------------------------------------------------------------------------
#  losses + eval
# --------------------------------------------------------------------------
def _pool_labels(L):
    return L[:, ::4, ::4]                                # -> 30x40 (/4)


def seg_loss(seg, L4, cw=None):
    ce = optax.softmax_cross_entropy_with_integer_labels(
        seg.reshape(-1, seg.shape[-1]), L4.reshape(-1))
    if cw is not None:                       # class-balanced (inverse-freq; void=0)
        w = cw[L4.reshape(-1)]
        return (ce * w).sum() / (w.sum() + 1e-9)
    return ce.mean()


def desc_loss(dA, dB, tau=0.2, max_cells=256):
    B, h, w, bits = dA.shape
    a = dA.reshape(B, h * w, bits); b = dB.reshape(B, h * w, bits)
    a = a / jnp.maximum(jnp.linalg.norm(a, axis=-1, keepdims=True), 1e-6)
    b = b / jnp.maximum(jnp.linalg.norm(b, axis=-1, keepdims=True), 1e-6)
    idx = jnp.linspace(0, h * w - 1, min(max_cells, h * w)).astype(int)
    a, b = a[:, idx], b[:, idx]
    S = jnp.einsum("bmk,bnk->bmn", a, b) / tau
    lab = jnp.broadcast_to(jnp.arange(a.shape[1]), (B, a.shape[1]))
    return optax.softmax_cross_entropy_with_integer_labels(S, lab).mean()


def _miou(pred, L4, n_class):
    ious = []
    for c in range(n_class):
        p, g = pred == c, L4 == c
        if g.sum() > 0:
            ious.append((p & g).sum() / max((p | g).sum(), 1))
    return float(np.mean(ious)) if ious else 0.0


def train(steps=600, bs=8, lr=2e-3, seed=0, w_desc=0.5, real=None, save=True,
          ch=CH, rgb=False):
    print(f"jax devices: {jax.devices()}", flush=True)
    if real is None:
        real = NYU.exists() and NYU.stat().st_size > 2_500_000_000
    if real:
        X, Y = load_nyu(rgb=rgb); ncls = N_CLASS
        print(f"  NYUv2 (Y8 luma {X.shape[1:]}, {len(X)} imgs, {ncls} classes, "
              f"resized to pinned half-res)", flush=True)
    else:
        X, Y = synth_luma(240, seed=seed); ncls = 6
        print(f"  SYNTHETIC luma {X.shape[1:]}, {len(X)} imgs, {ncls} classes "
              f"(NYUv2 not ready — architecture validation at pinned geometry)",
              flush=True)
    ntr = int(0.8 * len(X))
    Xtr, Ytr, Xte, Yte = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]
    # class-balanced weights (inverse sqrt-freq; void class 0 -> weight 0) so the
    # tiny net cannot collapse to the dominant wall/floor/void classes.
    cnt = np.bincount(_pool_labels(Ytr).ravel(), minlength=ncls).astype(float)
    cw = np.zeros(ncls, np.float32)
    cw[1:] = 1.0 / np.sqrt(cnt[1:] + 1.0)
    cw = jnp.asarray(cw / cw[1:].mean())

    model = UnifiedVisionNet(n_class=ncls, ch=ch)
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1,) + X.shape[1:]))
    n_par = sum(p.size for p in jax.tree.leaves(params))
    opt = optax.adamw(lr); st = opt.init(params)
    tr = model.apply(params, jnp.asarray(Xte[:1]))["trunk"].shape
    print(f"  UnifiedVisionNet params {n_par} (~{n_par/1024:.0f} KB int8; BNN "
          f"{n_par/8/1024:.1f} KB); trunk {tr[1]}x{tr[2]}x{tr[3]} per-cell heads; "
          f"DESC_BITS {DESC_BITS} LABEL_BITS {LABEL_BITS}", flush=True)
    rng = np.random.default_rng(seed)

    @jax.jit
    def step(params, st, xa, xb, L4):
        def loss_fn(p):
            oa = model.apply(p, xa); ob = model.apply(p, xb)
            ls = seg_loss(oa["seg"], L4, cw)
            ld = desc_loss(oa["desc"], ob["desc"])
            return ls + w_desc * ld, (ls, ld)
        (l, (ls, ld)), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
        u, st2 = opt.update(g, st, params)
        return optax.apply_updates(params, u), st2, ls, ld

    t0 = time.time()
    for s in range(steps):
        b = rng.integers(0, ntr, bs)
        xa, xb = luma_jitter(Xtr[b], rng), luma_jitter(Xtr[b], rng)
        L4 = _pool_labels(Ytr[b])
        params, st, ls, ld = step(params, st, jnp.asarray(xa),
                                  jnp.asarray(xb), jnp.asarray(L4))
        if s % 100 == 0 or s == steps - 1:
            print(f"  step {s:4d} seg {float(ls):.3f} desc {float(ld):.3f} "
                  f"t={time.time()-t0:.0f}s", flush=True)

    out = model.apply(params, jnp.asarray(Xte))
    pred = np.asarray(out["seg"]).argmax(-1); L4 = _pool_labels(Yte)
    miou = _miou(pred, L4, ncls)
    nz = L4 > 0                                       # pixel-acc on non-void
    pixacc = float((pred[nz] == L4[nz]).mean()) if nz.any() else float("nan")
    nuse = len(np.unique(pred))
    ob = model.apply(params, jnp.asarray(luma_jitter(Xte, rng)))
    bA, bB = binarize(out["desc"]), binarize(ob["desc"])
    stab = float((bA == bB).mean())
    a = np.asarray(out["desc"])[0].reshape(-1, DESC_BITS)
    bb = np.asarray(ob["desc"])[0].reshape(-1, DESC_BITS)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
    bb = bb / (np.linalg.norm(bb, axis=1, keepdims=True) + 1e-6)
    ret = float(((a @ bb.T).argmax(1) == np.arange(len(a))).mean())
    print(f"\n  EVAL: seg mIoU {miou:.3f} pixacc(non-void) {pixacc:.3f} "
          f"classes-used {nuse}/{ncls} | desc bit-stability {stab:.3f} "
          f"| per-cell retrieval {ret:.3f}", flush=True)
    fpga_cost(X.shape[1:], ncls)
    if save:
        import pickle
        f = ROOT / "scratch" / ("segnet_nyu.pkl" if real else "segnet_synth.pkl")
        pickle.dump(jax.tree.map(np.asarray, params), open(f, "wb"))
        print(f"  saved -> {f}", flush=True)
    return params, model, dict(miou=miou, stab=stab, ret=ret, npar=n_par)


def forward_np(params, x):
    """numpy-only deploy forward. x (H,W,1) -> dict(seg_argmax, w, cut, desc bits)
    all at 30x40. Mirrors UnifiedVisionNet (CReLU, two stride-2, 1x1 heads)."""
    from sspax.nnconv import conv2d as _c
    P = params["params"]

    def cv(name, x, s=1):
        return np.asarray(_c(jnp.asarray(x[None]), jnp.asarray(P[name]["kernel"]),
                             jnp.asarray(P[name]["bias"]), s))[0]

    def cr(x):
        return np.concatenate([np.maximum(x, 0), np.maximum(-x, 0)], -1)
    cs = sorted([k for k in P if k.startswith("Conv_")],
                key=lambda s: int(s.split("_")[1]))
    h = cr(cv(cs[0], x, 2))          # /2  3x3 s2
    h = cr(cv(cs[1], h, 2))          # /4  3x3 s2
    h = cr(cv(cs[2], h))             # /4  1x1 mix
    trunk = cr(cv(cs[3], h))         # /4  1x1 trunk
    w = cv(cs[4], trunk)[..., 0]; cut = cv(cs[5], trunk)[..., 0]
    desc = cv(cs[6], trunk); seg = cv(cs[7], trunk)
    return dict(seg=seg.argmax(-1), w=w, cut=cut,
                desc=(desc > 0).astype(np.int8))


def fpga_cost(in_hw=(IN_HW[0], IN_HW[1], 1), n_class=N_CLASS):
    """MMAC + weight budget. CReLU doubles activation width -> next layer's input
    channels double (accounted). Trunk+track @frame rate; seg head @keyframe."""
    H, W, Cin = in_hw
    ch = CH
    # early stride-2 3x3 (spatial) then 1x1 (channel); CReLU doubles the input
    # width of the following layer (accounted via the 2x/4x factors).
    trunk = [("c0/2", 3, 3, Cin, ch, H // 2, W // 2),
             ("c1/4", 3, 3, 2 * ch, ch, H // 4, W // 4),
             ("mix/4", 3, 3, 2 * ch, 2 * ch, H // 4, W // 4),
             ("trunk/4", 1, 1, 4 * ch, 2 * ch, H // 4, W // 4)]
    track = [("w", 1, 1, 4 * ch, 1, H // 4, W // 4),
             ("cut", 1, 1, 4 * ch, 1, H // 4, W // 4),
             ("desc", 1, 1, 4 * ch, DESC_BITS, H // 4, W // 4)]
    seg = [("seg", 1, 1, 4 * ch, n_class, H // 4, W // 4)]

    def acc(layers):
        mac = wt = 0
        for _, kh, kw, cin, cout, oh, ow in layers:
            w = kh * kw * cin * cout; mac += w * oh * ow; wt += w
        return mac, wt
    tm, tw = acc(trunk); km, kw_ = acc(track); sm, sw = acc(seg)
    print(f"  FPGA @ {H}x{W}x{Cin} (pinned half-res): trunk+track "
          f"{(tm+km)/1e6:.1f} MMAC/frame @frame-rate, seg head {sm/1e6:.1f} "
          f"MMAC @keyframe; weights {(tw+kw_+sw)/1024:.1f}K "
          f"(BNN 1-bit {(tw+kw_+sw)/8/1024:.1f} KB EBR)", flush=True)
    return tm + km, sm, tw + kw_ + sw


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        train(steps=300, real=False)
    elif cmd == "train":
        train(steps=800, real=True)
    elif cmd == "cost":
        fpga_cost()
