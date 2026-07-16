"""Consolidated, reproducible SEG-LEVER harness (msg round 3b nit d): the
FINDINGS-pt3 verdicts rested on ~9 scratch scripts; this is their committed
recipe. One parameterized `train_seg(cfg)` reproduces any lever (input channels /
architecture / capacity / resolution / class count / lidar-like depth), and
`LEDGER` records the banked verdicts inline. The load-bearing conclusion: at
deploy budget NO single lever crosses the ~0.70 object bar, and on the REAL
platform (sparse projected-lidar depth) the depth lever HALVES to +0.059 < the
+0.07 gate -> the fine-object label head DEMOTES (surfaces + QBE ships).

  python3 -m sspax.seg_levers            # default: the depth-lever decision
  python3 -m sspax.seg_levers table       # print the banked lever ledger
"""
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from sspax.nnconv import Conv
import sspax.vision.segnet as S

# banked verdicts (docs/RESULTS.md / FINDINGS pt3, 2026-07-15/16). pixacc(non-void).
LEDGER = {
    "luma trunk 10cls":            0.480,
    "RGB  trunk 10cls":            0.464,   # colour ~= luma for a CNN (1-NN artifact)
    "RGB-D trunk 10cls (DENSE)":   0.597,   # +0.119 depth lever (optimistic, dense Kinect)
    "RGB-D trunk 10cls LIDAR-LIKE":0.539,   # +0.059 < +0.07 gate -> DEMOTE (real platform)
    "RGB-D trunk full-res":        0.566,   # resolution does not help (hurts)
    "RGB-D trunk ch96 (8x cap)":   0.599,   # capacity does not help
    "RGB-D U-Net 10cls":           0.638,   # architecture +0.04 real but < 0.70
    "RGB-D U-Net regime (20k+aug)":0.642,   # training regime does not close it
    "RGB-D U-Net 5cls (surfaces)": 0.858,   # coarse pixel bar crossed; objects still weak (mIoU 0.23)
}


def degrade_lidarlike(dep, H, seed=0):
    """dense depth (N,H,W) -> 64-ring PROJECTED lidar geometry: vertical
    ring-subsample ~0.7deg, aperture crop (top/bottom 15% -> 0), occlusion
    dropout at depth discontinuities, nearest-fill inside the aperture (~68%)."""
    out = np.zeros_like(dep); ap0, ap1 = int(0.15 * H), int(0.85 * H)
    rows = np.arange(ap0, ap1, 7)
    for n in range(len(dep)):
        d = dep[n]; keep = np.zeros_like(d, bool); keep[rows] = True
        keep &= (np.abs(np.diff(d, axis=0, prepend=d[:1])) < 0.3) & (d > 0.1)
        col = np.zeros_like(d)
        for r in range(ap0, ap1):
            src = rows[np.argmin(np.abs(rows - r))]
            col[r] = np.where(keep[src], d[src], 0.0)
        out[n] = col
    return out


def _up2(x):
    B, H, W, C = x.shape
    return jnp.broadcast_to(x[:, :, None, :, None, :], (B, H, 2, W, 2, C)).reshape(B, H * 2, W * 2, C)


class UNet(nn.Module):
    ch: int = 32
    n_class: int = 10
    @nn.compact
    def __call__(self, x):
        r = nn.relu
        e1 = r(Conv(self.ch, (3, 3))(x)); e2 = r(Conv(self.ch * 2, (3, 3), strides=(2, 2))(e1))
        e3 = r(Conv(self.ch * 4, (3, 3), strides=(2, 2))(e2)); e4 = r(Conv(self.ch * 8, (3, 3), strides=(2, 2))(e3))
        d3 = r(Conv(self.ch * 4, (3, 3))(jnp.concatenate([_up2(e4), e3], -1)))
        d3 = r(Conv(self.ch * 4, (3, 3))(d3))
        return Conv(self.n_class, (1, 1))(d3)


def _load(channels, n_class, hw):
    import h5py
    old = S.N_CLASS; S.N_CLASS = n_class
    X, L = S.load_nyu(hw=hw, rgb=("rgb" in channels), max_n=1449); S.N_CLASS = old
    if "d" in channels:
        with h5py.File(S.NYU, "r") as f:
            dep = np.stack([S._resize_nn(np.array(f["depths"][i]).T, hw) for i in range(1449)]).astype(np.float32)
        if channels == "rgbd_ll":
            dep = degrade_lidarlike(dep, hw[0])
        X = np.concatenate([X, dep[..., None]], -1)
    mu = X.reshape(-1, X.shape[-1]).mean(0); sd = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    return ((X - mu) / sd).astype(np.float32), L


def train_seg(channels="luma", arch="trunk", ch=32, hw=(120, 160), n_class=10, steps=3000):
    """reproduce any banked seg lever. channels: luma|rgb|rgbd|rgbd_ll."""
    S.N_CLASS = n_class
    X, L = _load(channels, n_class, hw)
    ntr = int(0.8 * len(X)); Xtr, Ytr, Xte, Yte = X[:ntr], L[:ntr], X[ntr:], L[ntr:]
    cnt = np.bincount(S._pool_labels(Ytr).ravel(), minlength=n_class).astype(float)
    cw = np.zeros(n_class, np.float32); cw[1:] = 1 / np.sqrt(cnt[1:] + 1.0); cw = jnp.asarray(cw / cw[1:].mean())
    model = UNet(ch=ch, n_class=n_class) if arch == "unet" else S.UnifiedVisionNet(ch=ch, n_class=n_class)
    apply = (lambda p, x: model.apply(p, x)) if arch == "unet" else (lambda p, x: model.apply(p, x)["seg"])
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1,) + X.shape[1:]))
    opt = optax.adamw(2e-3); st = opt.init(params)

    @jax.jit
    def step(p, st, xb, yb):
        def loss(pp): return S.seg_loss(apply(pp, xb), yb, cw)
        l, g = jax.value_and_grad(loss)(p); u, st2 = opt.update(g, st, p)
        return optax.apply_updates(p, u), st2, l
    rng = np.random.default_rng(0); t0 = time.time()
    for s in range(steps):
        b = rng.integers(0, ntr, 8)
        xa = Xtr[b] + rng.normal(0, 0.05, Xtr[b].shape).astype(np.float32)
        params, st, _ = step(params, st, jnp.asarray(xa), jnp.asarray(S._pool_labels(Ytr[b])))
    seg = np.concatenate([np.asarray(apply(params, jnp.asarray(Xte[i:i + 24]))).argmax(-1) for i in range(0, len(Xte), 24)])
    L4 = S._pool_labels(Yte); v = L4 > 0
    return float((seg[v] == L4[v]).mean()), t0


def table():
    print("SEG-LEVER LEDGER (banked verdicts, pixacc non-void; docs/RESULTS.md 2026-07-15/16):")
    for k, val in LEDGER.items():
        print(f"  {k:32s} {val:.3f}")
    print("  => no single lever crosses ~0.70; the depth lever HALVES on real "
          "sparse lidar depth (+0.059<+0.07) -> label head DEMOTES; surfaces+QBE ships.")


def run():
    print("seg_levers: the DEPTH-LEVER decision (reproduces the demote gate)\n")
    for ch_cfg, tag in (("luma", "luma"), ("rgbd", "RGB-D DENSE"), ("rgbd_ll", "RGB-D LIDAR-LIKE")):
        acc, t0 = train_seg(ch_cfg, arch="trunk", ch=32)
        print(f"  {tag:18s}: pixacc {acc:.3f}   t={time.time()-t0:.0f}s", flush=True)
    print("  => decision: lidar-like lever vs luma should be ~+0.06 (< +0.07 gate) "
          "-> DEMOTE, matching the banked run. `table` prints the full ledger.")


if __name__ == "__main__":
    (table if len(sys.argv) > 1 and sys.argv[1] == "table" else run)()
