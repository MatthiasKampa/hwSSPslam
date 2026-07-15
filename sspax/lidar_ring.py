"""Unified dual-objective LIDAR net at the DEPLOY ingest geometry (pinned by
commit 34989b1 / TRAINING_PROGRAM.md "Network geometry"): the ring-range raster
with RINGS-AS-CHANNELS (3x1024 full / 3x512 half), NOT a BEV grid (BEV is now a
mechanism-study surface only). Mirror of the vision net (sspax/vision/segnet.py):
a shared 1D-along-azimuth trunk with a TRACKING head (per-bin saliency weight +
descriptor) @20 Hz and a distilled per-bin LABEL head @keyframe (cross-modal
Regime B — placeholder here; school has no lidar semantic labels).

Azimuth is CYCLIC, so a yaw rotation is an exact circular ROLL of the raster —
the cleanest possible rotation model (no re-rasterisation, no permutation
search needed for integer-bin yaw). The tracking head is trained self-supervised
(adjacent frames + azimuth-roll positives; far frames negatives) and gated on the
HONEST adjacent-vs-far place metric — NOT a self-rotation positive (that metric
was shown to be a rotation-triviality artifact, docs/RESULTS.md 2026-07-15 P1
retraction). Learned saliency is compared against uniform at equal budget.

  PYTHONPATH=. python3 -m sspax.lidar_ring demo     # ring raster + net shapes
  PYTHONPATH=. python3 -m sspax.lidar_ring train    # self-supervised gate
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

ROOT = Path(__file__).resolve().parents[1]
RINGS = [16, 33, 50]           # 3 rings spanning the vertical FoV
N_AZ = 1024                    # full-res azimuth bins (half = 512)
CH = 16


# --------------------------------------------------------------------------
#  ring-range raster (rings-as-channels): (1, N_AZ, 2*len(RINGS))
# --------------------------------------------------------------------------
def ring_raster(xyz, ring, rings=RINGS, n_az=N_AZ):
    """nearest-surface range + mean intensity per (ring, azimuth bin).
    channels = [range_r0..range_rk, int_r0..int_rk]; missing bins -> 0."""
    rng_img = np.zeros((n_az, len(rings)), np.float32)
    az = (np.arctan2(xyz[:, 1], xyz[:, 0]) + np.pi) / (2 * np.pi)   # [0,1)
    ab = np.clip((az * n_az).astype(int), 0, n_az - 1)
    rr = np.linalg.norm(xyz[:, :2], axis=1)
    for ci, rg in enumerate(rings):
        m = ring == rg
        if not m.any():
            continue
        a, r = ab[m], rr[m]
        # nearest surface per bin (min range)
        order = np.argsort(r)
        a, r = a[order], r[order]
        first = np.zeros(n_az, np.float32)
        seen = np.full(n_az, False)
        for ai, ri in zip(a, r):
            if not seen[ai]:
                first[ai] = ri; seen[ai] = True
        rng_img[:, ci] = first
    return rng_img[None]                                   # (1, N_AZ, len(rings))


def raster_dirs(n_az=N_AZ):
    """unit (x,y) direction of each azimuth bin centre (for SSP placement)."""
    az = (np.arange(n_az) + 0.5) / n_az * 2 * np.pi - np.pi
    return np.stack([np.cos(az), np.sin(az)], 1)


DIRS2D = raster_dirs()


# --------------------------------------------------------------------------
#  net: shared 1D-azimuth trunk (/4) + tracking head + label head
# --------------------------------------------------------------------------
class UnifiedLidarNet(nn.Module):
    ch: int = CH
    n_class: int = 40
    desc_bits: int = 32
    use_crelu: bool = True

    @nn.compact
    def __call__(self, x):                                 # x (B,1,N_AZ,Cin)
        act = crelu if self.use_crelu else nn.relu
        # NOTE nnconv.Conv uses a SCALAR stride (strides[0]); height is 1 so a
        # (2,2) stride halves azimuth (width) and leaves height at 1 — the way to
        # get asymmetric az-downsampling through the scalar-stride conv.
        x = act(Conv(self.ch, (1, 7), strides=(2, 2))(x))          # /2 az
        x = act(Conv(self.ch, (1, 7), strides=(2, 2))(x))          # /4 az
        x = act(Conv(self.ch * 2, (1, 5))(x))              # /4 context
        trunk = act(Conv(self.ch * 2, (1, 1))(x))          # (B,1,N_AZ/4,64)
        w = Conv(1, (1, 1))(trunk)[..., 0]                 # per-bin saliency
        desc = Conv(self.desc_bits, (1, 1))(trunk)         # tracking descriptor
        lab = Conv(self.n_class, (1, 1))(trunk)            # distilled label logits
        return dict(w=w, desc=desc, lab=lab, trunk=trunk)


# --------------------------------------------------------------------------
#  SSP place encode from per-bin saliency (the tracking use of the net)
# --------------------------------------------------------------------------
OCT6 = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def _W_place(lams=OCT6):
    from sspax import core as C
    from sspax import sphere as SPH
    dirs = SPH.dirs_ring(60, spacing="arc", apportion="const",
                         stagger=0.5, n_rings=6)
    return jnp.asarray(C.W_of(dirs, lams))                 # (360,3)


def encode_bins(W, rng_img, weights):
    """place each azimuth bin's nearest surface (range along its direction) into
    the SSP, weighted by learned saliency. rng_img (1,Naz4,nch) downsampled range,
    weights (Naz4,). Uses ring 0 range as the bin's radial distance."""
    naz = weights.shape[0]
    dirs = jnp.asarray(DIRS2D[::(N_AZ // naz)][:naz])
    r = rng_img[0, :, 0]                                   # ring-0 range per bin
    pts = jnp.concatenate([dirs * r[:, None],
                           jnp.zeros((naz, 1))], 1)        # (naz,3) z=0
    ph = jnp.exp(1j * (pts @ W.T))                         # (naz,360)
    v = (ph * weights[:, None].astype(ph.dtype)).sum(0)
    return v / jnp.maximum(jnp.linalg.norm(v), 1e-12)


# --------------------------------------------------------------------------
#  data (real school_run2 ring rasters) + honest gate
# --------------------------------------------------------------------------
def load_rasters(run="school_run2", n_frames=180):
    import experiments.lattice3d as L3
    import experiments.lidarscale as LS
    pick, kts, pose, kind = L3.sample_kf(run, n_frames, need_ref=True,
                                         labels="est")
    clouds, _ = LS.load_ring_clouds(run, kts, rings=RINGS, sub=1)
    import io
    import pyarrow.parquet as pq
    shards, cts, where = L3.cloud_index(run)
    # re-read with ring field for the raster
    j = np.clip(np.searchsorted(cts, kts), 1, len(cts) - 1)
    j = j - (np.abs(cts[j - 1] - kts) < np.abs(cts[j] - kts))
    rasters, xy, keep = [], [], []
    by_shard = {}
    for i, k in enumerate(j):
        si, row = where[k]; by_shard.setdefault(si, []).append((row, i))
    store = {}
    for si in sorted(by_shard):
        t = pq.read_table(shards[si])
        for row, i in by_shard[si]:
            z = np.load(io.BytesIO(t["npz_bytes"][row].as_py()))
            store[i] = (z["xyz"].astype(np.float64), z["ring"])
    for i in range(len(kts)):
        if i in store:
            xyz, rg = store[i]
            rasters.append(ring_raster(xyz, rg))
            xy.append(pose[i, :2]); keep.append(i)
    print(f"  {run}: {len(rasters)} ring rasters {rasters[0].shape} "
          f"(labels={kind}; est poses label pairs only — anti-oracle)",
          flush=True)
    return np.stack(rasters), np.asarray(xy)


def _auc(pos, neg):
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
    return (r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) \
        / max(len(pos) * len(neg), 1)


def gate(params, model, rasters, xy, seg, W, far=4.0):
    """HONEST place gate: adjacent (same-place) vs far (>4 m) rot-searched sim,
    learned saliency vs uniform. rotation = azimuth roll (exact)."""
    naz4 = model.apply(params, jnp.asarray(rasters[:1]))["w"].shape[2]

    def enc(i, uniform):
        o = model.apply(params, jnp.asarray(rasters[i:i + 1]))
        w = jnp.ones(naz4) if uniform else jax.nn.softplus(o["w"][0, 0])
        rimg = jax.image.resize(jnp.asarray(rasters[i]),
                                (1, naz4, rasters.shape[-1]), "nearest")
        return encode_bins(W, rimg, w)
    rolls = [int(round(k * naz4 / 12)) for k in range(12)]

    def sim(a, b):
        return max(float(jnp.abs(jnp.vdot(jnp.roll(a, r), b))) for r in rolls)
    res = {}
    for name, uni in [("uniform", True), ("learned", False)]:
        V = [enc(i, uni) for i in seg]
        n = len(seg)
        d = np.linalg.norm(xy[seg][:, None] - xy[seg][None], axis=2)
        S = np.array([[sim(V[a], V[b]) for b in range(n)] for a in range(n)])
        adj = np.array([S[i, i + 1] for i in range(n - 1)])
        fo = d > far
        res[name] = _auc(adj, S[fo])
    print(f"    place gate (adjacent vs far>{far}m, azimuth-roll searched): "
          f"uniform {res['uniform']:.3f}  learned {res['learned']:.3f}  "
          f"(delta {res['learned']-res['uniform']:+.3f})", flush=True)
    return res


def demo():
    rasters, xy = load_rasters(n_frames=40)
    model = UnifiedLidarNet()
    params = model.init(jax.random.PRNGKey(0), jnp.asarray(rasters[:1]))
    npar = sum(p.size for p in jax.tree.leaves(params))
    o = model.apply(params, jnp.asarray(rasters[:2]))
    print(f"  UnifiedLidarNet params {npar} (BNN {npar/8/1024:.1f} KB); "
          f"trunk {o['trunk'].shape}, w {o['w'].shape}, desc {o['desc'].shape}",
          flush=True)
    fpga_cost()


def fpga_cost(n_az=N_AZ, ncls=40):
    ch = CH
    layers = [("c0/2", 1, 7, len(RINGS), ch, n_az // 2),
              ("c1/4", 1, 7, 2 * ch, ch, n_az // 4),
              ("c2/4", 1, 5, 2 * ch, 2 * ch, n_az // 4),
              ("trunk", 1, 1, 4 * ch, 2 * ch, n_az // 4),
              ("w", 1, 1, 4 * ch, 1, n_az // 4),
              ("desc", 1, 1, 4 * ch, 32, n_az // 4),
              ("lab", 1, 1, 4 * ch, ncls, n_az // 4)]
    mac = wt = 0
    for _, kh, kw, cin, cout, ow in layers:
        w = kh * kw * cin * cout; mac += w * ow; wt += w
    print(f"  FPGA @ {len(RINGS)}x{n_az} ring raster: {mac/1e6:.2f} MMAC/frame "
          f"(@20 Hz {mac*20/1e6:.0f} MMAC/s), {wt/1024:.1f}K weights "
          f"(BNN {wt/8/1024:.1f} KB EBR)", flush=True)
    return mac, wt


def train(run="school_run2", n_frames=180, steps=400, bs=16, lr=3e-3, seed=0):
    print(f"jax devices: {jax.devices()}", flush=True)
    rasters, xy = load_rasters(run, n_frames)
    n = len(rasters); split = int(0.6 * n)
    train_seg = np.arange(0, split); eval_seg = np.arange(split, n)
    W = _W_place()
    model = UnifiedLidarNet()
    params = model.init(jax.random.PRNGKey(seed), jnp.asarray(rasters[:1]))
    npar = sum(p.size for p in jax.tree.leaves(params))
    naz4 = model.apply(params, jnp.asarray(rasters[:1]))["w"].shape[2]
    opt = optax.adam(lr); st = opt.init(params)
    print(f"  UnifiedLidarNet params {npar}; TIME split train[0,{split}) "
          f"eval[{split},{n}); naz/4={naz4}", flush=True)

    nch = rasters.shape[-1]
    rolls = [int(round(k * naz4 / 8)) for k in range(8)]

    def enc_batch(p, X):                                   # X (B,1,N_AZ,nch)
        o = model.apply(p, X)
        w = jax.nn.softplus(o["w"][:, 0])                  # (B, naz4)
        rimg = jax.image.resize(X, (X.shape[0], 1, naz4, nch), "nearest")
        return jax.vmap(lambda ri, wi: encode_bins(W, ri, wi))(rimg, w)

    @jax.jit
    def loss_step(p, st, xa, xb, negmask):
        def loss(pp):
            va = enc_batch(pp, xa); vb = enc_batch(pp, xb)
            S = jnp.max(jnp.stack([jnp.abs(jnp.roll(va, r, axis=1) @ vb.conj().T)
                                   for r in rolls]), 0)
            eye = jnp.eye(xa.shape[0], dtype=bool)
            S = jnp.where(eye | negmask, S / 0.1, -1e9)
            return optax.softmax_cross_entropy_with_integer_labels(
                S, jnp.arange(xa.shape[0])).mean()
        l, g = jax.value_and_grad(loss)(p)
        u, st2 = opt.update(g, st, p)
        return optax.apply_updates(p, u), st2, l

    print("  pre:"); gate(params, model, rasters, xy, eval_seg, W)
    rng = np.random.default_rng(seed); t0 = time.time()
    for s in range(steps):
        ia = rng.choice(train_seg, bs, replace=False)
        ib = np.clip(ia + rng.integers(-1, 2, bs), train_seg[0], train_seg[-1])
        d = np.linalg.norm(xy[ia][:, None] - xy[ia][None], axis=2)
        params, st, l = loss_step(params, st, jnp.asarray(rasters[ia]),
                                  jnp.asarray(rasters[ib]), jnp.asarray(d > 4.0))
        if s % 100 == 0 or s == steps - 1:
            print(f"    step {s} loss {float(l):.3f} t={time.time()-t0:.0f}s",
                  flush=True)
    print("  post:"); res = gate(params, model, rasters, xy, eval_seg, W)
    import pickle
    pickle.dump(jax.tree.map(np.asarray, params),
                open(ROOT / "scratch" / "lidar_ring.pkl", "wb"))
    return params, model, res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    {"demo": demo, "train": lambda: train()}[cmd]()
