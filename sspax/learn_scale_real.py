"""P1 (msg.txt round 2, CENTERPIECE): the REAL-DATA transfer gate for the
learned thermometer scale-modulation head (sspax/learn_scale.py gave the honest
synthetic NEGATIVE — box-rooms have no aliasing regime). Here the head is trained
and evaluated on REAL building-scale lidar (school_run2), where the banked
real-data law says fine rings alias under rotation and coarse rings carry the
place lever — exactly the regime where per-cell ring-blanking should help.

DISCIPLINE
- Anti-oracle (rule 2): own-estimate poses (BandSLAM `fin`, DIAGNOSTIC labels)
  only label pairs; they NEVER enter a raster, the net, or an encoded vector.
  Stated per-run in the log. run2 has no honest revisits (banked) so this is an
  est-DIAGNOSTIC gate, not an honest place number.
- TIME split (rule: no random splits — they self-leak at keyframe rate): train
  on the first 60% of the traverse, evaluate on pairs drawn ENTIRELY from the
  last 40%.
- Positives = temporally-adjacent frames (+-1-2 kf) OR a synthetic yaw of the
  SAME cloud, RE-RASTERISED (real raster effects, not a phase rotation).
  Negatives = frames >4 m apart in own-estimate. rot-NCE over the 24-perm yaw
  search (per-ring cyclic shift).
- Gates on the held-out segment: learned blanking vs (i) full ladder equal D,
  (ii) fixed global cutoffs, (iii) the ADOPTED fixed recipes (ring-coarse16
  D1920, azel-oct6 D240). Beating (i)+(ii) => mechanism transfers.

  PYTHONPATH=. python3 -m sspax.learn_scale_real train
"""
import sys
import time
import pickle
from pathlib import Path
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import optax

from sspax.learn_scale import ScaleNet
from sspax import core as C
from sspax import sphere as SPH
import experiments.lattice3d as L3
import experiments.lidarscale as LS

# ---- lattice: staggered-ring, full 6-ring ladder 0.25..90.5 m (D=360) --------
FULL_LAMS = list(np.geomspace(0.25, 90.5, 6))
RINGKW = dict(spacing="arc", apportion="const", stagger=0.5, n_rings=6)
N_DIR = 60
N_RING = 6
DIRS = SPH.dirs_ring(N_DIR, **RINGKW)
W_LAT = jnp.asarray(C.W_of(DIRS, FULL_LAMS))            # (360, 3), lam-major

# building-scale raster (clouds are clipped r<60 m -> +-60 m span); G=64 keeps
# ~3.75 m encoded cells so the fine rings have real structure to bite on (a
# 5 m raster starves the blanking test — the fine rungs can't resolve anything).
EXTENT = 120.0
G = 64
GD = G // 2

# 24-yaw per-ring permutation search (exact cyclic shift per ring)
_YAWP = [jnp.asarray(SPH.yaw_perm(N_DIR, k * 2 * np.pi / 24, n_lams=N_RING,
                                  **RINGKW)[0]) for k in range(24)]

FAR_LO = 4.0            # own-estimate metres: a valid negative
SEED = 0


# --------------------------------------------------------------------------
#  raster + cells (fixed building extent, sensor frame)
# --------------------------------------------------------------------------
def rasterize(cloud, extent=EXTENT, g=G):
    r = extent / 2
    xy = cloud[:, :2]
    ix = np.clip(((xy[:, 0] + r) / (2 * r) * g).astype(int), 0, g - 1)
    iy = np.clip(((xy[:, 1] + r) / (2 * r) * g).astype(int), 0, g - 1)
    occ = np.zeros((g, g)); hmax = np.full((g, g), -1e3); hsum = np.zeros((g, g))
    np.add.at(occ, (ix, iy), 1.0)
    np.maximum.at(hmax, (ix, iy), cloud[:, 2])
    np.add.at(hsum, (ix, iy), cloud[:, 2])
    hmax[hmax < -1e2] = 0.0
    hmean = np.where(occ > 0, hsum / np.maximum(occ, 1), 0.0)
    cell_m = np.log(extent / GD)                        # log-metric channel
    return np.stack([np.log1p(occ), hmax, hmean,
                     np.full((g, g), cell_m)], -1).astype(np.float64)


def cells_of(extent=EXTENT):
    r = extent / 2
    c = (np.arange(GD) + 0.5) / GD * (2 * r) - r
    cx, cy = np.meshgrid(c, c, indexing="ij")
    return np.stack([cx.ravel(), cy.ravel(), np.zeros(GD * GD)], -1)


CELLS = jnp.asarray(cells_of())


def yaw_cloud(cloud, theta):
    ct, st = np.cos(theta), np.sin(theta)
    R = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
    return cloud @ R.T


# --------------------------------------------------------------------------
#  thermometer encode (per-cell weight + per-cell ring cutoff)
# --------------------------------------------------------------------------
def encode_mod(params, model, grid, beta, hard=False, force_cut=None):
    """SSP with per-cell thermometer ring-blanking. force_cut overrides the net
    (a scalar global cutoff) -> the fixed-ladder / fixed-cutoff baselines share
    this exact encode path (only the mask differs)."""
    wlog, cutlog = model.apply(params, grid)
    B = grid.shape[0]
    w = jax.nn.softplus(wlog).reshape(B, -1)
    if force_cut is None:
        cutoff = (N_RING * jax.nn.sigmoid(cutlog)).reshape(B, -1)
    else:
        cutoff = jnp.full((B, GD * GD), float(force_cut))
    r = jnp.arange(N_RING)
    if hard:
        mask = (r[None, None, :] >= jnp.round(cutoff)[..., None]).astype(grid.dtype)
    else:
        mask = jax.nn.sigmoid(beta * (r[None, None, :] - cutoff[..., None]))
    pts = jnp.broadcast_to(CELLS, (B,) + CELLS.shape)
    ph = jnp.exp(1j * jnp.einsum("bnd,kd->bnk", pts, W_LAT))
    ph = ph.reshape(B, -1, N_RING, N_DIR) * mask[..., None]
    v = jnp.einsum("bnrd,bn->brd", ph, w.astype(ph.dtype)).reshape(B, -1)
    return v / jnp.maximum(jnp.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def rot_sim_mat(va, vb):
    """max over 24 yaw perms of |<va_i, vb_j>| -> (A,B) rot-searched sim."""
    return jnp.max(jnp.stack([jnp.abs(va[:, p] @ vb.conj().T)
                              for p in _YAWP]), 0)


def masked_nce(va, vb, negmask, tau=0.1):
    """InfoNCE, diagonal positive; off-diagonal negatives GATED by negmask
    (True = a valid >FAR_LO negative). Invalid off-diagonals -> -inf logit."""
    S = rot_sim_mat(va, vb) / tau
    B = S.shape[0]
    eye = jnp.eye(B, dtype=bool)
    keep = eye | negmask
    S = jnp.where(keep, S, -1e9)
    return optax.softmax_cross_entropy_with_integer_labels(
        S, jnp.arange(B)).mean()


# --------------------------------------------------------------------------
#  real data: clouds + own-estimate poses, time-ordered, split 60/40
# --------------------------------------------------------------------------
def load_real(run="school_run2", n_frames=180):
    pick, kts, pose, kind = L3.sample_kf(run, n_frames, need_ref=True,
                                         labels="est")
    full, _ = L3.load_clouds(run, kts)
    keep = [i for i, c in enumerate(full)
            if c is not None and len(c) > 300]
    clouds = [full[i] for i in keep]
    xy = pose[keep][:, :2]                              # own-estimate (DIAGNOSTIC)
    print(f"  real {run}: labels={kind}; {len(clouds)} usable frames "
          f"(anti-oracle: est poses label pairs only, never encoded)",
          flush=True)
    return clouds, xy


# --------------------------------------------------------------------------
#  training
# --------------------------------------------------------------------------
def make_batch(rasters, clouds, xy, seg, bs, rng, yaw_frac=0.5):
    """bs anchors spread across the training segment. positive = adjacent-kf
    raster OR yaw-reraster of the same cloud (real raster effects, cached).
    negmask = own-estimate dist > FAR_LO (a valid negative)."""
    last = int(seg[-1])
    idx = rng.choice(seg, size=bs, replace=False)
    A, B = [], []
    for i in idx:
        i = int(i)
        A.append(rasters[i])
        if rng.random() < yaw_frac or (i + 1) > last:
            th = float(rng.uniform(-np.pi, np.pi))
            key = (i, round(th, 3))
            if key not in _YCACHE:
                _YCACHE[key] = rasterize(yaw_cloud(clouds[i], th))
            B.append(_YCACHE[key])
        else:
            B.append(rasters[i + 1])                    # temporally adjacent
    d = np.linalg.norm(xy[idx][:, None] - xy[idx][None], axis=2)
    negmask = jnp.asarray(d > FAR_LO)
    return jnp.asarray(np.stack(A)), jnp.asarray(np.stack(B)), negmask


_YCACHE = {}


def train(run="school_run2", n_frames=180, steps=500, bs=24, lr=3e-3,
          seed=SEED, save=True, yaw_frac=0.5, tag=""):
    print(f"jax devices: {jax.devices()}", flush=True)
    clouds, xy = load_real(run, n_frames)
    n = len(clouds)
    rasters = [rasterize(c) for c in clouds]
    split = int(0.6 * n)
    train_seg = np.arange(0, split)
    eval_seg = np.arange(split, n)
    print(f"  TIME split: train frames [0,{split}) eval [{split},{n})",
          flush=True)

    model = ScaleNet()
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, G, G, 4)))
    n_par = sum(x.size for x in jax.tree.leaves(params))
    opt = optax.adam(lr); st = opt.init(params)
    print(f"  ScaleNet params {n_par}; ladder 0.25-90.5 m x6 D={W_LAT.shape[0]}; "
          f"extent {EXTENT} m raster {G}x{G} cells {GD}x{GD}", flush=True)

    @jax.jit
    def step(params, st, A, B, negmask, beta):
        def loss(p):
            va = encode_mod(p, model, A, beta)
            vb = encode_mod(p, model, B, beta)
            return masked_nce(va, vb, negmask)
        l, g = jax.value_and_grad(loss)(params)
        u, st2 = opt.update(g, st, params)
        return optax.apply_updates(params, u), st2, l

    rng = np.random.default_rng(seed)
    print("\n  pre-train gate:", flush=True)
    gate(params, model, rasters, clouds, xy, eval_seg)
    t0 = time.time(); losses = []
    for s in range(steps):
        A, B, negmask = make_batch(rasters, clouds, xy, train_seg, bs, rng,
                                   yaw_frac=yaw_frac)
        beta = 2.0 + 8.0 * min(1.0, s / (steps * 0.6))
        params, st, l = step(params, st, A, B, negmask, beta)
        losses.append(float(l))
        if s % 100 == 0 or s == steps - 1:
            print(f"  step {s:4d} loss {float(l):.3f} beta {beta:.1f} "
                  f"t={time.time()-t0:.0f}s", flush=True)

    print("\n  post-train gate:", flush=True)
    res = gate(params, model, rasters, clouds, xy, eval_seg, recipes=True)
    _cutoff_hist(params, model, rasters, eval_seg)
    if save:
        out = Path(__file__).resolve().parents[1] / "scratch" / \
            f"learned_scale_real{tag}.pkl"
        pickle.dump({"params": jax.tree.map(np.asarray, params),
                     "losses": losses, "gate": res}, open(out, "wb"))
        print(f"  saved -> {out}", flush=True)
    return params, model, res


# --------------------------------------------------------------------------
#  gate: rotation-searched same(self-rot + adjacent)-vs-far AUC
# --------------------------------------------------------------------------
def _auc(pos, neg):
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
    return (r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) \
        / max(len(pos) * len(neg), 1)


def _encode_all(enc_fn, seg):
    return jnp.stack([enc_fn(i) for i in seg])


def _gate_auc(V, Vr, xy_seg, test_yaw=True):
    """same = self-under-rotation (+ adjacent) ; diff = est>FAR_LO."""
    S = np.asarray(rot_sim_mat(V, V))
    n = len(V)
    d = np.linalg.norm(xy_seg[:, None] - xy_seg[None], axis=2)
    far = d > FAR_LO
    # adjacent same-place (gap 1) rot-searched
    adj = np.array([S[i, i + 1] for i in range(n - 1)])
    negs = S[far]
    out = {"adj": _auc(adj, negs)}
    if test_yaw:
        Sr = np.asarray(rot_sim_mat(V, Vr))
        selfrot = np.diag(Sr)                           # frame vs its rotated self
        out["selfrot"] = _auc(selfrot, negs)
    return out


def gate(params, model, rasters, clouds, xy, seg, recipes=False):
    xy_seg = xy[seg]
    Rseg = jnp.asarray(np.stack([rasters[i] for i in seg]))
    # rotated-self rasters (fixed 90 deg test rotation, re-rastered)
    Rrot = jnp.asarray(np.stack([rasterize(yaw_cloud(clouds[i], np.pi / 2))
                                 for i in seg]))

    def enc_mask(grid, cut):
        return encode_mod(params, model, grid, beta=8.0, hard=True,
                          force_cut=cut)
    rows = []
    # (i) full ladder (cutoff 0), (ii) fixed global cutoffs 1..3
    for label, cut in [("full-ladder c0 ", 0), ("fixed cutoff c1", 1),
                       ("fixed cutoff c2", 2), ("fixed cutoff c3", 3)]:
        V = enc_mask(Rseg, cut); Vr = enc_mask(Rrot, cut)
        a = _gate_auc(V, Vr, xy_seg)
        rows.append((label, a))
    # LEARNED
    Vl = encode_mod(params, model, Rseg, beta=8.0, hard=True)
    Vlr = encode_mod(params, model, Rrot, beta=8.0, hard=True)
    al = _gate_auc(Vl, Vlr, xy_seg)
    rows.append(("LEARNED thermo ", al))

    print(f"    {'method':16s} {'selfrot-AUC':>12s} {'adjacent-AUC':>13s}",
          flush=True)
    for label, a in rows:
        print(f"    {label:16s} {a.get('selfrot', float('nan')):12.3f} "
              f"{a['adj']:13.3f}", flush=True)

    res = {label: a for label, a in rows}
    if recipes:
        _recipe_refs(clouds, xy_seg, seg, res)
    return res


def _recipe_refs(clouds, xy_seg, seg, res):
    """(iii) ADOPTED fixed recipes on the SAME eval frames/pairs, each with its
    proper rotation search. Reference bars (different lattices)."""
    OCT6 = FULL_LAMS
    d = np.linalg.norm(xy_seg[:, None] - xy_seg[None], axis=2)
    far = d > FAR_LO

    def ring_W(nd, lams):
        dirs = SPH.dirs_ring(nd, **RINGKW)
        return np.concatenate([(2 * np.pi / lam) * dirs for lam in lams]), nd

    def ring_S(W, nd, nlam):
        V = np.stack([L3.encode(W, clouds[i]) for i in seg])
        mats = [np.abs(V @ V.conj().T)]
        for k in range(1, 24):
            th = k * 2 * np.pi / 24
            perm = SPH.yaw_perm(nd, th, n_lams=nlam, **RINGKW)[0]
            mats.append(np.abs(V[:, perm] @ V.conj().T))
        return np.max(np.stack(mats), 0)

    refs = [
        ("ring-coarse16 D1920", *ring_W(120, LS.LAMC16)),
        ("azel-oct6     D240 ", None, None),
    ]
    print("    -- adopted recipe reference bars (adjacent-AUC) --", flush=True)
    for name, W, nd in refs:
        if nd is not None:
            S = ring_S(W, nd, len(W) // nd)
        else:
            W = LS.azel_W(8, [-40, -20, 0, 20, 40], OCT6)
            V = np.stack([L3.encode(W, clouds[i]) for i in seg])
            S = LS.rot_sim(V, W, 24)
        n = len(seg)
        adj = np.array([S[i, i + 1] for i in range(n - 1)])
        auc = _auc(adj, S[far])
        print(f"    {name} (D{len(W):4d}): adjacent-AUC {auc:.3f}", flush=True)
        res[name] = {"adj": auc}


def _cutoff_hist(params, model, rasters, seg):
    R = jnp.asarray(np.stack([rasters[i] for i in seg]))
    _, cutlog = model.apply(params, R)
    cut = np.asarray(N_RING * jax.nn.sigmoid(cutlog)).ravel()
    print("\n  learned cutoff histogram (0=keep all rings .. 6=blank all):",
          flush=True)
    h, edges = np.histogram(cut, bins=6, range=(0, 6))
    for k in range(6):
        bar = "#" * int(40 * h[k] / max(h.max(), 1))
        print(f"    [{edges[k]:.0f},{edges[k+1]:.0f}) {h[k]:5d} {bar}",
              flush=True)
    print(f"  mean cutoff {cut.mean():.2f}  (const => single-venue "
          f"redundancy; spread => cells buy different fineness)", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train()
