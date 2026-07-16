"""P1 round 4 (PREP now, fire when the camera lands): platform-domain finetune of
the vision DESC head on OV5640 imagery — the query-by-example quality lever
(the gated head transfers real-but-PARTIAL cross-dataset; the lever is
platform-domain adaptation, not more NYUv2). Label-FREE luma-jitter InfoNCE on
the desc head; two arms (trunk FROZEN vs trunk LOW-LR, both printed); then
re-export via headio v2. DATALESS-RUNNABLE: uses build/snap.npz (hw_snap.py 'S',
key "img", 320x240 Y8) if present, else a synthetic luma smoke set — so it is
ready before the camera is.

Gate = the deploy-box pre-finetune reference (msg round 4): desc stability
0.750 adj / 0.520 far; objmap2 desc-key AUC ~0.59, err ~0.6 m. Report post vs
pre so the finetune's yield is measurable.

  PYTHONPATH=. python3 -m sspax.platform_finetune          # smoke (synthetic)
  PYTHONPATH=. python3 -m sspax.platform_finetune build/snap.npz   # real frames
"""
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

import sspax.vision.segnet as S
import sspax.headio as H

ROOT = Path(__file__).resolve().parents[1]
PKL = ROOT / "scratch" / "segnet_nyu.pkl"
OUT = ROOT / "sspax" / "artifacts" / "vision_head_ft.npz"


def load_frames(path=None, n_smoke=200):
    if path and Path(path).exists():
        img = np.load(path)["img"].astype(np.float32)        # (N,240,320) Y8
        if img.max() > 1.5:
            img = img / 255.0
        X = np.stack([S._resize_nn(f, S.IN_HW) for f in img])[..., None]
        print(f"  loaded {len(X)} OV5640 frames from {path} -> {X.shape}", flush=True)
        return X.astype(np.float32)
    X, _ = S.synth_luma(n_smoke, seed=0)                       # dataless smoke
    print(f"  SYNTHETIC luma smoke {X.shape} (no camera yet — harness dry-run)",
          flush=True)
    return X


def _descmask(params, mode):
    """grad-scale tree: desc head (Conv_6) full LR; trunk (Conv_0..3) 0 (frozen)
    or 0.1 (low-LR); other heads 0 (we only touch the desc channel)."""
    def scale(path):
        name = path[1].key if len(path) > 1 else ""
        if name == "Conv_6":
            return 1.0
        if name in ("Conv_0", "Conv_1", "Conv_2", "Conv_3"):
            return 0.1 if mode == "lowlr" else 0.0
        return 0.0
    flat = jax.tree_util.tree_leaves_with_path(params)
    return jax.tree_util.tree_map_with_path(lambda p, x: scale(p) * 0 + scale(p), params)


def _retr_stab(model, params, X, rng):
    o = model.apply(params, jnp.asarray(X[:1]))
    ob = model.apply(params, jnp.asarray(S.luma_jitter(X[:1], rng)))
    a = np.asarray(o["desc"])[0].reshape(-1, 32); b = np.asarray(ob["desc"])[0].reshape(-1, 32)
    stab = float(((a > 0) == (b > 0)).mean())
    an = a / (np.linalg.norm(a, 1, keepdims=True) + 1e-6)
    bn = b / (np.linalg.norm(b, 1, keepdims=True) + 1e-6)
    ret = float(((an @ bn.T).argmax(1) == np.arange(len(an))).mean())
    return ret, stab


def finetune(X, mode, steps=400, lr=1e-3, seed=0):
    import pickle
    model = S.UnifiedVisionNet(n_class=40)
    params = pickle.load(open(PKL, "rb"))
    smask = _descmask(params, mode)
    opt = optax.adamw(lr); st = opt.init(params)
    rng = np.random.default_rng(seed)
    pre = _retr_stab(model, params, X, rng)

    @jax.jit
    def step(p, st, xa, xb):
        def loss(pp):
            return S.desc_loss(model.apply(pp, xa)["desc"], model.apply(pp, xb)["desc"])
        l, g = jax.value_and_grad(loss)(p)
        g = jax.tree_util.tree_map(lambda gr, m: gr * m, g, smask)   # freeze/low-LR
        u, st2 = opt.update(g, st, p)
        return optax.apply_updates(p, u), st2, l
    for s in range(steps):
        b = rng.integers(0, len(X), 8)
        xa = S.luma_jitter(X[b], rng); xb = S.luma_jitter(X[b], rng)
        params, st, _ = step(params, st, jnp.asarray(xa), jnp.asarray(xb))
    post = _retr_stab(model, params, X, rng)
    return params, pre, post


def export(params):
    P = params["params"]
    def qc(name, k):
        w = np.asarray(P[name]["kernel"]).transpose(3, 2, 0, 1)
        b = np.asarray(P[name]["bias"]).astype(np.float32)
        s = (np.abs(w).reshape(w.shape[0], -1).max(1) / 127 + 1e-12).astype(np.float32)
        wi = np.clip(np.round(w / s[:, None, None, None]), -127, 127).astype(np.int8)
        return wi, s, b, {"kind": "conv", "k": k, "cin": w.shape[1], "cout": w.shape[0]}
    trunk = [("Conv_0", 3, 2), ("Conv_1", 3, 2), ("Conv_2", 3, 1), ("Conv_3", 1, 1)]
    meta = dict(version=2, modality="vision", in_h=120, in_w=160, in_ch=1,
                in_div=255.0, act="crelu", cell=4, desc_bits=32, k_bits=0, seg=[], thresh=[])
    meta["trunk"] = [dict(qc(n, k)[3], s=s) for n, k, s in trunk]
    meta["track"] = [dict(qc("Conv_4", 1)[3], s=1)]
    meta["desc"] = [dict(qc("Conv_6", 1)[3], s=1)]
    arrays = {}
    for pre_, items in (("T", trunk), ("A", [("Conv_4", 1, 1)]), ("D", [("Conv_6", 1, 1)])):
        for i, (n, k, _) in enumerate(items):
            wi, s, b, _ = qc(n, k); arrays[f"{pre_}{i}_w"] = wi
            arrays[f"{pre_}{i}_s"] = s; arrays[f"{pre_}{i}_b"] = b
    H.save_head(OUT, meta, arrays); H.load_head(OUT)          # validates
    return OUT


def run(path=None):
    print("PLATFORM-DOMAIN desc finetune (QBE lever; pre-finetune ref: "
          "stability 0.750/0.520, objmap2 ~0.59):", flush=True)
    X = load_frames(path)
    print(f"  {'arm':>14}  {'retr pre->post':>16}  {'stab pre->post':>16}")
    for mode in ("frozen (desc-only)", "lowlr (trunk 0.1x)"):
        params, pre, post = finetune(X, mode.split()[0])
        print(f"  {mode:>14}  {pre[0]:.3f} -> {post[0]:.3f}      "
              f"{pre[1]:.3f} -> {post[1]:.3f}", flush=True)
    out = export(params)
    print(f"  re-exported -> {out.name} (headio v2, validates)", flush=True)
    print("  => harness READY. On the camera day: point at build/snap.npz; the "
          "arm that improves objmap2 (deploy-side gate) is the ship head. "
          "Synthetic smoke is a dry-run — real yield needs OV5640 frames.")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
