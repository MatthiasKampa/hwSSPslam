"""Bridge: pretrained vision CNN -> semantic SSP map, with REAL images.

Closes the loop the directive asked for: a CNN pretrained on external images
(CIFAR-100) classifies each object; its predicted class -> the class's binary
role code; the softmax CONFIDENCE sets how many of the class's k bits are
committed (significance = confidence, per the user's "modulate significance by
setting bits"); bind with position and bundle. Query "chair" -> localize the
real chairs in the map. Cross-talk = the classifier's real confusions.

  python3 -m sspax.vision.semantic_bridge        # needs scratch/vision_cnn.pkl
"""
import pickle
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from sspax import semantic as SEM
from sspax.vision.tinycnn import TinyCNN, load, FURNITURE

ROOT = Path(__file__).resolve().parents[2]
PKL = ROOT / "scratch" / "vision_cnn.pkl"


def load_model():
    d = pickle.load(open(PKL, "rb"))
    return TinyCNN(), d["params"], d["classes"]


def _predict(model, params, X, bs=500):
    out = []
    for j in range(0, len(X), bs):
        logits, _ = model.apply(params, jnp.asarray(X[j:j + bs]))
        out.append(np.asarray(jax.nn.softmax(logits, -1)))
    return np.concatenate(out)


def demo(n_obj=16, seed=0, k=12, conf_sig=True):
    model, params, classes = load_model()
    _, _, Xte, yte, _ = load()
    probs = _predict(model, params, Xte)
    pred = probs.argmax(1)
    fidx = {c: classes.index(c) for c in FURNITURE}
    codes = SEM.class_codes(FURNITURE, k=k)

    # sample n_obj furniture images, place at random positions, bind predicted
    # class (confidence -> committed bit count)
    rng = np.random.default_rng(seed)
    furn_mask = np.isin(yte, list(fidx.values()))
    pool = np.flatnonzero(furn_mask)
    pick = rng.choice(pool, n_obj, replace=False)
    inv = {v: kk for kk, v in fidx.items()}
    objs = []
    mp = np.zeros(SEM.D, complex)
    for idx in pick:
        true_cls = inv[int(yte[idx])]
        p_cls = pred[idx]
        name = inv.get(int(p_cls), None)          # predicted furniture class?
        conf = float(probs[idx, p_cls])
        xy = rng.uniform([-5.5, -4.5], [5.5, 4.5])
        xyz = np.array([xy[0], xy[1], 0.5])
        objs.append((true_cls, xyz[:2]))
        if name is None:
            continue                              # predicted a non-furniture class
        nbits = max(1, int(round(conf * k))) if conf_sig else k
        mp += SEM.ROLES[codes[name][:nbits]].sum(0) * SEM.pos_ssp(xyz)

    grid, shape = SEM._grid((12.0, 10.0))
    dens = SEM.query(mp, codes["chair"], grid)
    pk = SEM.detect(dens, grid, shape)
    chairs = [xy for c, xy in objs if c == "chair"]
    tp, fp, errs = SEM._match(pk, chairs)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(len(chairs), 1)
    # classifier chair quality on this pool (upper bound the map can inherit)
    cm = yte[pick] == fidx["chair"]
    chair_prec = np.mean(yte[pick][pred[pick] == fidx["chair"]] == fidx["chair"]) \
        if (pred[pick] == fidx["chair"]).any() else float("nan")
    chair_rec = np.mean(pred[pick][cm] == fidx["chair"]) if cm.any() else float("nan")
    print(f"vision->semantic 'chair' query ({n_obj} real furniture images, "
          f"D={SEM.D}, k={k}, conf_sig={conf_sig}):")
    print(f"  classifier chair  precision {chair_prec:.2f}  recall {chair_rec:.2f}")
    print(f"  map localization  precision {prec:.2f}  recall {rec:.2f}  "
          f"pos err med {np.median(errs) if errs else float('nan'):.2f} m "
          f"({tp} TP / {fp} FP, {len(chairs)} true chairs)")
    return prec, rec


if __name__ == "__main__":
    print(f"jax devices: {jax.devices()}")
    if not PKL.exists():
        print(f"{PKL} not found — train the vision CNN first "
              f"(python3 -m sspax.vision.tinycnn train).")
    else:
        demo()
        print()
        demo(conf_sig=False)
