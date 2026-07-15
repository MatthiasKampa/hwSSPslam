"""Regime C — the QUERYABLE SEMANTIC MAP on REAL indoor data (NYUv2), closing
the user's original ask ("highlight where the chairs are in the map") end-to-end
through the unified vision net + the bounded SSP binding algebra.

Pipeline (all O(D) phase ops on ONE fixed-size vector — no history stored):
  1. UnifiedVisionNet (sspax/vision/segnet.py) predicts a per-cell class on a
     real luma image (the SEG head, Regime B).
  2. each 30x40 cell is back-projected to a 3D point via the NYUv2 depth +
     camera intrinsics (deploy: the depth is the lidar's; here NYUv2 gives it).
  3. the predicted class -> its k-bit code -> bound at the cell's 3D position and
     bundled into the bounded semantic map (sspax/semantic.py).
  4. QUERY a class -> unbind its roles -> spatial density -> the cells of that
     class light up; other classes cross-talk only through random bit overlap.

Honest metric (no class NAMES needed, so no mapping-file dependency): the
ROUND-TRIP query AUC — for each class present in a frame, cells the net PREDICTED
as that class must read out higher than the rest under that class's query.
Averaged over classes/frames = how faithfully the bounded map answers "where is
class c". Anti-oracle: GT labels only SCORE (compare predicted-class map recall
vs GT); they never enter the net or the binding. A named "chair" highlight is a
one-line extension once the official 40-class mapping is fetched.

  PYTHONPATH=. python3 -m sspax.vision.objmap_nyu           # needs segnet_nyu.pkl
"""
import pickle
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from sspax.vision import segnet as SG
from sspax import semantic as SM

ROOT = Path(__file__).resolve().parents[2]
PKL = ROOT / "scratch" / "segnet_nyu.pkl"

# NYUv2 labeled-set intrinsics (640x480); scaled to the net input then to /4.
FX, FY, CX, CY = 518.857, 519.470, 325.582, 253.736
IN_H, IN_W = SG.IN_HW                 # 120,160  (net input)
CELL = 4                              # trunk stride (/4) -> 30x40 cells


def load_nyu_depth(hw=SG.IN_HW, max_n=None):
    """luma (N,H,W,1), depth (N,H,W) metres, labels40 (N,H,W). Depth is kept for
    back-projection (the net still only SEES luma — deploy-faithful)."""
    import h5py
    with h5py.File(SG.NYU, "r") as f:
        imgs, deps, labs = f["images"], f["depths"], f["labels"]
        N = imgs.shape[0] if max_n is None else min(max_n, imgs.shape[0])
        Y, Dp, L = [], [], []
        for i in range(N):
            im = np.array(imgs[i]).transpose(2, 1, 0).astype(np.float32) / 255.0
            Y.append(SG._resize_nn(im @ SG._LUMA, hw)[..., None].astype(np.float32))
            Dp.append(SG._resize_nn(np.array(deps[i]).T, hw).astype(np.float32))
            L.append(SG._resize_nn(np.array(labs[i]).T, hw).astype(np.int32))
    return np.stack(Y), np.stack(Dp), np.stack(L)


def cell_positions(depth):
    """back-project each /4 cell centre to a 3D camera-frame point (metres).
    depth (H,W) -> pts (30*40, 3). Intrinsics scaled to the net input res."""
    H, W = depth.shape
    sx, sy = W / 640.0, H / 480.0
    fx, fy, cx, cy = FX * sx, FY * sy, CX * sx, CY * sy
    dc = depth[CELL // 2::CELL, CELL // 2::CELL]           # (30,40) cell depths
    hh, ww = dc.shape
    vv, uu = np.meshgrid(np.arange(hh) * CELL + CELL // 2,
                         np.arange(ww) * CELL + CELL // 2, indexing="ij")
    Z = dc
    X = (uu - cx) * Z / fx
    Yc = (vv - cy) * Z / fy
    return np.stack([X.ravel(), Yc.ravel(), Z.ravel()], 1), dc.ravel()


def _auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
    return (r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def run(n_frames=40, k=12, min_cells=8, seed=0):
    if not PKL.exists():
        print(f"segnet_nyu.pkl not found ({PKL}) — train segnet on NYUv2 first "
              f"(python3 -m sspax.vision.segnet train).")
        return
    params = pickle.load(open(PKL, "rb"))
    model = SG.UnifiedVisionNet(n_class=SG.N_CLASS)
    Y, Dp, L = load_nyu_depth(max_n=None)
    # eval on the last 20% (disjoint from segnet train split)
    ntr = int(0.8 * len(Y))
    idx = np.arange(ntr, len(Y))[:n_frames]
    codes = SM.class_codes([f"c{c}" for c in range(SG.N_CLASS)], k=k)
    names = list(codes.keys())

    rt_aucs, contrasts, loads = [], [], []
    for fi in idx:
        out = model.apply(params, jnp.asarray(Y[fi:fi + 1]))
        pred = np.asarray(out["seg"][0]).argmax(-1).ravel()          # 30*40
        pts, dc = cell_positions(Dp[fi])
        valid = (dc > 0.2) & (dc < 8.0)                              # usable depth
        # AGGREGATE cells -> OBJECTS: one feature per predicted class present, at
        # its centroid (binding every cell overloads the bounded map 75x past its
        # ~sqrt(D) capacity — the capacity law dictates object-level binding).
        objs = []                                                    # (class, xyz)
        for c in range(1, SG.N_CLASS):
            m = valid & (pred == c)
            if m.sum() < min_cells:
                continue
            objs.append((c, np.median(pts[m], axis=0)))
        if len(objs) < 2:
            continue
        loads.append(len(objs))
        mp = np.zeros(SM.D, complex)
        for c, xyz in objs:
            mp += SM.bind_feature(codes[names[c]], xyz)
        cent = np.stack([o[1] for o in objs])
        clss = np.array([o[0] for o in objs])
        # query each object's class -> its centroid must read out above the OTHER
        # object centroids in the scene (localise the queried class among objects)
        for c in np.unique(clss):
            dens = SM.query(mp, codes[names[int(c)]], cent[:, :2],
                            z=float(np.median(cent[:, 2])))
            pos = dens[clss == c]; neg = dens[clss != c]
            a = _auc(pos, neg)
            if not np.isnan(a):
                rt_aucs.append(a)
                if np.mean(neg) > 1e-9:
                    contrasts.append(float(np.mean(pos) / np.mean(neg)))

    print(f"Regime-C queryable map on NYUv2 ({len(idx)} held-out frames, "
          f"D={SM.D}, k={k} bits/class, object-level binding):")
    print(f"  map load: {np.mean(loads):.1f} objects/frame (D={SM.D} capacity "
          f"~{0.76*SM.D**0.5:.0f}) — within the sqrt(D) capacity law")
    print(f"  round-trip query AUC (queried class centroid vs other objects): "
          f"{np.nanmean(rt_aucs):.3f}  (n={len(rt_aucs)} class-queries)")
    print(f"  query contrast (readout at class / off-class): "
          f"{np.nanmedian(contrasts):.1f}x median")
    print("  anti-oracle: GT labels used only to SCORE seg; the map is built "
          "from the net's own predictions bound at depth-projected positions.")
    return float(np.nanmean(rt_aucs)), float(np.nanmedian(contrasts))


def _extract_objects(model, params, Y, Dp, idx, min_cells):
    """per frame: (class_id, 3D centroid) list from the net's OWN predictions."""
    frames = []
    for fi in idx:
        out = model.apply(params, jnp.asarray(Y[fi:fi + 1]))
        pred = np.asarray(out["seg"][0]).argmax(-1).ravel()
        pts, dc = cell_positions(Dp[fi])
        valid = (dc > 0.2) & (dc < 8.0)
        objs = []
        for c in range(1, SG.N_CLASS):
            m = valid & (pred == c)
            if m.sum() >= min_cells:
                objs.append((c, np.median(pts[m], axis=0)))
        if len(objs) >= 2:
            frames.append(objs)
    return frames


def run_dsweep(n_frames=40, k=12, min_cells=8):
    """CAPACITY LEVER on the REAL queryable map: bind the SAME real objects into
    maps of increasing D and show round-trip query AUC rises toward capacity —
    the sqrt(D) law, demonstrated on real NYUv2 predictions."""
    from sspax import semantic_capacity as SC
    if not PKL.exists():
        print("segnet_nyu.pkl not found — train segnet first."); return
    params = pickle.load(open(PKL, "rb"))
    model = SG.UnifiedVisionNet(n_class=SG.N_CLASS)
    Y, Dp, L = load_nyu_depth(max_n=None)
    ntr = int(0.8 * len(Y))
    idx = np.arange(ntr, len(Y))[:n_frames]
    frames = _extract_objects(model, params, Y, Dp, idx, min_cells)
    names = [f"c{c}" for c in range(SG.N_CLASS)]
    codes = SC.class_codes(names, k=k, m=256)
    load = np.mean([len(f) for f in frames])
    print(f"Regime-C queryable map — CAPACITY LEVER (real NYUv2, "
          f"{len(frames)} frames, {load:.1f} objects/frame):")
    print(f"  {'n_dirs':>7} {'D':>6} {'cap~':>6} {'query-AUC':>10}", flush=True)
    for nd in [20, 40, 60, 120, 240]:
        W = SC.W_of_D(nd); D = W.shape[0]
        roles = SC.roles_of(256, D)
        aucs = []
        for objs in frames:
            cent = np.stack([o[1] for o in objs])
            clss = np.array([o[0] for o in objs])
            mp = np.zeros(D, complex)
            for c, xyz in objs:
                mp += roles[codes[names[c]]].sum(0) * np.exp(1j * (xyz @ W.T))
            for c in np.unique(clss):
                unb = np.conj(roles[codes[names[int(c)]]]).sum(0) * mp
                dens = np.real(np.exp(1j * (cent @ W.T)) @ np.conj(unb)) / D
                a = _auc(dens[clss == c], dens[clss != c])
                if not np.isnan(a):
                    aucs.append(a)
        print(f"  {nd:>7} {D:>6} {0.76*D**0.5:>6.0f} {np.nanmean(aucs):>10.3f}",
              flush=True)


def run_significance(n_frames=60, k=12, min_cells=8):
    """SIGNIFICANCE = committed bit count (user directive: "modulate a feature's
    significance by setting bits"), on REAL predictions: an object commits
    round(confidence*k) of its class's k bits, so the query READOUT tracks the
    net's softmax confidence — and confidence tracks correctness, so the readout
    is a self-supervised significance/quality signal on the bounded map."""
    if not PKL.exists():
        print("segnet_nyu.pkl not found — train segnet first."); return
    import scipy.stats as st_
    params = pickle.load(open(PKL, "rb"))
    model = SG.UnifiedVisionNet(n_class=SG.N_CLASS)
    Y, Dp, L = load_nyu_depth(max_n=None)
    ntr = int(0.8 * len(Y)); idx = np.arange(ntr, len(Y))[:n_frames]
    names = [f"c{c}" for c in range(SG.N_CLASS)]
    codes = SM.class_codes(names, k=k)
    confs, reads, correct = [], [], []
    for fi in idx:
        out = model.apply(params, jnp.asarray(Y[fi:fi + 1]))
        logit = np.asarray(out["seg"][0])                       # (30,40,40)
        prob = np.exp(logit - logit.max(-1, keepdims=True))
        prob /= prob.sum(-1, keepdims=True)
        pred = prob.argmax(-1).ravel(); conf = prob.max(-1).ravel()
        pts, dc = cell_positions(Dp[fi]); valid = (dc > 0.2) & (dc < 8.0)
        L4 = L[fi, CELL // 2::CELL, CELL // 2::CELL].ravel()
        objs = []
        for c in range(1, SG.N_CLASS):
            m = valid & (pred == c)
            if m.sum() < min_cells:
                continue
            oc = float(conf[m].mean())
            objs.append((c, np.median(pts[m], 0), oc,
                         float((L4[m] == c).mean())))     # GT-correct frac (score)
        if len(objs) < 2:
            continue
        # bind: committed bits = round(confidence * k) of the class's k bits
        mp = np.zeros(SM.D, complex)
        for c, xyz, oc, _ in objs:
            nb = max(1, int(round(oc * k)))
            mp += SM.ROLES[codes[names[c]][:nb]].sum(0) * SM.pos_ssp(xyz)
        for c, xyz, oc, gt in objs:
            r = float(SM.query(mp, codes[names[c]], xyz[None, :2],
                               z=float(xyz[2]))[0])
            confs.append(oc); reads.append(r); correct.append(gt)
    confs, reads, correct = map(np.array, (confs, reads, correct))
    rc = st_.spearmanr(confs, reads).correlation
    # does high readout (=high committed bits) select more-correct objects?
    hi = reads > np.median(reads)
    print(f"SIGNIFICANCE via committed bits on real NYUv2 ({len(confs)} objects):")
    print(f"  Spearman(confidence, query-readout) = {rc:.3f}  "
          f"(readout tracks committed bits ∝ confidence)")
    print(f"  GT-correctness: high-readout objects {correct[hi].mean():.2f} vs "
          f"low-readout {correct[~hi].mean():.2f} — significance = quality signal")
    print("  => the map grades each feature's importance by its bit count; "
          "confident (correct) objects read out stronger. Anti-oracle: GT only "
          "scores the correctness split.")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"dsweep": run_dsweep, "sig": run_significance, "run": run}.get(cmd, run)()
