"""Feature-detector formulation zoo for the ECP5 camera front end
(user directive 2026-07-14: "continue experimenting, same opt corners as
before, various feature detector formulations, ...").

Every formulation is an INTEGER, line-buffer-streamable candidate for
hw/ecp5 RTL; each is characterized on the house corners:
  S (size)      static RTL cost: window rows (line buffers), multipliers
                per pixel, adder class — table below, printed by `cost`
  F (fidelity)  utility on the actual system metric: cam-only and
                lidar(+)cam place-AUC on the azel3d recipe lattice,
                adjacent-frame repeatability proxy, servo stability
  T (throughput) all are 1 px/clk single-pass; Harris-class needs a
                second window stage (Sobel then structure tensor) —
                noted as extra rows/latency, never a rate change

Detectors (det(img, t) -> (mask, score int)):
  fast9      FAST, >=9 contiguous of 16 (the shipped baseline)
  fast12     FAST, >=12 contiguous — stricter arc, same hardware
  susan16    SUSAN-style: few similar taps on the ring (n_sim <= 6),
             score = sum of |d|-t over dissimilar taps — comparators only
  extrema    strict 8-neighbour local extremum with margin t — cheapest
  harris     Sobel 3x3 -> structure tensor 5x5 -> det - tr^2/16
             (int64 exact here; RTL needs staged truncation + 5 mults)
  shitomasi  L1 min-eigenvalue bound: (Sxx+Syy) - (|Sxx-Syy| + 2|Sxy|)
             — 3 mults for the tensor, NO response multiplies
  edge       Sobel |dx|+|dy| + 3x3 NMS — edge features control (corridors
             are corner-poor; do edges carry the place signal?)

Depth augmentation (the "..."): camera features lifted to 3D landmarks
p = bearing * nearest-lidar-range within a 2-deg cone; the camera->body
axis mapping is picked ONCE by depth-hit rate (pure geometry, no labels).
Anti-oracle: labels (classroom withheld odometry / school own-estimate
DIAGNOSTIC) score pairs only.

Usage:
  python3 -m experiments.detzoo cost
  python3 -m experiments.detzoo bench spot|school_run2   # AUC + repeat
  python3 -m experiments.detzoo depth spot|school_run2   # 3D landmarks
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hw" / "ecp5"
                       / "host"))
import golden_cam as GC          # noqa: E402
import experiments.lattice3d as L3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "spot_telluride"
N_FRAMES = 110


# --------------------------------------------------------------------------
#  formulations
# --------------------------------------------------------------------------
def _fast_arc(img, t, arc):
    im = img.astype(np.int16)
    h, w = im.shape
    c = im[3:h - 3, 3:w - 3]
    d = np.stack([im[3 + dy:h - 3 + dy, 3 + dx:w - 3 + dx] - c
                  for dy, dx in GC.CIRCLE])
    bright, dark = d > t, d < -t
    corner = np.zeros(c.shape, bool)
    for armm in (bright, dark):
        for r in range(16):
            idx = [(r + j) % 16 for j in range(arc)]
            corner |= np.logical_and.reduce(armm[idx])
    bs = np.maximum(d - t, 0).sum(0)
    ds = np.maximum(-d - t, 0).sum(0)
    sc = np.where(corner, np.maximum(bs, ds), 0)
    return _pad(img, corner, sc)


def _pad(img, corner, sc, b=3):
    mask = np.zeros(img.shape, bool)
    score = np.zeros(img.shape, np.int64)
    mask[b:img.shape[0] - b, b:img.shape[1] - b] = corner
    score[b:img.shape[0] - b, b:img.shape[1] - b] = sc
    return mask, score


def det_fast9(img, t):
    return _fast_arc(img, t, 9)


def det_fast12(img, t):
    return _fast_arc(img, t, 12)


def det_susan16(img, t, n_sim_max=6):
    im = img.astype(np.int16)
    h, w = im.shape
    c = im[3:h - 3, 3:w - 3]
    d = np.stack([im[3 + dy:h - 3 + dy, 3 + dx:w - 3 + dx] - c
                  for dy, dx in GC.CIRCLE])
    sim = np.abs(d) <= t
    corner = sim.sum(0) <= n_sim_max
    sc = np.where(corner, np.maximum(np.abs(d) - t, 0).sum(0), 0)
    return _pad(img, corner, sc)


def det_extrema(img, t):
    im = img.astype(np.int16)
    h, w = im.shape
    c = im[1:h - 1, 1:w - 1]
    ns = [im[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
          for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    d = np.stack([n - c for n in ns])
    bright = np.all(d < -t, 0)
    dark = np.all(d > t, 0)
    corner = bright | dark
    sc = np.where(corner, np.abs(d).sum(0), 0)
    return _pad(img, corner, sc, b=1)


def _sobel(img):
    im = img.astype(np.int64)
    k = np.array([1, 2, 1])
    # separable: smooth one axis, diff the other (integer exact)
    sy = im[:-2] + 2 * im[1:-1] + im[2:]
    dx = sy[:, 2:] - sy[:, :-2]
    sx = im[:, :-2] + 2 * im[:, 1:-1] + im[:, 2:]
    dy = sx[2:] - sx[:-2]
    return dx, dy                      # both (h-2, w-2)


def _sum5(a):
    c = np.cumsum(np.cumsum(a, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[5:, 5:] - c[:-5, 5:] - c[5:, :-5] + c[:-5, :-5])


def _tensor(img):
    dx, dy = _sobel(img)
    return _sum5(dx * dx), _sum5(dy * dy), _sum5(dx * dy)


def det_harris(img, t):
    sxx, syy, sxy = _tensor(img)
    resp = (sxx * syy - sxy * sxy) - ((sxx + syy) ** 2 >> 4)
    sc = np.maximum(resp >> 16, 0)     # rescale into servo-able range
    corner = sc > t
    return _pad(img, corner, np.where(corner, sc, 0), b=3)


def det_shitomasi(img, t):
    sxx, syy, sxy = _tensor(img)
    lam = (sxx + syy) - (np.abs(sxx - syy) + 2 * np.abs(sxy))
    sc = np.maximum(lam >> 8, 0)
    corner = sc > t
    return _pad(img, corner, np.where(corner, sc, 0), b=3)


def det_edge(img, t):
    dx, dy = _sobel(img)
    m = (np.abs(dx) + np.abs(dy)) >> 3
    corner = m > t
    return _pad(img, corner, np.where(corner, m, 0), b=1)


DETS = dict(fast9=det_fast9, fast12=det_fast12, susan16=det_susan16,
            extrema=det_extrema, harris=det_harris,
            shitomasi=det_shitomasi, edge=det_edge)

# S-corner: (window rows incl. all stages, mults/pixel, adder class)
COST = dict(
    fast9=(7, 0, "16 cmp + 2x16 relu-add"),
    fast12=(7, 0, "same silicon as fast9 (arc param)"),
    susan16=(7, 0, "16 abs-cmp + count + relu-add"),
    extrema=(3, 0, "8 cmp + abs-add"),
    harris=(3 + 5, 5, "3 tensor mults + det/tr2 mults, staged trunc"),
    shitomasi=(3 + 5, 3, "3 tensor mults, response = abs/add only"),
    edge=(3, 0, "sep. Sobel adds + abs"),
)


def detect_target(det, img, target, tmin=1, tmax=4095):
    lo, hi = tmin, tmax
    best = None
    while lo <= hi:
        t = (lo + hi) // 2
        mask, score = det(img, t)
        keep = GC.nms3(score.astype(np.uint16) if score.max() < 65536
                       else (score >> 4).astype(np.uint16))
        ys, xs = np.nonzero(keep)
        n = len(ys)
        key = (abs(n - target), t)
        if best is None or key < best[0]:
            best = (key, t, (ys, xs, score[keep]))
        if n > target:
            lo = t + 1
        elif n < target:
            hi = t - 1
        else:
            break
    return best[1], best[2]


# --------------------------------------------------------------------------
#  frame set: decoded grays + clouds + poses (cached per venue)
# --------------------------------------------------------------------------
def frame_set(run, labels):
    from PIL import Image
    import io as _io
    import pyarrow.parquet as pq
    import runners.spot_cam as SC
    pick, kts, pose, kind = L3.sample_kf(run, N_FRAMES, need_ref=True,
                                         labels=labels)
    if pose is None:
        raise SystemExit(f"{run}: no labels")
    clouds, cok = L3.load_clouds(run, kts)
    shards, cts, where = SC._index(run)
    j = np.clip(np.searchsorted(cts, kts), 1, len(cts) - 1)
    j = j - (np.abs(cts[j - 1] - kts) < np.abs(cts[j] - kts))
    gok = np.abs(cts[j] - kts) < SC.ALIGN_TOL_MS * 1e6
    grays, K = [None] * len(pick), None
    by_shard = {}
    for i in range(len(pick)):
        if gok[i]:
            si, row = where[j[i]]
            by_shard.setdefault(si, []).append((row, i))
    for si in sorted(by_shard):
        t = pq.read_table(shards[si])
        if K is None:
            import json as _json
            K = np.array(_json.loads(t["camera_info"][0].as_py())["K"],
                         float).reshape(3, 3)
        for row, i in sorted(by_shard[si]):
            im = np.asarray(Image.open(_io.BytesIO(
                t["image"][row].as_py()["bytes"])).convert("RGB"))
            grays[i] = GC.bin2(GC.rgb_to_gray(im))
    keep = [i for i in range(len(pick))
            if cok[i] and clouds[i] is not None and len(clouds[i]) > 500
            and grays[i] is not None]
    return ([grays[i] for i in keep], [clouds[i] for i in keep],
            pose[keep], kind, K)


def bearings(K, ys, xs):
    Ki = np.linalg.inv(K)
    uv = np.stack([xs * 2 + 1.0, ys * 2 + 1.0, np.ones(len(xs))])
    b = (Ki @ uv).T
    return b / np.linalg.norm(b, axis=1, keepdims=True)


# --------------------------------------------------------------------------
#  benches
# --------------------------------------------------------------------------
def bench(run, only=None):
    labels = "ref" if run == "spot" else "est"
    grays, clouds, pose, kind, K = frame_set(run, labels)
    pr = dict(same_r=1.0, far_lo=4.0) if labels == "est" else {}
    (si, sj), (di, dj) = L3._pairs(pose, **pr)
    adj = np.arange(len(pose) - 1)
    W = L3.make_lattices()["azel3d"]
    Vl = np.stack([L3.encode(W, c) for c in clouds])
    Sl = np.abs(Vl @ Vl.conj().T)
    print(f"detector zoo ({run}, {len(grays)} frames, labels={kind}; "
          f"{len(si)} same / {len(di)} diff; lidar-only AUC "
          f"{L3._auc(Sl[si, sj], Sl[di, dj]):.3f}):")
    tgt = [min(len(c), 1000) for c in clouds]
    for name, det in DETS.items():
        if only and name not in only:
            continue
        Vc, ths, ns = [], [], []
        for g, tg in zip(grays, tgt):
            t, (ys, xs, sc) = detect_target(det, g, tg)
            b = bearings(K, ys, xs)
            Vc.append(L3.encode(W, b, sc.astype(float) + 1.0))
            ths.append(t)
            ns.append(len(ys))
        Vc = np.stack(Vc)
        Vb = Vc + Vl
        Vb /= np.linalg.norm(Vb, axis=1, keepdims=True)
        Sc = np.abs(Vc @ Vc.conj().T)
        Sb = np.abs(Vb @ Vb.conj().T)
        # repeatability proxy: adjacent-frame cam sim vs diff-pair cam sim
        rep = L3._auc(Sc[adj, adj + 1], Sc[di, dj])
        print(f"  {name:9s} n med {int(np.median(ns)):4d} "
              f"thr[{min(ths)}..{max(ths)}]  cam AUC "
              f"{L3._auc(Sc[si, sj], Sc[di, dj]):.3f}  fused "
              f"{L3._auc(Sb[si, sj], Sb[di, dj]):.3f}  adj-rep {rep:.3f}",
              flush=True)


def pick_cam2body(bear_list, clouds):
    """Axis mapping camera->body chosen by depth-hit rate (geometry only).
    Candidates map camera (x right, y down, z fwd) into body frames."""
    cands = {
        "z->+x": np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float),
        "z->-x": np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], float),
        "z->+y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float),
        "z->-y": np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], float),
    }
    best = None
    for name, R in cands.items():
        hits = 0
        tot = 0
        for b, c in zip(bear_list[:8], clouds[:8]):
            bb = b @ R.T
            cu = c / np.linalg.norm(c, axis=1, keepdims=True)
            m = (bb @ cu.T).max(1) > np.cos(np.deg2rad(2.0))
            hits += int(m.sum())
            tot += len(bb)
        rate = hits / max(tot, 1)
        if best is None or rate > best[1]:
            best = (name, rate, R)
    print(f"  cam->body: {best[0]} (depth-hit {best[1]*100:.0f}%)")
    return best[2]


def depth(run):
    labels = "ref" if run == "spot" else "est"
    grays, clouds, pose, kind, K = frame_set(run, labels)
    pr = dict(same_r=1.0, far_lo=4.0) if labels == "est" else {}
    (si, sj), (di, dj) = L3._pairs(pose, **pr)
    W = L3.make_lattices()["azel3d"]
    Vl = np.stack([L3.encode(W, c) for c in clouds])
    print(f"depth-augmented landmarks ({run}, {len(grays)} frames, "
          f"labels={kind}; fast9 features):")
    feats = []
    for g, c in zip(grays, clouds):
        t, (ys, xs, sc) = detect_target(det_fast9, g, min(len(c), 1000))
        feats.append((bearings(K, ys, xs), sc.astype(float) + 1.0))
    R = pick_cam2body([f[0] for f in feats], clouds)
    Vb_dir, Vb_3d = [], []
    hitrates = []
    for (b, wgt), c in zip(feats, clouds):
        bb = b @ R.T
        cu = c / np.linalg.norm(c, axis=1, keepdims=True)
        dot = bb @ cu.T
        jj = dot.argmax(1)
        ok = dot[np.arange(len(bb)), jj] > np.cos(np.deg2rad(2.0))
        rng = np.linalg.norm(c[jj], axis=1)
        p3 = bb[ok] * rng[ok, None]
        hitrates.append(ok.mean())
        Vb_dir.append(L3.encode(W, bb, wgt))
        Vb_3d.append(L3.encode(W, p3, wgt[ok]) if ok.sum() > 20
                     else L3.encode(W, bb, wgt))
    print(f"  depth-hit rate med {np.median(hitrates)*100:.0f}%")
    for tag, Vc in (("bearings ", np.stack(Vb_dir)),
                    ("3D points", np.stack(Vb_3d))):
        Vf = Vc + Vl
        Vf /= np.linalg.norm(Vf, axis=1, keepdims=True)
        Sc = np.abs(Vc @ Vc.conj().T)
        Sf = np.abs(Vf @ Vf.conj().T)
        print(f"  {tag}: cam AUC {L3._auc(Sc[si, sj], Sc[di, dj]):.3f}  "
              f"lidar+cam {L3._auc(Sf[si, sj], Sf[di, dj]):.3f}",
              flush=True)


def cost():
    print("S-corner (static RTL cost per formulation):")
    print(f"  {'det':9s} {'win rows':8s} {'mult/px':7s} adder class")
    for k, (rows, mults, note) in COST.items():
        print(f"  {k:9s} {rows:<8d} {mults:<7d} {note}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cost"
    if cmd == "cost":
        cost()
    elif cmd == "bench":
        bench(sys.argv[2] if len(sys.argv) > 2 else "spot",
              only=set(sys.argv[3].split(",")) if len(sys.argv) > 3
              else None)
    elif cmd == "depth":
        depth(sys.argv[2] if len(sys.argv) > 2 else "spot")
