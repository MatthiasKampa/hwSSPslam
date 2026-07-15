"""OBJECT-IN-MAP retrieval from camera data (user 2026-07-15: "push
towards being able to use the map for finding objects in the environment
map from cam data").

Formulation — the world-frame APPEARANCE MAP: camera cells are lifted to
3D world points (registered depth; TUM venue) and bound as
    m += w * exp(i (W . p_world + phi_app(code)))
bundled over keyframes into bounded vectors (one global, or one per
32-kf segment — the house bounded-map shape). An object query is a
template PATCH from a camera frame: its cells' appearance codes +
relative world offsets form a matched filter
    q_f = sum_j w_j exp(i (W_f . delta_j + phi_f(code_j)))
and the object's world position is the standard translation-correlation
decode  x* = argmax Re sum_f m_f conj(q_f) exp(-i W_f . x)  over a
coarse->fine grid — O(D) per candidate, the map's own algebra, no
history.

ANTI-ORACLE / DIAGNOSTIC LABEL: keyframe poses that PLACE map content
are mocap (the deployed builder would use its own anchor poses — this
measures the representation's capability bound, stated per PROTOCOL);
the query never sees GT; GT scores the decoded position afterwards.
Foil queries (patches from a different sequence) probe detection
separability. Everything deterministic (seed 11).

Corners:
  F  appearance family {census, grad, int} x {self, cross-view} err/AUC
  S  D ladder x phase-only 2b quant x {global, seg32} scope x kf budget
  T  encode = ncell x D cis-MACs per kf (place-encode class); query =
     D x grid cis-MACs, on-demand — printed analytically

Usage (repo root):
  python3 -m experiments.objmap selftest
  python3 -m experiments.objmap run   [seq]     # F-corner main table
  python3 -m experiments.objmap dcap  [seq]     # S-corner ladders
  python3 -m experiments.objmap anchor [seq]    # cheap which-anchor tier
"""
import sys
from pathlib import Path

import numpy as np

import experiments.vision6d as V6
import experiments.detzoo as DZ

SEQ = "rgbd_dataset_freiburg3_long_office_household"
FOIL_SEQ = "rgbd_dataset_freiburg1_desk"
RNG = 11
CS = 16                          # cell size on the 320x240 frame
N_QUERY = 24
PATCH = 2                        # query patch = (2P+1)^2 cells
BLOCK = 40                       # holdout block length (kf) for cross
SEG = 8                          # segment scope length (kf)
LAMS = [0.5, 0.707, 1.0, 1.414, 2.0, 2.828, 4.0, 5.657]
# room-scale ladder, COARSE-biased: real-data world points smear
# ~5-10 cm (depth noise + c16 cell-centre approximation), so sub-0.5 m
# rings decohere and only add map noise — D goes to directions instead


# --------------------------------------------------------------------------
#  lattice: isotropic 3D directions x spatial ladder (position-only decode;
#  no rotation algebra needed — appearance codes are view-variant anyway)
# --------------------------------------------------------------------------
def W_map(D, lams=None):
    import experiments.lattice3d as L3
    lams = LAMS if lams is None else lams
    nd = D // len(lams)
    dirs = L3.dirs_fib(nd)
    return np.concatenate([(2 * np.pi / lam) * dirs for lam in lams])


# --------------------------------------------------------------------------
#  cells: means, world points, appearance codes
# --------------------------------------------------------------------------
def cell_feats(gray, depth, K, gt):
    """-> dict per frame: world points, weights, appearance codes, and
    the cell grid (for patch queries). Inner cells with valid depth."""
    h, w = gray.shape[0] // CS * CS, gray.shape[1] // CS * CS
    m = gray[:h, :w].reshape(h // CS, CS, w // CS, CS).mean((1, 3))
    gh, gw = m.shape
    # census-8 on the cell-mean grid
    code = np.zeros(m.shape, np.int32)
    k = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                n = np.roll(np.roll(m, dy, 0), dx, 1)
                code |= (m > n).astype(np.int32) << k
                k += 1
    # gradient orientation (8-dir) + magnitude per cell
    dx, dy = DZ._sobel(gray)
    dxp = np.zeros(gray.shape); dxp[1:-1, 1:-1] = dx
    dyp = np.zeros(gray.shape); dyp[1:-1, 1:-1] = dy
    cdx = dxp[:h, :w].reshape(gh, CS, gw, CS).mean((1, 3))
    cdy = dyp[:h, :w].reshape(gh, CS, gw, CS).mean((1, 3))
    thq = np.round(np.arctan2(cdy, cdx) / (np.pi / 4)) * (np.pi / 4)
    mag = np.hypot(cdx, cdy)
    # depth at cell centres -> camera then world points
    ys, xs = np.mgrid[0:gh, 0:gw]
    cy, cx = ys * CS + CS // 2, xs * CS + CS // 2
    d = depth[cy, cx] / 1000.0
    inner = np.zeros(m.shape, bool)
    inner[1:-1, 1:-1] = True
    ok = inner & (d > 0.2) & (d < 8.0)
    Ki = np.linalg.inv(K)
    uv = np.stack([cx + 0.0, cy + 0.0, np.ones(m.shape)], 0)
    b = np.einsum("ij,jhw->ihw", Ki, uv)
    pc = (b / np.linalg.norm(b, axis=0)) * d
    R = V6.quat_R(gt[3:7])
    pw = np.einsum("ij,jhw->ihw", R, pc) + gt[:3, None, None]
    return dict(pw=pw.transpose(1, 2, 0), ok=ok, w=mag + 1.0,
                census=code, gradq=thq, inten=m, gh=gh, gw=gw)


_LUT = {}
_FAM_SEED = {"census": 1, "grad": 2, "int": 3}


def app_amp(fam, D):
    """(levels, D) COMPLEX appearance amplitude LUT, random-key bound.
    Binding law learned the hard way in this module's selftests: the
    place encoders' 3-harmonic scheme leaves cross-code terms
    ~1/sqrt(3)-coherent (bump forest clustered AT the true answer —
    difference-of-uniforms density peaks at zero); uniform per-bit
    phase keys kill the forest but fully decorrelate at ONE flipped
    bit; concentrated keys re-grow the forest from near-Hamming code
    pairs. The resolution:
      census  BIPOLAR SPATTER binding A[c] = sum_k s_k(c) exp(i K_k) /
              sqrt(8), s_k = +-1 per bit: cross-code expectation =
              (8 - 2*Hamming)/8 — ZERO-mean for random code pairs
              (per-value keys gave matching_bits/8 ~ 0.5 coherent
              clutter — measured), linear grace at 1-2 flips,
              pseudo-random cross-talk per row.
      grad    FPE phases on the doubled orientation (h_f in 1..4 keeps
              pi-periodicity), unit amplitude.
      int     FPE phases on intensity, s_f ~ U(0.5, 3), unit amplitude.
    Seeds fixed per (family, D) — no salted hash() (determinism rule).
    """
    key = (fam, D)
    if key not in _LUT:
        rng = np.random.default_rng(_FAM_SEED[fam] * 100_000 + D)
        if fam == "census":
            K = rng.uniform(0, 2 * np.pi, (8, D))
            bits = (np.arange(256)[:, None] >> np.arange(8)[None]) & 1
            sgn = 1.0 - 2.0 * bits                     # (256, 8) of +-1
            _LUT[key] = (sgn @ np.exp(1j * K)) / np.sqrt(8)
        elif fam == "grad":
            h = rng.integers(1, 5, D).astype(float)
            th = np.arange(8) * (np.pi / 4)
            _LUT[key] = np.exp(1j * (2.0 * th)[:, None] * h[None, :])
        else:                                          # int
            s = rng.uniform(0.5, 3.0, D)
            v = np.arange(256) / 256.0
            _LUT[key] = np.exp(1j * 2 * np.pi * v[:, None] * s[None, :])
    return _LUT[key]


def app_of(F, fam, D):
    """(gh, gw, D) complex appearance amplitudes via the LUTs."""
    lut = app_amp(fam, D)
    if fam == "census":
        idx = F["census"]
    elif fam == "grad":
        idx = np.round(F["gradq"] / (np.pi / 4)).astype(int) % 8
    else:
        idx = np.clip(np.round(F["inten"]), 0, 255).astype(int)
    return lut[idx]


# --------------------------------------------------------------------------
#  map build / query / decode
# --------------------------------------------------------------------------
def frame_vec(F, fam, W):
    ok = F["ok"]
    ph = np.exp(1j * (F["pw"][ok] @ W.T))
    w = np.sqrt(F["w"][ok])          # tame heavy gradmag tails
    app = app_of(F, fam, len(W))[ok]
    app = app - app.mean(0)          # scene-DC centering: real codes are
    # low-entropy (neighbouring cells correlate), so the cross-code
    # kernel is zero-mean only under UNIFORM codes — centering by the
    # empirical amplitude mean restores it under the scene distribution
    v = (w[:, None] * ph * app).sum(0)
    return v / max(np.linalg.norm(v), 1e-12)   # equal energy per kf


def build_maps(feats, kf_idx, fam, W, seg=None):
    """[(kf list, map vec)] — one global map (seg=None) or 32-kf
    segments (the bounded per-segment shape)."""
    groups = ([list(kf_idx)] if seg is None else
              [list(kf_idx[i:i + seg]) for i in range(0, len(kf_idx), seg)])
    out = []
    for g in groups:
        v = np.zeros(len(W), complex)
        for k in g:
            v += frame_vec(feats[k], fam, W)
        out.append((g, v))
    return out


def quant2b(v):
    ph = np.round(np.angle(v) * 4 / (2 * np.pi)) * (2 * np.pi / 4)
    return np.exp(1j * ph) * np.median(np.abs(v))


def make_query(F, fam, W, cy, cx):
    """Matched filter from the (2P+1)^2 patch at cell (cy, cx):
    q_f = sum_j w_j exp(i(W . delta_j + phi_j)), deltas in WORLD frame
    (query pose rotates offsets — in deploy that is the live pose
    estimate). -> (q, x_gt, n_cells)."""
    ok = F["ok"]
    ys, xs = np.mgrid[cy - PATCH:cy + PATCH + 1, cx - PATCH:cx + PATCH + 1]
    ys, xs = ys.ravel(), xs.ravel()
    keep = ok[ys, xs]
    ys, xs = ys[keep], xs[keep]
    if len(ys) < 6:
        return None
    pw = F["pw"][ys, xs]
    wgt = np.sqrt(F["w"][ys, xs])
    x0 = (pw * wgt[:, None]).sum(0) / wgt.sum()
    if np.linalg.norm(pw - x0, axis=1).max() > 0.5:
        return None                  # depth-discontinuous patch: x_gt
    ph = np.exp(1j * ((pw - x0) @ W.T))   # ill-defined (fore/background)
    app = app_of(F, fam, len(W))[ys, xs]
    app = app - app.mean(0)               # patch-DC centering (see map)
    q = (wgt[:, None] * ph * app).sum(0)
    return q, x0, len(ys)


def decode(maps, q, W, bbox, step=0.15, refine=0.02):
    """Scale-split coarse->fine: the wide search runs on the COARSE
    rings only (lam >= 1 m — kernel lobe wider than the grid step; the
    fine rings' 6 cm lobe would fall between 15 cm samples), the local
    refine on all rings. Max over map segments. -> (x_hat, score)."""
    nl = len([l for l in LAMS if True])
    # infer per-ring lam from row norms (W rows are (2pi/lam)*unit dirs)
    rn = 2 * np.pi / np.linalg.norm(W, axis=1)
    cmask = rn >= 1.0
    Wc = W[cmask]
    axes = [np.arange(bbox[0][k], bbox[1][k] + 1e-9, step)
            for k in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    best = None
    for _, m in maps:
        g = (m * np.conj(q)).astype(np.complex64)
        gc = g[cmask]
        sc = None
        for i in range(0, len(grid), 16384):
            E = np.exp(-1j * (grid[i:i + 16384] @ Wc.T)).astype(
                np.complex64)
            s = (E @ gc).real
            sc = s if sc is None else np.concatenate([sc, s])
        k = int(np.argmax(sc))
        if best is None or sc[k] > best[1]:
            best = (grid[k], float(sc[k]), g)
    x0, _, g = best
    # iterated re-centred refine: stage 1 must cover the coarse stage's
    # own uncertainty (half a lobe of the finest coarse ring ~ 0.25 m);
    # stage 2 re-centres so an edge-clipped peak is recovered
    for half, stp in ((0.35, 0.05), (0.10, refine)):
        off = np.arange(-half, half + 1e-9, stp)
        loc = x0[None] + np.stack(
            np.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3)
        sc = None
        for i in range(0, len(loc), 16384):
            E = np.exp(-1j * (loc[i:i + 16384] @ W.T)).astype(
                np.complex64)
            s = (E @ g).real
            sc = s if sc is None else np.concatenate([sc, s])
        k = int(np.argmax(sc))
        x0 = loc[k]
    return x0, float(sc[k])


# --------------------------------------------------------------------------
#  experiment scaffolding
# --------------------------------------------------------------------------
def load_feats(seq):
    gray, depth, gt, K = V6.load(seq)
    feats = [cell_feats(g, d, K, p) for g, d, p in zip(gray, depth, gt)]
    return feats, gt


def split(n):
    """map kf vs query kf: alternating 40-kf blocks; queries at excluded-
    block centres are >= BLOCK/2 kf (~4 s) from any map frame."""
    blk = (np.arange(n) // BLOCK) % 2 == 0
    return np.flatnonzero(blk), np.flatnonzero(~blk)


def pick_queries(feats, cand_kf, rng, n=N_QUERY):
    """textured cells with valid depth, deterministic."""
    out = []
    kfs = rng.permutation(cand_kf)
    for k in kfs:
        F = feats[k]
        ok = F["ok"].copy()
        ok[:PATCH + 1] = ok[-PATCH - 1:] = False
        ok[:, :PATCH + 1] = ok[:, -PATCH - 1:] = False
        mag = np.where(ok, F["w"], 0)
        if mag.max() <= 1.0:
            continue
        thr = np.percentile(mag[mag > 0], 75)
        cys, cxs = np.nonzero(mag >= thr)
        if not len(cys):
            continue
        j = rng.integers(len(cys))
        out.append((k, int(cys[j]), int(cxs[j])))
        if len(out) == n:
            break
    return out


def world_bbox(feats, kf_idx):
    pts = np.concatenate([feats[k]["pw"][feats[k]["ok"]][::7]
                          for k in kf_idx[::5]])
    lo = np.percentile(pts, 1, axis=0) - 0.3
    hi = np.percentile(pts, 99, axis=0) + 0.3
    return lo, hi


def run_tier(feats, maps, W, fam, queries, bbox):
    errs, scores, nc = [], [], []
    for k, cy, cx in queries:
        mq = make_query(feats[k], fam, W, cy, cx)
        if mq is None:
            continue
        q, x_gt, n = mq
        x_hat, sc = decode(maps, q, W, bbox)
        errs.append(np.linalg.norm(x_hat - x_gt))
        scores.append(sc / max(n, 1))
        nc.append(n)
    return np.array(errs), np.array(scores), nc


def foil_scores(maps, W, fam, foil_seq, bbox, rng):
    ff, _ = load_feats(foil_seq)
    qs = pick_queries(ff, np.arange(len(ff)), rng)
    out = []
    for k, cy, cx in qs:
        mq = make_query(ff[k], fam, W, cy, cx)
        if mq is None:
            continue
        q, _, n = mq
        _, sc = decode(maps, q, W, bbox)
        out.append(sc / max(n, 1))
    return np.array(out)


# --------------------------------------------------------------------------
#  benches
# --------------------------------------------------------------------------
def bench_run(seq=SEQ, D=960):
    import experiments.lattice3d as L3
    rng = np.random.default_rng(RNG)
    feats, gt = load_feats(seq)
    map_kf, qry_kf = split(len(feats))
    W = W_map(D)
    bbox = world_bbox(feats, map_kf)
    print(f"object-in-map ({seq.split('_freiburg')[-1]}, {len(feats)} kf: "
          f"{len(map_kf)} map / {len(qry_kf)} query blocks; D={len(W)}, "
          f"cells c{CS}, patch {2*PATCH+1}^2; poses DIAGNOSTIC mocap):")
    ncell = int(np.median([f['ok'].sum() for f in feats]))
    gsz = int(np.prod([(bbox[1][k] - bbox[0][k]) / 0.15 + 1
                       for k in range(3)]))
    print(f"  T-corner: encode ~{ncell * len(W) / 1e6:.1f}M cis-MAC/kf "
          f"(place-encode class); query ~{gsz * len(W) / 1e6:.0f}M "
          f"cis-MAC coarse (+refine), on-demand")
    q_self = pick_queries(feats, map_kf, rng)
    q_cross = pick_queries(feats, qry_kf, rng)
    for fam in ("census", "grad", "int"):
        maps = build_maps(feats, map_kf, fam, W, seg=SEG)
        e_s, s_s, _ = run_tier(feats, maps, W, fam, q_self, bbox)
        e_c, s_c, _ = run_tier(feats, maps, W, fam, q_cross, bbox)
        s_f = foil_scores(maps, W, fam, FOIL_SEQ, bbox, rng)
        auc = L3._auc(s_c, s_f)
        print(f"  {fam:6s} self err med {np.median(e_s):.3f} p90 "
              f"{np.percentile(e_s, 90):.3f} | cross med "
              f"{np.median(e_c):.3f} p90 {np.percentile(e_c, 90):.3f} "
              f"(n={len(e_c)}) | detect-AUC {auc:.3f}", flush=True)


def bench_dcap(seq=SEQ, fam="census"):
    rng = np.random.default_rng(RNG)
    feats, gt = load_feats(seq)
    map_kf, qry_kf = split(len(feats))
    bbox = world_bbox(feats, map_kf)
    q_cross = pick_queries(feats, qry_kf, rng)
    print(f"S-corner ladders ({seq.split('_freiburg')[-1]}, {fam}, "
          f"cross-view):")
    for D in (480, 960, 1920):
        W = W_map(D)
        maps = build_maps(feats, map_kf, fam, W)
        e, _, _ = run_tier(feats, maps, W, fam, q_cross, bbox)
        by = 2 * len(W) * 2 // 8
        print(f"  D{len(W):4d} global float: med {np.median(e):.3f} "
              f"p90 {np.percentile(e, 90):.3f}  ({by} B @2b)", flush=True)
    W = W_map(960)
    for sg in (32, 8, 4):
        maps = build_maps(feats, map_kf, fam, W, seg=sg)
        e, _, _ = run_tier(feats, maps, W, fam, q_cross, bbox)
        print(f"  D 960 seg{sg:2d} float: med {np.median(e):.3f} "
              f"p90 {np.percentile(e, 90):.3f}  ({len(maps)} segments)",
              flush=True)
    maps = build_maps(feats, map_kf, fam, W, seg=SEG)
    mq = [(g, quant2b(v)) for g, v in maps]
    e, _, _ = run_tier(feats, mq, W, fam, q_cross, bbox)
    print(f"  D 960 seg{SEG} 2b   : med {np.median(e):.3f} "
          f"p90 {np.percentile(e, 90):.3f}", flush=True)
    for frac in (0.25, 0.5):
        sub = map_kf[:int(len(map_kf) * frac)]
        maps = build_maps(feats, sub, fam, W)
        keep = [q for q in q_cross
                if any(abs(q[0] - k) < 3 * BLOCK for k in sub)]
        e, _, _ = run_tier(feats, maps, W, fam, keep, bbox)
        print(f"  D 960 kf x{frac:.2f}   : med {np.median(e):.3f} "
              f"(n={len(e)}; queries near covered span only)", flush=True)


def bench_anchor(seq=SEQ, fam="census"):
    """Cheap tier: WHICH-ANCHOR retrieval via per-anchor appearance BAGS
    (no geometry binding: v = sum w exp(i phi_app) — appearance
    histogram in phase). Answer = best anchor's position; the metric is
    anchor-granularity localization + top-k hit rate."""
    rng = np.random.default_rng(RNG)
    feats, gt = load_feats(seq)
    map_kf, qry_kf = split(len(feats))
    D = 240
    harm = (np.arange(D) % 3) + 1.0

    def bag(F, ys=None, xs=None):
        if ys is None:
            ok = F["ok"]
            c, w = F["census"][ok].astype(float), F["w"][ok]
        else:
            c, w = F["census"][ys, xs].astype(float), F["w"][ys, xs]
        ph = (2 * np.pi * c / 256.0)[:, None] * harm[None, :]
        v = (w[:, None] * np.exp(1j * ph)).sum(0)
        return v / max(np.linalg.norm(v), 1e-12)

    anchors = [list(map_kf[i:i + 8]) for i in range(0, len(map_kf), 8)]
    A = np.stack([sum(bag(feats[k]) for k in g) for g in anchors])
    A -= A.mean(0)                   # shared scene code statistics out
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    apos = np.stack([gt[g, :3].mean(0) for g in anchors])
    qs = pick_queries(feats, qry_kf, rng)
    hits1 = hits3 = 0
    errs = []
    for k, cy, cx in qs:
        F = feats[k]
        ys, xs = np.mgrid[cy - PATCH:cy + PATCH + 1,
                          cx - PATCH:cx + PATCH + 1]
        ys, xs = ys.ravel(), xs.ravel()
        keep = F["ok"][ys, xs]
        if keep.sum() < 6:
            continue
        q = bag(F, ys[keep], xs[keep])
        sim = np.abs(A @ np.conj(q))
        order = np.argsort(-sim)
        x_gt = F["pw"][ys[keep], xs[keep]].mean(0)
        d = np.linalg.norm(apos - x_gt, axis=1)
        errs.append(d[order[0]])
        hits1 += d[order[0]] <= np.sort(d)[2]
        hits3 += min(d[order[:3]]) <= np.sort(d)[2]
    print(f"anchor-bag tier ({seq.split('_freiburg')[-1]}, "
          f"{len(anchors)} anchors x D240 bags, {len(errs)} queries):")
    print(f"  top1-in-best3-anchors {hits1}/{len(errs)}  top3 "
          f"{hits3}/{len(errs)}  pos err med {np.median(errs):.3f} m "
          f"(anchor granularity)", flush=True)


def selftest():
    rng = np.random.default_rng(RNG)
    lams_fine = [0.25, 0.354, 0.5, 0.707, 1.0, 1.414, 2.0, 2.828,
                 4.0, 5.657, 8.0, 11.314]
    W = W_map(480, lams_fine)      # no-smear regime: machinery gate
    lut = app_amp("census", len(W))
    # synthetic: 400 random world points with random census codes
    pts = rng.uniform(-2, 2, (400, 3))
    codes = rng.integers(0, 256, 400)
    m = (np.exp(1j * (pts @ W.T)) * lut[codes]).sum(0)
    # STATISTICAL gate (5 query draws of 16 cells — the real config is
    # 20-25-cell patches): the decode landscape is stochastic at the
    # ~lam_min/4 = 12.5 cm lobe floor, so single draws swing 5-12 cm
    e_ex, e_cor, r_foil, ord_ok = [], [], [], 0
    for t in range(5):
        sel = rng.choice(400, 16, replace=False)
        x0 = pts[sel].mean(0)
        ph = np.exp(1j * ((pts[sel] - x0) @ W.T))
        q = (ph * lut[codes[sel]]).sum(0)
        x_hat, sc = decode([(None, m)], q, W, (x0 - 1.0, x0 + 1.0),
                           step=0.1)
        e_ex.append(np.linalg.norm(x_hat - x0))
        # foil (wrong codes): max of the noise field (extreme-value) —
        # gate ratio + ordering; real detection = the bench's AUC
        qf = (ph * lut[(codes[sel] + 91) % 256]).sum(0)
        _, scf = decode([(None, m)], qf, W, (x0 - 1.0, x0 + 1.0),
                        step=0.1)
        r_foil.append(scf / sc)
        # 1-bit-per-cell corruption, INDEPENDENT bits (same-bit-everywhere
        # would be a coherent ghost via the shared key vector): gate the
        # KERNEL LAW at the true position — (8-2)/8 = 0.75 exactly, a
        # deterministic binding property; corrupt-query DETECTION through
        # room-wide noise maxima is extreme-value territory and belongs
        # to the real bench's AUC, not an assert
        flip = 1 << rng.integers(0, 8, len(sel))
        qb = (ph * lut[codes[sel] ^ flip]).sum(0)
        at = np.exp(-1j * (x0[None] @ W.T))[0]
        s_true = float(np.real(at @ (m * np.conj(q))))
        s_corr = float(np.real(at @ (m * np.conj(qb))))
        e_cor.append(s_corr / max(s_true, 1e-9))
        ord_ok += sc > scf
    assert np.median(e_ex) < 0.06, e_ex
    assert 0.6 < np.median(e_cor) < 0.9, e_cor
    assert ord_ok >= 4, ord_ok
    # NOTE the foil ratio is NOT asserted: a wrong-code template's peak
    # is the max of the room-wide noise field, and at 16 cells vs 400
    # items that extreme value sits at 0.7-0.95x of the true match —
    # single-template detection margin IS thin (the real bench measures
    # it as AUC; the architectural answer is segment-scoped two-stage
    # retrieval and bigger/multi-view templates)
    print(f"selftest ok (5 draws): exact med {np.median(e_ex)*100:.1f} cm"
          f"; 1-flip kernel {np.median(e_cor):.2f} (theory 0.75); "
          f"exact>foil {ord_ok}/5; foil ratio med "
          f"{np.median(r_foil):.2f} (informational)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    seq = sys.argv[2] if len(sys.argv) > 2 else SEQ
    if cmd == "selftest":
        selftest()
    elif cmd == "run":
        bench_run(seq)
    elif cmd == "dcap":
        bench_dcap(seq)
    elif cmd == "anchor":
        bench_anchor(seq)
