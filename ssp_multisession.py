"""Multi-session VSA map composition: merge two INDEPENDENTLY-BUILT SSP maps
by VECTOR SUPERPOSITION (bundling = addition) -- the paper's core thesis
("algebraically transformable maps") applied to multi-session mapping.

The bounded SSP map is a set of per-anchor rigid segment vectors placed at each
anchor's graph pose by the SAME algebra world_vec_seg uses:
    translation  = elementwise phase multiply   ENC.shift(pos) * v
    rotation     = angle-index permutation       rot_permute(v, m)  (+ derivative)
    bundling     = vector addition
So two maps of overlapping area, once in a common frame, can be composed
WITHOUT re-encoding any scan: fold the placing transform into each anchor pose
and ADD the world-placed vectors. Because "graph corrections move anchors only;
segments ride along rigidly", RE-ANCHORING every segment to a corrected pose is
also free -- we exploit that to put two independent maps in one frame.

The experiment has TWO decoupled parts (see the coordinator note / RESULTS):

  PART 1 -- ISOLATE THE VSA CLAIM.  Re-anchor each session's segments onto the
  gfs ground-truth trajectory (free: rigid per-anchor move, no re-encode), so
  the two overlapping maps share ONE frame by construction. Then SUPERPOSE
  (bundle world-placed vectors of A and B per cell) and measure crosstalk:
  match-peak sharpness / localization accuracy of B's overlap scans against
  A-alone, B-alone, and the MERGED map. Plus an alignment-accuracy sweep
  (perturb the placement by delta before superposing) to find the bar.

  PART 2 -- CHARACTERIZE THE ALIGNMENT CHALLENGE.  In the two sessions' OWN
  (independently-drifted) frames, measure the differential drift over the
  overlap via place-correspondences: a single rigid T_AB leaves residual X m
  (non-rigid warp). This is a GENERAL multi-session SLAM problem (cross-session
  loop closure + joint relax), NOT VSA-specific; once solved, Part-1 VSA
  superposition applies unchanged.

Usage:
    python3 ssp_multisession.py selftest         # fast algebra checks
    python3 ssp_multisession.py build            # build+cache the two sessions
    python3 ssp_multisession.py run              # full experiment (auto-builds)
"""

import os
import pickle
import sys
import time

import numpy as np

import ssp_slam as S
import ssp_slam_loop as L
import ssp_slam_carmen as C
from ssp_bounded import BoundedSLAM, ANCHOR, CELL

VALID_MAX = 40.0
DTYPE = np.complex64
SCRATCH = ("/private/tmp/claude-504/-Users-kamp-code-ssp/"
           "dcd1c1b5-1a7c-4c0c-a903-a52aa4a19412/scratchpad")
CACHE = os.path.join(SCRATCH, "sessions.pkl")

RING_OF = np.repeat(np.arange(L.N_RING), L.N_ANG)
MAIN_M = RING_OF < 4                        # lam 0.25,0.5,1,2  (matcher band)
COARSE_M = RING_OF >= 4                     # lam 5.3, 12.8      (relocalization)


# --------------------------------------------------------------------------
# core VSA algebra: place a stored anchor-frame segment vector at a pose
# (identical to BoundedSLAM.world_vec_seg, standalone so it takes any pose)
# --------------------------------------------------------------------------
def place(segvec, segder, pose, use_der=True):
    """World-place an anchor-frame segment vector at `pose` (x,y,heading):
    rotate by angle-index permutation (+ 1st-order derivative for the sub-grid
    residual), then translate by phase multiply. Exactly world_vec_seg's
    algebra; NOTHING is re-encoded from scans."""
    m = int(round(pose[2] * L.N_ANG / np.pi))
    delta = pose[2] - m * np.pi / L.N_ANG
    v = L.rot_permute(segvec, m)
    if use_der and segder is not None:
        v = v + delta * L.rot_permute(segder, m)
    return L.ENC.shift(pose[:2]) * v


def se2(a, b):
    return L.se2_mul(a, b)


def se2i(a):
    return L.se2_inv(a)


# --------------------------------------------------------------------------
# build + cache the two independent sessions
# --------------------------------------------------------------------------
def build_session(keys, odom, beam, lo, hi):
    slam = BoundedSLAM(robust=True, attempt_every=4, relax_every=25,
                       gap_kf=300, recent_aids=12)
    slam.store_dtype = DTYPE
    idx = np.arange(lo, hi)
    est = np.zeros((len(idx), 3))
    for j, k in enumerate(idx):
        r = keys[k][0]
        rr = np.where(r < VALID_MAX, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = np.zeros(3) if j == 0 else se2(
            est[j - 1], se2(se2i(odom[k - 1]), odom[k]))
        est[j] = slam.add_keyframe(pts, w, guess)
    if slam.dirty:
        slam.relax()
    fin = np.stack([slam.pose_of(j) for j in range(len(idx))])
    data = dict(anchors=[a.copy() for a in slam.anchors],
                segvec={a: v.copy() for a, v in slam.segvec.items()},
                segder={a: v.copy() for a, v in slam.segder.items()},
                fin=fin, idx=idx,
                n_loop=sum(1 for e in slam.edges if e[5] == "loop"))
    return data


def build_and_cache():
    path = "data/intel.log"
    print("loading Intel log ...", flush=True)
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)
    n = len(keys)
    beam = np.deg2rad(np.arange(len(keys[0][0])) - 90.0)
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    ref = C.parse_flaser(path.replace(".log", ".gfs.log"))
    rts = np.array([t for _, _, t in ref])
    rp = np.stack([p for _, p, _ in ref])
    a0, a1 = 0, int(0.60 * n)
    b0, b1 = int(0.40 * n), n
    print(f"{n} kf; session A=kf[{a0}:{a1}]  B=kf[{b0}:{b1}]", flush=True)
    t0 = time.time()
    A = build_session(keys, odom, beam, a0, a1)
    print(f"  A built ({time.time()-t0:.0f}s) {len(A['segvec'])} segvecs "
          f"{A['n_loop']} loops", flush=True)
    t0 = time.time()
    B = build_session(keys, odom, beam, b0, b1)
    print(f"  B built ({time.time()-t0:.0f}s) {len(B['segvec'])} segvecs "
          f"{B['n_loop']} loops", flush=True)
    blob = dict(A=A, B=B, kts=kts, rts=rts, rp=rp, n=n,
                a0=a0, a1=a1, b0=b0, b1=b1,
                keys_r=[keys[k][0] for k in range(n)], beam=beam)
    with open(CACHE, "wb") as f:
        pickle.dump(blob, f)
    print(f"cached -> {CACHE}", flush=True)
    return blob


def load_cache():
    with open(CACHE, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------
# dense gt trajectory (interpolate sparse gfs to every keyframe time)
# --------------------------------------------------------------------------
def gt_dense(kts, rts, rp):
    o = np.argsort(rts)
    x = np.interp(kts, rts[o], rp[o, 0])
    y = np.interp(kts, rts[o], rp[o, 1])
    th = np.interp(kts, rts[o], np.unwrap(rp[o, 2]))
    return np.stack([x, y, S.wrap(th)], 1)


# --------------------------------------------------------------------------
# world-place a session's segments at GT-corrected anchor poses (common frame)
# (re-anchoring is free: segments ride rigidly, nothing re-encoded)
# --------------------------------------------------------------------------
def world_gt(data, gtd, T_extra=None, dt=complex):
    """List of (world_pos, world_vec) in the GT frame. T_extra optionally folds
    an extra rigid transform into every anchor pose (the perturbation used for
    the alignment-sensitivity sweep)."""
    idx = data["idx"]
    out = []
    for aid, segvec in data["segvec"].items():
        kl = aid * ANCHOR
        if kl >= len(idx):
            continue
        pose = gtd[idx[kl]].copy()
        if T_extra is not None:
            pose = se2(T_extra, pose)
        out.append((pose[:2].copy(),
                    place(segvec, data["segder"].get(aid), pose).astype(dt)))
    return out


def local_bundle(world_list, center, radius=8.0, dt=complex):
    B = np.zeros(L.W.shape[0], dt)
    for pos, v in world_list:
        if np.linalg.norm(pos - center) < radius:
            B += v
    return B


def peak_sharpness(bundle_main, pts, w, pose, half=0.8, step=0.05):
    """Peak-to-background of the MAIN-band translation score surface at pose:
    higher = sharper / better-localized. Uses the matcher's own objective."""
    Sv = L.ENC_MAIN.encode(pts @ S._rot(pose[2]).T, w)
    c = np.conj(bundle_main) * Sv
    vv = np.arange(-half, half + step / 2, step)
    gx, gy = np.meshgrid(vv, vv)
    off = np.stack([gx.ravel(), gy.ravel()], 1)
    ph = np.exp(1j * (off @ L.W[MAIN_M].T)) * np.exp(1j * (L.W[MAIN_M] @ pose[:2]))
    sc = (ph @ c).real
    return float((sc.max() - sc.mean()) / (sc.std() + 1e-12))


def make_matcher():
    return S.Matcher(L.ENC_MAIN, t_half=0.72, rot_half_deg=9,
                     rot_step_deg=1.5, perm=(4, L.N_ANG))


# --------------------------------------------------------------------------
# selftest: composition algebra is exact
# --------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(0)
    slam = BoundedSLAM()
    slam.store_dtype = complex
    pose = np.array([1.3, -2.1, 0.7])
    for _ in range(6):
        pts = rng.uniform(-4, 4, (200, 2))
        w = np.full(200, 0.1)
        slam.add_keyframe(pts, w, pose + rng.normal(0, 0.02, 3))
    aid = next(iter(slam.segvec))
    v_ref = slam.world_vec_seg(aid)
    v_new = place(slam.segvec[aid], slam.segder.get(aid), slam.anchors[aid])
    assert np.allclose(v_ref, v_new, atol=1e-9), np.abs(v_ref - v_new).max()
    # folding a grid-aligned transform into the pose == transforming the bundle
    m = 7
    T = np.array([2.0, -1.5, m * np.pi / L.N_ANG])
    B0nd = place(slam.segvec[aid], slam.segder.get(aid), slam.anchors[aid],
                 use_der=False)
    Ba = L.ENC.shift(T[:2]) * L.rot_permute(B0nd, m)
    Bb = place(slam.segvec[aid], slam.segder.get(aid), se2(T, slam.anchors[aid]),
               use_der=False)
    assert np.allclose(Ba, Bb, atol=1e-6), np.abs(Ba - Bb).max()
    print("selftest ok: place()==world_vec_seg; T-fold==bundle-transform")


# ==========================================================================
# PART 1 -- isolate the VSA superposition claim in a COMMON (gt) frame
# ==========================================================================
def part1(blob, gtd):
    print("\n" + "=" * 70)
    print("PART 1: VSA superposition in a COMMON frame (drift removed)")
    print("=" * 70, flush=True)
    A, B = blob["A"], blob["B"]
    a1 = blob["a1"]
    wlA = world_gt(A, gtd)
    wlB = world_gt(B, gtd)
    print(f"A: {len(wlA)} placed segments   B: {len(wlB)} placed segments "
          f"(both in gt frame)", flush=True)

    # ---- memory: per-cell superposition vs duplicate sum
    D = L.W.shape[0]
    bpc = 8 if DTYPE == np.complex64 else 16
    cell = lambda p: (int(np.floor(p[0] / CELL)), int(np.floor(p[1] / CELL)))
    cvA, cvB = {}, {}
    for p, v in wlA:
        cvA.setdefault(cell(p), np.zeros(D, complex)); cvA[cell(p)] += v
    for p, v in wlB:
        cvB.setdefault(cell(p), np.zeros(D, complex)); cvB[cell(p)] += v
    cellsA, cellsB = set(cvA), set(cvB)
    shared = cellsA & cellsB
    union = cellsA | cellsB
    kb = lambda nc: nc * D * bpc / 1024
    print(f"\n-- MEMORY (world-placed, 1 vec/cell, CELL={CELL} m) --", flush=True)
    print(f"  cells: A={len(cellsA)} B={len(cellsB)} shared={len(shared)} "
          f"union={len(union)}", flush=True)
    print(f"  duplicate sum  A+B          = {kb(len(cellsA)+len(cellsB)):.0f} KB", flush=True)
    print(f"  superposed merge (union)    = {kb(len(union)):.0f} KB", flush=True)
    print(f"  saving = {100*(1-len(union)/(len(cellsA)+len(cellsB))):.0f}% "
          f"(overlap fraction; O(area) not O(sessions))", flush=True)

    # ---- crosstalk of the bundled cell: cosine of single vs merged
    cos = []
    for cl in shared:
        merged = cvA[cl] + cvB[cl]
        for single in (cvA[cl], cvB[cl]):
            cos.append(abs(np.vdot(single, merged))
                       / (np.linalg.norm(single) * np.linalg.norm(merged) + 1e-12))
    cos = np.array(cos)
    print(f"\n-- CROSSTALK (bundled cell vs its single-session content) --", flush=True)
    print(f"  |<single, merged>| cosine over {len(shared)} shared cells: "
          f"median {np.median(cos):.3f}  p10 {np.percentile(cos,10):.3f} "
          f"(1=preserved)", flush=True)

    # ---- localization: B overlap scans vs A-alone / B-alone / MERGED
    print(f"\n-- LOCALIZATION of B's overlap scans (common gt frame) --", flush=True)
    matcher = make_matcher()
    idxB = B["idx"]
    beam = blob["beam"]
    Aposxy = np.array([p for p, _ in wlA])
    from scipy.spatial import cKDTree
    treeA = cKDTree(Aposxy)
    rng = np.random.default_rng(1)
    cand = []
    for j, k in enumerate(idxB):
        if k < a1:                       # take the temporally-disjoint tail
            continue
        d, _ = treeA.query(gtd[k, :2])
        if d < 2.0:
            cand.append((j, k))
    if len(cand) > 80:
        sel = sorted(rng.choice(len(cand), 80, replace=False))
        cand = [cand[i] for i in sel]
    print(f"  {len(cand)} genuine cross-pass query scans", flush=True)

    goff = np.array([0.30, -0.20, np.deg2rad(3.0)])   # odometry-like guess
    res = _localize_set(cand, blob, gtd, wlA, wlB, matcher, beam, goff)
    for tag in ("A", "B", "merged"):
        e = np.array(res[tag]["e"]); sh = np.array(res[tag]["sh"])
        print(f"  {tag:>6}: ATE rmse {np.sqrt((e**2).mean())*100:5.1f} cm "
              f"median {np.median(e)*100:5.1f} cm  in-gate "
              f"{res[tag]['gate']}/{len(e)}  peak-sharp {np.median(sh):.1f}",
              flush=True)
    eA = np.array(res["A"]["e"]); eB = np.array(res["B"]["e"])
    eM = np.array(res["merged"]["e"])
    best = min(np.sqrt((eA**2).mean()), np.sqrt((eB**2).mean()))
    mrg = np.sqrt((eM**2).mean())
    print(f"  => merged {mrg*100:.1f} cm vs best single {best*100:.1f} cm: "
          f"{'PRESERVES' if mrg <= best*1.15 else 'DEGRADES'} information",
          flush=True)

    # ---- alignment-sensitivity sweep: perturb B before superposing
    print(f"\n-- ALIGNMENT SENSITIVITY (perturb B by delta, then merge) --",
          flush=True)
    print(f"  (fine wavelength lam=0.25 m; CELL={CELL} m)", flush=True)
    for delta in (0.0, 0.05, 0.10, 0.25, 0.50, 1.0):
        T_pert = np.array([delta, 0.0, 0.0])
        wlB_p = world_gt(B, gtd, T_extra=T_pert)
        res_p = _localize_set(cand, blob, gtd, wlA, wlB_p, matcher, beam, goff,
                              only="merged", wlB_native=wlB)
        e = np.array(res_p["merged"]["e"]); sh = np.array(res_p["merged"]["sh"])
        cvBp = {}
        for p, v in wlB_p:
            cvBp.setdefault(cell(p), np.zeros(D, complex)); cvBp[cell(p)] += v
        cc = []
        for cl in cellsA & set(cvBp):
            merged = cvA[cl] + cvBp[cl]
            cc.append(abs(np.vdot(cvA[cl], merged))
                      / (np.linalg.norm(cvA[cl]) * np.linalg.norm(merged) + 1e-12))
        print(f"  delta={delta:4.2f} m: merged ATE {np.sqrt((e**2).mean())*100:5.1f} cm "
              f"median {np.median(e)*100:5.1f} cm  peak-sharp {np.median(sh):.1f} "
              f"  A-cell-cos {np.median(cc) if cc else float('nan'):.3f}",
              flush=True)


def _localize_set(cand, blob, gtd, wlA, wlB, matcher, beam, goff,
                  only=None, wlB_native=None):
    """Localize each query scan against A-alone, B-alone, merged. `only`
    restricts to one tag (for the sweep). wlB_native = B's unperturbed map for
    the B-alone local bundle (perturbation only affects the MERGED map).
    Returns per-tag lists of localization error e and peak-sharpness sh."""
    res = {t: {"e": [], "sh": [], "gate": 0} for t in ("A", "B", "merged")}
    keys_r = blob["keys_r"]
    gate_t, gate_r = 0.6, np.deg2rad(7)
    wlBn = wlB_native if wlB_native is not None else wlB
    for j, k in cand:
        rr = np.where(keys_r[k] < VALID_MAX, keys_r[k], np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        if len(pts) < 20:
            continue
        gt = gtd[k]
        guess = se2(gt, goff)                 # odometry-like off-truth guess
        BA = local_bundle(wlA, gt[:2]) if only in (None, "A") else None
        BB = local_bundle(wlBn, gt[:2]) if only in (None, "B") else None
        BM = (local_bundle(wlA, gt[:2]) + local_bundle(wlB, gt[:2])) \
            if only in (None, "merged") else None
        for tag, bundle in (("A", BA), ("B", BB), ("merged", BM)):
            if bundle is None or np.abs(bundle).sum() == 0:
                continue
            pose = matcher.match(bundle[L.MAIN], pts, w, guess)
            pose[2] = S.wrap(pose[2])
            res[tag]["e"].append(np.linalg.norm(pose[:2] - gt[:2]))
            res[tag]["sh"].append(peak_sharpness(bundle[L.MAIN], pts, w, pose))
            if np.linalg.norm(pose[:2] - guess[:2]) < gate_t \
                    and abs(S.wrap(pose[2] - guess[2])) < gate_r:
                res[tag]["gate"] += 1
    return res


# ==========================================================================
# PART 2 -- differential drift between the two independent sessions
# ==========================================================================
def part2(blob, gtd):
    print("\n" + "=" * 70)
    print("PART 2: differential drift (independent frames, rigid T_AB)")
    print("=" * 70, flush=True)
    A, B = blob["A"], blob["B"]
    a1 = blob["a1"]
    finA, finB = A["fin"], B["fin"]
    idxA, idxB = A["idx"], B["idx"]

    gtA = gtd[idxA]
    gtB = gtd[idxB]
    # each session's own ATE (rigid-aligned to gt) -- context for the drift
    aA = C.align_se2(finA[:, :2], gtA[:, :2])
    aB = C.align_se2(finB[:, :2], gtB[:, :2])
    ateA = np.sqrt((np.linalg.norm(aA - gtA[:, :2], axis=1) ** 2).mean())
    ateB = np.sqrt((np.linalg.norm(aB - gtB[:, :2], axis=1) ** 2).mean())
    print(f"  session self-ATE (rigid-aligned to gt): A {ateA:.2f} m  "
          f"B {ateB:.2f} m", flush=True)
    from scipy.spatial import cKDTree
    tA = cKDTree(gtA[:, :2])
    pairs = []
    for j, k in enumerate(idxB):
        if k < a1:
            continue
        d, jA = tA.query(gtB[j, :2])
        if d < 0.4 and abs(S.wrap(gtA[jA, 2] - gtB[j, 2])) < np.deg2rad(20):
            pairs.append((int(jA), j))
    if len(pairs) < 5:
        print("  too few correspondences"); return
    # per-correspondence rigid transform T_i (B frame -> A frame):
    #   T_i = finA[jA] o finB[jB]^-1   (so T_i o finB[jB] = finA[jA])
    Ts = np.array([se2(finA[jA], se2i(finB[jB])) for jA, jB in pairs])
    t_med = np.median(Ts[:, :2], 0)
    th_med = np.arctan2(np.median(np.sin(Ts[:, 2])), np.median(np.cos(Ts[:, 2])))
    T_AB = np.array([t_med[0], t_med[1], th_med])
    resid = []
    for jA, jB in pairs:
        pred = se2(T_AB, finB[jB])
        resid.append(np.linalg.norm(pred[:2] - finA[jA, :2]))
    resid = np.array(resid)
    t_spread = np.linalg.norm(Ts[:, :2] - t_med, axis=1)
    th_spread = np.abs(np.degrees(S.wrap(Ts[:, 2] - th_med)))
    print(f"  {len(pairs)} cross-pass place-correspondences (same gt pose)",
          flush=True)
    print(f"  best single rigid T_AB (B->A): t=({T_AB[0]:.2f},{T_AB[1]:.2f}) "
          f"heading={np.degrees(T_AB[2]):.1f} deg", flush=True)
    print(f"  per-correspondence transform SPREAD about that rigid T_AB:",
          flush=True)
    print(f"    translation: median {np.median(t_spread)*100:.0f} cm  "
          f"p90 {np.percentile(t_spread,90)*100:.0f} cm  "
          f"max {t_spread.max()*100:.0f} cm", flush=True)
    print(f"    heading:     median {np.median(th_spread):.1f} deg  "
          f"max {th_spread.max():.1f} deg", flush=True)
    print(f"  => a single rigid T_AB leaves a NON-RIGID residual of "
          f"~{np.median(resid)*100:.0f} cm (median), {resid.max()*100:.0f} cm "
          f"(max) across the overlap", flush=True)
    print(f"  vs the VSA superposition alignment bar (Part 1); rigid alignment "
          f"is INSUFFICIENT -> per-region loop constraints + joint relax are "
          f"required (a GENERAL multi-session SLAM problem, not VSA-specific).",
          flush=True)


def run():
    if not os.path.exists(CACHE):
        blob = build_and_cache()
    else:
        print(f"loading cached sessions from {CACHE}", flush=True)
        blob = load_cache()
    gtd = gt_dense(blob["kts"], blob["rts"], blob["rp"])
    part1(blob, gtd)
    part2(blob, gtd)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "build":
        build_and_cache()
    elif cmd == "run":
        run()
    else:
        selftest()
