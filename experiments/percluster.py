"""PER-CLUSTER vs BUNDLED map comparison: is the recurring verification wall
partly an artifact of BUNDLING candidate patches into one inner product, or is
it intrinsic to the appearance CONTENT?

The whole project bundles. The frontend `local_bundle` SUMS recent world-placed
segments into one vector B, then does ONE inner product with the query scan.
Drought/loop verification SUMS a candidate segment chain into one bundle, then
one coherence. A bundle of M patches has capacity ~sqrt(N/M): the patches
cross-talk, and a match against the sum cannot tell WHICH patch matched. This
module measures, oracle-free (GT only ever LABELS genuine-vs-twin, never
selects), whether PER-CLUSTER comparison (score the query against each candidate
patch SEPARATELY -> K scores -> best/consensus) preserves discrimination that
bundling averages away, at BOTH query paths, and quantifies the K-fold compute.

DISTINCTION from SeqSLAM (RESULTS.md, NEGATIVE on corridors): SeqSLAM tested
TEMPORAL sequence consistency (per-frame along the trajectory). This tests
SPATIAL per-patch comparison (query vs each map segment SEPARATELY). Different
axis, same wall.

Three experiments (subcommands):
  capacity  -- CROSSTALK: bundle M real recent segments, measure the genuine
               match's peak cosine + pose error vs M; does it follow ~1/sqrt(M),
               and does per-cluster (M=1) avoid it?  (frontend local-map model)
  closure   -- THE WALL: at ring-key-retrieved revisit candidates on MIT
               (corridor twins) + Intel, compare genuine-vs-twin SEPARATION of
               (a) BUNDLED chain coherence vs (b) PER-CLUSTER per-patch coherence
               (max/consensus). AUC + FP@recall, bundled vs per-cluster.
  frontend  -- ATE: BoundedSLAM with the shipped BUNDLED frontend vs a
               PER-CLUSTER frontend (match the scan against each recent segment
               separately, keep the best) on Intel/fr079/fr101.

NO shipped/prior module is edited. Imports by reference only:
  ssp_bounded.BoundedSLAM   -- frontend + map (subclassed for the frontend test)
  ssp_slam / ssp_slam_loop  -- SSP encode/match/SE(2) algebra
  ssp_slam_carmen           -- CARMEN parse/keyframe/align
  ssp_scancontext           -- ring_key / sc_distance / ref_positions / TOL_REV
  ssp_ringkey               -- grid_from_pts (Scan-Context grid from body pts)

Usage:
  python3 -m experiments.percluster selftest
  python3 -m experiments.percluster capacity [intel|mit]
  python3 -m experiments.percluster closure  [intel|mit]
  python3 -m experiments.percluster frontend [fr101|fr079|intel]
  python3 -m experiments.percluster all
"""

import sys
import time

import numpy as np

import sspslam.encoder as S
import sspslam.lattice as L
import sspslam.frontend as C
import sspslam.bounded as B
import baselines.scancontext as SC
from experiments.ringkey import grid_from_pts

ANCHOR = 5
GAP_A = 60                     # gap in anchors (~gap_kf 300) before a revisit
TOL_REV = SC.TOL_REV           # 5.0 m, REF-frame genuine-revisit radius (LABEL only)
MAIN = L.MAIN
WMAIN = L.W[MAIN]


# ---------------------------------------------------------------------------
# dataset loading (identical conventions to ssp_bounded_carmen)
# ---------------------------------------------------------------------------
def load(name, valid_max=40.0):
    path = f"data/{name}.log"
    keys = C.keyframes(C.parse_flaser(path))
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    return keys, beam, odom, kts, n_beams


def scan_pts(r, beam, valid_max=40.0):
    rr = np.where(r < valid_max, r, np.inf)
    pts, w, _ = S.scan_to_samples(rr, beam)
    return pts, w


def enc_main(pts, w, pose):
    """World-frame MAIN-ring SSP encoding of a scan placed at `pose`."""
    PW = pts @ S._rot(pose[2]).T + pose[:2]
    return np.exp(1j * (PW @ WMAIN.T)).T @ w


def cosine(a, b):
    return float((np.conj(a) * b).real.sum()
                 / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------------------------------------------------------------------------
# offline segment map: fold anchor-frame SSP segments exactly as BoundedSLAM
# does (odometry anchor poses; no backend). Stores per-anchor ring key + ref.
# ---------------------------------------------------------------------------
def build_map(name, valid_max=40.0, keep_pts=True, ref=True, verbose=True):
    keys, beam, odom, kts, n_beams = load(name, valid_max)
    n = len(keys)
    segvec = {}                 # aid -> anchor-frame full-ring SSP vector
    a_pose = {}                 # aid -> odom anchor pose (world)
    a_first = {}                # aid -> first keyframe index
    ringkey = {}                # aid -> yaw-invariant ring key (running-mean grid)
    grid_sum = {}
    grid_cnt = {}
    q_pts = {}                  # aid -> (pts, w) of its first keyframe (query probe)
    t0 = time.time()
    for k in range(n):
        pts, w = scan_pts(keys[k][0], beam, valid_max)
        aid = k // ANCHOR
        if aid not in segvec:
            segvec[aid] = np.zeros(L.W.shape[0], complex)
            a_pose[aid] = odom[k].copy()
            a_first[aid] = k
            if keep_pts:
                q_pts[aid] = (pts, w)
        # anchor-frame fold (rigid, exact rotation), identical to add_keyframe
        rel = L.se2_mul(L.se2_inv(a_pose[aid]), odom[k])
        if len(pts):
            PA = pts @ S._rot(rel[2]).T + rel[:2]
            segvec[aid] += np.exp(1j * (PA @ L.W.T)).T @ w
            g = grid_from_pts(pts)
            if g.sum() > 0:
                grid_sum[aid] = g if aid not in grid_sum else grid_sum[aid] + g
                grid_cnt[aid] = grid_cnt.get(aid, 0) + 1
    for aid in grid_sum:
        ringkey[aid] = SC.ring_key(grid_sum[aid] / grid_cnt[aid])
    out = dict(name=name, n=n, keys=keys, beam=beam, odom=odom, kts=kts,
               segvec=segvec, a_pose=a_pose, a_first=a_first, ringkey=ringkey,
               grid_mean={a: grid_sum[a] / grid_cnt[a] for a in grid_sum},
               q_pts=q_pts, n_beams=n_beams, valid_max=valid_max)
    if ref:
        kts2, refxy, valid = SC.ref_positions(name, keys)
        out["ref_kf"] = refxy
        out["valid_kf"] = valid
        # per-anchor ref = ref of its first keyframe
        out["ref_a"] = {a: refxy[a_first[a]] for a in a_first}
        out["valid_a"] = {a: bool(valid[a_first[a]]) for a in a_first}
    if verbose:
        print(f"  [{name}] {n} kf -> {len(segvec)} anchors, "
              f"{len(ringkey)} ring keys  ({time.time()-t0:.0f}s)")
    return out


# ---------------------------------------------------------------------------
# EXPERIMENT 3 -- CAPACITY / CROSSTALK (frontend local-map model)
#   Bundle M real RECENT segments (locally odom-consistent), match the query
#   against (a) the sum B_M and (b) each segment separately. Measure the genuine
#   match's peak cosine + translation error vs M.
# ---------------------------------------------------------------------------
def _peak_cosine(matcher, M, pts, w, guess):
    """matcher-aligned pose + normalized peak cosine of the scan vs map M."""
    pose = matcher.match(M, pts, w, guess)
    pose[2] = S.wrap(pose[2])
    sv = enc_main(pts, w, pose)
    return pose, cosine(M, sv)


def run_capacity(name, valid_max=40.0, n_sites=120, seed=0):
    print(f"\n=== CAPACITY / CROSSTALK  [{name}] "
          f"(frontend local-map bundle model) ===")
    mp = build_map(name, valid_max, keep_pts=True, ref=False)
    segvec, a_pose, q_pts = mp["segvec"], mp["a_pose"], mp["q_pts"]
    matcher = S.Matcher(L.ENC_MAIN, t_half=0.48, rot_half_deg=9,
                        rot_step_deg=1.5, perm=(4, L.N_ANG))
    aids = sorted(a for a in segvec if a in q_pts)
    Ms = [1, 2, 4, 8, 16, 32, 64]
    rng = np.random.default_rng(seed)
    # pick sites with >= max(Ms) available preceding recent anchors
    cand_sites = [a for a in aids if a - 1 in segvec and a >= max(Ms) + 1]
    if len(cand_sites) > n_sites:
        idx = np.linspace(0, len(cand_sites) - 1, n_sites).astype(int)
        cand_sites = [cand_sites[i] for i in idx]
    # per-M: genuine peak cosine (bundled vs per-cluster) and translation error
    cos_b = {m: [] for m in Ms}      # bundled: query vs sum of M recent segs
    cos_p = {m: [] for m in Ms}      # per-cluster: query vs best single seg
    err_b = {m: [] for m in Ms}      # bundled translation error vs per-cluster pose
    n_wrong = {m: 0 for m in Ms}     # bundled peak landed on a WRONG (non-genuine) seg
    for a in cand_sites:
        pts, w = q_pts[a]
        if len(pts) < 20:
            continue
        guess = a_pose[a]            # the site's own (odom) pose ~ where the scan is
        # genuine overlapping segment = immediately-preceding anchor (adjacent view)
        g = a - 1
        # world-place helper (local window -> odom is locally consistent)
        wv = {}
        recent = [a - 1 - j for j in range(max(Ms))]  # a-1, a-2, ...
        recent = [r for r in recent if r in segvec]
        for r in recent:
            wv[r] = _world_main(segvec[r], a_pose[r])
        wg = wv[g]
        # per-cluster reference: query vs the genuine segment ALONE
        pose_p, c_p = _peak_cosine(matcher, wg, pts, w, guess)
        for m in Ms:
            sel = recent[:m]
            Bm = sum(wv[r] for r in sel)
            pose_b, c_b = _peak_cosine(matcher, Bm, pts, w, guess)
            cos_b[m].append(c_b)
            cos_p[m].append(c_p)
            err_b[m].append(float(np.linalg.norm(pose_b[:2] - pose_p[:2])))
            # which single segment does the bundled peak actually explain best?
            best_r = max(sel, key=lambda r: cosine(wv[r], enc_main(pts, w, pose_b)))
            if best_r != g:
                n_wrong[m] += 1
    print(f"  sites: {len(cos_b[1])}   (genuine overlap = adjacent recent anchor)")
    print(f"  {'M':>4} {'cos_bundled':>12} {'cos_percluster':>15} "
          f"{'ratio b/p':>10} {'pose_err_m':>11} {'wrong-peak%':>11}")
    base = np.median(cos_p[1])
    for m in Ms:
        cb, cp = np.median(cos_b[m]), np.median(cos_p[m])
        pe = np.median(err_b[m])
        wr = 100.0 * n_wrong[m] / max(1, len(cos_b[m]))
        print(f"  {m:>4} {cb:>12.4f} {cp:>15.4f} {cb/(cp+1e-12):>10.3f} "
              f"{pe:>11.3f} {wr:>10.1f}%")
    # sqrt-law fit: cos_bundled(M) ~ c1 / sqrt(M) ?  (up to saturation)
    x = np.array(Ms, float)
    yb = np.array([np.median(cos_b[m]) for m in Ms])
    # fit on the log-log slope over the first decades
    sl = np.polyfit(np.log(x), np.log(yb), 1)[0]
    print(f"  bundled cos(M) log-log slope = {sl:+.3f}   "
          f"(-0.5 => exact 1/sqrt(M) capacity;  0 => no crosstalk)")
    print(f"  per-cluster cos is M-INVARIANT by construction "
          f"(median {base:.4f} at every M): K inner products, no crosstalk.")
    print(f"  COMPUTE: per-cluster = K matcher calls/query (K = window size, "
          f"here up to {max(Ms)}); bundled = 1.  Tradeoff is exactly K-fold.")
    return dict(cos_b=cos_b, cos_p=cos_p, err_b=err_b, slope=sl, Ms=Ms)


def _world_main(seg_anchor_frame, pose):
    """Place an anchor-frame full-ring segment into world frame, MAIN rings."""
    v = L.ENC.shift(pose[:2]) * (L.rot_permute(seg_anchor_frame,
                                               int(round(pose[2] * L.N_ANG / np.pi))))
    # first-order rotation correction skipped (segder not stored); sub-grid
    # residual (<1.5 deg) is absorbed by the matcher, adequate for this probe.
    return v[MAIN]


# ---------------------------------------------------------------------------
# EXPERIMENT 2 -- CLOSURE / RELOCALIZATION DISCRIMINATION (the wall)
#   At ring-key-retrieved revisit candidates, score the query against each
#   candidate patch (PER-CLUSTER) and against the candidate's contiguous chain
#   bundle (BUNDLED). GT labels genuine (<=TOL_REV) vs twin. Compare separation.
# ---------------------------------------------------------------------------
def _yaw_seed(qgrid, cand_grid, cand_heading):
    _, s = SC.sc_distance(qgrid, cand_grid)
    return S.wrap(cand_heading + s * (2 * np.pi / SC.N_SECTOR))


def _align_score(matcher, seg_main, pts, w, seed):
    """Wide local-frame alignment of the query to a single anchor-frame segment;
    returns (pose, per-ring coherence vector, fine-ring mean coherence)."""
    pose = matcher.match(seg_main, pts, w, seed)
    pose[2] = S.wrap(pose[2])
    return pose


def _coh(full_seg, pts, w, pose):
    """Per-ring coherence of the query at `pose` vs a full-ring segment vector
    (anchor frame). Query encoded in the same (anchor/local) frame at `pose`."""
    PW = pts @ S._rot(pose[2]).T + pose[:2]
    sv = np.exp(1j * (PW @ L.W.T)).T @ w
    c = (np.conj(full_seg) * sv).reshape(L.N_RING, L.N_ANG)
    Br = full_seg.reshape(L.N_RING, L.N_ANG)
    svr = sv.reshape(L.N_RING, L.N_ANG)
    coh = c.sum(1).real / (np.linalg.norm(Br, axis=1)
                           * np.linalg.norm(svr, axis=1) + 1e-12)
    return coh


def _chain_of(a, pool_set):
    """Contiguous anchor chain (step<=2) containing a, restricted to pool_set."""
    ch = [a]
    j = a - 1
    while j in pool_set and a - j <= 4:
        ch.insert(0, j); j -= 1
    j = a + 1
    while j in pool_set and j - a <= 4:
        ch.append(j); j += 1
    return ch


def run_closure(name, valid_max=40.0, topk=15, n_queries=140, seed=0):
    print(f"\n=== CLOSURE DISCRIMINATION  [{name}]  "
          f"(genuine vs twin; bundled chain vs per-cluster patch) ===")
    mp = build_map(name, valid_max, keep_pts=True, ref=True)
    segvec, ringkey = mp["segvec"], mp["ringkey"]
    ref_a, valid_a, gm = mp["ref_a"], mp["valid_a"], mp["grid_mean"]
    a_head = {a: mp["a_pose"][a][2] for a in mp["a_pose"]}
    # wide local-frame matcher: seed translation 0 (candidate's own frame), so it
    # must FIND the revisit basin within the TOL_REV disk -> t_half 2.5 m.
    matcher = S.Matcher(L.ENC_MAIN, t_half=2.5, rot_half_deg=9,
                        rot_step_deg=1.5, perm=(4, L.N_ANG))
    aids = sorted(a for a in segvec if a in ringkey and valid_a.get(a, False))
    pool_all = np.array(aids)
    # queries with BOTH a genuine and a twin in the ring-key top-K (the hard case)
    rng = np.random.default_rng(seed)
    order = list(aids)
    gp = []   # genuine per-patch coherences (pooled)
    tp = []   # twin per-patch coherences (pooled)
    gb = []   # genuine bundled chain coherence (per query, best genuine chain)
    tb = []   # twin bundled chain coherence (per query, best twin chain)
    pc_gen = []   # per-query per-cluster genuine fire = max over genuine patches
    pc_twn = []   # per-query per-cluster twin fire = max over twin patches
    n_used = 0
    t0 = time.time()
    pool_set = set(aids)
    for qi, a in enumerate(order):
        if n_used >= n_queries:
            break
        qref = ref_a[a]
        # candidate pool: >= GAP_A older, valid, has ring key
        older = [c for c in aids if a - c > GAP_A]
        if len(older) < 5:
            continue
        pts, w = mp["q_pts"][a]
        if len(pts) < 20:
            continue
        qgrid = grid_from_pts(pts)
        qk = SC.ring_key(qgrid)
        d = np.array([np.linalg.norm(ringkey[c] - qk) for c in older])
        cand = [older[i] for i in np.argsort(d, kind="stable")[:topk]]
        lab = [np.linalg.norm(ref_a[c] - qref) <= TOL_REV for c in cand]
        if not any(lab) or all(lab):
            continue        # need both genuine and twin to measure separation
        n_used += 1
        # ---- PER-CLUSTER: align + coherence for EACH candidate patch ----
        patch_coh = {}
        for c, is_g in zip(cand, lab):
            seed_pose = np.array([0.0, 0.0, _yaw_seed(qgrid, gm[c], 0.0)])
            pose = _align_score(matcher, segvec[c][MAIN], pts, w, seed_pose)
            coh = _coh(segvec[c], pts, w, pose)
            sc = float(coh[:2].mean())      # fine-ring coherence (shipped cue)
            patch_coh[c] = sc
            (gp if is_g else tp).append(sc)
        g_scores = [patch_coh[c] for c, is_g in zip(cand, lab) if is_g]
        t_scores = [patch_coh[c] for c, is_g in zip(cand, lab) if not is_g]
        pc_gen.append(max(g_scores)); pc_twn.append(max(t_scores))
        # ---- BUNDLED: best genuine chain vs best twin chain (sum then score) ----
        def chain_coh(anchor):
            ch = _chain_of(anchor, pool_set)
            Bb = np.zeros(L.W.shape[0], complex)
            for c in ch:
                Bb += segvec[c]
            seed_pose = np.array([0.0, 0.0,
                                  _yaw_seed(qgrid, gm[anchor], 0.0)])
            pose = _align_score(matcher, Bb[MAIN], pts, w, seed_pose)
            coh = _coh(Bb, pts, w, pose)
            return float(coh[:2].mean())
        gb.append(max(chain_coh(c) for c, is_g in zip(cand, lab) if is_g))
        tb.append(max(chain_coh(c) for c, is_g in zip(cand, lab) if not is_g))
        if n_used % 25 == 0:
            print(f"    ... {n_used} dual-queries  ({time.time()-t0:.0f}s)",
                  flush=True)
    if n_used == 0:
        print("  no dual (genuine+twin) queries found.")
        return None
    gp, tp = np.array(gp), np.array(tp)
    pc_gen, pc_twn = np.array(pc_gen), np.array(pc_twn)
    gb, tb = np.array(gb), np.array(tb)
    print(f"  dual-queries: {n_used}   pooled patches: {len(gp)} genuine / "
          f"{len(tp)} twin   ({time.time()-t0:.0f}s)")

    def stats(g, t, tag):
        auc = _auc(g, t)
        # FP rate at the threshold giving 90% genuine recall
        thr = np.quantile(g, 0.10)
        fp = float((t >= thr).mean())
        print(f"  {tag:28} genuine med {np.median(g):.4f} (p10 {np.quantile(g,.1):.4f}) "
              f"| twin med {np.median(t):.4f} (p90 {np.quantile(t,.9):.4f}) "
              f"| AUC {auc:.3f} | FP@90%recall {fp:.3f}")
        return auc, fp
    print("  --- POOLED per-patch (each candidate scored separately) ---")
    a_pp = stats(gp, tp, "PER-CLUSTER per-patch coh")
    print("  --- PER-QUERY fire score (best genuine vs best twin per query) ---")
    a_pc = stats(pc_gen, pc_twn, "PER-CLUSTER max-over-patches")
    a_bd = stats(gb, tb, "BUNDLED chain coherence")
    # paired separation per query
    sep_pc = pc_gen - pc_twn
    sep_bd = gb - tb
    print(f"  paired separation (genuine-fire - twin-fire), median:")
    print(f"    per-cluster (max)  {np.median(sep_pc):+.4f}  "
          f"(>0 in {100*(sep_pc>0).mean():.0f}% of queries)")
    print(f"    bundled (chain)    {np.median(sep_bd):+.4f}  "
          f"(>0 in {100*(sep_bd>0).mean():.0f}% of queries)")
    return dict(gp=gp, tp=tp, pc_gen=pc_gen, pc_twn=pc_twn, gb=gb, tb=tb,
                auc_pp=a_pp[0], auc_pc=a_pc[0], auc_bd=a_bd[0])


def _auc(pos, neg):
    """AUC = P(pos score > neg score) via Mann-Whitney, ties=0.5."""
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
# EXPERIMENT 1 -- FRONTEND: bundled vs per-cluster localization (ATE)
# ---------------------------------------------------------------------------
class _PerClusterMatcher:
    """Drop-in replacement for BoundedSLAM.matcher (FRONTEND only). Instead of a
    single inner product against the summed local bundle B, it matches the scan
    against EACH recent world-placed segment SEPARATELY and keeps the pose of the
    best-cosine one. Selection is crosstalk-free and knows WHICH patch fired."""

    def __init__(self, base, slam, radius=8.0):
        self.base, self.slam, self.radius = base, slam, radius
        self.n_calls = self.n_seg = 0

    def match(self, Mbundle, pts, w, guess):
        s = self.slam
        lo = len(s.anchors) - 1 - s.recent_aids
        segs = [aid for aid in s.segvec
                if aid >= lo
                and np.linalg.norm(s.anchors[aid][:2] - guess[:2]) < self.radius]
        if not segs:
            return self.base.match(Mbundle, pts, w, guess)
        self.n_calls += 1
        best = None
        for aid in segs:
            Bi = s.world_vec_seg(aid)[MAIN]
            if not np.abs(Bi).any():
                continue
            self.n_seg += 1
            pose = self.base.match(Bi, pts, w, guess)
            pose[2] = S.wrap(pose[2])
            sv = enc_main(pts, w, pose)
            sc = cosine(Bi, sv)              # normalized: fair across seg energies
            if best is None or sc > best[0]:
                best = (sc, pose)
        return best[1] if best is not None else self.base.match(Mbundle, pts, w, guess)


class PerClusterFrontendSLAM(B.BoundedSLAM):
    """BoundedSLAM whose FRONTEND localization is per-cluster (best single recent
    segment) instead of bundled (sum). Backend / closures / map UNCHANGED."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.matcher = _PerClusterMatcher(self.matcher, self)


def _run_slam(cls, name, valid_max=40.0, verbose=True, no_loop=False):
    keys, beam, odom, kts, n_beams = load(name, valid_max)
    n = len(keys)
    slam = cls(robust=True, attempt_every=4, relax_every=25,
               gap_kf=300, recent_aids=12)
    slam.store_dtype = np.complex64
    if no_loop:                       # AUDIT: disable loop closures -> isolate
        slam.try_constraint = lambda *a, **k: None   # the frontend contribution
    est = np.zeros((n, 3))
    t0 = time.time()
    for k in range(n):
        pts, w = scan_pts(keys[k][0], beam, valid_max)
        guess = odom[k] if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
        if verbose and k % 2000 == 0:
            print(f"    kf {k}/{n}  t={time.time()-t0:.0f}s  "
                  f"loops={sum(1 for e in slam.edges if e[5]=='loop')}", flush=True)
    if slam.dirty:
        slam.relax()
    dt = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    # ATE vs RBPF-corrected gfs reference (timestamp match, ssp_bounded_carmen)
    ref = C.parse_flaser(f"data/{name}.log".replace(".log", ".gfs.log"))
    rts = np.array([t for _, _, t in ref])
    rxy = np.stack([p[:2] for _, p, _ in ref])
    j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
    good = np.abs(rts - kts[j]) < 0.3
    al = C.align_se2(fin[j[good], :2], rxy[good])
    e = np.linalg.norm(al - rxy[good], axis=1)
    ate = float(np.sqrt((e ** 2).mean()))
    n_loop = sum(1 for e2 in slam.edges if e2[5] == "loop")
    return slam, ate, float(np.median(e)), float(e.max()), n_loop, dt


def run_frontend(name, valid_max=40.0):
    print(f"\n=== FRONTEND ATE  [{name}]  bundled vs per-cluster ===")
    sb, ab, mb, xb, lb, tb = _run_slam(B.BoundedSLAM, name, valid_max)
    print(f"  BUNDLED (shipped)   ATE {ab:.3f} m  med {mb:.3f}  max {xb:.2f}  "
          f"loops {lb}  {tb:.0f}s")
    sp, ap, mp2, xp, lp, tp = _run_slam(PerClusterFrontendSLAM, name, valid_max)
    kc, ks = sp.matcher.n_calls, sp.matcher.n_seg
    print(f"  PER-CLUSTER         ATE {ap:.3f} m  med {mp2:.3f}  max {xp:.2f}  "
          f"loops {lp}  {tp:.0f}s")
    print(f"  compute: per-cluster did {ks} single-segment matches over {kc} "
          f"frontend frames ({ks/max(1,kc):.1f} matches/frame vs 1 bundled)")
    print(f"  ATE delta (per-cluster - bundled): {ap-ab:+.3f} m")
    return dict(name=name, ate_bundled=ab, ate_percluster=ap)


# ---------------------------------------------------------------------------
def selftest():
    print("selftest: per-cluster plumbing")
    rng = np.random.default_rng(0)
    # (1) world placement round-trips: a segment folded at pose p, placed to
    #     world, re-encoded query at p -> high self-cosine
    pts = rng.uniform(-4, 4, (200, 2)); w = np.full(200, 0.1)
    seg = np.exp(1j * (pts @ L.W.T)).T @ w         # anchor-frame at identity pose
    wv = _world_main(seg, np.array([1.5, -2.0, 0.0]))
    sv = enc_main(pts, w, np.array([1.5, -2.0, 0.0]))
    c = cosine(wv, sv)
    assert c > 0.98, f"world-placement self-cosine {c:.3f} (rotation=0)"
    # (2) bundling DILUTES the constituent cosine ~1/sqrt(M) (crosstalk)
    others = [np.exp(1j * (rng.uniform(-4, 4, (200, 2)) @ WMAIN.T)).T @ w
              for _ in range(15)]
    c1 = cosine(wv, sv)
    cM = cosine(wv + sum(others), sv)
    assert cM < 0.6 * c1, f"bundle of 16 did not dilute: {c1:.3f}->{cM:.3f}"
    # (3) AUC sanity
    assert abs(_auc([2, 3, 4], [0, 1]) - 1.0) < 1e-9
    assert abs(_auc([0, 1], [0, 1]) - 0.5) < 1e-9
    # (4) yaw seed + matcher: a query scan seen ROTATED by th, aligned against
    #     the anchor-frame segment, must recover the rotation (seed within the
    #     matcher's +-9 deg window) and score a high coherence.
    matcher = S.Matcher(L.ENC_MAIN, t_half=0.5, rot_half_deg=9,
                        rot_step_deg=1.5, perm=(4, L.N_ANG))
    seg_full = np.exp(1j * (pts @ L.W.T)).T @ w        # anchor-frame, identity
    th = 0.30
    qpts = pts @ S._rot(th).T                            # query seen rotated +th
    qgrid = grid_from_pts(qpts); cgrid = grid_from_pts(pts)
    yaw = _yaw_seed(qgrid, cgrid, 0.0)
    seed = np.array([0.0, 0.0, yaw])
    pose = matcher.match(seg_full[MAIN], qpts, w, seed)
    coh = _coh(seg_full, qpts, w, pose)
    assert coh[:2].mean() > 0.9, f"aligned coherence {coh[:2].mean():.3f}"
    assert abs(S.wrap(pose[2] + th)) < np.deg2rad(2), \
        f"recovered rot {np.degrees(pose[2]):.1f} vs expected {-np.degrees(th):.1f}"
    print(f"  world self-cos {c:.3f}; bundle-16 dilution {c1:.3f}->{cM:.3f} "
          f"(ratio {cM/c1:.2f}, ~1/4 expected); AUC OK; "
          f"yaw-seed+matcher recover rot to {np.degrees(pose[2]):.2f} deg, "
          f"coh {coh[:2].mean():.3f}")
    print("selftest PASSED")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    ds = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd == "selftest":
        selftest()
    elif cmd == "capacity":
        for nm in ([ds] if ds else ["intel", "mit"]):
            run_capacity(nm, 40.0 if nm != "mit" else 50.0)
    elif cmd == "closure":
        for nm in ([ds] if ds else ["intel", "mit"]):
            run_closure(nm, 40.0 if nm != "mit" else 50.0)
    elif cmd == "frontend":
        for nm in ([ds] if ds else ["fr101", "fr079", "intel"]):
            run_frontend(nm)
    elif cmd == "audit":
        # closures DISABLED: pure frontend odometry, bundled vs per-cluster, so
        # the ATE gap cannot be attributed to loop-closure luck.
        nm = ds or "fr101"
        vm = 40.0 if nm != "mit" else 50.0
        print(f"\n=== FRONTEND AUDIT (closures OFF)  [{nm}] ===")
        _, ab, mb, xb, _, tb = _run_slam(B.BoundedSLAM, nm, vm, no_loop=True)
        print(f"  BUNDLED   no-loop ATE {ab:.3f} m  med {mb:.3f}  max {xb:.2f}  {tb:.0f}s")
        sp, ap, mp2, xp, _, tp = _run_slam(PerClusterFrontendSLAM, nm, vm, no_loop=True)
        print(f"  PER-CLUST no-loop ATE {ap:.3f} m  med {mp2:.3f}  max {xp:.2f}  {tp:.0f}s")
        print(f"  frontend-only ATE delta (per-cluster - bundled): {ap-ab:+.3f} m")
    elif cmd == "all":
        for nm in ["intel", "mit"]:
            run_capacity(nm, 40.0 if nm != "mit" else 50.0)
            run_closure(nm, 40.0 if nm != "mit" else 50.0)
        for nm in ["fr101", "fr079", "intel"]:
            run_frontend(nm)
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main()
