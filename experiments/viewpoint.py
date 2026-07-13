"""VIEWPOINT as a dual channel per sub-map cell (experiment; NEW file).

Question: the shipped bounded map (sspslam/bounded.py) stores, per anchor, a
CONTENT bundle phi_W(pts_body) encoded rigidly in the anchor frame, plus the
bare anchor graph pose (used only to geometrically PLACE the content:
world_vec_seg = ENC.shift(a[:2]) * rot_permute(v, m)). It stores WHAT was seen,
not WHERE it was seen FROM.

This experiment adds a SECOND, separately-weightable channel per anchor: a
VIEWPOINT CODE. For every observation (keyframe) folded into an anchor's
segment, we remember the OBSERVATION POSITION the content was sampled from --
the robot's pose relative to the anchor (kf_ref rel[:2], anchor frame, so it
NEVER moves with the graph). At query time the viewpoint code is (re)placed with
the SAME algebra as the content:

    V(g_hat) := sum_obs  psi_view( anchor_world  o  obs_rel )         (~20 comp)
    psi_view(t) = exp(i W_view . t)   on a COARSE, SEPARATE lattice W_view

so a graph correction that moves the anchor moves V correctly (Test 3), and with
the channel OFF the content path is bit-exact shipped behavior (Test 3).

VIEWPOINT-CONSISTENCY score at a candidate with matcher hypothesis pose g_hat:

    vp(aid, g_hat) = Re<V_world(aid), psi_view(g_hat[:2])> / (|V||psi|)  in [-1,1]

"does the cell's observed-from constellation agree with where the robot now
thinks it is?"  It is a SEPARATE score alongside the shipped content coherence,
NOT baked into the content vector (graph-correctability -- content stays
body-frame and rides the anchor rigidly).

Three tests, all oracle-free (GT = REF trajectory used ONLY to LABEL genuine vs
twin and to SCORE separation; never to select candidates or seed the matcher --
g_hat always comes from the matcher):

  1  DISCRIMINATION (intel / mit): at loop candidates, genuine-vs-twin
     separation of content coherence ALONE vs content x viewpoint-consistency.
     A spatial-distance channel is reported as a CONFOUND CONTROL (viewpoint
     must beat "genuine anchors are simply closer" to count as real signal).
     PREDICTION: helps on Intel (viewpoint well-determined), NOT on MIT
     (along-corridor viewpoint unobservable, g_hat slides -- the two-fold wall).

  2  COMPOSITION (compose): two-session Intel split; per overlapping cell use the
     viewpoint codes to call SAME-observation (redundant revisit, don't
     double-count) vs NEW coverage (distinct viewpoint mass). Dedup rate + a GT
     check that viewpoint overlap tracks true viewpoint proximity.

  3  GRAPH-CORRECTABILITY (selftest): V recomputed from the (graph-corrected)
     anchor pose moves rigidly with a pose update; channel OFF == shipped
     bit-exact.

Usage:  python3 -m experiments.viewpoint [selftest|intel|mit|compose|all]
Deterministic. No shipped/prior module is edited.
"""

import os
import sys
import time

import numpy as np

import sspslam.encoder as S
import sspslam.lattice as L
import sspslam.frontend as C
import baselines.scancontext as SC
from sspslam.bounded import BoundedSLAM, ANCHOR, CELL

SCRATCH = os.path.join(os.path.dirname(__file__), "..", "scratch")

# per-dataset config
CFG = {
    "intel": dict(log="data/intel.log", valid_max=40.0, gap_kf=200,
                  tol_gen=2.0, tol_twin=4.0),
    "mit":   dict(log="data/mit.log",   valid_max=50.0, gap_kf=1000,
                  tol_gen=3.0, tol_twin=6.0),
}
R_CAND = 6.0                       # loop-candidate radius (m, estimated frame)
LAM_VIEW = (2.0, 4.0, 8.0)         # W_view gate-width sweep (m)
N_DIR = 10                         # directions in the coarse viewpoint ring
#                                    -> 10 complex comps = ~20 real DOF
TOP_K = 8                          # appearance-retrieval shortlist (drought)
N_SECTOR = SC.N_SECTOR             # Scan-Context sectors (yaw seed)
YAW_SIGN = 1.0                     # sign convention (ssp_ringkey selftest pin)


def _grid_from_pts(pts):
    """Scan-Context point-density grid from body-frame samples (reuses
    SC.scan_context via per-point range/bearing). Same as ssp_ringkey."""
    if len(pts) == 0:
        return np.zeros((SC.N_RING, SC.N_SECTOR), float)
    r = np.linalg.norm(pts, axis=1)
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    return SC.scan_context(r, ang, 50.0, 50.0, SC.N_RING, SC.N_SECTOR)


# --------------------------------------------------------------------------
#  coarse viewpoint lattice + code
# --------------------------------------------------------------------------
def build_W_view(lam, n_dir=N_DIR):
    """A single COARSE ring: n_dir directions over [0,pi) at wavelength lam.
    Separate from the content lattice L.W; ~n_dir complex components."""
    a = np.arange(n_dir) * np.pi / n_dir
    u = np.stack([np.cos(a), np.sin(a)], 1)
    return (2 * np.pi / lam) * u                       # (n_dir, 2)


def vp_world(obs_rel, anchor, Wv):
    """World-frame viewpoint code V = sum_obs exp(i Wv . (anchor o obs_rel)).
    obs_rel = list/array of observation positions in the ANCHOR frame (never
    moves with the graph); anchor = LIVE graph pose. Recomputed at query, so a
    graph update to `anchor` moves V correctly."""
    obs_rel = np.atleast_2d(obs_rel)
    R = S._rot(anchor[2])
    ow = obs_rel @ R.T + anchor[:2]                    # world obs positions
    return np.exp(1j * (ow @ Wv.T)).sum(0)             # (n_dir,) complex


def vp_score(obs_rel, anchor, g_xy, Wv):
    """Viewpoint-consistency in [-1,1]: normalized Re<V_world, psi(g_hat)>."""
    V = vp_world(obs_rel, anchor, Wv)
    psi = np.exp(1j * (np.asarray(g_xy) @ Wv.T))       # (n_dir,)
    nV = np.linalg.norm(V)
    if nV < 1e-12:
        return 0.0
    return float((np.conj(V) @ psi).real / (nV * np.linalg.norm(psi)))


# --------------------------------------------------------------------------
#  ViewpointSLAM: BoundedSLAM + observation-viewpoint recording
# --------------------------------------------------------------------------
class ViewpointSLAM(BoundedSLAM):
    """Records, per anchor, the anchor-frame observation positions of every
    keyframe folded into its segment. Pure ADD-ON: the content path
    (segvec/segder/world_vec_seg/try_constraint/relax) is inherited UNCHANGED,
    so vp_on=False is bit-exact shipped behavior."""

    def __init__(self, vp_on=True, **kw):
        super().__init__(**kw)
        self.vp_on = vp_on
        self.obs_rel = {}          # aid -> list of anchor-frame obs positions
        # per-anchor Scan-Context ring-key + mean grid (appearance retrieval;
        # body-frame, yaw-invariant, never moves with the graph -- REUSED for
        # the drought candidate harvest, same construction as ssp_ringkey)
        self.ringkey = {}
        self.sc_grid = {}
        self._grid_sum = {}
        self._grid_cnt = {}

    def add_keyframe(self, pts, w, guess):
        grid = _grid_from_pts(pts) if len(pts) else None
        est = super().add_keyframe(pts, w, guess)
        if self.vp_on:
            aid, rel = self.kf_ref[-1]
            if aid in self.segvec:                     # not evicted this frame
                self.obs_rel.setdefault(aid, []).append(rel[:2].copy())
                if grid is not None and grid.sum() > 0:
                    s = self._grid_sum.get(aid)
                    self._grid_sum[aid] = grid if s is None else s + grid
                    self._grid_cnt[aid] = self._grid_cnt.get(aid, 0) + 1
                    mean = self._grid_sum[aid] / self._grid_cnt[aid]
                    self.sc_grid[aid] = mean
                    self.ringkey[aid] = SC.ring_key(mean)
            # drop records for anchors the cell-cap just evicted
            if len(self.obs_rel) > len(self.segvec):
                for a in [a for a in self.obs_rel if a not in self.segvec]:
                    for d in (self.obs_rel, self.ringkey, self.sc_grid,
                              self._grid_sum, self._grid_cnt):
                        d.pop(a, None)
        return est


# --------------------------------------------------------------------------
#  data loading
# --------------------------------------------------------------------------
def load_log(name):
    cfg = CFG[name]
    scans = C.parse_flaser(cfg["log"])
    keys = C.keyframes(scans)
    n = len(keys)
    beam = np.deg2rad(np.arange(len(keys[0][0])) - 90.0)
    odom = np.stack([k[1] for k in keys])
    kts, ref, valid = SC.ref_positions(name, keys)
    return dict(name=name, keys=keys, n=n, beam=beam, odom=odom,
                kts=kts, ref=ref, valid=valid, cfg=cfg)


def build_map(blob, lo=0, hi=None, vp_on=True, verbose=True):
    """Run ViewpointSLAM over kf[lo:hi]; return the built slam + est trajectory.
    Same frontend chaining as ssp_multisession.build_session."""
    keys, beam, odom = blob["keys"], blob["beam"], blob["odom"]
    cfg = blob["cfg"]
    hi = blob["n"] if hi is None else hi
    slam = ViewpointSLAM(vp_on=vp_on, robust=True, attempt_every=3,
                         relax_every=5, gap_kf=cfg["gap_kf"], recent_aids=12)
    slam.store_dtype = np.complex64
    idx = np.arange(lo, hi)
    est = np.zeros((len(idx), 3))
    t0 = time.time()
    for j, k in enumerate(idx):
        r = keys[k][0]
        rr = np.where(r < cfg["valid_max"], r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = np.zeros(3) if j == 0 else L.se2_mul(
            est[j - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[j] = slam.add_keyframe(pts, w, guess)
        if verbose and j % 2000 == 0 and j:
            print(f"    ...{j}/{len(idx)} kf ({time.time()-t0:.0f}s)", flush=True)
    if slam.dirty:
        slam.relax()
    fin = np.stack([slam.pose_of(j) for j in range(len(idx))])
    if verbose:
        nl = sum(1 for e in slam.edges if e[5] == "loop")
        print(f"  built {len(idx)} kf, {len(slam.segvec)} anchors, {nl} loop "
              f"edges ({time.time()-t0:.0f}s)", flush=True)
    return slam, est, fin, idx


# --------------------------------------------------------------------------
#  separation metrics
# --------------------------------------------------------------------------
def auc(pos, neg):
    """Rank AUC = P(score(genuine) > score(twin)). 0.5 = chance."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="stable")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fp_at_recall(pos, neg, recall=0.8):
    """False-positive rate (twins admitted) at a threshold giving `recall` of
    genuine. Lower is better."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    tau = np.quantile(pos, 1 - recall)                 # admit score >= tau
    return float((neg >= tau).mean())


# --------------------------------------------------------------------------
#  TEST 1 -- discrimination
# --------------------------------------------------------------------------
def _score_candidate(slam, aid, pts, w, seed, Wvs, gate_xy, gate_th=np.deg2rad(7)):
    """Match query scan to anchor `aid`'s content seeded at `seed`, gated.
    Returns (coh01, [vp per lam], g_hat_xy) or None if the match fails the gate.
    Oracle-free: g_hat is the matcher output."""
    B = slam.world_vec_seg(aid)
    pose = slam.cmatcher.match(B[L.MAIN], pts, w, seed)
    pose[2] = S.wrap(pose[2])
    if np.linalg.norm(pose[:2] - seed[:2]) > gate_xy \
            or abs(S.wrap(pose[2] - seed[2])) > gate_th:
        return None
    sv = L.ENC.shift(pose[:2]) * L.encode(pts @ S._rot(pose[2]).T, w)
    cc = (np.conj(B) * sv).reshape(L.N_RING, L.N_ANG)
    Br = B.reshape(L.N_RING, L.N_ANG)
    svr = sv.reshape(L.N_RING, L.N_ANG)
    coh = (cc.sum(1).real / (np.linalg.norm(Br, axis=1)
           * np.linalg.norm(svr, axis=1) + 1e-12))
    obs = np.array(slam.obs_rel[aid])
    vps = [vp_score(obs, slam.anchors[aid], pose[:2], Wv) for Wv in Wvs]
    return float(coh[:2].mean()), vps, pose[:2]


def harvest_loop(name, blob, slam, fin, idx, q_step=1):
    """At each (subsampled) query keyframe, take LOOP candidates = temporally
    distant anchors within R_CAND of the ESTIMATED pose, seed the matcher there
    (exactly the shipped try_constraint candidate path), and record for each:
    content coherence, viewpoint-consistency (per lam), anchor->me distance
    (confound control), and the GT genuine/twin label.  Oracle-free: GT only
    labels; the matcher produces g_hat."""
    keys, beam, ref, valid = blob["keys"], blob["beam"], blob["ref"], blob["valid"]
    cfg = blob["cfg"]
    gap_a = cfg["gap_kf"] // ANCHOR
    Wvs = [build_W_view(lam) for lam in LAM_VIEW]
    # anchor -> its center keyframe (global index) for GT labelling
    a_ref = {}
    for aid in slam.segvec:
        kc = aid * ANCHOR
        if kc < len(idx):
            a_ref[aid] = idx[kc]
    a_xy = {aid: slam.anchors[aid][:2] for aid in slam.segvec}

    rows = []                       # (coh, vp0,vp1,vp2, dist, label)  label 1=gen 0=twin
    n_q = 0
    for j in range(0, len(idx), q_step):
        k = idx[j]
        if not valid[k]:
            continue
        me = fin[j]
        r = keys[k][0]
        rr = np.where(r < cfg["valid_max"], r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        if len(pts) < 20:
            continue
        cands = [aid for aid in slam.segvec
                 if abs(aid - (j // ANCHOR)) > gap_a
                 and aid in a_ref
                 and np.linalg.norm(a_xy[aid] - me[:2]) < R_CAND]
        if not cands:
            continue
        n_q += 1
        for aid in cands:
            res = _score_candidate(slam, aid, pts, w, me, Wvs, gate_xy=0.6)
            if res is None:                            # failed shipped loop gate
                continue
            coh01, vps, ghat = res
            dist = float(np.linalg.norm(a_xy[aid] - me[:2]))
            # GT label (REF frame): is the query truly revisiting this anchor?
            dref = float(np.linalg.norm(ref[k] - ref[a_ref[aid]]))
            if dref < cfg["tol_gen"]:
                label = 1
            elif dref > cfg["tol_twin"] and coh01 > 0.0:
                label = 0
            else:
                continue                               # ambiguous band
            rows.append([coh01, vps[0], vps[1], vps[2], dist, label])
    return np.array(rows), n_q


def harvest_drought(name, blob, slam, fin, idx, q_step=3):
    """DROUGHT / relocalization candidates: rank temporally-distant anchors by
    APPEARANCE (ring-key kNN, oracle-free), seed the matcher at the candidate
    anchor's live graph pose + a Scan-Context column-shift yaw seed, and gate
    around the candidate. This surfaces genuine revisits even when drift keeps
    them far apart in the estimate (needed on MIT). g_hat lands near the
    candidate, so the confound here is CONTENT-VIEWPOINT COUPLING (does vp add
    signal beyond the content lock?) rather than spatial proximity."""
    keys, beam, ref, valid = blob["keys"], blob["beam"], blob["ref"], blob["valid"]
    cfg = blob["cfg"]
    gap_a = cfg["gap_kf"] // ANCHOR
    Wvs = [build_W_view(lam) for lam in LAM_VIEW]
    a_ref = {aid: idx[aid * ANCHOR] for aid in slam.segvec if aid * ANCHOR < len(idx)}
    old_all = np.array(sorted(a for a in slam.segvec
                              if a in slam.ringkey and a in a_ref), dtype=int)
    if old_all.size == 0:
        return np.zeros((0, 6)), 0
    RK_all = np.stack([slam.ringkey[a] for a in old_all])

    rows, n_q = [], 0
    for j in range(0, len(idx), q_step):
        k = idx[j]
        if not valid[k]:
            continue
        r = keys[k][0]
        rr = np.where(r < cfg["valid_max"], r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        if len(pts) < 20:
            continue
        my_aid = j // ANCHOR
        sel = np.abs(old_all - my_aid) > gap_a
        if not sel.any():
            continue
        qgrid = _grid_from_pts(pts)
        qk = SC.ring_key(qgrid)
        pool = old_all[sel]
        d = np.linalg.norm(RK_all[sel] - qk, axis=1)
        short = pool[np.argsort(d, kind="stable")[:TOP_K]]
        n_q += 1
        for aid in short:
            _, s = SC.sc_distance(qgrid, slam.sc_grid[aid])
            yaw = S.wrap(slam.anchors[aid][2] + YAW_SIGN * s * (2 * np.pi / N_SECTOR))
            seed = np.array([slam.anchors[aid][0], slam.anchors[aid][1], yaw])
            res = _score_candidate(slam, aid, pts, w, seed, Wvs, gate_xy=0.7,
                                   gate_th=np.deg2rad(9))
            if res is None:
                continue
            coh01, vps, ghat = res
            dref = float(np.linalg.norm(ref[k] - ref[a_ref[aid]]))
            if dref < cfg["tol_gen"]:
                label = 1
            elif dref > cfg["tol_twin"] and coh01 > 0.0:
                label = 0
            else:
                continue
            # 'dist' here = anchor->me distance in the estimate (info only)
            dist = float(np.linalg.norm(slam.anchors[aid][:2] - fin[j][:2]))
            rows.append([coh01, vps[0], vps[1], vps[2], dist, label])
    return np.array(rows), n_q


def report_discrimination(name, rows, n_q, log, mode="loop"):
    def pr(s):
        print(s, flush=True)
        log.write(s + "\n")
    pr(f"\n=== TEST 1 DISCRIMINATION: {name.upper()} [{mode}] ===")
    if rows.size == 0:
        pr(f"  queries {n_q}; no gated candidate matches."); return
    lab = rows[:, 5].astype(int)
    gen, twin = rows[lab == 1], rows[lab == 0]
    pr(f"  queries with candidates: {n_q};  candidate matches labelled: "
       f"{len(rows)}  (genuine {len(gen)}, twin {len(twin)})")
    if len(gen) == 0 or len(twin) == 0:
        pr("  insufficient population for AUC (need both genuine and twin)."); return
    coh_g, coh_t = gen[:, 0], twin[:, 0]
    # confound control: spatial proximity (closer = more genuine -> negate dist)
    dist_g, dist_t = -gen[:, 4], -twin[:, 4]
    pr(f"  content coherence   genuine med {np.median(coh_g):.3f}  "
       f"twin med {np.median(coh_t):.3f}")
    a_coh = auc(coh_g, coh_t)
    a_dist = auc(dist_g, dist_t)
    pr(f"  AUC  content-coherence ALONE      : {a_coh:.3f}")
    pr(f"  AUC  spatial-proximity (CONFOUND) : {a_dist:.3f}   "
       f"(viewpoint must beat this to be real)")
    pr(f"  FP@recall0.8  content-only        : {fp_at_recall(coh_g, coh_t):.3f}")
    for i, lam in enumerate(LAM_VIEW):
        vp_g, vp_t = gen[:, 1 + i], twin[:, 1 + i]
        a_vp = auc(vp_g, vp_t)
        # content x viewpoint (clip vp to [0,1] so it gates, not flips sign)
        prod_g = coh_g * np.clip(vp_g, 0, 1)
        prod_t = coh_t * np.clip(vp_t, 0, 1)
        a_prod = auc(prod_g, prod_t)
        pr(f"  lam_view={lam:>4.1f} m | AUC vp-alone {a_vp:.3f} | "
           f"AUC content x vp {a_prod:.3f} | "
           f"FP@0.8 content x vp {fp_at_recall(prod_g, prod_t):.3f} | "
           f"vp med gen {np.median(vp_g):+.3f} twin {np.median(vp_t):+.3f}")
    # verdict
    best_prod = max(auc(gen[:, 0] * np.clip(gen[:, 1 + i], 0, 1),
                        twin[:, 0] * np.clip(twin[:, 1 + i], 0, 1))
                    for i in range(len(LAM_VIEW)))
    best_vp = max(auc(gen[:, 1 + i], twin[:, 1 + i]) for i in range(len(LAM_VIEW)))
    pr(f"  -> content {a_coh:.3f}  best(content x vp) {best_prod:.3f}  "
       f"(delta {best_prod - a_coh:+.3f});  best vp-alone {best_vp:.3f} vs "
       f"confound {a_dist:.3f} (delta {best_vp - a_dist:+.3f})")


def test_discrimination(name, log):
    print(f"\n########## {name.upper()} ##########", flush=True)
    blob = load_log(name)
    print(f"  {name}: {blob['n']} keyframes, "
          f"{blob['valid'].sum()} in REF span", flush=True)
    slam, est, fin, idx = build_map(blob, vp_on=True)
    # honest map-quality note (ATE vs REF over valid span)
    v = blob["valid"][idx]
    ate = np.linalg.norm(fin[v, :2] - blob["ref"][idx[v]], axis=1)
    print(f"  map ATE vs REF (valid span): rmse {np.sqrt((ate**2).mean()):.2f} m",
          flush=True)
    qs = 2 if name == "intel" else 4
    # LOOP path (seed at estimated pose; the shipped try_constraint candidate
    # geometry) -- where g_hat reflects the robot's independent belief.
    rl, nl = harvest_loop(name, blob, slam, fin, idx, q_step=qs)
    report_discrimination(name, rl, nl, log, mode="loop: seed@me")
    # DROUGHT path (appearance retrieval; seed at candidate) -- surfaces genuine
    # revisits even under heavy drift (needed on MIT); tests coupling.
    rd, nd = harvest_drought(name, blob, slam, fin, idx, q_step=qs)
    report_discrimination(name, rd, nd, log, mode="drought: appearance+seed@anchor")
    # memory footprint of the channel
    nobs = sum(len(v) for v in slam.obs_rel.values())
    log.write(f"  viewpoint channel storage: {nobs} obs offsets over "
              f"{len(slam.obs_rel)} anchors = {2*nobs} floats "
              f"(~{2*nobs/max(len(slam.obs_rel),1):.0f} floats/anchor; "
              f"~{N_DIR} complex comps recomputed at query)\n")
    return rl, rd


# --------------------------------------------------------------------------
#  TEST 2 -- composition dedup on the two-session Intel split
# --------------------------------------------------------------------------
def test_compose(log):
    def pr(s):
        print(s, flush=True)
        log.write(s + "\n")
    pr("\n########## COMPOSE (two-session Intel split) ##########")
    blob = load_log("intel")
    n = blob["n"]
    a0, a1 = 0, int(0.60 * n)
    b0, b1 = int(0.40 * n), n
    pr(f"  A=kf[{a0}:{a1}]  B=kf[{b0}:{b1}]  (overlap kf[{b0}:{a1}])")
    slamA, _, _, idxA = build_map(blob, a0, a1, vp_on=True, verbose=False)
    slamB, _, _, idxB = build_map(blob, b0, b1, vp_on=True, verbose=False)
    ref = blob["ref"]

    # Re-anchor both sessions onto the GT (REF) frame so they share ONE frame
    # (free: segments ride rigidly). REF gives only position; use the session's
    # own heading (relative), consistent with ssp_multisession.world_gt intent.
    # For the dedup accounting we need, per (session, cell): the viewpoint code
    # V in the common frame + the true observation positions (REF).
    Wv = build_W_view(4.0)

    def session_cells(slam, idx):
        cells = {}                 # cell -> dict(V, obs_ref list)
        for aid in slam.segvec:
            kc = aid * ANCHOR
            if kc >= len(idx) or aid not in slam.obs_rel:
                continue
            kglob = idx[kc]
            if not blob["valid"][kglob]:
                continue
            # common-frame anchor pose: REF position of the anchor center kf +
            # the session-relative heading (heading cancels in same-cell overlap
            # comparison; position is what the viewpoint gate resolves)
            pose = np.array([ref[kglob][0], ref[kglob][1], slam.anchors[aid][2]])
            obs = np.array(slam.obs_rel[aid])
            R = S._rot(pose[2])
            obs_w = obs @ R.T + pose[:2]              # common-frame obs positions
            # true (REF) obs positions: REF of each contributing keyframe
            kk = [kc + t for t in range(len(obs)) if kc + t < len(idx)]
            obs_ref = np.array([ref[idx[q]] for q in kk if blob["valid"][idx[q]]])
            cell = tuple((pose[:2] // CELL).astype(int))
            d = cells.setdefault(cell, dict(V=np.zeros(N_DIR, complex),
                                            obs_w=[], obs_ref=[]))
            d["V"] += np.exp(1j * (obs_w @ Wv.T)).sum(0)
            d["obs_w"].append(obs_w)
            if len(obs_ref):
                d["obs_ref"].append(obs_ref)
        return cells

    cA, cB = session_cells(slamA, idxA), session_cells(slamB, idxB)
    overlap = sorted(set(cA) & set(cB))
    pr(f"  A cells {len(cA)}, B cells {len(cB)}, OVERLAP cells {len(overlap)}")
    if not overlap:
        pr("  no overlapping cells."); return

    THRESH = 0.5                    # viewpoint-overlap dedup threshold
    vp_ov, true_gap, redundant = [], [], 0
    for cell in overlap:
        VA, VB = cA[cell]["V"], cB[cell]["V"]
        ov = float((np.conj(VA) @ VB).real
                   / (np.linalg.norm(VA) * np.linalg.norm(VB) + 1e-12))
        vp_ov.append(ov)
        # GT check: true min viewpoint gap between an A-obs and a B-obs (REF)
        oa = np.vstack(cA[cell]["obs_ref"]) if cA[cell]["obs_ref"] else None
        ob = np.vstack(cB[cell]["obs_ref"]) if cB[cell]["obs_ref"] else None
        if oa is not None and ob is not None:
            gap = float(np.min(np.linalg.norm(
                oa[:, None, :] - ob[None, :, :], axis=2)))
        else:
            gap = np.nan
        true_gap.append(gap)
        if ov > THRESH:
            redundant += 1
    vp_ov = np.array(vp_ov)
    true_gap = np.array(true_gap)
    pr(f"  viewpoint-overlap over overlap cells: med {np.nanmedian(vp_ov):.3f}  "
       f"[{np.nanmin(vp_ov):+.2f},{np.nanmax(vp_ov):+.2f}]")
    pr(f"  DEDUP (overlap>{THRESH} = same viewpoint / redundant revisit): "
       f"{redundant}/{len(overlap)} = {redundant/len(overlap):.2f}")
    pr(f"  blind bundling double-counts ALL {len(overlap)} overlap cells; "
       f"viewpoint-aware flags {redundant} as revisits, "
       f"{len(overlap)-redundant} as NEW coverage")
    # GT validation: does viewpoint overlap track true viewpoint proximity?
    ok = np.isfinite(true_gap)
    if ok.sum() > 3:
        c = np.corrcoef(vp_ov[ok], -true_gap[ok])[0, 1]
        near = true_gap[ok] < 2.0
        pr(f"  GT: true viewpoint gap med {np.nanmedian(true_gap):.2f} m; "
           f"corr(vp-overlap, -true_gap) = {c:+.3f}")
        if near.any() and (~near).any():
            pr(f"    vp-overlap where true gap<2m: {np.median(vp_ov[ok][near]):.3f}"
               f"  vs gap>=2m: {np.median(vp_ov[ok][~near]):.3f}")


# --------------------------------------------------------------------------
#  TEST 3 -- graph-correctability + channel-off == shipped
# --------------------------------------------------------------------------
def test_selftest(log):
    def pr(s):
        print(s, flush=True)
        log.write(s + "\n")
    pr("\n########## SELFTEST (graph-correctability) ##########")
    rng = np.random.default_rng(0)
    Wv = build_W_view(4.0)

    # (a) rigid-invariance: apply a rigid transform T to BOTH the anchor pose
    #     and the hypothesis g_hat -> vp_score is INVARIANT (V is placed from
    #     the anchor pose, so it rides the graph correction exactly).
    obs = rng.uniform(-1.5, 1.5, (5, 2))
    anchor = np.array([3.0, -2.0, 0.7])
    g = np.array([3.4, -1.7])
    s0 = vp_score(obs, anchor, g, Wv)
    worst = 0.0
    for _ in range(200):
        th = rng.uniform(-np.pi, np.pi)
        t = rng.uniform(-20, 20, 2)
        R = S._rot(th)
        anchor2 = np.array([*(R @ anchor[:2] + t), S.wrap(anchor[2] + th)])
        g2 = R @ g + t
        worst = max(worst, abs(vp_score(obs, anchor2, g2, Wv) - s0))
    pr(f"  (a) vp_score rigid-invariance under graph correction: "
       f"max |delta| over 200 SE(2) moves = {worst:.2e}  "
       f"({'PASS' if worst < 1e-9 else 'FAIL'})")

    # (b) a pose UPDATE moves V correctly: recomputing V from a shifted anchor
    #     equals phase-shifting the world code by the same translation.
    d = np.array([0.37, -0.21])
    V1 = vp_world(obs, anchor, Wv)
    V2 = vp_world(obs, anchor + np.array([d[0], d[1], 0.0]), Wv)
    err = float(np.abs(V2 - np.exp(1j * (Wv @ d)) * V1).max())
    pr(f"  (b) V(anchor+d) == shift(d)*V(anchor): max |delta| = {err:.2e}  "
       f"({'PASS' if err < 1e-9 else 'FAIL'})")

    # (c) channel OFF == shipped BoundedSLAM bit-exact (content path untouched)
    blob = load_log("intel")
    N = 600
    keys, beam, odom, cfg = blob["keys"], blob["beam"], blob["odom"], blob["cfg"]

    def run(cls, **kw):
        S.RNG = np.random.default_rng(7)
        slam = cls(robust=True, attempt_every=3, relax_every=5,
                   gap_kf=cfg["gap_kf"], recent_aids=12, **kw)
        slam.store_dtype = np.complex64
        est = np.zeros((N, 3))
        for k in range(N):
            r = keys[k][0]
            rr = np.where(r < cfg["valid_max"], r, np.inf)
            pts, w, _ = S.scan_to_samples(rr, beam)
            guess = np.zeros(3) if k == 0 else L.se2_mul(
                est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
            est[k] = slam.add_keyframe(pts, w, guess)
        if slam.dirty:
            slam.relax()
        return np.stack([slam.pose_of(k) for k in range(N)])

    base = run(BoundedSLAM)
    off = run(ViewpointSLAM, vp_on=False)
    on = run(ViewpointSLAM, vp_on=True)
    d_off = float(np.abs(base - off).max())
    d_on = float(np.abs(base - on).max())
    pr(f"  (c) channel OFF vs shipped BoundedSLAM: max |traj delta| = {d_off:.2e}  "
       f"({'PASS bit-exact' if d_off == 0.0 else 'FAIL'})")
    pr(f"      channel ON  vs shipped (content path must ALSO be unchanged): "
       f"max |traj delta| = {d_on:.2e}  "
       f"({'PASS' if d_on == 0.0 else 'FAIL'})")


# --------------------------------------------------------------------------
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(SCRATCH, exist_ok=True)
    logpath = os.path.join(SCRATCH, f"scratch_viewpoint_{cmd}.log")
    with open(logpath, "w", buffering=1) as log:
        log.write(f"# viewpoint experiment [{cmd}] {time.ctime()}\n")
        if cmd in ("selftest", "all"):
            test_selftest(log)
        if cmd in ("intel", "all"):
            test_discrimination("intel", log)
        if cmd in ("mit", "all"):
            test_discrimination("mit", log)
        if cmd in ("compose", "all"):
            test_compose(log)
    print(f"\nwrote {logpath}", flush=True)


if __name__ == "__main__":
    main()
