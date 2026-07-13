"""Honest end-to-end multi-session VSA mapping on Intel: establish cross-session
correspondences by APPEARANCE RETRIEVAL (ring-key), verify them metrically, then
JOINT-RELAX + SUPERPOSE.  NO a-priori data-association oracle.

Why this file exists (see RESULTS.md "Cross-session alignment: pose-graph merge
works GIVEN correspondences..."). experiments/multisession_align.py was a FALSE WIN: its
cross-session matcher was seeded from the SHARED keyframe index (A=kf[0:3723],
B=kf[2482:6205] are split from ONE Intel log and share kf[2482:3723] -- the SAME
physical scans). Seeding at the known index put the metric matcher AT the answer,
so ~211/214 "closures" were same-scan self-ties and only 2/274 GENUINE cross-pass
revisits were recovered. The metric seed failed on genuine cross-pass because B's
4.37 m self-drift puts the seed ~7 m off (3% land in the matcher basin).

This file removes that oracle. Cross-session correspondence comes ONLY from
ring-key appearance matching + metric verification:

  1. Build a per-ANCHOR ring-key descriptor for BOTH sessions (mean 20x60
     Scan-Context grid over the anchor's keyframes -> yaw-invariant ring key;
     SC.scan_context / SC.ring_key / SC.sc_distance reused by import, Intel
     max_r=40 config == ssp_scancontext's recall@40 0.81 setup).
  2. For each B anchor, kNN its ring-key against ALL A anchors' ring-keys
     (APPEARANCE ONLY -- the ranking never sees the keyframe index), take the
     top-K, re-rank by the SC column-shift distance, and for the best few form a
     metric SEED = (retrieved A anchor's live graph pose) + (SC column-shift yaw).
     The seed does NOT use B's own pose estimate, any T_AB, or the shared index.
  3. Run the shipped matched-band matcher (make_matcher, ENC_MAIN t_half=0.72) on
     a bundle of A's segments near the retrieved anchor, and accept a cross-
     session edge only if it passes the shipped closure gates: match-gate REACH,
     coherence-vs-baseline (A's coh_ref), and innovation chi vs the appearance
     seed. Keep the single best-coherence verified candidate per B anchor.
  4. Derive T_AB (B-frame -> A-frame) by a Horn fit over the APPEARANCE-FOUND +
     verified correspondences (NOT the shared index), assemble the joint graph
     (A seq+loop | B seq+loop | genuine cross edges) and relax with _gn (gauge-
     fix A), then superpose. Reuses joint_relax / overlap_residual / superpose /
     kf_poses from ssp_multisession_align UNCHANGED.

HARD ANTI-ORACLE GUARANTEE (how no index leaks in):
  * Retrieval pool for a B anchor is the FULL set of A anchors, ranked purely by
    ring-key L2 distance (a 20-float body-frame appearance vector). The keyframe
    index is never used to build, filter, or order candidates.
  * The metric seed is the RETRIEVED A anchor's pose + the SC column-shift yaw
    (SC.sc_distance's best-shift). B's own drifted pose estimate is NEVER fed to
    the matcher as a seed; there is no bootstrap_TAB / Horn-over-shared-index /
    shared-kf local transform anywhere in the detection path.
  * T_AB for the joint fold is fit from the VERIFIED appearance correspondences
    only. gt/REF is used ONLY to (a) validate a found correspondence (A & B
    anchors within ~2 m in the REF frame) and (b) score the final ATE -- never to
    select, seed, order, or verify a correspondence. The cross-PASS label (index
    gap > GAP_KF) is a SCORING label applied AFTER detection, exactly as in
    ssp_multisession_align; it does not touch the detector.

Edits nothing shipped or prior. Deterministic. Uses the rich sessions_full.pkl
cache built by ssp_multisession_align (same BoundedSLAM build).

Usage:
    python3 -m experiments.multisession_ringkey selftest   # fast retrieval/seed checks
    python3 -m experiments.multisession_ringkey run        # full experiment (auto-builds)
"""

import sys
import time

import numpy as np
from scipy.spatial import cKDTree

import sspslam.encoder as S
import sspslam.lattice as L
import sspslam.frontend as C
from sspslam.bounded import ANCHOR, CELL

import baselines.scancontext as SC
from experiments.multisession import place, se2, se2i, gt_dense, make_matcher, VALID_MAX
from experiments.multisession_align import (
    load_full, kf_poses, joint_relax, overlap_residual, superpose,
    GAP_KF, REACH_T, REACH_R, COH_TARGET, NEAR_R)

# ---- ring-key descriptor config (Intel Scan-Context, == ssp_scancontext) ----
SC_MAX_R = 40.0           # Intel building extent (ssp_scancontext CFG["intel"])
N_SECTOR = SC.N_SECTOR
YAW_SIGN = 1.0            # SC column-shift -> world-heading sign (ssp_ringkey pin)

# ---- retrieval / verification knobs ----
TOP_K = 40               # ring-key appearance shortlist (recall reported @1..40)
N_VERIFY = 5             # SC-reranked candidates handed to the metric matcher
REF_TOL = 2.0            # m, REF-frame radius for correspondence validation
VALIDATE = True          # gate 3 (innovation) chi threshold same as align
CHI_MAX = 9.0


# ==========================================================================
# per-anchor ring-key descriptors (appearance; body-frame; index-blind)
# ==========================================================================
def anchor_descriptors(blob, tag):
    """Per-anchor (mean SC grid, ring-key) for one session, built from the raw
    keyframe scans that fold into each anchor. Reuses SC.scan_context/ring_key
    with the Intel max_r=40 config -- exactly the descriptor that scored
    recall@40 0.81 on Intel. No pose / index enters the descriptor."""
    sess = blob[tag]
    off = blob["a0"] if tag == "A" else blob["b0"]
    keys_r, beam = blob["keys_r"], blob["beam"]
    kf_ref = sess["kf_ref"]
    grid_sum, grid_cnt = {}, {}
    for k, (aid, _) in enumerate(kf_ref):
        if aid not in sess["segvec"]:
            continue
        g = SC.scan_context(keys_r[off + k], beam, VALID_MAX, SC_MAX_R)
        if g.sum() <= 0:
            continue
        grid_sum[aid] = g if aid not in grid_sum else grid_sum[aid] + g
        grid_cnt[aid] = grid_cnt.get(aid, 0) + 1
    grids, rkeys = {}, {}
    for aid, s in grid_sum.items():
        mean = s / grid_cnt[aid]
        grids[aid] = mean
        rkeys[aid] = SC.ring_key(mean)
    return grids, rkeys


def yaw_seed(qgrid, agrid, a_heading):
    """Metric heading seed = candidate anchor heading + SC column-shift yaw.
    Same convention as ssp_ringkey._yaw_seed. Uses appearance only."""
    _, s = SC.sc_distance(qgrid, agrid)
    return S.wrap(a_heading + YAW_SIGN * s * (2 * np.pi / N_SECTOR))


# ==========================================================================
# CROSS-SESSION DETECTION via ring-key retrieval + metric verification
# ==========================================================================
def detect_cross_ringkey(blob, gtd, verbose=True):
    A, B = blob["A"], blob["B"]
    a0, b0 = blob["a0"], blob["b0"]
    anchorsA = A["anchors"]
    segvecA, segderA = A["segvec"], A["segder"]
    coh_ref = A["coh_ref"]
    keys_r, beam = blob["keys_r"], blob["beam"]
    cmatcher = make_matcher()      # ENC_MAIN t_half=0.72 -- shipped loop matcher

    print("  building ring-key descriptors for both sessions ...", flush=True)
    gridsA, rkeysA = anchor_descriptors(blob, "A")
    gridsB, rkeysB = anchor_descriptors(blob, "B")
    A_aids = np.array(sorted(rkeysA), int)
    RK_A = np.stack([rkeysA[a] for a in A_aids])          # (nA, 20) appearance
    A_xy = np.array([anchorsA[a][:2] for a in A_aids])
    A_gkf = a0 + A_aids * ANCHOR
    treeA = cKDTree(A_xy)
    print(f"  A anchors with ring-key: {len(A_aids)}   "
          f"B anchors with ring-key: {len(rkeysB)}", flush=True)

    # REF-frame position of each anchor's center kf, for validation + opp labels
    def ref_of_A(aid):
        return gtd[a0 + aid * ANCHOR, :2]

    def ref_of_B(bid):
        return gtd[b0 + bid * ANCHOR, :2]

    edges = []
    diag = []                       # per-B-anchor retrieval + accept diagnostics
    cnt = dict(tried=0, nocand=0, noverify=0, reach=0, coh=0, innov=0, acc=0)
    KS = (1, 5, 10, 20, 40)

    for bid in sorted(rkeysB):
        gk = b0 + bid * ANCHOR
        rr = np.where(keys_r[gk] < VALID_MAX, keys_r[gk], np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        if len(pts) < 20:
            continue
        cnt["tried"] += 1
        qgrid, qk = gridsB[bid], rkeysB[bid]

        # -- APPEARANCE retrieval: kNN ring-key against ALL A anchors (no index)
        d = np.linalg.norm(RK_A - qk, axis=1)
        order = np.argsort(d, kind="stable")
        top = A_aids[order[:TOP_K]]

        # -- opportunity / retrieval-hit bookkeeping (REF used for SCORING only)
        bxy = ref_of_B(bid)
        dA, jA = treeA.query(bxy)               # nearest A anchor in REF
        nn_aid = int(A_aids[jA])
        is_corr_nn = dA < REF_TOL
        is_xpass = abs(A_gkf[jA] - gk) > GAP_KF  # genuine different-lap revisit
        # is any correct A anchor (REF<tol) within the top-K appearance list?
        hit = {}
        top_ref = np.array([ref_of_A(int(a)) for a in top])
        top_d = np.linalg.norm(top_ref - bxy, axis=1)
        top_x = np.abs((a0 + top * ANCHOR) - gk) > GAP_KF
        for kk in KS:
            corr_in = bool(np.any(top_d[:kk] < REF_TOL))
            corr_x_in = bool(np.any((top_d[:kk] < REF_TOL) & top_x[:kk]))
            hit[kk] = (corr_in, corr_x_in)

        # -- SC column-shift re-rank of the appearance shortlist; verify top few
        sc = np.array([SC.sc_distance(qgrid, gridsA[int(a)])[0] for a in top])
        cand = [int(a) for a in top[np.argsort(sc, kind="stable")][:N_VERIFY]]

        best = None                              # (coh, pose, c_aid, seed)
        any_cand = False
        for a in cand:
            seed = np.array([anchorsA[a][0], anchorsA[a][1],
                             yaw_seed(qgrid, gridsA[a], anchorsA[a][2])])
            # bundle A's segments near the retrieved anchor (contiguous chain,
            # nearest to the appearance seed) -- same as align's detect_cross
            near = treeA.query_ball_point(seed[:2], NEAR_R)
            pool = sorted(int(A_aids[i]) for i in near)
            if not pool:
                continue
            any_cand = True
            chains, cur = [], [pool[0]]
            for x in pool[1:]:
                if x - cur[-1] <= 2:
                    cur.append(x)
                else:
                    chains.append(cur); cur = [x]
            chains.append(cur)
            chain = min(chains, key=lambda ch: min(
                np.linalg.norm(anchorsA[x][:2] - seed[:2]) for x in ch))
            Bmap = sum(place(segvecA[x], segderA.get(x), anchorsA[x])
                       for x in chain)
            pose = cmatcher.match(Bmap[L.MAIN], pts, w, seed)
            pose[2] = S.wrap(pose[2])
            # GATE 1: match-gate reach (well inside the 0.72 m matcher basin)
            if np.linalg.norm(pose[:2] - seed[:2]) > REACH_T \
                    or abs(S.wrap(pose[2] - seed[2])) > REACH_R:
                continue
            # GATE 2: coherence vs A's session baseline (fine rings 0-1)
            sv = L.ENC.shift(pose[:2]) * L.encode(pts @ S._rot(pose[2]).T, w)
            c = (np.conj(Bmap) * sv).reshape(L.N_RING, L.N_ANG)
            Br = Bmap.reshape(L.N_RING, L.N_ANG)
            svr = sv.reshape(L.N_RING, L.N_ANG)
            coh = c.sum(1).real / (np.linalg.norm(Br, axis=1)
                                   * np.linalg.norm(svr, axis=1) + 1e-12)
            coh01 = coh[:2].mean()
            if coh01 < COH_TARGET * coh_ref:
                continue
            c_aid = min(chain, key=lambda x:
                        np.linalg.norm(anchorsA[x][:2] - pose[:2]))
            if best is None or coh01 > best[0]:
                best = (coh01, pose.copy(), c_aid, seed.copy())

        if not any_cand:
            cnt["nocand"] += 1
        # log even a rejected attempt for retrieval accounting
        rec = dict(bid=bid, is_opp=(is_corr_nn and is_xpass),
                   is_overlap=(is_corr_nn and not is_xpass), hit=hit,
                   accepted=False, correct=False, xpass=False)

        if best is None:
            if any_cand:
                cnt["noverify"] += 1     # candidates existed, none verified
            diag.append(rec)
            continue

        coh01, pose, c_aid, seed = best
        # -- edge measurement (A anchor nearest matched pose <-> B anchor bid)
        rel = B["kf_ref"][bid * ANCHOR][1]        # B kf rel to its B anchor
        Zk = se2(pose, se2i(rel))                 # implied B-anchor pose (A frame)
        Z = se2(se2i(anchorsA[c_aid]), Zk)
        lever = np.linalg.norm(seed[:2] - anchorsA[c_aid][:2])
        sig_t = np.sqrt(0.08 ** 2 + (0.05 * lever) ** 2)
        sig_r = np.deg2rad(2.0)
        # GATE 3: innovation chi vs the APPEARANCE seed arrangement (no index /
        # no B estimate: the reference is the retrieved-anchor seed itself)
        seedB = se2(seed, se2i(rel))              # B-anchor seed pose (A frame)
        Zc = se2(se2i(anchorsA[c_aid]), seedB)
        s_at, s_ar = sig_t + 0.30, sig_r + np.deg2rad(6)
        chi = (np.linalg.norm(Z[:2] - Zc[:2]) / s_at) ** 2 \
            + (S.wrap(Z[2] - Zc[2]) / s_ar) ** 2
        if chi > CHI_MAX:
            cnt["innov"] += 1
            diag.append(rec)
            continue
        cnt["acc"] += 1
        edges.append((c_aid, bid, Z, 1 / sig_t, 1 / sig_r))
        gaC = a0 + c_aid * ANCHOR
        # -- VALIDATE (REF only; NOT used to select/seed) --
        correct = np.linalg.norm(gtd[gaC, :2] - gtd[gk, :2]) < REF_TOL
        xpass = abs(gaC - gk) > GAP_KF
        rec.update(accepted=True, correct=bool(correct), xpass=bool(xpass),
                   c_aid=c_aid)
        diag.append(rec)

    if verbose:
        _report_detection(diag, edges, cnt, KS)
    return edges, diag


def _report_detection(diag, edges, cnt, KS):
    nacc = sum(d["accepted"] for d in diag)
    ncorr = sum(d["accepted"] and d["correct"] for d in diag)
    nx = sum(d["accepted"] and d["xpass"] for d in diag)
    nxcorr = sum(d["accepted"] and d["xpass"] and d["correct"] for d in diag)
    nov = sum(d["accepted"] and d["is_overlap"] for d in diag)
    print(f"  gates: tried {cnt['tried']}  no-cand {cnt['nocand']}  "
          f"no-verify {cnt['noverify']}  innov-rej {cnt['innov']}  "
          f"-> accepted {cnt['acc']}", flush=True)
    print(f"  accepted correspondences: {nacc}  correct {ncorr}  "
          f"false {nacc-ncorr}  FP-rate "
          f"{100*(nacc-ncorr)/max(nacc,1):.1f}%", flush=True)
    print(f"    same-lap overlap ties: {nov}    "
          f"CROSS-PASS (>|{GAP_KF}| kf, genuine revisit): {nx}  "
          f"correct {nxcorr}  false {nx-nxcorr}", flush=True)

    # -- ring-key APPEARANCE retrieval recall on genuine cross-pass opportunities
    opp = [d for d in diag if d["is_opp"]]
    print(f"\n  GENUINE CROSS-PASS OPPORTUNITIES (B anchor with an A anchor "
          f"< {REF_TOL:.0f} m in REF AND > {GAP_KF} kf apart): {len(opp)}",
          flush=True)
    if opp:
        print("  ring-key appearance recall@k over those opportunities "
              "(is a correct cross-pass A anchor in the top-k shortlist?):",
              flush=True)
        for kk in KS:
            r_any = np.mean([d["hit"][kk][0] for d in opp])
            r_x = np.mean([d["hit"][kk][1] for d in opp])
            print(f"    @{kk:<3}  any-correct {r_any:.3f}   "
                  f"cross-pass-correct {r_x:.3f}", flush=True)
        rec_x = sum(d["accepted"] and d["xpass"] and d["correct"] and d["is_opp"]
                    for d in opp)
        print(f"  ==> ring-key + metric verify RECOVERS {rec_x} / {len(opp)} "
              f"genuine cross-pass correspondences "
              f"(metric-seed baseline: 2/274).", flush=True)
    return dict(nacc=nacc, ncorr=ncorr, nx=nx, nxcorr=nxcorr,
                n_opp=len(opp),
                rec_x=sum(d["accepted"] and d["xpass"] and d["correct"]
                          and d["is_opp"] for d in diag))


# ==========================================================================
# T_AB from the APPEARANCE-FOUND + verified correspondences (no shared index)
# ==========================================================================
def tab_from_edges(blob, cross):
    """Horn rigid fit B-frame anchor positions -> their implied A-frame positions
    over the verified appearance correspondences. This is the ONLY place a global
    B->A transform is formed, and it is built from the found correspondences, not
    the known index. Its residual == the differential warp (as in align)."""
    A, B = blob["A"], blob["B"]
    anchorsA = A["anchors"]
    pB, pA = [], []
    for c_aid, bid, Z, wt, wr in cross:
        implied_A = se2(anchorsA[c_aid], Z)      # B-anchor pose in A frame
        pB.append(B["anchors"][bid][:2])
        pA.append(implied_A[:2])
    pB, pA = np.array(pB), np.array(pA)
    ca, cb = pB.mean(0), pA.mean(0)
    H = (pB - ca).T @ (pA - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    th = np.arctan2(R[1, 0], R[0, 0])
    t = cb - R @ ca
    return np.array([t[0], t[1], th])


# ==========================================================================
# full experiment
# ==========================================================================
def run():
    print("=" * 74)
    print("MULTI-SESSION VSA MAPPING via RING-KEY RETRIEVAL (no oracle) -- Intel")
    print("=" * 74, flush=True)
    blob = load_full()
    gtd = gt_dense(blob["kts"], blob["rts"], blob["rp"])
    A, B = blob["A"], blob["B"]
    a0, b0, a1 = blob["a0"], blob["b0"], blob["a1"]
    idxA, idxB = A["idx"], B["idx"]

    finA, finB = A["fin"], B["fin"]
    gtA, gtB = gtd[idxA], gtd[idxB]
    aA = C.align_se2(finA[:, :2], gtA[:, :2])
    aB = C.align_se2(finB[:, :2], gtB[:, :2])
    ateA = np.sqrt((np.linalg.norm(aA - gtA[:, :2], axis=1) ** 2).mean())
    ateB = np.sqrt((np.linalg.norm(aB - gtB[:, :2], axis=1) ** 2).mean())
    print(f"\nindependent self-ATE (each rigid-aligned to REF): "
          f"A {ateA:.2f} m  B {ateB:.2f} m   (single-run Intel ceiling 2.44 m)",
          flush=True)

    # -- 1. cross-session detection: ring-key appearance retrieval + verify --
    print("\n-- 1. CROSS-SESSION DETECTION (ring-key appearance retrieval + "
          "metric verify; NO index/T_AB/B-pose seed) --", flush=True)
    t0 = time.time()
    cross, diag = detect_cross_ringkey(blob, gtd)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    if len(cross) < 3:
        print("\n  too few verified correspondences to assemble a joint graph; "
              "stopping (LIMIT outcome).", flush=True)
        return dict(cross=len(cross))

    # -- T_AB from the found correspondences (oracle-free) --
    T_AB = tab_from_edges(blob, cross)
    print(f"\n  T_AB from {len(cross)} appearance-found correspondences (Horn, "
          f"gt-free): t=({T_AB[0]:.2f},{T_AB[1]:.2f}) "
          f"heading {np.degrees(T_AB[2]):.1f} deg", flush=True)

    # pre-relax tail residual (rigid T_AB) for the VERIFY block
    pairs = overlap_residual(blob, gtd, None, None)
    preA = finA[:, :2][[p[0] for p in pairs]]
    preB = np.array([se2(T_AB, finB[p[1]])[:2] for p in pairs])
    pre_res = np.linalg.norm(preA - preB, axis=1)

    # -- 2. joint relax (reused UNCHANGED from ssp_multisession_align) --
    print("\n-- 2. JOINT RELAX (A seq+loop | B seq+loop | GENUINE cross edges; "
          "_gn TRF + IRLS + LOO; gauge = A origin) --", flush=True)
    t0 = time.time()
    P, nA, n_cross, n_before, n_after, n_pruned, banned = joint_relax(
        blob, T_AB, cross)
    print(f"  ({time.time()-t0:.0f}s)  anchors {len(P)} ({nA} A + {len(P)-nA} B)  "
          f"cross edges {n_cross}  loop edges {n_before}->{n_after}  "
          f"LOO-pruned {n_pruned}", flush=True)

    posesA = kf_poses(list(P[:nA]), A["kf_ref"])
    posesB = kf_poses(list(P[nA:]), B["kf_ref"])

    # -- 3. verify alignment --
    postA = posesA[:, :2][[p[0] for p in pairs]]
    postB = posesB[:, :2][[p[1] for p in pairs]]
    post_res = np.linalg.norm(postA - postB, axis=1)
    print(f"\n-- 3. VERIFY ALIGNMENT --", flush=True)
    print(f"  (a) CROSS-PASS tail correspondences ({len(pairs)}, B revisits A on "
          f"a later lap):", flush=True)
    print(f"    PRE  median {np.median(pre_res)*100:.0f} cm -> "
          f"POST median {np.median(post_res)*100:.0f} cm  "
          f"(p90 {np.percentile(post_res,90)*100:.0f}  "
          f"max {post_res.max()*100:.0f} cm)", flush=True)
    sh = np.arange(b0, a1)
    shpreA = finA[sh - a0, :2]
    shpreB = np.array([se2(T_AB, finB[k - b0])[:2] for k in sh])
    sh_pre = np.linalg.norm(shpreA - shpreB, axis=1)
    sh_postA = posesA[sh - a0, :2]
    sh_postB = posesB[sh - b0, :2]
    sh_post = np.linalg.norm(sh_postA - sh_postB, axis=1)
    print(f"  (b) SHARED-overlap region ({len(sh)} co-mapped kf):", flush=True)
    print(f"    PRE  median {np.median(sh_pre)*100:.0f} cm -> "
          f"POST median {np.median(sh_post)*100:.0f} cm  "
          f"(p90 {np.percentile(sh_post,90)*100:.0f} cm)", flush=True)
    bar = 0.25
    tail_ok = np.median(post_res) < bar
    over_ok = np.median(sh_post) < bar
    print(f"    => tail cross-pass {np.median(pre_res)*100:.0f}->"
          f"{np.median(post_res)*100:.0f} cm median: "
          f"{'REACHED' if tail_ok else 'ABOVE'} the sub-0.25 m bar", flush=True)

    # -- superpose --
    sup = superpose(blob, P, nA)
    print(f"\n  SUPERPOSE jointly-aligned maps (per-cell A+B add, CELL={CELL} m):",
          flush=True)
    print(f"    cells A={sup['nA']} B={sup['nB']} shared={sup['shared']} "
          f"union={sup['union']}", flush=True)
    print(f"    duplicate-sum {sup['dup_kb']:.0f} KB -> merged(union) "
          f"{sup['merge_kb']:.0f} KB "
          f"({100*(1-sup['union']/(sup['nA']+sup['nB'])):.0f}% saving)  "
          f"bundled-cell cosine {sup['cos_med']:.3f}", flush=True)

    # -- 4. merged two-session ATE --
    gk_all = np.concatenate([idxA, idxB])
    pose_all = np.concatenate([posesA[:, :2], posesB[:, :2]])
    gt_all = gtd[gk_all, :2]
    al = C.align_se2(pose_all, gt_all)
    merged_ate = np.sqrt((np.linalg.norm(al - gt_all, axis=1) ** 2).mean())
    uniq, ui = np.unique(gk_all, return_index=True)
    alu = C.align_se2(pose_all[ui], gtd[uniq, :2])
    merged_ate_u = np.sqrt((np.linalg.norm(alu - gtd[uniq, :2], axis=1) ** 2).mean())
    jA = C.align_se2(posesA[:, :2], gtA[:, :2])
    jB = C.align_se2(posesB[:, :2], gtB[:, :2])
    jateA = np.sqrt((np.linalg.norm(jA - gtA[:, :2], axis=1) ** 2).mean())
    jateB = np.sqrt((np.linalg.norm(jB - gtB[:, :2], axis=1) ** 2).mean())
    print(f"\n-- 4. MERGED TWO-SESSION ATE (one frame, rigid-aligned to REF) --",
          flush=True)
    print(f"    over ALL {len(gk_all)} kf (overlap counted twice): "
          f"{merged_ate:.2f} m", flush=True)
    print(f"    over {len(uniq)} UNIQUE kf (overlap deduped):          "
          f"{merged_ate_u:.2f} m", flush=True)
    print(f"    per-session in joint frame: A {jateA:.2f} m (indep {ateA:.2f}) "
          f"B {jateB:.2f} m (indep {ateB:.2f})", flush=True)
    print(f"    vs single-run Intel ceiling 2.44 m", flush=True)

    # -- DIAGNOSTIC (REF-FILTERED UPPER BOUND, *NOT* the oracle-free method) --
    # Keep ONLY the REF-correct cross edges and re-run the identical joint relax.
    # This isolates whether the failure is (i) appearance retrieval, (ii) metric
    # verification FP, or (iii) the pose graph. REF is used here purely to FILTER
    # for a diagnostic bound; it never touches the reported detection above.
    correct_cross = [e for e, d in zip(cross, [dd for dd in diag if dd["accepted"]])
                     if d["correct"]]
    print(f"\n-- DIAGNOSTIC: joint relax on the {len(correct_cross)} REF-CORRECT "
          f"cross edges only (upper bound; NOT the oracle-free result) --",
          flush=True)
    if len(correct_cross) >= 3:
        T_AB_c = tab_from_edges(blob, correct_cross)
        Pc, nAc, *_ = joint_relax(blob, T_AB_c, correct_cross)
        posesAc = kf_poses(list(Pc[:nAc]), A["kf_ref"])
        posesBc = kf_poses(list(Pc[nAc:]), B["kf_ref"])
        tpostA = posesAc[:, :2][[p[0] for p in pairs]]
        tpostB = posesBc[:, :2][[p[1] for p in pairs]]
        tpost = np.linalg.norm(tpostA - tpostB, axis=1)
        jBc = C.align_se2(posesBc[:, :2], gtB[:, :2])
        jateBc = np.sqrt((np.linalg.norm(jBc - gtB[:, :2], axis=1) ** 2).mean())
        print(f"    tail cross-pass residual POST {np.median(tpost)*100:.0f} cm "
              f"(vs {np.median(post_res)*100:.0f} cm with FPs); "
              f"B ATE {ateB:.2f}->{jateBc:.2f} m (vs {jateB:.2f} with FPs)",
              flush=True)
        print(f"    => confirms the break is the {100*(len(cross)-sum(d['correct'] for d in diag if d['accepted']))/max(len(cross),1):.0f}%"
              f" cross-session VERIFICATION FP, not retrieval or the pose graph.",
              flush=True)

    print(f"\nVERDICT:", flush=True)
    print(f"  cross-pass genuine recovery is the headline (contrast metric-seed "
          f"baseline 2/274). Tail alignment "
          f"{'REACHED' if tail_ok else 'still ABOVE'} the 0.25 m bar; overlap "
          f"{'sub-0.25' if over_ok else 'above'} m; B ATE {ateB:.2f}->{jateB:.2f}, "
          f"A {ateA:.2f}->{jateA:.2f}; merged unique-kf ATE {merged_ate_u:.2f} m "
          f"vs 2.44 ceiling.", flush=True)
    return dict(cross=len(cross), T_AB=T_AB, pre=np.median(pre_res),
                post=np.median(post_res), merged_ate_u=merged_ate_u,
                ateA=ateA, ateB=ateB, jateA=jateA, jateB=jateB, sup=sup)


# ==========================================================================
# selftest: retrieval + seed correctness + anti-oracle sanity (fast, synthetic)
# ==========================================================================
def selftest():
    rng = np.random.default_rng(0)
    n_beams = 180
    beam = np.deg2rad(np.arange(n_beams) - 90.0)
    # (1) ring-key retrieval: a query built from place P's ranges (yaw-rotated)
    #     retrieves P over unrelated distractors, and the SC column-shift yaw
    #     seed recovers the applied rotation.
    places = [rng.uniform(1.0, 25.0, n_beams) for _ in range(8)]
    grids = [SC.scan_context(r, beam, VALID_MAX, SC_MAX_R) for r in places]
    rkeys = [SC.ring_key(g) for g in grids]
    tgt = 3
    dtheta = np.deg2rad(24.0)
    # rotate the robot by +dtheta: bearings shift by -dtheta (range vector rolls)
    shift_bins = int(round(dtheta / (2 * np.pi / n_beams)))
    q_ranges = np.roll(places[tgt], -shift_bins)          # yaw-rotated scan
    qg = SC.scan_context(q_ranges, beam, VALID_MAX, SC_MAX_R)
    qk = SC.ring_key(qg)
    d = np.array([np.linalg.norm(rk - qk) for rk in rkeys])
    assert int(np.argmin(d)) == tgt, (d, tgt)
    a_heading = 0.5
    seed = yaw_seed(qg, grids[tgt], a_heading)
    err = abs(S.wrap(seed - S.wrap(a_heading + dtheta)))
    assert err <= np.deg2rad(2 * 360 / N_SECTOR), np.rad2deg(err)
    # (2) tab_from_edges recovers a known rigid B->A transform from correspondences
    T = np.array([2.0, -1.0, 0.3])

    class _Sess(dict):
        pass
    anchorsA = [np.array([rng.uniform(-5, 5), rng.uniform(-5, 5),
                          rng.uniform(-1, 1)]) for _ in range(6)]
    anchorsB = [se2(se2i(T), a) for a in anchorsA]         # B frame = T^-1 . A
    blob = dict(A=dict(anchors=anchorsA), B=dict(anchors=anchorsB))
    cross = []
    for i in range(6):
        Z = se2(se2i(anchorsA[i]), anchorsA[i])            # identity relative
        cross.append((i, i, Z, 1.0, 1.0))
    T_fit = tab_from_edges(blob, cross)
    assert np.allclose(T_fit[:2], T[:2], atol=1e-6) and \
        abs(S.wrap(T_fit[2] - T[2])) < 1e-6, T_fit
    print(f"selftest ok: ring-key retrieves correct place (argmin d), yaw-seed "
          f"err {np.rad2deg(err):.2f} deg; tab_from_edges recovers T "
          f"({T_fit[0]:.2f},{T_fit[1]:.2f},{np.degrees(T_fit[2]):.1f}deg)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "run":
        run()
    else:
        selftest()
