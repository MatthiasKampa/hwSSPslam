"""Hybrid loop-closure optimization: shipped detect+warp+relax ANCHORS +
continual overlap-flow REFINEMENT.

The flow study showed continual overlap-flow (a) can only REFINE when already
in-basin (it can't do the long-range association detect+snap does) and (b) LOCKS
confidently-overlapping regions (within-1m anchors 8->40). The shipped BoundedSLAM
provides exactly the missing piece: detected+verified loop closures that get the
map in-basin (the topology). So the hybrid runs the shipped system (detect + gate
+ IRLS/LOO relax) and THEN adds overlap-flow refinement on top, with the detected
loop closures AND the sequential skeleton entered as stiff pose-pose constraints
in one FlowField energy:

    E = lam_edge*||seq + detected-loop edges||^2  -  lam_ov*sum rho(cos<P_i,P_j>)

Init at the shipped-relaxed poses (so lam_ov=0 is a fixed point = shipped), then
lam_ov>0 densely refines the deformation the sparse loop edges don't capture,
anchored so the noise-dominated overlap force can't drift to twins.

Reuses ssp_flow.FlowField (grad-L2-normalized force, FD-exact) and ate_vs_gfs.
No shipped/prior module edited.

Usage: python3 -m experiments.hybrid {selftest | single LOG | sweep LOG}
"""
import sys
import time

import numpy as np

import sspslam.encoder as S
import sspslam.frontend as C
import sspslam.lattice as L
import sspslam.bounded as B
import experiments.flow as F

VALID_MAX = 40.0


def run_shipped(path, limit=None, verbose=True):
    """Shipped BoundedSLAM with closure ENABLED. Returns (slam, kts, fin)."""
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)
    if limit:
        keys = keys[:limit]
    n = len(keys)
    nb = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(nb) * (180.0 / nb))
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    slam = B.BoundedSLAM(robust=True, attempt_every=4, relax_every=25,
                         gap_kf=300, recent_aids=12)
    slam.store_dtype = np.complex64
    est = np.zeros((n, 3))
    t0 = time.time()
    for k in range(n):
        rr = np.where(keys[k][0] < VALID_MAX, keys[k][0], np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = odom[0] if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
        if verbose and k % 1000 == 0:
            print(f"    build {k}/{n} t={time.time()-t0:.0f}s "
                  f"loops={sum(1 for e in slam.edges if e[5]=='loop')}", flush=True)
    if slam.dirty:
        slam.relax()
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    return slam, kts, fin


def field_from_shipped(slam, lam_ov, **kw):
    """Build a FlowField from a shipped slam's final state, with edges = seq +
    DETECTED LOOP closures (both stiff). Anchor poses initialise at the
    shipped-relaxed poses."""
    aids = sorted(slam.segvec)
    remap = {a: i for i, a in enumerate(aids)}
    pos = np.array([slam.anchors[a][:2] for a in aids])
    th = np.array([slam.anchors[a][2] for a in aids])
    segvec = np.array([slam.segvec[a].astype(complex) for a in aids])
    segder = np.array([slam.segder[a].astype(complex) for a in aids])
    # ALL pose-pose edges: sequential skeleton + detected loop closures
    edges = [(remap[a], remap[b], Z, wt, wr)
             for a, b, Z, wt, wr, kind in slam.edges
             if a in remap and b in remap]
    free = np.ones(len(aids), bool)
    free[0] = False                                 # gauge
    sess = np.zeros(len(aids), int)
    order = np.arange(len(aids))
    ff = F.FlowField(pos, th, segvec, segder, edges, free, sess, order,
                     lam_seq=1.0, lam_ov=lam_ov, **kw)
    return ff, aids, remap


def kf_poses_from_field(slam, ff, remap):
    """Reconstruct per-keyframe poses from flowed anchor poses (segments/kf ride
    their anchor rigidly)."""
    P = ff.poses()
    n = len(slam.kf_ref)
    out = np.zeros((n, 3))
    for k in range(n):
        aid, rel = slam.kf_ref[k]
        if aid in remap:
            out[k] = L.se2_mul(P[remap[aid]], rel)
        else:                                       # evicted anchor: shipped pose
            out[k] = slam.pose_of(k)
    return out


def ate(path, kts, fin):
    return F.ate_vs_gfs(path, kts, fin)


def run_single(path, lam_ov=0.3, iters=400, lr=0.4, gate_r=5.0, verbose=True):
    slam, kts, fin_ship = run_shipped(path, verbose=verbose)
    a_ship = ate(path, kts, fin_ship)
    ff, aids, remap = field_from_shipped(slam, lam_ov=lam_ov, gate_r=gate_r)
    ff.build_pairs()
    for _ in range(iters):
        ff.step(lr=lr)
    fin_hyb = kf_poses_from_field(slam, ff, remap)
    a_hyb = ate(path, kts, fin_hyb)
    name = path.rsplit("/", 1)[-1]
    print(f"  {name}: SHIPPED ATE {a_ship['rmse']:.3f} m (med {a_ship['median']:.3f}, "
          f"max {a_ship['max']:.2f})  loops={sum(1 for e in slam.edges if e[5]=='loop')}")
    print(f"  {name}: HYBRID (lam_ov={lam_ov}, {iters} it) ATE {a_hyb['rmse']:.3f} m "
          f"(med {a_hyb['median']:.3f}, max {a_hyb['max']:.2f})  "
          f"delta {a_hyb['rmse']-a_ship['rmse']:+.3f}", flush=True)
    return a_ship, a_hyb, slam


def run_multi(n_list=(0, 8, 30), lam_ov_list=(0.0, 2.0), init="coarse",
              verbose=True):
    """Multi-session hybrid: sparse cross-session ANCHOR edges (perfect closure
    RELATIVE-POSE at co-visible places -- a fenced diagnostic isolating the
    overlap-flow's refinement value from detection error) + continual overlap
    flow. Decomposes anchors-only (lam_ov=0 = pure pose-graph) vs anchors+overlap
    (hybrid) across anchor counts. A fixed (gauge), B flows.

    NB: GT is used ONLY to (a) pick which anchor pairs are genuinely co-visible
    and (b) supply the closure relative-pose Z and score -- never in the flow
    force. This is an UPPER-BOUND diagnostic (perfect detection+measurement), not
    a deployable system: it answers 'does overlap-flow converge B GIVEN correct
    sparse anchors?'."""
    import experiments.multisession as M
    from scipy.spatial import cKDTree

    blob = F._load_sessions()
    gtd = M.gt_dense(blob["kts"], blob["rts"], blob["rp"])
    A, Bs = blob["A"], blob["B"]
    aidsA = sorted(A["segvec"]); aidsB = sorted(Bs["segvec"])
    idxA, idxB = A["idx"], Bs["idx"]
    finA, finB = A["fin"], Bs["fin"]
    AN = F.ANCHOR
    nA, nB = len(aidsA), len(aidsB)

    # ---- coarse, GT-free, correspondence-free init for B (same as test_multi)
    b0 = blob["b0"]
    kA = np.searchsorted(idxA, b0); kB = 0
    T0 = M.se2(finA[kA], M.se2i(finB[kB]))
    if init == "coarse":
        pert = np.array([1.5, -1.0, np.deg2rad(6.0)])
        T_init = M.se2(T0, pert)
    else:
        T_init = T0

    posA = np.array([A["anchors"][a][:2] for a in aidsA])
    thA = np.array([A["anchors"][a][2] for a in aidsA])
    posB0 = np.array([M.se2(T_init, Bs["anchors"][a])[:2] for a in aidsB])
    thB0 = np.array([M.se2(T_init, Bs["anchors"][a])[2] for a in aidsB])
    pos = np.vstack([posA, posB0]); th = np.concatenate([thA, thB0])
    segvec = np.array([A["segvec"][a] for a in aidsA]
                      + [Bs["segvec"][a] for a in aidsB], complex)
    segder = np.array([A["segder"][a] for a in aidsA]
                      + [Bs["segder"][a] for a in aidsB], complex)
    free = np.concatenate([np.zeros(nA, bool), np.ones(nB, bool)])
    sess = np.concatenate([np.zeros(nA, int), np.ones(nB, int)])
    order = np.concatenate([np.arange(nA), np.arange(nB)])
    remapB = {a: nA + i for i, a in enumerate(aidsB)}
    seqB = [(remapB[a], remapB[b], Z, wt, wr)
            for a, b, Z, wt, wr, kind in F._session_seq(Bs)
            if a in remapB and b in remapB]

    # ---- co-visible anchor pairs (GT: which places overlap + closure Z) ----
    gtA = np.array([gtd[idxA[a * AN]] for a in aidsA])        # (nA,3) full pose
    gtB = np.array([gtd[idxB[a * AN]] for a in aidsB])        # (nB,3)
    treeA = cKDTree(gtA[:, :2])
    dco, jco = treeA.query(gtB[:, :2])
    corr = np.flatnonzero(dco < 0.5)                          # B locals co-vis
    jAmatch = jco[corr]                                       # matching A locals

    def anchor_edges(k):
        if k <= 0 or len(corr) == 0:
            return []
        sel = np.unique(np.linspace(0, len(corr) - 1, k).round().astype(int))
        edges = []
        for m in sel:
            bi, aj = corr[m], jAmatch[m]
            Z = L.se2_mul(L.se2_inv(gtA[aj]), gtB[bi])        # GT closure rel-pose
            edges.append((aj, nA + bi, Z, 1 / 0.10, 1 / np.deg2rad(1.0)))
        return edges

    # scoring: co-observed residual (B flowed vs A) + B-only ATE vs GT
    def score(ff):
        Bx = np.stack([ff.x[nA:], ff.y[nA:]], 1)
        Ax = np.stack([ff.x[:nA], ff.y[:nA]], 1)
        d = np.linalg.norm(Bx[corr] - Ax[jAmatch], axis=1)
        both = np.vstack([Ax, Bx]); gt_both = np.vstack([gtA[:, :2], gtB[:, :2]])
        al = C.align_se2(both, gt_both)
        e = np.linalg.norm(al - gt_both, axis=1)
        return float(np.median(d)), float(np.sqrt((e[nA:] ** 2).mean()))

    # coarse+mid rings only (skip the all-rings FINE stage: it is the memory hog
    # -- 88k cross-session pairs x 360 dims thrashes; MID's 240 dims answer
    # 'does overlap refine?' without cm registration). gate 6m keeps the pair set
    # tractable while still spanning the 6m init offset.
    stages = [(F.RINGS_COARSE, 6.0, 10 ** 9, 0.25, 0.48, 300),
              (F.RINGS_MID, 6.0, 10 ** 9, 0.28, 0.50, 500)]

    print(f"  multi-session: A={nA} B={nB} anchors, {len(corr)} GT co-visible, "
          f"init={init}", flush=True)
    ff0 = F.FlowField(pos, th, segvec, segder, seqB, free, sess, order,
                      lam_seq=1.0, lam_ov=0.0, gate_r=6.0, sep=10 ** 9)
    ff0.build_pairs()
    r0 = score(ff0)
    print(f"  init co-observed residual {r0[0]:.2f} m  B-ATE {r0[1]:.2f} m",
          flush=True)
    print(f"  {'n_anch':>7} {'lam_ov':>7} | {'coobs_med':>9} {'B_ATE':>7}",
          flush=True)
    results = {}
    for k in n_list:
        aedges = anchor_edges(k)
        for lov in lam_ov_list:
            edges = seqB + aedges
            ff = F.FlowField(pos.copy(), th.copy(), segvec, segder, edges,
                             free, sess, order, lam_seq=1.0, lam_ov=lov,
                             gate_r=6.0, sep=10 ** 9)
            F.run_flow(ff, stages, lr=0.4, mom=0.85, verbose=False)
            med, bate = score(ff)
            results[(k, lov)] = (med, bate)
            print(f"  {k:7d} {lov:7.1f} | {med:9.2f} {bate:7.2f}", flush=True)
    return results, r0


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        # lam_ov=0 from the shipped-relaxed init must be ~a fixed point (no drift)
        path = sys.argv[2] if len(sys.argv) > 2 else "data/fr101.log"
        slam, kts, fin_ship = run_shipped(path, limit=1893, verbose=False)
        a_ship = ate(path, kts, fin_ship)
        ff, aids, remap = field_from_shipped(slam, lam_ov=0.0, gate_r=5.0)
        ff.build_pairs()
        p0 = ff.poses().copy()
        for _ in range(100):
            ff.step(lr=0.4)
        drift = np.linalg.norm(ff.poses()[:, :2] - p0[:, :2], axis=1).max()
        print(f"selftest: lam_ov=0 max anchor drift over 100 steps = {drift:.4f} m "
              f"({'PASS' if drift < 0.05 else 'CHECK'})  shipped ATE {a_ship['rmse']:.3f}")
    elif cmd == "single":
        run_single(sys.argv[2])
    elif cmd == "sweep":
        path = sys.argv[2]
        for lam in [0.0, 0.1, 0.3, 0.6, 1.0]:
            run_single(path, lam_ov=lam, verbose=False)
    elif cmd == "multi":
        init = sys.argv[2] if len(sys.argv) > 2 else "coarse"
        run_multi(init=init)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
