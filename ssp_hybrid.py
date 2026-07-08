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

Usage: python3 ssp_hybrid.py {selftest | single LOG | sweep LOG}
"""
import sys
import time

import numpy as np

import ssp_slam as S
import ssp_slam_carmen as C
import ssp_slam_loop as L
import ssp_bounded as B
import ssp_flow as F

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
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
