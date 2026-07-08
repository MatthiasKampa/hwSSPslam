"""Hierarchical multi-resolution pose-graph correction (coarse-to-fine).

Experiment: does a COARSE-first, then FINE pose-graph relaxation -- keyed to
the SSP scale structure (large spatial patches = low-frequency global drift,
small patches = high-frequency detail) -- beat the shipped FLAT anchor-graph
relax of ssp_bounded.BoundedSLAM?

Architecture (subclass; NO shipped module is edited):
  * COARSE level. Group the fine anchors (the 5-keyframe segment nodes) into
    contiguous super-nodes of `coarse_span` anchors each (~span*ANCHOR kf, a
    large spatial patch). Condense the fine graph onto the super-nodes: each
    fine edge (a,b,Z) whose endpoints fall in DIFFERENT super-nodes A,B becomes
    a coarse edge S_A^-1 S_B = off_a * Z * off_b^-1, where off_i is anchor i's
    FIXED offset in its super-node reference frame (each patch treated rigid).
    Relax this small coarse graph FIRST (IRLS on loop edges) -> a smooth global
    deformation.
  * DRAG. Turn the per-super-node SE(2) corrections into a smooth deformation
    FIELD: each fine anchor is displaced by an inverse-distance-weighted blend
    (in the se(2) tangent) of the nearest super-node corrections. Distance to
    super-node centers, so the drag interpolates BETWEEN patches (smooth, not
    blocky). This warm-starts the fine graph with the coarse global fit.
  * FINE level. Run the SHIPPED fine relax (`_relax_solve(None)`: analytic-TRF
    GN + Cauchy-IRLS + leave-one-out pruning) on the dragged anchors.

Sanity: with the coarse level disabled (`hier=False`, or a single super-node)
the correction reduces BIT-EXACTLY to the shipped flat relax.

Run:
  python3 ssp_hiergraph.py selftest
  python3 ssp_hiergraph.py log data/intel.log flat
  python3 ssp_hiergraph.py log data/intel.log hier 6
  python3 ssp_hiergraph.py sweep data/fr079.log 2 4 6 8 12
"""

import sys
import time

import numpy as np

import ssp_slam as S
import ssp_slam_loop as L
import ssp_bounded as B


# --------------------------------------------------------------------------
# SE(2) exp/log (twist <-> pose), consistent with L.se2_mul / L.se2_inv
# --------------------------------------------------------------------------
def se2_log(T):
    """SE(2) pose (x,y,theta) -> twist (vx,vy,w)."""
    th = S.wrap(T[2])
    if abs(th) < 1e-9:
        return np.array([T[0], T[1], th])
    half = 0.5 * th
    a = half / np.tan(half)                 # (th/2) cot(th/2)
    Vinv = np.array([[a, half], [-half, a]])
    v = Vinv @ T[:2]
    return np.array([v[0], v[1], th])


def se2_exp(xi):
    """twist (vx,vy,w) -> SE(2) pose (x,y,theta)."""
    vx, vy, th = xi
    if abs(th) < 1e-9:
        return np.array([vx, vy, S.wrap(th)])
    s, c = np.sin(th), np.cos(th)
    V = np.array([[s / th, -(1 - c) / th], [(1 - c) / th, s / th]])
    t = V @ np.array([vx, vy])
    return np.array([t[0], t[1], S.wrap(th)])


# --------------------------------------------------------------------------
# Hierarchical multi-resolution SLAM
# --------------------------------------------------------------------------
class HierGraphSLAM(B.BoundedSLAM):
    def __init__(self, *args, coarse_span=6, hier=True, drag_k=3,
                 coarse_irls=True, **kw):
        super().__init__(*args, **kw)
        self.coarse_span = coarse_span      # anchors per super-node
        self.hier = hier                    # False => shipped flat relax
        self.drag_k = drag_k                # nearest super-nodes for the drag
        self.coarse_irls = coarse_irls
        # instrumentation
        self.n_gn = 0                       # total _gn (least_squares) calls
        self.n_coarse_solve = 0             # coarse-level _gn calls
        self.relax_time = 0.0               # wall time inside relax()
        self.n_hier = 0                     # hierarchical relax invocations

    # count every nonlinear solve (fine + coarse share this)
    def _gn(self, *a, **k):
        self.n_gn += 1
        return super()._gn(*a, **k)

    # ---- coarse graph construction & solve --------------------------------
    def _supernodes(self):
        """Contiguous index spans -> super-node id per anchor. Returns
        (grp array [A], n_super, centers [Nc,2], S0 refs [Nc,3], offs list)."""
        A = len(self.anchors)
        P = np.array(self.anchors)
        span = self.coarse_span
        grp = np.arange(A) // span
        nc = int(grp.max()) + 1
        centers = np.zeros((nc, 2))
        S0 = np.zeros((nc, 3))
        offs = [None] * A
        for g in range(nc):
            idx = np.where(grp == g)[0]
            c = P[idx, :2].mean(0)
            th = np.arctan2(np.sin(P[idx, 2]).mean(), np.cos(P[idx, 2]).mean())
            ref = np.array([c[0], c[1], th])
            centers[g] = c
            S0[g] = ref
            iref = L.se2_inv(ref)
            for i in idx:
                offs[i] = L.se2_mul(iref, P[i])
        return grp, nc, centers, S0, offs

    def _coarse_solve(self, grp, nc, S0, offs):
        """Condense fine edges onto super-nodes and relax the coarse graph.
        Returns the coarse pose array Pc [nc,3]."""
        ceds = []
        for a, b, Z, wt, wr, kind in self.edges:
            A, Bg = int(grp[a]), int(grp[b])
            if A == Bg:
                continue                    # internal to a patch: fine handles it
            Zc = L.se2_mul(offs[a], L.se2_mul(Z, L.se2_inv(offs[b])))
            ceds.append((A, Bg, Zc, wt, wr, kind))
        if not ceds:
            return S0.copy()
        free = np.array(sorted(set(range(1, nc))), int)   # super-node 0 = gauge
        Pc0 = S0.copy()
        Pc, r = self._gn(ceds, {}, free, Pc0)
        self.n_coarse_solve += 1
        if self.coarse_irls and self.robust:
            wl = {}
            for e, (A, Bg, Zc, wt, wr, kind) in enumerate(ceds):
                if kind != "loop":
                    continue
                rn = np.linalg.norm(r[3 * e:3 * e + 3])
                wl[e] = 1.0 / np.sqrt(1.0 + (rn / 2.0) ** 2)   # Cauchy-ish
            Pc, r = self._gn(ceds, wl, free, Pc0)
            self.n_coarse_solve += 1
        return Pc

    def _drag(self, grp, nc, centers, S0, Pc):
        """Apply the coarse corrections to the fine anchors as a smooth,
        distance-weighted deformation field (blend nearest-K super-node
        corrections in the se(2) tangent). Warm-starts the fine relax."""
        # world-frame correction twist per super-node: delta = Pc * S0^-1
        tw = np.zeros((nc, 3))
        for g in range(nc):
            delta = L.se2_mul(Pc[g], L.se2_inv(S0[g]))
            tw[g] = se2_log(delta)
        A = len(self.anchors)
        P = np.array(self.anchors)
        K = min(self.drag_k, nc)
        for i in range(1, A):               # keep anchor 0 as the fixed gauge
            if i in self.retired:
                continue
            d2 = ((centers - P[i, :2]) ** 2).sum(1)
            if K < nc:
                sel = np.argpartition(d2, K - 1)[:K]
            else:
                sel = np.arange(nc)
            w = 1.0 / (d2[sel] + 1e-6)
            w /= w.sum()
            xi = (w[:, None] * tw[sel]).sum(0)
            self.anchors[i] = L.se2_mul(se2_exp(xi), P[i])

    # ---- override relax: coarse-to-fine ----------------------------------
    def relax(self):
        t0 = time.perf_counter()
        # Non-hierarchical path: BIT-EXACT shipped flat relax.
        if not self.hier:
            super().relax()
            self.relax_time += time.perf_counter() - t0
            return
        # windowed mode is out of scope for this experiment (full-solve only)
        self.dirty = False
        self.n_relax += 1
        self.pending_new = []
        if len(self.anchors) < 2 or not self.edges:
            self.relax_time += time.perf_counter() - t0
            return
        self.n_hier += 1
        grp, nc, centers, S0, offs = self._supernodes()
        if nc >= 2:
            Pc = self._coarse_solve(grp, nc, S0, offs)
            self._drag(grp, nc, centers, S0, Pc)   # smooth deformation field
        # FINE level: shipped relax (analytic-TRF GN + IRLS + LOO), warm-started
        self._relax_solve(None)
        self.n_full += 1
        self.relax_time += time.perf_counter() - t0


# --------------------------------------------------------------------------
# Self-tests
# --------------------------------------------------------------------------
def _bent_graph(slam, n_anchors=40, bend=0.9, loop_noise=0.0, seed=0):
    """Build a synthetic bent chain that revisits its start: seq edges along a
    curved path, plus loop edges tying late anchors back to early ones (a
    global warp the pose graph must absorb). Populates slam.anchors/edges
    directly (bypasses the frontend)."""
    rng = np.random.default_rng(seed)
    # ground-truth: a semicircle-ish arc that comes back near the origin
    th = np.linspace(0, 2 * np.pi, n_anchors)
    R = 6.0
    gt = np.stack([R * np.sin(th), R * (1 - np.cos(th)),
                   S.wrap(th)], 1)
    # drifted odometry: accumulate a small per-step heading bias (=> the graph
    # is "floppy"/warped and needs the loop closures to unbend)
    est = [gt[0].copy()]
    for i in range(1, n_anchors):
        rel = L.se2_mul(L.se2_inv(gt[i - 1]), gt[i])
        rel[2] += bend * 0.01 * i            # cumulative heading drift
        est.append(L.se2_mul(est[-1], rel))
    slam.anchors = [p.copy() for p in est]
    slam.edges = []
    slam.edge_seen = {}
    for i in range(1, n_anchors):            # seq edges = true relative (GT)
        Z = L.se2_mul(L.se2_inv(gt[i - 1]), gt[i])
        slam.edges.append((i - 1, i, Z, 1 / 0.03, 1 / np.deg2rad(0.3), "seq"))
    # loop edges: last few anchors revisit the first few (GT relative)
    for j in range(5):
        a, b = j, n_anchors - 5 + j
        Z = L.se2_mul(L.se2_inv(gt[a]), gt[b])
        if loop_noise:
            Z = Z + rng.normal(0, loop_noise, 3)
        slam.edges.append((a, b, Z, 1 / 0.05, 1 / np.deg2rad(0.5), "loop"))
        slam.edge_seen[(a, b)] = len(slam.edges) - 1
    return gt


def synth_compare(seeds=8):
    """Clean offline isolation: flat relax vs hierarchical relax on the SAME
    fixed floppy graph (no online-feedback confound). Answers directly: does
    coarse-first absorb a global warp better than the flat relax? Sweeps warp
    severity (bend) and loop-edge noise. Reports mean ATE over seeds."""
    print("== synth_compare (flat vs hier on identical fixed graphs) ==")
    print(f"{'bend':>5} {'lnoise':>7} {'span':>5} | {'flat ATE':>9} "
          f"{'hier ATE':>9} {'delta%':>7}")
    for bend in (0.6, 1.2, 2.0):
        for lnoise in (0.0, 0.05):
            # baseline flat
            fa = []
            for sd in range(seeds):
                s = B.BoundedSLAM(robust=True)
                gt = _bent_graph(s, bend=bend, loop_noise=lnoise, seed=sd)
                s.dirty = True
                s.relax()
                fa.append(_ate(s, gt))
            fa = np.mean(fa)
            for span in (4, 8):
                ha = []
                for sd in range(seeds):
                    s = HierGraphSLAM(robust=True, hier=True, coarse_span=span)
                    gt = _bent_graph(s, bend=bend, loop_noise=lnoise, seed=sd)
                    s.dirty = True
                    s.relax()
                    ha.append(_ate(s, gt))
                ha = np.mean(ha)
                d = 100 * (ha - fa) / fa
                print(f"{bend:5.1f} {lnoise:7.2f} {span:5d} | {fa:9.3f} "
                      f"{ha:9.3f} {d:+7.1f}")


def _ate(slam, gt):
    P = np.array(slam.anchors)
    al = _umeyama(P[:, :2], gt[:, :2])
    return float(np.sqrt(((al - gt[:, :2]) ** 2).sum(1).mean()))


def self_test():
    print("== self_test ==")
    # (1) SE(2) exp/log round-trip
    rng = np.random.default_rng(1)
    for _ in range(1000):
        T = np.array([rng.normal(0, 3), rng.normal(0, 3),
                      S.wrap(rng.normal(0, 2))])
        assert np.allclose(se2_exp(se2_log(T)), T, atol=1e-9)
    # exp(log(delta)) applied == delta composition
    for _ in range(1000):
        d = np.array([rng.normal(0, 2), rng.normal(0, 2), S.wrap(rng.normal(0, 1.5))])
        p = np.array([rng.normal(0, 5), rng.normal(0, 5), S.wrap(rng.normal(0, 3))])
        assert np.allclose(L.se2_mul(se2_exp(se2_log(d)), p),
                           L.se2_mul(d, p), atol=1e-9)
    print("  se2 exp/log round-trip OK")

    # (2) BIT-EXACT: hier=False reproduces the shipped BoundedSLAM.relax
    sflat = B.BoundedSLAM(robust=True)
    _bent_graph(sflat, loop_noise=0.02, seed=3)
    sflat.dirty = True
    Pflat = [a.copy() for a in sflat.anchors]        # snapshot pre-relax
    sflat.relax()
    flat_after = np.array(sflat.anchors)

    shf = HierGraphSLAM(robust=True, hier=False)
    shf.anchors = [p.copy() for p in Pflat]
    # rebuild identical edges
    _bent_graph(shf, loop_noise=0.02, seed=3)
    # _bent_graph reset anchors to the drifted est; re-snapshot must match sflat
    assert np.allclose(np.array(shf.anchors), Pflat, atol=0), "setup mismatch"
    shf.dirty = True
    shf.relax()
    assert np.array_equal(np.array(shf.anchors), flat_after), \
        "hier=False NOT bit-exact to shipped flat relax"
    print("  hier=False == shipped flat relax  (bit-exact)")

    # (3) single super-node (span huge) reduces to flat relax bit-exactly
    shg = HierGraphSLAM(robust=True, hier=True, coarse_span=10 ** 9)
    _bent_graph(shg, loop_noise=0.02, seed=3)
    shg.dirty = True
    shg.relax()
    assert np.array_equal(np.array(shg.anchors), flat_after), \
        "single super-node NOT bit-exact to flat relax"
    print("  single super-node == flat relax  (bit-exact)")

    # (4) hierarchical actually corrects the warp (sane, finite, low residual)
    def ate(slam, gt):
        P = np.array(slam.anchors)
        al = _umeyama(P[:, :2], gt[:, :2])
        return np.sqrt(((al - gt[:, :2]) ** 2).sum(1).mean())
    s2 = HierGraphSLAM(robust=True, hier=True, coarse_span=6)
    gt = _bent_graph(s2, loop_noise=0.0, seed=3)
    pre = ate(s2, gt)
    s2.dirty = True
    s2.relax()
    post = ate(s2, gt)
    print(f"  warp bent graph: ATE pre={pre:.3f} -> hier post={post:.3f} m "
          f"(coarse solves={s2.n_coarse_solve})")
    assert post < pre and post < 0.5, "hierarchical did not correct the warp"
    print("  hierarchical corrects the warp  OK")
    print("ALL SELF-TESTS PASSED")


def _umeyama(src, dst):
    """2D rigid alignment src->dst (for the self-test ATE only)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    H = s.T @ d
    U, _, Vt = np.linalg.svd(H)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:
        Vt[-1] *= -1
        Rm = Vt.T @ U.T
    t = mu_d - Rm @ mu_s
    return src @ Rm.T + t


# --------------------------------------------------------------------------
# Real-log harness (mirrors ssp_bounded_carmen.py exactly; zero new tuning)
# --------------------------------------------------------------------------
def run_log(path, hier, span=6, drag_k=3, verbose=True):
    import ssp_slam_carmen as C
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    slam = HierGraphSLAM(robust=True, attempt_every=4, relax_every=25,
                         gap_kf=300, recent_aids=12,
                         hier=hier, coarse_span=span, drag_k=drag_k)
    slam.store_dtype = np.complex64
    est = np.zeros((n, 3))
    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < 40.0, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = opose if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
    if slam.dirty:
        slam.relax()
    dt = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    n_loop = sum(1 for e in slam.edges if e[5] == "loop")
    if "mit" in path.split("/")[-1]:
        # MIT gfs timestamps are corrupt -> per-keyframe REF recovered by
        # range-array identity (ssp_scancontext.ref_positions, same convention
        # as the shipped MIT ATE). align over the inside-REF-span keyframes.
        import ssp_scancontext as SC
        _, ref_xy, valid = SC.ref_positions("mit", keys)
        al = C.align_se2(fin[valid, :2], ref_xy[valid])
        e = np.linalg.norm(al - ref_xy[valid], axis=1)
    else:
        ref = C.parse_flaser(path.replace(".log", ".gfs.log"))
        rts = np.array([t for _, _, t in ref])
        rxy = np.stack([p[:2] for _, p, _ in ref])
        j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
        good = np.abs(rts - kts[j]) < 0.3
        al = C.align_se2(fin[j[good], :2], rxy[good])
        e = np.linalg.norm(al - rxy[good], axis=1)
    res = dict(path=path, hier=hier, span=span, drag_k=drag_k, n_kf=n,
               n_anchor=len(slam.anchors), n_loop=n_loop, pruned=slam.n_pruned,
               n_relax=slam.n_relax, n_gn=slam.n_gn,
               n_coarse=slam.n_coarse_solve, relax_time=slam.relax_time,
               wall=dt, max_jerk=slam.max_jerk,
               rmse=float(np.sqrt((e ** 2).mean())),
               med=float(np.median(e)), max=float(e.max()))
    if verbose:
        tag = f"HIER span={span} k={drag_k}" if hier else "FLAT (shipped)"
        print(f"[{tag}] {path.split('/')[-1]}  kf={n} anch={res['n_anchor']} "
              f"loops={n_loop} pruned={slam.n_pruned}")
        print(f"    ATE rmse={res['rmse']:.3f}  med={res['med']:.3f}  "
              f"max={res['max']:.3f} m   max_jerk={res['max_jerk']:.3f} m")
        print(f"    relax: n={res['n_relax']} gn_calls={res['n_gn']} "
              f"coarse_solves={res['n_coarse']} relax_time={res['relax_time']:.2f}s "
              f"wall={dt:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        self_test()
    elif cmd == "log":
        path = sys.argv[2]
        hier = sys.argv[3] == "hier"
        span = int(sys.argv[4]) if len(sys.argv) > 4 else 6
        dk = int(sys.argv[5]) if len(sys.argv) > 5 else 3
        run_log(path, hier, span, dk)
    elif cmd == "sweep":
        path = sys.argv[2]
        spans = [int(x) for x in sys.argv[3:]] or [2, 4, 6, 8, 12]
        run_log(path, False)                 # baseline
        for sp in spans:
            run_log(path, True, sp)
