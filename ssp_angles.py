"""Derivative vector vs angular resolution at equal storage (matched band).

Settles the open question from RESULTS.md "Derivative-vector novelty ablation":
for the loop-closure matched band (4 rings, lam 0.25/0.5/1/2), is a fixed
per-segment storage budget better spent on the d/dtheta DERIVATIVE vector
(sub-grid rotation correction) or on MORE ANGLES (finer angular lattice)?

Storage is counted in complex components per segment on the matched band:
  60+der    60 angles, segvec(240) + segder(240) = 480/seg   (SHIPPED)
  120-noder 120 angles, segvec(480), no derivative = 480/seg  (equal storage)
  60-noder  60 angles, segvec(240), no derivative = 240/seg   (half: cheap baseline)
  120+der   120 angles, segvec(480) + segder(480) = 960/seg   (optional check)

60+der and 120-noder are EQUAL STORAGE but differ in COMPUTE: the matcher works
on D = 4*n_ang complex features (240 vs 480), so 120 angles doubles the matcher
cost, while the derivative only adds a cheap per-query vector add.

Mechanism: this module REBUILDS the ssp_slam_loop lattice (W, N_ANG, N_RING,
MAIN, ENC, ENC_MAIN, and thereby rot_permute/encode which read those globals)
in place for a chosen n_ang on the 4-ring matched band, then runs the UNMODIFIED
ssp_bounded.BoundedSLAM through a thin subclass that (a) gates the derivative via
the existing use_der switch and (b) actually stores NO segder when it is off, so
the reported memory is honest. The 4-ring matched-band lattice is
decision-identical to the shipped 6-ring lattice for BoundedSLAM (the two coarse
rings feed only coh[4:6]/diagnostics, never a decision), so 60+der reproduces the
shipped Intel 2.440 m.

Does NOT edit ssp_bounded/ssp_slam/ssp_slam_loop (all reused by import).

Usage:
  python3 ssp_angles.py selftest
  python3 ssp_angles.py intel [data/intel.log]      # all 4 configs
  python3 ssp_angles.py bench                        # room/sparse/corridor
  python3 ssp_angles.py fr079 [data/fr079.log]       # top-2 configs
"""

import sys
import time

import numpy as np

import ssp_slam as S
import ssp_slam_carmen as C
import ssp_slam_loop as L
import ssp_bounded as B
from worlds import WORLDS

VALID_MAX = 40.0
MATCHED_RINGS = (0.25, 0.5, 1.0, 2.0)   # the 4-ring matched band


# ---------------------------------------------------------------------------
# Parametrized lattice: rebuild ssp_slam_loop's module globals in place.
# rot_permute/encode read N_RING/N_ANG/W from L's namespace at call time, so
# reassigning them generalizes both to any n_ang with zero edits to L.
# ---------------------------------------------------------------------------

def build_lattice(n_ang, rings=MATCHED_RINGS):
    lams = np.asarray(rings, float)
    a = np.arange(n_ang) * np.pi / n_ang
    u = np.stack([np.cos(a), np.sin(a)], 1)
    W = np.concatenate([(2 * np.pi / lam) * u for lam in lams])
    return W, len(lams)


def install(n_ang, rings=MATCHED_RINGS):
    """Rebuild L's lattice globals for n_ang on the given (matched-band) rings.
    All rings are the matched band here, so MAIN spans the whole lattice."""
    W, n_ring = build_lattice(n_ang, rings)
    L.LAMS = np.asarray(rings, float)
    L.N_ANG = n_ang
    L.N_RING = n_ring
    L.W = W
    L.MAIN = slice(0, n_ring * n_ang)
    L.ENC = S.SSPEncoder.__new__(S.SSPEncoder)
    L.ENC.W = W
    L.ENC_MAIN = S.SSPEncoder.__new__(S.SSPEncoder)
    L.ENC_MAIN.W = W[L.MAIN]
    return W, n_ring


# ---------------------------------------------------------------------------
# Derivative on/off with an HONEST memory number: a discard-store for segder.
# ---------------------------------------------------------------------------

class _NoStore(dict):
    """Segder replacement when the derivative is off: stores nothing, reads 0,
    reports absent. `v += x` becomes read-0 -> compute -> discard-write."""
    def __contains__(self, k):
        return False

    def __getitem__(self, k):
        return 0

    def __setitem__(self, k, v):
        pass

    def pop(self, k, d=None):
        return d


class AngleSLAM(B.BoundedSLAM):
    """BoundedSLAM with the derivative vector gated AND its storage skipped when
    off (world_vec_seg already short-circuits on use_der, so segder is never
    read when off). Everything else — matcher, coherence, backend — inherited
    unmodified. Construct AFTER install(n_ang): the matcher bases are built from
    L.ENC_MAIN in __init__."""

    def __init__(self, use_der=True, **kw):
        super().__init__(**kw)
        self.use_der = use_der
        if not use_der:
            self.segder = _NoStore()   # genuine zero-storage for the honest number

    def memory_kb(self):
        bytes_per = 8 if self.store_dtype == np.complex64 else 16
        nvec = 2 if self.use_der else 1
        return len(self.segvec) * L.W.shape[0] * bytes_per * nvec / 1024


CONFIGS = {              # tag -> (n_ang, use_der, components/seg on matched band)
    "60+der":    (60, True, 480),
    "120-noder": (120, False, 480),
    "60-noder":  (60, False, 240),
    "120+der":   (120, True, 960),
}


# ---------------------------------------------------------------------------
# Selftest: rot_permute and world_vec_seg exact at each n_ang.
# ---------------------------------------------------------------------------

def selftest():
    rng = np.random.default_rng(3)
    for n_ang in (60, 120):
        install(n_ang)
        pts = rng.uniform(-4, 4, (300, 2))
        w = np.full(300, 0.1)
        # (1) rot_permute is exact: encode(R(m*pi/n_ang) pts) == permute(encode)
        v = L.encode(pts, w)
        max_err = 0.0
        for m in (1, 7, 33, n_ang + 5, 2 * n_ang - 3):
            th = m * np.pi / n_ang
            exact = L.encode(pts @ S._rot(th).T, w)
            err = np.max(np.abs(exact - L.rot_permute(v, m)))
            max_err = max(max_err, err)
        assert max_err < 1e-9, (n_ang, max_err)
        # (2) world_vec_seg permutation identity: at an on-lattice anchor heading
        #     (delta == 0) it must equal the exact world re-encode of the segment.
        slam = AngleSLAM(use_der=True)
        m = 9 % n_ang
        pose = np.array([1.3, -0.7, m * np.pi / n_ang])   # heading on the grid
        slam.anchors = [pose.copy()]
        PA = pts @ S._rot(0.0).T                          # anchor-frame == rel 0
        slam.segvec = {0: (np.exp(1j * (PA @ L.W.T)).T @ w).astype(complex)}
        got = slam.world_vec_seg(0)
        world_pts = pts @ S._rot(pose[2]).T + pose[:2]
        exact = np.exp(1j * (world_pts @ L.W.T)).T @ w
        wv_err = np.max(np.abs(got - exact))
        assert wv_err < 1e-9, (n_ang, wv_err)
        # (3) derivative branch is a genuine 1st-order correction: at a small
        #     sub-grid delta the +der reconstruction beats permutation-only.
        delta = 0.6 * np.pi / n_ang
        pose2 = np.array([0.4, 0.2, m * np.pi / n_ang + delta])
        slam.anchors = [pose2.copy()]
        A = np.exp(1j * (PA @ L.W.T))
        Cx = PA[:, 0:1] * L.W[:, 1] - PA[:, 1:2] * L.W[:, 0]
        slam.segder = {0: ((1j * Cx * A).T @ w).astype(complex)}
        wp2 = pts @ S._rot(pose2[2]).T + pose2[:2]
        exact2 = np.exp(1j * (wp2 @ L.W.T)).T @ w
        slam.use_der = False
        e_perm = np.linalg.norm(slam.world_vec_seg(0) - exact2) / np.linalg.norm(exact2)
        slam.use_der = True
        e_der = np.linalg.norm(slam.world_vec_seg(0) - exact2) / np.linalg.norm(exact2)
        assert e_der < e_perm, (n_ang, e_perm, e_der)
        print(f"  n_ang={n_ang:3d}  D={L.W.shape[0]:3d}  rot_permute<{max_err:.1e}  "
              f"world_vec_seg<{wv_err:.1e}  recon perm {e_perm:.4f} -> +der {e_der:.4f}")
    print("selftest ok: rot_permute + world_vec_seg exact (1e-9) at n_ang=60,120")


# ---------------------------------------------------------------------------
# CARMEN log harness (Intel / fr079), mirrors ssp_bounded_carmen (deterministic).
# ---------------------------------------------------------------------------

def run_carmen(path, tag, keys, odom, kts, ref, verbose=False):
    n_ang, use_der, comp = CONFIGS[tag]
    install(n_ang)
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    slam = AngleSLAM(use_der=use_der, robust=True, attempt_every=4,
                     relax_every=25, gap_kf=300, recent_aids=12)
    slam.store_dtype = np.complex64
    est = np.zeros((n, 3))
    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < VALID_MAX, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = opose if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
        if verbose and k % 2000 == 0:
            print(f"    [{tag}] kf {k}/{n} t={time.time()-t0:.0f}s "
                  f"loops={sum(1 for e in slam.edges if e[5]=='loop')}", flush=True)
    if slam.dirty:
        slam.relax()
    dt = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    n_loop = sum(1 for e in slam.edges if e[5] == "loop")

    rts = np.array([t for _, _, t in ref])
    rxy = np.stack([p[:2] for _, p, _ in ref])
    j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
    good = np.abs(rts - kts[j]) < 0.3
    al = C.align_se2(fin[j[good], :2], rxy[good])
    e = np.linalg.norm(al - rxy[good], axis=1)
    return dict(tag=tag, comp=comp, D=L.W.shape[0], ate=np.sqrt((e ** 2).mean()),
                med=np.median(e), mx=e.max(), mem=slam.memory_kb(),
                segs=len(slam.segvec), loops=n_loop, secs=dt, mskf=dt / n * 1e3,
                ngrid=n_ang)


def carmen_suite(path, tags, label):
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    ref = C.parse_flaser(path.replace(".log", ".gfs.log"))
    print(f"== {label}: {len(scans)} scans -> {len(keys)} keyframes")
    rows = []
    for tag in tags:
        r = run_carmen(path, tag, keys, odom, kts, ref, verbose=True)
        rows.append(r)
        print(f"  {tag:<10} ATE {r['ate']:.3f} m (med {r['med']:.3f}, max {r['mx']:.2f})  "
              f"D={r['D']:<3d} mem {r['mem']:.0f} KB ({r['segs']} seg, {r['comp']}/seg)  "
              f"loops {r['loops']}  {r['mskf']:.1f} ms/kf  {r['secs']:.0f}s", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Synthetic bench: mirrors ssp_bounded.run with AngleSLAM + installed lattice.
# ---------------------------------------------------------------------------

def run_bench(world, tag, n=750, seed=1, laps=3):
    n_ang, use_der, comp = CONFIGS[tag]
    install(n_ang)
    S.RNG = np.random.default_rng(seed)
    segs = WORLDS[world]()
    gt = B.multiloop_traj(n, laps=laps)
    odo = B.sim_odometry(gt)
    slam = AngleSLAM(use_der=use_der, robust=True)
    slam.diag_gt = gt
    est = np.zeros((n, 3))
    t0 = time.time()
    for k in range(n):
        r = B.scan_at(segs, gt[k])
        pts, w, _ = S.scan_to_samples(r, B.BEAM)
        guess = odo[0] if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odo[k - 1]), odo[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    ate = np.linalg.norm(fin[:, :2] - gt[:, :2], axis=1)
    n_loop = sum(1 for e in slam.edges if e[5] == "loop")
    false = 0
    for a, b, Z, wt, wr, kind in slam.edges:
        if kind != "loop":
            continue
        Zt = L.se2_mul(L.se2_inv(gt[a * B.ANCHOR]), gt[b * B.ANCHOR])
        if np.linalg.norm(Z[:2] - Zt[:2]) > 0.30 or abs(S.wrap(Z[2] - Zt[2])) > np.deg2rad(3):
            false += 1
    return dict(ate=np.sqrt((ate ** 2).mean()), edges=n_loop, false=false,
                mem=slam.memory_kb(), secs=time.time() - t0)


def bench_suite(tags, worlds=("room", "sparse", "corridor"), seeds=(1, 2, 3)):
    print(f"== bench (paired seeds {seeds}, laps=3)")
    for world in worlds:
        print(f"-- {world}")
        base_ate = None
        for tag in tags:
            rs = [run_bench(world, tag, seed=s) for s in seeds]
            ates = np.array([r["ate"] for r in rs])
            if base_ate is None:
                base_ate = ates
            d = ates - base_ate
            n_ang, use_der, comp = CONFIGS[tag]
            print(f"   {tag:<10} ATE {100*ates.mean():6.1f} cm  "
                  f"(paired vs 60+der {100*d.mean():+6.1f}, {np.sum(d<0)}/{len(d)} better)  "
                  f"D={4*n_ang:<3d} false {np.mean([r['false'] for r in rs]):4.1f}  "
                  f"edges {np.mean([r['edges'] for r in rs]):4.1f}  "
                  f"mem {np.mean([r['mem'] for r in rs]):5.0f} KB  "
                  f"{np.mean([r['secs'] for r in rs]):4.1f}s", flush=True)


# ---------------------------------------------------------------------------

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        selftest()
    elif cmd == "intel":
        path = sys.argv[2] if len(sys.argv) > 2 else "data/intel.log"
        selftest()
        carmen_suite(path, ["60+der", "120-noder", "60-noder", "120+der"], "Intel")
    elif cmd == "bench":
        bench_suite(["60+der", "120-noder", "60-noder", "120+der"])
    elif cmd == "fr079":
        path = sys.argv[2] if len(sys.argv) > 2 else "data/fr079.log"
        carmen_suite(path, ["60+der", "120-noder"], "fr079")
    else:
        raise SystemExit(f"unknown command {cmd!r} (selftest | intel | bench | fr079)")


if __name__ == "__main__":
    main()
