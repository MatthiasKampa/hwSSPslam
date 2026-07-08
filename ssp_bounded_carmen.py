"""Bounded-memory continual SSP SLAM on a CARMEN log (Intel Research Lab).

Same BoundedSLAM core as the synthetic bench: no sensor history, segments as
rigid anchor-frame vectors (+ derivative vectors for sub-grid rotation),
continual constraint attempts + relaxations from frame 0, IRLS + LOO pruning.

Usage: python3 ssp_bounded_carmen.py [data/intel.log] [n_keyframes]
"""

import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ssp_slam as S
import ssp_slam_carmen as C
import ssp_slam_loop as L
import ssp_bounded as B

VALID_MAX = 40.0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/intel.log"
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        keys = keys[:int(sys.argv[2])]
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    print(f"{len(scans)} scans -> {n} keyframes")

    gap = int(sys.argv[sys.argv.index("--gap") + 1]) if "--gap" in sys.argv else 300
    # Empirically best on Intel: small recency window + large gap. The 1-5 min
    # age band in between is matched by nothing (deliberate: exposing the
    # frontend to it measured worse, 7.10 vs 4.48; stitching it with loop
    # edges measured 4.89).
    slam = B.BoundedSLAM(robust=True, attempt_every=4, relax_every=25,
                         gap_kf=gap, recent_aids=12)
    # complex64 segment storage: verified free (Intel 2.440 m bit-identical to
    # complex128, half the map memory: 7.85 -> 3.93 MB; match-peak shift
    # 0.000 mm on real geometry). "c128" arg restores full precision.
    if "c128" not in sys.argv:
        slam.store_dtype = np.complex64
    est = np.zeros((n, 3))
    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < VALID_MAX, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = opose if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
        if k % 1000 == 0:
            print(f"  kf {k}/{n}  t={time.time() - t0:.0f}s  "
                  f"loops={sum(1 for e in slam.edges if e[5] == 'loop')}  "
                  f"mem={slam.memory_kb():.0f} KB", flush=True)
    if slam.dirty:
        slam.relax()
    dt = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    n_loop = sum(1 for e in slam.edges if e[5] == "loop")
    print(f"done: {dt:.0f}s ({dt / n * 1e3:.0f} ms/kf)  loop edges={n_loop}  "
          f"pruned={slam.n_pruned}  relax={slam.n_relax}  "
          f"map memory={slam.memory_kb():.0f} KB "
          f"({len(slam.segvec)} segment vectors)")
    np.savez(path.rsplit("/", 1)[-1].replace(".log", "") + "_traj_bounded.npz", est=fin, odom=odom, kts=kts)

    ref = C.parse_flaser(path.replace(".log", ".gfs.log"))
    rts = np.array([t for _, _, t in ref])
    rxy = np.stack([p[:2] for _, p, _ in ref])
    j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
    good = np.abs(rts - kts[j]) < 0.3
    al = C.align_se2(fin[j[good], :2], rxy[good])
    e = np.linalg.norm(al - rxy[good], axis=1)
    print(f"ATE vs corrected log: rmse {np.sqrt((e ** 2).mean()):.3f} m   "
          f"median {np.median(e):.3f} m   max {e.max():.3f} m")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    ax = axes[0]
    ax.plot(fin[:, 0], fin[:, 1], "r-", lw=0.5)
    xs = np.array([slam.anchors[a][:2] for a in slam.segvec])
    ax.scatter(xs[:, 0], xs[:, 1], s=6, c="k", marker="s", alpha=0.5)
    ax.set_title(f"bounded SLAM trajectory + {len(slam.segvec)} segment anchors")
    ax.set_aspect("equal")
    ax = axes[1]
    ax.plot(np.flatnonzero(good), e)
    ax.grid(alpha=0.3)
    ax.set_title("ATE vs RBPF-corrected reference [m]")
    fig.tight_layout()
    fig.savefig("ssp_intel_bounded.png", dpi=110)
    print("wrote ssp_intel_bounded.png")


if __name__ == "__main__":
    main()
