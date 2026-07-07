"""Run SSP SLAM on a real CARMEN-format 2D lidar log (Intel Research Lab).

Usage: python3 ssp_slam_carmen.py [data/intel.log] [n_keyframes]

Differences from the synthetic pipeline:
- 180-beam, 180 deg FOV SICK scans; readings >= 40 m treated as no-return.
- Motion prior comes from the log's odometry (delta composed onto the last
  estimate) instead of a constant-velocity model.
- Keyframing: a scan is processed when odometry moved > 0.10 m or > 5 deg.
- Reference: the RBPF-corrected intel.gfs.log poses (matched by timestamp,
  rigidly aligned) give an approximate trajectory error; odometry dead
  reckoning is plotted for contrast.
"""

import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ssp_slam as S

VALID_MAX = 40.0
KEY_TRANS, KEY_ROT = 0.10, np.deg2rad(5.0)
BEAMS = None  # set from log


def parse_flaser(path):
    """Yield (ranges, laser_pose, timestamp) per FLASER line."""
    out = []
    with open(path) as f:
        for line in f:
            if not line.startswith("FLASER"):
                continue
            p = line.split()
            n = int(p[1])
            r = np.array(p[2:2 + n], float)
            x, y, th = (float(v) for v in p[2 + n:5 + n])
            ts = float(p[-1])  # logger-relative time, shared base with .gfs.log
            out.append((r, np.array([x, y, th]), ts))
    return out


def keyframes(scans):
    keys, last = [], None
    for r, pose, ts in scans:
        if last is None or np.linalg.norm(pose[:2] - last[:2]) > KEY_TRANS \
                or abs(S.wrap(pose[2] - last[2])) > KEY_ROT:
            keys.append((r, pose, ts))
            last = pose
    return keys


def compose_guess(est_prev, odom_prev, odom_now):
    """est_prev (+) (odom_prev^-1 (+) odom_now) in SE(2)."""
    dt_local = S._rot(-odom_prev[2]) @ (odom_now[:2] - odom_prev[:2])
    dth = S.wrap(odom_now[2] - odom_prev[2])
    g = np.empty(3)
    g[:2] = est_prev[:2] + S._rot(est_prev[2]) @ dt_local
    g[2] = est_prev[2] + dth
    return g


def align_se2(a, b):
    """Rigid 2D alignment b ~ R a + t (Horn); returns transformed a."""
    ca, cb = a.mean(0), b.mean(0)
    H = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(H)
    Rm = Vt.T @ np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return (a - ca) @ Rm.T + cb


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/intel.log"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9

    scans = parse_flaser(path)
    keys = keyframes(scans)[:limit]
    n_beams = len(keys[0][0])
    beam = np.deg2rad(np.arange(n_beams) - 90.0)  # SICK: 1 deg spacing, -90..+89
    print(f"{len(scans)} scans -> {len(keys)} keyframes, {n_beams} beams")

    N_ANG, SUBMAP_R = 60, 8.0
    GATE_T, GATE_R = 0.45, np.deg2rad(11.0)  # innovation gate: fall back to odometry
    lams = 0.25 * 2.0 ** np.arange(4)
    a = np.arange(N_ANG) * np.pi / N_ANG
    W = np.concatenate([(2 * np.pi / lam) * np.stack([np.cos(a), np.sin(a)], 1)
                        for lam in lams])
    enc = S.SSPEncoder.__new__(S.SSPEncoder)
    enc.W = W
    matcher = S.Matcher(enc, t_half=0.48, rot_half_deg=12.0, rot_step_deg=1.5)
    print(f"encoder: oct4 x {N_ANG} angles, D={W.shape[0]}   "
          f"submap radius {SUBMAP_R} m, odometry gate {GATE_T} m / {np.degrees(GATE_R):.1f} deg")

    # per-keyframe world-frame encodings: the map is re-bundled per query from
    # keyframes near the guess (spatial gating = cheap submaps, less ghosting)
    E = np.zeros((len(keys), W.shape[0]), complex)
    est = np.zeros((len(keys), 3))
    odom = np.stack([k[1] for k in keys])
    cloud, cloud_f = [], []
    n_gated = 0

    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < VALID_MAX, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        if k == 0:
            est[0] = opose
        else:
            guess = compose_guess(est[k - 1], odom[k - 1], odom[k])
            est[k] = guess
            near = np.linalg.norm(est[:k, :2] - guess[:2], axis=1) < SUBMAP_R
            near[:max(0, k - 150)] = False  # sliding local map: last 150 keyframes only
            if len(pts) >= 20 and near.any():
                M = E[:k][near].sum(0)
                cand = matcher.match(M, pts, w, guess)
                cand = matcher.match(M, pts, w, cand)  # second pass, re-centered
                cand[2] = S.wrap(cand[2])
                if (np.linalg.norm(cand[:2] - guess[:2]) < GATE_T
                        and abs(S.wrap(cand[2] - guess[2])) < GATE_R):
                    est[k] = cand
                else:
                    n_gated += 1
        wp = pts @ S._rot(est[k, 2]).T + est[k, :2]
        E[k] = enc.shift(est[k, :2]) * enc.encode(pts @ S._rot(est[k, 2]).T, w)
        cloud.append(wp[::2])
        cloud_f.append(np.full(len(wp[::2]), k))
        if k % 500 == 0:
            print(f"  frame {k}/{len(keys)}  t={time.time() - t0:.0f}s")
    dt = time.time() - t0
    print(f"done: {dt:.0f}s ({dt / len(keys) * 1e3:.0f} ms/keyframe)   "
          f"odometry fallbacks: {n_gated} ({100 * n_gated / len(keys):.1f}%)")

    np.savez("intel_traj.npz", est=est, odom=odom,
             kts=np.array([t for _, _, t in keys]))

    # reference comparison (corrected log, matched by timestamp)
    ref_err = None
    try:
        ref = parse_flaser(path.replace(".log", ".gfs.log"))
        rts = np.array([t for _, _, t in ref])
        rxy = np.stack([p[:2] for _, p, _ in ref])
        kts = np.array([t for _, _, t in keys])
        j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
        good = np.abs(rts - kts[j]) < 0.3  # keyframes ~0.3-0.5 s apart; adds a small floor to ATE
        aligned = align_se2(est[j[good], :2], rxy[good])
        ref_err = np.linalg.norm(aligned - rxy[good], axis=1)
        print(f"ATE vs corrected log over {good.sum()} matched poses: "
              f"rmse {np.sqrt((ref_err ** 2).mean()):.3f} m   "
              f"median {np.median(ref_err):.3f} m   max {ref_err.max():.3f} m")
    except FileNotFoundError:
        print("no corrected log found for reference")

    cl, cf = np.concatenate(cloud), np.concatenate(cloud_f)
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    ax = axes[0]
    ax.plot(odom[:, 0], odom[:, 1], "0.6", lw=0.8)
    ax.set_title("odometry dead reckoning (drift)")
    ax.set_aspect("equal")
    ax = axes[1]
    sc = ax.scatter(cl[:, 0], cl[:, 1], c=cf, s=0.3, cmap="viridis")
    ax.plot(est[:, 0], est[:, 1], "r-", lw=0.6, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="keyframe")
    ax.set_title("SSP SLAM map + trajectory")
    ax.set_aspect("equal")
    ax = axes[2]
    if ref_err is not None:
        ax.plot(np.flatnonzero(good), ref_err)
        ax.set_xlabel("matched reference pose #")
        ax.set_ylabel("position error vs corrected log [m]")
        ax.grid(alpha=0.3)
        ax.set_title("ATE vs RBPF-corrected reference")
    fig.suptitle(f"SSP SLAM on {path} — oct4 x {N_ANG} (D={W.shape[0]})")
    fig.tight_layout()
    fig.savefig("ssp_slam_intel.png", dpi=110)
    print("wrote ssp_slam_intel.png")


if __name__ == "__main__":
    main()
