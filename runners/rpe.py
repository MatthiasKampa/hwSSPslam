"""Relative pose error at fixed metric baselines (TUM-style) for Intel runs.

Usage: python3 -m runners.rpe [intel_traj.npz] [baselines...]
Compares SSP SLAM and raw odometry against the RBPF-corrected reference,
reporting median AND mean at each baseline (the median-only single-baseline
number quoted early in RESULTS.md was interval-dependent; this is the fix).
"""

import sys

import numpy as np

import sspslam.encoder as S
import sspslam.frontend as C


def rpe(traj, rp, jj, spans):
    dist = np.concatenate([[0], np.cumsum(
        np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=1))])
    out = {}
    for span in spans:
        terr, aerr = [], []
        for i in range(len(rp)):
            k = int(np.searchsorted(dist, dist[i] + span))
            if k >= len(rp):
                break
            dref = S._rot(-rp[i, 2]) @ (rp[k, :2] - rp[i, :2])
            a, b = traj[jj[i]], traj[jj[k]]
            dst = S._rot(-a[2]) @ (b[:2] - a[:2])
            terr.append(np.linalg.norm(dst - dref) / span)
            aerr.append(abs(S.wrap((b[2] - a[2]) - (rp[k, 2] - rp[i, 2]))) / span)
        out[span] = (100 * np.median(terr), 100 * np.mean(terr),
                     np.degrees(np.median(aerr)), np.degrees(np.mean(aerr)))
    return out


def main():
    npz = sys.argv[1] if len(sys.argv) > 1 else "intel_traj.npz"
    spans = [float(a) for a in sys.argv[2:]] or [1.0, 5.0, 10.0]
    d = np.load(npz)
    est, odom, kts = d["est"], d["odom"], d["kts"]
    ref = C.parse_flaser("data/intel.gfs.log")
    rts = np.array([t for _, _, t in ref])
    rp = np.stack([p for _, p, _ in ref])
    j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
    good = np.abs(rts - kts[j]) < 0.3
    rp, jj = rp[good], j[good]
    for name, traj in (("SSP SLAM", est), ("odometry", odom)):
        r = rpe(traj, rp, jj, spans)
        for span, (tm, ta, am, aa) in r.items():
            print(f"{name:<9} @{span:4.0f} m: trans {tm:5.1f} med / {ta:5.1f} mean cm/m"
                  f"   head {am:5.2f} med / {aa:5.2f} mean deg/m")
        al = C.align_se2(traj[jj, :2], rp[:, :2])
        e = np.linalg.norm(al - rp[:, :2], axis=1)
        print(f"{name:<9} ATE rmse {np.sqrt((e ** 2).mean()):.2f} m")


if __name__ == "__main__":
    main()
