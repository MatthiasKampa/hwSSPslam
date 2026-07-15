"""SPOT Telluride SCHOOL runs adapter (dataset drop 2: school/run1, run2).

https://huggingface.co/datasets/lorinachey/spot-telluride-workshop-dataset
Same 1024x64 Ouster-class head as the classroom run (runners/spot.py), new
shard format: pointclouds rows carry `npz_bytes` (xyz/intensity/ring arrays)
instead of ASCII `pcd_bytes`. This adapter DISTILLS the ~8.6 GB parquet drop
into two compact caches per run, which are all the pipeline and webvis need:

    data/spot_telluride/<run>/scans.npz    ring-sliced 1024-beam range scans
    data/spot_telluride/<run>/ref_lio.npz  gated LIO-SAM reference at kf times

DATA QUALITY (measured 2026-07-13, undocumented upstream): the school
`odometry_imu` stream is a single ~1 kHz source that oscillates +-20..300 m
for extended intervals in BOTH runs — kinematic odometry is UNUSABLE as a
reference here (the classroom drop's was fine). LIO-SAM is the only
reference: run2's is healthy for t < ~80 s (z flat +-0.1 m, ~5 Hz), then it
and the robot localization break simultaneously (z +-19 m, then frozen);
run1's is ~99% z=-1000 sentinels (23 usable kf -> run1 = loop/stability
stress set, no absolute eval). Gating: z-sentinel removal, |z| < 1 m
flat-floor window, 0.4 s windowed velocity <= 2.5 m/s with 1 s dilation.

Protocol stays LIDAR-ONLY (guess_mode='cv'; nothing external enters the
pipeline). The bundle is anchored in the LIO frame (first valid reference
pose seeds the world frame) so eval='exact' scores without alignment;
references are scoring/display-only (anti-oracle).

Usage:
  python3 -m runners.spot_school parse school_run1   # parquet -> scans.npz
  python3 -m runners.spot_school refs  school_run2   # build + report ref_lio
Run via the registry: python3 -m runners.datasets run school_run2
"""
import sys
from pathlib import Path

import numpy as np

import runners.spot as SP

BASE = Path(__file__).resolve().parents[1] / "data" / "spot_telluride"
STRIDE = 4                     # keyframe every Nth cloud (20 Hz -> 5 Hz)
REF_TOL_MS = 300               # kf must have a gated LIO pose this close


def parse(run):
    """pointclouds parquet (npz_bytes rows) -> scans.npz; mirrors
    runners.spot.parse (flattest-ring selection, min-range azimuth bins)."""
    import io
    import pyarrow.parquet as pq
    d = BASE / run
    shards = sorted((d / "pointclouds").glob("*.parquet"))
    ring_sel = None
    all_ts, all_rng = [], []
    for si, sh in enumerate(shards):
        t = pq.read_table(sh)
        for row in range(t.num_rows):
            ts = t["timestamp_ns"][row].as_py()
            z = np.load(io.BytesIO(t["npz_bytes"][row].as_py()))
            xyz, rings = z["xyz"], z["ring"].astype(int)
            if ring_sel is None:
                elev = np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))
                med = np.array([np.median(elev[rings == r])
                                for r in range(rings.max() + 1)])
                ring_sel = int(np.argmin(np.abs(med)))
                print(f"ring {ring_sel} selected "
                      f"(elev {np.degrees(med[ring_sel]):.2f} deg of "
                      f"{len(med)} rings)", flush=True)
            sel = xyz[rings == ring_sel]
            az = np.arctan2(sel[:, 1], sel[:, 0])
            rr = np.hypot(sel[:, 0], sel[:, 1])
            ok = (rr > SP.R_MIN) & (rr < SP.R_MAX)
            bins = ((az + np.pi) / (2 * np.pi)
                    * SP.N_BEAM).astype(int) % SP.N_BEAM
            out = np.full(SP.N_BEAM, np.inf, np.float32)
            np.minimum.at(out, bins[ok], rr[ok])
            all_ts.append(ts)
            all_rng.append(out)
        print(f"  shard {si + 1}/{len(shards)} done ({len(all_ts)} clouds)",
              flush=True)
    o = np.argsort(np.array(all_ts))
    np.savez_compressed(d / "scans.npz", ts=np.array(all_ts, np.int64)[o],
                        ranges=np.stack(all_rng)[o].astype(np.float32),
                        ring=ring_sel)
    print(f"wrote {d / 'scans.npz'} ({len(all_ts)} scans)", flush=True)


def _liosam_gated(run, win_s=0.4, vmax=2.5, dilate_s=1.0, zmax=1.0):
    """-> (ts_ns, x, y, yaw) of the gated LIO-SAM stream (see module doc)."""
    import pyarrow.parquet as pq
    t = pq.read_table(BASE / run / "odometry_lio_sam" / "train.parquet")
    ts = np.array(t["timestamp_ns"], dtype=np.int64)
    xyz = np.stack([np.array(t["x"]), np.array(t["y"]),
                    np.array(t["z"])], 1)
    yaw = SP._yaw(*(np.array(t[k]) for k in ("qx", "qy", "qz", "qw")))
    srt = np.argsort(ts, kind="stable")
    ts, xyz, yaw = ts[srt], xyz[srt], yaw[srt]
    sane = (np.abs(xyz[:, 2] + 1000.0) > 1.0) & (np.abs(xyz[:, 2]) < zmax)
    ts, xy, yaw = ts[sane], xyz[sane, :2], yaw[sane]
    j = np.clip(np.searchsorted(ts, ts + int(win_s * 1e9)), 0, len(ts) - 1)
    v = np.hypot(*(xy[j] - xy).T) / np.maximum((ts[j] - ts) / 1e9, 1e-3)
    ok = np.ones(len(ts), bool)
    tbad = ts[v > vmax]
    if len(tbad):
        a = np.searchsorted(ts, tbad - int(dilate_s * 1e9))
        b = np.searchsorted(ts, tbad + int((win_s + dilate_s) * 1e9))
        for i, k in zip(a, b):
            ok[i:k + 1] = False
    return ts[ok], xy[ok], yaw[ok]


def build_ref(run, stride=STRIDE):
    """Gated LIO-SAM at keyframe times -> ref_lio.npz {gt (N,3), ok (N)}."""
    z = np.load(BASE / run / "scans.npz")
    kts = z["ts"][np.arange(0, len(z["ts"]), stride)]
    lts, lxy, lyaw = _liosam_gated(run)
    gt = np.zeros((len(kts), 3))
    ok = np.zeros(len(kts), bool)
    if len(lts):
        j = np.clip(np.searchsorted(lts, kts), 1, len(lts) - 1)
        j = j - (np.abs(lts[j - 1] - kts) < np.abs(lts[j] - kts))
        ok = np.abs(lts[j] - kts) < REF_TOL_MS * 1e6
        gt = np.stack([lxy[j, 0], lxy[j, 1], lyaw[j]], 1)
        if ok.any():
            # rows under the mask (eval never reads them) hold the LAST valid
            # reference pose — the webvis draws gt as the odometry/reference
            # trace, and a constant placeholder made it teleport back and
            # forth through coverage gaps and park at the start pose after
            # t~80 s (user-visible glitch, 2026-07-14). Leading gap
            # back-fills from the first valid pose (also the world-frame
            # seed, so pipeline output is unchanged).
            idx = np.where(ok, np.arange(len(ok)), -1)
            np.maximum.accumulate(idx, out=idx)
            idx[idx < 0] = int(np.argmax(ok))
            gt = gt[idx]
    np.savez_compressed(BASE / run / "ref_lio.npz", gt=gt, ok=ok)
    print(f"wrote {BASE / run / 'ref_lio.npz'}: {int(ok.sum())}/{len(ok)} "
          f"kf covered", flush=True)
    return gt, ok


def make_bundle(run, stride=STRIDE):
    """runners.datasets-shaped bundle; LIO-frame anchored, lidar-only CV."""
    d = BASE / run
    if not (d / "scans.npz").exists():
        parse(run)
    z = np.load(d / "scans.npz")
    ts, ranges = z["ts"], z["ranges"]
    keys_idx = np.arange(0, len(ts), stride)
    if (d / "ref_lio.npz").exists():
        rz = np.load(d / "ref_lio.npz")
        gt, gt_ok = rz["gt"], rz["ok"]
    else:
        gt, gt_ok = build_ref(run, stride)
    beam = -np.pi + np.arange(SP.N_BEAM) * (2 * np.pi / SP.N_BEAM)
    keys = [(ranges[i], gt[k].copy(), ts[i] / 1e9)
            for k, i in enumerate(keys_idx)]
    return dict(name=run, kind="spot_school", eval="exact",
                path=str(d / "scans.npz"), keys=keys, beam=beam,
                odom=gt.copy(), kts=ts[keys_idx] / 1e9,
                rmin=SP.R_MIN, rmax=SP.R_MAX, gt=gt, gt_ok=gt_ok,
                guess_mode="cv",
                gt_name="LIO-SAM reference (gated flat window; kinematic "
                        "odometry broken in this drop)")


if __name__ == "__main__":
    cmd, run = sys.argv[1], sys.argv[2]
    if cmd == "parse":
        parse(run)
    elif cmd == "refs":
        gt, ok = build_ref(run)
        if ok.any():
            path = float(np.hypot(*np.diff(gt[ok, :2], axis=0).T).sum())
            print(f"{run}: ref covers {int(ok.sum())}/{len(ok)} kf, "
                  f"path {path:.1f} m, extent "
                  f"x[{gt[ok, 0].min():.0f},{gt[ok, 0].max():.0f}] "
                  f"y[{gt[ok, 1].min():.0f},{gt[ok, 1].max():.0f}]")
