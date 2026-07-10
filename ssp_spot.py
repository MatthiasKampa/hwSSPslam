"""SPOT Telluride workshop dataset adapter — the first TARGET-PLATFORM data.

https://huggingface.co/datasets/lorinachey/spot-telluride-workshop-dataset
Ouster-class clouds: 65536 pts = 1024 azimuth x 64 rings @ 20 Hz (the
custom-head spec), PCD v0.7 ASCII with a ring field; SPOT kinematic
odometry at 528 Hz; LIO-SAM trajectory (496 poses) as a secondary
reference. Session: 78 s, 36.5 m path, ~7x7 m room, z flat.

Protocol (user, 2026-07-10): the pipeline runs LIDAR-ONLY — the guess is
constant-velocity chaining of the system's own estimates; SPOT odometry is
WITHHELD from the system and used as ground truth ("essentially spot on").
Keyframing is time-based (every Nth cloud — input-legal), eval = 'exact'
vs odometry xy at keyframe times.

2D slice: the ring with median |elevation| closest to 0 (of 64), binned to
1024 uniform azimuth beams; parsed once into data/spot_telluride/scans.npz.

Usage:
  python3 ssp_spot.py parse          # build the npz cache from parquet
  python3 ssp_spot.py sweep          # settings sweep (lidar-only, GT=odom)
"""
import sys
from pathlib import Path

import numpy as np

import ssp_slam as S
import ssp_slam_loop as L
import ssp_fpga as F

DIR = Path("data/spot_telluride")
CACHE = DIR / "scans.npz"
N_BEAM = 1024
STRIDE = 4                    # keyframe every Nth cloud (20 Hz -> 5 Hz)
R_MIN, R_MAX = 0.3, 60.0


def _yaw(qx, qy, qz, qw):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def parse():
    import pyarrow.parquet as pq
    shards = sorted((DIR / "pointclouds").glob("*.parquet"))
    ring_sel = None
    all_ts, all_rng = [], []
    for si, sh in enumerate(shards):
        t = pq.read_table(sh)
        for row in range(t.num_rows):
            ts = t["timestamp_ns"][row].as_py()
            blob = t["pcd_bytes"][row].as_py()
            head_end = blob.index(b"DATA ascii") + len(b"DATA ascii\n")
            lines = blob[head_end:].split(b"\n")
            if ring_sel is None:
                arr = np.genfromtxt([ln for ln in lines if ln], dtype=np.float32)
                elev = np.arctan2(arr[:, 2], np.hypot(arr[:, 0], arr[:, 1]))
                rings = arr[:, 4].astype(int)
                med = np.array([np.median(elev[rings == r])
                                for r in range(rings.max() + 1)])
                ring_sel = int(np.argmin(np.abs(med)))
                print(f"ring {ring_sel} selected "
                      f"(elev {np.degrees(med[ring_sel]):.2f} deg of "
                      f"{len(med)} rings)", flush=True)
                sel = arr[rings == ring_sel]
            else:
                lo = ring_sel * N_BEAM
                arr = np.genfromtxt(lines[lo:lo + N_BEAM], dtype=np.float32)
                sel = arr
            az = np.arctan2(sel[:, 1], sel[:, 0])
            rr = np.hypot(sel[:, 0], sel[:, 1])
            ok = (rr > R_MIN) & (rr < R_MAX)
            bins = ((az + np.pi) / (2 * np.pi) * N_BEAM).astype(int) % N_BEAM
            out = np.full(N_BEAM, np.inf, np.float32)
            np.minimum.at(out, bins[ok], rr[ok])
            all_ts.append(ts)
            all_rng.append(out)
        print(f"  shard {si + 1}/{len(shards)} done ({len(all_ts)} clouds)",
              flush=True)
    o = np.argsort(np.array(all_ts))
    np.savez_compressed(CACHE, ts=np.array(all_ts, np.int64)[o],
                        ranges=np.stack(all_rng)[o].astype(np.float32),
                        ring=ring_sel)
    print(f"wrote {CACHE} ({len(all_ts)} scans)", flush=True)


def make_bundle(stride=STRIDE):
    """ssp_datasets-shaped bundle. guess_mode='cv' => lidar-only runs;
    bundle['odom'] carries the WITHHELD GT (display/eval only)."""
    import pyarrow.parquet as pq
    z = np.load(CACHE)
    ts, ranges = z["ts"], z["ranges"]
    keys_idx = np.arange(0, len(ts), stride)
    o = pq.read_table(DIR / "odometry_imu" / "train.parquet")
    ots = np.array(o["timestamp_ns"])
    oxy = np.stack([np.array(o["x"]), np.array(o["y"])], 1)
    oyaw = _yaw(*(np.array(o[k]) for k in ("qx", "qy", "qz", "qw")))
    # the parquet is NOT time-sorted (a ~62 s block sits out of order and
    # teleported 3 keyframes' GT by 3.6 m) — sort before any lookup
    srt = np.argsort(ots, kind="stable")
    ots, oxy, oyaw = ots[srt], oxy[srt], oyaw[srt]
    j = np.clip(np.searchsorted(ots, ts[keys_idx]), 0, len(ots) - 1)
    gt = np.stack([oxy[j, 0], oxy[j, 1], oyaw[j]], 1)
    # belt-and-suspenders GT hygiene: mask physically impossible reference
    # steps (> 0.5 m between ~0.2 s keyframes) from eval + display
    step = np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1)
    gt_ok = np.ones(len(gt), bool)
    bad = np.flatnonzero(step > 0.5)
    for i in bad:
        gt_ok[max(0, i):i + 2] = False
    if not gt_ok.all():
        print(f"  spot GT hygiene: masked {int((~gt_ok).sum())} keyframes "
              f"(impossible reference steps)", flush=True)
    beam = -np.pi + np.arange(N_BEAM) * (2 * np.pi / N_BEAM)
    keys = [(ranges[i], gt[k].copy(), ts[i] / 1e9)
            for k, i in enumerate(keys_idx)]
    return dict(name="spot", kind="spot", eval="exact", path=str(CACHE),
                keys=keys, beam=beam, odom=gt.copy(),
                kts=ts[keys_idx] / 1e9, rmin=R_MIN, rmax=R_MAX, gt=gt,
                gt_ok=gt_ok, guess_mode="cv")


def run_cv(cls=F.BandSLAM, sample="point", stride=STRIDE, use_odom=False,
           t_half=0.48, rot_half=9.0, **kw):
    """Lidar-only run: CV guess from own estimates (use_odom=True is the
    LABELED DIAGNOSTIC arm that consumes the withheld odometry)."""
    import ssp_datasets as DS
    b = make_bundle(stride)
    keys, beam = b["keys"], b["beam"]
    n = len(keys)
    slam = cls(robust=True, attempt_every=4, relax_every=25, gap_kf=300,
               recent_aids=12, **kw)
    slam.store_dtype = np.complex64
    slam.matcher = S.Matcher(L.ENC_MAIN, t_half=t_half, rot_half_deg=rot_half,
                             rot_step_deg=1.5, perm=(4, L.N_ANG))
    est = np.zeros((n, 3))
    for k, (r, gtp, ts) in enumerate(keys):
        rr = DS.clean(b, r)
        if callable(sample):
            pts, w = sample(rr, beam)
        elif sample == "point":
            pts, w = F.points_from_scan(rr, beam)
        else:
            pts, w, _ = S.scan_to_samples(rr, beam)
        if use_odom:
            guess = gtp if k == 0 else L.se2_mul(
                est[k - 1], L.se2_mul(L.se2_inv(keys[k - 1][1]), gtp))
        elif k == 0:
            guess = keys[0][1].copy()       # world frame anchor only
        elif k == 1:
            guess = est[0].copy()
        else:
            v = est[k - 1] - est[k - 2]     # constant-velocity extrapolation
            vn = np.hypot(v[0], v[1])
            if vn > 0.30:
                v[:2] *= 0.30 / vn
            guess = np.array([est[k - 1][0] + v[0], est[k - 1][1] + v[1],
                              est[k - 1][2] + S.wrap(v[2])])
        est[k] = slam.add_keyframe(pts, w, guess)
    if slam.dirty:
        slam.relax()
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    sc = DS.evaluate(b, fin)
    nloop = sum(1 for e in slam.edges if e[5] == "loop")
    return dict(ate=sc["ate"], med=sc["med"], loops=nloop, n=n,
                mem_kb=slam.memory_kb(), fin=fin, est=est, slam=slam,
                bundle=b)


def sweep():
    import ssp_sampling as SP
    z = np.load(CACHE)
    print(f"{len(z['ts'])} scans cached, ring {z['ring']}", flush=True)
    b2 = lambda rr, bm: SP.sample_interp(rr, bm, 2, 63.4)   # noqa: E731
    arms = (
        ("point  oct  s4 t.48", dict(sample="point")),
        ("bridge2 oct s4 t.48", dict(sample=b2)),
        ("seg    oct  s4 t.48", dict(sample="seg")),
        ("bridge2 oct s4 t.72r15", dict(sample=b2, t_half=0.72, rot_half=15)),
        ("bridge2 oct s2 t.48", dict(sample=b2, stride=2)),
        ("bridge2 oct s8 t.48", dict(sample=b2, stride=8)),
        ("bridge2 2bit-store  ", dict(sample=b2, nph=4, nmag=1,
                                      ring_scales=True)),
        ("bridge2 WITH-ODOM (diagnostic)", dict(sample=b2, use_odom=True)),
    )
    for tag, kw in arms:
        kw.setdefault("spec", None)
        kw.setdefault("nph", 0)
        r = run_cv(**kw)
        print(f"  spot {tag}: ATE {r['ate']:6.3f}  med {r['med']:6.3f}  "
              f"loops {r['loops']}  ({r['n']} kf, {r['mem_kb']:.0f} KB)",
              flush=True)


def sweep_lattice():
    import ssp_lattice
    import ssp_sampling as SP
    b2 = lambda rr, bm: SP.sample_interp(rr, bm, 2, 63.4)   # noqa: E731
    for tag, lams, nang in (
            ("oct60 (shipped)", (0.25, 0.5, 1.0, 2.0), 60),
            ("fine60 {.125-1}", (0.125, 0.25, 0.5, 1.0), 60),
            ("oct90          ", (0.25, 0.5, 1.0, 2.0), 90),
            ("oct36          ", (0.25, 0.5, 1.0, 2.0), 36)):
        ssp_lattice.set_polar(lams, nang)
        r = run_cv(sample=b2, spec=None, nph=0)
        print(f"  spot lattice {tag}: ATE {r['ate']:6.3f}  "
              f"med {r['med']:6.3f}  loops {r['loops']}", flush=True)
    ssp_lattice.shipped()


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "parse"
    dict(parse=parse, sweep=sweep, lattice=sweep_lattice)[what]()
