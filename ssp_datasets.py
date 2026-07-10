"""Dataset registry + the ONE shared run/eval harness.

Before this module the suite carried three copies of the run loop and three
eval recipes (ssp_fpga.run_log for CARMEN+gfs, ssp_stata.run/evaluate for the
bag, scratch_belgioioso/scratch_fpga_mit2 for the range-identity logs). This
consolidates them behind one registry; `run()` reproduces `ssp_fpga.run_log`
BIT-EXACTLY on the CARMEN logs (asserted in selftest) and the banked shipped
numbers on belg (2.644) / stata (0.202).

Eval families (scoring only — anti-oracle rules apply, PROTOCOL 2):
  gfs    each reference pose -> nearest keyframe timestamp within 0.3 s,
         align_se2, RMSE (the standard recipe).
  ident  belg/mit: the gfs timestamps are corrupt, so reference poses are
         matched by RANGE-ARRAY IDENTITY to raw scans, then interpolated to
         keyframe times (convention from ssp_scancontext).
  stata  floorplan-anchored GT poses (independent-class reference),
         nearest keyframe within 30 ms.

Usage:
  python3 ssp_datasets.py selftest              # bit-exact vs run_log
  python3 ssp_datasets.py run belg              # shipped config on one name
  python3 ssp_datasets.py run stata lean        # named config
"""
import sys
import time
from pathlib import Path

import numpy as np

import ssp_slam as S
import ssp_slam_carmen as C
import ssp_slam_loop as L
import ssp_fpga as F

VALID_MAX = 40.0

DATASETS = {
    "intel": dict(kind="carmen", path="data/intel.log", eval="gfs"),
    "fr079": dict(kind="carmen", path="data/fr079.log", eval="gfs"),
    "fr101": dict(kind="carmen", path="data/fr101.log", eval="gfs"),
    "aces": dict(kind="carmen", path="data/aces_publicb.log", eval="gfs"),
    "fhw": dict(kind="carmen", path="data/fhw.log", eval="gfs"),
    "belg": dict(kind="carmen", path="data/belgioioso.log", eval="ident"),
    "mit": dict(kind="carmen", eval="ident",
                path="data/MIT_Infinite_Corridor_2002_09_11_same_floor.log"),
    "stata": dict(kind="stata", path="data/stata/2012-01-27-07-37-01.bag",
                  eval="stata"),
    # SPOT Telluride (target platform): lidar-only protocol — guess_mode
    # 'cv' withholds the 528-Hz kinematic odometry from the system and
    # uses it as GT ('exact' eval). See ssp_spot.py.
    "spot": dict(kind="spot", path="data/spot_telluride/scans.npz",
                 eval="exact"),
}


def load(name, cap=None):
    """-> bundle dict: keys [(ranges, odom_pose, ts)], beam angles, odom
    stack, keyframe timestamps, range-validity params."""
    d = DATASETS[name]
    if d["kind"] == "spot":
        import ssp_spot
        b = ssp_spot.make_bundle()
        if cap:
            for key in ("keys", "kts", "odom", "gt"):
                b[key] = b[key][:cap]
        return b
    if d["kind"] == "carmen":
        keys = C.keyframes(C.parse_flaser(d["path"]))
        if cap:
            keys = keys[:cap]
        nb = len(keys[0][0])
        beam = np.deg2rad(-90.0 + np.arange(nb) * (180.0 / nb))
        rmin, rmax = None, VALID_MAX
    else:                                        # stata bag (npz cache)
        import ssp_stata as ST
        cache = Path(d["path"] + ".npz")
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            scans = list(zip(z["sts"].tolist(), z["ranges"]))
            beam, poses = z["beam"], z["poses"]
            rmax = float(z["range_max"]) - 0.1
        else:
            scans, beam, poses, rm = ST.load_bag(d["path"])
            np.savez_compressed(
                cache, sts=np.array([s[0] for s in scans], np.int64),
                ranges=np.stack([s[1] for s in scans]), beam=beam,
                poses=poses, range_max=rm)
            rmax = rm - 0.1
        keys = ST.keyframes(scans, poses)
        if cap:
            keys = keys[:cap]
        rmin = ST.RANGE_MIN
    return dict(name=name, kind=d["kind"], eval=d["eval"], path=d["path"],
                keys=keys, beam=beam,
                odom=np.stack([k[1] for k in keys]),
                kts=np.array([t for _, _, t in keys]),
                rmin=rmin, rmax=rmax)


def clean(bundle, r):
    """Invalid returns -> inf, per dataset kind."""
    if bundle["rmin"] is None:
        return np.where(r < bundle["rmax"], r, np.inf)
    return np.where((r > bundle["rmin"]) & (r < bundle["rmax"]), r,
                    np.inf).astype(float)


def _ref_by_identity(bundle):
    """Range-array-identity reference: per-keyframe interpolated xy + valid
    mask (belg/mit convention; gfs timestamps corrupt there)."""
    path = bundle["path"]
    raw = C.parse_flaser(path)
    gfs = C.parse_flaser(path.replace(".log", ".gfs.log"))
    gxy = np.stack([p[:2] for _, p, _ in gfs])
    idx = {}
    for i, (r, _, _) in enumerate(raw):
        idx.setdefault(r.tobytes(), []).append(i)
    gts, keep = [], []
    for m, (r, _, _) in enumerate(gfs):
        b = r.tobytes()
        if b in idx:
            gts.append(raw[idx[b][0]][2])
            keep.append(m)
    gts = np.array(gts)
    gxy = gxy[keep]
    o = np.argsort(gts)
    gts, gxy = gts[o], gxy[o]
    uniq = np.concatenate([[True], np.diff(gts) > 0])
    gts, gxy = gts[uniq], gxy[uniq]
    kts = bundle["kts"]
    ref = np.stack([np.interp(kts, gts, gxy[:, 0]),
                    np.interp(kts, gts, gxy[:, 1])], 1)
    valid = (kts >= gts[0]) & (kts <= gts[-1])
    return ref, valid


def _gt_stata():
    import ssp_stata as ST
    return ST.load_gt()


def evaluate(bundle, fin):
    """Score a final trajectory. -> dict(ate, med, mx, n_ref)."""
    kts = bundle["kts"]
    if bundle["eval"] == "gfs":
        ref = C.parse_flaser(bundle["path"].replace(".log", ".gfs.log"))
        rts = np.array([t for _, _, t in ref])
        rxy = np.stack([p[:2] for _, p, _ in ref])
        j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
        good = np.abs(rts - kts[j]) < 0.3
        al = C.align_se2(fin[j[good], :2], rxy[good])
        e = np.linalg.norm(al - rxy[good], axis=1)
    elif bundle["eval"] == "ident":
        ref, valid = _ref_by_identity(bundle)
        al = C.align_se2(fin[valid, :2], ref[valid])
        e = np.linalg.norm(al - ref[valid], axis=1)
    elif bundle["eval"] == "exact":              # synthetic bench (ssp_synth)
        gt = bundle["gt"]
        al = C.align_se2(fin[:, :2], gt[:, :2])
        e = np.linalg.norm(al - gt[:, :2], axis=1)
    else:                                        # stata floorplan GT
        gts, gxy = _gt_stata()
        j = np.abs(gts[:, None] - kts[None, :]).argmin(1)
        good = np.abs(gts - kts[j]) < 30_000
        al = C.align_se2(fin[j[good], :2], gxy[good])
        e = np.linalg.norm(al - gxy[good], axis=1)
    return dict(ate=float(np.sqrt((e ** 2).mean())), med=float(np.median(e)),
                mx=float(e.max()), n_ref=int(len(e)))


def ref_for_keys(bundle):
    """Per-KEYFRAME reference xy + ok mask (display/packing direction;
    scoring uses evaluate()'s reference->keyframe direction)."""
    kts = bundle["kts"]
    n = len(kts)
    if bundle["eval"] == "ident":
        ref, valid = _ref_by_identity(bundle)
        return ref.astype(np.float32), valid.astype(np.uint8)
    if bundle["eval"] == "exact":                # synthetic bench (ssp_synth)
        return (bundle["gt"][:, :2].astype(np.float32),
                np.ones(n, np.uint8))
    if bundle["eval"] == "gfs":
        gfs = C.parse_flaser(bundle["path"].replace(".log", ".gfs.log"))
        rts = np.array([t for _, _, t in gfs])
        rxy = np.stack([p[:2] for _, p, _ in gfs])
        tol = 0.3
    else:
        rts, rxy = _gt_stata()
        tol = 30_000
    jj = np.abs(kts[:, None] - rts[None, :]).argmin(1)
    kgood = np.abs(kts - rts[jj]) < tol
    ref_xy = np.zeros((n, 2), np.float32)
    ref_xy[kgood] = rxy[jj[kgood]].astype(np.float32)
    return ref_xy, kgood.astype(np.uint8)


def run(name, cls=F.BandSLAM, cap=None, sample="seg", slam=None, **kw):
    """The shared run loop — a faithful transcription of ssp_fpga.run_log
    (bit-exact on CARMEN logs, asserted in selftest), dataset-generic.
    Harness defaults may be overridden via kw (no collision trap)."""
    bundle = name if isinstance(name, dict) else load(name, cap)
    keys, beam, odom = bundle["keys"], bundle["beam"], bundle["odom"]
    n = len(keys)
    if slam is None:
        args = dict(robust=True, attempt_every=4, relax_every=25,
                    gap_kf=300, recent_aids=12)
        args.update(kw)
        slam = cls(**args)
    slam.store_dtype = np.complex64
    est = np.zeros((n, 3))
    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = clean(bundle, r)
        if callable(sample):
            pts, w = sample(rr, beam)
        elif sample == "point":
            pts, w = F.points_from_scan(rr, beam)
        elif sample == "pointcap":
            pts, w = F.points_from_scan(rr, beam, wcap=0.25)
        elif sample == "hitw":
            pts, w = F.points_from_scan_occw(rr, beam)
        else:
            pts, w, _ = S.scan_to_samples(rr, beam)
        if bundle.get("guess_mode") == "cv" and k > 0:
            # lidar-only: constant-velocity extrapolation of own estimates
            # (bundle odom is WITHHELD ground truth, display/eval only)
            if k == 1:
                guess = est[0].copy()
            else:
                v = est[k - 1] - est[k - 2]
                vn = np.hypot(v[0], v[1])
                if vn > 0.30:
                    v[:2] *= 0.30 / vn
                guess = np.array([est[k - 1][0] + v[0],
                                  est[k - 1][1] + v[1],
                                  est[k - 1][2] + S.wrap(v[2])])
        else:
            guess = opose if k == 0 else L.se2_mul(
                est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
    if slam.dirty:
        slam.relax()
    dt = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    sc = evaluate(bundle, fin)
    nloop = sum(1 for e in slam.edges if e[5] == "loop")
    return dict(ate=sc["ate"], med=sc["med"], mx=sc["mx"],
                n_ref=sc["n_ref"], loops=nloop, ms=dt / n * 1e3,
                mem_kb=slam.memory_kb(), pruned=slam.n_pruned,
                veto=slam.n_veto, infl=slam.n_inflate,
                innov=slam.n_innov_rej, coh_ref=slam.coh_ref,
                jit=getattr(slam, "n_jit_rej", 0), fin=fin, est=est,
                slam=slam, bundle=bundle)


CONFIGS = {
    "shipped": dict(sample="seg", kw=dict(spec=None, nph=0)),
    "e2": dict(sample="point", kw=dict(spec=None, nph=0)),
    "lean": dict(sample="point", kw=dict(spec="i8", nph=4, nmag=1,
                                         ring_scales=True)),
}


def _spec(kw):
    kw = dict(kw)
    if kw.get("spec") == "i8":
        kw["spec"] = F.IntSpec(8, 7, 7)
    return kw


def selftest():
    cap = 1200
    a = F.run_log(F.LOGS["fr101"], F.BandSLAM, cap=cap, spec=None, nph=0)
    b = run("fr101", F.BandSLAM, cap=cap, spec=None, nph=0)
    d = float(np.abs(a["fin"] - b["fin"]).max())
    print(f"selftest fr101[:{cap}]  run_log {a['ate']:.4f}  "
          f"registry {b['ate']:.4f}  max|dpose| {d:.2e}", flush=True)
    assert d == 0.0 and a["ate"] == b["ate"], "registry != run_log"
    print("selftest ok: registry run == ssp_fpga.run_log bit-exact")


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if what == "selftest":
        selftest()
        return
    assert what == "run"
    name = sys.argv[2]
    cfg = CONFIGS[sys.argv[3] if len(sys.argv) > 3 else "shipped"]
    r = run(name, F.BandSLAM, sample=cfg["sample"], **_spec(cfg["kw"]))
    print(f"{name}: ATE {r['ate']:.3f}  med {r['med']:.3f}  "
          f"loops {r['loops']}  mem {r['mem_kb']:.0f} KB  "
          f"({r['n_ref']} ref poses, {r['ms']:.0f} ms/kf)")


if __name__ == "__main__":
    main()
