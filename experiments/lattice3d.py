"""3D feature-vector lattices for the ECP5 fusion track (user directive
2026-07-14: "try both a 2D version and a 3D version — e.g. constructing the
feature vec via Fibonacci sphere (multiple scales) or via other structured
layouts").

Layouts (all equal-D so comparisons are fair; scales = the shipped matched
ladder from sspslam.lattice):
  az2d    n_az azimuth directions in the xy-plane x ladder — the shipped 2D
          polar lattice's shape (z ignored); EXACT yaw permutation.
  fib3d   Fibonacci-sphere directions x ladder — isotropic 3D coverage;
          yaw rotation is NOT an index permutation (nearest-direction
          approximation only — quantified in bench rot).
  azel3d  azimuth rings x elevation bands x ladder — structured 3D layout
          that KEEPS the exact-yaw-permutation algebra per (elev, lam) ring
          (the house rot_permute generalizes band-wise).
  rand3d  random directions control (the 2D encoder study's loser).

Encoding: phi(P) = sum_i w_i exp(i W . p_i), W rows = (2*pi/lam) * u_dir.
Anti-oracle: references label test pairs only; nothing enters an encoder.

Benches (python3 -m experiments.lattice3d <bench> [run]):
  selftest  layout invariants + azel/az2d exact-permutation checks
  disp      SE(3) displacement decode error on real school full clouds
  rot       yaw algebra: exact for az2d/azel3d; fib3d approximation error
  place     same-place separability AUC, 2D ring-slice vs 3D full-cloud
            (labels from gated LIO where available, else the shipped run's
            own estimate — DIAGNOSTIC labels, stated per PROTOCOL)
  cam       camera-bearing channel (unit vectors via K^-1) alone + bundled
            with the lidar cloud vector

Full clouds are decoded on demand from the school pointclouds parquets
(npz_bytes xyz), subsampled deterministically.
"""
import io
import sys
from pathlib import Path

import numpy as np

import sspslam.lattice as L

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "spot_telluride"
LAMS = [float(x) for x in np.unique(np.round(
    np.asarray(L.LAMS, float), 6))][:4] or [0.5, 1.0, 2.0, 4.0]
N_DIR = 60                      # directions per scale -> D = 240 everywhere
SUB = 16                        # cloud subsample stride (65536 -> 4096 pts)
RNG = 11


# --------------------------------------------------------------------------
#  lattice layouts
# --------------------------------------------------------------------------
def dirs_az(n):
    a = np.arange(n) * (np.pi / n)          # half-circle, house convention
    return np.stack([np.cos(a), np.sin(a), np.zeros(n)], 1)


def dirs_fib(n):
    """Fibonacci sphere on the HALF sphere (phasors give the antipode for
    free, matching the half-circle convention of the 2D lattice)."""
    g = (1 + 5 ** 0.5) / 2
    k = np.arange(n) + 0.5
    z = k / n                                # upper half only: z in (0, 1)
    r = np.sqrt(1 - z * z)
    th = 2 * np.pi * k / g
    return np.stack([r * np.cos(th), r * np.sin(th), z], 1)


def dirs_azel(n_az, elevs_deg):
    out = []
    for e in np.deg2rad(elevs_deg):
        c, s = np.cos(e), np.sin(e)
        a = np.arange(n_az) * (np.pi / n_az)
        out.append(np.stack([c * np.cos(a), c * np.sin(a),
                             np.full(n_az, s)], 1))
    return np.concatenate(out)


def dirs_rand(n, seed=RNG):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v[v[:, 2] < 0] *= -1.0
    return v


def W_of(dirs, lams=LAMS):
    return np.concatenate([(2 * np.pi / lam) * dirs for lam in lams])


def make_lattices():
    # azel: 12 az x 5 elevation bands (-40..40 deg) x 4 lams = 240
    return {
        "az2d":   W_of(dirs_az(N_DIR)),
        "fib3d":  W_of(dirs_fib(N_DIR)),
        "azel3d": W_of(dirs_azel(12, [-40, -20, 0, 20, 40])),
        "rand3d": W_of(dirs_rand(N_DIR)),
    }


def encode(W, pts, w=None):
    v = np.exp(1j * (pts @ W.T))
    v = v.sum(0) if w is None else v.T @ w
    return v / max(np.linalg.norm(v), 1e-12)


# --------------------------------------------------------------------------
#  data access: full clouds + labels
# --------------------------------------------------------------------------
def cloud_index(run):
    import pyarrow.parquet as pq
    d = BASE if run == "spot" else BASE / run       # classroom lives flat
    shards = sorted((d / "pointclouds").glob("*.parquet"))
    ts, where = [], []
    for si, sh in enumerate(shards):
        v = np.array(pq.read_table(sh, columns=["timestamp_ns"])
                     ["timestamp_ns"], dtype=np.int64)
        ts.append(v)
        where += [(si, r) for r in range(len(v))]
    ts = np.concatenate(ts)
    srt = np.argsort(ts)
    return shards, ts[srt], [where[i] for i in srt]


def load_clouds(run, ts_want, tol_ms=60):
    """Full xyz clouds nearest each wanted timestamp (deterministic SUB)."""
    import pyarrow.parquet as pq
    shards, cts, where = cloud_index(run)
    j = np.clip(np.searchsorted(cts, ts_want), 1, len(cts) - 1)
    j = j - (np.abs(cts[j - 1] - ts_want) < np.abs(cts[j] - ts_want))
    ok = np.abs(cts[j] - ts_want) < tol_ms * 1e6
    by_shard = {}
    for i, k in enumerate(j):
        if ok[i]:
            si, row = where[k]
            by_shard.setdefault(si, []).append((row, i))
    out = [None] * len(ts_want)
    for si in sorted(by_shard):
        t = pq.read_table(shards[si])
        pcd = "pcd_bytes" in t.column_names          # classroom ASCII PCD
        for row, i in sorted(by_shard[si]):
            if pcd:
                blob = t["pcd_bytes"][row].as_py()
                head = blob.index(b"DATA ascii") + len(b"DATA ascii\n")
                lines = blob[head:].split(b"\n")
                arr = np.genfromtxt([ln for ln in lines[::SUB] if ln],
                                    dtype=np.float64)
                xyz = arr[:, :3]
            else:
                z = np.load(io.BytesIO(t["npz_bytes"][row].as_py()))
                xyz = z["xyz"][::SUB].astype(np.float64)
            r = np.linalg.norm(xyz[:, :2], axis=1)
            out[i] = xyz[(r > 0.3) & (r < 60.0)]
    return out, ok


def sample_kf(run, n_frames, need_ref=True, labels="ref"):
    """(kf indices, ts, pose xyz-yaw, label_kind) — classroom uses its
    (excellent) withheld kinematic odometry; school the gated LIO window.
    labels='est': the canonical run's OWN estimate labels the pairs
    (DIAGNOSTIC per PROTOCOL — labels never enter an encoder, but they are
    lidar-derived with ~1 m error on school_run2; pair margins widened
    accordingly by the caller)."""
    d = BASE if run == "spot" else BASE / run
    z = np.load(d / "scans.npz")
    kts = z["ts"][np.arange(0, len(z["ts"]), 4)]
    if labels == "est":
        cache = d / "est_cache.npz"
        if not cache.exists():
            import runners.datasets as DS
            r = DS.run(run, DS.F.BandSLAM, spec=None, nph=0)
            np.savez_compressed(cache, fin=r["fin"])
        fin = np.load(cache)["fin"]
        pick = np.linspace(0, len(kts) - 1, n_frames).astype(int)
        return pick, kts[pick], fin[pick], "OWN ESTIMATE (diagnostic)"
    if run == "spot":
        import runners.spot as SP
        b = SP.make_bundle()
        gt, ok = b["gt"], b["gt_ok"]
    else:
        rz = np.load(d / "ref_lio.npz")
        gt, ok = rz["gt"], rz["ok"]
    if need_ref and ok.sum() >= n_frames:
        idx = np.flatnonzero(ok)
        pick = idx[np.linspace(0, len(idx) - 1, n_frames).astype(int)]
        lbl = "withheld odometry" if run == "spot" else "gated LIO-SAM"
        return pick, kts[pick], gt[pick], lbl
    pick = np.linspace(0, len(kts) - 1, n_frames).astype(int)
    return pick, kts[pick], None, "none"


# --------------------------------------------------------------------------
#  benches
# --------------------------------------------------------------------------
def bench_disp(run="school_run2", n_frames=24, seed=RNG):
    """Known SE(3) displacement decode: encode cloud, shift by known d
    (incl. z), decode via correlation over a 3D candidate grid; report
    |dhat - d|. The 2D lattice cannot see dz (its grid searches dz=0)."""
    lat = make_lattices()
    _, kts, _, _ = sample_kf(run, n_frames, need_ref=False)
    clouds, ok = load_clouds(run, kts)
    rng = np.random.default_rng(seed)
    steps = np.arange(-1.2, 1.21, 0.15)     # candidate grid per axis
    print(f"displacement decode ({run}, {int(ok.sum())} clouds, "
          f"grid step 0.15 m, D=240):")
    grids = {}
    for name in lat:
        zs = steps if name != "az2d" else np.array([0.0])
        gx, gy, gz = np.meshgrid(steps, steps, zs, indexing="ij")
        grids[name] = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    for mag in (0.3, 0.8):
        errs = {k: [] for k in lat}
        for c in clouds:
            if c is None or len(c) < 500:
                continue
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            d = mag * u
            for name, W in lat.items():
                v0 = encode(W, c)
                v1 = encode(W, c + d)
                g = np.conj(v1) * v0
                # score(cand) = Re sum_f g_f exp(i W_f . cand): one matmul
                sc = np.real(np.exp(1j * (grids[name] @ W.T)) @ g)
                best = grids[name][int(np.argmax(sc))]
                errs[name].append(np.linalg.norm(best - d))
        for name in lat:
            e = np.array(errs[name])
            print(f"  |d|={mag:.1f}: {name:7s} err med {np.median(e):.3f} "
                  f"p90 {np.percentile(e, 90):.3f}  (n={len(e)})",
                  flush=True)


def bench_rot(run="school_run2", n_frames=24):
    """Yaw algebra: az2d/azel3d rotation by the ring step == EXACT index
    permutation; fib3d gets nearest-direction permutation (approx)."""
    lat = make_lattices()
    _, kts, _, _ = sample_kf(run, n_frames, need_ref=False)
    clouds, _ = load_clouds(run, kts)
    clouds = [c for c in clouds if c is not None and len(c) > 500]
    th = np.pi / 12                        # 15 deg = az ring step
    R = np.array([[np.cos(th), -np.sin(th), 0],
                  [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    print(f"yaw algebra at {np.degrees(th):.0f} deg ({len(clouds)} clouds):")
    for name, W in make_lattices().items():
        # v_rot_f = sum exp(i W_f . (R p)) = v0 evaluated at row W_f @ R
        # (row-vector convention: p' = p @ R.T) -> permutation maps W @ R
        # rows onto original rows, conj where the half-sphere flips sign.
        WR = W @ R
        n = len(W)
        # match RAW rows (scale rings share directions — normalized
        # matching collides across rings); allow the antipodal conj-flip
        dp = np.linalg.norm(WR[:, None, :] - W[None, :, :], axis=2)
        dm = np.linalg.norm(WR[:, None, :] + W[None, :, :], axis=2)
        d = np.minimum(dp, dm)
        perm = d.argmin(1)
        ar = np.arange(n)
        sgn = np.where(dp[ar, perm] <= dm[ar, perm], 1.0, -1.0)
        exact = float(d[ar, perm].max()) < 1e-9 * np.linalg.norm(W, axis=1).max()
        cs = []
        for c in clouds:
            v_rot = encode(W, c @ R.T)             # truth: re-encode rotated
            v0 = encode(W, c)
            v_perm = np.where(sgn > 0, v0[perm], np.conj(v0[perm]))
            v_perm = v_perm / max(np.linalg.norm(v_perm), 1e-12)
            cs.append(np.abs(np.vdot(v_rot, v_perm)))
        print(f"  {name:7s} perm-exact={exact}  cos(re-encode, permuted) "
              f"med {np.median(cs):.4f} min {np.min(cs):.4f}", flush=True)


def _pairs(pose, same_r=0.7, far_lo=3.0, far_hi=10.0, gap=40):
    n = len(pose)
    d = np.linalg.norm(pose[:, None, :2] - pose[None, :, :2], axis=2)
    ii, jj = np.triu_indices(n, 1)
    tgap = np.abs(ii - jj) >= gap          # exclude trivially-adjacent
    same = (d[ii, jj] < same_r) & tgap
    diff = (d[ii, jj] > far_lo) & (d[ii, jj] < far_hi)
    return (ii[same], jj[same]), (ii[diff], jj[diff])


def _auc(pos, neg):
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x)
    r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1)
    return (r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) \
        / max(len(pos) * len(neg), 1)


def bench_place(run="school_run2", n_frames=110, labels="ref"):
    """Same-place separability: cosine sim AUC of same-place vs
    different-place frame pairs; 2D slice encoding vs 3D full-cloud.
    labels='est' widens the pair margins (same<1.0 m, diff>4 m) to clear
    the ~1 m estimate error."""
    pick, kts, pose, kind = sample_kf(run, n_frames, need_ref=True,
                                      labels=labels)
    if pose is None:
        print(f"{run}: no reference labels — place bench refused")
        return
    clouds, ok = load_clouds(run, kts)
    keep = [i for i in range(len(pick)) if ok[i] and clouds[i] is not None
            and len(clouds[i]) > 500]
    pose = pose[keep]
    clouds = [clouds[i] for i in keep]
    pr = dict(same_r=1.0, far_lo=4.0) if labels == "est" else {}
    (si, sj), (di, dj) = _pairs(pose, **pr)
    if len(si) < 8:
        print(f"{run}: only {len(si)} same-place pairs in the labeled "
              f"window — report is indicative only")
    dyaw = np.abs(np.arctan2(
        np.sin(pose[:, None, 2] - pose[None, :, 2]),
        np.cos(pose[:, None, 2] - pose[None, :, 2])))
    fwd = dyaw[si, sj] < np.deg2rad(60)
    rev = dyaw[si, sj] > np.deg2rad(120)
    print(f"place separability ({run}, {len(clouds)} frames, labels="
          f"{kind}; {len(si)} same [{int(fwd.sum())} fwd/"
          f"{int(rev.sum())} rev] / {len(di)} diff pairs):")
    lat = make_lattices()
    for name, W in lat.items():
        V = np.stack([encode(W, c) for c in clouds])
        S = np.abs(V @ V.conj().T)          # raw (no rotation search)
        # rotation-searched sim for the exact-permutation layouts: max
        # |<perm_k(V), V'>| over all yaw ring steps (the house matcher's
        # rotation search; fib3d/rand3d have no exact permutation)
        Sr = None
        if name in ("az2d", "azel3d"):
            n_az = 60 if name == "az2d" else 12
            Vs = [V]
            for k in range(1, n_az):
                th = k * np.pi / n_az
                R = np.array([[np.cos(th), -np.sin(th), 0],
                              [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
                WR = W @ R
                dp = np.linalg.norm(WR[:, None] - W[None], axis=2)
                dm = np.linalg.norm(WR[:, None] + W[None], axis=2)
                perm = np.minimum(dp, dm).argmin(1)
                sgn = dp[np.arange(len(W)), perm] <= \
                    dm[np.arange(len(W)), perm]
                Vp = np.where(sgn, V[:, perm], np.conj(V[:, perm]))
                Vs.append(Vp)
            Sr = np.max(np.stack([np.abs(vp @ V.conj().T) for vp in Vs]),
                        axis=0)
        for tag, M in (("raw", S),) + ((("rot", Sr),) if Sr is not None
                                       else ()):
            auc = _auc(M[si, sj], M[di, dj])
            afw = _auc(M[si, sj][fwd], M[di, dj]) if fwd.any() else \
                float("nan")
            arv = _auc(M[si, sj][rev], M[di, dj]) if rev.any() else \
                float("nan")
            print(f"  {name:7s} {tag:3s} AUC {auc:.3f}  "
                  f"(fwd {afw:.3f} / rev {arv:.3f})", flush=True)


def bench_cam(run="school_run2", n_frames=110, labels="ref"):
    """Camera-bearing channel on the 3D lattices (bearings = K^-1 [u,v,1]
    unit vectors of FAST features, weights = score) alone and bundled
    50/50 with the lidar cloud vector."""
    pick, kts, pose, kind = sample_kf(run, n_frames, need_ref=True,
                                      labels=labels)
    if pose is None:
        print(f"{run}: no reference labels — cam bench refused")
        return
    cz = np.load((BASE if run == "spot" else BASE / run)
                 / "cam_features.npz")
    K = cz["K"]
    Ki = np.linalg.inv(K)
    off, fy, fx, fs = cz["feat_off"], cz["feat_y"], cz["feat_x"], \
        cz["feat_score"]
    clouds, ok = load_clouds(run, kts)
    keep, bear, wts, cl = [], [], [], []
    for i, k in enumerate(pick):
        if not (ok[i] and clouds[i] is not None and len(clouds[i]) > 500
                and cz["cam_ok"][k] and off[k + 1] > off[k]):
            continue
        sl = slice(off[k], off[k + 1])
        # bin-2 pixel -> full-res pixel centre
        uv = np.stack([cz["feat_x"][sl] * 2 + 1.0,
                       cz["feat_y"][sl] * 2 + 1.0,
                       np.ones(off[k + 1] - off[k])])
        b = (Ki @ uv).T
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        keep.append(i)
        bear.append(b)
        wts.append(fs[sl].astype(float))
        cl.append(clouds[i])
    pose = pose[keep]
    pr = dict(same_r=1.0, far_lo=4.0) if labels == "est" else {}
    (si, sj), (di, dj) = _pairs(pose, **pr)
    print(f"camera channel ({run}, {len(keep)} frames, labels={kind}; "
          f"{len(si)} same / {len(di)} diff pairs):")
    for name, W in make_lattices().items():
        if name == "az2d":
            continue
        # bearings live on the unit sphere: scale ladder acts as angular
        # concentration ladder
        Vc = np.stack([encode(W, b, w) for b, w in zip(bear, wts)])
        Vl = np.stack([encode(W, c) for c in cl])
        Vb = Vc + Vl
        Vb /= np.linalg.norm(Vb, axis=1, keepdims=True)
        Sc = np.abs(Vc @ Vc.conj().T)
        Sb = np.abs(Vb @ Vb.conj().T)
        print(f"  {name:7s} cam-only AUC {_auc(Sc[si, sj], Sc[di, dj]):.3f}"
              f"  lidar+cam AUC {_auc(Sb[si, sj], Sb[di, dj]):.3f}",
              flush=True)


def _axis_rot(axis, deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    if axis == "yaw":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    if axis == "pitch":
        return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _perm_of(W, R):
    WR = W @ R
    dp = np.linalg.norm(WR[:, None] - W[None], axis=2)
    dm = np.linalg.norm(WR[:, None] + W[None], axis=2)
    perm = np.minimum(dp, dm).argmin(1)
    ar = np.arange(len(W))
    sgn = dp[ar, perm] <= dm[ar, perm]
    exact = float(np.minimum(dp, dm)[ar, perm].max()) \
        < 1e-9 * np.linalg.norm(W, axis=1).max()
    return perm, sgn, exact


def _apply_perm(V, perm, sgn):
    Vp = np.where(sgn, V[..., perm], np.conj(V[..., perm]))
    return Vp / np.maximum(np.linalg.norm(Vp, axis=-1, keepdims=True),
                           1e-12)


def bench_so3(run="school_run2", n_frames=16, seed=RNG):
    """2D+heading vs 3D+pitch/yaw/roll capability, on FROZEN vectors
    (the algebra's whole point: transform without re-encoding).
    Part 1: per-axis rotation quality cos(re-encode, permuted).
    Part 2: SO(3) decode — unknown small rotation recovered by permutation
    search over a yaw/pitch/roll grid (+-15 deg, 5-deg steps); geodesic
    error in degrees. az2d = the 2D+heading system: yaw-only by design."""
    lat = {k: v for k, v in make_lattices().items() if k != "rand3d"}
    _, kts, _, _ = sample_kf(run, n_frames, need_ref=False)
    clouds, _ = load_clouds(run, kts)
    clouds = [c for c in clouds if c is not None and len(c) > 500]
    print(f"per-axis rotation quality ({run}, {len(clouds)} clouds):")
    for name, W in lat.items():
        row = [name]
        for axis in ("yaw", "pitch", "roll"):
            for deg in (5, 15):
                perm, sgn, exact = _perm_of(W, _axis_rot(axis, deg))
                cs = [np.abs(np.vdot(encode(W, c @ _axis_rot(axis, deg).T),
                                     _apply_perm(encode(W, c), perm, sgn)))
                      for c in clouds]
                row.append(f"{axis[0]}{deg}:{np.median(cs):.2f}"
                           f"{'*' if exact else ''}")
        print("  " + "  ".join(f"{x:11s}" if i else f"{x:7s}"
                               for i, x in enumerate(row)), flush=True)
    print("  (* = exact permutation)")
    # ---- SO(3) decode over a +-15deg grid --------------------------------
    degs = np.arange(-15, 16, 5)
    grid = [(y, p, r) for y in degs for p in degs for r in degs]
    Rg = [_axis_rot("yaw", y) @ _axis_rot("pitch", p) @ _axis_rot("roll", r)
          for y, p, r in grid]
    rng = np.random.default_rng(seed)
    print(f"SO(3) decode (+-15deg grid, 5-deg steps, {len(grid)} cands):")
    for name, W in lat.items():
        perms = [_perm_of(W, R)[:2] for R in Rg]
        errs = []
        for c in clouds[:10]:
            y, p, r = rng.uniform(-12, 12, 3)
            Rt = _axis_rot("yaw", y) @ _axis_rot("pitch", p) \
                @ _axis_rot("roll", r)
            v0, v1 = encode(W, c), encode(W, c @ Rt.T)
            best, bs = 0, -1
            for gi, (perm, sgn) in enumerate(perms):
                s = np.abs(np.vdot(v1, _apply_perm(v0, perm, sgn)))
                if s > bs:
                    bs, best = s, gi
            Rh = Rg[best]
            ang = np.degrees(np.arccos(np.clip(
                (np.trace(Rh @ Rt.T) - 1) / 2, -1, 1)))
            errs.append(ang)
        print(f"  {name:7s} rot err med {np.median(errs):.1f} deg  "
              f"p90 {np.percentile(errs, 90):.1f}  (grid floor ~3.5)",
              flush=True)


def selftest():
    lat = make_lattices()
    for k, W in lat.items():
        assert W.shape == (240, 3), (k, W.shape)
    # az2d ignores z by construction
    assert np.abs(lat["az2d"][:, 2]).max() == 0.0
    # azel3d: yaw by ring step is an exact permutation (checked in bench_rot
    # machinery): here just direction-set closure under 15-deg yaw
    th = np.pi / 12
    R2 = np.array([[np.cos(th), -np.sin(th), 0],
                   [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    for name in ("az2d", "azel3d"):
        W = lat[name]
        WR = W @ R2.T
        d = np.abs((WR / np.linalg.norm(WR, axis=1, keepdims=True))
                   @ (W / np.linalg.norm(W, axis=1, keepdims=True)).T)
        assert np.allclose(d.max(1), 1.0, atol=1e-9), name
    print("selftest ok: 4 layouts D=240; az2d/azel3d closed under "
          "15-deg yaw (exact permutation); fib3d deliberately not")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    run = sys.argv[2] if len(sys.argv) > 2 else "school_run2"
    if cmd == "selftest":
        selftest()
    elif cmd == "disp":
        bench_disp(run)
    elif cmd == "rot":
        bench_rot(run)
    elif cmd == "place":
        bench_place(run, labels=(sys.argv[3] if len(sys.argv) > 3
                                 else "ref"))
    elif cmd == "cam":
        bench_cam(run, labels=(sys.argv[3] if len(sys.argv) > 3
                               else "ref"))
    elif cmd == "so3":
        bench_so3(run)
