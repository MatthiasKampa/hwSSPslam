"""LIDAR SCALING study (user 2026-07-14: "lidar needs to be scaled").

Three ladders on school_run2 (est-labels DIAGNOSTIC for place; SE(3)
displacement decode is label-free), all on the azel3d recipe family with
ROTATION-SEARCHED similarity (the matcher-faithful sim):

  input   what enters the encoder at D=240:
            slice1  ring-33 binned 1024-beam scan (the shipped 2D input,
                    z=0 points)
            rings3  rings {16, 33, 50} (2.5D — 3 slices of the head)
            full    full 64-ring cloud (SUB=16, ~4k pts)
  D       azel3d capacity ladder at full-cloud input:
            240  = 12 az x 5 elev x 4 lam   (the recipe)
            480  = 24 az x 5 elev x 4 lam   (finer az -> finer rot search)
            960  = 24 az x 5 elev x 8 lam   (sqrt2 ladder 0.25..8 m)
  SUB     full-cloud subsample at D=240: 32 / 16 / 8 (~2k/4k/8k pts)
          — the ingest-throughput corner (pts/frame vs quality)

Usage: python3 -m experiments.lidarscale [input|dcap|sub|disp] [run]
"""
import io
import sys

import numpy as np

import experiments.lattice3d as L3

RUN = "school_run2"


def azel_W(n_az, elevs, lams):
    return np.concatenate([(2 * np.pi / lam) * L3.dirs_azel(n_az, elevs)
                           for lam in [lams[0]]]) if False else \
        np.concatenate([(2 * np.pi / lam)
                        * L3.dirs_azel(n_az, elevs) for lam in lams])


LADDERS = {
    "D240": (12, [-40, -20, 0, 20, 40], list(L3.LAMS)),
    "D480": (24, [-40, -20, 0, 20, 40], list(L3.LAMS)),
    "D960": (24, [-40, -20, 0, 20, 40],
             [0.25, 0.354, 0.5, 0.707, 1.0, 1.414, 2.0, 2.828]),
}


def rot_sim(V, W, n_az):
    mats = [np.abs(V @ V.conj().T)]
    for k in range(1, n_az):
        th = k * np.pi / n_az
        R = np.array([[np.cos(th), -np.sin(th), 0],
                      [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
        WR = W @ R
        dp = np.linalg.norm(WR[:, None] - W[None], axis=2)
        dm = np.linalg.norm(WR[:, None] + W[None], axis=2)
        perm = np.minimum(dp, dm).argmin(1)
        sgn = dp[np.arange(len(W)), perm] <= dm[np.arange(len(W)), perm]
        Vp = np.where(sgn, V[:, perm], np.conj(V[:, perm]))
        mats.append(np.abs(Vp @ V.conj().T))
    return np.max(np.stack(mats), 0)


def load_ring_clouds(run, ts_want, rings, sub=4, tol_ms=60):
    """xyz restricted to a ring subset (school npz shards carry ring)."""
    import pyarrow.parquet as pq
    shards, cts, where = L3.cloud_index(run)
    j = np.clip(np.searchsorted(cts, ts_want), 1, len(cts) - 1)
    j = j - (np.abs(cts[j - 1] - ts_want) < np.abs(cts[j] - ts_want))
    ok = np.abs(cts[j] - ts_want) < tol_ms * 1e6
    by_shard = {}
    for i, k in enumerate(j):
        if ok[i]:
            si, row = where[k]
            by_shard.setdefault(si, []).append((row, i))
    out = [None] * len(ts_want)
    rset = set(rings)
    for si in sorted(by_shard):
        t = pq.read_table(shards[si])
        for row, i in sorted(by_shard[si]):
            z = np.load(io.BytesIO(t["npz_bytes"][row].as_py()))
            m = np.isin(z["ring"], list(rset))
            xyz = z["xyz"][m][::sub].astype(np.float64)
            r = np.linalg.norm(xyz[:, :2], axis=1)
            out[i] = xyz[(r > 0.3) & (r < 60.0)]
    return out, ok


def slice_points(run, pick):
    d = L3.BASE if run == "spot" else L3.BASE / run
    z = np.load(d / "scans.npz")
    idx = np.arange(0, len(z["ts"]), 4)[pick]
    beam = -np.pi + np.arange(1024) * (2 * np.pi / 1024)
    out = []
    for i in idx:
        r = z["ranges"][i]
        m = np.isfinite(r)
        out.append(np.stack([r[m] * np.cos(beam[m]),
                             r[m] * np.sin(beam[m]),
                             np.zeros(m.sum())], 1)[::1])
    return out


def _setup(run, n=110):
    pick, kts, pose, kind = L3.sample_kf(run, n, need_ref=True,
                                         labels="est")
    pr = dict(same_r=1.0, far_lo=4.0)
    return pick, kts, pose, kind, pr


def _place(clouds, pose, pr, name, n_az, W):
    keep = [i for i, c in enumerate(clouds)
            if c is not None and len(c) > 300]
    V = np.stack([L3.encode(W, clouds[i]) for i in keep])
    (si, sj), (di, dj) = L3._pairs(pose[keep], **pr)
    S = rot_sim(V, W, n_az)
    adj = np.arange(len(keep) - 1)
    print(f"  {name:12s} AUC {L3._auc(S[si, sj], S[di, dj]):.3f}  "
          f"adj {L3._auc(S[adj, adj + 1], S[di, dj]):.3f}  "
          f"({len(keep)} frames, pts med "
          f"{int(np.median([len(clouds[i]) for i in keep]))})", flush=True)


def bench_input(run=RUN):
    pick, kts, pose, kind, pr = _setup(run)
    n_az, el, lams = LADDERS["D240"]
    W = azel_W(n_az, el, lams)
    print(f"input ladder ({run}, labels={kind}, azel3d D=240, "
          f"rot-searched):")
    sl = slice_points(run, np.arange(len(pick)))
    _place(sl, pose, pr, "slice1(2D)", n_az, W)
    r3, _ = load_ring_clouds(run, kts, rings=[16, 33, 50])
    _place(r3, pose, pr, "rings3", n_az, W)
    full, _ = L3.load_clouds(run, kts)
    _place(full, pose, pr, "full(SUB16)", n_az, W)


def bench_dcap(run=RUN):
    pick, kts, pose, kind, pr = _setup(run)
    full, _ = L3.load_clouds(run, kts)
    print(f"D ladder ({run}, labels={kind}, full cloud, rot-searched):")
    for name, (n_az, el, lams) in LADDERS.items():
        _place(full, pose, pr, name, n_az, azel_W(n_az, el, lams))


def bench_sub(run=RUN):
    pick, kts, pose, kind, pr = _setup(run)
    n_az, el, lams = LADDERS["D240"]
    W = azel_W(n_az, el, lams)
    print(f"SUB ladder ({run}, labels={kind}, azel3d D=240):")
    for sub in (32, 16, 8):
        old = L3.SUB
        L3.SUB = sub                     # module-global patch (harness)
        clouds, _ = L3.load_clouds(run, kts)
        L3.SUB = old
        _place(clouds, pose, pr, f"SUB{sub}", n_az, W)


def bench_disp(run=RUN):
    """Label-free: SE(3) decode error at the D ladder."""
    pick, kts, _, _, _ = _setup(run, n=20)
    full, _ = L3.load_clouds(run, kts)
    clouds = [c for c in full if c is not None and len(c) > 500]
    rng = np.random.default_rng(11)
    steps = np.arange(-1.2, 1.21, 0.15)
    gx, gy, gz = np.meshgrid(steps, steps, steps, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    print(f"disp decode at D ladder ({run}, {len(clouds)} clouds):")
    for name, (n_az, el, lams) in LADDERS.items():
        W = azel_W(n_az, el, lams)
        errs = []
        for c in clouds:
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            d = 0.8 * u
            v0, v1 = L3.encode(W, c), L3.encode(W, c + d)
            g = np.conj(v1) * v0
            sc = np.real(np.exp(1j * (grid @ W.T)) @ g)
            errs.append(np.linalg.norm(grid[int(np.argmax(sc))] - d))
        print(f"  {name:6s} err med {np.median(errs):.3f} "
              f"p90 {np.percentile(errs, 90):.3f}", flush=True)


AXIS = {
    # equal-D 960 variants: which axis buys the win?
    "az48xel5xL4": (48, [-40, -20, 0, 20, 40], None),
    "az24xel5xL8": (24, [-40, -20, 0, 20, 40],
                    [0.25, 0.354, 0.5, 0.707, 1.0, 1.414, 2.0, 2.828]),
    "az24xel10xL4": (24, [-45, -35, -25, -15, -5, 5, 15, 25, 35, 45],
                     None),
    # headroom probe
    "D1920": (48, [-40, -20, 0, 20, 40],
              [0.25, 0.354, 0.5, 0.707, 1.0, 1.414, 2.0, 2.828]),
    # ECP5-ceiling probes (user: take full advantage of the larger part)
    "D3840az": (96, [-40, -20, 0, 20, 40],
                [0.25, 0.354, 0.5, 0.707, 1.0, 1.414, 2.0, 2.828]),
    "D3840lam": (48, [-40, -20, 0, 20, 40],
                 [0.125, 0.177, 0.25, 0.354, 0.5, 0.707, 1.0, 1.414,
                  2.0, 2.828, 4.0, 5.657, 8.0, 11.314, 16.0, 22.627]),
}


def rot_sim_refined(V, W, n_az):
    """rot_sim + parabolic peak refinement across the rotation index —
    cheap sub-step rotation (the d/dtheta idea without the derivative
    vector): fit s(k-1), s(k), s(k+1) per pair, take the interpolated
    peak VALUE (similarity at the refined rotation)."""
    mats = []
    for k in range(n_az):
        th = k * np.pi / n_az
        R = np.array([[np.cos(th), -np.sin(th), 0],
                      [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
        WR = W @ R
        dp = np.linalg.norm(WR[:, None] - W[None], axis=2)
        dm = np.linalg.norm(WR[:, None] + W[None], axis=2)
        perm = np.minimum(dp, dm).argmin(1)
        sgn = dp[np.arange(len(W)), perm] <= dm[np.arange(len(W)), perm]
        Vp = np.where(sgn, V[:, perm], np.conj(V[:, perm]))
        mats.append(np.abs(Vp @ V.conj().T))
    S = np.stack(mats)                     # (n_az, n, n)
    k = S.argmax(0)
    n = S.shape[1]
    ii, jj = np.mgrid[0:n, 0:n]
    sk = S[k, ii, jj]
    sm = S[(k - 1) % n_az, ii, jj]
    sp = S[(k + 1) % n_az, ii, jj]
    den = np.maximum(sm - 2 * sk + sp, 1e-12)
    return sk + np.where(den < 0, -((sp - sm) ** 2) / (8 * den), 0.0)


def bench_axis(run=RUN):
    pick, kts, pose, kind, pr = _setup(run)
    full, _ = L3.load_clouds(run, kts)
    print(f"axis decomposition ({run}, labels={kind}, full cloud, "
          f"rot-searched; D240/D480/D960 banked 0.700/0.814/0.858):")
    for name, (n_az, el, lams) in AXIS.items():
        W = azel_W(n_az, el, lams or list(L3.LAMS))
        _place(full, pose, pr, f"{name}(D{len(W)})", n_az, W)
    # parabolic rotation refinement on the banked D960 recipe
    n_az, el, lams = LADDERS["D960"]
    W = azel_W(n_az, el, lams)
    keep = [i for i, c in enumerate(full) if c is not None and len(c) > 300]
    V = np.stack([L3.encode(W, full[i]) for i in keep])
    (si, sj), (di, dj) = L3._pairs(pose[keep], **pr)
    Sr = rot_sim_refined(V, W, n_az)
    print(f"  D960+parab    AUC {L3._auc(Sr[si, sj], Sr[di, dj]):.3f}  "
          f"(refined rotation peak)", flush=True)
    # ingest recipe at the scaled lattice: 3 rings x D960
    r3, _ = load_ring_clouds(run, kts, rings=[16, 33, 50])
    _place(r3, pose, pr, "rings3xD960", n_az, W)


L8 = [0.25, 0.354, 0.5, 0.707, 1.0, 1.414, 2.0, 2.828]
L16_FULL = [0.125, 0.177, 0.25, 0.354, 0.5, 0.707, 1.0, 1.414,
            2.0, 2.828, 4.0, 5.657, 8.0, 11.314, 16.0, 22.627]
TUNE = {
    # ladder composition at D1920 (24az x 5el x 16lam = 1920)
    "lam16-full": (24, [-40, -20, 0, 20, 40], L16_FULL),
    "lam16-coarse": (24, [-40, -20, 0, 20, 40],
                     [0.5, 0.707, 1.0, 1.414, 2.0, 2.828, 4.0, 5.657,
                      8.0, 11.314, 16.0, 22.627, 32.0, 45.255, 64.0,
                      90.51]),
    "lam16-fine": (24, [-40, -20, 0, 20, 40],
                   [0.088, 0.125, 0.177, 0.25, 0.354, 0.5, 0.707, 1.0,
                    1.414, 2.0, 2.828, 4.0, 5.657, 8.0, 11.314, 16.0]),
    "lam8-oct-wide": (48, [-40, -20, 0, 20, 40],
                      [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]),
    # elevation distribution at D1920 (dirs x 8lam)
    "el5+-20": (48, [-20, -10, 0, 10, 20], L8),
    "el5+-22sens": (48, [-22, -11, 0, 11, 22], L8),
    "el3+-15": (80, [-15, 0, 15], L8),
    "el9+-40": (27, [-40, -30, -20, -10, 0, 10, 20, 30, 40], L8),
    # ceiling recipe candidates at D3840
    "ceil-a": (48, [-22, -11, 0, 11, 22], L16_FULL),
    "ceil-b": (24, [-22, -11, 0, 11, 22],
               [x for x in L16_FULL for _ in (0,)] * 2),
}
TUNE["ceil-b"] = (24, [-33, -22, -11, 0, 11, 22, 33, 44, -44, 55],
                  L16_FULL)


def bench_tune(run=RUN):
    """All-elements tuning at the ECP5 ceiling (user: scales, vec
    distribution, ... across the three corners). Banked refs: D1920
    0.876; D3840az 0.880; D3840lam 0.915."""
    pick, kts, pose, kind, pr = _setup(run)
    full, _ = L3.load_clouds(run, kts)
    print(f"ceiling tuning ({run}, labels={kind}):")
    for name, (n_az, el, lams) in TUNE.items():
        W = azel_W(n_az, el, lams)
        _place(full, pose, pr, f"{name}(D{len(W)})", n_az, W)


LAMC16 = [0.5, 0.707, 1.0, 1.414, 2.0, 2.828, 4.0, 5.657, 8.0, 11.314,
          16.0, 22.627, 32.0, 45.255, 64.0, 90.51]


def _quant_vec(V, nph):
    """Phase-only quantization (house 2b = nph 4, nmag 1 semantics)."""
    if nph == 0:
        return V
    ph = np.round(np.angle(V) * nph / (2 * np.pi)) * (2 * np.pi / nph)
    Q = np.exp(1j * ph)
    return Q / np.linalg.norm(Q, axis=-1, keepdims=True)


def bench_fidelity(run=RUN):
    """Map fidelity + matching granularity/space at the ceiling recipe
    (D1920 = 24az x 5el x lam16-coarse; banked float 0.928):
      quant   snapshot store at nph {4(2b), 8(3b), 16(4b), float}
      rings   matching SPACE: coarse-8 rings only / fine-8 only / all
      rotstep rotation-search granularity: 24 / 12 / 6 perms
    Corners: S = bytes/anchor at each nph; T = perms x D per query."""
    pick, kts, pose, kind, pr = _setup(run)
    full, _ = L3.load_clouds(run, kts)
    n_az, el = 24, [-40, -20, 0, 20, 40]
    W = azel_W(n_az, el, LAMC16)
    keep = [i for i, c in enumerate(full) if c is not None and len(c) > 300]
    V = np.stack([L3.encode(W, full[i]) for i in keep])
    (si, sj), (di, dj) = L3._pairs(pose[keep], **pr)
    print(f"fidelity + granularity at D1920-coarse ({run}, labels={kind}; "
          f"float/24-perm banked 0.928):")
    for nph, tag in ((4, "2b"), (8, "3b"), (16, "4b"), (0, "float")):
        Vq = _quant_vec(V, nph)
        S = rot_sim(Vq, W, n_az)
        by = 2 * 1920 * (2 if nph == 0 else int(np.log2(max(nph, 2)))) // 8
        print(f"  quant {tag:5s}: AUC {L3._auc(S[si, sj], S[di, dj]):.3f}"
              f"  ({by} B/anchor)", flush=True)
    half = len(W) // 2
    for name, m in (("coarse-8 rings", np.arange(len(W)) >= half),
                    ("fine-8 rings", np.arange(len(W)) < half)):
        Wm, Vm = W[m], V[:, m]
        Vm = Vm / np.linalg.norm(Vm, axis=1, keepdims=True)
        S = rot_sim(Vm, Wm, n_az)
        print(f"  space {name}: AUC {L3._auc(S[si, sj], S[di, dj]):.3f}"
              f"  (D{int(m.sum())})", flush=True)
    for step in (24, 12, 6):
        S = rot_sim(V, W, step)
        print(f"  rot-search {step:2d} perms: AUC "
              f"{L3._auc(S[si, sj], S[di, dj]):.3f}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "input"
    run = sys.argv[2] if len(sys.argv) > 2 else RUN
    dict(input=bench_input, dcap=bench_dcap, sub=bench_sub,
         disp=bench_disp, axis=bench_axis, tune=bench_tune,
         fidelity=bench_fidelity)[cmd](run)
