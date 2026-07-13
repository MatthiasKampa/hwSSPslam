"""Baseline 3: Rao-Blackwellized particle filter grid SLAM (GMapping-lite).

Grisetti-style RBPF, reduced to its essentials:
  - 30 particles, each with its OWN occupancy grid (int8 log-odds, 10 cm).
  - Motion model: odometry delta between keyframes composed onto each
    particle's pose, plus sampled Gaussian noise (alpha-style, scaled by
    the magnitude of the translation/rotation).
  - Improved proposal: a small correlative search (+-0.2 m x +-6 deg) around
    each particle's sampled predicted pose, scored on ITS OWN grid
    (endpoint hit count minus a Gaussian motion prior penalty).
  - Weight update: endpoint measurement model -- per-beam likelihood
    0.2 + 0.8 * p_occ(endpoint cell), log-summed, tempered by a gain
    (GMapping's "likelihood gain") so weights do not degenerate instantly.
  - Adaptive resampling: systematic resampling when N_eff < N/2; particle
    grids are COPIED on resample (the honest memory cost of RBPF).

Implementation notes:
  - All particle grids live in one (N, H, W) int8 array so the correlative
    search is a single fancy-indexed gather per keyframe: (N, R, B, T) int8.
  - Translation search offsets are whole cells (0.1 m), so candidate scores
    reuse one floor() per rotation; sub-cell accuracy comes from the sampled
    proposal noise + resampling, as in coarse GMapping settings.
  - Free space is carved by sampling points every 10 cm along each beam
    (capped at 15 m) instead of Bresenham; hits get +8, misses -2,
    clamped to +-100. Scatter is gather-modify-write (duplicate cells in one
    scan collapse to a single update -- acceptable for a lite baseline).
  - Beams are subsampled to ~90 per scan (every 2nd for 180-beam logs,
    every 4th for fr079's 360-beam scans) for both scoring and mapping.

Fairness (identical across baselines): parse_flaser / keyframes
(0.10 m / 5 deg) / align_se2 reused from ssp_slam_carmen; VALID_MAX = 40;
beam angles deg2rad(-90 + arange(n) * 180/n); ATE vs .gfs.log with 0.3 s
timestamp gate; fixed seed. Trajectory of the highest-weight particle is
reconstructed at the end (per-particle histories follow the ancestry through
every resample) and saved to <log>_traj_rbpf.npz.

Usage: python3 -m baselines.rbpf [data/intel.log ...] [--limit K]
"""

import os
import sys
import time

import numpy as np

from sspslam.frontend import parse_flaser, keyframes, align_se2, VALID_MAX

SEED = 1
N_PART = 30
RES = 0.10                       # grid resolution [m]
MARGIN = 12.0                    # grid bbox margin beyond odometry bounds [m]
L_HIT, L_MISS, L_CLAMP = 8, 2, 100
SUB_TARGET = 90                  # beams kept per scan (subsample step = n//90)
FREE_STEP, FREE_MAX = 0.10, 15.0 # free-space carve spacing / max carve range
# correlative search window (around each particle's sampled predicted pose)
DXY_CELLS = np.arange(-2, 3)                       # +-0.2 m in 0.1 m cells
DTH = np.deg2rad(np.arange(-6.0, 6.01, 1.0))       # +-6 deg in 1 deg steps
PRIOR_SXY, PRIOR_STH, PRIOR_W = 0.12, np.deg2rad(3.0), 2.0
HIT_MIN_ABS, HIT_MIN_FRAC = 8, 0.15  # accept match only above this support
GAIN = 0.10                      # likelihood tempering (GMapping-style gain)
ALPHA_P = 0.05                   # log-odds -> probability scale for weights
NEFF_FRAC = 0.5                  # resample when N_eff < NEFF_FRAC * N

# motion noise (alpha model): sigma_xy = A0 + A1*|trans| + A2*|rot|,
#                             sigma_th = B0 + B1*|rot|  + B2*|trans|
A0, A1, A2 = 0.01, 0.08, 0.02
B0, B1, B2 = np.deg2rad(0.5), 0.08, 0.02


def _build_search_tables():
    """Flattened translation offsets (T,), rotation grid, and prior penalty (R,T)."""
    dxc, dyc = np.meshgrid(DXY_CELLS, DXY_CELLS, indexing="ij")
    dxc, dyc = dxc.ravel(), dyc.ravel()                       # (T,) cells
    offm = np.stack([dxc * RES, dyc * RES], 1)                # (T,2) metres
    pen_t = (offm ** 2).sum(1) / PRIOR_SXY ** 2               # (T,)
    pen_r = DTH ** 2 / PRIOR_STH ** 2                         # (R,)
    pen = PRIOR_W * 0.5 * (pen_r[:, None] + pen_t[None, :])   # (R,T)
    return dxc.astype(np.int32), dyc.astype(np.int32), offm, pen


DXC, DYC, OFFM, PRIOR_PEN = _build_search_tables()
R_SEARCH, T_SEARCH = len(DTH), len(DXC)
R_MID, T_MID = R_SEARCH // 2, T_SEARCH // 2
# log-odds (int8) -> occupancy probability lookup, index = value + 128
PTAB = 1.0 / (1.0 + np.exp(-ALPHA_P * (np.arange(256) - 128.0)))


def scan_arrays(r, beam, step):
    """Per-keyframe precomputation shared by all particles.

    Returns hit endpoints in the sensor frame (B,2), the free-space carve
    points in the sensor frame (F,2), and the valid-beam count."""
    rs, bs = r[::step], beam[::step]
    valid = (rs < VALID_MAX) & (rs > 0.10)
    rv, bv = rs[valid], bs[valid]
    hit = np.stack([rv * np.cos(bv), rv * np.sin(bv)], 1)
    rc = np.minimum(rv, FREE_MAX) - 1.5 * RES        # stop short of the wall
    counts = np.maximum(np.floor(rc / FREE_STEP).astype(np.int64), 0)
    tot = int(counts.sum())
    bidx = np.repeat(np.arange(len(counts)), counts)
    within = np.arange(tot) - np.repeat(np.cumsum(counts) - counts, counts)
    d = (within + 0.5) * FREE_STEP
    free = np.stack([d * np.cos(bv[bidx]), d * np.sin(bv[bidx])], 1)
    return hit, free, int(valid.sum())


def update_grids(grids_flat, HW, W, H_cells, x0, y0, pts_rel, xs, ys, ths, delta):
    """Scatter +delta (hits) or -delta (misses) for pts_rel placed at each
    particle pose; one flat gather-modify-write across all particles."""
    if len(pts_rel) == 0:
        return
    c, s = np.cos(ths)[:, None], np.sin(ths)[:, None]
    px, py = pts_rel[:, 0][None, :], pts_rel[:, 1][None, :]
    wx = px * c - py * s + xs[:, None]              # (N, P)
    wy = px * s + py * c + ys[:, None]
    ix = np.floor((wx - x0) / RES).astype(np.int64)
    iy = np.floor((wy - y0) / RES).astype(np.int64)
    inb = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H_cells)
    base = (np.arange(len(xs), dtype=np.int64) * HW)[:, None]
    fidx = (base + iy * W + ix)[inb]
    v = grids_flat[fidx].astype(np.int16) + delta
    grids_flat[fidx] = np.clip(v, -L_CLAMP, L_CLAMP).astype(np.int8)


def run(path, limit=10 ** 9, seed=SEED):
    rng = np.random.default_rng(seed)
    scans = parse_flaser(path)
    keys = keyframes(scans)[:limit]
    K = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    step = max(1, n_beams // SUB_TARGET)
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    print(f"[{path}] {len(scans)} scans -> {K} keyframes, {n_beams} beams "
          f"(subsample every {step} -> {n_beams // step})")

    # grid bbox from odometry bounds + margin (identical rule on every log)
    x0, y0 = odom[:, 0].min() - MARGIN, odom[:, 1].min() - MARGIN
    x1, y1 = odom[:, 0].max() + MARGIN, odom[:, 1].max() + MARGIN
    W = int(np.ceil((x1 - x0) / RES))
    H = int(np.ceil((y1 - y0) / RES))
    HW = W * H
    grids = np.zeros((N_PART, H, W), np.int8)
    grids_flat = grids.reshape(-1)
    print(f"  grid {W} x {H} cells ({(x1-x0):.0f} x {(y1-y0):.0f} m), "
          f"{N_PART} particles -> {grids.nbytes / 1e6:.1f} MB of grids")

    poses = np.tile(odom[0], (N_PART, 1)).astype(np.float64)
    logw = np.zeros(N_PART)
    traj = np.zeros((N_PART, K, 3), np.float32)
    traj[:, 0] = odom[0]
    pidx4 = np.arange(N_PART)[:, None, None, None]

    hit0, free0, _ = scan_arrays(keys[0][0], beam, step)
    update_grids(grids_flat, HW, W, H, x0, y0, free0,
                 poses[:, 0], poses[:, 1], poses[:, 2], -L_MISS)
    update_grids(grids_flat, HW, W, H, x0, y0, hit0,
                 poses[:, 0], poses[:, 1], poses[:, 2], L_HIT)

    neff_hist = np.empty(K - 1) if K > 1 else np.zeros(0)
    n_resample = n_unmatched = 0
    resample_copies = []
    t0 = time.time()

    for k in range(1, K):
        r, _, _ = keys[k]
        hit_rel, free_rel, n_valid = scan_arrays(r, beam, step)
        B = len(hit_rel)

        # --- motion model: odometry delta + sampled noise (proposal seed)
        op, on = odom[k - 1], odom[k]
        co, so = np.cos(-op[2]), np.sin(-op[2])
        dx, dy = on[0] - op[0], on[1] - op[1]
        dt_local = np.array([co * dx - so * dy, so * dx + co * dy])
        dth = np.arctan2(np.sin(on[2] - op[2]), np.cos(on[2] - op[2]))
        trans, rot = np.linalg.norm(dt_local), abs(dth)
        sxy = A0 + A1 * trans + A2 * rot
        sth = B0 + B1 * rot + B2 * trans
        ndt = dt_local[None, :] + rng.normal(0, sxy, (N_PART, 2))
        ndth = dth + rng.normal(0, sth, N_PART)
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        px = poses[:, 0] + c * ndt[:, 0] - s * ndt[:, 1]
        py = poses[:, 1] + s * ndt[:, 0] + c * ndt[:, 1]
        pth = poses[:, 2] + ndth

        if B >= 10:
            # --- correlative search on each particle's own grid
            thc = pth[:, None] + DTH[None, :]                       # (N,R)
            cc, ss = np.cos(thc)[..., None], np.sin(thc)[..., None]
            hx, hy = hit_rel[:, 0][None, None, :], hit_rel[:, 1][None, None, :]
            wx = hx * cc - hy * ss + px[:, None, None]              # (N,R,B)
            wy = hx * ss + hy * cc + py[:, None, None]
            gx = np.floor((wx - x0) / RES).astype(np.int32)
            gy = np.floor((wy - y0) / RES).astype(np.int32)
            gxo = np.clip(gx[..., None] + DXC, 0, W - 1)            # (N,R,B,T)
            gyo = np.clip(gy[..., None] + DYC, 0, H - 1)
            gathered = grids[pidx4, gyo, gxo]                       # int8
            hits = (gathered > 0).sum(2)                            # (N,R,T)
            score = hits - PRIOR_PEN[None]
            flat = score.reshape(N_PART, -1).argmax(1)
            br, bt = flat // T_SEARCH, flat % T_SEARCH
            ar = np.arange(N_PART)
            matched = hits[ar, br, bt] >= max(HIT_MIN_ABS, HIT_MIN_FRAC * B)
            n_unmatched += int((~matched).sum())
            sel_r = np.where(matched, br, R_MID)
            sel_t = np.where(matched, bt, T_MID)
            # --- weight update: endpoint model at the selected pose
            vals = gathered[ar, sel_r, :, sel_t].astype(np.int64)   # (N,B)
            p = PTAB[vals + 128]
            logw += GAIN * np.log(0.2 + 0.8 * p).sum(1)
            # --- apply accepted refinement
            use = matched.astype(np.float64)
            px = px + use * OFFM[bt, 0]
            py = py + use * OFFM[bt, 1]
            pth = pth + use * DTH[br]
        else:
            n_unmatched += N_PART

        poses = np.stack([px, py, pth], 1)
        traj[:, k] = poses

        # --- map update on each particle's own grid, at its own pose
        update_grids(grids_flat, HW, W, H, x0, y0, free_rel, px, py, pth, -L_MISS)
        update_grids(grids_flat, HW, W, H, x0, y0, hit_rel, px, py, pth, L_HIT)

        # --- adaptive resampling on N_eff
        lw = logw - logw.max()
        w = np.exp(lw)
        w /= w.sum()
        neff = 1.0 / (w ** 2).sum()
        neff_hist[k - 1] = neff
        if neff < NEFF_FRAC * N_PART:
            pos = (rng.random() + np.arange(N_PART)) / N_PART
            idx = np.searchsorted(np.cumsum(w), pos)
            idx = np.minimum(idx, N_PART - 1)
            grids = grids[idx].copy()          # honest RBPF cost: grid copies
            grids_flat = grids.reshape(-1)
            traj = traj[idx].copy()
            poses = poses[idx].copy()
            logw[:] = 0.0
            n_resample += 1
            resample_copies.append(N_PART - len(np.unique(idx)))

        if k % 500 == 0:
            print(f"  kf {k}/{K}  t={time.time() - t0:.0f}s  "
                  f"N_eff={neff:.1f}  resamples={n_resample}")

    dt = time.time() - t0
    ms_kf = dt / max(K - 1, 1) * 1e3
    best = int(np.argmax(logw))
    est = traj[best].astype(np.float64)

    stem = os.path.splitext(os.path.basename(path))[0]
    out = f"{stem}_traj_rbpf.npz"
    np.savez(out, est=est, odom=odom, kts=kts)
    print(f"done: {dt:.0f}s ({ms_kf:.0f} ms/kf)   best particle {best}   -> {out}")

    # --- N_eff / resample statistics
    if len(neff_hist):
        print(f"  N_eff: mean {neff_hist.mean():.1f}  min {neff_hist.min():.1f} "
              f"(of {N_PART})   resamples: {n_resample} "
              f"(every {max(K - 1, 1) / max(n_resample, 1):.1f} kf, "
              f"avg {np.mean(resample_copies) if resample_copies else 0:.1f} "
              f"grids copied each)   "
              f"unmatched particle-steps: {n_unmatched}"
              f"/{(K - 1) * N_PART} ({100 * n_unmatched / ((K - 1) * N_PART):.1f}%)")

    # --- honest memory accounting
    grid_bytes = grids.nbytes                       # N x H x W int8, resident
    resample_peak = grid_bytes                      # grids[idx].copy() transient
    search_tmp = N_PART * R_SEARCH * (n_beams // step) * T_SEARCH * (1 + 4 + 4)
    misc = traj.nbytes
    print(f"  memory: grids {grid_bytes / 1e6:.1f} MB resident "
          f"({N_PART} x {HW / 1e6:.2f} Mcell int8)  "
          f"+ {resample_peak / 1e6:.1f} MB transient at resample (grid copies)  "
          f"+ {search_tmp / 1e6:.1f} MB search temporaries  "
          f"+ {misc / 1e6:.1f} MB trajectories; no scans stored  "
          f"=> peak ~{(2 * grid_bytes + search_tmp + 2 * misc) / 1e6:.0f} MB")

    # --- ATE vs corrected reference (0.3 s timestamp gate)
    ate = None
    try:
        ref = parse_flaser(path.replace(".log", ".gfs.log"))
        rts = np.array([t for _, _, t in ref])
        rxy = np.stack([p[:2] for _, p, _ in ref])
        if not np.all(np.isfinite(rts)):
            raise ValueError("non-finite reference timestamps")
        j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
        good = np.abs(rts - kts[j]) < 0.3
        if good.sum() < 10:
            raise ValueError(f"only {good.sum()} timestamp matches")
        aligned = align_se2(est[j[good], :2], rxy[good])
        err = np.linalg.norm(aligned - rxy[good], axis=1)
        ate = (np.sqrt((err ** 2).mean()), np.median(err), err.max(), good.sum())
        print(f"  ATE vs corrected log over {good.sum()} matched poses: "
              f"rmse {ate[0]:.3f} m   median {ate[1]:.3f} m   max {ate[2]:.3f} m")
    except FileNotFoundError:
        print("  no corrected log found for reference -- skipping ATE")
    except (ValueError, IndexError) as e:
        print(f"  reference unusable ({e}) -- skipping ATE")
    return ate, ms_kf


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit, seed = 10 ** 9, SEED
    for a in sys.argv[1:]:
        if a.startswith("--limit"):
            limit = int(a.split("=")[1])
        elif a.startswith("--seed"):
            seed = int(a.split("=")[1])
    paths = args if args else ["data/intel.log"]
    for p in paths:
        run(p, limit, seed)


if __name__ == "__main__":
    main()
