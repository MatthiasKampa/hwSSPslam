"""Scan-Context (Kim & Kim, IROS 2018) as a RICHER CLASSICAL control against
the coarse SSP relocalization rings on the MIT corridor limit.

The SSP drought relocalizer scores 0/106 stage-1 hits at deep MIT true
revisits (RESULTS.md R3/R4): place-recognition information is ABSENT in the
coarse SSP band (lam 5.3/12.8) — corridor self-similarity washes out the
door/alcove-scale detail. The open question that verdict leaves: is the
information GONE FROM THE RAW SCAN (environment/sensor limit), or merely lost
by the *coarse SSP summary* (representation limit)?

This module answers it with the standard rotation-invariant 2D place
descriptor, built from each keyframe's RAW scan (not the SSP encoding):

  * Scan-Context matrix: a polar bird's-eye grid, n_ring x n_sector, each bin
    holding the point density (number of returns) that fall in it.
  * RING KEY: the per-ring row statistic (mean occupancy over sectors) — a
    n_ring vector that is invariant to a yaw rotation (which is a circular
    column shift of the matrix). This is the coarse-retrieval shortlister,
    directly comparable to the coarse SSP ring.
  * Column-shift alignment: the fine SC distance = min over all n_sector yaw
    shifts of the mean column cosine distance (the standard SC scoring).

We reuse the CARMEN parsing / keyframing from ssp_slam_carmen (import; no edit
to any SSP module). We evaluate retrieval recall@k on the SAME true-revisit
definition the drought relocalizer was scored on:

  true revisit  = a keyframe within TOL_REV m (REFERENCE frame) of a keyframe
                  >= GAP keyframes older.
  correct match = any such older keyframe within TOL_REV m in the REF frame.
  recall@k      = fraction of true-revisit queries whose top-k ring-key
                  shortlist (over all >= GAP-older keyframes) contains a
                  correct match.

REF frame:
  * MIT: gfs timestamps are corrupt upstream, so we use range-array identity
    (each gfs scan's range vector is bit-identical to a raw scan's) to recover
    a valid timestamp per REF pose, then interpolate a REF position onto every
    keyframe. Exactly the mit_hy4 / r4_recall / mit_capacity eval convention.
  * Intel: gfs timestamps share the raw base, matched directly.

Usage:
  python3 ssp_scancontext.py            # MIT + Intel recall tables + verdict
  python3 ssp_scancontext.py mit
  python3 ssp_scancontext.py intel
  python3 ssp_scancontext.py selftest
"""

import sys
import time

import numpy as np

import ssp_slam_carmen as C  # parse_flaser, keyframes (reused, not modified)

# ---- Scan-Context parameters (Kim & Kim defaults adapted to planar lidar) ----
N_RING = 20
N_SECTOR = 60

# ---- evaluation parameters (identical to the drought recall eval) ----
TOL_REV = 5.0     # m, revisit / correct-match radius in the REF frame
GAP = 1500        # keyframes; a revisit must be against a >= GAP-older keyframe
KS = (1, 5, 20, 40)

# per-dataset sensor / grid extent
CFG = {
    "mit":   dict(log="data/mit.log",   valid_max=50.0, max_r=50.0),  # sentinel ~51.1 m
    "intel": dict(log="data/intel.log", valid_max=40.0, max_r=40.0),  # 28 m building
}


# ------------------------------------------------------------------ descriptor
def scan_context(ranges, beam, valid_max, max_r, n_ring=N_RING, n_sector=N_SECTOR):
    """Point-density Scan-Context matrix (n_ring x n_sector) from one raw scan.

    Sensor (body) frame: rotation invariance is handled by the ring key and the
    column-shift alignment, so no pose is used. Bin value = number of returns.
    """
    r = np.asarray(ranges, float)
    ok = np.isfinite(r) & (r > 0.0) & (r < valid_max) & (r < max_r)
    if not ok.any():
        return np.zeros((n_ring, n_sector), float)
    rr = r[ok]
    ang = beam[ok]  # bearing in the sensor frame
    ring = np.clip((rr / max_r * n_ring).astype(int), 0, n_ring - 1)
    # sector over full 2*pi so any yaw maps to a pure column shift
    sec = np.clip(((ang % (2 * np.pi)) / (2 * np.pi) * n_sector).astype(int),
                  0, n_sector - 1)
    grid = np.zeros((n_ring, n_sector), float)
    np.add.at(grid, (ring, sec), 1.0)
    return grid


def ring_key(grid):
    """Per-ring row statistic — invariant to sector (column) order, hence to
    yaw. Mean point density per ring (Kim's ring-key form)."""
    return grid.mean(axis=1)


def _unit_columns(grid):
    n = np.linalg.norm(grid, axis=0)
    u = np.zeros_like(grid)
    nz = n > 0
    u[:, nz] = grid[:, nz] / n[nz]
    return u, nz


def sc_distance(g1, g2):
    """Scan-Context distance: min over all n_sector yaw (column) shifts of the
    mean per-column cosine distance, counting only columns non-empty in both.
    Vectorized: cosine(col_i(g1), col_j(g2)) = (U1^T U2)[i,j]; the shift-s score
    is the wrapped diagonal sum. Returns (dist in [0,1], best_shift)."""
    u1, nz1 = _unit_columns(g1)
    u2, nz2 = _unit_columns(g2)
    n_sec = g1.shape[1]
    Cm = u1.T @ u2                      # (n_sec, n_sec) column cosines
    both = np.outer(nz1, nz2).astype(float)
    idx = np.arange(n_sec)
    best, best_s = 2.0, 0
    for s in range(n_sec):
        j = (idx + s) % n_sec
        v = both[idx, j].sum()
        if v <= 0:
            continue
        cos = Cm[idx, j].sum() / v
        d = 1.0 - cos
        if d < best:
            best, best_s = d, s
    return best, best_s


# --------------------------------------------------------------- REF positions
def ref_positions(name, keys):
    """Per-keyframe REFERENCE (ground-truth-corrected) position, interpolated
    from the gfs poses. Returns (kts, ref_xy, valid_mask)."""
    cfg = CFG[name]
    kts = np.array([t for _, _, t in keys])
    gfs = C.parse_flaser(cfg["log"].replace(".log", ".gfs.log"))
    gxy = np.stack([p[:2] for _, p, _ in gfs])
    if name == "mit":
        # gfs timestamps corrupt -> recover a valid timestamp per gfs pose by
        # range-array identity against the raw log (mit_hy4 / r4_recall trick).
        raw = C.parse_flaser(cfg["log"])
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
        matched = len(gts) / len(gfs)
        print(f"  [mit] range-identity matched {len(gts)}/{len(gfs)} gfs poses "
              f"({100 * matched:.0f}%)")
    else:
        gts = np.array([t for _, _, t in gfs])  # gfs timestamps share the base

    o = np.argsort(gts)
    gts, gxy = gts[o], gxy[o]
    # dedupe identical timestamps for np.interp
    uniq = np.concatenate([[True], np.diff(gts) > 0])
    gts, gxy = gts[uniq], gxy[uniq]
    rx = np.interp(kts, gts, gxy[:, 0])
    ry = np.interp(kts, gts, gxy[:, 1])
    ref = np.stack([rx, ry], 1)
    valid = (kts >= gts[0]) & (kts <= gts[-1])  # inside the REF span
    return kts, ref, valid


# ------------------------------------------------------------------ evaluation
def evaluate(name):
    cfg = CFG[name]
    t0 = time.time()
    scans = C.parse_flaser(cfg["log"])
    keys = C.keyframes(scans)
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(np.arange(n_beams) - 90.0)  # SICK: 1 deg, -90..+89
    print(f"\n=== {name.upper()}: {len(scans)} scans -> {n} keyframes, "
          f"{n_beams} beams, R_max={cfg['max_r']:.0f} m ===")

    # descriptors
    grids = np.zeros((n, N_RING, N_SECTOR), float)
    for k, (r, _, _) in enumerate(keys):
        grids[k] = scan_context(r, beam, cfg["valid_max"], cfg["max_r"])
    rkeys = grids.mean(axis=2)  # (n, N_RING) ring keys
    print(f"  built {n} Scan-Contexts ({N_RING}x{N_SECTOR}) + ring keys "
          f"in {time.time() - t0:.0f}s")

    # reference positions
    kts, ref, valid = ref_positions(name, keys)

    # true-revisit set + correct matches (REF frame), over labelled keyframes
    idx_all = np.arange(n)
    queries = []          # (query_kf, correct_older_kfs)
    for k in range(GAP, n):
        if not valid[k]:
            continue
        older = idx_all[:k - GAP + 1]
        older = older[valid[older]]
        if older.size == 0:
            continue
        d = np.linalg.norm(ref[older] - ref[k], axis=1)
        corr = older[d <= TOL_REV]
        if corr.size:
            queries.append((k, corr))
    n_rev = len(queries)
    print(f"  true revisits (>= {GAP} kf older, within {TOL_REV:.0f} m REF): "
          f"{n_rev}")
    if n_rev == 0:
        return name, n_rev, {kk: float("nan") for kk in KS}, {kk: float("nan") for kk in KS}

    # ring-key retrieval recall@k, and SC-aligned re-rank recall@k.
    # Store per-query hit booleans so we can stratify (e.g. the deep final
    # third = the corridor drought the SSP capacity study scored 0/25 on).
    RERANK = 100  # ring-key shortlist size fed to the SC column-shift scorer
    qk = np.array([k for k, _ in queries])
    rk_hit = {kk: np.zeros(n_rev, bool) for kk in KS}
    sc_hit = {kk: np.zeros(n_rev, bool) for kk in KS}
    rand = {kk: np.zeros(n_rev) for kk in KS}  # chance recall@k expectation
    for qi, (k, corr) in enumerate(queries):
        db = idx_all[:k - GAP + 1]           # all >= GAP-older keyframes
        frac = len(corr) / len(db)           # correct fraction of the database
        for kk in KS:
            rand[kk][qi] = 1.0 - (1.0 - frac) ** kk
        # (correctness is only defined for labelled db entries; unlabelled ones
        #  act as honest distractors and are never counted as hits)
        corr_set = set(int(c) for c in corr)
        # ---- coarse: ring-key L2 nearest ----
        dd = np.linalg.norm(rkeys[db] - rkeys[k], axis=1)
        order = db[np.argsort(dd, kind="stable")]
        for kk in KS:
            rk_hit[kk][qi] = bool(corr_set & set(int(x) for x in order[:kk]))
        # ---- fine: SC column-shift re-rank of the ring-key shortlist ----
        shortlist = order[:RERANK]
        sc = np.array([sc_distance(grids[k], grids[j])[0] for j in shortlist])
        sc_order = shortlist[np.argsort(sc, kind="stable")]
        for kk in KS:
            sc_hit[kk][qi] = bool(corr_set & set(int(x) for x in sc_order[:kk]))

    rk = {kk: rk_hit[kk].mean() for kk in KS}
    sc = {kk: sc_hit[kk].mean() for kk in KS}
    print(f"  ring-key recall@k : " +
          "  ".join(f"@{kk}={rk[kk]:.3f}" for kk in KS))
    print(f"  SC re-rank recall@k: " +
          "  ".join(f"@{kk}={sc[kk]:.3f}" for kk in KS)
          + f"   (shortlist {RERANK})")
    print(f"  random-chance recall: " +
          "  ".join(f"@{kk}={rand[kk].mean():.3f}" for kk in KS))
    # stratify by run third (final third = the deep corridor drought region)
    print("  by run-third (ring-key):")
    for lo, hi, lab in ((0, n / 3, "early"), (n / 3, 2 * n / 3, "mid"),
                        (2 * n / 3, n, "late/deep-corridor")):
        sel = (qk >= lo) & (qk < hi)
        if sel.sum() == 0:
            continue
        print(f"    {lab:20} ({sel.sum():4d} revisits): " +
              "  ".join(f"@{kk}={rk_hit[kk][sel].mean():.3f}" for kk in KS))
    return name, n_rev, rk, sc


# ---------------------------------------------------------------------- report
def report(results):
    print("\n" + "=" * 70)
    print("RECALL@k  (ring-key coarse retrieval | SC-aligned re-rank)")
    print("=" * 70)
    hdr = f"{'dataset':7} {'revisits':>8}  " + "  ".join(f"@{kk:<5}" for kk in KS)
    print(hdr)
    for name, n_rev, rk, sc in results:
        row = f"{name:7} {n_rev:8d}  " + "  ".join(
            f"{rk[kk]:.3f}" for kk in KS)
        print("ring-key " + row)
        row = f"{'':7} {'':8}  " + "  ".join(f"{sc[kk]:.3f}" for kk in KS)
        print("SC-rerank " + row)
    print("(SSP coarse relocalization ring at MIT true revisits: 0/106 = 0.000)")


def selftest():
    print("selftest: Scan-Context invariants")
    rng = np.random.default_rng(0)
    n_beams = 180
    beam = np.deg2rad(np.arange(n_beams) - 90.0)
    r = rng.uniform(1.0, 30.0, n_beams)
    g = scan_context(r, beam, 50.0, 50.0)
    # ring key invariant to sector permutation
    perm = rng.permutation(N_SECTOR)
    assert np.allclose(ring_key(g), ring_key(g[:, perm])), "ring-key not order-invariant"
    # SC distance ~0 to a yaw-rotated copy (pure column shift), min at that shift
    shift = 17
    g2 = np.roll(g, shift, axis=1)
    d, s = sc_distance(g, g2)
    assert d < 1e-9, f"SC distance to yaw-shifted self should be ~0, got {d}"
    assert s == shift, f"best shift {s} != {shift}"
    # self distance zero
    d0, _ = sc_distance(g, g)
    assert d0 < 1e-9
    # distance to unrelated scan is larger
    r3 = rng.uniform(1.0, 30.0, n_beams)
    g3 = scan_context(r3, beam, 50.0, 50.0)
    d3, _ = sc_distance(g, g3)
    assert d3 > d, "unrelated scan should be farther than yaw-rotated self"
    print("  ring-key order-invariance OK; yaw-shift SC dist ~0 at correct shift OK;"
          f" unrelated dist {d3:.3f} > 0 OK")
    print("selftest PASSED")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "selftest":
        selftest()
        return
    names = [arg] if arg in CFG else ["mit", "intel"]
    results = [evaluate(nm) for nm in names]
    report(results)


if __name__ == "__main__":
    main()
