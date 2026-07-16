#!/usr/bin/env python3
"""GOLDEN ON-CHIP TRACKER — the bit-exact integer spec for the ECP5
track engine (rung #43). Frame-to-frame tracking: match the CURRENT
scan's encode accumulators (i32, still in the encoder RAM after VEC
TX) against the PREVIOUS frame's 2-bit QPSK map slot (already written
by the VEC path; live parity 16/16 banked 2026-07-16), integrate the
winning delta into a pose. Everything integer; the RTL must match this
model bit-exactly.

Search (window pinned by the hunter retunes, t0.72/rot9/cap0.30):
  coarse: rho in prho+[-6..6] (13 x 1.5 deg), (dx,dy) in pdelta_clamped
          + 61*[-12..12]^2 U  (61 U = 59.6 mm step, +-0.73 m)
  fine:   winner rho fixed, (dx,dy) winner + 20*[-3..3]^2 U (19.5 mm)
Score = sum over rings of Re(match_int partial) — equal ring weights
(the deployed matcher's convention). Argmax scan order: rho outer,
dy middle, dx inner; STRICT > (first winner sticks) — RTL-pinned.
Guess: previous solved delta, translation clamped to 307 U (0.30 m),
rho clamped to +-6.

Pose integration (i32 U, theta in rho steps mod 240 = full circle):
  the matcher solves D aligning MAP(prev) to QUERY(cur); the robot's
  motion in the prev frame is -D and heading delta is -drho (sign
  pinned by the odom cross-check in `validate`).
  x += (-dx)*cos(th) - (-dy)*sin(th) etc. via the same F_ANG cis
  convention (240-entry heading LUT, i15).

  python3 hw/ecp5/host/golden_track.py validate [capture.npz] [N]
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import hw.ice40.golden as G                                    # noqa: E402

N_ANG = G.N_ANG
COARSE_STEP = 61            # U (59.6 mm)
COARSE_N = 12               # +-12 steps
FINE_STEP = 20              # U (19.5 mm)
FINE_N = 3
RHO_N = 6                   # +-6 x 1.5 deg = +-9 deg
CAP_U = 307                 # cv clamp 0.30 m
F_H = 14                    # heading LUT fractional bits


def heading_lut():
    th = np.arange(240) * np.pi / 120.0          # 1.5 deg steps, full
    return (np.round(np.cos(th) * (1 << F_H)).astype(np.int64),
            np.round(np.sin(th) * (1 << F_H)).astype(np.int64))


HC, HS = heading_lut()


def score_re(Q, mcode, dx, dy, rho, s, luts):
    """Re-only total score (i64) — the RTL computes exactly this."""
    p = G.match_int(Q, mcode, dx, dy, rho, s, luts)
    return int(p[:, 0].astype(np.int64).sum())


def _score_grid(Q, mcode, dxs, dys, rho, s, luts):
    """Vectorized Re scores for a (dx,dy) grid at one rho — element-wise
    IDENTICAL integer ops to score_re (>> on int64 numpy is arithmetic
    floor, same as the scalar path); verified in selftest."""
    n_ang = luts["n_ang"]
    QS = Q.astype(np.int64) >> int(s)
    QR = G.rot_grid_int(QS, rho, n_ang)
    hre, him = QR[:, 0], -QR[:, 1]
    c = np.asarray(mcode, np.int64) & 3
    HRE = np.select([c == 0, c == 1, c == 2], [hre, -him, -hre], him)
    HIM = np.select([c == 0, c == 1, c == 2], [him, hre, -him], -hre)
    ha = 1 << (G.F_ANG - 1)
    u = (dxs[:, None] * luts["ang_c"][None, :].astype(np.int64)
         + dys[:, None] * luts["ang_s"][None, :].astype(np.int64)
         + ha) >> G.F_ANG                       # (ncand, n_ang)
    tot = np.zeros(len(dxs), np.int64)
    for k in range(G.N_RING):
        addr = (u >> k) & 255
        cre = luts["cis_re"][addr].astype(np.int64)
        cim = luts["cis_im"][addr].astype(np.int64)
        h_re = HRE[k * n_ang:(k + 1) * n_ang]
        h_im = HIM[k * n_ang:(k + 1) * n_ang]
        tot += (h_re[None, :] * cre - h_im[None, :] * cim).sum(1)
    return tot


def track_step(Q, mc_prev, pdx, pdy, prho, luts):
    """One frame: -> (dx, dy, rho, score, s). Integer, deterministic.
    Argmax order: rho outer, dy, dx inner; strict > (first winner)."""
    s = G.shift_for(Q)
    cx = max(-CAP_U, min(CAP_U, int(pdx)))
    cy = max(-CAP_U, min(CAP_U, int(pdy)))
    cr = max(-RHO_N, min(RHO_N, int(prho)))
    off = np.arange(-COARSE_N, COARSE_N + 1) * COARSE_STEP
    DX = cx + np.tile(off, len(off))            # dx inner
    DY = cy + np.repeat(off, len(off))          # dy middle
    best = None
    for rho in range(cr - RHO_N, cr + RHO_N + 1):
        sc = _score_grid(Q, mc_prev, DX, DY, rho, s, luts)
        i = int(np.argmax(sc))                  # first max = scan order
        if best is None or int(sc[i]) > best[0]:
            best = (int(sc[i]), int(DX[i]), int(DY[i]), rho)
    _, bx, by, brho = best
    offf = np.arange(-FINE_N, FINE_N + 1) * FINE_STEP
    FX = bx + np.tile(offf, len(offf))
    FY = by + np.repeat(offf, len(offf))
    sc = _score_grid(Q, mc_prev, FX, FY, brho, s, luts)
    i = int(np.argmax(sc))
    if int(sc[i]) > best[0]:
        best = (int(sc[i]), int(FX[i]), int(FY[i]), brho)
    sc_, dx, dy, rho = best
    return dx, dy, rho, sc_, s


F_SUB = 6                   # theta stored in rho/64 units
TH_RE_U = 512               # re-anchor when |delta t| > 0.5 m
TH_RE_R = 5                 # ... or |delta rho| >= 5 steps (7.5 deg)


def subgrid_rho(sm1, s0, sp1):
    """Parabolic sub-grid rotation at re-anchor, in rho/64 units
    (integer): vertex = (sm1-sp1) / (2*(sm1-2*s0+sp1)), clamped +-1/2
    step. Integer division with round-to-nearest (RTL-exact)."""
    den = 2 * (sm1 - 2 * s0 + sp1)
    if den >= 0:
        return 0
    num = (sm1 - sp1) << F_SUB
    b = -den                       # > 0
    # vertex*64 = num/den = -num/b; round-to-nearest, RTL-exact
    off = -((num + b // 2) // b) if num >= 0 \
        else ((-num + b // 2) // b)
    return max(-32, min(32, off))


def _world_add(pose, dx, dy, drho64):
    """pose (x_U, y_U, th64 mod 240*64) += motion(-delta) in world."""
    x, y, th64 = pose
    mx, my = -dx, -dy
    c, s = HC[(th64 >> F_SUB) % 240], HS[(th64 >> F_SUB) % 240]
    x += (mx * c - my * s) >> F_H
    y += (mx * s + my * c) >> F_H
    # SIGN LAW (pin_signs, ground-truth motion test): motion = -D,
    # heading delta = +rho. (-rho here cost 13 m ATE on spot: clean
    # local RPE, globally curled trail — banked.)
    return (int(x), int(y), int((th64 + drho64) % (240 << F_SUB)))


def track(mm_all, n, luts, progress=200):
    """REFERENCE-FRAME tracker (v2): match each frame against a held
    ANCHOR slot; re-anchor when the delta exceeds TH_RE (with sub-grid
    rotation) — the pose is re-solved against the same reference every
    frame, so drift accrues per re-anchor (~0.5 m), not per frame.
    v1 frame-to-frame was REFUTED on capture3: at 10 Hz the per-frame
    rotation (~0.5 deg med) sits under the 1.5 deg grid -> heading
    deadband (dyaw90 0.0), pair-cos 0.911 < raw odom."""
    out = np.zeros((n, 3))
    ref_pose = (0, 0, 0)
    ref_mc = None
    pdx = pdy = prho = 0
    for k in range(n):
        mm = mm_all[k]
        keep = (mm > 0) & (mm <= G.R_MASK_MM)
        az = np.flatnonzero(keep).astype(np.int32)
        r_mm = mm[keep].astype(np.int32)
        # ARC-MASS weights (scan_to_ints w_unit=False — the python
        # tracker's emphasis; far points carry rotation observability;
        # chip cost = a range-based weight LUT in the feeder)
        w = np.maximum(np.minimum((r_mm.astype(np.int64) * 6434) >> 21,
                                  127), 1).astype(np.int32)
        Q = G.encode_int(az, r_mm, w, luts)
        if ref_mc is None:
            ref_mc = G.mcode_from_vec(Q[:, 0] + 1j * Q[:, 1])
            pose = ref_pose
        else:
            dx, dy, rho, sc, s = track_step(Q, ref_mc, pdx, pdy, prho,
                                            luts)
            pdx, pdy, prho = dx, dy, rho
            # sub-grid rotation EVERY frame (the reported pose): a
            # +-0.75 deg heading quantum costs 13-40 cm at range
            sm1 = score_re(Q, ref_mc, dx, dy, rho - 1, s, luts)
            sp1 = score_re(Q, ref_mc, dx, dy, rho + 1, s, luts)
            sub = subgrid_rho(sm1, sc, sp1)
            # UNIT LAW: rot_grid_int rho = 3-deg steps (pi/60); th64 is
            # in 1.5deg/64 units (the heading LUT grid) -> rho counts
            # DOUBLE. The missing <<1 integrated heading at HALF rate:
            # spot ATE 9-13 m, turns curled — THE dominant tracker bug.
            drho64 = (rho << (F_SUB + 1)) + 2 * sub
            pose = _world_add(ref_pose, dx, dy, drho64)
            if (abs(dx) > TH_RE_U or abs(dy) > TH_RE_U
                    or abs(rho) >= TH_RE_R):
                ref_pose = _world_add(ref_pose, dx, dy, drho64)
                ref_mc = G.mcode_from_vec(Q[:, 0] + 1j * Q[:, 1])
                pdx = pdy = prho = 0
                pose = ref_pose
        x, y, th64 = pose
        out[k] = (x * 250.0 / 256.0 / 1000.0,
                  y * 250.0 / 256.0 / 1000.0,
                  th64 * np.pi / (120 << F_SUB))
        if progress and k and k % progress == 0:
            print(f"  kf {k}/{n}", flush=True)
    return out


def validate(cap, n=600):
    import runners.datasets  # noqa: F401  (repo-root path check)
    z = np.load(cap, allow_pickle=True)
    mm_all, odom = z["mm"], np.asarray(z["est"], float)
    luts = G.make_luts()
    est = track(mm_all, n, luts)
    # sign pin + quality vs odom (short-window RPE) and pair-cos
    sys.path.insert(0, str(ROOT / "scratch"))
    import scratch_hunter_rpe as H
    H.CAP = str(cap)
    pts_w, vecs, _ = H.load()
    cs = H.pair_cos(est, pts_w[:n], vecs[:n])
    e1t, e1y = H.rpe(est, odom[:n], 10)
    dy = np.rad2deg(np.abs(np.diff(est[:, 2])))
    dy = np.minimum(dy, 360 - dy)
    print(f"GOLDEN CHIP TRACKER on {Path(cap).name} ({n} kf @ 10 Hz):")
    print(f"  pair-cos med {np.median(cs):.3f} p10 "
          f"{np.percentile(cs, 10):.3f}  (python nph=4 tracker: "
          f"0.956/0.892; raw odom: 0.950/0.506)")
    print(f"  RPE1s vs odom {np.median(e1t)*100:.1f} cm / "
          f"{np.median(e1y):.2f} deg | dyaw90 "
          f"{np.percentile(dy, 90):.1f} deg/kf")


def pin_signs():
    """Ground-truth motion test: scan the same world from pose A
    (origin) and pose B (known SE(2)); the tracker's reported motion
    must equal B. Prints the solved delta and what each convention
    yields — the one matching B is the LAW for _world_add."""
    luts = G.make_luts()
    rng = np.random.default_rng(4)
    a = np.arange(G.N_BEAM) * (2 * np.pi / G.N_BEAM) - np.pi
    r = np.minimum(3.5 / np.maximum(np.abs(np.cos(a)), 1e-9),
                   2.5 / np.maximum(np.abs(np.sin(a)), 1e-9))
    r += rng.normal(0, 0.01, G.N_BEAM)
    pw = np.stack([r * np.cos(a), r * np.sin(a)], 1)   # world pts (A=I)

    def scan_from(pose):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        d = pw - pose[:2]
        pl = d @ np.array([[c, s], [-s, c]]).T          # world->body
        az = np.round((np.arctan2(pl[:, 1], pl[:, 0]) + np.pi)
                      / (2 * np.pi / G.N_BEAM)).astype(int) % G.N_BEAM
        rm = np.round(np.hypot(pl[:, 0], pl[:, 1]) * 1000).astype(int)
        keep = rm <= G.R_MASK_MM
        return az[keep].astype(np.int32), rm[keep].astype(np.int32)

    azA, rA = scan_from(np.array([0.0, 0.0, 0.0]))
    QA = G.encode_int(azA, rA, np.full(len(azA), 127, np.int32), luts)
    mcA = G.mcode_from_vec(QA[:, 0] + 1j * QA[:, 1])
    B = np.array([0.30, -0.20, np.deg2rad(4.5)])
    azB, rB = scan_from(B)
    QB = G.encode_int(azB, rB, np.full(len(azB), 127, np.int32), luts)
    dx, dy, rho, sc, s = track_step(QB, mcA, 0, 0, 0, luts)
    u2m = 250.0 / 256.0 / 1000.0
    print(f"true B = ({B[0]:.3f}, {B[1]:.3f}, {np.rad2deg(B[2]):.1f} "
          f"deg); solved delta dx {dx} dy {dy} U, rho {rho} "
          f"(= {dx*u2m:+.3f}, {dy*u2m:+.3f} m, {rho*1.5:+.1f} deg)")
    for st_, sr in ((+1, +1), (+1, -1), (-1, +1), (-1, -1)):
        # candidate law: motion = st*(dx,dy) rotated by heading; dth=sr*rho
        mx, my = st_ * dx * u2m, st_ * dy * u2m
        print(f"  motion=({'+' if st_>0 else '-'}D, "
              f"dth={'+' if sr>0 else '-'}rho): "
              f"({mx:+.3f}, {my:+.3f}, {sr*rho*1.5:+.1f} deg)"
              + ("   <-- MATCHES B" if
                 abs(mx-B[0]) < 0.05 and abs(my-B[1]) < 0.05
                 and abs(sr*rho*1.5 - np.rad2deg(B[2])) < 1.6 else ""))


def selftest_grid():
    """Vectorized grid == scalar match_int path, bit-exact."""
    luts = G.make_luts()
    rng = np.random.default_rng(2)
    Q = rng.integers(-(1 << 22), 1 << 22, (240, 2)).astype(np.int32)
    mc = rng.integers(0, 4, 240)
    s = G.shift_for(Q)
    dxs = rng.integers(-700, 700, 40)
    dys = rng.integers(-700, 700, 40)
    for rho in (-6, 0, 5):
        v = _score_grid(Q, mc, dxs, dys, rho, s, luts)
        for i in range(40):
            assert v[i] == score_re(Q, mc, int(dxs[i]), int(dys[i]),
                                    rho, s, luts), (i, rho)
    print("grid==scalar bit-exact (120 candidates x 3 rho)")


def validate_spot(n=413):
    """The previous-results venue: spot classroom vs WITHHELD odometry.
    Frame-to-frame chip tracker (NO segments/loop closures — drift
    accumulates by design; the banked 0.039 is the FULL BoundedSLAM).
    Honest comparables: same-data python nph=4 frame tracker + RPE."""
    import runners.spot as SP
    b = SP.make_bundle()
    keys = b["keys"]
    n = min(n, len(keys))
    mm_all = np.stack([np.where(
        np.isfinite(np.asarray(k[0], float)),
        np.clip(np.round(np.asarray(k[0], float) * 1e3), 0, 65535),
        0).astype(np.uint16) for k in keys[:n]])
    gt = np.stack([np.asarray(k[1], float)[:3] for k in keys[:n]])
    luts = G.make_luts()
    est = track(mm_all, n, luts)
    # align start (gt frame anchor) for display-honest ATE + RPE
    est = est + (gt[0] - est[0])
    def rel(a, b):
        d = b[:2] - a[:2]
        c, s = np.cos(-a[2]), np.sin(-a[2])
        return np.array([c*d[0]-s*d[1], s*d[0]+c*d[1],
                         np.arctan2(np.sin(b[2]-a[2]),
                                    np.cos(b[2]-a[2]))])
    e1 = [np.linalg.norm(rel(est[k], est[k+5])[:2]
                         - rel(gt[k], gt[k+5])[:2])
          for k in range(0, n - 5, 2)]
    ate = np.linalg.norm(est[:, :2] - gt[:, :2], axis=1)
    print(f"GOLDEN CHIP TRACKER on spot ({n} kf @ 5 Hz, vs withheld "
          f"odometry):")
    print(f"  RPE(1s/5kf) med {np.median(e1)*100:.1f} cm | ATE med "
          f"{np.median(ate):.3f} p90 {np.percentile(ate, 90):.3f} m "
          f"(frame-to-frame, no loop closure; banked FULL SLAM: 0.039)")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if what == "selftest":
        selftest_grid()
    elif what == "spot":
        validate_spot()
    else:
        cap = sys.argv[2] if len(sys.argv) > 2 else \
            ("/private/tmp/claude-504/-Users-kamp-code-ssp/"
             "3b401c4b-a371-4a64-a26d-b61589c02a38/scratchpad/"
             "capture3.npz")
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 600
        validate(cap, n)
