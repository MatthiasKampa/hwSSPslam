"""SSP SLAM: 2D lidar SLAM with Spatial Semantic Pointers (random Fourier features).

Pipeline
--------
1. Synthetic 2D world made of wall segments; robot follows a smooth loop.
2. Simulated 2D lidar: raycasting + range noise, angular jitter, dropout.
3. Each scan is encoded as an SSP: hits are connected into line segments,
   sampled uniformly in *space* (not per-beam), and each sample point p is
   encoded as phi(p) = exp(i W p) with random frequencies W ~ N(0, 1/ell^2).
   Uniform spatial sampling along segments removes the lidar's bias toward
   close surfaces (which receive more beams per meter).
4. Occlusion filter: a segment between neighboring hits is rejected when the
   range difference exceeds twice the tangential (perpendicular-to-ray) gap,
   i.e. the segment points within ~26.6 deg of the viewing ray. This kills
   phantom walls drawn across depth discontinuities (near edge -> far wall).
   Note: the naive form |dr| > 2*|dp| can never trigger (reverse triangle
   inequality gives |dr| <= |dp|), hence the tangential formulation.
5. Registration: translation by vector d multiplies an SSP elementwise by
   exp(i W d), so the correlation of scan and map over candidate poses is
   score(theta, d) = Re sum_k conj(M_k) * S_theta_k * exp(i w_k . d),
   a random-feature approximation of the spatial cross-correlation of the two
   occupancy densities. Coarse-to-fine grid search over theta and d around a
   constant-velocity guess, with parabolic sub-cell refinement.
6. The aligned scan is bundled (added) into the accumulated map SSP M.
"""

import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(7)

# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------

def make_environment():
    """World as an array of wall segments, shape (S, 2, 2)."""
    segs = []

    def rect(x0, y0, x1, y1):
        segs.extend([
            [(x0, y0), (x1, y0)], [(x1, y0), (x1, y1)],
            [(x1, y1), (x0, y1)], [(x0, y1), (x0, y0)],
        ])

    rect(0, 0, 16, 10)                    # outer room
    rect(6.8, 4.2, 9.2, 5.8)              # central box (main occluder)
    rect(3.0, 2.2, 3.8, 3.0)              # pillar
    segs.append([(0.0, 7.5), (6.0, 7.5)])   # interior wall ...
    segs.append([(7.5, 7.5), (12.0, 7.5)])  # ... with a doorway at x=6..7.5
    segs.append([(12.0, 7.5), (12.0, 9.0)]) # stub into top-right area
    segs.append([(13.0, 1.0), (15.0, 3.0)]) # diagonal corner cut
    return np.array(segs, dtype=float)


# ----------------------------------------------------------------------------
# Lidar simulation
# ----------------------------------------------------------------------------

N_BEAMS = 240
MAX_RANGE = 12.0
RANGE_NOISE = 0.02       # sigma [m]
ANGLE_JITTER = np.deg2rad(0.05)
DROPOUT = 0.02


def raycast(segs, origin, angles):
    """Min-range ray/segment intersection for all beams. Returns ranges (inf = miss)."""
    d = np.stack([np.cos(angles), np.sin(angles)], axis=1)      # B x 2
    a = segs[:, 0]                                              # S x 2
    e = segs[:, 1] - segs[:, 0]                                 # S x 2
    ao = a - origin                                             # S x 2

    denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]  # B x S
    cross_ao_e = ao[:, 0] * e[:, 1] - ao[:, 1] * e[:, 0]                   # S
    cross_ao_d = ao[None, :, 0] * d[:, None, 1] - ao[None, :, 1] * d[:, None, 0]

    with np.errstate(divide="ignore", invalid="ignore"):
        t = cross_ao_e[None, :] / denom
        u = cross_ao_d / denom
    valid = (np.abs(denom) > 1e-12) & (t > 1e-9) & (u >= 0.0) & (u <= 1.0)
    t = np.where(valid, t, np.inf)
    return t.min(axis=1)


def simulate_scan(segs, pose):
    """Noisy scan from pose (x, y, heading). Returns (ranges, nominal beam angles),
    ranges = inf for misses/dropouts. Angles are in the robot frame."""
    beam = np.linspace(0, 2 * np.pi, N_BEAMS, endpoint=False)
    true_angles = pose[2] + beam + RNG.normal(0, ANGLE_JITTER, N_BEAMS)
    r = raycast(segs, pose[:2], true_angles)
    r = np.where(r <= MAX_RANGE, r + RNG.normal(0, RANGE_NOISE, N_BEAMS), np.inf)
    r[RNG.random(N_BEAMS) < DROPOUT] = np.inf
    return r, beam


# ----------------------------------------------------------------------------
# Scan -> weighted sample points (segment construction + occlusion filter)
# ----------------------------------------------------------------------------

SAMPLE_DS = 0.12         # spatial sampling step along segments [m]
MAX_CHORD = 1.5          # never bridge gaps larger than this [m]
OCC_RATIO = 2.0          # reject if |dr| > OCC_RATIO * tangential gap


def scan_to_samples(ranges, beam_angles):
    """Connect neighboring hits into segments, filter occlusion streaks, and
    sample the surviving polyline uniformly in space.

    Returns (points Nx2 in robot frame, weights N, rejected segments Rx2x2)."""
    hit = np.isfinite(ranges)
    idx = np.flatnonzero(hit)
    if len(idx) < 2:
        return np.zeros((0, 2)), np.zeros(0), np.zeros((0, 2, 2))
    pts = np.stack([ranges[idx] * np.cos(beam_angles[idx]),
                    ranges[idx] * np.sin(beam_angles[idx])], axis=1)
    r = ranges[idx]

    # candidate segments between consecutive beams (allow 1 dropped beam between)
    consec = np.diff(idx) <= 2
    p0, p1 = pts[:-1], pts[1:]
    chord_v = p1 - p0
    chord = np.linalg.norm(chord_v, axis=1)
    dr = np.abs(np.diff(r))
    # tangential gap = component of the chord perpendicular to the line of sight
    tang = np.sqrt(np.maximum(chord**2 - dr**2, 0.0))
    keep = consec & (chord <= MAX_CHORD) & (dr <= OCC_RATIO * tang)
    rej = consec & (chord <= MAX_CHORD) & ~keep
    rejected = np.stack([p0[rej], p1[rej]], axis=1) if rej.any() else np.zeros((0, 2, 2))

    sample_pts, sample_w = [], []
    used = np.zeros(len(idx), dtype=bool)
    for i in np.flatnonzero(keep):
        n = max(2, int(np.ceil(chord[i] / SAMPLE_DS)))
        ts = (np.arange(n) + 0.5) / n   # midpoint rule: no shared-vertex double mass
        sample_pts.append(p0[i] + ts[:, None] * chord_v[i])
        sample_w.append(np.full(n, chord[i] / n))
        used[i] = used[i + 1] = True
    # isolated hits (e.g. both adjacent segments rejected) still mark a surface
    lone = ~used
    if lone.any():
        sample_pts.append(pts[lone])
        sample_w.append(np.full(int(lone.sum()), SAMPLE_DS))
    return np.concatenate(sample_pts), np.concatenate(sample_w), rejected


# ----------------------------------------------------------------------------
# SSP encoder (random Fourier features)
# ----------------------------------------------------------------------------

class SSPEncoder:
    def __init__(self, dim=1024, length_scale=0.25):
        self.W = RNG.normal(0.0, 1.0 / length_scale, (dim, 2))  # frequencies

    def encode(self, pts, weights):
        """Weighted bundle of point encodings: sum_j w_j exp(i W p_j) -> (D,) complex."""
        return np.exp(1j * (pts @ self.W.T)).T @ weights

    def shift(self, d):
        """Fourier-domain translation operator: phi(p + d) = shift(d) * phi(p)."""
        return np.exp(1j * (self.W @ np.asarray(d)))

    def density(self, ssp, xs, ys):
        """Decode: similarity of phi(x) with the bundle on a grid (len(ys), len(xs))."""
        gx, gy = np.meshgrid(xs, ys)
        g = np.stack([gx.ravel(), gy.ravel()], axis=1)
        val = np.exp(1j * (g @ self.W.T)) @ np.conj(ssp)
        return val.real.reshape(len(ys), len(xs)) / self.W.shape[0]


class HexSSPEncoder(SSPEncoder):
    """Hexagonal SSP: grid-cell-style frequency modules instead of random draws.

    Each scale module contributes 3 unit vectors 120 deg apart (their negatives
    are the complex conjugates, so 3 suffice for a real kernel). Wavelengths
    follow a fixed geometric progression lam_min * ratio^k up to lam_max, so a
    handful of modules covers coarse alignment through fine registration.
    Module orientation is FIXED (rot_step=0, all modules aligned); rot_step>0
    deterministically fans module orientations across the 120 deg sector."""

    def __init__(self, lam_min=0.25, lam_max=8.0, ratio=1.05, rot_step=0.0):
        n = int(np.floor(np.log(lam_max / lam_min) / np.log(ratio))) + 1
        lams = lam_min * ratio ** np.arange(n)
        rows = []
        for k, lam in enumerate(lams):
            ang = k * rot_step + np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
            rows.append((2 * np.pi / lam) * np.stack([np.cos(ang), np.sin(ang)], axis=1))
        self.W = np.concatenate(rows)
        self.lams = lams


class PolarSSPEncoder(SSPEncoder):
    """No hex structure: each module is a SINGLE frequency direction (its negative
    is the conjugate, so distinct axes live in a 180 deg sector). Scales follow a
    fixed geometric progression. fan=True gives one angle per scale, marching
    equally across the sector (D = n_scales); fan=False gives every scale the
    same n_ang equally spaced angles (D = n_scales * n_ang)."""

    def __init__(self, lam_min=0.25, lam_max=8.0, n_scales=6, n_ang=36, fan=False):
        if fan:
            lams = np.geomspace(lam_min, lam_max, n_scales)
            angs = np.arange(n_scales) * np.pi / n_scales
        else:
            lams = np.repeat(np.geomspace(lam_min, lam_max, n_scales), n_ang)
            angs = np.tile(np.arange(n_ang) * np.pi / n_ang, n_scales)
        self.W = (2 * np.pi / lams)[:, None] * np.stack([np.cos(angs), np.sin(angs)], axis=1)
        self.lams = np.geomspace(lam_min, lam_max, n_scales)


class HexGridSSPEncoder(SSPEncoder):
    """Hex SSP with the module budget spread as an even grid over BOTH axes:
    n_scales wavelengths in a fixed geometric progression [lam_min, lam_max],
    and per scale the SAME n_rot orientations equally spaced across the 120 deg
    sector (beyond which the 3-fold hex symmetry repeats). D = 3*n_scales*n_rot."""

    def __init__(self, lam_min=0.25, lam_max=8.0, n_scales=6, n_rot=12):
        self.lams = np.geomspace(lam_min, lam_max, n_scales)
        rots = np.arange(n_rot) * (2 * np.pi / 3) / n_rot
        rows = []
        for lam in self.lams:
            for rot in rots:
                ang = rot + np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
                rows.append((2 * np.pi / lam) * np.stack([np.cos(ang), np.sin(ang)], axis=1))
        self.W = np.concatenate(rows)


# ----------------------------------------------------------------------------
# Registration: coarse-to-fine search over (theta, translation)
# ----------------------------------------------------------------------------

def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _grid_offsets(half, step):
    v = np.arange(-half, half + step / 2, step)
    gx, gy = np.meshgrid(v, v)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def _parabolic(sm1, s0, sp1, h):
    denom = sm1 - 2 * s0 + sp1
    return 0.0 if abs(denom) < 1e-12 else 0.5 * h * (sm1 - sp1) / denom


class Matcher:
    def __init__(self, enc, t_half=0.36, rot_half_deg=6.0, rot_step_deg=1.5,
                 perm=None):
        """perm=(n_ring, n_ang): enc.W is a ring-major polar lattice (angles
        j*pi/n_ang per ring), so coarse rotation candidates are generated by
        EXACT index permutation of TWO encodings (lattice-aligned + half-step
        pre-rotated) instead of re-encoding per candidate. Permuting the pair
        yields the full pi/(2*n_ang) grid — the same 1.5 deg coarse density
        as the re-encoding path (a single encode on the 3 deg lattice leaves
        the fine stage, +-1.5 deg around the winner, with ZERO margin at
        lattice midpoints and lets the coarse translation pick drift beyond
        the fine +-0.06 m window; measured on Intel, that halves loop-closure
        recall). perm=None keeps the original path."""
        self.enc = enc
        self.perm = perm
        self.rot_half = np.deg2rad(rot_half_deg)
        self.rot_step = np.pi / perm[1] if perm is not None else np.deg2rad(rot_step_deg)
        self.coarse_off = _grid_offsets(t_half, 0.06)
        self.fine_off = _grid_offsets(0.06, 0.02)
        # translation-score basis, reused every frame: exp(i off . W)
        self.E_coarse = np.exp(1j * (self.coarse_off @ enc.W.T))
        self.E_fine = np.exp(1j * (self.fine_off @ enc.W.T))

    def _best_translation(self, M, S, center, offsets, E):
        c = np.conj(M) * S * self.enc.shift(center)
        scores = (E @ c).real
        k = int(np.argmax(scores))
        return center + offsets[k], scores[k], scores, k

    def _rot_permute(self, Svec, m):
        """Encoding of the scan rotated by m*pi/n_ang: exact circular shift of
        the angle index (conjugate on sector wrap) on the ring-major lattice."""
        n_ring, n_ang = self.perm
        A = Svec.reshape(n_ring, n_ang)
        ext = np.concatenate([A, np.conj(A)], axis=1)      # angle period 2pi
        return ext[:, (np.arange(n_ang) - m) % (2 * n_ang)].reshape(-1)

    def match(self, M, pts, weights, guess):
        """Maximize correlation of the scan SSP with map M over SE(2) around guess."""
        # -- coarse
        best = (-np.inf, None, None)
        if self.perm is not None:
            # encode TWICE (lattice-aligned + half-step pre-rotated); index
            # permutations of the pair cover the pi/(2*n_ang) grid exactly,
            # restoring 1.5 deg coarse density at 2 encodes instead of 13
            n_ang = self.perm[1]
            lat = np.pi / n_ang
            bases = ((self.enc.encode(pts, weights), 0.0),
                     (self.enc.encode(pts @ _rot(lat / 2).T, weights), lat / 2))
            m0 = int(round(guess[2] / lat))
            h = int(np.ceil(self.rot_half / lat - 1e-9))
            for m in range(m0 - h, m0 + h + 1):
                for S0, off in bases:
                    S = self._rot_permute(S0, m)
                    t, sc, _, _ = self._best_translation(M, S, guess[:2], self.coarse_off, self.E_coarse)
                    if sc > best[0]:
                        best = (sc, m * lat + off, t)
        else:
            thetas = guess[2] + np.arange(-self.rot_half, self.rot_half + 1e-9, self.rot_step)
            for th in thetas:
                S = self.enc.encode(pts @ _rot(th).T, weights)
                t, sc, _, _ = self._best_translation(M, S, guess[:2], self.coarse_off, self.E_coarse)
                if sc > best[0]:
                    best = (sc, th, t)
        _, th_c, t_c = best

        # -- fine (translation grid recentred on coarse best)
        fine_step = np.deg2rad(0.375)
        thetas = th_c + fine_step * np.arange(-4, 5)
        th_scores, results = [], []
        for th in thetas:
            S = self.enc.encode(pts @ _rot(th).T, weights)
            t, sc, scores, k = self._best_translation(M, S, t_c, self.fine_off, self.E_fine)
            th_scores.append(sc)
            results.append((t, scores, k))
        j = int(np.argmax(th_scores))
        th_f = thetas[j]
        t_f, scores, k = results[j]

        # -- parabolic sub-cell refinement
        if 0 < j < len(thetas) - 1:
            th_f += _parabolic(th_scores[j - 1], th_scores[j], th_scores[j + 1], fine_step)
        n = int(round(np.sqrt(len(self.fine_off))))
        ky, kx = divmod(k, n)
        grid = scores.reshape(n, n)
        dt = np.zeros(2)
        if 0 < kx < n - 1:
            dt[0] = _parabolic(grid[ky, kx - 1], grid[ky, kx], grid[ky, kx + 1], 0.02)
        if 0 < ky < n - 1:
            dt[1] = _parabolic(grid[ky - 1, kx], grid[ky, kx], grid[ky + 1, kx], 0.02)
        return np.array([t_f[0] + dt[0], t_f[1] + dt[1], th_f])


# ----------------------------------------------------------------------------
# Trajectory (smooth loop around the central box)
# ----------------------------------------------------------------------------

def make_trajectory(n_frames):
    s = np.linspace(0, 2 * np.pi * 1.05, n_frames)
    a, b, cx, cy = 4.8, 1.9, 8.0, 5.0
    x = cx + a * np.cos(s)
    y = cy + b * np.sin(s)
    heading = np.arctan2(b * np.cos(s), -a * np.sin(s)) + RNG.normal(0, np.deg2rad(0.3), n_frames)
    return np.stack([x, y, heading], axis=1)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ----------------------------------------------------------------------------
# Main SLAM loop
# ----------------------------------------------------------------------------

def main(mode="rff", enc=None, make_plots=True):
    segs = make_environment()
    if enc is not None:
        pass
    elif mode == "rff":
        enc = SSPEncoder(dim=2048, length_scale=0.25)
    elif mode == "hex":
        enc = HexSSPEncoder(rot_step=0.0)
    elif mode == "hexrot":  # deterministic golden-angle orientation per module:
        # quasi-uniform coverage of the 120 deg sector, no helical rotation/scale
        # coupling (a constant small fan step k*(120/n) creates one: rotating the
        # scan by one step ~ shifting every module index by one ~ rescaling space)
        enc = HexSSPEncoder(rot_step=np.pi * (3 - np.sqrt(5)))
    elif mode == "hexgrid":  # equal split of modules across rotations x scales
        enc = HexGridSSPEncoder()
    elif mode == "polar":  # no hex: single-direction modules, angle grid per scale
        enc = PolarSSPEncoder()
    elif mode == "oct":  # recommended: 4 octaves x 24 angles, D=96 (see RESULTS.md)
        enc = PolarSSPEncoder(lam_min=0.25, lam_max=2.0, n_scales=4, n_ang=24)
    elif mode == "fan":  # no hex: one angle per scale, equal march across 180 deg
        enc = PolarSSPEncoder(n_scales=216, fan=True)
    else:
        raise SystemExit(
            f"unknown mode {mode!r} (use oct | rff | hex | hexrot | hexgrid | polar | fan)")
    print(f"mode: {mode}   D = {enc.W.shape[0]} complex features"
          + (f"   scales: {len(enc.lams)} (lam {enc.lams[0]:.2f}..{enc.lams[-1]:.2f} m)"
             if hasattr(enc, "lams") else ""))
    matcher = Matcher(enc)
    gt = make_trajectory(140)

    M = np.zeros(enc.W.shape[0], dtype=complex)
    est = np.zeros_like(gt)
    cloud, cloud_frame = [], []
    total_rejected = 0
    demo = (None, -1)  # (scan data, n_rejected) for the phantom-wall figure

    t0 = time.time()
    for k, pose in enumerate(gt):
        ranges, beams = simulate_scan(segs, pose)
        pts, w, rej = scan_to_samples(ranges, beams)
        total_rejected += len(rej)
        # demo frame: most long rejected bridges (true phantom walls, not grazing)
        n_long = int(np.sum(np.linalg.norm(rej[:, 1] - rej[:, 0], axis=1) > 0.6)) if len(rej) else 0
        if n_long > demo[1]:
            demo = ((ranges.copy(), beams.copy(), pose.copy()), n_long)

        if k == 0:
            est[0] = gt[0]  # anchor world frame to the first pose
        else:
            guess = est[k - 1].copy()
            if k >= 2:  # constant-velocity prediction
                guess[:2] += est[k - 1, :2] - est[k - 2, :2]
                guess[2] += wrap(est[k - 1, 2] - est[k - 2, 2])
            est[k] = matcher.match(M, pts, w, guess)
            est[k, 2] = wrap(est[k, 2])

        world_pts = pts @ _rot(est[k, 2]).T + est[k, :2]
        M += enc.shift(est[k, :2]) * enc.encode(pts @ _rot(est[k, 2]).T, w)
        cloud.append(world_pts[:: 3])
        cloud_frame.append(np.full(len(world_pts[::3]), k))
    dt = time.time() - t0

    terr = np.linalg.norm(est[:, :2] - gt[:, :2], axis=1)
    aerr = np.abs(np.degrees(wrap(est[:, 2] - gt[:, 2])))
    print(f"frames: {len(gt)}  time: {dt:.1f}s ({dt / len(gt) * 1e3:.0f} ms/frame)")
    print(f"translation error  mean {terr.mean() * 100:.1f} cm   max {terr.max() * 100:.1f} cm   final {terr[-1] * 100:.1f} cm")
    print(f"heading error      mean {aerr.mean():.2f} deg  max {aerr.max():.2f} deg  final {aerr[-1]:.2f} deg")
    print(f"occlusion filter rejected {total_rejected} segments total")

    if make_plots:
        plot(segs, gt, est, terr, aerr, enc, M,
             np.concatenate(cloud), np.concatenate(cloud_frame), demo[0], mode)
    return {"D": enc.W.shape[0], "ms": dt / len(gt) * 1e3,
            "terr_mean": terr.mean(), "terr_max": terr.max(),
            "aerr_mean": aerr.mean(), "aerr_max": aerr.max()}


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def draw_walls(ax, segs, **kw):
    for (p, q) in segs:
        ax.plot([p[0], q[0]], [p[1], q[1]], **kw)


def plot(segs, gt, est, terr, aerr, enc, M, cloud, cloud_frame, demo_scan, mode):
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    ax = axes[0, 0]
    draw_walls(ax, segs, color="k", lw=1.5)
    ax.plot(gt[:, 0], gt[:, 1], "g-", lw=2, label="ground truth")
    ax.plot(est[:, 0], est[:, 1], "r--", lw=1.5, label="SSP estimate")
    ax.plot(gt[0, 0], gt[0, 1], "go", ms=8)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Environment & trajectories")
    ax.set_aspect("equal")

    ax = axes[0, 1]
    sc = ax.scatter(cloud[:, 0], cloud[:, 1], c=cloud_frame, s=0.5, cmap="viridis")
    draw_walls(ax, segs, color="r", lw=0.5, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="frame")
    ax.set_title("Registered scan points (est. poses), true walls in red")
    ax.set_aspect("equal")

    ax = axes[0, 2]
    ax.plot(terr * 100, label="translation [cm]")
    ax.plot(aerr * 10, label="heading [0.1 deg]")
    ax.set_xlabel("frame")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Pose error vs ground truth")

    ax = axes[1, 0]
    xs, ys = np.arange(-0.5, 16.51, 0.05), np.arange(-0.5, 10.51, 0.05)
    dens = enc.density(M, xs, ys)
    ax.imshow(dens, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
              cmap="magma", aspect="equal")
    ax.set_title("Decoded SSP map:  Re < M, phi(x) >")

    if demo_scan is not None:
        ranges, beams, pose = demo_scan
        pts, w, rej = scan_to_samples(ranges, beams)
        R, t = _rot(pose[2]), pose[:2]
        wpts = pts @ R.T + t

        ax = axes[1, 1]
        draw_walls(ax, segs, color="0.7", lw=1)
        ax.scatter(wpts[:, 0], wpts[:, 1], s=2, c="tab:blue", label="kept segment samples")
        for seg in rej:
            wseg = seg @ R.T + t
            ax.plot(wseg[:, 0], wseg[:, 1], "r-", lw=1.2)
        ax.plot([], [], "r-", lw=1.2, label="rejected (phantom) segments")
        ax.plot(*t, "k^", ms=9)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title("Occlusion filter on one scan (red = rejected bridges)")
        ax.set_aspect("equal")

        # same scan encoded WITHOUT the filter -> phantom walls in the density
        global OCC_RATIO
        saved, OCC_RATIO = OCC_RATIO, np.inf
        pts_nf, w_nf, _ = scan_to_samples(ranges, beams)
        OCC_RATIO = saved
        S = enc.shift(t) * enc.encode(pts_nf @ R.T, w_nf)
        dens = enc.density(S, xs, ys)
        ax = axes[1, 2]
        ax.imshow(dens, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                  cmap="magma", aspect="equal")
        draw_walls(ax, segs, color="cyan", lw=0.4, alpha=0.6)
        ax.plot(*t, "w^", ms=8)
        ax.set_title("Same scan encoded WITHOUT filter: phantom walls")

    fig.suptitle(f"SSP SLAM — scan matching, encoder: {mode}", fontsize=14)
    fig.tight_layout()
    out = f"ssp_slam_results_{mode}.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "rff")
