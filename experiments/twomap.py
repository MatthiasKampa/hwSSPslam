"""TWO-MAP fusion architecture + vision detector menu (user directive
2026-07-14: school focus; heatmap/CNN-class detectors; "use 2 different
maps for lidar and vision so we can opt the systems individually").

Architecture under test: each modality keeps ITS OWN map/vector space —
  lidar map   az2d D=240 on the full cloud, similarity = the house
              ROTATION-SEARCHED |cos| (exact yaw permutations — what
              survives the school's all-reverse revisits)
  vision map  bearing lattice OWN layout/scales (fib half-sphere or a
              FOV-CONE-concentrated set — the camera only sees ~86x57
              deg; uniform sphere coverage wastes D), similarity plain
              |cos| (rot-search is pointless for a forward camera)
  fusion      LATE, score-level: z-normed (label-free, off-diagonal
              stats) weighted sum, alpha in {0.25, 0.5, 0.75}
Control: the single SHARED map (azel3d vector addition, banked
2026-07-14) on the same frame set.

Vision arms (all integer / ECP5-budget; ~7 MMAC/frame for the CNN class
= ~2.5% of 28 MULT18 @ 50 MHz at 5 Hz):
  fast9-pk / shito-pk   zoo detectors, peak features (controls)
  dog-pk / dog-raw      integer DoG (box3*9 - box9) blob heatmap,
                        peaks / RAW-heatmap-encode (dense weighted grid)
  ucnn-pk / ucnn-raw    UNTRAINED int8 3-layer micro-CNN (seeded random
                        filters — architecture lower bound; a trained
                        tiny-YOLO-class head is the same budget), peaks /
                        raw heatmap
  gridint               8x8 mean-intensity grid encode (no detector at
                        all — does a raw downsampled image bundle carry
                        the signal?)

Usage: python3 -m experiments.twomap bench school_run2|spot
"""
import sys
from pathlib import Path

import numpy as np

import experiments.lattice3d as L3
import experiments.detzoo as DZ

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hw" / "ecp5"
                       / "host"))
import golden_cam as GC          # noqa: E402

RNG = 11
ANG_SCALES = (0.3, 0.6, 1.2, 2.4)     # angular ladder (unit-sphere chords)


# --------------------------------------------------------------------------
#  vision lattices (the vision map's own space)
# --------------------------------------------------------------------------
def W_vis_fib(n=60, scales=ANG_SCALES):
    return np.concatenate([(2 * np.pi / s) * L3.dirs_fib(n) for s in scales])


def W_vis_cone(n_az=10, n_el=6, half_az=50.0, half_el=35.0,
               scales=ANG_SCALES):
    """Directions concentrated in the camera FOV cone (around +z of the
    CAMERA frame — bearings are encoded in-camera-frame per map)."""
    out = []
    for e in np.linspace(-half_el, half_el, n_el):
        for a in np.linspace(-half_az, half_az, n_az):
            ca, sa = np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))
            ce, se = np.cos(np.deg2rad(e)), np.sin(np.deg2rad(e))
            out.append([sa * ce, se, ca * ce])       # cam frame, z fwd
    u = np.asarray(out)
    return np.concatenate([(2 * np.pi / s) * u for s in scales])


# --------------------------------------------------------------------------
#  heatmap detectors (integer)
# --------------------------------------------------------------------------
def _boxsum(a, r):
    c = np.cumsum(np.cumsum(a.astype(np.int64), 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def dog_heat(gray):
    s3 = _boxsum(gray, 1)          # (h-2, w-2) 9-px sums
    s9 = _boxsum(gray, 4)          # (h-8, w-8) 81-px sums
    h = np.abs(9 * s3[3:-3, 3:-3] - s9) >> 6   # keep < 255 (no NMS plateaus)
    out = np.zeros(gray.shape, np.int64)
    out[4:gray.shape[0] - 4, 4:gray.shape[1] - 4] = h
    return out                     # full-res heat, borders zero


_UC = None


def _ucnn_weights():
    global _UC
    if _UC is None:
        r = np.random.default_rng(RNG)
        _UC = (r.integers(-2, 3, (8, 1, 3, 3)),
               r.integers(-2, 3, (16, 8, 3, 3)),
               r.integers(-2, 3, (1, 16, 1, 1)))
    return _UC


def _conv_s2(x, w, shift):
    from numpy.lib.stride_tricks import sliding_window_view
    v = sliding_window_view(x, (3, 3), axis=(1, 2))[:, ::2, ::2]
    y = np.tensordot(w, v, axes=([1, 2, 3], [0, 3, 4]))
    # |abs| activation: relu after zero-mean random filters zeroes ~half
    # of every layer and the 3-layer chain collapsed to an all-zero heat
    # (degenerate first run, banked) — abs keeps the energy, still 1
    # comparator + mux in RTL
    return np.clip(np.abs(y) >> shift, 0, 255)


def ucnn_heat(gray):
    """3-layer untrained int8 CNN -> heat at 1/4 the bin-2 res.
    ~7 MMAC/frame — the tiny-YOLO complexity class."""
    w1, w2, w3 = _ucnn_weights()
    x = gray[None].astype(np.int64)
    x = _conv_s2(x, w1, 4)
    x = _conv_s2(x, w2, 6)
    y = np.tensordot(w3[:, :, 0, 0], x, axes=([1], [0]))[0]
    return np.clip(np.abs(y) >> 3, 0, 255)     # (~59, ~79) 2-D heat


# --------------------------------------------------------------------------
#  encodings
# --------------------------------------------------------------------------
def peaks(heat, target, scale_uv):
    def det(_img, t):
        m = heat > t
        return m, np.where(m, heat, 0)
    t, (ys, xs, sc) = DZ.detect_target(det, heat.astype(np.uint8)
                                       if heat.max() < 256 else
                                       np.clip(heat, 0, 255).astype(
                                           np.uint8), target)
    return ys * scale_uv, xs * scale_uv, sc.astype(float) + 1.0


def raw_grid(heat, scale_uv, cap=3000):
    ys, xs = np.nonzero(heat > 0)
    w = heat[ys, xs].astype(float)
    if len(ys) > cap:
        o = np.argsort(-w)[:cap]
        ys, xs, w = ys[o], xs[o], w[o]
    return ys * scale_uv, xs * scale_uv, w


def grid_int(gray):
    """8x8 mean-intensity cells as weights (no detector)."""
    h, w = gray.shape[0] // 8 * 8, gray.shape[1] // 8 * 8
    g = gray[:h, :w].reshape(h // 8, 8, w // 8, 8).mean((1, 3))
    ys, xs = np.mgrid[0:h // 8, 0:w // 8]
    return (ys.ravel() * 8 + 4, xs.ravel() * 8 + 4,
            g.ravel().astype(float) + 1.0)


def cam_bearings(K, ys, xs):
    Ki = np.linalg.inv(K)
    uv = np.stack([xs * 2 + 1.0, ys * 2 + 1.0, np.ones(len(xs))])
    b = (Ki @ uv).T
    return b / np.linalg.norm(b, axis=1, keepdims=True)


# --------------------------------------------------------------------------
#  appearance-IN-PHASE dense encoders (architecture note in hw/ecp5/README:
#  bind what with where — weights alone cannot separate same-bearing-
#  different-appearance; all integer + cis-ROM addressable)
# --------------------------------------------------------------------------
def _cells(gray, cs=8):
    h, w = gray.shape[0] // cs * cs, gray.shape[1] // cs * cs
    m = gray[:h, :w].reshape(h // cs, cs, w // cs, cs).mean((1, 3))
    ys, xs = np.mgrid[0:h // cs, 0:w // cs]
    return m, ys * cs + cs // 2, xs * cs + cs // 2


def enc_intphase(gray, K, Wdir):
    """Cell intensity bound into phase (3 intensity wavelengths cycled
    across lattice rows). Predicted exposure-fragile."""
    m, ys, xs = _cells(gray)
    b = cam_bearings(K, ys.ravel(), xs.ravel())
    lam = np.array([0.375, 0.75, 1.5])[np.arange(len(Wdir)) % 3]
    ph_app = (2 * np.pi / lam)[None, :] * (m.ravel() / 256.0)[:, None]
    v = np.exp(1j * (b @ Wdir.T + ph_app)).sum(0)
    return v / max(np.linalg.norm(v), 1e-12)


def enc_gradhog(gray, K, Wdir):
    """Cell gradient ORIENTATION (8-direction integer quantization,
    pi-periodic doubled) in phase, magnitude as weight. HOG-in-VSA."""
    import experiments.detzoo as DZ
    dx, dy = DZ._sobel(gray)
    dxp = np.zeros(gray.shape); dxp[1:-1, 1:-1] = dx
    dyp = np.zeros(gray.shape); dyp[1:-1, 1:-1] = dy
    cs = 8
    h, w = gray.shape[0] // cs * cs, gray.shape[1] // cs * cs
    cdx = dxp[:h, :w].reshape(h // cs, cs, w // cs, cs).mean((1, 3))
    cdy = dyp[:h, :w].reshape(h // cs, cs, w // cs, cs).mean((1, 3))
    ys, xs = np.mgrid[0:h // cs, 0:w // cs]
    b = cam_bearings(K, (ys * cs + cs // 2).ravel(),
                     (xs * cs + cs // 2).ravel())
    th = np.arctan2(cdy, cdx).ravel()
    thq = np.round(th / (np.pi / 4)) * (np.pi / 4)     # 8-dir integer RTL
    mag = np.hypot(cdx, cdy).ravel() + 1.0
    harm = (np.arange(len(Wdir)) % 2) + 1              # 1..2 harmonics
    ph_app = 2.0 * thq[:, None] * harm[None, :]        # pi-periodic -> 2th
    v = (mag[:, None] * np.exp(1j * (b @ Wdir.T + ph_app))).sum(0)
    return v / max(np.linalg.norm(v), 1e-12)


def enc_census(gray, K, Wdir):
    """Cell-level census: cell mean vs its 8 neighbours -> 8-bit code ->
    binary phase decomposition (bit k shifts phase by pi/2^k).
    Illumination invariant — predicted deploy winner."""
    m, ys, xs = _cells(gray)
    code = np.zeros(m.shape, np.int32)
    k = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                n = np.roll(np.roll(m, dy, 0), dx, 1)
                code |= (m > n).astype(np.int32) << k
                k += 1
    inner = np.zeros(m.shape, bool)
    inner[1:-1, 1:-1] = True
    b = cam_bearings(K, ys[inner], xs[inner])
    c = code[inner].astype(float)
    ph_app = 2 * np.pi * c / 256.0
    harm = (np.arange(len(Wdir)) % 3) + 1
    v = np.exp(1j * (b @ Wdir.T + ph_app[:, None] * harm[None, :])).sum(0)
    return v / max(np.linalg.norm(v), 1e-12)


# --------------------------------------------------------------------------
#  lidar map: rotation-searched az2d similarity
# --------------------------------------------------------------------------
def lidar_rotsim(clouds):
    W = L3.make_lattices()["az2d"]
    V = np.stack([L3.encode(W, c) for c in clouds])
    n_az = 60
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


def _znorm(S):
    m = ~np.eye(len(S), dtype=bool)
    return (S - S[m].mean()) / max(S[m].std(), 1e-12)


# --------------------------------------------------------------------------
def bench(run):
    labels = "ref" if run == "spot" else "est"
    grays, clouds, pose, kind, K = DZ.frame_set(run, labels)
    pr = dict(same_r=1.0, far_lo=4.0) if labels == "est" else {}
    (si, sj), (di, dj) = L3._pairs(pose, **pr)
    adj = np.arange(len(pose) - 1)
    tgt = [min(len(c), 1000) for c in clouds]

    Sl = _znorm(lidar_rotsim(clouds))
    auc_l = L3._auc(Sl[si, sj], Sl[di, dj])
    # shared-map control (banked architecture): azel3d vector addition
    Wsh = L3.make_lattices()["azel3d"]
    Vl_sh = np.stack([L3.encode(Wsh, c) for c in clouds])
    Vc_sh = []
    for g, tg in zip(grays, tgt):
        t, (ys, xs, sc) = DZ.detect_target(DZ.det_fast9, g, tg)
        Vc_sh.append(L3.encode(Wsh, cam_bearings(K, ys, xs),
                               sc.astype(float) + 1.0))
    Vsh = np.stack(Vc_sh) + Vl_sh
    Vsh /= np.linalg.norm(Vsh, axis=1, keepdims=True)
    Ssh = np.abs(Vsh @ Vsh.conj().T)
    print(f"two-map bench ({run}, {len(grays)} frames, labels={kind}; "
          f"{len(si)} same / {len(di)} diff)")
    print(f"  lidar map (az2d rot-searched): AUC {auc_l:.3f} | shared-map "
          f"control (azel3d vec-add): "
          f"{L3._auc(Ssh[si, sj], Ssh[di, dj]):.3f}")

    VIS = {"vis-fib": W_vis_fib(), "vis-cone": W_vis_cone()}
    arms = {}
    for g_i, (g, tg) in enumerate(zip(grays, tgt)):
        heats = {"dog": (dog_heat(g), 1), "ucnn": (ucnn_heat(g), 4)}
        for name, fn in (("fast9", DZ.det_fast9),
                         ("shito", DZ.det_shitomasi)):
            t, (ys, xs, sc) = DZ.detect_target(fn, g, tg)
            arms.setdefault(f"{name}-pk", []).append(
                (ys, xs, sc.astype(float) + 1.0))
        for hname, (h, sc_uv) in heats.items():
            hy, hx, hw = peaks(np.clip(h, 0, 255).astype(np.uint8), tg, 1)
            arms.setdefault(f"{hname}-pk", []).append(
                (hy * sc_uv, hx * sc_uv, hw))
            ry, rx, rw = raw_grid(h, sc_uv)
            arms.setdefault(f"{hname}-raw", []).append((ry, rx, rw))
        arms.setdefault("gridint", []).append(grid_int(g))

    for arm, feats in arms.items():
        for vname, Wv in VIS.items():
            Vv = np.stack([L3.encode(Wv, cam_bearings(K, ys, xs), w)
                           for ys, xs, w in feats])
            Sv = _znorm(np.abs(Vv @ Vv.conj().T))
            a_v = L3._auc(Sv[si, sj], Sv[di, dj])
            rep = L3._auc(Sv[adj, adj + 1], Sv[di, dj])
            fused = []
            for al in (0.25, 0.5, 0.75):
                Sf = al * Sl + (1 - al) * Sv
                fused.append(L3._auc(Sf[si, sj], Sf[di, dj]))
            print(f"  {arm:9s} x {vname:8s} vis {a_v:.3f}  fused(.25/.5/"
                  f".75) {fused[0]:.3f}/{fused[1]:.3f}/{fused[2]:.3f}  "
                  f"adj-rep {rep:.3f}", flush=True)


def phase_bench(run):
    """Appearance-in-phase encoder family vs gridint (weights-only ref)."""
    labels = "ref" if run == "spot" else "est"
    grays, clouds, pose, kind, K = DZ.frame_set(run, labels)
    pr = dict(same_r=1.0, far_lo=4.0) if labels == "est" else {}
    (si, sj), (di, dj) = L3._pairs(pose, **pr)
    adj = np.arange(len(pose) - 1)
    Sl = _znorm(lidar_rotsim(clouds))
    Wv = W_vis_fib()
    print(f"appearance-in-phase encoders ({run}, labels={kind}; lidar "
          f"{L3._auc(Sl[si, sj], Sl[di, dj]):.3f}):")
    arms = {
        "gridint(w)": lambda g: L3.encode(Wv, cam_bearings(
            K, *grid_int(g)[:2]), grid_int(g)[2]),
        "intphase": lambda g: enc_intphase(g, K, Wv),
        "gradhog ": lambda g: enc_gradhog(g, K, Wv),
        "census  ": lambda g: enc_census(g, K, Wv),
    }
    for name, fn in arms.items():
        V = np.stack([fn(g) for g in grays])
        Sv = _znorm(np.abs(V @ V.conj().T))
        a = L3._auc(Sv[si, sj], Sv[di, dj])
        rep = L3._auc(Sv[adj, adj + 1], Sv[di, dj])
        Sm = np.maximum(Sl, Sv)
        S5 = 0.5 * Sl + 0.5 * Sv
        print(f"  {name} vis {a:.3f}  max-fused "
              f"{L3._auc(Sm[si, sj], Sm[di, dj]):.3f}  sum.5 "
              f"{L3._auc(S5[si, sj], S5[di, dj]):.3f}  adj-rep {rep:.3f}",
              flush=True)


def vissweep(run):
    """Vision-system individual optimization: gridint design space (cell
    size x weight transform x lattice D) + gain-invariant intphase
    (per-frame DC removal — cheap RTL fix for exposure fragility)."""
    import experiments.detzoo as DZ_
    labels = "ref" if run == "spot" else "est"
    grays, clouds, pose, kind, K = DZ.frame_set(run, labels)
    pr = dict(same_r=1.0, far_lo=4.0) if labels == "est" else {}
    (si, sj), (di, dj) = L3._pairs(pose, **pr)
    adj = np.arange(len(pose) - 1)
    Sl = _znorm(lidar_rotsim(clouds))
    print(f"vision sweep ({run}, labels={kind}; lidar rot "
          f"{L3._auc(Sl[si, sj], Sl[di, dj]):.3f}):")

    def cells_w(gray, cs, wmode):
        h, w = gray.shape[0] // cs * cs, gray.shape[1] // cs * cs
        m = gray[:h, :w].reshape(h // cs, cs, w // cs, cs).mean((1, 3))
        if wmode == "gradmag":
            dx, dy = DZ_._sobel(gray)
            gm = np.zeros(gray.shape)
            gm[1:-1, 1:-1] = np.abs(dx) + np.abs(dy)
            wgt = gm[:h, :w].reshape(h // cs, cs, w // cs, cs).mean((1, 3))
        elif wmode == "sqrt":
            wgt = np.sqrt(m + 1.0)
        else:
            wgt = m + 1.0
        ys, xs = np.mgrid[0:h // cs, 0:w // cs]
        return (ys.ravel() * cs + cs // 2, xs.ravel() * cs + cs // 2,
                wgt.ravel().astype(float) + 1e-6)

    Wv240 = W_vis_fib()
    for cs in (4, 8, 16):
        for wmode in ("int", "gradmag", "sqrt"):
            V = np.stack([L3.encode(Wv240, cam_bearings(
                K, *cells_w(g, cs, wmode)[:2]),
                cells_w(g, cs, wmode)[2]) for g in grays])
            Sv = _znorm(np.abs(V @ V.conj().T))
            Sm = np.maximum(Sl, Sv)
            print(f"  grid c{cs:2d} {wmode:7s} vis "
                  f"{L3._auc(Sv[si, sj], Sv[di, dj]):.3f}  max-f "
                  f"{L3._auc(Sm[si, sj], Sm[di, dj]):.3f}  adj "
                  f"{L3._auc(Sv[adj, adj + 1], Sv[di, dj]):.3f}",
                  flush=True)
    for n in (120, 240, 480):
        Wv = W_vis_fib(n=n // 4 * 1, scales=ANG_SCALES) if False else \
            np.concatenate([(2 * np.pi / s) * L3.dirs_fib(n // 4)
                            for s in ANG_SCALES])
        V = np.stack([L3.encode(Wv, cam_bearings(
            K, *grid_int(g)[:2]), grid_int(g)[2]) for g in grays])
        Sv = _znorm(np.abs(V @ V.conj().T))
        Sm = np.maximum(Sl, Sv)
        print(f"  gridint D{len(Wv):4d}     vis "
              f"{L3._auc(Sv[si, sj], Sv[di, dj]):.3f}  max-f "
              f"{L3._auc(Sm[si, sj], Sm[di, dj]):.3f}", flush=True)
    # gain-invariant intphase: per-frame DC-removed cell intensity
    def enc_intphase_dc(gray):
        m, ys, xs = _cells(gray)
        m = m - m.mean()                     # DC removal (RTL: subtract
        b = cam_bearings(K, ys.ravel(), xs.ravel())   # running frame mean)
        lam = np.array([0.375, 0.75, 1.5])[np.arange(len(Wv240)) % 3]
        ph = (2 * np.pi / lam)[None, :] * ((m.ravel() + 128) / 256.0)[:, None]
        v = np.exp(1j * (b @ Wv240.T + ph)).sum(0)
        return v / max(np.linalg.norm(v), 1e-12)
    V = np.stack([enc_intphase_dc(g) for g in grays])
    Sv = _znorm(np.abs(V @ V.conj().T))
    Sm = np.maximum(Sl, Sv)
    print(f"  intphase-DC       vis {L3._auc(Sv[si, sj], Sv[di, dj]):.3f}"
          f"  max-f {L3._auc(Sm[si, sj], Sm[di, dj]):.3f}  adj "
          f"{L3._auc(Sv[adj, adj + 1], Sv[di, dj]):.3f}", flush=True)


def enc_hog8(gray, K, Wdir):
    """FULL 8-bin gradient histogram per cell: each bin = one bound
    phasor (orientation phase x bin magnitude weight) — the gradhog
    extension (all bins, not just dominant)."""
    import experiments.detzoo as DZ_
    dx, dy = DZ_._sobel(gray)
    dxp = np.zeros(gray.shape); dxp[1:-1, 1:-1] = dx
    dyp = np.zeros(gray.shape); dyp[1:-1, 1:-1] = dy
    cs = 16
    h, w = gray.shape[0] // cs * cs, gray.shape[1] // cs * cs
    th = np.arctan2(dyp[:h, :w], dxp[:h, :w])
    mag = np.hypot(dxp[:h, :w], dyp[:h, :w])
    binq = ((th + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    ys, xs = np.mgrid[0:h // cs, 0:w // cs]
    b = cam_bearings(K, (ys * cs + cs // 2).ravel(),
                     (xs * cs + cs // 2).ravel())
    harm = (np.arange(len(Wdir)) % 2) + 1
    v = np.zeros(len(Wdir), complex)
    for k in range(8):
        m = (binq == k) * mag
        cw = m.reshape(h // cs, cs, w // cs, cs).mean((1, 3)).ravel()
        ph = 2 * np.pi * k / 8.0
        v += (cw[:, None] * np.exp(1j * (b @ Wdir.T
                                         + ph * harm[None, :]))).sum(0)
    return v / max(np.linalg.norm(v), 1e-12)


def extract_bench(run):
    """Feature-extraction variants vs the banked tiers: multi-scale grid,
    full HOG-8bin, combined snapshot (gridint (+) intphase sub-bundles)."""
    labels = "ref" if run == "spot" else "est"
    grays, clouds, pose, kind, K = DZ.frame_set(run, labels)
    pr = dict(same_r=1.0, far_lo=4.0) if labels == "est" else {}
    (si, sj), (di, dj) = L3._pairs(pose, **pr)
    adj = np.arange(len(pose) - 1)
    Sl = _znorm(lidar_rotsim(clouds))
    Wv = W_vis_fib()
    print(f"extraction variants ({run}, labels={kind}; lidar "
          f"{L3._auc(Sl[si, sj], Sl[di, dj]):.3f}):")

    def multi_grid(g):
        v = np.zeros(len(Wv), complex)
        for cs in (8, 32):
            h, w = g.shape[0] // cs * cs, g.shape[1] // cs * cs
            m = g[:h, :w].reshape(h // cs, cs, w // cs, cs).mean((1, 3))
            ys, xs = np.mgrid[0:h // cs, 0:w // cs]
            b = cam_bearings(K, (ys * cs + cs // 2).ravel(),
                             (xs * cs + cs // 2).ravel())
            vv = ((m.ravel() + 1.0)[:, None]
                  * np.exp(1j * (b @ Wv.T))).sum(0)
            v += vv / max(np.linalg.norm(vv), 1e-12)
        return v / max(np.linalg.norm(v), 1e-12)

    arms = {
        "gridint(ref)": lambda g: L3.encode(Wv, cam_bearings(
            K, *grid_int(g)[:2]), grid_int(g)[2]),
        "multigrid8+32": multi_grid,
        "hog8bin": lambda g: enc_hog8(g, K, Wv),
        "grid+intphase": lambda g: (lambda a, b: (a + b)
                                    / np.linalg.norm(a + b))(
            L3.encode(Wv, cam_bearings(K, *grid_int(g)[:2]),
                      grid_int(g)[2]), enc_intphase(g, K, Wv)),
    }
    for name, fn in arms.items():
        V = np.stack([fn(g) for g in grays])
        Sv = _znorm(np.abs(V @ V.conj().T))
        Sm = np.maximum(Sl, Sv)
        print(f"  {name:14s} vis {L3._auc(Sv[si, sj], Sv[di, dj]):.3f}  "
              f"max-f {L3._auc(Sm[si, sj], Sm[di, dj]):.3f}  adj "
              f"{L3._auc(Sv[adj, adj + 1], Sv[di, dj]):.3f}", flush=True)


def policy(run):
    """Fusion-policy study on the best vision channel (gridint): alpha
    sweep + max rule, two-map z-normed sims."""
    labels = "ref" if run == "spot" else "est"
    grays, clouds, pose, kind, K = DZ.frame_set(run, labels)
    pr = dict(same_r=1.0, far_lo=4.0) if labels == "est" else {}
    (si, sj), (di, dj) = L3._pairs(pose, **pr)
    Sl = _znorm(lidar_rotsim(clouds))
    Wv = W_vis_fib()
    Vv = np.stack([L3.encode(Wv, cam_bearings(K, *f[:2]), f[2])
                   for f in (grid_int(g) for g in grays)])
    Sv = _znorm(np.abs(Vv @ Vv.conj().T))
    print(f"fusion policy ({run}, labels={kind}): lidar "
          f"{L3._auc(Sl[si, sj], Sl[di, dj]):.3f}  gridint "
          f"{L3._auc(Sv[si, sj], Sv[di, dj]):.3f}")
    for al in (0.1, 0.3, 0.5, 0.7, 0.9):
        S = al * Sl + (1 - al) * Sv
        print(f"  sum a={al:.1f}: {L3._auc(S[si, sj], S[di, dj]):.3f}",
              flush=True)
    Sm = np.maximum(Sl, Sv)
    print(f"  max rule : {L3._auc(Sm[si, sj], Sm[di, dj]):.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bench"
    run = sys.argv[2] if len(sys.argv) > 2 else "school_run2"
    if cmd == "policy":
        policy(run)
    elif cmd == "phase":
        phase_bench(run)
    elif cmd == "vissweep":
        vissweep(run)
    elif cmd == "extract":
        extract_bench(run)
    else:
        bench(run if cmd == "bench" else cmd)
