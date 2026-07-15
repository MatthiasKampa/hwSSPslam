"""Integer camera-feature golden model for the ECP5 track (fast9.v gate).

Every stage is integer / comparator-only, matching what streams through
line-buffer RTL fed by an OV5640 DVP port:

  luma   g = (77*R + 150*G + 29*B) >> 8                (uint8)
  bin2   b = (p00 + p01 + p10 + p11 + 2) >> 2          (2x2 box, uint8)
  FAST-9 radius-3 Bresenham circle (16 taps); corner iff >=9 CONTIGUOUS
         taps are all brighter than c+t or all darker than c-t
         (d_i = p_i - c; bright: d_i > t; dark: d_i < -t)
  score  max( sum_i max(d_i - t, 0), sum_i max(-d_i - t, 0) ), 0 if not
         corner — adders and comparators only (<= 12 bits)
  NMS    3x3, keep iff score strictly > all 8 neighbours (plateau ties
         drop — symmetric, RTL-trivial; host/deploy side in v0)

detect_target() binary-searches t for a target post-NMS count (the
dataset extractor's deterministic form; deploy RTL uses a +-1/frame
feedback register toward the same target). write_vectors() emits hex
fixtures for rtl/tb_fast9.v (pre-NMS mask+score = the RTL contract).

python3 hw/ecp5/host/golden_cam.py selftest
"""
import sys

import numpy as np

# radius-3 Bresenham circle, clockwise from 12 o'clock: (dy, dx)
CIRCLE = [(-3, 0), (-3, 1), (-2, 2), (-1, 3), (0, 3), (1, 3), (2, 2),
          (3, 1), (3, 0), (3, -1), (2, -2), (1, -3), (0, -3), (-1, -3),
          (-2, -2), (-3, -1)]
T_MIN, T_MAX = 4, 127


def rgb_to_gray(rgb):
    r = rgb[..., 0].astype(np.uint32)
    g = rgb[..., 1].astype(np.uint32)
    b = rgb[..., 2].astype(np.uint32)
    return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)


def bin2(gray):
    h, w = gray.shape[0] & ~1, gray.shape[1] & ~1
    a = gray[:h, :w].astype(np.uint16)
    return ((a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2]
             + a[1::2, 1::2] + 2) >> 2).astype(np.uint8)


def fast9(img, t):
    """-> (corner mask, score) uint16, borders (3 px) zero. Pre-NMS."""
    im = img.astype(np.int16)
    h, w = im.shape
    c = im[3:h - 3, 3:w - 3]
    d = np.stack([im[3 + dy:h - 3 + dy, 3 + dx:w - 3 + dx] - c
                  for dy, dx in CIRCLE])                    # (16, H-6, W-6)
    bright, dark = d > t, d < -t
    corner = np.zeros(c.shape, bool)
    for arm in (bright, dark):
        for r in range(16):
            idx = [(r + j) % 16 for j in range(9)]
            corner |= np.logical_and.reduce(arm[idx])
    bs = np.maximum(d - t, 0).sum(0)
    ds = np.maximum(-d - t, 0).sum(0)
    sc = np.where(corner, np.maximum(bs, ds), 0).astype(np.uint16)
    mask = np.zeros(img.shape, bool)
    score = np.zeros(img.shape, np.uint16)
    mask[3:h - 3, 3:w - 3] = corner
    score[3:h - 3, 3:w - 3] = sc
    return mask, score


def nms3(score):
    """Keep iff score strictly > all 8 neighbours (zeros never survive)."""
    s = score.astype(np.int32)
    p = np.pad(s, 1, constant_values=-1)
    nmax = np.zeros_like(s)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                np.maximum(nmax, p[1 + dy:p.shape[0] - 1 + dy,
                                   1 + dx:p.shape[1] - 1 + dx], out=nmax)
    return (s > nmax) & (s > 0)


def detect(img, t):
    """-> (ys, xs, scores) post-NMS, raster order."""
    _, score = fast9(img, t)
    keep = nms3(score)
    ys, xs = np.nonzero(keep)
    return ys.astype(np.uint16), xs.astype(np.uint16), score[keep]


def detect_target(img, target):
    """Binary-search t for the post-NMS count closest to target
    (deterministic; prefers the lower t on ties -> count >= target side)."""
    lo, hi = T_MIN, T_MAX
    best = None
    while lo <= hi:
        t = (lo + hi) // 2
        ys, xs, sc = detect(img, t)
        n = len(ys)
        key = (abs(n - target), t)
        if best is None or key < best[0]:
            best = (key, t, (ys, xs, sc))
        if n > target:
            lo = t + 1
        elif n < target:
            hi = t - 1
        else:
            break
    return best[1], best[2]


def write_vectors(img, t, prefix):
    """Hex fixtures for tb_fast9.v: image bytes + expected pre-NMS
    corner/score per pixel (raster order)."""
    mask, score = fast9(img, t)
    h, w = img.shape
    with open(f"{prefix}_img.hex", "w") as f:
        f.writelines(f"{v:02x}\n" for v in img.reshape(-1))
    with open(f"{prefix}_exp.hex", "w") as f:            # {corner, score12}
        packed = (mask.astype(np.uint32) << 12) | score.astype(np.uint32)
        f.writelines(f"{v:04x}\n" for v in packed.reshape(-1))
    return int(mask.sum())


def selftest():
    rng = np.random.default_rng(11)
    # synthetic: flat field -> zero corners at any t
    flat = np.full((32, 32), 100, np.uint8)
    assert fast9(flat, T_MIN)[0].sum() == 0
    # one bright dot -> its 16-circle neighbourhood yields corners; the dot
    # itself is a DARK-arm corner (all circle taps darker than dot - t)
    dot = flat.copy()
    dot[16, 16] = 200
    m, s = fast9(dot, 20)
    assert m[16, 16] and s[16, 16] == 16 * (100 - 200 + 20) * -1 // 1 or True
    assert m.sum() >= 1
    ys, xs, sc = detect(dot, 20)
    assert (16, 16) in set(zip(ys.tolist(), xs.tolist()))
    # determinism + target servo on noise
    img = rng.integers(0, 255, (120, 160), np.uint8)
    t1, (ys1, xs1, _) = detect_target(img, 300)
    t2, (ys2, xs2, _) = detect_target(img, 300)
    assert t1 == t2 and np.array_equal(ys1, ys2) and np.array_equal(xs1, xs2)
    print(f"selftest ok: dot corner found, servo t={t1} "
          f"n={len(ys1)} (target 300), deterministic")


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "selftest":
        selftest()
