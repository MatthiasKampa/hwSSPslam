"""ringstag3d — the approximate-permutable Fibonacci-sphere substitute.

Motivation (user directive 2026-07-15): fib3d covers the sphere isotropically
(points ~equidistant) but a yaw rotation is NOT an index permutation of it;
azel3d keeps the exact yaw permutation but wastes directions near the poles
(constant azimuth count per band). The reconciliation the user asked for:

    "structure a sphere out of rings with half offset between layers to have
     approx equal distance between points"

Construction:
  * latitude rings on the UPPER half sphere (elev in (0, pi/2), full 2*pi
    azimuth per ring — the antipode is the phasor conjugate, free, exactly as
    in fib3d);
  * per-ring azimuth count proportional to the ring circumference cos(elev),
    apportioned (largest-remainder) to hit an exact direction budget, so the
    nearest-neighbour arc is ~constant across the sphere (fib3d's virtue);
  * consecutive rings staggered by half an azimuth step (brick-laying) so
    neighbouring rings interleave rather than align — tightens the packing.

The algebra this buys:
  * each ring is equal-azimuth over a FULL circle -> a yaw by a multiple of
    that ring's step is an EXACT cyclic shift (a true permutation, no
    conjugation, unlike the half-circle 2D lattice);
  * rings have DIFFERENT counts, so a common yaw is exact for all rings only at
    theta = 0 -> a general yaw is *approximately* a permutation: snap each ring
    to its nearest integer shift, residual <= pi/n_az(ring). `yaw_perm` returns
    that per-ring-snapped permutation and the worst residual.

This is the deliberate middle ground: more isotropic than azel3d, far more
permutable than fib3d.
"""
import numpy as np


# --------------------------------------------------------------------------
#  layout
# --------------------------------------------------------------------------
def _apportion(weights, total, floor=1):
    """Largest-remainder apportionment of `total` across bins ~ weights, each
    bin >= floor. Deterministic."""
    weights = np.asarray(weights, float)
    n = len(weights)
    assert total >= floor * n, (total, floor, n)
    ideal = weights / weights.sum() * (total - floor * n)
    base = np.floor(ideal).astype(int) + floor
    short = total - int(base.sum())
    if short > 0:                       # hand extras to the largest remainders
        rem = ideal - np.floor(ideal)
        order = np.argsort(-rem, kind="stable")
        base[order[:short]] += 1
    elif short < 0:                     # take from the smallest remainders
        rem = ideal - np.floor(ideal)
        order = np.argsort(rem, kind="stable")
        take, i = -short, 0
        while take > 0:
            j = order[i % n]
            if base[j] > floor:
                base[j] -= 1
                take -= 1
            i += 1
    return base


def _default_rings(n_total):
    """Rings chosen so meridian spacing ~ azimuth spacing (hex-ish packing).
    d ~ nearest-neighbour arc for n points on the half sphere (area 2*pi)."""
    d = np.sqrt(4 * np.pi / (np.sqrt(3.0) * n_total))   # hex cell edge
    return int(max(2, round((np.pi / 2) / d)))


def ring_layout(n_total, n_rings=None, *, spacing="arc", apportion="cos",
                stagger=0.5):
    """Return a ring-sphere layout hitting EXACTLY n_total directions on the
    upper half sphere. Knobs (the sweep axes):
      spacing   'arc'  equal elevation spacing (equal meridian arc between rings)
                'area' equal-area bands (equal spacing in z = sin(elev))
      apportion 'cos'  per-ring azimuth count ~ circumference cos(elev)
                       (=> ~equal nearest-neighbour distance)
                'const' equal azimuth count per ring (the azel3d convention)
      stagger   fraction of an azimuth step to offset alternate rings (0 or 0.5)
    Keys: elevs (R,), n_az (R,) int, offsets (R,), ring_id/local (n_total,),
    n_rings."""
    if n_rings is None:
        n_rings = _default_rings(n_total)
    u = (np.arange(n_rings) + 0.5) / n_rings
    if spacing == "area":
        elevs = np.arcsin(u)                     # equal area <=> equal z
    else:
        elevs = u * (np.pi / 2)                  # equal arc
    if apportion == "const":
        ideal = np.ones(n_rings)
    else:
        ideal = np.cos(elevs)                    # ~ ring circumference
    n_az = _apportion(ideal, n_total, floor=1)
    offsets = np.where(np.arange(n_rings) % 2 == 1, stagger, 0.0) \
        * (2 * np.pi / n_az)
    ring_id = np.repeat(np.arange(n_rings), n_az)
    local = np.concatenate([np.arange(k) for k in n_az])
    return dict(elevs=elevs, n_az=n_az, offsets=offsets, ring_id=ring_id,
                local=local, n_rings=int(n_rings))


def dirs_ring(n_total, n_rings=None, **kw):
    """(n_total, 3) ring-major directions on the upper half sphere; **kw are
    the ring_layout knobs (spacing / apportion / stagger)."""
    lay = ring_layout(n_total, n_rings, **kw)
    out = []
    for r, (elev, k, off) in enumerate(zip(lay["elevs"], lay["n_az"],
                                           lay["offsets"])):
        a = off + np.arange(k) * (2 * np.pi / k)
        c, s = np.cos(elev), np.sin(elev)
        out.append(np.stack([c * np.cos(a), c * np.sin(a),
                             np.full(k, s)], 1))
    return np.concatenate(out)


def dirs_ringstag(n_total, n_rings=None):
    """The default 2026-07-15 layout: equal-arc rings, cos apportionment, half
    stagger. Backward-compatible shorthand for dirs_ring(...)."""
    return dirs_ring(n_total, n_rings, spacing="arc", apportion="cos",
                     stagger=0.5)


# --------------------------------------------------------------------------
#  the approximate-permutable yaw
# --------------------------------------------------------------------------
def yaw_perm(n_total, theta, n_rings=None, n_lams=1, **kw):
    """Per-ring-snapped yaw permutation of a ringstag encoding.

    Each ring is cyclically shifted by the integer nearest to theta/(ring step);
    a full circle => a pure permutation, NO sign flip. Returns:
      perm  (n_total * n_lams,) int   index map (tiled across the lam blocks),
      sgn   (n_total * n_lams,) bool  all True (kept for a uniform apply_perm),
      resid float                     worst per-ring snap residual (radians).
    Rotating a FROZEN vector: v_rot = apply_perm(v, perm, sgn) (see core)."""
    lay = ring_layout(n_total, n_rings, **kw)
    perm1 = np.empty(n_total, np.int64)
    base = 0
    resid = 0.0
    for k in lay["n_az"]:
        step = 2 * np.pi / k
        m = int(round(theta / step))
        idx = base + (np.arange(k) - m) % k
        perm1[base:base + k] = idx
        resid = max(resid, abs(theta - m * step))
        base += k
    # tile the direction-level permutation across the lam blocks (ring-major
    # W = [lam0 dirs | lam1 dirs | ...], each block the same n_total ordering)
    perm = np.concatenate([perm1 + b * n_total for b in range(n_lams)])
    sgn = np.ones(n_total * n_lams, bool)
    return perm, sgn, float(resid)


def yaw_residuals(n_total, theta, n_rings=None, n_lams=1, **kw):
    """Per-component sub-step yaw residual delta_r (radians) left AFTER the
    integer per-ring shift of `yaw_perm`. Same ring-major tiling across lams.
    A first-order correction v_rot ~= shift(v) - delta * shift(dv/dalpha)
    removes it (see bench rot, ringstag3d derivative row)."""
    lay = ring_layout(n_total, n_rings, **kw)
    delta1 = np.empty(n_total, float)
    base = 0
    for k in lay["n_az"]:
        step = 2 * np.pi / k
        m = round(theta / step)
        delta1[base:base + k] = theta - m * step
        base += k
    return np.tile(delta1, n_lams)


# --------------------------------------------------------------------------
#  uniformity metric
# --------------------------------------------------------------------------
def nn_uniformity(dirs):
    """Nearest-neighbour angular-distance stats over the FULL direction set
    (each u and its antipode -u, since the phasor encodes both). Lower cv and
    higher min => more equidistant. Returns dict(min, mean, cv) in degrees /
    dimensionless."""
    d = np.asarray(dirs, float)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    full = np.concatenate([d, -d])                    # antipodes are real dirs
    cos = np.clip(full @ full.T, -1, 1)
    np.fill_diagonal(cos, -1.0)
    ang = np.degrees(np.arccos(cos))
    nn = ang.min(1)
    return dict(min=float(nn.min()), mean=float(nn.mean()),
                cv=float(nn.std() / nn.mean()))
