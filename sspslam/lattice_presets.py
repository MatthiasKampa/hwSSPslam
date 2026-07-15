"""Lattice configuration helpers — the ONE committed home for lattice patching.

The shipped lattice lives as module globals in `ssp_slam_loop` (LAMS, N_ANG,
N_RING, MAIN, W, ENC.W, ENC_MAIN.W, _RINGS, WIDE, WIDER). Experiments that
sweep it each carried a private copy of the patch code
(`scratch_lattice_sweep.set_lattice`, `scratch_binscale.set_lattice_n`,
`ssp_hexreal.set_lattice_hex`); this module consolidates them. Patch-only —
no shipped module is edited (PROTOCOL 1); every setter leaves the globals in
a state the shipped pipeline runs on, and `shipped()` restores the exact
import-time state.

Conventions (from the original studies):
  polar  half-circle direction grid (conjugate-wrap rotation permutation),
         directions at k*pi/n_ang — the shipped layout.
  hex    FULL-circle direction grid (plain-shift rotation permutation),
         n_ang = 3k odd for exact 120-deg triplets — see ssp_hexreal.
  Matched rings first (MAIN), relocalization rings last. For n_m matched
  rings: WIDE = rings >= n_m-2, WIDER = rings >= n_m-1 (shipped n_m=4 =>
  WIDE>=2, WIDER>=3, verified in selftest).

NOTE: the snapshot is taken at import; import ssp_lattice BEFORE patching
lattice globals by other means.
"""
import numpy as np

import sspslam.lattice as L

OCT = (0.25, 0.5, 1.0, 2.0)      # shipped matched-ring ladder (octaves)
RELO = (5.3, 12.8)               # shipped relocalization rings

_SHIPPED = dict(LAMS=L.LAMS.copy(), N_ANG=L.N_ANG, N_RING=L.N_RING,
                MAIN=L.MAIN, W=L.W.copy(), _RINGS=L._RINGS.copy(),
                WIDE=L.WIDE.copy(), WIDER=L.WIDER.copy())


def _apply(lams_matched, n_ang, relo, full_circle):
    n_m = len(lams_matched)
    L.LAMS = np.array(list(lams_matched) + list(relo))
    L.N_ANG = n_ang
    L.N_RING = n_m + len(relo)
    L.MAIN = slice(0, n_m * n_ang)
    span = 2 * np.pi if full_circle else np.pi
    a = np.arange(n_ang) * span / n_ang
    u = np.stack([np.cos(a), np.sin(a)], 1)
    L.W = np.concatenate([(2 * np.pi / lam) * u for lam in L.LAMS])
    L.ENC.W = L.W
    L.ENC_MAIN.W = L.W[L.MAIN]
    L._RINGS = np.repeat(np.arange(L.N_RING), n_ang)
    L.WIDE = L._RINGS >= n_m - 2
    L.WIDER = L._RINGS >= n_m - 1
    return n_m


def set_polar(lams_matched=OCT, n_ang=60, relo=RELO):
    """Half-circle grid (shipped layout). Returns n_matched."""
    return _apply(lams_matched, n_ang, relo, full_circle=False)


def set_hex(n_ang, lams_matched=OCT, relo=RELO):
    """Full-circle grid for hex/plain-shift permutation (ssp_hexreal).
    n_ang should be 3k odd (exact 120-deg triplets, no negation dupes)."""
    return _apply(lams_matched, n_ang, relo, full_circle=True)


def shipped():
    """Restore the exact import-time (shipped) lattice globals."""
    L.LAMS = _SHIPPED["LAMS"].copy()
    L.N_ANG = _SHIPPED["N_ANG"]
    L.N_RING = _SHIPPED["N_RING"]
    L.MAIN = _SHIPPED["MAIN"]
    L.W = _SHIPPED["W"].copy()
    L.ENC.W = L.W
    L.ENC_MAIN.W = L.W[L.MAIN]
    L._RINGS = _SHIPPED["_RINGS"].copy()
    L.WIDE = _SHIPPED["WIDE"].copy()
    L.WIDER = _SHIPPED["WIDER"].copy()


def describe():
    n_m = int(L.MAIN.stop // L.N_ANG)
    return (f"{L.N_RING} rings x {L.N_ANG} ang (D={len(L.W)}, "
            f"MAIN={L.MAIN.stop}), lams={np.round(L.LAMS, 3).tolist()}, "
            f"span={'2pi' if _is_full() else 'pi'}, n_matched={n_m}")


def _is_full():
    # direction of the last lattice angle distinguishes pi vs 2pi span
    a1 = np.arctan2(L.W[L.N_ANG - 1, 1], L.W[L.N_ANG - 1, 0]) % (2 * np.pi)
    return a1 > np.pi


# --------------------------------------------------------------------------
#  venue-adaptive place ladder (2026-07-15) — the deployable static cousin
#  of the learned scale head. sspax/ladder_extent.py measured the mechanism
#  behind the multi-venue verdict (adopted azel-oct6 wins rooms, ring-
#  coarse16 wins buildings): the best coarsest wavelength TRACKS THE VENUE
#  EXTENT (extent 8 -> lam_max 8, 30 -> 22.6, 80 -> 90.5 on the synthetic
#  diagonal). Both ADOPTED ladders are instances of one rule,
#      ladder = geomspace(lam_min, EXTENT_C * extent, n_rings):
#  oct6 = geomspace(0.25, 8, 6) at extent 8; LAMC16 = geomspace(0.5, 90.5,
#  16) at extent ~90.
#  P5 DENSIFIED-FIT VERDICT (sspax/ladder_rule.py, banked 2026-07-15): the
#  law is MONOTONE (Spearman 0.90-0.97) but NOT linear — the through-origin
#  constant is not identifiable (0.75 at n_rings=6 / 1.25 at n_rings=8,
#  R^2 0.39/0.78; ring count rescales every rung, confounding the fit), so
#  EXTENT_C stays 1.0 and will NOT be "refined" from that surface. The rule
#  is retained because only its REACH constraint binds: ladder choice is
#  nearly free below ~50 m (AUC spread 0.03-0.09) and matters hugely at
#  building scale (0.21-0.22 at 80-120 m, where a too-fine ladder aliases
#  catastrophically) — lam_max = extent guarantees scene-scale reach there
#  and is harmless where the choice is free, i.e. the continuous formula
#  IS the recommended coarse bucket in behavior. Caveat: ring-count
#  invariance did NOT hold in the dense sweep; the preset is calibrated at
#  the two adopted (n_rings, lam_min) combos above.
#  Extent comes from the OWN map bbox (est-registered points) — never from
#  reference poses (anti-oracle).
# --------------------------------------------------------------------------
EXTENT_C = 1.0                   # not identifiable beyond ~[0.75, 1.25]; keep 1
COH_LAM_MIN = 0.25               # sensor-coherence floor (lam_min ~ 2*pi*sig_r)


def ladder_of_extent(extent_m, n_rings=6, lam_min=COH_LAM_MIN, c=EXTENT_C):
    """Place-ladder preset for a venue of horizontal extent `extent_m`
    (map bbox max span, own-estimate-registered): geometric ladder from
    the sensor-coherence floor up to the venue scale. Returns a lam list
    for the 3D place lattice builders (azel/ring x ladder)."""
    lam_max = max(c * float(extent_m), 2.0 * lam_min)
    return list(np.geomspace(lam_min, lam_max, n_rings))


def extent_of_points(xy, lo=2.0, hi=98.0):
    """Robust horizontal extent (max bbox span, percentile-fenced) of
    own-estimate-registered map points or trajectory. Deployable: uses
    the map's own content, no reference."""
    p = np.asarray(xy, float).reshape(-1, np.asarray(xy).shape[-1])[:, :2]
    span = np.percentile(p, hi, axis=0) - np.percentile(p, lo, axis=0)
    return float(np.max(span))


def selftest_extent():
    oct6 = ladder_of_extent(8.0)
    assert np.allclose(oct6, [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]), oct6
    print("  ladder_of_extent(8) == the adopted azel-oct6 ladder (exact)")
    c16 = ladder_of_extent(90.5, n_rings=16, lam_min=0.5)
    assert len(c16) == 16 and abs(c16[0] - 0.5) < 1e-9 \
        and abs(c16[-1] - 90.5) < 1e-6
    r = np.diff(np.log(c16))
    assert np.allclose(r, r[0]), "geometric spacing"
    assert abs(np.exp(r[0]) - np.sqrt(2)) < 1e-3   # half-octave steps
    print("  ladder_of_extent(90.5, 16, 0.5) == the adopted ring-coarse16 "
          "ladder (half-octave steps)")
    rng = np.random.default_rng(7)
    pts = rng.uniform([-4, -3], [4, 3], (5000, 2))
    e = extent_of_points(pts)
    assert 7.0 < e < 8.1, e
    lam = ladder_of_extent(0.1)
    assert lam[-1] >= 2 * COH_LAM_MIN     # degenerate-extent floor
    print("  extent_of_points on an 8x6 box -> %.2f; degenerate floor ok"
          % e)
    print("selftest_extent ok")


def selftest():
    set_polar()
    for k, v in _SHIPPED.items():
        cur = getattr(L, k)
        if isinstance(v, np.ndarray):
            assert np.array_equal(cur, v), k
        else:
            assert cur == v, k
    assert np.array_equal(L.ENC.W, _SHIPPED["W"])
    assert np.array_equal(L.ENC_MAIN.W, _SHIPPED["W"][_SHIPPED["MAIN"]])
    print("  set_polar() == shipped globals (exact)")
    n_m = set_polar((0.125, 0.25, 0.5, 1.0, 2.0), 60)
    assert n_m == 5 and L.MAIN.stop == 300 and len(L.W) == 420
    assert L.WIDE.sum() == 4 * 60 and L.WIDER.sum() == 3 * 60
    print("  5-matched-ring polar layout consistent")
    set_hex(63)
    assert len(L.W) == 6 * 63 and _is_full()
    import sspslam.encoder as S
    rng = np.random.default_rng(3)
    pts = rng.uniform(-4, 4, (200, 2))
    w = np.full(200, 0.1)
    for m in (5, 17, 61):
        th = m * 2 * np.pi / 63
        a = np.exp(1j * ((pts @ S._rot(th).T) @ L.W.T)).T @ w
        A = (np.exp(1j * (pts @ L.W.T)).T @ w).reshape(-1, 63)
        b = A[:, (np.arange(63) - m) % 63].reshape(-1)
        assert np.allclose(a, b, atol=1e-9), m
    print("  hex63: rotation == plain-permutation")
    shipped()
    assert np.array_equal(L.W, _SHIPPED["W"]) and L.N_ANG == 60
    print("  shipped() restores import-time state")
    print("selftest ok")


if __name__ == "__main__":
    selftest()
    selftest_extent()
