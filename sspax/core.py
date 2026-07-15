"""Computational core: lattice construction (numpy, tiny) + the hot algebra
(encode / rotation-permutation / SO(3) decode) in JAX.

Parity contract (checked in `sspax.bench parity`): `encode`, the lattice
constructors, and `perm_of` reproduce `experiments.lattice3d` to <1e-9 in
float64. We enable x64 so the port is bit-faithful; the benches expose a
float32 fast path where only speed matters.

House conventions carried over verbatim from experiments.lattice3d:
  phi(P) = sum_i exp(i W . p_i),  W rows = (2*pi/lam) * u_dir   (u on the
  upper half sphere; the antipode is the complex conjugate, free).
  Rotation R acts as a nearest-direction index permutation of W rows, with a
  sign flip where the half-sphere wraps to its antipode. p' = p @ R.T  <=>
  W' rows = W @ R.
"""
import os

# --- GPU courtesy: cap our footprint so a second agent can share the card.
# Must be set BEFORE jax/XLA initializes. Uses <=50% of VRAM (user directive:
# leave 50% headroom); honours any pre-set env override.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)   # bit-faithful to the numpy core
import jax.numpy as jnp
from functools import partial

import sspslam.lattice as L

# matched ladder (first 4 unique house wavelengths) -> D = 4 * N_DIR = 240
LAMS = [float(x) for x in np.unique(np.round(
    np.asarray(L.LAMS, float), 6))][:4] or [0.5, 1.0, 2.0, 4.0]
N_DIR = 60


# --------------------------------------------------------------------------
#  lattice layouts  (numpy — construction is one-shot and small)
# --------------------------------------------------------------------------
def dirs_az(n=N_DIR):
    """n azimuths in the xy-plane, half-circle (house 2D convention)."""
    a = np.arange(n) * (np.pi / n)
    return np.stack([np.cos(a), np.sin(a), np.zeros(n)], 1)


def dirs_fib(n=N_DIR):
    """Fibonacci spiral on the upper half sphere — isotropic, NOT permutable."""
    g = (1 + 5 ** 0.5) / 2
    k = np.arange(n) + 0.5
    z = k / n
    r = np.sqrt(1 - z * z)
    th = 2 * np.pi * k / g
    return np.stack([r * np.cos(th), r * np.sin(th), z], 1)


def dirs_azel(n_az=12, elevs_deg=(-40, -20, 0, 20, 40)):
    """azimuth rings x elevation bands — exact common yaw, but anisotropic
    (constant az count crowds points toward the poles)."""
    out = []
    for e in np.deg2rad(elevs_deg):
        c, s = np.cos(e), np.sin(e)
        a = np.arange(n_az) * (np.pi / n_az)
        out.append(np.stack([c * np.cos(a), c * np.sin(a),
                             np.full(n_az, s)], 1))
    return np.concatenate(out)


def dirs_rand(n=N_DIR, seed=11):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v[v[:, 2] < 0] *= -1.0
    return v


def W_of(dirs, lams=LAMS):
    return np.concatenate([(2 * np.pi / lam) * np.asarray(dirs, float)
                           for lam in lams])


def make_lattices(lams=LAMS):
    """The catalogued set + the new ringstag3d, all at D = 4*N_DIR = 240."""
    from sspax.sphere import dirs_ringstag
    return {
        "az2d":      W_of(dirs_az(N_DIR), lams),
        "fib3d":     W_of(dirs_fib(N_DIR), lams),
        "azel3d":    W_of(dirs_azel(12, [-40, -20, 0, 20, 40]), lams),
        "ringstag3d": W_of(dirs_ringstag(N_DIR), lams),
        "rand3d":    W_of(dirs_rand(N_DIR), lams),
    }


# --------------------------------------------------------------------------
#  encode  (JAX — the hot path)
# --------------------------------------------------------------------------
@partial(jax.jit, static_argnames=())
def _encode_unw(W, pts):
    v = jnp.exp(1j * (pts @ W.T)).sum(0)
    return v / jnp.maximum(jnp.linalg.norm(v), 1e-12)


@partial(jax.jit, static_argnames=())
def _encode_w(W, pts, w):
    v = jnp.exp(1j * (pts @ W.T)).T @ w.astype(pts.dtype)
    return v / jnp.maximum(jnp.linalg.norm(v), 1e-12)


def encode(W, pts, w=None, quant=None):
    """phi(P) normalized, matching experiments.lattice3d.encode bit-for-bit.
    Accepts numpy or jax arrays; returns a jax array. quant=(nph, nmag)
    applies the house polar quantizer (FPGA storage model) to the result."""
    W = jnp.asarray(W)
    pts = jnp.asarray(pts)
    v = _encode_unw(W, pts) if w is None else _encode_w(W, pts, jnp.asarray(w))
    if quant is not None:
        nph, nmag = quant
        v = q_polar(v, nph, nmag)
        v = v / jnp.maximum(jnp.linalg.norm(v), 1e-12)
    return v


# --------------------------------------------------------------------------
#  polar quantization  (FPGA storage model — mirrors sspslam.quantized.q_polar)
# --------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("nph", "nmag"))
def q_polar(v, nph, nmag):
    """nph phase bins (mid-tread) x nmag magnitude levels; per-vector scale =
    99th-percentile magnitude. Bit-faithful to sspslam.quantized.q_polar."""
    mag = jnp.abs(v)
    s = jnp.percentile(mag, 99.0) + 1e-12
    dph = 2 * jnp.pi / nph
    phq = dph * jnp.round(jnp.angle(v) / dph)
    mstep = s / nmag
    magq = jnp.clip(mstep * (jnp.floor(jnp.clip(mag, 0, s) / mstep) + 0.5), 0, s)
    return magq * jnp.exp(1j * phq)


def q_polar_np(v, nph, nmag):
    """numpy reference (exact copy of sspslam.quantized.q_polar) for parity."""
    s = float(np.percentile(np.abs(v), 99)) + 1e-12
    dph = 2 * np.pi / nph
    phq = dph * np.round(np.angle(v) / dph)
    mstep = s / nmag
    magq = np.clip(mstep * (np.floor(np.clip(np.abs(v), 0, s) / mstep) + 0.5),
                   0, s)
    return magq * np.exp(1j * phq)


@partial(jax.jit, static_argnames=())
def _encode_raw(W, pts):
    """Unnormalized bundle (for the derivative-correction algebra)."""
    return jnp.exp(1j * (pts @ W.T)).sum(0)


@partial(jax.jit, static_argnames=())
def _encode_dvda_raw(W, pts):
    """Azimuthal (yaw) derivative d(phi)/d(alpha), unnormalized. The yaw
    generator rotates each frequency's xy part by 90deg, so the derivative
    frequency is W' = (-W_y, W_x, 0):  dv_f/dalpha = sum_p i (W'_f . p) e^{i W_f . p}."""
    Wp = jnp.stack([-W[:, 1], W[:, 0], jnp.zeros(W.shape[0], W.dtype)], 1)
    ph = jnp.exp(1j * (pts @ W.T))
    proj = pts @ Wp.T
    return (1j * proj * ph).sum(0)


def encode_raw(W, pts):
    return _encode_raw(jnp.asarray(W), jnp.asarray(pts))


def encode_dvda(W, pts):
    """Stored yaw-derivative vector (same D as the encoding); enables
    continuous sub-grid yaw on a FROZEN encoding via a first-order correction."""
    return _encode_dvda_raw(jnp.asarray(W), jnp.asarray(pts))


@partial(jax.jit, static_argnames=())
def _encode_batch(W, P):
    # P: (B, M, 3) fixed-size clouds -> (B, D) normalized
    v = jnp.exp(1j * jnp.einsum("bmd,kd->bmk", P, W)).sum(1)
    return v / jnp.maximum(jnp.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def encode_batch(W, P):
    """Batched encode of fixed-size clouds P (B,M,3) on the accelerator."""
    return _encode_batch(jnp.asarray(W), jnp.asarray(P))


# --------------------------------------------------------------------------
#  rotation as a nearest-direction index permutation  (numpy solver)
# --------------------------------------------------------------------------
def perm_of(W, R):
    """(perm, sgn, exact, resid) for rotation R, matching lattice3d._perm_of.
    perm: rows of W@R matched to nearest row of W (or its antipode);
    sgn (bool): True where the match is direct, False where antipodal (conj);
    exact: whether the match is a true permutation (max residual ~ 0);
    resid: the worst matched-direction gap (radians-ish, in frequency units)."""
    W = np.asarray(W, float)
    WR = W @ np.asarray(R, float)
    dp = np.linalg.norm(WR[:, None] - W[None], axis=2)
    dm = np.linalg.norm(WR[:, None] + W[None], axis=2)
    d = np.minimum(dp, dm)
    perm = d.argmin(1)
    ar = np.arange(len(W))
    sgn = dp[ar, perm] <= dm[ar, perm]
    resid = float(d[ar, perm].max())
    exact = resid < 1e-9 * np.linalg.norm(W, axis=1).max()
    return perm, sgn, exact, resid


@partial(jax.jit, static_argnames=())
def apply_perm(V, perm, sgn):
    """Rotate a FROZEN encoding by permuting its lattice indices (+ conj on the
    antipodal wrap), then renormalize. V (..., D); perm/sgn (D,)."""
    Vp = jnp.where(sgn, V[..., perm], jnp.conj(V[..., perm]))
    return Vp / jnp.maximum(jnp.linalg.norm(Vp, axis=-1, keepdims=True), 1e-12)


def _axis_rot(axis, deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    if axis == "yaw":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    if axis == "pitch":
        return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def so3_perms(W, grid_deg=range(-15, 16, 5)):
    """Precompute permutation tables for an Euler yaw/pitch/roll grid — the
    expensive-once step that turns SO(3) decode into a batched gather. Returns
    (Rg (G,3,3), perms (G,D) int32, sgns (G,D) bool, exact (G,) bool)."""
    degs = list(grid_deg)
    grid = [(y, p, r) for y in degs for p in degs for r in degs]
    Rg, perms, sgns, exact = [], [], [], []
    for y, p, r in grid:
        R = _axis_rot("yaw", y) @ _axis_rot("pitch", p) @ _axis_rot("roll", r)
        pm, sg, ex, _ = perm_of(W, R)
        Rg.append(R)
        perms.append(pm.astype(np.int32))
        sgns.append(sg)
        exact.append(ex)
    return (np.array(Rg), np.array(perms), np.array(sgns), np.array(exact))
