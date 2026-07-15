"""sspax — a JAX re-implementation of the SSP lattice algebra, for fast 6-DoF
experimentation (encode / rotate-by-permutation / SO(3) decode on GPU).

This is an *efficient experimentation surface*, NOT a shipped deliverable and
NOT a replacement for the frozen `sspslam/` core. It imports the house
conventions (LAMS ladder, half-sphere phasor convention, nearest-direction
rotation permutation) and reproduces the numpy `experiments.lattice3d`
algebra bit-for-bit (`python3 -m sspax.bench parity`), then runs the same
studies orders of magnitude faster under jit/vmap on the accelerator.

New here (2026-07-15 directive): `sphere.py` — the *approximate-permutable*
Fibonacci-sphere substitute. Points ~equidistant on the sphere (fib3d's virtue)
built as latitude rings with a half-azimuth stagger between layers, while each
ring stays exactly yaw-permutable (azel3d's virtue). A common yaw is therefore
*approximately* an index permutation — the deliberate middle ground.
"""
from sspax.core import (
    LAMS, N_DIR, encode, encode_batch, make_lattices, W_of,
    dirs_az, dirs_fib, dirs_azel, dirs_rand,
    perm_of, apply_perm, so3_perms, q_polar,
)
from sspax.sphere import (
    dirs_ringstag, ring_layout, nn_uniformity, yaw_perm,
)

__all__ = [
    "LAMS", "N_DIR", "encode", "encode_batch", "make_lattices", "W_of",
    "dirs_az", "dirs_fib", "dirs_azel", "dirs_rand",
    "perm_of", "apply_perm", "so3_perms", "q_polar",
    "dirs_ringstag", "ring_layout", "nn_uniformity", "yaw_perm",
]
