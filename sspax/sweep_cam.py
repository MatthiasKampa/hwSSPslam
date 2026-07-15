"""Batch 2 — CAMERA geometry sweep on synthetic bearing vectors (S^2).

Carries a curated geometry subset from batch 1 (the uniformity / permutability /
decode representatives) and tests them on camera bearings: SO(3) decode on an
OMNI camera (rotation-equivariant, the sphere lattice's home turf) and place
separability for OMNI vs NARROW FOV. Float + quantized. Question: does FOV
concentration change the geometry winner, and is the sphere lattice's isotropy
worth it for appearance place-rec?

  PYTHONPATH=. python3 -m sspax.sweep_cam
"""
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from sspax import core as C
from sspax import sphere as SPH
from sspax import worlds as Wd
from sspax import fusion as F
from sspax.sweep import m_uniform, m_so3, SEEDS, N_SCENES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch" / "sweep_b2_cam.csv"

# curated from batch 1: decode-winner, balanced baseline, uniformity ref,
# best-uniformity ring, default ringstag, best-permutability ring, area variant
GEOMS = {
    "rand3d":     lambda n: C.dirs_rand(n),
    "fib3d":      lambda n: C.dirs_fib(n),
    "azel3d":     lambda n: C.dirs_azel(12, [-40, -20, 0, 20, 40]),
    "ring_arc_cos_s0_r4":   lambda n: SPH.dirs_ring(n, spacing="arc",
                                                    apportion="cos", stagger=0, n_rings=4),
    "ring_arc_cos_s0.5_r5": lambda n: SPH.dirs_ring(n, spacing="arc",
                                                    apportion="cos", stagger=0.5, n_rings=5),
    "ring_arc_const_s0.5_r6": lambda n: SPH.dirs_ring(n, spacing="arc",
                                                      apportion="const", stagger=0.5, n_rings=6),
    "ring_area_cos_s0.5_r6": lambda n: SPH.dirs_ring(n, spacing="area",
                                                     apportion="cos", stagger=0.5, n_rings=6),
}

QUANTS = [None, (16, 4)]
HDR = ("config,quant,uni_cv,uni_min,so3omni_med,so3omni_p90,"
       "placeOmni_auc,placeNarrow_auc,secs")


def _bearings_by_seed(fov=(360.0, 120.0)):
    out = []
    for s in SEEDS:
        B, Wt = Wd.bearing_batch(N_SCENES, 400, seed0=s, fov_deg=fov)
        # m_so3 wants a list of clouds; carry weights implicitly (uniform for
        # decode — rotation acts on directions, weights are viewpoint-fixed)
        out.append([B[i] for i in range(len(B))])
    return out


def run():
    OUT.write_text(HDR + "\n")
    bears = _bearings_by_seed()
    pl_omni = [F.gen_places(N_SCENES, 2, seed0=s, yaw_jit=40,
                            fov=(360.0, 120.0)) for s in SEEDS[:3]]
    pl_narrow = [F.gen_places(N_SCENES, 2, seed0=s, yaw_jit=40,
                              fov=(90.0, 60.0)) for s in SEEDS[:3]]
    t0 = time.time()
    print(f"batch2 cam: {len(GEOMS)} geoms x {len(QUANTS)} quant -> {OUT}",
          flush=True)
    for name, fn in GEOMS.items():
        dirs = fn(60)
        Wl = jnp.asarray(C.W_of(dirs))
        cv, mn = m_uniform(dirs)
        for quant in QUANTS:
            ts = time.time()
            sm, sp = m_so3(Wl, bears, quant, range(-15, 16, 3))
            ao = np.mean([F.place_auc(np.asarray(Wl), p, mode="cam",
                                      quant=quant)[0] for p in pl_omni])
            an = np.mean([F.place_auc(np.asarray(Wl), p, mode="cam",
                                      quant=quant)[0] for p in pl_narrow])
            q = "float" if quant is None else f"{quant[0]}ph{quant[1]}mag"
            row = [name, q, f"{cv:.4f}", f"{mn:.2f}", f"{sm:.2f}", f"{sp:.2f}",
                   f"{ao:.4f}", f"{an:.4f}", f"{time.time()-ts:.1f}"]
            with open(OUT, "a") as f:
                f.write(",".join(str(x) for x in row) + "\n")
            print(f"  [{time.time()-t0:5.0f}s] {name:24s} {q:9s} "
                  f"so3omni {sm:.1f}  placeOmni {ao:.3f}  "
                  f"placeNarrow {an:.3f}", flush=True)
    print(f"batch2 done: {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    print(f"jax devices: {jax.devices()}", flush=True)
    run()
