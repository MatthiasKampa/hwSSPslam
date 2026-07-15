"""Batch 3 — COMBINED cam+lidar fusion on a SHARED lattice.

Sweeps the bundle weight alpha (0=lidar only .. 1=cam only), bundle (add, D —
memory-free) vs concat (2D), for OMNI and NARROW cameras, float + quant.
Question: is there an alpha where the fused place-separability beats BOTH single
channels, and does bundle match concat at half the storage?

  PYTHONPATH=. python3 -m sspax.sweep_fuse
"""
import time
from pathlib import Path

import numpy as np
import jax

from sspax import core as C
from sspax import sphere as SPH
from sspax import fusion as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch" / "sweep_b3_fuse.csv"

# shared backbone geometries (carry a permutability-leaning and a uniformity-
# leaning ring, plus azel as the balanced baseline)
GEOMS = {
    "azel3d": C.dirs_azel(12, [-40, -20, 0, 20, 40]),
    "ring_arc_cos_s0.5_r5": SPH.dirs_ring(60, spacing="arc", apportion="cos",
                                          stagger=0.5, n_rings=5),
    "ring_arc_const_s0.5_r6": SPH.dirs_ring(60, spacing="arc", apportion="const",
                                            stagger=0.5, n_rings=6),
}
ALPHAS = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
FOVS = {"omni": (360.0, 120.0), "narrow": (90.0, 60.0)}
QUANTS = [None, (16, 4)]
SEEDS = (0, 100, 200)
K = 24
HDR = "config,fov,quant,mode,alpha,auc,secs"


def run():
    OUT.write_text(HDR + "\n")
    places = {fov: [F.gen_places(K, 2, seed0=s, yaw_jit=40, fov=FOVS[fov])
                    for s in SEEDS] for fov in FOVS}
    t0 = time.time()
    print(f"batch3 fuse: {len(GEOMS)} geoms x {len(FOVS)} fov x {len(ALPHAS)} "
          f"alpha -> {OUT}", flush=True)
    for name, dirs in GEOMS.items():
        W = C.W_of(dirs)
        for fov in FOVS:
            for quant in QUANTS:
                q = "float" if quant is None else f"{quant[0]}ph{quant[1]}mag"
                # single-channel references (alpha-independent)
                for mode in ("lidar", "cam"):
                    ts = time.time()
                    auc = np.mean([F.place_auc(W, p, mode=mode, quant=quant)[0]
                                   for p in places[fov]])
                    _row(name, fov, q, mode, "-", auc, time.time() - ts)
                # fused: bundle across alpha, + concat at alpha 0.5
                for concat in (False, True):
                    alphas = ALPHAS if not concat else [0.5]
                    for a in alphas:
                        ts = time.time()
                        auc = np.mean([F.place_auc(W, p, mode="fused", alpha=a,
                                                   quant=quant, concat=concat)[0]
                                       for p in places[fov]])
                        mode = "concat" if concat else "bundle"
                        _row(name, fov, q, mode, a, auc, time.time() - ts)
            print(f"  [{time.time()-t0:5.0f}s] {name} {fov} done", flush=True)
    print(f"batch3 done: {time.time()-t0:.0f}s -> {OUT}", flush=True)


def _row(*vals):
    with open(OUT, "a") as f:
        f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                         for v in vals) + "\n")


if __name__ == "__main__":
    print(f"jax devices: {jax.devices()}", flush=True)
    run()
