"""Batch 5 — quantization / deploy sweep on the finalist geometries.

The FPGA storage model (q_polar: nph phase bins x nmag magnitude levels) across
a bit grid, on the batch-1..4 finalists. Reports bytes/anchor and the accuracy
of lidar SO(3) decode + yaw and camera place-rec at each budget, so the
store-vs-fidelity deploy point is explicit.

  PYTHONPATH=. python3 -m sspax.sweep_quant
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
from sspax.sweep import m_uniform, m_so3, m_yaw_nn, SEEDS, N_SCENES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch" / "sweep_b5_quant.csv"

GEOMS = {
    "ring_arc_const_s0.5_r6": SPH.dirs_ring(60, spacing="arc", apportion="const",
                                            stagger=0.5, n_rings=6),
    "ring_arc_cos_s0.5_r5": SPH.dirs_ring(60, spacing="arc", apportion="cos",
                                          stagger=0.5, n_rings=5),
    "azel3d": C.dirs_azel(12, [-40, -20, 0, 20, 40]),
}
QGRID = [None, (16, 4), (8, 4), (16, 2), (8, 2), (4, 2)]
HDR = "config,quant,bits_per_phasor,bytes_per_anchor,so3_med,yawNN7,placeOmni,placeNarrow,secs"


def _bytes(D, quant):
    if quant is None:
        return D * 8                       # complex64 (2 x f32) ... use 8 B/phasor
    nph, nmag = quant
    bits = np.log2(nph) + np.log2(nmag)
    return int(np.ceil(D * bits / 8)) + 4  # + 1 f32 per-vector scale


def run():
    OUT.write_text(HDR + "\n")
    clouds = [Wd.rooms(N_SCENES, 3000, seed0=s) for s in SEEDS]
    pl_o = [F.gen_places(N_SCENES, 2, seed0=s, yaw_jit=40, fov=(360.0, 120.0))
            for s in SEEDS[:3]]
    pl_n = [F.gen_places(N_SCENES, 2, seed0=s, yaw_jit=40, fov=(90.0, 60.0))
            for s in SEEDS[:3]]
    t0 = time.time()
    print(f"batch5 quant -> {OUT}", flush=True)
    for name, dirs in GEOMS.items():
        Wl = jnp.asarray(C.W_of(dirs))
        D = Wl.shape[0]
        for quant in QGRID:
            ts = time.time()
            sm, _ = m_so3(Wl, clouds, quant, range(-15, 16, 3))
            y7 = m_yaw_nn(Wl, clouds, 7, quant)
            ao = np.mean([F.place_auc(np.asarray(Wl), p, mode="cam",
                                      quant=quant)[0] for p in pl_o])
            an = np.mean([F.place_auc(np.asarray(Wl), p, mode="cam",
                                      quant=quant)[0] for p in pl_n])
            q = "float" if quant is None else f"{quant[0]}ph{quant[1]}mag"
            bits = 32 if quant is None else \
                round(np.log2(quant[0]) + np.log2(quant[1]), 1)
            with open(OUT, "a") as f:
                f.write(f"{name},{q},{bits},{_bytes(D, quant)},{sm:.2f},"
                        f"{y7:.4f},{ao:.4f},{an:.4f},{time.time()-ts:.1f}\n")
            print(f"  [{time.time()-t0:5.0f}s] {name:24s} {q:9s} "
                  f"{_bytes(D, quant):4d}B so3 {sm:.1f} yaw {y7:.3f} "
                  f"placeO {ao:.3f} placeN {an:.3f}", flush=True)
    print(f"batch5 done: {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    print(f"jax devices: {jax.devices()}", flush=True)
    run()
