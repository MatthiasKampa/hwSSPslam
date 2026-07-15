"""Batch 4 — resolution + scale-ladder sweep on the batch-1/2/3 winners.

Two studies on the carried geometries (lidar-leaning ring_arc_cos, cam-leaning
ring_arc_const, azel baseline):
  RES     directions/scale n_dir in {60,120,240,480}: does the geometry ranking
          hold as D grows, and how fast do yaw/place climb?
  LADDER  the wavelength ladder (n_scales x range) at n_dir=60: lidar lambda =
          spatial wavelength (metres); camera lambda = angular concentration on
          S^2 (|b|=1). The two modalities may want different ladders.

  PYTHONPATH=. python3 -m sspax.sweep_res
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
from sspax.sweep import m_uniform, m_so3, m_yaw_nn, m_disp, SEEDS, N_SCENES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch" / "sweep_b4_res.csv"

GEOMS = {
    "ring_arc_cos_s0.5_r5": dict(spacing="arc", apportion="cos", stagger=0.5, n_rings=5),
    "ring_arc_const_s0.5_r6": dict(spacing="arc", apportion="const", stagger=0.5, n_rings=6),
    "azel3d": None,     # fixed layout, resolution via n_az scaling
}
LADDERS = {
    "oct3_.5-2":  [0.5, 1.0, 2.0],
    "oct4_.5-4":  [0.5, 1.0, 2.0, 4.0],
    "oct5_.25-4": [0.25, 0.5, 1.0, 2.0, 4.0],
    "oct6_.25-8": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    "fine_.25-1": [0.25, 0.4, 0.6, 1.0],       # camera-tight (high ang freq)
}
HDR = "study,config,n_dir,n_scales,D,uni_cv,yawNN7,so3_med,disp_med,placeOmni,secs"


def _dirs(name, n_dir):
    if name == "azel3d":
        n_az = max(4, round(n_dir / 5))
        return C.dirs_azel(n_az, [-40, -20, 0, 20, 40])[:n_dir] if False \
            else C.dirs_azel(n_az, [-40, -20, 0, 20, 40])
    return SPH.dirs_ring(n_dir, **GEOMS[name])


def _row(*v):
    with open(OUT, "a") as f:
        f.write(",".join(f"{x:.4f}" if isinstance(x, float) else str(x)
                         for x in v) + "\n")


def run():
    OUT.write_text(HDR + "\n")
    clouds = [Wd.rooms(N_SCENES, 3000, seed0=s) for s in SEEDS]
    pl_omni = [F.gen_places(N_SCENES, 2, seed0=s, yaw_jit=40, fov=(360.0, 120.0))
               for s in SEEDS[:3]]
    t0 = time.time()
    print(f"batch4 res/ladder -> {OUT}", flush=True)

    # -- RES: resolution sweep, matched 4-lam ladder
    for name in GEOMS:
        for n_dir in (60, 120, 240, 480):
            dirs = _dirs(name, n_dir)
            nd = len(dirs)
            Wl = jnp.asarray(C.W_of(dirs))
            cv, _ = m_uniform(dirs)
            y7 = m_yaw_nn(Wl, clouds, 7, None)
            sm, _ = m_so3(Wl, clouds, None, range(-15, 16, 5)) if n_dir <= 120 \
                else ("", None)
            dm = m_disp(name, Wl, clouds, None)
            ao = np.mean([F.place_auc(np.asarray(Wl), p, mode="cam")[0]
                          for p in pl_omni])
            _row("res", name, nd, 4, 4 * nd, cv,
                 y7, sm if sm == "" else float(sm), dm, float(ao),
                 time.time() - t0)
            print(f"  [{time.time()-t0:5.0f}s] RES {name:22s} n{nd:<4d} "
                  f"cv{cv:.3f} yaw{y7:.3f} so3 {sm} disp{dm:.3f} "
                  f"place{ao:.3f}", flush=True)

    # -- LADDER: wavelength ladder at n_dir=60
    for name in GEOMS:
        dirs = _dirs(name, 60)
        for lname, lams in LADDERS.items():
            Wl = jnp.asarray(C.W_of(dirs, lams))
            cv, _ = m_uniform(dirs)
            y7 = m_yaw_nn(Wl, clouds, 7, None)
            sm, _ = m_so3(Wl, clouds, None, range(-15, 16, 5))
            dm = m_disp(name, Wl, clouds, None)
            ao = np.mean([F.place_auc(np.asarray(Wl), p, mode="cam")[0]
                          for p in pl_omni])
            _row("ladder:" + lname, name, len(dirs), len(lams),
                 len(dirs) * len(lams), cv, y7, float(sm), dm, float(ao),
                 time.time() - t0)
            print(f"  [{time.time()-t0:5.0f}s] LAD {lname:11s} {name:22s} "
                  f"so3 {sm:.1f} disp{dm:.3f} place{ao:.3f}", flush=True)
    print(f"batch4 done: {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    print(f"jax devices: {jax.devices()}", flush=True)
    run()
