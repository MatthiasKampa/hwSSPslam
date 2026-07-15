"""Batch 6 — verification of the finalists with a FINER SO(3) grid and
bootstrap confidence intervals, so the ranking is not grid/seed noise.

Top configs from batches 1-5 (ring_arc_const permutable winner, ring_arc_cos
uniformity winner, azel3d baseline, fib3d isotropic ref) at the shipped D=240
and the wider oct6 ladder, float + quant. Fine SO(3) grid (+-12deg @ 2deg =
2197 candidates). Place-rec AUC with a 1000-sample bootstrap 90% CI.

  PYTHONPATH=. python3 -m sspax.sweep_verify
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
from sspax.sweep import m_so3, SEEDS, N_SCENES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch" / "sweep_b6_verify.csv"

OCT6 = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
GEOMS = {
    "ring_arc_const_s0.5_r6": SPH.dirs_ring(60, spacing="arc", apportion="const",
                                            stagger=0.5, n_rings=6),
    "ring_arc_cos_s0.5_r5": SPH.dirs_ring(60, spacing="arc", apportion="cos",
                                          stagger=0.5, n_rings=5),
    "azel3d": C.dirs_azel(12, [-40, -20, 0, 20, 40]),
    "fib3d": C.dirs_fib(60),
}
HDR = "config,ladder,quant,so3_med,so3_p90,placeOmni,placeOmni_lo,placeOmni_hi,secs"


def _boot_auc(pos, neg, n=1000, seed=7):
    rng = np.random.default_rng(seed)
    pos, neg = np.array(pos), np.array(neg)
    vals = []
    for _ in range(n):
        p = rng.choice(pos, len(pos))
        q = rng.choice(neg, len(neg))
        vals.append(F._auc(p, q))
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def _place_pool(W, places, quant):
    """Collect same/different sims across seeds for one CI."""
    pos, neg = [], []
    for pl in places:
        K = len(pl)
        Vs = [[F.enc_cam(jnp.asarray(W), b, wb, quant) for (_, b, wb) in place]
              for place in pl]
        perms, sgns = F._yaw_perm_set(W, 12)

        def sim(a, b):
            return max(float(jnp.abs(jnp.vdot(C.apply_perm(a, p, s), b)))
                       for p, s in zip(perms, sgns))
        for k in range(K):
            pos.append(sim(Vs[k][0], Vs[k][1]))
            for j in range(K):
                if j != k:
                    neg.append(sim(Vs[k][0], Vs[j][0]))
    return pos, neg


def run():
    OUT.write_text(HDR + "\n")
    clouds = [Wd.rooms(N_SCENES, 3000, seed0=s) for s in SEEDS]
    pl_o = [F.gen_places(20, 2, seed0=s, yaw_jit=40, fov=(360.0, 120.0))
            for s in SEEDS[:3]]
    t0 = time.time()
    print(f"batch6 verify -> {OUT}", flush=True)
    for name, dirs in GEOMS.items():
        for lname, lams in (("oct4", None), ("oct6", OCT6)):
            W = C.W_of(dirs, lams) if lams else C.W_of(dirs)
            Wl = jnp.asarray(W)
            for quant in (None, (16, 4)):
                ts = time.time()
                sm, sp = m_so3(Wl, clouds, quant, range(-12, 13, 2))
                pos, neg = _place_pool(W, pl_o, quant)
                auc = F._auc(np.array(pos), np.array(neg))
                lo, hi = _boot_auc(pos, neg)
                q = "float" if quant is None else f"{quant[0]}ph{quant[1]}mag"
                with open(OUT, "a") as f:
                    f.write(f"{name},{lname},{q},{sm:.2f},{sp:.2f},{auc:.4f},"
                            f"{lo:.4f},{hi:.4f},{time.time()-ts:.1f}\n")
                print(f"  [{time.time()-t0:5.0f}s] {name:24s} {lname} {q:9s} "
                      f"so3 {sm:.1f}/{sp:.1f}  placeO {auc:.3f} "
                      f"[{lo:.3f},{hi:.3f}]", flush=True)
    print(f"batch6 done: {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    print(f"jax devices: {jax.devices()}", flush=True)
    run()
