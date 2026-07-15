"""Large 6-DoF formulation sweep over the SSP lattice family (JAX, ~1h budget).

Explores the isotropy <-> permutability <-> decode-accuracy trade across the
whole ring-sphere family (elevation spacing x azimuth apportionment x stagger x
ring count) against the non-permutable references (fib3d, rand3d) and the 2D
baseline (az2d), in FLOAT and QUANTIZED, over multiple deterministic seeds.

Checkpoints every config to scratch/sweep_6dof.csv (append-only) so a partial
run is still useful; a human log goes to stdout. Anti-oracle: self-contained
synthetic rooms, no reference labels. GPU capped at <=45% VRAM (see core).

  PYTHONPATH=. python3 -m sspax.sweep [tierA|tierB|all]

Metrics per config:
  uni_cv/uni_min   nearest-neighbour arc uniformity (seed-independent)
  yawNN_5/13       yaw fidelity via nearest-direction perm_of (fair, all)
  yawRing_5/13     ring-native per-ring snap + gated d/dalpha (ring layouts)
  so3_med/p90      SE(3)-rotation decode geodesic error over a +-15deg grid
  disp_med         SE(3)-translation decode error at |d|=0.5 m
"""
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from sspax import core as C
from sspax import sphere as SPH
from sspax import worlds as Wd
from sspax.bench import _decode_jit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch" / "sweep_6dof.csv"
SEEDS = (0, 100, 200, 300, 500)
N_SCENES = 16
N_PTS = 3000
GATE = np.deg2rad(3.5)
QUANTS = [None, (16, 4)]


# --------------------------------------------------------------------------
#  the formulation catalogue
# --------------------------------------------------------------------------
def catalogue():
    """name -> dict(fn=dirs(n_dir), ring=bool, kw=ring_layout knobs|None)."""
    cat = {
        "az2d":  dict(fn=lambda n: C.dirs_az(n), ring=False, kw=None),
        "fib3d": dict(fn=lambda n: C.dirs_fib(n), ring=False, kw=None),
        "rand3d": dict(fn=lambda n: C.dirs_rand(n), ring=False, kw=None),
        "azel3d": dict(fn=lambda n: C.dirs_azel(
            12, [-40, -20, 0, 20, 40]), ring=False, kw=None),
    }
    for spacing in ("arc", "area"):
        for apportion in ("cos", "const"):
            for stagger in (0.0, 0.5):
                for nr in (4, 5, 6, 7, 8):
                    kw = dict(spacing=spacing, apportion=apportion,
                              stagger=stagger, n_rings=nr)
                    name = f"ring_{spacing}_{apportion}_s{stagger:g}_r{nr}"
                    cat[name] = dict(
                        fn=(lambda n, kw=kw: SPH.dirs_ring(n, **kw)),
                        ring=True, kw=kw)
    return cat


# --------------------------------------------------------------------------
#  metrics
# --------------------------------------------------------------------------
def _rooms(seed):
    return Wd.rooms(N_SCENES, N_PTS, seed0=seed)


def m_uniform(dirs):
    u = SPH.nn_uniformity(dirs)
    return u["cv"], u["min"]


def m_yaw_nn(Wl, clouds_by_seed, deg, quant):
    R = C._axis_rot("yaw", deg)
    perm, sgn, _, _ = C.perm_of(np.asarray(Wl), R)
    perm, sgn = jnp.asarray(perm), jnp.asarray(sgn)
    cs = []
    for clouds in clouds_by_seed:
        for c in clouds:
            v0 = C.encode(Wl, c, quant=quant)
            v1 = C.encode(Wl, c @ R.T, quant=quant)
            cs.append(float(jnp.abs(jnp.vdot(v1, C.apply_perm(v0, perm, sgn)))))
    return float(np.median(cs))


def m_yaw_ring(Wl, n_dir, kw, clouds_by_seed, deg):
    """Native per-ring snap + gated d/dalpha correction (float only — the
    derivative vector doubles storage, so quant is out of scope here)."""
    th = np.deg2rad(deg)
    perm, sgn, _ = SPH.yaw_perm(n_dir, th, n_lams=4, **kw)
    dnp = SPH.yaw_residuals(n_dir, th, n_lams=4, **kw)
    dg = jnp.asarray(np.where(np.abs(dnp) < GATE, dnp, 0.0))
    permj = jnp.asarray(perm)
    cs = []
    for clouds in clouds_by_seed:
        for c in clouds:
            v1 = C.encode(Wl, c @ C._axis_rot("yaw", deg).T)
            v0r = C.encode_raw(Wl, c)
            dv = C.encode_dvda(Wl, c)
            vg = v0r[permj] - dg * dv[permj]
            vg = vg / jnp.maximum(jnp.linalg.norm(vg), 1e-12)
            cs.append(float(jnp.abs(jnp.vdot(v1, vg))))
    return float(np.median(cs))


def m_so3(Wl, clouds_by_seed, quant, grid_deg, seed=11):
    Rg, perms, sgns, exact = C.so3_perms(np.asarray(Wl), grid_deg=grid_deg)
    perms_j, sgns_j = jnp.asarray(perms), jnp.asarray(sgns)
    errs = []
    rng = np.random.default_rng(seed)
    for clouds in clouds_by_seed:
        truths = rng.uniform(-12, 12, (len(clouds), 3))
        Rts = np.stack([C._axis_rot("yaw", y) @ C._axis_rot("pitch", p)
                        @ C._axis_rot("roll", r) for y, p, r in truths])
        v0 = jnp.stack([C.encode(Wl, c, quant=quant) for c in clouds])
        v1 = jnp.stack([C.encode(Wl, c @ Rt.T, quant=quant)
                        for c, Rt in zip(clouds, Rts)])
        sc = np.asarray(_decode_jit(v0, v1, perms_j, sgns_j))
        best = sc.argmax(1)
        for b in range(len(clouds)):
            Rh = Rg[best[b]]
            errs.append(np.degrees(np.arccos(np.clip(
                (np.trace(Rh @ Rts[b].T) - 1) / 2, -1, 1))))
    return float(np.median(errs)), float(np.percentile(errs, 90))


def m_disp(name, Wl, clouds_by_seed, quant, mag=0.5, seed=11):
    steps = np.arange(-1.2, 1.21, 0.15)
    zs = steps if name != "az2d" else np.array([0.0])
    gx, gy, gz = np.meshgrid(steps, steps, zs, indexing="ij")
    grid = jnp.asarray(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1))
    rng = np.random.default_rng(seed)
    errs = []
    for clouds in clouds_by_seed:
        for c in clouds:
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            d = mag * u
            v0 = C.encode(Wl, c, quant=quant)
            v1 = C.encode(Wl, c + d, quant=quant)
            g = jnp.conj(v1) * v0
            sc = jnp.real(jnp.exp(1j * (grid @ Wl.T)) @ g)
            best = np.asarray(grid)[int(jnp.argmax(sc))]
            errs.append(float(np.linalg.norm(best - d)))
    return float(np.median(errs))


# --------------------------------------------------------------------------
#  runner
# --------------------------------------------------------------------------
HDR = ("config,n_dir,D,quant,uni_cv,uni_min,yawNN_5,yawNN_13,"
       "yawRing_5,yawRing_13,so3_med,so3_p90,disp_med,secs")


def _log_row(row):
    with open(OUT, "a") as f:
        f.write(",".join(str(x) for x in row) + "\n")


def run(tier="all", out=None):
    global OUT
    if out is not None:
        OUT = Path(out)
    OUT.write_text(HDR + "\n")                 # fresh per batch
    cat = catalogue()
    clouds_by_seed = [_rooms(s) for s in SEEDS]
    t_start = time.time()
    print(f"sweep start: {len(cat)} configs, {len(SEEDS)} seeds x {N_SCENES} "
          f"scenes; -> {OUT}", flush=True)

    # -- Tier A: every formulation @ n_dir=60, full metrics, float+quant
    if tier in ("all", "tierA"):
        so3_grid = range(-15, 16, 3)          # 11^3 = 1331 candidates
        for name, spec in cat.items():
            for quant in QUANTS:
                t0 = time.time()
                n_dir = 60
                dirs = spec["fn"](n_dir)
                Wl = jnp.asarray(C.W_of(dirs))
                cv, mn = m_uniform(dirs)
                y5 = m_yaw_nn(Wl, clouds_by_seed, 5, quant)
                y13 = m_yaw_nn(Wl, clouds_by_seed, 13, quant)
                if spec["ring"] and quant is None:
                    yr5 = m_yaw_ring(Wl, n_dir, spec["kw"], clouds_by_seed, 5)
                    yr13 = m_yaw_ring(Wl, n_dir, spec["kw"], clouds_by_seed, 13)
                else:
                    yr5 = yr13 = ""
                s_med, s_p90 = m_so3(Wl, clouds_by_seed, quant, so3_grid)
                dm = m_disp(name, Wl, clouds_by_seed, quant)
                secs = time.time() - t0
                q = "float" if quant is None else f"{quant[0]}ph{quant[1]}mag"
                row = [name, n_dir, 4 * n_dir, q, f"{cv:.4f}", f"{mn:.2f}",
                       f"{y5:.4f}", f"{y13:.4f}", yr5 and f"{yr5:.4f}",
                       yr13 and f"{yr13:.4f}", f"{s_med:.2f}", f"{s_p90:.2f}",
                       f"{dm:.4f}", f"{secs:.1f}"]
                _log_row(row)
                print(f"  [{time.time()-t_start:5.0f}s] {name:24s} {q:9s} "
                      f"cv{cv:.3f} yNN{y5:.3f}/{y13:.3f} so3 {s_med:.1f} "
                      f"disp {dm:.3f}", flush=True)

    # -- Tier B: resolution sweep of the ring family + fib @ higher n_dir
    if tier in ("all", "tierB"):
        pick = [n for n in cat if n.startswith("ring_") or n == "fib3d"]
        for n_dir in (120, 240, 480):
            so3_grid = range(-15, 16, 5) if n_dir <= 120 else None
            for name in pick:
                spec = cat[name]
                t0 = time.time()
                dirs = spec["fn"](n_dir)
                Wl = jnp.asarray(C.W_of(dirs))
                cv, mn = m_uniform(dirs)
                y7 = m_yaw_nn(Wl, clouds_by_seed, 7, None)
                if spec["ring"]:
                    yr7 = m_yaw_ring(Wl, n_dir, spec["kw"], clouds_by_seed, 7)
                else:
                    yr7 = ""
                if so3_grid is not None:
                    s_med, s_p90 = m_so3(Wl, clouds_by_seed, None, so3_grid)
                else:
                    s_med = s_p90 = ""
                dm = m_disp(name, Wl, clouds_by_seed, None)
                secs = time.time() - t0
                row = [name, n_dir, 4 * n_dir, "float", f"{cv:.4f}",
                       f"{mn:.2f}", f"{y7:.4f}", "", yr7 and f"{yr7:.4f}", "",
                       s_med and f"{s_med:.2f}", s_p90 and f"{s_p90:.2f}",
                       f"{dm:.4f}", f"{secs:.1f}"]
                _log_row(row)
                print(f"  [{time.time()-t_start:5.0f}s] {name:24s} n{n_dir:<4d} "
                      f"cv{cv:.3f} yNN7 {y7:.3f} yRing7 "
                      f"{yr7 if yr7=='' else f'{yr7:.3f}'} disp {dm:.3f}",
                      flush=True)

    print(f"sweep done: {time.time()-t_start:.0f}s total -> {OUT}", flush=True)


if __name__ == "__main__":
    print(f"jax devices: {jax.devices()}", flush=True)
    tier = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = sys.argv[2] if len(sys.argv) > 2 else None
    run(tier, out)
