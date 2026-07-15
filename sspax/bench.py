"""sspax benches — the 6-DoF lattice-algebra studies, JAX-accelerated.

  python3 -m sspax.bench parity           port is bit-faithful to numpy core
  python3 -m sspax.bench uniform          point-equidistance per layout
  python3 -m sspax.bench rot   [--quant nph,nmag]   yaw permutation fidelity
  python3 -m sspax.bench so3   [--quant nph,nmag]   SE(3)-rotation decode error
  python3 -m sspax.bench disp  [--quant nph,nmag]   SE(3)-translation decode
  python3 -m sspax.bench speed            numpy-vs-JAX wall time

Every result is deterministic (seeded synthetic rooms). --quant applies the
FPGA polar quantizer (q_polar) to the STORED vectors, giving the "quantized"
column next to the float one. Anti-oracle: no reference labels anywhere here —
these are self-contained algebra measurements on synthetic geometry.
"""
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

from sspax import core as C
from sspax import sphere as SPH
from sspax import worlds as W

LAYOUTS = ["az2d", "fib3d", "azel3d", "ringstag3d", "rand3d"]


def _parse_quant(argv):
    if "--quant" in argv:
        i = argv.index("--quant")
        nph, nmag = (int(x) for x in argv[i + 1].split(","))
        return (nph, nmag)
    return None


# --------------------------------------------------------------------------
def bench_parity():
    """The JAX core must reproduce the numpy experiments.lattice3d algebra."""
    import experiments.lattice3d as N
    rng = np.random.default_rng(3)
    pts = rng.uniform(-4, 4, (2000, 3))
    lat_np = N.make_lattices()
    lat_ax = C.make_lattices()
    print("parity vs experiments.lattice3d (max abs diff):")
    emax = pmax = qmax = 0.0
    for name in ("az2d", "fib3d", "azel3d", "rand3d"):
        W_np = lat_np[name]
        W_ax = np.asarray(lat_ax[name])
        emax = max(emax, np.abs(W_np - W_ax).max())          # same lattices
        v_np = N.encode(W_np, pts)
        v_ax = np.asarray(C.encode(W_np, pts))
        emax = max(emax, np.abs(v_np - v_ax).max())          # same encode
        R = N._axis_rot("yaw", 15) @ N._axis_rot("pitch", 7)
        p_np, s_np, ex_np = N._perm_of(W_np, R)
        p_ax, s_ax, ex_ax, _ = C.perm_of(W_np, R)
        pmax = max(pmax, int(np.abs(p_np - p_ax).max()),
                   int(np.abs(s_np.astype(int) - s_ax.astype(int)).max()))
        q_np = C.q_polar_np(v_np, 16, 4)
        q_ax = np.asarray(C.q_polar(jnp.asarray(v_np), 16, 4))
        qmax = max(qmax, np.abs(q_np - q_ax).max())
    print(f"  lattice+encode: {emax:.2e}   perm(idx+sgn): {pmax}   "
          f"q_polar: {qmax:.2e}")
    ok = emax < 1e-9 and pmax == 0 and qmax < 1e-9
    print("  PARITY OK" if ok else "  PARITY FAIL")
    return ok


# --------------------------------------------------------------------------
def bench_uniform():
    """How equidistant are the 60 base directions? (full antipodal set)."""
    print("point equidistance (nearest-neighbour arc over u & -u, N=60 dirs):")
    print(f"  {'layout':11s} {'min deg':>8s} {'mean deg':>9s} {'cv':>7s}"
          f"   (lower cv = more uniform)")
    builders = {
        "az2d": C.dirs_az(60), "fib3d": C.dirs_fib(60),
        "azel3d": C.dirs_azel(12, [-40, -20, 0, 20, 40]),
        "ringstag3d": SPH.dirs_ringstag(60), "rand3d": C.dirs_rand(60),
    }
    for name in LAYOUTS:
        u = SPH.nn_uniformity(builders[name])
        print(f"  {name:11s} {u['min']:8.2f} {u['mean']:9.2f} {u['cv']:7.3f}",
              flush=True)
    lay = SPH.ring_layout(60)
    print(f"  ringstag3d layout: {lay['n_rings']} rings, "
          f"az counts {list(lay['n_az'])}")


# --------------------------------------------------------------------------
def _encode_clouds(Wl, clouds, quant):
    return jnp.stack([C.encode(Wl, c, quant=quant) for c in clouds])


def bench_rot(quant=None, n_scenes=16, n_pts=3000):
    """Yaw fidelity: cos(re-encode rotated cloud, permutation of frozen v0).
    perm_of (fair, nearest-direction, all layouts) + ringstag3d's native
    per-ring-snapped yaw_perm. * = exact permutation."""
    clouds = W.rooms(n_scenes, n_pts)
    lat = C.make_lattices()
    tag = f" [quant {quant[0]}ph/{quant[1]}mag]" if quant else " [float]"
    print(f"yaw permutation fidelity{tag} ({n_scenes} rooms):")
    print(f"  {'layout':11s}  {'  5deg':>7s} {'  15deg':>7s} {' 30deg':>7s}"
          f"  {'exact@':>8s}")
    for name in LAYOUTS:
        Wl = jnp.asarray(lat[name])
        row = []
        exact_deg = "-"
        for deg in (5, 15, 30):
            R = C._axis_rot("yaw", deg)
            perm, sgn, exact, _ = C.perm_of(np.asarray(Wl), R)
            perm, sgn = jnp.asarray(perm), jnp.asarray(sgn)
            cs = []
            for c in clouds:
                v0 = C.encode(Wl, c, quant=quant)
                v1 = C.encode(Wl, c @ R.T, quant=quant)
                vp = C.apply_perm(v0, perm, sgn)
                cs.append(float(jnp.abs(jnp.vdot(v1, vp))))
            row.append(np.median(cs))
            if exact and exact_deg == "-":
                exact_deg = f"{deg}d"
        print(f"  {name:11s}  {row[0]:7.4f} {row[1]:7.4f} {row[2]:7.4f}"
              f"  {exact_deg:>8s}", flush=True)
    # ringstag3d native yaw (per-ring cyclic shift) at NON-lattice angles,
    # with and without the first-order d/dalpha derivative correction.
    Wl = jnp.asarray(lat["ringstag3d"])
    print("  ringstag3d native yaw_perm (per-ring snap vs snap+derivative, "
          "D=240):")
    for deg in (5, 15, 30, 37):
        th = np.deg2rad(deg)
        perm, sgn, resid = SPH.yaw_perm(60, th, n_lams=4)
        delta = jnp.asarray(SPH.yaw_residuals(60, th, n_lams=4))
        permj = jnp.asarray(perm)
        cs, cd = [], []
        for c in clouds:
            v1 = C.encode(Wl, c @ C._axis_rot("yaw", deg).T)   # truth
            v0r = C.encode_raw(Wl, c)                          # unnormalized
            dv = C.encode_dvda(Wl, c)
            v_shift = v0r[permj]
            v_snap = v_shift / jnp.maximum(jnp.linalg.norm(v_shift), 1e-12)
            v_corr = v_shift - delta * dv[permj]
            v_corr = v_corr / jnp.maximum(jnp.linalg.norm(v_corr), 1e-12)
            cs.append(float(jnp.abs(jnp.vdot(v1, v_snap))))
            cd.append(float(jnp.abs(jnp.vdot(v1, v_corr))))
        print(f"    {deg:3d}deg: snap {np.median(cs):.4f}  "
              f"snap+deriv {np.median(cd):.4f}  "
              f"(worst residual {np.degrees(resid):.1f} deg)", flush=True)


# --------------------------------------------------------------------------
def bench_resolution(n_scenes=16, n_pts=3000, yaw_deg=7.0):
    """When does ringstag3d's approximate yaw become GOOD? Sweep directions/
    scale: as ring azimuth steps shrink, the per-ring snap residual shrinks and
    the first-order d/dalpha correction crosses from hurting to rescuing the
    sub-step yaw. Fixed off-lattice yaw (7deg), float. D = 4 * n_dir."""
    clouds = W.rooms(n_scenes, n_pts)
    th = np.deg2rad(yaw_deg)
    print(f"ringstag3d yaw resolution sweep (off-lattice yaw {yaw_deg}deg, "
          f"{n_scenes} rooms):")
    print(f"  {'n_dir':>6s} {'D':>5s} {'rings':>6s} {'finest step':>12s} "
          f"{'residual':>9s}  {'snap':>7s} {'+deriv':>7s} {'+gated':>7s}")
    gate = np.deg2rad(3.5)                   # first-order valid regime
    for n_dir in (60, 120, 240, 480, 960):
        Wl = jnp.asarray(C.W_of(SPH.dirs_ringstag(n_dir)))
        lay = SPH.ring_layout(n_dir)
        perm, sgn, resid = SPH.yaw_perm(n_dir, th, n_lams=4)
        delta_np = SPH.yaw_residuals(n_dir, th, n_lams=4)
        delta = jnp.asarray(delta_np)
        # gated: correct only rings whose residual is within first-order range
        delta_g = jnp.asarray(np.where(np.abs(delta_np) < gate, delta_np, 0.0))
        permj = jnp.asarray(perm)
        cs, cd, cg = [], [], []
        for c in clouds:
            v1 = C.encode(Wl, c @ C._axis_rot("yaw", yaw_deg).T)
            v0r = C.encode_raw(Wl, c)
            dv = C.encode_dvda(Wl, c)
            vs = v0r[permj]
            vs_n = vs / jnp.maximum(jnp.linalg.norm(vs), 1e-12)
            vc = vs - delta * dv[permj]
            vc_n = vc / jnp.maximum(jnp.linalg.norm(vc), 1e-12)
            vg = vs - delta_g * dv[permj]
            vg_n = vg / jnp.maximum(jnp.linalg.norm(vg), 1e-12)
            cs.append(float(jnp.abs(jnp.vdot(v1, vs_n))))
            cd.append(float(jnp.abs(jnp.vdot(v1, vc_n))))
            cg.append(float(jnp.abs(jnp.vdot(v1, vg_n))))
        finest_step = 360.0 / lay["n_az"].max()
        print(f"  {n_dir:6d} {4*n_dir:5d} {lay['n_rings']:6d} "
              f"{finest_step:10.1f}d {np.degrees(resid):7.1f}d  "
              f"{np.median(cs):7.4f} {np.median(cd):7.4f} {np.median(cg):7.4f}",
              flush=True)


def _decode_scores(v0, v1, perms, sgns):
    """(B,D),(B,D),(G,D),(G,D) -> (B,G) |<v1, perm_g(v0)>|. Batched on device."""
    rot = jax.vmap(lambda p, s: C.apply_perm(v0, p, s))(perms, sgns)  # (G,B,D)
    sc = jnp.abs(jnp.einsum("gbd,bd->bg", jnp.conj(rot), v1))
    return sc


_decode_jit = jax.jit(_decode_scores)


def bench_so3(quant=None, n_scenes=12, n_pts=3000, seed=11):
    """SE(3)-rotation decode: recover an unknown small rotation by permutation
    search over a +-15deg yaw/pitch/roll grid (5deg steps); geodesic error."""
    clouds = W.rooms(n_scenes, n_pts)
    lat = C.make_lattices()
    rng = np.random.default_rng(seed)
    truths = rng.uniform(-12, 12, (n_scenes, 3))
    Rts = np.stack([C._axis_rot("yaw", y) @ C._axis_rot("pitch", p)
                    @ C._axis_rot("roll", r) for y, p, r in truths])
    tag = f" [quant {quant[0]}ph/{quant[1]}mag]" if quant else " [float]"
    print(f"SE(3) rotation decode{tag} (+-15deg grid 5deg steps, "
          f"{n_scenes} rooms, grid floor ~3.5deg):")
    print(f"  {'layout':11s} {'med deg':>8s} {'p90 deg':>8s}  frac-exact-cands")
    for name in LAYOUTS:
        Wl = jnp.asarray(lat[name])
        Rg, perms, sgns, exact = C.so3_perms(np.asarray(Wl))
        perms_j, sgns_j = jnp.asarray(perms), jnp.asarray(sgns)
        v0 = jnp.stack([C.encode(Wl, c, quant=quant) for c in clouds])
        v1 = jnp.stack([C.encode(Wl, c @ Rt.T, quant=quant)
                        for c, Rt in zip(clouds, Rts)])
        sc = np.asarray(_decode_jit(v0, v1, perms_j, sgns_j))    # (B,G)
        best = sc.argmax(1)
        errs = []
        for b in range(n_scenes):
            Rh = Rg[best[b]]
            errs.append(np.degrees(np.arccos(np.clip(
                (np.trace(Rh @ Rts[b].T) - 1) / 2, -1, 1))))
        print(f"  {name:11s} {np.median(errs):8.1f} {np.percentile(errs,90):8.1f}"
              f"  {exact.mean():.2f}", flush=True)


# --------------------------------------------------------------------------
def bench_disp(quant=None, n_scenes=12, n_pts=3000, seed=11):
    """SE(3)-translation decode: encode cloud, shift by a known 3D d, recover d
    by correlation over a 3D candidate grid. az2d searches dz=0 (cannot see z)."""
    clouds = W.rooms(n_scenes, n_pts)
    lat = C.make_lattices()
    rng = np.random.default_rng(seed)
    steps = np.arange(-1.2, 1.21, 0.15)
    grids = {}
    for name in LAYOUTS:
        zs = steps if name != "az2d" else np.array([0.0])
        gx, gy, gz = np.meshgrid(steps, steps, zs, indexing="ij")
        grids[name] = jnp.asarray(np.stack([gx.ravel(), gy.ravel(),
                                            gz.ravel()], 1))
    tag = f" [quant {quant[0]}ph/{quant[1]}mag]" if quant else " [float]"
    print(f"SE(3) translation decode{tag} (grid step 0.15m, "
          f"{n_scenes} rooms):")
    for mag in (0.3, 0.8):
        errs = {k: [] for k in LAYOUTS}
        for c in clouds:
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            d = mag * u
            for name in LAYOUTS:
                Wl = jnp.asarray(lat[name])
                v0 = C.encode(Wl, c, quant=quant)
                v1 = C.encode(Wl, c + d, quant=quant)
                g = jnp.conj(v1) * v0
                sc = jnp.real(jnp.exp(1j * (grids[name] @ Wl.T)) @ g)
                best = np.asarray(grids[name])[int(jnp.argmax(sc))]
                errs[name].append(np.linalg.norm(best - d))
        for name in LAYOUTS:
            e = np.array(errs[name])
            print(f"  |d|={mag:.1f}: {name:11s} med {np.median(e):.3f} "
                  f"p90 {np.percentile(e, 90):.3f}", flush=True)


# --------------------------------------------------------------------------
def bench_speed(n_scenes=64, n_pts=4000):
    """numpy (experiments.lattice3d) vs JAX SO(3) decode wall time."""
    import experiments.lattice3d as N
    clouds = W.rooms(n_scenes, n_pts)
    Wl_np = N.make_lattices()["fib3d"]
    Wl = jnp.asarray(Wl_np)
    Rg, perms, sgns, exact = C.so3_perms(Wl_np)
    perms_j, sgns_j = jnp.asarray(perms), jnp.asarray(sgns)

    t0 = time.time()
    v0 = jnp.stack([C.encode(Wl, c) for c in clouds])
    v1 = jnp.stack([C.encode(Wl, c @ Rg[10].T) for c in clouds])
    sc = _decode_jit(v0, v1, perms_j, sgns_j).block_until_ready()   # warm
    t_warm = time.time() - t0
    t0 = time.time()
    for _ in range(5):
        sc = _decode_jit(v0, v1, perms_j, sgns_j).block_until_ready()
    t_jax = (time.time() - t0) / 5

    t0 = time.time()
    for c in clouds[:8]:
        v0n, v1n = N.encode(Wl_np, c), N.encode(Wl_np, c @ Rg[10].T)
        for (perm, sgn) in [N._perm_of(Wl_np, R)[:2] for R in Rg]:
            _ = np.abs(np.vdot(v1n, N._apply_perm(v0n, perm, sgn)))
    t_np = (time.time() - t0) / 8
    print(f"SO(3) decode ({n_scenes} clouds x {len(Rg)} cands, D=240):")
    print(f"  JAX  {t_jax*1e3:7.1f} ms/batch  (warm+compile {t_warm:.2f}s)")
    print(f"  numpy{t_np*1e3:7.1f} ms/CLOUD (perm rebuilt per cand)")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "parity"
    q = _parse_quant(sys.argv)
    print(f"jax devices: {jax.devices()}  x64={jax.config.jax_enable_x64}\n")
    if cmd == "parity":
        bench_parity()
    elif cmd == "uniform":
        bench_uniform()
    elif cmd == "rot":
        bench_rot(quant=q)
    elif cmd == "so3":
        bench_so3(quant=q)
    elif cmd == "disp":
        bench_disp(quant=q)
    elif cmd == "resolution":
        bench_resolution()
    elif cmd == "speed":
        bench_speed()
    else:
        print(f"unknown bench {cmd!r}")
