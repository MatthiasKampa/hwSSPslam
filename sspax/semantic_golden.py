"""P3 (msg round 3): GOLDEN VECTORS + a numpy-ONLY checker for the semantic map,
so the deploy box (no JAX) can verify the FPGA-relevant semantic numbers. The
semantic suite imports JAX at module level; this file re-implements the binding /
query / polar-quant in pure numpy (bit-faithful to sspslam.quantized.q_polar) and
freezes a fixed scene + its expected readouts + the quantization-recall sweep into
`sspax/artifacts/semantic_golden.npz`. `check` reloads and recomputes with NO
jax import and asserts a match — the deploy-box acceptance gate for the map.

  python3 -m sspax.semantic_golden generate   # writes the golden npz
  python3 -m sspax.semantic_golden check       # numpy-only verify (deploy box)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "sspax" / "artifacts" / "semantic_golden.npz"
OCT6 = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
CLASSES = ["chair", "table", "bed", "couch", "wardrobe"]


# -------- pure-numpy semantic primitives (no jax) --------------------------
def dirs_ring_np(n_total=60, n_rings=6):
    """staggered const-azimuth rings (numpy port of sphere.dirs_ring const)."""
    per = n_total // n_rings
    elevs = np.linspace(-40, 40, n_rings)
    dirs = []
    for r, el in enumerate(elevs):
        e = np.deg2rad(el)
        az = (np.arange(per) + 0.5 * (r % 2)) / per * 2 * np.pi
        dirs.append(np.stack([np.cos(e) * np.cos(az), np.cos(e) * np.sin(az),
                              np.full(per, np.sin(e))], 1))
    return np.concatenate(dirs)


def W_of_np(dirs, lams=OCT6):
    return np.concatenate([(2 * np.pi / lam) * dirs for lam in lams])


def q_polar_np(v, nph, nmag):
    """bit-identical to sspslam.quantized.q_polar / core.q_polar_np."""
    s = float(np.percentile(np.abs(v), 99)) + 1e-12
    dph = 2 * np.pi / nph
    phq = dph * np.round(np.angle(v) / dph)
    mstep = s / nmag
    magq = np.clip(mstep * (np.floor(np.clip(np.abs(v), 0, s) / mstep) + 0.5), 0, s)
    return magq * np.exp(1j * phq)


def build_and_query(W, roles, codes, objs, qbits, grid, nph=0, nmag=0):
    mp = np.zeros(W.shape[0], complex)
    for name, xyz in objs:
        mp += roles[codes[name]].sum(0) * np.exp(1j * (xyz @ W.T))
    if nph:
        mp = q_polar_np(mp, nph, nmag)
    unb = np.conj(roles[qbits]).sum(0) * mp
    g = np.concatenate([grid, np.full((len(grid), 1), 0.5)], 1)
    return np.real(np.exp(1j * (g @ W.T)) @ np.conj(unb)) / W.shape[0]


def _detect(dens, grid, shape, n_sigma=4.0):
    D2 = dens.reshape(shape); med = np.median(dens)
    std = 1.4826 * np.median(np.abs(dens - med)) + 1e-9
    thr = med + n_sigma * std; out = []; nx, ny = shape
    for i in range(nx):
        for j in range(ny):
            v = D2[i, j]
            if v >= thr and v >= D2[max(0, i - 1):i + 2, max(0, j - 1):j + 2].max():
                out.append(grid[i * ny + j])
    return out


def _scene(n, seed):
    rng = np.random.default_rng(seed); objs = []
    for _ in range(n):
        objs.append((CLASSES[rng.integers(len(CLASSES))],
                     np.array([rng.uniform(-6, 6), rng.uniform(-5, 5), 0.5])))
    return objs


def _grid(step=0.25):
    xs = np.arange(-6, 6, step); ys = np.arange(-5, 5, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], 1), (len(xs), len(ys))


def _codes(m=256, k=12, seed=1):
    rng = np.random.default_rng(seed)
    return {c: np.sort(rng.choice(m, k, replace=False)) for c in CLASSES}


def _recall(W, roles, codes, seed0, nph, nmag, seeds=12, n_obj=8):
    grid, shape = _grid(); rs = []
    for s in range(seeds):
        objs = _scene(n_obj, seed0 + s)
        chairs = [x[:2] for n, x in objs if n == "chair"]
        if not chairs:
            continue
        dens = build_and_query(W, roles, codes, objs, codes["chair"], grid, nph, nmag)
        pk = _detect(dens, grid, shape)
        tp = sum(any(np.linalg.norm(p - c) < 0.8 for c in chairs) for p in pk[:len(chairs) + 3])
        rs.append(min(tp, len(chairs)) / len(chairs))
    return float(np.mean(rs))


QUANT = [(0, 0), (16, 4), (8, 2), (4, 2)]        # float, 6-bit, 4-bit, 3-bit


def generate():
    dirs = dirs_ring_np(); W = W_of_np(dirs)
    rng = np.random.default_rng(0)
    roles = np.exp(1j * rng.uniform(0, 2 * np.pi, (256, W.shape[0])))
    codes = _codes()
    recalls = np.array([_recall(W, roles, codes, 5000, nph, nmag) for nph, nmag in QUANT])
    GOLD.parent.mkdir(exist_ok=True)
    np.savez_compressed(GOLD, W=W.astype(np.float32), roles_seed=0, codes_seed=1,
                        quant=np.array(QUANT), recalls=recalls.astype(np.float32),
                        m=256, k=12)
    print(f"wrote {GOLD}  ({GOLD.stat().st_size/1024:.1f} KB)")
    for (nph, nmag), r in zip(QUANT, recalls):
        tag = "float" if nph == 0 else f"{int(np.log2(nph)+np.log2(nmag))}-bit/cell"
        print(f"  quant nph={nph} nmag={nmag} ({tag}): chair recall {r:.3f}")


def check(tol=0.02):
    z = np.load(GOLD)
    W = z["W"].astype(float)
    rng = np.random.default_rng(int(z["roles_seed"]))
    roles = np.exp(1j * rng.uniform(0, 2 * np.pi, (int(z["m"]), W.shape[0])))
    codes = _codes(int(z["m"]), int(z["k"]), int(z["codes_seed"]))
    print("numpy-ONLY golden check (no jax):")
    ok = True
    for (nph, nmag), exp in zip(z["quant"], z["recalls"]):
        got = _recall(W, roles, codes, 5000, int(nph), int(nmag))
        d = abs(got - exp); pas = d <= tol; ok &= pas
        print(f"  nph={nph} nmag={nmag}: expected {exp:.3f} got {got:.3f}  "
              f"|d|={d:.3f}  {'PASS' if pas else 'FAIL'}")
    print("GOLDEN CHECK", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    (generate if cmd == "generate" else check)()
