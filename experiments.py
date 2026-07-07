"""Encoder recipe experiments for ssp_slam. Run: python3 experiments.py <phase>

Each recipe builds a frequency matrix W; everything downstream is unchanged.
The sim RNG is re-seeded after encoder construction so every recipe sees the
IDENTICAL world, noise, and trajectory for a given seed.
"""

import sys

import numpy as np

import ssp_slam as S


class FreqEnc(S.SSPEncoder):
    def __init__(self, W):
        self.W = np.asarray(W, float)


def grid(lams, n_ang, stagger=0.0):
    """Scales x equally-spaced-angles grid (180 deg sector, single directions).
    stagger shifts scale k's angles by k*stagger of one angle step."""
    rows = []
    step = np.pi / n_ang
    for k, lam in enumerate(np.atleast_1d(lams)):
        a = np.arange(n_ang) * step + k * stagger * step
        rows.append((2 * np.pi / lam) * np.stack([np.cos(a), np.sin(a)], 1))
    return np.concatenate(rows)


def dft(L, N):
    """Integer frequency lattice (Fourier series of an L-periodic plane),
    half-plane so conjugates aren't duplicated."""
    ks = [(kx, ky) for kx in range(0, N + 1) for ky in range(-N, N + 1)
          if (kx > 0 or ky > 0)]
    return (2 * np.pi / L) * np.array(ks, float)


def octaves(lo, hi):
    return lo * 2.0 ** np.arange(int(np.log2(hi / lo)) + 1)


def banded_random(D, seed=42):
    """Random frequencies matched to the octave band: separates the
    spectral-envelope effect from the deterministic-grid effect."""
    rng = np.random.default_rng(seed)
    lam = np.exp(rng.uniform(np.log(0.25), np.log(2.0), D))
    ang = rng.uniform(0, np.pi, D)
    return (2 * np.pi / lam)[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)


def jitter_grid(lams, n_ang, seed=42):
    """Octave grid with per-module angle jitter (uniform +-half step)."""
    rng = np.random.default_rng(seed)
    rows = []
    step = np.pi / n_ang
    for lam in lams:
        a = np.arange(n_ang) * step + rng.uniform(-0.5, 0.5, n_ang) * step
        rows.append((2 * np.pi / lam) * np.stack([np.cos(a), np.sin(a)], 1))
    return np.concatenate(rows)


RECIPES = {
    # controls
    "rff-216": lambda: S.SSPEncoder(dim=216, length_scale=0.25),
    "rff-512": lambda: S.SSPEncoder(dim=512, length_scale=0.25),
    # winners so far
    "geo1.05-6x36": lambda: FreqEnc(grid(np.geomspace(0.25, 8, 6), 36)),
    # octave ladders (no tuned ratio -- simplest scale rule)
    "oct.25-16x30": lambda: FreqEnc(grid(octaves(0.25, 16), 30)),   # 7x30=210
    "oct.25-8x36": lambda: FreqEnc(grid(octaves(0.25, 8), 36)),     # 6x36=216
    "oct.25-4x36": lambda: FreqEnc(grid(octaves(0.25, 4), 36)),     # 5x36=180
    "oct.25-2x36": lambda: FreqEnc(grid(octaves(0.25, 2), 36)),     # 4x36=144
    "oct.25-2x18": lambda: FreqEnc(grid(octaves(0.25, 2), 18)),     # 4x18=72
    "oct.25-2x12": lambda: FreqEnc(grid(octaves(0.25, 2), 12)),     # 4x12=48
    "oct.5-2x18": lambda: FreqEnc(grid(octaves(0.5, 2), 18)),       # 3x18=54
    # scale-count extremes
    "1scale.5x36": lambda: FreqEnc(grid([0.5], 36)),
    "2scale.25-2x36": lambda: FreqEnc(grid([0.25, 2.0], 36)),
    # spacing recipes
    "linlam6x36": lambda: FreqEnc(grid(np.linspace(0.25, 8, 6), 36)),
    "linfreq6x36": lambda: FreqEnc(grid(2 * np.pi / np.linspace(2 * np.pi / 8, 2 * np.pi / 0.25, 6), 36)),
    "stagger-oct.25-2x18": lambda: FreqEnc(grid(octaves(0.25, 2), 18, stagger=0.25)),
    # integer DFT lattice (one parameter pair, fully regular)
    "dft-L20-N10": lambda: FreqEnc(dft(20.0, 10)),                  # D=220
    "dft-L16-N8": lambda: FreqEnc(dft(16.0, 8)),                    # D=144
    # minimal octave ladders
    "oct.25-2x9": lambda: FreqEnc(grid(octaves(0.25, 2), 9)),       # D=36
    "oct.25-2x6": lambda: FreqEnc(grid(octaves(0.25, 2), 6)),       # D=24
    "oct.25-2x4": lambda: FreqEnc(grid(octaves(0.25, 2), 4)),       # D=16
    "oct.25-1x12": lambda: FreqEnc(grid(octaves(0.25, 1), 12)),     # D=36
    "oct.25-1x6": lambda: FreqEnc(grid(octaves(0.25, 1), 6)),       # D=18
    # controls referenced in RESULTS.md tables
    "oct.25-2x24": lambda: FreqEnc(grid(octaves(0.25, 2), 24)),     # D=96
    "rff-2048": lambda: S.SSPEncoder(dim=2048, length_scale=0.25),
    "banded-216": lambda: FreqEnc(banded_random(216)),
    "banded-96": lambda: FreqEnc(banded_random(96)),
    "jit-x36": lambda: FreqEnc(jitter_grid(octaves(0.25, 2), 36)),
    "jit-x24": lambda: FreqEnc(jitter_grid(octaves(0.25, 2), 24)),
}


def run(name, make_enc, seed=7, plots=False):
    S.RNG = np.random.default_rng(seed)     # encoder draw (only used by rff)
    enc = make_enc()
    S.RNG = np.random.default_rng(seed + 1000)  # identical sim for every recipe
    r = S.main(mode=name, enc=enc, make_plots=plots)
    r["name"], r["seed"] = name, seed
    return r


def table(rows, title):
    rows = sorted(rows, key=lambda r: r["terr_mean"])
    print(f"\n== {title}")
    print(f"{'recipe':<22} {'D':>4} {'ms/f':>5} {'trans mean/max cm':>18} {'head mean/max deg':>18}")
    for r in rows:
        print(f"{r['name']:<22} {r['D']:>4} {r['ms']:>5.0f} "
              f"{100 * r['terr_mean']:>8.1f} /{100 * r['terr_max']:>7.1f} "
              f"{r['aerr_mean']:>9.2f} /{r['aerr_max']:>7.2f}")


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    if phase in ("all", "sweep"):
        rows = [run(n, f) for n, f in RECIPES.items()]
        table(rows, "single-seed sweep (seed 7)")
    if phase == "worlds":
        # six-world battery from RESULTS.md: recipes x worlds x seeds 1-3
        from worlds import WORLDS
        names = (sys.argv[2].split(",") if len(sys.argv) > 2 else
                 ["oct.25-2x12", "oct.25-2x24", "oct.25-2x36", "jit-x36", "rff-216"])
        for wname, wfn in WORLDS.items():
            S.make_environment = wfn
            print(f"== world {wname}")
            table([r for name in names
                   for r in [run(name, RECIPES[name], seed=s) for s in (1, 2, 3)]],
                  wname)
    if phase == "seeds":
        names = sys.argv[2].split(",")
        for name in names:
            rows = [run(name, RECIPES[name], seed=s) for s in (1, 2, 3, 4, 5)]
            tm = [r["terr_mean"] for r in rows]
            am = [r["aerr_mean"] for r in rows]
            tx = max(r["terr_max"] for r in rows)
            ax = max(r["aerr_max"] for r in rows)
            print(f"{name:<22} trans mean {100 * np.mean(tm):.1f} (worst-seed {100 * np.max(tm):.1f}, "
                  f"abs max {100 * tx:.1f}) cm   head mean {np.mean(am):.2f} "
                  f"(worst {np.max(am):.2f}, abs max {ax:.2f}) deg")
