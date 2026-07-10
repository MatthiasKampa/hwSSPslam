"""Structured hex SSP vs the vanilla structured polar lattice — REAL data e2e.

The original encoder study settled this on synthetic worlds (oct-polar best
or tied everywhere; hex loses on flat walls, wins on curved geometry) but a
real-log e2e comparison was never run: the shipped pipeline is polar-native
(half-circle direction grid + conjugate-wrap permutation). This module adds
the hex path (subclass/patch only):

  hex lattice   6 rings x nA directions over the FULL circle, nA = 3k odd
                (exact 120-deg triplets; odd => no negation-duplicates, so
                the direction set effectively covers pi at ~2x density for
                the same D — at the cost of 2x coarser rotation-permutation
                granularity 2*pi/nA).
  HexMatcher    rotation candidates = PLAIN index shift (no conjugate wrap),
                lattice step 2*pi/nA, fine stage span widened to cover the
                half-step grid quantization.
  HexSLAM       world_vec_seg with hex permutation + d/dtheta correction
                (theta snapped to 2*pi/nA instead of pi/N_ANG).

Configs: hex63 (D=378, matched to shipped D=360) and hex111 (dense, webvis
parity). Prediction from the synthetic study: belgioioso (non-Manhattan
castle) is the hex-friendly candidate; intel (Manhattan) the control.
"""
import sys

import numpy as np

import ssp_slam as S
import ssp_slam_loop as L
import ssp_fpga as F
import ssp_lattice
import ssp_datasets as DS

RELO = ssp_lattice.RELO
OCT = ssp_lattice.OCT


def set_lattice_hex(n_ang):
    """Patch module lattice globals to a full-circle hex layout
    (consolidated into ssp_lattice.set_hex; kept as the RESULTS-cited name)."""
    ssp_lattice.set_hex(n_ang)


def hex_permute(v, m, n_ang):
    """Rotation by m*(2pi/n_ang): plain circular shift, no conjugation."""
    A = v.reshape(-1, n_ang)
    return A[:, (np.arange(n_ang) - m) % n_ang].reshape(-1)


class HexMatcher(S.Matcher):
    """Full-circle permutation rotations + adaptive fine span."""

    def __init__(self, enc, n_ang, **kw):
        kw.pop("perm", None)
        super().__init__(enc, **kw)
        self.hex_n = n_ang
        self.lat = 2 * np.pi / n_ang
        fine_step = np.deg2rad(0.375)
        self.n_side = max(4, int(np.ceil((self.lat / 4) / fine_step)) + 1)

    def match(self, M, pts, weights, guess):
        lat = self.lat
        bases = ((self.enc.encode(pts, weights), 0.0),
                 (self.enc.encode(pts @ S._rot(lat / 2).T, weights), lat / 2))
        m0 = int(round(guess[2] / lat))
        h = int(np.ceil(self.rot_half / lat - 1e-9))
        best = (-np.inf, None, None)
        for m in range(m0 - h, m0 + h + 1):
            for S0, off in bases:
                Sv = hex_permute(S0, m, self.hex_n)
                t, sc, _, _ = self._best_translation(
                    M, Sv, guess[:2], self.coarse_off, self.E_coarse)
                if sc > best[0]:
                    best = (sc, m * lat + off, t)
        _, th_c, t_c = best
        fine_step = np.deg2rad(0.375)
        thetas = th_c + fine_step * np.arange(-self.n_side, self.n_side + 1)
        th_scores, results = [], []
        for th in thetas:
            Sv = self.enc.encode(pts @ S._rot(th).T, weights)
            t, sc, scores, kk = self._best_translation(
                M, Sv, t_c, self.fine_off, self.E_fine)
            th_scores.append(sc)
            results.append((t, scores, kk))
        j = int(np.argmax(th_scores))
        th_f = thetas[j]
        t_f, scores, kk = results[j]
        if 0 < j < len(thetas) - 1:
            th_f += S._parabolic(th_scores[j - 1], th_scores[j],
                                 th_scores[j + 1], fine_step)
        n = int(round(np.sqrt(len(self.fine_off))))
        ky, kx = divmod(kk, n)
        grid = scores.reshape(n, n)
        dt = np.zeros(2)
        if 0 < kx < n - 1:
            dt[0] = S._parabolic(grid[ky, kx - 1], grid[ky, kx],
                                 grid[ky, kx + 1], 0.02)
        if 0 < ky < n - 1:
            dt[1] = S._parabolic(grid[ky - 1, kx], grid[ky, kx],
                                 grid[ky + 1, kx], 0.02)
        return np.array([t_f[0] + dt[0], t_f[1] + dt[1], th_f])


class HexSLAM(F.BandSLAM):
    """Bounded pipeline over the hex lattice (globals must be patched via
    set_lattice_hex BEFORE construction)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        n = L.N_ANG
        self.matcher = HexMatcher(L.ENC_MAIN, n, t_half=0.48, rot_half_deg=9,
                                  rot_step_deg=1.5)
        self.cmatcher = HexMatcher(L.ENC_MAIN, n, t_half=0.72, rot_half_deg=9,
                                   rot_step_deg=1.5)

    def world_vec_seg(self, aid):
        a = self.anchors[aid]
        lat = 2 * np.pi / L.N_ANG
        m = int(round(a[2] / lat))
        delta = a[2] - m * lat
        v = hex_permute(self.segvec[aid], m, L.N_ANG)
        if self.use_der and aid in self.segder:
            v = v + delta * hex_permute(self.segder[aid], m, L.N_ANG)
        return L.ENC.shift(a[:2]) * v


def self_test(n_ang):
    set_lattice_hex(n_ang)
    rng = np.random.default_rng(3)
    pts = rng.uniform(-4, 4, (200, 2))
    w = np.full(200, 0.1)
    for m in (5, 17, n_ang - 2):
        th = m * 2 * np.pi / n_ang
        a = np.exp(1j * ((pts @ S._rot(th).T) @ L.W.T)).T @ w
        b = hex_permute(np.exp(1j * (pts @ L.W.T)).T @ w, m, n_ang)
        assert np.allclose(a, b, atol=1e-9), m
    print(f"  hex{n_ang}: rotation==plain-permutation self-test ok", flush=True)


def run_intel(n_ang):
    set_lattice_hex(n_ang)
    r = DS.run("intel", HexSLAM, spec=None, nph=0)
    return r["ate"], r["med"], r["loops"]


def run_belg(n_ang):
    set_lattice_hex(n_ang)
    r = DS.run("belg", HexSLAM, spec=None, nph=0)
    return r["ate"], r["med"], r["loops"]


def main():
    what = sys.argv[1:] or ["belg", "intel"]
    for n in (63, 111):
        self_test(n)
    print("references (shipped polar-60, zero-shot): belg 2.644 / intel 2.440",
          flush=True)
    for lg in what:
        for n in (63, 111):
            ate_, med, nl = (run_belg if lg == "belg" else run_intel)(n)
            print(f"  {lg} hex{n} (D={6 * n}): ATE {ate_:.3f}  med {med:.3f} "
                  f" loops {nl}", flush=True)


if __name__ == "__main__":
    main()
