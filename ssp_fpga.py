"""FPGA-track SSP SLAM: quantized map store + front-path sizing model.

Goal (session 2026-07-10): progress the shipped bounded-memory SSP SLAM toward
an FPGA deployment, step by step — float first, then a binary/integer VSA
version. The per-keyframe hot path (encode, frontend match, segment fold, loop
verify) is O(D) phase arithmetic + small dense MACs and maps onto fabric; the
sparse pose-graph relax stays host-side (it runs every ~25 kf on a few hundred
anchors). This module adds, on top of the UNTOUCHED shipped BoundedSLAM
(PROTOCOL sec 1, subclass/import only):

  q_polar          the validated polar quantizer (phase bins x magnitude
                   levels, per-vector scale) — the static winner from
                   scratch_quant.py (16ph x 4mag = 6 bits/phasor: reg 0.0040
                   vs 0.0036 full, coherence fidelity 0.972 on real Intel).
  QuantStoreSLAM   WRITE-TIME quantized segment store. A segment accumulates
                   at full precision while its anchor is the active one (in
                   hardware: a 1-deep high-precision accumulation buffer) and
                   is quantized ONCE when the active anchor advances; every
                   later read — frontend recency bundle, loop-candidate
                   bundles, coherence checks — sees only the quantized
                   content. This is the deployable memory claim, unlike the
                   read-time requantization of scratch_quant_e2e.py (which
                   models nothing storable). nph=0 neutralises to the parent
                   bit-exactly (asserted by `selftest`).
  ops_report       analytic per-keyframe op/bit counts of the shipped
                   pipeline at its real operating point (the FPGA sizing
                   story: DSP-MACs per keyframe, LUT lookups, map BRAM bits).

Anti-oracle: the .gfs reference only ever scores (run harness below).
Deterministic: no RNG anywhere in the pipeline.

Usage:
  python3 ssp_fpga.py selftest              # neutralised == parent, quick
  python3 ssp_fpga.py sweep intel fr101     # write-time ladder on given logs
  python3 ssp_fpga.py ops                   # op/memory sizing report
"""
import sys
import time

import numpy as np

import ssp_slam as S
import ssp_slam_carmen as C
import ssp_slam_loop as L
import ssp_bounded as B

VALID_MAX = 40.0
LOGS = {"intel": "data/intel.log", "fr101": "data/fr101.log",
        "fr079": "data/fr079.log", "aces": "data/aces_publicb.log"}


# ---------------------------------------------------------------------------
# Polar quantizer (identical numerics to scratch_quant.q_polar /
# scratch_quant_e2e._quant, kept verbatim for continuity with the static study)
# ---------------------------------------------------------------------------

def q_polar(v, nph, nmag):
    """Quantize a complex phasor vector to nph phase bins x nmag magnitude
    levels (mid-tread), per-vector scale = 99th percentile magnitude.
    Storage cost: log2(nph)+log2(nmag) bits/phasor + one f32 scale."""
    s = float(np.percentile(np.abs(v), 99)) + 1e-12
    dph = 2 * np.pi / nph
    phq = dph * np.round(np.angle(v) / dph)
    mstep = s / nmag
    magq = np.clip(mstep * (np.floor(np.clip(np.abs(v), 0, s) / mstep) + 0.5),
                   0, s)
    return magq * np.exp(1j * phq)


def q_polar_rings(v, nph, nmag, dead=False):
    """q_polar with a PER-RING magnitude scale (6 f32 per vector instead of
    1). Rationale: segment magnitudes are wildly ring-dependent — the coarse
    relocalization rings (lam 5.3/12.8 m) stay phase-coherent over a segment's
    ~2 m extent and dominate a per-vector 99th-percentile scale, so the two
    FINEST rings (whose coherence is the closure veto's decision statistic)
    collapse into the bottom magnitude levels. The static quantization bench
    (scratch_quant.py) used a 3-octave lattice without the relo rings, which
    is exactly why it did not predict the end-to-end closure regression.

    dead=True switches the magnitude code from mid-tread (which maps an EMPTY
    component to mstep/2 — a spurious noise-floor phasor at a junk phase, D of
    which add cross-correlation mass between unrelated segments) to a uniform
    code with a TRUE ZERO level: k*s/(nmag-1), k = 0..nmag-1."""
    A = v.reshape(L.N_RING, L.N_ANG)
    s = np.percentile(np.abs(A), 99, axis=1)[:, None] + 1e-12
    dph = 2 * np.pi / nph
    phq = dph * np.round(np.angle(A) / dph)
    if dead:
        lev = s / (nmag - 1)
        magq = lev * np.clip(np.round(np.abs(A) / lev), 0, nmag - 1)
    else:
        mstep = s / nmag
        magq = np.clip(
            mstep * (np.floor(np.clip(np.abs(A), 0, s) / mstep) + 0.5), 0, s)
    return (magq * np.exp(1j * phq)).reshape(-1)


class QuantStoreSLAM(B.BoundedSLAM):
    """Write-time polar-quantized segment store (freeze on anchor advance)."""

    def __init__(self, nph=16, nmag=4, ring_scales=False, dead=False, **kw):
        super().__init__(**kw)
        self.nph, self.nmag = nph, nmag
        self.ring_scales = ring_scales
        self.dead = dead
        self._frozen = set()
        self._active = -1

    def _freeze(self, a):
        if self.ring_scales:
            def q(v, nph, nmag):
                return q_polar_rings(v, nph, nmag, dead=self.dead)
        else:
            q = q_polar
        self.segvec[a] = q(self.segvec[a], self.nph, self.nmag) \
            .astype(self.store_dtype)
        if a in self.segder:
            self.segder[a] = q(self.segder[a], self.nph, self.nmag) \
                .astype(self.store_dtype)
        self._frozen.add(a)

    def add_keyframe(self, pts, w, guess):
        est = super().add_keyframe(pts, w, guess)
        if self.nph:
            aid = len(self.anchors) - 1
            if aid != self._active:
                for a in list(self.segvec):
                    if a < aid and a not in self._frozen:
                        self._freeze(a)
                self._active = aid
        return est

    def memory_kb(self):
        if not self.nph:
            return super().memory_kb()
        bits = np.log2(self.nph) + np.log2(self.nmag)
        n = len(self.segvec)
        payload = n * L.W.shape[0] * bits * 2 / 8      # segvec + segder
        n_sc = L.N_RING if self.ring_scales else 1     # f32 scales per vector
        return (payload + n * 2 * 4 * n_sc) / 1024


# ---------------------------------------------------------------------------
# E1: fine rotation via the scan's d/dtheta derivative (FPGA: 11 -> 3 encodes
# per match — the fine stage becomes permutations + AXPY instead of re-encodes)
# ---------------------------------------------------------------------------

class DerFineMatcher(S.Matcher):
    """Matcher whose FINE rotation stage approximates the scan encoding at
    theta_c + delta by  perm_m[S0] + delta * perm_m[D0]  (first-order Lie
    correction, |delta| <= 1.5 deg — exactly the regime the stored-derivative
    ablation validated as beneficial). Saves the 9 fine-stage re-encodes; in
    hardware the encoder then runs 3x per match (2 coarse bases + 1 derivative)
    instead of 11x. der_fine=False neutralises to the parent (bit-exact)."""

    def __init__(self, *a, der_fine=True, **kw):
        super().__init__(*a, **kw)
        self.der_fine = der_fine

    def _encode_der(self, pts, weights):
        """d/dtheta of encode(R_theta pts) at theta=0 (rotation about origin)."""
        A = np.exp(1j * (pts @ self.enc.W.T))
        Cx = pts[:, 0:1] * self.enc.W[:, 1] - pts[:, 1:2] * self.enc.W[:, 0]
        return (weights[:, None] * 1j * Cx * A).sum(0)

    def match(self, M, pts, weights, guess):
        if not self.der_fine or self.perm is None:
            return super().match(M, pts, weights, guess)
        n_ang = self.perm[1]
        lat = np.pi / n_ang
        S00 = self.enc.encode(pts, weights)
        p_h = pts @ S._rot(lat / 2).T
        S0h = self.enc.encode(p_h, weights)
        bases = ((S00, 0.0), (S0h, lat / 2))
        m0 = int(round(guess[2] / lat))
        h = int(np.ceil(self.rot_half / lat - 1e-9))
        best = (-np.inf, None, None, None, None)
        for m in range(m0 - h, m0 + h + 1):
            for bi, (S0, off) in enumerate(bases):
                Sv = self._rot_permute(S0, m)
                t, sc, _, _ = self._best_translation(
                    M, Sv, guess[:2], self.coarse_off, self.E_coarse)
                if sc > best[0]:
                    best = (sc, m * lat + off, t, m, bi)
        _, th_c, t_c, m_c, bi_c = best

        # fine stage: first-order rotation about the winning coarse angle
        D0 = self._encode_der((pts, p_h)[bi_c], weights)
        Sc = self._rot_permute(bases[bi_c][0], m_c)
        Dc = self._rot_permute(D0, m_c)
        fine_step = np.deg2rad(0.375)
        deltas = fine_step * np.arange(-4, 5)
        th_scores, results = [], []
        for d in deltas:
            Sv = Sc + d * Dc
            t, sc, scores, kk = self._best_translation(
                M, Sv, t_c, self.fine_off, self.E_fine)
            th_scores.append(sc)
            results.append((t, scores, kk))
        j = int(np.argmax(th_scores))
        th_f = th_c + deltas[j]
        t_f, scores, kk = results[j]
        if 0 < j < len(deltas) - 1:
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


class FrontSLAM(QuantStoreSLAM):
    """FPGA-frontend variants on top of the quantized store: der_fine swaps
    both matchers for DerFineMatcher. nph=0 + der_fine=False == parent."""

    def __init__(self, der_fine=False, **kw):
        super().__init__(**kw)
        if der_fine:
            self.matcher = DerFineMatcher(L.ENC_MAIN, t_half=0.48,
                                          rot_half_deg=9, rot_step_deg=1.5,
                                          perm=(4, L.N_ANG))
            self.cmatcher = DerFineMatcher(L.ENC_MAIN, t_half=0.72,
                                           rot_half_deg=9, rot_step_deg=1.5,
                                           perm=(4, L.N_ANG))


# ---------------------------------------------------------------------------
# Chaos control: how sensitive is the closure cascade to ANY map perturbation?
# Adds i.i.d. relative Gaussian noise to each segment at freeze time (same
# hook as quantization, deterministic seed). If a numerically-negligible
# perturbation (1e-6) already moves a log's ATE by O(1 m), then quant-level
# ATE deltas on that log measure CHAOS, not quantization damage, and the
# verdict must rest on the noise-response curve, not a single number.
# ---------------------------------------------------------------------------

class NoiseStoreSLAM(QuantStoreSLAM):
    def __init__(self, eps=1e-6, **kw):
        super().__init__(nph=1, **kw)          # reuse freeze hook, no quant
        self.eps = eps
        self._nrng = np.random.default_rng(1234)

    def _freeze(self, a):
        for d in (self.segvec, self.segder):
            if a in d:
                v = d[a]
                s = float(np.abs(v).mean())
                d[a] = (v + self.eps * s * (
                    self._nrng.standard_normal(len(v))
                    + 1j * self._nrng.standard_normal(len(v)))
                ).astype(self.store_dtype)
        self._frozen.add(a)

    def memory_kb(self):
        return B.BoundedSLAM.memory_kb(self)


def chaos_sweep(logs):
    print("CHAOS CONTROL: freeze-time relative map noise vs ATE\n")
    for lg in logs:
        r0 = run_log(LOGS[lg], B.BoundedSLAM)
        print(f"  {lg}: SHIPPED               ATE {r0['ate']:.3f}  "
              f"med {r0['med']:.3f}  loops {r0['loops']}", flush=True)
        for eps in (1e-6, 1e-3, 1e-2, 3e-2):
            r = run_log(LOGS[lg], NoiseStoreSLAM, eps=eps)
            print(f"     noise {eps:7.0e}        ATE {r['ate']:.3f}  "
                  f"med {r['med']:.3f}  loops {r['loops']}  "
                  f"[veto {r['veto']} infl {r['infl']} innov {r['innov']}]",
                  flush=True)
        print(flush=True)


# ---------------------------------------------------------------------------
# Integer front-path model (the binary-VSA track). Values are quantized onto
# the exact grids the hardware would produce (ROM-rounded cis, fixed-point
# weights) but carried in float64, which is exact for integer-valued data
# below 2^53 — so this IS the integer pipeline, bit-accurately, minus only
# wrap-around overflow (accumulators are assumed wide enough; the report
# tracks magnitudes). HW split modeled: encoder + matcher correlations + map
# store/bundle-shift on fabric (integer); pose math, gates, per-ring
# coherence statistics, and the sparse relax on the host CPU (float).
# ---------------------------------------------------------------------------

class IntSpec:
    """Bit widths of the fabric arithmetic.
    lut_addr: cis ROM address bits (phase resolution = 2pi / 2^lut_addr)
    lut_bits: signed cis ROM value bits (cos/sin scaled by 2^(lut_bits-1)-1)
    w_frac:   sample-weight fixed-point fraction bits (grid 2^-w_frac)
    unit_w:   drop weights entirely (binary-VSA style, 1 per point)"""

    def __init__(self, lut_addr=10, lut_bits=9, w_frac=9, unit_w=False):
        self.lut_addr, self.lut_bits = lut_addr, lut_bits
        self.w_frac, self.unit_w = w_frac, unit_w

    def __str__(self):
        w = "unit" if self.unit_w else f"w{self.w_frac}"
        return f"addr{self.lut_addr}/val{self.lut_bits}/{w}"


def cis_int(phi, spec):
    """Phase -> ROM-quantized unit phasor: nearest of 2^lut_addr bin centers,
    cos/sin each rounded to lut_bits signed integer (divided back by the
    scale, so values sit exactly on the hardware grid)."""
    n = 1 << spec.lut_addr
    idx = np.round(np.asarray(phi) * (n / (2 * np.pi))).astype(np.int64) % n
    M = (1 << (spec.lut_bits - 1)) - 1
    ang = idx * (2 * np.pi / n)
    return (np.round(M * np.cos(ang)) + 1j * np.round(M * np.sin(ang))) / M


class IntEncoder:
    """Drop-in for SSPEncoder on the fabric: phase MAC -> cis ROM -> weighted
    integer accumulate."""

    def __init__(self, W, spec):
        self.W = W
        self.spec = spec

    def encode(self, pts, weights):
        c = cis_int(pts @ self.W.T, self.spec)
        if self.spec.unit_w:
            return c.sum(0)
        g = 1 << self.spec.w_frac
        return c.T @ (np.round(weights * g) / g)

    def shift(self, d):
        return cis_int(self.W @ np.asarray(d), self.spec)


class IntMatcher(DerFineMatcher):
    """Matcher over the integer encoder; translation-bank ROMs (E matrices)
    quantized to the same cis grid. der_fine optional (E1)."""

    def __init__(self, W, spec, der_fine=False, **kw):
        super().__init__(IntEncoder(W, spec), der_fine=der_fine, **kw)
        self.E_coarse = cis_int(np.angle(self.E_coarse), spec)
        self.E_fine = cis_int(np.angle(self.E_fine), spec)

    def _encode_der(self, pts, weights):
        # derivative encode on the fabric: Cx fixed-point, cis from ROM
        A = cis_int(pts @ self.enc.W.T, self.enc.spec)
        Cx = pts[:, 0:1] * self.enc.W[:, 1] - pts[:, 1:2] * self.enc.W[:, 0]
        if self.enc.spec.unit_w:
            return (1j * Cx * A).sum(0)
        g = 1 << self.enc.spec.w_frac
        return (np.round(weights * g)[:, None] / g * 1j * Cx * A).sum(0)


class IntSLAM(FrontSLAM):
    """Full FPGA-arithmetic pipeline model: integer encoder + matcher +
    quantized store + integer bundle-shift. spec=None neutralises the
    arithmetic (parent float path)."""

    def __init__(self, spec=None, der_fine=False, **kw):
        super().__init__(**kw)
        self.spec = spec
        if spec is not None:
            self.matcher = IntMatcher(L.ENC_MAIN.W, spec, der_fine=der_fine,
                                      t_half=0.48, rot_half_deg=9,
                                      rot_step_deg=1.5, perm=(4, L.N_ANG))
            self.cmatcher = IntMatcher(L.ENC_MAIN.W, spec, der_fine=der_fine,
                                       t_half=0.72, rot_half_deg=9,
                                       rot_step_deg=1.5, perm=(4, L.N_ANG))
        elif der_fine:
            raise ValueError("der_fine without spec: use FrontSLAM")

    def world_vec_seg(self, aid):
        if self.spec is None:
            return super().world_vec_seg(aid)
        a = self.anchors[aid]
        m = int(round(a[2] * L.N_ANG / np.pi))
        delta = a[2] - m * np.pi / L.N_ANG
        v = L.rot_permute(self.segvec[aid], m)
        if self.use_der and aid in self.segder:
            v = v + delta * L.rot_permute(self.segder[aid], m)
        return cis_int(L.W @ a[:2], self.spec) * v


# ---------------------------------------------------------------------------
# E2: per-beam point encoding (FPGA: drops the segment-resampling / occlusion
# geometry preprocessing — each hit streams straight into the encoder with an
# arc-length weight r*dtheta; no hit-to-hit bridges, so no phantom walls)
# ---------------------------------------------------------------------------

def points_from_scan_occw(rr, beam):
    """E3 mechanism probe: samples ONLY at real hit positions (like E2) but
    with the SHIPPED occlusion-filtered chord weighting (each hit carries half
    of each kept adjacent chord; lone hits get SAMPLE_DS) — the same surface
    integral as shipped WITHOUT interpolated positions. Discriminates
    'interpolated chord mass is the damage' from 'the r*dtheta weighting /
    missing occlusion filter is the difference'."""
    hit = np.isfinite(rr)
    idx = np.flatnonzero(hit)
    if len(idx) < 2:
        return np.zeros((0, 2)), np.zeros(0)
    r = rr[idx]
    pts = np.stack([r * np.cos(beam[idx]), r * np.sin(beam[idx])], 1)
    consec = np.diff(idx) <= 2
    chord_v = pts[1:] - pts[:-1]
    chord = np.linalg.norm(chord_v, axis=1)
    dr = np.abs(np.diff(r))
    tang = np.sqrt(np.maximum(chord ** 2 - dr ** 2, 0.0))
    keep = consec & (chord <= S.MAX_CHORD) & (dr <= S.OCC_RATIO * tang)
    w = np.zeros(len(idx))
    w[:-1] += np.where(keep, chord / 2, 0.0)
    w[1:] += np.where(keep, chord / 2, 0.0)
    lone = w == 0.0
    w[lone] = S.SAMPLE_DS
    return pts, w


def points_from_scan(rr, beam, wcap=None):
    """Per-hit sample points + arc-length weights (r * beam spacing).
    wcap caps the per-point weight: a far return is one point carrying its
    whole (large) arc footprint, so a single far clutter hit (glass, person)
    can out-weigh 4+ near samples — capping bounds that leverage."""
    hit = np.isfinite(rr)
    idx = np.flatnonzero(hit)
    if len(idx) < 2:
        return np.zeros((0, 2)), np.zeros(0)
    r = rr[idx]
    pts = np.stack([r * np.cos(beam[idx]), r * np.sin(beam[idx])], 1)
    dth = abs(beam[1] - beam[0]) if len(beam) > 1 else np.deg2rad(1.0)
    w = r * dth
    if wcap is not None:
        w = np.minimum(w, wcap)
    return pts, w


# ---------------------------------------------------------------------------
# Standard harness (same parsing / keyframing / eval as ssp_bounded_carmen)
# ---------------------------------------------------------------------------

def run_log(path, cls, cap=None, sample="seg", **kw):
    keys = C.keyframes(C.parse_flaser(path))
    if cap:
        keys = keys[:cap]
    n = len(keys)
    nb = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(nb) * (180.0 / nb))
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    slam = cls(robust=True, attempt_every=4, relax_every=25, gap_kf=300,
               recent_aids=12, **kw)
    slam.store_dtype = np.complex64
    est = np.zeros((n, 3))
    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < VALID_MAX, r, np.inf)
        if sample == "point":
            pts, w = points_from_scan(rr, beam)
        elif sample == "pointcap":
            pts, w = points_from_scan(rr, beam, wcap=0.25)
        elif sample == "hitw":
            pts, w = points_from_scan_occw(rr, beam)
        else:
            pts, w, _ = S.scan_to_samples(rr, beam)
        guess = opose if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
    if slam.dirty:
        slam.relax()
    dt = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    nloop = sum(1 for e in slam.edges if e[5] == "loop")
    ref = C.parse_flaser(path.replace(".log", ".gfs.log"))
    rts = np.array([t for _, _, t in ref])
    rxy = np.stack([p[:2] for _, p, _ in ref])
    j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
    good = np.abs(rts - kts[j]) < 0.3
    al = C.align_se2(fin[j[good], :2], rxy[good])
    e = np.linalg.norm(al - rxy[good], axis=1)
    return dict(ate=float(np.sqrt((e ** 2).mean())), med=float(np.median(e)),
                loops=nloop, ms=dt / n * 1e3, mem_kb=slam.memory_kb(),
                pruned=slam.n_pruned, veto=slam.n_veto, infl=slam.n_inflate,
                innov=slam.n_innov_rej, coh_ref=slam.coh_ref,
                jit=getattr(slam, "n_jit_rej", 0), fin=fin)


# ---------------------------------------------------------------------------
# FPGA sizing: analytic op counts of the shipped hot path
# ---------------------------------------------------------------------------

def ops_report(n_pts=200):
    D_main = 4 * L.N_ANG            # matched band the matcher touches
    D_full = L.W.shape[0]           # + relocalization rings for storage/coh
    n_coarse = len(S._grid_offsets(0.48, 0.06))    # frontend coarse offsets
    n_fine = len(S._grid_offsets(0.06, 0.02))
    h = int(np.ceil(np.deg2rad(9) / (np.pi / L.N_ANG) - 1e-9))
    n_rot_banks = (2 * h + 1) * 2   # coarse rotations x 2 encode bases
    enc = n_pts * D_main            # one scan encode (phase MAC + LUT + acc)
    coarse = n_rot_banks * (D_main + n_coarse * D_main)   # c=conj(M)S + E@c
    fine_enc = 9 * enc              # fine stage re-encodes 9 thetas
    fine = 9 * (D_main + n_fine * D_main)
    fold = n_pts * D_full * 3       # segvec + segder fold (der ~2x work)
    # loop attempt every 4 kf: cmatcher match (t_half=0.72 -> bigger grid)
    n_coarse_c = len(S._grid_offsets(0.72, 0.06))
    coarse_c = n_rot_banks * (D_main + n_coarse_c * D_main)
    loop = (2 * enc + coarse_c + fine_enc + fine
            + 2 * n_pts * D_full + 8 * D_full) / 4        # + coh/ridge probes
    per_kf = dict(frontend_encode=2 * enc, frontend_coarse=coarse,
                  frontend_fine=fine_enc + fine, segment_fold=fold,
                  loop_amortized=loop)
    tot = sum(per_kf.values())
    print(f"shipped hot path @ n_pts={n_pts}, D_main={D_main}, D={D_full}")
    for k, v in per_kf.items():
        print(f"  {k:>18}: {v / 1e6:6.2f} M MAC-equiv")
    print(f"  {'TOTAL':>18}: {tot / 1e6:6.2f} M MAC-equiv per keyframe")
    for hz in (10, 20):
        print(f"    @ {hz} Hz keyframes: {tot * hz / 1e9:.2f} GMAC/s "
              f"(~{tot * hz / 200e6:.0f} DSP48 @ 200 MHz)")
    for tag, n_seg, bits in (("Intel c64", 698, 64), ("Intel 6-bit", 698, 6),
                             ("MIT c64", 2401, 64), ("MIT 6-bit", 2401, 6)):
        kb = n_seg * D_full * 2 * bits / 8 / 1024
        print(f"  map store {tag:>12}: {kb:8.0f} KB "
              f"({n_seg} segments x {D_full} x 2 x {bits} b)")


# ---------------------------------------------------------------------------

class BandSLAM(IntSLAM):
    """Any pipeline config + freeze-time relative map noise, for perturbation
    BAND probes. Lesson that forced this harness (combo-aces): a single run's
    ATE on a perturbation-sensitive log is a basin draw — the combo scored
    2.149 on aces at eps=0 but 8.65 at eps=1e-6 — so every claimed config is
    reported as its band over {0, 1e-6, 1e-3 x 2 seeds}, not a point."""

    def __init__(self, eps=0.0, seed=1, **kw):
        super().__init__(**kw)
        self.eps = eps
        self._brng = np.random.default_rng(seed)
        self._active2 = -1

    def _noise(self, a):
        for d in (self.segvec, self.segder):
            if a in d:
                v = d[a]
                s = float(np.abs(v).mean())
                d[a] = (v + self.eps * s * (
                    self._brng.standard_normal(len(v))
                    + 1j * self._brng.standard_normal(len(v)))
                ).astype(self.store_dtype)

    def _freeze(self, a):            # noise BEFORE quantization
        if self.eps:
            self._noise(a)
        super()._freeze(a)

    def add_keyframe(self, pts, w, guess):
        est = super().add_keyframe(pts, w, guess)
        if self.eps and not self.nph:            # noise-only (unquantized)
            aid = len(self.anchors) - 1
            if aid != self._active2:
                for a in list(self.segvec):
                    if a < aid and a not in self._frozen:
                        self._noise(a)
                        self._frozen.add(a)
                self._active2 = aid
        return est


BAND_EPS = ((0.0, 1), (1e-6, 1), (1e-3, 1), (1e-3, 2))

BAND_CONFIGS = (
    ("shipped ", dict(spec=None, nph=0), "seg"),
    ("E2 point", dict(spec=None, nph=0), "point"),
    ("FPGA8   ", dict(spec="i8", nph=16, nmag=4, ring_scales=True), "point"),
    ("BINARY  ", dict(spec="qpsk", nph=4, nmag=1, ring_scales=True), "point"),
)


def band_table(logs):
    """The session deliverable: config x log perturbation bands."""
    for lg in logs:
        for name, kw0, sm in BAND_CONFIGS:
            kw = dict(kw0)
            if kw["spec"] == "i8":
                kw["spec"] = IntSpec(8, 7, 7)
            elif kw["spec"] == "qpsk":
                kw["spec"] = IntSpec(2, 2, 0, unit_w=True)
            vals = []
            for eps, seed in BAND_EPS:
                r = run_log(LOGS[lg], BandSLAM, sample=sm, eps=eps,
                            seed=seed, **kw)
                vals.append(r["ate"])
                print(f"  {lg} {name} eps{eps:7.0e}/s{seed}  "
                      f"ATE {r['ate']:.3f}  med {r['med']:.3f}  "
                      f"loops {r['loops']}  mem {r['mem_kb']:.0f}", flush=True)
            v = np.array(vals)
            print(f"  {lg} {name} BAND [{v.min():.2f} .. {v.max():.2f}] "
                  f"median {np.median(v):.2f}", flush=True)
        print(flush=True)


def ops_report_binary():
    """Datapath sketch + resource budget for the quantized/binary system.

    Store: per-ring polar codes — nph phase bins (log2 nph bits) x nmag mag
    levels (log2 nmag bits) + 6 per-ring f32 scales/vector. Arithmetic: cis
    ROM at 2^addr entries x 2 x val bits; correlation banks are ROMs of the
    same format; a map-x-scan product needs (phase add mod nph) -> cos LUT
    (nph entries) x (mag x mag -> product LUT, nmag^2 entries) -> signed
    accumulate. At the QPSK extreme the cos LUT degenerates to {+1, 0, -1}:
    the correlate is a masked signed popcount over mag codes — the classic
    binary-VSA datapath."""
    for tag, n_seg, bits in (("Intel 6b store", 698, 6),
                             ("Intel 4b (16ph phase-only)", 698, 4),
                             ("MIT 6b store", 2401, 6),
                             ("MIT 4b phase-only", 2401, 4)):
        kb = n_seg * L.W.shape[0] * 2 * bits / 8 / 1024 \
            + n_seg * 2 * L.N_RING * 4 / 1024
        print(f"  map {tag:>28}: {kb:7.0f} KB "
              f"(+ per-ring scales included)")
    # per-keyframe fabric work at the shipped operating point (n_pts=200)
    n_pts = 200
    D = 4 * L.N_ANG
    enc = n_pts * D                       # phase MAC + ROM lookup + acc
    banks = 14 * (D + 289 * D) + 9 * (D + 49 * D)
    print(f"  fabric/kf: {2 * enc + 9 * enc:,} encode ops + {banks:,} "
          f"correlate MACs -> int8/int4 LUT-adds (no DSP needed at QPSK)")
    print("  BRAM fit: Artix-7 100T ~607 KB BRAM -> the Intel 6b map "
          "(368 KB + scales) fits on-chip with headroom; MIT needs 1.3 MB "
          "-> Zynq US+ BRAM/URAM or DDR streaming of cold cells")


def selftest():
    """Neutralised subclass must be bit-exact to the parent (PROTOCOL sec 1)."""
    cap = 1200
    a = run_log(LOGS["fr101"], B.BoundedSLAM, cap=cap)
    b = run_log(LOGS["fr101"], QuantStoreSLAM, cap=cap, nph=0)
    d = float(np.abs(a["fin"] - b["fin"]).max())
    print(f"selftest fr101[:{cap}]  parent ATE {a['ate']:.4f}  "
          f"neutralised ATE {b['ate']:.4f}  max|dpose| {d:.2e}")
    assert d == 0.0, "neutralised QuantStoreSLAM is NOT bit-exact to parent"
    print("selftest ok: neutralised == parent bit-exact")


def sweep(logs, ring=False):
    print("WRITE-TIME quantized segment store, end-to-end ATE "
          f"(freeze-on-advance; deployed 60/4-oct lattice; "
          f"{'PER-RING' if ring else 'per-vector'} scales)\n")
    for lg in logs:
        r0 = run_log(LOGS[lg], B.BoundedSLAM)
        print(f"  {lg}: SHIPPED c64            ATE {r0['ate']:.3f}  "
              f"med {r0['med']:.3f}  loops {r0['loops']}  "
              f"mem {r0['mem_kb']:.0f} KB  "
              f"[veto {r0['veto']} infl {r0['infl']} innov {r0['innov']}]",
              flush=True)
        for nph, nmag in ((32, 8), (16, 4), (8, 2)):
            r = run_log(LOGS[lg], QuantStoreSLAM, nph=nph, nmag=nmag,
                        ring_scales=ring)
            bits = np.log2(nph) + np.log2(nmag)
            tag = "  <-free" if abs(r["ate"] - r0["ate"]) < 0.15 else \
                  ("  <-ok" if r["ate"] < r0["ate"] + 0.4 else "  <-COSTS")
            print(f"     wq {nph:2d}ph x {nmag}mag ({bits:.0f}b)   "
                  f"ATE {r['ate']:.3f}  med {r['med']:.3f}  "
                  f"loops {r['loops']}  mem {r['mem_kb']:.0f} KB  "
                  f"[veto {r['veto']} infl {r['infl']} innov {r['innov']}]"
                  f"{tag}", flush=True)
        print(flush=True)


def int_sweep(logs, store=(0, 0)):
    """Integer-arithmetic ladder: fabric bit widths down to the binary (QPSK)
    extreme. store=(nph,nmag) optionally combines the quantized store."""
    nph, nmag = store
    print("INTEGER front-path arithmetic ladder, end-to-end ATE "
          f"(store={'c64' if not nph else f'{nph}ph x {nmag}mag ring'})\n")
    specs = [IntSpec(10, 9, 9), IntSpec(8, 7, 7), IntSpec(6, 5, 5),
             IntSpec(4, 4, 4), IntSpec(3, 3, 3),
             IntSpec(2, 2, 0, unit_w=True)]
    for lg in logs:
        r0 = run_log(LOGS[lg], B.BoundedSLAM)
        print(f"  {lg}: SHIPPED float         ATE {r0['ate']:.3f}  "
              f"med {r0['med']:.3f}  loops {r0['loops']}", flush=True)
        for sp in specs:
            r = run_log(LOGS[lg], IntSLAM, spec=sp, nph=nph, nmag=nmag,
                        ring_scales=bool(nph))
            d = r["ate"] - r0["ate"]
            tag = "  <-free" if abs(d) < 0.15 else \
                  ("  <-ok" if abs(d) < 0.4 else "  <-COSTS")
            print(f"     int {str(sp):>18}  ATE {r['ate']:.3f}  "
                  f"med {r['med']:.3f}  loops {r['loops']}{tag}", flush=True)
        print(flush=True)


def front_sweep(logs):
    """E1 (derivative fine rotation) and E2 (per-beam point encoding) vs
    shipped, full pipeline, ATE must hold multi-log."""
    print("FPGA-frontend simplification candidates, end-to-end ATE\n")
    for lg in logs:
        r0 = run_log(LOGS[lg], B.BoundedSLAM)
        print(f"  {lg}: SHIPPED               ATE {r0['ate']:.3f}  "
              f"med {r0['med']:.3f}  loops {r0['loops']}", flush=True)
        for tag, kw, sm in (("E1 der-fine   ", dict(der_fine=True, nph=0), "seg"),
                            ("E2 point-enc  ", dict(nph=0), "point"),
                            ("E1+E2         ", dict(der_fine=True, nph=0), "point")):
            r = run_log(LOGS[lg], FrontSLAM, sample=sm, **kw)
            d = r["ate"] - r0["ate"]
            tag2 = "  <-ok" if abs(d) < 0.3 else ("  <-WIN" if d < 0 else "  <-COSTS")
            print(f"     {tag} ATE {r['ate']:.3f}  med {r['med']:.3f}  "
                  f"loops {r['loops']}{tag2}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if what == "selftest":
        selftest()
    elif what == "sweep":
        args = [a for a in sys.argv[2:] if a != "--ring"]
        sweep(args or ["fr101", "intel"], ring="--ring" in sys.argv)
    elif what == "front":
        front_sweep(sys.argv[2:] or ["fr101", "intel"])
    elif what == "int":
        int_sweep(sys.argv[2:] or ["fr101", "intel"])
    elif what == "chaos":
        chaos_sweep(sys.argv[2:] or ["intel"])
    elif what == "ops":
        ops_report()
    elif what == "opsbin":
        ops_report_binary()
    elif what == "band":
        band_table(sys.argv[2:] or ["fr101"])
