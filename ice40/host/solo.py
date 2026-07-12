#!/usr/bin/env python3
"""solo.py — GOLDEN INTEGER MODEL of the v7 standalone ("dumb-cable") loop.

GATE STATUS (2026-07-12 build session): TWO gated RTL cores reproduce
this model BIT-EXACTLY in sim. (a) The v6-parallel pair (solo_tracker.v
+ encoder_solo.v): k1 220/220 + rl 160/160 (REFINE=1 fixtures, historic
float cs_of anchors baked in the hex). (b) The v7-LEAN stack
(solo_tracker.v REFINE=0 + encoder_lean.v + top_solo.v): bridge 32/32 +
readback 240/240, l1 220/220, and the FULL UART top end-to-end
(solo_top.py: pose frames + map dump incl. liveness/scales planes,
byte-for-byte). Fixtures via solo_vectors.py (main / lean); gates via
solo_bridge.py, solo_step.py, solo_top.py. Change anything here and the
RTL gates are the acceptance for the change. NOTE: cs_of is TABLE-
DEFINED (quarter-wave fold) and REFINE/liveness_int carry build-session
contracts — see the module comments at each.

User directive (2026-07-12): the chip must run SLAM standalone; host is a
sensor cable (points in, pose out) + map dump reader. This file defines the
bit-exact integer semantics the v7 RTL will implement, built ON TOP of the
accepted v3–v6 goldens in ssp_ice40 (imported, never edited):

  encode_int / shift_for / match_int / mcode_from_vec   (accepted on silicon)

New golden pieces (v7 contract):
  - Anchor table ints: xy in U units (i32), heading in TAU/960 units (i16),
    cos/sin as Q15 from the heading ROM mapping.
  - rel/pose transforms: Q15 rotate with round-half-up (+2^14) >> 15.
  - SoloTracker: the FabricLoc state machine in integers — 7x7x5 sweep grid
    around the odometry prediction, first-max argmax (== np.argmax == RTL),
    one edge-recenter retry, EMA score gate (ema += (score-ema)>>6; commit
    iff score > 29*(ema>>6); relock iff score > 35*(ema>>6); re-search every
    12 lost kf on a 15x15x7 grid), integer parabolic sub-grid refine
    (trunc-toward-zero, clamped ±SWEEP_T/2; live-parity semantics).
  - encode_int_at: encode with an integer SE2 pre-transform between the
    az->xy stage and the u-projection stage (the fold-at-pose path).
  - mcode_int: comparator-form QPSK code from (I,Q) ints; equivalent to
    G.mcode_from_vec incl. the |I|==|Q| tie (half-even round -> I-axis).
  - fold/freeze: 5-kf segments, open-segment accumulators, freeze to codes.
    Derivative vector note: cross-projection = u index-shifted by 30 angles
    (x sin - y cos = u(theta - pi/2)), so der-accumulate reuses the encode
    projections; the per-ring 2pi/lambda scale is applied at LAPTOP decode.

Anti-oracle: GT appears in the bench scoring only. Determinism: pure int
state, no wall clock, no RNG in the tracker.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ssp_ice40 as G                                     # noqa: E402

U = 0.25 / 256                     # position quantum [m] (fabric contract)
TAU_Q = 960                        # full circle in heading units (2pi/960)
RHO_Q = TAU_Q // 120               # pi/60 = 8 units
SWEEP_T, SWEEP_N, SWEEP_R = 12, 3, 2
RS_T, RS_N, RS_R = 48, 7, 6       # re-search grid (matches live _research)
EMA_SH, COMMIT_NUM, RELOCK_NUM = 6, 29, 35
LOST_PERIOD = 12
SEG_KF = 5
GATE_FOLD = False    # fold-only-on-commit REJECTED: starvation feedback
                     # (mixed: 32 segs, 1149 holds, err 0.85->1.31/2.15)
K_TRY = 1            # v7.1: match best-of-K nearest segments (2b codes are
                     # norm-equal, so raw totals compare across segments —
                     # unlike the float live-v2 handoff that thrashed).
                     # K=1 preserves the validated v7.0 parity semantics.
HYST_NUM = 66        # challenger must beat the incumbent by 66/64 (~3%)
REFINE = True        # integer parabolic sub-grid refine. False = the v7-lean
                     # RTL build (divider + tot RAM dropped for LC fit); the
                     # banked claim is p90-only gain — the lean fixture gen
                     # sets False and the delta is banked in RESULTS.


def q15(v):
    return int(round(32767 * v))


# Heading ROM: QUARTER-WAVE TABLE-DEFINED (v7-lean contract). The float
# form round(32767*cos(a)) is NOT quarter-symmetric at exact-half points
# (np.sin(pi/6) evaluates 0.4999999999999999 while np.cos(pi/3) gives
# 0.5000000000000001 -> 16383 vs 16384), so the golden is DEFINED as the
# folded 241-entry table — structurally identical to the RTL ROM. Deltas
# vs the old float definition: +-1 LSB at slop points only.
_CS_T = np.array([round(32767 * float(np.cos(o * 2 * np.pi / TAU_Q)))
                  for o in range(241)], np.int64)


def _cs_fold(h):
    q, o = h // 240, h % 240
    if q == 0:
        return int(_CS_T[o])
    if q == 1:
        return -int(_CS_T[240 - o]) if o else 0
    if q == 2:
        return -int(_CS_T[o])
    return int(_CS_T[240 - o]) if o else 0


def cs_of(h_q):
    """Heading units (TAU/960) -> (cos, sin) Q15 via the quarter-wave ROM
    fold (cq = fold(h); sq = fold(h + 720 mod 960))."""
    h = int(h_q) % TAU_Q
    return _cs_fold(h), _cs_fold((h + 720) % TAU_Q)


def rot_q15(cq, sq, x, y):
    """R(h) @ (x, y) in ints: round-half-up Q15."""
    xr = (cq * x - sq * y + (1 << 14)) >> 15
    yr = (sq * x + cq * y + (1 << 14)) >> 15
    return int(xr), int(yr)


def irot_q15(cq, sq, x, y):
    """R(-h) @ (x, y)."""
    xr = (cq * x + sq * y + (1 << 14)) >> 15
    yr = (-sq * x + cq * y + (1 << 14)) >> 15
    return int(xr), int(yr)


def wrap_q(dh):
    return (dh + TAU_Q // 2) % TAU_Q - TAU_Q // 2


def mcode_int(re, im):
    """Axis-quadrant QPSK code from int I/Q. Equivalent to
    G.mcode_from_vec(re + 1j*im): |I|>=|Q| -> (I>=0 ? 0 : 2),
    else (Q>=0 ? 1 : 3). Tie |I|==|Q| -> I-axis (== np.round half-even)."""
    re = np.asarray(re, np.int64)
    im = np.asarray(im, np.int64)
    iax = np.abs(re) >= np.abs(im)
    return np.where(iax, np.where(re >= 0, 0, 2),
                    np.where(im >= 0, 1, 3)).astype(np.int64)


def liveness_int(acc, n_ang=60):
    """3-bit dead-zero freeze plane + per-ring scales (BANKED 2026-07-12,
    scratch_liveness, school8+stata x span11/oct4): the RTL freeze scan.
      M      = max(|I|,|Q|) + (min(|I|,|Q|) >> 1)   (Chebyshev |A| proxy)
      Mmax_r = max over the ring
      alive  = 2*M >= Mmax_r                        (theta = 1/2; 5/8 won
               school but LOST stata both lattices — multi-log pick)
      scale_r = (e, m): serial normalize m = Mmax_r; m >>= 1 until < 256
    liveness+scales beats the FLOAT store on extraction (school .6138 vs
    .5984; stata .6714 vs .6677). DUMP-side only: the on-chip matcher
    keeps raw 2b codes (masked local reads are measurably worse).
    Returns (alive bool (N_RING*n_ang,), [(e, m)] per ring)."""
    acc = np.asarray(acc, np.int64)
    aI = np.abs(acc[:, 0]).reshape(G.N_RING, n_ang)
    aQ = np.abs(acc[:, 1]).reshape(G.N_RING, n_ang)
    M = np.maximum(aI, aQ) + (np.minimum(aI, aQ) >> 1)
    mmax = M.max(axis=1)
    alive = (2 * M >= mmax[:, None]).reshape(-1)
    scales = []
    for v in mmax:
        e, m = 0, int(v)
        while m >= 256:
            m >>= 1
            e += 1
        scales.append((e, m))
    return alive, scales


def encode_int_at(az, r_mm, w, cq, sq, tx_u, ty_u, luts):
    """encode_int with an integer SE2 pre-transform: body points are rotated
    by (cq, sq) and translated by (tx_u, ty_u) BETWEEN the az->xy stage and
    the u-projection stage (the v7 fold-at-pose datapath). Stage arithmetic
    is copied verbatim from the accepted G.encode_int."""
    n_ang = luts["n_ang"]
    half = 1 << (G.F_AZ - 1)
    x = (r_mm * luts["az_c"][az] + half) >> G.F_AZ
    y = (r_mm * luts["az_s"][az] + half) >> G.F_AZ
    xr = (cq * x - sq * y + (1 << 14)) >> 15
    yr = (sq * x + cq * y + (1 << 14)) >> 15
    x2 = xr + tx_u
    y2 = yr + ty_u
    assert len(x2) == 0 or max(np.abs(x2).max(), np.abs(y2).max()) < (1 << 15), \
        "transformed x,y exceed i16 (RTL envelope)"
    ha = 1 << (G.F_ANG - 1)
    u = (x2[:, None] * luts["ang_c"][None, :]
         + y2[:, None] * luts["ang_s"][None, :] + ha) >> G.F_ANG
    assert u.size == 0 or np.abs(u).max() < (1 << 17), "u exceeds i18"
    acc = np.zeros((G.N_RING * n_ang, 2), np.int64)
    # derivative (cross) projection = u index-shifted by n_ang//2 with sign:
    # cross_a = u_{a - n_ang/2} for a >= n_ang/2 else -u_{a + n_ang/2}
    hshift = n_ang // 2
    cross = np.concatenate([-u[:, hshift:], u[:, :hshift]], axis=1)
    dacc = np.zeros((G.N_RING * n_ang, 2), np.int64)
    for k in range(G.N_RING):
        addr = (u >> k) & 255
        re = w[:, None] * luts["cis_re"][addr]
        im = w[:, None] * luts["cis_im"][addr]
        acc[k * n_ang:(k + 1) * n_ang, 0] = re.sum(0)
        acc[k * n_ang:(k + 1) * n_ang, 1] = im.sum(0)
        # der accumulate: d/dtheta phasor = i * (2pi/lam_k) * cross * phasor;
        # the i and the per-ring 2pi/lam scale are applied at DECODE (host).
        # fabric stores sum(w * (cross>>CD_SH) * cis) to keep i16 operands.
        cd = cross >> CD_SH
        dacc[k * n_ang:(k + 1) * n_ang, 0] = (cd * re).sum(0) >> 8
        dacc[k * n_ang:(k + 1) * n_ang, 1] = (cd * im).sum(0) >> 8
    assert np.abs(acc).max() < (1 << 31), "i32 acc overflow"
    assert np.abs(dacc).max() < (1 << 31), "i32 dacc overflow"
    return acc.astype(np.int32), dacc.astype(np.int32)


CD_SH = 2                          # cross >> 2: |cross| < 2^17 -> i15 operand


class FakeFab:
    """Fabric-free match backend: G.match_int per candidate (the function the
    RTL is bit-exact against). first-max argmax == np.argmax == silicon."""

    def __init__(self, luts):
        self.luts = luts
        self.codes = None

    def load_seg(self, codes):
        self.codes = codes

    def sweep(self, Q, s, cands):
        """Ring-summed Re totals — the deployed 0x08/argmax criterion
        (== vectors.py: gold[:, :, 0].sum(1))."""
        return np.array([int(G.match_int(Q, self.codes, dx, dy, rho, s,
                                         self.luts)[:, 0].sum())
                         for (dx, dy, rho) in cands], np.int64)


class SoloTracker:
    """Integer FabricLoc: the v7 on-chip tracker contract."""

    def __init__(self, fab, anchors_q, codes):
        # anchors_q: list of (ax_u, ay_u, ah_q); codes: list of int arrays
        self.fab = fab
        self.anc = list(anchors_q)
        self.codes = list(codes)
        self.axy = (np.array([(a[0], a[1]) for a in self.anc], np.int64)
                    if self.anc else np.zeros((0, 2), np.int64))
        self.pose = None           # (x_u, y_u, h_q) ints
        self.ema = None
        self.lost = 0
        self.n_edge = 0
        self.state = "tracking"

    def add_segment(self, anchor_q, seg_codes):
        """v7b: register a self-frozen segment (RTL: append to the SPRAM
        store + anchor table). Tracker state is NOT touched."""
        self.anc.append(anchor_q)
        self.codes.append(np.asarray(seg_codes, np.int64))
        self.axy = np.array([(a[0], a[1]) for a in self.anc], np.int64)

    def nearest(self, x_u, y_u):
        # RTL envelope contract: deltas SATURATE at +/-32767 u (~32 m) so
        # the fabric d^2 fits one 16x16 MAC. argmin is invariant as long as
        # some anchor lies within 32 m (always true while tracking); beyond
        # that all saturated anchors tie and first-min wins in both models.
        dx = np.clip(self.axy[:, 0] - x_u, -32767, 32767)
        dy = np.clip(self.axy[:, 1] - y_u, -32767, 32767)
        d2 = dx * dx + dy * dy
        return int(np.argmin(d2))

    def _rel(self, x_u, y_u, h_q, ai):
        ax, ay, ah = self.anc[ai]
        cq, sq = cs_of(ah)
        tx, ty = irot_q15(cq, sq, x_u - ax, y_u - ay)
        return tx, ty, wrap_q(h_q - ah)

    def _pose_from(self, ai, tx_u, ty_u, rho, h_pred_q):
        ax, ay, ah = self.anc[ai]
        cq, sq = cs_of(ah)
        px, py = rot_q15(cq, sq, tx_u, ty_u)
        best = None
        for b in (-TAU_Q, -TAU_Q // 2, 0, TAU_Q // 2, TAU_Q):
            h = ah + rho * RHO_Q + b
            d = abs(wrap_q(h - h_pred_q))
            if best is None or d < best[0]:
                best = (d, h)
        return (ax + px, ay + py, best[1] % TAU_Q)

    def _grid(self, Q, s, ai, d0x, d0y, rho0, T, N, R, rstep=1):
        dU = d0x + T * np.arange(-N, N + 1)
        dV = d0y + T * np.arange(-N, N + 1)
        rhos = [(rho0 + r) % 60 for r in range(-R, R + 1, rstep)]
        cands = [(int(dx), int(dy), int(rho))
                 for rho in rhos for dx in dU for dy in dV]
        tot = self.fab.sweep(Q, s, cands).reshape(len(rhos), len(dU), len(dV))
        ir, ix, iy = np.unravel_index(int(np.argmax(tot)), tot.shape)
        return dU, dV, rhos, tot, ir, ix, iy

    def _sweep_seg(self, Q, s, ai, pred_q):
        """Full tracking sweep against ONE segment (with edge-recenter
        retry). Returns (score, ai, dU, dV, rhos, tot, ir, ix, iy)."""
        self.fab.load_seg(self.codes[ai])
        tx, ty, dh = self._rel(*pred_q, ai)
        d0x, d0y = -tx, -ty                          # sgn_d = -1 (golden)
        rho0 = ((dh + RHO_Q // 2) // RHO_Q) % 60
        for attempt in range(2):
            dU, dV, rhos, tot, ir, ix, iy = self._grid(
                Q, s, ai, d0x, d0y, rho0, SWEEP_T, SWEEP_N, SWEEP_R)
            on_edge = ix in (0, len(dU) - 1) or iy in (0, len(dV) - 1)
            if not on_edge or attempt:
                break
            self.n_edge += 1
            d0x, d0y, rho0 = int(dU[ix]), int(dV[iy]), rhos[ir]
        return (int(tot[ir, ix, iy]), ai, dU, dV, rhos, tot, ir, ix, iy)

    def step(self, q_ints, pred_q):
        """One keyframe: q_ints = (az, r_mm, w); pred_q = odometry-composed
        prediction (x_u, y_u, h_q). Returns (pose_q, ai, score, state).
        v7.1 (K_TRY>1): best-of-K nearest segments with incumbent
        hysteresis — 2b codes are norm-equal so raw totals compare; the
        challenger must beat the incumbent by HYST_NUM/64."""
        Q = G.encode_int(*q_ints, self.fab.luts)
        s = G.shift_for(Q)
        if K_TRY <= 1:
            ai = self.nearest(pred_q[0], pred_q[1])
            best = self._sweep_seg(Q, s, ai, pred_q)
        else:
            d2 = ((self.axy[:, 0] - pred_q[0]) ** 2
                  + (self.axy[:, 1] - pred_q[1]) ** 2)
            order = np.argsort(d2)[:K_TRY]
            best = None
            inc = getattr(self, "last_aid", -1)
            for ai in order:
                r = self._sweep_seg(Q, s, int(ai), pred_q)
                eff = r[0] if int(ai) != inc \
                    else (r[0] * HYST_NUM) >> 6      # incumbent bonus
                if best is None or eff > best[1]:
                    best = (r, eff)
            best = best[0]
        score, ai, dU, dV, rhos, tot, ir, ix, iy = best
        self.last_aid = ai
        # integer parabolic sub-grid refine (guarded, matches live's rule:
        # px += T/2 * (a-c)/(a-2b+c) iff den<0). RTL: small serial divide
        # off the critical path; golden = trunc-toward-zero int division.
        px, py = int(dU[ix]), int(dV[iy])
        if REFINE and 0 < ix < len(dU) - 1:
            a, b, c = (int(tot[ir, ix - 1, iy]), int(tot[ir, ix, iy]),
                       int(tot[ir, ix + 1, iy]))
            den = a - 2 * b + c
            if den < 0:
                px += max(-SWEEP_T // 2,
                          min(SWEEP_T // 2,
                              int(SWEEP_T * (a - c) / (2 * den))))
        if REFINE and 0 < iy < len(dV) - 1:
            a, b, c = (int(tot[ir, ix, iy - 1]), int(tot[ir, ix, iy]),
                       int(tot[ir, ix, iy + 1]))
            den = a - 2 * b + c
            if den < 0:
                py += max(-SWEEP_T // 2,
                          min(SWEEP_T // 2,
                              int(SWEEP_T * (a - c) / (2 * den))))
        pose = self._pose_from(ai, -px, -py, rhos[ir], pred_q[2])
        if self.ema is None:
            self.ema = score
        if score > COMMIT_NUM * (self.ema >> EMA_SH):
            self.ema += (score - self.ema) >> EMA_SH
            self.pose = pose
            self.lost = 0
            self.state = "tracking"
            return pose, ai, score, self.state
        self.lost += 1
        self.state = "hold"
        self.pose = tuple(pred_q)
        if self.lost % LOST_PERIOD == 0:
            rp = self._research(Q, s, pred_q)
            if rp is not None and rp[3] > RELOCK_NUM * (self.ema >> EMA_SH):
                self.pose = rp[:3]
                self.lost = 0
                self.state = "relock"
                self.last_aid = rp[4]
                return rp[:3], rp[4], rp[3], self.state
        return tuple(pred_q), ai, score, self.state

    def _research(self, Q, s, pred_q):
        ai = self.nearest(pred_q[0], pred_q[1])
        self.fab.load_seg(self.codes[ai])
        tx, ty, dh = self._rel(*pred_q, ai)
        rho0 = ((dh + RHO_Q // 2) // RHO_Q) % 60
        dU, dV, rhos, tot, ir, ix, iy = self._grid(
            Q, s, ai, -tx, -ty, rho0, RS_T, RS_N, RS_R, rstep=2)
        pose = self._pose_from(ai, -int(dU[ix]), -int(dV[iy]),
                               rhos[ir], pred_q[2])
        return (*pose, int(tot[ir, ix, iy]), ai)


class SoloMapper:
    """v7b fold/freeze: open-segment accumulators in the segment-anchor
    frame; freeze every SEG_KF keyframes to axis-quadrant codes."""

    def __init__(self, luts):
        self.luts = luts
        self.open_anchor = None    # (x_u, y_u, h_q)
        self.kf_in_seg = 0
        self.acc = None
        self.dacc = None
        self.frozen = []           # list of (anchor_q, codes, dcodes_raw)

    def fold(self, q_ints, pose_q):
        if self.open_anchor is None:
            self.open_anchor = pose_q
            n = G.N_RING * self.luts["n_ang"]
            self.acc = np.zeros((n, 2), np.int64)
            self.dacc = np.zeros((n, 2), np.int64)
        ax, ay, ah = self.open_anchor
        dh = wrap_q(pose_q[2] - ah)
        cq, sq = cs_of(dh)
        acs, asn = cs_of(ah)
        tx, ty = irot_q15(acs, asn, pose_q[0] - ax, pose_q[1] - ay)
        a, d = encode_int_at(*q_ints, cq, sq, tx, ty, self.luts)
        self.acc += a
        self.dacc += d
        self.kf_in_seg += 1
        if self.kf_in_seg >= SEG_KF:
            self.freeze()

    def freeze(self):
        if self.open_anchor is None or self.kf_in_seg == 0:
            return
        assert np.abs(self.acc).max() < (1 << 31), \
            "seg-acc exceeds i32 (RTL envelope)"
        codes = mcode_int(self.acc[:, 0], self.acc[:, 1])
        dcodes = mcode_int(self.dacc[:, 0], self.dacc[:, 1])
        alive, scales = liveness_int(self.acc, self.luts["n_ang"])
        self.frozen.append((self.open_anchor, codes, dcodes, alive, scales))
        self.open_anchor = None
        self.kf_in_seg = 0


class ShimBoard:
    """vectors.Board stand-in: golden match_int totals + first-max argmax."""

    def __init__(self, shim):
        self.shim = shim

    def _tot(self, cands):
        sh = self.shim
        return np.array([int(G.match_int(sh.Q, sh.codes, dx, dy, rho, s,
                                         sh.luts)[:, 0].sum())
                         for (dx, dy, rho, s) in cands], np.int64)

    def match_argmax(self, cands):
        t = self._tot(cands)
        i = int(np.argmax(t))
        return i, int(t[i])


class ShimFab:
    """live.Fabric stand-in for offline parity runs (no serial)."""

    def __init__(self, luts):
        self.luts = luts
        self.b = ShimBoard(self)
        self.Q = None
        self.codes = None
        self.cur_aid = None

    def set_scan(self, q_ints):
        self.Q = G.encode_int(*q_ints, self.luts)

    def load_seg(self, aid, codes):
        self.codes = np.asarray(codes, np.int64)
        self.cur_aid = aid

    def bsweep(self, cands, ints=None, codes=None):
        return self.b._tot(cands)

    def recover(self, q_ints):
        self.set_scan(q_ints)


# ------------------------------------------------------------------- bench
def bench(env="classroom", traj="orbit", passes=3, seed=11):
    """End-to-end golden bench: pass-1 python SLAM builds the frozen map
    (exactly live.py's pass 1); passes 2+ run the INTEGER SoloTracker over
    the looped feed with integer odometry composition — the v7 standalone
    contract. GT is scoring-only. Reports pos-err quantiles + state counts
    vs the banked live numbers (fx err ~0.032 m on classroom/orbit)."""
    import live as LV
    import ssp_fpga as F
    import ssp_dynenv as DE
    import ssp_slam as S
    import ssp_slam_loop as L

    feed = LV.Feed(seed=seed, laps=2, env=env, traj=traj)
    luts = G.make_luts()
    slam = F.BandSLAM(robust=True, attempt_every=4, relax_every=25,
                      gap_kf=300, recent_aids=12, spec=None, nph=0)
    slam.store_dtype = np.complex64
    it = iter(feed)
    items = [next(it) for _ in range(feed.n_loop)]
    est_prev = odom_prev = None
    for item in items:                        # pass 1: python SLAM mapping
        rr = np.where((item["r"] > 0.05) & (item["r"] < S.MAX_RANGE - 0.05),
                      item["r"], np.inf)
        pts, w = DE._bridge2(rr, feed.beam)
        if est_prev is None:
            guess = item["odom"].copy()
        else:
            guess = L.se2_mul(est_prev, L.se2_mul(
                L.se2_inv(odom_prev), item["odom"]))
        est = slam.add_keyframe(pts, w, guess)
        est_prev, odom_prev = est.copy(), item["odom"].copy()
    fmap = LV.FrozenMap(slam)
    anchors_q = [(int(round(fmap.anchor[a][0] / U)),
                  int(round(fmap.anchor[a][1] / U)),
                  int(round(fmap.anchor[a][2] * TAU_Q / (2 * np.pi))) % TAU_Q)
                 for a in fmap.aids]
    codes = [np.asarray(fmap.codes[a], np.int64) for a in fmap.aids]
    trk = SoloTracker(FakeFab(luts), anchors_q, codes)
    shim = ShimFab(luts)
    pyloc = LV.FabricLoc(shim, fmap)
    print(f"[bench {env}/{traj}] map: {len(codes)} segments; pass-1 python "
          f"SLAM done; solo + live-FabricLoc parity, passes "
          f"2..{passes + 1}", flush=True)
    errs, perrs, dxy = [], [], []
    states = {"tracking": 0, "hold": 0, "relock": 0}
    pose_q = None
    py_pose = None
    odom_prev = None
    for _ in range(passes * feed.n_loop):
        item = next(it)
        q = G.scan_to_ints(item["r"])
        if odom_prev is None:
            d_q = (0, 0, 0)
            d = np.zeros(3)
            pose_q = (int(round(item["odom"][0] / U)),
                      int(round(item["odom"][1] / U)),
                      int(round(item["odom"][2] * TAU_Q / (2 * np.pi)))
                      % TAU_Q)
            py_pose = item["odom"].copy()
        else:
            d = L.se2_mul(L.se2_inv(odom_prev), item["odom"])
            d_q = (int(round(d[0] / U)), int(round(d[1] / U)),
                   int(round(d[2] * TAU_Q / (2 * np.pi))))
        odom_prev = item["odom"].copy()
        cq, sq = cs_of(pose_q[2])
        dxw, dyw = rot_q15(cq, sq, d_q[0], d_q[1])
        pred_q = (pose_q[0] + dxw, pose_q[1] + dyw,
                  (pose_q[2] + d_q[2]) % TAU_Q)
        py_pred = L.se2_mul(py_pose, d)
        if len(q[0]) < 5:
            pose_q = pred_q
            py_pose = py_pred
            continue
        pose_q, ai, score, st = trk.step(q, pred_q)
        states[st] += 1
        shim.set_scan(q)
        py_pose, _, _ = pyloc.locate(q, py_pred)
        gt = item["gt"]                                   # SCORING ONLY
        errs.append(np.hypot(pose_q[0] * U - gt[0], pose_q[1] * U - gt[1]))
        perrs.append(np.hypot(py_pose[0] - gt[0], py_pose[1] - gt[1]))
        dxy.append(np.hypot(pose_q[0] * U - py_pose[0],
                            pose_q[1] * U - py_pose[1]))
    e, pe, dd = np.array(errs), np.array(perrs), np.array(dxy)
    print(f"  solo (int) pos err: med {np.median(e):.3f}  p90 "
          f"{np.percentile(e, 90):.3f}  max {e.max():.3f}  | states "
          f"{states} | edge-recenters {trk.n_edge}", flush=True)
    print(f"  live FabricLoc    : med {np.median(pe):.3f}  p90 "
          f"{np.percentile(pe, 90):.3f}  max {pe.max():.3f}  | "
          f"edge-recenters {pyloc.n_edge}", flush=True)
    print(f"  solo-vs-live trace: med {np.median(dd):.3f}  p90 "
          f"{np.percentile(dd, 90):.3f}  max {dd.max():.3f}", flush=True)
    return e, pe, dd


# ---------------------------------------------------------- bench: self-map
def bench_selfmap(env="classroom", traj="orbit", passes=3, seed=11):
    """v7b acceptance question: can the chip BUILD the map itself? No
    python SLAM anywhere: pass 1 folds scans at the tracker's own
    committed poses (odometry-only until the first segment freezes),
    freezing 5-kf segments into the tracker's map as it goes; passes 2+
    track against the self-built map (map growth stops after pass 1,
    mirroring the live freeze). GT = scoring only."""
    import live as LV
    import ssp_slam_loop as L

    feed = LV.Feed(seed=seed, laps=2, env=env, traj=traj)
    luts = G.make_luts()
    trk = SoloTracker(FakeFab(luts), [], [])
    mapper = SoloMapper(luts)
    it = iter(feed)
    errs1, errs2, odo_errs = [], [], []
    states = {"tracking": 0, "hold": 0, "relock": 0, "odo": 0}
    pose_q = None
    odom_prev = None
    n_frozen = 0
    for kk in range((passes + 1) * feed.n_loop):
        item = next(it)
        mapping = kk < feed.n_loop
        q = G.scan_to_ints(item["r"])
        if odom_prev is None:
            d_q = (0, 0, 0)
            pose_q = (int(round(item["odom"][0] / U)),
                      int(round(item["odom"][1] / U)),
                      int(round(item["odom"][2] * TAU_Q / (2 * np.pi)))
                      % TAU_Q)
        else:
            d = L.se2_mul(L.se2_inv(odom_prev), item["odom"])
            d_q = (int(round(d[0] / U)), int(round(d[1] / U)),
                   int(round(d[2] * TAU_Q / (2 * np.pi))))
        odom_prev = item["odom"].copy()
        cq, sq = cs_of(pose_q[2])
        dxw, dyw = rot_q15(cq, sq, d_q[0], d_q[1])
        pred_q = (pose_q[0] + dxw, pose_q[1] + dyw,
                  (pose_q[2] + d_q[2]) % TAU_Q)
        if len(q[0]) < 5:
            pose_q = pred_q
            continue
        st = "odo"
        if len(trk.codes) == 0:
            pose_q = pred_q                       # boot: pure odometry
            states["odo"] += 1
        else:
            pose_q, ai, score, st = trk.step(q, pred_q)
            states[st] += 1
        if mapping and (st in ("tracking", "relock", "odo")
                        or not GATE_FOLD):
            n_before = len(mapper.frozen)
            mapper.fold(q, pose_q)
            if len(mapper.frozen) > n_before:     # a segment froze
                anc, codes = mapper.frozen[-1][0], mapper.frozen[-1][1]
                trk.add_segment(anc, codes)
                n_frozen += 1
        gt = item["gt"]                            # SCORING ONLY
        e = np.hypot(pose_q[0] * U - gt[0], pose_q[1] * U - gt[1])
        (errs1 if mapping else errs2).append(e)
        odo_errs.append(np.hypot(item["odom"][0] - gt[0],
                                 item["odom"][1] - gt[1]))
    e1, e2 = np.array(errs1), np.array(errs2)
    eo = np.array(odo_errs)
    print(f"[selfmap {env}/{traj}] {n_frozen} self-frozen segments; "
          f"states {states} | gate_fold {GATE_FOLD}", flush=True)
    print(f"  pass-1 (mapping) err: med {np.median(e1):.3f}  p90 "
          f"{np.percentile(e1, 90):.3f}  max {e1.max():.3f}", flush=True)
    print(f"  replay (self-map) err: med {np.median(e2):.3f}  p90 "
          f"{np.percentile(e2, 90):.3f}  max {e2.max():.3f}", flush=True)
    print(f"  raw odometry chain  : med {np.median(eo):.3f}  p90 "
          f"{np.percentile(eo, 90):.3f}  max {eo.max():.3f}", flush=True)
    return e1, e2


# ------------------------------------------------- bench: sample reservoir
RES_N = 16          # slots (one spare SPRAM at ~2 KB/raw scan)
RES_P = 0.25        # per-kf overwrite chance; decay tau ~ N/p kf (user
                    # lever: linear in mem/chance)
RES_Q = 4           # attempt one re-match every RES_Q kf (idle matcher duty)
BLACKOUT = None     # DIAGNOSTIC: (period, len) forced blind windows in
                    # replay — the honest pathology (transient loss on a
                    # mature map) the reservoir exists to repair


def bench_reservoir(env="classroom", traj="orbit", passes=3, seed=11):
    """User proposal: spare-SPRAM reservoir of raw past samples (random
    overwrite -> exponentially decaying age mix); opportunistically
    re-match a random slot against the matured resident map and re-fold
    at the matched pose as a NEW correction segment (fidelity stream;
    quantized once — no negate, no requant-per-touch; matcher space
    untouched). Golden bench scores REGISTRATION of folds: capture err
    vs refold err (GT scoring only), split by capture state. Deployable
    iff refold errs improve on their captures (esp. hold-captured) and
    acceptance is healthy."""
    import live as LV
    import ssp_slam_loop as L

    feed = LV.Feed(seed=seed, laps=2, env=env, traj=traj)
    luts = G.make_luts()
    trk = SoloTracker(FakeFab(luts), [], [])
    mapper = SoloMapper(luts)
    rng = np.random.default_rng(seed * 31 + 7)
    slots = [None] * RES_N          # (q_ints, seed_pose_q, gt, state, kf)
    it = iter(feed)
    states = {"tracking": 0, "hold": 0, "relock": 0, "odo": 0}
    pose_q = None
    odom_prev = None
    rec = []                        # (cap_err, refold_err, cap_state, age)
    rej = 0
    for kk in range((passes + 1) * feed.n_loop):
        item = next(it)
        mapping = kk < feed.n_loop
        q = G.scan_to_ints(item["r"])
        if odom_prev is None:
            d_q = (0, 0, 0)
            pose_q = (int(round(item["odom"][0] / U)),
                      int(round(item["odom"][1] / U)),
                      int(round(item["odom"][2] * TAU_Q / (2 * np.pi)))
                      % TAU_Q)
        else:
            d = L.se2_mul(L.se2_inv(odom_prev), item["odom"])
            d_q = (int(round(d[0] / U)), int(round(d[1] / U)),
                   int(round(d[2] * TAU_Q / (2 * np.pi))))
        odom_prev = item["odom"].copy()
        cq, sq = cs_of(pose_q[2])
        dxw, dyw = rot_q15(cq, sq, d_q[0], d_q[1])
        pred_q = (pose_q[0] + dxw, pose_q[1] + dyw,
                  (pose_q[2] + d_q[2]) % TAU_Q)
        if len(q[0]) < 5:
            pose_q = pred_q
            continue
        st = "odo"
        blackout = (BLACKOUT and not mapping
                    and (kk % BLACKOUT[0]) < BLACKOUT[1])
        if len(trk.codes) == 0 or blackout:
            pose_q = pred_q                       # blind: odometry only
            states["odo"] += 1
            if blackout:
                st = "hold"                       # state-tag (GT-free)
        else:
            pose_q, ai, score, st = trk.step(q, pred_q)
            states[st] += 1
        if mapping:
            n_before = len(mapper.frozen)
            mapper.fold(q, pose_q)
            if len(mapper.frozen) > n_before:
                anc, codes = mapper.frozen[-1][0], mapper.frozen[-1][1]
                trk.add_segment(anc, codes)
        # reservoir write: random chance, random slot (exponential ages)
        if rng.random() < RES_P:
            si = int(rng.integers(RES_N))
            slots[si] = (q, pose_q, item["gt"].copy(), st, kk)
        # reservoir re-match: one random occupied slot every RES_Q kf
        # (not during blackout — if tracking is blind, so is the fabric)
        if kk % RES_Q == 0 and len(trk.codes) > 0 and not blackout:
            occ = [s for s in slots if s is not None]
            if occ:
                sq_, spose, sgt, sst, skf = occ[int(rng.integers(len(occ)))]
                Q2 = G.encode_int(*sq_, luts)
                s2 = G.shift_for(Q2)
                ai2 = trk.nearest(spose[0], spose[1])
                trk.fab.load_seg(trk.codes[ai2])
                tx, ty, dh = trk._rel(*spose, ai2)
                rho0 = ((dh + RHO_Q // 2) // RHO_Q) % 60
                # do-no-harm reference: score of the CAPTURE pose itself
                sc_cap = int(trk.fab.sweep(Q2, s2,
                                           [(-tx, -ty, rho0)])[0])
                # two-stage re-match: coarse re-search, then the fine
                # tracking grid + integer parabolic (same as step())
                dU, dV, rhos, tot, ir, ix, iy = trk._grid(
                    Q2, s2, ai2, -tx, -ty, rho0, RS_T, RS_N, RS_R, rstep=2)
                dU, dV, rhos, tot, ir, ix, iy = trk._grid(
                    Q2, s2, ai2, int(dU[ix]), int(dV[iy]), rhos[ir],
                    SWEEP_T, SWEEP_N, SWEEP_R)
                sc = int(tot[ir, ix, iy])
                px, py = int(dU[ix]), int(dV[iy])
                if 0 < ix < len(dU) - 1:
                    a, b, c = (int(tot[ir, ix - 1, iy]),
                               int(tot[ir, ix, iy]),
                               int(tot[ir, ix + 1, iy]))
                    den = a - 2 * b + c
                    if den < 0:
                        px += max(-SWEEP_T // 2,
                                  min(SWEEP_T // 2,
                                      int(SWEEP_T * (a - c) / (2 * den))))
                if 0 < iy < len(dV) - 1:
                    a, b, c = (int(tot[ir, ix, iy - 1]),
                               int(tot[ir, ix, iy]),
                               int(tot[ir, ix, iy + 1]))
                    den = a - 2 * b + c
                    if den < 0:
                        py += max(-SWEEP_T // 2,
                                  min(SWEEP_T // 2,
                                      int(SWEEP_T * (a - c) / (2 * den))))
                gate_ok = (trk.ema is not None
                           and sc > COMMIT_NUM * (trk.ema >> EMA_SH))
                if gate_ok and sc > sc_cap:       # strict improvement only
                    rp = trk._pose_from(ai2, -px, -py, rhos[ir], spose[2])
                    cap_e = np.hypot(spose[0] * U - sgt[0],
                                     spose[1] * U - sgt[1])
                    ref_e = np.hypot(rp[0] * U - sgt[0],
                                     rp[1] * U - sgt[1])
                    rec.append((cap_e, ref_e, sst, kk - skf))
                    # deploy: encode_int_at fold at rp -> correction segment
                else:
                    rej += 1
    r = np.array([(a, b, d) for a, b, _, d in rec])
    print(f"[reservoir {env}/{traj}] {len(rec)} re-folds accepted, {rej} "
          f"rejected | states {states}", flush=True)
    if len(rec):
        print(f"  capture err -> refold err: med {np.median(r[:, 0]):.3f}"
              f" -> {np.median(r[:, 1]):.3f}  p90 "
              f"{np.percentile(r[:, 0], 90):.3f} -> "
              f"{np.percentile(r[:, 1], 90):.3f} | improved "
              f"{(r[:, 1] < r[:, 0]).mean():.2f} | age med "
              f"{np.median(r[:, 2]):.0f} p90 {np.percentile(r[:, 2], 90):.0f} kf",
              flush=True)
        hold = np.array([(a, b) for a, b, s, _ in rec if s != "tracking"])
        if len(hold):
            print(f"  hold/odo-captured ({len(hold)}): med "
                  f"{np.median(hold[:, 0]):.3f} -> {np.median(hold[:, 1]):.3f}"
                  f"  p90 {np.percentile(hold[:, 0], 90):.3f} -> "
                  f"{np.percentile(hold[:, 1], 90):.3f}", flush=True)
    return rec


# ------------------------------------------------------------------ selftest
def selftest():
    luts = G.make_luts()
    rng = np.random.default_rng(7)
    # 1. mcode_int == G.mcode_from_vec on random ints incl. exact ties
    re = rng.integers(-1000, 1000, 4096)
    im = rng.integers(-1000, 1000, 4096)
    re[:64] = im[:64]                                   # force |I|==|Q| ties
    ref = G.mcode_from_vec(re + 1j * im)
    assert np.array_equal(mcode_int(re, im), ref), "mcode tie/quadrant"
    # 2. encode_int_at with identity transform == encode_int
    az = rng.integers(0, 1024, 1500)
    r_mm = rng.integers(300, 7000, 1500)
    w = rng.integers(1, 127, 1500)
    a0 = G.encode_int(az, r_mm, w, luts)
    a1, d1 = encode_int_at(az, r_mm, w, 32767, 0, 0, 0, luts)
    assert np.array_equal(a0, a1), "identity transform mismatch"
    # 3. transform consistency: encode at t == matcher peak at -t
    Q = G.encode_int(az, r_mm, w, luts)
    s = G.shift_for(Q)
    a2, _ = encode_int_at(az, r_mm, w, 32767, 0, 40, -24, luts)
    codes = mcode_int(a2[:, 0], a2[:, 1])
    # golden convention (match_int docstring): map content displaced +d
    # relative to the query peaks at D = -d. Fold encoded at t=(40,-24)
    # -> peak at (-40, 24). Coarse pass step 8, then step-2 refine.
    best, bat = None, None
    for dx in range(-64, 65, 8):
        for dy in range(-64, 65, 8):
            v = int(G.match_int(Q, codes, dx, dy, 0, s, luts)[:, 0].sum())
            if best is None or v > best:
                best, bat = v, (dx, dy)
    for dx in range(bat[0] - 8, bat[0] + 9, 2):
        for dy in range(bat[1] - 8, bat[1] + 9, 2):
            v = int(G.match_int(Q, codes, dx, dy, 0, s, luts)[:, 0].sum())
            if v > best:
                best, bat = v, (dx, dy)
    # tol: 2b-code peak wander on finite content; fold-path acceptance is
    # ~1 cm registration (8 U = 7.8 mm)
    assert abs(bat[0] + 40) <= 8 and abs(bat[1] - 24) <= 8, \
        f"fold/match convention: peak at {bat}, want ~(-40, 24)"
    # 4. Q15 rotate round-trip
    cq, sq = cs_of(123)
    x, y = rot_q15(cq, sq, 1200, -800)
    xb, yb = irot_q15(cq, sq, x, y)
    assert abs(xb - 1200) <= 2 and abs(yb + 800) <= 2, "Q15 round-trip"
    print("[solo selftest] mcode ok; identity-encode ok; "
          f"fold/match peak at {bat} == encode offset; Q15 ok", flush=True)


if __name__ == "__main__":
    selftest()
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        bench(env=sys.argv[2] if len(sys.argv) > 2 else "classroom",
              traj=sys.argv[3] if len(sys.argv) > 3 else "orbit")
    if len(sys.argv) > 1 and sys.argv[1] == "selfmap":
        bench_selfmap(env=sys.argv[2] if len(sys.argv) > 2 else "classroom",
                      traj=sys.argv[3] if len(sys.argv) > 3 else "orbit")
    if len(sys.argv) > 1 and sys.argv[1] == "reservoir":
        bench_reservoir(env=sys.argv[2] if len(sys.argv) > 2 else "classroom",
                        traj=sys.argv[3] if len(sys.argv) > 3 else "orbit")
