#!/usr/bin/env python3
"""solo2 — GOLDEN INTEGER MODEL of the v8 SOLO stack (ECP5 track).

v8 = v7-lean (REFINE=0, K_TRY=1) + ONE change: NOVELTY-GATED FOLDING.
A new segment opens only when the tracker's own nearest-anchor scan
flags the prediction as > NOV_M from every anchor — the resident map's
64-segment budget buys AREA, not time. Tracking semantics are
BIT-IDENTICAL to v7 (solo_tracker2.v re-passes the untouched l1/lr
lean fixtures verbatim).

Decomposition bench (2026-07-16, hunter capture 2810 kf @ 10 Hz, odom
arm; score = MAP CRISPNESS at the tracker's own poses, pts per
occupied 0.1 m cell — GT-free display-truth):

    raw odom display        7.27
    v7 stock                9.29   (64 segs eaten in the first 32 s)
    novelty ONLY   (v8)    10.73   <- shipped
    + rho window/retry     10.12   REFUTED (EMA-gate inflation — the
                                   same failure shape as K_TRY=3)
    + staged local RS      10.12   never fired usefully
    full E bundle          10.45   global RS partially repaid what the
                                   window lost; still < novelty alone
    heading clamps       7.7-8.2   REFUTED (matcher heading corrections
                                   IMPROVE the map; hunter odom yaw is
                                   long-term good — the scan-xcorr yaw
                                   integral is NOT an absolute ref)
    abs floor 25% self      7.66   REFUTED (kills honest commits)

Spot no-regression gate: crispness 15.31 -> 15.38, self-map ATE p90
4.48 -> 2.37 (novelty spends only 21/64 segs on the small venue).
NSEG=128 measured better still (10.88) — banked as the next rung (the
segment-index width surgery through top_solo is post-demo work).

RTL: hw/ecp5/rtl/solo_tracker2.v (novel output) + top_solo2.v (fold
gate) + top_solo_ecp5_v2.v; gates: solo2_gates.py.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hw" / "ice40" / "host"))

import hw.ice40.golden as G                                    # noqa: E402
import solo                                                    # noqa: E402

U, TAU_Q = solo.U, solo.TAU_Q
NSEG = 64
NOV_M = 0.8                                   # novelty radius [m]
NOV_D2 = int(round(NOV_M / U)) ** 2           # in u^2 (sat16 semantics)


class Solo2Tracker(solo.SoloTracker):
    """v7 tracker + the RTL's `novel` bit (nearest-anchor d2 > NOV_D2
    at the prediction, saturated like the fabric). Nothing else."""

    def __init__(self, fab, anchors_q, codes):
        super().__init__(fab, anchors_q, codes)
        self.novel = True

    def nearest(self, x_u, y_u):
        dx = np.clip(self.axy[:, 0] - x_u, -32767, 32767)
        dy = np.clip(self.axy[:, 1] - y_u, -32767, 32767)
        d2 = dx * dx + dy * dy
        i = int(np.argmin(d2))
        self.novel = bool(int(d2[i]) > NOV_D2)
        return i


def fold_gate(trk, mapper, n_seg):
    """v8 fold policy (top_solo2 K_FOLDQ): always finish an OPEN
    segment; open a new one only at a novel prediction (or boot)."""
    if mapper.open_anchor is not None:        # seg_open
        return True
    if n_seg == 0 or n_seg >= NSEG:
        return n_seg == 0
    return trk.novel


def selftest():
    """(1) novel-bit semantics incl. sat16; (2) tracking parity: the
    v8 tracker's step() must be BIT-IDENTICAL to stock v7 on a live
    scene (the RTL analog is the untouched l1/lr fixture pass)."""
    rng = np.random.default_rng(7)
    luts = G.make_luts()
    solo.REFINE = False                       # lean build semantics

    def cast(pose, n_beam=1024):
        wallseg = [((-3.0, -2.1), (3.4, -2.1)), ((3.4, -2.1), (3.4, 2.6)),
                   ((3.4, 2.6), (-3.0, 2.6)), ((-3.0, 2.6), (-3.0, -2.1)),
                   ((-1.6, 0.9), (-1.1, 1.4))]
        a = pose[2] + -np.pi + np.arange(n_beam) * (2 * np.pi / n_beam)
        r = np.full(n_beam, np.inf)
        ca, sa = np.cos(a), np.sin(a)
        for (x0, y0), (x1, y1) in wallseg:
            dx, dy = x1 - x0, y1 - y0
            den = dx * sa - dy * ca
            ok = np.abs(den) > 1e-9
            t = np.where(ok, ((pose[0] - x0) * sa - (pose[1] - y0) * ca)
                         / np.where(ok, den, 1.0), -1.0)
            rr = np.where((t >= 0) & (t <= 1),
                          (x0 + t * dx - pose[0]) * ca
                          + (y0 + t * dy - pose[1]) * sa, np.inf)
            r = np.minimum(r, np.where(rr > 0.05, rr, np.inf))
        return r

    q0 = G.scan_to_ints(cast((0.2, -0.3, 0.1)))
    acc = G.encode_int(*q0, luts)
    codes = solo.mcode_int(acc[:, 0], acc[:, 1])
    anchors = [(205, -307, 27)]
    t2 = Solo2Tracker(solo.FakeFab(luts), anchors, [codes])
    t1 = solo.SoloTracker(solo.FakeFab(luts), anchors, [codes])
    # novel bit
    t2.nearest(205, -307)
    assert not t2.novel
    t2.nearest(205 + int(1.0 / U), -307)
    assert t2.novel
    t2.nearest(120000, 120000)               # sat16 saturation -> novel
    assert t2.novel
    # step parity over a jittered trajectory (commits + holds)
    mism = 0
    for i in range(12):
        pose = (0.2 + 0.05 * i, -0.3 + 0.03 * i, 0.1 + 0.05 * i)
        q = G.scan_to_ints(cast(pose)
                           + rng.normal(0, 0.01, 1024))
        pred = (int(round(pose[0] / U)) + int(rng.integers(-20, 20)),
                int(round(pose[1] / U)) + int(rng.integers(-20, 20)),
                (int(round(pose[2] * TAU_Q / (2 * np.pi)))
                 + int(rng.integers(-4, 4))) % TAU_Q)
        r1 = t1.step(q, pred)
        r2 = t2.step(q, pred)
        mism += (r1[0] != r2[0]) + (r1[1] != r2[1]) \
            + (r1[2] != r2[2]) + (r1[3] != r2[3])
    assert mism == 0, f"v8 tracking diverged from v7 ({mism})"
    print("[solo2 selftest] novel bit ok (incl. sat16); 12-step "
          "tracking parity v8 == v7 bit-exact — PASS")
    solo.REFINE = True


if __name__ == "__main__":
    selftest()
