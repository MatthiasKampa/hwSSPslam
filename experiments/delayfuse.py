"""TEMPORAL fusion under measurement DELAY (user 2026-07-15: "one
important aspect of the system will be the delay - handle two spaces
and two position estimates, combine temporally?").

Deploy timing: the vision chain is the fast tier (120 Hz on-fabric;
the 15 Hz stride-2 TUM parse is the proxy), the cloud keyframe
registration the slow tier (5 Hz = every KF=3 fast steps), and the
slow result arrives LATE by d fast steps (processing latency; d=0..3
at 15 Hz = 0..200 ms spans the expected 50-100 ms cloud latency).

Bench: horizons of H=12 fast steps (0.8 s). Vision-only per-step GN
increments are chained; cloud-only GN measurements over each 3-step
keyframe interval arrive d steps late. Policies (identical fusion
math — omega-average with the banked per-channel precisions; ONLY the
window bookkeeping differs):

  fast-only  no fusion — the vision-chain drift baseline
  naive      timestamp-ignorant: on arrival the slow estimate is fused
             against the MOST RECENT 3-step fast segment, as if it
             described "now" (latest-sample mixing)
  matched    interval-matched buffered fusion: a ring buffer of
             per-step increments; the slow estimate is fused against
             exactly its own interval's buffered segment, then the
             newer buffered steps are re-applied on top (delayed-state
             re-anchoring)

At d=0 naive == matched by construction (sanity gate). Both policies
receive the identical measurement set per d (a measurement arriving
after the horizon end is dropped for both) — the timestamp handling is
the only degree of freedom measured. GT (mocap) gates window motion
and scores the horizon-end relative pose — DIAGNOSTIC per PROTOCOL;
nothing from GT enters an estimate or a policy decision.

Usage: python3 -m experiments.delayfuse [bench|selftest] [seq]
"""
import sys
from pathlib import Path

import numpy as np

import experiments.lattice3d as L3
import experiments.vision6d as V6
import experiments.lidar6d as L6
import experiments.deploy6d as D6

H = 12                      # horizon in fast steps (0.8 s @ 15 Hz)
KF = 3                      # fast steps per keyframe interval (5 Hz)
DELAYS = (0, 1, 2, 3)       # slow-arrival latency in fast steps
SIG_V, SIG_C = 1.82, 1.34   # banked per-estimate rot sigmas (deg)


def load_s2(seq=D6.SEQ):
    f = (Path(__file__).resolve().parents[1] / "data" / "tum"
         / f"{seq}_s2.npz")
    z = np.load(f)
    return z["gray"], z["depth_mm"], z["gt"], z["K"]


# --------------------------------------------------------------------------
#  SE(3) increment algebra (newest factor on the LEFT, deploy6d.chain
#  convention)
# --------------------------------------------------------------------------
def compose(steps):
    R, t = np.eye(3), np.zeros(3)
    for Rs, ts in steps:
        R, t = Rs @ R, Rs @ t + ts
    return R, t


def inv(T):
    R, t = T
    return R.T, -R.T @ t


def chain(A, B):
    """A after B (A is the newer increment)."""
    return A[0] @ B[0], A[0] @ B[1] + A[1]


def fuse(F, S, pf, ps):
    """Omega-average two relative-pose estimates of the SAME interval
    (the banked post-hoc recipe: rotation-vector + translation blend
    with fixed precisions)."""
    w = (pf * D6.rotvec(F[0]) + ps * D6.rotvec(S[0])) / (pf + ps)
    t = (pf * F[1] + ps * S[1]) / (pf + ps)
    return D6._R_of_w(w), t


# --------------------------------------------------------------------------
#  policies: identical inputs (per-step fast increments S, keyframe
#  measurements M with arrival times), different bookkeeping
# --------------------------------------------------------------------------
def _pf_ps():
    pf = 1.0 / (KF * SIG_V ** 2)      # KF chained fast steps
    ps = 1.0 / SIG_C ** 2
    return pf, ps


def run_matched(S, M, d):
    """Segments aligned to keyframe intervals; a measurement replaces
    (by fusion) exactly its own interval's buffered fast segment iff it
    arrives by the horizon end; newer buffered steps re-apply on top by
    ordinary composition."""
    pf, ps = _pf_ps()
    segs = [compose(S[m * KF:(m + 1) * KF]) for m in range(len(M))]
    for m in range(len(M)):
        if (m + 1) * KF + d <= H:                 # arrived in time
            segs[m] = fuse(segs[m], M[m], pf, ps)
    out = segs[0]
    for sg in segs[1:]:
        out = chain(sg, out)
    return out


def run_naive(S, M, d):
    """Single running state; on arrival the newest KF-step segment is
    swapped for its fusion with the (stale) measurement."""
    pf, ps = _pf_ps()
    arrive = {(m + 1) * KF + d: m for m in range(len(M))}
    T = np.eye(3), np.zeros(3)
    for s in range(H):
        T = chain(S[s], T)
        a = s + 1
        if a in arrive and a >= KF:
            seg = compose(S[a - KF:a])            # what "now" looks like
            F = fuse(seg, M[arrive[a]], pf, ps)
            T = chain(F, chain(inv(seg), T))
    return T


# --------------------------------------------------------------------------
#  bench
# --------------------------------------------------------------------------
def bench(seq=D6.SEQ, max_h=40):
    gray, depth, gt, K = load_s2(seq)
    Wl = L3.make_lattices()["azel3d"]
    Wv = D6.W_vis3d()
    n = len(gray)

    def gated(k0):
        for m in range(H // KF):
            i, j = k0 + m * KF, k0 + (m + 1) * KF
            Rt, tt = V6._rel_pose(gt[i], gt[j])
            ang = np.degrees(np.arccos(np.clip((np.trace(Rt) - 1) / 2,
                                               -1, 1)))
            if not (0.5 <= ang <= 9.0) or np.linalg.norm(tt) > 0.18:
                return False
        return True

    res = {("fast", 0): ([], [])}
    for d in DELAYS:
        res[("naive", d)] = ([], [])
        res[("matched", d)] = ([], [])
    nh = 0
    for k0 in range(0, n - H - 1, 6):
        if nh >= max_h:
            break
        if not gated(k0):
            continue
        F = {i: V6.feats(gray[i], K, "gridint", depth[i])
             for i in range(k0, k0 + H + 1)}
        S = []
        for s in range(H):
            i, j = k0 + s, k0 + s + 1
            v1 = V6._enc_raw(Wv, *F[j])
            S.append(L6._gn_iterate(Wv, F[i][0], F[i][1], v1,
                                    np.eye(3), np.zeros(3), iters=2))
        M = []
        for m in range(H // KF):
            i, j = k0 + m * KF, k0 + (m + 1) * KF
            P0, w0 = L6.depth_cloud(depth[i], K)
            P1, w1 = L6.depth_cloud(depth[j], K)
            v1 = V6._enc_raw(Wl, P1, w1)
            M.append(L6._gn_iterate(Wl, P0, w0, v1,
                                    np.eye(3), np.zeros(3), iters=2))
        Rt, tt = V6._rel_pose(gt[k0], gt[k0 + H])

        def score(T, key):
            er = np.degrees(np.linalg.norm(D6.rotvec(T[0] @ Rt.T)))
            et = np.linalg.norm(T[1] - tt)
            res[key][0].append(er)
            res[key][1].append(et)

        score(compose(S), ("fast", 0))
        for d in DELAYS:
            score(run_naive(S, M, d), ("naive", d))
            score(run_matched(S, M, d), ("matched", d))
        nh += 1

    print(f"delay fusion ({seq.split('_freiburg')[-1]} @15 Hz proxy, "
          f"{nh} horizons of {H} steps = {H/15:.1f} s; slow tier "
          f"cloud-only per {KF}-step interval, d = arrival latency in "
          f"fast steps):")
    e, t = res[("fast", 0)]
    print(f"  fast-only (vision chain) : rot med {np.median(e):.2f} "
          f"p90 {np.percentile(e, 90):.2f} | transl med "
          f"{np.median(t)*1000:.1f} mm p90 "
          f"{np.percentile(t, 90)*1000:.1f}", flush=True)
    for d in DELAYS:
        for pol in ("naive", "matched"):
            e, t = res[(pol, d)]
            print(f"  {pol:7s} d={d} ({d*1000//15:3d} ms) : rot med "
                  f"{np.median(e):.2f} p90 {np.percentile(e, 90):.2f} | "
                  f"transl med {np.median(t)*1000:.1f} mm p90 "
                  f"{np.percentile(t, 90)*1000:.1f}", flush=True)


# --------------------------------------------------------------------------
#  selftest: pure algebra, no data
# --------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(D6.RNG)

    def rnd_T(scale=0.05):
        return (D6._R_of_w(rng.normal(size=3) * scale),
                rng.normal(size=3) * scale)

    # compose/inv identities
    A, B = rnd_T(), rnd_T()
    C = chain(A, B)
    Ii = chain(inv(A), chain(A, B))
    assert np.allclose(Ii[0], B[0]) and np.allclose(Ii[1], B[1])
    assert np.allclose(compose([B, A])[0], C[0])

    # d=0: naive == matched exactly (same windows fuse the same way)
    S = [rnd_T() for _ in range(H)]
    M = [rnd_T() for _ in range(H // KF)]
    a = run_naive(S, M, 0)
    b = run_matched(S, M, 0)
    assert np.allclose(a[0], b[0], atol=1e-12)
    assert np.allclose(a[1], b[1], atol=1e-12)

    # motion reversal + perfect slow measurements: matched must beat
    # naive at d=2 (the stale window straddles the reversal)
    true = [(D6._R_of_w(np.array([0, 0, (1 if s < 6 else -1) * 0.04])),
             np.array([(1 if s < 6 else -1) * 0.03, 0, 0]))
            for s in range(H)]
    Sn = [(D6._R_of_w(D6.rotvec(R) + rng.normal(size=3) * 0.02),
           t + rng.normal(size=3) * 0.015) for R, t in true]
    Mt = [compose(true[m * KF:(m + 1) * KF]) for m in range(H // KF)]
    Tt = compose(true)
    errs = {}
    for pol, fn in (("naive", run_naive), ("matched", run_matched)):
        T = fn(Sn, Mt, 2)
        errs[pol] = (np.linalg.norm(D6.rotvec(T[0] @ Tt[0].T))
                     + np.linalg.norm(T[1] - Tt[1]))
    assert errs["matched"] < errs["naive"], errs
    print(f"delayfuse selftest OK (reversal d=2: matched "
          f"{errs['matched']:.4f} < naive {errs['naive']:.4f})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if cmd == "selftest":
        selftest()
    else:
        bench(sys.argv[2] if len(sys.argv) > 2 else D6.SEQ)
