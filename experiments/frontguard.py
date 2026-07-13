"""Frontend odometry-trust guard experiment (do-no-harm study).

Question: can a FRONTEND guard recover the two odometry-EXCELLENT logs (ACES,
belgioioso) toward odometry parity WITHOUT regressing the odometry-DRIFTING logs
(Intel, fr079, fr101)?

Across six CARMEN logs the scan-matching frontend helps IFF odometry drifts and
HURTS when odometry is already excellent (RESULTS.md "the frontend do-no-harm
gap"). The shipped frontend (ssp_bounded.add_keyframe) matches every keyframe and,
when the match passes the 0.45 m / 11 deg gate, commits FULLY to the matched pose
`cand`. The hook `_frontend_accept(self, guess, cand, c01)` returns `cand` by
default. Here we subclass BoundedSLAM and override ONLY that hook to shrink `cand`
toward the odometry-propagated `guess` when the match is low-confidence.

Rules compared (all deterministic):
  identity  -- return cand  (== shipped baseline; sanity check)
  blend     -- est = guess + alpha*(cand-guess),
               alpha = clip((r - lo)/(hi - lo), 0, 1), r = c01/coh_ref
  bayes     -- Gaussian odometry-prior blend: prior stiffness rises as r falls,
               alpha = 1/(1 + s^2), s = kappa*(1-conf)*[mag term],
               conf = clip((r-lo)/(hi-lo),0,1); mag_couple couples s to |cand-guess|
  cap       -- correction-magnitude prior: shrink so |cand-guess| <= base*conf

Entry points:
  python3 -m experiments.frontguard selftest        # fast determinism / identity check
  python3 -m experiments.frontguard diag            # per-frame c01/coh_ref & |corr| dists
  python3 -m experiments.frontguard sweep           # full 5-log rule/param sweep + verdict
  python3 -m experiments.frontguard run <log> <rule> [k=v ...]
"""
import sys
import time

import numpy as np

import sspslam.encoder as S
import sspslam.frontend as C
import sspslam.lattice as L
import sspslam.bounded as B

VALID_MAX = 40.0

# raw-odometry and shipped-frontend reference ATEs (RESULTS.md), for context.
ODO = {"intel": 24.2, "fr079": 14.4, "aces": 5.41, "fr101": 8.56, "belgioioso": 1.72}


# ----------------------------------------------------------------------------
class FrontGuardSLAM(B.BoundedSLAM):
    """BoundedSLAM + an odometry-trust guard on the frontend accept hook.

    Everything except `_frontend_accept` is inherited unchanged, so guard_rule
    == 'identity' reproduces the shipped trajectory bit-for-bit."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.guard_rule = "identity"
        # blend / bayes ratio window
        self.lo = 0.6
        self.hi = 1.0
        # bayes
        self.kappa = 3.0
        self.mag_couple = False
        self.dref = 0.20
        # cap
        self.cap_base = 0.20
        # diagnostics: per accepted-frame records
        self.record = False
        self.frames = []   # (k, c01, coh_ref_pre, ratio, corr_norm, alpha)

    def _alpha(self, r, corr_norm):
        """Trust weight in [0,1] for the frontend correction."""
        rule = self.guard_rule
        if rule == "identity":
            return 1.0
        if rule == "const":
            # coherence-BLIND uniform damping: est = guess + alpha0*(cand-guess)
            # for every accepted match. Control to show the coherence gate adds
            # no separation over a plain global "trust odometry more" bias.
            return float(self.alpha0)
        if rule == "blend":
            if self.hi <= self.lo:
                return 1.0 if r >= self.hi else 0.0
            return float(np.clip((r - self.lo) / (self.hi - self.lo), 0.0, 1.0))
        if rule == "bayes":
            if self.hi <= self.lo:
                conf = 1.0 if r >= self.hi else 0.0
            else:
                conf = float(np.clip((r - self.lo) / (self.hi - self.lo), 0.0, 1.0))
            s = self.kappa * (1.0 - conf)
            if self.mag_couple:
                s *= corr_norm / max(self.dref, 1e-9)
            return 1.0 / (1.0 + s * s)
        if rule == "cap":
            if self.hi <= self.lo:
                conf = 1.0 if r >= self.hi else 0.0
            else:
                conf = float(np.clip((r - self.lo) / (self.hi - self.lo), 0.0, 1.0))
            budget = self.cap_base * conf
            if corr_norm <= budget or corr_norm < 1e-9:
                return 1.0
            return float(budget / corr_norm)
        raise ValueError(f"unknown guard_rule {rule!r}")

    def _frontend_accept(self, guess, cand, c01):
        # coh_ref has already been EMA-updated with this frame's c01 upstream;
        # recover the PRE-update baseline so the session reference is not
        # contaminated by the very frame we are judging.
        cr_post = self.coh_ref
        cr_pre = (cr_post - 0.05 * c01) / 0.95 if cr_post is not None else c01
        cr_pre = max(cr_pre, 1e-12)
        r = c01 / cr_pre
        corr = cand - guess
        corr_norm = float(np.hypot(corr[0], corr[1]))
        alpha = self._alpha(r, corr_norm)
        if self.record:
            self.frames.append((self.k, float(c01), float(cr_pre), float(r),
                                corr_norm, float(alpha)))
        if alpha >= 1.0 - 1e-12:
            return cand
        est = guess.copy()
        est[0] = guess[0] + alpha * corr[0]
        est[1] = guess[1] + alpha * corr[1]
        est[2] = S.wrap(guess[2] + alpha * S.wrap(cand[2] - guess[2]))
        return est


# ----------------------------------------------------------------------------
def _slam(rule, record=False, **over):
    slam = FrontGuardSLAM(robust=True, attempt_every=4, relax_every=25,
                          gap_kf=300, recent_aids=12)
    slam.store_dtype = np.complex64
    slam.guard_rule = rule
    slam.record = record
    for k, v in over.items():
        setattr(slam, k, v)
    return slam


def _run(keys, odom, rule, record=False, **over):
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    slam = _slam(rule, record=record, **over)
    est = np.zeros((n, 3))
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < VALID_MAX, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = opose if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
    if slam.dirty:
        slam.relax()
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    return fin, slam


# ---- evaluation -------------------------------------------------------------
def _eval_timestamp(path, fin, kts):
    ref = C.parse_flaser(path.replace(".log", ".gfs.log"))
    rts = np.array([t for _, _, t in ref])
    rxy = np.stack([p[:2] for _, p, _ in ref])
    j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
    good = np.abs(rts - kts[j]) < 0.3
    al = C.align_se2(fin[j[good], :2], rxy[good])
    e = np.linalg.norm(al - rxy[good], axis=1)
    return np.sqrt((e ** 2).mean()), np.median(e), e.max()


def _ref_identity(path, keys):
    kts = np.array([t for _, _, t in keys])
    raw = C.parse_flaser(path)
    gfs = C.parse_flaser(path.replace(".log", ".gfs.log"))
    gxy = np.stack([p[:2] for _, p, _ in gfs])
    idx = {}
    for i, (r, _, _) in enumerate(raw):
        idx.setdefault(r.tobytes(), []).append(i)
    gts, keep = [], []
    for m, (r, _, _) in enumerate(gfs):
        b = r.tobytes()
        if b in idx:
            gts.append(raw[idx[b][0]][2])
            keep.append(m)
    gts = np.array(gts)
    gxy = gxy[keep]
    o = np.argsort(gts)
    gts, gxy = gts[o], gxy[o]
    uniq = np.concatenate([[True], np.diff(gts) > 0])
    gts, gxy = gts[uniq], gxy[uniq]
    rx = np.interp(kts, gts, gxy[:, 0])
    ry = np.interp(kts, gts, gxy[:, 1])
    valid = (kts >= gts[0]) & (kts <= gts[-1])
    return np.stack([rx, ry], 1), valid


def _eval_identity(path, fin, keys):
    ref, valid = _ref_identity(path, keys)
    al = C.align_se2(fin[valid, :2], ref[valid])
    e = np.linalg.norm(al - ref[valid], axis=1)
    return np.sqrt((e ** 2).mean()), np.median(e), e.max()


LOGS = {
    "intel": ("data/intel.log", "ts"),
    "fr079": ("data/fr079.log", "ts"),
    "aces": ("data/aces.log", "ts"),
    "fr101": ("data/fr101.log", "ts"),
    "belgioioso": ("data/belgioioso.log", "id"),
}


def evaluate_run(name, rule, record=False, **over):
    path, mode = LOGS[name]
    keys = C.keyframes(C.parse_flaser(path))
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    t0 = time.time()
    fin, slam = _run(keys, odom, rule, record=record, **over)
    if mode == "ts":
        rmse, med, mx = _eval_timestamp(path, fin, kts)
        # raw odometry ATE, same eval
        odo_r, _, _ = _eval_timestamp(path, odom, kts)
    else:
        rmse, med, mx = _eval_identity(path, fin, keys)
        odo_r, _, _ = _eval_identity(path, odom, keys)
    dt = time.time() - t0
    nloop = sum(1 for e in slam.edges if e[5] == "loop")
    return dict(name=name, rmse=rmse, med=med, mx=mx, odo=odo_r,
                nloop=nloop, veto=slam.n_veto, dt=dt, slam=slam, fin=fin)


# ---- entry points -----------------------------------------------------------
def selftest():
    """Determinism + identity-reproduces-shipped on a short prefix."""
    path = "data/aces.log"
    keys = C.keyframes(C.parse_flaser(path))[:400]
    odom = np.stack([k[1] for k in keys])
    finA, _ = _run(keys, odom, "identity")
    finB, _ = _run(keys, odom, "identity")
    assert np.array_equal(finA, finB), "identity non-deterministic!"
    # compare against the shipped BoundedSLAM directly (must be bit-identical)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    ship = B.BoundedSLAM(robust=True, attempt_every=4, relax_every=25,
                         gap_kf=300, recent_aids=12)
    ship.store_dtype = np.complex64
    est = np.zeros((len(keys), 3))
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < VALID_MAX, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = opose if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = ship.add_keyframe(pts, w, guess)
    if ship.dirty:
        ship.relax()
    finS = np.stack([ship.pose_of(k) for k in range(len(keys))])
    assert np.allclose(finA, finS, atol=0, rtol=0), "identity != shipped!"
    # a guard rule must actually change something
    finC, _ = _run(keys, odom, "blend", lo=0.9, hi=1.0)
    assert not np.array_equal(finA, finC), "blend had no effect!"
    finC2, _ = _run(keys, odom, "blend", lo=0.9, hi=1.0)
    assert np.array_equal(finC, finC2), "blend non-deterministic!"
    print("selftest OK: identity==shipped (bit-identical), guard deterministic & active")


def diag():
    """Per-frame c01/coh_ref ratio and |cand-guess| distributions on the three
    diagnostic logs. Are ACES/belgioioso distinguishable per-frame from Intel?"""
    for name in ("aces", "belgioioso", "intel", "fr101"):
        path, mode = LOGS[name]
        keys = C.keyframes(C.parse_flaser(path))
        odom = np.stack([k[1] for k in keys])
        _, slam = _run(keys, odom, "identity", record=True)
        F = np.array(slam.frames)  # k,c01,coh_ref_pre,ratio,corr_norm,alpha
        if len(F) == 0:
            print(f"{name}: no accepted frontend matches")
            continue
        ratio = F[:, 3]
        corr = F[:, 4]
        q = lambda a, p: np.percentile(a, p)
        print(f"\n== {name}  ({len(F)} accepted matches)")
        print(f"  c01/coh_ref  min {ratio.min():.2f}  p10 {q(ratio,10):.2f}  "
              f"p25 {q(ratio,25):.2f}  med {np.median(ratio):.2f}  "
              f"p75 {q(ratio,75):.2f}  p90 {q(ratio,90):.2f}  max {ratio.max():.2f}")
        print(f"  |cand-guess| med {np.median(corr):.3f}  p75 {q(corr,75):.3f}  "
              f"p90 {q(corr,90):.3f}  p99 {q(corr,99):.3f}  max {corr.max():.3f} m")
        for thr in (0.5, 0.7, 0.85, 1.0):
            print(f"    frac ratio<{thr:.2f}: {np.mean(ratio < thr):.3f}")


BEST = None  # filled by sweep


def _fmt_row(r, base):
    d_ship = 100 * (r["rmse"] - base) / base if base else float("nan")
    d_odo = 100 * (r["rmse"] - r["odo"]) / r["odo"]
    return (f"  {r['name']:<11} ATE {r['rmse']:6.3f}  (odo {r['odo']:5.2f}, "
            f"vs-ship {d_ship:+6.1f}%, vs-odo {d_odo:+6.1f}%)  "
            f"loops {r['nloop']:3d}  {r['dt']:.0f}s")


def sweep():
    logs = ["intel", "fr079", "aces", "fr101", "belgioioso"]
    # baseline (identity == shipped) for every log first
    print("=== baseline: identity (== shipped frontend) ===", flush=True)
    base = {}
    for nm in logs:
        r = evaluate_run(nm, "identity")
        base[nm] = r["rmse"]
        print(_fmt_row(r, base[nm]), flush=True)

    # candidate rules/params (focused decisive set; the diag showed the ratio
    # distributions are indistinguishable across logs, so we probe gentle ->
    # aggressive selective damping + the one theoretical hope, magnitude
    # coupling, which exploits that ACES's harmful corrections are LARGER).
    cands = [
        ("blend", dict(lo=0.5, hi=1.0)),           # gentle selective
        ("blend", dict(lo=0.7, hi=1.0)),           # moderate selective
        ("blend", dict(lo=0.85, hi=1.15)),         # aggressive selective
        ("bayes", dict(lo=0.6, hi=1.1, kappa=4.0)),                     # smooth
        ("bayes", dict(lo=0.6, hi=1.1, kappa=4.0, mag_couple=True)),    # mag-coupled
        ("cap",   dict(lo=0.5, hi=1.0, cap_base=0.15)),                 # mag cap
    ]

    results = []
    for rule, over in cands:
        tag = rule + " " + ",".join(f"{k}={v}" for k, v in over.items())
        print(f"\n=== {tag} ===", flush=True)
        rows = {}
        for nm in logs:
            r = evaluate_run(nm, rule, **over)
            rows[nm] = r
            print(_fmt_row(r, base[nm]), flush=True)
        # do-no-harm scoring
        drift = ["intel", "fr079", "fr101"]
        excellent = ["aces", "belgioioso"]
        worst_regress = max(100 * (rows[nm]["rmse"] - base[nm]) / base[nm]
                            for nm in drift)
        # gain toward odometry on excellent logs (positive = improved toward odo)
        gains = {}
        for nm in excellent:
            span = base[nm] - rows[nm]["odo"]  # how much above odometry we were
            gains[nm] = (base[nm] - rows[nm]["rmse"]) / span if span > 1e-9 else 0.0
        worst_gain = min(gains.values())
        results.append((tag, worst_regress, worst_gain, gains, rows))
        print(f"  -> worst drift regress {worst_regress:+.1f}%   "
              f"worst excellent gain-toward-odo {100*worst_gain:+.1f}%  "
              f"(aces {100*gains['aces']:+.0f}%, belg {100*gains['belgioioso']:+.0f}%)",
              flush=True)

    # verdict: do-no-harm = drift regress < 5% AND both excellent improve materially
    print("\n=== VERDICT ===")
    ok = [x for x in results if x[1] < 5.0 and x[2] > 0.15]
    if ok:
        best = max(ok, key=lambda x: x[2])  # most excellent gain among do-no-harm
        print(f"DO-NO-HARM ACHIEVED by: {best[0]}")
        print(f"  worst drift regress {best[1]:+.1f}%, worst excellent gain "
              f"{100*best[2]:+.1f}%")
    else:
        # pick least-bad by worst-case-over-logs ratio to best-known, like shipped
        print("NO rule achieves do-no-harm (drift<5% AND both excellent gain>15%).")
        # report the frontier: rule with best excellent gain at <5% regress
        safe = [x for x in results if x[1] < 5.0]
        if safe:
            b = max(safe, key=lambda x: x[2])
            print(f"  best do-no-harm-SAFE (regress<5%): {b[0]} -> "
                  f"excellent gain only {100*b[2]:+.1f}% "
                  f"(aces {100*b[3]['aces']:+.0f}%, belg {100*b[3]['belgioioso']:+.0f}%)")
        # rule with best excellent gain regardless of regress
        g = max(results, key=lambda x: x[2])
        print(f"  best excellent gain overall: {g[0]} -> "
              f"gain {100*g[2]:+.1f}% but drift regress {g[1]:+.1f}%")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        selftest()
    elif cmd == "diag":
        diag()
    elif cmd == "sweep":
        sweep()
    elif cmd == "run":
        nm, rule = sys.argv[2], sys.argv[3]
        over = {}
        for a in sys.argv[4:]:
            k, v = a.split("=")
            over[k] = (v == "True") if v in ("True", "False") else (
                float(v) if "." in v or v.replace("-", "").isdigit() else v)
        r = evaluate_run(nm, rule, **over)
        print(_fmt_row(r, r["rmse"]))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
