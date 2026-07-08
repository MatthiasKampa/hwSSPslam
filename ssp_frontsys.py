"""Windowed systematic-vs-random frontend guard (the LAST do-no-harm escape hatch).

Context (RESULTS.md "The frontend do-no-harm guard: a clean negative"): the
scan-matching frontend HELPS on odometry-drifting logs (Intel 24->2.4, fr101
8.6->1.9) and HURTS on odometry-excellent logs (ACES 5.41->6.38, belgioioso
1.72->2.45). A full per-frame sweep proved NO coherence/magnitude rule achieves
do-no-harm: the c01/coh_ref distributions are identical across regimes. THE KEY
UNEXPLAINED DATUM: ACES's per-frame corrections |cand-guess| are LARGER (med
0.036) than Intel's (0.017) even though ACES odometry is far better. The only
reconciliation: ACES's corrections are RANDOM (they cancel, don't accumulate) --
the frontend fighting good odometry with scan-match noise -- while Intel's are
SYSTEMATIC (they accumulate coherently to fix 24 m of real drift). Per-frame
signals are blind to this; a WINDOWED signal might see it.

This module subclasses BoundedSLAM and overrides ONLY the `_frontend_accept`
hook, maintaining a sliding window of the last W accepted incremental frontend
corrections delta_i = cand_i - guess_i. It computes a "systematicness" ratio

    rho = || sum_i delta_i_xy ||  /  ( sum_i || delta_i_xy || + eps )   in [0,1]

rho -> 1: corrections align (SYSTEMATIC drift-fixing); rho -> 0: corrections
cancel (RANDOM noise). Heading systematicness is the scalar analogue.

Reference FRAME. delta_i is intrinsically a WORLD-frame vector (difference of two
world SE2 poses). Accumulating odometry drift is a world-frame phenomenon: the
net world displacement the frontend imposes over a window, ||sum delta_i||, is
exactly "how far the frontend dragged the trajectory net" -- large relative to
the path length sum||delta_i|| iff it is systematically pushing one way (fixing
drift). rho is that ratio. A BODY-frame variant (rotate each delta by -guess
heading) asks "are corrections consistent in the robot's instantaneous frame",
which is more invariant to turns but mixes heading error into the xy signal. We
take WORLD as primary and report BODY as a robustness check; diag computes both.

Entry points:
  python3 ssp_frontsys.py selftest     # determinism / identity == shipped
  python3 ssp_frontsys.py diag         # rho distributions, all logs/W/frames
  python3 ssp_frontsys.py sweep        # guard (lo,hi,W) sweep + do-no-harm verdict
  python3 ssp_frontsys.py run <log> <rule> [k=v ...]
"""
import sys
import time

import numpy as np

import ssp_slam as S
import ssp_slam_carmen as C
import ssp_slam_loop as L
import ssp_bounded as B
import ssp_frontguard as FG   # reuse eval harness (LOGS, _eval_*, VALID_MAX)

VALID_MAX = FG.VALID_MAX
LOGS = FG.LOGS
# shipped-frontend (identity) reference ATEs and raw-odometry ATEs (RESULTS.md).
SHIP = {"intel": 2.44, "fr079": 5.52, "aces": 6.21, "fr101": 1.88, "belgioioso": 2.64}
ODO = {"intel": 24.2, "fr079": 14.4, "aces": 5.41, "fr101": 8.56, "belgioioso": 1.72}


# ----------------------------------------------------------------------------
def _rho_xy(win):
    """Systematicness of a window of xy correction vectors, in [0,1]."""
    s = win.sum(0)
    num = float(np.hypot(s[0], s[1]))
    den = float(np.hypot(win[:, 0], win[:, 1]).sum()) + 1e-9
    return num / den


def _rho_scalar(win):
    """Systematicness of a window of scalar (heading) corrections, in [0,1]."""
    num = abs(float(np.sum(win)))
    den = float(np.sum(np.abs(win))) + 1e-9
    return num / den


def rho_series(dxy, W):
    """Per-frame rho over a trailing window of length W (nan during warm-up)."""
    n = len(dxy)
    out = np.full(n, np.nan)
    for i in range(W - 1, n):
        out[i] = _rho_xy(dxy[i - W + 1:i + 1])
    return out


def rho_series_scalar(dh, W):
    n = len(dh)
    out = np.full(n, np.nan)
    for i in range(W - 1, n):
        out[i] = _rho_scalar(dh[i - W + 1:i + 1])
    return out


# ----------------------------------------------------------------------------
class FrontSysSLAM(B.BoundedSLAM):
    """BoundedSLAM + a windowed systematic-vs-random frontend guard.

    guard_rule == 'identity' inherits everything unchanged -> reproduces the
    shipped trajectory bit-for-bit (asserted in selftest)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.guard_rule = "identity"   # identity | blend
        self.W = 30                    # sliding-window length (accepted corrs)
        self.lo = 0.2                  # rho below lo  -> full damping (alpha 0)
        self.hi = 0.6                  # rho above hi  -> full trust  (alpha 1)
        self.frame = "world"           # world | body
        self.rho_key = "xy"            # xy | heading  (which window drives alpha)
        # diagnostics: per-accepted-frame raw records (frame-agnostic)
        self.record = False
        self.frames = []               # (k, dxw, dyw, dth, guess_heading)
        # online guard state
        self._dxy = []                 # frame-'self.frame' xy corrections
        self._dh = []                  # heading corrections

    def _alpha(self, rho):
        """Trust weight in [0,1]. Warm-up (rho is None) -> full trust: never
        damp before W corrections accumulate (Intel's early drift-fixing is
        critical)."""
        rule = self.guard_rule
        if rule == "identity" or rho is None:
            return 1.0
        if rule == "blend":
            if self.hi <= self.lo:
                return 1.0 if rho >= self.hi else 0.0
            return float(np.clip((rho - self.lo) / (self.hi - self.lo), 0.0, 1.0))
        raise ValueError(f"unknown guard_rule {rule!r}")

    def _frontend_accept(self, guess, cand, c01):
        corr = cand - guess
        dth = S.wrap(cand[2] - guess[2])
        dxy_w = corr[:2].astype(float).copy()
        # window vector in the configured reference frame
        if self.frame == "body":
            dxy = S._rot(-guess[2]) @ dxy_w
        else:
            dxy = dxy_w
        self._dxy.append(dxy)
        self._dh.append(dth)
        if self.record:
            self.frames.append((self.k, float(dxy_w[0]), float(dxy_w[1]),
                                float(dth), float(guess[2])))
        # rho over the trailing window (None until W corrections accumulate)
        rho = None
        if len(self._dxy) >= self.W:
            if self.rho_key == "heading":
                rho = _rho_scalar(np.array(self._dh[-self.W:]))
            else:
                rho = _rho_xy(np.array(self._dxy[-self.W:]))
        alpha = self._alpha(rho)
        if alpha >= 1.0 - 1e-12:
            return cand
        est = guess.copy()
        est[0] = guess[0] + alpha * corr[0]
        est[1] = guess[1] + alpha * corr[1]
        est[2] = S.wrap(guess[2] + alpha * dth)
        return est


# ----------------------------------------------------------------------------
def _slam(rule, record=False, **over):
    slam = FrontSysSLAM(robust=True, attempt_every=4, relax_every=25,
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


def evaluate_run(name, rule, record=False, **over):
    path, mode = LOGS[name]
    keys = C.keyframes(C.parse_flaser(path))
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    t0 = time.time()
    fin, slam = _run(keys, odom, rule, record=record, **over)
    if mode == "ts":
        rmse, med, mx = FG._eval_timestamp(path, fin, kts)
        odo_r, _, _ = FG._eval_timestamp(path, odom, kts)
    else:
        rmse, med, mx = FG._eval_identity(path, fin, keys)
        odo_r, _, _ = FG._eval_identity(path, odom, keys)
    dt = time.time() - t0
    nloop = sum(1 for e in slam.edges if e[5] == "loop")
    return dict(name=name, rmse=rmse, med=med, mx=mx, odo=odo_r,
                nloop=nloop, dt=dt, slam=slam, fin=fin)


# ---- entry points -----------------------------------------------------------
def selftest():
    """Determinism + identity-reproduces-shipped on a short prefix."""
    path = "data/aces.log"
    keys = C.keyframes(C.parse_flaser(path))[:400]
    odom = np.stack([k[1] for k in keys])
    finA, _ = _run(keys, odom, "identity")
    finB, _ = _run(keys, odom, "identity")
    assert np.array_equal(finA, finB), "identity non-deterministic!"
    # bit-identical to shipped BoundedSLAM
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
    # a guard rule must actually change something, and be deterministic
    finC, _ = _run(keys, odom, "blend", lo=0.9, hi=1.0, W=10)
    assert not np.array_equal(finA, finC), "blend had no effect!"
    finC2, _ = _run(keys, odom, "blend", lo=0.9, hi=1.0, W=10)
    assert np.array_equal(finC, finC2), "blend non-deterministic!"
    print("selftest OK: identity==shipped (bit-identical), guard deterministic & active")


def _record_deltas(name):
    """Run identity+record; return (dxw, dyw, dth, gh) arrays of corrections."""
    path, _ = LOGS[name]
    keys = C.keyframes(C.parse_flaser(path))
    odom = np.stack([k[1] for k in keys])
    _, slam = _run(keys, odom, "identity", record=True)
    F = np.array(slam.frames)  # (n,5): k, dxw, dyw, dth, gh
    return F


def _body_dxy(F):
    """Rotate world-frame xy corrections into per-frame body frame (-gh)."""
    dxy = F[:, 1:3]
    gh = F[:, 4]
    c, s = np.cos(-gh), np.sin(-gh)
    bx = c * dxy[:, 0] - s * dxy[:, 1]
    by = s * dxy[:, 0] + c * dxy[:, 1]
    return np.stack([bx, by], 1)


def diag():
    """rho distributions per log, for W in {10,30,100}, world & body frames.
    THE TEST: does rho separate high on Intel/fr079/fr101 (systematic) vs low
    on ACES/belgioioso (random)?"""
    logs = ["intel", "fr079", "fr101", "aces", "belgioioso"]
    drift = {"intel", "fr079", "fr101"}
    Ws = [10, 30, 100]
    q = lambda a, p: float(np.percentile(a, p))
    recs = {}
    for nm in logs:
        F = _record_deltas(nm)
        recs[nm] = F
        print(f"{nm:<11} accepted frontend corrections: {len(F)}", flush=True)

    for frame in ("world", "body"):
        for W in Ws:
            print(f"\n===== rho[xy]  frame={frame}  W={W} "
                  f"  (regime: DRIFT=systematic-expected, EXCELLENT=random-expected)")
            print(f"  {'log':<11} {'regime':<10} {'n':>5}  "
                  f"{'p10':>6} {'med':>6} {'p90':>6}  {'<0.3':>6} {'<0.5':>6}")
            for nm in logs:
                F = recs[nm]
                dxy = _body_dxy(F) if frame == "body" else F[:, 1:3]
                rho = rho_series(np.ascontiguousarray(dxy), W)
                rho = rho[~np.isnan(rho)]
                reg = "DRIFT" if nm in drift else "EXCELLENT"
                if len(rho) == 0:
                    print(f"  {nm:<11} {reg:<10} {0:>5}  (window never fills)")
                    continue
                print(f"  {nm:<11} {reg:<10} {len(rho):>5}  "
                      f"{q(rho,10):6.3f} {np.median(rho):6.3f} {q(rho,90):6.3f}  "
                      f"{np.mean(rho<0.3):6.3f} {np.mean(rho<0.5):6.3f}")

    # heading systematicness, world frame, W=30 (single representative view)
    print(f"\n===== rho[heading]  W=30")
    print(f"  {'log':<11} {'regime':<10} {'p10':>6} {'med':>6} {'p90':>6}")
    for nm in logs:
        F = recs[nm]
        rho = rho_series_scalar(F[:, 3], 30)
        rho = rho[~np.isnan(rho)]
        reg = "DRIFT" if nm in drift else "EXCELLENT"
        if len(rho) == 0:
            print(f"  {nm:<11} {reg:<10} (window never fills)")
            continue
        print(f"  {nm:<11} {reg:<10} {q(rho,10):6.3f} {np.median(rho):6.3f} "
              f"{q(rho,90):6.3f}")

    # separation verdict: compare median rho of drift vs excellent (world W=30)
    print("\n===== SEPARATION VERDICT (world xy) =====")
    for W in Ws:
        meds = {}
        for nm in logs:
            F = recs[nm]
            rho = rho_series(np.ascontiguousarray(F[:, 1:3]), W)
            rho = rho[~np.isnan(rho)]
            meds[nm] = np.median(rho) if len(rho) else np.nan
        dmin = min(meds[nm] for nm in drift)
        emax = max(meds[nm] for nm in logs if nm not in drift)
        sep = dmin - emax
        print(f"  W={W:>3}: min(drift med)={dmin:.3f}  max(excellent med)={emax:.3f}"
              f"  -> {'SEPARATES' if sep > 0.05 else 'OVERLAP'} (gap {sep:+.3f})")
    print("  (SEPARATES => build guard in sweep(); OVERLAP => decisive NEGATIVE, "
          "the 'systematic' story is wrong.)")


def _fmt_row(r, base):
    d_ship = 100 * (r["rmse"] - base) / base if base else float("nan")
    d_odo = 100 * (r["rmse"] - r["odo"]) / r["odo"]
    return (f"  {r['name']:<11} ATE {r['rmse']:6.3f}  (odo {r['odo']:5.2f}, "
            f"vs-ship {d_ship:+6.1f}%, vs-odo {d_odo:+6.1f}%)  "
            f"loops {r['nloop']:3d}  {r['dt']:.0f}s")


def sweep(cands=None):
    logs = ["intel", "fr079", "aces", "fr101", "belgioioso"]
    print("=== baseline: identity (== shipped frontend) ===", flush=True)
    base = {}
    for nm in logs:
        r = evaluate_run(nm, "identity")
        base[nm] = r["rmse"]
        print(_fmt_row(r, base[nm]), flush=True)

    if cands is None:
        cands = [
            ("blend", dict(W=30, lo=0.2, hi=0.5)),
            ("blend", dict(W=30, lo=0.3, hi=0.6)),
            ("blend", dict(W=10, lo=0.3, hi=0.6)),
            ("blend", dict(W=100, lo=0.2, hi=0.5)),
            ("blend", dict(W=30, lo=0.4, hi=0.7)),
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
        drift = ["intel", "fr079", "fr101"]
        excellent = ["aces", "belgioioso"]
        worst_regress = max(100 * (rows[nm]["rmse"] - base[nm]) / base[nm]
                            for nm in drift)
        gains = {}
        for nm in excellent:
            span = base[nm] - rows[nm]["odo"]
            gains[nm] = (base[nm] - rows[nm]["rmse"]) / span if span > 1e-9 else 0.0
        worst_gain = min(gains.values())
        results.append((tag, worst_regress, worst_gain, gains, rows))
        print(f"  -> worst drift regress {worst_regress:+.1f}%   "
              f"worst excellent gain-toward-odo {100*worst_gain:+.1f}%  "
              f"(aces {100*gains['aces']:+.0f}%, belg {100*gains['belgioioso']:+.0f}%)",
              flush=True)

    print("\n=== VERDICT ===")
    ok = [x for x in results if x[1] < 5.0 and x[2] > 0.15]
    if ok:
        best = max(ok, key=lambda x: x[2])
        print(f"DO-NO-HARM ACHIEVED by: {best[0]}")
        print(f"  worst drift regress {best[1]:+.1f}%, worst excellent gain "
              f"{100*best[2]:+.1f}%")
    else:
        print("NO rule achieves do-no-harm (drift<5% AND both excellent gain>15%).")
        safe = [x for x in results if x[1] < 5.0]
        if safe:
            b = max(safe, key=lambda x: x[2])
            print(f"  best do-no-harm-SAFE (regress<5%): {b[0]} -> "
                  f"excellent gain only {100*b[2]:+.1f}% "
                  f"(aces {100*b[3]['aces']:+.0f}%, belg {100*b[3]['belgioioso']:+.0f}%)")
        g = max(results, key=lambda x: x[2])
        print(f"  best excellent gain overall: {g[0]} -> "
              f"gain {100*g[2]:+.1f}% but drift regress {g[1]:+.1f}%")
    return results


def sweeph():
    """Heading-keyed guard sweep: rho over the HEADING window is the only signal
    whose per-log MEDIAN orders the regimes correctly (drift > excellent). Test
    whether that median separation survives per-frame (tail overlap) as an actual
    do-no-harm guard."""
    cands = [
        ("blend", dict(rho_key="heading", W=30, lo=0.05, hi=0.20)),
        ("blend", dict(rho_key="heading", W=30, lo=0.08, hi=0.25)),
        ("blend", dict(rho_key="heading", W=30, lo=0.10, hi=0.35)),
        ("blend", dict(rho_key="heading", W=10, lo=0.10, hi=0.35)),
        ("blend", dict(rho_key="heading", W=100, lo=0.05, hi=0.20)),
    ]
    sweep(cands)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        selftest()
    elif cmd == "diag":
        diag()
    elif cmd == "sweep":
        sweep()
    elif cmd == "sweeph":
        sweeph()
    elif cmd == "run":
        nm, rule = sys.argv[2], sys.argv[3]
        over = {}
        for a in sys.argv[4:]:
            k, v = a.split("=")
            over[k] = (v == "True") if v in ("True", "False") else (
                float(v) if "." in v else (int(v) if v.replace("-", "").isdigit()
                                           else v))
        r = evaluate_run(nm, rule, **over)
        print(_fmt_row(r, SHIP.get(nm, r["rmse"])))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
