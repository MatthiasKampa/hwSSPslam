#!/usr/bin/env python3
"""Live FPGA-in-the-loop visualization (classroom, iCEbreaker matcher build).

One long-lived process:
  - FEED: the dynenv classroom tour, looped FOREVER in real time (5 Hz
    keyframes = the tour's native 0.1 m / 0.5 m/s stride). The tour is
    closed with INTERPOLATED BRIDGE poses (GT is not a perfect loop);
    bridge scans are raycast in the same world with the same noise model.
    Every pass redraws sensor + odometry noise from a per-pass RNG
    ("slightly perturbed data"); nothing is ever reset — the python SLAM
    and the odometry chain persist across passes.
  - PYTHON SLAM: the shipped pipeline (BandSLAM, bench config, bridge2
    sampler) runs continuously — the bounded-map property live: endless
    revisits bundle into existing cells instead of growing the map.
  - FABRIC (the FPGA in the loop, top_match bitstream):
      * every keyframe's scan is ENCODED on the device (Q stays in the
        accumulator banks); every CHECK_EVERY-th keyframe is read back and
        compared bit-exact against the golden model — a live acceptance
        counter;
      * from pass 2 on, the fabric LOCALIZES against the frozen pass-1 map
        (2-bit QPSK codes per segment, the deploy store): paced candidate
        sweeps around the odometry-propagated prediction, argmax + parabolic
        refine → an independent position track;
      * the MAP IMAGE is probed from silicon: a delta probe (single point at
        the origin — rotation-invariant on this lattice) makes score(D) the
        matched-filter image of the stored map; world-grid pixels are probed
        against their nearest segments and stitched → "global walls";
      * "walls visible from a pickable position" = host ray-march on that
        fabric-probed image (first ridge crossing per ray).
  - UI: http://localhost:PORT — canvas + SSE. GT is drawn as a ghost and
    used for on-screen error DISPLAY ONLY (never enters any estimate).

Geometry between the matcher contract and SE2 poses is not hand-derived:
`calibrate()` empirically pins the (D, rho) <-> relative-pose mapping on
pass-1 keyframes with known poses and FAILS LOUDLY on mismatch.

Usage:
  python3 ice40/host/live.py selftest   # short headless run, numbers only
  python3 ice40/host/live.py serve [port] [laps]
"""
import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ice40" / "host"))

import ssp_dynenv as DE                                  # noqa: E402
import ssp_fpga as F                                     # noqa: E402
import ssp_ice40 as G                                    # noqa: E402
import ssp_slam as S                                     # noqa: E402
import ssp_slam_loop as L                                # noqa: E402
from vectors import Board, SweepError                    # noqa: E402

U = 0.25 / 256.0                 # position unit [m] (lambda_min / 256)
OBS_CURV = 0.0                   # per-axis curvature gate (0 = off; v2
                                 # blind-constant attempt REGRESSED — to be
                                 # calibrated from logged den/spread data)
RHO_FLAT = 0.0                   # rotation-flatness gate (0 = off)
HANDOFF_M = -1.0                 # two-segment handoff margin (<0 = off;
                                 # raw-score pick was content-biased)
USE_ARGMAX = True                # v6 fabric argmax for tracking sweeps
                                 # (6-byte reply + 9-cand refine batch)
CHECK_EVERY = 5                  # fabric encode readback cross-check cadence
SWEEP_T = 12                     # sweep grid step [U] (~1.2 cm)
SWEEP_N = 3                      # grid half-extent in steps -> 7x7
SWEEP_R = 2                      # rho half-extent -> 5 rotations
                                 # (heading odo noise ~0.25 deg/kf
                                 #  outgrew +-3 deg on long runs)
PACE = 220e-6                    # paced-sweep period (+jitter margin)
IMG_STEP = 0.12                  # map-image world grid [m]
IMG_RAD = 2.4                    # query probe radius [m] (MUST stay
                                 # 2.4 — 3.6 lands hits on exterior
                                 # teeth per the readout study)
IMG_RAD_MAP = 3.6                # map-image probe radius (full wall
                                 # coverage; readout-study recipe)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi



def _resample_step(xy, step):
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    scum = np.r_[0, np.cumsum(d)]
    n = max(16, int(scum[-1] / step))
    si = np.arange(n) * scum[-1] / n
    return np.stack([np.interp(si, scum, xy[:, 0]),
                     np.interp(si, scum, xy[:, 1])], 1)


def _clear_ok(segs, xy, margin=0.40):
    ang = np.arange(12) * (2 * np.pi / 12)
    for pt in xy[::7]:
        if S.raycast(segs, pt, ang).min() < margin:
            return False
    return True


def synth_traj(segs, kind, laps, rng):
    # Trajectory variants for synth worlds (STEP-spaced, tangent
    # headings + bench jitter); clearance-checked; fig8 shrinks
    # deterministically until it clears the world.
    if kind == "fig8":
        found = False
        for cx, cy, rot in ((8, 5, 0.0), (8, 5, 0.6), (7, 4.5, 0.3),
                            (9, 5.5, -0.4), (8, 4, 0.9)):
            for scale in (1.0, 0.85, 0.7, 0.6, 0.5):
                u = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
                bx = 5.6 * scale * np.sin(u)
                by = 6.0 * scale * np.sin(u) * np.cos(u)
                c, sn = np.cos(rot), np.sin(rot)
                base = np.stack([cx + c * bx - sn * by,
                                 cy + sn * bx + c * by], 1)
                xy = _resample_step(np.concatenate([base] * laps), DE.STEP)
                if _clear_ok(segs, xy):
                    found = True
                    break
            if found:
                break
        if not found:
            raise ValueError("fig8 cannot clear this world")
    elif kind == "waypoints":
        lo, hi = np.array([2.0, 1.5]), np.array([14.0, 8.5])

        def _leg_ok(a, b):
            d = b - a
            L2 = float(np.linalg.norm(d))
            if L2 < 1e-6:
                return True
            hit = S.raycast(segs, a,
                            np.array([np.arctan2(d[1], d[0])]))[0]
            if hit < L2 + 0.5:
                return False
            mid = np.stack([a + t * d for t in (0.25, 0.5, 0.75)])
            return _clear_ok(segs, mid, 0.45)

        xy = None
        for _ in range(24):                 # whole-chain retries
            pts = [np.array([8.0, 5.0])]
            tries = 0
            while len(pts) < 6 + 3 * laps and tries < 4000:
                tries += 1
                c = lo + rng.random(2) * (hi - lo)
                if np.linalg.norm(c - pts[-1]) < 2.0:
                    continue
                if not _leg_ok(pts[-1], c):
                    continue
                pts.append(c)
            if len(pts) < 6 or not _leg_ok(pts[-1], pts[0]):
                continue                    # home leg must be clear too
            cand = _resample_step(np.array(pts + [pts[0]]), DE.STEP)
            if _clear_ok(segs, cand):
                xy = cand
                break
        if xy is None:
            raise ValueError("waypoint path not clear")
    elif kind == "wobble":
        # orbit with a higher-frequency radial oscillation — feasible in
        # orbit worlds by construction (shrinks until clear), stresses
        # sustained heading-rate changes
        for amp in (0.22, 0.16, 0.10, 0.06):
            u = np.linspace(0, laps * 2 * np.pi, 4000, endpoint=False)
            sbase = 1.0 + 0.12 * np.sin(0.9 * u)
            swob = sbase * (1.0 + amp * np.sin(4.3 * u))
            a, b = 4.8 * swob, 1.9 * swob
            base = np.stack([8 + a * np.cos(u), 5 + b * np.sin(u)], 1)
            xy = _resample_step(base, DE.STEP)
            if _clear_ok(segs, xy):
                break
        else:
            raise ValueError("wobble cannot clear this world")
    elif kind == "reverse":
        # half the orbit forward, then backtrack — one 180-degree heading
        # event mid-run (the recovery-path stressor) plus reversed
        # revisit geometry
        peri = 2 * np.pi * (4.8 + 1.9) / 2 * laps
        g0 = None
        import ssp_synth as SY
        g0 = SY._traj(laps, np.random.default_rng(0), int(peri / DE.STEP))
        half = len(g0) // 2
        xy = np.concatenate([g0[:half, :2], g0[:half, :2][::-1]])
    else:
        raise ValueError(kind)
    dx, dy = np.gradient(xy[:, 0]), np.gradient(xy[:, 1])
    th = np.arctan2(dy, dx) + rng.normal(0, np.deg2rad(0.3), len(xy))
    return np.concatenate([xy, th[:, None]], 1)


# ------------------------------------------------------------------- feed
class Feed:
    """Looped tour + interpolated bridge, per-pass noise redraw.
    env: 'classroom' (dynenv) or an ssp_synth bench world ('mixed',
    'corridor', 'office', ...) — the acceptance-suite synthetic pair,
    raycast at 1024 beams so the fabric az grid applies unchanged."""

    def __init__(self, seed=11, laps=2, n_beams=1024, env="classroom",
                 traj="orbit"):
        self.seed, self.n_beams, self.env, self.traj = (seed, n_beams,
                                                        env, traj)
        if env == "classroom":
            self.segs = DE.classroom_segs(None)
            xy = DE.classroom_path(laps)
            th = DE._headings(xy, np.random.default_rng(seed))
            gt = np.concatenate([xy, th[:, None]], 1)
        else:
            import ssp_synth as SY
            self.segs = SY.WORLDS[env]()
            rngt = np.random.default_rng(seed)
            if traj == "orbit":
                peri = 2 * np.pi * (4.8 + 1.9) / 2 * laps
                gt = SY._traj(laps, rngt, int(peri / DE.STEP))
            else:
                gt = synth_traj(self.segs, traj, laps, rngt)
            xy, th = gt[:, :2], gt[:, 2]
        # bridge: interpolate last -> first (positions + shortest-arc heading)
        gap = float(np.linalg.norm(xy[0] - xy[-1]))
        nb = max(2, int(np.ceil(gap / DE.STEP)))
        t = np.linspace(0, 1, nb + 2)[1:-1, None]
        bxy = xy[-1] * (1 - t) + xy[0] * t
        bth = wrap(th[-1] + t[:, 0] * wrap(th[0] - th[-1]))
        self.gt_loop = np.concatenate(
            [gt, np.concatenate([bxy, bth[:, None]], 1)])
        self.n_tour, self.n_loop = len(gt), len(self.gt_loop)
        self.beam = -np.pi + np.arange(n_beams) * (2 * np.pi / n_beams)
        self.k_global = 0
        self.odom = None             # continuous noisy odometry chain
        self._prev_gt = None

    def scan_at(self, pose, rng):
        ang = pose[2] + self.beam + rng.normal(0, S.ANGLE_JITTER,
                                               self.n_beams)
        r = S.raycast(self.segs, pose[:2], ang)
        r = np.where(r <= S.MAX_RANGE,
                     r + rng.normal(0, S.RANGE_NOISE, self.n_beams), np.inf)
        r[rng.random(self.n_beams) < S.DROPOUT] = np.inf
        return r

    def scan_of(self, i):
        """Pass-0 scan of kf i, REGENERATED (deterministic per-kf stream —
        the freemask carve's original inline derivation, verbatim)."""
        rng = np.random.default_rng(self.seed * 100003 + 0 * 7919 + i)
        return self.scan_at(self.gt_loop[i], rng)

    def __iter__(self):
        while True:
            p = self.k_global // self.n_loop      # pass number, 0-based
            i = self.k_global % self.n_loop
            rng = np.random.default_rng(
                self.seed * 100003 + p * 7919 + i)  # per-kf stream, per-pass
            gt = self.gt_loop[i]
            r = self.scan_at(gt, rng)
            if self._prev_gt is None:
                self.odom = gt.copy()
            else:
                d = L.se2_mul(L.se2_inv(self._prev_gt), gt)
                d[:2] += (rng.normal(0, DE.ODO_T, 2)
                          + 0.02 * np.abs(d[:2]) * rng.normal(0, 1, 2))
                d[2] += rng.normal(0, DE.ODO_R)
                self.odom = L.se2_mul(self.odom, d)
            self._prev_gt = gt.copy()
            yield dict(k=self.k_global, p=p, i=i, gt=gt.copy(),
                       odom=self.odom.copy(), r=r,
                       bridge=(i >= self.n_tour))
            self.k_global += 1


class DrivenFeed(Feed):
    """WASD-DRIVEN synthetic feed (live demo): the browser holds the key
    set (/drive endpoint), motion integrates at keyframe rate (full speed
    = the scripted tour's stride), scans raycast from the DRIVEN pose,
    odometry = true delta + Feed's noise model. Pass-1 scans are RECORDED
    for the freemask carve (scan_of); the map freezes on the UI 'freeze
    map' action (_req_freeze), which pins n_tour to the driven keyframe
    count. No wall collision (you can drive through walls — the scans
    are raycast from wherever you are)."""

    def __init__(self, seed=11, laps=2, n_beams=1024, env="classroom",
                 traj="driven"):
        self.seed, self.n_beams, self.env, self.traj = (seed, n_beams,
                                                        env, "driven")
        self.driven = True
        if env == "classroom":
            self.segs = DE.classroom_segs(None)
        else:
            import ssp_synth as SY
            self.segs = SY.WORLDS[env]()
        sa = np.asarray(self.segs, float).reshape(-1, 2)
        self.pose = np.array([sa[:, 0].mean(), sa[:, 1].mean(), 0.0])
        self._bbox = (sa[:, 0].min(), sa[:, 0].max(),
                      sa[:, 1].min(), sa[:, 1].max())
        self.keys = set()
        self.beam = -np.pi + np.arange(n_beams) * (2 * np.pi / n_beams)
        self.n_tour = self.n_loop = 10 ** 9        # until the freeze
        self.k_global = 0
        self.odom = None
        self._prev_gt = None
        self._scans = []                           # pass-1 recordings

    def set_keys(self, ks):
        self.keys = set(ks) & set("wasd")

    def _step_pose(self):
        dt = DE.STEP / DE.ROBOT_V
        v = (("w" in self.keys) - ("s" in self.keys)) * DE.ROBOT_V
        om = (("a" in self.keys) - ("d" in self.keys)) * 1.6
        self.pose[2] = wrap(self.pose[2] + om * dt)
        self.pose[0] += v * np.cos(self.pose[2]) * dt
        self.pose[1] += v * np.sin(self.pose[2]) * dt
        m = 0.3                                    # stay near the world
        self.pose[0] = np.clip(self.pose[0], self._bbox[0] - m,
                               self._bbox[1] + m)
        self.pose[1] = np.clip(self.pose[1], self._bbox[2] - m,
                               self._bbox[3] + m)

    def scan_of(self, i):
        return self._scans[i]

    def __iter__(self):
        while True:
            self._step_pose()
            rng = np.random.default_rng(self.seed * 100003 + self.k_global)
            gt = self.pose.copy()
            r = self.scan_at(gt, rng)
            if self.k_global < self.n_tour and len(self._scans) < 6000:
                self._scans.append(r.copy())
            if self._prev_gt is None:
                self.odom = gt.copy()
            else:
                d = L.se2_mul(L.se2_inv(self._prev_gt), gt)
                d[:2] += (rng.normal(0, DE.ODO_T, 2)
                          + 0.02 * np.abs(d[:2]) * rng.normal(0, 1, 2))
                d[2] += rng.normal(0, DE.ODO_R)
                self.odom = L.se2_mul(self.odom, d)
            self._prev_gt = gt.copy()
            yield dict(k=self.k_global,
                       p=0 if self.k_global < self.n_tour else 1,
                       i=self.k_global, gt=gt,
                       odom=self.odom.copy(), r=r, bridge=False)
            self.k_global += 1


class SpotFeed:
    """REAL-DATA feed: the SPOT Telluride tour (data/spot_telluride/
    scans.npz via ssp_spot.make_bundle — 1024 beams on the fabric's exact
    az grid, 5 Hz keyframes). LIDAR-ONLY POSTURE (anti-oracle): item
    ['odom'] is a ZERO-MOTION chain anchored at the reference's first
    pose, so the pass-1 guess degenerates to the previous estimate (the
    banked lidar-only spot runs chain CV from own estimates; at these
    keyframe strides the matcher window covers the difference). item
    ['gt'] carries the WITHHELD odometry reference — display/eval ONLY.

    LOOP SEMANTICS (mirrors the synthetic Feed): the end->start gap
    (measured 0.381 m + 24.1 deg on this tour) is closed by an
    INTERPOLATED BRIDGE. Real data cannot re-raycast, so each bridge
    keyframe replays the nearest ENDPOINT scan CIRCULARLY ROLLED to the
    interpolated heading — exact for the rotation part on a 360 x 1024
    head (the 24 deg snap dies); the residual position parallax is
    <= half the gap (~0.19 m) on ~9 of 414+9 keyframes. From the SECOND
    loop on, scans are PERTURBED per pass (range noise sigma =
    S.RANGE_NOISE + S.DROPOUT, deterministic per (pass, kf) — the
    synthetic feed's redraw analog), so localization passes never match
    the exact bytes that built the map. Pass 0 maps the pristine data."""

    def __init__(self, seed=11, laps=1, n_beams=1024, traj="orbit"):
        import ssp_spot as SP
        b = SP.make_bundle()
        self.seed = seed
        self.keys = b["keys"]
        self.beam = b["beam"]
        self.gt_ok = b["gt_ok"]
        self.segs = []                    # no analytic walls (real data)
        self.n_tour = len(self.keys)
        gt = np.array([k[1] for k in self.keys], float)
        # interpolated bridge last -> first (positions + shortest arc)
        gap = float(np.linalg.norm(gt[0, :2] - gt[-1, :2]))
        step = float(np.median(np.linalg.norm(np.diff(gt[:, :2], axis=0),
                                              axis=1)))
        nb = max(2, int(np.ceil(gap / max(step, 1e-3))))
        t = np.linspace(0, 1, nb + 2)[1:-1]
        bxy = gt[-1, :2] * (1 - t[:, None]) + gt[0, :2] * t[:, None]
        bth = wrap(gt[-1, 2] + t * wrap(gt[0, 2] - gt[-1, 2]))
        self.bridge = np.concatenate([bxy, bth[:, None]], 1)
        self.bridge_t = t
        self.n_loop = self.n_tour + nb
        self.k_global = 0
        self.odom0 = np.array(self.keys[0][1], float)   # boot datum only
        self._dbeam = 2 * np.pi / len(self.beam)

    def scan_of(self, i):
        """Pass-0 scan of loop index i (recorded; bridge = rolled copy)."""
        if i < self.n_tour:
            return np.asarray(self.keys[i][0], float)
        b = i - self.n_tour
        src = self.n_tour - 1 if self.bridge_t[b] < 0.5 else 0
        r = np.asarray(self.keys[src][0], float)
        th_s = float(self.keys[src][1][2])
        th_b = float(self.bridge[b, 2])
        m = int(np.round(wrap(th_b - th_s) / self._dbeam))
        return np.roll(r, -m)             # rotation-exact re-heading

    def _perturb(self, r, p, i):
        """Per-pass lidar redraw analog (p >= 1): range noise + dropout,
        deterministic per (pass, kf) — same stream law as Feed."""
        rng = np.random.default_rng(self.seed * 100003 + p * 7919 + i)
        r = r.copy()
        ok = np.isfinite(r)
        r[ok] += rng.normal(0, S.RANGE_NOISE, int(ok.sum()))
        r[rng.random(len(r)) < S.DROPOUT] = np.inf
        return r

    def __iter__(self):
        while True:
            p = self.k_global // self.n_loop
            i = self.k_global % self.n_loop
            r = self.scan_of(i)
            if p >= 1:
                r = self._perturb(r, p, i)
            if i < self.n_tour:
                gt = np.array(self.keys[i][1], float)
            else:
                gt = self.bridge[i - self.n_tour].copy()
            yield dict(k=self.k_global, p=p, i=i, gt=gt,
                       odom=self.odom0.copy(), r=r,
                       bridge=(i >= self.n_tour))
            self.k_global += 1


# ------------------------------------------------------------- fabric side
class Fabric:
    """Board wrapper: live encode + cross-check, M loads, localization
    sweeps, delta-probe map imaging. All numbers vs ssp_ice40 golden."""

    def __init__(self, port=None):
        self.b = Board(port)
        self.luts = G.make_luts()
        self.cur_aid = None
        self.n_enc = self.n_ok = self.n_bad = 0

    def encode(self, r):
        az, r_mm, w = G.scan_to_ints(r)
        if len(az) < 5:
            return None
        self.b.clear()
        self.b.stream_bulk(az, r_mm, w)
        # scene-change observable (VSA-native, GT-free): cosine of this
        # keyframe's int encode to the previous one — occlusion events
        # (doorways) show as sharp dips; logged for glitch analysis
        q = G.encode_int(az, r_mm, w, self.luts).astype(np.float64)
        v = q[:, 0] + 1j * q[:, 1]
        if getattr(self, "_prev_v", None) is not None:
            den = np.linalg.norm(v) * np.linalg.norm(self._prev_v) + 1e-12
            self.scene_cos = float(np.abs(np.vdot(v, self._prev_v)) / den)
        else:
            self.scene_cos = 1.0
        self._prev_v = v
        self._last_q = q.astype(np.int32)
        return (az, r_mm, w)

    def crosscheck(self, ints):
        gold = G.encode_int(*ints, self.luts)
        hw = self.b.readback()
        self.n_enc += 1
        if np.array_equal(hw, gold):
            self.n_ok += 1
            return True
        self.n_bad += 1
        return False

    def load_seg(self, aid, mcodes):
        if aid != self.cur_aid:
            self.b.load_m(mcodes)
            self.cur_aid = aid

    def bsweep(self, cands, ints=None, codes=None):
        """Batched totals sweep (v5 cmd 0x08, FIFO-buffered — no pacing
        invariant). On a short read (real desync) recover and retry once:
        re-encode the scan and reload M if the caller provides them."""
        try:
            return self.b.match_batch(cands, totals=True)
        except AssertionError:
            if ints is None:
                raise
            self.recover(ints)
            if codes is not None:
                self.b.load_m(codes)
                self.cur_aid = None
            return self.b.match_batch(cands, totals=True)

    def bsweep_full(self, cands, ints=None, codes=None):
        """Batched FULL sweep: per-ring (4,2) partials per candidate —
        the refined image readers need rings, not totals."""
        try:
            return self.b.match_batch(cands, totals=False)
        except AssertionError:
            if ints is None:
                raise
            self.recover(ints)
            if codes is not None:
                self.b.load_m(codes)
                self.cur_aid = None
            return self.b.match_batch(cands, totals=False)

    def recover(self, ints, fmap=None):
        """Full protocol recovery after a lost-reply sweep: resync the
        parser, then rebuild the fabric state exactly (clear + re-encode
        the keyframe scan wipes any padding-injected garbage point; M
        codes reload on the next load_seg)."""
        self.n_recover = getattr(self, "n_recover", 0) + 1
        self.b.resync()
        self.b.clear()
        self.b.stream_bulk(*ints)
        self.cur_aid = None
        print(f"[recover] protocol resynced (#{self.n_recover})",
              flush=True)


# ------------------------------------------------------ frozen 2b map + loc
class FrozenMap:
    def __init__(self, slam):
        self.aids = sorted(slam.segvec.keys())
        self.anchor = {a: slam.anchors[a].copy() for a in self.aids}
        self.codes = {a: G.mcode_from_vec(slam.segvec[a][L.MAIN])
                      for a in self.aids}
        self.axy = np.stack([self.anchor[a][:2] for a in self.aids])

    def nearest(self, xy, k=1):
        d = np.linalg.norm(self.axy - xy[None, :], axis=1)
        o = np.argsort(d)
        return [(self.aids[i], d[i]) for i in o[:k]]

    def rel(self, pose, aid):
        """(t, phi): pose expressed relative to the anchor frame."""
        a = self.anchor[aid]
        t = S._rot(-a[2]) @ (pose[:2] - a[:2])
        return t, wrap(pose[2] - a[2])


class FabricLoc:
    """Localization against the frozen 2b map. Sign conventions between the
    matcher contract and SE2 are pinned EMPIRICALLY by calibrate()."""

    def __init__(self, fab, fmap):
        self.fab, self.map = fab, fmap
        self.sgn_d = -1          # peak at D = -t (golden convention); pinned
        self.pose = None
        self.n_edge = 0          # re-centered sweeps (peak on grid edge)
        self.s_ema = None        # running scale of a healthy peak score
        self.lost = 0            # consecutive weak-peak keyframes
        self.state = "tracking"  # tracking | hold | relock

    def rho_of(self, phi):
        r = int(round(phi * 60 / np.pi))
        return r % 60

    def phi_from(self, rho, phi_pred):
        cands = [rho * np.pi / 60 + b for b in (-2 * np.pi, -np.pi, 0,
                                                np.pi, 2 * np.pi)]
        return min(cands, key=lambda c: abs(wrap(c - phi_pred)))

    def locate(self, q_ints, pred):
        """One fabric localization step around prediction `pred`.
        v2 (battery forensics): degeneracy-aware commits — corridor-class
        content gives confident WRONG peaks (along-axis slide + heading
        twist); observability is read off the score surface itself
        (rotation-axis flatness, per-axis curvature), GT-free. Plus
        two-segment handoff: near segment boundaries both neighbors are
        swept and the better peak wins (the 1.6-3x glitch lift at
        handoffs from the trajectory battery)."""
        sh = G.shift_for(G.encode_int(*q_ints, self.fab.luts))
        near = self.map.nearest(pred[:2], k=2)
        aid = near[0][0]
        if HANDOFF_M >= 0 and len(near) > 1 \
                and near[1][1] < near[0][1] + HANDOFF_M:
            r1 = self._locate_seg(q_ints, pred, near[0][0], sh)
            r2 = self._locate_seg(q_ints, pred, near[1][0], sh)
            return r1 if r1[2] >= r2[2] else r2
        return self._locate_seg(q_ints, pred, aid, sh)

    def _locate_seg(self, q_ints, pred, aid, sh):
        self.fab.load_seg(aid, self.map.codes[aid])
        t_pred, phi_pred = self.map.rel(pred, aid)
        d0 = self.sgn_d * np.round(t_pred / U).astype(int)
        rho0 = self.rho_of(phi_pred)
        for attempt in range(2):
            dU = d0[0] + SWEEP_T * np.arange(-SWEEP_N, SWEEP_N + 1)
            dV = d0[1] + SWEEP_T * np.arange(-SWEEP_N, SWEEP_N + 1)
            rhos = [rho0 + r for r in range(-SWEEP_R, SWEEP_R + 1)]
            cands = [(int(dx), int(dy), int(rho) % 60, sh)
                     for rho in rhos for dx in dU for dy in dV]
            if USE_ARGMAX:
                # fabric argmax (6-byte reply) + 3x3 refine batch — the
                # tot grid is materialized sparsely (argmax cell + its
                # in-plane neighbors); everything downstream unchanged
                try:
                    bi, bv = self.fab.b.match_argmax(cands)
                except AssertionError:
                    self.fab.recover(q_ints)
                    self.fab.load_seg(aid, self.map.codes[aid])
                    bi, bv = self.fab.b.match_argmax(cands)
                nU, nV = len(dU), len(dV)
                ir = bi // (nU * nV)
                ix = (bi % (nU * nV)) // nV
                iy = bi % nV
                tot = np.full((len(rhos), nU, nV), -np.inf)
                tot[ir, ix, iy] = float(bv)
                nbr = [(jx, jy) for jx in (ix - 1, ix, ix + 1)
                       for jy in (iy - 1, iy, iy + 1)
                       if 0 <= jx < nU and 0 <= jy < nV
                       and (jx, jy) != (ix, iy)]
                if nbr:
                    rc = [(int(dU[jx]), int(dV[jy]), int(rhos[ir]) % 60,
                           sh) for jx, jy in nbr]
                    vals = self.fab.bsweep(rc)
                    for (jx, jy), v in zip(nbr, vals):
                        tot[ir, jx, jy] = float(v)
            else:
                tot = self.fab.bsweep(cands, ints=q_ints,
                                      codes=self.map.codes[aid])
                tot = tot.astype(np.float64).reshape(
                    len(rhos), len(dU), len(dV))
                ir, ix, iy = np.unravel_index(np.argmax(tot), tot.shape)
            on_edge = ix in (0, len(dU) - 1) or iy in (0, len(dV) - 1)
            if not on_edge or attempt:
                break
            self.n_edge += 1
            d0 = np.array([dU[ix], dV[iy]])
            rho0 = rhos[ir] % 60
        # parabolic sub-grid refine (guarded) + observability readout
        px, py = float(dU[ix]), float(dV[iy])
        pk = float(tot[ir, ix, iy])
        fin = tot[np.isfinite(tot)]
        spread = max(float(fin.max() - np.median(fin)), 1e-9)
        den_x = den_y = 0.0
        if 0 < ix < len(dU) - 1:
            a, b, c = tot[ir, ix - 1, iy], tot[ir, ix, iy], tot[ir, ix + 1, iy]
            den_x = float(a - 2 * b + c)
            if den_x < 0:
                px += SWEEP_T * 0.5 * (a - c) / den_x
        if 0 < iy < len(dV) - 1:
            a, b, c = tot[ir, ix, iy - 1], tot[ir, ix, iy], tot[ir, ix, iy + 1]
            den_y = float(a - 2 * b + c)
            if den_y < 0:
                py += SWEEP_T * 0.5 * (a - c) / den_y
        # per-axis observability: flat curvature (relative to the
        # surface's own spread) = unobservable axis -> hold prediction
        obs_x = OBS_CURV <= 0 or -den_x > OBS_CURV * spread
        obs_y = OBS_CURV <= 0 or -den_y > OBS_CURV * spread
        d0f = self.sgn_d * np.array([t_pred[0], t_pred[1]]) / U
        if not obs_x:
            px = float(d0f[0])
        if not obs_y:
            py = float(d0f[1])
        # rotation observability: flatness of the per-rho maxima
        rho_prof = tot.reshape(len(rhos), -1).max(1)
        rho_flat = RHO_FLAT > 0 and \
            (rho_prof.max() - np.median(rho_prof)) < RHO_FLAT * spread
        self.diag = (float(-den_x / spread), float(-den_y / spread),
                     float((rho_prof.max() - np.median(rho_prof)) / spread))
        t_est = self.sgn_d * np.array([px, py]) * U
        if rho_flat:
            phi = phi_pred
        else:
            phi = self.phi_from(rhos[ir] % 60, phi_pred)
        a = self.map.anchor[aid]
        pose = np.array([*(a[:2] + S._rot(a[2]) @ t_est), wrap(a[2] + phi)])
        score = float(tot[ir, ix, iy])
        # pi-BRANCH AUDIT (host arithmetic only; real-data pivot finding,
        # spot kf-504 event): the half-circle lattice folds heading mod
        # pi and phi_from picks the branch nearest the prediction — a
        # sustained in-place pivot can walk the bookkeeping across the
        # boundary, after which the tracker CONFIDENTLY tracks the
        # antipodal interpretation (observed: fx heading settled ~167
        # deg wrong with state=tracking, so holds/relock never armed).
        # The conjugated query scores the antipodal branch exactly (the
        # conjugate-wrap physics distinguishes pi); a decisive win flips
        # phi back. Costs one python match_int per committed kf.
        if score > 0:
            if not hasattr(self, "_luts"):
                self._luts = G.make_luts()
            Q = G.encode_int(*q_ints, self._luts)
            Qc = Q.copy()
            Qc[:, 1] = -Qc[:, 1]
            cand = (int(dU[ix]), int(dV[iy]), int(rhos[ir]) % 60)
            s_0 = int(G.match_int(Q, self.map.codes[aid], *cand, sh,
                                  self._luts)[:, 0].sum())
            s_pi = int(G.match_int(Qc, self.map.codes[aid], *cand, sh,
                                   self._luts)[:, 0].sum())
            if s_pi > 1.15 * max(s_0, 1):
                pose[2] = wrap(pose[2] + np.pi)
                self.n_piflip = getattr(self, "n_piflip", 0) + 1
        # score-gated commit: a weak peak means the prediction left the
        # basin (or aliased geometry) — HOLD on odometry instead of
        # snapping, and periodically run a wide re-search around the
        # odometry pose. Fabric + odometry only; nothing leaks in from
        # the python SLAM or GT.
        if self.s_ema is None:
            self.s_ema = score
        if score > 0.45 * self.s_ema:
            self.s_ema = 0.98 * self.s_ema + 0.02 * score
            self.pose = pose
            self.lost = 0
            self.state = "tracking"
            return pose, aid, score
        self.lost += 1
        self.state = "hold"
        self.pose = pred                        # trust odometry this kf
        if self.lost % 12 == 0:
            rp, raid, rsc = self._research(pred, sh)
            if rsc is not None and rsc > 0.55 * self.s_ema:
                self.pose = rp
                self.lost = 0
                self.state = "relock"
                return rp, raid, rsc
        return pred, aid, score

    def _research(self, pred, sh):
        """Wide coarse re-search around the odometry pose: +-0.35 m at
        4.7 cm, rho GLOBAL-IN-HEADING (all 60 half-degree-lattice steps,
        stride 2) — ~6.8k candidates, ~0.9 s, occasional (every 12th
        hold). The +-6-step heading window lost sustained IN-PLACE PIVOTS
        on real data (spot kf-504 event: ~5.8 deg/kf for ~10 kf = ~55
        deg while position holds — fine reach +-6 deg, old re-search
        +-18 deg; the tracker then wandered off a confident wrong basin).
        Position stays local (the robot cannot teleport); heading can
        pivot arbitrarily while stopped — relock must be global in
        exactly that axis."""
        aid, _ = self.map.nearest(pred[:2])[0]
        self.fab.load_seg(aid, self.map.codes[aid])
        t_pred, phi_pred = self.map.rel(pred, aid)
        d0 = self.sgn_d * np.round(t_pred / U).astype(int)
        rho0 = self.rho_of(phi_pred)
        dU = d0[0] + 48 * np.arange(-10, 11)
        dV = d0[1] + 48 * np.arange(-10, 11)
        rhos = [rho0 + r for r in range(0, 60, 2)]
        cands = [(int(dx), int(dy), int(rho) % 60, sh)
                 for rho in rhos for dx in dU for dy in dV]
        tot = self.fab.bsweep(cands).astype(np.float64).reshape(
            len(rhos), len(dU), len(dV))
        ir, ix, iy = np.unravel_index(np.argmax(tot), tot.shape)
        t_est = self.sgn_d * np.array([dU[ix], dV[iy]], float) * U
        phi = self.phi_from(rhos[ir] % 60, phi_pred)
        a = self.map.anchor[aid]
        pose = np.array([*(a[:2] + S._rot(a[2]) @ t_est), wrap(a[2] + phi)])
        return pose, aid, float(tot[ir, ix, iy])


# --------------------------------------------------------------- map image
def _delta_probe(fab):
    """Encode the delta probe (one point at the origin) — rotation-
    invariant on this lattice, so score(D) is the matched-filter image."""
    fab.b.clear()
    fab.b.stream_bulk(np.array([0], np.int32), np.array([0], np.int32),
                      np.array([127], np.int32))
    fab.cur_aid = None                    # force M reload bookkeeping


def probe_image(fab, fmap, bounds, progress=None):
    """Fabric matched-filter image of the frozen map on a world grid;
    every pixel is probed against its NEAREST segment (vectorized
    assignment), chunked into paced sweeps."""
    x0, x1, y0, y1 = bounds
    xs = np.arange(x0, x1 + 1e-9, IMG_STEP)
    ys = np.arange(y0, y1 + 1e-9, IMG_STEP)
    gx, gy = np.meshgrid(xs, ys)
    pix = np.stack([gx.ravel(), gy.ravel()], 1)          # (P, 2)
    d2 = ((pix[:, None, :] - fmap.axy[None, :, :]) ** 2).sum(2)
    owner = np.argmin(d2, 1)
    dist = np.sqrt(d2[np.arange(len(pix)), owner])
    img = np.full(len(pix), np.nan)
    m3 = np.full(len(pix), np.nan)
    _delta_probe(fab)
    total = int((dist <= IMG_RAD).sum())
    done = 0
    for oi, aid in enumerate(fmap.aids):
        sel = np.flatnonzero((owner == oi) & (dist <= IMG_RAD_MAP))
        if not len(sel):
            continue
        a = fmap.anchor[aid]
        t = (pix[sel] - a[:2]) @ S._rot(-a[2]).T
        d = (-np.round(t / U)).astype(int)               # golden: peak at -t
        keep = np.abs(d).max(1) < 32000
        sel, d = sel[keep], d[keep]
        fab.load_seg(aid, fmap.codes[aid])
        for c0 in range(0, len(sel), 1024):
            sl = slice(c0, min(c0 + 1024, len(sel)))
            cands = [(int(dx), int(dy), 0, 0) for dx, dy in d[sl]]
            fr = fab.bsweep_full(cands)
            img[sel[sl]] = fr[:, :, 0].sum(1)
            m3[sel[sl]] = fr[:, 1:, 0].min(1)   # min of rings 1..3
            done += len(cands)
            if progress:
                progress(done, total)
    img = img.reshape(len(ys), len(xs))
    m3 = m3.reshape(len(ys), len(xs))

    def _norm(a):
        lo, hi = np.nanpercentile(a, [5, 99.5])
        z = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
        return np.where(np.isnan(a), 0.0, z)
    disp, disp3 = _norm(img), _norm(m3)
    return dict(x0=float(xs[0]), y0=float(ys[0]), step=IMG_STEP,
                w=len(xs), h=len(ys),
                img=[[round(float(v), 3) for v in rw] for rw in disp],
                min3=[[round(float(v), 3) for v in rw] for rw in disp3])


def query_viewpoint_direct(fab, fmap, p, n_rays=120, r_max=5.0,
                           r_step=0.10, frac=0.62, rad=2.4,
                           freemask=None, trail=None):
    """Viewpoint query DIRECTLY from phasor space (no image cache): for
    each ray direction, march range samples where each sample is one
    fabric matched-filter readout of the stored 2-bit map at that world
    point (delta probe; viewpoint translation is exact in phase). The
    only non-phasor step is the FIRST-RETURN selection along each ray —
    occlusion is a nonlinear choice the linear algebra cannot express.
    Probes the <=2 nearest segments per sample, max-combined."""
    p = np.asarray(p, float)
    ths = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
    rs = np.arange(0.25, r_max, r_step)
    pts = p[None, None, :] + rs[None, :, None] * np.stack(
        [np.cos(ths), np.sin(ths)], 1)[:, None, :]       # (rays, R, 2)
    P = pts.reshape(-1, 2)
    d2 = ((P[:, None, :] - fmap.axy[None, :, :]) ** 2).sum(2)
    order = np.argsort(d2, 1)[:, :2]
    val = np.full((len(P), 2), np.nan)
    _delta_probe(fab)
    for col in range(2):
        for oi in np.unique(order[:, col]):
            sel = np.flatnonzero(order[:, col] == oi)
            aid = fmap.aids[oi]
            a = fmap.anchor[aid]
            near = np.sqrt(d2[sel, oi]) <= rad
            sel = sel[near]
            if not len(sel):
                continue
            t = (P[sel] - a[:2]) @ S._rot(-a[2]).T
            d = (-np.round(t / U)).astype(int)
            keep = np.abs(d).max(1) < 32000
            sel, d = sel[keep], d[keep]
            fab.load_seg(aid, fmap.codes[aid])
            for c0 in range(0, len(sel), 1024):
                sl = slice(c0, min(c0 + 1024, len(sel)))
                cands = [(int(dx), int(dy), 0, 0) for dx, dy in d[sl]]
                val[sel[sl], col] = fab.bsweep(cands)
    v = np.nanmax(val, 1).reshape(len(ths), len(rs))
    v = np.where(np.isnan(v), -np.inf, v)
    fin = v[np.isfinite(v)]
    if not len(fin):
        return []
    print(f"[direct] coverage {len(fin)}/{v.size} samples "
          f"({len(fin) / v.size:.0%})", flush=True)
    lo, hi = np.percentile(fin, [10, 99.5])
    floor = lo + 0.30 * (hi - lo)
    prom = 0.10 * (hi - lo)
    hits = []
    refine = []                         # (ray idx, r at peak)
    # peak detection + veto ported VERBATIM from the readout study's
    # winning "(sum, mask)" rule (scratch_readout.py): scipy find_peaks
    # on a [.25,.5,.25]-smoothed profile padded BELOW at the ends
    # (coverage-truncated wall ridges still register), whole-signal
    # prominence; veto = own-trajectory corridor (0.35 m) + local convex
    # hull (3 m) + the pass-1 free-space mask; first surviving peak
    # >= 0.55 x surviving max.
    from scipy.signal import find_peaks
    kd_tri = None
    if trail is not None and len(trail):
        from scipy.spatial import Delaunay, cKDTree
        kd = cKDTree(trail)
        loc = trail[np.linalg.norm(trail - p[None, :], axis=1) <= 3.0]
        tri = None
        if len(loc) >= 4:
            try:
                tri = Delaunay(loc)
            except Exception:
                tri = None
        kd_tri = (kd, tri)

    vdbg = dict(mask=0, corr=0, hull=0)

    def _veto(g):
        if freemask is not None and _mask_free(freemask, g):
            vdbg["mask"] += 1
            return True
        if kd_tri is not None:
            kd, tri = kd_tri
            if kd.query(g)[0] <= 0.35:
                vdbg["corr"] += 1
                return True
            if tri is not None and tri.find_simplex(g) >= 0:
                vdbg["hull"] += 1
                return True
        return False

    dbg = dict(rays=0, weak=0, no_peak=0, vetoed=0, surv=0)
    for i in range(len(ths)):
        ray = v[i]
        dbg["rays"] += 1
        mx = ray.max()
        if not np.isfinite(mx) or mx < floor:
            dbg["weak"] += 1
            continue
        ray3 = np.where(np.isfinite(ray), ray, lo)
        sm = np.convolve(np.r_[ray3[0], ray3, ray3[-1]],
                         [0.25, 0.5, 0.25], "valid")
        pk, _ = find_peaks(np.r_[lo - 1, sm, lo - 1], height=floor,
                           prominence=prom)
        pk = pk - 1
        if not len(pk):
            dbg["no_peak"] += 1
            continue
        uv = np.array([np.cos(ths[i]), np.sin(ths[i])])
        surv = [j for j in pk if not _veto(p + rs[min(int(j),
                len(rs) - 1)] * uv)]
        if not surv:
            dbg["vetoed"] += 1
            if freemask is not None:      # maskfb tier: free->unfree edge
                inside = False
                for rq in np.arange(0.25, rs[-1], 0.05):
                    g = p + rq * uv
                    if _mask_free(freemask, g):
                        inside = True
                    elif inside:
                        refine.append((i, float(rq)))
                        dbg["fb"] = dbg.get("fb", 0) + 1
                        break
            continue
        dbg["surv"] += 1
        mxs = max(sm[min(int(j), len(sm) - 1)] for j in surv)
        for j in surv:
            jj = min(int(j), len(sm) - 1)
            if sm[jj] >= 0.55 * mxs:
                rr = rs[min(jj, len(rs) - 1)]
                if 0 < jj < len(rs) - 1:
                    a, b, c = sm[jj - 1], sm[jj], sm[jj + 1]
                    den = a - 2 * b + c
                    if den < 0:
                        rr += (rs[1] - rs[0]) * 0.5 * (a - c) / den
                refine.append((i, float(rr)))
                break
    print(f"[direct] rays {dbg['rays']} weak {dbg['weak']} no-peak "
          f"{dbg['no_peak']} vetoed {dbg['vetoed']} surviving "
          f"{dbg['surv']} fb {dbg.get('fb', 0)} | veto by mask "
          f"{vdbg['mask']} corr {vdbg['corr']} hull {vdbg['hull']}",
          flush=True)
    import os
    if os.environ.get("DQ_DUMP"):
        np.savez(os.environ["DQ_DUMP"], ths=ths, rs=rs, v=v, p=p,
                 mask_free=freemask["free"] if freemask else np.zeros(0),
                 mask_x0=freemask["x0"] if freemask else 0.0,
                 mask_y0=freemask["y0"] if freemask else 0.0,
                 mask_cell=freemask["cell"] if freemask else 0.0,
                 trail=trail if trail is not None else np.zeros((0, 2)),
                 lo=lo, hi=hi, floor=floor, prom=prom)
    # fabric refine: +-0.12 m @ 0.02 m along each surviving ray
    for i, r0 in refine:
        u = np.array([np.cos(ths[i]), np.sin(ths[i])])
        cand_r = r0 + np.arange(-0.12, 0.121, 0.02)
        pts = p[None, :] + cand_r[:, None] * u[None, :]
        best, bv = None, -np.inf
        d2r = ((pts[:, None, :] - fmap.axy[None, :, :]) ** 2).sum(2)
        oi = int(np.argmin(d2r.min(0)))
        aid = fmap.aids[oi]
        a = fmap.anchor[aid]
        t = (pts - a[:2]) @ S._rot(-a[2]).T
        d = (-np.round(t / U)).astype(int)
        keep = np.abs(d).max(1) < 32000
        if not keep.any():
            continue
        fab.load_seg(aid, fmap.codes[aid])
        vals = fab.bsweep([(int(dx), int(dy), 0, 0)
                           for dx, dy in d[keep]])
        kk = np.flatnonzero(keep)
        j = int(np.argmax(vals))
        g = pts[kk[j]]
        hits.append([round(float(g[0]), 3), round(float(g[1]), 3)])
    return hits


def visible_walls(imgd, p, n_rays=180, thresh=0.50, freemask=None):
    """Host ray-march on the fabric image: first ridge crossing per ray,
    with the free-space mask vetoing interior ghost crossings (marching
    the raw min3 layer without the veto re-admits the 2 m combs —
    measured 2.1 m p50 vs GT; sum layer + mask gives wall-true hits)."""
    im = np.array(imgd["img"])
    hits = []
    for th in np.linspace(-np.pi, np.pi, n_rays, endpoint=False):
        d, step = np.array([np.cos(th), np.sin(th)]), imgd["step"] * 0.5
        for s in np.arange(0.2, 6.0, step):
            g = p + d * s
            if freemask is not None and _mask_free(freemask, g):
                continue
            ix = (g[0] - imgd["x0"]) / imgd["step"]
            iy = (g[1] - imgd["y0"]) / imgd["step"]
            if not (0 <= ix < imgd["w"] - 1 and 0 <= iy < imgd["h"] - 1):
                break
            fx, fy = ix - int(ix), iy - int(iy)
            i0, j0 = int(iy), int(ix)
            v = (im[i0, j0] * (1 - fx) * (1 - fy)
                 + im[i0, j0 + 1] * fx * (1 - fy)
                 + im[i0 + 1, j0] * (1 - fx) * fy
                 + im[i0 + 1, j0 + 1] * fx * fy)
            if v > thresh:
                hits.append([round(float(g[0]), 3), round(float(g[1]), 3)])
                break
    return hits


def carve_freemask(feed, poses, bounds, cell=0.30, stop=0.45):
    """Self-certified free-space mask: ray-carve from the system's OWN
    pass-1 scans at its OWN estimated poses (GT never enters). O(area),
    ~0.1-1 KB. Vetoes the 2 m octave ghost-comb in first-return queries
    (the readout-study mechanism)."""
    x0, x1, y0, y1 = bounds
    w = int(np.ceil((x1 - x0) / cell)) + 1
    h = int(np.ceil((y1 - y0) / cell)) + 1
    free = np.zeros((h, w), bool)
    hitpts = []
    for i, pose in enumerate(poses):
        r = feed.scan_of(i)      # pass-0 scan (synthetic: regenerated
        ok = np.isfinite(r)      # deterministically; real data: recorded)
        rr = np.clip(r[ok] - stop, 0.0, None)
        ang = pose[2] + feed.beam[ok]
        for t in np.arange(0.15, 1.0 + 1e-9, 0.12):
            pts = pose[:2] + (t * rr)[:, None] * np.stack(
                [np.cos(ang), np.sin(ang)], 1)
            ix = ((pts[:, 0] - x0) / cell).astype(int)
            iy = ((pts[:, 1] - y0) / cell).astype(int)
            m = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            free[iy[m], ix[m]] = True
        hp = pose[:2] + r[ok][:, None] * np.stack(
            [np.cos(ang), np.sin(ang)], 1)
        hitpts.append(hp[::4])
    # un-free every cell whose CENTER is near an observed hit — rays
    # passing tangentially near a wall otherwise bleed 'free' into
    # wall cells and veto true wall peaks (measured: 22/38 wrongly
    # vetoed before this pass)
    hp = np.concatenate(hitpts)
    iy, ix = np.nonzero(free)
    cx = x0 + (ix + 0.5) * cell
    cy = y0 + (iy + 0.5) * cell
    cc = np.stack([cx, cy], 1)
    # chunked nearest-hit distance (hp can be ~100k points)
    step = 4000
    near = np.zeros(len(cc), bool)
    for c0 in range(0, len(cc), step):
        blk = cc[c0:c0 + step]
        dmin = np.full(len(blk), np.inf)
        for h0 in range(0, len(hp), 20000):
            d = np.linalg.norm(blk[:, None, :] - hp[None, h0:h0 + 20000, :],
                               axis=2).min(1)
            dmin = np.minimum(dmin, d)
        near[c0:c0 + step] = dmin < 0.35
    free[iy[near], ix[near]] = False
    return dict(x0=x0, y0=y0, cell=cell, free=free)


def _mask_free(mask, g):
    ix = int((g[0] - mask["x0"]) / mask["cell"])
    iy = int((g[1] - mask["y0"]) / mask["cell"])
    f = mask["free"]
    return bool(0 <= ix < f.shape[1] and 0 <= iy < f.shape[0]
                and f[iy, ix])


# ------------------------------------------------------------------ system
class Live:
    def __init__(self, laps=2, seed=11, port=None, env="classroom",
                 traj="orbit"):
        self.feed = (SpotFeed(seed=seed, laps=laps) if env == "spot"
                     else Feed(seed=seed, laps=laps, env=env, traj=traj))
        self.slam = F.BandSLAM(robust=True, attempt_every=4, relax_every=25,
                               gap_kf=300, recent_aids=12, spec=None, nph=0)
        self.slam.store_dtype = np.complex64
        self.fab = Fabric(port)
        self.fmap = None
        self.loc = None
        self.imgd = None
        self.state = {}
        self.lock = threading.Lock()
        self.serlock = threading.RLock()   # one board, two threads (/pick)
        self.clients = []
        self.est_prev = None
        self.bench_log = []
        self.trail_py = []
        self.trail_fx = []
        self.trail_gt = []
        self.status = "pass 1: mapping (python SLAM + fabric encode)"

    def reset_for(self, env=None, traj="orbit", laps=2, seed=11):
        """UI 'reset map' / env switch: fresh feed + python SLAM + frozen
        map + tracker state; the BOARD stays (per-scan encode clears it).
        Called from the run loop between keyframes (handler threads only
        set _req_reset/_req_env)."""
        if env is not None:
            drive = bool(getattr(self, "_req_drive", False))
            self.feed = (SpotFeed(seed=seed, laps=laps) if env == "spot"
                         else DrivenFeed(seed=seed, env=env) if drive
                         else Feed(seed=seed, laps=laps, env=env,
                                   traj=traj))
        else:
            f = self.feed
            self.feed = (SpotFeed() if isinstance(f, SpotFeed)
                         else DrivenFeed(seed=f.seed, env=f.env)
                         if isinstance(f, DrivenFeed)
                         else Feed(seed=f.seed, laps=2, env=f.env,
                                   traj=f.traj))
        self._req_freeze = False
        self.slam = F.BandSLAM(robust=True, attempt_every=4,
                               relax_every=25, gap_kf=300, recent_aids=12,
                               spec=None, nph=0)
        self.slam.store_dtype = np.complex64
        self.fmap = None
        self.loc = None
        self.imgd = None
        self.est_prev = None
        self._odom_prev2 = None
        self._fx_prev = None
        self._fx_vel = np.zeros(3)
        for a in ("freemask", "fm_trail"):
            if hasattr(self, a):
                delattr(self, a)
        self.bench_log.clear()
        with self.lock:
            self.trail_py.clear()
            self.trail_fx.clear()
            self.trail_gt.clear()
        self.status = "pass 1: mapping (python SLAM + fabric encode)"
        self.broadcast(dict(reset=True))

    # ---- pass-1 python SLAM step (bench config, bridge2 sampler)
    def slam_step(self, item):
        r = item["r"]
        rr = np.where((r > 0.05) & (r < S.MAX_RANGE - 0.05), r, np.inf)
        pts, w = DE._bridge2(rr, self.feed.beam)
        k = item["k"]
        if k == 0:
            guess = item["odom"].copy()
        else:
            guess = L.se2_mul(self.est_prev, L.se2_mul(
                L.se2_inv(self._odom_prev), item["odom"]))
        est = self.slam.add_keyframe(pts, w, guess)
        self.est_prev, self._odom_prev = est.copy(), item["odom"].copy()
        return est

    def freeze(self):
        self.fmap = FrozenMap(self.slam)
        self.loc = FabricLoc(self.fab, self.fmap)
        print(f"[freeze] {len(self.fmap.aids)} segments -> 2b codes "
              f"({len(self.fmap.aids) * 60} B)", flush=True)

    def calibrate(self, items):
        """Pin the matcher<->SE2 mapping on known pass-1 poses; hard-fail.
        Uses post-relax poses (pose_of), not the online estimates."""
        ok = 0
        for item, _ in items:
            est = self.slam.pose_of(item["k"])
            ints = self.fab.encode(item["r"])
            if ints is None:
                continue
            pose, aid, _ = self.loc.locate(ints, est)
            err = np.linalg.norm(pose[:2] - est[:2])
            derr = abs(wrap(pose[2] - est[2]))
            print(f"[calib] kf {item['k']}: |dxy| {err * 100:.1f} cm  "
                  f"|dth| {np.rad2deg(derr):.2f} deg  (aid {aid})",
                  flush=True)
            ok += (err < 0.08 and derr < np.deg2rad(4.0))
        need = max(1, len(items) - 1) if len(items) < 2 \
            else max(2, len(items) - 1)
        assert ok >= need, \
            f"calibration failed ({ok}/{len(items)}) — geometry mapping wrong"
        print(f"[calib] OK {ok}/{len(items)} (convention sgn_d=-1)",
              flush=True)

    # ---- SSE
    def broadcast(self, obj):
        dead = []
        for q in self.clients:
            try:
                q.put_nowait(obj)
            except queue.Full:
                dead.append(q)
        for q in dead:
            self.clients.remove(q)

    def push_state(self, item, est, fx_pose, score, ms):
        gt = item["gt"]
        st = dict(
            k=item["k"], pss=item["p"] + 1, i=item["i"],
            bridge=bool(item["bridge"]), ms=round(ms, 1),
            gt=[round(float(v), 4) for v in gt],
            py=[round(float(v), 4) for v in est],
            fx=None if fx_pose is None else
            [round(float(v), 4) for v in fx_pose],
            err_py=round(float(np.linalg.norm(est[:2] - gt[:2])), 4),
            err_fx=None if fx_pose is None else
            round(float(np.linalg.norm(fx_pose[:2] - gt[:2])), 4),
            score=None if score is None else round(score / 1e6, 2),
            enc=f"{self.fab.n_ok}/{self.fab.n_enc}",
            enc_bad=self.fab.n_bad,
            segs=len(self.slam.segvec), mem=round(self.slam.memory_kb(), 1),
            env=("spot" if isinstance(self.feed, SpotFeed)
                 else self.feed.env),
            driven=bool(getattr(self.feed, "driven", False)),
            frozen=self.fmap is not None,
            edge=0 if self.loc is None else self.loc.n_edge,
            fx_state=None if self.loc is None else self.loc.state,
            status=self.status)
        st_code = {"tracking": 0, "hold": 1, "relock": 2}.get(
            None if self.loc is None else self.loc.state, -1)
        self.bench_log.append((
            item["k"], gt.copy(), est.copy(),
            None if fx_pose is None else fx_pose.copy(),
            st_code, -1.0 if score is None else float(score),
            bool(item["bridge"]), float(ms),
            float(getattr(self.fab, "scene_cos", 1.0)),
            -1 if self.fab.cur_aid is None else int(self.fab.cur_aid),
            getattr(self.loc, "diag", (np.nan,) * 3) if self.loc else
            (np.nan,) * 3))
        if len(self.bench_log) > 12000:
            del self.bench_log[:6000]
        with self.lock:
            self.state = st
            self.trail_py.append(st["py"][:2])
            self.trail_gt.append(st["gt"][:2])
            if st["fx"]:
                self.trail_fx.append(st["fx"][:2])
            for tr in (self.trail_py, self.trail_gt, self.trail_fx):
                if len(tr) > 4000:
                    del tr[:len(tr) - 4000]
        self.broadcast(st)

    # ---- main loop
    def run(self, realtime=True, stop_after=None, image=True):
        dt_kf = DE.STEP / DE.ROBOT_V
        self._req_reset = False
        self._req_env = None
        while True:
            calib_items = []
            t_next = time.perf_counter()
            for item in self.feed:
                if self._req_reset or self._req_env:
                    with self.serlock:
                        self.reset_for(self._req_env)
                    self._req_reset = False
                    self._req_env = None
                    break         # fresh feed iterator
                t0 = time.perf_counter()
                est = self.slam_step(item)
                with self.serlock:
                    ints = self.fab.encode(item["r"])
                    fx_pose = score = None
                    if ints is not None and item["k"] % CHECK_EVERY == 0:
                        self.fab.crosscheck(ints)   # readback is non-destructive
                    if self.fmap is None:
                        driven = getattr(self.feed, "driven", False)
                        if item["p"] == 0 and ints is not None and (
                                (not driven and item["i"] in (
                                    self.feed.n_tour // 3,
                                    2 * self.feed.n_tour // 3,
                                    self.feed.n_tour - 2))
                                or (driven and item["i"] % 60 == 30)):
                            calib_items.append((item, est.copy()))
                            calib_items = calib_items[-3:]
                        if getattr(self, "_req_freeze", False) and driven:
                            # driven mode: the UI freeze pins the tour
                            self.feed.n_tour = item["i"] + 1
                            self.feed.n_loop = item["i"] + 1
                            self._req_freeze = False
                            if ints is not None:
                                calib_items.append((item, est.copy()))
                                calib_items = calib_items[-3:]
                        if item["i"] == self.feed.n_tour - 1:  # tour end: freeze
                            if self.slam.dirty:
                                self.slam.relax()
                            self.freeze()
                            self.status = "calibrating matcher geometry"
                            try:
                                self.calibrate(calib_items)
                            except AssertionError as e:
                                # demo server must NOT die: report loudly
                                # and remap instead (the board session and
                                # UI stay alive)
                                print(f"[calib] FAILED — remapping: {e}",
                                      flush=True)
                                self.status = ("calibration FAILED — "
                                               "remapping")
                                with self.serlock:
                                    self.reset_for(None)
                                break
                            if image:
                                self.status = "probing map image from fabric"
                                self.probe()
                            self.loc.pose = est.copy()
                            self.status = ("looping: python SLAM + fabric "
                                           "localization")
                    elif ints is not None:
                        pred = (self.loc.pose if self.loc.pose is not None
                                else est)
                        if getattr(self, "_odom_prev2", None) is not None:
                            d = L.se2_mul(L.se2_inv(self._odom_prev2),
                                          item["odom"])
                            if (np.abs(d).max() < 1e-12
                                    and self.loc.pose is not None):
                                # LIDAR-ONLY feeds (SpotFeed: odom deltas are
                                # EXACTLY zero): constant-velocity pred from
                                # the tracker's OWN history — the banked spot
                                # posture. The velocity estimate updates ONLY
                                # from tracking-state steps and DECAYS through
                                # holds (translation 0.5x/kf — stops are
                                # common; heading 0.9x/kf — pivots are
                                # sustained). Without the decay, holds feed
                                # the velocity back to itself and the pred
                                # runs away at exactly the clamp rate
                                # (measured: the kf-504 stop-and-pivot event,
                                # +0.13 m/kf through 8 holds, then a
                                # confident wrong relock 1.8 m off).
                                if not hasattr(self, "_fx_vel"):
                                    self._fx_vel = np.zeros(3)
                                if (self.loc.state == "tracking"
                                        and getattr(self, "_fx_prev", None)
                                        is not None):
                                    v = L.se2_mul(L.se2_inv(self._fx_prev),
                                                  self.loc.pose)
                                    n = float(np.hypot(v[0], v[1]))
                                    if n > 0.13:
                                        v[:2] *= 0.13 / n
                                    self._fx_vel = v
                                else:
                                    # FREEZE on holds: score collapses are
                                    # scene-driven and coincide with stops
                                    # (kf-504: score 124->46 at 3 cm error);
                                    # any residual velocity slides the pred
                                    # out of the re-search reach before the
                                    # scene recovers (measured 0.6 m in 8
                                    # holds with 0.5x decay).
                                    self._fx_vel = np.zeros(3)
                                d = self._fx_vel
                            pred = L.se2_mul(pred, d)
                        self._fx_prev = (None if self.loc.pose is None
                                         else self.loc.pose.copy())
                        fx_pose, aid, score = self.loc.locate(ints, pred)
                self._odom_prev2 = item["odom"].copy()
                ms = (time.perf_counter() - t0) * 1e3
                self.push_state(item, est, fx_pose, score, ms)
                if item["k"] % 25 == 0:
                    st = self.state
                    print(f"[kf {st['k']:5d} pass {st['pss']}] "
                          f"py {st['err_py']:.3f} m  "
                          f"fx {st['err_fx'] if st['err_fx'] is not None else '-'}"
                          f"  enc {st['enc']}  {st['ms']:.0f} ms", flush=True)
                if stop_after is not None and item["k"] >= stop_after:
                    return
                if realtime:
                    t_next += dt_kf
                    pause = t_next - time.perf_counter()
                    if pause > 0:
                        time.sleep(pause)
                    else:
                        t_next = time.perf_counter()

    def probe(self):
        g = np.array(self.trail_gt)
        bounds = (g[:, 0].min() - 4, g[:, 0].max() + 4,
                  g[:, 1].min() - 4, g[:, 1].max() + 4)

        def prog(done, total):
            if done % 500 < 40:
                self.broadcast(dict(probe=f"{done}/{total}"))
                print(f"[image] {done}/{total} pixels", flush=True)
        t0 = time.time()
        poses = [self.slam.pose_of(i) for i in range(self.feed.n_tour)]
        self.freemask = carve_freemask(self.feed, poses, bounds)
        self.fm_trail = np.array([q[:2] for q in poses])[::3]
        print(f"[mask] free-space mask "
              f"{int(self.freemask['free'].sum())} cells", flush=True)
        self.imgd = probe_image(self.fab, self.fmap, bounds, prog)
        print(f"[image] fabric map image {self.imgd['w']}x{self.imgd['h']} "
              f"in {time.time() - t0:.1f} s", flush=True)
        self.broadcast(dict(map_ready=True))


# --------------------------------------------------------------- http/sse
def make_handler(live):
    html = (Path(__file__).parent / "live.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
            elif self.path == "/state":
                with live.lock:
                    self._json(live.state)
            elif self.path == "/map":
                self._json(live.imgd or dict(pending=True))
            elif self.path.startswith("/reset"):
                live._req_reset = True
                self._json(dict(ok=True))
            elif self.path.startswith("/drive"):
                from urllib.parse import parse_qs, urlparse
                q = parse_qs(urlparse(self.path).query)
                ks = q.get("keys", [""])[0]
                if hasattr(live.feed, "set_keys"):
                    live.feed.set_keys(ks)
                self._json(dict(ok=True))
            elif self.path.startswith("/freeze"):
                live._req_freeze = True
                self._json(dict(ok=True))
            elif self.path.startswith("/env"):
                from urllib.parse import parse_qs, urlparse
                q = parse_qs(urlparse(self.path).query)
                name = q.get("name", ["classroom"])[0]
                live._req_drive = q.get("drive", ["0"])[0] == "1"
                if name in ("classroom", "mixed", "corridor", "office",
                            "spot"):
                    live._req_env = name
                    self._json(dict(ok=True, env=name))
                else:
                    self._json(dict(ok=False, err="unknown env"))
            elif self.path == "/world":
                # ground-truth wall segments (display overlay only)
                segs = np.asarray(live.feed.segs, float)
                self._json(dict(segs=[[round(float(a), 3) for a in
                                       sg.reshape(-1)] for sg in segs]))
            elif self.path.startswith("/trails"):
                with live.lock:
                    self._json(dict(py=live.trail_py, fx=live.trail_fx,
                                    gt=live.trail_gt))
            elif self.path.startswith("/pick"):
                from urllib.parse import parse_qs, urlparse
                q = parse_qs(urlparse(self.path).query)
                p = np.array([float(q["x"][0]), float(q["y"][0])])
                if q.get("mode", [""])[0] == "direct":
                    if live.fmap is None:
                        self._json(dict(hits=[], mode="direct"))
                        return
                    live.broadcast(dict(probe="direct viewpoint query "
                                              "(replay paused)"))
                    t0 = time.time()
                    with live.serlock:
                        hits = query_viewpoint_direct(
                            live.fab, live.fmap, p,
                            freemask=getattr(live, "freemask", None),
                            trail=getattr(live, "fm_trail", None))
                    self._json(dict(hits=hits, mode="direct",
                                    secs=round(time.time() - t0, 1)))
                elif live.imgd is None:
                    self._json(dict(hits=[]))
                else:
                    self._json(dict(
                        hits=visible_walls(
                            live.imgd, p,
                            freemask=getattr(live, "freemask", None)),
                        mode="cached"))
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                qq = queue.Queue(maxsize=64)
                live.clients.append(qq)
                try:
                    while True:
                        obj = qq.get()
                        self.wfile.write(
                            f"data: {json.dumps(obj)}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    if qq in live.clients:
                        live.clients.remove(qq)
            else:
                self.send_error(404)

    return H


def serve(port=8642, laps=2, env="classroom", traj="orbit"):
    live = Live(laps=int(laps), env=env, traj=traj)
    srv = ThreadingHTTPServer(("127.0.0.1", int(port)), make_handler(live))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[live] http://127.0.0.1:{port}/  (tour {live.feed.n_tour} kf + "
          f"bridge {live.feed.n_loop - live.feed.n_tour} kf, 5 Hz)",
          flush=True)
    live.run(realtime=True)


def bench(env="mixed", passes=2, laps=2, traj="orbit"):
    """Headless synth hw-in-the-loop bench: pass-1 maps (python SLAM +
    fabric encode cross-check), then `passes` perturbed localization
    passes with the fabric tracking the frozen 2b map. Numbers only;
    exact GT (same frame by construction — no alignment)."""
    passes, laps = int(passes), int(laps)
    live = Live(laps=laps, env=env, traj=traj)
    n_stop = live.feed.n_loop * (1 + passes) + 5
    print(f"[bench] env={env} traj={traj} tour {live.feed.n_tour} + "
          f"bridge {live.feed.n_loop - live.feed.n_tour} kf, "
          f"passes {passes}", flush=True)
    live.run(realtime=False, stop_after=n_stop, image=False)
    log = live.bench_log
    last = [r for r in log if r[0] >= live.feed.n_loop * passes]
    gt = np.array([r[1] for r in last])
    py = np.array([r[2] for r in last])
    e_py = np.linalg.norm(py[:, :2] - gt[:, :2], axis=1)
    fx_rows = [(r[1], r[3]) for r in last if r[3] is not None]
    e_fx = np.array([np.linalg.norm(f[:2] - g[:2]) for g, f in fx_rows])
    e_th = np.array([abs(wrap(f[2] - g[2])) for g, f in fx_rows])
    rec = getattr(live.fab, "n_recover", 0)
    rows = live.bench_log
    np.savez(str(ROOT / f"scratch_livebench_{env}_{traj}.npz"),
             k=np.array([r[0] for r in rows]),
             gt=np.stack([r[1] for r in rows]),
             py=np.stack([r[2] for r in rows]),
             fx=np.stack([r[3] if r[3] is not None
                          else np.full(3, np.nan) for r in rows]),
             state=np.array([r[4] for r in rows]),
             score=np.array([r[5] for r in rows]),
             bridge=np.array([r[6] for r in rows]),
             ms=np.array([r[7] for r in rows]),
             scene=np.array([r[8] for r in rows]),
             seg=np.array([r[9] for r in rows]),
             obs=np.array([r[10] for r in rows], float),
             n_loop=live.feed.n_loop,
             counters=np.array([live.fab.n_ok, live.fab.n_bad,
                                live.loc.n_edge, rec]))
    print(f"[bench {env}/{traj}] last pass ({len(last)} kf): "
          f"py RMSE {np.sqrt((e_py ** 2).mean()):.3f} p90 "
          f"{np.percentile(e_py, 90):.3f} | fx RMSE "
          f"{np.sqrt((e_fx ** 2).mean()):.3f} p90 "
          f"{np.percentile(e_fx, 90):.3f} ({len(e_fx)} kf) | fx-th p90 "
          f"{np.rad2deg(np.percentile(e_th, 90)):.2f} deg | enc "
          f"{live.fab.n_ok}/{live.fab.n_enc} bad {live.fab.n_bad} | "
          f"edge-recenters {live.loc.n_edge} | recovers {rec}",
          flush=True)
    assert live.fab.n_bad == 0, "fabric encode mismatches"
    assert np.isfinite(e_fx).all() and len(e_fx) > 0
    print("bench ok")


def selftest():
    """Headless: short tour, freeze, calibration, small image, few loc
    steps — numbers only, hard asserts."""
    global IMG_STEP, IMG_RAD
    IMG_STEP, IMG_RAD = 0.3, 1.6
    live = Live(laps=1)
    live.run(realtime=False, stop_after=live.feed.n_loop + 25, image=True)
    st = live.state
    assert live.fab.n_bad == 0, "fabric encode mismatches"
    assert st["err_fx"] is not None and st["err_fx"] < 0.15, st
    im = np.array(live.imgd["img"])
    frac = float((im > 0.55).mean())
    print(f"[selftest] enc {st['enc']} bad {live.fab.n_bad}; "
          f"fx err {st['err_fx']} m; image wall-frac {frac:.3f}; "
          f"loc edge-recenters {live.loc.n_edge}")
    assert 0.005 < frac < 0.5, "image looks degenerate"
    pc = np.array(live.trail_gt).mean(0)      # room center, not a corner
    hits = visible_walls(live.imgd, pc,
                         freemask=getattr(live, "freemask", None))
    print(f"[selftest] visibility from center {np.round(pc, 2)}: "
          f"{len(hits)} wall hits (cached fabric image)")
    assert len(hits) > 25, "visibility ray-march found too few walls"
    hits2 = query_viewpoint_direct(live.fab, live.fmap, pc,
                                   n_rays=60, r_max=4.5,
                                   freemask=getattr(live, "freemask", None),
                                   trail=getattr(live, "fm_trail", None))
    print(f"[selftest] visibility from center: {len(hits2)} wall hits "
          f"(DIRECT from phasor space, per-sample fabric readouts)")
    assert len(hits2) > 20, "direct phasor viewpoint query degenerate"
    # score BOTH hit sets against the analytic world (scoring only)
    segs = np.asarray(DE.classroom_segs(None), float)

    def wdist(h):
        h = np.asarray(h, float)
        dd = np.full(len(h), np.inf)
        for a, b in segs:
            a, b = np.asarray(a), np.asarray(b)
            vv = b - a
            t = np.clip(((h - a) @ vv) / max(vv @ vv, 1e-9), 0, 1)
            dd = np.minimum(dd, np.linalg.norm(h - (a + t[:, None] * vv),
                                               axis=1))
        return dd
    d1, d2 = wdist(hits), wdist(hits2)
    print(f"[selftest] hit-to-GT-wall p50/p90: cached "
          f"{np.median(d1):.3f}/{np.percentile(d1, 90):.3f}  direct "
          f"{np.median(d2):.3f}/{np.percentile(d2, 90):.3f}")
    assert np.median(d2) < 0.20, "direct hits off-wall (study: 0.072)"
    print("selftest ok")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if what == "serve":
        serve(*sys.argv[2:])
    elif what == "bench":
        bench(*sys.argv[2:])
    else:
        selftest()
