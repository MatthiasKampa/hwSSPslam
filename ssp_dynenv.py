"""Dynamic / multi-room synthetic environments for the iCE40 deployment study.

Regime: the SPOT target head (360 deg x 1024 beams) in (a) a single 7x7 m
classroom (the Telluride room's proxy) and (b) "school10x" — a hallway with
N connected, deliberately IDENTICAL classrooms (~10x the map area, the
aliasing stressor at the twin-wall scale) — with PEOPLE as circle obstacles,
standing or walking on deterministic scripted paths.

Ablation discipline:
  - Paired seeds: all noise draws (trajectory heading, odometry, range,
    dropout, angle jitter) are INDEPENDENT of the people configuration, so
    `make(..., people=0)` and `make(..., people=k)` with the same seed see
    bit-identical noise — the delta is the people, nothing else.
  - People yield to the robot (walkers pause inside 0.45 m), so GT clearance
    holds by construction and frames stay physically sane.
  - jitter=0 keeps every classroom identical (max aliasing); jitter>0
    perturbs furniture per room (distinguishability knob).
  - Walkers are frozen within each frame (motion-during-sweep is the
    de-skew ablation, handled separately).

Bundles are ssp_datasets-shaped with EXACT GT (eval='exact'); the fenced
bench-only GT hook (slam.diag_gt, PROTOCOL 2 diagnostic) labels accepted
loop edges for precision reporting. No shipped module is edited.

Usage:
  python3 ssp_dynenv.py check     # geometry, clearance, determinism, pairing
  python3 ssp_dynenv.py quick     # classroom +-people, 360 beams (fast)
  python3 ssp_dynenv.py bench     # full grid, 1024 beams, bridge2 sampler
  python3 ssp_dynenv.py tenx      # school10x map-size pressure (iCE40 SPRAM)
"""
import sys

import numpy as np

import ssp_slam as S
import ssp_slam_loop as L
from worlds import _rect

STEP = 0.10                    # keyframe stride along the path [m]
ROBOT_V = 0.5                  # m/s (spot workshop pace) -> dt = STEP/ROBOT_V
ODO_T, ODO_R = 0.010, np.deg2rad(0.25)
PERSON_R = 0.18                # circle radius [m]
WALK_V = 1.2                   # m/s
YIELD_D = 0.45                 # walkers pause inside this distance to robot

ROOM_W = 7.0                   # classroom side
HALL_W = 2.4                   # hallway width
DOOR_W = 1.1


# ---------------------------------------------------------------- geometry
def _room_furniture(cx, cy, rng=None, front=1.0):
    """4 student desks + teacher desk, IDENTICAL layout unless rng given.
    front=+1/-1 selects the far-from-door wall for the teacher desk."""
    segs = []
    spots = [(-2.2, -2.2, 0.0), (2.2, -2.2, 0.0), (-2.2, 2.2, 0.0),
             (2.2, 2.2, 0.0)]
    for dx, dy, rot in spots:
        if rng is not None:
            dx += rng.uniform(-0.35, 0.35)
            dy += rng.uniform(-0.35, 0.35)
            rot = rng.uniform(0, 60)
        segs += _rect(cx + dx - 0.30, cy + dy - 0.25,
                      cx + dx + 0.30, cy + dy + 0.25, rot=rot)
    ty = 0.9 if rng is None else 0.9 + rng.uniform(-0.2, 0.2)
    yc = cy + front * (ROOM_W / 2 - ty)
    segs += _rect(cx - 0.7, yc - 0.3, cx + 0.7, yc + 0.3)
    return segs


def classroom_segs(seed_jitter=None):
    rng = None if seed_jitter is None else np.random.default_rng(seed_jitter)
    segs = _rect(0, 0, ROOM_W, ROOM_W)
    segs += _room_furniture(ROOM_W / 2, ROOM_W / 2, rng)
    return np.array(segs, float)


def school_segs(rooms=8, jitter=0.0, seed=0):
    """Hallway along x with rooms/2 classrooms above and below. Returns
    (segs, meta). Rooms are identical when jitter==0."""
    per_side = (rooms + 1) // 2
    hall_len = per_side * (ROOM_W + 2.0) + 2.0
    y0t, y1t = HALL_W, HALL_W + ROOM_W          # top rooms
    y0b, y1b = -ROOM_W, 0.0                     # bottom rooms
    segs = []
    segs.append([(0.0, 0.0), (0.0, HALL_W)])            # hall end caps
    segs.append([(hall_len, 0.0), (hall_len, HALL_W)])
    meta = dict(hall_len=hall_len, rooms=[], hall_y=HALL_W / 2)
    rng = np.random.default_rng(seed) if jitter > 0 else None
    for i in range(rooms):
        top = (i % 2 == 0)
        col = i // 2
        x0 = 2.0 + col * (ROOM_W + 2.0)
        x1 = x0 + ROOM_W
        y0, y1 = (y0t, y1t) if top else (y0b, y1b)
        dxc = (x0 + x1) / 2
        dl, dr = dxc - DOOR_W / 2, dxc + DOOR_W / 2
        yw = y0 if top else y1                          # wall shared w/ hall
        # room walls, door gap in the hallway-side wall
        segs.append([(x0, yw), (dl, yw)])
        segs.append([(dr, yw), (x1, yw)])
        yf = y1 if top else y0                          # far wall
        segs.append([(x0, yf), (x1, yf)])
        segs.append([(x0, y0), (x0, y1)])
        segs.append([(x1, y0), (x1, y1)])
        jr = None
        if rng is not None:
            jr = np.random.default_rng(rng.integers(1 << 31))
        segs += _room_furniture(dxc, (y0 + y1) / 2, jr,
                                front=1.0 if top else -1.0)
        meta["rooms"].append(dict(c=(dxc, (y0 + y1) / 2), top=top,
                                  door=(dxc, yw)))
    # hallway wall segments between doors (top edge y=HALL_W, bottom y=0)
    for y, side_top in ((HALL_W, True), (0.0, False)):
        xs = [0.0]
        for r in meta["rooms"]:
            if r["top"] == side_top:
                xs += [r["door"][0] - DOOR_W / 2, r["door"][0] + DOOR_W / 2]
        xs.append(hall_len)
        for a, b in zip(xs[0::2], xs[1::2]):
            if b > a + 1e-9:
                segs.append([(a, y), (b, y)])
    return np.array(segs, float), meta


# ------------------------------------------------------------- trajectories
def _orbit(cx, cy, r, n0=0.0):
    def f(t):                                   # t in [0,1)
        a = n0 + 2 * np.pi * t
        return np.array([cx + r * np.cos(a), cy + 0.8 * r * np.sin(a)])
    return f, 2 * np.pi * r * 0.9               # approx perimeter


def _polyline(points, step):
    pts = np.asarray(points, float)
    seg = np.diff(pts, axis=0)
    ln = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0], np.cumsum(ln)])
    s = np.arange(0, cum[-1], step)
    x = np.interp(s, cum, pts[:, 0])
    y = np.interp(s, cum, pts[:, 1])
    return np.stack([x, y], 1)


def classroom_path(laps=5):
    c = ROOM_W / 2
    ang = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    loop = np.stack([c + 1.6 * np.cos(ang), c + 1.3 * np.sin(ang)], 1)
    return _polyline(np.tile(loop, (laps, 1)), STEP)


def school_path(meta, laps=1):
    """Patrol: down the hallway visiting each room (small in-room loop),
    then straight back along the hallway (the revisit pass)."""
    hy = meta["hall_y"]
    way = [(1.0, hy)]
    for r in meta["rooms"]:
        dx, dy = r["door"]
        cx, cy = r["c"]
        inward = 1.0 if r["top"] else -1.0
        way += [(dx, hy), (dx, dy + inward * 0.8)]
        ang0 = -np.pi / 2 if r["top"] else np.pi / 2
        ang = ang0 + np.linspace(0, 2 * np.pi, 24, endpoint=False)
        way += [(cx + 1.5 * np.cos(a), cy + 1.2 * np.sin(a)) for a in ang]
        way += [(dx, dy + inward * 0.8), (dx, hy)]
    way += [(meta["hall_len"] - 1.0, hy), (1.0, hy)]
    return _polyline(np.tile(np.array(way, float), (laps, 1)), STEP)


def _headings(xy, rng):
    d = np.gradient(xy, axis=0)
    th = np.arctan2(d[:, 1], d[:, 0])
    # unwrap, boxcar-smooth over ~0.9 m so corners turn at a finite rate
    thu = np.unwrap(th)
    k = np.ones(9) / 9.0
    thu = np.convolve(np.pad(thu, 4, mode="edge"), k, "valid")
    return thu + rng.normal(0, np.deg2rad(0.3), len(xy))


# ----------------------------------------------------------------- people
def place_people(env, meta, count, moving, gt_xy, seed):
    """-> per-frame people positions (n, count, 2), deterministic.
    Standing: fixed spots >=0.6 m off the robot path. Walking: waypoint
    loops (hallway pacing / in-room wander), yield inside YIELD_D."""
    if count == 0:
        return np.zeros((len(gt_xy), 0, 2))
    rng = np.random.default_rng(seed + 7777)
    n = len(gt_xy)
    dt = STEP / ROBOT_V
    if env == "classroom":
        lo, hi = np.array([0.6, 0.6]), np.array([ROOM_W - 0.6, ROOM_W - 0.6])
        loops = [np.array([[1.2, 1.2], [ROOM_W - 1.2, 1.2],
                           [ROOM_W - 1.2, ROOM_W - 1.2], [1.2, ROOM_W - 1.2]])
                 + rng.uniform(-0.3, 0.3, (4, 2)) for _ in range(count)]
    else:
        hl, hy = meta["hall_len"], meta["hall_y"]
        lo, hi = np.array([0.5, 0.4]), np.array([hl - 0.5, HALL_W - 0.4])
        loops = []
        for j in range(count):
            xa = rng.uniform(1.0, hl / 2)
            xb = rng.uniform(hl / 2, hl - 1.0)
            lane = rng.uniform(0.45, HALL_W - 0.45)
            loops.append(np.array([[xa, lane], [xb, lane]]))
    if not moving:
        spots = []
        while len(spots) < count:
            p = rng.uniform(lo, hi)
            if np.linalg.norm(gt_xy - p, axis=1).min() > 0.6 and \
               all(np.linalg.norm(p - q) > 0.5 for q in spots):
                spots.append(p)
        return np.tile(np.array(spots)[None], (n, 1, 1))
    # walkers: arc-length position on their loop, advance v*dt, yield
    out = np.zeros((n, count, 2))
    for j, wp in enumerate(loops):
        closed = np.vstack([wp, wp[0]])
        seg = np.diff(closed, axis=0)
        ln = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0], np.cumsum(ln)])
        peri = cum[-1]
        s = rng.uniform(0, peri)
        for k in range(n):
            sk = s % peri
            i = np.searchsorted(cum, sk, "right") - 1
            i = min(i, len(seg) - 1)
            p = closed[i] + (sk - cum[i]) / max(ln[i], 1e-9) * seg[i]
            out[k, j] = p
            nxt = (s + WALK_V * dt) % peri
            i2 = min(np.searchsorted(cum, nxt, "right") - 1, len(seg) - 1)
            p2 = closed[i2] + (nxt - cum[i2]) / max(ln[i2], 1e-9) * seg[i2]
            if np.linalg.norm(p2 - gt_xy[min(k + 1, n - 1)]) > YIELD_D:
                s += WALK_V * dt                # else: yield (stand still)
    return out


def ray_circles(origin, angles, centers, radius=PERSON_R):
    """Min positive ray-circle intersection per beam; inf where none."""
    if len(centers) == 0:
        return np.full(len(angles), np.inf)
    d = np.stack([np.cos(angles), np.sin(angles)], 1)      # (B,2)
    oc = centers[None, :, :] - origin[None, None, :2].reshape(1, 1, 2)
    oc = np.broadcast_to(oc, (len(angles), len(centers), 2))
    b = (d[:, None, :] * oc).sum(-1)                        # (B,C)
    cc = (oc * oc).sum(-1) - radius ** 2
    disc = b * b - cc
    ok = (disc >= 0)
    t = np.where(ok, b - np.sqrt(np.maximum(disc, 0)), np.inf)
    t = np.where(t > 0, t, np.inf)
    return t.min(1)


# ----------------------------------------------------------------- bundles
def make(env="classroom", rooms=8, people=0, moving=False, laps=None,
         seed=11, n_beams=1024, jitter=0.0, cap=None):
    """-> ssp_datasets-shaped bundle, exact GT. Noise draws are people-
    independent (paired +-people at fixed seed)."""
    rng = np.random.default_rng(seed)
    if env == "classroom":
        segs = classroom_segs(None if jitter == 0 else seed)
        meta = None
        xy = classroom_path(laps or 5)
    else:
        segs, meta = school_segs(rooms=rooms, jitter=jitter, seed=seed)
        xy = school_path(meta, laps or 1)
    if cap:
        xy = xy[:cap]
    n = len(xy)
    th = _headings(xy, rng)
    gt = np.concatenate([xy, th[:, None]], 1)
    beam = -np.pi + np.arange(n_beams) * (2 * np.pi / n_beams)
    # people AFTER gt (yield needs the path), rng stream separate from noise
    ppl = place_people(env, meta, people, moving, xy, seed)
    odom = gt.copy()
    keys = []
    for k in range(n):
        ang = gt[k, 2] + beam + rng.normal(0, S.ANGLE_JITTER, n_beams)
        r_w = S.raycast(segs, gt[k, :2], ang)
        r_p = ray_circles(gt[k, :2], ang, ppl[k])
        r = np.minimum(r_w, r_p)
        r = np.where(r <= S.MAX_RANGE,
                     r + rng.normal(0, S.RANGE_NOISE, n_beams), np.inf)
        r[rng.random(n_beams) < S.DROPOUT] = np.inf
        if k:
            d = L.se2_mul(L.se2_inv(gt[k - 1]), gt[k])
            d[:2] += (rng.normal(0, ODO_T, 2)
                      + 0.02 * np.abs(d[:2]) * rng.normal(0, 1, 2))
            d[2] += rng.normal(0, ODO_R)
            odom[k] = L.se2_mul(odom[k - 1], d)
        keys.append((r.astype(np.float32), odom[k].copy(),
                     k * STEP / ROBOT_V))
    tag = f"{env}{rooms if env == 'school' else ''}-p{people}" \
          f"{'w' if moving else 's'}-j{jitter:g}-s{seed}"
    return dict(name=f"dyn-{tag}", kind="synth", eval="exact", path=None,
                keys=keys, beam=beam,
                odom=np.stack([kk[1] for kk in keys]),
                kts=np.array([t for _, _, t in keys]),
                rmin=0.05, rmax=S.MAX_RANGE - 0.05, gt=gt, _people=ppl)


# ------------------------------------------------------------------ bench
def _bridge2(rr, bm):
    import ssp_sampling as SP
    return SP.sample_interp(rr, bm, 2, 63.4)


def run_one(bundle, sample="bridge2", **kw):
    import ssp_datasets as DS
    import ssp_fpga as F
    args = dict(robust=True, attempt_every=4, relax_every=25, gap_kf=300,
                recent_aids=12, spec=None, nph=0)
    args.update(kw)
    slam = F.BandSLAM(**args)
    slam.diag_gt = bundle["gt"]                 # fenced: diagnostics only
    sm = _bridge2 if sample == "bridge2" else sample
    r = DS.run(dict(bundle), slam=slam, sample=sm)
    d = np.array([row for row in slam.diag], float)
    acc = d[d[:, -1] > 0] if len(d) else np.zeros((0, 1))
    ok = float((acc[:, 0] < 0.3).mean()) if len(acc) else float("nan")
    r["loop_prec"] = ok
    r["loop_gt_max"] = float(acc[:, 0].max()) if len(acc) else float("nan")
    return r


def _row(tag, r):
    print(f"  {tag:<34} ATE {r['ate']:6.3f}  med {r['med']:6.3f}  "
          f"loops {r['loops']:3d}  prec {r['loop_prec']:.2f}  "
          f"gtmax {r['loop_gt_max']:5.2f}  veto {r['veto']:3d}  "
          f"infl {r['infl']:3d}  mem {r['mem_kb']:5.0f} KB", flush=True)


def check():
    segs, meta = school_segs(rooms=8)
    xy = school_path(meta, laps=1)
    a, b = segs[:, 0], segs[:, 1]
    e = b - a
    ee = (e * e).sum(1)
    dmin = np.inf
    for p in xy[::7]:
        t = np.clip(((p - a) * e).sum(1) / np.maximum(ee, 1e-12), 0, 1)
        dmin = min(dmin, np.linalg.norm(a + t[:, None] * e - p, axis=1).min())
    print(f"school8: {len(segs)} segs, path {len(xy)} kf "
          f"({len(xy) * STEP:.0f} m), wall clearance {dmin:.2f} m")
    area = meta["hall_len"] * HALL_W + 8 * ROOM_W * ROOM_W
    print(f"school8 area ~{area:.0f} m^2 (classroom {ROOM_W * ROOM_W:.0f}"
          f" -> {area / ROOM_W ** 2:.1f}x)")
    b1 = make("classroom", people=0, seed=11, n_beams=360, cap=300)
    b2 = make("classroom", people=0, seed=11, n_beams=360, cap=300)
    same = all(np.array_equal(x[0], y[0])
               for x, y in zip(b1["keys"], b2["keys"]))
    print(f"determinism (same seed twice): {'bit-equal' if same else 'FAIL'}")
    b3 = make("classroom", people=4, moving=True, seed=11, n_beams=360,
              cap=300)
    diff = [int((~np.isclose(x[0], y[0], equal_nan=True)).sum())
            for x, y in zip(b1["keys"], b3["keys"])]
    pmin = min(np.linalg.norm(b3["_people"][k] - b3["gt"][k, :2],
                              axis=1).min() for k in range(len(b3["keys"])))
    print(f"pairing +-people: {sum(1 for d in diff if d)} of "
          f"{len(diff)} frames touched, mean {np.mean(diff):.1f} beams; "
          f"min robot-person dist {pmin:.2f} m (yield {YIELD_D})")
    # honest pairing check: beams the people rays did NOT shorten must be
    # bit-equal between the +-people bundles (same noise realization)
    bad = 0
    for k, (x, y) in enumerate(zip(b1["keys"], b3["keys"])):
        ang = b3["gt"][k, 2] + b3["beam"]       # jitter-free ray directions
        rp = ray_circles(b3["gt"][k, :2], ang, b3["_people"][k],
                         radius=PERSON_R + 0.08)  # margin covers angle jitter
        untouched = np.isinf(rp)
        if not np.array_equal(x[0][untouched], y[0][untouched]):
            bad += 1
    print(f"untouched beams bit-equal: {'yes' if bad == 0 else f'FAIL {bad}'}")


def quick():
    for people, moving in ((0, False), (5, True)):
        b = make("classroom", people=people, moving=moving, seed=11,
                 n_beams=360)
        _row(b["name"] + " [bridge2]", run_one(b))


def bench(seeds=(11, 12), n_beams=1024):
    for env, rooms in (("classroom", 0), ("school", 8)):
        for seed in seeds:
            b0 = None
            for people, moving in ((0, False), (2, False), (5, False),
                                   (5, True), (10, True)):
                b = make(env, rooms=rooms, people=people, moving=moving,
                         seed=seed, n_beams=n_beams)
                if b0 is None:
                    odo, gt = b["odom"], b["gt"]
                    import ssp_slam_carmen as C
                    e = np.linalg.norm(C.align_se2(odo[:, :2], gt[:, :2])
                                       - gt[:, :2], axis=1)
                    print(f"== {env}{rooms or ''} s{seed} "
                          f"({len(b['keys'])} kf, raw odom "
                          f"{np.sqrt((e ** 2).mean()):.3f}):", flush=True)
                    b0 = b
                _row(b["name"], run_one(b))


def tenx():
    b = make("school", rooms=8, people=0, seed=11, n_beams=1024)
    r = run_one(b)
    _row(b["name"], r)
    segn = len(r["slam"].segvec)
    print(f"  segments {segn}  -> 2b no-relo est {segn * 136 / 1024:.0f} KB "
          f"(iCE40 SPRAM 128 KB), oct36 est {segn * 136 * 0.6 / 1024:.0f} KB")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "check"
    dict(check=check, quick=quick, bench=bench, tenx=tenx)[what]()
