"""Small-scale loop-closure benches with ground truth.

Synthetic multi-loop runs with DRIFTING odometry and a 180-deg FOV lidar
(mirrors the Intel difficulty), so the continual graph-maintenance design can
be developed and stress-tested where every constraint can be labeled
true/false against ground truth.

Design under test ("continual"): no closure events. From the first frame,
- every frame: sequential graph edge from the frontend estimate;
- every ATTEMPT_EVERY frames: try ONE map constraint against non-adjacent
  geometry (segment-restricted or bundle-consensus, per config);
- every RELAX_EVERY frames: bounded relaxation of the whole graph and blended
  application to all poses (a no-op when nothing new was accepted).

Metrics per run: ATE vs GT, max pose jerk applied by any single relaxation
(stability), edge count + false-edge count (GT-labeled), wall time.

Usage: python3 -m sspslam.bench [quick|full]
"""

import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import sspslam.encoder as S
import sspslam.lattice as L
from sspslam.worlds import WORLDS

FOV_BEAMS = 180
BEAM = np.deg2rad(np.arange(FOV_BEAMS) - 90.0)   # 180 deg forward FOV
ODO_T, ODO_R = 0.010, np.deg2rad(0.25)           # per-frame odometry noise
ANCHOR = 5
GAP = 60                                          # non-adjacency for constraints


def multiloop_traj(n_frames, laps=3):
    """Repeated loops with a CONTINUOUSLY varying scale (no teleports at lap
    boundaries), so passes overlap without being identical."""
    u = np.linspace(0, laps * 2 * np.pi, n_frames, endpoint=False)
    s = 1.0 + 0.12 * np.sin(0.9 * u)
    a, b = 4.8 * s, 1.9 * s
    x, y = 8 + a * np.cos(u), 5 + b * np.sin(u)
    # heading from the actual path tangent (finite difference, wrap-safe)
    dx = np.gradient(x)
    dy = np.gradient(y)
    th = np.arctan2(dy, dx)
    gt = np.stack([x, y, th], 1)
    gt[:, 2] += S.RNG.normal(0, np.deg2rad(0.3), n_frames)
    return gt


def sim_odometry(gt):
    """Compose noisy GT deltas -> drifting odometry track."""
    odo = gt.copy()
    for k in range(1, len(gt)):
        d = L.se2_mul(L.se2_inv(gt[k - 1]), gt[k])
        d[:2] += S.RNG.normal(0, ODO_T, 2) + 0.02 * np.abs(d[:2]) * S.RNG.normal(0, 1, 2)
        d[2] += S.RNG.normal(0, ODO_R)
        odo[k] = L.se2_mul(odo[k - 1], d)
    return odo


def scan_at(segs, pose):
    angles = pose[2] + BEAM + S.RNG.normal(0, S.ANGLE_JITTER, FOV_BEAMS)
    r = S.raycast(segs, pose[:2], angles)
    r = np.where(r <= S.MAX_RANGE, r + S.RNG.normal(0, S.RANGE_NOISE, FOV_BEAMS), np.inf)
    r[S.RNG.random(FOV_BEAMS) < S.DROPOUT] = np.inf
    return r


class Graph:
    def __init__(self):
        self.edges = []

    def add(self, a, b, Z, wt, wr, kind, is_false):
        self.edges.append((a, b, np.asarray(Z, float), wt, wr, kind, is_false))

    def optimize(self, P0, robust=True):
        A = len(P0)
        sp = lil_matrix((3 * len(self.edges), 3 * (A - 1)))
        for e, (a, b, *_) in enumerate(self.edges):
            for nd in (a, b):
                if nd > 0:
                    sp[3 * e:3 * e + 3, 3 * (nd - 1):3 * nd] = 1

        def resid(x):
            P = np.vstack([P0[0], x.reshape(-1, 3)])
            out = np.empty(3 * len(self.edges))
            for e, (a, b, Z, wt, wr, *_ ) in enumerate(self.edges):
                d = S._rot(-P[a, 2]) @ (P[b, :2] - P[a, :2])
                out[3 * e:3 * e + 2] = (d - Z[:2]) * wt
                out[3 * e + 2] = S.wrap(P[b, 2] - P[a, 2] - Z[2]) * wr
            return out

        kw = dict(loss="soft_l1", f_scale=2.0) if robust else {}
        res = least_squares(resid, P0[1:].ravel(), jac_sparsity=sp.tocsr(),
                            method="trf", max_nfev=30, verbose=0, **kw)
        return np.vstack([P0[0], res.x.reshape(-1, 3)])


def run_bench(world="room", n=420, seed=1, mode="continual",
              constraint="segment", robust=True, attempt_every=3,
              relax_every=5, verbose=False):
    S.RNG = np.random.default_rng(seed)
    segs = WORLDS[world]()
    gt = multiloop_traj(n)
    odo = sim_odometry(gt)

    enc, enc_main = L.ENC, L.ENC_MAIN
    matcher = S.Matcher(enc_main, t_half=0.48, rot_half_deg=9, rot_step_deg=1.5)
    cmatcher = S.Matcher(enc_main, t_half=0.72, rot_half_deg=9, rot_step_deg=1.5)

    store = L.KeyframeStore(n)
    est = np.zeros((n, 3))
    graph = Graph()
    anchors = np.arange(0, n, ANCHOR)
    a_of = np.searchsorted(anchors, np.arange(n), side="right") - 1
    seq_done = -1
    edges_dirty = False
    max_jerk, n_relax = 0.0, 0

    def add_seq(upto_a):
        nonlocal seq_done
        while seq_done + 1 <= upto_a:
            i = seq_done + 1
            if i > 0:
                a, b = anchors[i - 1], anchors[i]
                graph.add(i - 1, i, L.se2_mul(L.se2_inv(est[a]), est[b]),
                          1 / 0.10, 1 / np.deg2rad(1.0), "seq", False)
            seq_done = i

    def apply_all(P):
        nonlocal max_jerk
        Pold = est[anchors].copy()
        corrs = np.stack([L.se2_mul(P[i], L.se2_inv(Pold[i]))
                          for i in range(len(P))])
        jerk = 0.0
        for i in range(len(P)):
            lo = anchors[i]
            hi = anchors[i + 1] if i + 1 < len(anchors) else n
            cn = corrs[min(i + 1, len(P) - 1)]
            span = max(1, hi - lo)
            for k2 in range(lo, hi):
                if store.pts[k2] is None:
                    break
                f = (k2 - lo) / span
                corr = np.array([
                    (1 - f) * corrs[i, 0] + f * cn[0],
                    (1 - f) * corrs[i, 1] + f * cn[1],
                    corrs[i, 2] + f * S.wrap(cn[2] - corrs[i, 2])])
                new = L.se2_mul(corr, est[k2])
                jerk = max(jerk, np.linalg.norm(new[:2] - est[k2, :2]))
                if np.linalg.norm(new[:2] - est[k2, :2]) > 0.01 \
                        or abs(S.wrap(new[2] - est[k2, 2])) > np.deg2rad(0.1):
                    est[k2] = new
                    store.pver[k2] += 1
        max_jerk = max(max_jerk, jerk)

    def try_constraint(k):
        nonlocal edges_dirty
        pts, w = store.pts[k], store.w[k]
        if pts is None or len(pts) < 20:
            return
        idx = np.arange(k)
        far = idx[np.abs(idx - k) > GAP]
        near = far[np.linalg.norm(est[far, :2] - est[k, :2], axis=1) < 5.0]
        if len(near) < 10:
            return
        if constraint == "segment":
            # restrict to the single contiguous pass containing the nearest kf
            c0 = int(near[np.argmin(np.linalg.norm(est[near, :2] - est[k, :2], axis=1))])
            seg = np.arange(max(0, c0 - 25), min(k, c0 + 25))
            seg = seg[np.abs(seg - k) > GAP]
            use = seg[[store.pts[i] is not None for i in seg]]
        else:                      # bundle-consensus (known-bad control)
            use = near
        if len(use) < 10:
            return
        B = store.bundle(use, est)
        pose = cmatcher.match(B[L.MAIN], pts, w, est[k])
        pose[2] = S.wrap(pose[2])
        if np.linalg.norm(pose[:2] - est[k, :2]) > 0.7 \
                or abs(S.wrap(pose[2] - est[k, 2])) > np.deg2rad(10):
            return
        c = int(use[np.argmin(np.linalg.norm(est[use, :2] - pose[:2], axis=1))])
        if a_of[c] == a_of[k]:
            return
        Z = L.se2_mul(L.se2_inv(est[anchors[a_of[c]]]),
                      L.se2_mul(pose, L.se2_mul(L.se2_inv(est[k]),
                                                est[anchors[a_of[k]]])))
        # GT label: does Z match the true relative anchor pose?
        Zt = L.se2_mul(L.se2_inv(gt[anchors[a_of[c]]]), gt[anchors[a_of[k]]])
        is_false = np.linalg.norm(Z[:2] - Zt[:2]) > 0.30 \
            or abs(S.wrap(Z[2] - Zt[2])) > np.deg2rad(3)
        graph.add(a_of[c], a_of[k], Z, 1 / 0.15, 1 / np.deg2rad(1.5),
                  "loop", bool(is_false))
        edges_dirty = True

    t0 = time.time()
    for k in range(n):
        r = scan_at(segs, gt[k])
        pts, w, _ = S.scan_to_samples(r, BEAM)
        store.set_scan(k, pts, w)
        if k == 0:
            est[0] = odo[0]
        else:
            guess = L.se2_mul(est[k - 1], L.se2_mul(L.se2_inv(odo[k - 1]), odo[k]))
            est[k] = guess
            if len(pts) >= 20:
                nearw = np.flatnonzero(
                    np.linalg.norm(est[:k, :2] - guess[:2], axis=1) < 8.0)
                nearw = nearw[nearw >= k - 40]     # short local window -> drift
                if len(nearw):
                    M = store.bundle(nearw, est)[L.MAIN]
                    cand = matcher.match(M, pts, w, guess)
                    cand[2] = S.wrap(cand[2])
                    if np.linalg.norm(cand[:2] - guess[:2]) < 0.45 \
                            and abs(S.wrap(cand[2] - guess[2])) < np.deg2rad(11):
                        est[k] = cand
        store.world_enc(k, est[k])
        add_seq(a_of[k])

        if mode == "continual":
            if k % attempt_every == 0:
                try_constraint(k)
            if k % relax_every == 0 and edges_dirty:
                P = graph.optimize(est[anchors[:a_of[k] + 1]], robust=robust)
                apply_all(P)
                edges_dirty = False
                n_relax += 1
        elif mode == "event":
            if k % attempt_every == 0:
                try_constraint(k)
                if edges_dirty:    # relax immediately on acceptance (event-style)
                    P = graph.optimize(est[anchors[:a_of[k] + 1]], robust=robust)
                    apply_all(P)
                    edges_dirty = False
                    n_relax += 1

    if edges_dirty:
        P = graph.optimize(est[anchors], robust=robust)
        apply_all(P)
        n_relax += 1
    dt = time.time() - t0

    ate = np.linalg.norm(est[:, :2] - gt[:, :2], axis=1)
    loops = [e for e in graph.edges if e[5] == "loop"]
    falses = sum(1 for e in loops if e[6])
    odo_ate = np.linalg.norm(odo[:, :2] - gt[:, :2], axis=1)
    return dict(ate=np.sqrt((ate ** 2).mean()), ate_max=ate.max(),
                odo_ate=np.sqrt((odo_ate ** 2).mean()),
                edges=len(loops), false=falses, relax=n_relax,
                jerk=max_jerk, secs=dt)


def row(tag, rs):
    ate = [r["ate"] for r in rs]
    print(f"{tag:<38} ATE {100 * np.mean(ate):6.1f} cm (worst {100 * np.max(ate):6.1f})  "
          f"edges {np.mean([r['edges'] for r in rs]):5.1f}  "
          f"false {np.mean([r['false'] for r in rs]):4.1f}  "
          f"jerk {np.mean([r['jerk'] for r in rs]):5.2f} m  "
          f"relax {np.mean([r['relax'] for r in rs]):4.0f}  "
          f"{np.mean([r['secs'] for r in rs]):4.0f}s")


if __name__ == "__main__":
    quick = (sys.argv[1] if len(sys.argv) > 1 else "quick") == "quick"
    seeds = (1, 2) if quick else (1, 2, 3)
    worlds = ["room"] if quick else ["room", "mixed"]
    base = [run_bench(w, seed=s, mode="off") for w in worlds for s in seeds]
    print(f"odometry-only ATE {100 * np.mean([r['odo_ate'] for r in base]):.1f} cm | "
          f"frontend-only ATE {100 * np.mean([r['ate'] for r in base]):.1f} cm "
          f"(no graph, drifting local window)")
    for cfg in [
        dict(mode="continual", constraint="segment", robust=True),
        dict(mode="continual", constraint="segment", robust=False),
        dict(mode="continual", constraint="bundle", robust=True),
        dict(mode="event", constraint="segment", robust=True),
        dict(mode="event", constraint="bundle", robust=True),
    ]:
        rs = [run_bench(w, seed=s, **cfg) for w in worlds for s in seeds]
        row(f"{cfg['mode']}/{cfg['constraint']}/robust={cfg['robust']}", rs)
