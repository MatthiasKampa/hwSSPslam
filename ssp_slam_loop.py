"""SSP SLAM with loop closure on a CARMEN log (Intel Research Lab).

SSP-native loop closure on top of the ssp_slam_carmen frontend:
- Polar frequency lattice, ring-major: 6 rings (lam 0.25,0.5,1,2 | 5.3,12.8
  relocalization rings, incommensurate with the octaves) x 60 equally spaced
  angles. Rotation by a grid step m*pi/60 is EXACTLY a circular shift of the
  angle index (conjugate on sector wrap): no re-encoding.
- NOTE on place recognition: the |S| magnitude descriptor (rotation-equivariant
  circular correlation) survives only in self_test/rel_rot_index — it fails on
  this 180-deg-FOV sensor (rotation changes scan CONTENT, not just its index),
  so the pipeline instead relocalizes in a prior window around the estimate:
  +-30 deg x +-6 m normally, +-42 deg x +-12 m (coarser rings) when no closure
  has been accepted for a while.
- Acceptance: z-score over the hypothesis surface, fine refinement, innovation
  gates, and two consecutive consistent relocalizations.
- Backend: anchor pose graph (every 10th keyframe), scipy least_squares with
  a sparse jacobian after each accepted closure; corrections blended into
  intermediate keyframes; world encodings repaired lazily (re-encode dirty
  keyframes from stored scan points only when a query touches them).

Usage: python3 ssp_slam_loop.py [data/intel.log]
"""

import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import ssp_slam as S
import ssp_slam_carmen as C

LAMS = np.array([0.25, 0.5, 1.0, 2.0, 5.3, 12.8])  # relo rings incommensurate
                                                    # with the octave ladder
N_ANG = 60
N_RING = len(LAMS)
MAIN = slice(0, 4 * N_ANG)          # rings used by the frontend matcher
VALID_MAX = 40.0

LOCAL_WIN = 150                     # frontend sliding map, keyframes
LOCAL_EXCL = 400                    # closures only against keyframes older than this
RELOC_EVERY = 6                     # min keyframes between relocalization attempts
RELOC_Z = 3.0                       # peak z-score over the hypothesis surface
ROT_WIN = 10                        # heading window: +-10 grid steps = +-30 deg
OPT_EVERY = 20                      # min keyframes between graph optimisations
ANCHOR = 10


def build_W():
    a = np.arange(N_ANG) * np.pi / N_ANG
    u = np.stack([np.cos(a), np.sin(a)], 1)
    return np.concatenate([(2 * np.pi / lam) * u for lam in LAMS])


W = build_W()
ENC = S.SSPEncoder.__new__(S.SSPEncoder)
ENC.W = W
ENC_MAIN = S.SSPEncoder.__new__(S.SSPEncoder)
ENC_MAIN.W = W[MAIN]


def encode(pts, w):
    return np.exp(1j * (pts @ W.T)).T @ w


def rot_permute(Svec, m):
    """Encoding of the scan rotated by m*pi/60, via index shift (exact)."""
    A = Svec.reshape(N_RING, N_ANG)
    ext = np.concatenate([A, np.conj(A)], axis=1)          # angle period 2pi
    idx = (np.arange(N_ANG) - m) % (2 * N_ANG)
    return ext[:, idx].reshape(-1)


def descriptor(Svec):
    """Rotation-equivariant, translation-invariant signature (N_RING, N_ANG)."""
    A = np.abs(Svec.reshape(N_RING, N_ANG))
    A = A - A.mean(axis=1, keepdims=True)
    n = np.linalg.norm(A)
    return A / n if n > 0 else A


# ---------------------------------------------------------------------------
# SE(2) helpers
# ---------------------------------------------------------------------------

def se2_mul(A, B):
    return np.array([*(A[:2] + S._rot(A[2]) @ B[:2]), S.wrap(A[2] + B[2])])


def se2_inv(A):
    return np.array([*(-(S._rot(-A[2]) @ A[:2])), -A[2]])


# ---------------------------------------------------------------------------
# Self-tests of the rotation/descriptor conventions (synthetic, fast)
# ---------------------------------------------------------------------------

def rel_rot_index(F_c, d_now):
    """Place-recognition correlation: index m0 such that the current scan
    rotated by (m_base(theta_c) + m0) grid steps aligns with keyframe c."""
    corr = np.fft.ifft((F_c * np.conj(np.fft.fft(d_now, axis=1))).sum(0)).real
    return int(np.argmax(corr)), corr.max()


def self_test():
    rng = np.random.default_rng(3)
    pts = rng.uniform(-4, 4, (300, 2))
    w = np.full(300, 0.1)
    for m_true in (7, 33, 95):
        th = m_true * np.pi / N_ANG
        Sr = encode(pts @ S._rot(th).T, w)
        assert np.allclose(Sr, rot_permute(encode(pts, w), m_true), atol=1e-9)
    # end-to-end heading recovery through the exact place-rec path:
    # two robot-frame views of the same world points at different headings
    q = rng.uniform(-4, 4, (300, 2))
    th_c, th_now = 0.83, 2.31
    S_c = encode(q @ S._rot(-th_c).T @ S._rot(0).T, w)      # p_c = R(th_c)^T q
    S_now = encode(q @ S._rot(-th_now).T, w)
    F_c = np.fft.fft(descriptor(S_c), axis=1)
    m0, _ = rel_rot_index(F_c, descriptor(S_now))
    m_base = int(round(th_c * N_ANG / np.pi))
    cands = [S.wrap((m_base + s * m0 + p) * np.pi / N_ANG - th_now)
             for s in (1, -1) for p in (0, N_ANG)]
    best = min(np.abs(cands))
    assert best < np.pi / N_ANG, np.degrees(cands)
    which = int(np.argmin(np.abs(cands)))
    assert which // 2 == 0, "sign convention: use m_base + m0"
    print("self-test ok: rotation permutation + place-rec heading recovery")


# ---------------------------------------------------------------------------
# Lazy world-frame encodings
# ---------------------------------------------------------------------------

class KeyframeStore:
    def __init__(self, n):
        self.pts = [None] * n           # robot-frame sample points
        self.w = [None] * n
        self.E = np.zeros((n, W.shape[0]), complex)  # world-frame encodings
        self.ever = np.full(n, -1)      # pose version E was built with
        self.pver = np.zeros(n, int)    # current pose version

    def set_scan(self, k, pts, w):
        self.pts[k], self.w[k] = pts, w

    def world_enc(self, k, pose):
        if self.ever[k] != self.pver[k]:
            pw = self.pts[k] @ S._rot(pose[2]).T
            self.E[k] = ENC.shift(pose[:2]) * encode(pw, self.w[k])
            self.ever[k] = self.pver[k]
        return self.E[k]

    def bundle(self, idx, poses):
        for k in idx:
            self.world_enc(k, poses[k])
        return self.E[idx].sum(0)


# ---------------------------------------------------------------------------
# Verification: coarse-to-fine phase correlation against an old submap
# ---------------------------------------------------------------------------

def grid_scores(B, Svec, center, half, step, ring_mask):
    v = np.arange(-half, half + step / 2, step)
    gx, gy = np.meshgrid(v, v)
    G = center + np.stack([gx.ravel(), gy.ravel()], 1)
    c = (np.conj(B) * Svec)[ring_mask]
    sc = (np.exp(1j * (G @ W[ring_mask].T)) @ c).real
    return G, sc


_RINGS = np.repeat(np.arange(N_RING), N_ANG)
WIDE = _RINGS >= 2                   # lam 1, 2, 5.3, 12.8
WIDER = _RINGS >= 3                  # lam 2, 5.3, 12.8 (adaptive wide stage)


def _basis(half, step, mask):
    v = np.arange(-half, half + step / 2, step)
    gx, gy = np.meshgrid(v, v)
    off = np.stack([gx.ravel(), gy.ravel()], 1)
    return off, np.exp(1j * (off @ W[mask].T))


RELOC_OFF, RELOC_E = _basis(6.0, 0.4, WIDE)
RELOC_OFF2, RELOC_E2 = _basis(12.0, 0.7, WIDER)


def relocalize(B, S_now, guess, matcher_bundle, matcher, pts, w, wide=False):
    """Prior-windowed basin search against the old-map bundle. Normal mode:
    +-30 deg x +-6 m (lam >= 1). Wide mode (long time since last closure):
    +-42 deg x +-12 m on coarser rings, then a mid refine. -> (pose, z) | None."""
    Bc = np.conj(B)
    mask, off, E = (WIDER, RELOC_OFF2, RELOC_E2) if wide else (WIDE, RELOC_OFF, RELOC_E)
    rot_win = ROT_WIN + 4 if wide else ROT_WIN
    ph = np.exp(1j * (W[mask] @ guess[:2]))
    m_est = int(round(guess[2] * N_ANG / np.pi))
    ms = (m_est + np.arange(-rot_win, rot_win + 1)) % (2 * N_ANG)
    sc = np.empty((len(ms), len(off)))
    for i, m in enumerate(ms):
        sc[i] = (E @ ((Bc * rot_permute(S_now, m))[mask] * ph)).real
    z = (sc.max() - sc.mean()) / sc.std()
    if z < RELOC_Z:
        return None
    i_pk, g_pk = np.unravel_index(int(sc.argmax()), sc.shape)
    t1 = guess[:2] + off[g_pk]
    m_pk = ms[i_pk]
    if wide:  # mid refine on lam >= 1 before the fine stage
        G15, s15 = grid_scores(B, rot_permute(S_now, m_pk), t1, 1.2, 0.15, WIDE)
        t1 = G15[int(np.argmax(s15))]
    G3, s3 = grid_scores(B, rot_permute(S_now, m_pk), t1, 0.5, 0.06,
                         np.ones_like(WIDE))
    t3 = G3[int(np.argmax(s3))]
    g2 = np.array([t3[0], t3[1], m_pk * np.pi / N_ANG])
    pose = matcher.match(matcher_bundle, pts, w, g2)
    pose[2] = S.wrap(pose[2])
    if np.linalg.norm(pose[:2] - g2[:2]) > 0.45 \
            or abs(S.wrap(pose[2] - g2[2])) > np.deg2rad(6):
        return None
    return pose, z


# ---------------------------------------------------------------------------
# Anchor pose graph
# ---------------------------------------------------------------------------

class PoseGraph:
    def __init__(self):
        self.edges = []              # (a, b, Z, wt, wr)

    def add(self, a, b, Z, wt, wr):
        self.edges.append((a, b, np.asarray(Z, float), wt, wr))

    def optimize(self, P0):
        A = len(P0)
        sp = lil_matrix((3 * len(self.edges), 3 * (A - 1)))
        for e, (a, b, _, _, _) in enumerate(self.edges):
            for n in (a, b):
                if n > 0:
                    sp[3 * e:3 * e + 3, 3 * (n - 1):3 * n] = 1

        def resid(x):
            P = np.vstack([P0[0], x.reshape(-1, 3)])
            out = np.empty(3 * len(self.edges))
            for e, (a, b, Z, wt, wr) in enumerate(self.edges):
                d = S._rot(-P[a, 2]) @ (P[b, :2] - P[a, :2])
                out[3 * e:3 * e + 2] = (d - Z[:2]) * wt
                out[3 * e + 2] = S.wrap(P[b, 2] - P[a, 2] - Z[2]) * wr
            return out

        res = least_squares(resid, P0[1:].ravel(), jac_sparsity=sp.tocsr(),
                            method="trf", loss="soft_l1", f_scale=2.0,
                            max_nfev=40, verbose=0)
        return np.vstack([P0[0], res.x.reshape(-1, 3)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    self_test()
    path = sys.argv[1] if len(sys.argv) > 1 else "data/intel.log"
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)
    for arg in sys.argv[2:]:
        if arg.isdigit():          # flags and the numeric limit can mix freely
            keys = keys[:int(arg)]
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(np.arange(n_beams) - 90.0)  # SICK: 1 deg spacing, -90..+89
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    print(f"{len(scans)} scans -> {n} keyframes   D={W.shape[0]}")

    matcher = S.Matcher(ENC_MAIN, t_half=0.48, rot_half_deg=12.0, rot_step_deg=1.5)
    store = KeyframeStore(n)
    est = np.zeros((n, 3))
    S_robot = np.zeros((n, W.shape[0]), complex)
    pending = None                  # last unconfirmed relocalization

    graph = PoseGraph()
    anchors = np.arange(0, n, ANCHOR)
    a_of = np.searchsorted(anchors, np.arange(n), side="right") - 1
    closures, n_gated = [], 0
    last_verify = last_opt = -10 ** 9
    last_accept = 0
    seq_done = -1

    def add_seq_edges(upto):
        nonlocal seq_done
        while seq_done + 1 <= upto:
            i = seq_done + 1
            if i > 0:
                a, b = anchors[i - 1], anchors[i]
                Z = se2_mul(se2_inv(est[a]), est[b])
                graph.add(i - 1, i, Z, wt=1 / 0.15, wr=1 / np.deg2rad(1.5))
            seq_done = i

    def apply_optimized(Pnew, upto_anchor):
        Pold = est[anchors[:upto_anchor + 1]].copy()
        corrs = np.stack([se2_mul(Pnew[i], se2_inv(Pold[i]))
                          for i in range(upto_anchor + 1)])
        for i in range(upto_anchor + 1):
            lo = anchors[i]
            hi = anchors[i + 1] if i + 1 <= upto_anchor else min(lo + ANCHOR, n)
            cn = corrs[i + 1] if i + 1 <= upto_anchor else corrs[i]
            span = max(1, hi - lo)
            for k in range(lo, min(hi, n)):
                if store.pts[k] is None:
                    break
                f = (k - lo) / span   # blend corrections across the segment
                corr = np.array([
                    (1 - f) * corrs[i, 0] + f * cn[0],
                    (1 - f) * corrs[i, 1] + f * cn[1],
                    corrs[i, 2] + f * S.wrap(cn[2] - corrs[i, 2])])
                new = se2_mul(corr, est[k])
                if np.linalg.norm(new[:2] - est[k, :2]) > 0.02 \
                        or abs(S.wrap(new[2] - est[k, 2])) > np.deg2rad(0.2):
                    est[k] = new
                    store.pver[k] += 1

    refine_matcher = S.Matcher(ENC_MAIN, t_half=0.72, rot_half_deg=9.0,
                               rot_step_deg=1.5)
    edge_seen = set()
    all_idx = np.arange(n)

    def harvest_edges(sample, exclude=250, radius=6.0, gate_t=0.7, gate_r=10.0):
        """Match sampled keyframes against non-adjacent overlapping map regions
        and add the resulting constraints. Returns number of edges added."""
        added = 0
        for k2 in sample:
            pts2, w2 = store.pts[k2], store.w[k2]
            if pts2 is None or len(pts2) < 20:
                continue
            cand = all_idx[np.abs(all_idx - k2) > exclude]
            cand = cand[cand < len(est)]
            cand = cand[[store.pts[i] is not None for i in cand]]
            nearby = cand[np.linalg.norm(est[cand, :2] - est[k2, :2],
                                         axis=1) < radius]
            if len(nearby) < 20:
                continue
            B2 = store.bundle(nearby, est)
            pose2 = refine_matcher.match(B2[MAIN], pts2, w2, est[k2])
            pose2[2] = S.wrap(pose2[2])
            if np.linalg.norm(pose2[:2] - est[k2, :2]) > gate_t \
                    or abs(S.wrap(pose2[2] - est[k2, 2])) > np.deg2rad(gate_r):
                continue
            c2 = int(nearby[np.argmin(
                np.linalg.norm(est[nearby, :2] - pose2[:2], axis=1))])
            key = (a_of[c2], a_of[k2])
            if key in edge_seen or key[0] == key[1]:
                continue
            edge_seen.add(key)
            Z2 = se2_mul(se2_inv(est[anchors[a_of[c2]]]),
                         se2_mul(pose2, se2_mul(se2_inv(est[k2]),
                                                est[anchors[a_of[k2]]])))
            graph.add(a_of[c2], a_of[k2], Z2,
                      wt=1 / 0.08, wr=1 / np.deg2rad(0.8))
            added += 1
        return added

    def closure_event(k):
        """Propagate a closure through the whole map: optimize, apply to every
        keyframe, then lock in newly-overlapping regions with extra edges."""
        pre = est[:k + 1].copy()
        P = graph.optimize(est[anchors[:a_of[k] + 1]])
        apply_optimized(P, a_of[k])
        moved = np.flatnonzero(
            np.linalg.norm(est[:k + 1, :2] - pre[:, :2], axis=1) > 0.05)
        extra = 0
        if len(moved) and "lockin" in sys.argv:
            # NOTE: off by default — bundle-consensus edges are self-fulfilling
            # (they freeze the current arrangement and fight later closures)
            extra = harvest_edges(moved[::8][:100])
            if extra:
                P = graph.optimize(est[anchors[:a_of[k] + 1]])
                apply_optimized(P, a_of[k])
        print(f"  closure event @kf {k}: {len(moved)} kf moved"
              + (f", +{extra} lock-in edges" if extra else ""), flush=True)

    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < VALID_MAX, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        if k == 0:
            est[0] = opose
        else:
            guess = C.compose_guess(est[k - 1], odom[k - 1], odom[k])
            est[k] = guess
            if len(pts) >= 20:
                near = np.flatnonzero(
                    np.linalg.norm(est[:k, :2] - guess[:2], axis=1) < 8.0)
                near = near[near >= k - LOCAL_WIN]
                if len(near):
                    M = store.bundle(near, est)[MAIN]
                    cand = matcher.match(M, pts, w, guess)
                    cand = matcher.match(M, pts, w, cand)
                    cand[2] = S.wrap(cand[2])
                    if (np.linalg.norm(cand[:2] - guess[:2]) < 0.45
                            and abs(S.wrap(cand[2] - guess[2])) < np.deg2rad(11)):
                        est[k] = cand
                    else:
                        n_gated += 1
        store.set_scan(k, pts, w)
        S_robot[k] = encode(pts, w) if len(pts) else 0
        store.world_enc(k, est[k])

        # ---- relocalization against the OLD map around the current estimate
        old_lim = k - LOCAL_EXCL
        if old_lim > 50 and k - last_verify >= RELOC_EVERY and len(pts) >= 20:
            old = np.flatnonzero(
                np.linalg.norm(est[:old_lim, :2] - est[k, :2], axis=1) < 10.0)
            if len(old) >= 30:
                last_verify = k
                B = store.bundle(old, est)
                wide = k - last_accept > 600 and "nowide" not in sys.argv
                got = relocalize(B, S_robot[k], est[k], B[MAIN], matcher,
                                 pts, w, wide=wide)
                if got is not None:
                    pose_hyp, z = got
                    corr_t = pose_hyp[:2] - est[k, :2]
                    corr_r = S.wrap(pose_hyp[2] - est[k, 2])
                    ok_mag = np.linalg.norm(corr_t) < (14.0 if wide else 8.0) \
                        and abs(corr_r) < np.deg2rad(45 if wide else 35)
                    # require two consistent relocalizations close in time
                    consistent = (pending is not None and k - pending[0] <= 15
                                  and np.linalg.norm(corr_t - pending[1]) < 0.6
                                  and abs(S.wrap(corr_r - pending[2])) < np.deg2rad(6))
                    if ok_mag and consistent:
                        pending = None
                        last_accept = k
                        c = int(old[np.argmin(
                            np.linalg.norm(est[old, :2] - pose_hyp[:2], axis=1))])
                        add_seq_edges(a_of[k])
                        Zloop = se2_mul(
                            se2_inv(est[anchors[a_of[c]]]),
                            se2_mul(pose_hyp,
                                    se2_mul(se2_inv(est[k]),
                                            est[anchors[a_of[k]]])))
                        graph.add(a_of[c], a_of[k], Zloop,
                                  wt=1 / 0.10, wr=1 / np.deg2rad(1.0))
                        closures.append((c, k))
                        if k - last_opt >= OPT_EVERY:
                            last_opt = k
                            closure_event(k)
                    elif ok_mag:
                        pending = (k, corr_t, corr_r)
        if k % 500 == 0:
            print(f"  frame {k}/{n}  t={time.time() - t0:.0f}s  "
                  f"closures={len(closures)}", flush=True)

    if closures:
        add_seq_edges(a_of[n - 1])
        P = graph.optimize(est[anchors])
        apply_optimized(P, len(anchors) - 1)
    dt = time.time() - t0
    print(f"online done: {dt:.0f}s ({dt / n * 1e3:.0f} ms/kf)  closures={len(closures)}  "
          f"odometry fallbacks={n_gated} ({100 * n_gated / n:.1f}%)", flush=True)

    # ------------------------------------------------------------------
    # Global refinement: densely harvest constraints across the whole
    # trajectory, optimize the full graph, straighten every keyframe, repeat.
    # ------------------------------------------------------------------
    add_seq_edges(a_of[n - 1])
    n_rounds = 3 if "refine" in sys.argv else 0   # optional offline polish
    for rd in range(n_rounds):
        t_rd = time.time()
        added = harvest_edges(range(0, n, 4))
        P = graph.optimize(est[anchors])
        apply_optimized(P, len(anchors) - 1)
        print(f"refine round {rd}: +{added} edges "
              f"({len(graph.edges)} total)  {time.time() - t_rd:.0f}s", flush=True)
        if added == 0:
            break
    dt = time.time() - t0
    print(f"done: {dt:.0f}s total")

    np.savez("intel_traj_loop.npz", est=est, odom=odom, kts=kts,
             closures=np.array(closures))

    # ---- evaluation vs corrected reference
    try:
        ref = C.parse_flaser(path.replace(".log", ".gfs.log"))
    except FileNotFoundError:
        print("no corrected reference log; skipping ATE")
        ref = None
    good = e = None
    if ref is not None:
        rts = np.array([t for _, _, t in ref])
        rxy = np.stack([p[:2] for _, p, _ in ref])
        j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
        good = np.abs(rts - kts[j]) < 0.3
        al = C.align_se2(est[j[good], :2], rxy[good])
        e = np.linalg.norm(al - rxy[good], axis=1)
        print(f"ATE vs corrected log: rmse {np.sqrt((e ** 2).mean()):.3f} m   "
              f"median {np.median(e):.3f} m   max {e.max():.3f} m")

    # ---- figure: final map from corrected poses
    cl, cf = [], []
    for k in range(0, n, 2):
        if store.pts[k] is None or not len(store.pts[k]):
            continue
        wp = store.pts[k] @ S._rot(est[k, 2]).T + est[k, :2]
        cl.append(wp[::2])
        cf.append(np.full(len(wp[::2]), k))
    cl, cf = np.concatenate(cl), np.concatenate(cf)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    ax = axes[0]
    sc = ax.scatter(cl[:, 0], cl[:, 1], c=cf, s=0.3, cmap="viridis")
    ax.plot(est[:, 0], est[:, 1], "r-", lw=0.5, alpha=0.7)
    for c, k in closures:
        ax.plot([est[c, 0], est[k, 0]], [est[c, 1], est[k, 1]], "m-", lw=0.8)
    plt.colorbar(sc, ax=ax, label="keyframe")
    ax.set_title(f"SSP SLAM + loop closure ({len(closures)} closures, magenta)")
    ax.set_aspect("equal")
    ax = axes[1]
    if e is not None:
        ax.plot(np.flatnonzero(good), e)
        ax.set_title("ATE vs RBPF-corrected reference [m]")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("ssp_slam_intel_loop.png", dpi=110)
    print("wrote ssp_slam_intel_loop.png")


if __name__ == "__main__":
    main()
