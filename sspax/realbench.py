"""REAL-DATA TRANSFER GATE for the sspax sweep winner (pure numpy — runs
on the no-JAX deploy box; house rule 5: synthetic wins are accepted only
after they hold on real logs — the phi-ladder / hybrid-sampler
precedent).

The sweep's recommended formulation (ring_arc_const_s0.5_r6 directions ×
oct6 0.25-8 m ladder) is run through the REAL venues with the numpy
harnesses, against the banked recipes:

  school   school_run2 full lidar clouds, ROT-SEARCHED place AUC (est
           labels DIAGNOSTIC, the aliased building-scale venue) at two
           budgets — D240 matcher-class and D1920 anchor-class. The
           LADDER-vs-VENUE-SCALE question is the load-bearing one: the
           sweep's oct6 tops out at 8 m on a synthetic room; the banked
           real-data law says building-scale coarse rings (0.5-90.5 m)
           are the place lever (azel 24x5 x coarse16 = 0.928).
  tum      fr3 vision gridint-3D landmarks: place AUC (fwd/rev split) ±
           ring-yaw rotation search (the sweep claims camera rewards
           permutability) + real-motion SO(3) decode (5-20 deg pairs,
           lattice-agnostic nearest-direction permutation grid).
  verify   the deploy small-motion 6x6 verify with ringstag as the
           VISION space vs the tuned camera-range W_vis3d.

Usage: python3 -m sspax.realbench school|tum|verify
"""
import sys

import numpy as np

import sspax.sphere as SPH
import experiments.lattice3d as L3
import experiments.lidarscale as LS
import experiments.vision6d as V6
import experiments.deploy6d as D6

OCT6 = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
RINGKW = dict(spacing="arc", apportion="const", stagger=0.5, n_rings=6)


def W_ring(nd, lams):
    dirs = SPH.dirs_ring(nd, **RINGKW)
    return np.concatenate([(2 * np.pi / lam) * dirs for lam in lams]), nd


def ring_rotsim(V, nd, nlam, n_step=24):
    """rot-searched |cos| for the staggered-ring lattice: per-ring snap
    permutations over n_step yaw angles spanning the full circle."""
    mats = [np.abs(V @ V.conj().T)]
    for k in range(1, n_step):
        th = k * 2 * np.pi / n_step
        perm, sgn, _ = SPH.yaw_perm(nd, th, n_lams=nlam, **RINGKW)
        mats.append(np.abs(V[:, perm] @ V.conj().T))
    return np.max(np.stack(mats), 0)


def bench_school(run="school_run2"):
    pick, kts, pose, kind, pr = LS._setup(run)
    full, _ = L3.load_clouds(run, kts)
    keep = [i for i, c in enumerate(full) if c is not None and len(c) > 300]
    pose_k = pose[keep]
    (si, sj), (di, dj) = L3._pairs(pose_k, **pr)
    adj = np.arange(len(keep) - 1)
    print(f"TRANSFER school place ({run}, labels={kind}, {len(keep)} "
          f"frames, rot-searched 24; banked azel refs: D240 0.700 / "
          f"D1920-coarse 0.928):")
    arms = [
        ("ring-oct6      D240 ", *W_ring(40, OCT6)),
        ("ring-coarse16  D1920", *W_ring(120, LS.LAMC16)),
        ("ring-oct6      D1920", *W_ring(320, OCT6)),
        ("azel-oct6      D1920", LS.azel_W(24, [-40, -20, 0, 20, 40],
                                           OCT6[:5] + [8.0]), None),
    ]
    for name, W, nd in arms:
        V = np.stack([L3.encode(W, full[i]) for i in keep])
        if nd is not None:
            S = ring_rotsim(V, nd, len(W) // nd)
        else:
            S = LS.rot_sim(V, W, 24)
        auc = L3._auc(S[si, sj], S[di, dj])
        rep = L3._auc(S[adj, adj + 1], S[di, dj])
        print(f"  {name} (D{len(W):4d}): AUC {auc:.3f}  adj {rep:.3f}",
              flush=True)


def bench_tum(seq=V6.SEQ):
    gray, depth, gt, K = V6.load(seq)
    pos = gt[:, :3]
    yaw = np.array([np.arctan2(V6.quat_R(q)[1, 0], V6.quat_R(q)[0, 0])
                    for q in gt[:, 3:7]])
    n = len(gray)
    d3 = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    ii, jj = np.triu_indices(n, 1)
    gap = np.abs(ii - jj) >= 30
    same = (d3[ii, jj] < 0.5) & gap
    diff = (d3[ii, jj] > 2.0) & (d3[ii, jj] < 6.0)
    si, sj, di, dj = ii[same], jj[same], ii[diff], jj[diff]
    dyaw = np.abs(np.arctan2(np.sin(yaw[si] - yaw[sj]),
                             np.cos(yaw[si] - yaw[sj])))
    fwd = dyaw < np.deg2rad(60)
    F = [V6.feats(g, K, "gridint", d) for g, d in zip(gray, depth)]
    print(f"TRANSFER tum place+rot ({seq.split('_freiburg')[-1]}, {n} kf, "
          f"mocap labels; banked azel3d gridint-3D place 0.794 rev "
          f"0.791):")
    Wr, nd = W_ring(60, OCT6)                     # the semantic default
    arms = {
        "ring-oct6 D360 raw": (Wr, nd),
        "azel3d house D240 ": (L3.make_lattices()["azel3d"], None),
    }
    for name, (W, nd_) in arms.items():
        V = np.stack([L3.encode(W, *f) for f in F])
        S = np.abs(V @ V.conj().T)
        if nd_ is not None:                        # + ring-yaw search
            Sr = ring_rotsim(V, nd_, len(W) // nd_)
        auc = L3._auc(S[si, sj], S[di, dj])
        ar = L3._auc(S[si, sj][~fwd], S[di, dj]) if (~fwd).any() else \
            float("nan")
        line = f"  {name}: AUC {auc:.3f} (rev {ar:.3f})"
        if nd_ is not None:
            aucr = L3._auc(Sr[si, sj], Sr[di, dj])
            arr = L3._auc(Sr[si, sj][~fwd], Sr[di, dj])
            line += f" | rot-searched {aucr:.3f} (rev {arr:.3f})"
        print(line, flush=True)
    # real-motion SO(3) decode, lattice-agnostic permutation grid
    rp = []
    for i in range(n):
        for j in range(i + 1, min(i + 12, n)):
            Rt, tt = V6._rel_pose(gt[i], gt[j])
            ang = np.degrees(np.arccos(np.clip((np.trace(Rt) - 1) / 2,
                                               -1, 1)))
            if np.linalg.norm(tt) < 0.15 and 5 <= ang <= 20:
                rp.append((i, j, Rt))
    rp = rp[:: max(1, len(rp) // 30)][:30]
    degs = np.arange(-20, 21, 5)
    Rg = [L3._axis_rot("yaw", y) @ L3._axis_rot("pitch", p)
          @ L3._axis_rot("roll", r)
          for y in degs for p in degs for r in degs]
    print(f"  real-motion SO(3) ({len(rp)} pairs, banked azel3d 8.4 / "
          f"fib3d 6.7 deg):")
    for name, (W, _) in arms.items():
        perms = [L3._perm_of(W, R)[:2] for R in Rg]
        errs = []
        for i, j, Rt in rp:
            v0 = L3.encode(W, *F[i])
            v1 = L3.encode(W, *F[j])
            bs, best = -1, 0
            for gi, (perm, sgn) in enumerate(perms):
                s = np.abs(np.vdot(v1, L3._apply_perm(v0, perm, sgn)))
                if s > bs:
                    bs, best = s, gi
            errs.append(np.degrees(np.arccos(np.clip(
                (np.trace(Rg[best] @ Rt.T) - 1) / 2, -1, 1))))
        print(f"    {name}: med {np.median(errs):.1f} p90 "
              f"{np.percentile(errs, 90):.1f}", flush=True)


def bench_verify(seq=V6.SEQ):
    gray, depth, gt, K = V6.load(seq)
    pairs = D6.small_pairs(gt)
    print(f"TRANSFER verify ({seq.split('_freiburg')[-1]}, {len(pairs)} "
          f"small-motion pairs; banked W_vis3d float 1.32 deg / 31.8 mm):")
    arms = {
        "W_vis3d 0.35-2.8 D240": D6.W_vis3d(),
        "ring-oct6       D240 ": W_ring(40, OCT6)[0],
        "ring-vislams    D240 ": W_ring(60, D6.VIS_LAMS)[0],
    }
    for name, W in arms.items():
        er, et = [], []
        for i, j, Rt, tt in pairs:
            P0, w0 = V6.feats(gray[i], K, "gridint", depth[i])
            P1, w1 = V6.feats(gray[j], K, "gridint", depth[j])
            v0 = V6._enc_raw(W, P0, w0)
            v1 = V6._enc_raw(W, P1, w1)
            Dm = np.concatenate([V6._deriv_axes(W, P0, w0),
                                 V6._deriv_transl(W, P0, w0)], 1)
            th = D6.solve66(v0, Dm, v1)
            er.append(D6.gyro_err(th, Rt))
            et.append(np.linalg.norm(th[3:6] - tt))
        print(f"  {name}: rot med {np.median(er):.2f} p90 "
              f"{np.percentile(er, 90):.2f} | transl med "
              f"{np.median(et)*1000:.1f} mm", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "school"
    dict(school=bench_school, tum=bench_tum,
         verify=bench_verify)[cmd]()
