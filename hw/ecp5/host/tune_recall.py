#!/usr/bin/env python3
"""Optimize the cam-VSA map's per-object recall on SYNTHETIC data
(user directive 2026-07-16). Synthetic scenes with known object classes/
positions drive the REAL CamMap code (webvis.py) — GT scores only
(anti-oracle). Knobs, each from a banked law:

  centroid vs INSTANCE-SPLIT binding   (same-class multiplicity thread)
  target-class significance boost w    (significance-buys-capacity law)
  fusion across kf vectors max vs SUM  (persistence/re-observation)
  support gating (marks need >=S kf)   (one-view flicker suppression)
  z threshold                          (4-sigma discipline)

crossed with bit-flip noise (measured head stability ~0.75 adj -> up to
25% flips) and position noise (bearing-cell quantization + nominal
extrinsics). Metric: recall = TARGET-class GT objects with a mark
within 0.6 m; precision = marks within 0.6 m of a same-class GT object.

  python3 hw/ecp5/host/tune_recall.py [seeds]
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import importlib.util

spec = importlib.util.spec_from_file_location("wv", HERE / "webvis.py")
wv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wv)

W = wv.W_MAIN
FOV = np.deg2rad(69.0)


def scene(seed, n_cls=8, n_per=3, n_kf=90):
    rng = np.random.default_rng(seed)
    codes = rng.random((n_cls, 32)) > 0.5
    objs = []                                    # (cls, x, y)
    for c in range(n_cls):
        for _ in range(n_per):
            objs.append((c, rng.uniform(1, 11), rng.uniform(1, 9)))
    # wandering trajectory
    p = np.array([6.0, 5.0, 0.0])
    traj = []
    for _ in range(n_kf):
        p[2] += rng.normal(0, 0.25)
        p[0] = np.clip(p[0] + 0.18 * np.cos(p[2]), 0.8, 11.2)
        p[1] = np.clip(p[1] + 0.18 * np.sin(p[2]), 0.8, 9.2)
        traj.append(p.copy())
    return codes, objs, traj, rng


def build(cm, codes, objs, traj, rng, p_bit, sig_p, split, boost,
          targets):
    seen = set()
    """Feed observations through CamMap's OWN binding math (mirrors
    ingest: per-class groups, optional instance split, boost)."""
    for c in range(len(codes)):                  # fixed clusters = classes
        cm.cents.append(codes[c].astype(float))
        cm.counts.append(999)
        cm.example.append((0, 0, 0))
    cm.boost = {c: boost for c in targets} if boost != 1.0 else {}
    for k, pose in enumerate(traj):
        obs = []
        for c, ox, oy in objs:
            d = np.hypot(ox - pose[0], oy - pose[1])
            brg = wv.S.wrap(np.arctan2(oy - pose[1], ox - pose[0])
                            - pose[2])
            if d < 8.0 and abs(brg) < FOV / 2:
                px = ox + rng.normal(0, sig_p)
                py = oy + rng.normal(0, sig_p)
                obs.append((c, px, py))
                seen.add((c, ox, oy))
        if not obs:
            continue
        v = np.zeros(len(W), complex)
        byc = {}
        for c, x, y in obs:
            byc.setdefault(c, []).append((x, y))
        for c, ps in byc.items():
            bits = codes[c] ^ (rng.random(32) < p_bit)   # noisy obs code
            w_ = cm.boost.get(c, 1.0)
            groups = []
            if split:
                for pp in ps:
                    for g in groups:
                        if (pp[0] - g[0][0]) ** 2 \
                                + (pp[1] - g[0][1]) ** 2 < 1.44:
                            g.append(pp)
                            break
                    else:
                        groups.append([pp])
            else:
                groups = [ps]
            for g in groups:
                pm = np.mean(g, axis=0)
                v += w_ * cm._amp(bits) * np.exp(
                    1j * ((pm - pose[:2]) @ W.T))
        cm.vecs[k] = (pose.copy(), v)
    return seen


def evaluate(cm, codes, objs, targets, z_th, sup_min, seen=None):
    cm.Z_TH, cm.SUP_MIN = z_th, sup_min
    lo, hi = np.array([0.0, 0.0]), np.array([12.0, 10.0])
    rec = prec = nq = 0
    for c in targets:
        res = cm._zquery(np.conj(cm._amp(codes[c])), cm._rand_amp(),
                         lo, hi, c, ngrid=96)
        gts = [(x, y) for cc, x, y in objs if cc == c
               and (seen is None or (cc, x, y) in seen)]
        if not gts:
            continue
        hits = sum(any((mx - x) ** 2 + (my - y) ** 2 < 0.36
                       for mx, my, _ in res["marks"]) for x, y in gts)
        good = sum(any((mx - x) ** 2 + (my - y) ** 2 < 0.36
                       for x, y in gts) for mx, my, _ in res["marks"])
        rec += hits / len(gts)
        prec += good / max(len(res["marks"]), 1)
        nq += 1
    if nq == 0:
        return np.nan, np.nan
    return rec / nq, prec / nq


def arm(name, seeds, p_bit=0.15, sig_p=0.2, split=False, boost=1.0,
        fuse="max", z_th=4.0, sup_min=1):
    R, P = [], []
    for s in range(seeds):
        codes, objs, traj, rng = scene(s)
        cm = wv.CamMap()
        cm.FUSE = fuse
        targets = [0, 1]
        seen = build(cm, codes, objs, traj, rng, p_bit, sig_p, split,
                     boost, targets)
        r, p = evaluate(cm, codes, objs, targets, z_th, sup_min, seen)
        R.append(r)
        P.append(p)
    print(f"  {name:38s} recall {np.nanmean(R):.3f}+-{np.nanstd(R):.3f}  "
          f"prec {np.nanmean(P):.3f}  (n={np.isfinite(R).sum()})")
    return float(np.nanmean(R)), float(np.nanmean(P))


def main(seeds=8):
    print(f"TARGET-CLASS map recall, synthetic (seeds={seeds}; "
          f"2 target + 6 bg classes x3 objects; GT scores only):")
    print("-- knob ladder (bit-flip 0.15, sig_p 0.20 m) --")
    arm("base: centroid, max-fuse, z4", seeds)
    arm("+instance split", seeds, split=True)
    arm("+sum fusion", seeds, split=True, fuse="sum")
    arm("+support>=2", seeds, split=True, fuse="sum", sup_min=2)
    arm("+target boost x2", seeds, split=True, fuse="sum", sup_min=2,
        boost=2.0)
    arm("+target boost x3", seeds, split=True, fuse="sum", sup_min=2,
        boost=3.0)
    arm("boost x2, z3", seeds, split=True, fuse="sum", sup_min=2,
        boost=2.0, z_th=3.0)
    arm("boost x2, z5", seeds, split=True, fuse="sum", sup_min=2,
        boost=2.0, z_th=5.0)
    print("-- max-fusion factorial (sum lost above) --")
    arm("max+split+boost x2", seeds, split=True, boost=2.0)
    arm("max+split+boost x3", seeds, split=True, boost=3.0)
    arm("max+split+boost x2, z3", seeds, split=True, boost=2.0, z_th=3.0)
    arm("SANITY: no noise, max+split", seeds, p_bit=0.0, sig_p=0.05,
        split=True)
    print("-- noise robustness (winner: max+split+boost x3) --")
    for pb, sp in ((0.10, 0.15), (0.25, 0.20), (0.15, 0.35), (0.25, 0.35)):
        arm(f"win @ bit{pb} pos{sp}", seeds, p_bit=pb, sig_p=sp,
            split=True, boost=3.0)
    print("-- control: same config, boost OFF --")
    arm("winner config, boost 1.0", seeds, split=True, boost=1.0)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
