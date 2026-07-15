"""Object-in-map ROUND 2 (objmap.py frozen at its banked verdict; this
module imports it). The banked wall was room-wide EXTREME-VALUE noise
against thin single-template signal (foil max 0.7-0.9x true even with
exact codes). The two filed levers:

  twostage   stage 1 = whole-QUERY-FRAME place similarity against the
             per-map-kf gridint-bearing snapshots (the EXISTING deploy
             library channel — zero new storage) -> top-K segment
             shortlist; stage 2 = the objmap matched-filter decode
             INSIDE the shortlisted segments' content bboxes only.
             The EV pool shrinks from the room to ~K small boxes, and
             wrong-scene foils must now ALSO win stage 1.
  multiview  query = patch cells gathered from THREE query frames
             (k-2, k, k+2), world-gated to the patch centroid (0.4 m)
             via each frame's own pose+depth (deploy: gyro-chained
             poses; here mocap DIAGNOSTIC, as the whole map is) ->
             Nq x ~2-3 matched signal.

Metrics vs banked single-stage/single-view (int 0.735 AUC / ~0.97 m;
census 0.609 / 1.33 m): detection AUC (same foil protocol, foils run
through the SAME two-stage pipeline), localization error, stage-1
shortlist hit rate.

Usage: python3 -m experiments.objmap2 run [seq]
"""
import sys

import numpy as np

import experiments.objmap as OM
import experiments.lattice3d as L3
import experiments.twomap as TM


def snap_vec(gray, K, Wv):
    """The deploy place-snapshot channel: gridint cells as bearings."""
    ys, xs, w = TM.grid_int(gray)
    b = TM.cam_bearings(K, ys, xs)
    v = (w[:, None] * np.exp(1j * (b @ Wv.T))).sum(0)
    return v / max(np.linalg.norm(v), 1e-12)


def seg_bboxes(feats, groups, margin=0.5):
    out = []
    for g in groups:
        pts = np.concatenate([feats[k]["pw"][feats[k]["ok"]][::5]
                              for k in g])
        lo = np.percentile(pts, 2, axis=0) - margin
        hi = np.percentile(pts, 98, axis=0) + margin
        out.append((lo, hi))
    return out


def mv_query(feats, k, cy, cx, fam, W, span=2, rad=0.4):
    """Multi-view query: base patch on frame k defines the centroid;
    frames k+-span contribute their cells within rad of it (world-gated
    via their own pose+depth). Amplitudes patch-centered per frame."""
    base = OM.make_query(feats[k], fam, W, cy, cx)
    if base is None:
        return None
    _, x0, n0 = base
    q = np.zeros(len(W), complex)
    ntot = 0
    for kk in (k - span, k, k + span):
        if kk < 0 or kk >= len(feats):
            continue
        F = feats[kk]
        ok = F["ok"]
        m = ok & (np.linalg.norm(F["pw"] - x0, axis=2) < rad)
        if m.sum() < 3:
            continue
        pw = F["pw"][m]
        wgt = np.sqrt(F["w"][m])
        app = OM.app_of(F, fam, len(W))[m]
        app = app - app.mean(0)
        q += (wgt[:, None] * np.exp(1j * ((pw - x0) @ W.T)) * app).sum(0)
        ntot += int(m.sum())
    return q, x0, ntot


def run(seq=OM.SEQ, fam="int", topk=3, nq=24):
    import experiments.vision6d as V6
    rng = np.random.default_rng(OM.RNG)
    feats, gt = OM.load_feats(seq)
    gray, depth, _, K = V6.load(seq)
    map_kf, qry_kf = OM.split(len(feats))
    W = OM.W_map(960)
    maps = OM.build_maps(feats, map_kf, fam, W, seg=OM.SEG)
    boxes = seg_bboxes(feats, [g for g, _ in maps])
    bbox_room = OM.world_bbox(feats, map_kf)
    # stage-1 library: per-map-kf place snapshots (existing channel)
    Wv = TM.W_vis_fib()
    snaps = {k: snap_vec(gray[k], K, Wv) for k in map_kf}
    kf2seg = {}
    for si, (g, _) in enumerate(maps):
        for k in g:
            kf2seg[k] = si
    queries = OM.pick_queries(feats, qry_kf, rng, n=nq)

    def stage1(qk):
        vq = snap_vec(gray[qk], K, Wv)
        sims = {}
        for k, v in snaps.items():
            si = kf2seg[k]
            s = abs(np.vdot(vq, v))
            sims[si] = max(sims.get(si, 0.0), s)
        order = sorted(sims, key=lambda s: -sims[s])
        return order[:topk], sims[order[0]]

    def decode_scoped(q, segs):
        best = None
        for si in segs:
            x, sc = OM.decode([maps[si]], q, W, boxes[si])
            if best is None or sc > best[1]:
                best = (x, sc)
        return best

    arms = {}
    for tag in ("1stage-1view", "2stage-1view", "2stage-3view"):
        arms[tag] = dict(err=[], sc=[], hit=0, n=0)
    for k, cy, cx in queries:
        mq = OM.make_query(feats[k], fam, W, cy, cx)
        mv = mv_query(feats, k, cy, cx, fam, W)
        if mq is None or mv is None:
            continue
        q1, x_gt, n1 = mq
        qm, _, nm = mv
        segs, s1 = stage1(k)
        hit = any(np.all(x_gt > boxes[si][0]) and np.all(x_gt < boxes[si][1])
                  for si in segs)
        arms["2stage-1view"].setdefault("s1", []).append(s1)
        x, sc = OM.decode(maps, q1, W, bbox_room)
        arms["1stage-1view"]["err"].append(np.linalg.norm(x - x_gt))
        arms["1stage-1view"]["sc"].append(sc / max(n1, 1))
        x, sc = decode_scoped(q1, segs)
        a = arms["2stage-1view"]
        a["err"].append(np.linalg.norm(x - x_gt))
        a["sc"].append(sc / max(n1, 1))
        a["hit"] += hit
        a["n"] += 1
        x, sc = decode_scoped(qm, segs)
        a = arms["2stage-3view"]
        a["err"].append(np.linalg.norm(x - x_gt))
        a["sc"].append(sc / max(nm, 1))
    # foils through the SAME pipelines (fr1_desk frames + patches)
    ff, fgt = OM.load_feats(OM.FOIL_SEQ)
    fgray, fdepth, _, fK = V6.load(OM.FOIL_SEQ)
    fq = OM.pick_queries(ff, np.arange(len(ff)), rng, n=nq)
    foil = {t: [] for t in arms}
    for k, cy, cx in fq:
        mq = OM.make_query(ff[k], fam, W, cy, cx)
        mv = mv_query(ff, k, cy, cx, fam, W)
        if mq is None or mv is None:
            continue
        q1, _, n1 = mq
        qm, _, nm = mv
        vq = snap_vec(fgray[k], fK, Wv)
        sims = {}
        for kk, v in snaps.items():
            si = kf2seg[kk]
            sims[si] = max(sims.get(si, 0.0), abs(np.vdot(vq, v)))
        order = sorted(sims, key=lambda s: -sims[s])
        segs = order[:topk]
        foil.setdefault("s1", []).append(sims[order[0]])
        _, sc = OM.decode(maps, q1, W, bbox_room)
        foil["1stage-1view"].append(sc / max(n1, 1))
        _, sc = decode_scoped(q1, segs)
        foil["2stage-1view"].append(sc / max(n1, 1))
        _, sc = decode_scoped(qm, segs)
        foil["2stage-3view"].append(sc / max(nm, 1))
    print(f"object-in-map round 2 ({seq.split('_freiburg')[-1]}, fam="
          f"{fam}, {len(arms['1stage-1view']['err'])} queries, top-{topk}"
          f" shortlist; banked 1-stage {fam} AUC "
          f"{'0.735' if fam == 'int' else '0.609'}):")
    a = arms["2stage-1view"]
    print(f"  stage-1 shortlist hit rate: {a['hit']}/{a['n']}")
    s1r, s1f = np.array(a["s1"]), np.array(foil["s1"])
    auc1 = L3._auc(s1r, s1f)
    print(f"  stage-1 frame-sim ALONE as detector: AUC {auc1:.3f}")
    for tag in ("1stage-1view", "2stage-1view", "2stage-3view"):
        e = np.array(arms[tag]["err"])
        sr, sf = np.array(arms[tag]["sc"]), np.array(foil[tag])
        auc = L3._auc(sr, sf)
        # combined detector: z-normed stage-1 + stage-2 evidence (both
        # exist in deploy; a foreign-scene foil must beat BOTH)
        mu, sd = np.concatenate([sr, sf]).mean(),             max(np.concatenate([sr, sf]).std(), 1e-9)
        m1, d1 = np.concatenate([s1r, s1f]).mean(),             max(np.concatenate([s1r, s1f]).std(), 1e-9)
        zc = L3._auc((sr - mu) / sd + (s1r - m1) / d1,
                     (sf - mu) / sd + (s1f - m1) / d1)
        print(f"  {tag}: err med {np.median(e):.3f} p90 "
              f"{np.percentile(e, 90):.3f} | AUC stage2 {auc:.3f} "
              f"combined {zc:.3f}", flush=True)


if __name__ == "__main__":
    seq = sys.argv[2] if len(sys.argv) > 2 else OM.SEQ
    fam = sys.argv[3] if len(sys.argv) > 3 else "int"
    nq = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    run(seq, fam, nq=nq)
