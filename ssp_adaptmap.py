"""Adaptive content-keyed map clustering for bounded SSP SLAM (experiment).

Subclass of BoundedSLAM. The shipped map keys segments by robot pose into a
fixed 2 m grid, capped at CELL_CAP per cell (memory O(area), same spend on a
bland corridor as on a distinctive junction). This experiment replaces the
"always create a new patch per anchor" rule with an INFORMATION-DRIVEN one:
when a new anchor would open a patch, compute the SSP similarity (normalized
world-frame inner product, MAIN band) of its content to nearby recent patches;
if the max similarity exceeds tau the content is REDUNDANT -> bundle-merge it
into that patch (no new vector); if below tau it is NOVEL -> open a new patch.

Guard against the scale-arrays / smeared-closure failure: merge candidates are
restricted to SAME-PASS anchors (within merge_age_a anchors) and a spatial
neighborhood (merge_radius). A corridor revisited at a different drift state
(far in anchor index) is NEVER merged into the earlier pass — it opens a fresh
patch and is linked only through gated loop closures, exactly as shipped.

The shipped cell-cap is kept as an across-pass backstop (cell_cap, default 6),
so tau -> +inf reproduces the shipped map bit-for-bit and merging can only
REMOVE patches the cap would have kept: memory is monotone in tau and the
sweep is a controlled A/B against the shipped single point.

Usage:
  python3 ssp_adaptmap.py probe   [log]        # similarity distribution
  python3 ssp_adaptmap.py sweep   [log]        # tau sweep, one log
  python3 ssp_adaptmap.py all                  # full table, all four logs
  python3 ssp_adaptmap.py selftest             # invariants
"""

import sys
import time

import numpy as np

import ssp_slam as S
import ssp_slam_carmen as C
import ssp_slam_loop as L
import ssp_bounded as B
from ssp_bounded import ANCHOR, CELL

VALID_MAX = 40.0


class AdaptiveMapSLAM(B.BoundedSLAM):
    """BoundedSLAM with adaptive content-keyed patch clustering."""

    def __init__(self, *a, tau=0.97, merge_radius=3.0, merge_age_a=20,
                 use_adaptive=True, cell_cap=6, **kw):
        super().__init__(*a, **kw)
        self.tau = tau                    # similarity above which content merges
        self.merge_radius = merge_radius  # spatial neighborhood [m]
        self.merge_age_a = merge_age_a    # same-pass window [anchors]
        self.use_adaptive = use_adaptive
        self.cell_cap = cell_cap          # across-pass backstop (0 disables)
        self.fold_target = {}             # anchor_id -> owner anchor it folds into
        self.owner_last_aid = {}          # owner -> highest anchor folded into it
        self.n_merge = 0
        self.merge_sims = []              # diagnostics: chosen similarity per anchor

    # --- content-novelty decision -----------------------------------------
    def _choose_target(self, aid, pts, w, est):
        """Return the anchor whose patch this anchor's content folds into:
        itself (open a new patch) or a redundant same-pass neighbour."""
        if not self.use_adaptive or not self.segvec or len(pts) < 20:
            return aid
        p = est[:2]
        cands = [m for m in self.segvec
                 if 0 < aid - m <= self.merge_age_a
                 and np.linalg.norm(self.anchors[m][:2] - p) < self.merge_radius]
        if not cands:
            return aid
        PW = pts @ S._rot(est[2]).T + est[:2]           # world-frame points
        vnew = np.exp(1j * (PW @ L.W[L.MAIN].T)).T @ w   # world content (MAIN)
        nn = np.linalg.norm(vnew) + 1e-12
        best_sim, best_m = -1.0, aid
        for m in cands:
            vm = self.world_vec_seg(m)[L.MAIN]
            sim = float((np.conj(vm) @ vnew).real
                        / (np.linalg.norm(vm) * nn + 1e-12))
            if sim > best_sim:
                best_sim, best_m = sim, m
        self.merge_sims.append(best_sim)
        if best_sim > self.tau:
            self.n_merge += 1
            return best_m
        return aid

    # --- frontend recency by last-fold, not owner age ---------------------
    def local_bundle(self, center, radius=8.0):
        """Same as shipped, but an owner counts as RECENT if content was folded
        into it recently (owner_last_aid), not merely by its own anchor index —
        merged content concentrates into fewer owners, some older than the raw
        recency cut-off yet still holding this pass's latest folds."""
        lo = len(self.anchors) - 1 - self.recent_aids
        Bd = np.zeros(L.W.shape[0], complex)
        for aid in self.segvec:
            if self.owner_last_aid.get(aid, aid) >= lo \
                    and np.linalg.norm(self.anchors[aid][:2] - center) < radius:
                Bd += self.world_vec_seg(aid)
        return Bd

    # --- lifecycle: copy of BoundedSLAM.add_keyframe with adaptive fold ----
    def add_keyframe(self, pts, w, guess):
        self.k += 1
        k = self.k
        if k % ANCHOR == 0:
            self.anchors.append(guess.copy())
        aid = len(self.anchors) - 1
        est = guess
        fell_back = True
        if k > 0 and len(pts) >= 20:
            Bl = self.local_bundle(guess[:2])[L.MAIN]
            if np.abs(Bl).sum() > 0:
                cand = self.matcher.match(Bl, pts, w, guess)
                cand[2] = S.wrap(cand[2])
                if np.linalg.norm(cand[:2] - guess[:2]) < 0.45 \
                        and abs(S.wrap(cand[2] - guess[2])) < np.deg2rad(11):
                    est = cand
                    fell_back = False
                    W01 = L.W[:2 * L.N_ANG]
                    PW = pts @ S._rot(est[2]).T + est[:2]
                    sv01 = np.exp(1j * (PW @ W01.T)).T @ w
                    B01 = Bl[:2 * L.N_ANG].reshape(2, L.N_ANG)
                    s01 = sv01.reshape(2, L.N_ANG)
                    c01 = float(np.mean(
                        (np.conj(B01) * s01).sum(1).real
                        / (np.linalg.norm(B01, axis=1)
                           * np.linalg.norm(s01, axis=1) + 1e-12)))
                    self.coh_ref = c01 if self.coh_ref is None \
                        else 0.95 * self.coh_ref + 0.05 * c01
                    est = self._frontend_accept(guess, cand, c01)
        self._span_fallback = getattr(self, "_span_fallback", False) or fell_back
        if k % ANCHOR == 0:
            self.anchors[aid] = est.copy()
        rel = L.se2_mul(L.se2_inv(self.anchors[aid]), est)
        self.kf_ref.append((aid, rel))

        # ---- adaptive content-keyed consolidation -------------------------
        if len(pts):
            if k % ANCHOR == 0:            # decide fold target once per anchor
                self.fold_target[aid] = self._choose_target(aid, pts, w, est)
            target = self.fold_target.get(aid, aid)
            relt = L.se2_mul(L.se2_inv(self.anchors[target]), est)
            PA = pts @ S._rot(relt[2]).T + relt[:2]      # target-frame points
            A = np.exp(1j * (PA @ L.W.T))
            v = A.T @ w
            Cx = PA[:, 0:1] * L.W[:, 1] - PA[:, 1:2] * L.W[:, 0]
            vd = (1j * Cx * A).T @ w
            if target == aid and aid not in self.segvec:   # open a new patch
                self.segvec[aid] = np.zeros(L.W.shape[0], self.store_dtype)
                self.segder[aid] = np.zeros(L.W.shape[0], self.store_dtype)
                if self.cell_cap:          # across-pass spatial backstop
                    cell = tuple((self.anchors[aid][:2] // CELL).astype(int))
                    self.cells.setdefault(cell, []).append(aid)
                    if len(self.cells[cell]) > self.cell_cap:
                        drop = self.cells[cell].pop(0)
                        self.segvec.pop(drop, None)
                        self.segder.pop(drop, None)
                        self.owner_last_aid.pop(drop, None)
                        self.n_evict += 1
                        if self.windowed:
                            self._maybe_retire(drop)
            self.segvec[target] += v
            self.segder[target] += vd
            self.owner_last_aid[target] = aid

        if aid > 0 and k % ANCHOR == 0:   # sequential edge between anchors
            Z = L.se2_mul(L.se2_inv(self.anchors[aid - 1]), self.anchors[aid])
            if getattr(self, "_span_fallback", False):
                st, sr = 0.10, np.deg2rad(1.5)
            else:
                st, sr = 0.03, np.deg2rad(0.3)
            self._span_fallback = False
            self.edges.append((aid - 1, aid, Z, 1 / st, 1 / sr, "seq"))

        if k % self.attempt_every == 0:
            self.try_constraint(pts, w)
        if k % self.relax_every == 0 and self.dirty:
            self.relax()
            return self.pose_of(k)
        return est


# --------------------------------------------------------------------------
# Evaluation harness (numbers only; no plots)
# --------------------------------------------------------------------------

def _ref_by_timestamp(keys, path, kts):
    ref = C.parse_flaser(path.replace(".log", ".gfs.log"))
    rts = np.array([t for _, _, t in ref])
    rxy = np.stack([p[:2] for _, p, _ in ref])
    j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
    good = np.abs(rts - kts[j]) < 0.3
    # map ref->kf; return per-kf ref by nearest matched, plus mask over kf
    kf_ref = np.full((len(keys), 2), np.nan)
    kf_ok = np.zeros(len(keys), bool)
    for m in np.flatnonzero(good):
        kf_ref[j[m]] = rxy[m]
        kf_ok[j[m]] = True
    return kf_ref, kf_ok


def _ref_by_identity(keys, path, kts):
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
    ok = (kts >= gts[0]) & (kts <= gts[-1])
    return np.stack([rx, ry], 1), ok


def eval_ate(fin_xy, ref, ok):
    al = C.align_se2(fin_xy[ok], ref[ok])
    e = np.linalg.norm(al - ref[ok], axis=1)
    return float(np.sqrt((e ** 2).mean())), float(np.median(e)), float(e.max())


def traj_length(fin_xy):
    return float(np.linalg.norm(np.diff(fin_xy, axis=0), axis=1).sum())


def load(path, n_max=None):
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)
    if n_max:
        keys = keys[:n_max]
    return keys


def run_one(keys, path, tag="", n_max=None, **cfg):
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    slam = AdaptiveMapSLAM(robust=True, attempt_every=4, relax_every=25,
                           gap_kf=300, recent_aids=12, **cfg)
    slam.store_dtype = np.complex64
    est = np.zeros((n, 3))
    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < VALID_MAX, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = opose if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
    if slam.dirty:
        slam.relax()
    secs = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])

    is_mit = "MIT" in path or "mit" in path
    if is_mit:
        ref, ok = _ref_by_identity(keys, path, kts)
    else:
        ref, ok = _ref_by_timestamp(keys, path, kts)
    rmse, med, mx = eval_ate(fin[:, :2], ref, ok)
    length = traj_length(fin[:, :2])
    nseg = len(slam.segvec)
    mb = slam.memory_kb() / 1024.0
    nloop = sum(1 for e in slam.edges if e[5] == "loop")
    return dict(tag=tag, rmse=rmse, med=med, mx=mx, nseg=nseg, mb=mb,
                segpm=nseg / max(length, 1e-9), length=length, nloop=nloop,
                merge=slam.n_merge, evict=slam.n_evict, veto=slam.n_veto,
                innov=slam.n_innov_rej, secs=secs, n=n,
                sims=np.array(slam.merge_sims))


def print_row(r):
    print(f"  {r['tag']:<22} ATE {r['rmse']:6.3f} m (med {r['med']:.3f}) "
          f"| segs {r['nseg']:5d}  {r['mb']:5.2f} MB  {r['segpm']:.3f} seg/m "
          f"| merges {r['merge']:5d}  evict {r['evict']:4d}  loops {r['nloop']:3d} "
          f"| {r['secs']:.0f}s", flush=True)


LOGS = {"intel": "data/intel.log", "fr079": "data/fr079.log",
        "fr101": "data/fr101.log", "mit": "data/mit.log"}
TAUS = [1.01, 0.90, 0.85, 0.80, 0.75, 0.70]   # 1.01 == never merge (baseline)


def cmd_probe(log):
    path = LOGS.get(log, log)
    keys = load(path)
    print(f"== probe {path}: {len(keys)} keyframes")
    r = run_one(keys, path, tag="tau=0.90", tau=0.90)
    s = r["sims"]
    print(f"  candidate-anchor similarities: n={len(s)}  "
          f"min {s.min():.3f}  p10 {np.percentile(s,10):.3f}  "
          f"p50 {np.percentile(s,50):.3f}  p90 {np.percentile(s,90):.3f}  "
          f"max {s.max():.3f}")
    for t in [0.99, 0.97, 0.94, 0.90, 0.85, 0.80]:
        print(f"  tau={t:.2f} -> would merge {int((s > t).sum())}/{len(s)} "
              f"({100*(s>t).mean():.0f}%)")


def cmd_sweep(log, n_max=None):
    path = LOGS.get(log, log)
    keys = load(path, n_max)
    print(f"== sweep {path}: {len(keys)} keyframes"
          + (f" (capped {n_max})" if n_max else ""))
    base = run_one(keys, path, tag="shipped (no merge)", use_adaptive=False)
    print_row(base)
    for t in TAUS[1:]:
        r = run_one(keys, path, tag=f"adaptive tau={t:.2f}", tau=t)
        print_row(r)


def cmd_all():
    for log in ["intel", "fr079", "fr101", "mit"]:
        cmd_sweep(log)
        print()


def cmd_selftest():
    """use_adaptive=False must reproduce shipped BoundedSLAM bit-for-bit."""
    path = LOGS["fr101"]
    keys = load(path, n_max=400)
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    odom = np.stack([k[1] for k in keys])

    def drive(slam):
        est = np.zeros((n, 3))
        for k, (r, opose, ts) in enumerate(keys):
            rr = np.where(r < VALID_MAX, r, np.inf)
            pts, w, _ = S.scan_to_samples(rr, beam)
            guess = opose if k == 0 else L.se2_mul(
                est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
            est[k] = slam.add_keyframe(pts, w, guess)
        if slam.dirty:
            slam.relax()
        return np.stack([slam.pose_of(k) for k in range(n)])

    s0 = B.BoundedSLAM(robust=True, attempt_every=4, relax_every=25,
                       gap_kf=300, recent_aids=12)
    s0.store_dtype = np.complex64
    s1 = AdaptiveMapSLAM(robust=True, attempt_every=4, relax_every=25,
                         gap_kf=300, recent_aids=12, use_adaptive=False)
    s1.store_dtype = np.complex64
    f0, f1 = drive(s0), drive(s1)
    d = np.abs(f0 - f1).max()
    print(f"selftest: max |pose diff| shipped-vs-adaptive(off) = {d:.2e}  "
          f"segs {len(s0.segvec)} vs {len(s1.segvec)}  "
          f"-> {'PASS' if d < 1e-9 and len(s0.segvec)==len(s1.segvec) else 'FAIL'}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    arg = sys.argv[2] if len(sys.argv) > 2 else "intel"
    if mode == "probe":
        cmd_probe(arg)
    elif mode == "sweep":
        cmd_sweep(arg)
    elif mode == "all":
        cmd_all()
    else:
        cmd_selftest()
