"""Spatially-anchored per-scale submap arrays (SA).

The matched band (rings lam 0.25/0.5/1/2) reworked as per-ring ARRAYS OF
CELLS, each cell a small graph-anchored submap:

1. PER-RING CELL ARRAYS. Ring s gets cells of size c*lam_s (c=8 default) on a
   staggered dual grid (second grid offset half a cell; a write adds 0.5 of a
   sample to each covering cell, queries sum every gathered cell, so total
   mass is 1 and content near a grid-0 edge is centered in a grid-1 cell —
   bounded anchor lever without seams).
2. HIT-KEYED WRITES. Every occlusion-filtered sample is placed at its WORLD
   position (current estimate) and accumulated, per ring the range gate
   allows (lam >= beta*r*dtheta, beta=3, r = robot-frame range), into the
   ring's cell containing THE HIT — not the robot. Content therefore lives
   where the geometry is: a wall seen across open space is queryable from
   anywhere near the WALL, which anchor-keyed segment gathering (content
   keyed at the observing robot's anchor) structurally misses.
3. GRAPH ANCHORING. A cell's frame attaches to the trajectory anchor node of
   its FIRST writer: stored as (writer_aid, rel pose of the cell origin in
   that anchor's frame at creation; origin = cell center, heading 0 at
   creation). Cell world pose = current graph pose of writer_aid composed
   with rel — graph corrections move cells RIGIDLY, content is never
   re-encoded. Each cell stores a 60-angle block + a d/dtheta derivative
   block (rotation about the cell origin); queries transform to world via
   translation phase + nearest-3-deg exact permutation + first-order
   derivative correction (HY4 semantics, per cell). Cell-origin anchoring
   caps the rotation lever at the cell radius (segments carry sensor-range
   levers).
4. CROSS-PASS WRITE GATE. A keyframe may write into a cell last written by a
   different pass (last_writer older than gap_kf; pass epoch = aid //
   (gap_kf/ANCHOR), i.e. aid//60 at the CARMEN gap 300) ONLY if a verified
   loop constraint was accepted within the last ~50 kf within ~8 m (the
   write is then graph-consistent at write time). Otherwise the write opens
   a PARALLEL per-pass cell (key extended by the pass epoch). Writes are
   permanent; every historical failure of world-frame maps entered at write
   time — hence discipline on writes, liberality on queries:
5. QUERIES ARE FREE (VSA superposition). The frontend local map liberally
   SUMS every cell within radius — both staggered grids, base and ALL
   parallel pass layers, no merge step, no relevance logic (over-inclusion
   costs only sqrt(N) crosstalk; local queries gather far fewer than the
   ~400-vector capacity knee from the aliasing/capacity study). Merge-on-
   closure was DROPPED entirely — the two-layer additive query replaces it.
   LOOP CONSTRAINTS keep evidence independence: matched against a SINGLE
   old pass's cell set (cells the current epoch never wrote, grouped by
   last-writer epoch, nearest group only), Z taken against the dominant
   cell's frame anchor at its CURRENT graph pose (cells ride with the
   graph — no write-time snapshot needed). Backend inherited verbatim from
   BoundedSLAM (seq edges, innovation gate, soft coherence response,
   IRLS + LOO, TRF max_nfev=30). Coarse relocalization band: out of scope
   (matched band only).
6. EVICTION. Per-cell content-mass cap (sum of |w| written; saturated cells
   stop accepting writes) and a per-(ring, 8 m region) cell-count cap
   (stalest cell evicted, parallels first) keep the store O(area).

Usage:
  python3 -m experiments.scale_arrays selftest
  python3 -m experiments.scale_arrays cell <world> <seed> [nogate|recent]
  python3 -m experiments.scale_arrays bench [quick]        # paired vs G1 + HY4
  python3 -m experiments.scale_arrays carmen [path]        # mirrors ssp_bounded_carmen
"""

import json
import sys
import time

import numpy as np

import sspslam.encoder as S
import sspslam.lattice as L
import sspslam.bounded as B
from sspslam.worlds import WORLDS
from sspslam.bench import multiloop_traj, FOV_BEAMS, BEAM, scan_at

ANCHOR = B.ANCHOR
NA = L.N_ANG
LAMS = L.LAMS[:4]                       # matched band only
WR = [L.W[r * NA:(r + 1) * NA] for r in range(4)]


def _perm(vec, m):
    """Exact rotation of one ring's 60-angle block by m lattice steps."""
    ext = np.concatenate([vec, np.conj(vec)])
    return ext[(np.arange(NA) - m) % (2 * NA)]


class Cell:
    __slots__ = ("v", "vd", "aid", "rel", "last_writer", "epochs",
                 "mass", "key", "ring")

    def __init__(self, ring, key, aid, rel):
        self.v = np.zeros(NA, complex)     # angle block (cell frame)
        self.vd = np.zeros(NA, complex)    # d/dtheta about the cell origin
        self.aid = aid                     # graph frame: first writer's anchor
        self.rel = rel                     # cell-origin pose in anchors[aid]
        self.last_writer = aid
        self.epochs = set()                # pass epochs that wrote here
        self.mass = 0.0                    # sum |w| written (eviction budget)
        self.key = key                     # (g,ix,iy) or (g,ix,iy,epoch)
        self.ring = ring


class ScaleArraySLAM(B.BoundedSLAM):
    """BoundedSLAM with the segment store replaced by per-scale cell arrays.

    Inherits the whole backend (seq edges, innovation gate, soft coherence
    response, IRLS + LOO, TRF max_nfev=30) and both matchers; overrides map
    writes/queries and the loop-constraint frontend."""

    def __init__(self, c_cell=8.0, beta=3.0, mass_cap=500.0,
                 region_s=8.0, region_factor=3.0,
                 gate_recent_kf=50, gate_radius=8.0, xpass_gate=True,
                 front_mode="all", robust=True, attempt_every=3,
                 relax_every=5, gap_kf=B.GAP, recent_aids=12):
        super().__init__(robust=robust, attempt_every=attempt_every,
                         relax_every=relax_every, gap_kf=gap_kf,
                         recent_aids=recent_aids)
        self.c_cell = c_cell
        self.beta = beta
        self.cell_s = [c_cell * lam for lam in LAMS]     # 2/4/8/16 m at c=8
        self.dtheta_beam = np.pi / FOV_BEAMS   # bench; CARMEN: pi/n_beams
        self.mass_cap = mass_cap
        self.region_s = region_s
        self.region_cap = [max(8, int(region_factor * 2
                                      * max(1.0, region_s / s) ** 2))
                           for s in self.cell_s]
        self.gate_recent_kf = gate_recent_kf
        self.gate_radius = gate_radius
        self.xpass_gate = xpass_gate       # False = ungated ablation (R1-like)
        # front_mode "all": liberal additive query (default) — every cell in
        # radius, all pass layers. "recent": recency-restricted control arm
        # (shipped BoundedSLAM law) to measure the crosstalk cost of "all".
        self.front_mode = front_mode
        self.epoch_aids = max(1, gap_kf // ANCHOR)   # 60 at CARMEN gap 300
        self.rings = [dict() for _ in range(4)]      # key -> Cell
        self.regions = [dict() for _ in range(4)]    # region key -> [keys]
        self.accepts = []                  # (k, xy) of accepted closures
        self.n_sat = self.n_parallel = 0
        self.n_gate_open = 0               # cross-pass writes let through
        # Q1 instrumentation: loop-candidate availability per attempt,
        # anchor-keyed (G1 criterion: old anchor within 5 m) vs cell-keyed
        # (old-pass cell within 5 m)
        self.att_both = self.att_cell_only = 0
        self.att_anchor_only = self.att_neither = 0
        self.acc_cell_only = 0             # accepted closures anchor-keying misses

    # ---- epoch / gate helpers ----------------------------------------------
    def _epoch(self, aid):
        return aid // self.epoch_aids

    def _gate_open(self, xy, k):
        """Cross-pass write permission: a verified loop constraint accepted
        within the last gate_recent_kf keyframes, within gate_radius."""
        self.accepts = [(ka, p) for ka, p in self.accepts
                        if k - ka <= self.gate_recent_kf]
        return any(np.linalg.norm(xy - p) <= self.gate_radius
                   for _, p in self.accepts)

    # ---- cell creation / eviction -------------------------------------------
    def _new_cell(self, ring, key):
        s = self.cell_s[ring]
        g, ix, iy = key[0], key[1], key[2]
        cx = (ix + 0.5) * s + 0.5 * s * g
        cy = (iy + 0.5) * s + 0.5 * s * g
        aid = len(self.anchors) - 1
        rel = L.se2_mul(L.se2_inv(self.anchors[aid]),
                        np.array([cx, cy, 0.0]))
        cell = Cell(ring, key, aid, rel)
        self.rings[ring][key] = cell
        rk = (int(np.floor(cx / self.region_s)),
              int(np.floor(cy / self.region_s)))
        lst = self.regions[ring].setdefault(rk, [])
        lst[:] = [kk for kk in lst if kk in self.rings[ring]]  # lazy clean
        lst.append(key)
        if len(lst) > self.region_cap[ring]:      # O(area) bound
            par = [kk for kk in lst if len(kk) == 4 and kk != key]
            pool = par if par else [kk for kk in lst if kk != key]
            drop = min(pool, key=lambda kk: self.rings[ring][kk].last_writer)
            lst.remove(drop)
            del self.rings[ring][drop]
            self.n_evict += 1
        return cell

    def _target_cell(self, ring, key, xy, k, my_aid, cur_e):
        """Write-side pass discipline: same-pass -> base cell; cross-pass ->
        base only if the gate is open (graph-consistent at write time), else
        the epoch's parallel cell."""
        G = self.rings[ring]
        base = G.get(key)
        if base is None:
            return self._new_cell(ring, key)
        if not self.xpass_gate \
                or my_aid - base.last_writer <= self.epoch_aids:
            return base
        if self._gate_open(xy, k):
            self.n_gate_open += 1
            return base
        pk = key + (cur_e,)
        par = G.get(pk)
        if par is None:
            par = self._new_cell(ring, pk)
            self.n_parallel += 1
        return par

    # ---- writes --------------------------------------------------------------
    def _write_hits(self, PW, w, rngr):
        """Hit-keyed writes: per ring allowed by the range gate, accumulate
        each sample into the cell containing the HIT, encoded in the cell's
        anchor frame (angle block + derivative about the cell origin)."""
        k = self.k
        my_aid = len(self.anchors) - 1
        cur_e = self._epoch(my_aid)
        for ring in range(4):
            if np.isfinite(self.beta):
                ok = LAMS[ring] >= self.beta * rngr * self.dtheta_beam
                if not ok.any():
                    continue
                P, ww = PW[ok], 0.5 * w[ok]
            else:
                P, ww = PW, 0.5 * w
            s = self.cell_s[ring]
            Wp = WR[ring]
            for g in (0, 1):
                ij = np.stack([np.floor((P[:, 0] - 0.5 * s * g) / s),
                               np.floor((P[:, 1] - 0.5 * s * g) / s)],
                              1).astype(np.int64)
                uq, inv = np.unique(ij, axis=0, return_inverse=True)
                inv = inv.ravel()
                for u in range(len(uq)):
                    sel = inv == u
                    grp, gw = P[sel], ww[sel]
                    key = (g, int(uq[u, 0]), int(uq[u, 1]))
                    cell = self._target_cell(ring, key, grp.mean(0),
                                             k, my_aid, cur_e)
                    if cell.mass >= self.mass_cap:
                        self.n_sat += 1
                        cell.last_writer = my_aid    # pass still owns it
                        cell.epochs.add(cur_e)
                        continue
                    C = L.se2_mul(self.anchors[cell.aid], cell.rel)
                    pc = (grp - C[:2]) @ S._rot(C[2])   # cell frame
                    A = np.exp(1j * (pc @ Wp.T))
                    cell.v += A.T @ gw
                    Cx = pc[:, 0:1] * Wp[:, 1] - pc[:, 1:2] * Wp[:, 0]
                    cell.vd += (1j * Cx * A).T @ gw
                    cell.mass += float(gw.sum())
                    cell.last_writer = my_aid
                    cell.epochs.add(cur_e)

    # ---- queries ---------------------------------------------------------------
    def _gather(self, ring, center, radius, mode):
        """Cells within radius (by CURRENT graph pose of the cell origin, so
        corrected cells are found where they now are). mode 'front': liberal
        — everything (or the recency control arm); 'loop': old-pass only —
        cells the current epoch never touched."""
        G = self.rings[ring]
        if not G:
            return [], None, None
        cells = list(G.values())
        P = np.array(self.anchors)
        aidv = np.array([c.aid for c in cells])
        rel = np.stack([c.rel for c in cells])
        ca, sa = np.cos(P[aidv, 2]), np.sin(P[aidv, 2])
        cx = P[aidv, 0] + ca * rel[:, 0] - sa * rel[:, 1]
        cy = P[aidv, 1] + sa * rel[:, 0] + ca * rel[:, 1]
        th = P[aidv, 2] + rel[:, 2]
        d = np.hypot(cx - center[0], cy - center[1])
        near = d <= radius + 0.75 * self.cell_s[ring]
        my_aid = len(self.anchors) - 1
        cur_e = self._epoch(my_aid)
        keep = []
        for i in np.flatnonzero(near):
            c = cells[i]
            if mode == "front":
                ok = self.front_mode == "all" \
                    or my_aid - c.last_writer <= self.recent_aids
            else:
                ok = (my_aid - c.last_writer > self.epoch_aids
                      and cur_e not in c.epochs)
            if ok:
                keep.append(i)
        if not keep:
            return [], None, None
        poses = np.stack([cx, cy, th], 1)[keep]
        return [cells[i] for i in keep], poses, d[keep]

    def _ring_world(self, ring, cells, poses):
        """World-frame ring block: per cell nearest-3-deg exact permutation +
        first-order derivative correction + translation phase, summed."""
        V = np.stack([c.v for c in cells])
        VD = np.stack([c.vd for c in cells])
        m = np.round(poses[:, 2] * NA / np.pi).astype(int)
        delta = poses[:, 2] - m * np.pi / NA
        idx = (np.arange(NA)[None, :] - m[:, None]) % (2 * NA)
        Vr = np.take_along_axis(np.concatenate([V, np.conj(V)], 1), idx, 1)
        if self.use_der:
            VDr = np.take_along_axis(
                np.concatenate([VD, np.conj(VD)], 1), idx, 1)
            Vr = Vr + delta[:, None] * VDr
        ph = np.exp(1j * (poses[:, :2] @ WR[ring].T))
        return (ph * Vr).sum(0)

    def local_bundle(self, center, radius=8.0):
        """Frontend local map: liberal additive superposition of every cell
        in radius (both grids, all pass layers — queries are free)."""
        Bf = np.zeros(L.W.shape[0], complex)
        for ring in range(4):
            cells, poses, _ = self._gather(ring, center, radius, "front")
            if cells:
                Bf[ring * NA:(ring + 1) * NA] = \
                    self._ring_world(ring, cells, poses)
        return Bf

    def map_memory_kb(self):
        n = sum(len(G) for G in self.rings)
        return n * 2 * NA * 16 / 1024          # v + vd per cell

    def cell_counts(self):
        return {"cells": sum(len(G) for G in self.rings),
                "per_ring": [len(G) for G in self.rings],
                "parallel": sum(1 for G in self.rings for kk in G
                                if len(kk) == 4)}

    # ---- lifecycle (mirrors BoundedSLAM.add_keyframe) ------------------------
    def add_keyframe(self, pts, w, guess):
        self.k += 1
        k = self.k
        if k % ANCHOR == 0:
            self.anchors.append(guess.copy())
        aid = len(self.anchors) - 1
        est = guess
        fell_back = True
        if k > 0 and len(pts) >= 20:
            Bq = self.local_bundle(guess[:2])[L.MAIN]
            if np.abs(Bq).sum() > 0:
                cand = self.matcher.match(Bq, pts, w, guess)
                cand[2] = S.wrap(cand[2])
                if np.linalg.norm(cand[:2] - guess[:2]) < 0.45 \
                        and abs(S.wrap(cand[2] - guess[2])) < np.deg2rad(11):
                    est = cand
                    fell_back = False
                    # session coherence baseline: fine rings, as shipped
                    B01 = Bq[:2 * NA].reshape(2, NA)
                    PWc = pts @ S._rot(est[2]).T + est[:2]
                    sv01 = np.exp(1j * (PWc @ L.W[:2 * NA].T)).T @ w
                    s01 = sv01.reshape(2, NA)
                    c01 = float(np.mean(
                        (np.conj(B01) * s01).sum(1).real
                        / (np.linalg.norm(B01, axis=1)
                           * np.linalg.norm(s01, axis=1) + 1e-12)))
                    self.coh_ref = c01 if self.coh_ref is None \
                        else 0.95 * self.coh_ref + 0.05 * c01
        self._span_fallback = getattr(self, "_span_fallback", False) or fell_back
        if k % ANCHOR == 0:
            self.anchors[aid] = est.copy()
        rel = L.se2_mul(L.se2_inv(self.anchors[aid]), est)
        self.kf_ref.append((aid, rel))

        if len(pts):
            rngr = np.linalg.norm(pts, axis=1)   # robot-frame ranges
            PW = pts @ S._rot(est[2]).T + est[:2]
            self._write_hits(PW, w, rngr)

        if aid > 0 and k % ANCHOR == 0:          # seq edge, shipped sigmas
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

    # ---- loop constraints: single old-pass cell set --------------------------
    def try_constraint(self, pts, w):
        k = self.k
        if len(pts) < 20:
            return
        me = self.pose_of(k)
        my_aid = self.aid_of(k)
        cur_e = self._epoch(my_aid)
        # Q1: would G1's anchor-keyed gathering find a candidate here?
        anchor_avail = any(
            abs(aid - my_aid) > self.epoch_aids
            and np.linalg.norm(self.anchors[aid][:2] - me[:2]) < 5.0
            for aid in range(len(self.anchors)))
        # gather old-pass cells, group by last-writer epoch, keep the single
        # NEAREST pass (evidence independence: one old pass per measurement)
        groups = {}                       # epoch -> [(cell, pose, dist)]
        for ring in range(4):
            cells, poses, dist = self._gather(ring, me[:2], 8.0, "loop")
            for c, p, dd in zip(cells, poses if cells else [],
                                dist if cells else []):
                groups.setdefault(self._epoch(c.last_writer), []).append(
                    (c, p, dd))
        best_e, best_d = None, np.inf
        for e, lst in groups.items():
            d0 = min(dd for _, _, dd in lst)
            if d0 < best_d:
                best_e, best_d = e, d0
        cell_avail = best_e is not None and best_d < 5.0
        if anchor_avail and cell_avail:
            self.att_both += 1
        elif cell_avail:
            self.att_cell_only += 1
        elif anchor_avail:
            self.att_anchor_only += 1
        else:
            self.att_neither += 1
        if not cell_avail:
            return
        sel = groups[best_e]
        Bb = np.zeros(L.W.shape[0], complex)
        for ring in range(4):
            rc = [(c, p) for c, p, _ in sel if c.ring == ring]
            if rc:
                Bb[ring * NA:(ring + 1) * NA] = self._ring_world(
                    ring, [c for c, _ in rc], np.stack([p for _, p in rc]))
        # dominant cell (max mass within 5 m) carries the edge: its frame
        # anchor is what the content rides with
        dom = max((c for c, _, dd in sel if dd < 5.0), key=lambda c: c.mass)
        c_aid = dom.aid
        if abs(c_aid - my_aid) <= self.epoch_aids:
            return                        # frame anchor too recent: skip
        pose = self.cmatcher.match(Bb[L.MAIN], pts, w, me)
        pose[2] = S.wrap(pose[2])
        if np.linalg.norm(pose[:2] - me[:2]) > 0.6 \
                or abs(S.wrap(pose[2] - me[2])) > np.deg2rad(7):
            return
        # per-ring coherence + degeneracy indicators at the matched pose
        sv = L.ENC.shift(pose[:2]) * L.encode(pts @ S._rot(pose[2]).T, w)
        c = (np.conj(Bb) * sv).reshape(L.N_RING, NA)
        Br = Bb.reshape(L.N_RING, NA)
        svr = sv.reshape(L.N_RING, NA)
        coh = c.sum(1).real / (np.linalg.norm(Br, axis=1)
                               * np.linalg.norm(svr, axis=1) + 1e-12)
        cM = c[:4].ravel()
        nrmM = (np.linalg.norm(Br[:4].ravel())
                * np.linalg.norm(svr[:4].ravel()) + 1e-12)
        K = (L.W[L.MAIN] * (cM.real / nrmM)[:, None]).T @ L.W[L.MAIN]
        evals, evecs = np.linalg.eigh(K)
        l_weak, l_strong = float(evals[0]), float(evals[1])
        u_weak = evecs[:, 0]
        ts = np.arange(-1.2, 1.21, 0.06)
        s0 = float(cM.real.sum())
        far = np.abs(ts) >= 0.35
        ridge = [0.0, 0.0]
        for j, u in enumerate((u_weak, evecs[:, 1])):
            proj = L.W[L.MAIN] @ u
            sc = (np.exp(1j * ts[:, None] * proj[None, :]) @ cM).real
            ridge[j] = float(sc[far].max() / max(s0, 1e-12))
        ridge_w, ridge_s = ridge
        lever = np.linalg.norm(me[:2] - self.anchors[c_aid][:2])
        sig_t = np.sqrt(0.08 ** 2 + (0.05 * lever) ** 2)
        sig_r = np.deg2rad(2.0)
        self.diag.append([np.linalg.norm(pose[:2] - me[:2]),
                          abs(S.wrap(pose[2] - me[2])), len(sel), *coh,
                          l_weak, l_strong, ridge_w, ridge_s,
                          self.coh_ref if self.coh_ref is not None else np.nan,
                          0.0])
        infl = 1.0
        if self.use_coh and self.coh_ref is not None:
            if not self.coh_soft:
                if coh[:2].mean() < self.coh_target * self.coh_ref:
                    self.n_veto += 1
                    return
            else:
                infl = self._coh_response(coh, l_weak, l_strong,
                                          ridge_w, ridge_s, me, k)
                if infl < 0:
                    return
        _, rel = self.kf_ref[k]
        Zk = L.se2_mul(pose, L.se2_inv(rel))     # implied anchor pose of k
        # cells ride with the graph, so Z is against the CURRENT frame-anchor
        # pose — exactly the shipped rigid-segment semantics
        Z = L.se2_mul(L.se2_inv(self.anchors[c_aid]), Zk)
        Zc = L.se2_mul(L.se2_inv(self.anchors[c_aid]), self.anchors[my_aid])
        since = k - self.last_accept_k
        s_at = sig_t + min(0.30, 0.002 * since)
        s_ar = sig_r + min(np.deg2rad(6), np.deg2rad(0.03) * since)
        chi = (np.linalg.norm(Z[:2] - Zc[:2]) / s_at) ** 2 \
            + (S.wrap(Z[2] - Zc[2]) / s_ar) ** 2
        if self.use_innov and chi > 9.0:
            self.n_innov_rej += 1
            return
        edge = (c_aid, my_aid, Z, 1 / (sig_t * infl), 1 / (sig_r * infl),
                "loop")
        key = (c_aid, my_aid)
        if key in self.banned:
            return
        if key in self.edge_seen:
            if edge[3] * 3.5 < self.edges[self.edge_seen[key]][3]:
                return
            self.edges[self.edge_seen[key]] = edge
        else:
            self.edge_seen[key] = len(self.edges)
            self.edges.append(edge)
        if infl <= self.infl_weak:
            self.dirty = True
            self.pending_new.append(key)
            self.last_accept_k = k
            self._streak = 0
            self._streak_xy = None
            # verified closure: opens the cross-pass write gate here
            self.accepts.append((k, me[:2].copy()))
            if not anchor_avail:
                self.acc_cell_only += 1
        else:
            self._suppress(me[:2], k)
        self.diag[-1][-1] = 1.0


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _enc_ref(P, w, ring):
    return np.exp(1j * (P @ WR[ring].T)).T @ w


def self_test():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-4, 4, (200, 2))
    w = np.full(200, 0.1)
    # --- 1. write/query round trip, then rigid anchor motion ---------------
    sa = ScaleArraySLAM()
    sa.beta = np.inf
    sa.k = 0
    sa.anchors = [np.array([1.0, -2.0, 4 * np.pi / NA])]
    sa.kf_ref = [(0, np.zeros(3))]
    ctr = np.array([8.0, 5.0])
    PW = pts + ctr
    sa._write_hits(PW, w, np.linalg.norm(pts, axis=1))
    Bq = sa.local_bundle(ctr, radius=12.0)
    for ring in range(4):
        assert np.allclose(Bq[ring * NA:(ring + 1) * NA],
                           _enc_ref(PW, w, ring), atol=1e-9)
    # exact-lattice rigid graph correction: content must ride exactly
    G7 = np.array([0.7, -1.3, 7 * np.pi / NA])
    sa.anchors[0] = L.se2_mul(G7, sa.anchors[0])
    PW2 = PW @ S._rot(G7[2]).T + G7[:2]
    ctr2 = ctr @ S._rot(G7[2]).T + G7[:2]
    Bq2 = sa.local_bundle(ctr2, radius=12.0)
    for ring in range(4):
        assert np.allclose(Bq2[ring * NA:(ring + 1) * NA],
                           _enc_ref(PW2, w, ring), atol=1e-9)
    sa.anchors[0] = L.se2_mul(L.se2_inv(G7), sa.anchors[0])   # undo
    # off-lattice correction: first-order derivative must beat no-derivative
    Gd = np.array([0.3, 0.2, 7 * np.pi / NA + np.deg2rad(0.4)])
    sa.anchors[0] = L.se2_mul(Gd, sa.anchors[0])
    PW3 = PW @ S._rot(Gd[2]).T + Gd[:2]
    ctr3 = ctr @ S._rot(Gd[2]).T + Gd[:2]
    Bq3 = sa.local_bundle(ctr3, radius=12.0)
    err_d = err_n = ref_n = 0.0
    sa.use_der = False
    Bq3n = sa.local_bundle(ctr3, radius=12.0)
    sa.use_der = True
    for ring in range(4):
        ref = _enc_ref(PW3, w, ring)
        err_d += np.linalg.norm(Bq3[ring * NA:(ring + 1) * NA] - ref) ** 2
        err_n += np.linalg.norm(Bq3n[ring * NA:(ring + 1) * NA] - ref) ** 2
        ref_n += np.linalg.norm(ref) ** 2
    err_d, err_n = np.sqrt(err_d / ref_n), np.sqrt(err_n / ref_n)
    assert err_d < 0.10 and err_d < 0.35 * err_n, (err_d, err_n)
    # --- 2. cross-pass write gate -------------------------------------------
    sb = ScaleArraySLAM(gap_kf=60)          # epoch_aids = 12
    sb.beta = np.inf
    sb.k = 0
    sb.anchors = [np.zeros(3)]
    sb.kf_ref = [(0, np.zeros(3))]
    Pc = rng.uniform(2.5, 3.5, (50, 2))
    wc = np.full(50, 0.1)
    sb._write_hits(Pc, wc, np.linalg.norm(Pc, axis=1))
    base_keys = {(r, kk) for r in range(4) for kk in sb.rings[r]}
    assert all(len(kk) == 3 for _, kk in base_keys)
    m0 = {(r, kk): sb.rings[r][kk].mass for r, kk in base_keys}
    sb.anchors += [np.zeros(3)] * 20        # aid 20 -> epoch 1 (cross-pass)
    sb.k = 100
    sb._write_hits(Pc, wc, np.linalg.norm(Pc, axis=1))
    assert sb.n_parallel > 0                # gate shut: parallel cells opened
    assert all(abs(sb.rings[r][kk].mass - m0[(r, kk)]) < 1e-12
               for r, kk in base_keys)      # base untouched
    # --- 3. two-layer additive query (merge-free superposition) -------------
    # base holds pass-0 content (frame anchor 0), the parallel layer holds
    # pass-1 content (frame anchor 20). A graph correction applied to the
    # pass-1 anchors must move ONLY the parallel layer, and the liberal
    # frontend query must equal the sum of both world contents — no merge.
    Gc = np.array([0.4, -0.2, 5 * np.pi / NA])   # correction on pass 1 only
    for kk in range(20, len(sb.anchors)):
        sb.anchors[kk] = L.se2_mul(Gc, sb.anchors[kk])
    Pc2 = Pc @ S._rot(Gc[2]).T + Gc[:2]     # pass-1 copy of the content rides
    q = Pc.mean(0)
    Bq = sb.local_bundle(q, radius=10.0)
    for ring in range(4):
        ref = _enc_ref(Pc, wc, ring) + _enc_ref(Pc2, wc, ring)
        assert np.allclose(Bq[ring * NA:(ring + 1) * NA], ref, atol=1e-8), ring
    # loop gather must see ONLY old passes (epoch segregation)
    sb.anchors += [np.zeros(3)] * 60        # current epoch far ahead
    sb.k = 400
    cells, poses, _ = sb._gather(0, q, 8.0, "loop")
    es = {sb._epoch(c.last_writer) for c in cells}
    assert es and sb._epoch(len(sb.anchors) - 1) not in es
    # gate-open path: a verified closure nearby re-admits cross-pass writes
    # into the base cell
    sb.accepts.append((400, Pc.mean(0)))
    sb._write_hits(Pc, wc, np.linalg.norm(Pc, axis=1))
    assert sb.n_gate_open > 0
    assert any(sb.rings[r][kk].mass > m0[(r, kk)] + 1e-9
               for r, kk in base_keys)      # gate open: base written again
    print("self-test ok: hit-keyed write/query round trip, rigid ride "
          "(exact + first-order), cross-pass gate, two-layer additive "
          "query, loop epoch segregation")


# ---------------------------------------------------------------------------
# Bench (paired protocol: identical RNG stream to B.run / ssp_hier.run_hybrid)
# ---------------------------------------------------------------------------

def run(world="room", n=750, seed=1, laps=3, **kw):
    S.RNG = np.random.default_rng(seed)
    segs = WORLDS[world]()
    gt = multiloop_traj(n, laps=laps)
    odo = B.sim_odometry(gt)
    slam = ScaleArraySLAM(**kw)
    est = np.zeros((n, 3))
    t0 = time.time()
    for k in range(n):
        r = scan_at(segs, gt[k])
        pts, w, _ = S.scan_to_samples(r, BEAM)
        guess = odo[0] if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odo[k - 1]), odo[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
    secs = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    ate = np.linalg.norm(fin[:, :2] - gt[:, :2], axis=1)
    n_loop = false = 0
    for a, b, Z, wt, wr, kind in slam.edges:
        if kind != "loop":
            continue
        n_loop += 1
        Zt = L.se2_mul(L.se2_inv(gt[a * ANCHOR]), gt[b * ANCHOR])
        if np.linalg.norm(Z[:2] - Zt[:2]) > 0.30 \
                or abs(S.wrap(Z[2] - Zt[2])) > np.deg2rad(3):
            false += 1
    cc = slam.cell_counts()
    return dict(ate=float(np.sqrt((ate ** 2).mean())),
                ate_max=float(ate.max()), edges=n_loop, false=false,
                pruned=slam.n_pruned, veto=slam.n_veto,
                innov=slam.n_innov_rej, inflate=slam.n_inflate,
                mem=float(slam.map_memory_kb()), ms=1000 * secs / n,
                cells=cc["cells"], parallel=cc["parallel"],
                evict=slam.n_evict, sat=slam.n_sat,
                gate_open=slam.n_gate_open,
                att_both=slam.att_both, att_cell_only=slam.att_cell_only,
                att_anchor_only=slam.att_anchor_only,
                att_neither=slam.att_neither,
                acc_cell_only=slam.acc_cell_only)


SA_VARIANTS = {
    "SA": {},
    "SA-nogate": dict(xpass_gate=False),
    "SA-recent": dict(front_mode="recent"),
}

BENCH_WORLDS = ("room", "sparse", "corridor")


def run_cell(gran, world, seed, n=750):
    if gran == "G1":
        r = B.run(world, n=n, seed=seed)
        out = dict(ate=float(r["ate"]), edges=int(r["edges"]),
                   false=int(r["false"]), mem=float(r["mem"]),
                   ms=1000 * r["secs"] / n)
    elif gran == "HY4":
        import experiments.hier as H                # run only, never edited
        r = H.run_hybrid(world, n=n, seed=seed, beta=3.0, seg_nring=4)
        out = {k: r[k] for k in ("ate", "edges", "false", "mem", "ms")}
    else:
        r = run(world, n=n, seed=seed, **SA_VARIANTS[gran])
        out = dict(r)
    out.update(gran=gran, world=world, seed=seed)
    return out


def bench(quick=False):
    self_test()
    seeds = (1, 2) if quick else (1, 2, 3, 4)
    results = {}

    def do(gran, worlds=BENCH_WORLDS):
        for wl in worlds:
            for s in seeds:
                r = run_cell(gran, wl, s)
                results[(gran, wl, s)] = r
                print(json.dumps(r), flush=True)

    for gran in ("G1", "HY4", "SA"):
        do(gran)
    do("SA-nogate", worlds=("room", "sparse"))     # Q2 ablation
    do("SA-recent", worlds=("room", "sparse"))     # crosstalk control arm

    hdr = f"{'config':<12}" + "".join(
        f"{wl + ' ATE cm':>14}{'f/e':>9}{'mem KB':>8}{'ms':>5}"
        for wl in BENCH_WORLDS)
    print("\n" + hdr)
    for gran in ("G1", "HY4", "SA", "SA-nogate", "SA-recent"):
        line = f"{gran:<12}"
        for wl in BENCH_WORLDS:
            cell = [results[(gran, wl, s)] for s in seeds
                    if (gran, wl, s) in results]
            if not cell:
                line += f"{'-':>14}{'-':>9}{'-':>8}{'-':>5}"
                continue
            line += (f"{100 * np.mean([c['ate'] for c in cell]):>14.1f}"
                     f"{np.mean([c['false'] for c in cell]):>5.1f}/"
                     f"{np.mean([c['edges'] for c in cell]):<3.0f}"
                     f"{np.mean([c['mem'] for c in cell]):>8.0f}"
                     f"{np.mean([c['ms'] for c in cell]):>5.0f}")
        print(line, flush=True)
    # pre-registered gate: SA pooled mean ATE within 1.15x of the better of
    # G1 / HY4 pooled
    pool = {g: np.mean([results[(g, wl, s)]["ate"]
                        for wl in BENCH_WORLDS for s in seeds])
            for g in ("G1", "HY4", "SA")}
    ref = min(pool["G1"], pool["HY4"])
    print(f"\npooled mean ATE m: G1 {pool['G1']:.3f}  HY4 {pool['HY4']:.3f}  "
          f"SA {pool['SA']:.3f}  ratio vs better {pool['SA'] / ref:.3f}  "
          f"gate(<=1.15) {'PASS' if pool['SA'] <= 1.15 * ref else 'FAIL'}")
    # Q1 aggregate
    for wl in BENCH_WORLDS:
        cs = [results[("SA", wl, s)] for s in seeds]
        print(f"Q1 {wl}: attempts both/cell-only/anchor-only/neither = "
              f"{sum(c['att_both'] for c in cs)}/"
              f"{sum(c['att_cell_only'] for c in cs)}/"
              f"{sum(c['att_anchor_only'] for c in cs)}/"
              f"{sum(c['att_neither'] for c in cs)}   "
              f"accepted-where-anchor-keying-misses = "
              f"{sum(c['acc_cell_only'] for c in cs)}")
    return results


# ---------------------------------------------------------------------------
# CARMEN driver (mirrors runners/carmen.py; zero new tuning)
# ---------------------------------------------------------------------------

def main_carmen(argv):
    import sspslam.frontend as C
    path = argv[0] if argv else "data/intel.log"
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)
    if len(argv) > 1 and argv[1].isdigit():
        keys = keys[:int(argv[1])]
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    print(f"{len(scans)} scans -> {n} keyframes   scale-array map")
    slam = ScaleArraySLAM(robust=True, attempt_every=4, relax_every=25,
                          gap_kf=300, recent_aids=12)
    slam.dtheta_beam = np.pi / n_beams
    est = np.zeros((n, 3))
    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < 40.0, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = opose if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
        if k % 1000 == 0:
            print(f"  kf {k}/{n}  t={time.time() - t0:.0f}s  "
                  f"loops={sum(1 for e in slam.edges if e[5] == 'loop')}  "
                  f"mem={slam.map_memory_kb():.0f} KB", flush=True)
    if slam.dirty:
        slam.relax()
    dt = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    n_loop = sum(1 for e in slam.edges if e[5] == "loop")
    cc = slam.cell_counts()
    print(f"done: {dt:.0f}s ({dt / n * 1e3:.0f} ms/kf)  loop edges={n_loop}  "
          f"pruned={slam.n_pruned}  relax={slam.n_relax}  "
          f"map={slam.map_memory_kb():.0f} KB "
          f"({cc['cells']} cells, per-ring {cc['per_ring']}, "
          f"{cc['parallel']} parallel)  evict={slam.n_evict}  "
          f"sat={slam.n_sat}  gate_open={slam.n_gate_open}")
    print(f"Q1: both/cell-only/anchor-only/neither = {slam.att_both}/"
          f"{slam.att_cell_only}/{slam.att_anchor_only}/{slam.att_neither}  "
          f"acc-cell-only={slam.acc_cell_only}")
    ref = C.parse_flaser(path.replace(".log", ".gfs.log"))
    rts = np.array([t for _, _, t in ref])
    rxy = np.stack([p[:2] for _, p, _ in ref])
    j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
    good = np.abs(rts - kts[j]) < 0.3
    al = C.align_se2(fin[j[good], :2], rxy[good])
    e = np.linalg.norm(al - rxy[good], axis=1)
    print(f"ATE vs corrected log: rmse {np.sqrt((e ** 2).mean()):.3f} m   "
          f"median {np.median(e):.3f} m   max {e.max():.3f} m")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if cmd == "selftest":
        self_test()
    elif cmd == "cell":
        world, seed = sys.argv[2], int(sys.argv[3])
        gran = {"nogate": "SA-nogate", "recent": "SA-recent"}.get(
            sys.argv[4] if len(sys.argv) > 4 else "", "SA")
        print(json.dumps(run_cell(gran, world, seed)))
    elif cmd == "carmen":
        main_carmen(sys.argv[2:])
    else:
        bench(quick=(cmd == "quick" or "quick" in sys.argv[2:]))
