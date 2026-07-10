"""Sampling/encoding family: point vs segment-INTEGRAL vs multi-sub-point.

The shipped sampler (S.scan_to_samples) is already a multi-sub-point
interpolator: consecutive hits (<=1 dropped beam) bridge into segments gated
by dr <= OCC_RATIO * tang (OCC_RATIO 2.0 == a 63.4-deg cutoff angle from the
tangential) and chord <= 1.5 m, resampled at 0.12 m spacing with chord mass;
both-side-rejected hits survive as LONE points at fixed w = 0.12. The E2
point mode (ssp_fpga.points_from_scan) drops interpolation entirely: one
point per return at arc weight w = r*dtheta — the audited win mechanism was
arc weighting + keeping silhouette hits at full weight. This module spans
the space between and beyond them:

  sample_interp(n_sub, cut_deg, w_mode)   [deploy-friendly]
      n_sub midpoint sub-points per kept segment; NON-bridged hits stay
      single points at w = r*dtheta (E2-style silhouette retention).
      w_mode "arc": pair mass = mean(r0,r1) * dtheta_gap; "chord": chord
      length (shipped-style). cut_deg parameterizes the occlusion gate
      dr <= tan(cut_deg) * tang. cut_deg < 0 => no bridging == E2 exactly.

  pack_segint(cut_deg, lone)              [exact integral]
      EXACT line integral of the plane-wave phasor along each kept segment:
        integral_0^1 e^{i k.(p0 + t(p1-p0))} dt = sinc(k.d/2) e^{i k.c}
      (c = midpoint, d = chord). The sinc factor is PRINCIPLED BLANKING:
      components with |k.d| >> 1 (rings finer than the segment extent) are
      suppressed automatically while coarse rings keep full weight — the
      sandbox "thermometer" is a hand-tuned approximation of exactly this.
      lone="point": non-bridged hits as degenerate segments (sinc=1);
      lone="arc": non-bridged hits as their TANGENTIAL BEAM FOOTPRINT
      (a segment of length r*dtheta perpendicular to the ray) — per-beam
      soft blanking with zero extra constants.

  pack_arcint()
      every hit as its tangential footprint segment, no bridging — isolates
      the blanking question from the interpolation question.

Representation: packed = vstack([P0, P1]) (2N x 2). Both halves transform
correctly under the pipeline's linear ops (pts @ R.T + t), so the matcher
needs no changes; SegIntEncoder derives (c, d) from the halves. Rotation by
index permutation stays valid: sinc is even, so v(-k) = conj(v(k)) holds.

SegIntSLAM carries copied add_keyframe/try_constraint bodies (the two spots
that hand-roll point encodes: the coh_ref probe and the segment fold — the
fold's d/dtheta derivative gains the exact sinc' term). Neutralisation:
degenerate packs (P0 == P1) reduce EXACTLY to the parent point pipeline
(asserted bit-exact in selftest).

renorm_alpha (the blanking-vs-weighting question): when the sinc blanks
fine-ring mass of a sample, scale its surviving components by
(1/rho)^(alpha/2), rho = mean_D sinc^2 (fraction of lattice mass kept).
alpha=0 control (linear arc weight only); alpha=1 restores the sample's
total lattice energy (the "upweight coarse even more" hypothesis).

Usage:
  python3 ssp_sampling.py selftest
  python3 ssp_sampling.py interp fr079           # stage 1 sweep
  python3 ssp_sampling.py segint fr079           # stage 2: integral modes
  python3 ssp_sampling.py renorm fr079           # stage 3: alpha sweep
"""
import sys

import numpy as np

import ssp_slam as S
import ssp_slam_loop as L
import ssp_bounded as B
import ssp_bounded as BB      # alias for constants inside copied bodies
import ssp_fpga as F          # (the verbatim body assigns a local `B`)
import ssp_datasets as DS

MAX_CHORD = 1.5          # shipped bridge cap [m]


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------

def _hits(rr, beam):
    hit = np.isfinite(rr)
    idx = np.flatnonzero(hit)
    r = rr[idx]
    pts = np.stack([r * np.cos(beam[idx]), r * np.sin(beam[idx])], 1)
    dth = abs(beam[1] - beam[0]) if len(beam) > 1 else np.deg2rad(1.0)
    return idx, r, pts, dth


def _bridge(idx, r, pts, cut_deg):
    """keep mask over consecutive-pair candidates (shipped gate, angle-
    parameterized): consec (<=1 dropped beam), chord cap, occlusion angle."""
    consec = np.diff(idx) <= 2
    chord_v = pts[1:] - pts[:-1]
    chord = np.linalg.norm(chord_v, axis=1)
    dr = np.abs(np.diff(r))
    tang = np.sqrt(np.maximum(chord ** 2 - dr ** 2, 0.0))
    ratio = np.tan(np.deg2rad(cut_deg)) if cut_deg >= 0 else -1.0
    keep = consec & (chord <= MAX_CHORD) & (dr <= ratio * tang)
    return keep, chord_v, chord


def sample_interp(rr, beam, n_sub=3, cut_deg=63.4, w_mode="arc"):
    """Multi-sub-point interpolation -> plain (pts, w); drop-in sampler."""
    idx, r, pts, dth = _hits(rr, beam)
    if len(idx) < 2:
        return np.zeros((0, 2)), np.zeros(0)
    keep, chord_v, chord = _bridge(idx, r, pts, cut_deg)
    out_p, out_w = [], []
    used = np.zeros(len(idx), bool)
    ts = (np.arange(n_sub) + 0.5) / n_sub
    for i in np.flatnonzero(keep):
        out_p.append(pts[i] + ts[:, None] * chord_v[i])
        mass = chord[i] if w_mode == "chord" \
            else 0.5 * (r[i] + r[i + 1]) * dth * (idx[i + 1] - idx[i])
        out_w.append(np.full(n_sub, mass / n_sub))
        used[i] = used[i + 1] = True
    lone = ~used
    if lone.any():
        out_p.append(pts[lone])
        out_w.append(r[lone] * dth)
    return np.concatenate(out_p), np.concatenate(out_w)


def _foot(r, pts, dth):
    """Tangential beam-footprint chord vectors (length r*dtheta, perp ray)."""
    u = pts / np.maximum(r[:, None], 1e-12)          # ray directions
    t = np.stack([-u[:, 1], u[:, 0]], 1)             # tangentials
    return (r * dth)[:, None] * t


def pack_points(rr, beam):
    """E2 points as degenerate segments (neutralisation control)."""
    pts, w = F.points_from_scan(rr, beam)
    return np.vstack([pts, pts]), w


def pack_arcint(rr, beam):
    """Every hit = its tangential footprint segment (pure blanking probe)."""
    idx, r, pts, dth = _hits(rr, beam)
    if len(idx) < 2:
        return np.zeros((0, 2)), np.zeros(0)
    d = _foot(r, pts, dth)
    return np.vstack([pts - d / 2, pts + d / 2]), r * dth


def pack_segint(rr, beam, cut_deg=63.4, lone="point"):
    """Bridged exact-integral segments + lone hits (point or footprint)."""
    idx, r, pts, dth = _hits(rr, beam)
    if len(idx) < 2:
        return np.zeros((0, 2)), np.zeros(0)
    keep, chord_v, chord = _bridge(idx, r, pts, cut_deg)
    P0, P1, W = [], [], []
    used = np.zeros(len(idx), bool)
    for i in np.flatnonzero(keep):
        P0.append(pts[i])
        P1.append(pts[i + 1])
        W.append(0.5 * (r[i] + r[i + 1]) * dth * (idx[i + 1] - idx[i]))
        used[i] = used[i + 1] = True
    lm = ~used
    if lm.any():
        pl, rl = pts[lm], r[lm]
        if lone == "arc":
            d = _foot(rl, pl, dth)
            P0.append(pl - d / 2)
            P1.append(pl + d / 2)
        else:
            P0.append(pl)
            P1.append(pl)
        W.append(rl * dth)
    P0 = np.concatenate([np.atleast_2d(p) for p in P0])
    P1 = np.concatenate([np.atleast_2d(p) for p in P1])
    return np.vstack([P0, P1]), np.concatenate(
        [np.atleast_1d(x) for x in W])


# ---------------------------------------------------------------------------
# segment-integral encoder
# ---------------------------------------------------------------------------

class SegIntEncoder:
    """encode(packed 2Nx2, w N) = sum_j w_j sinc(k.d_j/2) e^{i k.c_j};
    optional per-sample energy renormalisation (renorm_alpha)."""

    def __init__(self, W, renorm_alpha=0.0):
        self.W = W
        self.alpha = renorm_alpha

    def _cd(self, packed):
        n = len(packed) // 2
        P0, P1 = packed[:n], packed[n:]
        return (P0 + P1) / 2, P1 - P0

    def encode(self, packed, weights):
        c, d = self._cd(packed)
        a = 0.5 * (d @ self.W.T)
        sf = np.sinc(a / np.pi)                       # sin(a)/a, sinc(0)=1
        if self.alpha:
            rho = np.maximum((sf ** 2).mean(1, keepdims=True), 1e-12)
            sf = sf * rho ** (-self.alpha / 2)
        return (np.exp(1j * (c @ self.W.T)) * sf).T @ weights

    def shift(self, d):
        return np.exp(1j * (self.W @ np.asarray(d)))


def _dsinc(a):
    """d/da [sin a / a], stable at 0 (Taylor: -a/3)."""
    out = np.empty_like(a)
    small = np.abs(a) < 1e-4
    ax = np.where(small, 1.0, a)
    out = (np.cos(ax) - np.sin(ax) / ax) / ax
    return np.where(small, -a / 3.0, out)


def seg_encode_der(W, packed, weights, alpha=0.0):
    """(v, dv/dtheta) of the segment-integral encode about the origin —
    the fold pair. Exact: d/dth[e^{ik.Rc} sinc(k.Rd/2)] at th=0 =
    [i Cx(c) sinc(a) + Cx(d)/2 dsinc(a)] e^{ik.c},  Cx(x) = x_x k_y - x_y k_x."""
    n = len(weights)
    P0, P1 = packed[:n], packed[n:]
    c, d = (P0 + P1) / 2, P1 - P0
    A = np.exp(1j * (c @ W.T))
    a = 0.5 * (d @ W.T)
    sf = np.sinc(a / np.pi)
    if alpha:
        rho = np.maximum((sf ** 2).mean(1, keepdims=True), 1e-12)
        g = rho ** (-alpha / 2)
        sf = sf * g
        dsf = _dsinc(a) * g          # rho treated locally constant in theta
    else:
        dsf = _dsinc(a)
    Cxc = c[:, 0:1] * W[:, 1] - c[:, 1:2] * W[:, 0]
    Cxd = d[:, 0:1] * W[:, 1] - d[:, 1:2] * W[:, 0]
    v = (A * sf).T @ weights
    vd = (A * (1j * Cxc * sf + 0.5 * Cxd * dsf)).T @ weights
    return v, vd


# ---------------------------------------------------------------------------
# pipeline over packed segments (copied-body overrides)
# ---------------------------------------------------------------------------

class SegCore(B.BoundedSLAM):
    """BoundedSLAM core with every encode swapped for the packed segment
    integral — overridden AT THE BoundedSLAM LEVEL so the Quant/Band layers'
    super() chains (freeze-on-advance, band noise) still run above it.
    Compose as class X(F.BandSLAM, SegCore): MRO puts SegCore between
    QuantStoreSLAM and BoundedSLAM."""

    def add_keyframe(self, pts, w, guess):
        # ---- verbatim ssp_bounded.BoundedSLAM.add_keyframe with (a) point
        # counts -> len(w), (b) the coh_ref probe + the fold encodes swapped
        # for the packed segment-integral versions ----
        self.k += 1
        k = self.k
        ANCHOR, CELL, CELL_CAP = BB.ANCHOR, BB.CELL, BB.CELL_CAP
        if k % ANCHOR == 0:
            self.anchors.append(guess.copy())
        aid = len(self.anchors) - 1
        est = guess
        fell_back = True
        if k > 0 and len(w) >= 20:
            B = self.local_bundle(guess[:2])[L.MAIN]
            if np.abs(B).sum() > 0:
                cand = self.matcher.match(B, pts, w, guess)
                cand[2] = S.wrap(cand[2])
                if np.linalg.norm(cand[:2] - guess[:2]) < 0.45 \
                        and abs(S.wrap(cand[2] - guess[2])) < np.deg2rad(11):
                    est = cand
                    fell_back = False
                    PW = pts @ S._rot(est[2]).T + est[:2]
                    sv01 = self.enc01.encode(PW, w)
                    B01 = B[:2 * L.N_ANG].reshape(2, L.N_ANG)
                    s01 = sv01.reshape(2, L.N_ANG)
                    c01 = float(np.mean(
                        (np.conj(B01) * s01).sum(1).real
                        / (np.linalg.norm(B01, axis=1)
                           * np.linalg.norm(s01, axis=1) + 1e-12)))
                    self.coh_ref = c01 if self.coh_ref is None \
                        else 0.95 * self.coh_ref + 0.05 * c01
                    est = self._frontend_accept(guess, cand, c01)
        self._span_fallback = getattr(self, "_span_fallback", False) \
            or fell_back
        if k % ANCHOR == 0:
            self.anchors[aid] = est.copy()
        rel = L.se2_mul(L.se2_inv(self.anchors[aid]), est)
        self.kf_ref.append((aid, rel))

        if len(w):
            PA = pts @ S._rot(rel[2]).T + rel[:2]
            v, vd = seg_encode_der(L.W, PA, w, self.alpha)
            if aid not in self.segvec:
                self.segvec[aid] = np.zeros(L.W.shape[0], self.store_dtype)
                self.segder[aid] = np.zeros(L.W.shape[0], self.store_dtype)
                cell = tuple((self.anchors[aid][:2] // CELL).astype(int))
                self.cells.setdefault(cell, []).append(aid)
                if len(self.cells[cell]) > CELL_CAP:
                    drop = self.cells[cell].pop(0)
                    self.segvec.pop(drop, None)
                    self.segder.pop(drop, None)
                    self.n_evict += 1
                    if self.windowed:
                        self._maybe_retire(drop)
            self.segvec[aid] += v
            self.segder[aid] += vd

        if aid > 0 and k % ANCHOR == 0:
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
            return self.pose_of(k)   # post-relax pose, keeps chaining
        return est

    def try_constraint(self, pts, w):
        # ---- verbatim ssp_bounded.BoundedSLAM.try_constraint with (a) point
        # count -> len(w), (b) L.encode -> packed segment-integral encode ----
        k = self.k
        if len(w) < 20:
            return
        me = self.pose_of(k)
        my_aid = self.aid_of(k)
        cands = [aid for aid in self.segvec
                 if abs(aid - my_aid) > self.gap_kf // 5
                 and np.linalg.norm(self.anchors[aid][:2] - me[:2]) < 5.0]
        if not cands:
            return
        cands.sort()
        chains, cur = [], [cands[0]]
        for aid in cands[1:]:
            (cur.append(aid) if aid - cur[-1] <= 2
             else (chains.append(cur), cur := [aid]))
        chains.append(cur)
        chain = min(chains, key=lambda ch: min(
            np.linalg.norm(self.anchors[a][:2] - me[:2]) for a in ch))
        B = sum(self.world_vec_seg(a) for a in chain)
        pose = self.cmatcher.match(B[L.MAIN], pts, w, me)
        pose[2] = S.wrap(pose[2])
        if np.linalg.norm(pose[:2] - me[:2]) > 0.6 \
                or abs(S.wrap(pose[2] - me[2])) > np.deg2rad(7):
            return
        sv = L.ENC.shift(pose[:2]) * self.enc_full.encode(
            pts @ S._rot(pose[2]).T, w)
        c = (np.conj(B) * sv).reshape(L.N_RING, L.N_ANG)
        Br = B.reshape(L.N_RING, L.N_ANG)
        svr = sv.reshape(L.N_RING, L.N_ANG)
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
        c_aid = chain[len(chain) // 2]
        lever = np.linalg.norm(me[:2] - self.anchors[c_aid][:2])
        sig_t = np.sqrt(0.08 ** 2 + (0.05 * lever) ** 2)
        sig_r = np.deg2rad(2.0)
        err = e_weak = e_strong = np.nan
        if self.diag_gt is not None:
            gtk = self.diag_gt
            Zm = L.se2_mul(L.se2_inv(self.anchors[c_aid]),
                           L.se2_mul(pose, L.se2_inv(self.kf_ref[k][1])))
            Zt = L.se2_mul(L.se2_inv(gtk[c_aid * 5]), gtk[k - (k % 5)])
            err = np.linalg.norm(Zm[:2] - Zt[:2])
            errW = S._rot(self.anchors[c_aid][2]) @ (Zm[:2] - Zt[:2])
            e_weak = abs(float(errW @ u_weak))
            e_strong = abs(float(errW @ evecs[:, 1]))
        iv_weak = abs(float((pose[:2] - me[:2]) @ u_weak))
        self.diag.append([err, np.linalg.norm(pose[:2] - me[:2]),
                          abs(S.wrap(pose[2] - me[2])), len(chain), *coh,
                          l_weak, l_strong, u_weak[0], u_weak[1],
                          e_weak, e_strong, iv_weak, ridge_w, ridge_s,
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
        Zk = L.se2_mul(pose, L.se2_inv(rel))
        Z = L.se2_mul(L.se2_inv(self.anchors[c_aid]), Zk)
        if self.inject_rate and self._inj_rng.random() < self.inject_rate:
            if self.inject_mode == "aliased":
                if self._alias is None:
                    a0 = self._inj_rng.uniform(0, 2 * np.pi)
                    self._alias = (self._inj_rng.uniform(0.6, 1.2)
                                   * np.array([np.cos(a0), np.sin(a0)]),
                                   np.deg2rad(self._inj_rng.uniform(6, 12)))
                Z[:2] += self._alias[0]
                Z[2] = S.wrap(Z[2] + self._alias[1])
            else:
                ang = self._inj_rng.uniform(0, 2 * np.pi)
                Z[:2] += self._inj_rng.uniform(0.5, 1.5) \
                    * np.array([np.cos(ang), np.sin(ang)])
                Z[2] = S.wrap(Z[2] + np.deg2rad(self._inj_rng.uniform(5, 15))
                              * self._inj_rng.choice([-1, 1]))
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
        else:
            self._suppress(me[:2], k)
        self.diag[-1][-1] = 1.0


class SegIntSLAM(F.BandSLAM, SegCore):
    """Band/Quant layers over the segment-integral core (MRO:
    BandSLAM -> IntSLAM -> QuantStoreSLAM -> SegCore -> BoundedSLAM)."""

    def __init__(self, renorm_alpha=0.0, **kw):
        super().__init__(**kw)
        self.alpha = renorm_alpha
        enc_main = SegIntEncoder(L.W[L.MAIN], renorm_alpha)
        self.enc_full = SegIntEncoder(L.W, renorm_alpha)
        self.enc01 = SegIntEncoder(L.W[:2 * L.N_ANG], renorm_alpha)
        self.matcher = S.Matcher(enc_main, t_half=0.48, rot_half_deg=9,
                                 rot_step_deg=1.5, perm=(4, L.N_ANG))
        self.cmatcher = S.Matcher(enc_main, t_half=0.72, rot_half_deg=9,
                                  rot_step_deg=1.5, perm=(4, L.N_ANG))


# ---------------------------------------------------------------------------

def selftest():
    rng = np.random.default_rng(7)
    W = L.W
    # 1) quadrature: segint == dense midpoint sub-point limit
    P0 = rng.uniform(-4, 4, (40, 2))
    P1 = P0 + rng.uniform(-0.8, 0.8, (40, 2))
    w = rng.uniform(0.05, 0.3, 40)
    enc = SegIntEncoder(W)
    v = enc.encode(np.vstack([P0, P1]), w)
    n = 400
    ts = (np.arange(n) + 0.5) / n
    pts = (P0[:, None, :] + ts[None, :, None] * (P1 - P0)[:, None, :]
           ).reshape(-1, 2)
    wq = np.repeat(w / n, n)
    vq = np.exp(1j * (pts @ W.T)).T @ wq
    err = np.abs(v - vq).max() / np.abs(v).max()
    print(f"  quadrature (n=400): rel err {err:.2e}")
    assert err < 1e-5
    # 2) derivative: analytic vd == finite difference of rotated encode
    packed = np.vstack([P0, P1])
    v0, vd = seg_encode_der(W, packed, w)
    dth = 1e-6
    R = S._rot(dth)
    vp = enc.encode(packed @ R.T, w)
    vm = enc.encode(packed @ S._rot(-dth).T, w)
    fd = (vp - vm) / (2 * dth)
    err = np.abs(vd - fd).max() / np.abs(fd).max()
    print(f"  d/dtheta analytic vs numeric: rel err {err:.2e}")
    assert err < 1e-4
    assert np.abs(v0 - v).max() == 0.0
    # 3) degenerate-pack primitives == point primitives, BIT-exact on a real
    # scan (encode / fold v / fold vd / shift). The e2e comparison cannot be
    # bit-exact by construction: BLAS GEMM rounds duplicated rows at
    # different row positions differently by ~1 ULP (measured: first est
    # delta 1.8e-15 at fr101 kf 62), and the documented closure-cascade
    # knife-edge amplifies ULPs — so e2e asserts ATE equivalence instead.
    bundle = DS.load("fr101", cap=70)
    rr = DS.clean(bundle, bundle["keys"][62][0])
    beam = bundle["beam"]
    p1, w1 = F.points_from_scan(rr, beam)
    pk, wk = pack_points(rr, beam)
    assert np.abs(L.encode(p1, w1)
                  - SegIntEncoder(L.W).encode(pk, wk)).max() == 0.0
    A = np.exp(1j * (p1 @ L.W.T))
    Cx = p1[:, 0:1] * L.W[:, 1] - p1[:, 1:2] * L.W[:, 0]
    vS, vdS = seg_encode_der(L.W, pk, wk)
    assert np.abs(A.T @ w1 - vS).max() == 0.0
    assert np.abs((1j * Cx * A).T @ w1 - vdS).max() == 0.0
    print("  degenerate-pack primitives == point primitives (bit-exact)")
    cap = 1200
    a = DS.run("fr101", F.BandSLAM, cap=cap, sample="point",
               spec=None, nph=0)
    b = DS.run("fr101", SegIntSLAM, cap=cap, sample=pack_points,
               spec=None, nph=0)
    d = float(np.abs(a["fin"] - b["fin"]).max())
    print(f"  degenerate-pack e2e fr101[:{cap}]: point {a['ate']:.4f}  "
          f"segint {b['ate']:.4f}  max|dpose| {d:.2e} (ULP-amplified)")
    assert abs(a["ate"] - b["ate"]) < 0.02 and d < 5e-3
    # 4) sample_interp(cut<0) == points_from_scan exactly
    bundle = DS.load("fr101", cap=5)
    rr = DS.clean(bundle, bundle["keys"][2][0])
    p1, w1 = F.points_from_scan(rr, bundle["beam"])
    p2, w2 = sample_interp(rr, bundle["beam"], cut_deg=-1)
    assert np.array_equal(p1, p2) and np.array_equal(w1, w2)
    print("  sample_interp(cut<0) == E2 points exactly")
    print("selftest ok", flush=True)


def stage_interp(lg):
    print(f"== {lg}: controls", flush=True)
    for tag, sm in (("shipped seg", "seg"), ("E2 point  ", "point")):
        r = DS.run(lg, F.BandSLAM, sample=sm, spec=None, nph=0)
        print(f"  {lg} {tag}            ATE {r['ate']:6.3f}  med "
              f"{r['med']:6.3f}  loops {r['loops']}", flush=True)
    print(f"== {lg}: interp n_sub x cut_deg (w_mode=arc)", flush=True)
    for n_sub in (2, 3, 5):
        for cut in (45.0, 63.4, 75.0):
            r = DS.run(lg, F.BandSLAM, spec=None, nph=0,
                       sample=lambda rr, b, n=n_sub, c=cut:
                       sample_interp(rr, b, n, c))
            print(f"  {lg} interp n{n_sub} cut{cut:4.1f}  ATE {r['ate']:6.3f}"
                  f"  med {r['med']:6.3f}  loops {r['loops']}", flush=True)
    r = DS.run(lg, F.BandSLAM, spec=None, nph=0,
               sample=lambda rr, b: sample_interp(rr, b, 3, 63.4, "chord"))
    print(f"  {lg} interp n3 cut63.4 CHORD-mass  ATE {r['ate']:6.3f}  "
          f"med {r['med']:6.3f}  loops {r['loops']}", flush=True)


def stage_segint(lg):
    print(f"== {lg}: exact segment integrals", flush=True)
    for tag, sampler in (
            ("arcint (footprint only)  ", pack_arcint),
            ("segint cut63.4 lone=point",
             lambda rr, b: pack_segint(rr, b, 63.4, "point")),
            ("segint cut63.4 lone=arc  ",
             lambda rr, b: pack_segint(rr, b, 63.4, "arc")),
            ("segint cut45.0 lone=arc  ",
             lambda rr, b: pack_segint(rr, b, 45.0, "arc"))):
        r = DS.run(lg, SegIntSLAM, sample=sampler, spec=None, nph=0)
        print(f"  {lg} {tag}  ATE {r['ate']:6.3f}  med {r['med']:6.3f}  "
              f"loops {r['loops']}", flush=True)


def stage_renorm(lg):
    print(f"== {lg}: blanking-vs-weighting (renorm alpha; arcint base)",
          flush=True)
    for alpha in (0.0, 0.5, 1.0):
        r = DS.run(lg, SegIntSLAM, sample=pack_arcint, spec=None, nph=0,
                   renorm_alpha=alpha)
        print(f"  {lg} arcint alpha {alpha:.1f}  ATE {r['ate']:6.3f}  "
              f"med {r['med']:6.3f}  loops {r['loops']}", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if what == "selftest":
        selftest()
    elif what == "interp":
        for lg in sys.argv[2:] or ["fr079"]:
            stage_interp(lg)
    elif what == "segint":
        for lg in sys.argv[2:] or ["fr079"]:
            stage_segint(lg)
    elif what == "renorm":
        for lg in sys.argv[2:] or ["fr079"]:
            stage_renorm(lg)
