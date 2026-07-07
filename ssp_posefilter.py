"""VSA pose-posterior filter: a bandlimited Bayes filter over pose, carried
as a phasor vector (characteristic function on the SSP frequency lattice).

CONCEPT AS BUILT. The translation posterior p(x) over a local window is
represented by its characteristic function on the matched-band lattice
Wm = L.W[MAIN] (4 rings lam 0.25..2 x 60 angles, D=240):

    rho = sum_j p_j * exp(i Wm x_j)          (a bundled SSP *is* the CF)

Heading is a small discrete mixture: H hypotheses theta_h = theta_ref + off_h
(1-deg lattice), each with its own weight pi_h and translation CF rho_h —
the RBPF insight (heading error levers translation, so the two must stay
jointly represented) in O(H*D) memory instead of 30 particle grids.

Operations per keyframe:
  MOTION (predict), exact on the CF, O(H*D): compose the odometry delta —
  each hypothesis shifts by its OWN world-frame translation R(theta_h) d
  (phase multiply), then multiply elementwise by the Gaussian noise kernel's
  CF exp(-0.5 sig_t^2 |w_k|^2); heading noise = second-moment-matched
  tridiagonal diffusion of the mixture (weights AND CFs mix — densities are
  linear).
  MEASUREMENT (correct), hybrid on the coarse grid the matcher already
  scans: decode the prior density on the grid (Re<rho, phi(x)>, clipped),
  multiply by the scan-match likelihood exp(beta * s01) where s01 is the
  normalized correlation score of the scan (encoded at theta_h) against the
  local bundle, renormalize jointly over (h, grid), re-encode each
  hypothesis row to a fresh CF. The soft Bayes product replaces the hard
  innovation gate: an inconsistent peak is suppressed by the prior instead
  of triggering odometry fallback.

COMMIT RULES (replaces single-hypothesis commit): posterior tight
(sig_max < commit_sig_t, heading std < commit_sig_r, no boundary
saturation) -> est = sub-cell-refined posterior mode, segment content
written as usual. Loose (corridor ambiguity) -> carry the posterior, est =
posterior mean for navigation, SUPPRESS segment writes (ambiguous poses
never enter the map; a 1-in-max_suppress forced write keeps the local
bundle alive on long ambiguous spans) and write the span's seq edge at the
posterior's own spread (the fallback-sigma machinery, made
information-honest). On collapse (junction / wall notch), the wide-window
likelihood re-localizes along the corridor, the posterior tightens, and
writes resume. Loop-closure machinery (try_constraint, IRLS+LOO backend)
is inherited unchanged from BoundedSLAM.

Usage:
  python3 ssp_posefilter.py selftest
  python3 ssp_posefilter.py bench [quick]        (corridor/room/sparse, paired)
  python3 ssp_posefilter.py carmen <log> [n] [--base]
"""

import sys
import time

import numpy as np

import ssp_slam as S
import ssp_slam_loop as L
import ssp_bounded as B
from worlds import WORLDS
from bench_loop import multiloop_traj, BEAM, scan_at

ANCHOR = B.ANCHOR
Wm = L.W[L.MAIN]                       # (240, 2) matched-band lattice
W2 = (Wm ** 2).sum(1)                  # |w_k|^2 for the Gaussian noise CF


def _grid(half, step=0.06):
    v = np.arange(-half, half + step / 2, step)
    gx, gy = np.meshgrid(v, v)
    offs = np.stack([gx.ravel(), gy.ravel()], 1)
    E = np.exp(1j * (offs @ Wm.T))                    # (G, D)
    edge = (np.abs(offs).max(1) > half - step * 0.75)  # boundary cells
    return offs, E, edge, len(v)


class PoseFilterSLAM(B.BoundedSLAM):
    """BoundedSLAM with the frontend's gate-and-commit replaced by the CF
    pose filter. Backend (anchors, seq/loop edges, relax) inherited."""

    def __init__(self, robust=True, attempt_every=3, relax_every=5,
                 gap_kf=B.GAP, recent_aids=12):
        super().__init__(robust=robust, attempt_every=attempt_every,
                         relax_every=relax_every, gap_kf=gap_kf,
                         recent_aids=recent_aids)
        # --- filter parameters
        # +-6 deg mixture span is load-bearing on real logs (Intel keyframes
        # rotate up to 5 deg): the +-3 deg lattice clips genuine heading
        # innovation and the accumulated bias kills early loop closures
        # (prefix A/B: rmse 0.874/0 loops at +-3 vs 0.280/7 at +-6)
        self.H = 9
        self.h_spacing = np.deg2rad(1.5)
        self.offs_th = (np.arange(self.H) - (self.H - 1) / 2) * self.h_spacing
        self.meas_beta = 40.0          # likelihood tempering on s01
        self.q_floor = 0.04            # min normalized peak score to use a meas
        # COMMIT = "locally consistent", not "narrow": segments are written
        # anchor-relative, so a coherently-drifted span writes clean content
        # the graph can still fix (wide-isotropic posteriors commit).
        # What corrupts the map is intra-span inconsistency: multimodality,
        # aperture ridges (mean wanders along the corridor), heading spread,
        # window saturation — those suppress.
        self.commit_sig_cap = 0.30     # [m] absolute posterior-spread cap
        self.ridge_sig = 0.15          # [m] weak... strong-axis spread AND
        self.ridge_aniso = 2.5         #     anisotropy -> aperture ridge
        self.mode_rel = 0.30           # secondary mode counts above this
        self.mode_sep = 0.18           # [m] min mode separation
        self.commit_sig_r = np.deg2rad(1.2)
        self.edge_mass_max = 0.05      # boundary saturation -> loose
        # Decode is the inner product with the clipped lattice kernel: the
        # polar lattice holds only 4 radial frequencies, so a density decode
        # is inherently kernel-blurred with ~0.2-rel sidelobes. Subtracting
        # decode_clip*max before clipping kills the sidelobe floor (measured:
        # mean error 30 mm -> <1 mm on tight densities); the residual kernel
        # blur inflates PRIOR spread by a near-constant (~+9-13 cm on tight
        # densities). Harmless downstream: the posterior spread that drives
        # the commit gate is likelihood-dominated wherever it is small
        # (prior sidelobes x likelihood tempering ~ e^-8 cross-terms).
        self.decode_clip = 0.15
        self.prior_floor = 1e-4        # relative density floor after clip
        self.max_suppress = 12         # forced map write every N suppressed kf
        # motion noise (per keyframe): sig = c0 + c1 * |motion|
        self.sig_t0, self.sig_t1 = 0.02, 0.10
        self.sig_r0, self.sig_r1 = np.deg2rad(0.6), 0.10
        # --- grids (tight + wide window; wide used while posterior is loose)
        self.grid_t = _grid(0.48)
        self.grid_w = _grid(0.90)
        # committed-estimate fine stage (mirrors the base matcher's): the
        # mixture lattice is 1 deg / 6 cm; a committed unimodal estimate is
        # refined on the 0.375 deg x 2 cm stage exactly as the base frontend
        self.fine_off = S._grid_offsets(0.06, 0.02)
        self.E_fine = np.exp(1j * (self.fine_off @ Wm.T))
        self.fine_dth = np.deg2rad(0.375) * np.arange(-2, 3)
        # --- state
        self.pf_pi = None              # (H,) mixture weights
        self.pf_rho = None             # (H, D) complex translation CFs
        self.pf_mean = None            # (3,) posterior mean pose
        self.theta_ref = 0.0           # heading lattice center
        self.pf_sig_dr = 0.0           # dead-reckoned spread since last meas
        self.pf_hsig_dr = 0.0
        self._loose = False
        self._supp_run = 0
        self._span_st = 0.03           # info-honest seq sigma accumulators
        self._span_sr = np.deg2rad(0.3)
        self._prev_ret = None
        # --- stats
        self.n_commit = self.n_suppress = self.n_forced = 0
        self.n_nomeas = self.n_qskip = 0
        self.max_supp_run = 0
        self.spread_log = []           # (k, sig_max, hstd, tight)

    # ---- filter core -------------------------------------------------------
    def _pf_init(self, pose):
        self.theta_ref = pose[2]
        self.pf_mean = pose.copy()
        self.pf_pi = np.zeros(self.H)
        self.pf_pi[(self.H - 1) // 2] = 1.0
        phi = np.exp(1j * (Wm @ pose[:2]))
        self.pf_rho = np.tile(phi, (self.H, 1))
        self.pf_sig_dr = 0.0
        self.pf_hsig_dr = 0.0

    def _pf_predict(self, delta):
        """Motion update: exact CF composition with the odometry delta and
        its Gaussian noise kernel; heading diffusion mixes the mixture."""
        trans = np.linalg.norm(delta[:2])
        rot = abs(delta[2])
        sig_t = self.sig_t0 + self.sig_t1 * trans
        sig_r = self.sig_r0 + self.sig_r1 * rot
        th = self.theta_ref + self.offs_th
        # per-hypothesis world-frame shift (heading error levers translation)
        ca, sa = np.cos(th), np.sin(th)
        tx = ca * delta[0] - sa * delta[1]
        ty = sa * delta[0] + ca * delta[1]
        damp = np.exp(-0.5 * sig_t ** 2 * W2)
        self.pf_rho *= np.exp(1j * (tx[:, None] * Wm[:, 0] +
                                    ty[:, None] * Wm[:, 1])) * damp
        # heading: lattice rides the odometry; noise = tridiagonal diffusion
        self.theta_ref = S.wrap(self.theta_ref + delta[2])
        p_n = min(0.5 * (sig_r / self.h_spacing) ** 2, 1 / 3)
        if p_n > 1e-6:
            m = self.pf_pi[:, None] * self.pf_rho
            m_new = (1 - 2 * p_n) * m
            m_new[1:] += p_n * m[:-1]
            m_new[:-1] += p_n * m[1:]
            m_new[0] += p_n * m[0]          # reflect at lattice ends
            m_new[-1] += p_n * m[-1]
            pi_new = (1 - 2 * p_n) * self.pf_pi
            pi_new[1:] += p_n * self.pf_pi[:-1]
            pi_new[:-1] += p_n * self.pf_pi[1:]
            pi_new[0] += p_n * self.pf_pi[0]
            pi_new[-1] += p_n * self.pf_pi[-1]
            self.pf_pi = pi_new / pi_new.sum()
            self.pf_rho = m_new / np.maximum(pi_new, 1e-300)[:, None]
        # navigation mean + dead-reckoned spread
        c, s = np.cos(self.pf_mean[2]), np.sin(self.pf_mean[2])
        self.pf_mean[0] += c * delta[0] - s * delta[1]
        self.pf_mean[1] += s * delta[0] + c * delta[1]
        self.pf_mean[2] = S.wrap(self.pf_mean[2] + delta[2])
        self.pf_sig_dr = np.sqrt(self.pf_sig_dr ** 2 + sig_t ** 2)
        self.pf_hsig_dr = np.sqrt(self.pf_hsig_dr ** 2 + sig_r ** 2)

    def _pf_scores(self, Bv, pts, w, c, offs, E):
        """Normalized correlation surface per heading hypothesis: exactly the
        matcher's translation scan, evaluated for the H mixture headings."""
        th = self.theta_ref + self.offs_th
        Ec = np.exp(1j * (Wm @ c))
        nB = np.linalg.norm(Bv) + 1e-12
        s01 = np.empty((self.H, len(offs)))
        for h in range(self.H):
            P = pts @ S._rot(th[h]).T
            Sv = np.exp(1j * (P @ Wm.T)).T @ w
            cv = np.conj(Bv) * Sv * Ec
            s01[h] = (E @ cv).real / (nB * np.linalg.norm(Sv) + 1e-12)
        return s01

    def _pf_correct(self, s01, c, offs, E, edge):
        """Measurement update on the grid: decode prior, multiply by
        exp(beta*s01), renormalize, re-encode. Returns stats dict."""
        Ec = np.exp(1j * (Wm @ c))
        dec = ((np.conj(self.pf_rho) * Ec) @ E.T).real          # (H, G)
        P = np.clip(dec - self.decode_clip * dec.max(1, keepdims=True),
                    0.0, None)
        P += self.prior_floor * (P.max(1, keepdims=True) + 1e-300)
        P /= P.sum(1, keepdims=True)
        lam = np.exp(self.meas_beta * (s01 - s01.max()))
        Q = self.pf_pi[:, None] * P * lam
        tot = Q.sum()
        if tot <= 0 or not np.isfinite(tot):
            return None
        Q /= tot
        # --- joint moments
        xy = c + offs
        m_g = Q.sum(0)
        mu = m_g @ xy
        d = xy - mu
        cov = (m_g[:, None] * d).T @ d
        evals = np.linalg.eigvalsh(cov)
        sig_max = float(np.sqrt(max(evals[-1], 0.0)))
        sig_min = float(np.sqrt(max(evals[0], 0.0)))
        pi_h = Q.sum(1)
        th = self.theta_ref + self.offs_th
        th_mean = float(pi_h @ th)         # offsets are small: wrap-safe
        hstd = float(np.sqrt(pi_h @ (th - th_mean) ** 2))
        edge_mass = float(Q[:, edge].sum())
        modes = _modes(m_g, offs, rel=self.mode_rel, min_sep=self.mode_sep)
        ridge = (sig_max > self.ridge_sig
                 and sig_max > self.ridge_aniso * max(sig_min, 1e-6))
        tight = (len(modes) < 2 and not ridge
                 and sig_max < self.commit_sig_cap
                 and hstd < self.commit_sig_r
                 and edge_mass < self.edge_mass_max)
        # --- committed estimate: sub-cell-refined posterior mode (fixes grid
        # discretization; identical to the mean for a tight unimodal peak)
        est = np.array([mu[0], mu[1], th_mean])
        if tight:
            n = int(round(np.sqrt(len(offs))))
            g0 = int(np.argmax(m_g))
            ky, kx = divmod(g0, n)
            grid = m_g.reshape(n, n)
            ex, ey = xy[g0]
            step = offs[1, 0] - offs[0, 0]
            if 0 < kx < n - 1:
                ex += S._parabolic(grid[ky, kx - 1], grid[ky, kx],
                                   grid[ky, kx + 1], step)
            if 0 < ky < n - 1:
                ey += S._parabolic(grid[ky - 1, kx], grid[ky, kx],
                                   grid[ky + 1, kx], step)
            j = int(np.argmax(pi_h))
            eth = th[j]
            if 0 < j < self.H - 1:
                eth += S._parabolic(pi_h[j - 1], pi_h[j], pi_h[j + 1],
                                    self.h_spacing)
            est = np.array([ex, ey, eth])
        # --- recenter the heading lattice at the posterior mean; linear
        # split of each old row between the two nearest new bins preserves
        # the heading first moment exactly
        new_ref = th_mean
        Qn = np.zeros_like(Q)
        pos = (th - new_ref) / self.h_spacing + (self.H - 1) / 2
        for j in range(self.H):
            i0 = int(np.floor(pos[j]))
            f = pos[j] - i0
            i0c = min(max(i0, 0), self.H - 1)
            i1c = min(max(i0 + 1, 0), self.H - 1)
            Qn[i0c] += (1 - f) * Q[j]
            Qn[i1c] += f * Q[j]
        pi_new = Qn.sum(1)
        rho_new = np.empty_like(self.pf_rho)
        for i in range(self.H):
            if pi_new[i] > 1e-12:
                q = Qn[i] / pi_new[i]
                rho_new[i] = Ec * (q @ E)
            else:
                rho_new[i] = np.exp(1j * (Wm @ mu))
        self.theta_ref = S.wrap(new_ref)
        self.pf_pi = pi_new / pi_new.sum()
        self.pf_rho = rho_new
        self.pf_mean = np.array([mu[0], mu[1], S.wrap(th_mean)])
        self.pf_sig_dr = sig_max
        self.pf_hsig_dr = hstd
        return dict(tight=tight, sig_max=sig_max, sig_min=sig_min,
                    hstd=hstd, edge_mass=edge_mass, n_modes=len(modes),
                    ridge=ridge, est=est, mu=mu, m_g=m_g)

    def _pf_meas(self, Bv, pts, w):
        offs, E, edge, _ = self.grid_w if self._loose else self.grid_t
        c = self.pf_mean[:2].copy()
        s01 = self._pf_scores(Bv, pts, w, c, offs, E)
        if s01.max() < self.q_floor:
            self.n_qskip += 1
            return None
        st = self._pf_correct(s01, c, offs, E, edge)
        if st is not None:
            self._loose = not st["tight"]
        return st

    def _pf_refine(self, Bv, pts, w, est):
        """Sub-lattice refinement of a COMMITTED estimate (0.375 deg x 2 cm
        stage + parabolic, mirroring the base matcher's fine stage). Touches
        only where the pose is written; the carried posterior is untouched."""
        th_scores, results = [], []
        Ec = np.exp(1j * (Wm @ est[:2]))
        for dth in self.fine_dth:
            P = pts @ S._rot(est[2] + dth).T
            Sv = np.exp(1j * (P @ Wm.T)).T @ w
            cv = np.conj(Bv) * Sv * Ec
            sc = (self.E_fine @ cv).real
            k = int(np.argmax(sc))
            th_scores.append(sc[k])
            results.append((sc, k))
        j = int(np.argmax(th_scores))
        th_f = est[2] + self.fine_dth[j]
        if 0 < j < len(self.fine_dth) - 1:
            th_f += S._parabolic(th_scores[j - 1], th_scores[j],
                                 th_scores[j + 1], np.deg2rad(0.375))
        sc, k = results[j]
        n = int(round(np.sqrt(len(self.fine_off))))
        ky, kx = divmod(k, n)
        grid = sc.reshape(n, n)
        t = est[:2] + self.fine_off[k]
        if 0 < kx < n - 1:
            t[0] += S._parabolic(grid[ky, kx - 1], grid[ky, kx],
                                 grid[ky, kx + 1], 0.02)
        if 0 < ky < n - 1:
            t[1] += S._parabolic(grid[ky - 1, kx], grid[ky, kx],
                                 grid[ky + 1, kx], 0.02)
        return np.array([t[0], t[1], th_f])

    def _pf_frame_jump(self, old, new):
        """Backend relaxation moved the current pose: ride the posterior
        along (translation = exact phase multiply; the small residual
        rotation about the robot is applied to the lattice center only —
        first-order, same order as the graph's own linearization)."""
        dxy = new[:2] - old[:2]
        dth = S.wrap(new[2] - old[2])
        self.pf_rho *= np.exp(1j * (Wm @ dxy))
        self.theta_ref = S.wrap(self.theta_ref + dth)
        self.pf_mean = new.copy()

    # ---- lifecycle (mirrors BoundedSLAM.add_keyframe; frontend replaced) ---
    def add_keyframe(self, pts, w, guess):
        self.k += 1
        k = self.k
        if k == 0:
            self._pf_init(guess)
        else:
            delta = L.se2_mul(L.se2_inv(self._prev_ret), guess)
            self._pf_predict(delta)
        if k % ANCHOR == 0:
            self.anchors.append(guess.copy())
        aid = len(self.anchors) - 1

        committed = (k == 0)
        est = guess.copy() if k == 0 else self.pf_mean.copy()
        st = None
        if k > 0 and len(pts) >= 20:
            Bv = self.local_bundle(self.pf_mean[:2])[L.MAIN]
            if np.abs(Bv).sum() > 0:
                st = self._pf_meas(Bv, pts, w)
                if st is not None:
                    committed = st["tight"]
                    est = st["est"].copy() if committed \
                        else self.pf_mean.copy()
                    if committed:
                        est = self._pf_refine(Bv, pts, w, est)
                        est[2] = S.wrap(est[2])
                        self.pf_mean = est.copy()
                        # session-relative coherence baseline (as in base)
                        W01 = L.W[:2 * L.N_ANG]
                        PW = pts @ S._rot(est[2]).T + est[:2]
                        sv01 = np.exp(1j * (PW @ W01.T)).T @ w
                        B01 = Bv[:2 * L.N_ANG].reshape(2, L.N_ANG)
                        s01r = sv01.reshape(2, L.N_ANG)
                        c01 = float(np.mean(
                            (np.conj(B01) * s01r).sum(1).real
                            / (np.linalg.norm(B01, axis=1)
                               * np.linalg.norm(s01r, axis=1) + 1e-12)))
                        self.coh_ref = c01 if self.coh_ref is None \
                            else 0.95 * self.coh_ref + 0.05 * c01
                else:
                    self.n_nomeas += 1
            else:
                self.n_nomeas += 1
        elif k > 0:
            self.n_nomeas += 1
        est[2] = S.wrap(est[2])

        suppress = (k > 0) and not committed
        if suppress:
            self.n_suppress += 1
            self._supp_run += 1
            self.max_supp_run = max(self.max_supp_run, self._supp_run)
        else:
            self.n_commit += int(k > 0)
            self._supp_run = 0
        self.spread_log.append((k, self.pf_sig_dr, self.pf_hsig_dr,
                                not suppress))
        self._span_fallback = getattr(self, "_span_fallback", False) or suppress
        # seq sigma: committed frames leave the base 0.03 (the posterior
        # spread is ABSOLUTE uncertainty; the 5-frame relative chain error
        # stays 2-3 cm because consecutive likelihood errors are shared).
        # Only suppressed frames — where the mean can slide within the span —
        # inflate the edge, by their own posterior spread, AND ONLY IN THE
        # AMBIGUOUS COMPONENT: heading-spread fires must not soften the
        # translation chain (isotropic inflation measured on Intel: 35% of
        # spans went soft on hstd flapping alone, relax jerk 0.60 -> 1.27 m
        # and closures warped the early chain, rmse 2.44 -> 4.6).
        if suppress:
            t_amb = st is None or st["ridge"] or st["n_modes"] >= 2 \
                or st["sig_max"] >= self.commit_sig_cap \
                or st["edge_mass"] >= self.edge_mass_max
            h_amb = st is None or st["hstd"] >= self.commit_sig_r
            if t_amb:
                self._span_st = max(self._span_st, 0.10,
                                    min(self.pf_sig_dr, 0.30))
            if h_amb:
                self._span_sr = max(self._span_sr, np.deg2rad(1.5),
                                    min(self.pf_hsig_dr, np.deg2rad(3.0)))

        if k % ANCHOR == 0:
            self.anchors[aid] = est.copy()
        rel = L.se2_mul(L.se2_inv(self.anchors[aid]), est)
        self.kf_ref.append((aid, rel))

        # WRITE-ON-COMMIT: ambiguous poses never enter the map (forced
        # 1-in-max_suppress write keeps the local bundle alive on long spans)
        write = len(pts) > 0 and not suppress
        if len(pts) > 0 and suppress and self._supp_run >= self.max_suppress:
            write = True
            self._supp_run = 0
            self.n_forced += 1
        if write:
            PA = pts @ S._rot(rel[2]).T + rel[:2]
            A = np.exp(1j * (PA @ L.W.T))
            v = A.T @ w
            Cx = PA[:, 0:1] * L.W[:, 1] - PA[:, 1:2] * L.W[:, 0]
            vd = (1j * Cx * A).T @ w
            if aid not in self.segvec:
                self.segvec[aid] = np.zeros(L.W.shape[0], complex)
                self.segder[aid] = np.zeros(L.W.shape[0], complex)
                cell = tuple((self.anchors[aid][:2] // B.CELL).astype(int))
                self.cells.setdefault(cell, []).append(aid)
                if len(self.cells[cell]) > B.CELL_CAP:
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
            st_, sr_ = self._span_st, self._span_sr
            self._span_fallback = False
            self._span_st = 0.03
            self._span_sr = np.deg2rad(0.3)
            self.edges.append((aid - 1, aid, Z, 1 / st_, 1 / sr_, "seq"))

        if k % self.attempt_every == 0:
            self.try_constraint(pts, w)
        ret = est
        if k % self.relax_every == 0 and self.dirty:
            self.relax()
            newp = self.pose_of(k)
            self._pf_frame_jump(est, newp)
            ret = newp
        self._prev_ret = ret.copy()
        return ret


# ---------------------------------------------------------------------------
# Selftests
# ---------------------------------------------------------------------------

def _gauss2(offs, mu, sig):
    d = offs - mu
    return np.exp(-0.5 * ((d[:, 0] / sig[0]) ** 2 + (d[:, 1] / sig[1]) ** 2))


def _decode(rho, c, offs, E, clip=0.15):
    Ec = np.exp(1j * (Wm @ c))
    dec = (E @ np.conj(rho * np.conj(Ec))).real
    return np.clip(dec - clip * dec.max(), 0, None)


def _moments(p, xy):
    p = p / p.sum()
    mu = p @ xy
    d = xy - mu
    return mu, (p[:, None] * d).T @ d


def _modes(m, offs, rel=0.25, min_sep=0.15):
    n = int(round(np.sqrt(len(m))))
    g = m.reshape(n, n)
    thr = rel * g.max()
    peaks = []
    for ky in range(n):
        for kx in range(n):
            v = g[ky, kx]
            if v < thr:
                continue
            nb = g[max(0, ky - 1):ky + 2, max(0, kx - 1):kx + 2]
            if v >= nb.max():
                peaks.append((v, offs[ky * n + kx]))
    peaks.sort(key=lambda t: -t[0])
    kept = []
    for v, p in peaks:
        if all(np.linalg.norm(p - q) >= min_sep for _, q in kept):
            kept.append((v, p))
    return kept


def selftest():
    rng = np.random.default_rng(0)
    offs, E, edge, nside = _grid(0.48)
    xy = offs  # grid in local coords, c = origin of the test frame

    # --- 1. CF round trip: encode density -> decode -> moments -------------
    c = np.array([3.7, -1.2])
    p = _gauss2(offs, np.array([0.10, -0.05]), (0.09, 0.06)) \
        + 0.5 * _gauss2(offs, np.array([-0.20, 0.12]), (0.05, 0.05))
    p /= p.sum()
    rho = np.exp(1j * (Wm @ c)) * (p @ E)
    dec = _decode(rho, c, offs, E)
    mu_t, cov_t = _moments(p, xy)
    mu_d, cov_d = _moments(dec, xy)
    err_mu = np.linalg.norm(mu_d - mu_t)
    dsig = np.sqrt(np.abs(np.diag(cov_d))) - np.sqrt(np.abs(np.diag(cov_t)))
    corr = np.corrcoef(p, dec)[0, 1]
    print(f"ST1 roundtrip: mean err {err_mu * 1000:.2f} mm  "
          f"kernel sigma inflation {dsig[0] * 100:+.2f}/{dsig[1] * 100:+.2f} cm  "
          f"corr {corr:.3f}")
    # means are near-exact; spreads live in kernel-inflated space (see
    # decode_clip note in PoseFilterSLAM.__init__) — bound, don't deny it
    assert err_mu < 0.006 and corr > 0.55 and np.all(dsig > -0.01) \
        and np.all(dsig < 0.16)
    # delta round trip: argmax lands on the encoded cell
    g0 = 7 * nside + 3
    rho_d = np.exp(1j * (Wm @ (c + offs[g0])))
    assert int(np.argmax(_decode(rho_d, c, offs, E))) == g0
    # bimodal round trip: both modes recovered at their true positions
    pb = _gauss2(offs, np.array([-0.18, 0.0]), (0.05, 0.03)) \
        + _gauss2(offs, np.array([0.18, 0.0]), (0.05, 0.03))
    pb /= pb.sum()
    rho_b = np.exp(1j * (Wm @ c)) * (pb @ E)
    mb = _modes(_decode(rho_b, c, offs, E), offs)
    assert len(mb) == 2, f"bimodal roundtrip lost a mode: {mb}"
    locs = sorted(m[1][0] for m in mb)
    assert abs(locs[0] + 0.18) < 0.05 and abs(locs[1] - 0.18) < 0.05
    print(f"ST1 bimodal: modes at x = {locs[0]:+.3f}, {locs[1]:+.3f} "
          f"(true -0.180, +0.180)")

    # --- 2. motion update vs Monte-Carlo convolution ------------------------
    p0 = _gauss2(offs, np.array([0.0, 0.0]), (0.07, 0.07))
    p0 /= p0.sum()
    delta, sig_n = np.array([0.31, -0.17]), 0.06
    rho0 = np.exp(1j * (Wm @ c)) * (p0 @ E)
    rho1 = rho0 * np.exp(1j * (Wm @ delta)) * np.exp(-0.5 * sig_n ** 2 * W2)
    c1 = c + delta
    dec1 = _decode(rho1, c1, offs, E)
    # MC: sample from p0 (cell + in-cell jitter), push through the same noise
    N = 200_000
    cells = rng.choice(len(offs), N, p=p0)
    x = c + offs[cells] + rng.uniform(-0.03, 0.03, (N, 2)) \
        + delta + rng.normal(0, sig_n, (N, 2))
    rho_mc = np.exp(1j * (x @ Wm.T)).mean(0)
    dec_mc = _decode(rho_mc, c1, offs, E)
    mu_a, cov_a = _moments(dec1, xy)
    mu_b, cov_b = _moments(dec_mc, xy)
    err = np.linalg.norm(mu_a - mu_b)
    dsg = np.sqrt(np.abs(np.diag(cov_a))) - np.sqrt(np.abs(np.diag(cov_b)))
    print(f"ST2 motion vs MC: mean err {err * 1000:.2f} mm  "
          f"sigma diff {dsg[0] * 100:+.2f}/{dsg[1] * 100:+.2f} cm")
    assert err < 0.005 and np.all(np.abs(dsg) < 0.01)

    # --- 3. bimodal carry-and-collapse (corridor-fork toy) ------------------
    slam = PoseFilterSLAM()
    pose0 = np.array([2.0, 1.0, 0.15])
    slam._pf_init(pose0)
    step = np.array([0.05, 0.0, 0.0])
    modes_xy = None
    for t in range(8):
        slam._pf_predict(step)
        offs_c, E_c, edge_c, _ = slam.grid_t
        cgr = slam.pf_mean[:2].copy()
        th = slam.theta_ref + slam.offs_th
        # ambiguous likelihood: two along-x modes +-0.18 m about the mean,
        # tight in y; heading peaked at the true heading
        hfac = np.exp(-0.5 * ((th - pose0[2]) / np.deg2rad(1.2)) ** 2)
        sA = _gauss2(offs_c, np.array([-0.18, 0.0]), (0.05, 0.03))
        sB = _gauss2(offs_c, np.array([+0.18, 0.0]), (0.05, 0.03))
        s01 = 0.02 + 0.6 * hfac[:, None] * (sA + sB)[None, :]
        st = slam._pf_correct(s01, cgr, offs_c, E_c, edge_c)
        modes_xy = _modes(st["m_g"], offs_c)
    assert not st["tight"], "ambiguous posterior must not commit"
    assert len(modes_xy) == 2, f"expected 2 carried modes, got {len(modes_xy)}"
    split = min(modes_xy[0][0], modes_xy[1][0]) / max(modes_xy[0][0],
                                                      modes_xy[1][0])
    print(f"ST3a carry: 2 modes held over 8 frames, sig_max "
          f"{st['sig_max'] * 100:.1f} cm (loose), peak ratio {split:.2f}")
    assert split > 0.5
    # collapse: junction likelihood keeps only mode B (the true one)
    true_xy = slam.pf_mean[:2] + np.array([0.18, 0.0])
    for t in range(3):
        slam._pf_predict(step)
        true_xy = true_xy + S._rot(pose0[2]) @ step[:2]
        offs_c, E_c, edge_c, _ = slam.grid_t
        cgr = slam.pf_mean[:2].copy()
        th = slam.theta_ref + slam.offs_th
        hfac = np.exp(-0.5 * ((th - pose0[2]) / np.deg2rad(1.2)) ** 2)
        rel = true_xy - cgr
        sB = _gauss2(offs_c, rel, (0.05, 0.03))
        s01 = 0.02 + 0.6 * hfac[:, None] * sB[None, :]
        st = slam._pf_correct(s01, cgr, offs_c, E_c, edge_c)
    err_c = np.linalg.norm(st["est"][:2] - true_xy)
    print(f"ST3b collapse: tight={st['tight']}  sig_max "
          f"{st['sig_max'] * 100:.1f} cm  est err {err_c * 100:.1f} cm  "
          f"heading err {np.degrees(abs(S.wrap(st['est'][2] - pose0[2]))):.2f} deg")
    assert st["tight"], "posterior must collapse and commit at the junction"
    assert err_c < 0.05
    assert abs(S.wrap(st["est"][2] - pose0[2])) < np.deg2rad(0.5)
    print("selftest ok")


# ---------------------------------------------------------------------------
# Bench: paired vs BoundedSLAM on the synthetic worlds
# ---------------------------------------------------------------------------

def run(world, n=750, seed=1, cls=B.BoundedSLAM, laps=3):
    S.RNG = np.random.default_rng(seed)
    segs = WORLDS[world]()
    gt = multiloop_traj(n, laps=laps)
    odo = B.sim_odometry(gt)
    slam = cls(robust=True)
    slam.diag_gt = gt
    est = np.zeros((n, 3))
    t0 = time.time()
    for k in range(n):
        r = scan_at(segs, gt[k])
        pts, w, _ = S.scan_to_samples(r, BEAM)
        guess = odo[0] if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odo[k - 1]), odo[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
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
    seqs = [e for e in slam.edges if e[5] == "seq"]
    fb = float(np.mean([1 / e[3] > 0.09 for e in seqs])) if seqs else 0.0
    out = dict(ate=np.sqrt((ate ** 2).mean()), ate_max=ate.max(),
               edges=n_loop, false=false, pruned=slam.n_pruned,
               fb=fb, secs=time.time() - t0)
    if isinstance(slam, PoseFilterSLAM):
        tot = max(slam.n_commit + slam.n_suppress, 1)
        out.update(commit=slam.n_commit / tot, forced=slam.n_forced,
                   nomeas=slam.n_nomeas, qskip=slam.n_qskip,
                   maxrun=slam.max_supp_run)
    return out


def bench(worlds=("corridor", "room", "sparse"), seeds=(1, 2, 3, 4)):
    for world in worlds:
        print(f"== {world}")
        base = []
        for tag, cls in (("BoundedSLAM", B.BoundedSLAM),
                         ("PoseFilter", PoseFilterSLAM)):
            rs = [run(world, seed=s, cls=cls) for s in seeds]
            ates = np.array([r["ate"] for r in rs])
            if tag == "BoundedSLAM":
                base = ates
            d = ates - base
            extra = ""
            if "commit" in rs[0]:
                extra = (f"  commit {100 * np.mean([r['commit'] for r in rs]):4.1f}%"
                         f"  forced {np.mean([r['forced'] for r in rs]):4.1f}"
                         f"  maxrun {np.mean([r['maxrun'] for r in rs]):4.0f}")
            print(f"{tag:<12} ATE {100 * ates.mean():7.1f} cm "
                  f"[{' '.join(f'{100 * a:.0f}' for a in ates)}] "
                  f"(paired {100 * d.mean():+7.1f}, {np.sum(d < 0)}/{len(d)} better)  "
                  f"edges {np.mean([r['edges'] for r in rs]):5.1f}  "
                  f"false {np.mean([r['false'] for r in rs]):4.1f}  "
                  f"fb-seq {100 * np.mean([r['fb'] for r in rs]):4.1f}%  "
                  f"{np.mean([r['secs'] for r in rs]):5.1f}s{extra}", flush=True)


# ---------------------------------------------------------------------------
# CARMEN driver (mirrors ssp_bounded_carmen.py)
# ---------------------------------------------------------------------------

def carmen(path, limit=10 ** 9, base=False):
    import ssp_slam_carmen as C
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)[:limit]
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])
    print(f"[{path}] {len(scans)} scans -> {n} keyframes "
          f"({'BoundedSLAM' if base else 'PoseFilterSLAM'})")
    cls = B.BoundedSLAM if base else PoseFilterSLAM
    slam = cls(robust=True, attempt_every=4, relax_every=25,
               gap_kf=300, recent_aids=12)
    if not base:
        slam.sig_t0, slam.sig_t1 = 0.01, 0.10   # real odom, keyframed 0.10 m
        slam.sig_r0 = np.deg2rad(0.4)
    est = np.zeros((n, 3))
    t0 = time.time()
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < 40.0, r, np.inf)
        pts, w, _ = S.scan_to_samples(rr, beam)
        guess = opose if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
        if k % 1000 == 0:
            print(f"  kf {k}/{n} t={time.time() - t0:.0f}s "
                  f"loops={sum(1 for e in slam.edges if e[5] == 'loop')}",
                  flush=True)
    if slam.dirty:
        slam.relax()
    dt = time.time() - t0
    fin = np.stack([slam.pose_of(k) for k in range(n)])
    seqs = [e for e in slam.edges if e[5] == "seq"]
    fb = float(np.mean([1 / e[3] > 0.05 for e in seqs])) if seqs else 0.0
    n_loop = sum(1 for e in slam.edges if e[5] == "loop")
    msg = (f"done: {dt:.0f}s ({dt / n * 1e3:.0f} ms/kf) loops={n_loop} "
           f"pruned={slam.n_pruned} fb-seq={100 * fb:.1f}%")
    if not base:
        tot = max(slam.n_commit + slam.n_suppress, 1)
        msg += (f" commit={100 * slam.n_commit / tot:.1f}% "
                f"forced={slam.n_forced} nomeas={slam.n_nomeas} "
                f"qskip={slam.n_qskip} maxrun={slam.max_supp_run}")
    print(msg)
    ref = C.parse_flaser(path.replace(".log", ".gfs.log"))
    rts = np.array([t for _, _, t in ref])
    rxy = np.stack([p[:2] for _, p, _ in ref])
    j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
    good = np.abs(rts - kts[j]) < 0.3
    al = C.align_se2(fin[j[good], :2], rxy[good])
    e = np.linalg.norm(al - rxy[good], axis=1)
    print(f"ATE vs corrected log: rmse {np.sqrt((e ** 2).mean()):.3f} m  "
          f"median {np.median(e):.3f} m  max {e.max():.3f} m")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if mode == "selftest":
        selftest()
    elif mode == "bench":
        if "quick" in sys.argv:
            bench(worlds=("corridor",), seeds=(1, 2))
        else:
            bench()
    elif mode == "carmen":
        path = sys.argv[2]
        lim = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() \
            else 10 ** 9
        carmen(path, lim, base="--base" in sys.argv)
    else:
        raise SystemExit(f"unknown mode {mode}")
