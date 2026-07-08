"""Iterative slow-to-fast scale-cascaded loop-closure verification.

This is classic multi-wavelength phase unwrapping applied to loop closure, on
the shipped ring-major polar lattice (ssp_slam_loop.py: rings lam
0.25/0.5/1/2/5.3/12.8 x 60 angles, D=360). Given a candidate closure
(query scan S, old-pass map bundle B, seed pose), the cascade walks the rings
SLOW->FAST (12.8, 5.3, 2, 1, 0.5, 0.25 m):

  Stage 0  slowest ring: wide phase-correlation search (rotation via exact
           lattice permutation x translation) -> rough (theta_0, t_0).
  Stage i  seed from (theta_{i-1}, t_{i-1}); translation window ~= precision of
           the previous stage (~lam_{i-1}/4) + small rotation refinement, using
           ring i's phase only. Ring i resolves within +-lam_i before wrapping,
           so the seed must be within lam_i of truth (the unwrapping condition).
           Produce (theta_i, t_i), a per-stage correction d_i = |t_i - t_{i-1}|,
           and a per-ring coherence coh_i at the stage-i pose.

THE ACCEPTANCE TEST is per-match multi-scale consistency (the corridor-twin
discriminator this file exists to evaluate):

  unwrap-consistency : reject if any d_i exceeds the predecessor ring's
                       ambiguity radius (~lam_{i-1}/2): the scales failed to
                       unwrap to one consistent transform.
  coherence-profile  : a GENUINE closure keeps coh_i roughly flat or rising as
                       scales sharpen; a corridor-twin ALIAS shows coh
                       collapsing coarse->fine (coarse locks the twin, fine
                       sees noise). The statistic is coh_fine/coh_coarse (a
                       ratio, hence SCALE-INVARIANT: it factors out the
                       per-candidate absolute coherence level that does not
                       transfer across domains, per the veto history).
  final-precision    : the finest stage's coherence and correction within noise.

Everything is reused by import; nothing here edits the shipped modules.

Run:  python3 ssp_cascade.py [selftest|roc|intel|all]
"""

import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ssp_slam as S
import ssp_slam_loop as L
import ssp_slam_carmen as C
import worlds

# ---------------------------------------------------------------------------
# Lattice handles (reused from ssp_slam_loop)
# ---------------------------------------------------------------------------
LAMS = L.LAMS                       # [0.25, 0.5, 1, 2, 5.3, 12.8]
N_ANG = L.N_ANG
N_RING = L.N_RING
RINGS = L._RINGS                    # ring index of each of the 360 features
ORDER = [5, 4, 3, 2, 1, 0]          # slow -> fast: 12.8, 5.3, 2, 1, 0.5, 0.25
FINE_STAGES = (4, 5)                # lam 0.5, 0.25  (finest two in cascade order)
COARSE_STAGES = (0, 1)              # lam 12.8, 5.3


def world_enc(pts, w, pose):
    """World-frame encoding of a scan observed at `pose` (x, y, theta)."""
    return L.ENC.shift(pose[:2]) * L.encode(pts @ S._rot(pose[2]).T, w)


def ring_coherence(B, sv, ring):
    """Normalized per-ring correlation (cosine similarity) between the map
    bundle B and the query encoding sv, on `ring` only. Matches the shipped
    per-ring coherence definition in ssp_hier."""
    m = RINGS == ring
    br, svr = B[m], sv[m]
    den = np.linalg.norm(br) * np.linalg.norm(svr)
    if den < 1e-12:
        return 0.0
    return float(np.real(np.conj(br) @ svr) / den)


# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------

def cascade(B, pts, w, seed, coarse_half=2.5, coarse_rot=6, verbose=False):
    """Slow-to-fast scale-cascaded registration of query scan (pts, w) against
    world-frame bundle B, seeded at `seed`=(x, y, theta).

    Returns a dict with the per-stage track and the acceptance statistics."""
    S0 = L.encode(pts, w)                       # robot-frame full encoding
    t = np.asarray(seed[:2], float).copy()
    m_cur = int(round(seed[2] * N_ANG / np.pi))  # heading as a lattice index
    theta = m_cur * np.pi / N_ANG

    stages = []
    prev_lam = None
    for si, ring in enumerate(ORDER):
        lam = LAMS[ring]
        mask = RINGS == ring
        if si == 0:
            half = coarse_half
            step = max(lam / 8.0, half / 12.0)
            rot_range = range(m_cur - coarse_rot, m_cur + coarse_rot + 1)
        else:
            half = prev_lam / 4.0               # window ~ precision of stage i-1
            step = lam / 8.0
            rot_range = range(m_cur - 1, m_cur + 2)  # small heading refinement

        best = (-np.inf, m_cur, t)
        for m in rot_range:
            Sm = L.rot_permute(S0, m)
            G, sc = L.grid_scores(B, Sm, t, half, step, mask)
            k = int(np.argmax(sc))
            if sc[k] > best[0]:
                best = (sc[k], m, G[k])
        _, m_cur, t_new = best
        theta = m_cur * np.pi / N_ANG
        d_i = float(np.linalg.norm(t_new - t))
        t = t_new

        sv = world_enc(pts, w, np.array([t[0], t[1], theta]))
        coh_i = ring_coherence(B, sv, ring)
        stages.append(dict(stage=si, ring=ring, lam=lam, t=t.copy(),
                           theta=theta, d=d_i, coh=coh_i))
        prev_lam = lam
        if verbose:
            print(f"  stage {si} ring{ring} lam{lam:5.2f}  "
                  f"t=({t[0]:7.3f},{t[1]:7.3f}) th={np.degrees(theta):6.1f}  "
                  f"d={d_i:.3f}  coh={coh_i:+.3f}")

    coh = np.array([s["coh"] for s in stages])
    d = np.array([s["d"] for s in stages])
    lams = np.array([s["lam"] for s in stages])

    # ---- statistics -------------------------------------------------------
    coh_coarse = coh[list(COARSE_STAGES)].mean()
    coh_fine = coh[list(FINE_STAGES)].mean()
    # profile ratio: scale-invariant coarse->fine coherence transfer
    profile_ratio = coh_fine / max(coh_coarse, 1e-6)
    # slope of coh vs stage index (rising = genuine, falling = twin alias)
    idx = np.arange(len(coh))
    profile_slope = float(np.polyfit(idx, coh, 1)[0])
    # single-scale baseline: the shipped veto's fine-ring coherence
    single_fine = coh_fine

    # ---- acceptance tests -------------------------------------------------
    # unwrap-consistency: correction at stage i must stay within the previous
    # ring's ambiguity radius lam_{i-1}/2
    unwrap_ok = True
    unwrap_viol = 0.0
    for si in range(1, len(stages)):
        radius = lams[si - 1] / 2.0
        if d[si] > radius:
            unwrap_ok = False
            unwrap_viol = max(unwrap_viol, d[si] / radius)
    # final precision: finest correction within its own ambiguity radius
    final_ok = d[-1] <= lams[-1] / 2.0

    return dict(stages=stages, coh=coh, d=d, pose=np.array([t[0], t[1], theta]),
                coh_coarse=coh_coarse, coh_fine=coh_fine,
                profile_ratio=profile_ratio, profile_slope=profile_slope,
                single_fine=single_fine, unwrap_ok=unwrap_ok,
                unwrap_viol=unwrap_viol, final_ok=final_ok)


def accept(B, pts, w, seed, profile_thresh=0.55, final_coh=0.15, **kw):
    """Drop-in acceptance predicate. Returns (accepted, refined_pose, info)."""
    r = cascade(B, pts, w, seed, **kw)
    ok = (r["unwrap_ok"] and r["final_ok"]
          and r["profile_ratio"] >= profile_thresh
          and r["coh_fine"] >= final_coh)
    return ok, r["pose"], r


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def auc(pos, neg):
    """AUC via Mann-Whitney: P(score(genuine) > score(twin)). pos/neg are the
    statistic values on genuine / twin candidates (higher = more genuine)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return wins / (len(pos) * len(neg))


def cohens_d(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    sp = np.sqrt((pos.var() + neg.var()) / 2) + 1e-12
    return (pos.mean() - neg.mean()) / sp


# ===========================================================================
# 1. SELF-TEST on synthetic geometry
# ===========================================================================

def _bay_world(n_bays=6, period=6.0, width=3.0, texture=0.28, shared=0.0):
    """Straight self-similar corridor built to isolate the corridor-twin regime.

    COARSE envelope (aliases across bays): identical wide alcove recess every
    `period` m, so at lam >= 5 m every bay looks the same. FINE fingerprint
    (unique per bay): the corridor walls are drawn as a jagged polyline whose
    vertical bumps (amplitude `texture` ~ the fine wavelengths, spacing 0.3 m)
    are seeded by the bay index. At lam 0.25/0.5 m the bumps are resolved and
    every bay is distinct; at lam >= 2 m they blur into a smooth wall. A twin
    (bay k scan vs bay j map) therefore matches at coarse rings and decoheres
    at fine rings, exactly the aperture/self-similarity trap of R3/R4."""
    segs = []

    def seg(a, b):
        segs.append([a, b])

    x1 = n_bays * period
    yb, yt = 0.0, width
    dx = 0.3
    aw, ad = 2.2, 1.2               # alcove width, depth (coarse periodic feature)
    nx = int(period / dx) + 2
    # shared fine profile: same in every bay (aligns for ANY integer-period
    # twin) -> a floor on twin fine coherence, modelling common corridor detail.
    # Fixed seed so the world is deterministic regardless of call order.
    shrng = np.random.default_rng(77)
    sh_b = shrng.standard_normal(nx)
    sh_t = shrng.standard_normal(nx)
    for b in range(n_bays):
        rng = np.random.default_rng(1000 + b)      # per-bay fine fingerprint
        x = b * period
        cx = b * period + period / 2
        al, ar = cx - aw / 2, cx + aw / 2
        # jagged bottom + top walls across this bay, bumps = shared + per-bay
        xs = np.arange(x, x + period + dx / 2, dx)
        m = len(xs)
        bump_b = texture * (shared * sh_b[:m] + (1 - shared) * rng.standard_normal(m))
        bump_t = texture * (shared * sh_t[:m] + (1 - shared) * rng.standard_normal(m))
        for i in range(len(xs) - 1):
            seg((xs[i], yb + bump_b[i]), (xs[i + 1], yb + bump_b[i + 1]))
            # top wall detours into the alcove recess over [al, ar]
            xa, xbn = xs[i], xs[i + 1]
            if xbn <= al or xa >= ar:
                seg((xa, yt + bump_t[i]), (xbn, yt + bump_t[i + 1]))
        # alcove box (shared coarse feature)
        seg((al, yt), (al, yt + ad))
        seg((al, yt + ad), (ar, yt + ad))
        seg((ar, yt + ad), (ar, yt))
    return np.array(segs, float), period, width


def _bay_scan(segs, pose, rng):
    """Simulate one 360-deg scan at `pose` with an independent RNG."""
    saved = S.RNG
    S.RNG = rng
    try:
        r, beam = S.simulate_scan(segs, pose)
        pts, w, _ = S.scan_to_samples(r, beam)
    finally:
        S.RNG = saved
    return pts, w


def _add_clutter(pts, w, pose, rng, frac=1.0, extent=6.0):
    """Dilute a scan with uncorrelated clutter points in the robot frame,
    modelling furniture/people not present in the old map. Adds broadband
    energy that lowers per-ring coherence at ALL wavelengths ~proportionally
    (common-mode), unlike range noise which selectively kills the fine rings."""
    if len(pts) == 0:
        return pts, w
    n_add = int(frac * len(pts))
    cl = rng.uniform(-extent, extent, (n_add, 2))
    cl = cl[np.linalg.norm(cl, axis=1) < extent]
    wc = np.full(len(cl), float(np.mean(w)))
    return np.vstack([pts, cl]), np.concatenate([w, wc])


def _bay_bundle(segs, center, heading, rng, n=5, spread=0.35):
    """Old-pass map bundle: a few scans near (center, heading)."""
    B = np.zeros(L.W.shape[0], complex)
    for _ in range(n):
        p = np.array([center[0] + rng.normal(0, spread),
                      center[1] + rng.normal(0, spread * 0.4),
                      heading + rng.normal(0, np.deg2rad(2))])
        pts, w = _bay_scan(segs, p, rng)
        if len(pts) >= 20:
            B += world_enc(pts, w, p)
    return B


def self_test():
    print("=" * 70)
    print("SELF-TEST: genuine revisit vs constructed corridor twin")
    print("=" * 70)
    # shared=0.55: adjacent bays share ~half their fine wall detail (common
    # corridor style), so a twin keeps MODERATE fine coherence — enough to
    # clear a session-calibrated single-scale veto, yet the coarse->fine
    # profile still collapses.
    segs, period, width = _bay_world(n_bays=8, shared=0.55)
    rng = np.random.default_rng(11)
    yc = 1.4
    heading = 0.0

    # -- GENUINE: query and map at the SAME bay (bay 3) -------------------
    bay = 3
    cx = bay * period + period / 2
    old_center = np.array([cx, yc])
    B = _bay_bundle(segs, old_center, heading, rng)
    qpose = np.array([cx + 0.25, yc + 0.1, np.deg2rad(3)])   # true revisit pose
    pts, w = _bay_scan(segs, qpose, rng)
    seed = np.array([cx - 0.3, yc, 0.0])                     # rough prior
    print("\n[genuine] query bay 3, map bay 3, seed near truth:")
    g = cascade(B, pts, w, seed, verbose=True)
    print(f"  profile_ratio={g['profile_ratio']:.3f}  slope={g['profile_slope']:+.4f}"
          f"  coh_coarse={g['coh_coarse']:.3f} coh_fine={g['coh_fine']:.3f}"
          f"  unwrap_ok={g['unwrap_ok']} final_ok={g['final_ok']}")

    # -- WEAK GENUINE: a genuine revisit with only PARTIAL overlap (the old
    #    pass swept ~3.4 m further down the corridor, so query and map share
    #    only the middle stretch). The non-overlapping ends are real, spectrally
    #    matched corridor, so they dilute EVERY ring proportionally (common
    #    mode): absolute fine coherence sinks to (below!) the twin level, yet
    #    the coarse->fine profile stays high. This is the regime a single-scale
    #    absolute-coherence veto cannot handle but the ratio can.
    Bw = _bay_bundle(segs, np.array([cx + 3.4, yc]), heading, rng, n=3, spread=0.3)
    qposew = np.array([cx + 0.25, yc + 0.1, np.deg2rad(3)])
    ptsw, ww = _bay_scan(segs, qposew, rng)
    print("\n[weak genuine] genuine partial-overlap revisit (common-mode dilution):")
    gw = cascade(Bw, ptsw, ww, seed, verbose=True)
    print(f"  profile_ratio={gw['profile_ratio']:.3f}  slope={gw['profile_slope']:+.4f}"
          f"  coh_coarse={gw['coh_coarse']:.3f} coh_fine={gw['coh_fine']:.3f}"
          f"  unwrap_ok={gw['unwrap_ok']} final_ok={gw['final_ok']}")

    # -- TWIN: query bay 3, map bay 4, seed pointed at bay 4 (wrong) ------
    bay2 = 4
    cx2 = bay2 * period + period / 2
    old_center2 = np.array([cx2, yc])
    B2 = _bay_bundle(segs, old_center2, heading, rng)
    # same physical query scan (bay 3), but seed placed a full period away so
    # the coarse ring locks onto bay 4's identical alcove
    seed2 = np.array([cx2 - 0.3, yc, 0.0])
    print("\n[twin] query bay 3, map bay 4, seed one period off (alcove alias):")
    tw = cascade(B2, pts, w, seed2, verbose=True)
    print(f"  profile_ratio={tw['profile_ratio']:.3f}  slope={tw['profile_slope']:+.4f}"
          f"  coh_coarse={tw['coh_coarse']:.3f} coh_fine={tw['coh_fine']:.3f}"
          f"  unwrap_ok={tw['unwrap_ok']} final_ok={tw['final_ok']}")

    # -- decisions --------------------------------------------------------
    prof_thr = 0.60
    # single-scale veto: to keep RECALL it must admit the weak (partial-overlap)
    # genuine, so its absolute fine-coherence threshold sits at the weakest
    # genuine's level — which then also admits the twin (twin fine coh actually
    # EXCEEDS the weak genuine's). The profile ratio needs no such low absolute
    # bar: it admits the weak genuine and rejects the twin.
    single_thr = gw["coh_fine"] - 1e-9

    def dec_prof(r):
        return r["unwrap_ok"] and r["final_ok"] and r["profile_ratio"] >= prof_thr

    def dec_single(r):
        return r["coh_fine"] >= single_thr

    print("\nDECISIONS (single-scale threshold calibrated to admit weak genuine):")
    for lab, r in [("strong genuine", g), ("weak genuine", gw), ("twin", tw)]:
        print(f"  {lab:<15} profile {'ACCEPT' if dec_prof(r) else 'REJECT'}"
              f"   single-scale {'ACCEPT' if dec_single(r) else 'REJECT'}"
              f"   (ratio {r['profile_ratio']:.2f}, fine coh {r['coh_fine']:.2f})")

    ok = dec_prof(g) and dec_prof(gw) and (not dec_prof(tw))
    # the point: with the single-scale threshold set low enough to keep the
    # weak genuine, the twin sneaks through; the profile keeps both genuines
    # and rejects the twin.
    point = (not dec_prof(tw)) and dec_single(tw) and dec_prof(gw)
    print(f"\n  -> both genuine closures pass the profile predicate: "
          f"{dec_prof(g) and dec_prof(gw)}")
    print(f"  -> twin REJECTED by profile while single-scale ACCEPTS it: {point}")
    assert dec_prof(g) and dec_prof(gw), "genuine closures must pass the profile"
    assert not dec_prof(tw), "twin must be rejected by the coherence-profile test"
    print("SELF-TEST OK" if ok else "SELF-TEST WEAK")
    return dict(genuine=g, weak=gw, twin=tw, point=point)


# ===========================================================================
# 2. GT-LABELLED ROC on the corridor bench
# ===========================================================================

def _make_candidates(segs, period, width, n_each=120, seed=0):
    """Generate labelled candidate closures on a self-similar corridor.

    Genuine: query re-observes the SAME place the map covers, with a mix of
             realistic degradations that spread the ABSOLUTE coherence widely:
             (a) partial overlap (old pass swept further down the corridor) —
             COMMON-MODE, keeps the profile flat; (b) range noise + dropout —
             FINE-SELECTIVE, drops the profile like a twin would. A robust
             discriminator must survive both.
    Twin   : query re-observes a bay `delta` periods from the map bay; the seed
             is placed on the map bay so the coarse alcove aliases; twins also
             get varied noise so their fine coherence spans a range."""
    rng = np.random.default_rng(seed)
    n_bays = int(segs[:, :, 0].max() // period)
    yc = 1.4
    gen, twn = [], []
    saved_noise = (S.RANGE_NOISE, S.DROPOUT)
    for _ in range(n_each):
        bay = rng.integers(2, n_bays - 2)
        cx = bay * period + period / 2
        heading = rng.uniform(-np.pi, np.pi) if rng.random() < 0.25 else 0.0
        S.RANGE_NOISE = rng.uniform(0.02, 0.08)
        S.DROPOUT = rng.uniform(0.02, 0.25)
        # --- genuine: same place, partial-overlap old pass ----------------
        overlap_off = rng.uniform(-3.6, 3.6)              # common-mode dilution
        B = _bay_bundle(segs, np.array([cx + overlap_off, yc]), heading, rng,
                        n=rng.integers(2, 5), spread=rng.uniform(0.2, 0.5))
        qp = np.array([cx + rng.normal(0, 0.35), yc + rng.normal(0, 0.12),
                       heading + rng.normal(0, np.deg2rad(3))])
        pts, w = _bay_scan(segs, qp, rng)
        if len(pts) >= 20:
            seedg = np.array([cx + rng.normal(0, 0.3), yc, heading])
            gen.append(cascade(B, pts, w, seedg))
        # --- twin: same query, map+seed delta periods away ----------------
        delta = int(rng.choice([-2, -1, 1, 2]))
        b2 = bay + delta
        if 2 <= b2 <= n_bays - 2 and len(pts) >= 20:
            cx2 = b2 * period + period / 2
            B2 = _bay_bundle(segs, np.array([cx2, yc]), heading, rng,
                             n=rng.integers(2, 5), spread=rng.uniform(0.2, 0.5))
            seedt = np.array([cx2 + rng.normal(0, 0.3), yc, heading])
            twn.append(cascade(B2, pts, w, seedt))
    S.RANGE_NOISE, S.DROPOUT = saved_noise
    return gen, twn


def roc_bench():
    print("=" * 70)
    print("ROC: multi-scale profile vs single-scale coherence (corridor bench)")
    print("=" * 70)
    segs, period, width = _bay_world(n_bays=10, period=6.0, width=3.0, shared=0.5)
    t0 = time.time()
    gen, twn = _make_candidates(segs, period, width, n_each=160, seed=1)
    print(f"generated {len(gen)} genuine + {len(twn)} twin candidates "
          f"in {time.time() - t0:.0f}s")

    def col(rs, key):
        return np.array([r[key] for r in rs])

    stats = {
        "multi-scale profile ratio (coh_fine/coh_coarse)":
            (col(gen, "profile_ratio"), col(twn, "profile_ratio")),
        "multi-scale profile slope (coh vs ring)":
            (col(gen, "profile_slope"), col(twn, "profile_slope")),
        "single-scale fine-ring coherence (shipped veto)":
            (col(gen, "single_fine"), col(twn, "single_fine")),
        "single-scale coarse-ring coherence":
            (col(gen, "coh_coarse"), col(twn, "coh_coarse")),
    }
    print(f"\n{'statistic':<52} {'AUC':>6} {'d':>7} "
          f"{'gen mean':>9} {'twin mean':>9}")
    print("-" * 90)
    results = {}
    for name, (pos, neg) in stats.items():
        a = auc(pos, neg)
        dd = cohens_d(pos, neg)
        results[name] = (a, dd)
        print(f"{name:<52} {a:6.3f} {dd:7.2f} "
              f"{pos.mean():9.3f} {neg.mean():9.3f}")

    # unwrap-consistency contribution
    g_uw = np.mean([r["unwrap_ok"] for r in gen])
    t_uw = np.mean([r["unwrap_ok"] for r in twn])
    print(f"\nunwrap-consistency pass rate: genuine {g_uw:.2f}  twin {t_uw:.2f}")

    auc_prof = results["multi-scale profile ratio (coh_fine/coh_coarse)"][0]
    auc_single = results["single-scale fine-ring coherence (shipped veto)"][0]
    print(f"\nSEPARATION: profile-ratio AUC {auc_prof:.3f}  vs  "
          f"single-scale AUC {auc_single:.3f}   "
          f"(delta {auc_prof - auc_single:+.3f})")

    _plot_roc(stats, "cascade_roc.png")
    return results, gen, twn


def _roc_curve(pos, neg):
    thr = np.unique(np.concatenate([pos, neg]))
    thr = np.concatenate([[thr.min() - 1], thr, [thr.max() + 1]])
    tpr = np.array([(pos >= t).mean() for t in thr])
    fpr = np.array([(neg >= t).mean() for t in thr])
    o = np.argsort(fpr)
    return fpr[o], tpr[o]


def _plot_roc(stats, out):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for name, (pos, neg) in stats.items():
        fpr, tpr = _roc_curve(pos, neg)
        ax.plot(fpr, tpr, lw=1.8, label=f"{name.split('(')[0].strip()} "
                                        f"(AUC {auc(pos, neg):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
    ax.set_xlabel("false-positive rate (twins accepted)")
    ax.set_ylabel("true-positive rate (genuine accepted)")
    ax.set_title("ROC: separating genuine from corridor twins")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    pr_g, pr_t = stats["multi-scale profile ratio (coh_fine/coh_coarse)"]
    sf_g, sf_t = stats["single-scale fine-ring coherence (shipped veto)"]
    ax2.scatter(sf_g, pr_g, s=14, c="tab:green", label="genuine", alpha=0.6)
    ax2.scatter(sf_t, pr_t, s=14, c="tab:red", label="twin", alpha=0.6)
    ax2.set_xlabel("single-scale fine-ring coherence")
    ax2.set_ylabel("multi-scale profile ratio")
    ax2.set_title("Why the profile separates: absolute coherence overlaps")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


def ab_bench(gen, twn):
    """A/B the profile predicate against the single-scale veto at matched TPR:
    pick each rule's threshold to accept ~90% of genuine, compare twin
    (false-edge) acceptance."""
    print("\n" + "=" * 70)
    print("A/B: false-edge count at matched genuine recall (>=0.90 TPR)")
    print("=" * 70)

    def at_recall(pos, neg, target=0.90):
        thr = np.quantile(pos, 1 - target)   # accept top `target` of genuine
        return (pos >= thr).mean(), (neg >= thr).mean(), thr

    for name, key in [("multi-scale profile ratio", "profile_ratio"),
                      ("single-scale fine coherence", "single_fine")]:
        pos = np.array([r[key] for r in gen])
        neg = np.array([r[key] for r in twn])
        tpr, fpr, thr = at_recall(pos, neg)
        print(f"  {name:<28} thr={thr:6.3f}  genuine {tpr:.2f}  "
              f"twin-accept {fpr:.2f}  ({int(fpr * len(neg))}/{len(neg)} false edges)")


# ===========================================================================
# 3. INTEL REGRESSION: profile test must not reject genuine drifted closures
# ===========================================================================

def intel_regression(path="data/intel.log", max_pairs=80):
    print("=" * 70)
    print("INTEL REGRESSION: genuine drifted closures must survive the profile")
    print("=" * 70)
    scans = C.parse_flaser(path)
    keys = C.keyframes(scans)
    n = len(keys)
    n_beams = len(keys[0][0])
    beam = np.deg2rad(np.arange(n_beams) - 90.0)
    print(f"{len(scans)} scans -> {n} keyframes, {n_beams} beams")

    # Use the RBPF-corrected reference as near-GT geometry to locate genuine
    # revisits (poses close in space, far in keyframe index).
    ref = C.parse_flaser(path.replace(".log", ".gfs.log"))
    rts = np.array([t for _, _, t in ref])
    rp = np.stack([p for _, p, _ in ref])
    kts = np.array([t for _, _, t in keys])
    j = np.abs(kts[:, None] - rts[None, :]).argmin(1)   # nearest ref per keyframe
    good = np.abs(kts - rts[j]) < 0.3
    gt = np.full((n, 3), np.nan)
    gt[good] = rp[j[good]]

    # sample scans once
    pts_all, w_all = [None] * n, [None] * n
    for k, (r, _, _) in enumerate(keys):
        rr = np.where(r < C.VALID_MAX, r, np.inf)
        p, ww, _ = S.scan_to_samples(rr, beam)
        pts_all[k], w_all[k] = p, ww

    # find genuine revisit pairs on GT geometry
    idx = np.flatnonzero(good)
    rng = np.random.default_rng(3)
    rng.shuffle(idx)
    pairs = []
    for k in idx:
        if len(pairs) >= max_pairs:
            break
        if pts_all[k] is None or len(pts_all[k]) < 25:
            continue
        # older keyframes revisited by k: GT-close, >400 kf apart
        cand = np.flatnonzero(good)
        cand = cand[(np.abs(cand - k) > 400)]
        if len(cand) == 0:
            continue
        dist = np.linalg.norm(gt[cand, :2] - gt[k, :2], axis=1)
        near = cand[dist < 1.5]
        near = [c for c in near if pts_all[c] is not None and len(pts_all[c]) >= 25]
        if len(near) < 5:
            continue
        pairs.append((k, near))

    print(f"found {len(pairs)} genuine revisit candidates (GT-close, >400 kf apart)")
    profs, slopes, singles, coarse, uw, fin = [], [], [], [], [], []
    for k, near in pairs:
        # old-map bundle from GT-corrected old poses (world-frame, locally rigid)
        B = np.zeros(L.W.shape[0], complex)
        for c in near[:30]:
            B += world_enc(pts_all[c], w_all[c], gt[c])
        seed = gt[k].copy()          # genuine closure seeds near the true revisit
        r = cascade(B, pts_all[k], w_all[k], seed)
        profs.append(r["profile_ratio"]); slopes.append(r["profile_slope"])
        singles.append(r["coh_fine"]); coarse.append(r["coh_coarse"])
        uw.append(r["unwrap_ok"]); fin.append(r["final_ok"])
    profs = np.array(profs); singles = np.array(singles); coarse = np.array(coarse)
    m = len(profs)
    q = np.quantile(profs, [0.1, 0.5, 0.9])
    print(f"\ngenuine Intel profile ratio: mean {profs.mean():.3f}  "
          f"median {q[1]:.3f}  (p10 {q[0]:.3f}, p90 {q[2]:.3f})")
    print(f"genuine Intel coarse coh {coarse.mean():.3f}  fine coh {singles.mean():.3f}"
          f"  -> fine<coarse on real revisits (fine-selective viewpoint/overlap loss)")
    print(f"unwrap-consistency pass rate: {np.mean(uw):.2f}   final-precision {np.mean(fin):.2f}")
    print("\ngenuine closures kept vs profile threshold (the transfer test):")
    for thr in (0.35, 0.45, 0.55):
        kept = int(((profs >= thr) & np.array(uw) & np.array(fin)).sum())
        print(f"  fixed profile thr {thr:.2f}: {kept}/{m} kept "
              f"({100*kept/m:.0f}%)")
    # session-relative operating point (like shipped coh_target * coh_ref):
    rel = 0.55 * np.median(profs)
    kept_rel = int(((profs >= rel) & np.array(uw) & np.array(fin)).sum())
    print(f"  session-relative thr 0.55*median={rel:.2f}: {kept_rel}/{m} kept "
          f"({100*kept_rel/m:.0f}%)")
    print("\nVERDICT: the corridor-tuned absolute profile threshold (0.55) "
          f"rejects {100*(profs<0.55).sum()/m:.0f}% of genuine Intel closures — "
          "genuine real-log revisits share the twins' coarse>fine collapse "
          "(180-deg FOV, viewpoint change, clutter are fine-SELECTIVE), so the "
          "RATIO LEVEL is itself domain-dependent. The profile separates within "
          "a domain (see ROC) but its threshold does NOT transfer as an absolute "
          "value; a drop-in predicate must be session-calibrated, exactly like "
          "the shipped veto it aims to replace.")
    return dict(n=m, profs=profs, singles=singles, coarse=coarse)


# ===========================================================================
def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("selftest", "all"):
        self_test()
        print()
    res = gen = twn = None
    if what in ("roc", "all"):
        res, gen, twn = roc_bench()
        ab_bench(gen, twn)
        print()
    if what in ("intel", "all"):
        try:
            intel_regression()
        except FileNotFoundError as e:
            print(f"intel regression skipped: {e}")


if __name__ == "__main__":
    main()
