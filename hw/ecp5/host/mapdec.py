#!/usr/bin/env python3
"""mapdec — ECP5 map-RECONSTRUCTION consumer: chip dump -> walls.

Input = exactly what the deploy chip produces, in either lane:
  * SOLO dump (solo_host.SoloChip.dump()): per-segment 2b QPSK codes +
    liveness plane + per-ring scales + anchor pose (5-kf segments);
  * STREAM map bank (webvis.Fpga.map_seg): per-scan 2b codes at the
    registration pose (no liveness/scales planes);
  * the GOLDEN OFFLINE FOLD (fold_offline): hw/ice40/host/solo.py
    SoloMapper — the bit-exact algebra of the chip's own fold/freeze —
    run over a dataset's scans at the replay pose estimates, so dataset
    replays get a chip-exact resident map without hardware attached.

Reconstruction = the banked 2026-07-12 read-side recipe (per-segment
CLEAN line pursuit + driven-path veto + cross-segment consensus +
Cr^2-weighted Wiener mosaic; hw/ice40/host/decode.py, IMPORTED never
edited) with one ECP5-side upgrade: the pursuit runs in each segment's
OWN ANCHOR FRAME with the world transform applied to the recovered
line parameters afterwards. The ice40 recipe rotated segment content
onto the world lattice by snapping the anchor heading to the 3-degree
permutation grid (up to 1.5 deg residual -> ~15 cm wall smear at 6 m);
the segment-frame form is EXACT for any heading and needs no vector
rotation at all. The probe grid + atom bank are shared across segments
(one precompute).

Scoring is GT-free: the reference is the venue's own scan scatter at
the SAME pose estimates the fold used (reconstruction fidelity of what
was encoded — not localization accuracy). decode.py's classroom
GT-wall self-score stays available there (SCORING ONLY).

  python3 hw/ecp5/host/mapdec.py selftest   # fold->pursuit line recovery
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hw" / "ice40" / "host"))

import hw.ice40.golden as G                                    # noqa: E402
import solo as SOLO                                            # noqa: E402
import decode as DEC                                           # noqa: E402

U, TAU_Q = SOLO.U, SOLO.TAU_Q
W_LAT = DEC.build_W()                  # (240, 2) matcher lattice
N_ANG = DEC.NANG
N_RING = len(DEC.LAMS)

# FIDELITY BAND (reconstruction-only second fold). The matcher octave
# ladder (0.25/0.5/1/2 — ring scale = BIT SHIFT in the RTL) shares a
# 2 m period across ALL rings, so a wall decodes as a ghost comb with
# 2 m teeth: parallel structure at multiples of 2 m is irreducibly
# aliased (see selftest). The fidelity ladder is INCOMMENSURATE
# (pairwise non-integer ratios -> no common period within the venue
# scale) and spans room scale. RTL cost: ring-specific angle LUTs
# (u_r = (x*C_r[a] + y*S_r[a]) >> F_ANG, LUT scale 0.25/lam_r) instead
# of the shared-u shift — 4x angle-ROM entries, same multiplier
# datapath; fold/freeze/store identical (60 B/seg, same permutation
# algebra per ring). Encoded here with golden integer semantics so the
# RTL port is a transcription, not a design.
FID_LAMS = (0.31, 0.83, 2.17, 5.71)


def make_luts_gen(lams=FID_LAMS, n_ang=N_ANG):
    """Golden-faithful LUTs for an arbitrary ring ladder: az stage
    copied from G.make_luts; angle stage per-ring with the 0.25/lam
    scale folded in (matcher band: scale 1 + bit shift)."""
    base = G.make_luts(n_ang)
    th = np.pi * np.arange(n_ang) / n_ang
    ang_c = np.stack([np.round(np.cos(th) * (0.25 / lam)
                               * (1 << G.F_ANG)).astype(np.int32)
                      for lam in lams])
    ang_s = np.stack([np.round(np.sin(th) * (0.25 / lam)
                               * (1 << G.F_ANG)).astype(np.int32)
                      for lam in lams])
    return dict(az_c=base["az_c"], az_s=base["az_s"],
                ang_c=ang_c, ang_s=ang_s,
                cis_re=base["cis_re"], cis_im=base["cis_im"],
                n_ang=n_ang, lams=tuple(lams))


def encode_gen_at(az, r_mm, w, cq, sq, tx_u, ty_u, gluts):
    """encode_int_at for a general ladder (fidelity band): stage
    arithmetic verbatim from the golden path; the ring loop uses its
    own projection instead of the shared-u shift. No derivative plane
    (reconstruction-only band)."""
    n_ang = gluts["n_ang"]
    half = 1 << (G.F_AZ - 1)
    x = (r_mm * gluts["az_c"][az] + half) >> G.F_AZ
    y = (r_mm * gluts["az_s"][az] + half) >> G.F_AZ
    xr = (cq * x - sq * y + (1 << 14)) >> 15
    yr = (sq * x + cq * y + (1 << 14)) >> 15
    x2 = xr + tx_u
    y2 = yr + ty_u
    assert len(x2) == 0 or max(np.abs(x2).max(), np.abs(y2).max()) \
        < (1 << 15), "transformed x,y exceed i16 (RTL envelope)"
    ha = 1 << (G.F_ANG - 1)
    n_ring = len(gluts["lams"])
    acc = np.zeros((n_ring * n_ang, 2), np.int64)
    for k in range(n_ring):
        u = (x2[:, None] * gluts["ang_c"][k][None, :]
             + y2[:, None] * gluts["ang_s"][k][None, :] + ha) >> G.F_ANG
        addr = u & 255
        acc[k * n_ang:(k + 1) * n_ang, 0] = \
            (w[:, None] * gluts["cis_re"][addr]).sum(0)
        acc[k * n_ang:(k + 1) * n_ang, 1] = \
            (w[:, None] * gluts["cis_im"][addr]).sum(0)
    assert np.abs(acc).max() < (1 << 31), "i32 acc overflow"
    return acc.astype(np.int32)


def liveness_gen(acc, n_ring, n_ang=N_ANG):
    """SOLO.liveness_int with a parametric ring count (the banked ops
    verbatim; solo.py hardcodes G.N_RING=4)."""
    acc = np.asarray(acc, np.int64)
    aI = np.abs(acc[:, 0]).reshape(n_ring, n_ang)
    aQ = np.abs(acc[:, 1]).reshape(n_ring, n_ang)
    M = np.maximum(aI, aQ) + (np.minimum(aI, aQ) >> 1)
    mmax = M.max(axis=1)
    alive = (2 * M >= mmax[:, None]).reshape(-1)
    scales = []
    for v in mmax:
        e, m = 0, int(v)
        while m >= 256:
            m >>= 1
            e += 1
        scales.append((e, m))
    return alive, scales


FID6_LAMS = (0.31, 0.61, 1.13, 2.17, 4.03, 7.87)   # 6-ring readout arm


def join_segs(a_segs, b_segs):
    """Concatenate two bands folded at the SAME anchors (dual-band
    joint readout: e.g. matcher + fidelity = 480 components/seg)."""
    out = []
    for sa, sb in zip(a_segs, b_segs):
        assert np.allclose(sa[0], sb[0]), "anchor mismatch between bands"
        out.append((sa[0], np.concatenate([sa[1], sb[1]]),
                    None if sa[2] is None or sb[2] is None
                    else np.concatenate([sa[2], sb[2]]),
                    None if sa[3] is None or sb[3] is None
                    else np.concatenate([sa[3], sb[3]])))
    return out


class FidMapper:
    """SoloMapper's fold/freeze for the fidelity band (same anchors,
    same SEG_KF cadence, same freeze scan — codes/liveness/scales via
    the accepted SOLO integer ops)."""

    def __init__(self, gluts, seg_kf=SOLO.SEG_KF):
        self.gluts = gluts
        self.seg_kf = seg_kf
        self.open_anchor = None
        self.kf_in_seg = 0
        self.acc = None
        self.frozen = []               # (anchor_q, codes, alive, scales)

    def fold(self, q_ints, pose_q):
        if self.open_anchor is None:
            self.open_anchor = pose_q
            n = len(self.gluts["lams"]) * self.gluts["n_ang"]
            self.acc = np.zeros((n, 2), np.int64)
        ax, ay, ah = self.open_anchor
        dh = SOLO.wrap_q(pose_q[2] - ah)
        cq, sq = SOLO.cs_of(dh)
        acs, asn = SOLO.cs_of(ah)
        tx, ty = SOLO.irot_q15(acs, asn, pose_q[0] - ax, pose_q[1] - ay)
        self.acc += encode_gen_at(*q_ints, cq, sq, tx, ty, self.gluts)
        self.kf_in_seg += 1
        if self.kf_in_seg >= self.seg_kf:
            self.freeze()

    def freeze(self):
        if self.open_anchor is None or self.kf_in_seg == 0:
            return
        codes = SOLO.mcode_int(self.acc[:, 0], self.acc[:, 1])
        alive, scales = liveness_gen(self.acc, len(self.gluts["lams"]),
                                     self.gluts["n_ang"])
        self.frozen.append((self.open_anchor, codes, alive, scales))
        self.open_anchor = None
        self.kf_in_seg = 0

# ---- tunable recipe (bench: scratch_walls.py; defaults = banked) ---------
PARAMS = dict(
    grid_r=4.0,       # local probe box half-size around the anchor [m]
    step=0.10,        # probe cell size [m]
    k_max=12,         # pursuit iterations per segment
    tau=0.25,         # stop when peak < tau * first peak (walls2:
                      # recall 0.87->0.95/0.54->0.63/0.59->0.69 at
                      # p50 <= 4 cm across the three venues)
    gamma=1.0,        # CLEAN loop gain (precision vs recall dial)
    lens=(0.6, 1.2, 2.4),   # atom half-length menu [m]
    n_th=24,          # atom orientation grid over pi
    veto_r=0.25,      # driven-path veto radius [m]
    cons_d=0.20,      # consensus: center distance [m]
    cons_th=np.deg2rad(15.0),   # consensus: orientation window
    use_planes=True,  # liveness mask + per-ring scales (when present)
)


# ------------------------------------------------------------ seg adapters
def seg_tuple(anchor_m, codes, alive=None, wr=None):
    """Normal form: (anchor(3) m/rad, codes uint8(240), alive bool|None,
    wr float(4)|None — per-ring weights, mean-normalized at dequant)."""
    return (np.asarray(anchor_m, float),
            np.asarray(codes, np.uint8),
            None if alive is None else np.asarray(alive, bool),
            None if wr is None else np.asarray(wr, float))


def segs_from_dump(dump):
    """solo_host.SoloChip.dump() -> segs. Chip scale words are
    {e[4:0], m[7:0]} -> w = m * 2^e (decode.load_stream convention)."""
    out = []
    for anchor, codes, alive, scales in dump:
        wr = np.array([(s & 0xFF) * float(1 << (s >> 8)) for s in scales],
                      float)
        out.append(seg_tuple(anchor, codes, alive, wr))
    return out


def segs_from_mapper(mapper):
    """solo.SoloMapper / FidMapper .frozen -> segs (int anchors ->
    m/rad; (e, m) ring scales -> m * 2^e)."""
    out = []
    for fz in mapper.frozen:
        anchor_q, codes = fz[0], fz[1]
        alive, scales = fz[-2], fz[-1]
        anchor = (anchor_q[0] * U, anchor_q[1] * U,
                  anchor_q[2] * 2 * np.pi / TAU_Q)
        wr = np.array([m * float(1 << e) for (e, m) in scales], float)
        out.append(seg_tuple(anchor, codes, alive, wr))
    return out


def segs_from_stream(chip_segs):
    """webvis Demo.chip_segs {slot: (pose, codes)} -> segs (per-scan
    'segments' at the registration pose; no liveness/scales planes)."""
    return [seg_tuple(pose, codes) for pose, codes in chip_segs.values()
            if pose is not None]


def fold_offline(scans_m, est, n_max=None, band="matcher"):
    """GOLDEN FOLD: run the chip's fold/freeze algebra over scans at
    the given pose estimates. band: "matcher" = bit-exact
    solo.SoloMapper (what the SOLO silicon builds today); "fidelity" =
    FidMapper on FID_LAMS (the reconstruction band — RTL rung specced
    above). scans_m: iterable of 1024 ranges [m] (inf = miss); est:
    (n, 3) world poses. n_max: stop after that many frozen segments
    (64 = the chip's resident-map envelope; None = laptop fidelity
    stream)."""
    mapper = SOLO.SoloMapper(G.make_luts()) if band == "matcher" \
        else FidMapper(make_luts_gen(
            FID6_LAMS if band == "fid6" else FID_LAMS))
    for r, e in zip(scans_m, est):
        q = G.scan_to_ints(np.asarray(r, float))
        if len(q[0]) < 5:
            continue
        pose_q = (int(round(e[0] / U)), int(round(e[1] / U)),
                  int(round(e[2] * TAU_Q / (2 * np.pi))) % TAU_Q)
        mapper.fold(q, pose_q)
        if n_max and len(mapper.frozen) >= n_max:
            break
    mapper.freeze()                       # flush a partial tail segment
    return segs_from_mapper(mapper)


def dequant(seg, use_planes=True):
    """codes (+ optional planes) -> complex content vector in the
    SEGMENT frame. wr is mean-normalized so pursuit amplitudes stay
    comparable across segments."""
    _, codes, alive, wr = seg
    v = np.exp(1j * (np.pi / 2) * codes.astype(np.float64))
    if use_planes and alive is not None:
        v = v * alive
    if use_planes and wr is not None and wr.mean() > 0:
        v = v * np.repeat(wr / wr.mean(), N_ANG)
    return v


# ------------------------------------------------- segment-frame pursuit
class AtomBank:
    """Shared probe grid + atom envelope bank (one precompute for ALL
    segments — the segment-frame form needs no per-segment rotation).
    E: probe-cell conjugate phasors; ENV: sinc envelopes per (th, len);
    offsets refine the peak cell like the banked recipe."""

    def __init__(self, p=PARAMS, W=None):
        self.W = W_LAT if W is None else np.asarray(W)
        g = np.arange(-p["grid_r"], p["grid_r"] + 1e-9, p["step"])
        gx, gy = np.meshgrid(g, g)
        self.cells = np.stack([gx.ravel(), gy.ravel()], 1)
        self.E = np.exp(-1j * (self.cells @ self.W.T))
        self.ths = np.pi * np.arange(p["n_th"]) / p["n_th"]
        self.lens = np.asarray(p["lens"], float)
        combos = [(th, ln) for th in self.ths for ln in self.lens]
        self.combo = np.asarray(combos)
        H = 0.5 * self.combo[:, 1:2] * np.stack(
            [np.cos(self.combo[:, 0]), np.sin(self.combo[:, 0])], 1)
        self.ENV = np.sinc((H @ self.W.T) / np.pi)     # (n_combo, D) real
        self.na2 = (self.ENV ** 2).sum(1)
        off = p["step"] * 1.5
        self.offs = np.array([(dx, dy) for dx in (-off, 0, off)
                              for dy in (-off, 0, off)])
        self.p = dict(p)

    def pursuit(self, v):
        """CLEAN on one segment-frame vector -> [(c(2), th, ln, amp)]
        in the SEGMENT frame."""
        p = self.p
        res = v.astype(complex).copy()
        out, first = [], None
        for _ in range(p["k_max"]):
            f = (self.E @ res).real
            pk = int(np.argmax(f))
            if first is None:
                first = f[pk]
            if f[pk] < p["tau"] * first or f[pk] <= 0:
                break
            best = None
            for off in self.offs:
                c = self.cells[pk] + off
                ph = np.exp(-1j * (self.W @ c))
                amps = (self.ENV @ (ph * res)).real / (self.na2 + 1e-12)
                en = amps * amps * self.na2
                en[amps <= 0] = -1.0
                j = int(np.argmax(en))
                if en[j] > 0 and (best is None or en[j] > best[0]):
                    best = (en[j], c, j, amps[j])
            if best is None:
                break
            _, c, j, amp = best
            th, ln = self.combo[j]
            a = np.exp(1j * (self.W @ c)) * self.ENV[j]
            res -= p["gamma"] * amp * a
            out.append((c, float(th), float(ln), float(p["gamma"] * amp)))
        return out


def seg_lines_world(seg, bank, use_planes=True):
    """Pursue one segment in its own frame; return world-frame lines
    [(c_w(2), th_w, ln, amp)] — EXACT heading transform (no 3-deg snap)."""
    ax, ay, ah = seg[0]
    v = dequant(seg, use_planes)
    R = np.array([[np.cos(ah), -np.sin(ah)],
                  [np.sin(ah), np.cos(ah)]])
    out = []
    for c, th, ln, amp in bank.pursuit(v):
        cw = R @ c + [ax, ay]
        out.append((cw, (th + ah) % np.pi, ln, amp))
    return out


def path_veto(lines, trail, r_veto=0.25):
    """A line crossing the DRIVEN PATH cannot be a wall (banked physics
    prior). trail: (n, 2) est positions (denser than anchors)."""
    if not len(lines):
        return lines
    T = np.asarray(trail, float)[:, :2]
    keep = []
    for (c, th, ln, amp, si) in lines:
        h = 0.5 * ln * np.array([np.cos(th), np.sin(th)])
        t = np.linspace(-1, 1, max(2, int(np.ceil(ln / 0.05))))[:, None]
        pts = c[None] + t * h
        if np.sqrt(((pts[:, None] - T[None]) ** 2).sum(-1)).min() > r_veto:
            keep.append((c, th, ln, amp, si))
    return keep


COMB = max(DEC.LAMS)               # ghost-comb period of the octave ladder


def comb_collapse(lines, comb=COMB, tol=0.20, th_tol=np.deg2rad(10.0),
                  keep_frac=0.60):
    """GHOST-COMB cleanup (map-alone physics). The octave ladder decodes
    a wall as a comb of parallel rows spaced by lambda_max; CLEAN then
    tiles one wall's along-length extent across SEVERAL rows (each
    subtraction removes that tile's energy from every row, so the next
    tile lands on an arbitrary row). Group near-parallel lines whose
    perpendicular offsets differ by ~k*comb; per group score each row
    (sum len*amp of its members); KEEP every row scoring >= keep_frac of
    the best (a real corridor pair keeps both walls); SNAP weaker rows'
    tiles onto their nearest strong row. Along-wall positions are never
    aliased — only the row is ambiguous."""
    if not lines:
        return lines
    n = len(lines)
    C = np.stack([l[0] for l in lines])
    TH = np.array([l[1] % np.pi for l in lines])
    par = np.zeros(n, int) - 1                 # union-find (tiny n)

    def find(i):
        while par[i] >= 0:
            i = par[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            dth = abs((TH[j] - TH[i] + np.pi / 2) % np.pi - np.pi / 2)
            if dth > th_tol:
                continue
            nrm = np.array([-np.sin(TH[i]), np.cos(TH[i])])
            doff = (C[j] - C[i]) @ nrm
            k = round(doff / comb)
            if abs(doff - k * comb) < tol:     # k=0 -> same row, still group
                ri, rj = find(i), find(j)
                if ri != rj:
                    par[rj] = ri
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for g in groups.values():
        if len(g) == 1:
            out.append(lines[g[0]])
            continue
        i0 = g[0]
        nrm = np.array([-np.sin(TH[i0]), np.cos(TH[i0])])
        offs = np.array([(C[j] - C[i0]) @ nrm for j in g])
        rows = {}                              # row id -> member idxs
        for jj, off in zip(g, offs):
            for r in rows:
                if abs(off - rows[r][0]) < tol:
                    rows[r][1].append(jj)
                    break
            else:
                rows[len(rows)] = [off, [jj]]
        score = {r: sum(lines[j][2] * lines[j][3] for j in m)
                 for r, (o, m) in rows.items()}
        smax = max(score.values())
        strong = {r: rows[r][0] for r in rows
                  if score[r] >= keep_frac * smax}
        for r, (o, mem) in rows.items():
            if r in strong:
                out.extend(lines[j] for j in mem)
                continue
            tgt = min(strong.values(), key=lambda so: abs(so - o))
            for j in mem:                      # snap tile onto strong row
                c, th, ln, amp, si = lines[j]
                out.append((c + (tgt - o) * nrm, th, ln, amp, si))
    return out


def consensus_collinear(lines, d_perp=0.20, th_max=np.deg2rad(15.0),
                        gap=0.75):
    """Cross-segment consensus, COLLINEARITY form. The ice40 recipe
    keyed on line-CENTER distance, which rejects two views of the same
    wall whose centers sit metres apart along it (different segments see
    different portions). Keep a line iff some OTHER segment produced a
    line that is (a) parallel within th_max, (b) whose center lies
    within d_perp of this line's infinite extension, and (c) whose span
    along the line overlaps or comes within `gap` of ours."""
    if not lines:
        return lines
    C = np.stack([l[0] for l in lines])
    TH = np.array([l[1] % np.pi for l in lines])
    LN = np.array([l[2] for l in lines])
    SEG = np.array([l[4] for l in lines])
    Ux = np.stack([np.cos(TH), np.sin(TH)], 1)
    keep = np.zeros(len(lines), bool)
    for i in range(len(lines)):
        d = C - C[i]
        dth = np.abs((TH - TH[i] + np.pi / 2) % np.pi - np.pi / 2)
        perp = np.abs(d @ [-Ux[i, 1], Ux[i, 0]])
        along = np.abs(d @ Ux[i])
        ok = ((dth < th_max) & (perp < d_perp) & (SEG != SEG[i])
              & (along <= 0.5 * LN + 0.5 * LN[i] + gap))
        if ok.any():
            keep[i] = True
    return [l for l, k in zip(lines, keep) if k]


def walls(segs, trail=None, params=None, bank=None, image=False,
          band="matcher"):
    """Full reconstruction: per-segment pursuit -> path veto ->
    (matcher band only) ghost-comb collapse -> cross-segment
    collinearity consensus. band selects the lattice: "matcher" =
    W_LAT (what the SOLO silicon dumps today, 2 m ghost comb),
    "fidelity" = FID_LAMS (incommensurate — comb-free by
    construction, so no collapse step). Returns
    dict(lines=[(c, th, ln, amp, si)], and with image=True the
    Cr^2-Wiener mosaic + per-ring coherence via the ice40 consumer)."""
    p = dict(PARAMS, **(params or {}))
    W = (W_LAT if band == "matcher"
         else DEC.build_W(list(FID6_LAMS)) if band == "fid6"
         else np.vstack([W_LAT, DEC.build_W(list(FID_LAMS))])
         if band == "joint" else DEC.build_W(list(FID_LAMS)))
    bank = bank or AtomBank(p, W)
    lines = []
    model = p.get("hybrid_model")
    for si, seg in enumerate(segs):
        if model is not None:
            ls = seg_lines_hybrid(seg, bank, model)
        else:
            ls = seg_lines_world(seg, bank, p["use_planes"])
        for c, th, ln, amp in ls:
            lines.append((c, th, ln, amp, si))
    anchors = [tuple(s[0]) for s in segs]
    tr = np.asarray(trail, float)[:, :2] if trail is not None \
        else np.array([a[:2] for a in anchors])
    lines = path_veto(lines, tr, p["veto_r"])
    if band == "matcher":
        lines = comb_collapse(lines)
    lines = consensus_collinear(lines, d_perp=p["cons_d"],
                                th_max=p["cons_th"])
    out = dict(lines=lines, n_seg=len(segs))
    if image:
        codes = [s[1].astype(np.int64) for s in segs]
        alive = [s[2] for s in segs] if all(
            s[2] is not None for s in segs) else None
        wr = [s[3] for s in segs] if all(
            s[3] is not None for s in segs) else None
        Wv = DEC.world_vectors(codes, anchors, W, alive=alive, wr=wr)
        Cr = DEC.coherence(Wv, anchors)
        apos = np.array([a[:2] for a in anchors])
        bounds = (apos[:, 0].min() - 4, apos[:, 0].max() + 4,
                  apos[:, 1].min() - 4, apos[:, 1].max() + 4)
        cells, img = DEC.wiener_image(Wv, anchors, W, Cr, bounds)
        out.update(Cr=Cr, img=img, bounds=bounds)
    return out


# ------------------------------------------------------------ GT-free score
def line_samples(lines, ds=0.05):
    ps = []
    for (c, th, ln, amp, _si) in lines:
        h = 0.5 * ln * np.array([np.cos(th), np.sin(th)])
        t = np.linspace(-1, 1, max(2, int(np.ceil(ln / ds))))[:, None]
        ps.append(c[None] + t * h)
    return np.concatenate(ps) if ps else np.zeros((0, 2))


def score_walls(lines, pts, r_rec=(0.15, 0.30), cell=0.10, min_hits=3):
    """Reconstruction fidelity vs the venue's own scan scatter at the
    fold poses (GT-FREE). precision: line-sample -> nearest scan point
    (p50/p90); recall: occupied cells (>= min_hits) within r of a line.
    Returns dict; NaNs when there are no lines."""
    from scipy.spatial import cKDTree
    pts = np.asarray(pts, float)
    ls = line_samples(lines)
    out = dict(n_lines=len(lines),
               len_m=float(sum(l[2] for l in lines)))
    if not len(ls) or len(pts) < 100:
        out.update(prec_p50=np.nan, prec_p90=np.nan,
                   **{f"rec{int(r * 100)}": np.nan for r in r_rec})
        return out
    d = cKDTree(pts).query(ls, k=1)[0]
    out["prec_p50"] = float(np.median(d))
    out["prec_p90"] = float(np.percentile(d, 90))
    lo = pts.min(0)
    ij = np.floor((pts - lo) / cell).astype(int)
    _, idx, cnt = np.unique(ij, axis=0, return_index=True,
                            return_counts=True)
    occ = (np.floor((pts[idx] - lo) / cell) + 0.5) * cell + lo
    occ = occ[cnt >= min_hits]
    dr = cKDTree(ls).query(occ, k=1)[0]
    for r in r_rec:
        out[f"rec{int(r * 100)}"] = float((dr <= r).mean())
    out["n_occ"] = int(len(occ))
    return out


def fmt_score(s):
    return (f"{s['n_lines']:4d} lines {s['len_m']:6.1f} m | prec p50 "
            f"{s['prec_p50']:.3f} p90 {s['prec_p90']:.3f} | recall@15 "
            f"{s.get('rec15', float('nan')):.2f} @30 "
            f"{s.get('rec30', float('nan')):.2f}")


# ----------------------------------------------------------------- selftest
def selftest():
    """Dual-band gate on the WORST-CASE room: a bare rectangle whose
    wall separations are exact multiples of the matcher ladder's 2 m
    comb (y walls 4 m apart, x walls 6 m apart) + two pillars. The
    MATCHER band cannot place such walls absolutely (true row and
    +-2k m ghost rows are mathematically identical — physics limit,
    asserted only mod-comb); the FIDELITY band (incommensurate ladder)
    must recover them ABSOLUTELY. Both go through the same golden
    integer fold -> 2b freeze -> pursuit -> veto/consensus chain."""
    rng = np.random.default_rng(3)
    walls_gt = [((-3.0, -2.0), (3.0, -2.0)), ((3.0, -2.0), (3.0, 2.0)),
                ((3.0, 2.0), (-3.0, 2.0)), ((-3.0, 2.0), (-3.0, -2.0)),
                ((-1.8, 1.0), (-1.4, 1.3)), ((1.2, -1.0), (1.5, -0.7))]

    def cast(pose, n_beam=1024):
        a = pose[2] + -np.pi + np.arange(n_beam) * (2 * np.pi / n_beam)
        r = np.full(n_beam, np.inf)
        ca, sa = np.cos(a), np.sin(a)
        for (x0, y0), (x1, y1) in walls_gt:
            dx, dy = x1 - x0, y1 - y0
            den = dx * sa - dy * ca
            ok = np.abs(den) > 1e-9
            t = np.where(ok, ((pose[0] - x0) * sa
                              - (pose[1] - y0) * ca)
                         / np.where(ok, den, 1.0), -1.0)
            rr = np.where((t >= 0) & (t <= 1),
                          (x0 + t * dx - pose[0]) * ca
                          + (y0 + t * dy - pose[1]) * sa, np.inf)
            r = np.minimum(r, np.where(rr > 0.05, rr, np.inf))
        return r

    poses = np.stack([np.array([rng.uniform(-1.5, 1.2),
                                rng.uniform(-0.8, 1.0),
                                rng.uniform(0, 2 * np.pi)])
                      for _ in range(10)])
    scans = [cast(p) + rng.normal(0, 0.01, 1024) for p in poses]
    gt_pts = []
    for (x0, y0), (x1, y1) in walls_gt:
        t = np.linspace(0, 1, 200)[:, None]
        gt_pts.append(np.stack([x0 + t[:, 0] * (x1 - x0),
                                y0 + t[:, 0] * (y1 - y0)], 1))
    gt_pts = np.concatenate(gt_pts)

    # ---- fidelity band: ABSOLUTE recovery required
    fseg = fold_offline(scans, poses, band="fidelity")
    assert len(fseg) == 2, f"want 2 segments, got {len(fseg)}"
    rf = walls(fseg, trail=poses, band="fidelity")
    sf = score_walls(rf["lines"], gt_pts)
    print(f"[mapdec selftest] fidelity band: {fmt_score(sf)}")
    # ---- matcher band: mod-comb correctness only (comb physics)
    mseg = fold_offline(scans, poses, band="matcher")
    rm = walls(mseg, trail=poses, band="matcher")
    devs = []
    for (c, th, ln, amp, si) in rm["lines"]:
        nrm = np.array([-np.sin(th), np.cos(th)])
        d = (gt_pts - c) @ nrm
        j = np.abs((gt_pts - c) @ [np.cos(th), np.sin(th)]) < 0.5 * ln
        if j.any():
            dd = np.abs(d[j])
            devs.append(np.abs(dd - np.round(dd / COMB) * COMB).min())
    print(f"[mapdec selftest] matcher band: {len(rm['lines'])} lines, "
          f"mod-comb dev med "
          f"{np.median(devs) if devs else float('nan'):.3f} m")
    assert sf["n_lines"] >= 4, "fidelity: too few lines"
    assert sf["prec_p50"] < 0.15, f"fidelity p50 {sf['prec_p50']:.3f}"
    assert sf["rec30"] > 0.5, f"fidelity recall@30 {sf['rec30']:.2f}"
    assert devs and np.median(devs) < 0.20, "matcher mod-comb"
    # stream-form (no planes) must also run
    r2 = walls([seg_tuple(sg[0], sg[1]) for sg in fseg], trail=poses,
               band="fidelity")
    print(f"[mapdec selftest] planes-off arm ok ({len(r2['lines'])} "
          "lines) — PASS")


if __name__ == "__main__":
    selftest()


# ------------------------------------------------- learned-recall hybrid
# wall_unet.npz (GPU-agent P1, 2026-07-17): (96,96,1) correlation FIELD
# of a fidelity segment -> (96,96) occupancy logits. Numpy forward.
# HYBRID (round-9 contract): logits are the pursuit's PROPOSAL DENSITY;
# CLEAN atom fitting keeps sub-cell line precision.
UNET = ROOT / "sspax" / "artifacts" / "wall_unet.npz"


def unet_load(path=UNET):
    z = np.load(path, allow_pickle=True)
    Ls = [(z[f"{n}_kernel"].astype(np.float32),
           z[f"{n}_bias"].astype(np.float32)) for n in z["layers"]]
    return dict(layers=Ls, mu=float(z["mu"]), sd=float(z["sd"]),
                cell=float(z["cell"]))


def _c2d(x, k, b, s=1):
    from numpy.lib.stride_tricks import sliding_window_view
    kh = k.shape[0]
    p = kh // 2
    xp = np.pad(x, ((p, p), (p, p), (0, 0))) if p else x
    win = sliding_window_view(xp, (kh, kh), axis=(0, 1))[::s, ::s]
    return np.einsum("hwcab,abco->hwo", win, k) + b


def unet_forward(model, field, pool="max", up="nearest"):
    """6-conv U-Net-small: enc 16/32/64 with 2x pools, dec with 2x
    upsample + skip concat, 1x1 head. pool/up are the two semantics
    the export note left open — pinned by reproducing the banked
    school numbers (see RESULTS)."""
    Ls = model["layers"]
    x = ((np.asarray(field, np.float32) - model["mu"])
         / model["sd"])[:, :, None]
    a0 = np.maximum(_c2d(x, *Ls[0]), 0)              # 96x96x16
    d0 = (a0.reshape(48, 2, 48, 2, -1).max((1, 3)) if pool == "max"
          else a0.reshape(48, 2, 48, 2, -1).mean((1, 3)))
    a1 = np.maximum(_c2d(d0, *Ls[1]), 0)             # 48x48x32
    d1 = (a1.reshape(24, 2, 24, 2, -1).max((1, 3)) if pool == "max"
          else a1.reshape(24, 2, 24, 2, -1).mean((1, 3)))
    a2 = np.maximum(_c2d(d1, *Ls[2]), 0)             # 24x24x64
    u1 = np.repeat(np.repeat(a2, 2, 0), 2, 1)        # 48x48x64
    a3 = np.maximum(_c2d(np.concatenate([u1, a1], -1), *Ls[3]), 0)
    u0 = np.repeat(np.repeat(a3, 2, 0), 2, 1)        # 96x96x32
    a4 = np.maximum(_c2d(np.concatenate([u0, a0], -1), *Ls[4]), 0)
    return _c2d(a4, *Ls[5])[:, :, 0]                 # logits 96x96


FIELD_SCALE = 0.5      # wall_unet input calibration: the net trained
                       # on mantissa-weighted fields (artifact sd 1687
                       # ~= school m-field sd 3400 x 0.5); recall is
                       # scale-robust (0.75-0.80 across 0.35-1.0), the
                       # hybrid consumes RANKING not the threshold


def _mantissa(wr):
    """Recover the freeze-scan mantissa m (<256) from wr = m * 2^e."""
    out = []
    for w in np.asarray(wr):
        m = int(round(float(w)))
        while m >= 256:
            m >>= 1
        out.append(m)
    return np.array(out, float)


def seg_field(seg, W=None, n=96, step=0.10):
    """The wall_unet input: correlation field of one segment in its
    own frame (n x n at `step`), MANTISSA-weighted phasors x alive
    (the training distribution — fingerprinted vs the banked school
    numbers, RESULTS 2026-07-17 hybrid entry)."""
    W = DEC.build_W(list(FID_LAMS)) if W is None else W
    g = (np.arange(n) - (n - 1) / 2) * step
    gx, gy = np.meshgrid(g, g)
    cells = np.stack([gx.ravel(), gy.ravel()], 1)
    _, codes, alive, wr = seg
    v = np.exp(1j * (np.pi / 2) * codes.astype(np.float64))
    if alive is not None:
        v = v * alive
    if wr is not None:
        v = v * np.repeat(_mantissa(wr), N_ANG)
    v = v * FIELD_SCALE
    return (np.exp(-1j * (cells @ W.T)) @ v).real.reshape(n, n), cells


def seg_lines_hybrid(seg, bank, model, thr=0.0, use_planes=True,
                     pool="max", up="nearest"):  # noqa: D401
    """HYBRID pursuit: U-Net logits (recall) propose peak cells; each
    proposal seeds ONE CLEAN atom fit (+-0.2 m window) against the
    residual (precision); subtraction stays CLEAN's. World-frame lines
    out, same tuple form as seg_lines_world."""
    ax, ay, ah = seg[0]
    field, cells = seg_field(seg, bank.W)
    logits = unet_forward(model, field, pool=pool, up=up)
    v = dequant(seg, use_planes).astype(complex)
    R = np.array([[np.cos(ah), -np.sin(ah)], [np.sin(ah), np.cos(ah)]])
    n = logits.shape[0]
    order = np.argsort(-logits.ravel())
    used = np.zeros(n * n, bool)
    out = []
    res = v.copy()
    p = bank.p
    # CLEAN's amplitude discipline, calibrated CLEAN's way: fit the
    # segment's strongest FIELD peak first and hold every proposal to
    # tau x that bar (v2's running-max let early junk in: school p50
    # 0.147; v1 no floor: 0.218 vs CLEAN 0.040).
    f0 = (bank.E @ res).real
    pk = int(np.argmax(f0))
    amp_ref = 0.0
    for off in bank.offs:
        c = bank.cells[pk] + off
        ph = np.exp(-1j * (bank.W @ c))
        amps = (bank.ENV @ (ph * res)).real / (bank.na2 + 1e-12)
        j = int(np.argmax(amps))
        if amps[j] > amp_ref:
            amp_ref = float(amps[j])
    for idx in order[:400]:
        if logits.ravel()[idx] <= thr or len(out) >= p["k_max"]:
            break
        if used[idx]:
            continue
        c0 = cells[idx]
        best = None
        for off in bank.offs:
            c = c0 + off
            ph = np.exp(-1j * (bank.W @ c))
            amps = (bank.ENV @ (ph * res)).real / (bank.na2 + 1e-12)
            en = amps * amps * bank.na2
            en[amps <= 0] = -1.0
            j = int(np.argmax(en))
            if en[j] > 0 and (best is None or en[j] > best[0]):
                best = (en[j], c, j, amps[j])
        if best is None:
            continue
        _, c, j, amp = best
        if amp < p["tau"] * amp_ref:
            continue
        th, ln = bank.combo[j]
        a = np.exp(1j * (bank.W @ c)) * bank.ENV[j]
        res -= p["gamma"] * amp * a
        # suppress proposals along the fitted line (they are explained)
        h = 0.5 * ln * np.array([np.cos(th), np.sin(th)])
        d = cells - c
        along = d @ [np.cos(th), np.sin(th)]
        perp = d @ [-np.sin(th), np.cos(th)]
        used |= (np.abs(along) < 0.5 * ln + 0.15) & (np.abs(perp) < 0.25)
        cw = R @ c + [ax, ay]
        out.append((cw, float((th + ah) % np.pi), float(ln),
                    float(p["gamma"] * amp)))
    return out
