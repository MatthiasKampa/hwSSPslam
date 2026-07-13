"""iCE40 golden model — the bit-exact integer reference for the RTL datapath.

Implements the design-note v1 integer spec (SotA/fpga_design.md, iCE40 track):
  - position unit U = lambda_min/256 = 0.9765625 mm; the mm->U factor 1.024
    is FOLDED into the az-LUT constants (no extra multiply anywhere),
  - hits as (az_bin, r_mm): x,y from a 1024-entry az LUT (beams exactly on
    the grid), i32 with |p| <= 32 m,
  - projections u_j = (x*C[j] + y*S[j]) >> G against N_ANG (cos,sin) ROM
    pairs (the A1 form: ONE projection per angle),
  - ring cis addresses as BIT-SLICES of u: addr_k = (u >> k) & 255 (exact
    because the octave ladder is x2 — A1),
  - cis ROM 256 x 2 x int8 (127-scaled), int32 accumulators.

Every operation is integer with defined rounding, so the RTL must match this
model BIT-EXACTLY (golden vectors: hw/ice40/host/vectors.py). Float encode is
used only to BOUND the quantization error (cosine fidelity), never as the
spec. No shipped module is edited; lattice geometry mirrors ssp_slam_loop's
MAIN band (4 octave rings x N_ANG half-circle angles, ring-major).

Usage:
  python3 -m hw.ice40.golden selftest       # LUT sanity + fidelity vs float encode
  python3 -m hw.ice40.golden fidelity       # per-scan cosine on spot + synth data
"""
import sys

import numpy as np

import sspslam.lattice as L

N_BEAM = 1024
N_ANG = 60                    # deployment knob (oct36/45 study pending)
N_RING = 4
F_AZ = 14                     # az LUT fractional bits (value scale 2^14*1.024)
F_ANG = 14                    # angle ROM fractional bits
U_PER_MM = 256.0 / 250.0      # position-unit conversion folded into az LUT
R_MASK_MM = 31200             # |p| beyond this masked: x,y in U units must
                              # fit i16 (31200 mm * 1.024 = 31949 < 32767)


def make_luts(n_ang=N_ANG):
    az = -np.pi + np.arange(N_BEAM) * (2 * np.pi / N_BEAM)
    az_c = np.round(np.cos(az) * U_PER_MM * (1 << F_AZ)).astype(np.int32)
    az_s = np.round(np.sin(az) * U_PER_MM * (1 << F_AZ)).astype(np.int32)
    th = np.arange(n_ang) * np.pi / n_ang
    ang_c = np.round(np.cos(th) * (1 << F_ANG)).astype(np.int32)
    ang_s = np.round(np.sin(th) * (1 << F_ANG)).astype(np.int32)
    a = np.arange(256) * (2 * np.pi / 256)
    cis_re = np.round(127 * np.cos(a)).astype(np.int32)
    cis_im = np.round(127 * np.sin(a)).astype(np.int32)
    return dict(az_c=az_c, az_s=az_s, ang_c=ang_c, ang_s=ang_s,
                cis_re=cis_re, cis_im=cis_im, n_ang=n_ang)


def scan_to_ints(r, beam=None, w_unit=True):
    """Ranges (N_BEAM, inf=miss) -> (az_bin i32, r_mm i32, w u8 arrays).
    v1 weights: unit (127) or arc-mass r*dtheta quantized to u8."""
    ok = np.isfinite(r)
    r_mm = np.round(r[ok] * 1000).astype(np.int64)
    keep = r_mm <= R_MASK_MM
    az = np.flatnonzero(ok)[keep].astype(np.int32)
    r_mm = r_mm[keep].astype(np.int32)
    if w_unit:
        w = np.full(len(az), 127, np.int32)
    else:                       # w = r*dtheta arc mass, scaled to <=127
        w = np.minimum((r_mm * 6434) >> 21, 127).astype(np.int32)  # ~r*dθ mm
        w = np.maximum(w, 1)
    return az, r_mm, w


def encode_int(az, r_mm, w, luts):
    """THE RTL-EXACT ENCODE. -> (n_ring*n_ang, 2) int32 accumulators.
    Integer ops only; >> is arithmetic (floor) shift, matching RTL."""
    n_ang = luts["n_ang"]
    # x,y in U units: (r_mm * lut + half) >> F_AZ, round-half-up like RTL
    half = 1 << (F_AZ - 1)
    x = (r_mm * luts["az_c"][az] + half) >> F_AZ
    y = (r_mm * luts["az_s"][az] + half) >> F_AZ
    assert len(x) == 0 or max(np.abs(x).max(), np.abs(y).max()) < (1 << 15), \
        "x,y exceed i16 (RTL envelope)"
    # projections u_j: (x*C + y*S + half) >> F_ANG   -> (P, n_ang) i64->i32
    ha = 1 << (F_ANG - 1)
    u = (x[:, None] * luts["ang_c"][None, :]
         + y[:, None] * luts["ang_s"][None, :] + ha) >> F_ANG
    assert u.size == 0 or np.abs(u).max() < (1 << 17), \
        "u exceeds i18 (RTL envelope)"
    acc = np.zeros((N_RING * n_ang, 2), np.int64)
    for k in range(N_RING):
        addr = (u >> k) & 255                          # A1 bit-slice
        re = w[:, None] * luts["cis_re"][addr]
        im = w[:, None] * luts["cis_im"][addr]
        acc[k * n_ang:(k + 1) * n_ang, 0] = re.sum(0)
        acc[k * n_ang:(k + 1) * n_ang, 1] = im.sum(0)
    assert np.abs(acc).max() < (1 << 31), "i32 accumulator overflow"
    return acc.astype(np.int32)


def encode_float_ref(az, r_mm, w, n_ang=N_ANG):
    """Float reference on the SAME masked points/weights, exact lattice —
    bounds the int path's quantization error (never the spec)."""
    beam = -np.pi + az * (2 * np.pi / N_BEAM)
    r = r_mm / 1000.0
    pts = np.stack([r * np.cos(beam), r * np.sin(beam)], 1)
    th = np.arange(n_ang) * np.pi / n_ang
    lams = 0.25 * 2.0 ** np.arange(N_RING)
    W = np.concatenate([(2 * np.pi / lam)
                        * np.stack([np.cos(th), np.sin(th)], 1)
                        for lam in lams])
    return np.exp(1j * (pts @ W.T)).T @ w.astype(float)


def fidelity(vec_int, vec_ref):
    v = vec_int[:, 0] + 1j * vec_int[:, 1]
    n = np.linalg.norm
    return float(np.abs(np.vdot(v, vec_ref)) / (n(v) * n(vec_ref) + 1e-12))


# ------------------------------------------------------- matcher (corner F)
def rot_grid_int(Q, rho, n_ang=N_ANG):
    """Grid rotation by rho steps (theta = rho*pi/n_ang) of an integer
    (N_RING*n_ang, 2) vector: index shift with CONJUGATE WRAP (polar
    half-circle lattice — the antipodal direction is the conjugate
    component). Geometry pinned against the float encoder in selftest."""
    Q = Q.reshape(N_RING, n_ang, 2)
    j = np.arange(n_ang)
    src = j - (rho % n_ang)
    wrap = src < 0
    src = np.where(wrap, src + n_ang, src)
    out = np.empty_like(Q)
    out[:, :, 0] = Q[:, src, 0]
    out[:, :, 1] = np.where(wrap[None, :], -Q[:, src, 1], Q[:, src, 1])
    return out.reshape(-1, 2)


def mcode_from_vec(v):
    """QPSK codes (0..3 ~ 1,i,-1,-i) from a complex map vector — the
    nph=4 phase store. Input convention only (fabric takes codes as-is)."""
    return (np.round(np.angle(v) / (np.pi / 2)).astype(np.int64) % 4)


def shift_for(Q):
    """Per-scan pre-shift: smallest s with |Q >> s| < 2^15 (i16 DSP
    operands). Real 1024-beam scans reach |Q| ~ 2^23 (measured spot max
    2^23.0, school 2^23.3) -> s ~ 9 typical; quiet scans keep more bits."""
    m = int(np.abs(Q).max())
    return max(0, m.bit_length() - 15)


def match_int(Q, mcode, dx, dy, rho, s, luts):
    """THE RTL-EXACT MATCHER (one candidate pose). Integer contract:
      Q      (N_RING*n_ang, 2) i32 encode accumulators,
      mcode  (N_RING*n_ang,)   uint2 QPSK map codes (2b store),
      dx,dy  i16 candidate translation in position units (lambda_min/256),
      rho    grid rotation steps (theta = rho*pi/n_ang),
      s      per-scan pre-shift (host: shift_for(Q)); |Q >> s| must be i16.
    Returns (N_RING, 2) i32 per-ring partial sums
      s[k] = sum_j conj(rot(Q >> s, rho))[k,j] * i^mcode[k,j]
                   * cis(((dx*C[j] + dy*S[j] + ha) >> F_ANG) >> k & 255).
    Per-ring scales / normalization / sub-grid rotation are HOST-side
    (scales factor out of each ring's partial; derivative refinement is
    storage-side). Scores are linear in Q, so the shift only costs
    truncation precision — bounded by the fidelity probe, never the spec.
    Convention (pinned in selftest): cis(u(D)) translates MAP content by
    +D, so Re[s] peaks at the D that ALIGNS the map to the query — map
    content displaced +d relative to the query peaks at D = -d."""
    n_ang = luts["n_ang"]
    QS = (Q.astype(np.int64) >> int(s))
    assert np.abs(QS).max() < (1 << 15), "Q >> s exceeds i16 (bad shift)"
    QR = rot_grid_int(QS, rho, n_ang)
    hre, him = QR[:, 0], -QR[:, 1]                  # conj
    c = np.asarray(mcode, np.int64) & 3             # * i^c (rot90 — no mult)
    HRE = np.select([c == 0, c == 1, c == 2], [hre, -him, -hre], him)
    HIM = np.select([c == 0, c == 1, c == 2], [him, hre, -him], -hre)
    ha = 1 << (F_ANG - 1)
    u = (int(dx) * luts["ang_c"].astype(np.int64)
         + int(dy) * luts["ang_s"].astype(np.int64) + ha) >> F_ANG
    s = np.zeros((N_RING, 2), np.int64)
    for k in range(N_RING):
        addr = (u >> k) & 255
        cre = luts["cis_re"][addr].astype(np.int64)
        cim = luts["cis_im"][addr].astype(np.int64)
        h_re = HRE[k * n_ang:(k + 1) * n_ang]
        h_im = HIM[k * n_ang:(k + 1) * n_ang]
        s[k, 0] = np.sum(h_re * cre - h_im * cim)
        s[k, 1] = np.sum(h_re * cim + h_im * cre)
    assert np.abs(s).max() < (1 << 31), "matcher i32 overflow"
    return s.astype(np.int32)


def selftest_match():
    luts = make_luts()
    rng = np.random.default_rng(7)
    # --- geometry pin 1: grid rotation == permutation-with-conjugate-wrap
    pts = rng.uniform(-4, 4, (150, 2))
    w = rng.integers(1, 128, 150)
    az0 = np.arctan2(pts[:, 1], pts[:, 0])
    r0 = np.hypot(pts[:, 0], pts[:, 1])
    for m in (1, 7, 33, 59):
        th = m * np.pi / N_ANG
        pr = np.stack([r0 * np.cos(az0 + th), r0 * np.sin(az0 + th)], 1)
        vr = np.exp(1j * (pr @ _W_float().T)).T @ w.astype(float)
        v0 = np.exp(1j * (pts @ _W_float().T)).T @ w.astype(float)
        vp = _rot_grid_c(v0, m)
        assert np.allclose(vr, vp, rtol=1e-9, atol=1e-6), m
    print("rotation: float rotate == permute+conj-wrap (m=1,7,33,59)")
    # --- geometry pin 2: score peak localizes a known displacement + rot
    r = 3.5 / np.maximum(np.abs(np.cos(np.arange(N_BEAM) * 2 * np.pi
                                       / N_BEAM - np.pi)), 0.3)
    r += rng.normal(0, 0.02, N_BEAM)
    az, r_mm, ww = scan_to_ints(r)
    Q = encode_int(az, r_mm, ww, luts)
    vfl = encode_float_ref(az, r_mm, ww)
    mc = mcode_from_vec(vfl)
    sh = shift_for(Q)
    grid = np.arange(-96, 97, 32)                   # +-9.4 cm in U units
    best, sc0 = None, None
    for dx in grid:
        for dy in grid:
            s = match_int(Q, mc, dx, dy, 0, sh, luts)
            tot = float(s[:, 0].sum())              # Re, unit ring weights
            if best is None or tot > best[0]:
                best = (tot, dx, dy)
            if dx == 0 and dy == 0:
                sc0 = tot
    assert best[1] == 0 and best[2] == 0, f"peak off-origin {best}"
    s_rot = [match_int(Q, mc, 0, 0, m, sh, luts)[:, 0].sum()
             for m in range(N_ANG)]
    assert int(np.argmax(s_rot)) == 0, "rotation peak off-zero"
    print(f"match peak at (0,0,rho=0) on synth room "
          f"(score {sc0:.3g}; rot margin "
          f"{s_rot[0] / max(s_rot[1], s_rot[-1]):.2f}x)")
    # --- shifted map -> peak at the shift (sign convention pinned)
    dxu = 64                                        # 62.5 mm in U units
    pts_s = np.stack([r * np.cos(np.arange(N_BEAM) * 2 * np.pi / N_BEAM
                                 - np.pi) + dxu * 250 / 256 / 1000,
                      r * np.sin(np.arange(N_BEAM) * 2 * np.pi / N_BEAM
                                 - np.pi)], 1)
    vs = np.exp(1j * (pts_s @ _W_float().T)).T @ ww.astype(float)
    mcs = mcode_from_vec(vs)
    best = max(((match_int(Q, mcs, dx, 0, 0, sh, luts)[:, 0].sum(), dx)
                for dx in grid), key=lambda t: t[0])
    assert best[1] == -dxu, f"align peak at {best[1]} != {-dxu}"
    print(f"map displaced +{dxu} U aligns at D={best[1]} "
          f"(cis(+u) moves map content by +D — convention pinned)")
    # --- determinism
    s1 = match_int(Q, mc, 33, -47, 13, sh, luts)
    s2 = match_int(Q, mc, 33, -47, 13, sh, luts)
    assert np.array_equal(s1, s2)
    print("determinism: bit-equal on repeat")
    print("selftest_match ok")


def _W_float(n_ang=N_ANG):
    th = np.arange(n_ang) * np.pi / n_ang
    lams = 0.25 * 2.0 ** np.arange(N_RING)
    return np.concatenate([(2 * np.pi / lam)
                           * np.stack([np.cos(th), np.sin(th)], 1)
                           for lam in lams])


def _rot_grid_c(v, m, n_ang=N_ANG):
    v = v.reshape(N_RING, n_ang)
    j = np.arange(n_ang)
    src = j - m
    wrap = src < 0
    src = np.where(wrap, src + n_ang, src)
    return np.where(wrap[None, :], np.conj(v[:, src]), v[:, src]).reshape(-1)


# ------------------------------------------------------------------ checks
def selftest():
    luts = make_luts()
    # LUT exactness vs float (<= 1 LSB by construction of round())
    az = -np.pi + np.arange(N_BEAM) * (2 * np.pi / N_BEAM)
    e = np.abs(luts["az_c"] - np.cos(az) * U_PER_MM * (1 << F_AZ)).max()
    print(f"az LUT max rounding err {e:.3f} LSB (<=0.5 by construction)")
    # ring bit-slice == direct per-ring address computation
    rng = np.random.default_rng(3)
    u = rng.integers(-(1 << 22), 1 << 22, 4096)
    for k in range(N_RING):
        direct = np.floor_divide(u, 1 << k) % 256
        assert np.array_equal((u >> k) & 255, direct)
    print("ring addresses: bit-slice == floor-div mod 256 (exact, 4 rings)")
    # synthetic scan fidelity int vs float
    r = np.full(N_BEAM, np.inf)
    a = np.arange(N_BEAM) * (2 * np.pi / N_BEAM) - np.pi
    for wall in range(4):                       # a 7x7 square room at center
        pass
    r = np.minimum(3.5 / np.maximum(np.abs(np.cos(a)), 1e-9),
                   3.5 / np.maximum(np.abs(np.sin(a)), 1e-9))
    r += rng.normal(0, 0.02, N_BEAM)
    az, r_mm, w = scan_to_ints(r)
    vi = encode_int(az, r_mm, w, luts)
    vr = encode_float_ref(az, r_mm, w)
    print(f"synthetic room scan: {len(az)} pts, int-vs-float cosine "
          f"{fidelity(vi, vr):.5f}")
    # determinism: same input twice -> bit-equal
    assert np.array_equal(vi, encode_int(az, r_mm, w, luts))
    print("determinism: bit-equal on repeat")
    print("selftest ok")


def fidelity_data():
    luts = make_luts()
    import runners.datasets as DS
    for name in ("spot",):
        try:
            b = DS.load(name, cap=60)
        except Exception as ex:
            print(f"{name}: unavailable ({ex})")
            continue
        cs = []
        for r, _, _ in b["keys"]:
            rr = DS.clean(b, r)
            az, r_mm, w = scan_to_ints(rr)
            if len(az) < 20:
                continue
            cs.append(fidelity(encode_int(az, r_mm, w, luts),
                               encode_float_ref(az, r_mm, w)))
        cs = np.array(cs)
        print(f"{name}: {len(cs)} scans, int-vs-float cosine "
              f"min {cs.min():.5f} med {np.median(cs):.5f}")
    import sspslam.worlds_dyn as DE
    b = DE.make("classroom", people=0, seed=11, n_beams=N_BEAM, cap=40)
    cs = []
    for r, _, _ in b["keys"]:
        az, r_mm, w = scan_to_ints(np.where(r < 30, r, np.inf))
        cs.append(fidelity(encode_int(az, r_mm, w, luts),
                           encode_float_ref(az, r_mm, w)))
    cs = np.array(cs)
    print(f"dyn-classroom: {len(cs)} scans, cosine min {cs.min():.5f} "
          f"med {np.median(cs):.5f}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    dict(selftest=selftest, fidelity=fidelity_data,
         match=selftest_match)[what]()
