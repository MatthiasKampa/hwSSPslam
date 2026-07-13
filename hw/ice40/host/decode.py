#!/usr/bin/env python3
"""decode.py — v7 FIDELITY-STREAM CONSUMER (laptop side of the dual-space
architecture): everything the host does with a map the chip dumps.

Input = exactly what top_solo produces: the resident-map image (2b codes,
seg-major, 4 codes/byte LSB-first — build/solo_map.hex format) + the
anchor table (16 B/entry: x_u i32, y_u i32, ah_q i16, cq i16, sq i16 —
build/solo_anchors.hex format). Lattice is a parameter (matcher space
oct60 today; the span11x60 fidelity space uses the same consumer with
the banked better extraction).

Implements the 2026-07-12 banked read-side recipe (RESULTS "cleanup
formulations program"):
  - per-ring COHERENCE profile C_r (GT-free map-health metric: mean |cos|
    over overlapping world-frame segment pairs; low fine-ring C_r = warp)
  - Cr^2-weighted (Wiener) mosaic imaging  [+.016 warped / +.005 clean]
  - per-segment CLEAN line pursuit (gamma dial: precision vs recall) +
    cross-segment consensus -> parametric wall map (~5 cm own-gauge)
  - (store note: the 3b dead-zero freeze — phase 2b + liveness bit —
    beats even the float store on both fixtures; RTL stream-format
    upgrade filed in the v7 spec)

Usage:
  python3 decode.py [map_hex] [anchors_hex] [--gamma G] [--out out.npz]
Defaults consume the v7 classroom fixture in ../build. Prints stats;
with --out writes lines/image/C_r for a UI. GT never enters the decode;
the classroom self-score at the end is labeled SCORING ONLY.
"""
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import solo                                                # noqa: E402

BUILD = Path(__file__).resolve().parents[1] / "build"

# ---- lattice (parameter; default = matcher space oct60, D=240) ----------
LAMS = [0.25, 0.5, 1.0, 2.0]
NANG = 60


def build_W(lams=LAMS, nang=NANG):
    ks = []
    for lam in lams:
        for a in range(nang):
            th = np.pi * a / nang
            ks.append([2 * np.pi / lam * np.cos(th),
                       2 * np.pi / lam * np.sin(th)])
    return np.array(ks)


def load_stream(map_hex, anchors_hex, live_hex=None, scl_hex=None,
                nang=NANG, n_ring=len(LAMS)):
    """Chip dump -> (codes, anchors, alive, ring-weights). alive/weights
    are None when the liveness/scales planes are absent (pre-v7 dumps).
    Liveness = the 3-bit dead-zero freeze (30 B/seg, bit k*60+j LSB-first);
    scales = per-ring Mmax as {e[4:0], m[7:0]} words (8 B/seg) -> w = m*2^e
    (BANKED: liveness theta=1/2 + scales beats the float store on
    extraction — school .6138 vs .5984, stata .6714 vs .6677)."""
    by = bytes(int(l, 16) for l in open(map_hex).read().split())
    seg_b = n_ring * nang // 4
    codes = []
    for s in range(len(by) // seg_b):
        c = []
        for b in by[s * seg_b:(s + 1) * seg_b]:
            for j in range(4):
                c.append((b >> (2 * j)) & 3)
        codes.append(np.array(c, np.int64))
    ab = bytes(int(l, 16) for l in open(anchors_hex).read().split())
    anchors = []
    for s in range(len(ab) // 16):
        ax, ay, ah, cq, sq = struct.unpack("<iihhh", ab[s * 16:s * 16 + 14])
        anchors.append((ax * solo.U, ay * solo.U,
                        2 * np.pi * ah / solo.TAU_Q))
    alive = wr = None
    if live_hex is not None:
        lb = bytes(int(l, 16) for l in open(live_hex).read().split())
        lseg = n_ring * nang // 8
        alive = []
        for s in range(len(lb) // lseg):
            bits = int.from_bytes(lb[s * lseg:(s + 1) * lseg], "little")
            alive.append(np.array([(bits >> i) & 1
                                   for i in range(n_ring * nang)], bool))
    if scl_hex is not None:
        sb = bytes(int(l, 16) for l in open(scl_hex).read().split())
        wr = []
        for s in range(len(sb) // (2 * n_ring)):
            ws = []
            for r in range(n_ring):
                wv = struct.unpack_from("<H", sb, s * 2 * n_ring + 2 * r)[0]
                ws.append((wv & 0xFF) * float(1 << (wv >> 8)))
            wr.append(np.array(ws))
    return codes, anchors[:len(codes)], alive, wr


def world_vectors(codes, anchors, W, nang=NANG, alive=None, wr=None):
    """Dequantize (phase-only QPSK; optional liveness mask + per-ring
    scale weights from the freeze scan) + place in the world frame:
    rotation = index permutation with conjugate wrap (grid part; the
    sub-grid der correction is a refinement the phase-only dump omits),
    translation = phase multiply."""
    out = []
    lat = np.pi / nang
    n_ring = len(codes[0]) // nang
    for si, (c, (ax, ay, ah)) in enumerate(zip(codes, anchors)):
        v = np.exp(1j * (np.pi / 2) * c)
        if alive is not None:
            v = v * alive[si]
        if wr is not None:
            v = v * np.repeat(wr[si], nang)
        m = int(round(ah / lat))
        vr = np.empty_like(v)
        for r in range(n_ring):
            blk = v[r * nang:(r + 1) * nang]
            idx = (np.arange(nang) - m) % (2 * nang)
            wrap = idx >= nang
            src = np.where(wrap, idx - nang, idx)
            b = blk[src]
            b = np.where(wrap, np.conj(b), b)
            vr[r * nang:(r + 1) * nang] = b
        out.append(vr * np.exp(1j * (W @ [ax, ay])))
    return out


def coherence(Wv, anchors, nang=NANG, pair_r=4.0):
    n_ring = len(Wv[0]) // nang
    apos = np.array([(a[0], a[1]) for a in anchors])
    pc = [[] for _ in range(n_ring)]
    for i in range(len(Wv)):
        for j in range(i + 1, len(Wv)):
            if np.linalg.norm(apos[i] - apos[j]) > pair_r:
                continue
            for r in range(n_ring):
                va = Wv[i][r * nang:(r + 1) * nang]
                vb = Wv[j][r * nang:(r + 1) * nang]
                den = np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12
                pc[r].append(abs(np.vdot(va, vb)) / den)
    return np.array([np.mean(p) if p else 0.0 for p in pc])


def wiener_image(Wv, anchors, W, Cr, bounds, step=0.10, dom_r=8.0,
                 nang=NANG):
    x0, x1, y0, y1 = bounds
    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys)
    cells = np.stack([gx.ravel(), gy.ravel()], 1)
    wc = np.repeat(Cr ** 2, nang)
    apos = np.array([(a[0], a[1]) for a in anchors])
    Bv = np.zeros(len(W), complex)
    for v in Wv:
        Bv += v
    Bv = Bv * wc
    d = np.sqrt(((cells[:, None, :] - apos[None]) ** 2).sum(-1)).min(1)
    keep = d <= dom_r
    img = np.full(len(cells), np.nan)
    E = np.exp(-1j * (cells[keep] @ W.T))
    img[keep] = (E @ Bv).real
    return cells, img.reshape(len(ys), len(xs))


# ---- per-segment CLEAN pursuit (lattice-generic port of the banked
# scratch_lineprior recipe; gamma = the banked recall/precision dial) ----
def atom(W, c, th, ln):
    h = 0.5 * ln * np.array([np.cos(th), np.sin(th)])
    return np.exp(1j * (W @ c)) * np.sinc((W @ h) / np.pi)


def pursuit(W, Bv, cells, k_max=12, tau=0.35, gamma=1.0):
    res = Bv.copy()
    E = np.exp(-1j * (cells @ W.T))
    ths = np.pi * np.arange(24) / 24
    offs = np.array([(dx, dy) for dx in (-0.15, 0, 0.15)
                     for dy in (-0.15, 0, 0.15)])
    out, first = [], None
    for _ in range(k_max):
        f = (E @ res).real
        pk = int(np.argmax(f))
        if first is None:
            first = f[pk]
        if f[pk] < tau * first or f[pk] <= 0:
            break
        best = None
        for off in offs:
            c = cells[pk] + off
            for th in ths:
                for ln in (0.6, 1.2, 2.4):
                    a = atom(W, c, th, ln)
                    na2 = float(np.real(np.vdot(a, a)))
                    amp = float(np.real(np.vdot(a, res))) / (na2 + 1e-12)
                    energy = amp * amp * na2
                    if amp > 0 and (best is None or energy > best[0]):
                        best = (energy, c, th, ln, amp, a)
        if best is None:
            break
        _, c, th, ln, amp, a = best
        res -= gamma * amp * a
        out.append((c, th, ln, gamma * amp))
    return out


def path_veto(lines, anchors, r_veto=0.25):
    """Physics prior (banked 2026-07-12 P7: school p50 0.226->0.087, p90
    1.413->0.731, recall free): a line crossing the DRIVEN PATH cannot
    be a wall — the anchor trail is the path, already in the stream."""
    trail = np.array([(a[0], a[1]) for a in anchors])
    keep = []
    for (c, th, ln, amp, seg) in lines:
        h = 0.5 * ln * np.array([np.cos(th), np.sin(th)])
        t = np.linspace(-1, 1, max(2, int(np.ceil(ln / 0.05))))[:, None]
        pts = c[None] + t * h
        if np.sqrt(((pts[:, None] - trail[None]) ** 2).sum(-1)).min() \
                > r_veto:
            keep.append((c, th, ln, amp, seg))
    return keep


def consensus(lines, d_max=0.20, th_max=np.deg2rad(15)):
    if not lines:
        return lines
    Cc = np.stack([l[0] for l in lines])
    TH = np.array([l[1] % np.pi for l in lines])
    SEG = np.array([l[4] for l in lines])
    keep = np.zeros(len(lines), bool)
    for i in range(len(lines)):
        d = np.linalg.norm(Cc - Cc[i], axis=1)
        dth = np.abs((TH - TH[i] + np.pi / 2) % np.pi - np.pi / 2)
        if ((d < d_max) & (dth < th_max) & (SEG != SEG[i])).any():
            keep[i] = True
    return [l for l, k in zip(lines, keep) if k]


def decode(map_hex, anchors_hex, live_hex=None, scl_hex=None, gamma=1.0):
    W = build_W()
    codes, anchors, alive, wr = load_stream(map_hex, anchors_hex,
                                            live_hex, scl_hex)
    Wv = world_vectors(codes, anchors, W, alive=alive, wr=wr)
    Cr = coherence(Wv, anchors)
    apos = np.array([(a[0], a[1]) for a in anchors])
    bounds = (apos[:, 0].min() - 4, apos[:, 0].max() + 4,
              apos[:, 1].min() - 4, apos[:, 1].max() + 4)
    cells, img = wiener_image(Wv, anchors, W, Cr, bounds)
    g = np.arange(-4.0, 4.0 + 1e-9, 0.10)
    gx, gy = np.meshgrid(g, g)
    loc = np.stack([gx.ravel(), gy.ravel()], 1)
    lines = []
    for i, (v, (ax, ay, _)) in enumerate(zip(Wv, anchors)):
        for l in pursuit(W, v, loc + [ax, ay], gamma=gamma):
            lines.append((l[0], l[1], l[2], l[3], i))
    lines = consensus(path_veto(lines, anchors))
    return dict(codes=codes, anchors=anchors, Cr=Cr, cells=cells, img=img,
                lines=lines, bounds=bounds)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    map_hex = args[0] if args else BUILD / "solo_map.hex"
    anc_hex = args[1] if len(args) > 1 else BUILD / "solo_anchors.hex"
    live_hex = args[2] if len(args) > 2 else None
    scl_hex = args[3] if len(args) > 3 else None
    # v7 chip-dump convention: sibling live/scl planes auto-detected
    if live_hex is None:
        cand = Path(str(map_hex).replace("map", "live"))
        if cand != Path(str(map_hex)) and cand.exists():
            live_hex = cand
    if scl_hex is None:
        cand = Path(str(map_hex).replace("map", "scl"))
        if cand != Path(str(map_hex)) and cand.exists():
            scl_hex = cand
    gamma = 1.0
    out = None
    for i, a in enumerate(sys.argv):
        if a == "--gamma":
            gamma = float(sys.argv[i + 1])
        if a == "--out":
            out = sys.argv[i + 1]
    if live_hex or scl_hex:
        print(f"dump planes: liveness {'ON' if live_hex else 'off'}, "
              f"ring scales {'ON' if scl_hex else 'off'}", flush=True)
    d = decode(map_hex, anc_hex, live_hex, scl_hex, gamma=gamma)
    print(f"decode: {len(d['codes'])} segments | C_r profile "
          + "/".join(f"{c:.2f}" for c in d["Cr"])
          + f" | {len(d['lines'])} consensus lines (gamma={gamma})",
          flush=True)
    if out:
        np.savez(out, Cr=d["Cr"], img=d["img"],
                 lines=np.array([(l[0][0], l[0][1], l[1], l[2], l[3])
                                 for l in d["lines"]]),
                 bounds=np.array(d["bounds"]))
        print(f"wrote {out}")
    # ---- classroom self-score (SCORING ONLY; GT walls from the feed) ----
    if "solo_map" in str(map_hex):
        import live as LV
        feed = LV.Feed(seed=11, laps=2, env="classroom", traj="orbit")
        segs = feed.segs.reshape(-1, 4)
        pts = []
        for x0, y0, x1, y1 in segs:
            n = max(2, int(np.hypot(x1 - x0, y1 - y0) / 0.05))
            t = np.linspace(0, 1, n)[:, None]
            pts.append(np.stack([x0 + t[:, 0] * (x1 - x0),
                                 y0 + t[:, 0] * (y1 - y0)], 1))
        Wp = np.concatenate(pts)
        lp = []
        for c, th, ln, amp, _ in d["lines"]:
            h = 0.5 * ln * np.array([np.cos(th), np.sin(th)])
            t = np.linspace(-1, 1, max(2, int(ln / 0.05)))[:, None]
            lp.append(c[None] + t * h[None])
        if lp:
            lp = np.concatenate(lp)
            dmin = np.sqrt(((lp[:, None] - Wp[None]) ** 2).sum(-1)).min(1)
            drec = np.sqrt(((Wp[:, None] - lp[None]) ** 2).sum(-1)).min(1)
            print(f"classroom self-score (SCORING ONLY): line precision "
                  f"p50 {np.median(dmin):.3f} p90 "
                  f"{np.percentile(dmin, 90):.3f} | wall recall@0.15 "
                  f"{(drec <= 0.15).mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
