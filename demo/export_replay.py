#!/usr/bin/env python3
"""Export a SHIPPED-pipeline replay of a real log for the browser demo.

The demo's live-JS Intel pipeline is a hand-ported approximation that has
repeatedly drifted from the Python system (and is currently broken by the
sandbox hand-tuning — its lattice/config is coupled to the synth sandbox
cfg). This exporter makes Python the source of truth: it runs the REAL
`ssp_bounded` deliverable on the log and records everything a thin JS player
needs to replay it faithfully — online per-keyframe estimates, per-keyframe
anchor references, anchor-pose snapshots at every relaxation (so graph snaps
replay exactly), surviving loop edges, and the final ATE-aligned reference.

Output: demo/replay_<log>.json (small enough to inline; the ranges for
drawing live in the existing INTEL_B64 blob — this file carries poses only).
JS wiring: a "shipped replay" mode reads REPLAY_META/REPLAY_B64 spliced
between /*__REPLAY_START__*/ ... /*__REPLAY_END__*/ markers in index.html.

Usage: python3 demo/export_replay.py [data/intel.log] [n_keyframes]
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ssp_slam as S            # noqa: E402
import ssp_slam_carmen as C     # noqa: E402
import ssp_slam_loop as L       # noqa: E402
import ssp_bounded as B         # noqa: E402

VALID_MAX = 40.0


class RecordingSLAM(B.BoundedSLAM):
    """Shipped pipeline + snapshot of anchor poses after every relaxation."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.snaps = []          # (keyframe k, anchor pose array copy)

    def relax(self):
        super().relax()
        self.snaps.append((self.k, np.array(self.anchors, np.float32)))


# --config variants: run a NON-shipped pipeline (e.g. the FPGA/binary config
# from ssp_fpga) and export it as replay slot 2 so the demo can show shipped
# vs quantized side by side. Sampling per config; label lands in the player.
CONFIGS = {
    "shipped": dict(label="shipped replay (Python)", sample="seg", kw={}),
    "fpga8": dict(label="FPGA replay: point+6b store+int8", sample="point",
                  kw=dict(spec=("i", 8, 7, 7), nph=16, nmag=4,
                          ring_scales=True)),
    "binary": dict(label="BINARY replay: point+2b store+QPSK", sample="point",
                   kw=dict(spec=("q",), nph=4, nmag=1, ring_scales=True)),
    # the FLOAT winner: per-beam point encoding, full-precision store —
    # band-dominant on fr079/aces/belg (RESULTS 2026-07-10 band table)
    "e2": dict(label="float winner: E2 point encoding",
               sample="point", kw=dict(spec=None, nph=0)),
    # the BINARY winner: point + 2-bit phase-only ring store + 8-bit integer
    # arithmetic (fr101 75 KB / fr079 110 KB; MIT 625 KB, band-equal)
    "lean": dict(label="binary winner: FPGA-lean (2b store + int8)",
                 sample="point",
                 kw=dict(spec=("i", 8, 7, 7), nph=4, nmag=1,
                         ring_scales=True)),
}


def make_slam(cfg):
    if not cfg["kw"]:
        return RecordingSLAM(robust=True, attempt_every=4, relax_every=25,
                             gap_kf=300, recent_aids=12)
    import ssp_fpga as F

    class RecFPGA(F.BandSLAM):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.snaps = []

        def relax(self):
            super().relax()
            self.snaps.append((self.k, np.array(self.anchors, np.float32)))

    kw = dict(cfg["kw"])
    if kw.get("spec") is not None:
        kw["spec"] = F.IntSpec(2, 2, 0, unit_w=True) \
            if kw["spec"][0] == "q" else F.IntSpec(*kw["spec"][1:])
    return RecFPGA(robust=True, attempt_every=4, relax_every=25,
                   gap_kf=300, recent_aids=12, eps=0.0, **kw)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/intel.log"
    cfg_name = "shipped"
    slot = 1
    for a in sys.argv[2:]:
        if a.startswith("--config="):
            cfg_name = a.split("=", 1)[1]
        elif a.startswith("--slot="):
            slot = int(a.split("=", 1)[1])
    cfg = CONFIGS[cfg_name]
    keys = C.keyframes(C.parse_flaser(path))
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        keys = keys[:int(sys.argv[2])]
    n = len(keys)
    nb = len(keys[0][0])
    beam = np.deg2rad(-90.0 + np.arange(nb) * (180.0 / nb))
    odom = np.stack([k[1] for k in keys])
    kts = np.array([t for _, _, t in keys])

    slam = make_slam(cfg)
    slam.store_dtype = np.complex64
    # est stays float64: float32 chaining of the guess alone perturbs the
    # chaotic closure cascade (measured: Intel 2.44 -> 3.97 m). Cast at pack.
    est = np.zeros((n, 3))
    for k, (r, opose, ts) in enumerate(keys):
        rr = np.where(r < VALID_MAX, r, np.inf)
        if cfg["sample"] == "point":
            import ssp_fpga as F
            pts, w = F.points_from_scan(rr, beam)
        else:
            pts, w, _ = S.scan_to_samples(rr, beam)
        guess = odom[0] if k == 0 else L.se2_mul(
            est[k - 1], L.se2_mul(L.se2_inv(odom[k - 1]), odom[k]))
        est[k] = slam.add_keyframe(pts, w, guess)
        if k % 1000 == 0:
            print(f"  kf {k}/{n}", flush=True)
    if slam.dirty:
        slam.relax()
    fin = np.stack([slam.pose_of(k) for k in range(n)]).astype(np.float32)
    aid = np.array([slam.aid_of(k) for k in range(n)], np.uint16)
    rel = np.stack([slam.kf_ref[k][1] for k in range(n)]).astype(np.float32)
    loops = np.array([(a, b) for a, b, Z, wt, wr, kind in slam.edges
                      if kind == "loop"], np.uint16).reshape(-1, 2)

    # ---- ATE vs reference (scoring only)
    ref = C.parse_flaser(path.replace(".log", ".gfs.log"))
    rts = np.array([t for _, _, t in ref])
    rxy = np.stack([p[:2] for _, p, _ in ref])
    j = np.abs(rts[:, None] - kts[None, :]).argmin(1)
    good = np.abs(rts - kts[j]) < 0.3
    al = C.align_se2(fin[j[good], :2].astype(float), rxy[good])
    e = np.linalg.norm(al - rxy[good], axis=1)
    ate = float(np.sqrt((e ** 2).mean()))
    print(f"{cfg_name} replay: ATE {ate:.3f} m over {int(good.sum())} ref "
          f"poses, {len(loops)} loop edges, {len(slam.snaps)} relax snapshots")

    # ---- pack a SELF-CONTAINED replay: the shipped keyframing (0.10 m/5 deg)
    # differs from the demo's dense no-odometry stride, so the replay carries
    # its own ranges + odometry + reference alongside the pose streams.
    # Layout (little-endian):
    #   u16[n*nBeams] ranges cm; f32[n*3] odom; f32[n*2] refXY; u8[n] refOk;
    #   f32[n*3] est(online); f32[n*3] fin; f32[n*3] rel; u16[n] aid;
    #   u16[nLoops*2] loop anchor pairs; u32[nSnaps] snapK;
    #   f32[nSnaps*nAnchor*3] snapshot anchor poses
    n_anchor = len(slam.anchors)
    snap_k = np.array([k for k, _ in slam.snaps], np.uint32)
    snap_poses = np.zeros((len(slam.snaps), n_anchor, 3), np.float32)
    for i, (_, P) in enumerate(slam.snaps):
        snap_poses[i, :len(P)] = P
    ranges = np.stack([k[0] for k in keys])
    r_cm = np.clip(np.round(ranges * 100), 0, 65535).astype("<u2")
    # per-keyframe reference match (same recipe, keyframe-indexed)
    jj = np.abs(kts[:, None] - rts[None, :]).argmin(1)
    kgood = np.abs(kts - rts[jj]) < 0.3
    ref_xy = np.zeros((n, 2), np.float32)
    ref_ok = np.zeros(n, np.uint8)
    ref_xy[kgood] = rxy[jj[kgood]].astype(np.float32)
    ref_ok[kgood] = 1
    est = est.astype(np.float32)
    blob = (r_cm.tobytes() + odom.astype("<f4").tobytes()
            + ref_xy.tobytes() + ref_ok.tobytes()
            + est.tobytes() + fin.tobytes() + rel.tobytes() + aid.tobytes()
            + loops.astype("<u2").tobytes()
            + snap_k.astype("<u4").tobytes() + snap_poses.tobytes())
    meta = dict(n=n, nBeams=nb, validMax=VALID_MAX, beamStartDeg=-90.0,
                beamStepDeg=180.0 / nb, nAnchor=n_anchor,
                nLoops=int(len(loops)), nSnaps=len(slam.snaps),
                anchorEvery=B.ANCHOR, ate=ate, nRefOk=int(ref_ok.sum()),
                keyTrans=C.KEY_TRANS, keyRotDeg=float(np.degrees(C.KEY_ROT)),
                log=path.rsplit("/", 1)[-1], config=cfg_name,
                label=cfg["label"])
    b64 = base64.b64encode(blob).decode()
    suffix = "" if slot == 1 else f"_{cfg_name}"
    out = ROOT / "demo" / ("replay_" + meta["log"].replace(".log", "")
                           + suffix + ".json")
    out.write_text(json.dumps(dict(meta=meta, b64=b64)))
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")

    if "--embed" in sys.argv:
        import re
        tag = "REPLAY" if slot == 1 else f"REPLAY{slot}"
        html_p = ROOT / "demo" / "index.html"
        html = html_p.read_text()
        payload = (f"/*__{tag}_START__*/\n"
                   f"const {tag}_META = {json.dumps(meta)};\n"
                   f'const {tag}_B64 = "{b64}";\n'
                   f"/*__{tag}_END__*/")
        if f"/*__{tag}_START__*/" in html:
            html2, ns = re.subn(
                rf"/\*__{tag}_START__\*/.*?/\*__{tag}_END__\*/",
                lambda _: payload, html, flags=re.S)
        else:
            html2, ns = re.subn(r"/\*__DATA_END__\*/",
                                lambda _: "/*__DATA_END__*/\n" + payload,
                                html, count=1)
        assert ns == 1, "marker splice failed"
        html_p.write_text(html2)
        print(f"embedded {tag} into {html_p} "
              f"({html_p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
