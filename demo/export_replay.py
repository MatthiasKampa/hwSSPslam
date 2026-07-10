#!/usr/bin/env python3
"""Export a Python-pipeline replay of any registry dataset for the browser demo.

Python is the source of truth: this runs the REAL pipeline (shipped or an
ssp_fpga/ssp_hexreal config) via the ssp_datasets registry and records
everything the thin JS player needs — online per-keyframe estimates,
per-keyframe anchor references, anchor-pose snapshots at every relaxation
(graph snaps replay exactly), surviving loop edges, and the per-keyframe
scoring reference (gfs / range-identity / floorplan GT, per dataset).

Blob layout (little-endian), one base64 string per replay:
  u16[n*nBeams] ranges cm (beam-STRIDED for display; poses use full scans)
  f32[n*3] odom; f32[n*2] refXY; u8[n] refOk;
  f32[n*3] est(online); f32[n*3] fin; f32[n*3] rel; u16[n] aid;
  u16[nLoops*2] loop anchor pairs; u32[nSnaps] snapK;
  f32[nSnaps*nAnchor*3] snapshot anchor poses

Files land in demo/replays/ + are listed in demo/replays/manifest.json;
`--embed` splices every manifest entry as one REPLAY_PACK array between the
/*__REPLAYS_START__*/ ... /*__REPLAYS_END__*/ markers in index.html.

Usage:
  python3 demo/export_replay.py intel                      # shipped config
  python3 demo/export_replay.py belg --config=hex63
  python3 demo/export_replay.py stata --config=shipped
  python3 demo/export_replay.py --embed                    # splice manifest
"""
import base64
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ssp_slam_carmen as C     # noqa: E402
import ssp_fpga as F            # noqa: E402
import ssp_datasets as DS       # noqa: E402

RDIR = ROOT / "demo" / "replays"
MANIFEST = RDIR / "manifest.json"

# label lands in the player's source selector; sample/kw as in the studies
CONFIGS = {
    "shipped": dict(label="shipped (float)", sample="seg",
                    kw=dict(spec=None, nph=0)),
    "e2": dict(label="float winner: E2 point encoding", sample="point",
               kw=dict(spec=None, nph=0)),
    "fpga8": dict(label="FPGA: point+6b store+int8", sample="point",
                  kw=dict(spec=("i", 8, 7, 7), nph=16, nmag=4,
                          ring_scales=True)),
    "binary": dict(label="BINARY: point+2b store+QPSK", sample="point",
                   kw=dict(spec=("q",), nph=4, nmag=1, ring_scales=True)),
    # the BINARY winner: point + 2-bit phase-only ring store + int8 matcher
    "lean": dict(label="binary winner: FPGA-lean (2b store + int8)",
                 sample="point",
                 kw=dict(spec=("i", 8, 7, 7), nph=4, nmag=1,
                         ring_scales=True)),
    # full-circle hex lattice (non-Manhattan winner: belg 2.07 vs 2.64)
    "hex63": dict(label="hex SSP lattice (63 dirs, full circle)",
                  sample="seg", kw=dict(spec=None, nph=0), hex=63),
    # the wide-FOV/dense-head deploy sampler (encoder study 2026-07-10):
    # bridged PAIR sub-points at the 63.4-deg occlusion gate — on the
    # 1040-beam stata target proxy 0.196 vs raw points 1.659
    "interp2": dict(label="deploy sampler: bridged pairs @ 63.4°",
                    sample=None, kw=dict(spec=None, nph=0)),
}


def _interp2(rr, beam):
    import ssp_sampling as SP
    return SP.sample_interp(rr, beam, 2, 63.4)

GT_NAMES = dict(gfs="GMapping-corrected reference",
                ident="GMapping-corrected reference (range-identity match)",
                stata="floorplan-anchored GT (independent)")


def recording(base):
    """base pipeline + snapshot of anchor poses after every relaxation."""

    class Rec(base):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.snaps = []

        def relax(self):
            super().relax()
            self.snaps.append((self.k, np.array(self.anchors, np.float32)))

    return Rec


def export(name, cfg_name, cap=None, stride=None):
    cfg = CONFIGS[cfg_name]
    kw = dict(cfg["kw"])
    if kw.get("spec") is not None:
        kw["spec"] = F.IntSpec(2, 2, 0, unit_w=True) \
            if kw["spec"][0] == "q" else F.IntSpec(*kw["spec"][1:])
    if cfg.get("hex"):
        import ssp_lattice
        import ssp_hexreal as H
        ssp_lattice.set_hex(cfg["hex"])
        base = H.HexSLAM
    else:
        base = F.BandSLAM
    sample = _interp2 if cfg_name == "interp2" else cfg["sample"]
    # est stays float64 inside DS.run: float32 chaining of the guess alone
    # perturbs the chaotic closure cascade (measured: Intel 2.44 -> 3.97 m).
    r = DS.run(name, recording(base), cap=cap, sample=sample,
               eps=0.0, **kw)
    slam, bundle, est = r["slam"], r["bundle"], r["est"].astype(np.float32)
    keys, beam = bundle["keys"], bundle["beam"]
    n = len(keys)
    print(f"{name}/{cfg_name}: ATE {r['ate']:.3f} m (med {r['med']:.3f}) "
          f"over {r['n_ref']} ref poses, {r['loops']} loop edges, "
          f"{len(slam.snaps)} relax snapshots, mem {r['mem_kb']:.0f} KB")

    fin = r["fin"].astype(np.float32)
    aid = np.array([slam.aid_of(k) for k in range(n)], np.uint16)
    rel = np.stack([slam.kf_ref[k][1] for k in range(n)]).astype(np.float32)
    loops = np.array([(a, b) for a, b, Z, wt, wr, kind in slam.edges
                      if kind == "loop"], np.uint16).reshape(-1, 2)
    n_anchor = len(slam.anchors)
    snap_k = np.array([k for k, _ in slam.snaps], np.uint32)
    snap_poses = np.zeros((len(slam.snaps), n_anchor, 3), np.float32)
    for i, (_, P) in enumerate(slam.snaps):
        snap_poses[i, :len(P)] = P

    # display ranges: stride the beams to ~<=260 for blob size (poses were
    # computed on FULL scans; the stride only thins the drawn point cloud)
    nb = len(beam)
    if stride is None:
        stride = max(1, int(np.ceil(nb / 260)))
    ranges = np.stack([k[0] for k in keys])[:, ::stride]
    beam_s = beam[::stride]
    r_cm = np.clip(np.round(ranges * 100), 0, 65535).astype("<u2")
    ref_xy, ref_ok = DS.ref_for_keys(bundle)

    odom = bundle["odom"]
    blob = (r_cm.tobytes() + odom.astype("<f4").tobytes()
            + ref_xy.tobytes() + ref_ok.tobytes()
            + est.tobytes() + fin.tobytes() + rel.tobytes() + aid.tobytes()
            + loops.astype("<u2").tobytes()
            + snap_k.astype("<u4").tobytes() + snap_poses.tobytes())
    step_deg = float(np.degrees(beam_s[1] - beam_s[0]))
    meta = dict(n=n, nBeams=len(beam_s), validMax=float(bundle["rmax"]),
                beamStartDeg=float(np.degrees(beam_s[0])),
                beamStepDeg=step_deg, nAnchor=n_anchor,
                nLoops=int(len(loops)), nSnaps=len(slam.snaps),
                anchorEvery=5, ate=r["ate"], med=r["med"],
                nRefOk=int(ref_ok.sum()), memKb=round(r["mem_kb"]),
                keyTrans=C.KEY_TRANS, keyRotDeg=float(np.degrees(C.KEY_ROT)),
                dataset=name, evalKind=bundle["eval"],
                gt=GT_NAMES[bundle["eval"]],
                log=Path(bundle["path"]).name, config=cfg_name,
                label=cfg["label"])
    RDIR.mkdir(exist_ok=True)
    out = RDIR / f"replay_{name}_{cfg_name}.json"
    out.write_text(json.dumps(dict(meta=meta, b64=base64.b64encode(
        blob).decode())))
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    manifest_add(out.name)


def manifest_add(fname):
    order = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    if fname not in order:
        order.append(fname)
        MANIFEST.write_text(json.dumps(order, indent=1))
        print(f"manifest + {fname}")


def embed():
    order = json.loads(MANIFEST.read_text())
    parts = []
    for fname in order:
        d = json.loads((RDIR / fname).read_text())
        parts.append("{\"meta\": " + json.dumps(d["meta"])
                     + ", \"b64\": \"" + d["b64"] + "\"}")
        m = d["meta"]
        print(f"  pack + {m.get('dataset', m['log'])}/{m.get('config', '?')}"
              f"  ATE {m['ate']:.3f}  ({len(d['b64']) // 1024} KB b64)")
    payload = ("/*__REPLAYS_START__*/\nconst REPLAY_PACK = [\n"
               + ",\n".join(parts) + "\n];\n/*__REPLAYS_END__*/")
    html_p = ROOT / "demo" / "index.html"
    html = html_p.read_text()
    html2, ns = re.subn(r"/\*__REPLAYS_START__\*/.*?/\*__REPLAYS_END__\*/",
                        lambda _: payload, html, flags=re.S)
    assert ns == 1, "REPLAYS markers not found in index.html (JS rework?)"
    html_p.write_text(html2)
    print(f"embedded {len(order)} replays into {html_p} "
          f"({html_p.stat().st_size:,} bytes)")


def main():
    if "--embed" in sys.argv:
        embed()
        return
    name = sys.argv[1]
    cfg_name, cap, stride = "shipped", None, None
    for a in sys.argv[2:]:
        if a.startswith("--config="):
            cfg_name = a.split("=", 1)[1]
        elif a.startswith("--cap="):
            cap = int(a.split("=", 1)[1])
        elif a.startswith("--stride="):
            stride = int(a.split("=", 1)[1])
    export(name, cfg_name, cap, stride)


if __name__ == "__main__":
    main()
