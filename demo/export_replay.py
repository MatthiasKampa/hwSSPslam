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
  python3 demo/export_replay.py dynsch8-p0 --config=interp2  # dynenv arm
  python3 demo/export_replay.py --embed                    # splice manifest

Dynenv arms (DYNENV below) build their bundle via ssp_dynenv.make() — the
reference is the synthetic ground truth — and are pinned to the banked
bench-grid numbers: the export aborts if ATE/loops drift.
"""
import base64
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sspslam.frontend as C     # noqa: E402
import sspslam.quantized as F            # noqa: E402
import runners.datasets as DS       # noqa: E402

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
    import experiments.sampling as SP
    return SP.sample_interp(rr, beam, 2, 63.4)

GT_NAMES = dict(gfs="GMapping-corrected reference",
                ident="GMapping-corrected reference (range-identity match)",
                stata="floorplan-anchored GT (independent)",
                exact="held-out reference (withheld odometry / synthetic GT)")

# --- dynamic synthetic environments (sspslam/worlds_dyn.py) ------------------------
# Registry of exportable dynenv arms. The bundle is BUILT by ssp_dynenv.make()
# (ssp_datasets-shaped, eval='exact'): the scoring/display reference is the
# SYNTHETIC GROUND TRUTH itself — exact poses, no external log. Config is
# PINNED to interp2 (bridge2 deploy sampler + shipped encoder), matching the
# RESULTS.md 2026-07-11 "Dynamic multi-room environments" bench grid (1024
# beams, seed 11); `expect` pins that grid's ATE/loops and the export ABORTS
# on any mismatch (protocol: no silent config drift).
DYN_GROUP = "Dynamic synth (people)"
DYNENV = {
    # small-room nominal: classroom ATE is people-invariant under pairing
    "dyncls-p5w": dict(
        make=dict(env="classroom", people=5, moving=True, seed=11,
                  n_beams=1024),
        short="classroom, 5 walking — small-room nominal (people-robust)",
        note="7×7 m room, 5 walkers with deterministic yield: closure "
             "precision 1.00, ATE 8 mm and people-invariant — walker "
             "dynamics at this density do not hurt the small room.",
        expect=dict(ate=0.008, loops=14),
    ),
    # DELIBERATE NEGATIVE SHOWCASE: identical-rooms aliasing, nobody present
    "dynsch8-p0": dict(
        make=dict(env="school", rooms=8, people=0, moving=False, seed=11,
                  n_beams=1024),
        short="school8, nobody — aliasing admits wrong closures "
              "(worse than odometry)",
        note="hallway + 8 IDENTICAL classrooms (483 m²), nobody present: "
             "the twin rooms alias, wrong closures are accepted (precision "
             "0.53, constraint errors to 1.5 m) and SLAM lands WORSE than "
             "raw odometry (0.95 vs 0.39 m). The verification wall at "
             "building scale.",
        negative=True,
        expect=dict(ate=0.954, loops=93),
    ),
    "dynsch8-p5s": dict(
        make=dict(env="school", rooms=8, people=5, moving=False, seed=11,
                  n_beams=1024),
        short="school8, 5 standing — people de-alias the twin rooms",
        note="same building + 5 standing people as symmetry-breaking "
             "landmarks: precision 0.78, ATE 0.35 m vs 0.95 with nobody — "
             "persistent bodies de-alias the identical rooms.",
        expect=dict(ate=0.348, loops=112),
    ),
}


def dyn_bundle(name, cap=None):
    import sspslam.worlds_dyn as ssp_dynenv
    return ssp_dynenv.make(cap=cap, **DYNENV[name]["make"])


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
    dyn = DYNENV.get(name)
    if dyn is not None:
        assert cfg_name == "interp2", \
            "dynenv arms are benched ONLY with the bridge2 deploy sampler " \
            "(run with --config=interp2; RESULTS.md 2026-07-11 grid)"
    cfg = CONFIGS[cfg_name]
    kw = dict(cfg["kw"])
    if kw.get("spec") is not None:
        kw["spec"] = F.IntSpec(2, 2, 0, unit_w=True) \
            if kw["spec"][0] == "q" else F.IntSpec(*kw["spec"][1:])
    if cfg.get("hex"):
        import sspslam.lattice_presets as ssp_lattice
        import experiments.hexreal as H
        ssp_lattice.set_hex(cfg["hex"])
        base = H.HexSLAM
    else:
        base = F.BandSLAM
    sample = _interp2 if cfg_name == "interp2" else cfg["sample"]
    # est stays float64 inside DS.run: float32 chaining of the guess alone
    # perturbs the chaotic closure cascade (measured: Intel 2.44 -> 3.97 m).
    src = dyn_bundle(name, cap) if dyn is not None else name
    r = DS.run(src, recording(base), cap=cap, sample=sample,
               eps=0.0, **kw)
    slam, bundle, est = r["slam"], r["bundle"], r["est"].astype(np.float32)
    keys, beam = bundle["keys"], bundle["beam"]
    n = len(keys)
    print(f"{name}/{cfg_name}: ATE {r['ate']:.3f} m (med {r['med']:.3f}) "
          f"over {r['n_ref']} ref poses, {r['loops']} loop edges, "
          f"{len(slam.snaps)} relax snapshots, mem {r['mem_kb']:.0f} KB")
    if dyn is not None:                 # pinned to the banked bench grid
        exp = dyn["expect"]
        if abs(r["ate"] - exp["ate"]) > 5e-4 or r["loops"] != exp["loops"]:
            raise SystemExit(
                f"ABORT {name}: export path gives ATE {r['ate']:.4f}/"
                f"{r['loops']} loops, bench grid says {exp['ate']:.3f}/"
                f"{exp['loops']} — config drift, do not ship")

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
                log=Path(bundle["path"]).name if bundle.get("path")
                else bundle["name"], config=cfg_name,
                label=cfg["label"])
    if dyn is not None:                 # selector group + one-line verdicts
        meta.update(group=DYN_GROUP, short=dyn["short"], note=dyn["note"],
                    gt="synthetic ground truth (exact poses)")
        if dyn.get("negative"):
            meta["negative"] = True
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
