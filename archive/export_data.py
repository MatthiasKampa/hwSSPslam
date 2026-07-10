#!/usr/bin/env python3
"""Export a keyframed subset of the Intel Research Lab CARMEN log for the
browser demo, embedding it INLINE into demo/index.html (single-file, strict
CSP: no fetch, no external files).

Pipeline (mirrors ssp_slam_carmen.py):
  - parse FLASER lines from data/intel.log
  - keyframe at 0.10 m / 5 deg odometry motion (KEY_TRANS/KEY_ROT)
  - take the first N_KF keyframes (early corridor loops)
  - timestamp-match the RBPF-corrected intel.gfs.log poses (0.3 s slop) and
    rigidly transform them into the raw-odometry start frame (first matched
    pair anchors translation + heading), so the page can show an ATE proxy
    without running an alignment solver in JS.

Binary layout (little-endian), then base64:
  uint16[N_KF * n_beams]  ranges in cm (0 = invalid; >= VALID_MAX treated as
                          no-return by the page)
  float32[N_KF * 3]       raw odometry poses x, y, theta  [m, m, rad]
  float32[N_KF * 2]       reference positions x, y in the odom start frame
                          (garbage where invalid)
  uint8[N_KF]             reference validity flags

The blob is spliced into index.html between the __DATA_START__/__DATA_END__
markers as  const INTEL_META = {...}; const INTEL_B64 = "...";

Usage: python3 demo/export_data.py [n_keyframes]   (default 2200; run from repo root
       or from demo/ — paths are resolved relative to this file)
"""
import base64
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data" / "intel.log"
REF_LOG = ROOT / "data" / "intel.gfs.log"
HTML = Path(__file__).resolve().parent / "index.html"

# DENSE keyframing (0.05 m / 2.5 deg): the demo pipeline runs WITHOUT
# odometry (constant-velocity guess from previous estimates only), so it
# needs small per-step deltas; the classic 0.10/5.0 stride was tuned for an
# odometry-composed motion prior. Override: argv[2]=trans_m argv[3]=rot_deg.
KEY_TRANS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
KEY_ROT = np.deg2rad(float(sys.argv[3]) if len(sys.argv) > 3 else 2.5)
VALID_MAX = 40.0
N_KF = int(sys.argv[1]) if len(sys.argv) > 1 else 4122


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def parse_flaser(path):
    out = []
    with open(path) as f:
        for line in f:
            if not line.startswith("FLASER"):
                continue
            p = line.split()
            n = int(p[1])
            r = np.array(p[2:2 + n], float)
            x, y, th = (float(v) for v in p[2 + n:5 + n])
            out.append((r, np.array([x, y, th]), float(p[-1])))
    return out


def keyframes(scans):
    keys, last = [], None
    for r, pose, ts in scans:
        if last is None or np.linalg.norm(pose[:2] - last[:2]) > KEY_TRANS \
                or abs(wrap(pose[2] - last[2])) > KEY_ROT:
            keys.append((r, pose, ts))
            last = pose
    return keys


def main():
    scans = parse_flaser(LOG)
    keys = keyframes(scans)
    print(f"{len(scans)} scans -> {len(keys)} keyframes; exporting first {N_KF}")
    keys = keys[:N_KF]
    n_kf = len(keys)
    n_beams = len(keys[0][0])
    assert all(len(k[0]) == n_beams for k in keys)

    ranges = np.stack([k[0] for k in keys])                  # (n_kf, n_beams) m
    odom = np.stack([k[1] for k in keys]).astype(np.float32)  # (n_kf, 3)
    kts = np.array([k[2] for k in keys])

    # ---- reference: timestamp match + rigid transform into odom start frame
    ref_xy = np.zeros((n_kf, 2), np.float32)
    ref_ok = np.zeros(n_kf, np.uint8)
    try:
        ref = parse_flaser(REF_LOG)
        rts = np.array([t for _, _, t in ref])
        rps = np.stack([p for _, p, _ in ref])
        j = np.abs(kts[:, None] - rts[None, :]).argmin(1)
        good = np.abs(kts - rts[j]) < 0.3
        i0 = int(np.flatnonzero(good)[0])
        # anchor: first matched pair defines the SE(2) transform ref -> odom
        r0, o0 = rps[j[i0]], keys[i0][1]
        dth = wrap(o0[2] - r0[2])
        c, s = np.cos(dth), np.sin(dth)
        R = np.array([[c, -s], [s, c]])
        t = o0[:2] - R @ r0[:2]
        xy = rps[j][:, :2] @ R.T + t
        ref_xy[good] = xy[good].astype(np.float32)
        ref_ok[good] = 1
        print(f"reference: {int(good.sum())}/{n_kf} keyframes matched "
              f"(anchor kf {i0}, heading offset {np.degrees(dth):.2f} deg)")
    except FileNotFoundError:
        print("no intel.gfs.log — page will show odometry only")

    # ---- pack
    r_cm = np.clip(np.round(ranges * 100), 0, 65535).astype("<u2")
    blob = (r_cm.tobytes()
            + odom.astype("<f4").tobytes()
            + ref_xy.astype("<f4").tobytes()
            + ref_ok.tobytes())
    b64 = base64.b64encode(blob).decode()

    # bbox over odom + valid reference, padded (est stays inside this envelope)
    xs = np.concatenate([odom[:, 0], ref_xy[ref_ok > 0, 0]])
    ys = np.concatenate([odom[:, 1], ref_xy[ref_ok > 0, 1]])
    pad = 3.0
    bbox = [float(xs.min() - pad), float(ys.min() - pad),
            float(xs.max() + pad), float(ys.max() + pad)]

    meta = {
        "nKF": n_kf, "nBeams": n_beams, "validMax": VALID_MAX,
        "beamStartDeg": -90.0, "beamStepDeg": 1.0,   # SICK: CCW, -90..+89
        "keyTrans": KEY_TRANS, "keyRotDeg": float(np.degrees(KEY_ROT)),
        "bbox": bbox, "nRefOk": int(ref_ok.sum()),
        "source": "data/intel.log (Radish / StachnissLab, Dirk Haehnel)",
    }

    html = HTML.read_text()
    payload = ("/*__DATA_START__*/\n"
               f"const INTEL_META = {json.dumps(meta)};\n"
               f'const INTEL_B64 = "{b64}";\n'
               "/*__DATA_END__*/")
    out, n_sub = re.subn(r"/\*__DATA_START__\*/.*?/\*__DATA_END__\*/",
                         lambda _: payload, html, flags=re.S)
    assert n_sub == 1, "data markers not found in index.html"
    HTML.write_text(out)
    print(f"embedded {len(blob):,} bytes raw -> {len(b64):,} b64 chars; "
          f"index.html now {HTML.stat().st_size:,} bytes")
    print(f"bbox {np.round(bbox, 1).tolist()}  beams {n_beams}  kf {n_kf}")


if __name__ == "__main__":
    main()
