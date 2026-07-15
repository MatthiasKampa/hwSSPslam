"""TUM RGB-D adapter (6-DoF mocap GT; user 2026-07-14: "get a 3D/6D SLAM
vision data set").

Sequences land as tgz under data/tum/ (rgb 640x480 @30 Hz, registered
16-bit depth, groundtruth.txt = 100 Hz mocap [ts tx ty tz qx qy qz qw]).
parse() associates rgb<->depth<->gt by nearest timestamp (20 ms tol),
keyframes at stride 6 (~5 Hz — the house keyframe rate), bins rgb to the
QVGA working resolution via the integer pipeline (golden_cam), and caches
data/tum/<seq>.npz: gray (n,240,320 u8), depth_mm (n,240,320 u16, 0 =
invalid), gt (n,7 pose), K (3x3, bin-2-adjusted).

Intrinsics (TUM defaults per freiburg station, full-res):
  fr2: fx 520.9 fy 521.0 cx 325.1 cy 249.7
  fr3: fx 535.4 fy 539.2 cx 320.1 cy 247.6
Depth factor 5000/m. K is stored PRE-DIVIDED by 2 (bin-2 frame) so
bearings come from K^-1 [x, y, 1] on binned pixel coords directly.

Usage: python3 -m runners.tum parse rgbd_dataset_freiburg3_long_office_household
       python3 -m runners.tum list
"""
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hw" / "ecp5"
                       / "host"))
import golden_cam as GC          # noqa: E402

BASE = Path(__file__).resolve().parents[1] / "data" / "tum"
K_FULL = {"freiburg1": (517.3, 516.5, 318.6, 255.3),
          "freiburg2": (520.9, 521.0, 325.1, 249.7),
          "freiburg3": (535.4, 539.2, 320.1, 247.6)}
STRIDE = 6                       # 30 Hz -> 5 Hz keyframes
TOL_S = 0.02


def _k_of(seq):
    for k, v in K_FULL.items():
        if k in seq:
            fx, fy, cx, cy = v
            return np.array([[fx / 2, 0, cx / 2], [0, fy / 2, cy / 2],
                             [0, 0, 1.0]])
    raise SystemExit(f"unknown freiburg station in {seq}")


def parse(seq, stride=STRIDE):
    from PIL import Image
    tgz = BASE / f"{seq}.tgz"
    out = BASE / (f"{seq}.npz" if stride == STRIDE
                  else f"{seq}_s{stride}.npz")
    tf = tarfile.open(tgz)
    names = tf.getnames()
    root = names[0].split("/")[0]

    def read_index(fname):
        txt = tf.extractfile(f"{root}/{fname}").read().decode()
        rows = [ln.split() for ln in txt.splitlines()
                if ln and not ln.startswith("#")]
        return (np.array([float(r[0]) for r in rows]),
                [r[1] for r in rows])

    rts, rfiles = read_index("rgb.txt")
    dts, dfiles = read_index("depth.txt")
    gtxt = tf.extractfile(f"{root}/groundtruth.txt").read().decode()
    g = np.array([[float(x) for x in ln.split()]
                  for ln in gtxt.splitlines()
                  if ln and not ln.startswith("#")])
    gts, gpose = g[:, 0], g[:, 1:8]

    pick = np.arange(0, len(rts), stride)
    grays, depths, poses, kts = [], [], [], []
    for i in pick:
        jd = int(np.abs(dts - rts[i]).argmin())
        jg = int(np.abs(gts - rts[i]).argmin())
        if abs(dts[jd] - rts[i]) > TOL_S or abs(gts[jg] - rts[i]) > TOL_S:
            continue
        rgb = np.asarray(Image.open(tf.extractfile(
            f"{root}/{rfiles[i]}")).convert("RGB"))
        dep = np.asarray(Image.open(tf.extractfile(
            f"{root}/{dfiles[jd]}")))
        grays.append(GC.bin2(GC.rgb_to_gray(rgb)))
        d = dep.astype(np.uint32)
        d2 = np.minimum(np.minimum(d[0::2, 0::2], d[0::2, 1::2]),
                        np.minimum(d[1::2, 0::2], d[1::2, 1::2]))
        depths.append((d2 * 1000 // 5000).astype(np.uint16))  # -> mm
        poses.append(gpose[jg])
        kts.append(rts[i])
        if len(grays) % 100 == 0:
            print(f"  {len(grays)} keyframes", flush=True)
    np.savez_compressed(out, gray=np.stack(grays),
                        depth_mm=np.stack(depths), gt=np.stack(poses),
                        kts=np.array(kts), K=_k_of(seq))
    print(f"wrote {out}: {len(grays)} kf, "
          f"{(out.stat().st_size / 1e6):.0f} MB", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "list":
        for f in sorted(BASE.glob("*.tgz")):
            print(f.stem)
    else:
        parse(sys.argv[2] if len(sys.argv) > 2 else sys.argv[1],
              stride=int(sys.argv[3]) if len(sys.argv) > 3 else STRIDE)
