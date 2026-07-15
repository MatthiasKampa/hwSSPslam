"""Camera-feature extraction for the ECP5 fusion track (dataset stand-in
for the OV5640 that will hang off the FPGA later).

Pipeline per frame (every stage integer / RTL-faithful — the golden model
is hw/ecp5/host/golden_cam.py, gated bit-exact by hw/ecp5/rtl/fast9.v):
  640x480 RGB JPEG -> integer luma -> 2x2 box bin (320x240 uint8)
  -> FAST-9 corners, threshold servoed per keyframe to the LIDAR
  VALID-BEAM COUNT of the aligned scan (point parity: "lidar and cam
  have approx the same amount of data").

Frames are aligned nearest-timestamp to the LIDAR KEYFRAMES of the run's
registry bundle; only aligned frames are decoded. Cache:
  data/spot_telluride/<rgbdir>/../cam_features.npz
    kf_ts, cam_ts, cam_ok, thresh, n_feat, n_lidar,
    feat_off (CSR offsets), feat_y, feat_x, feat_score, K (intrinsics)

Usage (repo root):
  python3 -m runners.spot_cam parse school_run2 | spot
  python3 -m runners.spot_cam stats school_run2
  python3 -m runners.spot_cam vectors school_run2   # tb fixtures (64x64)
"""
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hw" / "ecp5"
                       / "host"))
import golden_cam as GC          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "spot_telluride"
# run key -> (rgb shard dir, cache path); "spot" = the classroom session
RGB = {
    "spot": (BASE / "classroom_rgb", BASE / "cam_features.npz"),
    "school_run1": (BASE / "school_run1" / "rgb_d455",
                    BASE / "school_run1" / "cam_features.npz"),
    "school_run2": (BASE / "school_run2" / "rgb_d455",
                    BASE / "school_run2" / "cam_features.npz"),
}
ALIGN_TOL_MS = 60                # cam ~27-30 Hz -> nearest within 2 frames


def _index(run):
    """First pass: (shard, row) index + timestamps, no image decode."""
    import pyarrow.parquet as pq
    shards = sorted(RGB[run][0].glob("*.parquet"))
    assert shards, f"no rgb shards for {run} under {RGB[run][0]}"
    ts, where = [], []
    for si, sh in enumerate(shards):
        t = pq.read_table(sh, columns=["timestamp_ns"])
        v = np.array(t["timestamp_ns"], dtype=np.int64)
        ts.append(v)
        where += [(si, r) for r in range(len(v))]
    ts = np.concatenate(ts)
    srt = np.argsort(ts, kind="stable")
    return shards, ts[srt], [where[i] for i in srt]


def _lidar_kts_counts(run):
    """Keyframe timestamps (ns) + per-kf valid-beam counts from the run's
    scans.npz (registry keyframing: stride 4)."""
    d = BASE if run == "spot" else BASE / run
    z = np.load(d / "scans.npz")
    idx = np.arange(0, len(z["ts"]), 4)
    return z["ts"][idx], np.isfinite(z["ranges"][idx]).sum(1)


def parse(run):
    from PIL import Image
    import pyarrow.parquet as pq
    shards, cts, where = _index(run)
    kts, nlidar = _lidar_kts_counts(run)
    j = np.clip(np.searchsorted(cts, kts), 1, len(cts) - 1)
    j = j - (np.abs(cts[j - 1] - kts) < np.abs(cts[j] - kts))
    ok = np.abs(cts[j] - kts) < ALIGN_TOL_MS * 1e6
    need = {}                              # (shard,row) -> [kf indices]
    for k in np.flatnonzero(ok):
        need.setdefault(where[j[k]], []).append(k)
    print(f"{run}: {len(kts)} lidar kf, {int(ok.sum())} aligned cam frames "
          f"({len(cts)} available), decoding {len(need)} unique", flush=True)
    n = len(kts)
    thresh = np.zeros(n, np.uint8)
    nfeat = np.zeros(n, np.int32)
    cam_ts = np.where(ok, cts[j], 0)
    fy, fx, fs = [[] for _ in range(n)], [[] for _ in range(n)], \
                 [[] for _ in range(n)]
    K = None
    by_shard = {}
    for (si, row), ks in need.items():
        by_shard.setdefault(si, []).append((row, ks))
    done = 0
    for si in sorted(by_shard):
        t = pq.read_table(shards[si])
        if K is None:
            K = np.array(json.loads(t["camera_info"][0].as_py())["K"],
                         float).reshape(3, 3)
        for row, ks in sorted(by_shard[si]):
            im = np.asarray(Image.open(
                io.BytesIO(t["image"][row].as_py()["bytes"])).convert("RGB"))
            g = GC.bin2(GC.rgb_to_gray(im))
            for k in ks:                   # same frame may serve 2 kf
                tt, (ys, xs, sc) = GC.detect_target(g, int(nlidar[k]))
                thresh[k], nfeat[k] = tt, len(ys)
                fy[k], fx[k], fs[k] = ys, xs, sc
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(need)} frames", flush=True)
    off = np.zeros(n + 1, np.int64)
    off[1:] = np.cumsum(nfeat)
    cat = (lambda L, d: np.concatenate([np.asarray(v, d) for v in L])
           if off[-1] else np.zeros(0, d))
    np.savez_compressed(
        RGB[run][1], kf_ts=kts, cam_ts=cam_ts, cam_ok=ok, thresh=thresh,
        n_feat=nfeat, n_lidar=nlidar, feat_off=off,
        feat_y=cat(fy, np.uint16), feat_x=cat(fx, np.uint16),
        feat_score=cat(fs, np.uint16), K=K,
        img_w=320, img_h=240, bin=2)
    print(f"wrote {RGB[run][1]} ({int(ok.sum())} kf with features)",
          flush=True)


def stats(run):
    z = np.load(RGB[run][1])
    ok = z["cam_ok"]
    nf, nl = z["n_feat"][ok], z["n_lidar"][ok]
    dt = np.abs(z["cam_ts"][ok] - z["kf_ts"][ok]) / 1e6
    print(f"{run}: {int(ok.sum())}/{len(ok)} kf aligned "
          f"(cam-kf dt med {np.median(dt):.1f} ms)")
    print(f"  lidar valid beams/kf: med {int(np.median(nl))} "
          f"[{nl.min()}..{nl.max()}]")
    print(f"  cam features/kf:      med {int(np.median(nf))} "
          f"[{nf.min()}..{nf.max()}]  (thresh med "
          f"{int(np.median(z['thresh'][ok]))})")
    r = nf / np.maximum(nl, 1)
    print(f"  parity nf/nl: med {np.median(r):.2f} p10 "
          f"{np.percentile(r, 10):.2f} p90 {np.percentile(r, 90):.2f}")
    print(f"  bytes/kf: lidar slice 2048 | cam feats {int(np.median(nf))*6}"
          f" | scaled gray 76800 | full cloud 131072")


def vectors(run, k=None):
    """tb fixture: 64x64 crop of an aligned frame at its served threshold."""
    from PIL import Image
    import pyarrow.parquet as pq
    z = np.load(RGB[run][1])
    k = int(np.flatnonzero(z["cam_ok"])[10]) if k is None else k
    shards, cts, where = _index(run)
    j = int(np.searchsorted(cts, z["cam_ts"][k]))
    j = min(max(j, 0), len(cts) - 1)
    si, row = where[j]
    t = pq.read_table(shards[si])
    im = np.asarray(Image.open(
        io.BytesIO(t["image"][row].as_py()["bytes"])).convert("RGB"))
    g = GC.bin2(GC.rgb_to_gray(im))
    crop = g[88:152, 128:192].copy()       # 64x64 centre-ish
    tt = int(z["thresh"][k])
    out = ROOT / "hw" / "ecp5" / "build"
    out.mkdir(exist_ok=True)
    n = GC.write_vectors(crop, tt, str(out / "fast9"))
    (out / "fast9_thresh.txt").write_text(str(tt))
    print(f"vectors: kf {k} thresh {tt} -> {out}/fast9_{{img,exp}}.hex "
          f"({n} pre-NMS corners in crop)")


if __name__ == "__main__":
    cmd, run = sys.argv[1], sys.argv[2]
    dict(parse=parse, stats=stats, vectors=vectors)[cmd](run)
