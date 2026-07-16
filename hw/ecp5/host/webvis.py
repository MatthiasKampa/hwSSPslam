#!/usr/bin/env python3
"""ECP5 FPGA-in-the-loop WEB VISUALIZATION v3 (STREAM.md v1.1).

ARCHITECTURE (2026-07-16 directive): the host is a SHIM. NO python SLAM
runs anywhere in the serving path — SLAM lives on the FPGA system.
Every scan streams through the board (digest-verified), is ENCODED ON
CHIP (0x92 vector golden-crosschecked bit-exact), and its 2-bit QPSK
codes land in the ON-CHIP MAP BANK; the laptop fetches the compressed
segments (0x0F -> 0x93) and DECODES them into the chip-map image.
Lidar points are added to the display map at the POSE ESTIMATE:
  live      the pose in the scan datagram (robot odometry today, the
            on-chip tracker when that rung lands)
  datasets  data/spot_telluride/est_demo.npz — trajectories computed
            ONCE OFFLINE by `webvis.py estcache` with the same algebra
            the chip tracker will run (make_slam); webvis only replays
            them. Fallback: the bundle's recorded reference.

CAMERA lane: frames -> the exported vision head's DESC BITS (cbits C
kernel when built, headio numpy fallback) -> per-keyframe appearance
grids + the cam VSA object map. Query-by-example, class-chip, patch and
reverse queries; no class labels (appearance is what the encode
honestly allows). GT/withheld reference is a display-only ghost.

  python3 hw/ecp5/host/webvis.py selftest        # headless, numbers only
  python3 hw/ecp5/host/webvis.py serve [port]    # default 8790
  python3 hw/ecp5/host/webvis.py estcache [name] # build est_demo.npz
"""
import base64
import json
import os
import queue
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import serial                                                  # noqa: E402

import hw_stream as HS                                         # noqa: E402
import hw.ice40.golden as G                                    # noqa: E402
import runners.datasets as DS                                  # noqa: E402
import runners.spot as SP                                      # noqa: E402
import sspslam.encoder as S                                    # noqa: E402
import sspslam.lattice as L                                    # noqa: E402
import sspslam.quantized as F                                  # noqa: E402

PORT_SER = "/dev/cu.usbserial-DK0GEIG0"
KF_DT = 0.2                     # 5 Hz keyframes
MAX_PTS = 60_000
W_MAIN = L.ENC_MAIN.W           # (240, 2) — the chip's lattice band
HEAD_NPZ = ROOT / "sspax" / "artifacts" / "vision_head.npz"


# ---------------------------------------------------------------- FPGA lane
class Fpga:
    """scan -> silicon: digest verify + ON-CHIP encode (golden-checked)
    + on-chip compressed map bank (fetchable)."""

    def __init__(self, port=PORT_SER, baud=2_000_000):
        self.s = serial.Serial(port, baud, timeout=0.02)
        time.sleep(0.3)
        self.s.reset_input_buffer()
        self.tx = HS.Sender(self.s)
        self.tx.ctrl(0)
        self.rxbuf = bytearray()
        self.luts = G.make_luts()
        self.dig_ok = self.vec_ok = self.total = 0
        self.bytes = 0
        self.t0 = time.time()

    def _await(self, want, deadline):
        out = []
        while time.time() < deadline:
            out += HS.read_pkts(self.s, 1, timeout=0.05, buf=self.rxbuf)
            if any(t == want for t, _ in out):
                break
        return out

    def verify_encode(self, fid, r, t_us=0):
        """-> (dig_ok, vec_ok, ms). vec_ok = on-chip vector bit-exact
        vs hw.ice40.golden on the transmitted integers."""
        mm = np.where(np.isfinite(r), np.clip(np.round(r * 1000.0), 0,
                                              65535), 0.0)
        mm = mm.astype(np.uint16)[None, :]
        t0 = time.time()
        self.tx.lidar_frame(fid, mm, [33], t_us=t_us, cols_per_pkt=8)
        self.total += 1
        self.bytes += mm.nbytes
        ref = HS.digest_ref(np.ascontiguousarray(mm.T).tobytes())
        keep = (mm[0] > 0) & (mm[0] <= G.R_MASK_MM)
        az = np.flatnonzero(keep).astype(np.int32)
        exp = G.encode_int(az, mm[0][keep].astype(np.int32),
                           np.full(len(az), 127, np.int32), self.luts)
        dig = vec = False
        for typ, pl in self._await(0x92, t0 + 0.6):
            if typ == 0x90:
                _, rfid, dg, cnt = struct.unpack("<BIHI", pl)
                dig = rfid == fid and dg == ref and cnt == mm.nbytes
            elif typ == 0x92:
                rfid, = struct.unpack_from("<I", pl, 0)
                acc = np.frombuffer(pl[4:], "<i4").reshape(240, 2)
                vec = rfid == fid and np.array_equal(acc, exp)
        self.dig_ok += dig
        self.vec_ok += vec
        return dig, vec, (time.time() - t0) * 1e3

    def map_seg(self, slot):
        """Fetch the on-chip compressed segment -> 240 QPSK codes."""
        self.tx.pkt(0x0F, struct.pack("<B", slot & 63))
        for typ, pl in self._await(0x93, time.time() + 0.4):
            if typ == 0x93 and pl[0] == (slot & 63):
                codes = np.frombuffer(pl[1:], np.uint8)
                out = np.empty(240, np.uint8)
                for j in range(4):
                    out[j::4] = (codes >> (2 * j)) & 3
                return out
        return None

    def kbps(self):
        return self.bytes / max(time.time() - self.t0, 1e-6) / 1024.0


# --------------------------------------------------------------- data feeds
DATASETS = ("spot", "school_run1", "school_run2")   # REAL data only
# live /capture dumps double as replay datasets ("inspect without
# driving"): capture_<ts>.npz in SSP_CAPTURE_DIR (default ~) appear in
# the selector as drive_<ts>. Discovered at process start. Only the
# LATEST capture WITH cam bits is offered (user directive — cam-less
# drives, e.g. pre-fix drive 1, replay with a dead camera panel).
CAPTURE_DIR = Path(os.environ.get("SSP_CAPTURE_DIR", str(Path.home())))


def _has_cam(p):
    try:
        z = np.load(p, allow_pickle=True)      # lazy: reads cam_kf only
        return "cam_kf" in z.files and len(z["cam_kf"]) > 0
    except Exception:
        return False


CAPTURES = {f"drive_{p.stem.split('_', 1)[1]}": p
            for p in [q for q in sorted(CAPTURE_DIR.glob("capture_*.npz"))
                      if _has_cam(q)][-1:]}


def avail_datasets():
    d = ROOT / "data" / "spot_telluride"
    out = []
    for n in DATASETS:
        if (d / "scans.npz" if n == "spot" else d / n).exists():
            out.append(n)
    return out + sorted(CAPTURES)


def capture_bundle(name):
    """Replay a live /capture dump: scans register at the RECORDED
    odom pose (exactly what live showed); the cam lane replays the
    stored desc-bit grids (C-kernel output) — queries work offline."""
    z = np.load(CAPTURES[name], allow_pickle=True)
    mm = z["mm"]
    ts = np.asarray(z["ts"], float)
    est = np.asarray(z["est"], float)
    keys = [(np.where(mm[i] > 0, mm[i] / 1000.0, np.inf),
             est[i], ts[i]) for i in range(len(mm))]
    b = dict(name=name, kind="capture", keys=keys,
             beam=-np.pi + np.arange(1024) * (2 * np.pi / 1024),
             gt_ok=np.zeros(len(keys), bool), kts=ts,
             rmin=0.3, rmax=60.0, est=est,
             kdt=float(np.median(np.diff(ts))) if len(ts) > 1 else KF_DT)
    if "cam_kf" in z.files and len(z["cam_kf"]):
        b["cam_bits"] = dict(zip([int(x) for x in z["cam_kf"]],
                                 np.asarray(z["cam_bits"], bool)))
        b["cam_jpeg"] = dict(zip([int(x) for x in z["jpeg_kf"]],
                                 [bytes(j) for j in z["jpegs"]]))
    return b


EST_DEMO = ROOT / "data" / "spot_telluride" / "est_demo.npz"


def _attach_est(b):
    """Attach the offline-built pose-estimate trajectories
    (est_demo.npz, keyed by dataset name; '<name>__q' = the QUANTIZED
    recipe variant, nph=4 chip-precision store). With no python SLAM in
    the serving path this is where dataset points register; without it
    the recorded reference is the fallback."""
    if EST_DEMO.exists():
        z = np.load(EST_DEMO)
        if b["name"] in z.files and len(z[b["name"]]) >= len(b["keys"]):
            b["est"] = np.asarray(z[b["name"]], float)
        qk = b["name"] + "__q"
        if qk in z.files and len(z[qk]) >= len(b["keys"]):
            b["est_q"] = np.asarray(z[qk], float)
    return b


def load_bundle(name):
    """'spot' = the real classroom tour; 'school_run1'/'school_run2' =
    the real school sessions (ring-33 1024-beam slices, stride-4
    keyframes). run1 has NO reference (est-only: gt_ok all-False, ghost
    hidden); run2 carries the gated-LIO window (342/836 kf, display/eval
    only there)."""
    if name in CAPTURES:
        return capture_bundle(name)
    if name == "spot":
        b = SP.make_bundle()
        b["name"] = "spot"
        return _attach_est(b)
    d = ROOT / "data" / "spot_telluride" / name
    z = np.load(d / "scans.npz")
    ts = z["ts"]                                # materialize ONCE — npz
    ranges = z["ranges"]                        # members decompress fully
    idx = np.arange(0, len(ts), SP.STRIDE)      # on EVERY access
    ref = np.load(d / "ref_lio.npz")
    gt = np.asarray(ref["gt"], float)[:len(idx)]
    ok = np.asarray(ref["ok"], bool)[:len(idx)]
    if len(gt) < len(idx):                      # ref shorter than run
        pad = len(idx) - len(gt)
        gt = np.vstack([gt, np.zeros((pad, gt.shape[1]))])
        ok = np.concatenate([ok, np.zeros(pad, bool)])
    keys = [(ranges[i], gt[k][:3] if gt.shape[1] >= 3 else
             np.append(gt[k], 0.0), ts[i] / 1e9)
            for k, i in enumerate(idx)]
    return _attach_est(dict(
        name=name, kind="spot", keys=keys,
        beam=-np.pi + np.arange(1024) * (2 * np.pi / 1024),
        gt_ok=ok, kts=ts[idx] / 1e9,
        rmin=SP.R_MIN, rmax=SP.R_MAX))


def convex_hull(pts):
    """monotone chain; pts (n,2) -> hull vertices (m,2) CCW."""
    P = sorted(map(tuple, np.round(np.asarray(pts, float), 3)))
    if len(P) < 3:
        return np.asarray(P, float)
    def half(seq):
        h = []
        for q in seq:
            while len(h) > 1 and (
                (h[-1][0]-h[-2][0])*(q[1]-h[-2][1])
                - (h[-1][1]-h[-2][1])*(q[0]-h[-2][0])) <= 0:
                h.pop()
            h.append(q)
        return h
    lo, hi = half(P), half(P[::-1])
    return np.asarray(lo[:-1] + hi[:-1], float)


def hull_mask(gx, gy, hull):
    """(ny,nx) bool: grid cells inside the convex hull (CCW)."""
    XX, YY = np.meshgrid(gx, gy)
    m = np.ones(XX.shape, bool)
    n = len(hull)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        m &= ((b[0]-a[0])*(YY-a[1]) - (b[1]-a[1])*(XX-a[0])) >= -1e-9
    return m


# ------------------------------------------------------- camera VSA obj map
class CamMap:
    """The camera-side VSA OBJECT MAP (laptop-side, per the decode-on-
    laptop directive). Cell codes (the exported head's desc bits — honest
    post-demotion classes are APPEARANCE CLUSTERS; real class bits slot in
    when the bottleneck artifact lands) are BOUND at world positions
    lifted from the LIDAR: cell bearing (D455 HFOV, nominal yaw-aligned
    extrinsics — demo-grade, stated) -> ring-33 range at that bearing ->
    world point. Per keyframe ONE bounded D=240 vector holds its
    object-aggregated bindings (capacity law: cluster cells -> centroids,
    3-6 bindings/kf). Selecting a class UNBINDS its code from every kf
    vector and paints the response density on the map."""

    HFOV = np.deg2rad(69.0)          # D455 RGB horizontal FOV (nominal)
    HAM_T = 7                        # cluster threshold (of 32).
                                     # NOISE-TRACKED: 0.28 med flip at
                                     # 4 Hz kf wanted 11; at 10 Hz the
                                     # flip is 0.128 med (4/32) and the
                                     # capture3 sweep shows T=7 is the
                                     # only positive-margin setting
                                     # (intra 5 / inter-min 6); T>=9
                                     # over-merges. Rate < ~8 Hz -> 11.
    MAX_C = 12

    def __init__(self):
        from sspax import headio as HIO
        self.keys = HIO._head_keys(32, len(W_MAIN))   # (32, D) spatter
        self.cents = []              # cluster centroid bit-freqs (float)
        self.counts = []
        self.example = []            # (kf, cy, cx) per cluster
        self.vecs = {}               # kf -> (anchor_pose, D complex)
        self.cellpos = {}            # kf -> {(cy,cx): (wx,wy)}
        self.boost = {}              # class id -> bind-weight (targets>1)
        self.lock = threading.Lock()

    def _amp(self, bits):
        return ((1.0 - 2.0 * bits.astype(np.float64)) @ self.keys
                / np.sqrt(32.0))

    def _cluster(self, b):
        """leader clustering on hamming; -> cluster id."""
        for i, c in enumerate(self.cents):
            if np.abs((c > 0.5).astype(int)
                      - b.astype(int)).sum() <= self.HAM_T:
                n = self.counts[i]
                self.cents[i] = (c * n + b) / (n + 1)
                self.counts[i] += 1
                return i
        if len(self.cents) < self.MAX_C:
            self.cents.append(b.astype(np.float64))
            self.counts.append(1)
            self.example.append(None)
            return len(self.cents) - 1
        d = [np.abs((c > 0.5).astype(int) - b.astype(int)).sum()
             for c in self.cents]
        i = int(np.argmin(d))
        self.counts[i] += 1
        return i

    def ingest(self, k, bits, scan, pose):
        """bits (15,20,32); bind this keyframe's object aggregates."""
        beam_n = len(scan)
        self.cellpos.setdefault(int(k), {})
        cells = []                   # (cid, x, y, cy, cx)
        for cy in range(3, 13):      # content rows (skip ceiling/floor)
            for cx in range(20):
                brg = self.HFOV / 2 - (cx + 0.5) / 20.0 * self.HFOV
                bi = int(round((brg + np.pi) / (2 * np.pi) * beam_n)) \
                    % beam_n
                r = np.nanmedian(np.where(
                    np.isfinite(scan[max(0, bi - 2):bi + 3]),
                    scan[max(0, bi - 2):bi + 3], np.nan))
                if not np.isfinite(r) or r < 0.3 or r > 12.0:
                    continue
                a = pose[2] + brg
                wx, wy = (pose[0] + r * np.cos(a),
                          pose[1] + r * np.sin(a))
                self.cellpos[int(k)][(cy, cx)] = (wx, wy)
                cells.append((self._cluster(bits[cy, cx]), wx, wy, cy, cx))
        if not cells:
            return
        v = np.zeros(len(W_MAIN), complex)
        with self.lock:
            byc = {}
            for cid, x, y, cy, cx in cells:
                byc.setdefault(cid, []).append((x, y))
                if self.example[cid] is None:
                    self.example[cid] = (int(k), cy, cx)
            for cid, ps in byc.items():
                code = (self.cents[cid] > 0.5)
                w = self.boost.get(cid, 1.0)
                # INSTANCE SPLIT (tuned, tune_recall.py): same-class
                # observations >1.2 m apart bind SEPARATELY — a blind
                # centroid averages two instances into a phantom between
                # them (the same-class-multiplicity law, live).
                groups = []
                for p in ps:
                    for g in groups:
                        if (p[0] - g[0][0]) ** 2 + (p[1] - g[0][1]) ** 2 \
                                < 1.44:
                            g.append(p)
                            break
                    else:
                        groups.append([p])
                for g in groups:
                    p = np.mean(g, axis=0)
                    v += w * self._amp(code) * np.exp(
                        1j * ((p - pose[:2]) @ W_MAIN.T))
            self.vecs[int(k)] = (pose.copy(), v)

    def classes(self):
        with self.lock:
            return [dict(id=i, n=int(self.counts[i]),
                         ex=self.example[i])
                    for i in range(len(self.cents)) if self.counts[i] >= 8]

    # ---- calibrated matched-filter sweep (shared by all query forms) --
    # Defaults TUNED on synthetic recall (tune_recall.py, banked
    # 2026-07-16): MAX-fusion wins (sum integrates cross-talk and halves
    # recall — my sum prior was refuted by the sweep); instance-split +
    # target boost x2-3 give recall 1.0 on observed objects and boost
    # also buys precision. Support image kept for display only.
    FUSE = "max"

    def _sweep(self, qa, grid_lo, grid_hi, ngrid=96):
        """qa: conj query amplitude (D,) -> (density image, support
        image = #kf vectors responding at each cell) over the world
        grid. Fusion across kf vectors per FUSE ("sum" integrates
        re-observations — persistence; "max" = single best view)."""
        with self.lock:
            vecs = list(self.vecs.items())
        gx = np.linspace(grid_lo[0], grid_hi[0], ngrid)
        gy = np.linspace(grid_lo[1], grid_hi[1], ngrid)
        XX, YY = np.meshgrid(gx, gy)
        P = np.stack([XX.ravel(), YY.ravel()], 1)
        img = np.zeros(ngrid * ngrid, np.float64) if self.FUSE == "sum" \
            else np.full(ngrid * ngrid, -1e9, np.float64)
        sup = np.zeros(ngrid * ngrid, np.int32)
        touched = np.zeros(ngrid * ngrid, bool)
        for k, (pose, v) in vecs:
            d = P - pose[:2]
            m = (np.abs(d) < 8.0).all(1)
            if not m.any():
                continue
            unb = qa * v
            sc = (np.exp(-1j * (d[m] @ W_MAIN.T)) @ unb).real / len(W_MAIN)
            mi = np.flatnonzero(m)
            touched[mi] = True
            sup[mi] += sc > 0.5 * np.abs(sc).max()
            if self.FUSE == "sum":
                img[mi] += sc
            else:
                np.maximum.at(img, mi, sc)
        if self.FUSE == "sum":
            img[~touched] = -1e9
        return img.reshape(ngrid, ngrid), sup.reshape(ngrid, ngrid), gx, gy

    Z_TH = 4.0                       # 4-sigma discipline
    SUP_MIN = 1                      # support gating OFF by default —
                                     # the sweep showed it buys nothing
                                     # once fusion is max (tune_recall)

    def _zquery(self, qa, qa_ctrl, grid_lo, grid_hi, tag, ngrid=96,
                hull=None):
        """CALIBRATED query: z-score the density against a seeded
        RANDOM-CODE control sweep (the honest 'no match' answer — a
        garbage query yields no marks instead of a normalized peak).
        Marks additionally require SUP_MIN keyframes of support."""
        img, sup, gx, gy = self._sweep(qa, grid_lo, grid_hi, ngrid)
        ctl, _, _, _ = self._sweep(qa_ctrl, grid_lo, grid_hi, ngrid)
        val = ctl[ctl > -1e8]
        mu, sd = (float(np.median(val)), float(val.std() + 1e-9)) \
            if val.size else (0.0, 1.0)
        z = np.where(img > -1e8, (img - mu) / sd, 0.0)
        if hull is not None and len(hull) >= 3:
            z[~hull_mask(gx, gy, hull)] = 0.0
        marks = []
        ok = (z > self.Z_TH) & (sup >= self.SUP_MIN)
        cand = np.argwhere(ok)
        order = np.argsort(-z[ok]) if cand.size else []
        for yy, xx in cand[order][:32]:
            wx, wy = gx[xx], gy[yy]
            if all((wx - a) ** 2 + (wy - b) ** 2 > 0.36
                   for a, b, _ in marks):
                marks.append((float(wx), float(wy),
                              float(round(z[yy, xx], 1))))
        im = np.clip(z / 8.0, 0, 1)                  # display scale z=8
        return dict(x0=float(grid_lo[0]), y0=float(grid_lo[1]),
                    x1=float(grid_hi[0]), y1=float(grid_hi[1]),
                    png=base64.b64encode(          # flipud: client draws
                        np.flipud(im * 255)        # row 0 at MAX y
                        .astype(np.uint8).tobytes()).decode(),
                    n=ngrid, marks=marks[:12], cls=tag,
                    zmax=float(round(z.max(), 1)))

    def _rand_amp(self, seed=99):
        rng = np.random.default_rng(seed)
        return np.conj(self._amp(rng.random(32) > 0.5))

    def query(self, cid, grid_lo, grid_hi, ngrid=96):
        """class-chip query: UNBIND cluster cid's code, z-calibrated."""
        with self.lock:
            if cid >= len(self.cents):
                return None
            code = (self.cents[cid] > 0.5)
        return self._zquery(np.conj(self._amp(code)), self._rand_amp(),
                            grid_lo, grid_hi, int(cid), ngrid,
                            hull=getattr(self, "hull_ref", None))

    def query_patch(self, k, cells, grid_lo, grid_hi, bits_grid):
        """REGION query (objmap patch form): the dragged cells form a
        position-STRUCTURED multi-cell query q = sum_c A(bits_c) *
        exp(iW.(p_c - x0)) — matches object-shaped content, far sharper
        than a single centroid code. Control = same structure, random
        codes (calibrates away the spatial envelope)."""
        with self.lock:
            ps = self.cellpos.get(int(k), {})
        pts = [(ps[(cy, cx)], bits_grid[cy, cx])
               for cy, cx in cells if (cy, cx) in ps]
        if len(pts) < 2:
            return None
        x0 = np.mean([p for p, _ in pts], axis=0)
        rng = np.random.default_rng(7)
        q = np.zeros(len(W_MAIN), complex)
        qc = np.zeros(len(W_MAIN), complex)
        for (p, b) in pts:
            ph = np.exp(1j * ((np.asarray(p) - x0) @ W_MAIN.T))
            q += self._amp(b) * ph
            qc += self._amp(rng.random(32) > 0.5) * ph
        n = np.sqrt(len(pts))
        return self._zquery(np.conj(q / n), np.conj(qc / n),
                            grid_lo, grid_hi, "patch",
                            hull=getattr(self, "hull_ref", None))

    def whatis(self, wx, wy):
        """REVERSE readout: click the MAP -> decode the code bits at
        that spot (project the local readout onto the spatter keys) ->
        nearest cluster + bit confidence."""
        with self.lock:
            vecs = list(self.vecs.items())
            cents = [c > 0.5 for c in self.cents]
        r = np.zeros(len(W_MAIN), complex)
        nk = 0
        for k, (pose, v) in vecs:
            d = np.array([wx, wy]) - pose[:2]
            if np.abs(d).max() > 8.0:
                continue
            r += v * np.exp(-1j * (d @ W_MAIN.T))
            nk += 1
        if nk == 0 or not cents:
            return None
        c = (self.keys @ np.conj(r)).real          # per-bit correlation
        bits = c < 0                               # A = (1-2b): b=1 -> -key
        conf = float(np.abs(c).mean() / (np.abs(c).std() + 1e-9))
        ham = [int(np.sum(bits != cc)) for cc in cents]
        cid = int(np.argmin(ham))
        return dict(cls=cid, ham=int(ham[cid]), nk=nk,
                    conf=round(conf, 2))


# --------------------------------------------------------------- camera/QBE
class CaptureCam:
    """Replay cam lane: serves the desc-bit grids the LIVE cam worker
    computed (cbits C kernel) and the jpegs its bounded cache kept
    (last ~300 kf). Same .bits/.jpeg/.lock/.compute/.query surface as
    CamLane, so the worker and the query endpoints don't care."""

    def __init__(self, b):
        self.bits = b["cam_bits"]
        self.jpeg = b.get("cam_jpeg", {})
        self.lock = threading.Lock()

    def compute(self, k):
        return self.bits.get(int(k))

    def query(self, k, cy, cx):
        with self.lock:
            if int(k) not in self.bits:
                return {}
            q = self.bits[int(k)][cy, cx]
            return {int(kk): 1.0 - float((bb != q).sum(-1).min()) / 32.0
                    for kk, bb in self.bits.items()}


class CamLane:
    """Aligned D455 frames -> exported vision head desc bits (laptop).
    The appearance map: per-keyframe (15, 20, 32) bit grids anchored at
    estimated poses; QBE = hamming match of one clicked cell's bits."""

    def __init__(self, run, kts_s):
        from sspax import headio as HIO
        import runners.spot_cam as SC
        import golden_cam as GC
        self.HIO, self.GC = HIO, GC
        self.head = HIO.load_head(str(HEAD_NPZ))
        self.cb = None
        try:
            from cbits import CBits
            self.cb = CBits(str(HEAD_NPZ))
        except Exception as e:
            print(f"[webvis] cbits unavailable ({e}); numpy bits",
                  flush=True)
        self.shards, self.cts, self.where = SC._index(run)
        kns = (np.asarray(kts_s) * 1e9).astype(np.int64)
        j = np.clip(np.searchsorted(self.cts, kns), 1, len(self.cts) - 1)
        j = j - (np.abs(self.cts[j - 1] - kns) < np.abs(self.cts[j] - kns))
        self.kf2cam = j
        self._tbl = {}                    # shard idx -> pyarrow table
        self.bits = {}                    # k -> (15,20,32) bool
        self.jpeg = {}                    # k -> bytes (bounded cache)
        self.lock = threading.Lock()

    def _gray(self, k):
        from PIL import Image
        import io as _io
        import pyarrow.parquet as pq
        si, row = self.where[self.kf2cam[k]]
        if si not in self._tbl:
            self._tbl = {si: pq.read_table(self.shards[si])}  # keep one
        jb = self._tbl[si]["image"][row].as_py()["bytes"]
        im = np.asarray(Image.open(_io.BytesIO(jb)).convert("RGB"))
        g = self.GC.bin2(self.GC.rgb_to_gray(im))   # deploy-faithful path
        return g, jb

    def compute(self, k):
        try:
            g, jb = self._gray(k)
        except Exception:
            return None
        try:
            b = self.cb.bits(g) if self.cb else None
        except Exception:
            b = None                       # e.g. non-QVGA gray shape
        if b is None:
            b = self.HIO.cell_bits(self.head, g, source="desc")
        with self.lock:
            self.bits[k] = b
            self.jpeg[k] = jb
            if len(self.jpeg) > 40:       # bound the jpeg cache
                self.jpeg.pop(next(iter(self.jpeg)))
        return b

    def query(self, k, cy, cx):
        """QBE: bits of cell (cy,cx) in frame k vs every stored grid.
        -> {kf: score 0..1} (1 = identical bits somewhere in that kf)."""
        with self.lock:
            if k not in self.bits:
                return {}
            q = self.bits[k][cy, cx]
            out = {}
            for kk, b in self.bits.items():
                d = (b != q).sum(-1).min()
                out[int(kk)] = 1.0 - float(d) / 32.0
        return out


# ----------------------------------------------- offline tracker + estcache
# make_slam is OFFLINE-ONLY since the 2026-07-16 shim directive: the
# serving path runs NO python SLAM. It is (a) the estcache builder's
# engine and (b) the configuration home for the future ON-CHIP tracker —
# the chip runs this same algebra.
CV_CAP = 0.30          # cv-guess translation cap per kf (m). The
                       # hunter pair-cos sweep REFUTED the uncapped
                       # prior: cap 0.60 arms are knife-edged (one
                       # diverges outright); cap 0.30 + the matcher
                       # window covering the residual is uniformly
                       # stable. scratch_hunter_retune 2026-07-16.


def make_slam(nph=0):
    # nph=4 = the QUANTIZED recipe: the chip's 2-bit QPSK map store
    # (measured lossless at rate on the hunter corpus: pair-cos
    # 0.956/0.892 vs float 0.957/0.894, map memory 1058 -> 68 KB).
    slam = F.BandSLAM(robust=True, attempt_every=4, relax_every=25,
                      gap_kf=300, recent_aids=12, spec=None, nph=nph)
    slam.store_dtype = np.complex64
    # HUNTER RETUNE 2 (capture_1784219440 929 kf @ 4 Hz, pair-cos
    # evaluator — re-encode scan k at the estimated delta vs scan k-1):
    # t 0.72 / rot 9 / cap 0.30 sits centered on the robust plateau
    # (cos med 0.940 p10 0.766 vs the smoothness-tuned t.72/rot18/
    # cap.60's p10 0.173 — smoothness alone was the wrong objective).
    # Stress arm: at 2 Hz keyframes EVERY window collapses -> keep
    # live keyframes >= 4 Hz; the earlier live divergence was rate
    # stalls, not the window. Acceptance recipes in runners/ are
    # UNTOUCHED — this is the offline/chip-tracker config.
    slam.matcher = S.Matcher(L.ENC_MAIN, t_half=0.72, rot_half_deg=9.0,
                             rot_step_deg=1.5, perm=(4, L.N_ANG))
    return slam


def cv_guess(est, k, cap=CV_CAP):
    """Constant-velocity pose guess from est[:k] (the replay/tracker
    prior — lived in the old webvis step loop, now offline-only)."""
    if k == 0:
        return None
    if k == 1:
        return est[0].copy()
    v = est[k - 1] - est[k - 2]
    vn = np.hypot(v[0], v[1])
    if vn > cap:
        v[:2] *= cap / vn
    return np.array([est[k - 1][0] + v[0], est[k - 1][1] + v[1],
                     est[k - 1][2] + S.wrap(v[2])])


def build_est_cache(names=None):
    """Compute each dataset's pose-estimate trajectory ONCE with the
    offline tracker (make_slam — the chip-tracker algebra) and store it
    in est_demo.npz for webvis to REPLAY. Rerun after any tracker
    retune. Prints per-run smoothness + err vs the recorded reference
    where one exists (display-honesty numbers, not acceptance)."""
    out = {}
    if EST_DEMO.exists():
        z = np.load(EST_DEMO)
        out = {k: np.asarray(z[k]) for k in z.files}
    for name, suffix, nph in [(n_, s_, q_) for n_ in (names or DATASETS)
                              for s_, q_ in (("", 0), ("__q", 4))]:
        b = load_bundle(name)
        b.pop("est", None)                     # never seed from a cache
        b.pop("est_q", None)
        keys, beam = b["keys"], b["beam"]
        n = len(keys)
        slam = make_slam(nph=nph)
        est = np.zeros((n, 3))
        t0 = time.time()
        for k in range(n):
            rr = DS.clean(b, np.asarray(keys[k][0], float))
            pts, w = F.points_from_scan(rr, beam)
            g = cv_guess(est, k)
            if g is None:
                g = np.asarray(keys[k][1], float).copy()   # start anchor
            e = slam.add_keyframe(pts, w, g)
            if slam.dirty:
                slam.relax()
                e = slam.pose_of(k)
            est[k] = e
        dy = np.abs(np.diff(est[:, 2]))
        dy = np.rad2deg(np.minimum(dy, 2 * np.pi - dy))
        gt_ok = np.asarray(b.get("gt_ok", np.zeros(n, bool)))[:n]
        line = (f"  {name}{suffix}: {n} kf in {time.time()-t0:.0f}s | "
                f"|dyaw|/kf med {np.median(dy):.2f} p90 "
                f"{np.percentile(dy, 90):.1f} deg")
        if gt_ok.any():
            err = [np.linalg.norm(est[k][:2]
                                  - np.asarray(keys[k][1])[:2])
                   for k in range(n) if gt_ok[k]]
            line += (f" | err vs ref med {np.median(err):.3f} "
                     f"p90 {np.percentile(err, 90):.3f} m "
                     f"({len(err)} kf)")
        print(line, flush=True)
        out[name + suffix] = est
    np.savez_compressed(EST_DEMO, **out)
    print(f"wrote {EST_DEMO} ({', '.join(sorted(out))})")


# ------------------------------------------------------------------- server


class Demo:
    def __init__(self, data="spot", use_fpga=True, cam=True,
                 port=PORT_SER):
        self.port = port
        self.use_fpga = use_fpga
        self.want_cam = cam
        self.fpga = None
        if use_fpga:
            try:
                self.fpga = Fpga(port)
            except Exception as e:
                print(f"[webvis] FPGA lane OFF (board unplugged?): {e}",
                      flush=True)
        self.clients = []
        self.lock = threading.Lock()
        self._req_reset = False
        self._req_data = None
        self._qbe = None                  # latest {kf: score}
        self._thumbs = {}
        self.camq = queue.Queue(maxsize=8)
        self._load(data)
        threading.Thread(target=self._cam_worker, daemon=True).start()

    def _load(self, data):
        print(f"[webvis] loading '{data}'...", flush=True)
        self.data = data
        self.b = load_bundle(data)
        self.keys, self.beam = self.b["keys"], self.b["beam"]
        self.n = len(self.keys)
        self.gt_ok = np.asarray(self.b.get("gt_ok", np.ones(self.n, bool)))
        rq = getattr(self, "recipe", "float") == "quantized"
        self.pose_src = ("odom (replay)" if self.b.get("kind") == "capture"
                         else "est-cache (QUANTIZED 2b)"
                         if rq and self.b.get("est_q") is not None
                         else "est-cache" if self.b.get("est") is not None
                         else "ref")
        self.cam = None
        self.cammap = None
        if self.want_cam:
            try:
                self.cam = CaptureCam(self.b) if self.b.get("cam_bits") \
                    else CamLane(data, np.asarray(self.b["kts"]))
                self.cammap = CamMap()
            except Exception as e:
                print(f"[webvis] cam lane off: {e}", flush=True)
        self.est = np.zeros((self.n, 3))
        self.k = 0
        self.done = False
        self.pts, self.trail_py, self.trail_gt = [], [], []
        self.chip_segs = {}               # slot -> (pose, codes)
        self.chip_img = None
        self.hull = None                  # convex hull of lidar points
        self._chip_cursor = 0             # batched sequential decode
        self.ms_slam = self.ms_fpga = 0.0
        self.overruns = 0
        if self.fpga:
            self.fpga.tx.ctrl(0)
            self.fpga.dig_ok = self.fpga.vec_ok = self.fpga.total = 0
            self.fpga.bytes = 0
            self.fpga.t0 = time.time()
        self.broadcast(dict(reset=True, data=data,
                            cam=bool(self.cam)))

    # ---- chip-map image: decode fetched codes at estimated poses -------
    DECODE_R = float(os.environ.get("SSP_DECODE_R", "6.0"))

    def _chip_image(self, batch=16):
        """Decode the CHIP map over the CONVEX HULL of all lidar points
        (not a bbox square) — outside-hull cells are never decoded. The
        per-segment decode is BATCHED (`batch` segments per call, cursor
        cycling) so laptops spread the cost across keyframes.

        REGION FIXES (user report: decoded map in the wrong region):
        the 3 m per-segment decode box hugged the trajectory and missed
        the walls the scans actually encode (fine in the classroom,
        wrong at the school/Hunter venue) -> DECODE_R, default 12 m;
        and the overlay rows were VERTICALLY MIRRORED vs the point map
        (grid row 0 = min-y but the client draws row 0 at max-y) ->
        flipud before shipping. A corr-vs-points readout ships with the
        image so the alignment is a NUMBER, not an impression."""
        self.hull = getattr(self, 'hull', None)
        self._chip_cursor = getattr(self, '_chip_cursor', 0)
        if not self.chip_segs or self.hull is None or len(self.hull) < 3:
            return None
        lo = self.hull.min(0) - 1.0
        hi = self.hull.max(0) + 1.0
        ngrid = 96
        gx = np.linspace(lo[0], hi[0], ngrid)
        gy = np.linspace(lo[1], hi[1], ngrid)
        mask = hull_mask(gx, gy, self.hull)
        img = np.zeros((ngrid, ngrid), np.float32)
        XX, YY = np.meshgrid(gx, gy)
        P = np.stack([XX.ravel(), YY.ravel()], 1)
        items = list(self.chip_segs.items())
        if len(items) > batch:             # sequential window
            s = self._chip_cursor % len(items)
            items = (items + items)[s:s + batch]
            self._chip_cursor += batch
        for slot, (pose, codes) in items:
            if pose is None:
                continue
            vec = np.exp(1j * np.pi / 2 * codes.astype(np.float32))
            d = P - pose[:2]
            m = (np.abs(d) < self.DECODE_R).all(1)
            if not m.any():
                continue
            c, s = np.cos(-pose[2]), np.sin(-pose[2])
            ds = d[m] @ np.array([[c, -s], [s, c]]).T
            sc = (np.exp(-1j * (ds @ W_MAIN.T)) @ vec).real / 240.0
            img.ravel()[np.flatnonzero(m)] = np.maximum(
                img.ravel()[np.flatnonzero(m)], sc)
        img = np.clip(img / max(img.max(), 1e-9), 0, 1)
        img[~mask] = 0.0                   # decode only inside the hull
        # alignment metric: Pearson corr of decode vs point occupancy
        corr = None
        with self.lock:
            pa = np.asarray(self.pts[-20000:], float) \
                if self.pts else None
        if pa is not None and len(pa) > 100:
            occ, _, _ = np.histogram2d(
                pa[:, 1], pa[:, 0], bins=ngrid,
                range=[[lo[1], hi[1]], [lo[0], hi[0]]])
            occ = np.minimum(occ, 5)
            m2 = mask & (img > 0)
            if m2.sum() > 50:
                a_, b_ = img[m2], occ[m2]
                corr = float(np.corrcoef(a_, b_)[0, 1])
                self.chip_corr = round(corr, 3)
        return dict(x0=float(lo[0]), y0=float(lo[1]),
                    x1=float(hi[0]), y1=float(hi[1]),
                    corr=corr,
                    png=base64.b64encode(
                        np.flipud(img * 255).astype(np.uint8)
                        .tobytes()).decode(),
                    n=ngrid)

    def _cam_track(self, k, bits):
        """camera shift-tracking: best horizontal cell shift vs the
        previous frame -> yaw rate estimate + agreement quality; compared
        against the lidar yaw over the same interval = DIVERGENCE."""
        prev = getattr(self, "_cam_prev", None)
        self._cam_prev = (k, bits)
        if prev is None or k <= prev[0]:
            return
        pk, pb = prev
        best_s, best_a = 0, 0.0
        for s_ in range(-3, 4):
            if s_ >= 0:
                a_ = (bits[:, s_:20] == pb[:, :20 - s_]).mean()
            else:
                a_ = (bits[:, :20 + s_] == pb[:, -s_:]).mean()
            if a_ > best_a:
                best_a, best_s = float(a_), s_
        fov = getattr(self.cammap, "HFOV", np.deg2rad(69.0)) \
            if self.cammap else np.deg2rad(69.0)
        dtk = max(k - pk, 1) * KF_DT
        yaw_cam = -best_s * (fov / 20.0) / dtk
        yaw_lid = float(S.wrap(self.est[k][2] - self.est[pk][2])) / dtk
        d = abs(float(S.wrap(yaw_cam - yaw_lid)))
        a = 0.25
        pv = getattr(self, "trk_cam", (best_a, 0.0, 0.0, 0.0))
        self.trk_cam = (a * best_a + (1 - a) * pv[0], yaw_cam, yaw_lid,
                        a * d + (1 - a) * pv[3])

    # ---- camera worker (desc bits + VSA obj-map bind, off the hot loop)
    def _cam_worker(self):
        while True:
            self._cam_step(self.camq.get())

    def _cam_step(self, k):
        from PIL import Image
        import io as _io
        if True:
            if self.cam is None or k >= len(self.keys):
                return
            b = self.cam.compute(k)
            if b is None:
                return
            self._cam_track(k, b)
            if self.cammap is not None:
                self.cammap.ingest(k, b,
                                   np.asarray(self.keys[k][0], float),
                                   self.est[k])
                # thumbnails for freshly-exampled clusters
                with self.cam.lock:
                    jb = self.cam.jpeg.get(k)
                cls = self.cammap.classes()
                if jb:
                    im = None
                    for c in cls:
                        ex = c["ex"]
                        if (ex and ex[0] == k
                                and c["id"] not in self._thumbs):
                            if im is None:
                                im = Image.open(_io.BytesIO(jb)) \
                                    .convert("L").resize((320, 240))
                            cy, cx = ex[1], ex[2]
                            crop = im.crop((max(0, cx * 16 - 8),
                                            max(0, cy * 16 - 8),
                                            min(320, cx * 16 + 24),
                                            min(240, cy * 16 + 24))) \
                                .resize((40, 40))
                            buf = _io.BytesIO()
                            crop.save(buf, "JPEG", quality=70)
                            self._thumbs[c["id"]] = base64.b64encode(
                                buf.getvalue()).decode()
                self.broadcast(dict(classes=[
                    dict(id=c["id"], n=c["n"],
                         thumb=self._thumbs.get(c["id"]),
                         boost=c["id"] in self.cammap.boost)
                    for c in cls]))
            with self.cam.lock:
                jb = self.cam.jpeg.get(k)
            if jb is None:
                # replay kf outside the stored-jpeg window: render the
                # DESC-BIT grid instead (panel stays live + clickable —
                # cell layout matches the QBE grid exactly)
                pc = (np.asarray(b).sum(-1)
                      * (255 // b.shape[-1])).astype(np.uint8)
                im = Image.fromarray(np.kron(
                    pc, np.ones((16, 16), np.uint8)))
                buf = _io.BytesIO()
                im.save(buf, "JPEG", quality=70)
                jb = buf.getvalue()
            self.broadcast(dict(cam_kf=int(k),
                                jpg=base64.b64encode(jb).decode()))

    recipe = "float"          # dataset replay: "float" | "quantized"

    def pose_for(self, k, r, gtp):
        """Pose estimate for keyframe k (the ONLY pose authority in the
        serving path). Base: bundle est cache (float or QUANTIZED chip-
        precision recipe per self.recipe) / recorded pose."""
        est_ref = (self.b.get("est_q") if self.recipe == "quantized"
                   else None)
        if est_ref is None:
            est_ref = self.b.get("est")
        return np.asarray(est_ref[k] if est_ref is not None else gtp,
                          float).copy()

    # ---- one keyframe ---------------------------------------------------
    def step(self):
        k = self.k
        r, gtp = self.keys[k][0], self.keys[k][1]
        ts = self.keys[k][2] if len(self.keys[k]) > 2 else 0.0
        dig = vec = False
        if self.fpga:
            dig, vec, self.ms_fpga = self.fpga.verify_encode(
                k, np.asarray(r, float), int(ts * 1e6))
        t0 = time.time()
        rr = DS.clean(self.b, np.asarray(r, float))
        pts, w = F.points_from_scan(rr, self.beam)
        # NO python SLAM (shim directive): points register at the POSE
        # ESTIMATE via the pose_for hook — datasets: offline est_demo
        # trajectory else recorded reference; live: datagram odom;
        # SOLO deploy: the ON-CHIP tracker (webvis_live.SoloDemo).
        e = self.pose_for(k, r, gtp)
        self.est[k] = e
        # chip-map readout: ONE slot every 4 kf, stamped with THIS
        # frame's pose (the chip wrote frame k to slot k%64 during
        # streaming, so the fetch pairs codes+pose exactly; over 64
        # frames 16 slots stay live = the rolling chip-map window)
        if self.fpga and k % 4 == 0:
            codes = self.fpga.map_seg(k % 64)
            if codes is not None:
                self.chip_segs[k % 64] = (e.copy(), codes)
        if self.cam and k % 2 == 0:
            try:
                self.camq.put_nowait(k)
            except queue.Full:
                pass

        c, s = np.cos(e[2]), np.sin(e[2])
        wpts = pts[::6] @ np.array([[c, -s], [s, c]]).T + e[:2]
        gt_ok = bool(self.gt_ok[k])
        if len(wpts):                      # grow-only hull, cheap update
            base = wpts if getattr(self, 'hull', None) is None else np.vstack(
                [self.hull, wpts])
            self.hull = convex_hull(base)
        with self.lock:
            self.pts.extend(np.round(wpts, 3).tolist())
            if len(self.pts) > MAX_PTS:
                self.pts = self.pts[-MAX_PTS:]
            self.trail_py.append([round(float(e[0]), 3),
                                  round(float(e[1]), 3)])
            self.trail_gt.append([round(float(gtp[0]), 3),
                                  round(float(gtp[1]), 3), gt_ok])
        err = (float(np.linalg.norm(e[:2] - np.asarray(gtp)[:2]))
               if gt_ok else None)
        self.ms_slam = (time.time() - t0) * 1e3      # shim python cost
        st = dict(k=k, n=self.n, data=self.data,
                  dig=bool(dig), vec=bool(vec),
                  fx_dig=self.fpga.dig_ok if self.fpga else 0,
                  fx_vec=self.fpga.vec_ok if self.fpga else 0,
                  fx_tot=self.fpga.total if self.fpga else 0,
                  kbps=round(self.fpga.kbps(), 1) if self.fpga else 0,
                  ms_fx=round(self.ms_fpga, 1),
                  ms_py=round(self.ms_slam, 1),
                  pose_src=getattr(self, "pose_src", "ref"),
                  est=[round(float(x), 3) for x in e],
                  gt=[round(float(x), 3) for x in gtp[:3]], gt_ok=gt_ok,
                  err=None if err is None else round(err, 3),
                  chip_segs=len(self.chip_segs),
                  overruns=self.overruns,
                  trk_cam=[round(v, 3) for v in
                           getattr(self, "trk_cam", (0, 0, 0, 0))],
                  wpts=np.round(wpts, 3).tolist())
        self.k += 1
        if self.k >= self.n:
            self.done = True
            st["done"] = True
        self.broadcast(st)
        if self.fpga and k and k % 40 == 0:
            img = self._chip_image()
            if img:
                self.broadcast(dict(chipmap=img))

    # ---- SSE ------------------------------------------------------------
    def broadcast(self, obj):
        dead = []
        for q_ in self.clients:
            try:
                q_.put_nowait(obj)
            except queue.Full:
                dead.append(q_)
        for q_ in dead:
            self.clients.remove(q_)

    live_capable = False        # LiveDemo overrides: "live" selectable

    def snapshot(self):
        with self.lock:
            return dict(init=True, n=self.n, k=self.k, data=self.data,
                        cam=bool(self.cam), pts=self.pts,
                        datasets=(["live"] if self.live_capable else [])
                        + avail_datasets(),
                        trail_py=self.trail_py, trail_gt=self.trail_gt)

    def run(self, realtime=True, stop_after=None, loop=False):
        while True:
            while not self.done:
                if self._req_data:
                    d, self._req_data = self._req_data, None
                    self._load(d)
                if self._req_reset:
                    self._req_reset = False
                    self._load(self.data)
                t0 = time.time()
                self.step()
                if stop_after and self.k >= stop_after:
                    break
                if realtime:
                    kdt = self.b.get("kdt", KF_DT)   # capture replays
                    dt = time.time() - t0            # at recorded rate
                    if dt > kdt:
                        self.overruns += 1
                    else:
                        time.sleep(kdt - dt)
            print(f"[webvis] '{self.data}' "
                  f"{'complete' if self.done else 'stopped'} at kf {self.k}",
                  flush=True)
            if stop_after or not loop:
                break
            time.sleep(5.0)
            self._load(self.data)


# ----------------------------------------------------------------- HTTP/SSE
def make_handler(demo):
    html = (HERE / "webvis.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path == "/reset":
                demo._req_reset = True
                self._json(dict(ok=True))
            elif self.path.startswith("/select"):
                d = self.path.split("=")[-1]
                if d in DATASETS or d in CAPTURES or (
                        d == "live"
                        and getattr(demo, "live_capable", False)):
                    demo._req_data = d
                    demo.done = False
                self._json(dict(ok=True, data=d))
            elif self.path.startswith("/query_patch"):
                ln = int(self.headers.get("Content-Length", 0))
                q = json.loads(self.rfile.read(ln) or b"{}")
                res = None
                if demo.cammap and demo.cam and demo.trail_py:
                    with demo.cam.lock:
                        bg = demo.cam.bits.get(int(q["k"]))
                    if bg is not None:
                        h = demo.hull
                        tr = np.array(demo.trail_py)
                        lo = (h.min(0) - 1.0) if h is not None \
                            and len(h) >= 3 else tr.min(0) - 3.0
                        hi = (h.max(0) + 1.0) if h is not None \
                            and len(h) >= 3 else tr.max(0) + 3.0
                        demo.cammap.hull_ref = h
                        res = demo.cammap.query_patch(
                            int(q["k"]),
                            [tuple(c) for c in q["cells"]], lo, hi, bg)
                    if res:
                        demo.broadcast(dict(objmap=res))
                self._json(dict(ok=res is not None))
            elif self.path.startswith("/recipe"):
                rr = self.path.split("=")[-1]
                if rr in ("float", "quantized"):
                    demo.recipe = rr
                    demo._req_reset = True
                self._json(dict(ok=True, recipe=rr))
            elif self.path.startswith("/whatis"):
                ln = int(self.headers.get("Content-Length", 0))
                q = json.loads(self.rfile.read(ln) or b"{}")
                res = demo.cammap.whatis(float(q["x"]), float(q["y"])) \
                    if demo.cammap else None
                if res:
                    demo.broadcast(dict(whatis=res, at=[q["x"], q["y"]]))
                self._json(res or dict(ok=False))
            elif self.path.startswith("/boost"):
                ln = int(self.headers.get("Content-Length", 0))
                q = json.loads(self.rfile.read(ln) or b"{}")
                if demo.cammap is not None:
                    cid = int(q["id"])
                    with demo.cammap.lock:
                        if cid in demo.cammap.boost:
                            del demo.cammap.boost[cid]
                        else:
                            demo.cammap.boost[cid] = 3.0
                        bl = sorted(demo.cammap.boost)
                    self._json(dict(ok=True, boosted=bl))
                else:
                    self._json(dict(ok=False), 400)
            elif self.path.startswith("/query_class"):
                ln = int(self.headers.get("Content-Length", 0))
                q = json.loads(self.rfile.read(ln) or b"{}")
                if demo.cammap and demo.trail_py:
                    h = demo.hull
                    tr = np.array(demo.trail_py)
                    lo = (h.min(0) - 1.0) if h is not None and len(h) >= 3 \
                        else tr.min(0) - 3.0
                    hi = (h.max(0) + 1.0) if h is not None and len(h) >= 3 \
                        else tr.max(0) + 3.0
                    demo.cammap.hull_ref = h
                    res = demo.cammap.query(int(q["id"]), lo, hi)
                    if res:
                        demo.broadcast(dict(objmap=res))
                    self._json(dict(ok=res is not None))
                else:
                    self._json(dict(ok=False), 400)
            elif self.path.startswith("/query"):
                ln = int(self.headers.get("Content-Length", 0))
                q = json.loads(self.rfile.read(ln) or b"{}")
                if demo.cam:
                    sc = demo.cam.query(int(q["k"]), int(q["cy"]),
                                        int(q["cx"]))
                    demo._qbe = sc
                    top = sorted(sc.items(), key=lambda kv: -kv[1])[:6]
                    demo.broadcast(dict(qbe=sc, qbe_top=top,
                                        qcell=[q["k"], q["cy"], q["cx"]]))
                    self._json(dict(ok=True, n=len(sc)))
                else:
                    self._json(dict(ok=False), 400)
            else:
                self.send_response(404)
                self.end_headers()

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html)
            elif self.path == "/capture":
                # dump the session so far (scans/poses/cam bits+jpegs)
                # for offline retuning. GT-free; est poses only.
                import time as _t
                out = str(Path.home() / f"capture_{int(_t.time())}.npz")
                with demo.lock:
                    n = demo.k
                    mm = np.stack([np.where(
                        np.isfinite(np.asarray(demo.keys[i][0], float)),
                        np.clip(np.asarray(demo.keys[i][0], float) * 1e3,
                                0, 65535), 0).astype(np.uint16)
                        for i in range(n)]) if n else np.zeros((0, 1024),
                                                               np.uint16)
                    ts = np.array([demo.keys[i][2] for i in range(n)])
                d = dict(mm=mm, ts=ts, est=demo.est[:n].copy())
                if demo.cam:
                    with demo.cam.lock:
                        ks = sorted(demo.cam.bits)
                        d["cam_kf"] = np.array(ks, np.int64)
                        d["cam_bits"] = np.stack(
                            [demo.cam.bits[i] for i in ks]) if ks else \
                            np.zeros((0, 15, 20, 32), bool)
                        d["jpeg_kf"] = np.array(
                            sorted(demo.cam.jpeg), np.int64)
                        d["jpegs"] = np.array(
                            [demo.cam.jpeg[i] for i in
                             sorted(demo.cam.jpeg)], object)
                np.savez_compressed(out, **{k: v for k, v in d.items()
                                            if not isinstance(v, dict)},
                                    allow_pickle=True)
                self._json(dict(ok=True, path=out, kf=int(n),
                                cam=len(d.get("cam_kf", []))))
            elif self.path == "/chipsegs":
                # raw fetched chip codes (hw-in-loop parity checks)
                with demo.lock:
                    segs = {str(s): dict(
                        pose=[float(x) for x in p] if p is not None
                        else None, codes=[int(c) for c in cd])
                        for s, (p, cd) in demo.chip_segs.items()}
                self._json(dict(k=demo.k, segs=segs))
            elif self.path == "/status":
                self._json(dict(
                    k=demo.k, n=demo.n, data=demo.data, done=demo.done,
                    fx_dig=demo.fpga.dig_ok if demo.fpga else 0,
                    fx_vec=demo.fpga.vec_ok if demo.fpga else 0,
                    fx_tot=demo.fpga.total if demo.fpga else 0,
                    chip_segs=len(demo.chip_segs),
                    chip_corr=getattr(demo, "chip_corr", None),
                    cam_kf=len(demo.cam.bits) if demo.cam else 0,
                    pose_src=getattr(demo, "pose_src", "ref"),
                    ms_py=round(demo.ms_slam, 1),
                    ms_fx=round(demo.ms_fpga, 1),
                    overruns=demo.overruns))
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                q_ = queue.Queue(maxsize=64)
                demo.clients.append(q_)
                try:
                    self.wfile.write(
                        f"data: {json.dumps(demo.snapshot())}\n\n".encode())
                    self.wfile.flush()
                    while True:
                        obj = q_.get(timeout=30)
                        self.wfile.write(
                            f"data: {json.dumps(obj)}\n\n".encode())
                        self.wfile.flush()
                except (queue.Empty, BrokenPipeError,
                        ConnectionResetError):
                    pass
                finally:
                    if q_ in demo.clients:
                        demo.clients.remove(q_)
            else:
                self.send_response(404)
                self.end_headers()

    return H


def serve(port=8790, use_fpga=True):
    import os
    # SSP_DATA: spot | school_run1 | school_run2 (needs the dataset dirs)
    demo = Demo(data=os.environ.get("SSP_DATA", "spot"),
                use_fpga=use_fpga)
    threading.Thread(target=demo.run, kwargs=dict(loop=True),
                     daemon=True).start()
    bind = os.environ.get("SSP_BIND", "127.0.0.1")  # 0.0.0.0 = LAN-visible
    srv = ThreadingHTTPServer((bind, port), make_handler(demo))
    print(f"[webvis] http://{bind}:{port}  (FPGA "
          f"{'IN THE LOOP' if use_fpga else 'OFF'})", flush=True)
    srv.serve_forever()


def selftest(n=40, use_fpga=True):
    demo = Demo(use_fpga=use_fpga)
    demo.run(realtime=False, stop_after=n)
    errs = [np.linalg.norm(demo.est[k][:2]
                           - np.asarray(demo.keys[k][1])[:2])
            for k in range(n) if demo.gt_ok[k]]
    time.sleep(3.0)                        # let the cam worker drain
    ncam = len(demo.cam.bits) if demo.cam else 0
    line = f"selftest: {n} kf | "
    if demo.fpga:
        f = demo.fpga
        line += (f"digests {f.dig_ok}/{f.total} | on-chip vectors "
                 f"{f.vec_ok}/{f.total} bit-exact | chip map "
                 f"{len(demo.chip_segs)} segs | ")
    else:
        line += "FPGA lane off | "
    line += (f"cam grids {ncam} | pose {demo.pose_src}"
             + (f" | med err vs ref {np.median(errs):.3f} m"
                if errs else ""))
    print(line)
    if demo.cammap:
        cls = demo.cammap.classes()
        nb = len(demo.cammap.vecs)
        tr = np.array(demo.trail_py)
        res = demo.cammap.query(cls[0]["id"], tr.min(0) - 3,
                                tr.max(0) + 3) if cls else None
        print(f"  cam VSA map: {nb} kf vectors | {len(cls)} appearance "
              f"classes (n>=8) | class-{cls[0]['id'] if cls else '-'} "
              f"unbind -> {len(res['marks']) if res else 0} object "
              f"marks (peak {max(m[2] for m in res['marks']):.2f})"
              if res else "  cam VSA map: query returned nothing")
        assert cls and res and res["marks"], "cam obj-map query failed"
    if demo.fpga:
        assert demo.fpga.dig_ok == demo.fpga.vec_ok == demo.fpga.total == n
        assert demo.chip_segs
    print("WEBVIS SELFTEST PASS "
          f"({'full' if demo.fpga else 'laptop-lanes'} mode)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "selftest":
        selftest(int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    elif cmd == "estcache":
        build_est_cache([sys.argv[2]] if len(sys.argv) > 2 else None)
    elif cmd == "serve":
        serve(int(sys.argv[2]) if len(sys.argv) > 2 else 8790,
              use_fpga=not (len(sys.argv) > 3 and sys.argv[3] == "nofpga"))
