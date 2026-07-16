#!/usr/bin/env python3
"""ECP5 FPGA-in-the-loop WEB VISUALIZATION v2 (STREAM.md v1.1).

The plugged Icepi Zero runs top_stream_enc: every keyframe's scan streams
through the board, is digest-verified, and is ENCODED ON CHIP — the VSA
vector returns (0x92) and is golden-crosschecked bit-exact on the laptop;
its 2-bit QPSK codes land in the ON-CHIP MAP BANK, whose compressed
segments the laptop fetches (0x0F -> 0x93) and DECODES into the chip-map
image. SLAM (shipped run_cv recipe) + all decode run laptop-side.

Lanes:
  LIDAR : real scans -> FPGA (digest + on-chip encode + on-chip map) ->
          shipped BandSLAM (python) -> map/trails; chip-map image decoded
          from the FETCHED compressed codes at estimated poses.
  CAMERA: aligned D455 frames (spot dataset) -> exported vision head's
          DESC BITS (headio, numpy, laptop) -> per-keyframe appearance
          grids anchored on the trajectory. QUERY-BY-EXAMPLE: click a
          cell in the camera panel; hamming-match against every stored
          grid; matches highlight along the trail. (Post-demotion the
          CNN's honest map query IS appearance QBE — no class labels.)

UI: reset button, dataset selection (spot real / classroom+school
synthetic), lidar map (points + chip-map layer), camera panel + QBE.
GT/withheld reference is a display-only ghost (anti-oracle).

  python3 hw/ecp5/host/webvis.py selftest        # headless, numbers only
  python3 hw/ecp5/host/webvis.py serve [port]    # default 8790
"""
import base64
import json
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
import sspslam.worlds_dyn as DE                                # noqa: E402

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
def load_bundle(name):
    """'spot' = the real classroom tour; 'classroom'/'school' = synthetic
    dynenv worlds (exact GT, same 1024-beam format)."""
    if name == "spot":
        return SP.make_bundle()
    return DE.make(env=name, laps=1, seed=11)


# --------------------------------------------------------------- camera/QBE
class CamLane:
    """Aligned D455 frames -> exported vision head desc bits (laptop).
    The appearance map: per-keyframe (15, 20, 32) bit grids anchored at
    estimated poses; QBE = hamming match of one clicked cell's bits."""

    def __init__(self, kts_s):
        from sspax import headio as HIO
        import runners.spot_cam as SC
        import golden_cam as GC
        self.HIO, self.GC = HIO, GC
        self.head = HIO.load_head(str(HEAD_NPZ))
        self.shards, self.cts, self.where = SC._index("spot")
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


# ------------------------------------------------------------ SLAM + server
def make_slam():
    slam = F.BandSLAM(robust=True, attempt_every=4, relax_every=25,
                      gap_kf=300, recent_aids=12, spec=None, nph=0)
    slam.store_dtype = np.complex64
    slam.matcher = S.Matcher(L.ENC_MAIN, t_half=0.48, rot_half_deg=9.0,
                             rot_step_deg=1.5, perm=(4, L.N_ANG))
    return slam


class Demo:
    def __init__(self, data="spot", use_fpga=True, cam=True,
                 port=PORT_SER):
        self.port = port
        self.use_fpga = use_fpga
        self.want_cam = cam
        self.fpga = Fpga(port) if use_fpga else None
        self.clients = []
        self.lock = threading.Lock()
        self._req_reset = False
        self._req_data = None
        self._qbe = None                  # latest {kf: score}
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
        self.slam = make_slam()
        self.cam = None
        if data == "spot" and self.want_cam:
            try:
                self.cam = CamLane(np.asarray(self.b["kts"]))
            except Exception as e:
                print(f"[webvis] cam lane off: {e}", flush=True)
        self.est = np.zeros((self.n, 3))
        self.k = 0
        self.done = False
        self.pts, self.trail_py, self.trail_gt = [], [], []
        self.chip_segs = {}               # slot -> (pose, codes)
        self.chip_img = None
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
    def _chip_image(self):
        if not self.chip_segs:
            return None
        tr = np.array(self.trail_py) if self.trail_py else np.zeros((1, 2))
        lo = tr.min(0) - 3.0
        hi = tr.max(0) + 3.0
        ngrid = 96
        gx = np.linspace(lo[0], hi[0], ngrid)
        gy = np.linspace(lo[1], hi[1], ngrid)
        img = np.zeros((ngrid, ngrid), np.float32)
        XX, YY = np.meshgrid(gx, gy)
        P = np.stack([XX.ravel(), YY.ravel()], 1)
        for slot, (pose, codes) in list(self.chip_segs.items()):
            vec = np.exp(1j * np.pi / 2 * codes.astype(np.float32))
            d = P - pose[:2]
            m = (np.abs(d) < 3.0).all(1)
            if not m.any():
                continue
            c, s = np.cos(-pose[2]), np.sin(-pose[2])
            ds = d[m] @ np.array([[c, -s], [s, c]]).T
            sc = (np.exp(-1j * (ds @ W_MAIN.T)) @ vec).real / 240.0
            img.ravel()[np.flatnonzero(m)] = np.maximum(
                img.ravel()[np.flatnonzero(m)], sc)
        img = np.clip(img / max(img.max(), 1e-9), 0, 1)
        return dict(x0=float(lo[0]), y0=float(lo[1]),
                    x1=float(hi[0]), y1=float(hi[1]),
                    png=base64.b64encode(
                        (img * 255).astype(np.uint8).tobytes()).decode(),
                    n=ngrid)

    # ---- camera worker (desc bits off the hot loop) --------------------
    def _cam_worker(self):
        while True:
            k = self.camq.get()
            if self.cam is None:
                continue
            b = self.cam.compute(k)
            if b is None:
                continue
            with self.cam.lock:
                jb = self.cam.jpeg.get(k)
            self.broadcast(dict(cam_kf=int(k),
                                jpg=base64.b64encode(jb).decode()
                                if jb else None))

    # ---- one keyframe ---------------------------------------------------
    def step(self):
        k = self.k
        r, gtp = self.keys[k][0], self.keys[k][1]
        ts = self.keys[k][2] if len(self.keys[k]) > 2 else 0.0
        dig = vec = False
        if self.fpga:
            dig, vec, self.ms_fpga = self.fpga.verify_encode(
                k, np.asarray(r, float), int(ts * 1e6))
            if k % 8 == 0:
                codes = self.fpga.map_seg(k % 64)
                if codes is not None:
                    self.chip_segs[k % 64] = (None, codes)  # pose after est
        t0 = time.time()
        rr = DS.clean(self.b, np.asarray(r, float))
        pts, w = F.points_from_scan(rr, self.beam)
        if k == 0:
            guess = np.asarray(gtp, float).copy()
        elif k == 1:
            guess = self.est[0].copy()
        else:
            v = self.est[k - 1] - self.est[k - 2]
            vn = np.hypot(v[0], v[1])
            if vn > 0.30:
                v[:2] *= 0.30 / vn
            guess = np.array([self.est[k - 1][0] + v[0],
                              self.est[k - 1][1] + v[1],
                              self.est[k - 1][2] + S.wrap(v[2])])
        e = self.slam.add_keyframe(pts, w, guess)
        if self.slam.dirty:
            self.slam.relax()
            e = self.slam.pose_of(k)
        self.est[k] = e
        self.ms_slam = (time.time() - t0) * 1e3
        if self.fpga and k % 8 == 0 and (k % 64) in self.chip_segs:
            self.chip_segs[k % 64] = (e.copy(),
                                      self.chip_segs[k % 64][1])
        if self.cam and k % 2 == 0:
            try:
                self.camq.put_nowait(k)
            except queue.Full:
                pass

        c, s = np.cos(e[2]), np.sin(e[2])
        wpts = pts[::6] @ np.array([[c, -s], [s, c]]).T + e[:2]
        gt_ok = bool(self.gt_ok[k])
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
        st = dict(k=k, n=self.n, data=self.data,
                  dig=bool(dig), vec=bool(vec),
                  fx_dig=self.fpga.dig_ok if self.fpga else 0,
                  fx_vec=self.fpga.vec_ok if self.fpga else 0,
                  fx_tot=self.fpga.total if self.fpga else 0,
                  kbps=round(self.fpga.kbps(), 1) if self.fpga else 0,
                  ms_fx=round(self.ms_fpga, 1),
                  ms_py=round(self.ms_slam, 1),
                  est=[round(float(x), 3) for x in e],
                  gt=[round(float(x), 3) for x in gtp[:3]], gt_ok=gt_ok,
                  err=None if err is None else round(err, 3),
                  mem_kb=round(self.slam.memory_kb(), 1),
                  segs=len(self.slam.segvec),
                  loops=sum(1 for ed in self.slam.edges
                            if ed[5] == "loop"),
                  chip_segs=len(self.chip_segs),
                  overruns=self.overruns,
                  wpts=np.round(wpts, 3).tolist())
        self.k += 1
        if self.k >= self.n:
            self.done = True
            st["done"] = True
        self.broadcast(st)
        if self.fpga and k and k % 25 == 0:
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

    def snapshot(self):
        with self.lock:
            return dict(init=True, n=self.n, k=self.k, data=self.data,
                        cam=bool(self.cam), pts=self.pts,
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
                    dt = time.time() - t0
                    if dt > KF_DT:
                        self.overruns += 1
                    else:
                        time.sleep(KF_DT - dt)
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
                if d in ("spot", "classroom", "school"):
                    demo._req_data = d
                    demo.done = False
                self._json(dict(ok=True, data=d))
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
            elif self.path == "/status":
                self._json(dict(
                    k=demo.k, n=demo.n, data=demo.data, done=demo.done,
                    fx_dig=demo.fpga.dig_ok if demo.fpga else 0,
                    fx_vec=demo.fpga.vec_ok if demo.fpga else 0,
                    fx_tot=demo.fpga.total if demo.fpga else 0,
                    chip_segs=len(demo.chip_segs),
                    cam_kf=len(demo.cam.bits) if demo.cam else 0,
                    mem_kb=round(demo.slam.memory_kb(), 1),
                    segs=len(demo.slam.segvec),
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
    demo = Demo(use_fpga=use_fpga)
    threading.Thread(target=demo.run, kwargs=dict(loop=True),
                     daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(demo))
    print(f"[webvis] http://localhost:{port}  (FPGA "
          f"{'IN THE LOOP' if use_fpga else 'OFF'})", flush=True)
    srv.serve_forever()


def selftest(n=40):
    demo = Demo(use_fpga=True)
    demo.run(realtime=False, stop_after=n)
    f = demo.fpga
    errs = [np.linalg.norm(demo.est[k][:2]
                           - np.asarray(demo.keys[k][1])[:2])
            for k in range(n) if demo.gt_ok[k]]
    time.sleep(2.0)                        # let the cam worker drain
    ncam = len(demo.cam.bits) if demo.cam else 0
    print(f"selftest: {n} kf | digests {f.dig_ok}/{f.total} | on-chip "
          f"vectors {f.vec_ok}/{f.total} bit-exact | chip map "
          f"{len(demo.chip_segs)} segs fetched | cam desc grids {ncam} | "
          f"med err vs ref {np.median(errs):.3f} m | "
          f"mem {demo.slam.memory_kb():.1f} KB")
    assert f.dig_ok == f.vec_ok == f.total == n
    assert demo.chip_segs
    print("WEBVIS v2 SELFTEST PASS: every scan digest-verified AND "
          "on-chip-encoded bit-exact; chip map fetched + decoded")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "selftest":
        selftest(int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    elif cmd == "serve":
        serve(int(sys.argv[2]) if len(sys.argv) > 2 else 8790,
              use_fpga=not (len(sys.argv) > 3 and sys.argv[3] == "nofpga"))
