#!/usr/bin/env python3
"""ECP5 FPGA-in-the-loop WEB VISUALIZATION on real dataset data (STREAM.md).

The plugged Icepi Zero runs top_stream (silicon-gated 2026-07-16). Every
keyframe of the REAL SPOT classroom tour (data/spot_telluride, 1024-beam
ring-33 slice @ 5 Hz — the shipped 2D input) is:

  1. streamed THROUGH the FPGA as a lidar frame (n_rings=1, ring 33
     declared in the header per the STREAM.md honesty rule; uint16 mm)
     and DIGEST-VERIFIED bit-exact on silicon — a live acceptance
     counter on the deploy ingest path;
  2. fed to the SHIPPED lidar-only pipeline in real time (BandSLAM +
     matcher + CV-guess chain replicated-with-cite from
     runners/spot.run_cv — the 0.039-ATE flagship recipe);
  3. rendered live in the browser: registered world points, python-SLAM
     trail, GT ghost (the WITHHELD odometry — display/eval ONLY,
     anti-oracle), the bounded-memory counter, and FPGA link stats.

  python3 hw/ecp5/host/webvis.py selftest        # headless, numbers only
  python3 hw/ecp5/host/webvis.py serve [port]    # default 8790
"""
import json
import queue
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
import runners.datasets as DS                                  # noqa: E402
import runners.spot as SP                                      # noqa: E402
import sspslam.encoder as S                                    # noqa: E402
import sspslam.lattice as L                                    # noqa: E402
import sspslam.quantized as F                                  # noqa: E402

PORT_SER = "/dev/cu.usbserial-DK0GEIG0"
KF_DT = 0.2                     # 5 Hz keyframes (the tour's native rate)
MAX_PTS = 60_000                # world-point display cap (viz only)


# ---------------------------------------------------------------- FPGA lane
class Fpga:
    """One keyframe scan -> stream packet -> silicon -> digest verify."""

    def __init__(self, port=PORT_SER, baud=2_000_000):
        self.s = serial.Serial(port, baud, timeout=0.02)
        time.sleep(0.3)
        self.s.reset_input_buffer()
        self.tx = HS.Sender(self.s)
        self.tx.ctrl(0)                         # reset counters
        self.verified = 0
        self.total = 0
        self.bytes = 0
        self.t0 = time.time()

    def verify_scan(self, fid, r, t_us=0):
        """r: 1024 float ranges (m, inf=miss) -> (ok, roundtrip_ms).
        Polls for THIS frame's echo (stale packets from earlier sessions
        are skipped); short port timeout keeps the roundtrip ~tens of ms."""
        import struct
        mm = np.where(np.isfinite(r), np.clip(r * 1000.0, 0, 65535), 0.0)
        mm = mm.astype(np.uint16)[None, :]      # (rings=1, az=1024)
        t0 = time.time()
        self.tx.lidar_frame(fid, mm, [33], t_us=t_us)
        self.total += 1
        self.bytes += mm.nbytes
        ref = HS.digest_ref(np.ascontiguousarray(mm.T).tobytes())
        deadline = t0 + 0.4
        while time.time() < deadline:
            for typ, pl in HS.read_pkts(self.s, 1, timeout=0.05):
                if typ != 0x90:
                    continue
                _, rfid, dg, cnt = struct.unpack("<BIHI", pl)
                if rfid != fid:
                    continue                    # stale echo — skip
                ok = dg == ref and cnt == mm.nbytes
                self.verified += ok
                return ok, (time.time() - t0) * 1e3
        return False, (time.time() - t0) * 1e3

    def kbps(self):
        dt = max(time.time() - self.t0, 1e-6)
        return self.bytes / dt / 1024.0


# ------------------------------------------------------------ SLAM + server
class Demo:
    def __init__(self, use_fpga=True, port=PORT_SER):
        print("[webvis] loading SPOT classroom bundle...", flush=True)
        self.b = SP.make_bundle()
        self.keys, self.beam = self.b["keys"], self.b["beam"]
        self.n = len(self.keys)
        # the shipped flagship recipe — replicated from runners/spot.run_cv
        self.slam = F.BandSLAM(robust=True, attempt_every=4, relax_every=25,
                               gap_kf=300, recent_aids=12, spec=None, nph=0)
        self.slam.store_dtype = np.complex64
        self.slam.matcher = S.Matcher(L.ENC_MAIN, t_half=0.48,
                                      rot_half_deg=9.0, rot_step_deg=1.5,
                                      perm=(4, L.N_ANG))
        self.fpga = Fpga(port) if use_fpga else None
        self.est = np.zeros((self.n, 3))
        self.k = 0
        self.done = False
        self.pts = []                   # world points (viz)
        self.trail_py, self.trail_gt = [], []
        self.lock = threading.Lock()
        self.clients = []
        self.ms_slam = self.ms_fpga = 0.0
        self.overruns = 0

    # ---- one keyframe --------------------------------------------------
    def step(self):
        k = self.k
        r, gtp, ts = self.keys[k]
        ok_fx, self.ms_fpga = (self.fpga.verify_scan(k, np.asarray(r, float),
                                                     int(ts * 1e6))
                               if self.fpga else (False, 0.0))
        t0 = time.time()
        rr = DS.clean(self.b, np.asarray(r, float))
        pts, w = F.points_from_scan(rr, self.beam)
        if k == 0:
            guess = np.asarray(self.keys[0][1], float).copy()
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

        c, s = np.cos(e[2]), np.sin(e[2])
        wpts = pts[::6] @ np.array([[c, -s], [s, c]]).T + e[:2]
        gt_ok = bool(self.b["gt_ok"][k])
        with self.lock:
            self.pts.extend(np.round(wpts, 3).tolist())
            if len(self.pts) > MAX_PTS:
                self.pts = self.pts[-MAX_PTS:]
            self.trail_py.append([round(float(e[0]), 3),
                                  round(float(e[1]), 3)])
            self.trail_gt.append(
                [round(float(gtp[0]), 3), round(float(gtp[1]), 3), gt_ok])
        err = (float(np.linalg.norm(e[:2] - np.asarray(gtp)[:2]))
               if gt_ok else None)
        nloop = sum(1 for ed in self.slam.edges if ed[5] == "loop")
        st = dict(k=k, n=self.n, ok_fx=ok_fx,
                  fx_ok=self.fpga.verified if self.fpga else 0,
                  fx_tot=self.fpga.total if self.fpga else 0,
                  kbps=round(self.fpga.kbps(), 1) if self.fpga else 0,
                  ms_fx=round(self.ms_fpga, 1),
                  ms_py=round(self.ms_slam, 1),
                  est=[round(float(x), 3) for x in e],
                  gt=[round(float(x), 3) for x in gtp], gt_ok=gt_ok,
                  err=None if err is None else round(err, 3),
                  mem_kb=round(self.slam.memory_kb(), 1),
                  segs=len(self.slam.segvec), loops=nloop,
                  overruns=self.overruns,
                  wpts=np.round(wpts, 3).tolist())
        self.k += 1
        if self.k >= self.n:
            self.done = True
            st["done"] = True
        self.broadcast(st)
        return st

    # ---- SSE -----------------------------------------------------------
    def broadcast(self, obj):
        dead = []
        for q in self.clients:
            try:
                q.put_nowait(obj)
            except queue.Full:
                dead.append(q)
        for q in dead:
            self.clients.remove(q)

    def snapshot(self):
        with self.lock:
            return dict(init=True, n=self.n, k=self.k, pts=self.pts,
                        trail_py=self.trail_py, trail_gt=self.trail_gt)

    def reset(self):
        """Fresh pass: new SLAM + counters; clients keep their stream."""
        self.slam = F.BandSLAM(robust=True, attempt_every=4, relax_every=25,
                               gap_kf=300, recent_aids=12, spec=None, nph=0)
        self.slam.store_dtype = np.complex64
        self.slam.matcher = S.Matcher(L.ENC_MAIN, t_half=0.48,
                                      rot_half_deg=9.0, rot_step_deg=1.5,
                                      perm=(4, L.N_ANG))
        self.est = np.zeros((self.n, 3))
        self.k = 0
        self.done = False
        self.overruns = 0
        if self.fpga:
            self.fpga.tx.ctrl(0)
            self.fpga.verified = self.fpga.total = self.fpga.bytes = 0
            self.fpga.t0 = time.time()
        with self.lock:
            self.pts.clear()
            self.trail_py.clear()
            self.trail_gt.clear()
        self.broadcast(dict(reset=True))

    def run(self, realtime=True, stop_after=None, loop=False):
        while True:
            while not self.done:
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
            print(f"[webvis] tour {'complete' if self.done else 'stopped'} "
                  f"at kf {self.k}", flush=True)
            if not (loop and self.done):
                break
            time.sleep(5.0)
            self.reset()


# ----------------------------------------------------------------- HTTP/SSE
def make_handler(demo):
    html = (HERE / "webvis.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html)
            elif self.path == "/status":
                st = dict(k=demo.k, n=demo.n, done=demo.done,
                          fx_ok=demo.fpga.verified if demo.fpga else 0,
                          fx_tot=demo.fpga.total if demo.fpga else 0,
                          kbps=round(demo.fpga.kbps(), 1) if demo.fpga
                          else 0,
                          mem_kb=round(demo.slam.memory_kb(), 1),
                          segs=len(demo.slam.segvec),
                          ms_py=round(demo.ms_slam, 1),
                          ms_fx=round(demo.ms_fpga, 1),
                          overruns=demo.overruns)
                body = json.dumps(st).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                q = queue.Queue(maxsize=64)
                demo.clients.append(q)
                try:
                    snap = json.dumps(demo.snapshot())
                    self.wfile.write(f"data: {snap}\n\n".encode())
                    self.wfile.flush()
                    while True:
                        obj = q.get(timeout=30)
                        self.wfile.write(
                            f"data: {json.dumps(obj)}\n\n".encode())
                        self.wfile.flush()
                except (queue.Empty, BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    if q in demo.clients:
                        demo.clients.remove(q)
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
    errs = [np.linalg.norm(demo.est[k][:2] - np.asarray(demo.keys[k][1])[:2])
            for k in range(n) if demo.b["gt_ok"][k]]
    print(f"selftest: {n} kf | fpga verified {f.verified}/{f.total} "
          f"({f.kbps():.0f} KB/s) | slam {demo.ms_slam:.0f} ms/kf last | "
          f"med err vs withheld ref {np.median(errs):.3f} m | "
          f"mem {demo.slam.memory_kb():.1f} KB segs {len(demo.slam.segvec)}")
    assert f.verified == f.total == n, "FPGA digest verification failed"
    print("WEBVIS SELFTEST PASS: every scan digest-verified on silicon")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "selftest":
        selftest(int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    elif cmd == "serve":
        serve(int(sys.argv[2]) if len(sys.argv) > 2 else 8790,
              use_fpga=not (len(sys.argv) > 3 and sys.argv[3] == "nofpga"))
