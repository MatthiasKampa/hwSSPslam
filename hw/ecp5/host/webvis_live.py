#!/usr/bin/env python3
"""webvis LIVE mode: sensor data arrives over UDP from the robot's ROS
bridge (hunter_ws lidar_fpga_bridge, webvis_udp:=...) instead of a
recorded bundle. webvis keeps FULL ownership of the FPGA serial (one
owner) — the ROS side never opens the port. Designed by the robot-side
bring-up agent; extended at HEAD with the CAMERA lane (OAK).

Datagram formats (loopback-sized, type-tagged since the cam extension):
  SCAN (legacy, untyped, len == 8 + 2048):
      t_us u64 LE | mm u16 LE x 1024 (ranges, fixed az grid, 0 = miss)
  TYPED (first byte):
      0x01 SCAN : t_us u64 | mm u16 x 1024
      0x02 CAM  : t_us u64 | JPEG bytes (any resolution; converted to
                  gray 320x240 -> exported head desc bits -> the cam VSA
                  object map at the CURRENT pose, bearings lifted by the
                  LATEST scan; SSP_CAM_FOV degrees, default 69 = OAK RGB)

Usage (robot):
    SSP_BIND=0.0.0.0 SSP_PORT=/dev/ttyUSB0 \
        python3 hw/ecp5/host/webvis_live.py [http] [udp]
    ros2 launch lidar_fpga_bridge stream_bridge.launch.py \
        webvis_udp:=127.0.0.1:8791 cam_topic:=/oak/rgb/image_raw
Browser: http://<robot-ip>:8790
"""
import io
import os
import socket
import struct
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import webvis  # noqa: E402

N_BEAM = 1024
BIG = 200_000                    # preallocated est/gt_ok capacity
MAX_LAG = 12                     # drop incoming kf if SLAM this far behind
CAM_FOV = np.deg2rad(float(os.environ.get("SSP_CAM_FOV", "69")))


def live_bundle():
    beam = -np.pi + np.arange(N_BEAM) * (2 * np.pi / N_BEAM)
    return dict(name="live", kind="live", eval="none", path="",
                keys=[], beam=beam, odom=None, kts=[],
                rmin=0.3, rmax=60.0, gt=None,
                gt_ok=np.zeros(BIG, bool), guess_mode="cv")


class LiveCam:
    """OAK jpegs -> exported head desc bits (the CamLane analog with no
    dataset alignment: frames pair with the CURRENT pose/scan)."""

    def __init__(self):
        from sspax import headio as HIO
        self.HIO = HIO
        self.head = HIO.load_head(str(webvis.HEAD_NPZ))
        self.bits = {}
        self.jpeg = {}
        self.lock = threading.Lock()

    def compute(self, k, jb):
        from PIL import Image
        im = Image.open(io.BytesIO(jb)).convert("L").resize((320, 240))
        g = np.asarray(im, np.uint8)
        b = self.HIO.cell_bits(self.head, g, source="desc")
        with self.lock:
            self.bits[k] = b
            self.jpeg[k] = jb
            if len(self.jpeg) > 300:
                self.jpeg.pop(next(iter(self.jpeg)))
        return b

    def query(self, k, cy, cx):
        with self.lock:
            if k not in self.bits:
                return {}
            q = self.bits[k][cy, cx]
            return {int(kk): 1.0 - float((b != q).sum(-1).min()) / 32.0
                    for kk, b in self.bits.items()}


class LiveDemo(webvis.Demo):
    """Dataset player -> unbounded live consumer. keys grows from the UDP
    thread; est/gt_ok preallocated; done never latches. The cam lane
    binds OAK desc bits at the live pose (nominal yaw extrinsics,
    SSP_CAM_FOV)."""

    def _load(self, data):
        if data != "live":
            return super()._load(data)
        print("[webvis] live mode: waiting for UDP scans...", flush=True)
        self.data = data
        self.b = live_bundle()
        self.keys, self.beam = self.b["keys"], self.b["beam"]
        self.n = 0
        self.gt_ok = self.b["gt_ok"]
        self.slam = webvis.make_slam()
        self.cam = LiveCam() if self.want_cam else None
        self.cammap = webvis.CamMap() if self.want_cam else None
        self.cammap_fov = CAM_FOV
        if self.cammap:
            self.cammap.HFOV = CAM_FOV
        self.est = np.zeros((BIG, 3))
        self.k = 0
        self.done = False
        self.pts, self.trail_py, self.trail_gt = [], [], []
        self.chip_segs = {}
        self.chip_img = None
        self.ms_slam = self.ms_fpga = 0.0
        self.overruns = 0
        self._camq_live = []             # (t_us, jpeg) pending frames
        if self.fpga:
            self.fpga.tx.ctrl(0)
            self.fpga.dig_ok = self.fpga.vec_ok = self.fpga.total = 0
            self.fpga.bytes = 0
            self.fpga.t0 = time.time()
        self.broadcast(dict(reset=True, data=data,
                            cam=bool(self.cam)))

    def _cam_worker(self):
        while True:                      # live cam runs in _drain_cam
            time.sleep(3600)

    def _drain_cam(self):
        """Bind pending camera frames at the current pose + latest scan
        (live analog of the dataset cam worker)."""
        while self._camq_live and self.cam and self.k:
            t_us, jb = self._camq_live.pop(0)
            k = self.k - 1
            try:
                b = self.cam.compute(k, jb)
            except Exception:
                continue
            self._cam_track(k, b)
            self.cammap.ingest(k, b, np.asarray(self.keys[k][0], float),
                               self.est[k])
            self.broadcast(dict(cam_kf=int(k), jpg=__import__(
                "base64").b64encode(jb).decode()))
            cls = self.cammap.classes()
            if cls:
                self.broadcast(dict(classes=[
                    dict(id=c["id"], n=c["n"], thumb=None,
                         boost=c["id"] in self.cammap.boost)
                    for c in cls]))

    def run(self, realtime=True, stop_after=None, loop=False):
        while True:
            if self._req_reset:
                self._req_reset = False
                self._load(self.data)
            if self._req_data:                 # UI switch to a dataset
                d, self._req_data = self._req_data, None
                self._load(d)
            if self.data != "live":            # recorded tour: one pass,
                if not self.done:              # then back to live
                    t0 = time.time()
                    self.step()
                    dt = time.time() - t0
                    if dt < webvis.KF_DT:
                        time.sleep(webvis.KF_DT - dt)
                else:
                    time.sleep(2.0)
                    self._load("live")
                continue
            if self.k < len(self.keys) and self.k < BIG:
                self.n = len(self.keys)
                self.step()
                self.done = False              # never latch in live mode
                self._drain_cam()
            else:
                time.sleep(0.02)


def udp_rx(demo, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", port))
    scan_len = 8 + 2 * N_BEAM
    while True:
        pkt, _ = s.recvfrom(65535)
        if demo.data != "live":
            continue
        if len(pkt) == scan_len:                       # legacy scan
            typ, off = 0x01, 0
        else:
            typ, off = pkt[0], 1
        if typ == 0x01 and len(pkt) - off == scan_len:
            if len(demo.keys) - demo.k > MAX_LAG:
                demo.overruns += 1
                continue
            t_us, = struct.unpack_from("<Q", pkt, off)
            mm = np.frombuffer(pkt, "<u2", N_BEAM, off + 8)
            r = np.where(mm > 0, mm / 1000.0, np.inf)
            demo.keys.append((r, np.zeros(3), t_us / 1e6))
        elif typ == 0x02 and len(pkt) > off + 12:      # cam jpeg
            t_us, = struct.unpack_from("<Q", pkt, off)
            if len(getattr(demo, "_camq_live", [])) < 4:
                demo._camq_live.append((t_us, pkt[off + 8:]))


def serve(http_port=8790, udp_port=8791):
    bind = os.environ.get("SSP_BIND", "127.0.0.1")
    demo = LiveDemo(data="live", use_fpga=True, cam=True,
                    port=os.environ.get("SSP_PORT",
                                        webvis.PORT_SER))
    threading.Thread(target=udp_rx, args=(demo, udp_port),
                     daemon=True).start()
    threading.Thread(target=demo.run, daemon=True).start()
    srv = ThreadingHTTPServer((bind, http_port),
                              webvis.make_handler(demo))
    print(f"[webvis-live] http://{bind}:{http_port}  UDP on "
          f"127.0.0.1:{udp_port}  (FPGA IN THE LOOP; cam lane "
          f"FOV {np.rad2deg(CAM_FOV):.0f} deg)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8790,
          int(sys.argv[2]) if len(sys.argv) > 2 else 8791)
