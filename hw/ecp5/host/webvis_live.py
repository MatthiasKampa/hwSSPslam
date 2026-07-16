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
import queue
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
    """cam jpegs -> exported head desc bits: the cbits C KERNEL when it
    builds (1.25 ms/frame, 100% parity gate — full-frame-rate capable),
    headio numpy fallback otherwise. Frames pair with the CURRENT
    pose/scan (no dataset alignment)."""

    def __init__(self):
        from sspax import headio as HIO
        self.HIO = HIO
        self.head = HIO.load_head(str(webvis.HEAD_NPZ))
        self.cb = None
        try:
            from cbits import CBits
            self.cb = CBits(str(webvis.HEAD_NPZ))
            print("[webvis] cam desc bits: cbits C kernel", flush=True)
        except Exception as e:
            print(f"[webvis] cbits unavailable ({e}); numpy fallback",
                  flush=True)
        self.bits = {}
        self.jpeg = {}
        self.lock = threading.Lock()

    def compute(self, k, jb):
        from PIL import Image
        im = Image.open(io.BytesIO(jb)).convert("L").resize((320, 240))
        g = np.asarray(im, np.uint8)
        b = self.cb.bits(g) if self.cb else \
            self.HIO.cell_bits(self.head, g, source="desc")
        with self.lock:
            self.bits[k] = b
            self.jpeg[k] = jb
            if len(self.jpeg) > 3000:      # ~5 min of kf — captures now
                self.jpeg.pop(next(iter(self.jpeg)))   # replay with FULL
        return b                                       # video (~45 MB)

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

    live_capable = True         # "live" appears in the selector

    def _load(self, data):
        if data != "live":
            return super()._load(data)
        print("[webvis] live mode: waiting for UDP scans...", flush=True)
        self.data = data
        self.b = live_bundle()
        self.keys, self.beam = self.b["keys"], self.b["beam"]
        self.n = 0
        self.gt_ok = self.b["gt_ok"]
        self.pose_src = "odom (datagram)"    # chip tracker when it lands
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
        self.walls = None      # STREAM rolling slots: no wall lane;
                               # SoloDemo swaps in a matcher-band one
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
        """FULL-RATE bits (cbits C kernel, ~1-3 ms/frame): every pending
        frame's desc bits run at wire rate, newest-first. The VSA map
        INGESTS once per new keyframe — re-binding the same kf every
        frame would only inflate cluster counts, not add information."""
        last_ingest = -1
        while True:
            try:
                # snapshot the queue REFERENCE: _load() (reset) swaps in
                # a fresh list concurrently — check-then-pop on the
                # attribute raced that and an escaped IndexError killed
                # this thread for good (cam_kf froze at 0 live).
                q = getattr(self, "_camq_live", None)
                if self.data != "live" or not q or not self.k:
                    if self.data != "live":  # replay mode ONLY: drain
                        try:                 # the DATASET cam queue —
                            k = self.camq.get_nowait()   # in live mode
                            self._cam_step(k)  # cam is LiveCam(k, jb)
                            continue
                        except queue.Empty:
                            pass
                    time.sleep(0.005)
                    continue
                try:
                    t_us, jb = q.pop()            # newest
                    q.clear()                     # drop stale
                except IndexError:
                    continue
                k = self.k - 1
                if k < 0 or k >= len(self.keys):   # reset transient
                    continue
                try:
                    b = self.cam.compute(k, jb)
                except Exception:
                    continue
                self._cam_track(k, b)
                if k == last_ingest:
                    continue
                last_ingest = k
                self.cammap.ingest(k, b,
                                   np.asarray(self.keys[k][0], float),
                                   self.est[k])
                cls = self.cammap.classes()
                if cls:
                    self.broadcast(dict(classes=[
                        dict(id=c["id"], n=c["n"], thumb=None,
                             boost=c["id"] in self.cammap.boost)
                        for c in cls]))
            except Exception as e:                # NEVER die silently
                print(f"[webvis] cam worker error (recovered): {e}",
                      flush=True)
                time.sleep(0.1)

    def run(self, realtime=True, stop_after=None, loop=False):
        while True:
            if self._req_reset:
                self._req_reset = False
                self._load(self.data)
            if self._req_data:                 # UI switch to a dataset
                d, self._req_data = self._req_data, None
                try:
                    self._load(d)
                except Exception as e:         # no dataset dirs on the
                    print(f"[webvis] '{d}' unavailable ({e}); "
                          f"back to live", flush=True)   # robot -> live
                    self._load("live")
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
            else:
                time.sleep(0.02)


class SoloDemo(LiveDemo):
    """AUTONOMY deploy (SSP_SOLO=1, top_solo_ecp5.bit flashed): the
    ECP5 SOLO chip IS the tracker — every scan goes down the wire as a
    keyframe (points + body-frame odom delta from the datagram pose),
    and the CHIP pose comes back as THE pose estimate. The datagram
    odom remains only the prediction input + display ghost. The chip
    also BUILDS the map (fold/freeze); frozen segments are dumped and
    fed to the decode layer. Host = vis + data transport, nothing else."""

    def __init__(self, *a, **kw):
        kw["use_fpga"] = False              # no STREAM board — SOLO owns
        self._chip = None                   # the serial
        super().__init__(*a, **kw)
        import solo_host
        self._chip = solo_host.SoloChip(
            os.environ.get("SSP_PORT", solo_host.PORT))
        assert self._chip.ping(), "SOLO chip not replying (flashed?)"
        self._chip.set_pose(0, 0, 0)
        self._chip.set_mode(mapping=True, stream=True)
        self.pose_src = "CHIP solo tracker"
        self.chip_state = 3
        print("[webvis] SOLO autonomy: on-chip tracker is the pose "
              "source", flush=True)

    def _load(self, data):
        super()._load(data)
        if data == "live" and self._chip is not None:
            self._chip.reset_map()
            self._chip.set_pose(0, 0, 0)
            self.pose_src = "CHIP solo tracker"
            # walls from the CHIP's own dumped segments (matcher band,
            # liveness+scales planes ride the dump)
            self.walls = webvis.WallLane("matcher")

    def pose_for(self, k, r, gtp):
        if self.data != "live" or self._chip is None:
            return super().pose_for(k, r, gtp)
        res = self._chip.keyframe(np.asarray(r, float),
                                  np.asarray(gtp, float))
        if res is None:                     # lost frame: odom fallback
            self.chip_state = 4
            return super().pose_for(k, r, gtp)
        x, y, yaw, st, frozen, n_seg = res
        self.chip_state = st
        if frozen:                          # new frozen segment: refresh
            segs = self._chip.dump()        # the decode layer (~n*114 B)
            if segs:
                self.chip_segs = {i: (s[0], s[1])
                                  for i, s in enumerate(segs)}
                if self.walls is not None:  # matcher-band wall lane
                    try:
                        self.walls.set_segs(webvis.MD.segs_from_dump(segs))
                    except Exception as ex:
                        print(f"[webvis] wall lane off ({ex})",
                              flush=True)
                        self.walls = None
        return np.array([x, y, yaw])


def udp_rx(demo, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", port))
    scan_len = 8 + 2 * N_BEAM
    last_jpg = [0.0]
    while True:
        pkt, _ = s.recvfrom(65535)
        if demo.data != "live":
            continue
        if len(pkt) in (scan_len, scan_len + 12):     # legacy (+pose)
            typ, off = 0x01, 0
        else:
            typ, off = pkt[0], 1
        if typ == 0x01 and len(pkt) - off in (scan_len, scan_len + 12):
            if len(demo.keys) - demo.k > MAX_LAG:
                demo.overruns += 1
                continue
            t_us, = struct.unpack_from("<Q", pkt, off)
            mm = np.frombuffer(pkt, "<u2", N_BEAM, off + 8)
            r = np.where(mm > 0, mm / 1000.0, np.inf)
            pose = np.array(struct.unpack_from("<fff", pkt,
                            off + 8 + 2 * N_BEAM), float) \
                if len(pkt) - off == scan_len + 12 else \
                (np.asarray(demo.keys[-1][1]) if demo.keys
                 else np.zeros(3))
            demo.keys.append((r, pose, t_us / 1e6))
        elif typ == 0x02 and len(pkt) > off + 12:      # cam jpeg
            t_us, = struct.unpack_from("<Q", pkt, off)
            jb = pkt[off + 8:]
            # display decimated to ~20 fps (full wire rate would flood
            # slow SSE clients out of the queue); COMPUTE stays full rate
            now = time.time()
            if now - last_jpg[0] >= 0.05:
                last_jpg[0] = now
                demo.broadcast(dict(cam_kf=max(demo.k - 1, 0),
                                    jpg=__import__("base64")
                                    .b64encode(jb).decode()))
            if getattr(demo, "_camq_live", None) is not None:
                demo._camq_live.append((t_us, jb))
                if len(demo._camq_live) > 3:
                    demo._camq_live.pop(0)


def serve(http_port=8790, udp_port=8791):
    bind = os.environ.get("SSP_BIND", "127.0.0.1")
    if os.environ.get("SSP_SOLO", "") == "1":
        demo = SoloDemo(data="live", cam=True)
    else:
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
