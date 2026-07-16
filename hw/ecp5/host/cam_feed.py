#!/usr/bin/env python3
"""cam_feed — stream a camera view into webvis_live's UDP cam lane
(STREAM datagram 0x02: t_us u64 LE + JPEG). Runs on the robot beside
the lidar bridge; webvis_live turns the frames into desc bits + the cam
VSA object map and shows the live view in the browser.

Modes:
  ROS (default): subscribe `--topic` (sensor_msgs Image OR
      CompressedImage — compressed jpeg forwards as-is), decimate to
      `--fps`, send to `--dst`.
  --test: NO ROS — synthesize a moving test-pattern JPEG at `--fps`
      (proves the lane end-to-end before camera hardware arrives).

  python3 cam_feed.py --test --fps 1
  python3 cam_feed.py --topic /oak/rgb/image_raw --fps 2
"""
import argparse
import socket
import struct
import time

import numpy as np


def send(sock, dst, jpeg, t_us=None):
    t_us = int(time.time() * 1e6) if t_us is None else t_us
    sock.sendto(b"\x02" + struct.pack("<Q", t_us) + jpeg, dst)


def test_mode(dst, fps):
    import cv2
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rng = np.random.default_rng(0)
    tex = [rng.integers(40, 220, (60, 60), np.uint8) for _ in range(6)]
    k = 0
    print(f"[cam_feed] TEST pattern -> {dst} @ {fps} fps", flush=True)
    while True:
        img = np.full((240, 320), 30, np.uint8)
        for i, t in enumerate(tex):                # blocks orbit slowly
            a = k * 0.05 + i * 1.05
            x = int(130 + 100 * np.cos(a))
            y = int(90 + 60 * np.sin(a))
            img[y:y + 60, x:x + 60] = t
        cv2.putText(img, "TEST PATTERN (no camera attached)",
                    (12, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1)
        ok, jb = cv2.imencode(".jpg", img,
                              [cv2.IMWRITE_JPEG_QUALITY, 80])
        send(sock, dst, jb.tobytes())
        k += 1
        time.sleep(1.0 / fps)


def ros_mode(dst, topic, fps):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, Image
    import cv2

    class CamFeed(Node):
        def __init__(self):
            super().__init__("cam_feed")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.last = 0.0
            self.n = 0
            if topic.endswith(("compressed", "Compressed")):
                self.create_subscription(CompressedImage, topic,
                                         self.on_comp, 2)
            else:
                self.create_subscription(Image, topic, self.on_raw, 2)
            self.get_logger().info(
                f"cam_feed: {topic} -> {dst} @ {fps} fps")

        def _pace(self, stamp):
            now = time.time()
            if now - self.last < 1.0 / fps:
                return None
            self.last = now
            return int(stamp.sec * 1e6 + stamp.nanosec / 1e3)

        def on_comp(self, m):
            t = self._pace(m.header.stamp)
            if t is None:
                return
            send(self.sock, dst, bytes(m.data), t)
            self.n += 1

        def on_raw(self, m):
            t = self._pace(m.header.stamp)
            if t is None:
                return
            a = np.frombuffer(m.data, np.uint8)
            if m.encoding in ("rgb8", "bgr8"):
                img = a.reshape(m.height, m.width, 3)
                if m.encoding == "rgb8":
                    img = img[..., ::-1]
            elif m.encoding in ("mono8", "8UC1"):
                img = a.reshape(m.height, m.width)
            else:
                return
            if img.shape[1] > 320:          # transport thrift at high fps
                h2 = int(img.shape[0] * 320 / img.shape[1])
                img = cv2.resize(img, (320, h2))
            ok, jb = cv2.imencode(".jpg", img,
                                  [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                send(self.sock, dst, jb.tobytes(), t)
                self.n += 1

    rclpy.init()
    rclpy.spin(CamFeed())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="127.0.0.1:8791")
    ap.add_argument("--topic",
                    default="/camera/camera/color/image_raw")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    host, _, port = a.dst.partition(":")
    dst = (host, int(port or 8791))
    if a.test:
        test_mode(dst, a.fps)
    else:
        ros_mode(dst, a.topic, a.fps)
