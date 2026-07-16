# STREAM — virtual sensors over USB (robot PC → Icepi Zero)

Opened 2026-07-16 (user directive: sensor shipment delayed; lidar + camera
will stream from the robot's Linux machine over USB — "no preprocessing on
the PC, pick an efficient format"; **the OV5640 path is kept** — this is a
second, coexisting front end, not a replacement).

## Purpose and honesty rules

The FPGA ingests SENSOR-EQUIVALENT raw streams and does its own processing
on-chip, exactly as it would with physical sensors:

- The PC may **reformat and subset** (QVGA downscale, RGB→Y8 luma — the
  formats the physical sensors would deliver anyway; ring subsetting for
  bandwidth), but never compute features, never filter, never register.
  Every subset is DECLARED in the stream header (`ring_ids`, `w×h`), so
  nothing is silently preprocessed.
- Timestamps travel IN the stream (sensor time from the robot, µs). All
  temporal logic downstream (keyframing, delay fusion) consumes STREAM
  time, never arrival time — the delayfuse law (interval-matched fusion)
  holds at any link speed, and slower-than-real-time replay is
  semantically identical to live capture (deterministic pipeline).
- **Coexistence contract**: the camera assembler's output interface is
  pixel-stream-identical to `dvp_capture.v`'s output (Y8 pixel + valid +
  line/frame syncs). Downstream blocks (fast9, dense encoders) sit behind
  a SRC MUX (CTRL-selected, default DVP) and cannot tell replay from the
  physical OV5640. The OV bring-up tops (`top_ov5640_id/snap`) and ladder
  (#31) are untouched.

## Transport (measured board facts, 2026-07-16)

- The board's USB-UART is an **FT231X** (Full-Speed, max 3 Mbaud). Only
  TXD/RXD/RTS#/DTR# reach FPGA pins (LPF) — no FT245 FIFO option.
- **Two USB-C ports have D+/D− wired directly to FPGA pins**
  (`usb_dp/dn[0]` F15/E16 + pulls, `usb_dp/dn[1]` J16/J15 + pulls): a
  soft USB-FS CDC-ACM device (~0.9–1.1 MB/s practical) is a pure-RTL
  transport upgrade; the Linux side stays pyserial (`/dev/ttyACM0`).
- The packet layer below is TRANSPORT-AGNOSTIC (any ordered byte pipe).

Tiers:
| tier | link | rate | status |
|---|---|---|---|
| v0 | FT231X UART @ 2 Mbaud (DIV=25 @ 50 MHz) | 197 KB/s measured | **GATED on silicon** |
| v0.5 | 2.4 Mbaud (FTDI-exact; DIV=21 → −0.8 %) | ~235 KB/s | macOS VCP silently kept ~2M (+19 % mismatch, zero packets) — retest with Linux `ftdi_sio` on the robot box before enabling |
| v1 | soft USB-FS CDC on `usb_dp/dn[0]` | ~1 MB/s | RTL, planned |
| — | 3.0 Mbaud (DIV=17 → −2.0 %) | marginal timing | not default |

## Live-rate matrix (what fits, honestly)

| stream | rate | payload | v0 (195 KB/s) | v1 (~1 MB/s) |
|---|---|---|---|---|
| lidar 3-ring×1024, range16 | 20 Hz | 123 KB/s | **LIVE** | LIVE |
| lidar 64-ring×1024, range16 | 20 Hz | 2.62 MB/s | 0.07× | 0.38× |
| cam QVGA Y8 | 30 fps | 2.30 MB/s | 0.08× | 0.43× |
| cam QVGA Y8 | 5 Hz (keyframe tier) | 384 KB/s | 0.5× | **LIVE** |
| deploy-live combo: 3-ring @20 Hz + cam @5 Hz | | 507 KB/s | 0.38× | **LIVE** |

So: v0 runs the 3-ring lidar tier live plus camera at ~1–2 fps (or either
stream alone faster); v1 (soft-USB) runs the full deploy-live combo
(matcher-space lidar @ 20 Hz + keyframe vision @ 5 Hz). Full-fidelity
64-ring + 30 fps (4.9 MB/s) exceeds every link this board has → those runs
stream sub-real-time (results identical by determinism + stream time), and
AT-RATE gates use SDRAM-buffered burst replay (v2, needs #44): frames are
loaded slow, then a pacer releases them into the sensor FIFOs at true
20 Hz / 30–120 fps for seconds-long rate-corner tests.

## Packet format (little-endian; CRC16-CCITT poly 0x1021 init 0xFFFF)

```
| A5 5A | type u8 | flags u8 | len u16 | seq u16 | payload[len≤4096] | crc16 u16 |
```
CRC covers type..payload. `seq` is a global rolling counter (gap
detection). Corrupt packet → dropped whole (digest state rolls back),
parser re-hunts the magic.

Host→FPGA types:
- `0x01 LIDAR_HDR` payload: `frame_id u32, t_us u64, n_rings u8,
  ring_ids u8[n_rings], n_az u16, fmt u8` (fmt bit0 = range u16 mm;
  bit1 = +reflectivity u8 plane — appended per column when set)
- `0x02 LIDAR_COL` payload: `frame_id u32, az0 u16, n_cols u8,
  data[n_cols × n_rings × (2|3) B]` (columns az0..az0+n_cols−1, rings in
  ring_ids order; ~≤4 KB per packet)
- `0x03 CAM_HDR` payload: `frame_id u32, t_us u64, w u16, h u16, fmt u8`
  (fmt 0 = Y8)
- `0x04 CAM_ROW` payload: `frame_id u32, row0 u16, n_rows u8,
  data[n_rows × w]`
- `0x05 IMU_SAMPLE` (reserved — robot IMU later; mirrors the ism330
  stream frame content)
- `0x0F MAP_READ` payload: `seg u8` — fetch a stored on-chip map
  segment's compressed codes (reply 0x93)
- `0x10 CTRL` payload: `cmd u8, arg u32` — 0 reset counters, 1 status
  request, 2 echo en/dis (arg), 3 src-mux select (arg: 0 DVP, 1 stream;
  v2), 4 pacer config (v2)

FPGA→host types (same framing):
- `0x90 ECHO_DIGEST` payload: `stream u8 (1=lidar, 3=cam), frame_id u32,
  digest u16, count u32` — emitted when a frame completes (count =
  payload data bytes; digest = CRC16 over the frame's data bytes in
  arrival order, headers excluded)
- `0x92 VEC` payload: `frame_id u32, acc i32[480]` (1924 B) — the
  ON-CHIP-ENCODED VSA vector of a completed single-ring lidar frame:
  240 lattice components × (re, im) int32 LE, interleaved re,im per
  component — bit-exact to `hw/ice40/golden.encode_int` on the kept
  points (r_mm != 0, r_mm <= 31200; w = 127). Emitted automatically
  after the frame's ECHO when the top has the encoder (top_stream_enc);
  ~9.7 ms at 2 Mbaud, well inside the 5 Hz keyframe budget.
- `0x93 MAP_SEG` payload: `seg u8, codes u8[60]` — the on-chip map
  bank's compressed segment: 240 components × 2-bit QPSK codes
  (0..3 ≡ 1, i, −1, −i — `golden.mcode_from_vec` convention), packed 4
  components/byte little-first (comp 4b+k in bits 2k+1..2k). The map
  RESIDES on chip (64-segment ring, slot = frame_id % 64, written as
  each VEC streams out); the laptop fetches + DECODES for viz/matching.
- `0x80 STATUS` payload: `pkts_ok u32, crc_drops u16, seq_gaps u16,
  cam_frames u16, lidar_frames u16` (on CTRL request)
- `0x81 CREDIT` payload: `grants u16` — flow control for the SDRAM-ring
  era; v0 grants max once at start (the 50 MHz parser outruns every
  transport tier, ~6 MB/s ceiling)

## ROS-bridge implementer notes (robot side codes against THIS section)

- Transport: the FPGA's FT231X enumerates as a standard USB CDC/serial
  device on Linux (`/dev/ttyUSB*`, `ftdi_sio`). Open at **2,000,000
  baud, 8N1, no flow control**. (2.4 Mbaud may work with `ftdi_sio` —
  it silently fell back to ~2M on macOS's VCP; verify with the loopback
  gate before enabling.) All integers little-endian. CRC16-CCITT:
  poly 0x1021, init 0xFFFF, over `type..payload` (magic and the CRC
  itself excluded). `seq` is one global rolling u16 across all packets
  the sender emits.
- Lidar (Ouster → packets): pick the ring subset (deploy = ring 33
  alone for the matcher slice, or rings {16,33,50}); build per-frame
  `(n_rings, n_az=1024)` u16 range-in-MM images on the fixed azimuth
  grid `az_k = -pi + k*2pi/1024` (beam k = column k); missing/invalid
  returns = 0; ranges CLIP to 65535. Send LIDAR_HDR then LIDAR_COL
  column blocks (32 cols/packet is a good size). `t_us` = sensor
  timestamp in µs (authoritative downstream — never re-stamp).
  ON-CHIP ENCODE fires only for n_rings==1 frames (the matcher slice);
  multi-ring frames are digest-verified transport only.
- Camera: Y8 luma, resize/bin to 320×240 (or 120×160), rows top-down;
  CAM_HDR + CAM_ROW blocks (8 rows/packet). RGB→Y8 on the robot is
  format conversion, not preprocessing; declare the true resolution in
  the header. No other processing (no rectify, no features).
- Pacing: ~197 KB/s goodput at 2 Mbaud. Deploy-live budget: ring-33
  slice @ 5 Hz keyframes ≈ 10.4 KB/s up + ~10 KB/s down (ECHO+VEC) —
  1/10th of the link. 3-ring @ 20 Hz ≈ 123 KB/s also fits. Do NOT
  exceed ~190 KB/s aggregate; watch STATUS.crc_drops/seq_gaps (poll
  CTRL cmd=1 every few seconds) and back off if they move.
- **Encode-frame column cap (IMPORTANT)**: for single-ring frames (the
  on-chip-encode path) keep `n_cols <= 8` per LIDAR_COL packet at
  2 Mbaud. The serial encoder core drains ~840 cycles/point; larger
  packets outpace it and the feeder's one-deep commit queue drops the
  overflow (counted on-chip, never silent — this exact mode dropped
  half the points of 32-col packets before the queue existed; found on
  silicon 2026-07-16). Multi-ring (digest-only) frames may use up to
  ~32 cols/packet.
- Recommended node behavior: subscribe lidar + camera topics; per
  lidar keyframe (decimate to 5 Hz) send HDR+COLs, await 0x90 (digest
  match optional robot-side — the laptop/webvis also verifies) and
  0x92 if present; forward 0x92/0x93 upstream if a consumer wants the
  chip vectors; expose CTRL reset + STATUS as services.
- Reference implementation: `hw/ecp5/host/hw_stream.py` (Sender +
  read_pkts, pure pyserial — lift verbatim) and `webvis.py`'s
  `Fpga.verify_scan` for the echo/VEC handshake pattern.

## RTL (v0, this build)

`rtl/stream_ingest.v` — byte-at-a-time FSM: magic hunt → header → payload
dispatch-on-the-fly → CRC check. Per-packet SHADOW-COMMIT: stream digest/
meta registers are snapshotted at header time and restored on CRC failure
— no payload buffering, zero EBR. TX side shares the framing (digest/
status packets). `rtl/top_stream.v` — 50 MHz top, uart @ DIV=25, LEDs:
rx-activity / sticky-crc-drop / cam-frame / lidar-frame.
`rtl/tb_stream.v` — G0 sim: garbage resync, cam frame, lidar frame,
corrupt packet (drop + rollback), status readback.
`host/hw_stream.py` — reference sender + gates; pyserial, runs unchanged
on the robot's Linux box (it IS the robot-side reference implementation).

## Gates

- **G0 sim** (`make sim-stream`): parser + rollback + digests assert.
  **PASS 2026-07-16** (resync, 3 frame digests bit-exact, corrupt-packet
  rollback, status counters).
- **G1 silicon loopback** (`python3 host/hw_stream.py gate`): synthetic
  cam+lidar frames streamed at 2 Mbaud; every frame digest must match the
  host's local CRC16; zero crc_drops/seq_gaps; goodput printed.
  **PASS 2026-07-16**: 18/18 digests bit-exact (6 cam QVGA + 12 lidar
  3×1024), 583 pkts, 0 drops/gaps, 197 KB/s; soak: 241 lidar frames @
  **30.0 fps sustained = 1.5× the 20 Hz live target**, 7956 pkts, 0 drops
  (timing 54.2 MHz vs 50 constraint).
- **G2 fast9-through-ingest** (next): the 64×64 fixture sent as CAM
  packets → SRC MUX → fast9 → centres bit-exact vs golden (end-to-end
  virtual sensor; supersedes the ad-hoc loader in top_fast9_uart).
- **G3 encoder-on-streamed-lidar** (`top_stream_enc` + `hw_enc.py`):
  **PASS 2026-07-16 (sim + SILICON)** — sim: 480/480 vector words +
  60/60 map-code bytes == golden; silicon: 25/25 REAL spot scans, VSA
  vectors bit-exact vs `hw/ice40/golden.encode_int`, on-chip QPSK map
  codes == `mcode_from_vec`, 85 ms/kf roundtrip (53.4 MHz timing).
  Three bugs found en route, all banked: pl_idx skew (payload byte
  tagged one index late), EBR 2-edge read discipline (map readback
  shifted one component), and the commit-miss during multi-col replay
  (half the points dropped — hence the column cap + commit queue).
- **G4** (with #44): SDRAM-paced at-rate bursts (20 Hz / 30–120 fps),
  no-drop, pacer timing vs stream timestamps.
