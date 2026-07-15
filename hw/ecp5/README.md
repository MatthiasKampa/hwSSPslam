# ECP5 track — Icepi Zero: full-rate lidar + camera front end

Opened 2026-07-14 (user directive). Target: **Icepi Zero v1.3**
(<https://github.com/cheyao/icepi-zero>) — Lattice **ECP5 LFE5U-25F**
(CABGA256), 24k LUT4, ~112 KiB EBR, 28 MULT18 DSPs, **32 MB SDRAM**,
16 MB QSPI flash, 50 MHz oscillator (pin `M1`), 5 user LEDs, Pi-Zero
GPIO header, 3× USB-C, GPDI video. Build flow: yosys `synth_ecp5` →
`nextpnr-ecp5 --25k --package CABGA256` → `ecppack` (all in
oss-cad-suite); flash/run via `openFPGALoader -b icepi-zero` (board
upstream-supported). Pin constraints: `icepi-zero.lpf` (vendored
verbatim from the board repo).

## Charter

The iCE40 UP5K track (`hw/ice40/`) proved the VSA datapath on silicon at
deploy scale (encoder + matcher + argmax bit-exact; standalone v7
sim-gated) but is resource-walled: 5280 LC, 8 DSP, 128 KB SPRAM, one 2D
ring slice in, no headroom for a second sensor. The ECP5 version is the
**sensor-fusion front end**:

1. **Full lidar ingest** — the complete 1024×64 Ouster-class cloud @
   20 Hz (65536 pts × 2 B = 128 KB/frame = 2.6 MB/s), not just the
   ring-33 slice. The 32 MB SDRAM is the frame/reservoir store the UP5K
   never had; EBR holds the hot window. Ring-slice + fold logic moves
   on-chip (the UP5K golden consumed a host-parsed slice).
2. **Camera ingest** — an **OV5640** wired to the FPGA (later): DVP
   8-bit parallel mode (PCLK/HREF/VSYNC + D[7:0], SCCB/I²C config,
   XCLK from a PLL) on the Pi-Zero GPIO bank. Until the sensor is
   attached, the DATASET's D455 RGB frames stand in (640×480 JPEG,
   intrinsics in-band; `runners/spot_cam.py`).
3. **Budget-matched modalities** (the design rule this track starts
   from, user: "lidar and cam have approx the same amount of data"):
   the camera contributes **the same order of data per keyframe as the
   lidar**, by construction —

   | stream | per 5 Hz keyframe | rate |
   |---|---|---|
   | lidar 2D slice (matcher space) | 1024 beams × 2 B = 2 KB, ~937 valid pts | 40 KB/s @20 Hz |
   | lidar full cloud (fidelity/ingest) | 128 KB | 2.6 MB/s @20 Hz |
   | camera raw 640×480 gray | 307 KB | 9.2 MB/s @30 Hz |
   | camera after 2×2 bin → 320×240 | 76.8 KB (≈ full cloud / 1.7) | 2.3 MB/s |
   | **camera after FAST-9 + target-count** | **~937 features × 6 B ≈ 5.6 KB** (point parity with the lidar slice) | ~28 KB/s |

   Feature count is servoed to the per-keyframe **valid-lidar-beam
   count** (adaptive threshold), so downstream fusion (VSA bundling of
   bearing-weighted feature phasors next to range points) sees two
   point sets of the same cardinality.
4. **Detector = integer FAST-9** (radius-3 Bresenham circle, ≥9
   contiguous, comparator-only score, 3×3 NMS): line-buffer streaming
   RTL, no multipliers — the OV5640 pixel clock streams straight
   through it. `host/golden_cam.py` is the bit-exact golden model
   (integer luma `(77R+150G+29B)>>8`, 2×2 box bin, fixed-t detect);
   `rtl/fast9.v` must match it pixel-for-pixel (mask + score, pre-NMS;
   NMS is host/deploy-side v0, RTL v1 follow-up). Deploy threshold
   control = ±1/frame feedback toward the target count (bounded logic);
   the dataset extractor uses per-frame binary search (deterministic).
5. **Frame-rate constraint (user, 2026-07-14): run at the OV5640's MAX
   frame rate.** Operating point: **QVGA 320×240 @ 120 fps** — natively
   the binned working resolution (binning stage removed), pixel clock
   only ~9–12 MHz (vs 27.6 MHz VGA@90 / 55 MHz 720p@60), 24× keyframe
   temporal rate for tracking/rehearsal. T-corner at 120 fps:
   comparator class (fast9/extrema/edge) and shi-tomasi (3 MULT18 at
   pixel clock — DSP count is fps-invariant) are FREE; the 3-layer
   int8 micro-CNN tier ≈ 840 MMAC/s ≈ 60% of the 28-MULT18 budget
   (time-multiplexed, competes with the encoder) — quality-reference
   only; YOLO-nano-class (100+ MMAC/frame → 12+ GMAC/s) is OUT on this
   part. Preprocessing must be single-pass streaming; feature stream at
   120 Hz ≈ 0.7 MB/s (downstream keyframing decimates/accumulates).

## Architecture (2026-07-14 ultrathink; measured basis in RESULTS "4h block")

**Two maps of different KINDS, not just different tunings.** The phase
algebra transforms range points (translate + rotate) but not camera
appearance (bearings rotate; translation = parallax, untransformable on a
frozen vector). Therefore:
  - lidar map  = transformable world-frame bounded segments (existing
    BoundedSLAM lifecycle) on the scaled azel3d lattice (D960-class:
    place-AUC 0.86 vs 0.70 @ D240; 3 rings + ~2k pts of ingest suffice);
    full 6-DoF on frozen vectors (rot decode 3.9 deg, transl at grid
    floor).
  - vision map = per-anchor APPEARANCE SNAPSHOT library (bounded: const
    bytes/anchor), queried by max-similarity — yaw revisits still match
    via exact bearing permutation; parallax deliberately not modeled.
**Vision = two services at two rates:** (1) 120 fps rotation/ego-motion
aid (bearing-permutation decode between frames — a VSA visual gyro;
adj-rep 0.97), (2) keyframe-rate dense place channel. **Fusion = late,
score-level** (two-map): max-rule at the aliasing wall (school 0.798 >
lidar 0.720), vision-weighted sums under view overlap (classroom 0.954).
**One datapath serves both modalities:** the winning dense vision
encoders are per-cell `w * exp(i(W_dir . bearing + appearance-phase))` —
the lidar encoder's cis-ROM + complex-accumulate engine with a bearing
lattice and an appearance phase term (~134 M cis-MAC/s @ 1200 cells x
D240 x 120 fps — lidar-encode order). Dense vision needs NO feature
extractor on-chip; the sparse detector path (fast9.v) remains for the
ego-motion service where sparse+NMS is the cheap option.
**Encoder family = appearance bound IN PHASE** (weights alone cannot
separate same-bearing-different-appearance): intensity-phase (predicted
exposure-fragile), gradient-orientation phase (HOG-in-VSA; 8-direction
integer quantization = comparator-only), cell-census phase (8 neighbor
comparisons -> code -> binary phase decomposition; illumination
invariant — predicted the deploy winner). All integer, streaming,
cis-ROM addressable.

## Deploy variant v1 (2026-07-14 coherence; every number banked in RESULTS)

**Sensors → services**
- Lidar 1024×64 @ 20 Hz: ingest **3 rings (16/33/50), ~2k pts** (≈1/40
  bandwidth; quality-neutral). Cloud gyro at sensor rate (GN×2 on the 6
  analytic derivative vectors; 1.34° on 2.5° steps at 5 Hz test rate,
  better-conditioned at 20 Hz). Keyframes @ 5 Hz.
- OV5640 QVGA @ 120 fps: **gridint c16** (300 intensity-weighted cells —
  the cross-view place channel, ~free) at keyframes; **intphase** as the
  precision/verification channel; gradmag/intphase **visual gyro** @
  frame rate (1.8–2.0° on 2.6–3.0° steps). fast9.v sparse path retained
  (bit-exact RTL v0, 72 MHz); shi-tomasi v1 filed.

**Maps (two kinds, the asymmetry law)**
- Lidar: shipped SE(2) BoundedSLAM backend (acceptance-gated) + a
  per-anchor 3D place/verification layer on **azel3d D1920 (24az × 5el
  × 16 coarse √2 rings, 0.5–90.5 m)** — school place-AUC **0.928** at
  the 3-ring/2k-pt operating point (v1.1 ceiling recipe; sub-0.5 m
  rings are registration-only, building-scale rings are the place
  lever; curve saturates ~0.93). 2-bit store ≈ 480 B/anchor.
- Vision: per-anchor snapshot library at **D120–240** (vision does not
  scale with D): gridint V + intphase V + 6 derivative vectors for
  SE(3) checks ≈ ≤0.5 KB/anchor at 2 b. Bounded per anchor, like
  everything else.

**Fusion policy**
- Candidates: max-rule over z-normed {lidar rot-searched sim, vision
  gridint sim} — 0.798 at the aliasing wall vs 0.720 lidar-alone.
- Verification: intphase precision match under view overlap + a 6-DoF
  consistency check (short-baseline SE(3) decode against the anchor
  snapshot — the content-consistent regime, where the decode stack is
  near-exact: 0.29°/3 mm GN on school clouds). Wide-baseline SE(3) is
  content-overlap-limited (~6–9°/0.17 m) — treated as a coarse gate
  only, never as registration.
- Ego-motion: cloud gyro (20 Hz) + visual gyro (120 fps) feed the CV
  guess / odometry prior.
- The reverse-view wall is crossed only by depth-lifted 3D visual
  landmarks (TUM rev-AUC 0.941; graceful to 25% depth coverage) —
  **gated on calibrated lidar↔camera extrinsics, the standing unlock.**

## Headroom allocation (2026-07-14, user-directed; the part is bigger than v1)

Deploy v1 uses roughly half the LUTs, half the MULT18s, little EBR and
almost none of the 32 MB SDRAM. The measured/banked value curves say the
headroom goes to (in current expected-value order):

1. **Better map (place + objects).** The lattice-D law keeps paying:
   D3840 with a LONG √2 ladder (0.125–22.6 m incl. building-scale
   coarse rings) hits **0.915** school place-AUC (vs 0.858 @ the v1
   D960) at 960 B/anchor @2 b — SDRAM holds tens of thousands of
   anchors. The same ladder lever lifted 2D map EXTRACTION (banked
   fidelity-space thread: half-oct+coarse = +0.05..+0.10 AUCgt, ghost
   combs annihilated) → a D3840-class fidelity layer is the "find
   objects in the map" substrate: line/corner-atom pursuit + driven-
   path veto (decode.py machinery) over a sharper PSF. Ceiling-tuning
   sweep in flight (ladder composition × elevation distribution).
2. **History samples.** 32 MB SDRAM = ~16k raw ring-scans (2 KB each)
   — the reservoir thread at its natural scale: sparse-cadence
   rehearsal (school 10/10 benign, median wins banked; collapse is
   per-environment and cadence e64-class was safest), anneal-style
   in-basin refolds, and — the wall-relevant one — RE-VERIFICATION of
   loop candidates against RAW stored scans (independent of the map's
   own gauge; the commit-policy wall applies to selection, but a
   candidate-gated re-match against history is a verification signal
   the UP5K never had memory for). Multi-session map storage likewise.
3. **Better feature detection.** shi-tomasi v1 (3 MULT18) is the
   default; the freed multipliers make the trained tiny-CNN head
   affordable (~60% MULT18 @ QVGA120, less at reduced detector fps) —
   banked as the quality-reference tier with headroom (+0.03 with
   RANDOM weights); train offline, weights via flash.

## Status

- [x] Toolchain smoke: `make TOP=top_fast9 build` — synth+PnR+pack on
      LFE5U-25F, timing report vs 50 MHz.
- [x] `rtl/fast9.v` v0 streaming detector, bit-exact vs golden vectors
      (`make sim`).
- [x] **FIRST SILICON (2026-07-14, board-arrival day)**: fast9 UART gate
      HW PASS — 3364/3364 centres bit-exact on the Icepi Zero
      (`top_fast9_uart` @ 2 Mbaud, magic-armed protocol;
      `host/hw_fast9.py`). FT231X first-byte + TX-FIFO-overflow defect
      classes documented in the ledger.
- [x] Dataset camera pipeline: `python3 -m runners.spot_cam parse
      school_run2` → `cam_features.npz` aligned to lidar keyframes.
- [ ] OV5640 DVP capture front (SCCB init ROM, PCLK domain, async FIFO).
- [ ] Full-cloud lidar ingest + SDRAM frame store (UART/USB feed first,
      sensor later).
- [ ] Encoder port (iCE40 v6/v7 cores; SPRAM → EBR/SDRAM re-plumb —
      the map store and mc codes lived in SB_SPRAM256KA).
- [ ] Fusion experiment (Python first, PROTOCOL rules): bearing-only
      feature phasors bundled beside range points — does the camera
      channel help exactly where 2D lidar aliases (the wall)?

Numbers-only discipline applies (no board attached yet — sim + PnR
reports only; hw-replay gates when hardware arrives, mirroring the
iCE40 track's staging).
