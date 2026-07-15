# Learned front-end TRAINING PROGRAM (user directive 2026-07-15)

The goal, stated once: **networks improve the SLAM/map substrate in three
distinct regimes, for BOTH modalities** — and every trained artifact must
fit the ECP5 budget (BNN/int8, streaming, Y8-gray input for vision, 3-ring
range/BEV input for lidar) and pass the real-data gates below.

## Network geometry (user 2026-07-15b): TWO nets, shared trunks, full/half res

Binding constraints on every trained artifact from here on:

1. **Input resolution is pinned to FULL or HALF sensor res** — train AND
   gate at deploy geometry, never toy rasters:
   - vision: Y8 **320×240 (full)** or **160×120 (half)**; external
     datasets (NYUv2 etc.) are resized to these before anything sees them;
   - lidar: the deploy ingest ring-range raster, rings-as-channels —
     **3×1024 (full)** or **3×512 (half)** beams. (BEV grids remain a
     mechanism-study surface only; adoption form is the ingest raster.)
2. **The vision CNN is ONE dual-objective net**: a shared trunk with
   (a) a TRACKING head — per-cell weights (+ thermometer scale cutoff),
   the Regime-A outputs serving ego-motion/place/verify; and
   (b) a **SEGMENTED classification head** — per-cell class → k-bit label
   latent (Regime B). Per-cell is load-bearing, never whole-image: the
   map binds features AT POSITIONS, so only spatially-resolved labels can
   be encoded/queried (objmap/semantic map).
   Joint training: alternate SLAM-sequence batches (tracking losses,
   self-supervised) with segmented-dataset batches (per-cell CE);
   Regime C fine-tunes the same trunk through the VSA decode.
3. **The lidar CNN mirrors it**: shared trunk on the ring raster with a
   tracking head (saliency/weight + cutoff) @20 Hz and a distilled
   per-cell label head (Regime B cross-modal) @keyframe rate.
4. **Rate split on fabric** (verified in cnn_budget.py, this date):
   trunk + tracking head run at FRAME rate (vision full-res 11.3 MMAC =
   fits @120 int8-packed with 2× headroom; half-res 8×; lidar full
   2.5 MMAC @20 trivial); the seg/label heads re-use the latest trunk
   features at KEYFRAME rate (vision 40-class head 16.7 MMAC @5 Hz =
   32× headroom, 14k params EBR-resident). EBR caveat: full+full line
   buffers (13.1 KB vision + 22 KB lidar) + weights ≈ 54.5 of 55 KB —
   edge-exact int8; half-res lidar (11 KB) or BNN weights restores
   margin. Both full and half variants are sanctioned; report budget
   lines for whichever is trained.

## Regime A — end-to-end THROUGH the VSA, for SLAM feature quality

Self-supervised. The net emits per-feature WEIGHTS (and nothing else); the
differentiable SSP encode `Σ wᵢ·exp(iW·pᵢ)` carries the gradient; the loss
is the SLAM objective itself:

- losses: (i) contrastive place (positives = temporally-adjacent /
  own-estimate-near frames + synthetic rotations; negatives = far frames —
  ANTI-ORACLE: adjacency and own estimates only, never reference poses);
  (ii) differentiable registration (synthetic SE(3) on real frames,
  soft-argmax correlation decode, penalize pose error); (iii) verify
  conditioning (6×6 derivative-solve error on adjacent pairs).
- lidar: the TRACKING head of the unified lidar net (geometry section
  above) at DEPLOY scale — k≈2000 (not the k=32 toy), ring raster
  3×1024 full / 3×512 half in; existing `learn_lidar.py` is the seed.
- vision: the TRACKING head of the unified vision net — Y8 full/half
  QVGA in, per-cell weight (+cutoff) out, replacing the
  intensity/gradmag heuristics in gridint/gyro encodes.
- gates: school+classroom place vs the ADOPTED v1.4 recipes at equal D
  (multi-venue-gated 2026-07-15: azel-oct6 D240 = 0.947 classroom
  honest / 0.892 school est; ring-coarse16 D1920 = 0.976 / 0.940 —
  `python3 -m sspax.realbench mv` reproduces); gyro vs 1.13°; verify vs
  1.12°/27.6 mm. Two venues minimum (rule 5).

## Regime B — WITHOUT the VSA: label latents (semantic descriptors)

Supervised on SEGMENTED indoor data (NYUv2 primary, SUN RGB-D scale-up —
per-cell heads, never whole-image; images resized to the pinned deploy
res 320×240 / 160×120 before training):

- vision: the SEG head of the SAME unified net (shared trunk with the
  tracking head — geometry section above): per-cell class embedding →
  binarized **k-bit label vector**. The bits play TWO roles:
    1. semantic KEY — bound into the map (ROLES-binding / bipolar spatter;
       P4b compares the two at equal bits);
    2. **encoding STRENGTH** — the feature's write weight is a function of
       its label bits: `w = w_base · (1 + β·sig(bits))` with `sig` =
       popcount over significance bits (optionally IDF-weighted: rare
       classes localize, wall/floor bits ≈ 0). On-fabric: popcount → LUT →
       shift-add; no multipliers.
- lidar: labels are scarce → CROSS-MODAL DISTILLATION: the trained vision
  net labels pixels; registered depth (TUM today, lidar-projected depth
  after extrinsics) lifts them to 3D points; the LABEL head of the
  unified lidar net (ring-raster input, geometry section) learns to
  predict those labels from GEOMETRY alone. Vision teaches lidar; no
  manual lidar labels.
- gates: seg mIoU sanity; cross-view descriptor stability on TUM (adjacent
  vs far — the failure mode that killed raw census); semantic-map P/R on
  real instances (ScanNet when available).

## Regime C — end-to-end THROUGH the VSA, for OBJECT-FINDING quality

The alternative/complement to B: train the descriptor (+ weight head)
through the FULL objmap pipeline — bind → per-segment bundle → matched-
filter decode — with retrieval losses aimed at the measured walls:

- losses: (i) localization margin (score at true x₀ vs max-elsewhere in
  the segment box — attacks the extreme-value wall directly); (ii)
  cross-view code consistency (same physical cell from different frames →
  matching codes; correspondence via pose+depth, which deploy has); (iii)
  foil contrast (negatives from other scenes).
- curriculum: init from Regime B (semantic pretrain) → fine-tune through
  the VSA decode. A and C can share a trunk with two heads (SLAM-weight
  head + descriptor head).
- both modalities: vision cells AND lidar clusters/cells get descriptor
  bits (via B's distillation) — the semantic map is cross-modal.
- gates: objmap2 harness (TUM fr3, foils fr1_desk) vs banked 0.805
  combined AUC / 0.165 m census multi-view; capacity curve vs
  `semantic.py` synthetic reference.

## Scale-mapped modulation — LEARN the ladder allocation we hand-tuned

The label/latent vector can also map to SCALES (user 2026-07-15): the
feature's write is per-ring modulated,

    contribution_i = Σ_r m_r(latent_i) · w_i · exp(i W_r · p_i)

with `m_r` a small learned head (or a per-bit → per-ring LUT).

**Head form: THERMOMETER output, i.e. learned BLANKING** (user
2026-07-15). `m_r` is not R free values but a monotone thermometer code
over the ladder: the net emits one scalar per feature — the finest
USEFUL scale — and rings finer than the cutoff are BLANKED (optionally a
second scalar for a coarse cutoff → a scale BAND). Why this form:
  - it matches the physics: the banked sensor-coherence-length law
    (λ_min ≈ 2πσ_r — sub-coherence rings are dead bytes) says every
    feature HAS a finest useful scale; the net learns it per feature
    instead of one global hand pick;
  - the house already hand-built the static version twice: the
    smear-matched ladder (objmap) and the span15g thermometer
    MIN_SCALES anti-alias fade (webvis) — this replaces both with a
    learned, feature-conditioned cutoff;
  - 1-2 scalars/feature regularizes and stays interpretable (report the
    cutoff distributions per class);
  - on-fabric it is literally blanking: a comparator per ring
    (ring_idx ≥ cutoff → accumulate), zero multipliers — cheaper even
    than the LUT form.
Training: straight-through estimator on the thermometer step (or a
sigmoid ramp annealed to hard blanking); quantization-aware from the
start.

This subsumes, as special cases, everything the project hand-tuned:
  - m_r constant            = the fixed ladder recipes (oct6, coarse16,
                              half-oct, smear-matched VIS_LAMS — all of
                              lidarscale TUNE was hand-searching this);
  - m_r per CLASS           = walls/floor write coarse rings (place),
                              small textured objects write fine rings
                              (registration/object-finding) — the
                              banked "coarse = place lever, sub-0.5 m =
                              registration-only" law becomes a learned,
                              feature-conditioned policy;
  - m_r per FEATURE         = full scale profiles (regime A can learn
                              per-cell×per-ring weights with no labels).
Decode side gets the same treatment: the scale-split masks and ring
weights in the coarse/fine decode stages (hand-chosen today) can be
learned against the registration/retrieval losses (a Wiener-style
per-ring read weight — the banked Cr² read is the hand-derived cousin).
On-fabric: m_r = per-ring LUT indexed by descriptor bits → shift-add
scaling on the per-ring accumulators; no multipliers, and the per-RING
magnitude store (the verify recipe) already reserves exactly this slot.
Gates: must beat the fixed-ladder recipes at equal D and bytes (school
0.892/0.940, TUM place/verify), and the learned m_r profiles should be
REPORTED (they are interpretable — which scales did each class buy?).

## Shared constraints

- Determinism (fixed seeds), anti-oracle statements per bench, negatives
  banked in docs/RESULTS.md.
- Input geometry pinned (2026-07-15b): vision 320×240 or 160×120 Y8;
  lidar ring raster 3×1024 or 3×512. External datasets resized to these;
  no toy-resolution training runs count toward a gate.
- Budgets printed with every result: params, int8/BNN bytes, MACs (or
  XNOR-popcounts) per frame at QVGA120 / 20 Hz lidar, LUT/DSP estimate.
- HARD per-rate ceilings from the full-board envelope
  (hw/ecp5/host/cnn_budget.py, SLAM reserves subtracted): @120 fps
  22.5 MMAC int8-packed / 3.3 Gbop BNN; @60 45 / 6.6 G; @20 135 /
  19.7 G; @5 Hz 540 MMAC / 79 Gbop. Weights: ≤55 KB EBR-resident;
  ≤8 MB int8 via SDRAM streaming at keyframe rate. BNN-first is the
  search prior; int8 DSPs are the scarce resource.
- Quantization-aware from the start (the map store is 2-3b phase; the
  descriptor is binary by construction).
- Transfer-gate everything through `sspax/realbench.py`-style numpy runs
  on the deploy box before a recipe changes.
