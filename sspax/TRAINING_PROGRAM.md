# Learned front-end TRAINING PROGRAM (user directive 2026-07-15)

The goal, stated once: **networks improve the SLAM/map substrate in three
distinct regimes, for BOTH modalities** — and every trained artifact must
fit the ECP5 budget (BNN/int8, streaming, Y8-gray input for vision, 3-ring
range/BEV input for lidar) and pass the real-data gates below.

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
- lidar: SaliencyNet at DEPLOY scale — k≈2000 (not the k=32 toy), 3-ring
  ingest raster in; existing `learn_lidar.py` is the seed.
- vision: CellWeightNet — Y8 QVGA in, per-c16-cell weight out, replacing
  the intensity/gradmag heuristics in gridint/gyro encodes.
- gates: school+classroom place vs the banked/attributed recipes at equal
  D (azel-oct6 D240 0.892, ring-coarse16 D1920 0.940); gyro vs 1.13°;
  verify vs 1.12°/27.6 mm. Two venues minimum (rule 5).

## Regime B — WITHOUT the VSA: label latents (semantic descriptors)

Supervised on SEGMENTED indoor data (NYUv2 primary, SUN RGB-D scale-up —
per-cell heads, never whole-image):

- vision: per-c8/c16-cell class embedding → binarized **k-bit label
  vector**. The bits play TWO roles:
    1. semantic KEY — bound into the map (ROLES-binding / bipolar spatter;
       P4b compares the two at equal bits);
    2. **encoding STRENGTH** — the feature's write weight is a function of
       its label bits: `w = w_base · (1 + β·sig(bits))` with `sig` =
       popcount over significance bits (optionally IDF-weighted: rare
       classes localize, wall/floor bits ≈ 0). On-fabric: popcount → LUT →
       shift-add; no multipliers.
- lidar: labels are scarce → CROSS-MODAL DISTILLATION: the trained vision
  net labels pixels; registered depth (TUM today, lidar-projected depth
  after extrinsics) lifts them to 3D points; a small point/BEV net learns
  to predict those labels from GEOMETRY alone. Vision teaches lidar; no
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
