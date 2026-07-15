# 6-DoF formulation sweep — results (2026-07-15)

Six insight-driven batches (each folded into the next), JAX/GPU, float **and**
quantized, deterministic synthetic geometry (anti-oracle), ≤45% VRAM.
Raw CSVs: `scratch/sweep_b{1..6}_*.csv`. Runners: `sspax/sweep*.py`.

## Headline

The **permutable staggered-ring sphere** — equal-azimuth-per-ring, arc-spaced
elevations, half-azimuth stagger, ~6 rings (`ring_arc_const_s0.5_r6`) — on a
**wide octave ladder (0.25–8 m, oct6)** is the **cross-modality winner**
(lidar + camera): best SO(3) rotation decode (**3.0°**, tightest p90 6.0°), best
place-recognition (**0.910**, 90% CI [0.861, 0.956]), and near-invariant under
aggressive quantization. It beats the more-isotropic cos-apportioned ring (which
wins *pure uniformity* but carries decode outliers, p90 11.5°) and the coarser
`azel3d` baseline.

## Batch-by-batch

**B1 — lidar geometry (D=240, 44 formulations × float/quant).**
- SO(3) decode is nearly **geometry-independent** (3.0–4.5°); `rand3d` (worst
  uniformity, cv 0.43) has the *best* decode (3.0°). **Uniformity does not
  predict decode.**
- Uniformity ↔ permutability is a real trade: `cos` apportionment wins
  uniformity (`ring_arc_cos_s0_r4` cv **0.043**), `const` apportionment (more
  azimuths on the polar rings) wins native yaw (`…_const_s0.5_r6` yawRing
  **0.919**).
- Translation decode is **0.074 for every 3D layout** (grid-limited, not a
  discriminator); `az2d` is blind to dz (0.15–0.29).

**B2 — camera geometry (S² bearings).**
- `ring_arc_const_s0.5_r6` wins camera on **all three** metrics (so3omni 3.1°,
  placeOmni 0.857, placeNarrow 0.662) — camera *rewards permutability*, unlike
  lidar's decode-flatness, because rotation-searched appearance matching leans
  on the ring yaw algebra.
- Omni (360°) place-rec is strong (0.83–0.86, rotation-equivariant — the
  sphere lattice's home turf); narrow FOV is the hard case (0.64–0.66).

**B3 — cam+lidar fusion (shared lattice, α-sweep, bundle vs concat).**
- Geometric-only cam & lidar are **redundant** → naive α=0.5 bundle *dilutes*
  below lidar-alone; `concat` (2×D) is worse than `bundle` (D). With the best
  geometry and a narrow FOV, fusion gives a **small real lift** (lidar 0.892 →
  fused **0.909**).
- Deep link to the project thesis: fusion pays only with an **independent cue**
  (appearance/texture), exactly the "independent absolute cue" of
  `docs/FINDINGS.md` §5–6. A geometric camera is just another view of the same
  geometry. Bundling is **memory-free** (same D) — the system win stands
  regardless.

**B4 — resolution + scale ladder.**
- Resolution lifts yaw strongly (`ring_arc_cos` 0.82→0.99 as D 240→1920);
  place-rec **saturates ~0.86** (content-limited, not resolution-limited).
- A **wider octave ladder helps place-rec** (`oct6_.25-8` best for every
  geometry: const-ring **0.901**); a camera-tight high-frequency ladder
  (`fine_.25-1`) hurts (loses coarse structure).

**B5 — quantization / deploy (bytes/anchor).**
- Place-rec is **near quant-invariant**: `ring_arc_const` 0.857→0.844 at
  **94 B/anchor** (4ph/2mag). SO(3) decode degrades gracefully (3.7→5.2°).
- **Yaw registration is the bit-sensitive metric** (0.864→0.649 at 4ph/2mag).
- Deploy points: **place-recognition → 4ph2mag (94 B)**; **fine registration →
  16ph4mag (184 B)** (so3 3.7°, yaw 0.817). `ring_arc_const` dominates at every
  budget.

**B6 — verification (fine ±12° @ 2° SO(3) grid, 2197 cands; bootstrap CIs).**
- `ring_arc_const_s0.5_r6` + oct6: so3 **3.03°** (p90 6.05, the tightest tail),
  place **0.910 [0.861, 0.956]** — CI lower bound clears the others' means.
- The oct6 ladder lift is consistent (+0.03–0.04 place-rec across geometries);
  quant (16ph4mag) is negligible on place-rec, +0.5–1° on decode.
- Honest caveat: place-rec geometry CIs **overlap** (±~0.05) — the *geometry*
  differences are modest; the *ladder* effect and the const-ring's tail
  behaviour are the robust signals.

## Verdict → recommended formulation

| axis | choice | why |
|------|--------|-----|
| directions | staggered rings, **const** azimuth/ring, arc elevations, ½ stagger, ~6 rings | best rotation decode + place-rec across lidar & camera; permutability tail |
| ladder | **oct6 (0.25–8 m)** | consistently best place separability |
| rotation | ring-native snap (quant) / snap + gated d/dα (float) | continuous yaw on a frozen vector |
| fusion | shared-lattice **bundle** (not concat), small α, only with an independent cue | memory-free; geometric channels are redundant |
| deploy | 4ph2mag/94 B for place-rec, 16ph4mag/184 B for registration | place-rec is quant-robust; registration is bit-sensitive |

The more-uniform `ring_arc_cos` (the earlier default `ringstag3d`) remains the
choice when *isotropic coverage* is the goal; `const` wins when **rotation
decode + place-rec** are the goal — which for a SLAM front-end they are.
