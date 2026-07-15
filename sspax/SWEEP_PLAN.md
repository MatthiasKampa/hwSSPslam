# 6-DoF formulation sweep — architecture & batch plan (2026-07-15)

Run in **~10-min batches**; read each CSV, fold the winners into the next batch.
All synthetic + seeded (anti-oracle), GPU ≤45% VRAM, float **and** quantized.

## Algorithmic components under test
- **Lattice geometry** (directions u on S²): ring-sphere family
  {spacing arc|area} × {apportion cos|const} × {stagger 0|½} × {n_rings} +
  references fib3d / rand3d / az2d / azel3d. Shared backbone for all modalities.
- **Scale ladder** (λ): n_scales, ratio, min/max. Lidar λ = spatial wavelength
  (metres, resolves translation); camera λ = angular concentration on S²
  (|b|=1 ⇒ 2π/λ is an angular frequency).
- **Rotation representation**: nearest-direction `perm_of` (fair, all layouts) ·
  ring-native per-ring snap · snap + gated d/dα derivative (float only —
  derivative doubles storage, so quant uses snap alone).
- **Quantization**: float vs (phase-bits × mag-levels); bytes/anchor.
- **Fusion**: v = α·φ_cam + (1−α)·φ_lidar on a **shared** lattice (one
  permutation rotates both). Bundle (add, same D — memory-free) vs concat (2D).

## Modality geometry
- **Lidar**: R³ points, range-bearing; translation resolvable; metrics =
  uniformity · yaw/SO(3) decode · translation decode.
- **Camera**: S² bearings in a forward FOV cone (synthetic pinhole over the same
  rooms); rotation-dominated, no translation; metrics = SO(3) decode · place
  separability (appearance). FOV-vs-isotropy is a live question.
- **Combined**: shared-lattice bundle; metrics = place separability & SO(3)
  decode vs either channel alone; α sweep; bundle-vs-concat.

## System / deploy metrics
D (vector length), bytes/anchor under quant, encode MACs, rotation-search cost
(perm gather O(D) vs re-encode O(M·D)), derivative-vector storage cost.

## Batches (each ≈10 min, insight-driven)
1. **lidar geometry** @ D=240, float+quant: uniformity, yawNN 5/13°, yawRing,
   SO(3) decode, translation decode. → top geometries.
2. **camera geometry** on synthetic bearings: SO(3) decode + place separability;
   FOV-matched elevation range vs full isotropy. → cam geometry winner.
3. **combined fusion**: shared lattice (batch-1/2 winner), α ∈ [0,1], bundle vs
   concat; place separability & SO(3) decode of the fused vector.
4. **resolution / scale ladder** on winners: n_dir {60…480}, n_scales, ratio.
5. **quant / deploy**: phase-bits × mag-levels × derivative on/off; bytes &
   compute costing on the end-to-end winners.
6. **verification**: fine SO(3) grid + multi-seed CIs on the top 2–3
   lidar/cam/combined configs, float+quant. Honest negatives kept.

CSV per batch in `scratch/sweep_b<k>_*.csv`; distilled table → docs/RESULTS.md.
