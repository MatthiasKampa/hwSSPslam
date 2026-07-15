# sspax — JAX SSP lattice algebra + the approximate-permutable sphere

An **efficient experimentation surface** for the 6-DoF lattice question, not a
shipped deliverable. Two things live here:

1. **A JAX re-implementation** of the SSP encode / rotate-by-permutation /
   SO(3)-decode algebra, bit-faithful to the numpy `experiments/lattice3d.py`
   and `sspslam/` conventions (float64), running under `jit`/`vmap` on the
   accelerator. The SO(3) grid decode goes from **695 ms/cloud (numpy)** to
   **1.0 ms/batch** of 64 clouds × 343 candidates.
2. **`ringstag3d`** — the *approximate-permutable Fibonacci-sphere substitute*
   the 2026-07-15 directive asked for.

Everything imports the frozen core (rule 1); nothing here is tuned against a
reference (rule 2 — the benches are self-contained synthetic geometry).
GPU footprint is capped at ≤45 % VRAM (`XLA_PYTHON_CLIENT_MEM_FRACTION`) so a
second agent can share the card.

## The idea: rings with a half-offset stagger

`fib3d` covers the sphere isotropically (points ~equidistant) but a yaw is **not**
an index permutation of it. `azel3d` keeps the exact yaw permutation but wastes
directions near the poles (constant azimuth count per band). `ringstag3d`
reconciles them, per the directive *"structure a sphere out of rings with half
offset between layers to have approx equal distance between points"*:

- latitude rings on the upper half-sphere, **full 2π azimuth** per ring
  (antipode = phasor conjugate, free);
- per-ring azimuth count ∝ ring circumference `cos(elev)`, apportioned to an
  exact direction budget → ~constant nearest-neighbour arc (fib3d's virtue);
- consecutive rings **staggered by half an azimuth step** (brick-laying) →
  tighter packing;
- each ring is equal-azimuth over a full circle → a yaw by a multiple of that
  ring's step is an **exact cyclic shift**; rings have different counts, so a
  common yaw is *approximately* a permutation (per-ring integer snap, residual
  ≤ π/n_az). `yaw_perm` returns that permutation; `yaw_residuals` + the stored
  `encode_dvda` derivative vector give a first-order sub-step correction.

## Findings (all `python3 -m sspax.bench <name>`; see docs/RESULTS.md 2026-07-15)

| bench | headline |
|-------|----------|
| `parity` | JAX core = numpy `lattice3d` to 4e-16 (encode, perm, q_polar) |
| `uniform` | **ringstag3d is the MOST equidistant layout**: cv 0.081 vs fib3d 0.102, azel3d 0.111; largest min-arc 12.7° |
| `rot` | yaw fidelity at D=240 is coarse (~0.82) — the cost of spreading 60 dirs over a 2-sphere; azel3d/az2d stay exact only at lattice angles, ringstag3d beats azel3d at off-lattice 5° |
| `resolution` | ringstag3d yaw → **0.94 @ D=960, 0.99 @ D=1920**; the d/dα derivative correction crosses from hurting to helping at finest-step ≈6°; **gated** correction never hurts |
| `so3` | 2D `az2d` collapses on 3-axis rotation (16.7° vs grid floor 3.5°) — out-of-plane needs a 3D lattice; ringstag3d competitive (7.4°) but azel3d (4.7°)/rand3d (4.2°) edge it at this coarse grid |
| `disp` | all 3D lattices recover z-inclusive translation at the grid limit (0.079 m); az2d is blind to dz (0.15–0.29 m) |
| `speed` | JAX 1.0 ms/batch vs numpy 695 ms/cloud |

`--quant nph,nmag` adds the FPGA polar-quantized column (16ph/4mag costs ~1–2°
on SO(3) decode, ~0.05 on yaw cos — the storage model survives).

## Sweep + learned front-end (2026-07-15)

- `sspax/sweep*.py` — the 6-batch formulation sweep (lidar / camera / fusion /
  resolution / quant / verify). Winner: permutable const-ring + oct6 ladder.
  See `sspax/SWEEP_PLAN.md` + `sspax/SWEEP_RESULTS.md`.
- `sspax/learn_lidar.py` — differentiable learned lidar saliency (826-param CNN,
  end-to-end for place-rec; +0.10 AUC at a sparse k=32 budget).
- `sspax/semantic.py` — binary-descriptor binding → **queryable map**
  ("highlight the chairs": P=R=1.0, 4 cm, within capacity).
- `sspax/vision/tinycnn.py` — FPGA-fittable vision CNN + 64-bit descriptor head,
  pretrained on CIFAR-100 (whole-image; SUPERSEDED for the map by segnet below).
  See `sspax/LEARNED_FRONTEND.md`.
- `sspax/realbench.py` — pure-numpy REAL-DATA transfer gates (school/tum/
  verify/mv/preset), runs on the no-JAX deploy box.
- `sspax/headio.py` — the unified-net HEAD EXPORT CONTRACT v2 (int8 npz +
  pure-numpy forward; v2 mirrors the trained segnet/lidar_ring shapes —
  desc stack, track cout 1|2 (cutoff retired), in_div lidar-meters,
  seg optional until distillation) + deploy-box gates: contract selftest,
  cross-view bit stability, objmap2 semantic/appearance-key harness, with
  measured random-weights baselines for both. Train anywhere, gate here.
- `sspax/ladder_extent.py` — extent×λ_max mechanism study behind the
  venue-adaptive ladder preset (`sspslam/lattice_presets.ladder_of_extent`).

## Deploy front-ends + bounded queryable map (2026-07-15b)

Pinned network geometry (TRAINING_PROGRAM.md "Network geometry"): TWO unified
dual-objective CNNs at deploy resolution, each a shared trunk + a tracking head
(per-cell saliency/cutoff/descriptor) + a per-cell SEGMENTED-label head. Both
emit a SPATIAL map (never a global descriptor) so features bind at positions.

- `sspax/vision/segnet.py` — the vision net: Y8 luma 160×120 → /4 trunk
  30×40×64, seg (40-class) + tracking (desc) heads. ~22 MMAC/frame, 2.3 KB BNN,
  exact numpy-forward parity, CReLU ≈ 55 % of ReLU params at matched width.
- `sspax/lidar_ring.py` — the lidar mirror on the deploy ingest RING raster
  (3×1024 rings-as-channels, 1D-azimuth trunk; yaw = exact azimuth ROLL).
  4.1 MMAC/frame, 1.9 KB BNN. BEV is now mechanism-study only.
- `sspax/vision/objmap_nyu.py` — **Regime C**, the queryable map end-to-end:
  segnet per-cell class → back-projected to 3D → bound into the bounded SSP
  map → query a class → objects light up ("highlight the chairs"). On real
  NYUv2, object-level round-trip AUC **~1.0** (0.799 under a z-flattened query);
  a random-code control collapses to 0.49 → the map is genuinely SEMANTIC.
- `sspax/semantic_transform.py` — the queryable map is also ALGEBRAICALLY
  TRANSFORMABLE: translation = map ⊙ exp(iW·t) (bit-exact, any t), rotation =
  index permutation (exact on-lattice). A graph correction moves the whole map
  (objects + class bindings) with one O(D) op; queries still localise in the new
  frame. Unifies queryable + transformable on one bounded vector.
- `sspax/semantic_capacity.py` — the bounded-map CAPACITY trend: capacity is
  SUBLINEAR in D (empirical ~0.76·D^0.50, but the exponent is
  resolution-confounded — not a physics constant; only "sublinear" is
  load-bearing). One giant superposition scales poorly — so the bounded-memory
  win comes from TILING (per-segment maps), reinforcing the shipped architecture.
- `sspax/learn_scale_real.py` — thermometer scale head on real lidar (P1). Its
  "+0.183 place gain" was RETRACTED (rule-4 audit: a self-rotation-metric
  artifact; the honest adjacent control was flat). Reconfirms the loop-closure
  wall — the front-end cannot manufacture place separability.
- `sspax/ladder_rule.py` — densified venue-scale ladder rule: the extent→λ_max
  law is monotone (Spearman 0.9-0.97) but NOT linear; effect is large only at
  building scale. Recommend coarse venue-bucketing, not a formula.
- `sspax/semantic_capacity.py`/`semantic_lattice.py`/`semantic_quant.py`/
  `semantic_transform.py`/`dual_use.py` — the bounded map characterized: capacity
  sublinear (→tile) + SIGNIFICANCE allocates it (9× important-object protection);
  geometry-FREE but LADDER-contested (per-role ladders); survives FPGA polar-quant
  at ~4 bits/cell; place+semantic dual-use in one vector (one-directional). All
  positives audited (rule 4).

Honest place is scoreable on the CLASSROOM venue (withheld odometry, real revisit
pairs): the banked recipes reproduce cross-box (azel-oct6 D240 0.947,
ring-coarse16 D1920 0.976), and a DIRECT learned-vs-fixed test shows NO learned
place gain — reconfirming the loop-closure wall on the honest venue.

## Verdict

The directive's hypothesis **holds in principle**: the staggered-ring sphere is
simultaneously the most isotropic layout *and* per-ring exactly (globally
approximately) yaw-permutable. The quantified **cost** is resolution: at the
shipped D=240 the approximate yaw is coarse (0.82); it becomes accurate (>0.98)
only at D≈2k or with the gated derivative correction. It does **not** beat
azel3d on SO(3) *decode* accuracy — its win is uniformity, not decode. So it is
the layout of choice when isotropic coverage matters and a larger direction
budget is affordable; azel3d remains preferable when exact common-yaw at a tiny
D is the priority.
