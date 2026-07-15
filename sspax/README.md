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
  pretrained on CIFAR-100. See `sspax/LEARNED_FRONTEND.md`.
- `sspax/realbench.py` — pure-numpy REAL-DATA transfer gates (school/tum/
  verify/mv/preset), runs on the no-JAX deploy box.
- `sspax/headio.py` — the unified-net HEAD EXPORT CONTRACT (int8 npz +
  pure-numpy forward) + deploy-box gates: contract selftest, cross-view
  bit stability, objmap2 semantic-key harness. Train anywhere, gate here.
- `sspax/ladder_extent.py` — extent×λ_max mechanism study behind the
  venue-adaptive ladder preset (`sspslam/lattice_presets.ladder_of_extent`).

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
