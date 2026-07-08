# hwSSPslam — bounded-memory 2D lidar SLAM with Spatial Semantic Pointers

A SLAM system whose map is not a grid and not a point cloud: every 5-keyframe
segment of trajectory is a **fixed-size complex Fourier-feature vector**
(a Spatial Semantic Pointer / VSA hypervector) on a polar frequency lattice.
The algebra does the work:

- **translation** of map content = elementwise phase multiplication,
- **rotation** = an exact index permutation of the lattice (plus a stored
  d/dθ *derivative vector* for sub-grid angles — first-order Lie correction),
- **matching** = an inner product (the correlation of scan and map densities),
- **map updates & graph corrections** = vector addition and O(D) phase ops —
  nothing is ever re-encoded, and **no sensor history is stored anywhere**.

> **Status: active research.** This repository is a live snapshot of an
> ongoing agentic research session; more results land frequently. The full
> experiment ledger — every finding, negative result, retraction, and
> reviewer audit — is [`RESULTS.md`](RESULTS.md). Read it as a lab notebook
> with an abstract on top.

## Headline numbers (deterministic, reproducible)

ATE rmse vs RBPF-corrected references; `ssp_bounded_carmen.py <log>`:

| log | ours | raw odometry | map memory | speed |
|---|---|---|---|---|
| Intel Research Lab (full) | **2.44 m** (median 1.55) | 24.2 m | 5–8 MB | 15–17 ms/kf |
| Freiburg 101 (held-out) | **1.88 m** (median 1.55) | 8.56 m | 1.9 MB | 27 ms/kf |
| Freiburg 079 (no retune) | 5.52 m | 14.4 m | 3.5 MB | 30 ms/kf |
| ACES3 Austin (no retune) | 6.21 m | 5.41 m | 5.3 MB | 13 ms/kf |
| Belgioioso Castle (held-out) | 2.64 m | 1.72 m | 1.6 MB | 27 ms/kf |
| MIT Infinite Corridor, 1.9 km (held-out) | 42.66 m (38–58 band) | 189 m | 27 MB | 12–21 ms/kf |

*Transfer caveat:* Intel/fr079/ACES are **not** held out — the shipped config was
selected by minimizing the worst-of-three ATE ratio over those three, so they are
the selection set (run with no per-log retune, but not unseen). **fr101,
belgioioso, and MIT are the genuinely held-out logs**, all zero-retune.

In-repo baselines on identical parsing/keyframing/eval (`baseline_*.py`):
ICP+graph 1.70 m (35 MB, retains all scans), correlative grids 3.27 m,
RBPF/GMapping-lite **0.12 m** (56 MB peak, particle grids). *Caveat on the
0.12 m:* the ATE reference (`intel.gfs.log`) is itself GridFastSLAM (GMapping,
an RBPF grid SLAM) output, so the RBPF baseline is scored against its own
algorithm family and its number is **partly self-referential** — not
head-to-head comparable with the non-RBPF methods. The mutually comparable,
cross-family numbers are **ours 2.44, ICP 1.70, CSM 3.27** (all "distance to a
GMapping estimate"); treat RBPF's 0.12 as "reproduces the reference," not a
14× accuracy win. The honest positioning: this project's contribution is the
**representation** —
**O(area)-bounded** (not O(time)), **history-free**, **algebraically
transformable** maps at usable accuracy — not state-of-the-art accuracy, and
not raw spatial compactness. Per m² the SSP map is actually *denser* than an
occupancy grid (~3 KB/m² vs ~0.4 KB/m²); the memory win holds only against
*history-storing* baselines (ICP's retained scans are O(time); RBPF is
particles × grids). The defensible properties are the bound, the absence of
history, and the algebra — verified: Intel plateaus at 698 segments (84 % of
the cell-cap ceiling), MIT grows dead-linear at 1.26 seg/m as new corridor is
exposed. fr101 (a dense-revisit building, 1.88 m held-out) is where the closure
machinery pays off most; MIT demonstrates the revisit-density limit at scale.
ACES and belgioioso are frank negatives of a *specific* kind — logs whose
odometry is already excellent, where the scan-matching **frontend** (not loop
closure) is what costs accuracy; see RESULTS.md "the frontend do-no-harm gap".

## How it works (one screen)

**Encoder** (`ssp_slam.py`): scans → occlusion-filtered segment samples →
φ(p) = exp(iWp) on a lattice of 4 octave rings (λ = 0.25/0.5/1/2 m) × 60
angles (D = 240 complex), plus 2 incommensurate relocalization rings
(5.3/12.8 m). Octaves are provably the right ladder; the golden ratio is
uniquely catastrophic for ladders (additive resonance φ²=φ+1 — see
RESULTS.md finding 7, incl. the falsification history of its mechanism).

**Map** (`ssp_bounded.py`, the deliverable; `ssp_hier.py` = HY4 refinement):
rigid anchor-frame segment vectors + derivative vectors; coarse rings live in
a single global relocalization vector. Spatial cell-cap eviction bounds map
memory by area, not time.

**Backend**: anchor pose graph; sequential edges at measured frontend
accuracy (inflated over odometry-fallback spans); pass-segregated loop
constraints behind a drift-scaled innovation gate, session-relative soft
coherence response, Cauchy IRLS + leave-one-out pruning; analytic-Jacobian
TRF solver whose 30-evaluation cap is a *deliberate* early-stopping
regularizer (iterative regularization / semiconvergence — Hanke 1997; the
explicit Tikhonov alternative was built, swept, and rejected). Drought
relocalization against the global coarse vector breaks closure droughts on
sparse-revisit logs (R3/R4 sections in RESULTS.md).

**Three architectural laws**, each learned from a measured failure:
1. content the loop band matches against must live in **graph-consistent**
   memory (world-frame decay is never slow enough);
2. frontend recency small, loop gap large — **independent knobs**; sub-lap
   closures retro-corrupt, wide recency bends the sequential chain;
3. map content must be **viewpoint-neutral** — directional encode weighting
   wins same-heading benches and breaks real logs.

## Quickstart

```sh
pip install -r requirements.txt   # pins are load-bearing (see RESULTS.md)
cd data && ./fetch_datasets.sh && cd ..
python3 ssp_bounded_carmen.py data/intel.log     # full pipeline, ~100 s
python3 ssp_bounded.py quick                     # synthetic GT bench
python3 ssp_slam.py oct                          # original synthetic demo
python3 rpe.py intel_traj_bounded.npz            # RPE/ATE evaluation
open demo/index.html                             # interactive browser demo
```

Everything is deterministic: published numbers reproduce bit-exact.

## Repository map

| file | role |
|---|---|
| `ssp_slam.py` | encoder, matcher, synthetic world/sim (the original study) |
| `ssp_bounded.py` | **bounded-memory continual SLAM (deliverable)** + GT bench |
| `ssp_bounded_carmen.py` | CARMEN-log driver (Intel/fr079/ACES/MIT) |
| `ssp_hier.py` | hierarchical/HY4 refinement + drought relocalization (R1–R4) |
| `ssp_slam_loop.py`, `ssp_slam_carmen.py` | earlier unbounded pipeline (kept as stratigraphy) |
| `ssp_scale_arrays.py` | WIP: spatially-anchored per-scale submap arrays |
| `ssp_posefilter.py` | WIP: VSA pose-posterior (harmonic Bayes) filter |
| `baseline_icp.py` / `baseline_csm.py` / `baseline_rbpf.py` | reference baselines, same harness |
| `bench_loop.py`, `worlds.py`, `experiments.py`, `rpe.py` | benches, worlds, encoder sweeps, metrics |
| `demo/` | self-contained browser demo replaying the real Intel log |
| `SotA/` | literature scout notes (VSA/SSP theory, spectral registration, backends, golden-ratio sampling) |
| `RESULTS.md` | **the ledger**: all findings, tables, negatives, audits |
| `archive/` | superseded implementations kept for provenance |

## Datasets & acknowledgements

CARMEN logs from the Radish repository via the StachnissLab mirrors — please
credit the original recorders (see `data/README.md`). Built in an extended
agentic research session (Claude Code); the ledger records which results
were independently re-verified and which claims were corrected along the way.
