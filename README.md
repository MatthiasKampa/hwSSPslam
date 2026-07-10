# hwSSPslam — bounded-memory 2D lidar SLAM with Spatial Semantic Pointers

**The vision: SLAM as bounded vector algebra, headed for silicon.** A
complete 2D-lidar SLAM stack on an FPGA aboard a SPOT (custom 360° head,
1024 beams @ 20 rot/s) whose entire world model is a few hundred KB of
quantized hypervectors in BRAM — every operation the system ever does
(scan registration, map write, loop verification, graph correction) is
phase multiplication, index permutation, inner products, and vector
addition at a few bits per component. No occupancy grid, no point cloud,
**no sensor history**, memory bounded by *area* rather than time. The
Python in this repo is the bit-exact reference model for that datapath
(`SotA/fpga_design.md` is the RTL hand-off; the browser demo replays the
real pipeline).

Concretely: every 5-keyframe segment of trajectory is a **fixed-size
complex Fourier-feature vector** (a Spatial Semantic Pointer / VSA
hypervector) on a polar frequency lattice, and the algebra does the work —

- **translation** of map content = elementwise phase multiplication,
- **rotation** = an exact index permutation of the lattice (plus a stored
  d/dθ *derivative vector* for sub-grid angles — first-order Lie correction),
- **matching** = an inner product (the correlation of scan and map densities),
- **map updates & graph corrections** = vector addition and O(D) phase ops —
  nothing is ever re-encoded.

## Novel components (most interesting first)

1. **Dense scan-to-map registration against the map vector itself.** The
   frontend and the loop verifier both correlate the encoded scan directly
   with the *bundled* map hypervector over SE(2) (translation banks ×
   rotation permutations) — no landmarks, no descriptors, no retrieval
   index. The map is not looked up; it *is* the matching surface.
2. **Group-closed lattice + stored d/dθ derivative.** Rotation of frozen
   map content is an exact index permutation (conjugate wrap on the
   half-circle grid), and a per-segment derivative vector gives sub-grid
   angles to first order — so pose-graph corrections re-place frozen
   content with O(D) phase ops, never re-encoding (`ssp_slam_loop.py`,
   `ssp_bounded.py`).
3. **Segment-integral encoding (sinc family).** The exact line integral of
   the plane wave along a chord, sinc(k·d/2)·e^{ik·c}, unifies three
   things: interpolation between beams, *principled* fine-ring blanking
   (the range "thermometer" is its hand-tuned shadow), and **mass-exact
   beam decimation** for dense heads (GROUP/8 = 2.5× fewer encode terms,
   lossless on the 1024-beam bench). Sampling structure is per-sensor-
   regime: wide-FOV dense heads need bridging (stata 0.196 vs 1.659),
   sparse 180° heads prefer raw arc-weighted points (`ssp_sampling.py`).
4. **Rigid temporal-segment bounded map.** Map content lives in ephemeral
   anchor-frame segments, frozen on anchor advance, evicted by spatial
   cell-cap — O(area), history-free, and the measured law behind it: a
   closure primitive must hold *single-burst, drift-consistent* content
   (persistent spatial submaps smear write-time drift into every loop
   constraint and fail catastrophically).
5. **Binary/quantized VSA store.** Write-time per-ring polar codes (2–6
   bits/phasor + per-ring scales) keep the *algebra* intact after
   quantization; building-scale maps fit 75–625 KB of BRAM, and on fr101
   the 2-bit + int8 pipeline *beats* the float baseline (1.13 vs 1.88)
   (`ssp_fpga.py`).
6. **Perturbation-band acceptance methodology.** The closure cascade is a
   measured knife-edge (1e-4 relative map noise re-rolls discrete
   admission decisions, 100–1000× amplification), so config claims on
   sensitive logs are *bands over an ε-ladder*, not points — with the
   band's entry channel localized (frontend local-bundle reads) and a
   deterministic jitter-consensus mitigation that collapses it 14×
   (`BandSLAM`, PROTOCOL §6, `ssp_stablegate.py`).
7. **Aperture-failure admission classifier.** Loop candidates are gated by
   per-ring coherence *relative to the session's own accepted-match EMA*,
   then split by translation-Hessian anisotropy and off-peak ridge probes
   into well/ill tiers with smooth σ-inflation instead of hard vetoes —
   calibrated once, 9/17 of its constants measured flat (bakeable), the
   rest per-environment registers (`ssp_bounded.py`, veto-scan entry).
8. **Encoder lattice results.** Octave rings beat random Fourier features
   10–20× at equal D; the golden-ratio ladder is uniquely catastrophic
   (additive resonance φ² = φ+1, mechanism falsified and re-established in
   the ledger); full-circle hex lattices with plain-shift permutation are
   the non-Manhattan option (belg 2.07 vs 2.64) (`ssp_angles.py`,
   `ssp_hexreal.py`).

> **Status: active research.** This repository is a live snapshot of an
> ongoing agentic research session; more results land frequently. The full
> experiment ledger — every finding, negative result, retraction, and
> reviewer audit — is [`RESULTS.md`](RESULTS.md). Read it as a lab notebook
> with an abstract on top.

## Headline numbers (deterministic, reproducible)

ATE rmse vs RBPF-corrected references; `ssp_bounded_carmen.py <log>`:

| log | ours | raw odometry | map memory | speed |
|---|---|---|---|---|
| **MIT Stata (floorplan GT, 260° × 1040 beams)** | **0.202 m** (median 0.12; bridged sampler 0.196) | 3.21 m | 1.2 MB | 88 ms/kf |
| FHW exhibition hall | **0.98 m** (median 0.84) | — | 5.2 MB | 33 ms/kf |
| Freiburg 101 (held-out) | **1.88 m** (median 1.55) | 8.56 m | 1.9 MB | 27 ms/kf |
| Freiburg 079 (no retune) | 5.52 m | 14.4 m | 3.5 MB | 30 ms/kf |
| Belgioioso Castle (held-out) | 2.64 m (hex lattice 2.07) | 1.72 m | 1.6 MB | 27 ms/kf |
| ACES3 Austin (no retune) | 6.21 m | 5.41 m | 5.3 MB | 13 ms/kf |
| Intel Research Lab (historical¹) | 2.44 m (median 1.55) | 24.2 m | 5–8 MB | 15–17 ms/kf |
| MIT Infinite Corridor, 1.9 km (held-out) | 42.66 m (38–58 band) | 189 m | 27 MB | 12–21 ms/kf |

¹ intel was removed from the acceptance suite 2026-07-10 (knife-edge
perturbation band, GMapping-referenced GT, 180° FOV vs the 360° target
platform); the number is kept for history. **Stata is the flagship: its
ground truth is floorplan-anchored (~2–3 cm), independent of any SLAM
family, and its 1040-beam 260° sensor is the closest proxy to the SPOT
target head.**

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

## All mechanisms, end to end

### The ordinary (float) SSP pipeline

**Sampling** (`ssp_sampling.py`, per sensor regime — see novel component 3):
wide-FOV dense heads bridge consecutive returns at the 63.4° occlusion gate
(dr ≤ 2·tangential) with arc mass r̄·dθ, as n2/n3 midpoint sub-points or
exact sinc integrals; sparse 180° heads use one arc-weighted point per hit
(w = r·dθ); the historical fixed-0.12 m segment resampler survives only as
a control. Non-bridged (silhouette) hits are always kept at full arc weight
— dropping them was the measured source of the old frontend damage.

**Encoder** (`ssp_slam.py`, `ssp_slam_loop.py`): φ(p) = exp(iWp) on 4
octave matched rings (λ = 0.25/0.5/1/2 m) × 60 half-circle angles (D = 240
complex). Octaves are provably the right ladder; the golden-ratio ladder is
uniquely catastrophic (additive resonance φ² = φ+1); ring count/λ_min is a
per-sensor co-design (finer bottoms want SNR headroom). The 2 extra
relocalization rings (λ 5.3/12.8) are dead weight in the bounded deliverable
(bit-identical without them, −33 % map memory) and matter only for the
drought-relocalization extension.

**Frontend matcher** (`ssp_slam.Matcher`): coarse stage = 2 encodes
(lattice-aligned + half-step pre-rotated) expanded to 14 rotation candidates
by index permutation, each scored against a 17×17 translation bank
(E·(conj(M)⊙S⊙shift), one MAC tree per offset); fine stage = 9 re-encoded
θ steps (0.375°) × 7×7 offsets at 2 cm; parabolic sub-cell refinement in θ
and x/y. Guess = previous estimate composed with the odometry delta; accept
gate 0.45 m / 11°, else clean odometry fallback for that keyframe.

**Keyframing & map write**: keyframes at 0.10 m / 5°; every 5th keyframe
opens an anchor. Scans fold into the active segment in the anchor frame at
EXACT rotation (points still in hand — the segment never sees per-keyframe
quantization), together with the analytic d/dθ derivative vector. Segments
freeze when the anchor advances. A spatial cell grid (2.0 m cells, cap 6
segments/cell) evicts oldest-in-cell — memory bounded by area, not time.

**Local bundle read**: the frontend matches only against the RECENT window
(last 12 anchors within 8 m) — old passes may reach the pose only through
gated loop edges, never through the frontend (recency small / loop gap
large are independent knobs; violating either was a measured failure).

**Loop admission cascade** (`try_constraint`, attempted every 4 kf):
candidates = frozen segments ≥ 60 anchors away and < 5.0 m from the pose →
grouped into contiguous chains (index gap ≤ 2), nearest chain summed into
one bundle B → wide matcher (t_half 0.72) → match gate 0.6 m / 7° →
per-ring coherence of scan vs B, referenced to the session's own
accepted-match EMA (coh_ref; target 0.55) → aperture-failure classifier
(translation-Hessian anisotropy + off-peak ridge probes; calibrated
thresholds) routes into: hard veto (ratio < 0.20, or < 0.35 with the
failure signature), exempt WELL tier (sharp isotropic peak), or smooth
σ-inflation ((target/ratio)² ×8 if ill, capped 40) → drift-scaled
innovation gate (χ² 9 with allowance growing 0.002 m & 0.03°/kf since the
last accept, capped 0.30 m/6°) → factor σ from anchor lever arm
(√(0.08² + (0.05·lever)²), 2.0°), inflated edges don't reset the drift
clock → per-pair dedup (re-measured edges replace only if ≥3.5× weaker) →
repeated same-region suppressions (3 within 3 m) cap the allowance clock.

**Backend** (`ssp_bounded.py`): anchor pose graph; sequential edges at
measured frontend accuracy (0.03 m/0.3°, inflated to 0.10 m/1.5° over
odometry-fallback spans); Cauchy IRLS + leave-one-out pruning of loop
edges; analytic-Jacobian TRF solver whose 30-evaluation cap is a
*deliberate* early-stopping regularizer (semiconvergence — the explicit
Tikhonov alternative was built, swept, and rejected). Relax every 25 kf;
afterwards all frozen content is re-placed by O(D) permutation + phase ops.

**Optional extensions** (`ssp_hier.py`): HY4 hierarchical store (0.67×
memory, bit-identical) and drought relocalization against a global coarse
vector (needs the relo rings; breaks closure droughts on sparse-revisit
logs).

**Three architectural laws**, each learned from a measured failure:
1. content the loop band matches against must live in **graph-consistent**
   memory (world-frame decay is never slow enough);
2. frontend recency small, loop gap large — **independent knobs**; sub-lap
   closures retro-corrupt, wide recency bends the sequential chain;
3. map content must be **viewpoint-neutral** — directional encode weighting
   wins same-heading benches and breaks real logs.

### The binary datapath (what changes for silicon)

Same architecture; four substitutions (`ssp_fpga.py` is the bit-accurate
model, `SotA/fpga_design.md` the RTL hand-off):

- **Arithmetic**: every phase evaluation goes through an 8-bit cis ROM
  (256 entries × 2 × 7-bit signed); W·p as fixed-point MACs (W rows are
  constants); weights 7-bit (unit weights only on robust logs); int32
  accumulators. addr8 is the measured knee — addr6 does real damage.
- **Store**: the active segment accumulates in full precision (1-deep
  buffer); freeze-on-anchor-advance quantizes segvec+segder to per-ring
  polar codes — 6 bit/phasor (16 phase × 4 mag) general tier, 2 bit
  (4 phase, one-level mag) on closure-redundant deployments — with
  per-ring scales (REQUIRED: per-vector scales crush the fine rings the
  veto reads). The translate/rotate/bundle algebra survives the store
  (fidelities 0.986/0.925/0.9997 at 6 b).
- **Budgets**: 136 B/segment at 2-bit no-relo; building-scale maps
  75–625 KB BRAM; matcher banks are ROMs; the whole system fits ~155 KB
  BRAM + 1–2 DSPs. The 1024-beam target head costs ~1 DSP48-class encode
  budget at 20 Hz keyframes; GROUP/8 integral decimation halves it if
  wanted.
- **Fabric/host split**: fabric owns encode, match banks, fold, and the
  per-ring coherence sums; the host (any embedded core) owns SE(2)
  bookkeeping, the admission gates, and the sparse TRF relax
  (milliseconds per relax at ≤1k anchors).

Acceptance for every binary config is its **perturbation band**, not a
point (PROTOCOL §6); measured tiers: fr101-class 2-bit (1.13 e2e — beats
float), stata/fr079-class 6-bit, with per-log bands in the ledger. The
veto scan splits the admission constants into 9 bakeable constants vs 6
per-environment runtime registers ({gate_t, sig_r0, coh_target, ill_mult,
infl_pow, chain_gap}).

## Quickstart

```sh
pip install -r requirements.txt   # pins are load-bearing (see RESULTS.md)
cd data && ./fetch_datasets.sh && cd ..
python3 ssp_bounded_carmen.py data/fr101.log     # shipped deliverable + ATE
python3 ssp_datasets.py run stata                # flagship: floorplan GT
python3 ssp_synth.py bench                       # SPOT-proxy 360° GT bench
python3 ssp_bounded.py quick                     # synthetic GT bench
open demo/index.html                             # browser demo (9 replays + sandbox)
```

Everything is deterministic: published numbers reproduce bit-exact.

## Repository map

| file | role |
|---|---|
| `ssp_slam.py` | encoder, matcher, synthetic world/sim (the original study) |
| `ssp_bounded.py` | **bounded-memory continual SLAM (deliverable)** + GT bench |
| `ssp_bounded_carmen.py` | CARMEN-log driver (Intel/fr079/ACES/MIT) |
| `ssp_fpga.py` | **FPGA track**: write-time quantized store (per-ring polar), integer front-path model, per-beam point encoding, perturbation-band harness, op/BRAM sizing |
| `ssp_datasets.py` | dataset registry + the ONE shared run/eval harness (gfs / range-identity / floorplan-GT / exact-synth) |
| `ssp_sampling.py` | sampling/encoding family: bridged sub-points, exact segment integrals, group decimation, blanking compensation |
| `ssp_synth.py` | SPOT-proxy synthetic 360° bench (exact GT, seeded) |
| `ssp_lattice.py` | lattice patching (polar/hex, any ladder) with exact restore |
| `ssp_stata.py` | MIT Stata bag adapter (floorplan-anchored GT) |
| `ssp_hier.py` | hierarchical/HY4 refinement + drought relocalization (R1–R4) |
| `ssp_slam_loop.py`, `ssp_slam_carmen.py` | earlier unbounded pipeline (kept as stratigraphy) |
| `ssp_scale_arrays.py` | WIP: spatially-anchored per-scale submap arrays |
| `ssp_posefilter.py` | WIP: VSA pose-posterior (harmonic Bayes) filter |
| `baseline_icp.py` / `baseline_csm.py` / `baseline_rbpf.py` | reference baselines, same harness |
| `bench_loop.py`, `worlds.py`, `experiments.py`, `rpe.py` | benches, worlds, encoder sweeps, metrics |
| `demo/` | self-contained browser demo: interactive synth sandbox + 9 real-data replays (stata/fhw/fr101/fr079/belg incl. binary, hex, and the deploy-sampler configs) exported by `demo/export_replay.py` from the real pipeline — Python is the source of truth |
| `SotA/` | literature scout notes (VSA/SSP theory, spectral registration, backends, golden-ratio sampling) |
| `RESULTS.md` | **the ledger**: all findings, tables, negatives, audits |
| `archive/` | superseded implementations kept for provenance |

## Datasets & acknowledgements

CARMEN logs from the Radish repository via the StachnissLab mirrors — please
credit the original recorders (see `data/README.md`). Built in an extended
agentic research session (Claude Code); the ledger records which results
were independently re-verified and which claims were corrected along the way.
