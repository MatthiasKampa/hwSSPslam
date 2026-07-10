# SSP SLAM — encoder recipe experiments

**SHIPPED CONFIGURATION (final, 2026-07-07).** Solver: analytic-Jacobian TRF
(`_gn`; plain GN steps over-converge along flat-valley directions on floppy
graphs). Veto: soft sigma-inflation, session-relative, `coh_soft=True`,
`coh_target=0.55` — chosen by minimizing the worst-of-three-logs ratio
(1.97 vs 2.59 hard). Three-log confirmation, deterministic: **Intel 2.440 m
rmse / 1.553 median @ 17 ms/kf (best of session; day started at 14 m
diverging), fr079 5.523 (odometry 14.35), ACES 6.212 (honest negative:
odometry 5.41 is already better there)**. Session lesson, now THRICE-proven: "GN is a strict upgrade", "the soft veto
costs accuracy", and "TRF step control fixes the flat valleys" were all
artifacts — the final audit showed the shipped solver's max_nfev=30 acts as
load-bearing early-stopping regularization (fully-converged TRF slides down
the same valleys: fr079 12.7 m at 300 evals). The numbers are real,
deterministic, and reproduced bit-exact by an independent auditor (GO
verdict); the mechanism story required correction. The designed successor
(an explicit Tikhonov prior toward pre-relax anchors, tuned multi-log) was
BUILT AND REJECTED 2026-07-07: every tested lambda converges cleanly yet
loses to early stopping on fr079 (worst-of-three ratio 3.39 best-case vs
1.97 shipped) — see "Tikhonov prior sweep" below. The prior ships
default-off (`prior_lam_t=0`); max_nfev=30 remains the load-bearing
regularizer. Every future acceptance gate must be multi-log.

**Abstract.** Built: a 2D-lidar SLAM stack whose map is a fixed-size complex
Fourier-feature (SSP) vector per rigid segment — no sensor history — with
translation as phase multiplication, rotation as exact lattice permutation
plus a stored derivative vector for sub-grid angles (`ssp_bounded.py`, the
deliverable; the encoder sweep below is the supporting study that fixed its
frequency lattice). Claimed: bounded map memory at usable accuracy, not
state-of-the-art accuracy. Proven: deterministic polar-lattice encoders beat
Gaussian RFF ~10-20x in dimensions (multi-world, multi-seed); the bounded
continual system runs Intel at 2.440 m ATE / 3.6 cm-per-m RPE@10m median
(rpe.py on intel_traj_bounded.npz; odometry 20.9) in 8 MB. TRANSFER CAVEAT
(referee): the shipped config was selected by worst-of-three over
Intel/fr079/ACES, so those logs are no longer held-out. HELD-OUT TEST RUN
(MIT Infinite Corridor, 17,480 scans, 1.9 km, 2.1 h — 4-6x longer than any
tuning log; zero code/parameter changes): bounded SLAM 57.8 m rmse vs raw
odometry 189.3 (ORIGINAL held-out record; drought relocalization R3/R4/R5 later
improved this to the SHIPPED 42.66 m / ~4.4x — see those sections) at 12 ms/kf and
a 27 MB map, degrading gracefully (locally rigid, globally bent) rather than
diverging; only 77 closures fired (~0.5% density, none in the final hour),
demonstrating the revisit-density limit on real data at full scale. The
drift-proportional-payoff pattern held in relative form exactly as this
ledger predicted; the absolute accuracy envelope is confirmed
revisit-density-limited. (Eval note: the MIT gfs timestamps are corrupt
upstream; reference matching used exact range-array identity — script in
session scratchpad; SLAM run itself untouched.) Environment pinned in requirements.txt — the scipy
pin is load-bearing (max_nfev truncation is the solver regularizer).
Historical numbers below (4.48, 2.927, 7.37...) are superseded stratigraphy
kept for the record; current numbers live in the SHIPPED block above.

Setup: 16x10 m synthetic room (walls, box, pillar, doorway, diagonal), 140-frame
loop, 240-beam lidar with 2 cm range noise / angular jitter / 2% dropout.
Scans become uniformly-spaced samples along hit-to-hit segments (occlusion
filter kills radial "phantom wall" bridges), encoded as complex random-Fourier
features phi(p) = exp(i W p); registration = coarse-to-fine search of
Re<Map, exp(iWd) * Scan_theta> over (theta, d); aligned scan bundled into the map.
Errors below are vs ground truth over the whole run. `D` counts complex features.

## Headline table (single seed 7, axis-aligned world)

| recipe | D | ms/frame | trans mean/max cm | head mean/max deg |
|---|---|---|---|---|
| RFF gaussian, l=0.25 | 2048 | 353 | 3.0 / 6.9 | 0.37 / 0.74 |
| RFF gaussian, l=0.25 | 216 | 36 | 7.2 / 18.0 | 0.79 / 2.82 |
| hex, one fixed orientation | 216 | 28 | 25.2 / 124.6 | 0.09 / 0.50 |
| hex, golden-angle orient/scale | 216 | 32 | 2.4 / 5.1 | 0.30 / 0.60 |
| hexgrid 6 scales x 12 rots | 216 | 30 | 0.9 / 2.1 | 0.03 / 0.16 |
| polar grid 6 x 36 (no hex) | 216 | 30 | 1.1 / 2.6 | 0.04 / 0.15 |
| octaves 0.25-2 x 12 angles | 48 | 8 | 0.6 / 1.7 | 0.03 / 0.11 |
| octaves 0.25-2 x 4 angles | 16 | 3 | 0.7 / 1.8 | 0.03 / 0.13 |
| **octaves 0.25-2 x 24 angles** | **96** | **14** | **0.9 / 2.3** | **0.03 / 0.17** |
| DFT lattice L=20 N=10 | 220 | 28 | 7.4 / 18.6 | 1.02 / 1.89 |
| linear-lambda 6 x 36 | 216 | 29 | 3.1 / 35.3 | 0.27 / 6.78 |

Multi-seed (1-5): all octave-grid numbers reproduce within +-0.1 cm.

## Findings

1. **Structured beats random at small D — two separable causes.** A banded-random
   control (lambda log-uniform on [0.25,2], random angles) at D=216 scores 6.4 cm
   vs Gaussian-RFF-216's 15.1 (3 seeds, room world): matching the spectral band
   buys ~2.4x. The deterministic grid buys another ~10x on top (oct4 x 24 at
   D=96: 0.7 cm) in flat-wall worlds where grid angles hit wall normals exactly
   (see finding 6 for the orientation-luck caveat).

2. **Rotation/scale must be a symmetric grid.** Any scheme where orientation
   marches with scale index couples the two axes: rotating the scan by one
   angle step is nearly indistinguishable from shifting all scales one index
   (= zooming space by the ratio). Smooth fans diverge (127 cm / 22 deg);
   golden-angle survives but is 3x worse than the grid.

3. **>= ~6 well-spread axes are needed for a point-like correlation peak.**
   3 axes (aligned hex) give a 3-ridge kernel: heading superb (0.09 deg),
   translation slides along ridges (>1 m excursions).

4. **Hex triples add nothing** over single-direction modules at equal D
   (hexgrid 6x12 = 0.9 cm / 0.03 deg vs polar 6x36 = 1.1 / 0.04 at D=216).

5. **Scale ladder: octaves suffice; range can stop at the search window.**
   lam = 0.25/0.5/1/2 m is as good as 6-72 tuned geometric scales. Big
   wavelengths are unnecessary because the constant-velocity prediction keeps
   the search window at +-0.36 m. Gaps > ~octave x2 (e.g. 0.25 then 2.0)
   reintroduce sidelobe excursions. lam_min ~ noise scale; 0.5 costs 3x.

6. **Angle density is environment-coupled — the one real fragility.** A long
   straight wall only excites frequencies within ~lambda/L radians of its
   normal. In the axis-aligned world any even angle count (grid contains 0 and
   90 deg) works down to 4 angles; odd counts (7/9/11) miss 90 deg and lose an
   entire translation axis (score landscape becomes a 1D ridge). In a 37
   deg-rotated world (normals far from grid angles), sparse grids fail;
   24 angles (7.5 deg spacing) is the knee: 3.3 cm — same as RFF-2048 there.
   Staggering angles across scales to densify the union makes things worse
   (breaks the grid symmetry of finding 2). Area-uniform frequency sampling
   (angles proportional to frequency) also loses to the plain uniform grid.

7. **Ring-ratio study (targeted, fixed band [0.25,2] x 12 angles, 2026-07-07
   late).** Octaves (r=2.0) win outright (0.68 cm at the smallest dense-D);
   flat plateau r~1.4-2.8; excursions at r=8 (two-scale). Sharp negative:
   the GOLDEN RATIO is uniquely bad — 12.1 cm on all 6 seeds (neighbors
   1.57/1.635/1.68: 0.9-2.2) because phi is the ADDITIVE-resonance ratio
   (phi^2 = phi+1: each frequency is exactly the sum of the two below,
   aligning three-wave interference in the correlation), and its Fibonacci
   convergent 8/5 = 1.600 fails catastrophically-but-stochastically (one
   seed 50 m: exact rational coincidence -> occasional false lock). The
   "most irrational" intuition that makes golden ANGLES good is backwards
   for SCALES, where additive — not rational — coincidences bite. sqrt(e) =
   1.6487, 2% from phi, is fine (1.35 cm): the culprit is the additive
   identity, not the neighborhood.
   MECHANISM CORRECTED by the dither sweep (scratchpad golden_dither.log,
   60-angle rings): angular dithering does NOT rescue the golden ladder
   (aligned 8.2 / half-interleave 8.8 / golden-seq rotation 9.3 cm) —
   falsifying the collinear three-wave form; the resonance is a
   MAGNITUDE-only envelope beat (|w_i|-|w_j|=|w_m| difference-beats in the
   radial kernel, angle-independent, undithererable). The hazard is also
   ENVIRONMENT-COUPLED: in the curved blob world the golden ladder WINS
   (1.17 vs octave 2.83 cm — extra scale density pays where spectra spread
   over angles); flat-wall worlds expose the radial beats along wall
   normals. Vogel/Fibonacci-spiral frequency layouts are fragile (room:
   D=240 erratic 8.2, D=300 fine 1.21) and forfeit the permutation property.
   Literature (SotA/): the phenomenon is known in grid-cell theory (Vago &
   Ujfalussy 2018, interference at ~1.618) and the mechanism class in
   turbulence ("Fibonacci Turbulence" PRX 2021 builds shell models on
   k_{n+1}=k_n+k_{n-1} BECAUSE it maximizes three-wave resonance); no
   registration/interferometry source warns against phi ladders — the design
   rule appears publishable. Wei-Balasubramanian's optimal grid ratio
   sqrt(e)~1.649 sits in our safe plateau. RWTH reconciliation
   (SotA/golden_dithering.md): Schretter-Kobbelt's golden sampling is
   ADDITIVE (Weyl mod 1), where phi's continued fraction is a virtue; used
   MULTIPLICATIVELY as a ladder ratio, the same identity phi^2=phi+1 becomes
   additive closure — same constant, opposite side of the exponential map.
   Irrational ladders buy nothing inside the
   matched band (search window < coarsest wavelength); incommensurability
   matters only for wide-window relocalization rings (5.3/12.8, separate).

8. **Occlusion filter.** |dr| > 2 x tangential-gap rejection removes phantom
   bridges across depth discontinuities (the literal |dr| > 2|dp| form can
   never trigger — reverse triangle inequality). Side effect: grazing-angle
   views of real walls are also rejected; their hits survive as isolated
   points with small weight.

## World dependence (3 seeds each; trans mean cm)

| recipe | D | room (flat walls) | room rot 37 deg | curved blob world |
|---|---|---|---|---|
| oct4 x 12 angles, exact grid | 48 | 0.7 | 49.0 | 7.1 |
| oct4 x 24 angles, exact grid | 96 | 0.7 | 3.3 | 4.3 |
| oct4 x 36 angles, exact grid | 144 | 0.6 | 14.2 | 2.0 |
| oct4 x 24, jittered angles | 96 | 8.2 | 19.2 | 2.4 |
| oct4 x 36, jittered angles | 144 | 9.6 | 11.6 | 2.1 |
| RFF gaussian | 216 | 13.0 | 10.0 | 6.8 |
| RFF gaussian | 2048 | 3.0 | 4.0 | 2.1 |

The encoder's angular sampling must match the environment's angular spectrum:

- Long flat walls concentrate fine-scale spectral energy in razor-thin lines at
  the wall normals. Exact grids win when a grid angle hits a normal (room: 0/90
  in every even set) and lose by luck-of-alignment otherwise (rot37: x24 lands
  0.5 deg from a normal and beats the denser x36 at 2 deg away). Jitter or
  randomness forfeits exact hits and always pays several cm here.
- Curved worlds spread energy over all angles: any well-spread set works,
  density and coincidence-avoidance dominate — jittered x24 (D=96) matches
  RFF-2048 (D=2048).
- Random RFF is the flat-response choice: mediocre-to-good everywhere,
  ~10-20x the dimensions for the same accuracy.

Caveats: heading in the rot37 world carries a ~0.5 deg encoder-independent
bias (pipeline-level, undiagnosed). First curved-world attempt was degenerate
by construction — concentric boundary+pillar made rotation weakly observable,
and one pillar grazed the path; clearance is now asserted >= 0.5 m in
worlds.check_world (measured 0.6-1.9 m across the six worlds).

## Six-world battery (`experiments.py worlds`; seeds 1-3; trans mean cm)

Note on cross-table numbers: the world-dependence table above used seed 7's
sim stream (via `ssp_slam.py <mode>`-style runs); this battery uses seeds 1-3
with a different RNG protocol, so the same recipe can differ between tables
(e.g. RFF-2048 room 3.0 there vs 7.4 here, blob 2.1 vs 1.6). Excursion-driven
errors have heavy seed variance — compare within a table, not across.

| recipe | room | corridor | office | sparse | blob | mixed |
|---|---|---|---|---|---|---|
| oct4 x 12 (D=48) | 0.7 | 5.7 | 0.9 | 5.7 | 7.1 | 1.0 |
| oct4 x 24 (D=96) | 0.7 | 6.1 | 0.9 | 1.9 | 4.3 | 0.9 |
| **oct4 x 36 (D=144)** | **0.6** | **3.4** | **0.9** | **1.4** | **2.0** | **0.7** |
| jit-x36 (D=144) | 9.6 | 3.8 | 5.1 | 9.5 | 2.1 | 9.4 |
| RFF 216 | 15.0 | 5.9 | 7.7 | 25.5 | 6.8 | 15.1 |
| RFF 2048 (1 seed) | 7.4 | 5.0 | 3.3 | 15.4 | 1.6 | 9.3 |

oct4 x 36 is best or tied-best among the structured small-D recipes in every
world in THIS battery (in the earlier rot37 world x24 beat x36 by alignment
luck — 0.5 vs 2.0 deg from a wall normal; neither ordering is robust to
world orientation, which is finding 6) (RFF-2048 edges it out in the blob world, 1.6 vs 2.0, at 14x the D). The ring corridor is the hardest
world for every encoder (aperture problem; ~3.5 cm floor even for the winner —
wall notches are what keep it bounded). The sparse hall (53% beam hit rate)
punishes small D via occasional excursions. The jittered variant only pays off
in the curved blob world and loses everywhere flat walls dominate.

## Reference baselines (built in-repo, same parsing/keyframing/eval; ATE rmse m)

| method | Intel | fr079 | ACES | ms/kf | map+state memory | sensor history |
|---|---|---|---|---|---|---|
| raw odometry | 24.15 | 14.35 | 5.41 | — | — | — |
| **SSP bounded (ours)** | 2.44 | 5.52 | 6.21 | 15-17 | **5-8 MB** | **none** |
| ICP + pose graph (`baseline_icp.py`) | 1.70 | 3.01 | 4.93 | 22* | 15-35 MB | all scans |
| Correlative grids (`baseline_csm.py`) | 3.27 (med 0.98) | 2.27 | 0.22 | 15-26* | 19-28 MB | endpoints+grids |
| RBPF GMapping-lite (`baseline_rbpf.py`) | **0.12** | 2.74** | **0.12** | 6 | 39-56 MB peak | particle grids |

*under shared-CPU load. **high seed variance (1.26-2.74 across seeds); the
Intel 0.12 is single-seed and should carry a mean+/-std.
CIRCULARITY CAVEAT (do not read RBPF's 0.12 as a head-to-head win): the ATE
reference (`*.gfs.log`) is itself GridFastSLAM/GMapping output — an RBPF grid
SLAM — so the RBPF baseline is scored against its OWN algorithm family. Both
build occupancy grids from the same scans with the same motion model, so their
errors are correlated by construction and RBPF's error toward the reference
collapses toward zero. The mutually comparable, cross-family numbers are ours
2.44 / ICP 1.70 / CSM 3.27 (all "distance to a GMapping estimate"); RBPF's 0.12
is the one non-comparable outlier — "reproduces the reference," not 14x more
accurate in absolute terms. (An underpowered lite RBPF still hitting 0.12
reinforces this — it is structural, not skill.)
Honest verdict: RBPF is GMapping-class on the revisit-dense logs (sub-15 cm,
near the eval's timestamp-slop floor) but that number is partly self-referential
(see caveat); ICP beats ours everywhere at 2-4x our memory with full scan
retention; CSM beats ours on fr079/ACES and loses on Intel rmse (wins median).
Ours holds the
smallest, boundedly-sized, history-free state. This quantifies the ledger's
long-standing positioning: the contribution is the representation and its
memory/algebra properties, not absolute accuracy. Baseline details (tuning
ledgers, all <=3 iterations, no per-log tuning) in each agent report; notable
transferables found by the baseline builds: robust loss can neuter genuine
closures (linear + explicit pruning won there), and rotation-triggered
keyframes should not be inserted into maps (motion distortion during in-place
spins — likely relevant to OUR Intel spin segment too).

## Encode-side gating study (2026-07-07 eve) — closed negative

The HY4 hard range-gate is VESTIGIAL (gate_hits = 0 on all bench cells and
both real logs: fine rings are ungated in segments by design; coarse-ring
thresholds exceed every range cap) — R1's "gating is load-bearing" applied
to the R1 tiered architecture, not HY4. Refinements evaluated on the full
encode: soft von-Mises rolloff (A), anisotropic per-frequency weights (B),
measured-support gate variable (C). A+B win the same-heading bench decisively
(pooled 0.69x, corridor false edges 5.2 -> 0.8) and REGRESS the real logs
2.2-3.3x (Intel 2.440 -> 7.98): directional encode weights bake the writing
pass's ray geometry into rigid map content that later passes match from
other headings. NEW LAW completing the R1/R2 family: the matched band must
be graph-consistent AND VIEWPOINT-NEUTRAL. C is a bit-exact no-op and its
fr079 over-gating premise is refuted by measurement (spacing/(r*dtheta)
median ~1.1; the driver already scales dtheta by beam count). Encode-side
weighting is closed as a direction; full implementation archived in the
session scratchpad (ssp_hier_gating.py) with self-tests.

## Related work (literature scout, 2026-07-07)

- The closest prior "SSP SLAM" (Dumont/Furlong/Orchard/Eliasmith, Frontiers
  2023) does LANDMARK-level SLAM with given data association and a path
  integrator; it does not register raw lidar. This project's dense
  scan-to-map registration in SSP space appears to be new. Novelty claims
  CORRECTED after the deep scout (SotA/vsa_ssp_theory.md, 2026-07-07): (a)
  rotation-as-permutation is PARTIALLY ANTICIPATED by Krausse/Neftci/Sommer/
  Renner (NICE 2025, arXiv:2503.08608) in a grid-cell VSA — approximate,
  cognitive-maps-only; ours remains distinct as an exact group-closed lattice
  applied to lidar registration. (b) The d/dtheta companion vector appears
  novel in mechanism (they solve sub-grid rotation via bump-vector
  convolution); a head-to-head ablation would be the clean proof. Also
  relevant: VSA-OGM (npj 2026) is the nearest SSP mapping system but assumes
  given poses — registration remains ours.
- Our octave ladder = classic multi-wavelength phase unwrapping; the
  "gaps > ~2 octaves reintroduce sidelobes" finding is the standard unwrapping
  ambiguity condition, and the incommensurate relocalization rings are a
  residue-number-system trick (Kymn et al., Residue HDC, 2025).
- Highest-leverage borrowings queued: covariance from correlation-peak
  curvature as edge information matrices (Olson ICRA'09); branch-and-bound
  relocalization over the octave rings (Cartographer ICRA'16) to replace the
  fixed prior window; bundle-capacity theory (Frady/Kleyko/Sommer FHRR) to
  derive window/cell-cap sizes instead of hand-tuning; PCM max-clique
  (Mangelson ICRA'18) or GNC as stronger outlier filters than IRLS+LOO.

## Recommendation

`python3 ssp_slam.py oct` — wavelengths 0.25/0.5/1/2 m x 24 equally spaced
angles, D=96, ~14 ms/frame: 0.9 cm / 0.03 deg in the friendly world, 3-4 cm in
the hostile ones (roughly RFF-2048 territory at 1/21 the dimensions and 25x
the speed). Tune to the deployment: axis-aligned indoor -> exact grid, as few
as 4-12 angles (D=16-48); curved/organic -> jittered or 36-angle grid
(D=96-144); unknown mix and dimensions are cheap -> random RFF at D >= 1-2k.

Reproduce: `python3 experiments.py sweep` and `python3 experiments.py seeds <names>`;
`score_landscape.png` visualizes the odd-angle-count ridge pathology.

## Real dataset: Intel Research Lab (Radish / StachnissLab)

`data/intel.log` — the classic Intel Research Lab CARMEN log (Dirk Haehnel;
13631 SICK scans, 180 beams / 180 deg, 28.5 x 28.5 m building, ~45 min with
loops and per-room excursions). Source: StachnissLab pre-2014 2D-laser
datasets, http://www2.informatik.uni-freiburg.de/~stachnis/datasets.html
(intel.log.gz + RBPF-corrected intel.gfs.log.gz used as reference; timestamps
in both logs share the logger-relative base in the LAST field). Alternatives
noted there: ACES3 Austin, Freiburg 079, MIT Infinite Corridor.

`python3 ssp_slam_carmen.py data/intel.log` adapts the pipeline: odometry
motion prior (composed deltas) instead of constant velocity, 0.10 m / 5 deg
keyframing, oct4 x 60 angles (D=240), sliding local map (last 150 keyframes
within 8 m, bundled from stored per-keyframe encodings), two-pass matching,
and an innovation gate (0.45 m / 11 deg) that falls back to odometry.
46 ms/keyframe.

Results vs the RBPF-corrected reference (890 matched poses; `rpe.py`,
median/mean at fixed metric baselines — the advantage is baseline-dependent,
at 1 m the 0.3 s timestamp-matching slop dominates both):

| RPE trans cm/m (med/mean) | @1 m | @5 m | @10 m | ATE rmse |
|---|---|---|---|---|
| raw odometry | 13.8 / 15.6 | 14.7 / 15.2 | 20.9 / 21.1 | 24.2 m |
| SSP frontend | 15.5 / 25.2 | 6.9 / 10.7 | 6.4 / 9.0 | 8.0 m |

| RPE heading deg/m (med/mean) | @1 m | @5 m | @10 m |
|---|---|---|---|
| raw odometry | 5.72 / 6.24 | 3.77 / 3.78 | 3.49 / 3.53 |
| SSP frontend | 2.02 / 4.04 | 0.76 / 1.31 | 0.54 / 0.88 |

(Post-beam-fix numbers; regenerate with `ssp_slam_carmen.py` then `rpe.py`.
The beam fix moved frontend ATE 10.9 -> 8.0 m and fallbacks 16.4 -> 14.3%.)

## Loop closure (ssp_slam_loop.py) and bounded-memory SLAM (ssp_bounded.py)

Event-driven closure on the unbounded pipeline (`ssp_slam_loop.py`): prior-
windowed relocalization (rotation = exact lattice permutation; wide mode on
coarse incommensurate rings after long droughts), z-score + two-consistent-
relocalizations acceptance, anchor graph with soft_l1, corrections blended to
every keyframe. Full Intel: ATE 4.9 m with 9 closures (run predates the beam
fix; its frontend-only baseline was 10.9 then, 8.0 after the fix — the
loop-closure delta should be re-measured against the fixed frontend). Lock-in
edge harvesting HURT (self-fulfilling bundle-consensus edges freeze drift)
and is off by default.

`ssp_bounded.py` is the current architecture (bounded memory, continual):
no sensor history — each keyframe folds at frame time (exact rotation) into a
rigid anchor-frame segment vector plus its d/dtheta derivative vector (query
rotation = permutation + first-order correction; the ABLATION shows this is
load-bearing: removing it costs 4x on the bench). Frontend matches only
RECENT segments; old passes act through pass-segregated loop constraints with
lever-inflated sigma, drift-scaled innovation gating, Cauchy-IRLS (loops
only), and leave-one-out chi-square pruning. Sequential edges quadratic at
the frontend's true 5-frame accuracy (0.03 m / 0.3 deg). The frontend
recency and loop-candidate age bands MUST abut (recent_aids >= gap_kf/ANCHOR)
— violating it on Intel left a 2-minute dead band and 14 m divergence.

8-seed 4-world soak (paired): the graph's value scales with frontend drift —
sparse -33.2 cm (7/8 seeds better), room -0.9 (5/8), office -1.2 (3/8), and
corridor +5.5 cm WORSE (aperture world: 5.4 closures accepted, 5.2 false —
nearly every corridor closure is a false one; this is the case the
never-firing coherence veto was designed for and misses; open item).

Bench with outlier injection (`ssp_bounded.py quick`, room, 3 seeds): graph
beats frontend 3/3 seeds (-2.8 cm paired). 10%
injected I.I.D. wrong closures are absorbed by either protection layer alone
(innovation gate rejects ~21, or with the gate ablated LOO prunes ~18; ATE
unharmed either way). Correlated/aliased injection (one fixed wrong SE(2)
transform on every corrupted constraint, 8 paired seeds x 2 worlds): the
stack SURVIVES at 10% and 25% — zero alias edges in any final graph; the
innovation gate rejects 100% pre-insertion; with the gate ablated, LOO still
prunes ~85% and only 1 alias edge survived across 16 runs (sparse cost +1.1
cm mean). Caveat: the tested alias (1.05 m) exceeds the gate's allowance, so
rejection is deterministic — SMALL correlated aliases (0.2-0.35 m, repetitive
-bay scale) remain the unprobed hard case; prescribed (open): a cycle-
consistency consensus-cluster test over accepted loop edges (PCM-style),
which targets exactly the correlated signature per-edge tests cannot see. 10% frontend dropout + odometry bias also unharmed; fallback spans
now carry inflated seq sigma (0.10 m / 1.5 deg vs 0.03 / 0.3). The fine-ring
coherence veto has never fired anywhere and is unsupported pending a ROC pass
on the logged diagnostics.

Full Intel (`ssp_bounded_carmen.py`): ATE 4.48 m rmse / 2.69 m median at
54 ms/kf, map in ~8 MB of segment vectors, no sensor history — matching the
unbounded pipeline (4.9; one run each, differing in more than boundedness, so
no attribution). RPE tells the more defensible story: trans 11.2 / 4.4 / 3.5
cm/m median at 1/5/10 m, heading 1.38 / 0.39 / 0.28 deg/m. POSITIONING: this
is not accuracy-competitive with mature occupancy-grid SLAM (occupancy-grid SLAM is typically reported well under 1 m on this log —
order-of-magnitude yardstick, uncited); the claim is the
architecture — bounded map memory, algebraically transformable map, D=360
complex features — at a usable accuracy, not state-of-the-art accuracy. The
"O(area)" memory claim is by-construction until an eviction soak demonstrates
it (n_evict counter added; graph bookkeeping still grows O(t) pending task
#8 marginalization).

Zero-shot transfer (`ssp_bounded_carmen.py data/fr079.log`, Intel-tuned
settings unchanged except beam spacing generalized to 180/n_beams): Freiburg
079, 4934 scans at 360x0.5deg — ATE 7.37 m rmse (median 6.12) vs raw
odometry 14.35 (7.67); 87 loop edges, 15 pruned, 4.9 MB map at 71 ms/kf.
Halves dead-reckoning error on an unseen log with zero retuning. Third log
(ACES3 Austin, 7374 scans, 180x1deg, zero-shot, no code changes needed):
5.95 m rmse vs raw odometry's 5.41 — HONEST NEGATIVE: ACES odometry is
already excellent (325 m run) and only 17 closures fired on its single large
square loop, whose per-lap drift outruns the Intel-tuned gates. Transfer
verdict across three logs: the system pays off in proportion to dead-
reckoning drift and revisit density; it does not yet snap large sparse loops
(matches the corridor-world soak finding). Band-config
note: closing the Intel dead band by ENLARGING the frontend recency window
(recent_aids 12->60 at gap_kf=300) made Intel WORSE (4.48 -> 7.10 m): wide
recency re-exposes seq edges to drifted-geometry snaps, which empirically
outweighs the dead band. Shrinking the gap instead (gap=60, recent=12,
abutting, 288 loop edges of continual stitching) scored 4.89 m — close but
still behind the deliberate dead band (4.48). A GT-grounded bench study
(sparse world, 5 laps, dropout+bias, 4 paired seeds) reproduces the ordering
— (300,12) 0.13 m vs (300,60) 0.24 vs (60,12) 0.58, frontend-only 0.63 — and
supplies the mechanism: (a) SUB-LAP loop closures carry the current pass's
own drift; accepting them retro-corrupts even early anchors (lap-1 RMS 0.45
vs 0.13 without them) — a large gap forces closures onto high-innovation old
geometry; (b) wide recency lets the frontend snap onto drifted geometry,
bending the seq chain until honest closures get innovation-gated (rejections
6.8 -> 16.5) — the attenuated form of Intel's 7.10 m failure. The two knobs
are independent: keep recency SMALL and gap LARGE; the dead band between
them is benign (old co-located geometry covers it when it matters). Known scaling limit (profiled): full-graph relaxation is
the only O(t) compute term; windowed relax + seq-edge marginalization is the
designed fix, not yet built. Two verified perf upgrades landed meanwhile:
coarse rotation candidates as exact lattice permutations (one encode per
match instead of ~22; Matcher perm=(n_ring,n_ang) option) and analytic-
Jacobian damped Gauss-Newton replacing finite-difference TRF (converges
tighter than the old solver's max_nfev cap) — bench 2.6x wall, relax 9.1x,
match 2.2x, paired ATE unchanged-to-slightly-better. The single-encode
3-deg coarse lattice initially cost closure recall via compounding frontend
drift; the fix (verified independently) encodes twice — lattice-aligned plus
half-step pre-rotated — restoring the 1.5-deg coarse grid at 2 encodes.

FINAL Intel record (merged stack, `ssp_bounded_carmen.py data/intel.log`,
deterministic): **ATE 2.927 m rmse / 1.857 m median at 15 ms/kf** on 28
high-quality closures, 8 MB map — vs 4.48 m and 54 ms/kf at the start of the
day. The last gain came from fixing the corridor false-closure hole: the
coherence veto now compares fine-ring coherence against an EMA of the
frontend's own accepted matches (absolute thresholds do not transfer across
domains; ratio 0.55, response is cliff-like and pinned from above by Intel —
documented in-code). Corridor world: false edges 5.5 -> 0.8 per run, jerk
0.38 -> 0.08, now statistically indistinguishable from frontend-only; the
two residual leaks are GT-verified perceptual aliases that no available
statistic separates from true closures. Independent verification confirmed
all numbers but flags the 0.55 ratio as a BRITTLE operating point: the
response is non-monotone (0.60 scores worse than vetoing everything),
adjacent thresholds share almost no edges, and over-vetoing feeds back into
a loosened innovation gate via last_accept_k starvation. Prescribed
robustification (open): corroborate the veto with the already-computed
degeneracy indicators, soften the hard reject into sigma inflation, and cap
the innovation-allowance growth to break the feedback loop. Caveats: Intel max error rises
(9.7 -> 11.7 m; 28 edges = less redundancy), and the current stack's fr079
baseline is 10.36 m (veto improves it to 9.82) vs 7.37 pre-upgrade. BISECTED
(read-only, deterministic, all published numbers reproduced bit-exact): the
primary cause is the analytic damped-GN solver — on fr079's floppy graph
(sparse closures, 18% odometry-fallback spans) "converging tighter" slides
anchors up to 2.7 m along flat-valley directions the data does not
constrain; trust-region TRF stops 0.5% higher in cost and 3.7x better in
ATE (2.80 vs 10.36 m — BEATING the historical 7.37; Intel costs only
2.93 -> 3.09). The GN-favorable verification had tested only
well-conditioned graphs and Intel. Secondary: the hard 0.55 veto forfeits
the TRF recovery (2.80 -> 10.68). Exonerated: perm matcher (helps),
fallback sigmas (protective), bands (second-order), beam density (red
herring). Prescribed minimal fix: analytic-Jacobian TRF (keep the 9x speed
source, restore trust-region step control) + the veto softening already in
progress. The task-8 windowed backend (BFS cycle windows, boundary-fixed
solves, escalation ladder, eviction marginalization) is merged, independently
verified (escalate-don't-ban fix applied), and OPT-IN via
`BoundedSLAM(windowed=True)`: on Intel it costs accuracy (6.5 m) for no
speed gain because analytic GN already made full solves cheap at this scale;
it exists for genuinely unbounded runs, where its solve windows are proven
flat over time (-21% wall at n=3000 with 504 evictions exercised).

The frontend cuts translational drift ~2x at >=5 m baselines and heading
drift ~5x; the map is locally crisp but rotates globally over the 45-min run
— the expected failure mode of any frontend without loop closure. (The
loop-closure backend suggested here was subsequently built; see the next
section.) Lessons that transferred:
beam order in this log is CCW, exactly 1-deg spacing -90..+89 (A/B tested:
reversed order triples the match-failure rate, and the endpoint-inclusive
linspace(-90,+90,180) stretch cost 4 points of fallback rate: 10.5% -> 6.1%
after the fix); real odometry needs wider search windows (0.48 m / 12
deg) and denser keyframes than the synthetic constant-velocity prior;
matching against an ever-growing global bundle ghost-locks after drift —
the sliding local map cut match failures from 45% to 16%.

## Tikhonov prior sweep (2026-07-07): the principled regularizer LOSES to early stopping

The designed successor to the accidental max_nfev=30 regularizer was built:
explicit per-free-anchor prior rows `lam_t*(x-x0), lam_t*(y-y0),
lam_r*wrap(th-th0)` toward the PRE-RELAX poses (P0 = x0 of each `_gn` call),
appended to both the residual and the analytic sparse Jacobian (diagonal
block; FD-verified to 4.5e-8), with max_nfev raised to 200 as a true
convergence budget. Coupling `lam_r = 0.5*lam_t` (task-suggested rad/m
equivalent: prior tolerates 2 rad of heading per meter of translation slack —
deliberately loose on heading so closures can rotate spans). Sweep on the
three real logs, deterministic, `ssp_bounded_carmen.py` entry points
(ATE rmse, m; ratio = per-log rmse / best-known 2.440 / 2.80 / 5.41):

| lam_t (1/m) | lam_r (1/rad) | Intel | fr079 | ACES | worst-of-3 ratio |
|---|---|---|---|---|---|
| 0 = shipped early-stop | — | **2.440** | **5.523** | 6.212 | **1.97** (fr079) |
| 0.5 | 0.25 | 4.730 | 15.775 | 6.115 | 5.63 (fr079) |
| 1.0 | 0.5 | 3.704 | 9.480 | 5.988 | 3.39 (fr079) |
| 2.0 | 1.0 | 2.947 | 10.627 | 5.947 | 3.80 (fr079) |
| 4.0 | 2.0 | 7.325 | 11.979 | 5.876 | 4.28 (fr079) |

Convergence evidence (`n_nfev_cap` = solves exhausting max_nfev): with the
prior, 0 of 162/161/196/138 Intel-scale solves hit the 200-eval cap (fr079
0/48-62, ACES 0/4-16) — every solve CONVERGED, so the losses are not a
budget artifact. Shipped config for contrast: Intel 68 of 76 solves hit the
30-eval cap, fr079 36/46, ACES 12/12 — the shipped solver essentially NEVER
converges; early stopping is the regularizer, confirmed.

DECISION: keep the early-stop config (worst-of-3 ratio 1.97; every prior
lambda >= 3.39). The prior ships default-off (`prior_lam_t = 0.0` in
`BoundedSLAM.__init__`; lam_t = 0 reproduces the shipped numbers bit-exact —
re-verified post-change on all three logs: 2.440 / 5.523 / 6.212). Why the
principled story fails here: a quadratic prior toward P0 only stiffens each
direction independently, while 30-eval truncation limits the TOTAL applied
correction path-dependently — the trust region applies the well-conditioned
(data-consistent) components of the step first and quits before the sloppy
ones, and the partially-converged intermediate states feed the IRLS / LOO /
veto cascade, changing which loop edges are accepted at all (e.g. Intel
lam_t=4.0 accepted ~31 loop edges by kf 5000 vs ~120 shipped). ACES —
the one log dominated by a single stiff loop — does improve monotonically
with the prior (6.212 -> 5.876), which is consistent with the mechanism:
where the graph is not floppy, converging is good. On floppy graphs the
prior is either too weak to hold the valley (0.5-1.0) or fights the
closures (4.0). A regularizer that reproduces early stopping would need to
be trajectory-aware (e.g. iterate-path penalty or per-direction trust
scaling), not a static Tikhonov row block.

## Hierarchical multi-scale maps (ssp_hier.py, 2026-07-07): tiered world-frame maps LOSE ~1.5-2.8x accuracy for ~37x less map memory

Experiment: replace the rigid per-segment anchor-frame vectors with TIERED
WORLD-FRAME region maps — rings partitioned into tiers, each tier a pair of
staggered square cell grids (cell = cellk*max_lam, 50% overlap, writes
EMA-decayed at eps=0.02/write; coarsest tier(s) optionally one global
vector) — plus RANGE GATING at encode (a sample at range r feeds ring lam
only if lam >= beta*r*dtheta_beam). Backend identical to BoundedSLAM
(inherited: seq edges, innovation gate, soft coherence response, IRLS+LOO,
TRF max_nfev=30); frontend matches the fine+mid query bundle at the current
position; loop constraints match MID rings (lam 1,2) only, edges taken
against the candidate anchor's write-time pose snapshot (world-frame sums
never move with the graph; decay is the only map-side corrective).

Bench (room/sparse/corridor, seeds 1-4 paired, n=750, ATE rmse cm mean over
seeds; f/e = GT-false/accepted loop edges; map KB = touched cells x D_tier x
16 B, vs G1's segvec+segder):

| config | room | sparse | corridor | f/e room | map KB | ms/kf |
|---|---|---|---|---|---|---|
| G1 = BoundedSLAM control | 6.4 | 12.1 | 379 | 7/142 | 932 | 14-21 |
| G2 fine cells + rest global, beta=3 | 17.9 | 18.2 | 600 | 27/144 | 25 | 12-18 |
| G2 beta=2 | 18.6 | 18.1 | 643 | 29/142 | 25 | 12-18 |
| G2 beta=4 | 17.8 | 19.1 | 626 | 27/145 | 24 | 12-18 |
| G2 gating OFF (beta=inf) | 63.3 | 17.9 | 580 | 34/144 | 24 | 12-19 |
| G3 (+ regional {1,2} tier), beta=3 | 106.2 | 17.5 | 566 | 21/149 | 27 | 12-19 |
| G6 (every ring its own tier), beta=3 | 119.9 | 18.4 | 561 | 28/148 | 43 | 12-19 |

Paired vs G1: G2 b3 room 2.78x / sparse 1.51x / corridor 1.58x worse
(0/4, 0/4, 1/4 seeds better). Findings:

1. **Granularity: fewer tiers win.** G2 (fine {0.25,0.5} cells + everything
   else in one slow global vector) is the only stable hierarchy. G3/G6 blow
   up on 2/4 room seeds (~2 m): their mid/loop band lives in fast-decaying
   regional cells, so loop matches hit the current pass's own last-35-kf
   writes and CONFIRM the drifted estimate — confident wrong edges (Z-error
   p90 2.15 m vs 0.25 m in G2, same seed) that every gate is blind to
   (coherence high, innovation ~0; veto/innov-rej fire counts are literally
   0). The loop band must live in SLOW memory; recency-decay and
   loop-reference roles cannot share a tier.
2. **Range gating carries weight where fine rings matter.** Gating off
   diverges 1/4 room seeds (17.9 -> 63.3 cm mean): far samples at ~1 deg
   beam spacing alias into the 0.25/0.5 m rings and blur the frontend map.
   beta in {2,3,4} is flat (insensitive knob, 3 is fine); sparse/corridor
   don't care (fine rings barely constrain there anyway).
3. **Memory-vs-accuracy trade, the honest verdict:** ~37x smaller map
   (25 vs 932 KB; the gap grows with run length since G1 stores 2 vectors
   per 5-kf segment while tier cells are O(area)) for 1.5x (sparse,
   corridor) to 2.8x (room) worse ATE, never beating G1. The ~18 cm room
   floor is structural: unmixable world-frame sums retain drift smear that
   graph corrections cannot retro-fix, and self-written recent content
   biases every loop measurement toward zero innovation. The rigid-segment
   design keeps map geometry exactly consistent with its anchor by
   construction; that invariant is what the hierarchy gives up.
4. Intel NOT run: the pre-registered gate (any config within ~1.3x of G1's
   ATE) failed — closest is 1.44-1.6x on sparse/corridor, room >= 2.8x.

Caveats: eps_global=0.002 (global tiers are written every kf, so the
literal 0.02/write would forget a lap in ~35 kf; set by dwell-ratio
reasoning, not swept — G2's win over G3 partly IS this slow decay);
mid-band ridge probe rescaled 4x (band lam_min 1.0 vs 0.25 calibration);
corridor closures are ~100% GT-false for every hier config (114-141 edges,
vs G1's 5-edge near-abstention) yet ATE stays within 1.6x — the aperture
world aliases self-consistently; relocalize_global (coarse global vector,
prior-windowed) is implemented and self-tested but never exercised on the
bench. Reproduce: `python3 ssp_hier.py bench` (or `cell <gran> <world>
<seed> [beta]`; `selftest` covers gating/write-query/relocalization
invariants).

### R2 refinement (HybridSLAM, 2026-07-07): the fine/mid split-band hybrid FAILS its gate; the MAIN/coarse split is FREE — G1-identical at 2/3 map memory, Intel 2.440 m at 5.2 vs 8.1 MB; neither CSM transfer (spin exclusion, linear loss) survives contact with the VSA pipeline

R2 built `HybridSLAM` (ssp_hier.py, subclass of BoundedSLAM): the FINE band
(rings lam 0.25/0.5) keeps the SHIPPED design exactly — rigid anchor-relative
per-segment vectors + d/dtheta derivatives, recency-windowed frontend,
pass-segregated loop constraints, graph corrections move anchors — but
segments store only the fine sub-vectors (120 of 360 dims + derivative =
1/3 of G1's per-segment memory). The MID band (lam 1/2) goes to G2-style
world-frame region cells (staggered 20 m grids, slow EMA eps_mid=0.005/write)
as additional frontend context and loop-bundle support; the COARSE band
(lam 5.3/12.8) is one global vector (eps_global=0.002) for wide
relocalization (`relocalize_global`, self-tested). Loop constraints match
the fine-band segment chains (slow rigid memory, per the R1 law) with mid
support mixed into the bundle; Z is taken against the CURRENT candidate
anchor (fine band rides with the graph — no write-time snapshot needed).
Tier writes stay range-gated at beta=3; the fine FOLD is UNGATED
(shipped-exact): gating it costs 9.2 -> 14.0 cm on room/1 — the R1 gating
law is about cross-pass aliasing in world-frame sums, which rigid
anchor-frame segments do not have. `seg_nring=4` ("HY4") moves the split to
the MAIN/coarse boundary instead: segments keep the whole matched band
(240 dims + derivative = 2/3 memory), only the coarse band is tiered, and
the mid cells go unused.

Bench (room/sparse/corridor, seeds 1-4 paired, n=750; ATE rmse cm mean;
f/e = GT-false/accepted loop edges; map KB; ms/kf):

| config | room | f/e | KB | ms | sparse | f/e | KB | ms | corridor | f/e | KB | ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 = BoundedSLAM | 6.4 | 7.2/142 | 841 | 21 | 12.1 | 9.0/120 | 852 | 14 | 379 | 5.2/5 | 1102 | 18 |
| G2 (R1 best tier config) | 17.9 | 27.2/144 | 21 | 18 | 18.2 | 22.8/142 | 21 | 12 | 600 | 114/114 | 33 | 18 |
| HY fine/mid split | 8.4 | 3.0/140 | 289 | 20 | 24.2 | 29.2/141 | 294 | 13 | 473 | 22.8/23 | 400 | 18 |
| HY4 MAIN/coarse split | 6.4 | 7.2/142 | 562 | 21 | 12.1 | 9.0/120 | 570 | 14 | 379 | 5.2/5 | 737 | 18 |

1. **HY (the pre-registered main event) fails its 1.15x gate**: per-world
   mean-ATE ratios vs G1 = 1.30 (room, 0/4 seeds better) / 2.00 (sparse,
   0/4) / 1.25 (corridor, 1/4); pooled 1.27x. It does deliver the memory:
   0.34-0.36x of G1's map (segments 3.75 vs 11.25 KB each; mid cells +
   global add ~15-100 KB), and it beats G2 everywhere fine rings matter
   (room 8.4 vs 17.9 cm) with far fewer false edges (3.0 vs 27.2).
2. **Where it fails and why — the R1 law, seen from the other side.**
   Sparse is 2.0x with false edges tripled (29.2 vs 9.0). Component
   ablations (room/sparse seeds 1-2): mid support OUT of the loop bundle
   drops sparse false edges 32 -> 9 but halves accepted edges and still
   lands ~1.7x (fine chains alone cannot reference sparse loops); mid OUT
   of the frontend is catastrophic everywhere (room 74 cm, sparse 1.6-2.2 m
   — the 4-ring matcher needs the mid basin); norm-capping mid support to
   the fine band's mass changes nothing (sparse fine content is weak in
   COHERENCE, not in norm). So wherever the mid band carries the
   localization signal, SOME mid content must be in the matched bundle, and
   world-frame decaying mid content drags the R1 structural floor (stale
   drift smear + self-match bias) back in. R1 said the loop band must live
   in slow memory; R2 adds: slow WORLD-FRAME memory is not slow enough —
   the loop/matched band must live in GRAPH-CONSISTENT (anchor-relative)
   memory wherever it is informative. Fine chains can substitute for it
   only in fine-dominated worlds (room).
3. **The split that IS free: MAIN/coarse (HY4).** Rings 4-5 never enter the
   matcher, the coherence veto or the Hessian/ridge indicators — they exist
   for relocalization only — so dropping them from segments and keeping one
   global coarse vector is behaviorally invisible: HY4 tracks G1 to ~7
   significant digits on all 12 bench cells (paired ratio 1.000, float
   summation noise only) at 0.669x map memory, and reproduces the shipped
   Intel result to the millimeter: **2.440 m rmse at 5.24 MB vs 8.06 MB
   shipped (0.65x)**, 17 ms/kf, 80 loop edges / 6 pruned, 698 segments.
   fr079 (gate: Intel within 1.3x of 2.44 -> run): **5.523 m rmse, again
   exactly the shipped number, at 3.48 MB** (464 segments; G1 stores the
   same 464 at 11.25 KB each = 5.22 MB). This is the recommended default going
   forward: same trajectory, 1/3 less map memory, relocalization vector
   included.
4. **Spin exclusion (CSM transfer) does NOT transfer — negative on Intel.**
   Rotation-triggered keyframes (|odom dtheta| > 4.5 deg over the keyframe
   interval, CSM's INSERT_ROT_MAX) matched but not folded into
   segments/tiers. Bench: neutral (9 of 750 kf excluded; room -0.3 cm,
   sparse +0.6 cm on HY; -0.1/-0.1 on HY4 — the bench trajectory has no
   spins, so this only certifies no harm). Intel: 2286 of 6205 keyframes
   (36.8%) are rotation-triggered, and excluding their folds guts the
   recency-windowed segment frontend: **8.716 m vs 2.440 m**. The CSM
   baseline could afford exclusion because its persistent occupancy grid
   keeps full history density; the VSA frontend's map IS the last ~12
   anchors' folds — starving it of 37% of content (including the entire
   -200 deg spin segment at kf ~3525) breaks matching long before motion
   distortion matters. Shipped behavior (fold everything, let the soft
   coherence response absorb distorted frames) stands.
5. **Linear loss (CSM transfer) does NOT transfer either.** IRLS off, LOO +
   explicit chi2 pruning kept (the CSM baseline's winning combination).
   Bench: HY unchanged (junk never reaches the backend; 0 pruned either
   arm), HY4 slightly worse (room 6.4 -> 7.1 cm on 4/4 seeds, sparse 12.1
   -> 12.4 mixed). Intel: **4.320 m vs 2.440 m** (156 loop edges accepted
   vs 80, 8 pruned vs 6) — without Cauchy downweighting the mid-weight
   wrong edges that survive the innovation gate pull the graph between
   prunings. The CSM result came from a pipeline whose loop verifier
   (LC_SCORE 0.65 + ratio test + pairwise chain consistency) rejects junk
   BEFORE insertion; the VSA pipeline deliberately inserts low-confidence
   edges at inflated sigma and relies on IRLS to hold them — remove IRLS
   and that contract breaks. Genuine large closures being many-sigma is
   handled here by the drift-aware innovation allowance, not by the loss.

Memory accounting (measured, map stores only; trajectory bookkeeping is
O(time) for every config): per segment G1 = 2 x 360 x 16 B = 11.25 KB,
HY = 2 x 120 x 16 B = 3.75 KB, HY4 = 2 x 240 x 16 B = 7.5 KB; HY adds
20 m-cell mid grids (2-15 cells on the bench, 1.875 KB each) + one 1.875 KB
global vector. Bench means: G1 841/852/1102 KB, HY 289/294/400 KB (0.34x),
HY4 562/570/737 KB (0.67x). Intel: HY4 5237 KB vs shipped 8058 KB.

Caveats: HY's fine fold is ungated by DESIGN after measurement (see above)
— the "range gating stays on" pre-registration survives only for tier
writes; eps_mid=0.005 gives a ~139-write half-life, so on the small bench
worlds the current pass is ~40% of a mid cell's mass — the self-match bias
in finding 2 is partly THIS dwell ratio and would shrink on larger maps;
HY corridor beats G1 on the one seed (4) where G1 itself diverges (7.3 m)
— not evidence of robustness; spin exclusion was only falsified for THIS
frontend (a persistent-map variant might still want it); the Intel
ablation arms are single deterministic runs, as shipped comparisons are.
Reproduce: `python3 ssp_hier.py cell <HY|HY4|HY-spin|HY-noirls|HY4-spin|
HY4-noirls> <world> <seed>`; `python3 ssp_hier.py hyintel data/intel.log
[spin] [noirls] [s2]`; `selftest` covers the hybrid invariants (fine
fold/rotation identity, gating, mid roundtrip, coarse relocalization, spin
gate, IRLS switch).

## VSA binding experiments (2026-07-07, scratchpad-only): fusion mechanism test + heterodyne virtual rings

**1. Nonlinear per-ring score fusion — finding 7 mechanism REFINED: the
golden hazard is a CONJUNCTIVE alias, not per-ring-separable sidelobes;
fusion rescues nothing and costs everywhere.** Matcher variant computing
per-ring scores (normalized by ||M_r||*||S_r||) fused as product / geomean /
min of positively-clipped per-ring coherences, vs the shipped linear sum.
Room world, seeds 1-3, aligned 12-angle ladders (finding-7 config;
golden = 0.25*phi^k, 5 rings in [0.25,2], D=60; oct D=48):

| ladder-fusion | trans mean cm (s1/s2/s3) | mean | absmax |
|---|---|---|---|
| oct-linear (control) | 0.8/0.6/0.6 | 0.7 | 2.1 |
| oct-product | 1.5/1.3/1.4 | 1.4 | 4.5 |
| oct-geomean | 1.5/1.5/1.3 | 1.4 | 6.0 |
| oct-min | 1.8/1.8/1.7 | 1.8 | 7.5 |
| golden-linear (control) | 11.8/12.1/12.1 | 12.0 | 35.8 |
| golden-product | 13.9/14.8/14.8 | 14.5 | 93.9 |
| golden-geomean | 13.7/14.8/149.6 | 59.4 | 217.2 |
| golden-min | 15.5/16.4/151.9 | 61.2 | 220.3 |

Prediction FALSIFIED: golden is not rescued and octaves regress ~2x, so the
six-world battery was skipped per the pre-registered condition. Feedback-free
single-shot probe (map bundled at GT poses, leave-one-out, guess = GT +-15 cm
jitter): golden-linear is BIMODAL — median 0.80 cm but 31/210 frames lock
onto remote sidelobes 20-50 cm out (the 12-cm closed-loop mean is these
lock-ins blurring the map); product/min make lock-ins MORE frequent (60/68
of 210; oct-linear has 0). Per-ring scores at lock-in offsets: the false
peak is high in EVERY ring individually (e.g. +0.85/+0.54/+0.52/+0.72/+0.40
vs true +0.80/-0.08/+0.10/+0.45/+0.29) — mid rings PREFER the false offset
and the fine ring is aperture-blind between the two. Mechanism verdict: the
additive closure f_i - f_j = f_k puts the beat interference INSIDE ring k's
own score (map-bundle clutter coherent across rings at the same offset), so
no per-ring fusion can separate it; conjunctive fusion additionally lets any
one weak ring veto the true peak (the octave regression, absmax 2.1 -> 7.5).
Product and geomean share the argmax (monotone transform); their divergence
(seed 3) enters purely through parabolic sub-cell refinement curvature
feeding the closed loop. Cost: fused ~+10% ms/frame at D=60. Design rule:
keep linear score fusion; the golden hazard is a ladder-choice problem
(finding 7), not a fusion problem.

**2. Heterodyne virtual coarse rings — algebra exact, noise benign (1.5-3.6x,
NOT lam_virt/lam_fine), relo near-native at small baselines; but
generated-not-stored is NOT a memory win.** v_virt = v_i (x) conj(v_j)
elementwise on the shared 60-angle lattice, bound PER SEGMENT before
bundling, synthesizes frequency f_i - f_j with EXACT translation covariance
(and inherits the lattice rotation permutation). Designed pairs: lam
1/4.2=0.2381 vs 0.25 -> lam_virt 5 m; 1/2.05=0.4878 vs 0.5 -> 20 m.

- Fibonacci closure (golden ladder): f_k - f_{k+1} = f_{k+2} exact; single
  points close at cos = 1.000000; real scan bundles only cos 0.37-0.46 vs
  the native ring (w- or w^2-weighted alike) — >75% of the virtual ring's
  energy is cross-term bispectral clutter, not the clean coarse encoding.
  Golden self-extension tops out one Fibonacci step above the ladder
  (2.77 m from a 1.71 m coarsest) — never reaches the 5-13 m relo band;
  octave differences stay on-band (premise confirmed).
- Noise (same pose, two sim noise realizations): single-point control shows
  EXACT common-mode cancellation (virt phase std == native to 5 decimals —
  both rings encode the same noisy points). Bundles, position-equivalent
  sigma: native5 0.039 m / virt5 0.060 (1.5x); native20 0.061 / virt20
  0.220 (3.6x). Both naive theories falsified (lam_virt/lam_fine = 20x/40x;
  sqrt(f_i^2+f_j^2)/f_virt = 29x/57x). Correct model: sensor noise cancels
  common-mode; what remains is a scale-free clutter-decoherence phase floor
  ~0.07 rad that maps to position as 0.07*lam_virt/2pi (predicts
  0.056/0.22 m — matches measurement).
- Wide-window relo (+-12 m, 0.25 m grid, room, seeds 1-2, same-heading
  pairs): native {5,20} rings 100%/84% success (<1 m) at 0.5 m/2 m
  viewpoint baselines with peak-to-sidelobe 1.33/1.27; virt-vs-virt 90%/63%
  at PSR 1.09/1.05 (thin margins); the CROSS channel (native query x
  virtual map) FAILS (10-16%, median err 5.0 m = ring alias) — bispectral
  content only matches itself, so BOTH sides must use the virtual channel.
  Bundling per-segment-bound virt vectors holds at 5 segments (89% vs
  native 100%) and sags at 15 (72% vs 94%).
- Accounting: synthesizing {5,20} needs dedicated near-degenerate partner
  rings = exactly the dims of storing {5,20} natively, and the partners add
  nothing to matched-band registration (room, 12 angles, seeds 1-3: pairs6
  0.7 cm = oct4 0.7 = oct4+coarse 0.7). Useful side-negative for finding
  7's hazard model: near-degenerate spacings (0.2381/0.25) are HARMLESS
  when the beat wavelength (5 m) exceeds the search window (+-0.36 m).

VERDICT: the relo band stays STORED. Heterodyning is sound VSA algebra and
could regenerate a lost coarse channel at 1.5-3.6x noise and ~0.8x success,
but it is never cheaper than native storage: ladders whose differences come
free never reach the relo band, and pairs that reach it cost full price.
Scripts (session scratchpad): exp1_fusion.py, exp1_singleshot.py,
exp1_probe.py, exp2_heterodyne.py, exp2b_followup.py.

### R3 refinement (drought relocalization, 2026-07-07): the coarse global vector BREAKS CLOSURE DROUGHTS — bench 3.3-3.5 m -> 0.05 m (4/4 seeds); MIT 45.3 -> 38.0 m with 27 final-third closures (HY4 baseline 15, recorded G1 run 0); Intel 2.440 / fr079 5.523 bit-identical with the mechanism armed; the coarse band's real limit is corridor self-similarity — not capacity, not drift pollution

MECHANISM (HybridSLAM, armed by default, drought_kf=500; `nodrought`
disables). Trigger: no GENUINE accepted closure for drought_kf keyframes,
on a dedicated clock (last_accept_k cannot serve — the _suppress streak
cap advances it as part of its innovation-allowance job, which would
silently disarm the drought in exactly the veto-storm regions it exists
for). Then every 25 kf:

1. MAP-ANCHORED coarse search. Candidate positions = 0.8 m cells within
   4.8 m of old (pass-segregated; pre-drought once a constrained era
   exists — content mapped during the current drought lives in the same
   dangling frame and "relocalizing" against it only resets the clock)
   live segment anchors, minus a drift-scaled self-exclusion ball (fresh
   self-content correlates at offset ~0). NOT a window around the
   estimate: at MIT's true revisits the drifted estimate is 90-236 m
   (median 90) from where the old pass was MAPPED — the pre-registered
   20-40 m prior window is blind there, measured. Full-360 heading sweep
   (3 deg lattice, one zgemm over candidate cells) against the coarse
   global vector; two-stage translation (0.8 m sweep -> top-40 local
   0.2 m refinement: the coarse rings' phase turns ~0.5 rad across half a
   cell, so the true peak can score below junk on the raw grid);
   z >= 3.0 of the refined peak over the sweep statistics.
2. NORMAL-MACHINERY VERIFICATION. Nearest old segment chain to the
   hypothesis; cmatcher from a 5x5 seed grid (+-1.5 m — the coarse peak
   is only cell-accurate); fine-ring session-relative coherence must
   clear coh_target OUTRIGHT. No well-conditioned exemption: that escape
   hatch is calibrated for normal 5 m-radius closures, and the one
   sub-target exemption snap measured (ratio 0.41, variant run) broke
   its run.
3. DROUGHT INNOVATION ALLOWANCE. The Intel-tuned 0.30 m cap saturates at
   since=150 kf and was indeed the MIT binding constraint. Drought
   closures use s_t = sig_t + max(0.30, 0.5 x distance travelled since
   last accept) and s_r = sig_r + 20 deg: measured MIT revisit separation
   reaches ~0.4x the path travelled since the last accept (p90 213 m at
   ~554 m) — rotation-levered, linear in distance; a sqrt(since) cap
   lands ~6x low. The gate keeps only a sanity role (snaps implying
   >1.5x-path drift are physically impossible); precision gating is
   delegated to (2) and (4).
4. CONSISTENCY PAIRING, deliberately conservative: TWO verified fires at
   the 25-kf base cadence (>= ~3 m apart = independent viewpoints)
   within 200 kf whose implied rigid world corrections agree (3 m
   compared at the current anchor, 8 deg); single pending,
   replace-on-conflict.
5. PRE-ALIGNED SNAP. The dangling chain since the last accept is unbent
   (linear heading distribution about the pivot + spread residual
   translation) BEFORE the solver sees the edges: a building-scale snap
   cannot survive max_nfev=30 TRF + IRLS + LOO otherwise — the first
   solve leaves a ~100-sigma residual, IRLS zeroes the genuine edge, LOO
   bans it. Both paired edges then enter the normal backend at standard
   sigmas; a forced relax polishes.

eps_global default 0.002 -> 1e-4: the 0.002 half-life (~350 kf) is
minutes; MIT droughts pair with passes ~10,000 kf old (0.002 leaves 2e-9
of them). The relocalization store must remember for the REVISIT horizon.
It feeds relocalization only — invisible wherever no snap fires.

AGGRESSIVENESS IS THE ENEMY (five MIT variants, all of which pass the
bench): the conservative shape above scored 38.0 m; adding a hot retry
window (pair on the same corridor traverse) 59.6; hot + independence
floor + extra strictness guards 86.9; buffered multi-pending pairing
70.3; a triplet-consistency variant degraded the BENCH itself (1.75 m on
one seed). Every harmful snap came from faster pairing in self-similar
corridor stretches where the fine verifier slides along the corridor,
and a wrong pre-aligned snap is unrecoverable (after pre-alignment the
backend sees only small residuals). More snap opportunities = more wrong
snaps; the 25-kf cadence is load-bearing as the independence guarantee.

BENCH (`python3 ssp_hier.py drought`; sparse world, n=2000, 5 laps; two
70-kf dropout + 0.4 deg/kf heading-bias events at kf 800/1200, each
stepping the frame ~28 deg past every matcher gate; gap 350 =
cross-lap-only candidates; drought_kf=350, below the same-frame revisit
horizon). Scenario design is measured, not assumed — three shapes FAIL to
produce an MIT-like drought: a single concentrated fault (all post-fault
laps share one displaced frame and close on each other), persistent
odometry heading bias with scans up (the frontend erases it per
keyframe), and short 12-kf events (the recency window still holds
pre-event segments at relock and the frontend walks the frame back at up
to ~9 deg per keyframe). ATE rmse cm, paired seeds, baseline = HY4 with
the mechanism disarmed:

| seed | off | armed | snaps @ kf |
|---|---|---|---|
| 1 | 329.5 | 5.0 | 1176, 1575 |
| 2 | 332.1 | 5.4 | 1176, 1575 |
| 3 | 343.2 | 5.9 | 1176, 1575 |
| 4 | 352.5 | 4.8 | 1176, 1581 |

Both droughts are broken within ~380 kf of onset in every seed (false
edges 1-2 of 178-190); the baseline never recovers — its post-fault
"closures" are same-frame shortcuts that cannot fix the global bend.

MIT (driver mirroring hyintel + the range-array-identity eval, session
scratchpad mit_hy4.py; 14,499 keyframes, deterministic):

| config | rmse m | median | final-third rmse | loops (final third) | snaps |
|---|---|---|---|---|---|
| G1 bounded (held-out record) | 57.8 | - | - | 77 (0) | - |
| HY4 nodrought (R3 baseline) | 45.29 | 36.71 | 54.25 | 61 (15) | 0 |
| HY4 + drought (shipped) | 38.03 | 33.51 | 41.66 | 98 (27) | 2 |

The two snaps (kf 1368 and 2868, both at distinctive junction geometry,
verification ratios 0.80-1.45, chi < 1.1) break the early droughts and
re-anchor the run; downstream, normal closures nearly double (55 -> 98
by end incl. 27 in the final third, where the recorded G1 run had zero)
and the final-third rmse drops 54.2 -> 41.7. The deep final-hour drought
produces verified fires but they arrive ~600 kf apart in ambiguous
corridor geometry and never pair — by design; the four variants that
forced them to pair all measured worse than not snapping at all. 21
ms/kf (14 baseline); the target of 20-30 m was NOT reached.

CAVEAT — G1 and HY4 are NOT identical on MIT: first divergence at kf
2100 at 0.26 mm (a relax-solution difference), amplified chaotically
over 14,499 kf of floppy graph to 45.3 vs 57.8 m and 61 vs 77 loops
(probe in session scratchpad). The R2 identity holds bit-exactly on
bench/Intel/fr079 but does not survive MIT-scale chaos; before/after
above therefore uses the HY4 baseline, and ANY single-run MIT delta of
tens of meters — including this one — must be read against that
sensitivity. The bench, with paired seeds and 60-70x margins, is the
controlled evidence that the mechanism works; MIT shows it firing
correctly on real data at the places it can.

REGRESSION (mechanism armed, final code, deterministic): Intel 2.440 m /
1.553 median / 6.420 max, 80 loop edges, 6 pruned — bit-identical to
shipped; drought fired 43 tries -> 23 hypotheses -> 2 verified -> 0
snaps (pairing refused the singletons). fr079 5.523 / 3.533 / 14.617, 32
edges, 2 pruned — bit-identical; 23 tries -> 2 -> 2 verified -> 0 snaps.
The mechanism is inert where droughts do not bind.

DRIFT-POLLUTION AND CAPACITY VERDICT (the R1/R2 caveat and the sqrt(D/N)
question, answered with data):
1. Drift pollution is NOT the failure mode. The world-frame-summed
   coarse vector's geometry is the drifted one, and that is exactly what
   a loop closure needs — agreement with the map as built. Across 52
   fine-verified fires, the coarse hypothesis landed within 1.64 m
   median / 2.66 m max and ~3 deg of the fine-verified pose.
2. Capacity is NOT the failure mode either. Offline study (scratchpad
   mit_capacity.py): rebuild the coarse store from the base trajectory
   as (a) one global vector and (b) ORACLE 40 m staggered-cell shards (N
   cut ~40x — the sqrt(D/N) rescue if capacity-bound). On 25 true
   final-third revisits, the global peak hits the true old-pass location
   0/25 (median miss 26 m); the oracle shard rescues only 3/25 (12%,
   median miss 14.3 m at HIGHER z — i.e. confident within-shard
   along-corridor slides). Sharding is not the fix.
3. The real limit is CORRIDOR SELF-SIMILARITY at the coarse wavelengths:
   the lam 5.3/12.8 blur washes out door/alcove-scale distinguishing
   detail, so retrieval works at distinctive junctions (both MIT snaps
   and the verified-fire clusters sit at two such regions) and fails
   along corridor stretches — which is why the mechanism reliably breaks
   droughts the robot exits through distinctive geometry (every bench
   fault; MIT's early droughts) and only opportunistically the deep
   final-hour corridor drought. A finer relocalization band cannot be
   world-frame-summed (the R1/R2 law), so within this architecture the
   residual is a place-recognition information limit, not a tuning gap.

Reproduce: `python3 ssp_hier.py drought [seeds]`; `python3 ssp_hier.py
hyintel data/intel.log [nodrought]` (and fr079); MIT driver/eval
(mit_hy4.py) + capacity study (mit_capacity.py) + G1/HY4 divergence
probe (mit_probe.py) in the session scratchpad; trajectories
mit_traj_hy4_base.npz / mit_traj_hy4_dr5.npz; `selftest` covers the
coarse-hypothesis rotation+translation identity, chain pre-alignment,
and the candidate-cell raster.

## Aliasing and capacity in unbounded environments (numeric study, 2026-07-07)

Pure-numerics study of the shipped lattice (ssp_slam_loop.py: rings lam
0.25/0.5/1/2 matched + 5.3/12.8 relo, 60 angles, D=360) at MIT-and-beyond
scales. Scripts, JSONs and plots in the session scratchpad (m1c_alias.py,
m2c_capacity.py + m2d_finish.py, m3_shard.py; m1_alias.png, m2_capacity.png).
Scenes: 16x10 m wall room (structured) and 500 aperiodic random points (pure
lattice behavior). Noise: clean | 2 cm sample jitter | 1.0 m i.i.d. jitter as
the MIT drift proxy (3%-of-distance drift x ~30 m local map extent; i.i.d. is
the conservative bound - real drift is locally rigid).

**1. Joint unambiguous range: the binding limit is RED-DOMINANCE, not CRT
coincidences.** Encoded scenes concentrate energy in the coarsest ring
(coherent near-in-phase sums at lam ~ scene size): 74% of a random scene's
energy sits in the 12.8 ring (room: 67% in 5.3+12.8). The joint kernel
therefore behaves like a lone 12.8 m ring: measured worst-direction first
0.5x-alias (peak-based, 180-direction fan + 2D maps to +-100 m) is 12.9 m
(full band) / 13.3 m (relo band), nearly noise-independent, vs the 1D
robust-CRT (Xia-Wang / residue-HDC) equal-weight prediction of 10.1 m
({2,5.3,12.8}) with 0.9x-aliases at 26 m, and exact joint recurrences at
2.0 / 678.4 / 3392 m (matched / relo pair / all three). PER-RING WHITENING
(a one-line query-time gain, dividing each ring's correlation by its bundle
energy) removes EVERY >=0.5x alias out to the full tested +-100 m in 2D -
clean and drifted, D=120/180/240 - and sharpens the relo main lobe from 3.75
to 2.25 m half-width: with 60 angles the J0 angular falloff suppresses the
1D pair-coincidence at 26.2 m below 0.5x, so the whitened 2D range far
exceeds the 1D CRT bound. The matched-band 2 m periodicity claim is
CONFIRMED where it matters: in the wall room, C(2 m) = 0.83-0.88 of peak
along wall normals (first 0.5x-alias at 1.0 m from the lam<=1 sub-comb);
the comb is absent in the aperiodic scene. Under MIT drift the matched band
is DEAD (peak attenuation to 0.03-0.09 of clean; relo band keeps 0.71-0.86)
- drought relocalization has only the relo band to work with, and the room
scene adds a genuine self-similarity ridge out to ~12 m plus the 26.2 m
pair-coincidence under drift.

**2. Bundle capacity of the relo band (D=120), drought-relocalizer metric.**
N unit-norm synthetic segments (two 1.5-4 m wall pieces each) bundled into
one world-frame vector at 0.5 seg/m^2 (area = 2N m^2, 50 to 51,200 m^2);
query = one segment re-observed (2 cm jitter, 30% dropout, heading known);
stage-1 success = a top-40 local-max candidate (0.8 m grid + 0.2 m refine)
within 3 m of truth. Findings: (a) true-peak SNR follows sqrt(2D/N) only to
N ~ 10^2, then saturates at ~0.5-1 - interference is LOCAL (J0 kernel
support), so the real failure mode is shortlist CROWDING by junk maxima,
which grows with area and is only log-suppressed; consequently recall does
NOT improve with D (120 vs 180 {+7.9 ring} vs 240 {120 angles}
indistinguishable at 8-12 trials/pt) - more dims do not fix extreme-value
competition. (b) The RAW band is unusable at any N (recall <=0.33 even at
N=25: the 12.8-ring pile-up forms a blob at the bundle centroid that
outscores truth); whitening is load-bearing here too. (c) Whitened:
recall@40 plateaus at 0.75-0.85 up to N ~ 400, ~0.50 at N ~ 1000, 0 by
N >= 6400; the found peak is displaced 1-3 m at working N (kernel
resolution), so the fine-verify seed grid must span +-3 m. (d) Corridor at
MIT density (1.26 seg/m, 3 m wide): recall@40 = 1.00 at 125 m, 0.75 at
250 m, 0.50 at 1 km, 0.25 at the full 1.9 km, 0.12 at 4 km. Dataset
positions: fr079 (434), ACES (469), Intel (718) sit at 1.1-1.8x the 75%
capacity N* = 400 (one vector per building = marginal, workable across
retries); MIT (2402) is 6x over - a single run-global coarse vector cannot
shortlist MIT revisits. Two pipeline defects surfaced: _coarse_hyp takes
top-40 RAW cells (no non-max suppression), which collapses the shortlist
into one ~5 m blob - NMS at 2.5 m radius raises anchored recall at N=400
from 0.12 to 0.62; and drought_seed = (1.5, 0.75) is tighter than the
measured 1-3 m coarse-peak displacement.

**3. Shard sizing rule.** A per-shard coarse vector supports drought
relocalization iff N_shard <= N* AND shard diameter <= alias-free range:
R_area = min( sqrt(N*/(pi rho_a)), d_alias/2 ),
S_corridor = min( N*/rho_l, measured 250 m @ 75% ),
with N*(75%) = 400, N*(50%) = 1000 (whitened, NMS top-40, +-3 m verify;
D-independent over 120-240), d_alias >= 100 m whitened (13.25 m raw).

| lattice (whitened) | N*75 | R area rho=0.5 | R Intel rho=0.88 | S corridor MIT | KB/km (50% ovl) |
|---|---|---|---|---|---|
| D=120 (5.3/12.8 x 60) | 400 | 16 m | 12 m | 200-250 m | 15-19 |
| D=180 (+7.9 ring) | 400 | 16 m | 12 m | 200-250 m | 22-28 |
| D=240 (x 120 angles) | 400 | 16 m | 12 m | 200-250 m | 30-38 |
| D=120 raw | <25 | alias-bound R <= 6.6 m | - | - | unusable |

Doubling relo dims buys NO measured capacity; the only argument for a third
ring (7.9 m, pairwise-coprime with 53/128 x 0.1) is the 1D CRT guarantee
(0.9x-alias 26 -> 64 m, exact period 678 m -> 53.6 km) as insurance for
shards beyond the tested 100 m - otherwise keep D=120. Shard vectors are
~0.2% of map memory (HY4 segments: 9.4 MB/km at MIT density vs 15-19 KB/km
of shard vectors at 125 m spacing).

**Recommendation for the MIT drought-relocalization work (R3).** (1) Whiten
per-ring in _coarse_hyp / relocalize_global - one line, removes both the
13 m alias trap and the centroid-blob failure. (2) Take the top-40 over
NMS'd local maxima (suppression radius ~2.5 m), not raw cells. (3) Widen
drought_seed to (3.0, 0.75). (4) Shard the coarse vector by travel
distance: one write-once vector per ~250 m of path (retire the single
EMA-decayed run-global gvec; candidates near old anchors score against
THEIR shard's vector). At MIT scale that is ~8-15 vectors (+15-28 KB) and
takes measured stage-1 recall at true revisits from ~0.25 per attempt
(1.9 km single vector, and that is WITH whitening+NMS it currently lacks)
to 0.75-1.00, i.e. near-certain shortlisting within a few RELOC_EVERY
retries; final acceptance stays with the existing fine-band verification +
two-consistent gate. (5) Keep the matched band out of drought scoring - it
is dead at drift (M1) and only the relo band survives.

Caveats: synthetic unit-norm segments with independent content (real
adjacent segments share walls - spatial ambiguity, not vector crosstalk, so
capacity here is the cleaner bound); heading assumed known in M2 (pipeline
sweeps 120 headings, diluting the shortlist further - per-attempt recalls
are optimistic); the 1.0 m i.i.d. drift proxy over-dephases relative to
locally-rigid real drift; 8-12 trials/point (recall s.e. ~0.13), N*
stated to ~2x; whitened alias-free range is a tested bound (+-100 m), not
a proof.

### R4 refinement (study recommendations in the drought relocalizer, 2026-07-07): whitening + NMS + sharded write-once store land cleanly (bench 4/4 at 5.0-6.1 cm, Intel/fr079 bit-identical, +4-15 KB); the seed-grid widening is REJECTED by the controlled bench; in-pipeline corridor recall moves 0.00 -> 0.04 — the study's 0.62-0.75 does NOT transfer; sharpened retrieval in the deep corridor produces a CONSISTENT wrong pair that only a search-breadth escalation stops; MIT 42.95 m vs the 38.03 R3 record and 45.29 baseline

WHAT SHIPPED (HybridSLAM drought path, ssp_hier.py):

1. PER-RING WHITENING (rec 1) in `_coarse_hyp` and both
   `relocalize_global`s: each ring's correlation divided by that ring's
   bundle energy (the study's exact formula). Raw surfaces are
   red-dominated (74% energy in the 12.8 m ring; 13 m aliases, centroid
   blob); whitened surfaces drive the shortlist.
2. NMS SHORTLIST (rec 2): stage-1 takes the top-40 cells after greedy
   non-max suppression at 2.5 m (per-cell best over headings), per
   shard, instead of top-40 raw (cell, heading) pairs.
3. SHARDED WRITE-ONCE COARSE STORE (rec 4): one 120-dim vector per
   250 m of accumulated odometry travel, summed write-once (no decay —
   the store must remember for the revisit horizon; capacity is handled
   by the sharding, N/shard staying near the study's N*~400).
   `_drought_offsets` groups candidate cells by the era shard(s) of the
   old anchors that generated them; each group is scored against ITS
   era's vector with per-shard surface statistics. The EMA gvec remains
   only for the legacy `relocalize_global` recovery path. Memory: Intel
   3 shards (+5.6 KB on 5.2 MB), fr079 2 (+3.8 KB), MIT 8 (+15.0 KB) —
   at the study's predicted floor.
4. MATCHED BAND OUT OF DROUGHT SCORING (rec 5): verified true by
   construction and now ASSERTED (shard vectors and query projection are
   2*NA coarse-band only; selftest checks WC == W[COARSE]).

REC 3 (widen drought_seed to (3.0, 0.75)) IS REJECTED BY MEASUREMENT.
The wide grid reaches the matched band's ~2 m scene aliases (study M1:
joint C(2 m) = 0.83-0.88 along wall normals) and multi-meter corridor
slides, which VERIFY (session-relative ratio 0.9-1.2) and poison the
consistency pairing: bench seeds 1/4 went 5 cm -> 3.3-3.5 m. Every
single-viewpoint arbiter tried FAILED under measurement: raw-score pick
takes the alias where it wins (margin +0.3%); 4-ring and fine-ring
coherence ranks each pick an alias on some seed (fine rings are
drift-dephased noise for ranking); a hyp-distance tiebreak picks the
alias when the whitened peak's own ~1 m displacement leans that way;
rejecting near-ties starves self-similar corridors of ALL fires (0
verifies, droughts never break). drought_seed stays (1.5, 0.75) — reach
1.5 + 0.6 cmatcher capture = 2.1 m, deliberately below alias reach.

WHAT THE BENCH FORCED INSTEAD (all measured, ssp_hier.py comments carry
the numbers):

- TWO-REGIME STAGE-1 PLACEMENT: whitening reweights the main lobe, not
  just the aliases — the whitened peak sits ~1 m from the raw local
  peak, and the entire R3-validated fine verification is calibrated to
  RAW placement (seeding from whitened peaks re-aims it into aliases).
  `_coarse_hyp` computes BOTH surfaces: raw and whitened winners in the
  same basin (<= 1.2 m) -> R3-exact raw refined peak and raw z;
  disagreement (the blob/alias-corrupted regime whitening provably
  fixes — raw MIT revisit peaks are 26 m off, 0/25) -> whitened winner,
  re-centered on its basin's raw local maximum (+-1 m).
- VERIFICATION SUB-FLOOR TIEBREAK: convergent seed-grid poses are
  clustered into basins (0.7 m); the raw MAIN-band score decides
  outright above a 2% margin; within it the score is statistically
  blind (measured margins: decided basins differ >= 2.5%, alias/truth
  flips live at +-0.5%) and the coarse hypothesis picks the nearest
  near-tied basin. Correct on all 8 measured truth/alias cases; a
  plain nearest-to-hyp rule breaks seed 2 (its alias sits NEARER the
  hyp at a 5.8% score deficit).
- pair_tol_t 3.0 -> 1.2 m: BELOW the 2 m alias spacing. Genuine
  verified pairs agree to centimeters (measured snaps: 0.19-0.72 m);
  3.0 m let a truth fire pair with an alias fire (one 2 m-wrong edge at
  strong sigma = 3.5 m bench runs).
- DEEP-SEARCH ESCALATION (`deep_noff` = 6000): a consistent pair formed
  while stage-1 swept > 6000 candidate cells is HELD and must be
  re-confirmed by a third consecutive consistent fire. Measured need on
  MIT: the deep-corridor drought produced a WRONG pair (kf 9236 + 9264)
  agreeing to 0.54 m / 0.6 deg — TIGHTER than a genuine junction snap
  (0.72 m / 2.6 deg), so no tolerance separates it; what separates it
  is search breadth (10,672 cells over 4+ era shards vs 1948/2989 at
  the two genuine snaps, <= 3300 on every bench fire — the study's own
  extreme-value crowding law). Un-guarded, that snap cost 38 -> 54 m
  (final-third closures 27 -> 9, tail segments worse than BASELINE);
  held, the third confirmation never arrives and the run keeps R3's
  measured-best deep-corridor behavior (no snap).

BENCH (`python3 ssp_hier.py drought`, ship code; ATE rmse cm, paired
seeds, baseline = mechanism disarmed):

| seed | off | armed | snaps @ kf |
|---|---|---|---|
| 1 | 329.5 | 5.0 | 1176, 1575 |
| 2 | 332.1 | 5.4 | 1176, 1575 |
| 3 | 343.2 | 5.9 | 1176, 1575 |
| 4 | 352.5 | 6.1 | 1176, 1575 |

(R3: 5.0/5.4/5.9/4.8 with seed 4's second snap at 1581 — same events
broken at the same attempts, all verified poses within 3 cm of truth.)

REGRESSIONS (ship code, deterministic): Intel 2.440 / 1.553 / 6.420 m,
80 loop edges, 6 pruned — bit-identical to shipped/R3; drought 43 tries
-> 26 hyps -> 3 verified -> 0 snaps. fr079 5.523 / 3.533 / 14.617, 32
edges, 2 pruned — bit-identical; 23 -> 10 -> 3 -> 0. The mechanism
stays inert where droughts do not bind; whitening changes WHICH
sub-gate peaks appear (hyp counts moved 23 -> 26 and 2 -> 10) but the
verification + pairing chain refuses them all, exactly as designed.

MIT (mit_hy4.py driver, 14,499 kf, deterministic single runs):

| config | rmse m | median | final-third | loops (final third) | snaps |
|---|---|---|---|---|---|
| HY4 nodrought (baseline) | 45.29 | 36.71 | 54.25 | 61 (15) | 0 |
| HY4 + R3 drought (record) | 38.03 | 33.51 | 41.66 | 98 (27) | 2 |
| R4 without deep-search escalation | 54.28 | 39.28 | 69.51 | 65 (9) | 3 (1 WRONG) |
| R4 ship | 42.95 | 39.12 | 47.59 | 60 (10) | 2 |

(The literal five-recommendations-as-given configuration — wide seeds,
pair_tol 3.0, no guards — scored 53.36 with its own wrong third snap;
both un-guarded shapes land at ~54 via the deep-corridor pair.)

RECALL PER ATTEMPT AT TRUE REVISITS (gfs range-identity pairing; true
revisit = attempt kf within 5 m in REF frame of a >= 1500-kf-older kf;
hit = z >= 3 hypothesis within 5 m of the old pass's mapped position;
r4_recall.py in session scratchpad):

| run | attempts | at true revisits | stage-1 hits | recall/attempt |
|---|---|---|---|---|
| R3 | 295 | 75 | 0 | 0.00 |
| R4 ship | 331 | 106 | 4 | 0.04 |

All 104 R4 failures at true revisits are WRONG-PEAK (confident z >= 3
hypotheses elsewhere — overwhelmingly the junction hotspot at
(-20, 64)); zero are sub-gate misses. The study's projected
0.25 -> 0.75+ does NOT transfer to the pipeline: its own flagged
caveats bite (heading known there vs 120-heading sweep here;
independent unit-norm segment content vs real walls shared across
segments; no est-frame/write-frame mismatch). Both R4 snaps fire at
the SAME two distinctive junctions as R3 (kf 1340/2980 vs 1368/2868 —
the first one ~30 kf EARLIER, the one measurable retrieval gain).

CORRIDOR SELF-SIMILARITY VERDICT (the R4 question): whitening + NMS +
shards do NOT move it — it is information-limited at lam 5.3/12.8, and
R4 shows the limit is one level deeper than R3 could see. Retrieval
can now be made to shortlist confidently in the deep corridor drought
(z 5-6.9 fires at 25-kf cadence, 59-70 verified vs R3's 42), but the
hypotheses land at the wrong corridor twin, the fine verifier confirms
them (corridor content matches corridor content), and the two-fire
consistency test is defeated because self-similar wrongness is
SYSTEMATIC — the one wrong pair agreed tighter than the genuine snaps.
Retrieval was never the MIT bottleneck; verification information is.
Within this architecture the deep-corridor residual stands as a
place-recognition information limit: more retrieval sharpness converts
misses into consistent wrong snaps unless evidence requirements scale
with search breadth (the escalation). rmse note: 42.95 vs the R3
record 38.03 sits inside the measured MIT chaos band (a 0.26 mm relax
perturbation moved this dataset 12.5 m / 16 loop edges, R3 caveat);
per-segment analysis shows R4-ship better than baseline in the
structural segments (0-2k: 38.2 vs 53.3 — the earlier first snap;
10-14k tail better) with no segment systematically damaged, while the
un-escalated run's tail was worse than BASELINE (its wrong snap is a
real, non-chaotic 16 m harm — hence the guard).

Runtime: MIT 21 ms/kf (R3 21, baseline 14); Intel 18. Reproduce:
`python3 ssp_hier.py selftest | drought | hyintel data/intel.log`;
MIT driver mit_hy4.py, recall r4_recall.py, seed-4 falsification trail
r4_diag_seed4.py / r4_margin.py, all in the session scratchpad;
trajectories mit_traj_hy4_r4ship.npz (+ r4v2 = unguarded). Selftest
additionally covers whitened+NMS+sharded coarse-hyp identity, the
junk-shard robustness case, per-era offset grouping, and the
coarse-band-only assert.

## Spatially-anchored per-scale submap arrays (ssp_scale_arrays.py, 2026-07-07): the O(area) memory win is real for the FRONTEND but loop closure structurally fails — persistent spatial accumulation smears old-pass content, biasing every loop Z; a map primitive used for CLOSURE must hold drift-consistent single-burst content, which only EPHEMERAL segments guarantee

Motivation: HY4 segments are O(time) (2 vectors per 5-kf segment). The
memory-efficient alternative is O(area): per matched-band ring (lam
0.25/0.5/1/2) an array of graph-anchored cells (size c*lam, c=8 -> 2/4/8/16 m),
HIT-KEYED (each occlusion-filtered sample writes to the cell containing the
HIT, not the robot, per the range gate), each cell a small submap anchored to
its first-writer's trajectory node (rides the graph rigidly; 60-angle block +
d/dtheta derivative, HY4 semantics per cell). Content lives where the geometry
is, so a wall is queryable from anywhere near the WALL — the content-location
fix anchor-keyed segment gathering structurally cannot do. Writes are
pass-disciplined (cross-pass gate: a later pass writes the base cell only if a
verified closure just landed nearby, else opens a parallel per-pass cell);
queries are LIBERAL (VSA superposition — the frontend sums every cell in
radius, both staggered grids, all pass layers, no merge, no relevance logic;
loop constraints keep evidence independence, matched against a SINGLE nearest
old-pass cell set). Backend inherited verbatim from BoundedSLAM. Selftest
covers write/query round trip, exact + first-order rigid ride, cross-pass gate,
two-layer additive query, loop epoch segregation.

Bench (room/sparse/corridor, seeds 1-4 paired vs G1 segments and HY4; ATE cm
mean, f/e = GT-false/accepted loop edges, map KB):

| config | room | f/e | sparse | f/e | corridor | mem note |
|---|---|---|---|---|---|---|
| G1 = BoundedSLAM | 6.4 | 7.2/142 | 12.1 | 9.0/120 | 379 | 841/852/1102 KB |
| HY4 | 6.4 | 7.2/142 | 12.1 | 9.0/120 | 379 | 564/572/739 KB |
| SA | 15.5 | 28/82 | 52.2 | 20/52 | 318 | 623/675/350 KB |
| SA-nogate | 31.8 | 24/32 | 24.8 | 6/19 | - | 311/180 KB |
| SA-recent | 33.4 | 51/81 | 49.2 | 21/64 | - | 572/667 KB |

The pre-registered pooled-mean gate "PASSES" (SA 1.285 m vs min(G1,HY4)
1.325 m, ratio 0.969) but is MISLEADING — the pool is swamped by the corridor
world's 3.8 m absolute error; on the well-behaved worlds SA is 2.4x worse
(room) and 4.3x worse (sparse) even at 750 frames. The smoking gun is the
false-edge rate: **SA 34-38% vs G1/HY4 5-7%**. Ambiguous cell content ->
the matcher lands on wrong poses. The hit-keyed content-location benefit is
real but tiny: att_cell_only (candidates only hit-keying finds) = 0 on
room/sparse, 18 on corridor; accepted-where-anchor-keying-misses = 4 across
all corridor runs (40 on Intel) — nowhere near paying for the accuracy loss.
SA is also not reliably smaller than HY4 on the bench (parallel cells inflate
it: 159-272 of them).

Intel (`ssp_scale_arrays.py carmen data/intel.log` + parametrized probes):

| config | ATE m | map | note |
|---|---|---|---|
| HY4 (reference) | 2.44 | 5.24 MB | shipped |
| SA default (cap 500, all) | 15.51 | 3.4 MB | saturation kills it |
| SA cap=inf, c=8, all | 12.96 | 3.5 MB | saturation was ~2.5 m of it |
| SA cap=inf, c=4, all | 14.39 | 10.8 MB | smaller cells WORSE + 3x mem |
| SA cap=inf, c=8, recent frontend | 12.47 | 3.5 MB | recency barely helps |
| SA cap=inf, recent, hard veto | 13.88 | 3.7 MB | strict gating no help |
| SA cap=inf, recent, cohT=0.85 | 11.18 | 3.7 MB | strict gating no help |
| **SA FRONTEND-ONLY (no closures)** | **4.34** | **4.5 MB** | the map is a GOOD target |

THE REVERSAL AND THE MECHANISM. Frontend-only SA is 4.34 m — BETTER than any
closure-enabled SA (11-15 m). Loop closures make it WORSE by ~8 m. So the
O(area) cell map is a perfectly good FRONTEND matching target; the failure is
entirely in loop closure. Why: the frontend matches recent, still-sharp cell
content; loop constraints match the current scan against OLD-PASS cells
(epoch-segregated, the most-accumulated and most drift-smeared). A cell
persistently sums many observations made at different drift states at ONE
frame, so its geometry is smeared into a broad, weak correlation peak; the
cmatcher lands on a biased pose; the biased Z passes every gate (coherence
cannot separate smear-bias from truth — a smeared match is broad but still
correlated), so the graph believes it and is dragged off the good frontend
trajectory. Stricter coherence vetoing (hard veto; cohT 0.55->0.85) removes
edges roughly uniformly and never recovers toward 4.34 — the surviving
closures are still net-negative. The SAME backend that lifts HY4's ~8 m
frontend to 2.44 m corrupts SA's 4.34 m to 11-14 m, purely on Z quality.

Graph-anchoring the cell fixes the FRAME's drift-riding (verified exact in
selftest) but NOT the intra-cell WRITE-TIME smear — content written across
accumulated drift cannot be un-smeared by moving its frame. This is the R1
world-frame floor in a sharper form and it EXTENDS the law family: it is not
enough for the loop/matched band to live in graph-consistent memory (R2) — the
CONTENT of a primitive used for loop closure must come from a single
drift-consistent burst. Ephemeral per-segment vectors guarantee this (5 kf =
one consistent burst, then frozen); persistent per-area cells violate it by
construction. Memory efficiency via spatial accumulation trades away exactly
the closure sharpness that SLAM needs. A freeze-and-reopen fix (each cell one
burst, any large gap opens a parallel) would restore sharpness but converges
toward O(area x visits) ~ O(time) and reintroduces the parallel-cell memory
inflation already seen on the bench — i.e. back to segments. HY4 stands as the
memory-efficient matched-band representation; the next lever is precision /
dimensionality on the segments themselves, not spatial reorganization.

Reproduce: `python3 ssp_scale_arrays.py selftest | bench | carmen data/intel.log`;
parametrized Intel probes in the session scratchpad (sa_intel.py).

## Memory levers that preserve closure sharpness (2026-07-07): complex64 segment storage is a FREE 2x; derivative-vector resolution reduction is a dead end

After the scale-arrays negative (spatial O(area) accumulation smears closure
content), the remaining memory levers attack the segment vectors WITHOUT
touching matching sharpness.

**complex64 storage — verified free 2x, shipped.** Segment vectors (segvec +
segder) stored at complex64 (8 B/component) instead of complex128 (16 B).
Phasor bundles are numerically forgiving: round-trip relative vector error
~2.4e-8, and the match-peak shift on real Intel geometry is **0.000 mm /
0.00 millideg** (a wrong loop-closure decision needs cm-scale pose error;
7 significant digits is ample). Paired verification:
- Bench (room/sparse, seeds 1-2): ATE identical to 3 decimals (dATE +-0.000),
  memory exactly halved (832 -> 416 KB).
- **Full Intel: ATE 2.440 m / 1.553 median BIT-IDENTICAL, 80 loops, map
  7852 -> 3926 KB.** Runtime unchanged (query arithmetic upcasts to c128;
  only storage is 64-bit).
Exposed as `BoundedSLAM.store_dtype` (default complex128 to keep the ledger's
bit-exact HY4/drought/MIT numbers unperturbed — precision can shift a
chaos-sensitive drought threshold); `ssp_bounded_carmen.py` sets complex64 by
default (`c128` arg restores full precision). Stacks with HY4's matched-band
split: HY4 5.24 MB -> ~2.6 MB at complex64.

**Derivative-vector resolution reduction — dead end.** The d/dtheta vector
only corrects sub-3-deg rotation, so half angular resolution (30 of 60 angles,
nearest upsample) was tried for a further 0.75x. It FAILS: naive ::2
downsampling aliases the derivative's genuine full-resolution angular
structure, landing rel-error 0.0999 — WORSE than dropping the derivative
entirely (0.0898; full-resolution derivative is 0.0725). The derivative
carries real high-angular-frequency content (confirming the "4x load-bearing"
ablation); it cannot be cheaply compressed. complex64 remains the clean win.

CROSS-LOG VERIFICATION (independent audit, 2026-07-07, BLAS threads pinned for
a single-variable comparison): fr079 c128==c64 BIT-IDENTICAL (5.523 m, 5.22 ->
2.61 MB); ACES c128==c64 BIT-IDENTICAL (6.212 m, 5.30 -> 2.65 MB). MIT (14,499
kf, chaos-sensitive) is NOT bit-reproducible under c64: 42.95 -> 66.02 m, a
23 m swing — but same-order as MIT's intrinsic non-determinism (BLAS threading
alone moves it ~5 m; a 0.26 mm relax perturbation moves it 12 m), with NO new
failure mode (both degrade gracefully, revisit-density-limited). VERDICT
(validated): keep c64 OPT-IN — free on bounded/revisit-dense logs, unsafe to
make the global default on chaos-sensitive long runs. Shipped policy correct.

MEMORY-FRONTIER HONEST FRAMING (audit): the shipped representation
(HY4 matched-band + c64, 3.75 KB/segment) is 2.7 (MIT) - 4.8 (Intel) KB/m^2 —
i.e. 3-12x DENSER per m^2 than a 5 cm occupancy grid (400 B/m^2 at 1 byte/cell)
and 15-110x denser than 2D NDT. SSP is NOT bytes/m^2-competitive with classical
area maps; a 5 cm occupancy grid of Intel is ~0.2-0.9 MB vs our 2.6 MB. The
memory win over the in-repo baselines (ICP 15-35 MB, RBPF 39-56 MB) holds ONLY
because those store HISTORY (ICP retains every scan = O(time); RBPF = particles
x grids). The genuine, defensible properties are: O(area)-BOUNDED not O(time)
(Intel plateaus at 698 segs / 84% of the cell-cap ceiling; MIT grows dead-
linear at 1.26 seg/m as new corridor is exposed), HISTORY-FREE continual
folding, and an ALGEBRAICALLY TRANSFORMABLE map — not raw spatial compactness.

Net memory story after this session: HY4 matched-band split (0.67x) x
complex64 (0.5x) = ~0.33x of the original 8 MB shipped map at bit-identical
Intel accuracy, i.e. the deliverable's map is ~2.6 MB with no accuracy cost —
and the scale-arrays study establishes WHY going further (spatial O(area)) is
not free: it trades the closure sharpness SLAM needs.

## Iterative slow-to-fast scale-cascade loop closure (2026-07-07)

`ssp_cascade.py` (new, imports only — edits nothing). Applies classic
multi-wavelength phase unwrapping to loop-closure VERIFICATION on the shipped
ring lattice (lam 0.25/0.5/1/2/5.3/12.8 x 60 angles). Given a candidate
(query scan, old-pass bundle B, seed pose) the cascade walks rings SLOW->FAST:
stage 0 = wide phase-correlation on the 12.8 ring (rotation via exact
`rot_permute` x translation) for a rough (theta_0,t_0); each finer stage i
seeds from i-1, searches a translation window ~lam_{i-1}/4 + a 1-step rotation
refine on ring i's phase (`grid_scores` on that ring's mask), and records the
correction d_i=|t_i-t_{i-1}| and per-ring coherence coh_i. Three acceptance
tests: (a) unwrap-consistency (reject if any d_i > lam_{i-1}/2); (b)
coherence-profile ratio coh_fine/coh_coarse (fine=lam0.5/0.25, coarse=lam
12.8/5.3) — a ratio, so nominally scale-invariant; (c) final-precision
(finest coh + correction within noise). Unwrapping condition holds on the
octave ladder: the tightest rung is 5.3->2 (2.65x, window 1.33 m vs ring
ambiguity radius 1.0 m); empirically the coarse stage always lands inside the
radius (unwrap pass rate 1.00 on every bench below).

**The mechanism is real and the profile out-separates single-scale coherence
ON A LABELLED CORRIDOR BENCH.** Self-similar repeated-bay corridor world
(identical alcove every 6 m = coarse alias; a per-bay jagged wall fingerprint
that blurs out at lam>=2 m; `shared` blends a common fine texture so twins keep
MODERATE fine coherence). Selftest: a strong genuine passes all three tests; a
constructed corridor twin (query bay k vs map bay k+-1, seed one period off so
the coarse alcove aliases) collapses coh coarse->fine (0.96->0.49) and is
REJECTED by the profile while a single-scale fine-coherence veto ACCEPTS it —
and it must, because a genuine PARTIAL-OVERLAP revisit has fine coh 0.42
(BELOW the twin's 0.49) yet profile ratio 0.67 (ABOVE the twin's 0.51): any
absolute fine-coherence bar low enough to keep that genuine also admits the
twin. GT-labelled ROC (160 genuine w/ mixed partial-overlap + noise
degradation, 135 twins): profile ratio AUC 0.897 / profile slope AUC 0.948 vs
single-scale fine coherence AUC 0.782 (coarse coherence AUC 0.19 — coarse is
HIGHER on twins, it locks the alias). A/B at matched 90% genuine recall: the
profile admits 31% of twins (42/135 false edges) vs single-scale's 76%
(102/135) — false edges more than halved at equal recall. Whitening the
scatter (cascade_roc.png right panel): at equal fine coherence, genuine sit
above twins in ratio. WHY it works: the profile's denominator (coarse
coherence) divides out the per-candidate absolute level that varies with
overlap/clutter — the exact common-mode nuisance that swamps a single-scale
absolute threshold.

**But the absolute profile threshold does NOT transfer to real data — the
Intel regression is the honest bound.** 80 genuine Intel revisits (GT-close,
>400 kf apart; bundle from 30 GT-posed old keyframes, seed at GT) have profile
ratio mean 0.456 / median 0.438 (coarse coh 0.765, fine coh 0.350): genuine
real-log revisits show the SAME coarse>fine collapse as twins, because a
180-deg-FOV SICK revisit's viewpoint change, partial overlap and dynamic
clutter are all FINE-SELECTIVE (they kill the short wavelengths first, exactly
like a twin). The corridor-tuned absolute threshold 0.55 keeps only 17/80
(21%) of genuine Intel closures; even 0.35 keeps 50%; a session-relative
threshold (0.55 x session median = 0.24) recovers 66%. So the "scale-invariant,
absolute-level-free" hope is only HALF true: the ratio removes the
per-candidate absolute level WITHIN a domain (hence the better within-domain
AUC), but HOW MUCH fine degrades relative to coarse on a genuine closure is
itself domain-dependent (sensor FOV, viewpoint diversity, clutter), so a fixed
ratio threshold still does not transfer. A drop-in predicate (`accept()` is
provided) must be session-calibrated exactly like the shipped coherence veto
it aimed to replace — it relocates the transfer problem from an absolute
coherence bar to a ratio bar, it does not eliminate it.

**Two structural findings, honest either way (the task's anticipated result).**
(1) The unwrap-consistency test is INERT in smooth corridors: both genuine and
twin corrections stay < lam/2 (a corridor lets the finer rings slide to a local
fit rather than jump a period), so ALL discrimination comes from the
coherence-profile, none from unwrap magnitude — the geometric-consistency
signal the cascade was built around does not fire where the aperture problem
lives. (2) The cascade INHERITS the coarse rings' place-recognition limit
(R3/R4): if lam 5.3/12.8 cannot distinguish two bays (they can't — that IS the
corridor self-similarity), stage 0 cannot pick the right bay, so the cascade
cannot RECALL a genuine closure the coarse ring missed. What it CAN do is
REJECT a twin the coarse ring locked onto (precision, not recall) — within-
match multi-scale consistency raises precision on coarse-ambiguous candidates
but cannot exceed the coarsest ring's place-recognition information for recall.
Net verdict: a materially better twin DISCRIMINATOR on labelled synthetic
corridors (AUC +0.11, false edges halved at equal recall), NOT a drop-in
Intel-safe predicate at any fixed threshold; the fine band's viewpoint/overlap
fragility on real 180-deg-FOV logs is indistinguishable, within one match, from
a corridor twin's collapse. Reproduce: `python3 ssp_cascade.py
[selftest|roc|intel|all]`; figure cascade_roc.png.

### R5 refinement (PCM consensus-set admission for drought closures, 2026-07-07): the R3/R4 single-pending pairing is replaced by Mangelson et al.'s Pairwise-Consistent-Measurement-Set maximization (SE(2) cycle-consistency graph + max clique); a principled admission layer that SUBSUMES R4's deep-search-escalation hack and reproduces every shipped number BIT-IDENTICALLY (bench 4/4 5.0-6.1 cm, Intel 2.440, fr079 5.523, MIT 42.952) — and delivers the verdict R4 could only conjecture: cross-edge consistency does NOT break the deep-corridor limit because the deep corridor musters no exploitable consensus, true or false

WHAT SHIPPED (HybridSLAM drought path, ssp_hier.py; normal-closure path
untouched). R4 diagnosed the MIT bottleneck as VERIFICATION information: a
corridor-twin false closure is INTERNALLY consistent (the measured wrong pair
agreed tighter, 0.54 m/0.6 deg, than the genuine junction snaps at
0.72 m/2.6 deg), so no per-pair residual separates truth from twin, and R4's
two-consecutive-consistency gate is defeated. R5 implements the prescribed
fix — "verification information beyond a single pair":

1. PCM PAIRWISE CYCLE-CONSISTENCY (`_pcm_cycle`). Two candidate drought edges
   e1=(a1,b1,Z1), e2=(a2,b2,Z2) have cycle error
   C = Z1 . seq(b1->b2) . Z2^-1 . seq(a2->a1), seq(x->y) = anchors[x]^-1 .
   anchors[y] read off the anchor graph (odometry-rigid on the dangling side,
   constrained on the old side). C is frame-independent (all factors relative)
   and equals identity iff the two edges imply the SAME rigid correction of the
   dangling frame — the coordinate-clean generalization of R3/R4's implied-
   correction agreement. Selftest verifies C~=I for genuine and for twin pairs,
   ~5 m for a truth/twin cross pair.
2. CONSISTENCY GRAPH + MAX CLIQUE (`_pcm_admit`). A buffer of verified
   candidate edges (cand_window=200 kf) is turned into a graph (edge i~j iff
   independent — distinct anchors, >= drought_every kf apart — and the cycle
   clears the R4-calibrated box gate pair_tol_t/pair_tol_r); the MAXIMUM clique
   containing the freshest fire is found by Bron-Kerbosch with pivoting (tiny
   graphs, real-time). Only clique edges enter the pose graph; pre-alignment
   and insertion are the R3 machinery verbatim, applied to the whole clique.
3. BREADTH ESCALATION AS CONSENSUS SIZE (deep_noff recast). R4's ad-hoc "hold
   the pair, demand a third consecutive fire when the sweep exceeded 6000
   cells" becomes: if any admitted-clique member was found under deep search
   (noff > deep_noff), the required clique grows pcm_min_clique(2) ->
   pcm_deep_min(3). A deep closure must be part of a THREE-way consistent
   family, not just a pair. This is the search-breadth crowding law expressed
   as an evidence requirement.

The KEY INSIGHT PCM exploits and the old single-pending gate could not: a
genuine closure is consistent with OTHER genuine closures (they all imply one
global correction), while a twin family is consistent only with its own twins
and inconsistent with the true junction snaps + the odometry cycle — so the
trusted family is the larger clique. Selftest (extended): a synthetic graph of
3 mutually-consistent TRUE edges + 2 twin-FALSE edges (fixed 5 m wrong shift)
admits EXACTLY the 3 true (max clique containing the freshest); a lone twin
pair still admits at size 2 UNLESS a member is deep (then held for want of a
third); the 3-true set with one deep member still admits.

ACCEPTANCE (all four, ship code, deterministic):

1. selftest — PASS (`python3 ssp_hier.py selftest`): the PCM cycle identities,
   the truth/twin cross-inconsistency, the 3-true/2-twin clique selection, and
   the deep-escalation size gate, alongside the R3/R4 coarse-hyp/pre-align/
   raster asserts.
2. Drought bench (`python3 ssp_hier.py drought`) — BIT-IDENTICAL to R4: seeds
   1-4 armed 5.0 / 5.4 / 5.9 / 6.1 cm (baseline 329-352), 4/4, both engineered
   droughts broken, snaps at kf 1176/1575. The genuine bench pairs form a
   consistent clique of 2 at the same attempts the old gate paired them.
3. MIT (mit_hy4.py, 14,499 kf): 42.952 m / 39.118 median / 47.588 final-third,
   60 loop edges, 2 snaps at kf 1340 & 2980 (both distinctive junctions,
   clique=2, shallow noff 1943-3018) — TRAJECTORY BIT-IDENTICAL to the R4-ship
   run (max|est-diff| = 0.000, same 60 edges). Recall/attempt 0.04 (106 true
   revisits, 4 stage-1 hits, 102 wrong-peak, 0 sub-gate — retrieval untouched).
   No wrong snap.

   CONSENSUS-SET STATISTICS (the primary evidence, per the MIT chaos-band
   caveat). Of 41 verified fires, max-clique-size distribution is {1: 37,
   2: 4}; NO fire ever reached clique 3. The two snaps are the two clique-2
   fires at shallow junctions. In the DEEP corridor (noff 10672/15348, 31
   fires) clique_max reaches 2 only twice — kf 9264 (the EXACT R4 twin pair
   9236+9264) and kf 13940 — both HELD by pcm_deep_min=3 and never admitted.
   The many junction-hotspot fires near (-20, 63) are pairwise-INCONSISTENT
   scatter: clique_max stayed 1 even with 3-6 co-located candidates in the
   buffer (kf 11840-11980, 13380-13548) — they match different wrong twins /
   different drift states and imply DIFFERENT corrections, so they form no
   family at all.
4. Regression — Intel 2.440 / 1.553 / 6.420 m, 80 loop edges, 6 pruned,
   drought 43 try -> 26 hyp -> 3 verified -> 0 snaps; fr079 5.523 / 3.533 /
   14.617, 32 edges, 2 pruned, 23 -> 10 -> 3 -> 0. Both BIT-IDENTICAL to
   R3/R4/shipped. The 3 verified fires per run do not form a consistent
   independent pair, so PCM admits nothing — the mechanism is inert where
   droughts do not bind, and produces 0 false snaps.

VERDICT — PCM admission does NOT break the corridor limit, and now we know
exactly why. R4 said "retrieval was never the bottleneck; verification
information is," and conjectured that consistency beyond a single pair was the
missing evidence. R5 implements precisely that cross-edge consistency and
measures the answer: there is no consensus to find. (a) There is no TRUE
clique — retrieval scores 0/106 stage-1 hits at deep true revisits (all
wrong-peak), so no genuine deep closure exists for a clique to lock onto; the
place-recognition information is simply absent at lam 5.3/12.8. (b) There is no
large FALSE family either — the deep-corridor wrong fires are mostly pairwise-
inconsistent scatter (clique_max=1) with only sporadic 28-kf twin PAIRS, which
the consensus-size requirement correctly holds. This sharpens the corridor
verdict one level past R4: the limit is not a correlated-alias family fooling
the backend (DC-GM's case, which PCM/consensus clustering is built to catch) —
it is upstream information STARVATION that no backend consensus can undo, true
or false. [ROOT CAUSE LATER CORRECTED — see "Richer place descriptor" and
"Ring-key shortlister" below: the coarse-band starvation is a representation
artifact (retrieval recovers to 0.808); the true wall is verification/consensus,
which SeqSLAM then shows is sequence-ambiguous. This R5 line is kept as
stratigraphy.] The gain R5 books is not accuracy (it matches R4 bit-for-bit) but
architecture and evidence: R4's three separate heuristics (single-pending
replace-on-conflict, implied-correction-at-anchor agreement, and a bolt-on
deep-search hold) collapse into one principled PCM admission layer with 8 years
of literature behind it, the deep-search hack becomes a consensus-size rule,
and the "information-limited, not tuning-limited" verdict is now demonstrated
rather than argued. rmse 42.952 sits in the same MIT chaos band as the R3
record 38.03 (a 0.26 mm relax perturbation moves this dataset 12.5 m); the
closure/consensus statistics, not the single-run rmse, are the evidence.

Runtime: MIT 21 ms/kf (unchanged; the max-clique is over <= ~6-node buffers),
Intel 18. Reproduce: `python3 ssp_hier.py selftest | drought |
hyintel data/intel.log` (and data/fr079.log); MIT driver mit_hy4.py, recall
r4_recall.py (both session scratchpad); trajectory mit_traj_hy4_r5pcm.npz
(bit-identical to mit_traj_hy4_r4ship.npz). Selftest additionally covers the
PCM cycle identities, truth/twin cross-inconsistency, 3-true/2-twin clique
selection, and the deep-escalation consensus-size gate.

### Session-relative profile veto in the deliverable — tested and REJECTED (2026-07-07, Opus)

Follow-up to the scale-cascade result: wire its profile statistic (fine-ring
coherence / coarse-ring coherence) as a SESSION-RELATIVE veto in
`ssp_bounded.py` try_constraint (EMA baseline `prof_ref` over accepted
closures, like the coherence veto's `coh_ref`; veto when ratio <
prof_target * prof_ref). Hypothesis: catch corridor twins whose ABSOLUTE
coherence clears the veto but whose fine geometry collapsed.

RESULT — catastrophic over-veto, no operating point works. Paired bench (4
seeds), profile OFF -> ON: room 6.4 -> 15.1 cm (142 -> 10 edges), sparse
12.1 -> 41.1 cm (120 -> 4 edges), corridor 379 -> 380 cm (5 -> 4 edges,
aperture-limited regardless). IDENTICAL across prof_target in {0.2, 0.3, 0.4}
— the threshold is not the binding factor.

CORRECTED MECHANISM (the important part — it sharpens the cascade agent's
"relocates the transfer problem" verdict): the profile's discriminative
DIRECTION is geometry-specific. On the bench, GT-labelled accepted candidates
have GENUINE fine coh 0.760 / coarse 0.650 (ratio ~1.17) but FALSE fine 0.651
/ coarse **-0.031** (coarse near ZERO -> ratio EXPLODES to ~1e5). So the
bench's false closures (aperture aliases with weak coarse correlation) have
the OPPOSITE profile to the cascade's CONSTRUCTED self-similar-corridor twins
(coarse-locked, fine-collapsed, low ratio). The huge false-closure ratios
dominate the EMA `prof_ref`, which then vetoes the genuine closures (ratio
1.17 << 0.55 x huge). The cascade's AUC-0.948 separation is specific to
constructed corridor-twin geometry and does NOT generalise even to the
bench's own false-closure population — so the profile cannot serve as a
general veto. Reverted; the deliverable is unchanged. The scale-cascade
statistic remains a valid CORRIDOR-TWIN precision tool where that specific
geometry dominates (its ROC stands), but not a drop-in general veto — exactly
the transfer bound the cascade agent flagged, now with the sign-flip
mechanism measured.

## Belief-carrying frontend (ssp_belief.py, 2026-07-07): carrying pose belief through frontend ambiguity does NOT reduce drift — the aperture worlds have the TIGHTEST frontend surfaces, and wherever the belief fires at all it REGRESSES the result (bench +26 to +278 cm, Intel 2.44 -> 8.43 m). Honest negative, mechanism measured.

CONCEPT AS BUILT. `BeliefSLAM(BoundedSLAM)` (ssp_belief.py, NEW file; parents
untouched) replaces the frontend's gate-and-commit with a carried translation
BELIEF on the matcher's OWN coarse correlation surface — the 17x17 (+-0.48 m /
6 cm) grid `S.Matcher` already scans, no phasor CF, no new heavy compute. Per
keyframe: (i) LIKELIHOOD = the matcher's coarse score surface at the winning
heading, standardized and tempered `softmax(beta*zscore(s(d)))` — a sharp peak
concentrates, an aperture ridge / alias spreads or splits; (ii) MOTION PREDICT =
the previous posterior re-sampled onto the new grid (its center rides the
const-vel / odometry guess) and Gaussian-blurred by the per-step motion noise
(`ndimage.shift` + `gaussian_filter`, O(cells)); (iii) POSTERIOR = predict *
likelihood, renormalized. COMMIT RULE: unimodal AND tight (one mode, positional
spread < sig_cap) -> commit exactly as the shipped frontend (matcher argmax,
fold, 0.03 m seq sigma). Multimodal OR diffuse -> navigate on the belief mean,
inflate that span's seq sigma to the belief's own spread, optionally suppress
the map fold. Heading tracks the matcher argmax. Backend (seq/loop edges,
innovation gate, coherence response, IRLS+LOO, TRF relax) inherited UNCHANGED.
(Distinct from the abandoned `ssp_posefilter.py`, whose belief was a full
characteristic-function phasor vector + 9-way heading mixture — heavier; this
one lives on the grid the matcher already computes.)

SELFTEST (`python3 ssp_belief.py selftest`, deterministic) — the filter is
CORRECT in isolation on a corridor-fork toy: through 8 ambiguous frames the
carried belief HELD both forks (2 modes, sig_max ~18 cm, never commits) while a
single-commit argmax frontend flipped between forks (4/8 wrong) and baked those
jumps into its trajectory; at the distinctive fork frame the belief collapsed to
the true mode (tight, mean error 0 cm). Motion-predict re-centering and
mode/moment detectors are unit-tested (peak rides the moving grid center;
bimodal round-trip recovers both modes at +-0.18 m). So the negative below is
NOT a broken filter — it is the absence of the signal the filter needs.

BENCH (`python3 ssp_belief.py bench`, corridor/sparse/room, seeds 1-4 paired,
n=750; ATE cm mean [per-seed]; amb% = frontend keyframes the belief flagged
ambiguous; infl-seq = fraction of seq edges it inflated):

| world | config | ATE cm [seeds] | paired vs Bounded | amb% | infl-seq | false loops |
|---|---|---|---|---|---|---|
| corridor | BoundedSLAM (shipped) | 379.1 [288 280 222 726] | — | — | 6.7% | 5.2 |
| corridor | Belief (default) | 380.0 [258 281 222 759] | +0.9 (1/4) | 0.1% | 7.0% | 4.8 |
| corridor | Belief no-carry | 379.1 [288 280 222 726] | +0.0 (0/4) | 0.0% | 6.7% | 5.2 |
| corridor | Belief sensitive | 509.0 [620 386 418 613] | +129.9 (1/4) | 22.5% | 43.1% | 7.2 |
| corridor | Belief suppress-fold | 497.8 [597 435 524 436] | +118.7 (1/4) | 24.6% | 45.5% | 12.2 |
| sparse | BoundedSLAM (shipped) | 12.1 [15 11 9 13] | — | — | 1.2% | 9.0 |
| sparse | Belief (default) | 38.1 [17 112 12 13] | +26.1 (1/4) | 3.0% | 9.1% | 20.2 |
| sparse | Belief no-carry | 12.1 [15 11 9 13] | +0.0 (0/4) | 0.0% | 1.2% | 9.0 |
| sparse | Belief sensitive | 289.7 [718 92 158 192] | +277.7 (0/4) | 34.4% | 53.9% | 25.0 |
| sparse | Belief suppress-fold | 231.7 [265 214 217 231] | +219.6 (0/4) | 29.4% | 49.3% | 30.0 |
| room | BoundedSLAM (shipped) | 6.4 [7 7 7 5] | — | — | 0.7% | 7.2 |
| room | Belief (default) | 55.4 [7 7 101 106] | +49.0 (0/4) | 1.7% | 7.2% | 13.0 |
| room | Belief no-carry | 6.4 [7 7 7 5] | +0.0 (0/4) | 0.0% | 0.7% | 7.2 |
| room | Belief sensitive | 123.4 [19 248 162 64] | +116.9 (0/4) | 22.6% | 41.1% | 32.0 |

Findings:

1. **The aperture worlds have the TIGHTEST frontend surfaces — there is no
   ambiguity for the belief to catch.** Diagnostic (per-keyframe posterior
   positional spread sig_max, seed 1, 750 kf, beta=4): corridor p90/p99 =
   8.8/10.2 cm, 0.0% above 13 cm, 0.0% multimodal — vs room p90 8.7 and sparse
   p90 9.7. The classic-aperture corridor is if anything TIGHTER than the
   well-conditioned room, and there is NO sig_max separation between them.
   Mechanism: the frontend matches against the RECENT local bundle (last 12
   anchors) with a const-vel prior; aperture slip is a few cm/frame and resolves
   to a single confident peak, not a >0.48 m ridge. The corridor's large ATE is
   accumulated single-peak slip baked into an already-drifted local reference
   map — a GLOBAL-drift error the belief cannot see, because within the local
   window the match IS well-determined. This is the Rao-Blackwell split working
   AGAINST a frontend belief: the map is deterministic given poses and the
   frontend already extracts the ML pose crisply; the uncertainty that matters
   lives in the graph, not the per-frame surface. (Confirms the ledger prior:
   "the frontend map matching is generally good.")

2. **Carrying is the ONLY thing that makes the belief fire — and firing HURTS.**
   The no-carry ablation (fresh motion prior each frame) reproduces BoundedSLAM
   BIT-IDENTICALLY on ALL three worlds (corridor 379.1 [288 280 222 726], sparse
   12.1 [15 11 9 13], room 6.4 [7 7 7 5]; 0.0% amb each): a tight per-frame
   motion prior x a tight likelihood is always tight, so it commits the matcher
   argmax exactly like the shipped single-commit frontend. Only the CARRIED
   prior accumulates enough blur to trip the ambiguity threshold — and those
   trips are NOT the drifting frames. Acting on them (belief-mean navigation +
   seq-sigma inflation) monotonically hurts, scaling with how much it fires:
   room-default +49.0 (fires 1.7%, seeds 3-4 blow 7 cm -> ~1 m), sparse-default
   +26.1, corridor-sensitive +129.9, sparse-sensitive +277.7 (seed 1 718 cm).
   Inflating a large slice of the seq chain floppifies the graph and the extra
   false loop closures (room 7.2 -> 32, sparse 9 -> 25) then pull it. The
   belief's spread is honest uncertainty about a pose the frontend nonetheless
   nailed; treating it as edge uncertainty is strictly counterproductive.

3. **Even the conservative default regresses the room control** (+49.0, 0/4
   better). It is only near-neutral (corridor +0.9) where it essentially never
   fires (0.1%). There is no threshold that fires on the drift-causing frames
   without firing on well-conditioned ones — no sig_max separation exists
   (finding 1). Fold suppression makes it worse, not better (adds false loops).

REAL LOGS (`ssp_belief.py carmen data/intel.log`, same harness, belief on vs
off, deterministic):

| Intel | ATE rmse | median | ms/kf | infl-seq | amb% (multi/diff) |
|---|---|---|---|---|---|
| BoundedSLAM (belief off) | **2.440 m** | 1.553 | 16 | 8.9% | — |
| BeliefSLAM (default) | 8.433 m | 6.285 | 18 | 23.5% | 5.6% (192/129) |

Intel REGRESSES 3.5x (2.440 -> 8.433 m; the 2.440 is the shipped number
reproduced bit-exact in this harness with belief off). At the default threshold
the belief fires on 5.6% of keyframes and inflates 23.5% of the seq chain; on
Intel's floppy graph that destroys the seq-chain stiffness the shipped
early-stop (max_nfev=30) regularizer relies on to hold flat valleys — the same
flat-valley slide the Tikhonov and damped-GN experiments already documented
(fr079 2.8 -> 10.4 m under looser regularization). fr079 was NOT run: the
pre-registered gate ("helps materially on the ambiguity worlds WITHOUT
regressing room -> Intel + fr079") failed at the bench (room itself regresses)
and again on Intel; fr079 is the floppiest of the three graphs and would
regress hardest. The Intel run is the confirmatory real-log negative.

VERDICT (honest failure, exactly as the task's honest-failure clause
anticipated). Carrying pose belief through frontend ambiguity does NOT reduce
the drift that makes loop closures necessary, because on this stack that drift
is not represented as frontend ambiguity: the coarse correlation surface is
unimodal and tight even in the aperture corridor (the frontend is
confidently-and-consistently slightly-wrong), so the belief faithfully reports
"commit" on exactly the frames that drift. The shipped single-commit +
drift-scaled-seq-sigma + innovation-gate + loop-closure stack already handles
this regime; the belief adds a per-frame ambiguity signal that (a) reproduces
the shipped frontend BIT-FOR-BIT where it stays silent (no-carry, all worlds)
and (b) actively regresses every log wherever it is made to speak, because
seq-sigma inflation on non-drifting frames floppifies an already-floppy graph.
The RBPF advantage this tried to import — a carried multi-hypothesis pose
posterior — pays off in RBPF because each particle carries a GLOBAL grid
re-scored against OLD geometry (a loop-closure-like signal); a belief on the
frontend's LOCAL-recent surface has no access to that signal and cannot
substitute for the backend. Shipped `BoundedSLAM` frontend STANDS; `ssp_belief.py`
is retained with its passing selftest as a documented negative and a reusable
grid-Bayes belief primitive. Reproduce: `python3 ssp_belief.py selftest | bench
| carmen data/intel.log [--base]`.

## Derivative-vector novelty ablation (2026-07-07, Opus): the d/dtheta correction is a genuine first-order Lie correction; an equal-storage "more angles" alternative is an open question

The scout flagged the d/dtheta companion vector as likely-novel in mechanism
(Krausse et al. NICE 2025 solve sub-grid rotation via bump-vector convolution,
not an analytic derivative) and prescribed an ablation. On real Intel segment
geometry (rings x 60 angles), reconstruction rel-error of the world-frame ring
block vs the exact re-encode, as a function of sub-grid angle delta (the
nearest-3-deg permutation keeps delta in [0, 1.5 deg]):

| delta (deg) | permutation only | + d/dtheta | ratio |
|---|---|---|---|
| 0.5 | 0.040 | 0.014 | 2.8x |
| 1.0 | 0.077 | 0.052 | 1.5x |
| 1.5 | 0.107 | 0.100 | 1.1x |
| 2.0 | 0.129 | 0.150 | 0.9x |
| 3.0 | 0.165 | 0.242 | 0.7x |

The correction behaves EXACTLY as a genuine first-order (Lie-generator) term:
maximal benefit at small delta (2.8x at 0.5 deg), tapering to the lattice edge
(1.1x at 1.5 deg), and turning HARMFUL beyond ~2 deg (0.7x at 3 deg) — accurate
near the expansion point, wrong far from it. Since the operating range is
[0, 1.5 deg] it lives entirely in its beneficial regime. This is the clean
proof the scout wanted that it is an analytic derivative companion, not a
heuristic. (The ledger's "4x load-bearing" is the full-pipeline ATE impact,
which amplifies this modest per-query reconstruction benefit through the
matching + graph cascade.)

OPEN QUESTION surfaced: the derivative doubles per-segment storage (segvec +
segder); the SAME storage spent on doubling angular resolution (60 -> 120
angles, no derivative, lattice step 3 -> 1.5 deg) gives a flat ~0.04
reconstruction error with no large-delta blow-up. Whether 60-angle+derivative
or 120-angle-no-derivative MATCHES better at equal storage is untested (they
differ in COMPUTE — 120 angles doubles the matcher cost while the derivative
only adds storage — so it is a tradeoff, not a free swap). Needs a lattice
rebuild; flagged, not chased.

## Richer place descriptor vs the corridor limit (ssp_scancontext.py, 2026-07-07): the MIT corridor revisits ARE recoverable — Scan-Context ring-key recall@1 0.272 / @40 0.674 where the coarse SSP relo ring scored 0/106; late/deep-corridor third recall@40 0.717. VERDICT: the limit is REPRESENTATIONAL (coarse-band + world-frame summing), not the environment; the raw corridor scans are distinguishable, the lam 5.3/12.8 summary throws the signal away

R3/R4 left the MIT corridor verdict deliberately scoped: within the SSP
architecture (coarse relo band lam 5.3/12.8, world-frame-summed) the deep-
corridor drought is a "place-recognition information limit" — 0/106 stage-1
hits at true revisits (R4), 0/25 global-peak in the offline capacity study.
The unanswered question: is that information GONE FROM THE RAW SCAN (an
environment/sensor limit — corridors mutually indistinguishable) or merely
lost by the coarse SSP summary (a representation limit a richer descriptor
would recover)? This experiment settles it with the standard classical
control.

EXPERIMENT (`ssp_scancontext.py`, NEW; reuses `ssp_slam_carmen.parse_flaser`
/`keyframes` by import, edits no SSP module). Scan-Context (Kim & Kim, IROS
2018) built from each keyframe's RAW scan (not the SSP encoding): a polar
bird's-eye n_ring=20 x n_sector=60 point-density matrix in the sensor frame;
the rotation-invariant RING-KEY (per-ring mean occupancy — yaw is a pure
column shift, so the row statistic is heading-free) for coarse retrieval;
the column-shift min-cosine SC distance for fine re-scoring. Evaluated on the
SAME true-revisit set the drought relocalizer was scored on (true revisit =
keyframe within 5 m in the REF frame of a >= 1500-kf-older keyframe; REF from
gfs, MIT via range-array identity + interpolation exactly as mit_hy4 /
r4_recall / mit_capacity; Intel via shared-base timestamps). recall@k = the
fraction of true-revisit queries whose ring-key top-k over ALL >= 1500-older
keyframes contains a correct within-5 m match. `selftest` covers ring-key
order-invariance and SC-distance yaw-shift identity.

RECALL@k (ring-key coarse retrieval; SC-rerank = ring-key top-100 re-ordered
by the column-shift SC distance; random = chance top-k given the revisit
density):

| dataset | revisits | @1 | @5 | @20 | @40 |
|---|---|---|---|---|---|
| MIT ring-key | 4520 | **0.272** | 0.417 | 0.582 | **0.674** |
| MIT SC-rerank | " | 0.254 | 0.400 | 0.569 | 0.659 |
| MIT random-chance | " | 0.014 | 0.068 | 0.240 | 0.412 |
| Intel ring-key | 4667 | 0.334 | 0.446 | 0.612 | 0.716 |
| Intel SC-rerank | " | 0.313 | 0.444 | 0.613 | 0.712 |
| Intel random-chance | " | 0.139 | 0.459 | 0.854 | 0.952 |
| **SSP coarse relo ring, MIT** | 106 (R4 attempts) | **0.000** | — | — | **0.000** |

MIT by run-third (ring-key recall@k): early (207) 0.126/0.164/0.261/0.348,
mid (2193) 0.281/0.415/0.560/0.663, **late/deep-corridor (2120)
0.277/0.444/0.635/0.717**. The deep final-hour corridor — the exact region
the SSP capacity study scored 0/25 on and the drought relocalizer could only
break at distinctive junctions — is recovered by Scan-Context at recall@40 =
0.717, HIGHER than the mid third. (Early is thin because GAP=1500 leaves a
small database near the start.)

FINDINGS:

1. **The information is THERE in the raw scan — the corridor limit is
   REPRESENTATIONAL, not environmental.** A richer classical descriptor
   built from the same keyframes recovers the MIT corridor revisits the
   coarse SSP rings cannot. The density-robust head-to-head is recall@1
   (the SSP relocalizer emits ONE hypothesis per attempt = top-1 semantics):
   Scan-Context's single best ring-key match is a true within-5 m revisit
   0.272 of the time on MIT (19x the 0.014 chance), 0.277 in the deep
   corridor — vs the coarse SSP ring's 0.000 (0/106). The corridors are NOT
   mutually indistinguishable; the lam 5.3/12.8 blur (plus world-frame
   summing) discards the door/alcove-scale radial structure that
   distinguishes them, and that structure survives in the raw scan.

2. **Where the recoverable signal lives: the RADIAL OCCUPANCY PROFILE (ring
   key), not fine yaw alignment.** SC-rerank (full column-shift alignment
   over the ring-key top-100) does NOT beat the ring key and slightly HURTS
   (@40 0.659 vs 0.674 on MIT, 0.712 vs 0.716 on Intel). Self-similar
   corridor twins align nearly perfectly, so the fine SC distance ranks a
   wrong twin above truth — the same "systematic self-similar wrongness"
   R4 saw defeat the fine verifier and the two-consistent gate. The
   discriminating information is the gross 20-ring radial profile (how far
   the walls are at each range band), which is exactly what the SSP coarse
   band collapses to 2 wavelengths and then world-frame-sums away. This is
   the mechanism of the SSP loss, pinned.

3. **Intel sanity holds and isolates the effect.** Scan-Context recalls
   Intel revisits well (recall@1 0.334, early-third 0.561) — the descriptor
   works on distinctive geometry, so MIT's non-zero recall is not a bug.
   Intel's chance baseline is high (revisit-dense small building: @40 chance
   0.95 exceeds SC@40, so @40 is uninformative there); the density-robust
   recall@1 = 0.334 (2.4x chance) is the honest Intel number. The
   Intel/MIT recall@1 gap is small (0.334 vs 0.272): Scan-Context treats the
   two logs almost alike, whereas the SSP coarse ring works on Intel
   (revisit-dense, no drought fires) and dies on MIT (0/106). The
   corridor-specific SSP failure is thus specifically a coarse-summary
   failure, reproduced and localized.

VERDICT: **SSP-LOSSY, not environment-limited.** This sharpens — does not
overturn — the R3/R4 conclusion, which was carefully scoped "within this
architecture (a finer relo band cannot be world-frame-summed, per the R1/R2
law)". The raw sensor data carries recoverable place information in the
corridors; the residual 0/106 is a property of the coarse-band +
world-frame-summed relo representation, not of the MIT environment. R3/R4's
"place-recognition information limit" is real but is the SSP band's limit, not
the scan's.

SKETCH — augmenting the drought relocalizer with a ring-key-style descriptor
(finding only; NOT wired into ssp_hier). The ring key sidesteps the R1/R2 law
that killed a finer world-frame-summed band: it is a per-keyframe (unbundled)
YAW-INVARIANT APPEARANCE signature (20 floats/kf ~ 160 B; MIT 14,499 kf =
2.3 MB, ~0.25% of the segment map, or ~1/5 that at one per 5-kf segment),
stored ALONGSIDE each segment anchor. The descriptor never moves with the
graph (it is frame-relative appearance); the candidate POSE it points to is
the anchor's current graph pose, which does — so it is graph-consistent
without any world-frame accumulation or capacity crowding (the two things the
capacity study blamed). Concretely: (a) replace drought STAGE-1 only — at a
drought, kNN the query ring-key against pre-drought segment ring-keys ->
top-40 candidate anchors (recall@40 0.67 puts the correct twin in the
shortlist, vs the coarse SSP band's ~0), seeding the EXISTING fine-band
cmatcher + session-relative coherence verification + two-consistent /
deep-search-escalation gate UNCHANGED; the column-shift also hands the
matcher a free yaw seed. (b) Do NOT trust top-1 (0.27 right = 0.73 wrong,
usually a corridor twin) and do NOT lean on SC re-ranking to disambiguate
(finding 2: it cannot). Disambiguation must stay with the multi-viewpoint
two-consistent gate: because independent revisit viewpoints each retrieve
THEIR correct twin, the consistency pairing — which R4 showed is defeated
only because the SSP band supplied consistent WRONG twins — now has the
correct candidate present to pair on. The honest expectation is bounded:
retrieval was necessary-not-sufficient (R4: "retrieval was never the MIT
bottleneck; verification information is"), so this restores the missing
precondition (correct candidate in the shortlist) but the wrong-twin
verification hazard is unchanged; the payoff rides on the two-consistent gate
finally seeing truth among the candidates. A per-segment ring-key store is the
concrete, R1/R2-legal, ~2 MB way to give it that chance.

Reproduce: `python3 ssp_scancontext.py` (MIT + Intel, ~90 s + ~40 s);
`python3 ssp_scancontext.py {mit,intel,selftest}`. Caveats: point-density
Scan-Context (the max-height 3D form has no planar analogue; a max-range
variant is the untested alternative the code notes); REF positions
interpolated from gfs (~1 m spacing, well inside the 5 m tolerance); recall@40
is revisit-density-inflated (use recall@1 for the density-robust claim);
retrieval is over the labelled-REF-span keyframes with unlabelled frames as
honest distractors; this measures RETRIEVAL recall, not end-to-end SLAM ATE
(the sketch above is a finding, not a built-and-measured system).

## Derivative vs angular resolution at equal storage (2026-07-07, Opus): the d/dtheta derivative WINS decisively — spend the storage on the derivative, not on more angles

Answers the open question left by "Derivative-vector novelty ablation": the
derivative doubles per-segment matched-band storage (segvec 240 + segder 240 =
480 components/seg); is that budget better spent DERIVATIVE-ON at 60 angles, or
DERIVATIVE-OFF at 120 angles (same 480 = segvec only, but the matcher works on
D=480 not 240 — 2x compute)? Built `ssp_angles.py` (new; imports only, edits
nothing): a parametrized rebuild of the `ssp_slam_loop` lattice (W/N_ANG/N_RING/
MAIN/ENC/ENC_MAIN; `rot_permute`/`encode` read those globals so they generalize
with zero edits) on the 4-ring matched band (lam 0.25/0.5/1/2), driving the
UNMODIFIED `BoundedSLAM` via a thin subclass that gates the derivative on the
existing `use_der` switch AND stores NO segder when off (honest memory). The
4-ring matched band is decision-identical to the shipped 6-ring lattice for
BoundedSLAM (the two coarse rings feed only coh[4:6]/diagnostics, never a
decision), so 60+der reproduces every shipped number bit-for-bit. Selftest:
`rot_permute` and `world_vec_seg` exact to <1e-13 (permutation identity, delta=0)
at n_ang=60 AND 120; the +der branch beats permutation-only at sub-grid delta.

Storage counted in complex components/segment on the matched band. **60+der and
120-noder are EQUAL storage (480/seg) — memory identical per segment by
construction — and differ only in compute (D=240 vs 480).**

**Intel (6205 kf, deterministic, ATE rmse m):**

| config    | store/seg | D (compute) | ms/kf | ATE   | loops |
|-----------|-----------|-------------|-------|-------|-------|
| 60+der    | 480       | 240         | 16.3  | **2.440** | 80 |
| 120-noder | 480       | 480         | 29.9  | 7.926 | 101 |
| 60-noder  | 240       | 240         | 15.8  | 5.238 | 48 |
| 120+der   | 960       | 480         | 31.0  | 5.537 | 93 |

**Equal-storage verdict: the derivative wins 3.2x (2.44 vs 7.93 m) at HALF the
compute** (16 vs 30 ms/kf; memory identical by construction). 120-noder does not
just fail to beat 60+der — it is beaten even by the HALF-storage 60-noder
(5.24 m). SURPRISE: on Intel more angles HURT regardless of the derivative
(120+der 5.54 > 60+der 2.44), and 120-noder finds the MOST loops (101) yet has
the worst ATE — the sharper 1.5-deg lattice over-fits onto Intel's
internally-drifted old bundles (genuine closures there sit at accepted-coherence
median ~0.49) and accepts more geometrically-off closures, while the analytic
sub-grid derivative on the coarser 60-angle lattice keeps the accepted closures
honest. Angular resolution is not the bottleneck; rotational FIDELITY is, and
the derivative delivers it in half the components and half the compute.

**fr079 (3071 kf, top-2 = the equal-storage pair; transfer/regression):**

| config    | store/seg | D   | ms/kf | ATE    |
|-----------|-----------|-----|-------|--------|
| 60+der    | 480       | 240 | 26.9  | **5.523** |
| 120-noder | 480       | 480 | 46.4  | 14.801 |

60+der reproduces the shipped 5.523 m EXACTLY; the equal-storage angle swap
transfers its Intel loss (2.7x worse, 1.7x compute). Both real logs agree.

**Bench (`ssp_bounded.run`-style paired seeds 1-3, laps=3; ATE cm, paired vs 60+der):**

| world    | 60+der | 120-noder | 60-noder | 120+der | notes |
|----------|--------|-----------|----------|---------|-------|
| room     | 6.8    | 7.6 (+0.8)  | 26.0 (+19.2) | 6.5 (-0.3) | false 6-8 all |
| sparse   | 11.8   | 12.4 (+0.7) | 32.2 (+20.4) | 8.0 (-3.8) | false 7-14 |
| corridor | 263.5  | 415.9 (+152)| 623.6 (+360) | 268.3 (+4.8) | 60-noder false 10.7 vs 2.7 |

On the CLEAN synthetic worlds the two equal-storage options are within ~1 cm
(120-noder marginally worse both), but on the environment-coupled CORRIDOR the
derivative's edge widens sharply (60+der 263 vs 120-noder 416 cm). Finding-6
holds among the NO-derivative options — more angles help the rotation-coupled
corridor (120-noder 416 beats 60-noder 624) — but the derivative is the more
storage-efficient AND compute-cheaper route to the same rotational fidelity: it
beats 120-noder while using D=240. Dropping the derivative for its half of the
storage (60-noder) is catastrophic everywhere (+19-20 cm clean, +360 cm
corridor, and 4x the false-edge rate in corridor: no sub-grid correction ->
more geometrically-wrong closures admitted).

**Is the derivative worth its half of the storage vs spending it on angles?
Unambiguously yes.** Dropping it (60-noder) loses 2-4x accuracy for the memory
saved; reinvesting the freed storage in angular resolution (120-noder) does NOT
recover it and adds 2x matcher compute — worse than 60+der on BOTH real logs
(Intel 3.2x, fr079 2.7x) and the corridor, and no better on clean rooms. The
derivative is a genuine first-order Lie correction whose accuracy-per-component
and accuracy-per-flop both dominate raw angular density; 60 angles + derivative
is the operating point. (120+der, double storage, is marginally best only on the
clean rooms and REGRESSES on both real logs — buying angles on top of the
derivative is not worth it either.) No wash: the derivative is load-bearing, the
angle swap is a net loss.

---

## Two new held-out logs (fr101, belgioioso) + the frontend do-no-harm gap

Added two held-out CARMEN logs to fill gaps in the transfer suite (dataset
survey + selection in `SotA/datasets.md`): **fr101** (Freiburg building 101 —
a dense-revisit loopy multi-room building, the regime where loop closure should
excel, missing between Intel and MIT's sparse corridors) and **belgioioso**
(Belgioioso Castle — non-Manhattan stone-wall/courtyard structure, probing the
encoder's one documented environment fragility). Both are zero-adapter for the
CARMEN driver (FLASER, 180 deg FOV; fr101 360 beams, belgioioso 361). fr101 has
clean shared-base timestamps; belgioioso's gfs is low-precision (MIT-class
corrupt), so its ATE uses the range-array-identity convention (matched 395/395).

Shipped BoundedSLAM (`ssp_bounded_carmen.py`, complex64, no retuning), plus a
frontend-only / odometry breakdown (`scratch_aces_diag.py`,
`scratch_belgioioso.py`), ATE rmse m vs the RBPF-corrected reference:

| log            | raw odo | frontend-only | +loops (shipped) | regime |
|---|---|---|---|---|
| Intel          | 24.2 | ~ (frontend essential) | **2.44** | odo drifts |
| fr079          | 14.4 |  —   | **5.52** | odo drifts |
| **fr101**      | 8.56 | 3.16 | **1.88** | odo drifts — best transfer |
| ACES           | 5.41 | 6.38 | 6.21 | odo excellent |
| **belgioioso** | 1.72 | 2.45 | 2.64 | odo excellent |

**fr101 confirms the dense-revisit hypothesis.** Everything helps and stacks:
frontend cuts odometry 2.7x (8.56 -> 3.16), loop closures cut the remainder 1.7x
(3.16 -> 1.88), on 53 accepted closures (vs ACES's 11) — the loopy building is
exactly where the closure machinery pays off. 1.88 m is the best held-out
transfer number in the project, at 1.9 MB / 337 segments, zero retuning.

**belgioioso corroborates the ACES failure mode — and relocates it to the
frontend.** A do-no-harm diagnosis on the two logs where odometry already beats
the shipped result (ACES 5.41 vs 6.21; belgioioso 1.72 vs 2.64) shows the
regression is NOT in loop-closure admission (the coherence-veto tuning the
ledger spent so much on): on ACES the loop backend is net-POSITIVE
(frontend-only 6.38 -> 6.21 soft / 5.94 hard veto), and on belgioioso only 1
revisit exists to close at all. The damage is in the SCAN-MATCHING FRONTEND,
which replaces already-good odometry deltas with slightly worse scan-matched
ones. Two independent logs show it, so it is a generalizable regime, not an ACES
quirk (and belgioioso's non-Manhattan geometry is precisely where the octave-ring
matcher was predicted to underperform).

**The pattern across all six logs: the frontend helps iff odometry drifts, and
hurts when odometry is already excellent.** This is the do-no-harm gap. A
frontend guard that detects "odometry is already self-consistent, don't override
it" (e.g. shrink the frontend correction toward the odometry guess when the
match is low-coherence / ill-conditioned by the indicators already computed for
loop closures) could recover ACES + belgioioso toward odometry parity without
touching Intel/fr079/fr101, where odometry drifts and the frontend is the hero
(Intel 24.2 -> 2.44). Open experiment; a load-bearing frontend change, so it
needs the full six-log validation before it can ship. [OUTCOME: this guard was
built and REFUTED — see "The frontend do-no-harm guard: a clean negative" and
"... is CLOSED: a triple negative" below. No per-frame or windowed signal
separates the regimes; the gap is irreducible at the frontend.] NOTE the
hard-veto column
still edges soft@0.55 on both odo-excellent logs (ACES 5.94 vs 6.21, fr101 1.80
vs 1.88) exactly as the worst-of-three coh_target sweep predicted — soft's win
is concentrated on Intel; it is a small, known cost elsewhere.

---

## The frontend do-no-harm guard: a clean negative (per-frame signals can't do it)

Follow-up to "the frontend do-no-harm gap": can a frontend guard shrink the
scan-match correction toward the odometry guess on low-confidence frames,
recovering the odo-excellent logs (ACES, belgioioso) toward odometry parity
WITHOUT regressing the odo-drifting logs (Intel, fr079, fr101)? Built
`ssp_frontguard.py` (FrontGuardSLAM subclasses BoundedSLAM, overrides ONLY the
`_frontend_accept` hook; `identity` reproduces shipped bit-for-bit — asserted).

**DIAGNOSTIC (the decisive evidence).** Per-accepted-frame distributions of the
coherence ratio r = c01/coh_ref and the correction magnitude |cand-guess|:

| log | r p10 | r med | r p90 | frac r<0.85 | \|cand-guess\| med |
|---|---|---|---|---|---|
| aces (odo-excellent) | 0.70 | 1.02 | 1.26 | 0.223 | 0.036 |
| belgioioso (odo-excellent) | 0.64 | 1.03 | 1.30 | 0.261 | 0.027 |
| intel (odo-drifts) | 0.72 | 1.01 | 1.26 | 0.209 | 0.017 |
| fr101 (odo-drifts) | 0.72 | 1.00 | 1.25 | 0.237 | 0.026 |

The coherence-ratio distribution is **essentially identical** across the two
regimes — the session-relative EMA self-normalizes every log onto the same
curve (median ~1.0, ~21-26% of frames below 0.85). Coherence carries NO
per-frame signal for "odometry is already good." Correction magnitude inverts
the naive expectation: ACES's corrections are LARGER (med 0.036) than Intel's
(0.017) even though ACES odometry is far better — so "large correction" doesn't
flag needless override either.

**SWEEP (confirms the prediction).** ATE rmse m; blend/bayes/cap rules over the
coherence ratio, vs the identity baseline (= shipped):

| rule | Intel | fr079 | ACES | fr101 | belgioioso |
|---|---|---|---|---|---|
| identity (ship) | 2.44 | 5.52 | 6.21 | 1.88 | 2.64 |
| blend 0.5-1.0 | 10.82 (+343%) | 10.84 | 2.86 | 1.43 | 0.61 |
| blend 0.7-1.0 | 10.89 (+347%) | 6.62 | 3.65 | 0.26 | 1.97 |
| blend 0.85-1.15 | 11.45 (+370%) | 11.68 | 2.80 | 1.43 | 0.98 |
| bayes k=4 | 12.60 (+416%) | 8.03 | 2.86 | 0.81 | 0.95 |
| bayes k=4 mag | 7.40 (+203%) | 11.15 | 1.20 | 0.41 | 4.53 |
| cap 0.5/0.15 | 4.97 (+104%) | 11.23 | 2.21 | 1.18 | — |

Every rule that materially helps ACES/belgioioso **catastrophically regresses
Intel and fr079** — the two highest-drift logs, whose per-frame corrections ARE
the drift fix. Intel's genuine corrections are indistinguishable by coherence
from ACES's harmful ones (identical distributions), so any damping catches both;
the excellent-log gain and the drift-log regression move together. **No do-no-
harm rule exists in the per-frame coherence/magnitude signal.** (Note fr101, an
odo-drift log, IMPROVES under damping — 1.88->0.26 — so the frontend is
genuinely over-trusted in general; only the very-high-drift logs need full
frontend authority, and they are exactly the ones every rule breaks.)

**Control that seals it (the regression is intrinsic to damping, not a gate
artifact).** A COHERENCE-BLIND constant damping (alpha=0.25, no gate) reproduces
the identical frontier: it pulls ACES to 3.11 but drives Intel to 14.09
(+478%) — worse than any gated rule. The gate buys only a marginal ACES/Intel
tradeoff, never separation. So "odometry is already accurate" is a property of
the odometry vs unavailable ground truth; it is NOT observable from ANY scan-
match confidence signal, because a large frontend correction is per-frame
indistinguishable between "odometry drifted, trust the match" (Intel) and
"odometry fine, match is map-noise" (ACES). A do-no-harm frontend guard would
require an INDEPENDENT odometry-quality estimate — which is exactly what the
windowed systematic-vs-random probe below attempts.

**The one untested escape hatch (data-motivated).** ACES's corrections are
larger per-frame yet its odometry is better — reconcilable only if ACES's
corrections are RANDOM (cancel, don't accumulate) while Intel's are SYSTEMATIC
(accumulate coherently to fix 24 m of drift). Per-frame magnitude/coherence is
blind to this; a WINDOWED "are my recent corrections systematic or random?"
detector (cumulative-sum / autocorrelation of the correction vector over a
sliding window) could separate the regimes where per-frame signals provably
cannot. That is the next experiment; until it is tested, the shipped conclusion
stands: the ACES/belgioioso frontend regression is NOT closable at the frontend
with a per-frame confidence signal.

---

## Ring-key shortlister: MIT corridor RETRIEVAL solved, but the wall moves to consensus

The Scan-Context ring-key finding ("Richer place descriptor vs the corridor
limit") proved the MIT revisit signal survives in the raw scan. This experiment
(`ssp_ringkey.py`, HybridSLAM subclass; drought candidates come from a per-
segment ring-key kNN instead of / alongside the coarse-band cell sweep, feeding
the UNCHANGED `_drought_verify` + PCM admission) tests whether wiring that
retrieval into the drought path breaks the MIT ATE. An independent read-only
audit first hardened it (deep-consensus escalation was disabled because
`noff`=pool-size never reached the shipped `deep_noff` threshold — fixed; plus
per-snap REF-revisit validation added; eviction leak pruned).

**Retrieval: solved.** hit-rate@40 over MIT TRUE revisits (range-identity REF):

| region | ring-key | coarse-band (shipped) |
|---|---|---|
| all true revisits (104) | **0.808** | 0.317 |
| mid corridor (52) | 0.827 | 0.173 |
| late/deep corridor (33) | 0.788 | 0.273 |

Ring-key recovers 84/104 true revisits where the coarse SSP band gets 33/104 —
including the deep corridor the coarse band originally scored ~0 on. The R3/R4
"place-recognition starvation at lam 5.3/12.8" is a coarse-summary artifact, not
an environment limit: the radial-occupancy signal is there and retrievable.

**ATE: unchanged-to-worse, because the bottleneck was never retrieval.**

| log | verified fires | snaps | ATE (m) |
|---|---|---|---|
| MIT baseline HY4 | 70 | 2 | 42.66 |
| MIT ring-key | 252 | **0** | 45.24 |
| Intel baseline HY4 | 3 | 0 | 2.540 |
| Intel ring-key | 37 | 0 | 2.540 (bit-identical) |

Ring-key raises verified drought fires 3.6x (70->252 on MIT, 3->37 on Intel) yet
admits ZERO consistent cliques on MIT (vs the baseline's 2) — so ATE nudges
WORSE (45.24 vs 42.66, within the MIT chaos band) and Intel is untouched. The
252 verified fires include many corridor TWINS that each pass fine verification
(they align geometrically) but point to DIFFERENT wrong places, so PCM's
pairwise-consistent consensus finds no clique — and, crucially, the extra
genuine revisits do not rescue it because they arrive diluted by even more
twin scatter. **Better retrieval made consensus HARDER, not easier.**

**Verdict — the corridor limit is now definitively a TWIN-DISAMBIGUATION /
CONSENSUS problem, not a retrieval or information problem.** This closes the
loop opened by the drought/R4/R5/cascade/belief triangulation and the
Scan-Context control: (1) the info is in the raw scan (Scan-Context), (2) it is
retrievable into the drought pool (ring-key, 0.808), (3) per-fire verification
passes (252 fires), but (4) CONSENSUS admission cannot isolate a genuine
mutually-consistent subset from the twin scatter. The next lever is not more
retrieval or a tighter per-pair gate — it is SEQUENCE consistency: a genuine
revisit yields a temporally-consistent RUN of ring-key matches along the
trajectory, while twins are isolated coincidences (SeqSLAM, Milford & Wyeth
2012). That is the queued next experiment. (Implementation note: the 60-min MIT
run is the per-drought coarse-band A/B measurement scaffolding
`_coarse_score_anchors` — a per-anchor Python loop — NOT the ring-key mechanism,
which is a vectorized 20-dim kNN; production ring-key drought is cheap.)

## The frontend do-no-harm gap is CLOSED: a triple negative

Extending "The frontend do-no-harm guard: a clean negative", the windowed
systematic-vs-random probe (`ssp_frontsys.py`) tested whether correction
ACCUMULATION over a trailing window separates the regimes where per-frame
signals could not. rho = ||sum delta_i|| / sum||delta_i|| over the last W
accepted frontend corrections (delta = cand - guess); rho->1 systematic
(drift-fixing), rho->0 random (map-noise).

- **Translation rho: decisive negative** — the regimes interleave BACKWARDS.
  fr101 (drift log, frontend essential 8.56->1.88) has the LOWEST xy rho of all
  five logs (0.098 at W=30), below both odo-excellent logs; belgioioso
  (excellent) sits above two of three drift logs. Even Intel's genuine
  drift-fix is a small mean buried under scan-match jitter that dominates rho
  everywhere. The two-log Intel-vs-ACES intuition did not survive five logs.
- **Heading rho: orders correctly at the median but fails the sweep.** Heading
  corrections DO accumulate on real heading drift (drift med 0.15-0.38 vs
  excellent 0.067-0.085), but the tails overlap massively, so no threshold is
  per-frame separable. The guard sweep confirms it: every heading-keyed config
  regresses the drift logs +62% to +135% while only partially helping ACES —
  no do-no-harm frontier (worst-case identical to the per-frame rules).

**So do-no-harm is a TRIPLE negative: per-frame coherence/magnitude, windowed
translation-systematicness, and windowed heading-systematicness all fail.** The
frontend's two entangled jobs — absorbing gross odometry drift (essential) and
needlessly perturbing already-good odometry (harmful) — are not separable from
ANY scan-match-observable signal, because "odometry is already accurate" is a
property of the odometry vs unavailable ground truth. A do-no-harm frontend
guard would require an INDEPENDENT odometry-quality estimate (e.g. wheel-slip /
IMU residual), which this sensor suite does not provide. Shipped behavior
stands; the ACES/belgioioso regression is a documented, understood, irreducible
property of scan-matching against good odometry.

---

## SeqSLAM sequence consistency: the corridor is SEQUENCE-AMBIGUOUS (thread closed)

The ring-key experiment relocated the MIT corridor limit to CONSENSUS: retrieval
is solved (0.808) but PCM admits 0 cliques because twins verify geometrically
and scatter. PCM checks GEOMETRIC pairwise consistency; it does NOT check
TEMPORAL consistency. SeqSLAM (Milford & Wyeth 2012) is the SotA answer: a
genuine revisit should yield a temporally-consistent RUN of matches along the
trajectory (query k~old j, k+1~j+1, ...), a velocity-swept diagonal; a twin is
an isolated coincidence. `ssp_seqslam.py` stores a per-keyframe ring-key,
scores the velocity-swept diagonal (both traversal directions) at each drought
candidate, and admits either seq-only (single confirmed fire) or seq+PCM.

An independent read-only audit FIRST caught a decisive lifecycle bug (the
current keyframe's descriptor was stored AFTER `_try_drought` ran, so the seq
drought early-returned on every attempt — the first run was a degenerate
356/0/0/0 that would have been misread as "sequence cleanly avoids twins"). It
also flagged that seq-only's single-fire snap removes PCM's two-consistent
requirement and rests entirely on the fine verifier + a hand-set seq gate. Both
concerns proved central. After the fix (query window ends at the newest STORED
keyframe; verified d_hyp>0):

| config | ATE (m) | try/hyp/ver/snap | snaps |
|---|---|---|---|
| baseline HY4 (shipped drought) | 42.66 | 331/324/70/2 | 2 GENUINE |
| HY4 drought-OFF | 45.24 | 0/0/0/0 | 0 (drought worth +2.6 m) |
| SeqSLAM seq-only | **56.23** | 311/204/9/2 | **2 TWIN (false)** |
| SeqSLAM seq+PCM | 45.24 | 356/242/13/0 | 0 |

**Sequence does NOT separate genuine revisits from twins.** REF-aware diagonal
separation over 34 true-revisit attempts that had a genuine candidate:

- SEQUENCE picks genuine over the best twin **17/34 = 0.500 (chance)**;
  single-frame 10/34 = 0.294. The diagonal adds essentially nothing.
- Gate sensitivity: at EVERY threshold twins pass at >= the genuine rate
  (thresh 0.06: genuine 34/44, twin 36/44; thresh 0.10: genuine 38/44, twin
  43/44). No threshold separates them — a passing result would be pure
  threshold-fitting.
- seq-only therefore snapped 2 TWINS (ATE 56.23, max error 200 m — a
  catastrophic wrong building-scale snap, exactly the audit's predicted failure
  of single-fire admission). seq+PCM's clique requirement safely blocks the
  twins (0 snaps) but sequence gives PCM nothing new to consense on -> 45.24,
  identical to drought-off.
- WINDOW-LENGTH ALIASING SWEEP (does more temporal context help?): fraction of
  9 REF true-revisit droughts where the genuine diagonal beats the best twin,
  vs window travel: L=8 (4.8 m) 0.222 | L=15 (9 m) 0.222 | L=25 (15 m) 0.222 |
  L=40 (24 m) 0.111. At EVERY window up to 24 m of travel it is at-or-below
  chance (0.5) — twins routinely score BETTER. Longer sequences do NOT
  disambiguate: the corridor is geometrically identical over the entire
  practical sequence range, so the descriptor SEQUENCE aliases as badly as the
  single frame. This is the deepest confirmation that the limit is the
  sensor-environment pair, not the window.
- INTEL CONTROL (is the METHOD broken, or is MIT genuinely ambiguous?): on
  Intel's DISTINCTIVE geometry sequence DOES separate — genuine-over-twin 4/6 =
  0.667 (vs MIT 0.500), and seq-only snapped 1 GENUINE revisit (0 twins). So
  SeqSLAM works where geometry permits; MIT's failure is corridor
  self-similarity, not a method defect. BUT seq-only's aggressive single-fire
  admission REGRESSES even Intel (2.540 -> 3.984): Intel needs no drought
  relocalization (baseline snaps 0), and forcing an unnecessary — if genuine —
  building-scale snap + chain pre-alignment perturbs the already-converged
  graph. seq+PCM blocks the lone fire and holds 2.540. NET: seq-only is too
  aggressive EVERYWHERE (twins on MIT, needless snaps on Intel); seq+PCM is
  safe everywhere but adds nothing. The shipped conservative drought is right.

**Verdict — the MIT corridor limit is CLOSED, triangulated across all three
axes of place recognition:**
1. RETRIEVAL (Scan-Context / ring-key): the revisit signal is in the raw scan
   and recoverable (recall 0.317 -> 0.808). NOT the bottleneck.
2. GEOMETRIC CONSENSUS (PCM): twins verify geometrically and scatter; no
   pairwise-consistent clique isolates the genuine set. Unfixable by a tighter
   per-pair gate.
3. TEMPORAL SEQUENCE (SeqSLAM): corridor twins produce diagonals AS CONSISTENT
   AS genuine revisits — the geometry is identical over the whole sequence
   window, so the temporal signal is 50/50 (chance).

The MIT infinite corridor is fundamentally ambiguous from 2D-lidar appearance at
EVERY level available to a bounded appearance-only SLAM — per-frame, per-place-
descriptor, per-geometric-consensus, and per-temporal-sequence. Correct
relocalization there requires an INDEPENDENT ABSOLUTE CUE (global positioning,
surveyed landmarks, active perception, or a wider-aperture/3D sensor) that lidar
appearance alone cannot supply. The shipped conservative drought (2 genuine
snaps, 42.66 m) is at the achievable frontier; more aggressive relocalization
(seq-only) snaps twins and is strictly worse. This closes the corridor thread
opened by the R3/R4/R5/cascade/belief investigations: it was never a tuning or
representation deficiency — it is an information limit of the sensor-environment
pair, now proven from three independent directions.

---

## VSA map composition: superposition merges two independently-built maps (with scope caveats)

The paper's thesis is "algebraically transformable maps": translation = phase
multiply, rotation = index permutation, and — the untested part — BUNDLING =
vector addition, so two aligned maps of the same area should COMPOSE by adding
their world-placed segment vectors, with no re-encoding. `ssp_multisession.py`
(BoundedSLAM subclass/import, edits nothing shipped) tests this by splitting
Intel into two independently-built sessions A=kf[0:3723], B=kf[2482:6205] (each
from its own re-origined odometry) and merging them. A read-only audit checked
it adversarially; the VERIFIED (audit-caveated) result:

**What is solid (algebra + composition):**
- The merge is a genuine per-cell A+B vector ADDITION (audit confirmed no
  merged==single-session artifact), and the shift/permute/add algebra is
  self-verified BIT-EXACT (selftest: standalone place() == world_vec_seg;
  folding a rigid T into an anchor pose == transforming the world-placed
  bundle). Composition = re-place each B anchor at se2_mul(T, pose) and add —
  nothing re-encoded from scans.
- Overlapping 2 m cells BUNDLE rather than duplicate: merged union of 93 cells
  vs 147 duplicate-sum = a 37 % memory saving (O(area of the UNION), not sum of
  sessions). Arithmetic verified. (These are idealized 1-vector/cell figures,
  not the full BoundedSLAM footprint with CELL_CAP and segder — but the overlap
  saving is model-independent.)
- Adding session A does NOT destroy session B's signal: the bundled-cell cosine
  to each constituent is 0.940 median (p10 0.591 — the tail where one session's
  norm dominates, so the metric does surface washout, it is not hidden). No
  destructive interference at this load.
- Alignment sensitivity: perturbing B's placement before the merge is harmless
  to delta ~= 0.25 m (the fine wavelength lam=0.25) and degrades clearly by
  0.5-1.0 m. So the superposition tolerates SUB-WAVELENGTH misalignment.

**What is NOT shown (the audit's load-bearing caveat):** the localization
comparison (A-alone 46 cm / B-alone 31 / merged 33 cm) is a gt-SEEDED LOCAL
REFINEMENT of B's OWN build scans, not from-scratch relocalization — ground
truth sets both the 0.36 m starting guess and the 8 m local-map window, and the
query scans are the same scans B's segments were folded from (self-
localization). So the honest content is "adding A does not destroy B's self-
signal, at a small ~9 % RMSE crosstalk cost" — NOT "the fused map relocalizes
independently." Under a realistic odometry-only prior the absolute errors would
be larger and the 0.25 m alignment bar likely tighter. Treat the cm figures as
leak-assisted lower bounds; trust the qualitative "superposition composes maps
without destroying information," not the specific numbers.

**The alignment problem is real and NOT VSA-specific.** In the sessions' own
independently-drifted frames, 245 place-correspondences show a single best rigid
T_AB leaves a ~1.22 m median non-rigid residual (up to ~10 m; per-correspondence
transform spread 3.2 m) — far beyond the 0.25 m superposition bar, which is
exactly why naive rigid relocalization gave ~8 m errors. Two independently-
drifted maps have their own internal warp; a single rigid transform cannot align
them. This is a GENERAL multi-session SLAM problem (the standard fix is cross-
session loop constraints + a joint pose-graph relax), NOT a property of the SSP
representation — after alignment, the VSA superposition applies unchanged.

**Verdict:** VSA superposition genuinely composes two independently-built maps
by shift/permute/add — bit-exact algebra, O(area-of-union) memory, and adding a
map does not destroy the other's signal (small crosstalk cost) — demonstrating
the representation's unique algebraic-composition property (grids and point
clouds cannot be superposed). The demonstration is a gt-seeded local-refinement /
self-localization test, so it shows COMPOSITION-WITHOUT-CORRUPTION, not
from-scratch multi-session relocalization; and aligning two drifted maps to
sub-wavelength (the precondition) is a general multi-session problem this does
not solve. A clean positive capability with an honestly bounded scope.

---

## Cross-session alignment: pose-graph merge works GIVEN correspondences; establishing them is the (recurring) relocalization limit

Follow-up to "VSA map composition" (which proved superposition merges two
ALIGNED maps): can two independently-drifted sessions be aligned end-to-end via
cross-session loop closures + a joint pose-graph relax, then superposed?
`ssp_multisession_align.py` attempts it on the two Intel sessions. A read-only
audit found the initial "win" is a FALSE WIN, and the corrected result is the
honest — and more interesting — finding.

**The false win (what NOT to claim).** The reported "213 correct / 1 false
cross-session closures, overlap 31->3 cm, B improves 4.37->3.49 m" is real
arithmetic but rests on an A-PRIORI DATA-ASSOCIATION ORACLE, not genuine
relocalization. A=kf[0:3723] and B=kf[2482:6205] are split from ONE Intel log,
so keyframes [2482,3723) are the SAME physical scans in both sessions. The
coarse-alignment bootstrap uses that KNOWN per-keyframe index correspondence
(kf i in A == kf i in B) to seed the cross-session matcher AT the answer — for
an overlap anchor the seed IS A's own estimate of the same scan (cm away), so
acceptance is a foregone conclusion. Of 214 admitted edges, ~211 are the same
scan tied to itself across the artificial split; only 3 are genuine cross-pass.
The "213/1 correct" validation is trivially true for same-index ties ("a scan is
near itself"). The 31->3 cm residual is measured on the shared SAME-LAP
co-traversal (self-consistency of the tied nodes, near-tautological) — NOT the
~1.2 m non-rigid warp the experiment set out to fix, which lives in the
cross-pass tail. No GT leaks into the estimate; the oracle is the session
index-correspondence (worse, because it is disguised as "gt-free").

**Strip the oracle -> the genuine result (2/274).** On the truly independent
cross-pass revisits (B on a later lap over A's earlier-mapped area, no shared
index), metric relocalization recovers 2 correct closures out of ~274
opportunities; the tail overlap residual barely moves (565 -> 532 cm); no
alignment. Root cause (audit-confirmed, not FP and not the pose graph): SEED
QUALITY. A metric cross-session seed for a B tail keyframe comes from B's own
estimate, and B's 4.37 m self-drift puts the seed ~7 m off — only ~3 % land in
any matcher basin. This is EXACTLY the project's documented relocalization-prior
limit (the drought/R3/R4 story: the matched band needs a prior within its basin;
metric priors drift out of it).

**What IS validly demonstrated (the VSA-native parts, both sound):**
1. GIVEN correct cross-session correspondences, the joint pose-graph relax over
   both sessions' anchors + cross-session edges (shipped `_gn` TRF + IRLS + LOO,
   gauge-fixed at A) genuinely merges the maps: on the co-observed stretch it
   pulls the drifted session toward the tighter one (B 4.37 -> 3.49, A preserved
   3.30 -> 3.41) and drives the tied-region residual to sub-wavelength. Pose-
   graph merging is sound.
2. The superposition step (composition) then applies unchanged.

**Conclusion — multi-session VSA mapping decomposes into three parts:**
(a) map COMPOSITION (superposition) — WORKS (bit-exact, prior section);
(b) pose-graph MERGE given correspondences — WORKS (this section);
(c) establishing cross-session CORRESPONDENCES metrically — BLOCKED at the
    relocalization-prior/seed-drift limit (2/274). Parts (a) and (b) are the
VSA-native / backend contributions and are demonstrated; part (c) is the general
place-recognition problem the project has already characterized, and its
identified fix is the SAME appearance-based retrieval that solved MIT drought
retrieval — the ring-key (recall@40 0.81 on Intel), which never needs a metric
seed. So the multi-session thread REDISCOVERS the project's central recurring
limit: metric relocalization priors drift out of basin; appearance retrieval is
the missing precondition. End-to-end multi-session is therefore gated on wiring
ring-key place recognition into cross-session detection — a concrete next step,
not a new limit. The naive metric-matcher demonstration does NOT establish
cross-session correspondences and must not be reported as if it does.

---

## Ring-key cross-session correspondence: retrieval unblocks the seed, but the wall moves to cross-session VERIFICATION

The oracle-free completion of the multi-session thread: use ring-key APPEARANCE
retrieval (not a metric seed, not the shared index) to establish cross-session
correspondences, then joint-relax + superpose. `ssp_multisession_ringkey.py`
(imports the session/merge/ring-key helpers; edits nothing). Anti-oracle
guarantee VERIFIED by direct read: candidates are ALL A anchors ranked by
ring-key L2 (20-float body-frame appearance) — the keyframe index never selects,
orders, or seeds; the metric seed is the RETRIEVED anchor's pose + SC column-
shift yaw (drift-independent, not B's estimate); REF is used only to score
retrieval hits and validate ties, never to select/seed/verify.

**The result decomposes cleanly (a precisely-located LIMIT, not a win):**
1. APPEARANCE RETRIEVAL WORKS. Over 314 genuine cross-pass opportunities,
   ring-key recall@40 = 0.78 (any correct) / 0.64 (correct AND cross-pass),
   reproducing Intel's ~0.81. Its seed is drift-INDEPENDENT, so it supplies
   exactly the candidate the metric-seed baseline (2/274) missed on B's 4.37 m
   self-drift. The seed-drift limit is removed.
2. CROSS-SESSION METRIC VERIFICATION FAILS (the new break). Of 435 B anchors
   only 63 verify, and 51/63 are false = 81 % FP; cross-pass, 50 admitted / 5
   correct. So ring-key+verify recovers 5/314 genuine cross-pass ties — more than
   the metric-seed 2/274, but swamped by false positives. The 81 % FP poisons the
   joint relax: B ATE 4.37 -> 6.32 m (WORSE), tail residual 381 -> 469 cm.
3. MECHANISM (probed, not a bug): ring-key ranks the correct A anchor #0 and it
   scores coherence ~0.88 when reached — but the shipped acceptance gate is a
   SESSION-RELATIVE ratio (0.55 x coh_ref ~= 0.223 absolute), tuned for the
   same-session frontend where the odometry seed is ALREADY near-correct and the
   gate never had to DISCRIMINATE places. Under appearance seeding, wrong-but-
   nearby places (5-15 m off) routinely clear 0.223 (FPs) while viewpoint-changed
   genuine cross-pass ties fall just below (misses). The gate was doing
   confirmation, not discrimination — the (now-removed) oracle seed did the
   discrimination.
4. THE PIECES ARE SOUND — only verification is broken. Diagnostic (REF-filtered
   upper bound, explicitly NOT the oracle-free method): feeding the joint relax
   only the 12 REF-correct cross edges gives tail residual 172 cm and B 4.37 ->
   **2.49 m (beats the 2.44 single-run ceiling)**. So retrieval + pose-graph
   merge + superposition all work; the single missing link is a cross-session
   VERIFIER that can discriminate.

**Verdict — the multi-session thread is complete, and its bottleneck is the
project's recurring one (verification, not retrieval):**
- COMPOSITION (superposition) — works (bit-exact).
- MERGE given correspondences — works (joint relax; clean ties -> B 2.49 m).
- CORRESPONDENCE retrieval — works (ring-key appearance, drift-independent).
- CORRESPONDENCE verification — the WALL: the same-session-tuned coherence gate
  cannot reject cross-session twins (81 % FP). This mirrors the MIT corridor
  close (retrieval solved, verification is the wall) but on distinctive Intel and
  for a different reason: the gate does confirmation, not discrimination.
The capstone is one cross-session VERIFIER away (an ABSOLUTE coherence margin +
PCM geometric-consensus over the appearance shortlist, replacing the session-
relative confirmation gate) — a verification fix, not a new retrieval method, and
the diagnostic proves it would land (clean ties -> 2.49 m). This is the concrete,
scoped next step; the naive session-relative gate must not be used for
appearance-seeded cross-session admission.

---

## Cross-session verifier: the wall is irreducible (multi-session thread CLOSED)

The final multi-session experiment (`ssp_multisession_verify.py`, imports the
ring-key detector + joint-relax/superpose + `ssp_hier` PCM + `_gn`; anti-oracle
inherited — the detector reproduces `detect_cross_ringkey` bit-exact 63/12/5, and
the verifiers use Otsu-on-coherence-VALUES + REF-free SE(2) PCM, REF only labels
and scores). Tests whether an ORACLE-FREE verifier can discriminate cross-session
twins and land the 2.49 m diagnostic. Over 314 genuine cross-pass opportunities:

| verifier | admitted | FP | genuine cross-pass | merged ATE | B ATE |
|---|---|---|---|---|---|
| session-relative (reproduction) | 63 | 81 % | 5 | 5.54 | 4.37 -> 6.32 (worse) |
| absolute margin (Otsu tau=0.223) | 63 | 81 % | 5 | = (a) | = (a) |
| PCM clique >= 2 | 22 | 50 % | 6 | **3.35** | 4.37 -> 4.65 |
| PCM clique >= 3 | 5 | 100 % | 0 | 3.29 | 4.37 -> 4.36 |
| absolute AND PCM | 4 | 0 % | 1 | 5.52 | 4.37 -> 4.36 |

1. ABSOLUTE COHERENCE MARGIN — REFUTED oracle-free. Correct- and false-candidate
   fine-ring coherence distributions are INDISTINGUISHABLE (medians 0.171 vs
   0.172, a single 0.04-0.64 blob); Otsu lands exactly on the shipped 0.223 gate,
   so (b) === (a). The earlier "correct tie ~0.88" was a SAME-REGION (near-
   identical viewpoint) probe — viewpoint-CHANGED genuine cross-pass ties score
   no higher than wrong-nearby places. Coherence cannot discriminate cross-session
   twins on Intel.
2. PCM GEOMETRIC CONSENSUS is the ONLY mechanism that helps: clique>=2 halves the
   FP (81 % -> 50 %), RAISES genuine recovery 5 -> 6, and improves the merged
   two-session ATE 5.54 -> 3.35 m ORACLE-FREE (A preserved 3.30 -> 3.28). A real,
   honest gain over B-alone 4.37 m.
3. BUT THE WALL DOES NOT CLOSE. clique>=3 admits only SYSTEMATIC TWINS (5 edges,
   100 % FP — aliased wrong places form consistent triples while genuine ties are
   too sparse to, the exact PCM failure the MIT corridor also showed); abs+PCM
   gets 0 % FP but 4 clustered edges under-determine the rotation. And even a
   PERFECT cross-pass verifier is bounded: 22 REF-clean cross-pass-only edges
   reach only B 4.35 m — the 2.49 m diagnostic REQUIRED the dense same-lap
   OVERLAP-region ties (near-identical viewpoint, easy), not genuine different-lap
   revisits. So "one verifier from 2.49" was optimistic; the genuine cross-pass
   ceiling on Intel is ~3.35 m (PCM clique>=2), not 2.49.

**VERDICT — the multi-session thread is CLOSED, and its final decomposition is:**
- COMPOSITION (VSA superposition) — WORKS (bit-exact, the positive capstone).
- MERGE given correspondences (joint relax) — WORKS.
- RETRIEVAL (ring-key appearance) — WORKS (drift-independent).
- VERIFICATION (discriminating genuine cross-session ties from twins) — the
  IRREDUCIBLE WALL: absolute coherence can't separate viewpoint-changed ties from
  wrong-nearby places, and PCM consensus helps (3.35 m oracle-free) but cannot
  close it because the aliased twins are themselves geometrically consistent.
This MIRRORS THE MIT CORRIDOR CLOSE from the other side: on MIT the twins are
self-similar corridors defeating retrieval+consensus; on Intel the twins are
cross-session viewpoint changes defeating coherence+consensus. Both land on the
same structural limit — appearance-only place VERIFICATION cannot separate a
genuine revisit from a consistent alias without an independent cue. The VSA-native
contributions (composition, algebraic merge) are demonstrated and sound; the
recurring bottleneck across every thread this campaign is the SAME one:
verification/discrimination, not retrieval. Best honest oracle-free multi-session
result: merged two-session 3.35 m via PCM clique>=2 (better than the drifted
single session, short of the same-lap ceiling).

---

## Small-correlated-alias robustness floor (the last open question) — CLOSED

Outlier robustness was proven against i.i.d. and LARGE (0.6-1.2 m) correlated
aliases; the unprobed hard case was SMALL repetitive-bay-scale correlated aliases
(0.2-0.35 m), the magnitude regime the corridor limit lives in. `scratch_small
alias.py` sweeps a CONTROLLED correlated alias (one fixed wrong transform of
magnitude m + 9 deg, injected on 10 % of genuine candidates via a FixedAliasSLAM
subclass that pre-sets `_alias`) over m in [0.15, 1.0] m, paired seeds, on three
worlds. ATE lift over the clean baseline (cm):

| world | clean | lift @ all m (0.15-1.0) | surv @0.15 -> @1.0 |
|---|---|---|---|
| room     | 6.42  | -0.20 (flat)  | 13.5 -> 0.75 |
| corridor | 379.1 | -0.05 (flat)  | 5.2 -> 5.0 |
| sparse   | 12.08 | +0.26 (flat)  | 27.2 -> 0.0 |

**The backend is robust to small correlated aliases at EVERY magnitude** — ATE
lift is negligible (-0.20 to +0.26 cm) across the whole 0.15-1.0 m range,
including the 0.2-0.35 m regime. Mechanism = a TWO-LAYER defense, cleanly split
by magnitude:
- The INNOVATION GATE rejects only LARGE aliases (> ~0.9 m ~= 3x the 0.30 m drift
  allowance cap). `innov-rej` is CONSTANT across m (25 room / 25 sparse / 0.2
  corridor) — those are genuine-drift rejections, not the alias; the gate never
  sees the small aliases as anomalous because they sit within the drift ball.
- IRLS CAUCHY DOWNWEIGHTING neutralizes the small aliases that pass the gate.
  `surv` (edges corrupted past the genuine scale) RISES as m falls (sparse
  27->0, room 13->0.75) — the small aliases ARE admitted as edges — but `pruned`
  ~= 0.2 (LOO almost never fires) and ATE stays flat, so they are not removed,
  they are LOW-WEIGHTED: a fixed alias disagrees with the genuine closures at the
  same location, so the robust loss caps its pull. The gate handles the large
  outliers; IRLS handles the small ones the gate is blind to.

**Crucial distinction (why this does NOT contradict the corridor limit).** This
injection model adds the alias to GENUINE candidates, so a genuine closure is
always present for IRLS to disagree with — that is exactly why the small alias is
downweighted. The corridor-twin failure (RESULTS §corridor) is a DIFFERENT mode:
a drought relocalization to a self-similar twin where there is NO genuine
competitor at all, so IRLS has nothing to weigh it against and consensus/sequence
cannot discriminate. Outlier-injection robustness (this section) and the corridor
verification limit are distinct failure modes: the backend rejects/downweights
correlated outliers that COMPETE with genuine data, but cannot verify a place
that has no genuine competitor. This closes the small-correlated-alias open
question (robust, via IRLS not the gate) while sharpening — not softening — the
corridor verification finding: the two are orthogonal.

---

## Value of a perfect verification cue on the MIT corridor (GT-oracle diagnostic) — the wall is TWO-FOLD

The symmetric completion of the synthesis: the multi-session diagnostic showed
that GIVEN correct correspondences the VSA pieces close the loop (B 2.49 m). Does
the same hold on the MIT corridor? `ssp_mit_gtverify.py` (subclasses the ring-key
SLAM; imports only) replaces ONLY the drought VERIFICATION step with a GT oracle
(admit a candidate iff its query & candidate REF positions are within TOL_REV=5 m
— genuine revisit, reject twins), keeping ring-key retrieval, edge geometry
(cmatcher), and the backend as the real shipped machinery. GT-leak boundary
verified: the REF touches ONLY the admit/reject decision and the final ATE, never
retrieval, seeding, edge geometry, or the relax.

| run | MIT ATE | genuine snaps |
|---|---|---|
| raw odometry | 187.83 | — |
| shipped drought (coherence + PCM) | 42.66 | 2 |
| ORACLE B: GT place-verify + real geometry gate | **41.04** | 3 |
| ORACLE A: GT place-verify ONLY (coherence off) | 73.47 (WORSE) | 4 |

**Value of a perfect PLACE oracle: 42.66 -> 41.04 m (~+1.6 m, ~4 %) — it barely
moves MIT.** And this is the result: the corridor and multi-session are NOT
symmetric, and the asymmetry sharpens the whole synthesis.
- MULTI-SESSION: place verification was the ONLY missing piece — genuine ties had
  GOOD relative-pose geometry, so the oracle closed the loop (2.49 m).
- MIT CORRIDOR: perfect place recognition is NECESSARY BUT NOT SUFFICIENT.
  Corridor self-similarity ALSO corrupts the RELATIVE-POSE GEOMETRY — at a genuine
  revisit the matcher SLIDES along the corridor and returns a mis-aligned edge
  (coherence ~0.40). The wall is TWO-FOLD: place discrimination AND geometry.

The two oracle variants isolate this because the shipped coherence gate does
double duty (place discriminator + geometry-quality filter):
- B keeps coherence purely as a GEOMETRY filter (GT owns place): it admits only
  the genuine revisits the matcher can ALSO align well — just 3 over the whole
  1.9 km log -> 41.04 m ~= shipped. The shipped conservative drought (2 snaps,
  42.66) was already near the achievable frontier.
- A drops the geometry gate (GT owns all verification): it admits 4 genuine
  PLACES, but two are geometrically-bad along-corridor slides (coherence
  0.40/0.44) that, inserted through _distribute_correction + relax, drive ATE
  WORSE to 73.47 m. Place verification ALONE backfires; the geometry gate is what
  prevents it.

**Verdict (sharpens, does NOT reopen, the corridor limit).** This is a GT-oracle
UPPER BOUND, not a system — no real appearance-only verifier reaches even B
(SeqSLAM genuine-vs-twin 0.500 = chance; PCM twins; coherence indistinguishable).
What it reveals: the missing EXTERNAL cue on the corridor would have to fix BOTH
place discrimination AND relative-pose geometry, not place alone — because
corridor self-similarity corrupts the along-corridor translation of even a
correctly-recognized revisit. This is strictly DEEPER than the multi-session wall
(place only), and it explains why the shipped conservative drought is near the
frontier: on the MIT infinite corridor there are only ~3 genuine revisits whose
geometry a lidar matcher can pin down at all, so even oracular place recognition
buys ~4 %. The synthesis stands and is sharpened: retrieval and the VSA algebra
are sound; the residual limit is verification — and on aperture-degenerate
geometry, verification means place AND pose, both of which need an external cue.

---

## Observability-weighted anisotropic loop constraints: a regime-specific backend win

The concrete lever the two-fold corridor finding exposed: genuine corridor
revisits fail on GEOMETRY (the matcher slides along the unobservable corridor
axis), yet the code ALREADY computes each candidate's observability structure —
the translation Hessian K = sum_k Re(c_k) w_k w_k^T (MAIN band) with eigenvalues
l_weak <= l_strong — and uses it only to VETO. `ssp_aniso.py` (subclass; imports
only) instead inserts the closure as a RANK-DEFICIENT constraint: a per-edge 2x2
information matrix Lambda = wt^2 K / l_strong (condition number l_weak/l_strong
clamped to a floor, rotated into the anchor's body frame), whitening the
translation residual anisotropically in a reimplemented analytic-Jacobian _gn.
Correctness gates ALL pass: use_aniso=False reproduces shipped BoundedSLAM
BIT-EXACT (max|dpose|=0), the analytic Jacobian matches finite-difference to
2e-8, the body-frame weak eigenvector is exact, and Intel-isotropic = 2.4395 ==
shipped 2.440 — so the ATE numbers are the anisotropy, not a solver artifact.

**REAL LOGS (ATE m; floor 0.1-0.2 usable band):**

| log | shipped | anisotropic | verdict |
|---|---|---|---|
| fr101 (dense-revisit) | 1.881 | **1.47** | WIN ~22% (stable across all 4 floors) |
| fr079 (floppy) | 5.523 | 5.09 | WIN ~8% |
| ACES (sparse closures) | 6.212 | 6.16 | wash |
| Intel (well-conditioned) | 2.440 | 3.69 | REGRESS |

**It is REGIME-SPECIFIC: it helps ill-conditioned / dense-revisit graphs (fr101
+22 %, the best-transfer log; floppy fr079 +8 %), washes the sparse-closure log
(ACES), and regresses the well-conditioned one (Intel).** fr101's edge set is
stable (53->56) — a clean observable-direction gain; Intel's blows up (80->158)
and goes chaotic.

**The Intel regression is NOT a pruning artifact (net-positive hypothesis
REFUTED).** Hypothesis: down-weighting the weak direction shrinks the whitened
residual that LOO/chi2 pruning ranks on, so weak-geometry edges survive. Fix
tested: rank pruning on the ISOTROPIC (geometric) residual while keeping the
anisotropic SOLVE. It did NOT recover Intel — it made it WORSE (aniso 3.69 ->
aniso+fix 5.58 at floor 0.1). So the regression is deeper: the anisotropic
weighting itself discards weak-direction information that a WELL-CONDITIONED graph
genuinely uses (its "weak" direction still carries real signal isotropic
captured). Anisotropy is therefore not a safe global default; it would need to be
GATED (enabled only on detected ill-conditioning) or offered opt-in.

**CORRIDOR GT-ORACLE (the direct test of the geometry half of the two-fold
wall):** anisotropic insertion of the geometrically-partial genuine revisits
RESCUES the isotropic catastrophe (ISO oracle-A 73.47 -> ANISO 45.6-48.3 across
floors, admitting 5-6 genuine closures whose along-corridor slide is now
neutralized) but does NOT beat conservative rejection (ISO oracle-B 41.04). So
the observability-weighting mechanically does what it should — it turns a
catastrophic partial-DOF closure into a harmless one — but the salvaged
cross-corridor component is not enough to beat simply dropping those closures:
the along-corridor DOF was GENUINELY unobservable, exactly the two-fold finding.

**Verdict.** A genuine SotA backend technique (rank-deficient constraints from the
SSP correlation-surface observability structure, verified correct) with a real,
characterized regime-dependence: net-positive on ill-conditioned / dense-revisit
graphs (a usable opt-in / conditioning-gated improvement, notably +22 % on the
best-transfer fr101), neutral-to-negative on well-conditioned ones, and on the
corridor it neutralizes catastrophic slides but cannot beat conservative
rejection. It partially addresses the geometry half of the two-fold corridor wall
(rescue, not cure), consistent with that limit being genuine.

### Anisotropic constraints — FINAL VERDICT (corrected after the full follow-up sweep)

Two claims in the section above were too optimistic and are corrected here by the
complete pruning-fix + conditioning-gate follow-ups:

- **The real-log wins are FRAGILE, not a conditioning-gateable opt-in.** The
  per-closure conditioning gate (leave well-conditioned closures isotropic,
  anisotropize only degenerate ones) FAILED on a false premise: Intel's closures
  are mostly ILL-conditioned (48-64 % trip the gate), so gating cannot spare them
  — and selective anisotropy destabilizes MORE than uniform (fr079 -> 11.4,
  chaotic Intel). The iso-pruning fix also failed (doesn't recover Intel, removes
  fr079's win). Root cause: the Intel edge-flood is a TRAJECTORY-FEEDBACK loop in
  edge ADMISSION (anisotropic down-weighting -> more marginal closures admitted ->
  jitter -> more closures), not a fixable ranking/weighting knob. So fr101 +22 % /
  fr079 +8 % are real observations for ONE uniform weighting but are fragile
  artifacts of that feedback, NOT a robust, shippable, or gateable improvement.
- **The MIT oracle-B DID edge below conservative rejection** (41.04 -> 39.12,
  admitting a 4th near-pure-slide genuine revisit that contributes its observable
  cross-corridor component) — but ~2 m on the 38-58 m MIT chaos band is
  mechanistically real yet only marginally significant, not a robust gain.

**Net:** the technique (rank-deficient constraints from the SSP correlation-surface
observability structure) is correctly implemented (bit-exact isotropic
reproduction, FD-checked Jacobian) and mechanistically does what it should
(neutralizes catastrophic along-corridor slides; the geometry half of the
two-fold wall is PARTIALLY addressable in the oracle setting). But it is NOT a
shippable net-positive: on the real-log suite it is a fragile, feedback-entangled,
regime-specific effect with no robust operating point, and the corridor gain is
chaos-band-marginal. Honest outcome: a genuine but non-shippable backend
experiment — a clean mechanism demonstration, not an accuracy win. The shipped
isotropic backend stands.

---

## Iterative large-to-small scale map correlation for cross-session alignment — NEGATIVE

Tests whether whole-map multi-scale correlation (coarse ring first for a broad
basin, then refine down the wavelengths) solves cross-session alignment where the
metric seed failed (`ssp_iteralign.py`, oracle-free, REF scores only). Positive
control PASSES (recovers a known rigid move of A to 0.18 m via the shift-theorem
correlator <A, shift(d)·rot(m)·B>), so the negative is genuine.

- **Coarse ring is translation-AMBIGUOUS, not a broad basin** — the core claim
  inverts: ring 12.8 m gives Terr 10.8 m (WORST band; 12.8 m period over a ~30 m
  map aliases), 5.3+12.8 -> 19.2 m, MAIN band 5.2 m (least-bad, still out of
  basin). No basin at the true T_AB for the cascade to start from.
- **Cascade DIVERGES even seeded at REF** (4.0 -> 8.7 m): whole-map fine-band
  correlation's global max is not at truth (O(N^2) cross-term clutter + the
  ~1.2 m non-rigid warp bury the true peak). It loses the basin between scales.
- **Piecewise local refinement** is boundary-pinned noise (177-202 cm, 48-81 %
  at the search boundary) — LARGER than the 1.22 m rigid residual it should fix.

Root cause: <A, shift·rot·B> is a feature-weighted dense cross-correlation of the
two decoded density fields; it peaks at truth only if the map is distinctive at
that scale AND the two views share content. Coarse rings fail distinctiveness
(aliasing); all scales fail the cross-session viewpoint change + non-rigid warp.
This rediscovers the recurring result from a new angle — multi-scale map
correlation INHERITS the coarse band's place-recognition limit (cf. Scan-Context
vs coarse SSP relo) rather than escaping it. It is not a substitute for appearance
retrieval + a discriminating verifier. IMPLICATION for map handling: BUNDLED
whole-map comparison is fundamentally aliasing/clutter-limited, which motivates
PER-CLUSTER comparison (compare candidate patches individually, let consensus
select) over a single bundled inner product.

---

## Per-cluster vs bundled map comparison — SPLIT: bundling is a real signal cost, but NOT the verification wall

Direct test of the hypothesis the section above raises: does bundling M candidate
patches into ONE inner product throw away the genuine-vs-twin discrimination that
comparing patches individually preserves? Representation-level version of the
recurring verification wall (`ssp_percluster.py`, subclasses BoundedSLAM; oracle-
free — GT ever only LABELS genuine-vs-twin, never selects). Three measurements at
both query paths. DISTINCTION from SeqSLAM (temporal per-frame sequence, 0.500 on
MIT): this is SPATIAL per-patch comparison (query vs each map segment separately).

- **CAPACITY / crosstalk (frontend local-map model, 120 sites).** Bundle M real
  recent segments, match the scan against the sum vs each segment separately. The
  genuine match's peak cosine decays with bundle size — Intel M=1..64:
  0.886 -> 0.694 -> 0.518 -> 0.374 -> 0.222 (log-log slope **-0.35**; MIT slope
  -0.34), between 1/sqrt(M) capacity and the local-J0 saturation the capacity
  study already found. Per-cluster is **M-invariant** (0.886 at every M). And the
  BUNDLED peak increasingly lands on the WRONG segment — it cannot tell which
  patch fired: wrong-patch rate 6.7 % (M=2) -> 32 % (M=4) -> 50 % (M=16) ->
  **72 %** (M=64) on Intel (31 % on the more self-similar MIT). So bundling DOES
  throw away signal and provenance; per-cluster removes both, at exactly K-fold
  compute (~12 matches/frame here). This half of the hypothesis is CONFIRMED.

- **CLOSURE discrimination — genuine vs twin (140 dual-queries each, ring-key
  top-15 retrieval, GT labels).** On INTEL (distinctive content) per-cluster
  gives a real but modest edge: per-patch coherence AUC **0.557**, per-cluster
  max-over-patches AUC **0.573** (median separation +0.055, positive on 61 % of
  queries) vs BUNDLED chain coherence AUC **0.479** (separation -0.008, 49 %) —
  bundled chain sits AT/below chance because the 2-3-segment sum mixes in
  adjacent-place content. But on the MIT CORRIDOR — the actual wall — per-cluster
  does NOT help: per-patch AUC **0.484** (chance), and max-over-patches AUC
  **0.217** is WORSE than bundled 0.166, because taking the max over the ~12
  twins/query amplifies the best-aliasing twin (separation -0.070). FP@90 %-recall
  is ~0.90-1.0 in every configuration. Twins alias per-patch exactly as they alias
  per-sequence (SeqSLAM 0.500) and per-coherence-median (0.171 vs 0.172): the
  corridor wall is INTRINSIC to the appearance content, invariant to whether you
  bundle, sequence, or compare per-patch.

- **FRONTEND ATE (bundled shipped vs per-cluster localization).** Not a robust win
  — bundling's summation also AVERAGES OUT per-patch aliasing (protective), and
  per-cluster's greedy best-single-patch selection trades that robustness for
  crosstalk-free signal: **fr101 1.881 -> 0.914 m** (-0.97; closures-OFF audit
  3.158 -> 1.427, so the gain is genuinely frontend, not closure luck), but
  **fr079 5.523 -> 9.283 m** (+3.76, a single confident aliasing patch wins), and
  **Intel 2.440 -> 2.369 m** (neutral RMSE, median WORSE 1.55 -> 2.30). One big
  win, one big loss, one wash — high variance, no robust operating point.

VERDICT (decisive, split): bundling is a REAL, quantified representation-level cost
(the genuine match dilutes ~M^-0.35 and provenance is lost — the frontend and
distinctive-content closure feel it), and where content is distinctive per-cluster
recovers a modest discrimination edge (Intel closure AUC 0.573 vs 0.479; fr101 ATE
halved). BUT per-cluster is NOT the lever that breaks the wall: on the MIT corridor
it is chance-to-worse, and as a frontend it is a variance trade, not a net gain,
because bundling's averaging is itself protective. The recurring verification wall
is in the CONTENT, not the aggregation. Bundling stays the right default for the
frontend/closure; the actionable residue is that where the local map has a clean
dominant overlapping patch, a per-cluster best-patch frontend can sharply help
(fr101) — a conditional lever, not a shippable one.

---

## Adaptive content-keyed map clustering: a real O(area)-constant memory win on a brittle ATE knob

Information-driven patching (`ssp_adaptmap.py`, AdaptiveMapSLAM(BoundedSLAM), no
shipped edit): when an anchor opens, encode its world-frame MAIN-band content and
compare (normalized inner product) to nearby RECENT patches; above tau -> redundant
-> bundle-merge into that owner (no new patch); below -> novel -> new patch.
Same-pass drift guard (merge only within 20 anchors / 3 m -> a corridor revisited
at a different drift state is far in anchor index, never merged -> avoids the
scale-arrays smear). tau=inf reproduces the shipped map BIT-EXACT (selftest max
|dpose| 0.00e0); merging only removes patches -> memory monotone in tau.

**Memory mechanism WORKS (the robust result):** best memory at <= baseline ATE:
Intel tau .85 -> 2.338 m / 489 seg / 2.69 MB (-30 %); fr101 tau .70 -> 1.279 m /
138 seg / 0.76 MB (-59 % mem, -32 % ATE); MIT tau .75 -> 35.44 m / 1294 seg /
7.11 MB (-46 % mem) with **MIT's O(area) constant dropping 1.287 -> 0.69 seg/m**.
The linear-growth constant is genuinely compressible -> information-driven memory
efficiency is real (the original project goal). Controlled/monotone -> trustworthy.

**ATE-at-equal-memory is BRITTLE (the honest caveat):** the response is strongly
NON-MONOTONE in tau (Intel .90 -> 3.96 / .85 -> 2.34 / .80 -> 7.94) and NO single
tau transfers across logs. The swings are driven by merging RESHUFFLING the
loop-closure graph (accepted loops vary 27 -> 230) — consolidating the recent
bundle changes which closures fire and moves the frontend globally — the SAME
brittleness signature as the shipped coherence veto, not smooth signal loss. So
the ATE improvements are lucky operating points, not a robust Pareto gain; treat
the memory cut as the win and hold the ATE bonuses at arm's length.

**Key-tension answer:** "bland corridor content is redundant AND load-bearing"
holds ONLY on fr079 (floppy, sparse-closure — every tau worse; redundant content
is load-bearing there). Elsewhere merging is neutral-to-helpful, because on
revisit-DENSER graphs a consolidated owner is a STRONGER, less-ghosted loop-closure
target. So "redundant != compressible for SLAM" is a floppy-graph property, not
universal. VERDICT: a real memory-efficiency improvement on the O(area) constant
(30-60 %, esp. MIT), robust in the memory axis, brittle in the ATE axis — usable
as a memory lever (conservative tau ~0.85), not as an accuracy improvement.

---

## Hierarchical multi-resolution (SSP-scale) pose-graph correction — LIMIT: flat relax already optimal

Tests whether a coarse-first correction keyed to the SSP scale structure (group
5-kf anchors into large super-nodes = the low-frequency scale, relax the coarse
graph first, DRAG the fine anchors by a smooth inverse-distance-weighted
deformation field, then refine) beats the shipped flat anchor pose graph
(`ssp_hiergraph.py`, HierGraphSLAM(BoundedSLAM); sanity: hier=False and span=inf
both BIT-EXACT to BoundedSLAM.relax; flat baselines reproduce shipped exactly).

**The decisive isolation:** on a FIXED graph (flat vs hier over the identical
loop set, 8 seeds), both give the SAME ATE to <=0.1 % (0.002 vs 0.002 m). **The
flat single global least-squares solve already reaches the optimum — the 5-kf
anchor pose graph IS the multi-resolution deformation model, and coarse-first
converges to the same point.** So every real-log difference is an ONLINE-FEEDBACK
artifact, never a better optimum.

Real-log ATE (rmse m), flat vs hierarchical span sweep:
| log | flat | best hier | verdict |
|---|---|---|---|
| fr101 (dense) | 1.881 | 1.882 | neutral (well-constrained) |
| fr079 (floppy) | 5.523 | 7.16 | HURTS +30-95 % |
| intel | 2.440 | 3.57 | HURTS +46-126 % |
| MIT (bends) | 57.383 | 40.7 | apparent -29 % but a CHAOS ARTIFACT |

- Neutral where dense revisits over-constrain (fr101); WORSE on floppy/drifty logs
  (the transient coarse warp destabilizes the frontend + drift-gated loop
  admission — Intel loop count balloons 80 -> 147-171 with low-quality edges).
  Damage SHRINKS as span grows toward the flat limit — the best hierarchical
  config is the one that least resembles a hierarchy.
- SLOWER, refuting the multigrid hypothesis (Intel relax 8.95 s -> 17-34 s: the
  fine full solve still runs every relax, plus coarse overhead plus more edges).
- The MIT "win" is NOT better optimization — it is fewer/less-catastrophic
  closures (max-err 221 -> 79-97 m); the drag's online perturbation accidentally
  suppresses some false corridor snaps in the documented MIT chaos regime. The
  fixed-graph isolation proves the coarse level does not optimize better; same
  class as "drop corridor closures and ride odometry", not a multi-resolution
  correction absorbing the bend.

VERDICT: SSP-scale hierarchical correction does NOT improve on the flat anchor
pose graph — neutral at best, worse on floppy logs, slower. The flat relax already
distributes low-frequency drift optimally; the hierarchy only injects blocky-drag /
condensed-edge perturbation. Confirms the dispatch prediction: the 5-kf anchor is
already a deformation node. (No positive to audit — the MIT apparent-win is a
quantified chaos artifact.)

---

## Continual gradient-flow closure: valid non-folding mechanism, not a detection replacement

Detection-free paradigm (`ssp_flow.py`, subclass; no shipped edit): energy
E = lam_seq*||seq skeleton||^2 - lam_ov*sum rho(cos<P_i,P_j>) over ALL
spatially-near / temporally-distant anchor pairs (no detect/gate), fixed anchor
gauge. The overlap force is the ANALYTIC SSP correlation gradient (translation =
phase gradient iW, rotation = the stored d/dtheta derivative vector),
finite-difference-exact to rel 8e-8. Sanity: lam_ov=0 == odometry (0.0000 cm);
collapse guard holds (traj length 604->604 m with the stiff skeleton; +29 %
distortion without -> skeleton load-bearing).

- **Single-session (T1) — matches detected closure ONLY where the frontend is
  already in-basin.** fr101 flow 3.21 vs shipped 3.32 (median 1.18 < 1.37); but
  Intel flow 5.24 vs shipped 2.44, fr079 11.4 vs 5.52 (flow ~= frontend-only
  5.07/11.6). Detection-free flow CANNOT recover drift beyond the coarse-ring
  basin (~6 m) + spatial gate — it only REFINES, it can't do the long-range data
  association that detect+snap does. Annealing HURT (folded max 15-18 m).
- **MIT (T2) — GRACEFUL by construction.** Traj length 604->604 m, max anchor
  displacement 3.3 m, no twin fold / no 100 m blowup. The bounded robust kernel
  never COMMITS a corridor twin, so it cannot fold — graceful WITHOUT drought/PCM
  (but it also does not improve).
- **Multi-session (T3, the crown) — does NOT converge.** From a GT-free single-tie
  init (6 m) the residual stalls 6.00 -> 5.95 m; root-cause probe: genuine
  co-observed cross-pass cosine 0.097 vs noise p90 0.29 -> INDISTINGUISHABLE (the
  same cross-session data-association wall the discrete methods hit). From a
  favorable rigid T_AB init (2.64 m) the flow LOCKS confidently-overlapping
  regions (within-1 m anchors 8 -> 40) but the median doesn't converge —
  region-by-region locking only, not global alignment. (Anti-oracle confirmed:
  single-tie/rigid init, GT scores only, no shared-index.)

**User-suggested enhancements — both CONFIRMED as real conditioning/robustness
wins (not content fixes):** (a) GRADIENT-L2 normalization (divide the force by
|u_i||u_j|, keep stored vectors RAW so additivity is preserved) -> lr-ROBUST
(2.50-2.64 across lr 0.1-1.0) where the raw force DIVERGES at lr 0.1 (169 m); (b)
SOFT/HEX partition-of-unity feature distribution (bilinear-kernel /
Delaunay-barycentric) -> SMOOTHER descent (monotone % 46-50 vs hard 27) + wider
offset tolerance (kernel held 3.69 at a 1.5 m offset where hard degraded to 4.43),
point-mass conserved. Neither moves the cosine-indistinguishability wall.

VERDICT: continual gradient-flow closure is a valid, well-conditioned,
non-folding mechanism, but NOT a viable detection-free REPLACEMENT — it can only
refine when already in-basin and does not solve cross-session association. This
motivates a HYBRID (next): detected+verified closures do the long-range topology
(getting in-basin) that the flow can't, and the overlap flow does the dense
deformation refinement between them (the region-by-region locking it CAN do).

---

## Hybrid: detected closure + overlap-flow refinement — single-session is neutral-to-negative

The user's hybrid: run the shipped detect+gate+relax, then add continual
overlap-flow REFINEMENT on top, with seq + detected-loop closures as stiff
pose-pose anchors in one energy E = lam*||seq+loop||^2 - lam_ov*sum rho(cos)
(`ssp_hybrid.py`, reuses ssp_flow.FlowField grad-L2-normalized force; no shipped
edit). Init at the shipped-relaxed poses so lam_ov=0 is a fixed point (= shipped).
lam_ov sweep, ATE rmse m:

| lam_ov | fr101 | fr079 |
|---|---|---|
| shipped | 1.881 | 5.523 |
| 0.0 (no overlap) | 1.881 | 5.521 |
| 0.1 | 1.886 (+0.005) | 5.522 (-0.001) |
| 0.3 | 1.889 (+0.008) | 5.526 (+0.003) |
| 0.6 | 1.905 (+0.023) | 5.535 (+0.012) |
| 1.0 | 1.909 (+0.028) | — |

**On an already-well-optimized (detected-closure) map, anchored overlap-refinement
is NET-NEGATIVE** — monotone degradation with lam_ov (fr101 hurts, fr079 neutral
then hurts); lam_ov=0 (= shipped) is best. The overlap force is noise-dominated
(genuine cosine 0.097 vs noise 0.29-0.63, per the flow study), so on a tight map
it pulls toward spurious attractors rather than tightening. The anchoring does its
job — it keeps the degradation BOUNDED (fr101 +0.028, fr079 +0.012 max, no
drift-to-twins) — but bounded noise is not signal. The detected closures do all
the useful work; there is nothing left for the dense overlap force to add on a
map that is already solved, and to add value it would need to distinguish genuine
overlaps from noise — the verification wall. (Multi-session, where the map is loose
and there IS room, is the honest remaining test — the flow locked confident
regions from a loose init there, so anchored refinement of a loose alignment is
not a foregone negative.)

---

## Hybrid multi-session: sparse anchors do the work; overlap-flow has real local signal but is net-negative on ATE (the wall, precisely localized)

The honest test of the user's hybrid ("loop detection + direct warp AND relaxing
via overlaps"): a two-session field (Intel A/B split, A fixed as gauge, B free,
coarse GT-free 6 m init -- same as ssp_flow test_multi) with sparse cross-session
ANCHOR edges (`ssp_hybrid.run_multi`). The anchors are a FENCED DIAGNOSTIC: GT is
used ONLY to pick which anchor pairs genuinely co-observe and to supply the
closure relative-pose Z (perfect detection + measurement) -- never in the flow
force. This isolates the overlap-flow's refinement value from the detection wall.
Decompose anchors-only (lam_ov=0 = pure pose-graph) vs anchors+overlap (hybrid):

| n_anchors | lam_ov | coobs_med (B->A) | B_ATE (B->GT) |
|---|---|---|---|
| 0  | 0.0 | 6.00 | 6.60 |  (init, unchanged)
| 0  | 2.0 | **5.51** | 7.07 |  (pure overlap flow, no anchors)
| 8  | 0.0 | 4.36 | 5.92 |
| 8  | 2.0 | **3.97** | 6.30 |
| 30 | 0.0 | 0.91 | 6.09 |
| 30 | 2.0 | **0.76** | 6.37 |

The two metrics DIVERGE, and that divergence is the whole story:

1. **Sparse correct anchors do the real alignment work.** 30 perfect closures pull
   B's co-observed anchors onto A's frame: coobs 6.00 -> 0.91 m. The pose-graph
   backend works fine GIVEN closures -- consistent with every prior finding that
   the entire problem is DETECTION, not the backend.

2. **Overlap-flow carries GENUINE local signal.** It tightens the co-observed
   (truly overlapping) anchors at EVERY anchor level -- even standalone at k=0
   (6.00 -> 5.51, ~8%), and 0.91 -> 0.76 (~16%) atop 30 anchors. The overlap force
   is NOT pure noise: on genuinely-overlapping pairs it pulls inward correctly.

3. **But it is NET-NEGATIVE on global ATE at every level** (+0.3 to +0.5 m). Within
   the gate radius the same inward force also pulls TWIN / aliased near-pairs
   together (it cannot tell genuine overlap from a twin -- the content wall), and
   that distorts the non-overlapping trajectory. B-ATE is additionally floored by
   A's OWN drift (A map ATE 3.30 m rmse) since B is pinned to a drifted reference,
   plus B's raw drift in the 156/435 anchors that visit A-unseen regions.

**Verdict: the hybrid does NOT beat anchors-alone on the deployable metric (ATE).**
This is the most PRECISE localization of the wall to date: the overlap force is not
"no signal" -- it measurably tightens true overlaps (~8-16%) -- it is "signal not
SEPARABLE from twin-noise." Because aliased twins rival/outnumber genuine overlaps
inside any usable gate, the aggregate force is net-negative even with perfect
sparse anchors. Detected closures (the topology) do all the useful work; dense
overlap refinement cannot add net value without first solving genuine-vs-twin
discrimination -- the same verification wall, now shown to bound even the
gradient-flow refinement layer. (Fenced GT-anchor diagnostic, not deployable;
`ssp_hybrid.py run_multi`.)

**Independent audit (SOUND, two caveats surfaced).** An independent read confirmed
the anti-oracle fence (GT confined to co-visibility selection + anchor-edge Z +
scoring; the flow force is GT-free -- cleanest at the k=0/lam_ov=2 row, which has
ZERO anchor edges yet still moves coobs 6.00->5.51), the edge convention, and the
arithmetic. Two honest qualifications: (1) `coobs` measures exactly the quantity the
attractive force targets while ignoring its false attractions, so it is a
force-favorable/partial metric -- the load-bearing evidence is the fair aggregate
B-ATE (cleanly net-negative and monotone: +0.47/+0.38/+0.28 at k=0/8/30), not the
~8-16% coobs tightening. (2) The twin/aliased-contamination MECHANISM is inferred,
not isolated by ablation -- but it is corroborated by the pure-overlap ATE (k=0,
lam_ov=2) reaching 7.07 m, which OVERSHOOTS the perfect-30-anchor floor of 6.09 m:
overlap injects error BEYOND merely pulling B toward a drifted A, consistent with
spurious twin attraction. Neither caveat overturns the headline.

---

## Viewpoint-tagged dual-channel submap: compose/dedup WIN, but not a genuine-vs-twin discriminator

The user's submap-storage idea: alongside the content channel, store a second
VIEWPOINT channel encoding the robot POSES a cell's points were sampled from
(bind pose with a distinct symbol), so revisits can be told apart by where they
were seen from (`ssp_viewpoint.py`, agent-built, no shipped module edited; channel
rides the graph rigidly so a pose update shifts V exactly).

- **Test 3 correctness (selftest) — PASS.** vp_score rigid-invariant under graph
  correction (max |delta| 6.97e-11), V(anchor+d)==shift(d)*V(anchor) (1.6e-15),
  channel-OFF == shipped BoundedSLAM bit-exact (0.0), channel-ON leaves the
  content path bit-exact (0.0).
- **Test 2 composition/dedup (two-session Intel split) — WIN.** Over 53 overlap
  cells, viewpoint-overlap median 0.901; DEDUP (overlap>0.5 = redundant revisit)
  flags 48/53 = 0.91 as revisits vs 5 as NEW coverage, where blind bundling
  double-counts ALL 53. GT validates the signal: corr(vp-overlap, -true_viewpoint_
  gap) = **+0.929**. So the viewpoint channel is genuinely useful for redundancy
  detection in map COMPOSITION -- the original submap-storage use case.
- **Test 1 discrimination (genuine revisit vs twin) — NOT a net discriminator.**
  Intel (3866 genuine / 20765 twin candidates): content-coherence AUC 0.679;
  viewpoint best (lam_view=8m) vp-alone AUC 0.896 and content x vp 0.863 (a real
  +0.184 over content) -- BUT vp-alone 0.896 does NOT beat the SPATIAL-PROXIMITY
  CONFOUND (AUC 0.909, delta -0.013), and FP@recall0.8 = 1.000. On low-drift Intel
  the viewpoint channel is essentially re-encoding "are these anchors near each
  other in the estimate," which the pose graph already gives for free. MIT (deep
  corridor): EVERYTHING is weak -- content AUC 0.551, vp-alone 0.488 (BELOW chance),
  confound 0.634, content x vp 0.551 (no gain); the corridor wall admits no
  appearance discrimination at all.

**Verdict:** viewpoint dual-channel is a correctness-clean WIN for composition/
redundancy-dedup (corr +0.93) -- valuable for the bounded-map storage question --
but it is NOT a solution to the verification wall: on Intel it is confounded by
spatial proximity (adds nothing beyond the pose estimate), and on MIT it carries
no signal (below chance). Consistent with every prior thread: genuine-vs-twin
discrimination is the irreducible wall; where drift is low the pose already
discriminates, and where drift is high (the case that needs it) appearance --
content OR viewpoint -- cannot. (`ssp_viewpoint.py`; logs scratch_viewpoint_*.log.)

---

## Per-dataset lattice (scaling) + phasor-count re-evaluation (scratch_lattice_sweep.py)

Question raised in review: the shipped encoder lattice (oct4 = [0.25,0.5,1,2] m
matched rings x N_ANG=60 = D 240, + relo rings 5.3/12.8) is a FIXED module
constant applied to every log (never re-evaluated per dataset — deliberately, so
the held-out claims are zero-shot). Is some of the non-Intel accuracy gap an
UNMATCHED-ENCODER artifact? Swept scaling lam_max in {2,4,8} (4 matched rings =
geomspace(0.25,lam_max,4)) x phasors N_ANG in {36,60,90} on the SHIPPED pipeline
(`set_lattice` patches the ssp_slam_loop globals at runtime, no module edit;
harness reproduces shipped fr101 1.881 and Intel 2.440 BIT-EXACT at baseline).

ATE rmse (m), best per log in **bold**, (=shipped) marks lam2/N60/D240:

| log | lam2/N36 | **lam2/N60** | lam2/N90 | lam4/N36 | lam8/N36 | verdict |
|---|---|---|---|---|---|---|
| intel | 2.93 | **2.44 (=shipped)** | 3.09 | 9.93 | 13.03 | shipped optimal |
| fr079 | 11.66 | **5.52 (=shipped)** | 13.25 | 16.15 | 14.74 | shipped optimal |
| aces  | 10.26 | **6.21 (=shipped)** | 8.05 | 21.75 | 8.87 | shipped optimal |
| fr101 | 0.905 | 1.881 (=shipped) | 4.29 | 1.152 | **0.668** | COARSER wins 2.8x |

**Findings.**
1. For **Intel / fr079 / ACES the shipped lattice IS the per-dataset optimum** —
   every alternative is worse. MORE phasors hurt on Intel (N90 -> 3.09; loop edges
   80->136: extra angles admit more aliased closures), WIDER scaling is
   catastrophic (lam_max=4/8 -> 10-22 m: stretching past octaves loses mid-band
   wall structure and starves genuine closures, loops -> 1-3). So fr079's 5.52 and
   ACES's 6.21 are NOT unmatched-encoder artifacts; they are intrinsic (fr079
   floppy graph; ACES do-no-harm frontend gap, see section 6). The fixed choice
   was well-placed, not lucky.
2. **fr101 is the exception and has real headroom**: it improves MONOTONE with a
   coarser, lower-D lattice — lam2/N36/D144 -> 0.905, lam8/N36/D144 -> **0.668**
   (2.8x better than the 1.881 shipped; median 0.466 vs 1.551, a BROAD improvement
   not an outlier). Mechanism (plausible, AUDIT-PENDING): fr101 is a dense GENUINE-
   revisit building with little aliasing, so a coarser encoder that admits more
   closures (61 vs 53) helps — the opposite of Intel/MIT, which need the finer
   lattice to SUPPRESS aliased twins. Lower D is also ~cheaper (faster ms/kf).
3. **No single better fixed config exists** — the optimum is environment-dependent
   (Intel wants fine to suppress aliasing; fr101 wants coarse to admit revisits).
   This VALIDATES shipping one robust zero-shot lattice rather than tuning per log.

**Honest framing (PROTOCOL sec 4, sec 6).** The fr101 0.668 is (a) a per-dataset-
TUNED number, NOT comparable to the zero-shot held-out claim — fr101's honest
zero-shot ATE stays 1.881; and (b) a POSITIVE result that needs an independent
audit (closure-correctness: are the extra coarse-lattice closures genuine or
lucky twins?) before it is trusted as deployable headroom. Recorded as promising,
not banked.

---

## Submap SLAM (align-then-bundle hex patches) for loop closure — triangulated negative

User hypothesis: replace temporal 5-kf segments with persistent, area-indexed
HEXAGONAL patches; each scan soft-assigned to the 3 nearest hex centres
(barycentric), matched per-patch, position = weighted-average of the per-patch
matches, then bundled into the 3 at that pose (align-then-bundle). Repeated
aligned observations REINFORCE (signal ~N, noise ~sqrt N) instead of the
raw-pose SMEAR that killed ssp_scale_arrays. Revisits match into old-pass patch
content = closure. The premise: reinforcement makes submaps viable for closure.

Investigated end-to-end (scratch_submap_slam.py V1 frontend, scratch_patch_slam.py
redo with map/graph decoupled + prunable backend, scratch_submap_rearrange.py +
real-multipass mechanism tests) and cross-checked by a SotA-research agent and an
independent design-review agent. VERDICT: the reinforcement-across-passes premise
is architecturally unsound; it converges back to the shipped segment design.

**What is TRUE (mechanism tests, real Intel):** align-then-bundle at the CORRECT
pose reinforces (coherence +0.05..+0.12 vs raw-bundle -0.04..-0.10); a rigid
transform is a unit-magnitude phase multiply so it CANNOT restore magnitude lost
to destructive interference (best closure-rearrange recovery falls 0.999 -> 0.795
as per-cell drift grows 0.1 -> 1.0 m; fine 0.25 m ring survival 0.86 -> 0.19).
Real intra-cell write-pose disagreement on Intel: 1.24 m within-pass, 8.14 m
cross-pass -> both large enough to destroy the fine (cm-Z) rings.

**Why it fails for CLOSURE (triangulated):**
1. VSA theory (Frady/Kleyko/Sommer 2018; VSA-OGM 2024): bundling is LOSSY
   superposition -- crosstalk grows with item count, capacity ~D/log, and
   normalization makes a summed-in contribution only APPROXIMATELY subtractable.
   A loop closure fused into a patch HV is not cleanly prunable.
2. SotA (Cartographer, Hess 2016; Kimera): mature submap SLAM NEVER co-mingles
   passes -- a submap is built by short-window insertion, FROZEN at a fixed scan
   count (before drift smears it), never rewritten; a revisit makes a NEW submap
   + a PRUNABLE loop edge between frozen frames. Reinforcement is intra-pass;
   cross-pass linkage is edges. The freeze rule IS the write-drift-smear defense.
3. Independent design review of the redo: cross-pass patches ride only their
   FIRST-writer anchor, so later-pass content moves by the wrong correction --
   error maximal exactly when a loop closes (compounding); no pass-segregation;
   dropped LOO pruning; dead derivative-rotation term. "This is the scale_arrays
   negative in a new costume." The prescribed fix (per-pass frozen layers matched
   one-at-a-time) "converges toward ephemeral per-segment vectors" = shipped.

**Correction to an earlier note in this session:** the claim that the submap
frontend "beats the shipped frontend on fr101 (2.60 vs 3.16)" was apples-to-
oranges -- the submap 2.60 includes implicit revisit-closure while shipped 3.16
is frontend-ONLY. Fair comparison is submap 2.60 vs shipped-FULL 1.88 (worse),
and Intel submap 11.4 vs shipped 2.44 (far worse). The submap approach does NOT
beat shipped on either log.

**Net:** the shipped design (frozen single-burst 5-kf segments + prunable,
coherence/innovation-gated loop closures) already sits at the SotA-convergent
point that Cartographer/Kimera and VSA capacity theory independently prescribe.
Cross-pass reinforcement trades away prunability and re-incurs write-drift smear
for a benefit (noise-averaging) that only holds intra-pass, where shipped already
gets it. The genuinely open lever the research leaves is narrow: whether LARGER
frozen single-burst patches + hex spatial indexing improve retrieval over 5-kf
segments -- but that does not touch the verification wall and is close to shipped.

---

## fr101 lattice-headroom audit: FRAGILE knife-edge, NOT banked (PROTOCOL sec 4)

Audited the per-dataset-sweep positive (fr101 1.881 -> 0.668 at coarse lam8/N36/
D144). GT used only to label loop edges / score (anti-oracle). `scratch_fr101_audit.py`:

  config                 |  ATE   loops genuine twin (labeled subset)
  shipped lam2/N60/D240  | 1.869    53     4     0
  winner  lam8/N36/D144  | 0.664    61     5     1
  lam8/N36 (rerun)       | 0.664    61     5     1     <- deterministic
  lam6/N36/D144          | 4.817    14     -     -     <- nearby, CATASTROPHIC
  lam8/N48/D192          | 2.248     5     -     -     <- nearby, BAD

**VERDICT: the win is a knife-edge artifact, not a robust improvement.** It
reproduces bit-exact (deterministic), but the two NEAREST configs are 2.5x-7x
WORSE and admit wildly different closure counts (5 / 14 / 61) -- so 0.664 is one
config hitting a lucky closure set, not a generalizable "coarse-is-better-for-
fr101" law. Confirms the sweep's own caveat (no single better fixed config) and
vindicates audit-before-trust. fr101's honest ATE stays the zero-shot 1.881.

**SotA mechanism (research agent):** the coarse-helps-dense-revisit / fine-needed-
for-aliased effect is the generalization-vs-discrimination (matched-filter
BANDWIDTH <-> main-lobe-width) tradeoff -- coarse = wide capture range = more
closures (recall) but aliasing sidelobes; fine = sharp/discriminative but narrow,
fragile basin. Same tradeoff in BoW quantization (FAB-MAP, Cummins&Newman 2008),
Scan-Context resolution (Kim 2018), SSP kernel-width (Komer 2019; Frady/Kanerva/
Sommer 2021), radar ambiguity theory. The knife-edge fragility fits a wide-shallow
coarse basin: which closures lock is hypersensitive to config. Principled rule =
set finest resolved wavelength to the smallest scale at which distinct places stay
distinguishable; GT-FREE run-time proxy = descriptor-database self-similarity
(off-diagonal confusion mass) + top-1/top-2 score margin. Under-explored as an
ADAPTIVE-resolution method (most SLAM adapts the threshold, not the descriptor
scale) -- a genuine open direction, tested next.

---

## Windowed-relax marginalization: quick-fix falsified, real cause is solve-locality (scoped)

The O(t) full-graph relax is the shipped system's one remaining scalability
bottleneck (FINDINGS sec 9). Benchmarked the opt-in windowed marginalization mode
(slam.windowed) vs full relax (scratch_windowed_bench.py):

  log    | full ATE / ms/kf | windowed ATE / ms/kf | retired
  intel  | 2.440 / 15.2     | 6.196 / 14.5         | 320
  fr079  | 5.523 / 26.4     | 7.057 / 25.3         | 93
  fr101  | 1.881 / 25.8     | 1.878 / 25.1         | 27

Two findings: (1) windowed is barely faster (~5%) -- at bounded scale (hundreds of
anchors) full relax is already cheap, so O(t) has not kicked in on these logs;
(2) windowed badly hurts accuracy on drift-heavy logs (Intel 2.44->6.20, fr079
5.52->7.06), scaling with retirement count; low-drift fr101 unaffected. Confirms
windowed is correctly opt-in-OFF.

SotA scout (Schur/GLC/iSAM2 marginalization) named the class of bug: hard-freeze
!= marginalization. HYPOTHESIS: retired anchors are frozen at stale poses, so their
keyframes are frozen islands that never follow a later correction; fix = rigidly
attach each retired anchor to its bypass-parent and back-substitute its pose from
the corrected parent after each relax (scratch_windowed_fix.py, WFix subclass).
FALSIFIED: win+fix == windowed to <2 cm on all three logs (Intel 6.196->6.180). The
shipped _maybe_retire already emits a SOFT covariance-weighted bypass edge, so
graph-level propagation was never the issue. REAL CAUSE localized: the windowed
SOLVE itself is local -- _relax_solve(Sw) optimizes only the window of anchors
around a new closure and freezes the rest, so a loop correction that should shift
the whole trajectory only shifts the window; the full_every=20 global bleed catches
up periodically but error accumulates between bleeds. PROPER FIX (SotA rank 2):
iSAM2-style affected-subtree resolve (Kaess 2012) -- re-solve exactly the region a
loop perturbs (endpoint-to-common-ancestor), globally consistent, no freezing;
best via GTSAM ISAM2 if a dependency is acceptable. This is a substantial rework
whose payoff is ASYMPTOTIC (unbounded runs where O(t) full-relax dominates) -- the
current bounded logs are fine with full relax at 15-26 ms/kf. Scoped as a future
engineering task, not a quick win. Full relax stays the default.

---

## ROVER-style trajectory-deformation loop verifier (untried lever) — redundant with the innovation gate, not a wall-breaker

The alternative-paradigms scout flagged two constraint-legal levers we hadn't
tried; the passive-compatible one is ROVER (Yu et al. 2024): verify a candidate
closure by ADDING it to the pose graph, re-optimizing, and scoring the trajectory
DEFORMATION -- a correct loop deforms gracefully, a false one catastrophically.
Prototyped on a synthetic drifting loop (`scratch_rover.py`; minimal SE(2)
pose-graph relax; SE(2)-aligned pre/post deformation + post-relax residual),
compared head-to-head with the pre-relax innovation residual the shipped gate
already uses:

  candidate               | innov(m) | ROVER rms | postRes
  GENUINE (last=first)    |   3.26   |   0.823   |    38.75
  twin coincide w/ 1/4    |   7.89   |   4.115   |  1905.62
  twin coincide w/ 1/2    |  13.37   |   4.285   |  4140.99
  twin coincide w/ 3/4    |   9.65   |   2.237   |  1879.45
  DANGEROUS twin innov~0  |   0.00   |   0.000   |     0.00   <- ROVER passes it too

**Finding: for a SINGLE closure, ROVER adds no new discriminating axis over the
innovation gate we already ship.** (1) Geometrically-INCONSISTENT twins are caught
by BOTH -- the innovation residual already separates them (genuine 3.26 vs twins
7.9-13.4), and ROVER's post-relax residual separates them more sharply (39 vs
1900-4100) but redundantly. (2) A DANGEROUS twin whose Z AGREES with the drifted
estimate (innov ~ 0 -- the only kind that passes the innovation gate) produces
ZERO deformation and ZERO post-relax residual: **ROVER passes it exactly where the
innovation gate does.** ROVER's post-relax residual is a sharper proxy for the same
cycle-consistency signal; it cannot see a twin that is consistent with the current
estimate, which is precisely the dangerous case and the info-theoretic twin. Its
one genuinely distinct value (per the scout) is SET-level -- catching a *group* of
pairwise-consistent closures that jointly deform the trajectory -- an incremental
improvement over pairwise PCM, but it too passes a globally-consistent (symmetric)
twin. VERDICT: ROVER is a sharper, global-consistency verifier, NOT a new
verification axis and NOT a wall-breaker -- consistent with the campaign
conclusion. Not worth a real-log build for the single-closure case; the only part
worth revisiting is set-level global consistency, and only if pairwise PCM is
observed to admit a jointly-inconsistent cluster.


## 2026-07-09 — Extended-overlap consensus closure (negative) + dependency-ordered component ladder (synthetic→real)

### Extended-overlap consensus closure — TESTED, CLEAN NEGATIVE
User idea: close only after generating a significant *overlap* of the two loop
ends, match them n-to-n, joint-solve the alignment, then lerp the correction.
`scratch_overlap.py` / `scratch_overlapslam.py` (no shipped module edited; the
subclass intercepts the parent's full-confidence admission).

- **Core discriminator VALID in isolation** (`scratch_overlap.py`): genuine vs a
  POINT-TWIN — single-point coherence is IDENTICAL (0.962/0.962, provably blind)
  while n-to-n consensus separates them 1.00 vs 0.00.
- **Every full-SLAM integration HURTS BOTH regimes** (sanity: min_agree=0 ==
  shipped 1.881 EXACT, so not a bug):
  - temporal-hold: fr101 1.88→3.0 — STALE constraints (bounded memory can't
    re-measure a held closure; batch-release jerks the graph);
  - spatial hard-gate (bundle-to-bundle registration, staleness-free): fr101 3.0,
    intel 2.44→4.94 — rejection rolls back the drift clock → loosens the
    innovation gate → FLOODS 80→136 loops;
  - spatial soft-weight: fr101 3.14, intel 5.83.
- **Two principled walls.** (1) a brief GENUINE revisit has few overlapping recent
  anchors → low corroboration → indistinguishable from a point-twin (hurts dense-
  revisit fr101); (2) in real aliased envs the closures that DO corroborate are the
  EXTENDED-symmetry aliases (corridors) while brief-genuine don't → corroboration
  is ANTI-correlated with correctness (soft up-weights the twins). The verification
  wall from a new angle. The bundle-to-bundle registration primitive (segvec is
  anchor-frame → plugs straight into the Matcher's rotate/translate path; segder =
  coarse half-step) works and is reusable.

### Dependency-ordered component ladder, synthetic→real
User directive: every component by dependence, in synthetic envs with increasing +
component-specific complications, progressing to REAL benchmark subsets at the top
rung. `scratch_components.py` (roots/mid/upper/real/interp) + `scratch_realbench.py`
(real-lidar frontend registration bench: gfs poses as GT, register each scan vs its
gfs-posed neighbourhood — isolates the frontend from the closure-admission confound
that dominates ATE; anti-oracle: gfs only places the submap / seeds the offset /
scores).

- **Roots.** encoding robust to noise ~½ the finest λ (real margin 0.89, diff-place
  floor higher 0.106 vs synth 0.053 = real is more self-similar); translation
  capture = `t_half`=0.48 exactly; rotation derivative is N_ANG-dependent and
  CROSSES OVER (hurts coarse −0.075@N30, helps fine +0.020@113, marginal on real
  where noise erodes it); bundling FHRR capacity synth ~4-8 (repetitive room is
  pessimistic) but real ~8-16 → validates `ANCHOR=5`.
- **Mid.** multi-scale: the fixed lattice is scale-robust (cm precision across
  3–96 m extent); registration robust to 50% partial-overlap + 20% clutter (clutter
  is the one frontend weakness — the correlation sums all points, no outlier
  rejection).
- **Upper.** verification WALL quantified: synth AUC *inverts* below 0.5 for twin-
  similarity s≥0.5 (a geometric twin out-coheres a genuine revisit, which carries
  viewpoint change the twin doesn't → viewpoint-invariance and twin-rejection in
  direct tension); real intel AUC 0.82 average BUT a 4.6% aliased tail out-coheres
  the median genuine revisit and OUTNUMBERS the 204 genuine revisits ~7:1 — no
  coherence threshold separates them. backend relax absorbs ≤10% aliased outliers
  flat (ATE 6.8→6.5 cm, IRLS downweight, 0 hard-prune).
- **Per-component synthetic→real (frontend elements).** N_ANG 113 CONFIRMS (real
  registration 19–32% better than 60, saturates ~89–113); scales REVERSED (real
  prefers 3–4 octaves; more scales add clutter noise and HURT: intel 3/4/5-oct →
  0.044/0.053/0.059); thermometer DOESN'T transfer (real off best — local reg
  doesn't exercise range-gating); capture `t_half`=0.48 adequate.
- **META-FINDING (the payoff):** synthetic misleads DIFFERENTLY per component —
  OPTIMISTIC on richness (scales/thermo/derivative), PESSIMISTIC on capacity
  (bundling), and UNDERSTATES the discrimination floor (verification is worse on
  real). No single "synth easier/harder" rule; each component needs its own real
  validation. Real-validated LEAN frontend: **N_ANG=113, 3–4 oct, thermo off,
  t_half 0.48, segder marginal at 113 (droppable candidate)**. Frontend is
  registration-saturated (2–5 cm); the remaining real-ATE lever is closure
  admission — which the coherence gate re-tuning test below probes.

### N_ANG=113 end-to-end: the frontend gain is NON-deployable (coherence gate has zero purchase)
`scratch_cohsweep113.py` — sweep `coh_target` at 113 on fr101 vs shipped 60/0.55:

    fr101 SHIPPED 60/0.55 : ATE 1.881  med 1.551  53 loops
    fr101 113/0.45        : ATE 4.227  med 1.589  83 loops
    fr101 113/0.55        : ATE 4.227  med 1.588  85 loops
    fr101 113/0.65        : ATE 4.227  med 1.587  85 loops
    fr101 113/0.80        : ATE 4.228  med 1.593  83 loops

ATE is FLAT at 4.227 across *every* coh_target — the gate has zero effect. The
median (1.59) matches shipped (1.55), so 113's FRONTEND is healthy; the RMSE blowup
is the ~30 extra closures the finer angular sampling admits, and they are COHERENT
ALIASED TWINS (the 4.6% tail from the verification rung) that sit *above* any
coherence threshold — the verification wall, invariant to the gate. **Verdict:
113's real registration gain does NOT deploy to ATE.** The coarser shipped 60-lattice
is ATE-optimal precisely because coarser angular sampling forms fewer spurious
coherent matches — it is a natural twin-suppressor. There is a FRONTEND-PRECISION vs
TWIN-ADMISSION tradeoff, and 60 favors the ATE-dominant closure robustness. So 113
is a representation / quantization improvement (better registration, leaner, more
rotationally smooth), NOT an ATE improvement — adopt it only where the goal is the
representation, not the deployed benchmark. This is the whole synthetic→real
campaign's end-to-end closure: the frontend is real-validated and improvable, but
ATE is closure-wall-limited and *invariant to frontend quality*.

### Quantization headroom + segder-drop at N_ANG=113 (`scratch_quant.py`)
Representation-track work (the lean 113 config exists to be quantized).

- **segder-drop is free / better at 113.** Storage-path rotation registration
  (store a scan in anchor frame, reconstruct the world map at the anchor's sub-grid
  heading, register a query): permute-only med 0.0001 vs permute+derivative 0.0046.
  At the fine 113 lattice the matcher's own rotation search covers the ≤0.8° sub-grid
  offset exactly, so the first-order derivative only injects O(d²) noise into the
  stored map. The N_ANG crossover from the component ladder, cashed out: the
  derivative earns its keep at coarse 60 (3°/step) but is dead weight at 113 →
  **drop `segder`, half the phasor count** (needs an end-to-end relax-time check to
  bank; static evidence is strong).
- **Polar multi-bin phasor beats Cartesian at every bit budget** (real intel subset,
  3-oct 113; reg median / coh-fidelity):

      full complex128        128b  0.0036 / 1.000
      cartesian 3+3            6b  0.0045 / 0.955
      polar 16phase x 4mag     6b  0.0040 / 0.972   <- best
      cartesian 2+2            4b  0.0058 / 0.863
      polar 8phase x 2mag      4b  0.0046 / 0.906
      phase-only 8             3b  0.0045 / 0.768

  Quantizing PHASE (bins) + MAGNITUDE (levels) — "mag instead of sin/cos" — beats
  quantizing (real, imag) at matched bits on BOTH registration and coherence, because
  the SSP correlation is a phase inner product so polar aligns with the native
  structure while Cartesian spends bits fighting it. Sweet spot **16phase x 4mag =
  6 bits/phasor** = near-full registration + 0.97 coherence. Phase-only (2-3 b) is
  excellent for registration but sheds the magnitude closure coherence needs.
- **Combined:** segder-drop (2x) x polar-6-bit (~10x vs the shipped complex64's 64
  b/phasor) ≈ **20x smaller map** than the current deliverable, at cm registration
  and 0.97 coherence fidelity — a concrete advance on the bounded-memory thesis.
  NEXT: end-to-end confirm (segder-drop + polar-6bit map through the full pipeline —
  ATE + closure recall vs shipped).

## 2026-07-10 — FPGA track opened: write-time quantized store, the perturbation band, point encoding, Python-fed webvis replay

Session goal (user directive): progress toward an SSP SLAM system for FPGA
deployment — float groundwork first, then a binary/integer VSA version; rework
the Python side (the webvis had been hand-tuned on the synth sandbox and its
real-data part broke). New module `ssp_fpga.py` (subclass/import only; the
neutralised subclass is asserted bit-exact to the parent: `selftest` max|Δpose|
= 0.0). Shipped baseline reproduced first: Intel 2.440 / 80 loops / 15 ms/kf,
bit-exact.

### FPGA sizing: compute is trivial, memory + noise-sensitivity are the design pressures
Analytic op count of the shipped hot path (`ssp_fpga.py ops`, n_pts=200):
frontend encode 0.10 + coarse banks 0.97 + fine 0.54 + segment fold 0.22 +
amortized loop attempt 0.72 ≈ **2.55 M MAC-equiv per keyframe → 0.05 GMAC/s at
20 Hz** — a fraction of one DSP array at 200 MHz; even a small FPGA is compute-
idle. Map store at the shipped D=360×2 vectors: Intel 3.93 MB (c64) → 368 KB at
6 b/phasor; MIT 13.5 MB → 1.27 MB. The real design pressures are (a) map
memory (solved below, per-log caveats) and (b) the closure cascade's
perturbation sensitivity (measured below — the session's centerpiece).

### Write-time polar-quantized store (deployable, unlike read-time requant)
`QuantStoreSLAM`: a segment accumulates at full precision while its anchor is
active (hardware: a 1-deep high-precision accumulation buffer) and is polar-
quantized ONCE when the anchor advances; every later read — frontend recency
bundle, loop candidates, coherence probes — sees quantized content.

- Read-time requant (scratch_quant_e2e.py, the queued NEXT from 2026-07-09):
  fr101 6b 1.646 / 4b 0.866(!); intel 6b 4.890 / 4b 11.12. Models nothing
  storable; superseded by write-time.
- Write-time, per-VECTOR scale: fr101 8b 1.886 (free; med halves to 0.669) /
  6b 1.656 / 4b 2.594; intel 8b 5.937 / 6b 3.733 / 4b 7.728 with loop counts
  scattering 56/169/198 (vs 80).
- **Per-ring scale fix (mechanism confirmed static).** The per-vector 99th-pct
  scale is set by the phase-coherent relocalization rings (per-ring |v| p99 =
  7.9/14.4/19.7/25.9/24.9/35.1 across λ 0.25..12.8), so the two FINEST rings —
  the closure veto's `coh[:2]` statistic — collapse into the bottom magnitude
  levels: per-ring coherence fidelity at 6 b jumps 0.841/0.909 → 0.972/0.974
  with per-ring scales (6 extra f32/vector). This also explains why the static
  2026-07-09 bench (3-oct lattice, NO relo rings) failed to predict the e2e
  regression. A dead-zone (true-zero) magnitude code was also added; the
  floor-mass hypothesis it tests was NOT supported statically (distant-segment
  |xcorr|: raw 0.517 / mid-tread 0.502 / dead 0.496).
- Write-time, per-RING scales, intel: 8b 4.097 (loops 183) / 6b 5.648 (64) /
  4b 4.790 (62) / 10b 4.856 (113) — non-monotone in bits, counters wobbling
  around shipped [veto 51 infl 51 innov 26]. Interpretation resolved by the
  chaos control below.

### The perturbation band (chaos control) — intel's 2.440 is a knife-edge
`NoiseStoreSLAM` adds i.i.d. RELATIVE Gaussian noise (ε·mean|v|) to each
segment at freeze time (deterministic seed), isolating "any map perturbation"
from "quantization structure":

    intel:  ε=1e-6 → 2.440 (print-precision identical; loops/counters
                            IDENTICAL 80/51/51/26; audit: d_ATE 7.4e-7,
                            max|Δpose| 1.3e-5 — not strictly bit-identical)
            ε=1e-5 → 2.440 (identical at print precision)
            ε=1e-4 → 3.999            ε=1e-3 → 4.168 / 3.596 / 3.866 (3 seeds)
            ε=1e-2 → 5.232            ε=3e-2 → 3.464
    fr101:  ε=1e-3 → 1.883   ε=1e-2 → 1.884   (robust; ≈1.881 shipped)
    fr079:  ε=1e-3 → 13.547  ε=1e-2 → 15.294  (FRAGILE, 2.5–2.8×)
    aces:   ε=1e-3 → 6.963   ε=1e-2 → 7.457   (mild, +12–20%)

**Intel's response is a multi-meter, NON-MONOTONE band for every tested
sub-percent perturbation** (1e-4 → 4.0; 1e-2 → 5.2; 3e-2 → 3.5) — not
progressive degradation. AUDITED MECHANISM (read-only audit, CONFIRMED WITH
CAVEATS): the continuous channel alone predicts ~1 cm of pose change at
ε=1e-3 (measured gain ~13 m pose per unit relative map perturbation at
1e-6), yet the observed response is ~1.7 m with SHIFTED decision counters
(veto 64 vs 51) — i.e. 100–1000× super-linear, driven by **flipped discrete
decisions in the closure-admission cascade** (which then take different
early-stopped TRF solve paths), not the solver path alone. A float32
rounding of the frontend guess alone lands at 3.971, in the same band.
Scope caveats (audit): the band rests on 6 noise samples, intel-only; and
gross quantization damage IS still distinguishable — 4-bit per-vector quant
(7.728) falls OUTSIDE it. Defensible statements: (1) sub-percent freeze-time
perturbations move intel's ATE by multiple meters via closure-decision
flips, so individual intel deltas for ≥6-bit quantizers are NOT attributable
to quantization damage — intel verdicts must rest on the perturbation-
response comparison, not single numbers; (2) holding 2.440 (at print
precision) required ≤1e-5 relative map fidelity — no useful quantization
provides that, and 2.440 is a knife-edge configuration, not a robust
operating point (the robust perturbed-intel figure is the band, still ~5×
under raw odometry 24.2); (3) per-log tolerance spans 3+ orders of
magnitude — fr101 absorbs 1e-2 (and is quantization-FREE at 6–8 b:
1.66–1.89 at 183–246 KB vs 1.9 MB c64, a real ~10× map-memory win) while
fr079 degrades 2.5× at 1e-3 and aces +12–20%. Noise tolerance tracks
closure redundancy (fr101's 53 dense closures ≫ fr079's 32 on a floppy
graph).

### E1 — fine rotation via scan d/dθ derivative: CLEAN NEGATIVE
`DerFineMatcher` replaces the fine stage's 9 re-encodes with one derivative
encode + permutations (11 → 3 encodes/match, attractive for fabric): fr101
1.881→2.496, intel 2.440→4.666. The first-order Lie term is validated for
STORAGE reconstruction (≤1.5°, bundle-averaged) but is not accurate enough for
the fine stage's 0.375° argmax decisions on a single scan. Keep the re-encodes
(0.5 MMAC/kf — cheap). Rejected.

### E2 — per-beam point encoding: 3-of-4 win, intel regression (AUDIT PENDING)
`points_from_scan`: one phasor per raw hit, weight = r·dθ (arc-length
footprint — the surface integral as a per-beam Riemann sum), replacing the
whole segment-resampling + occlusion-filter preprocessing (the one
irregular-compute stage in the fabric path):

    fr079  5.523 → 2.210  (loops 32 → 66)   — best fr079 result in the repo
    aces   6.212 → 4.409  (11 → 39)         — beats shipped AND raw odo 5.41
    fr101  1.881 → 1.569  (53 → 64)
    intel  2.440 → 5.126  (80 → 250, loop-edge explosion)

**AUDIT VERDICT (read-only, PROTOCOL §4): CONFIRMED WITH CAVEATS.** The
auditor reproduced all three improvements end-to-end bit-consistently (fr079
2.210134/66, aces 4.409/39, fr101 1.569/64), verified run_log is line-for-line
the shipped recipe with GT touching only the score block, verified the
neutralised subclass is empirically bit-exact to the parent (so E2 is a PURE
sampling swap), verified identical eval populations between arms (fr079: 4286
matched reference poses in both), and found no fake-win channel (no duplicate/
self edges, no fallback asymmetry, determinism confirmed). Caveats that stand:
(1) intel regresses with a 3× loop-edge explosion (80→250) — a structural
change no noise perturbation produced (band loop counts stayed 51–183), so
the regression is real E2 behavior; E2 FAILS the multi-log gate as a
shipped-default replacement — it is a 3-of-4 result, adopt per sensor/log.
(2) The beam-count/Nyquist mechanism story is REFUTED: aces has 180 beams
like intel and MORE super-Nyquist hits (11.3% vs intel's 6.2%; intel's hits
are mostly near, median 1.96 m) yet improves. Mechanism: UNKNOWN. What E2
actually changes (audit-quantified): drops the occlusion filter, gives
isolated far hits up to ~0.7 weight (~6× shipped's 0.12 lone-hit weight),
underweights oblique surfaces by cos(incidence); per-scan weight sums stay
comparable (8–20 both arms). (3) The intel delta's SIZE is confounded by the
perturbation band; the loop explosion is the trustworthy regression signal.
Weight-cap probe (E2b, w≤0.25): does NOT rescue intel (6.720, loops 150) and
is ~neutral on fr079 (2.151) — far-hit leverage is not the intel mechanism.
Also E2-fr079 is far OUTSIDE fr079's perturbation response in the GOOD
direction (generic ε=1e-3 noise degrades fr079 to 13.5; E2 improves it to
2.21) — the improvement is signal, not a basin re-roll.

### Integer front-path model (binary-VSA track opened)
`IntSpec`/`cis_int`/`IntEncoder`/`IntMatcher`/`IntSLAM`: the fabric arithmetic
modeled value-exactly on integer grids (ROM-rounded cis at 2^addr phases ×
2^(bits-1) signed values, fixed-point weights, integer MACs carried in float64
— exact below 2^53), with the HW split: encoder + correlation banks + store on
fabric, pose math/gates/relax host-side. Smoke (fr101 cap 600): float 0.927 /
int addr10-val9-w9 0.920 / **QPSK (4-phase cis ∈ {±1,±i}, unit weights) 1.096**
— the binary extreme is NOT dead on arrival. Full ladder running (fr101 +
intel, addr10/8/6/4/3/QPSK).

### Webvis real-data path: fixed and Python-fed
Diagnosis: the Intel replay was COUPLED to the sandbox config — buildIntelLattice
used cfg.ladder/cfg.hex, intelProcessKF used cfg.occ/cfg.pointEnc, and the hex
toggle rebuilt the Intel lattice — so synth-env hand-tuning silently reconfigured
the real-data pipeline (the reported breakage). Fixes: (1) the Intel replay is
now PINNED to the shipped config; (2) NEW `demo/export_replay.py` runs the real
`ssp_bounded` deliverable and embeds a self-contained replay (ranges, odometry,
reference, online estimates, per-kf anchor refs, per-relax anchor snapshots,
loop edges; 4.06 MB → page 6.3 MB) — reproduces 2.440 / 80 loops / 30 snapshots
deterministically; (3) index.html gained a "shipped replay (Python)" source in
the player bar, driven by a slam-shaped shim over the recorded streams (graph
snaps replay exactly via the snapshots; Python is the source of truth, the JS
pipeline can no longer drift); (4) jsc parity harness: the page's own
loadReplay+makeReplayShim reconstruct the trajectory at RMSE 2.351 over its
2034 kf-indexed reference matches (the official 2.440 is over 890 ref-indexed
matches — matching-direction difference, understood), 80/80 loop edges, all
snapshots applied. Browser-runtime remains unverified (numbers-only rule).
Also documented from 2026-07-09 scratch outputs: the sandbox's odometry-MAP
fusion in Python on real logs — backend odo factor: fr101 1.744 wins but
intel 13.5 / fr079 14.3 catastrophic; frontend MAP prior γ=1: aces 6.21→1.39,
fr101 1.70, intel 8.5 / fr079 10.1 — fixed-weight fusion does not transfer
(consistent with FINDINGS §6); it stays a sandbox toggle, not a pipeline
change.

## 2026-07-10 (consolidation) — the perturbation-band table, E2's mechanism, and the binary verdict

The session's second half turned the individual findings into a uniform,
band-probed comparison (PROTOCOL §6 band rule; `ssp_fpga.py band`). Four
configs: **shipped** (segment resampling, float, c64 store), **E2** (per-beam
point encoding, float, c64), **FPGA8** (E2 + 16ph×4mag per-ring 6 b store +
int addr8/val7/w7 arithmetic), **BINARY** (E2 + 4ph phase-only 2 b store +
QPSK unit-weight arithmetic); bands over ε ∈ {0, 1e-6, 1e-3 × 2 seeds}
freeze-noise; plus **FPGA-lean** (E2 + 2 b store + int8 arithmetic — the
post-hoc winner combining the store floor with the arithmetic knee).

### The band table (ATE rmse m, [min .. max] median; map KB where quantized)

    log    shipped              E2 point             FPGA8 (≈6b, ~10x mem)   BINARY (2b+QPSK)      FPGA-lean (2b+int8)
    fr101  [1.88 .. 1.88] 1.88  [1.57 .. 3.74] 1.68  [2.21 .. 4.94] 3.57     [3.75 .. 5.93] 5.47   [1.13 .. 3.08] ~2.2 @ 75 KB
    fr079  [5.52 .. 12.4] 8.37  [2.21 .. 4.86] 2.77  [3.15 .. 6.71] 5.37     [10.4 .. 13.9]        {3.15, 6.25} @ ~110 KB
    aces   [6.21 .. 8.18] 7.03  [3.48 .. 6.66] 4.90  [2.15 .. 8.65] 7.57     [11.8 .. 22.4] 16.0   {8.04, 8.74} (fails: 5 loops)
    intel  [2.44 .. 6.72] 3.17  [4.35 .. 5.50] 4.91  [3.69 .. 5.60] 4.44     ~12.5-12.7            (not run)
    belg   {2.64,2.49,2.14}     {2.44,2.05,1.89}     —                       —                     —   (range-identity eval)

Per-log configuration guidance falling out of the table: fr101/fr079 →
FPGA-lean (25× less map, integer datapath, band at/below shipped); aces/belg
→ E2-float (the lean store starves aces's already-sparse closures: 5 loops);
intel-class sparse-beam clutter → shipped sampling (E2 band-worse), where
FPGA8 is still band-indistinguishable at 10× less map if memory matters.

Note on intel-FPGA8: its band [3.69 .. 5.60] sits INSIDE shipped's
[2.44 .. 6.72] — at ~10× less map memory the quantized-integer configuration
is band-indistinguishable from shipped on the tuning log.

Reading (band vs band): **E2 dominates shipped outright on fr079** (its whole
band clears shipped's best draw) **and on aces and belgioioso** (every rung
better); on fr101 it improves the median (1.68 vs 1.88) at the cost of
variance the shipped config doesn't have (fr101-shipped is the suite's one
point-stable configuration, flat at 1.88 even under 1% noise); **on intel it
is band-worse** (median 4.91 vs 3.17, overlapping ranges) — intel keeps
shipped sampling. FPGA8 buys ~10× map memory for roughly one band-notch of
accuracy on the E2 logs and dominates shipped on fr079; BINARY (QPSK
arithmetic) fails everywhere except fr101-median — the arithmetic, not the
store, is the binary bottleneck.

### E2's mechanism: it is a FRONTEND registration win (5/5 logs)

Frontend-only decomposition (loop attempts disabled):

    log    shipped-frontend  E2-frontend   (raw odo)
    fr079      11.62            2.75        14.35
    aces        6.38            4.33         5.41   <- E2 frontend BEATS odo
    intel       5.07            4.39        24.15
    fr101       3.16            3.00         8.56
    belg        2.45            2.17         1.72

Per-beam point encoding improves the scan-to-map registration on every log —
on fr079 the E2 frontend alone (2.75) nearly matches the full shipped system
(5.52 with closures). The full-system per-log differences are all downstream
closure-cascade behavior (intel's loop-count explosion to 205–286 with an
in-band-worse median; fr079's doubled genuine closures). Two candidate
mechanisms were tested and refuted en route: beam-count/Nyquist (audit: aces
has 180 beams and MORE super-Nyquist hits than intel, yet improves) and
viewpoint-neutrality of content (scratch_e2mech.py: point content's
cross-pass/same-pass coherence ratio is LOWER, 0.29 vs 0.43 on fr079). The
surviving explanation — the segment resampler's hit-to-hit chord
interpolation invents straight-line mass across furniture/clutter that the
outlier-blind correlation then optimizes against — was then tested DIRECTLY
and REFUTED as the position effect: **E3** (samples at REAL hit positions but
with the shipped occlusion-filtered chord weights) reproduces shipped's bad
fr079 frontend almost exactly (frontend-only 11.69 vs shipped 11.62 vs E2
2.75; full 7.92 in shipped's band). So the E2 win is carried by the
WEIGHTING and/or the occlusion filter's mass removal, not by where the
samples sit. **E4** (r·dθ weights WITH the occlusion filter) then splits the
two contributions: fr079 frontend-only shipped/E3 11.6 → E4 6.02 → E2 2.75.
**Half the win is the arc-length r·dθ weighting; the other half is NOT
running the occlusion filter.** The filter was designed to kill phantom
bridges in chord interpolation — with point sampling there are no bridges to
kill, and its weight-zeroing near depth discontinuities (door frames,
furniture edges) deletes exactly the most translation-informative scan
content. A shipped design feature is thus actively harmful in the
point-sampling regime (and its protective role is moot there). FINDINGS §6 gains the addendum: the
do-no-harm gap's SIZE was substantially sampling damage (aces frontend now
beats its own odometry); the guard's impossibility is untouched.

### Arithmetic and store ladders (fr101 = the stable probe; intel = band)

    int arithmetic (c64 store, fr101):  addr10 1.90 | addr8 2.10 | addr6 3.09
      | addr4 2.17 | addr3 2.98 | QPSK-unit 3.81 (med 1.57 = shipped median)
    int arithmetic (intel): >=addr8 lands in the perturbation band (4.7-6.1);
      below addr8 there is REAL damage beyond the band (addr6 10.5, addr4 9.0,
      addr3 8.5, QPSK 11.5) -> the arithmetic knee is addr8 (256-entry cis ROM,
      7-bit values); QPSK is median-viable on robust logs only.
    store floor (phase-only per-ring, float arith, fr101): 16ph/4b 2.98 |
      8ph/3b 2.34 | 4ph/2b 1.41 - the STORE tolerates 2 bits/phasor on the
      robust log; combining 2b store + int8 arithmetic + point encoding
      (FPGA-lean) gives fr101 [1.13..3.08] @ 75 KB (vs shipped flat 1.88 @
      1.9 MB) and fr079 {3.15, 6.25} @ ~110 KB (vs shipped band 5.5-12.4 @
      2.6 MB) - comparable-to-better accuracy at ~25x less map memory on an
      integer datapath.

### FPGA deployment picture after this session

Fabric budget (ops_report / ops_report_binary): the whole hot path is
~2.6 MMAC-equiv per keyframe (0.05 GMAC/s at 20 Hz — a fraction of one DSP
array; at the 8-bit knee these are LUT-adds, no DSPs needed); the map at the
lean config is 75–110 KB for building-scale logs (fits small-FPGA BRAM with
headroom). The honest per-regime accuracy statement is the band table above.

**MIT capstone (1.9 km, range-identity eval — the MIT gfs timestamps are
corrupt uint32 garbage, so the naive timestamp eval NaNs; convention as
ssp_scancontext/belgioioso):** raw odometry 187.8; shipped 57.38 (77 loops —
matching the historical closure count — 13.5 MB c64 map); E2 57.26 (92
loops); **FPGA-lean 58.08 at 625 KB** — band-equal accuracy (documented MIT
chaos band 38–58 m) at **22× less map memory** over 1.9 km of corridor. The
O(area) bound and the quantized store compose at scale. (Note the eval
convention differs from the historical 42.66 headline; within-table
comparisons are like-for-like.)

**Webvis:** slot-2 replay embedded — the demo's Intel player now offers
"shipped replay (Python)" (2.440) and "FPGA replay: point+6b store+int8"
(3.902, its in-band draw; 120 loops, 43 snapshots) side by side, both
exported from the real Python pipelines (`export_replay.py --config=fpga8
--slot=2 --embed`; page 10.6 MB).
