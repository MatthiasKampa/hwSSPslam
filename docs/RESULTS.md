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

Seed-firming (4 extra ε=1e-3 seeds per config on intel, n≈7 total): shipped
draws 2.78–6.43 (median ≈3.9; one draw approaches the knife-edge), E2 draws
4.14–9.72 (median ≈5.3, incl. a 9.7 tail) — the intel verdicts (shipped
sampling preferred; FPGA8 band-indistinguishable) stand on the larger
sample.

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
WEIGHTING and/or content selection, not by where the samples sit (audit
caveat: E3's rmse matches shipped but its median improves 22% — positions
are not literally nothing, they just don't touch the rmse-dominating
failure). **E4** (r·dθ weights WITH a filter) initially suggested a 50/50
split (11.6 → 6.02 → 2.75) — **the follow-up audit REFUTED that attribution
as worded**: the E4 probe's isolation rule silently deleted depth-jump
SILHOUETTE hits (hits whose neighbor chords all exceed MAX_CHORD — 0.45% of
fr079's hits) that the shipped pipeline always keeps as lone hits; a
faithful arm (E4a: dr/tang occlusion deletion only) scores **2.92 ≈ E2's
2.75**. Corrected decomposition: **the point-regime win = arc-length r·dθ
weighting + retaining isolated depth-jump silhouette hits; the dr/tang
occlusion masking itself is nearly FREE (±0.2 m) under point sampling.**
Those ~0.45% silhouette points carrying a 2× frontend factor is itself a
sharp finding (they are the translation-informative payload). The
chord-regime filter harm is carried by the separate occ-off experiment
below, which the audit confirmed exactly. FINDINGS §6 gains the addendum: the
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

## 2026-07-10 (late) — decision-stability thread: the perturbation band is a FRONTEND phenomenon; the occlusion filter is a per-environment tradeoff

`ssp_stablegate.py` (subclass-only; jit_k=0 asserted bit-exact to parent).
Motivation: if the band's decision flips could be made perturbation-stable,
quantization would become attributably free on every log.

### v1 — loop-gate jitter-consensus: NULL (and diagnostic)
Re-match every loop candidate against K=2 jittered old-map bundles (the exact
noise channel of the band study); reject jitter-unstable candidates. Result:
at ε=0 and 1e-6 the gate binds NEVER (bit-identical to shipped: intel
2.440/80, fr079 5.523/32, fr101 1.881/53); on a full intel run at ε=1e-3 the
counter shows **0 rejections** (seed 7, results bit-identical to no-gate).
fr079's gated band [5.52..12.41] = the ungated band exactly. The one apparent
improvement (intel s1 6.72→3.88) is a re-roll from a rare firing, not a
systematic clip. **The loop-candidate matched peaks are already jitter-stable
— the flips are not there.**

### Channel localization: the band enters ENTIRELY through the frontend
Inject 1e-3 relative noise into ONLY one bundle-read channel (intel):

    noise ONLY in loop-candidate reads:  ATE 2.440  med 1.553  loops 80
                                         [veto 51 innov 26] — EXACTLY shipped
    noise ONLY in frontend local-bundle: ATE 5.145  med 3.044  loops 91
                                         [veto 97 innov 24] — the full band

The closure layer (candidate match + coherence tiers + innovation + IRLS/LOO)
is fully robust to 1e-3 map noise. The band is produced by the frontend's
reads alone.

### Amplifier hunt: NOT the session-relative calibration
Pinning coh_ref to the unperturbed run's steady value (0.3851) during an
ε=1e-3 run does NOT collapse the band (4.668, loops 112) — the
coherence-calibration feedback (veto 97 above) is a symptom, not the
amplifier. By elimination the driver is the **est-chain divergence of the
frontend's own discrete layer** (per-keyframe accept-gate / coarse-argmax
flips integrating over ~6k keyframes; the audit's continuous-channel estimate
predicts only ~cm at 1e-3). v2 below tests exactly that layer.

### v2 — frontend jitter-consensus: the band COLLAPSES — onto its median
`ConsensusMatcher`: after each frontend match, re-match against K=2 jittered
local bundles; if the matched pose is unstable beyond (5 cm, 0.5°), decline
the match (odometry fallback for that keyframe). Intel band:

    front-consensus: [3.84 .. 4.15] median 3.87   (width 0.31)
    no gate:         [2.44 .. 6.72] median 3.17   (width 4.28)

The band narrows 14x — but re-centers at the band median instead of keeping
the knife-edge 2.44: the unperturbed optimum is BUILT FROM frontend matches
that do not survive 1e-3 jitter, so a perturbation-consistent system
converges to its noise-robust performance level. Interpretation for
deployment: front-consensus buys PREDICTABILITY (a hardware/quantized system
is guaranteed a perturbation anyway — a tight [3.84..4.15] spec beats an
unpredictable [3.5..5.5] draw) at the price of the fragile upside. It is a
choice, not a free win; not shipped by default.

### The occlusion filter is a per-environment tradeoff (frontend-only ATEs)
Disabling the day-1 occlusion filter (runtime `S.OCC_RATIO = inf`; shipped
segment sampling otherwise):

    fr079  11.62 → 4.83   aces  6.38 → 2.40 (!)   intel  5.07 → 6.99 (worse)

On the furnished-but-clean logs the filter's weight-zeroing near depth
discontinuities deletes the most translation-informative content (aces
occ-off even beats E2's 4.33 and raw odometry 5.41 at the frontend level);
on cluttered sparse-beam intel the phantom-bridge protection is genuinely
needed. This completes the E2 per-log pattern with a mechanism: E2 (which has
no bridges to protect) wins exactly where the filter hurts and loses where it
protects. Full-system occ-off on fr079 lands in-band (8.31; the closure layer
reshuffles), so this is a frontend-level finding — E2 remains the deployable
form of it.

### Divergence trace (the microscope pass): the band's FIRST FLIP is the closure edge's chain-median ATTRIBUTION
`scratch_divtrace.py` — run ε=0 and ε=1e-3 (the 4.168 draw) side by side on
intel, log per-keyframe poses + observable decisions, find the first
meaningful deltas:

    |d est| > 1 µm  : kf 6          (float noise)
    |d est| > 1 cm  : kf 250        (still only 1 cm!)
    ... d stays 0.1–0.4 mm through kf 3000 ...
    FIRST PERSISTENT/STRUCTURAL FLIP: kf 3424 — a DIFFERENT loop edge:
        A: (240 → 684)    B: (241 → 684)
    fallback flip kf 3547, first relax delta kf 3575, |d|>0.1 m kf 3575,
    |d|>1 m kf 3645; thereafter the loop-edge sets diverge wholesale
    (A 86 / B 101 insertions, only 36 common).

**The divergence is not integration — it is one discrete event.** For 3,400
keyframes the perturbed run tracks at sub-mm (the continuous response,
exactly as the audit predicted). The first flip is in `try_constraint`'s
edge ATTRIBUTION — `c_aid = chain[len(chain)//2]`: sub-mm pose noise flips a
boundary anchor in/out of the 5 m candidate set, shifting the chain median by
one anchor and re-anchoring the (otherwise nearly identical) constraint; the
early-stopped relax's path-dependence then amplifies the ~cm difference over
~200 kf into meters, after which the trajectories present different closure
candidates and the sets scatter. AUDIT (independent re-instrumentation,
identical numbers): the flip is verified to the millimeter — run A has
anchor 237 at 4.9999 m (inside the 5.0 m candidate radius), run B at
5.0003 m (outside). Caveats folded in: a ~1 cm matcher-internal argmax flip
occurs at kf 250 and SELF-HEALS by kf 300 (an unrecorded decision class that
fires early and harmlessly — reinforcing the v2 front-consensus reading),
so kf 3424 is the first PERSISTENT flip among recorded observables; the
fallback proxy is blind at relax keyframes (9 such, identical in both runs,
immaterial); insertion tallies are approximate at prune keyframes. The matcher, the coherence tiers, and the
innovation gate were all innocent — which is why both jitter-consensus gates
were no-ops. Targeted fix under test (`scratch_attrib.py`): attribute the
edge to the chain anchor SPATIALLY NEAREST the matched pose (invariant to
membership flips at the far radius boundary); gate = eps-0 points hold
multi-log AND the intel band tightens.

### Stable attribution (the targeted fix): the knife-edge is NOT recoverable by local rule-hardening — thread conclusion
Nearest-anchor attribution (the one-line fix the trace suggested): fr101
1.879 (≈shipped), aces 5.954 / fr079 7.03 (in their bands), but intel's ε=0
point itself moves to 3.951 and the band becomes [3.95 .. 5.75] med 4.51 —
narrower than shipped's [2.44 .. 6.72] but WITHOUT the 2.44 draw. Any rule
change that decides differently is a fresh draw from the band.

**Thread conclusion (four interventions, one trace).** The intel 2.440 is an
unstable fixed point whose basin has ~measure zero under any perturbation OR
local rule change: loop-gate jitter-consensus (no-op, 0–1 firings), coh_ref
pinning (no collapse), frontend jitter-consensus (band collapses 14× onto
its median 3.87, losing the fragile upside), nearest-anchor attribution
(relocates the draw). The divergence trace shows why: the system holds
sub-mm for thousands of keyframes, then ONE boundary decision flips (chain-
median attribution at kf 3424 in the traced pair) and the early-stopped
relax's path-dependence amplifies it; hardening that boundary just promotes
the next one. Deployment guidance unchanged and now fully grounded: report
bands (PROTOCOL §6); where determinism matters, frontend jitter-consensus
buys a 14×-tighter predictable band at the band-median accuracy level.
Practical corollary for the webvis report ("irrecoverable divergence at loop
closures on real data"): the live-JS port stacks three of tonight's causes —
an LM/PCG relax (the solver class the ledger rejected for flat-valley
sliding; shipped TRF step control is load-bearing), a no-odometry guess
chain (nothing pulls back a wrong snap), and knife-edge admission — so the
demo now DEFAULTS its real-data tab to the Python-fed replay; the live
pipeline remains selectable as a toy.

### Quantized composition: the algebra survives the FPGA store
`scratch_qcompose.py` (real intel 5-kf segments; integer operators = addr8
cis ROM): transform fidelity (shift+permute of quantized content vs float
pipeline) **6 b: 0.986** / 2 b: 0.800 — with INTEGER operators identical to
float operators at both levels; bundling part-retrievability float 0.930 →
**6 b 0.925 (free)** → 2 b 0.849 (graceful); correction path-independence
(two half-corrections vs one, integer ops) **0.9997** at both levels. The
thesis's algebraic claims — translate = phase multiply, rotate = permute,
compose = add, O(D) graph corrections on frozen content — carry over to the
quantized/integer store: intact at 6 bits, gracefully degraded at 2 bits
(consistent with the e2e band table's per-log verdicts). AUDIT confirmed all
numbers with framing caveats: T1 is best read as STORE fidelity with
error-free operators (the transform — float or addr8-integer — adds ≤4e-4 at
6 b on top of the static quantization cosine, which IS the algebra-survives
claim); the sub-grid δ·segder rotation path is untested here (the e2e band
table carries it); and nmag=1 mid-tread actually emits TWO magnitude levels
(top-coded ≥p99 components, ~1%), so the "2 b/phasor" memory figures are
~1%-of-components optimistic — a hardware clamp to one level is the honest
fix, untested.

### Session capstone (2026-07-10, ~10 h autonomous)
Directive: rework Python-first after the webvis synth hand-tuning broke the
real-data path; keep exploring submaps/loop closure; float groundwork toward
an FPGA SSP-SLAM, then a binary VSA version. Delivered, all audited (two
independent read-only audits; one attribution corrected): `ssp_fpga.py`
(quantized per-ring store / integer front path / point encoding / band
harness / sizing), the perturbation-band methodology (PROTOCOL §6) grounded
in a millimeter-verified divergence trace and closed with four honest
intervention negatives (`ssp_stablegate.py`), E2 point encoding (frontend
win 5/5 logs, band-dominant fr079/aces/belg), the binary verdict (8-bit
arithmetic knee, 2-bit store on robust logs, algebra survives the store),
the FPGA-lean configuration (fr101 75 KB / fr079 110 KB / MIT 625 KB at
band-equal accuracy), and a Python-fed two-source webvis replay (shipped +
FPGA8) that defaults the real-data tab to the true pipeline. The corridor /
verification walls were not re-litigated; nothing shipped was edited; every
neutralised subclass is bit-exact to its parent.

Audit-gap closures: (a) sub-grid rotation THROUGH the quantized derivative
(permute + δ·segder_q): 6 b median cos 0.9858 — identical to the plain
transform fidelity, so the derivative path adds no extra quantization damage
(2 b: 0.795, consistent); (b) TRUE one-level 2-bit store (hardware clamp, no
mid-tread top-coding): fr101 e2e ATE 1.820 (med 0.851, 52 loops, 76 KB) —
inside the lean band; the "2 b/phasor" memory figures survive the honest
encoding.

### Silhouette emphasis: an intel-specific frontend lever (from the audit's E4a insight)
Adding the depth-jump silhouette hits a SECOND time at r·dθ weight on top of
shipped chord sampling (~2.4× silhouette up-weighting): intel frontend-only
**5.07 → 3.109** — the best intel frontend of the session (better than E2's
4.39) — while fr101 3.16→3.61, aces 6.38→6.92, fr079 11.6→13.0 (neutral to
worse: those logs want the full r·dθ reweighting instead). Full-system on
intel: 4.41 / 3.86 / 6.52 across ε — in the band, so like every intel
frontend gain (cf. the N_ANG=113 precedent) it does NOT deploy to ATE.
Completed sampling map per log: fr079/aces/belg (+fr101 median) → E2;
intel frontend → silhouette-emphasized chords (frontend-level only). The
sampling recipe is a per-environment knob, exactly like the occlusion
filter it modulates.

## 2026-07-10 (later) — multi-bin phase-shifted binary + dithered fractional encoding (user's scheme): tested, decomposed, not adopted

The user's proposal: per lattice site, M binary bins with structured phase
shifts (a vernier, M≥2–3), dithering between the two nearest candidates so
the expectation carries the fractional phase. `scratch_ditherbin.py`
(deterministic splitmix dither — LFSR-equivalent in hardware) +
`scratch_ditherbin_e2e/band` runs. The scheme decomposes into two ideas with
different fates:

- **Dither (zero-mean quantization error).** Real and measurable exactly
  where its theory predicts: the coarsest single-bin ENCODE, where the error
  pool averages over ~200 points (1-bin QPSK encode fidelity 0.9662 →
  0.9779). Everywhere else round-to-nearest wins statics (store fidelity
  2-bin 0.974 vs 0.894; registration 0.0038 vs 0.0070) because dither's
  added variance outweighs the bias it removes once codes are ≥4 bit. E2E:
  the dithered-QPSK 2-bit store drew a striking fr101 1.205 — but the band
  probe (PROTOCOL §6) lands it at [1.20 .. 3.19] median 1.81, statistically
  indistinguishable from round-to-nearest 2-bit ([1.13 .. 3.08]; one-level
  1.82); fr079 lean-with-dither shows no transfer win (7.94/6.16 vs
  3.15/6.25). VERDICT: keep round-to-nearest for the store; dither is the
  right tool only if an ENCODE path is ever forced to 1-bin QPSK (it
  de-biases exactly there).
- **The vernier (M phase-shifted QPSK bins).** At matched bits/site the flat
  code always wins fidelity — effective step 2π/(4M) vs 2π/4^M (2×QPSK
  0.9962 < 16-PSK 0.9993 at 4 b; 0/π binary bins far worse: 0.92–0.96
  encode, 0.57–0.70 store). E2E on fr101 the vernier variants land 2.1–2.3
  (the log's coarse-loves-it pattern). Its genuine merit is HARDWARE SHAPE —
  per-bin sign ops instead of a phase LUT — which this design does not need
  (the audited knee is a trivial 256-entry cis ROM); it becomes relevant on
  LUT-free substrates (in-memory/analog computing). Noted in
  SotA/fpga_design.md territory as a fallback primitive, not adopted.

### Webvis: the two winning recipes integrated as replay sources
Player generalized to three self-contained replay slots (each carries its own
ranges/odometry/reference, so different logs coexist): slot 1 = shipped
reference (intel, 2.440), slot 2 = **float winner: E2 point encoding**
(fr079, 2.210, 66 loops), slot 3 = **binary winner: FPGA-lean, 2-bit ring
store + int8 arithmetic** (fr079, 3.149, 81 loops — vs shipped-fr079's
[5.5..12.4] band). Both exports reproduce the banked band-table numbers
exactly; jsc parity harness verifies all three slots (loops 80/80, 66/66,
81/81; RMSE within the known kf-vs-ref matching-direction offset). The old
slot-2 fpga8-intel replay is superseded (lives in git history). Page
13.3 MB, local-file.

### Webvis sandbox: binary FPGA mode (live)
The synth toy env gains a `binary FPGA` toggle implementing the audited
operating point live in JS: all trig (encode, shift, matcher banks) through a
256-entry × 7-bit cis ROM (mkTables rebuilds the shared table; matcher banks
rebaked on toggle), and the matcher + decode panel see a 2-bit phase-only
per-ring quantized VIEW of the map (the running sandbox map remains the
full-precision accumulation buffer — the sandbox has no segment freeze;
captioned). Combine with the existing point-encoding toggle for the full
FPGA-lean recipe. Intel-live (if selected) follows the mode explicitly;
replay slots are precomputed and immune. jsc functional test: per-ring
unit-mag view PASS, binary-mode registration on a synthetic room 0.00 cm /
0.00° PASS, ROM == addr8/val7 spec PASS.

## 2026-07-10 (v3) — bundle-frame smooth closure factors (BFC): CLEAN NEGATIVE, and the band's final localization

`ssp_bfc.py` (use_bfc=False bit-exact to parent). The ultrathink candidate:
replace single-anchor attribution + hard membership with smooth-membership
tapered ([4.0,5.5] m) per-anchor peak-contribution factor groups
(information-preserving split of the matched pose) + weighted innovation
chi². Physically the "edge the measurement actually made"; kills the traced
kf-3424 attribution flip class by construction. FALSIFIABLE PREDICTION: the
intel 1e-3 band collapses toward the ~cm continuous response.

    eps0: fr101 1.877 (holds) / fr079 10.31 (high draw) / aces 7.33 (band)
          / intel 4.17 (band); loop-edge counts inflate by design (weighted
          groups: intel 862 edges)
    intel band: [3.20 .. 6.78] median 4.13 — SAME WIDTH as shipped
    and eps=1e-6 ≠ eps=0 (4.092 vs 4.169) where shipped was stable to 1e-5:
    the group machinery (w-min cut, top-k, chain choice) adds boundaries.

**Verdict: REFUTED as a band fix — and the refutation completes the band's
localization.** Combining with the channel study (loop-side noise moves
nothing; frontend-side noise reproduces the band) and nearest-attribution
(hardening one boundary promotes the next): the band is a property of the
RECURRENT FRONTEND ESTIMATOR (est[k] feeds guess[k+1] through per-keyframe
argmax + accept gates), not of the closure formulation. Any loop-layer
reshaping — smooth or hard — is structurally incapable of removing it,
because the divergence enters upstream and will cross SOME structural
boundary downstream. Making the recurrence itself C¹ would require softmax-
pose blending (= the closed belief-frontend negative); the working levers
are consensus at the source (frontend jitter-consensus: band 14× tighter at
band-median accuracy) or hypothesis envelopes (report the band online). P2
(decision fragility) is hereby CHARACTERIZED-AND-BOUNDED: smooth admission
refuted as insufficient (BFC), frontend recurrence identified as the locus,
consensus/envelope as the deployable mitigations. P1 (twin verification)
unchanged — the isometry argument stands; embedding reshaping wins live on
the retrieval/memory/smoothness axes only.

## 2026-07-10 — better benchmarking datasets: the first independent-class reference (MIT Stata), suite growth, and the RAWSEEDS path

User directive: find better datasets. The ledger's highest-value gap
(FINDINGS §9 caveat 2): every ATE is agreement-with-GMapping. Targets, by
value: independent GT > second reference family > same-family diversity.

### MIT Stata Center — the caveat-2 counter-evidence (ssp_stata.py, NEW)
The dataset's GT is anchored to AS-BUILT FLOORPLANS (160-scan batches aligned
to the plan at both ends, iSAM-relaxed between; ~2-3 cm claimed) — a
reference class independent of any particle-filter SLAM. Adapter:
`ssp_stata.py` reads the PR2 bag's /base_scan (Hokuyo UTM-30LX, 1040 beams ×
0.25° = 260° FOV — the encoder/matcher are FOV-agnostic; only the CARMEN
harness assumed 180°, and the loader supplies the true beam array) +
/base_odometry composed into the laser frame; scores against the per-scan
`.gt.laser.poses` derivatives. Bag 2012-01-27-07-37-01 (7.1 GB, 6:48 min,
8,142 scans → 1,369 keyframes; GT parts 1+3 = 3,996 poses, 819 matched
keyframes):

    raw odometry   ATE 3.214  med 2.086  max  7.68
    shipped        ATE 0.202  med 0.123  max  0.49   (99 loops, 78 ms/kf)
    E2 point       ATE 1.659  (9 loops — closures collapse)
    FPGA-lean      ATE 2.122  (11 loops)
    shipped ε=1e-3 (2 seeds): 0.202 / 0.203 — PERTURBATION-STABLE
                              (fr101-class; dense closures = robustness)

**Headline: against floorplan-anchored truth the shipped system is a
0.202 m / 0.123 m-median system, and the number is perturbation-stable.**
The agreement-with-GMapping caveat now has counter-evidence on an
independent-class reference. Per-regime note: E2 REGRESSES here (1040-beam
Hokuyo; loop count collapses 99→9) — with fhw below, the sampling verdict
is confirmed per-log, not per-beam-count: shipped sampling for intel+stata,
E2 for fr079/aces/belg/fhw(+fr101 median). Scope caveats: one 6:48-min
session; GT covers 819/1369 keyframes; base→laser offset used PR2 nominal
0.275 m (not in this bag's /tf); GT is floorplan-anchored but
scanmatch-interpolated between anchors (weakest in long corridors, per its
authors). Data + gdown recipe in data/fetch_datasets.sh; parsed-bag cache
(.npz) written beside the bag.

### Suite growth (zero-adapter Radish logs)
- **fhw** (large open exhibition hall, 38,613 scans → 7,035 kf, dense
  483-pose ref, clean timestamps): shipped **0.981** / E2 **0.350 (med
  0.233)** / lean 1.108 @ 206 KB — E2's best absolute result in the repo,
  on a 180-beam log (further burying the beam-count story).
- **orebro**: DROPPED — frontend never engages (ATE 18.29 identical across
  shipped/E2/lean, 1-2 loops; beam/FOV mismatch with the FLASER driver
  assumptions). Noted in fetch script.

### RAWSEEDS Bicocca — located, blocked at one manual click
The genuinely independent camera-network GT + **published GMapping baseline
vs that GT (ATE 2.04 ± 1.87 on Bicocca_2009-02-25b)** — a direct
same-truth comparison target. rawseeds.org is alive (with per-file
inventories and baseline solutions); the files now live in AIRLab's Dropbox
folder, whose per-file addressing defeated headless access (subpath 200s are
interstitials; legacy list endpoint 403; no mirrors; no archive.org copy).
One manual download of Bicocca_2009-02-25b-{SICK_FRONT,ODOMETRY_XYT,
GROUNDTRUTH}.csv.bz2 unblocks it — the adapter is a small CSV loader away
(180° SICK = zero FOV work). Deutsches Museum (Cartographer bag, 493 MB,
verified reachable) remains the second-reference-family option.

## 2026-07-10 — per-dataset lattice/recipe study, hex-vs-polar on real data, the no-relo memory win, and the λ×bits co-design (all labeled per-dataset; PROTOCOL §6)

### φ-prime ray-count recipe (user's synth-sandbox winner): REFUTED on real logs
N_ANG = rays·(φ−1) rounded to the next prime — 113 (180 rays), 223 (360),
643 (1040) — loses everywhere e2e: stata 113→1.158 / 643→1.190 (vs 60:
0.202, at 16× compute); fhw 113→3.081 (vs 0.981); fr079 223→12.871 (vs
5.523); belg 113→4.353 (vs 2.644); intel 113→4.588 (band-worse). Full
ladders on the new envs put shipped-60 at the optimum with the documented
even-count mechanism confirmed twice more (60→61: stata 0.202→0.772 loops
99→19; fhw 0.981→2.748) — odd/prime counts miss the 90° wall normals.
Another instance of the synth→real gap: the sandbox recipe was a
crisp-walls artifact.

### Hex SSP vs vanilla polar (first real-data e2e; ssp_hexreal.py)
Full-circle triplet lattices (plain-shift permutation, no conjugate wrap;
self-tested exact): intel keeps polar-60 (hex57/63/111 → 4.8/5.7/4.7, in
band). **belg (non-Manhattan castle): hex63 → 2.071 vs shipped 2.644 (med
1.28 vs 1.86)** — the synthetic curved-world prediction confirmed on real
data for the first time; hex57 (3×19 prime) 3.52 and hex111 7.53 lose
(belg's 1-9 closure counts make deltas noisy; the prime-base claim actually
INVERTED here — composite 3×21 won). Verdict: hex is a per-environment
option for non-Manhattan deployments, not a default. Webvis integration
(pushed 6daccc1): sandbox angle-count selector + a validated non-Manhattan
"castle" world so the regime split can be demonstrated live.

### NO-RELO ablation — a free 33% of the map (component-inventory sleeper)
The λ 5.3/12.8 relocalization rings are DEAD WEIGHT in the bounded
deliverable: the matcher uses MAIN, the veto uses the finest two rings, and
the drought/relo consumer lives only in the ssp_hier extension. Dropping
them (D 360→240): **fr101 1.881 / intel 2.440 / fr079 5.523 — identical to
shipped to the last digit, identical loop counts — at −33% map memory**
(fr101 1.9→1.26 MB; binary segment 204→136 B; MIT-scale ~625→~417 KB).
Keep the rings only if the drought/global-relocalization extension is
wanted; drop for the FPGA deployment.

### Scale ladder × store bits (binary co-design; scratch_binscale.py)
The "binary needs finer λ_min" hypothesis resolves as PER-SENSOR, twice
inverted: fr101 — finer {0.125..1} is the best FLOAT config ever drawn there
(0.797! labeled, unbanked, config-wobble applies) but LOSES at 2 b (2.19 vs
1.41: sensor phase noise ±50° + code ±45° pushes the finest ring past the
decision boundary — coarse codes need per-component SNR headroom); stata —
finer HELPS at 2 b (3.33 vs 4.89) but hurts float (1.06 vs 0.202), and
nothing rescues 2 b there (closures 4-19 vs 99: a closure-redundancy limit,
not resolution). Store-tier verdict per log stands: 2 b for
closure-redundant fr101/fr079-class, 6 b tier for stata/intel-class; λ
ladder stays {0.25..2}; 5-ring ladders need the try_constraint 4-ring
Hessian slice generalized (deferred).

## 2026-07-10 — consolidation: dataset registry + lattice module, repo cleanup, 9-replay webvis

Directive: "read all code and results, clean up, integrate other datasets
into webvis, rework both python side and webvis, continue experiments."

### Infra consolidation (committed modules)
`ssp_datasets.py` — ONE registry + run/eval harness for all eight datasets
(intel/fr079/fr101/aces/fhw gfs-eval; belg/mit range-identity; stata
floorplan-GT), replacing the three parallel copies of the run loop
(ssp_fpga.run_log, ssp_stata.run, scratch_belgioioso/scratch_fpga_mit2).
Acceptance: `selftest` asserts the registry loop == run_log BIT-EXACT
(fr101[:1200], max|Δpose| 0); banked-number reproduction belg 2.644 /
fhw 0.981 / stata 0.202 — all three eval families exact. `ssp_lattice.py` —
set_polar (any ladder/ring count)/set_hex/shipped()-restore with an exact
shipped-globals selftest, consolidating the three per-scratch lattice
patch copies. `ssp_hexreal.py` rewired onto both (verbatim-run equivalence:
its runners now delegate to the registry loop, which is proven bit-exact);
its scratch_belgioioso import is gone (range-identity eval promoted into
the registry).

### Repo cleanup
16 committed root figures → `archive/plots/`; the two tracked scratch
scripts + ~200 untracked scratch files/logs → `archive/scratch/` (READMEs
in both; RESULTS references cite the same file names); regenerable root
`*.npz` → `archive/local/`. Repo root now holds docs + committed modules +
the active thread's scratch only.

### Webvis rework — replay pack + 9 datasets (the real-data side)
`demo/export_replay.py` rewritten registry-driven: any dataset ×
{shipped, e2, lean, fpga8, binary, hex63}; per-dataset scoring reference
travels with the blob (gfs / range-identity / floorplan GT + provenance
string); display beams strided to ≤260 (poses computed on full scans);
output → `demo/replays/` + manifest; `--embed` splices ONE lazy-decoding
REPLAY_PACK (replaces the 3 numbered slots). index.html: dataset/config
selector built from the pack, drag-drop/Load-replay for sidecar JSONs,
"Real data" tab naming, pack self-tests. Embedded set (ATE, all
reproduced by the exporter run itself):

| replay | ATE m | note |
|---|---|---|
| intel shipped | 2.440 | float64 est chaining preserved |
| fr101 shipped | 1.881 | |
| fr101 FPGA-lean | 1.132 | 2b+int8 @ 75 KB — this draw lands at the band floor [1.13..3.08]; binary BEATS float here |
| fr079 E2 | 2.210 | float winner |
| fr079 FPGA-lean | 3.149 | |
| belg shipped | 2.644 | range-identity eval in-blob |
| belg hex63 | 2.071 | the non-Manhattan hex win, now live |
| stata shipped | 0.202 | independent floorplan GT in-blob |
| fhw shipped | 0.981 | 7035 kf |

Verification: jsc parity harness decodes all 9 embedded blobs and walks the
JS replay shim over every keyframe — max |shim − Python fin| ≤ 4e-6 m
(fhw 3.2e-4, float32 snapshot accumulation over 157 relaxes; 3 orders below
drawing resolution). Full-page JS re-run through jsc (core executes to the
headless guard; UI parses); 46 referenced element ids all defined.
index.html self-contained at 26.2 MB.

### Loop-candidate radius sweep {4, 6, 8} vs shipped 5.0 (scratch_radius.py)
The divergence trace located the band's first flip AT this constant's
boundary; the value itself was never swept. RadiusBand = verbatim copied
try_constraint with the one literal lifted (cand_rad=5.0 asserted bit-exact
to parent). eps=0: intel 4.71 / 4.65 / 3.40 and fr079 10.38 / 8.57 / 10.29
for rad 4/6/8 — ALL far worse than shipped (2.440 / 5.523); fr101 alone
prefers 6.0 (1.648 med 1.005, +38 loops) but the gain does not transfer.
Bands: fr079 medians are RADIUS-INVARIANT (8.37 / 8.91 / 8.57 / 8.30 for
5/4/6/8 — the radius moves the eps=0 draw, not the distribution; the
shipped point 5.52 sits at its band floor). intel: rad 8 NARROWS the band
([3.37..4.86] vs [2.44..6.72]) at the cost of the floor — consistent with
the chain-membership-flip mechanism (a farther boundary crosses sparser
anchor shells), but its median 3.40 does not beat shipped 3.17. VERDICT:
5.0 stands (best floor + best median on 2/3 logs); radius is not a
cross-log tunable; the fr101-only 6.0 gain is the usual per-log trap.

### Bench redirect (user call, mid-burst): intel REMOVED; target = SPOT/360°
User: intel "doesn't work well in general (diverges) and the ground truth
seems off"; deployment target is a SPOT with a 360° lidar (intel is 180°);
"you can just remove that bench" — and the synthetic env joins the suite.
Actions: acceptance suite = stata (260° FOV, floorplan GT — nearest the
target) / fr101 / fhw / fr079(bands) / belg + the synthetic 360° bench;
intel dropped from EXPERIMENTS/PROTOCOL acceptance language (loader kept in
ssp_datasets only for historical ledger reproducibility — every pre-existing
intel number in this file stays as history). Webvis: intel replay dropped
from the pack (8 replays, stata first) and the LIVE-JS intel pipeline mode
removed entirely (blob + loadIntel + live option + cDer control + the
core-parity soak sub-check; the sandbox is the interactive toy; replays are
the real-data view). demo/export_data.py → archive/. index.html 26.2 → 20.0
MB; jsc + 8-replay parity re-verified. Consistent with this suite's own
findings: intel was the knife-edge log AND the agreement-with-GMapping
caveat's worst case.

## 2026-07-10 — encoder sampling study: point vs segment-integral vs multi-sub-point, the cutoff-angle gate, and blanking-vs-weighting (user thread; ssp_sampling.py + ssp_synth.py)

User directive: compare per-beam point encoding, LINE-SEGMENT-INTEGRAL
encoding, and n-sub-point interpolation of the segment; sweep the
interpolation gate (cutoff angle); test whether blanked fine-scale energy
should be re-weighted onto coarse scales harder than the linear r·dθ.
Mid-thread the target platform landed (SPOT, 360° × 1024 beams @ 20 Hz) and
the suite was redirected (intel removed; synth-360 added).

**Machinery** (committed fd49069/40c427a): `sample_interp(n_sub, cut_deg,
w_mode)` — the shipped occlusion test dr ≤ 2·tang IS a 63.4° cutoff angle,
now parameterized; non-bridged hits stay E2 points. `SegIntEncoder` — the
EXACT integral ∫e^{ik·x} along a chord = sinc(k·d/2)·e^{ik·c}: fine rings
with |k·d|≫1 are suppressed automatically at full coarse-ring weight, i.e.
THE SANDBOX THERMOMETER IS A HAND-TUNED APPROXIMATION OF THE SEGMENT
INTEGRAL. Endpoint packing [P0;P1] keeps every pipeline transform linear;
SegCore inserts at the BoundedSLAM MRO slot so Quant/Band layers compose.
`pack_group(k)` = decimation-by-integral (fold ≤k bridged hits into one
segment; per-hit footprint partition ⇒ mass-exact vs E2). `renorm_alpha`:
sinc-surviving components scaled ρ^(-α/2), ρ = kept lattice-mass fraction.
Selftests: quadrature (vs 400-pt sub-sampling, 2e-6), analytic dθ derivative
(1e-9), degenerate-pack primitives BIT-exact to the point pipeline (e2e is
ULP-equal only — BLAS rounds duplicated rows position-dependently; measured
first delta 1.8e-15 at kf 62, knife-edge amplified to 2.8e-5; documented).

**The verdict is FOV/density-dependent — one recipe per sensor regime:**

| bench | best | numbers |
|---|---|---|
| synth-mixed 360°×360 | any bridged/integral ≈ E2 ≫ shipped | 1.1–1.3 cm vs shipped 4.8–5.8 |
| synth-corridor 360° | insensitive (aperture-limited) | all 1.7–2.5 cm |
| synth FOV ladder | E2-vs-shipped RANKING FLIPS with FOV | 180°: 6.2 vs 5.9 cm (tie/lose); 260/360°: 2.6–2.7 vs 4.8 (2× win) |
| SPOT 1024-beam mixed | interp n3 / segint @63.4° | 1.1 cm; GROUP/4-8 hold 2.5 cm @ 367/276 terms; shipped resampler 6 cm @ 1636 terms |
| SPOT 1024-beam corridor | all bridged ≈ 1.5 cm; **shipped seg FAILS (1.76 m, 1 loop)** | GROUP/8 1.6 cm @ 410 terms; decim/8 degrades (3.2 cm) |
| stata 1040 beams (proxy) | **bridged @63.4° = 0.196 (74 loops), beats shipped 0.202**; raw E2 points COLLAPSE (1.659, 9 loops) | interp n2 ≡ n3 ≡ segint-lone-arc = 0.196; n5 3.43; CHORD-mass 2.58 (arc mass matters) |
| stata gate response | 63.4° is a REAL optimum | 45° → 1.66 (under-bridge); 75° → 8.25 (over-bridge) — keep the constant |
| fhw / fr079 / fr101 (180°) | E2 point stays champion | fhw 0.350 (interp 1.4–2.8 WORSE); fr079 2.21; fr101 1.57 |

**Blanking-vs-weighting (the α question):** at target density it is MOOT —
wall-sample spacing r·dθ < λ_min/2 inside ~20 m at 1024 beams, and synth-360
shows α inert (all 0.011–0.026 regardless). On the SPARSE 180° logs the
footprint-integral (arcint) base improves with α=0.5 on ALL THREE logs
(fr101 3.13→1.61, fr079 5.35→2.43, fhw 1.00→0.744) while α=1.0 overshoots
on all three — i.e. HALF-energy restoration is the right compensation where
blanking bites — but arcint+α0.5 never beats plain E2 points there (fhw
0.744 vs 0.350), so it is a mechanism note, not an adoption. stata: α flat
(1.60/1.48/1.48) — bridging, not weighting, is what the dense head needs.

**Deployment recipe (SPOT 1024×360°):** bridge consecutive returns at the
shipped 63.4° gate with arc mass; encode either as n2/n3 sub-points (zero
new hardware) or as exact integrals; if encode budget matters, GROUP/8
integral decimation is lossless on the bench (410 terms, 1.6 cm) — but the
budget analysis (ops_report(1024) = 6.3 M MAC/kf ≈ 1 DSP48 @ 200 MHz) says
even the full head is cheap. Webvis: `replay_stata_interp2.json` exported
(0.196, in the pack). Design note updated ("Target sensor" +
sampler-mux sections).

**Synth-360 bench** (`ssp_synth.py`, committed): worlds.py geometry, exact
GT, deterministic per-seed rng, ssp_datasets-shaped bundles ('exact' eval
family); shipped/E2 controls land at 1–6 cm — the cm-regime the sandbox
showed, now scriptable and seed-controlled.

### Veto/admission-tier constant sensitivity scan (scratch_vetoscan.py; fr101 + fhw stages, follow-ups on stata/belg/fr079)
Purpose: knife-edge detection over the ~17 one-shot-calibrated admission
constants (which must stay runtime registers vs bake into fabric), NOT
retuning. VetoBand = copied verbatim body with every literal lifted
(all-defaults asserted bit-exact).

**fr101 (point-stable log): 9/17 constants are FLAT-ZERO** —
chi2 {6,13}, allow_t {½,2×}, allow_r {½,2×}, repl {2.5,5}, coh_floor
{0.15,0.28}, coh_infl_cap {20,80}, aniso_min {0.32,0.48}, ridge_max
{0.5,0.7}, ridge_s_max {0.42,0.58}: ATE 1.881 ± ≤0.006 each way. The
calibrated ridge/aniso classifier and the whole innovation-gate cluster are
BAKEABLE. Movers: chain_gap→4 (+1.27), coh_target→0.65 (+1.16, the
documented cliff side), gate_t→0.8 (+0.38), ill_mult→4 (−0.31),
coh_target→0.45 (−0.24), infl_pow→3 (−0.20), gate_t→0.45 (−0.18),
sig_r0→2.8 (−0.17).

**fhw (dense-closure log, 559 loops): hypersensitive with OPPOSITE signs**
on most movers — gate_t→0.8 −0.71 (fr101 +0.38); sig_r0→1.4 −0.75 (fr101
+0.07); infl_pow→3 +0.19 (fr101 −0.20); chain_gap→4 −0.18 (fr101 +1.27).
The one-shot constants sit at a cross-env compromise; no cross-log retune
exists in this cluster. FPGA conclusion: {gate_t, sig_r0, coh_target,
ill_mult, infl_pow, chain_gap} = runtime registers; the other 9 = constants.

**coh_target→0.45, the single ALIGNED mover** (fr101 −0.24, fhw −0.67):
follow-up — stata 0.203 (≈ shipped 0.202, +2 loops), belg 2.644 (identical;
1-loop env has no decision surface), fr079 band [4.33..12.24] median 8.18 vs
shipped [5.52..12.41] median 8.37 — band-equal, floor slightly better.
Suite scorecard for 0.45: fr101 −0.24, fhw −0.67, stata +0.001, belg 0.000,
fr079 band-equal — WEAKLY DOMINANT on the post-redirect suite. The shipped
0.55 was chosen by worst-of-three INCLUDING the now-removed intel log, so
the calculus has genuinely changed. Caveats before adoption: the hypothesis
came from the scan itself (mild selection effect; the stata/belg/fr079
follow-up was the independent check and came back neutral-not-better), and
flipping the shipped default re-baselines every table. DECISION PREPARED,
not executed: recommend coh_target 0.45 as the new default under the
SPOT-era suite; awaiting user sign-off. (sig_r0 2.8, the weak second
candidate, is fr101-only — stata 0.208 vs 0.202 — dropped.)

## 2026-07-10 — global readout: bundle capacity × scale ladder × bits (user thread; scratch_capacity*.py, scratch_fullread*.py)

Question (user): how many vectors does the (binary) store need for a
global/full-map readout to work — sweeping both bundle size K and the
scale ladder ("large enough scales to not alias").

**Protocol.** Fixture = shipped-float run (stata 0.196, 229 live segments;
fr101 control 1.569/342); stores REBUILT per ladder from fixture poses
(readout isolated from pipeline dynamics); bundles = K nearest segments;
probes registered against the bundle. Arms: bits {float, 6 b, 2 b per-ring
codes}; ladders L4 {0.25..2} … L7 {+5.3, 12.8, 25.6} all-matched; readouts
LOCAL (guess truth⊕0.25 m, ±0.72 m window), REGION (rotation fenced,
coarse-ring stage-1 over the bundle extent), DESCENT (per-ring
phase-unwrapping, coarse→fine). Plus e2e full/global-read pipeline runs.

**1. Bundle capacity (local-prior readout, shipped ladder).** stata:
K* = 64 (float) = 64 (6 b) = 32 (2 b) under the strict criterion (med
< 0.10 m AND p90 < 0.30 m) — the 2-bit code-noise floor costs exactly one
octave of superposition capacity. Degradation is graceful (median only
1.3→6 cm at K = 128; the p90 tail fails first). fr101 reads K* ≈ 8–16 on
all tiers, but its fixture is only 1.57-ATE accurate — treat stata as the
credible curve. → A 229-segment building needs ≥ 4 readout vectors at
float/6 b, ≥ 8 at 2 b; **reads of ≤ 32 segments are safe even at 2 bits.**

**2. Scales: widening the matched band is NEGATIVE (dilution).** L4
dominates every K and both readouts; adding rings monotonically hurts
(L7 LOCAL med 0.7–0.9 m at K ≥ 128 — the coarse components add crosstalk
energy but no in-window discrimination; this re-derives why the original
design kept relo rings OUT of the matched band). Region-readout at K ≤ 32
already works on L4 alone (succ 0.88–0.96 — content correlation
disambiguates within a few-metre region despite λ-periodicity).

**3. Hierarchical descent over coarse rings FAILS at building scale.**
Ring-pair phase-unwrapping (λ/8 steps, shrinking windows, fine matcher
last): K = 32 succ 0.80–0.84 (worse than plain L4 region search), K = 128
succ ≤ 0.36, K = 229 succ ≤ 0.20 with 6–17 m medians — the coarse-ring
argmax picks the wrong part of the building. Large-λ rings cannot buy
global readout from appearance-only content: the corridor/place-SNR wall
again, now measured at the readout layer. 2-bit ≈ float throughout.

**4. e2e full/global reads.** Float full-read (r = 8) breaks 4/5 logs
(stata 0.196→9.46, fhw 0.35→3.16, fr079 2.21→11.6, belg 2.64→3.51) —
architectural law 2 stands — EXCEPT fr101, which FLIPS: full-read float
0.396 (vs 1.569 local; best fr101 number ever recorded) and even truly
global reads win (2-bit global 0.885). Synth (cm drift): benign. The law
is really about drift-consistency of old content: dense-revisit maps keep
it graph-tight; sparse-revisit maps poison the frontend with it.

**5. The bits interaction (open mechanism).** stata r = 8 full-read is
monotone in store coarseness: float 9.46 (14 loops) / 6 b 4.42 (24) /
**2 b 0.181 (77 loops — beats the local float reference 0.196)**. The
magnitude-clamp explanation is REFUTED (phase-only 256-level/1-mag store
reproduces the float failure, 8.83); the loop counts localize the channel
to the accept-gate/closure layer (float full-read produces
confident-but-wrong frontend matches that pass the gate and kill the
closure layer; 2-bit matches either land or fail the gate cleanly —
"quantization as decision hygiene"). Single-run points; mechanism OPEN.

**ANSWER.** One global vector: NO — at any bit width and any ladder
(capacity + place-SNR walls, measured independently). Radius-grouped
readout vectors of ≤ 32 segments at 2 bits: YES with margin — a
stata-scale building = ~8 such vectors (4 at 6 b/float). The matched band
stays 4 rings; coarse rings remain relocalization-extension material only
(and even there cannot deliver building-scale readout). Dense-revisit
environments (fr101-class) tolerate even ONE global 2-bit vector end to
end (0.885) — an environment-dependent bonus, not a spec. Deploy guidance
unchanged + sharpened: cell-grouped bundle reads, ≤ 32 segments per read.

## 2026-07-10 — ultrathink batch: the deadband mechanism, the corrected global-readout verdict, and FIRST CONTACT with the target platform

### The quantization deadband (mechanism of the 2-bit full-read rescue)
The stata full-read r=8 anomaly (float 9.46 / 6 b 4.42 / 2 b 0.181 —
monotone in store coarseness) decomposes cleanly:

| store | ATE | loops |
|---|---|---|
| float | 9.46 | 14 |
| phase-only (256 ph, 1 mag) | 8.83 | 11 |
| snap-only (4 ph, 64 mag) | 3.51 | 18 |
| 8 ph, 1 mag | 2.27 | 22 |
| **4 ph, 1 mag (2-bit)** | **0.181** | **77** |
| 2 ph, 1 mag (BPSK) | 1.51 | 20 |

Magnitude equalization alone does nothing; phase snapping alone recovers
part; TOGETHER they rescue fully — and the response is NON-monotone in
nph with the optimum at exactly 4 levels. Interpretation (the DEADBAND
theory): coarse phase codes snap drift-displaced copies of the same
geometry to the SAME code when the per-ring phase offset stays inside the
quantization cell — per-ring forgiveness ≈ λ/(2·nph). At nph=4 that is
λ/8 (25 cm on the 2 m ring ≈ stata's inter-pass drift): mid rings FUSE
the passes at the graph-mean pose while fine rings decohere the drifted
copies into noise, and the one-level magnitude weights fused content
uniformly against that noise. BPSK over-forgives (λ/4) and blurs the
current pass too. The loop counts localize the e2e effect at the accept
gate (float full-read passes confident-wrong matches that kill the
closure layer; snapped stores either land or fail cleanly). Band-checked:
stata r8+2 b = [0.16..0.18] median 0.17 — tighter than local float 0.196.
**The binary store is not a compression compromise; at the right nph it
is a per-scale drift-deadband — a robustness mechanism.** fr101 confirms
the regime logic: tiny drift → float full-read best (0.396, best-ever
fr101); wide reads need the deadband even there (float GLOBAL 3.44 vs 2 b
GLOBAL 0.885).

### Global readout, corrected (user: pick the joint alias outside the env)
The earlier "impossible" verdict conflated three things; the honest
decomposition (scratch_capacity4/5/5b):
- The ×2 matched ladder joint-aliases every λ_top — but the shipped
  incommensurate anchor pair {5.3, 12.8} has its first joint near-alias
  at 63.6 m (computed) — OUTSIDE a stata floor. Aliasing is solvable by
  design, exactly as the user said.
- With a CUMULATIVE multiresolution joint decode (anchor band over the
  extent → +2 m ring → +1 m ring → fine matcher; every level keeps the
  coarser rings), K=32 bundles localize GLOBALLY: L1-hit 0.96, succ
  0.76–0.80, float ≡ 2-bit. The K≥128 failure is PURE superposition SNR
  (240 anchor components vs 229 segments; capacity theory wants
  D_anchor ≳ K·ln #cells ≈ 1900), NOT aliasing.
- Grouped construction (12 cell-vectors ≤32 segs): within-group decode
  works; CROSS-group selection is place recognition and hits the
  documented wall — raw argmax 0.32–0.40 right-group; top-3 + fine-match
  verification lifts the MEDIAN to 0.039–0.040 m with succ 0.56–0.60.
VERDICT (replaces the earlier one): global readout is FEASIBLE at
suitable capacity — reliable given any ±10 m prior (group selection
trivial, then 0.96 within-group), median-accurate but tail-broken (~0.6)
with no prior at all. The prior-free tail is the §5 wall at the readout
layer, not a lattice or bit-width limit. Anchor rings earn a role again:
NOT in the matched band (dilution, measured), as the global-decode band.

### SPOT Telluride — first contact with the target platform (ssp_spot.py)
First dataset from the user's platform (HF
lorinachey/spot-telluride-workshop-dataset): Ouster-class 1024×64 clouds
@ 20 Hz (the custom-head spec), SPOT kinematic odometry @ 528 Hz, LIO-SAM
trajectory; 78 s / 36.5 m / ~7×7 m room. Adapter: ring 34 (elevation
0.00°) → 1024-beam 2D scans, npz-cached; keyframes = every Nth cloud
(time-based, input-legal); **protocol per user: lidar-only** (constant-
velocity guesses from own estimates; odometry WITHHELD and used as GT —
"essentially spot on"), eval = 'exact'. Registry: DATASETS["spot"],
guess_mode="cv" in the shared runner (selftest still bit-exact).

| arm (lidar-only, GT = withheld odometry) | ATE | med | loops | map |
|---|---|---|---|---|
| point / bridge2 / seg, oct60, stride 4 | 0.315–0.316 | 0.044–0.047 | 22 | 354 KB |
| **FPGA-lean 2 b + int8 (either sampler)** | **0.315** | **0.045** | 22 | **14 KB** |
| 6-bit store | 0.316 | 0.045 | 22 | 37 KB |
| stride 2 (10 Hz keyframes) | 0.289 | 0.043 | 52 | 461 KB |
| stride 8 (2.5 Hz; gap disables loops) | 0.260 | 0.038 | 0 | 225 KB |
| WITH-ODOM (labeled diagnostic) | 0.041 | 0.035 | 22 | — |
| lattices fine60/oct90/oct36 | 0.317–0.318 | 0.048–0.062 | 22 | — |

Read: **the full binary datapath reproduces float exactly on the target
platform's own data — 4.5 cm median, 25× less map memory (14 KB for the
room)** — and every setting ties within noise (registration-easy env;
the sampler/lattice/search knobs don't bind at 7×7 m). The RMSE tail
(0.32 vs 0.045 median) is guess/start-transient, not registration (the
with-odom diagnostic bounds it at 0.041). Config for the demo: shipped
lattice, bridged pairs, stride 4. Webvis: spot float + FPGA-lean replays
exported into the pack (reference track = the withheld odometry).

### SPOT jumps diagnosed: a GT artifact (unsorted odometry parquet), not the estimator; window hypothesis tested and refuted
User observed "long jumps/jitters" in the spot replay and proposed the
local search space as the cause. Census (worst |est-step − gt-step|
frames): kf 79–82 show the REFERENCE teleporting 3.5–3.8 m away and back
in ~0.6 s while the estimate steps 4–12 cm — and the odometry stream's
file order carries a ~62 s block with out-of-order timestamps (file-order
sample dt +62,614 ms at t=15.8 s, −66,884 ms at t=82.7 s); the adapter's
searchsorted assumed sorted times and pulled poses from the misplaced
block for 3 keyframes. Those 3 frames were the entire RMSE gap
(√(3·3.7²/414) ≈ 0.31 of the 0.315). Fix: sort odometry by timestamp in
make_bundle + a physical-step hygiene mask (>0.5 m per ~0.2 s keyframe →
masked from eval and the webvis reference track; after sorting the mask
fires zero times). Window sweep (t_half 0.48→0.12, rot 9→3°): ALL
IDENTICAL pre-fix (max err 3.79 in every row) — the matcher never used
the extra room; windows can be shrunk freely for compute, they were not
the jump source.

**Post-fix (lidar-only, GT = withheld sorted odometry): float 0.039 /
LEAN 2-bit+int8 0.039 ATE — medians 3.3/3.5 cm, max 12–13 cm, 22 loops,
14 KB binary map.** The lidar-only estimate now agrees with SPOT's own
kinematic odometry at the level of the earlier with-odometry diagnostic
(0.041) — i.e., at the reference's own noise floor. Remaining "jitters"
are the cm-scale frame noise (p99 ~0.10–0.12).

## 2026-07-11 — iCE40 hardware-in-the-loop track: the encoder on silicon (bit-exact), dynamic multi-room synth environments, and the school-aliasing quantification

**Session note (PROTOCOL §8 exception, user call 2026-07-11):** subagents ran
experiments in parallel this session (the dynenv bench grid below; solver /
consensus / lattice studies report separately and will be banked on arrival).
Verification for the banked numbers: the bench grid's longest arm reproduced
BIT-EXACT on an independent rerun (`tenx` == bench row: 0.954/93/0.53/1.52);
the classroom p0 row matches this session's hands-on smoke run; the loop-
precision metric's GT fence was audited read-only (GT enters only as a label
on already-accepted edges, ssp_bounded.py:456–473 — accept logic is
coherence/ridge/innovation, PROTOCOL §2 held).

### iCE40 track opened: target iCEbreaker v1.0e (UP5K), hw-in-the-loop

Toolchain: oss-cad-suite darwin-arm64 (yosys 0.67, nextpnr-ice40 0.10,
icestorm, icarus 14) installed fresh; board verified (flash W25Q128, CDONE
toggles; UART echo-plus-one at 1 Mbaud proved clock/RX/TX/fabric compute).

**Golden model (`ssp_ice40.py`)** implements the design-note v1 integer spec:
position unit λ_min/256 with the mm→unit factor folded into the az-LUT
constants, x/y from a 1024-entry az LUT (uniform head ⇒ beams exactly on
grid), 60 2-MAC angle projections, **ring cis addresses as bit-slices of one
projection word (A1: exact because the octave ladder is ×2)**, cis ROM
256×2×7b, int32 accumulators, i16/i18 envelope asserts, hits masked beyond
31.2 m (i16 bound; mask cost on long corridors unmeasured — flagged).
Fidelity vs float encode on the same masked points: cosine ≥ 0.99999 on
spot scans, synthetic room, and dyn-classroom (min over 100 scans). All
integer, deterministic (bit-equal on repeat).

**Encoder RTL, two generations, both BIT-EXACT vs the golden model in
simulation AND on the device** (`ice40/rtl/`, golden vectors
`ice40/host/vectors.py`):

| build | cycles/pt | LUT | DSP | EBR | fmax (icetime) |
|---|---|---|---|---|---|
| v1 serial (`encoder.v`) | ~845 | 1085/5280 | 6/8 | 18/30 | 37.4 MHz |
| v2 ring-parallel, angle-pipelined (`encoder_par.v`) | 63 | 3782/5280 | 4/8 | 27/30 | 26.3 MHz |

v2: banked per-ring accumulators (EBR dual-port shape: readback muxes
through the idle read port; write addr latched WITH its enable), 1
cycle/angle software pipeline, the four 16×16 mults in MAC16s, the eight
8×8 ring products hand-lowered to 7-tap shift-adds (yosys -dsp otherwise
claims 12 DSPs > 8; `use_dsp="no"` attribute is ignored by synth_ice40).
Run at 12 MHz (2.2× timing margin): 903-pt scan encodes in ~4.8 ms ≈ 208
scans/s — 13 encodes/kf at 5 Hz keyframes needs 65/s, so ~3× headroom
before the PLL.

**UART protocol lessons (hw-in-loop found both):** (1) a cmd-FSM that
ignores RX while the encoder runs DROPS BYTES and desyncs (first hw run
failed exactly so) → always-listening parser + 256-deep point FIFO; (2) at
3 Mbaud (exact 12 MHz/4) the encoder outruns the wire, so per-point acks
are pure USB-latency waste → bulk mode (cmd 0x05, no ack) + one counter
poll; per-scan wall time 477 → 141 → 68 ms, now bound by the FT2232H
~16 ms USB latency timer on 3 round-trips, datapath ~4.8 ms.

**Corner-S encode acceptance ("no glitches on test data"): PASS.** Full
replays through the fabric, every accumulator of every scan bit-exact:
spot 414/414 scans, dyn-classroom p5-walking 120/120 (1024-beam synth),
dyn-school8 p5 150/150 (long-range mask boundary exercised). 684 scans /
~600k points / 0 mismatches at ~70 ms/scan. `hw-replay` is the standing
acceptance gate for RTL changes.

### Dynamic multi-room environments (`ssp_dynenv.py`) + the people ablation

Design: classroom (7×7, spot-proxy) and school8 (hallway + 8 IDENTICAL
classrooms, 483 m² ≈ 9.9× classroom — deliberate aliasing stressor);
people as r=0.18 m circles, standing or waypoint-walking at 1.2 m/s with
deterministic yield-to-robot (min robot-person dist 0.72 m ≥ the 0.45
yield); **noise draws are people-independent, so ±people at fixed seed is
an exactly-paired comparison** (verified: untouched beams bit-equal).
Checks: clearance 0.55 m (door half-width), same-seed bit-equal.

Bench grid (1024 beams, bridge2 deploy sampler, shipped config + fenced
diag_gt labels; agent-run, 20 arms, no crashes):

| env | arm | s11 ATE/loops/prec | s12 ATE/loops/prec |
|---|---|---|---|
| classroom | p0 | 0.008 / 14 / 1.00 | 0.008 / 0 / — |
| classroom | p2 stand | 0.007 / 14 / 1.00 | 0.007 / 14 / 1.00 |
| classroom | p5 stand | 0.008 / 0 / — | 0.008 / 14 / 1.00 |
| classroom | p5 walk | 0.008 / 14 / 1.00 | 0.008 / 14 / 1.00 |
| classroom | p10 walk | 0.009 / 14 / 1.00 | 0.008 / 14 / 1.00 |
| school8 (odom 0.385 / 0.719) | p0 | 0.954 / 93 / 0.53 | 0.749 / 115 / 0.62 |
| school8 | p2 stand | 0.411 / 97 / 0.70 | 0.590 / 110 / 0.72 |
| school8 | p5 stand | 0.348 / 112 / 0.78 | **0.153** / 108 / **0.85** |
| school8 | p5 walk | 0.862 / 113 / 0.69 | 0.777 / 105 / 0.64 |
| school8 | p10 walk | 0.765 / 108 / 0.69 | 0.997 / 80 / 0.52 |

**Findings (seed-stable claims first):**
1. **The identical-rooms aliasing admits wrong closures with NOBODY
   present** — school8 p0 precision 0.53/0.62, constraint errors up to
   1.5 m — and the admitted-wrong edges are net-negative: SLAM ATE is
   WORSE than raw paired odometry in 7 of 10 school arms (p0-s11: 0.954
   vs 0.385). Wrong > missed, now quantified in the deployment-shaped
   environment; this is the §5/§7 verification wall reproduced at
   building scale with a 360° head.
2. **Standing people are symmetry-breaking landmarks**: precision climbs
   monotonically with standing density in BOTH seeds (0.53→0.70→0.78;
   0.62→0.72→0.85) and paired ATE improves by −0.16..−0.61 m. Persistent
   bodies de-alias the twin rooms.
3. **Walkers do not de-alias** (not position-stable across the revisit):
   precision ~p0-level, paired ATE deltas mixed at noise level; the only
   recall sag in the grid is heavy walker traffic (p10w-s12: 80 loops,
   prec 0.52). Closure RECALL is otherwise people-robust (80–115 loops
   in every school arm).
4. **Classroom closure on/off is a basin draw** (bistable 0-or-14,
   uncorrelated with people: s12 died at p0 with nobody around; s11 died
   at p5-standing; both survive 10 walkers). The earlier 360-beam smoke
   reading "5 walkers kill closures" does NOT replicate at 1024 beams —
   read small-room loop counts as bands per the PROTOCOL §6 rule.
   Classroom ATE is people-invariant under pairing (|Δ| ≤ 1 mm; frontend
   + odometry carry the small room regardless).
5. **The 10× map fits the UP5K with headroom**: school8 tour = 301
   segments → 40 KB at 2b/no-relo/oct60 (31% of the 128 KB SPRAM), 24 KB
   at oct36 (19%). The people arms change map size by <3%.

Open follow-ups filed: more seeds on the school grid (people-density
precision ladder is 2-seed so far); classroom bands via freeze-noise
probe; jitter>0 (distinguishable rooms) as the de-aliasing control that
separates "people as landmarks" from "any asymmetry as landmark"; walker
motion-during-sweep (de-skew ablation) once the fabric path carries it.

## 2026-07-11 — front-consensus on the current suite: NOT default-safe (the intel median-collapse does not generalize)

Agent-run per the session's PROTOCOL §8 exception; harness validated before
trust: `FrontConsensusSLAM(fc_k=0)` bit-exact to shipped BandSLAM on
fr101[:1200] (max|dpose| 0), and four banked numbers reproduced in-session
(fr101 1.881, belg 2.644, spot-registry 0.034, synth-mixed/corridor in
range). Config = the banked intel v2 consensus exactly (fc_k=2, fc_eps=1e-3,
tol 5 cm/0.5°); no per-dataset tuning. Runner: `scratch_consensus.py` (log
`scratch_consensus.log`).

| log | shipped | consensus | declines | note |
|---|---|---|---|---|
| fr101 | 1.881 (53 loops) | 1.882 (50) | 2/1893 kf | unharmed; gate never binds |
| fr101 @1e-3 | 1.881 | 1.881 | 0 | ditto under perturbation |
| fr079 BAND | [5.52..12.41] med 8.37 (banked) | **[12.17..14.84] med 12.32** (5 rungs) | 4–11/run | **collapses onto the band's WORST level** |
| belg | 2.644 (1 loop) | 2.305 (0 loops) | 1 | improvement, AUDIT-PENDING (do-no-harm shaped) |
| synth mixed/corridor | 0.048 / 0.020 | identical | 0 | exact-GT bench: no-op |
| spot (lidar-only) | 0.034 | identical | 0 | target platform: no-op |

Overhead: uniformly ~2.35–2.5× total ms/kf (fc_k=2 ⇒ 3 frontend matches/kf).

**Verdict (a clean negative with one pending positive).** (1) On every
stable/target log consensus is a no-op that costs 2.5× frontend compute —
the gate never fires. (2) On fr079 the jitter-robust performance level sits
at the TOP of the shipped band, not the median: all five consensus rungs
land at/above shipped's worst banked draw (+47% over the band median,
2.2× the ε=0 point). The intel finding ("band collapses 14× onto its
median") is therefore log-dependent: consensus guarantees *a* level, and
that level can be the worst draw. A hardware determinism mode bought this
way can be a guarantee of the worst case — the accuracy-contract row in
`SotA/fpga_design.md` is amended accordingly. (3) belg improves 2.644 →
2.305 (med 1.86→1.45, max 8.6→4.6) with a single decline and the one loop
dropped — mechanistically consistent with the frontend do-no-harm gap
(odo-excellent log; raw odo 1.72 still beats both), so it is evidence for
that thread, NOT for consensus adoption; labeled AUDIT-PENDING per
PROTOCOL §4. Consensus stays opt-in, default OFF, including for the iCE40
deployment posture (the ε-dither twin as an online band ESTIMATOR is
unaffected — it reports, it does not decline).

## 2026-07-11 (second burst) — iCE40 corners T and F on silicon: 36 MHz pipelined encoder, the matcher bit-exact on device, oct36 rejected

### Corner T: encoder v3 (`ice40/rtl/encoder_pipe.v`) + 36 MHz PLL — acceptance PASS

The v2 fmax (26.3 MHz) was not the arithmetic: yosys's cost model had
silently LUT-mapped the 4-read-port cis ROM (a 9-LUT-level cone into
`cre_q`), and after that every fix exposed the next ~30 ns cone. The
ladder, each step sim-verified bit-exact before building:

| fix | fmax (icetime) |
|---|---|
| v2 baseline (banked) | 26.3 MHz |
| + u_q proj register + registered w·cis products | 31.3 |
| + cis ROM forced to 4 EBR replicas (`ram_style="block"`) + FIFO 34→32 b (noack = mode reg, not a FIFO bit) → EBR 30/30 | 30.5 |
| + product tree split into 2 registered partials | 33.0 |
| + bit-sliced readback addressing {ring,j} (kills the ÷60 decode cone; wire byte order unchanged) | 33.3 |
| + registered fifo_empty/ack_pend flags + timing-driven PnR (`FREQ=36`) | **39.7** (nextpnr 39.75 @ 36) |

v3 is also smaller: 2933 LC (v2 3782), 4/8 DSP, 30/30 EBR, ~67 cyc/pt
(3 pipe stages deeper, still 1 cycle/angle). `top_pll.v` = SB_PLL40_PAD
12→36 MHz, UDIV=12 (3 Mbaud exact), margin 1.10×; the empirical gate is
the sweep. **Acceptance at 36 MHz: PASS — 64-pt vector + spot 414/414 +
dyn-classroom-p5 120/120 + dyn-school-p5 150/150, zero mismatches.**
Wall stays ~69 ms/scan (FT2232H USB-latency-bound); the datapath went
4.8 → 1.6 ms/scan ≈ 590 scans/s, keyframe headroom ~3× → ~9×.

### Corner S closed: N_ANG × bits verdict — oct60 STANDS, oct36/45 rejected

The krylov-agent N_ANG×bits sweep never banked; rerun directly
(`scratch_nang.py`, background, ~13 min). Harness self-validated: every
n_ang=60 arm reproduced its banked registry number exactly (spot 0.034,
stata 0.202/99 loops, fr101 1.881/53) before the 45/36 arms were read.
Config fixed across datasets (no per-log tuning); lattice patched only
via `ssp_lattice.set_polar`.

| n_ang | spot sh / lean | stata sh (loops) / lean | fr101 sh / lean | 2b seg | school301 |
|---|---|---|---|---|---|
| 60 | 0.034 / 0.036 | **0.202** (99) / 2.122 | 1.881 / 1.132 | 120 B | 35 KB |
| 45 | 0.040 / 0.039 | 1.203 (33) / 1.122 | 2.076 / 3.474 | 90 B | 26 KB |
| 36 | 0.042 / 0.036 | 0.975 (53) / 3.192 | 0.905 / 2.048 | 72 B | 21 KB |

Verdict: (1) spot ties at the reference noise floor at every N_ANG — the
7×7 room does not bind the knob (consistent with the banked "all
samplers/lattices tie" there). (2) The stata proxy — the closest real log
to the SPOT head — collapses below 60 angles: shipped float 0.202 →
0.975/1.203 with loops 99 → 53/33; lean-2b is bad on stata at every
N_ANG (the known closure-redundancy limit, 6b tier there). (3)
fr101@36 0.905 has the exact signature of the un-banked lam8/N36
knife-edge draw (2026-07-08 precedent, 0.668, fragile) — not banked.
**oct60 stands. The oct36 ROM option is REJECTED for deployment; its
memory case is moot anyway — school8 at 2b/no-relo/oct60 is 35 KB = 27%
of the UP5K's 128 KB SPRAM.** (2b bytes here = phase codes only,
2·D_MAIN·2b; the earlier 40 KB figure included per-ring scales +
overheads.)

### Corner F: the matcher on fabric — bit-exact on silicon at 24 MHz

Golden integer matcher added to `ssp_ice40.py` (`match_int`,
`rot_grid_int`, `shift_for`; `encode_int` untouched, selftest still
passes). Contract: per-ring partial sums

    s[k] = Σ_j conj(rot(Q >> sh, ρ))[k,j] · i^mc[k,j] · cis(u_kj(Δ))

with Q = the last-encoded scan's accumulators (read-only in match mode —
one encode serves many candidates), mc = the 240 2-bit QPSK map codes
(THE deploy store), Δ = (dx,dy) in λ_min/256 units, ρ = grid-rotation
steps, sh = per-scan pre-shift (real 1024-beam scans reach |Q| ≈ 2^23.3
— measured spot/classroom/school max 2^23.0/22.9/23.3 — so sh≈9 keeps
DSP operands i16; scores are linear in Q, the shift is truncation
precision only). Geometry pinned in `selftest_match`: float rotation ==
permute-with-conjugate-wrap at m ∈ {1,7,33,59}; score peak at (0,0,ρ=0)
on the synth room; alignment convention pinned (map content displaced +δ
peaks at Δ = −δ); deterministic.

RTL (`ice40/rtl/encoder_match.v` = v3 + match extension; encode schedule
untouched, match-mode gates are enable-muxes that free-run in encode
mode): H = conj(Q)·i^mc is pure sign/swap (no multiplier); the
translation phases REUSE the encoder's proj→cis datapath (x,y regs carry
dx,dy; same ang ROMs, same cis EBRs); the four 16×8 products per ring go
to the 4 free MAC16s (8/8 DSPs now used); 8 cycles/angle sequential
schedule (stage-A mux+barrel-shift → stage-B conj+negations → consume →
accumulate) = **483 cycles/candidate ≈ 20 µs @ 24 MHz ≈ 50k
candidate-poses/s fabric-side** (deploy-shaped: ~500 evals/kf → 10 ms/kf,
~100× under the 5 Hz budget). M codes live in a SPRAM (30/30 EBRs taken;
single-port is safe — writes only while idle, reads only during match
beats; its registered read IS the mc pipeline stage; a register-file
version cost ~900 LC and blew the device). Protocol (`enc_top_m.v`):
0x06 + 60 packed bytes → ack 0x2C; 0x07 dx dy ρ sh → 32-byte reply
(4 rings × re,im i32 LE). Build: 4716/5280 LC, 8/8 DSP, 30/30 EBR, 1/4
SPRAM; places at ~28–30 MHz (remaining limiters: exec-FSM decode + score
accumulate mux — follow-up filed) → ships at 24 MHz PLL (UDIV=8, 3 Mbaud
exact, margin ~1.17×). The 36 MHz corner remains the encoder-only build's.

**hw-match: BIT-EXACT — 22 candidates (translations to ±8 m, rotations
incl. conjugate-wrap cases, two shift values) × 4 rings, vs `match_int`,
on device.** Wall 16 ms/candidate = one USB round-trip each (a batched
0x07 is the obvious follow-up, as bulk 0x05 was for encode).
Re-acceptance on the match bitstream: spot 414/414 + school-p5 150/150,
zero mismatches — the encode path is untouched by the extension.

Hardware-in-the-loop earned its keep again, catching what sim + TB
missed: E_MWAIT trusted the STICKY m_done of the previous match, so ring
0 of every reply streamed the stale previous score while rings 1–3 were
"saved" by the UART being slower than the new match — an on-device-only
failure signature (the TB's polling timing never exposed it). Fixed with
a busy-based wait; sim re-verified, reflashed, bit-exact.

Files: `ice40/rtl/{encoder_pipe,encoder_match,enc_top,enc_top_m,
top_direct,top_pll,top_match,tb_encoder2,tb_match}.v`, Makefile FREQ=
knob, `ssp_ice40.py` matcher golden, `ice40/host/vectors.py`
{gen,sim,hw}-match + Board.load_m/match. Corner status: **S closed
(oct60, 35 KB school8 map), T closed at 36 MHz (encoder) / 24 MHz
(encoder+matcher), F = encoder AND matcher primitives bit-exact on
silicon.** Next on the track: batched match command (kill the USB
round-trip), match-build timing to 36 (est decode + accumulate mux),
on-fabric argmax/top-k, and the ε-dither twin for online band estimation.

**Addendum — pipelined match sweep (host-only, no RTL change).** The
16 ms/candidate wall was one FTDI latency-timer stall per synchronous
round-trip. The always-listening parser already buffers one pre-fed
command while a reply streams, and the fabric is deterministic (reply
106.7 µs + match 20.1 µs at 24 MHz), so time-paced writes + one bulk
read keep at most one undispatched command in the parser regs (the
overwrite invariant) and let replies pack the FTDI buffer back-to-back:
`Board.match_sweep_paced` / `vectors.py hw-match-sweep`. On device, 441
candidates (21×21 grid × 3 rotations), every reply bit-exact vs
`match_int`: **279 µs/cand = 3 582 poses/s at the safe 250 µs period
(2× wire margin); probes at 200/160 µs gave 4 325/5 189 poses/s, both
bit-exact** (wire floor ≈ 127 µs). Deploy-shaped: ~500 evals/kf ≈
140 ms/kf — the frontend correlation is real-time at 5 Hz keyframes
even THROUGH the 3 Mbaud bench UART; fabric-side it is 10 ms/kf. A
fabric candidate FIFO (for un-paced hosts) stays a follow-up.

## 2026-07-11 — webvis: dynenv showcases embedded (agent-run, spot-verified)

Agent-run per this session's §8 exception; verified in the main session:
pack parses with 14 entries, the three dynenv arms carry exactly the
banked bench-grid numbers, the exporter diff is registry-only + a pinned
`expect` gate that ABORTS export on any ATE/loop drift from the banked
grid (no silent config drift, permanent).

Embedded (seed 11, 1024 beams, bridge2 deploy sampler, shipped config;
reference = synthetic exact GT): dyn-classroom-p5w 0.008/14,
dyn-school8-p0 0.954/93 (labeled DELIBERATE NEGATIVE SHOWCASE — the
identical-rooms aliasing admitting wrong closures, worse than raw
odometry), dyn-school8-p5s 0.348/112 (standing people de-alias). Selector
grouped (Real-world logs / Dynamic synth (people)) with verdict
one-liners; docs-alongside captions. jsc parity: all 14 replays shim-walk
to Python fin (new arms ≤3.7e-6 m; the 11 old replays unchanged;
re-export of spot through the modified exporter is byte-identical). Page
21.8 → 25.5 MB (under the 26.2 precedent).

**Known-issue ledger (pre-existing, found by the agent's regression
sweep, NOT fixed):** webvis core self-test T8 ("segment fold: d/dθ
correction beats permutation-only 3×") fails at HEAD — broke at 1f2bab3
when the sandbox ladder moved to 5 cm finest (the test's fixed 0.6°
offset is ~3 rad of phase at a 5 cm ring, beyond first-order), and
`?test=1` threw entirely from 1f2bab3..9bfdaff (missing `LADDERS.phi`)
so the break went unnoticed. Sandbox/display toy only (replay poses are
Python-recorded); fix = give T8 a ladder-aware offset, deferred.

## 2026-07-11 — 2b max capacity: readout quality solved, raster extraction is representation-limited (agent-run, ledger-verified)

User directive ("solve the high-fidelity corner via 2-bit phase, max
capacity — global readout quality and map extraction"); agent-run per
this session's §8 exception. Main-session verification: the harness gate
reproduced the archived capacity scripts row-for-row bit-exact (JOINT
K=32 float med 0.033 / succ 0.76 / L1-hit 0.96; knee tables), and the
banked school8 fixture row (0.954/93/301 segs) — ledger
`scratch_capacity2b.log` + per-phase logs.

**1. Capacity at 2b (80 paired probes, stata + school8).** Member-level
2b code noise is NIL for readout: 2b-mem ≡ float at every K (and better
on school8 at K≥96). The knee is SUPERPOSITION (float succ 0.94→0.72
over K 32→128); the first genuinely-2b limit is BUNDLE-level
quantization of the K-sum, which binds only at K≥64–96 on a clean map
(paired +0.012→+0.061 by K=128) and never binds (≤128) on the
drift-warped school store. 6b ≡ float throughout.
**AMENDMENT to the banked story:** "2 bits costs one octave of capacity
(K* 32 vs 64)" does NOT survive 80 paired probes — the archived data
reproduces bit-exact, but that inference was p90-threshold noise at 40
unpaired guesses. The conservative ≤32-segments-per-read RULE stands
(float itself softens beyond 48–64); the REASON is amended to
superposition-only. Prior-free global joint decode: knee K≈32–48 on
both datasets, float ≡ 2b; school's succ ceiling ~0.8 at every K is the
twin-room alias (the banked environment wall, not a store property).

**2. Deadband × readout: snapping WASHES for decode quality** (≥4
phases tie float everywhere; 2ph hurts) — the e2e nph=4 deadband win
lives at the accept-gate/closure layer, exactly as banked. The real
readout-tail effect in the "2b" arm is **nmag=1 magnitude flattening**
(crosstalk whitening): present at 8/16/64 phases too (school K=64 p90
0.52→0.14–0.18; stata K=96 0.51→0.30), replicated at 80 probes.

**3. Map extraction (the centerpiece): a quantified NEGATIVE with a
mechanism.** GT-free matched-filter imaging (GT = scoring label only;
rand-phase NULL ≈ 0.5): occupancy AUC tops out at 0.63–0.69 (best-F1
0.39–0.50) and is FLAT in K (K=8→all costs 0.04–0.06) and FLAT in bits
(float→2b ≤0.05). Cell-level raster extraction is
REPRESENTATION-limited — 240-component Fourier aperture per read ×
multi-pass fine-ring phase decoherence (fine-rings-only imaging is the
WORST reader, −0.06..−0.13) × line-content ghost combs — not
capacity-limited. A 2b bundle carries as much extractable raster as
float; neither is thresholdable occupancy at 0.1–0.2 m. The map's
fidelity lives in CORRELATION reads (cm registration, ~3 cm global
decode), not raster geometry; the deployable "map view" stays the
signed local field (webvis refold / the live fabric image).

**4. Max-capacity recipe (UP5K).** school8-scale: per-segment
2b/no-relo/oct60 store (301 segs ≈ 31% SPRAM) + a readout layer of
⌈301/32⌉ = 10 group vectors (2b sum-quantized, 76–114 B each) ≈ **0.8–
1.1 KB** → global joint decode med 0.03 m, L1-hit 0.88–0.96, local
reads succ ≥0.94 (clean env). **Operating point: K=32 per readout group
(≤48 measured-safe), ⌈S/32⌉ groups.** Beyond it: bundle-quantization
tail first (clean maps, K≥64), then superposition; on aliased envs the
twin-alias ceiling binds at every K. Do not budget raster extraction.

Audits: the win-shaped "2b ≥ float at large K" was decomposed via the
nph ladder (it is the magnitude flattening, not the snap) and
LOSO-audited (self-tie Δ ≈ +0.0002 — no self-echo); one invalid AUC row
(rank-tie artifact on clipped zeros) excluded and documented. Scripts:
scratch_capacity2b*.py/log (gitignored).

## 2026-07-11 — live-system verification vs ground truth (map samples read off the silicon)

`scratch_livegt.py` scores the RUNNING live system through its own HTTP
endpoints (fabric map image + fresh direct-mode viewpoint readouts) against
the analytic dynenv world (GT = scoring only). At kf ~1300, pass 8:

- **Localization: verified.** Fabric-loc 4.5–6.2 cm vs GT (python SLAM
  5.5 cm), state `tracking`, 260/260 live encodes bit-exact, at the 8th
  perturbed pass with nothing ever reset.
- **Image vs GT: reproduces the capacity-study ceiling — after fixing a
  scoring confound.** Naive all-pixel rank-AUC read 0.390 (below chance)
  because probe coverage is path-anchored: wall pixels 56% covered vs
  free 89% (IMG_RAD=2.4 m from interior anchors), and uncovered pixels
  score 0. Covered-only: **rank-AUC 0.609, best-F1 0.31** — in line with
  the banked representation-limited ceiling (0.63–0.69). Ridge
  LOCALIZATION is good: 56% of GT wall samples have a strong ridge
  within 0.5 m, and those sit at **p50 0.133 m ≈ one image pixel** —
  correlation-sharp, raster-soft, exactly the banked verdict, now
  reproduced on silicon.
- **Direct first-return visibility: the weak reader, quantified.** Fresh
  fabric readouts (120 rays × 43 ranges × 3 picks): hits land p50
  0.80 m from the nearest GT wall, 81% > 25 cm — the per-ray-max
  first-return criterion triggers early on interior energy (furniture
  sidelobes + coarse-ring fill). Readout-refinement agent dispatched
  (fabric-exact offline emulation via match_int) targeting the reader,
  the first-return criterion, and the probe-coverage recipe.

**Ledger correction (found by the FPGA-implementation review):** the
top_match build on disk — and the FLASHED bitstream — carries **6/8
MAC16, 4931 LC** (the S_XY xm/ym products silently LUT-mapped in the
post-E_MWAIT-fix rebuild; nextpnr 28.97 MHz @ 24). The second-burst
entry's "4716 LC, 8/8 DSP" describes the earlier rebuild. Bit-exactness
is unaffected (all sweeps passed on this bitstream — a LUT multiplier
computes the same product); the lesson is DSP-inference instability
across rebuilds → add a DSP-count assert to the build flow (roadmap #4).

## 2026-07-11 — optimization review round: four read-only agents (python/FPGA × implementation/algorithm), consolidated

Full reports in the session transcripts; ranked digests + cross-checks:

**Cross-agent consensus:** the fabric datapath is ~100× under budget; the
binding costs are (1) the WIRE choreography (per-candidate 32 B replies,
µs-paced writes, 16 ms FTDI stalls on small acks) and (2) python trig in
the 11-encodes-per-match frontend. Both sides converge on the same
architectural move: grid auto-sweep + on-fabric totals/argmax.

**FPGA-algorithm (top: build the auto-sweep command first).** 0x08 grid
descriptor + on-fabric Re-sum totals stream (4 B/cand) or argmax reply
(winner + 3×3 neighborhood + margin telemetry, ~42 B): locate 49 ms →
5–8 ms/kf, kills the paced-host requirement; ~350–500 LC. Then: SPRAM
map residency (512 segs/SPRAM, segment switch → 0; school8 fits the
already-burned SPRAM), frozen-Q-in-SPRAM encode/match fusion (also
deletes the stage-A barrel shifter = a 36 MHz enabler), staged on-fabric
tracking loop (v1 host-predicted; margin-collapse = coast + flag, NEVER
auto-widen — §5 wall), ε-dither twin as band telemetry (report, never
decline — the front-consensus fence), de-skew = model-first experiment.
Ring-subset early termination PARKED (rings are interleaved, not
sequential; prune at grid level instead).

**FPGA-implementation (top: pipeline the measured 35.75 ns path).**
icetime attributes it: first_acc select cone (~13.5 ns) → 3-operand
32-bit accumulate (~15 ns). Fix = pre-registered one-hot write enables +
a pd=p0−p1 pipeline stage (+1 drain cycle) + one-hot est FSM + merged
reply streamers → expected 36 MHz closure for the match build (20.1 →
13.4 µs/cand). Also: xm/ym into the 2 idle DSPs (−215 LC + inference
determinism + build assert); half-wave az table VERIFIED numerically
exact (round-half-even is sign-symmetric; π/2 shift grid-aligned) → 6
EBRs freed (ang ROMs into EBR, M-store into EBR, 3 free for
FIFO/argmax/twin) with a loud symmetry assert in gen_luts.py; rot90
folded into the cis ADDRESS (i^mc·cis(a) ≡ cis(a+(mc<<6)), verified
exact) removes the SPRAM→mux→DSP cone; candidate FIFO reusing the point
FIFO (128 cands, zero new EBR) → 9.4k cand/s unpaced at 3 Mbaud; UART
6 Mbaud exact both clocks (12 M needs an RX rewrite — parked); match
8→4 cyc/angle only after the wire is fixed. SB_HFOSC rejected (±10%
breaks framing).

**Python-implementation (top quick win: host-serial merges).** Single
UART conversation per keyframe (prepend clear, append npts/readback —
same bytes, one 16 ms stall instead of 2–3) + PACE 200→160 µs
(device-validated) ≈ 25–45 ms/kf on the live loop, class bit-exact-safe,
host files only. Biggest single lever repo-wide: cos/sin encode kernel
replacing np.exp(1j·φ) — measured bit-EQUAL on this platform (±1100 rad
probe grid) and ~25–30% of frontend compute; platform-gated startup
assert, shipped-core maintenance pass required. Also: threading overlap
(SLAM ∥ serial), sampler vectorization (bit-equal), memoized
world_vec_seg (the KeyframeStore lazy pattern), _gn structure reuse.
Measured do-NOT list: 14 zgemv→zgemm is NOT bit-equal (band cost for
0.03 ms); der_fine stays banked-negative; c64 matcher arithmetic ≠ the
banked c64-store evidence.

**Python-algorithm (top experiment: per-ring bit allocation for the 2b
store).** nph = {16,16,4,4} over the octave ladder (~3 b/phasor mean):
fine rings keep veto-statistic fidelity (the banked per-ring-scale
mechanism), mid/coarse keep the deadband — aimed at the stata-lean gap
(2.122 vs 0.202), the one open per-regime compromise; multi-log gate,
fixed global allocation. Then: 8-base exact-permutation fine stage
(11→8 encodes/match, matcher+cmatcher share bases; exact algebra ≠ the
banked E1 first-order negative — band-vs-band framing), adaptive
run-length chord integrals (terms ~ scene complexity), bit-exact
frozen-map caching + spatial-index candidate generation (the cells hash
is maintained but never used for lookup), branch-and-bound coarse sweep
(argmax-identical), veto-retry second chain + relax cadence sweep
(school8-p0 precision as the guard), FHRR capacity-derived caps (chain
sum vs the K*≈32 knee), batched separable imaging.

Adopted into the roadmap now: build the auto-sweep+argmax command with
the accumulate-path pipeline and xm/ym DSP fix in one RTL pass (items
converge); host-serial merges + PACE 160 into live.py; per-ring bit
allocation queued as the next python experiment. The four full reports
remain the reference for the rest.

## 2026-07-11 — v5 RTL pass: the batched-sweep FIFO command on silicon (crash class eliminated, 6.6k poses/s unpaced)

Adopts the consolidated roadmap's converged top item. `encoder_match2.v`
(v5; v4 stays frozen per ledger discipline) + `enc_top_m2.v` +
`top_match2.v`:

- **cmd 0x08 batched sweep**: `mode cnt` + cnt × 6-byte candidates,
  buffered through the SHARED point FIFO (2 words each, ≤120/batch; the
  point-drain arm is hard-gated while a batch is pending). Replies
  stream back-to-back per candidate: full 32-byte per-ring partials, or
  4-byte ring-summed Re totals (the deployed unit-weight argmax
  criterion computed on-fabric). **No host pacing invariant — the
  OS-write-coalescing overwrite crash class (two live-server crashes
  today) is structurally dead.**
- Timing fixes from the measured v4 paths: pd = p0−p1 pipeline stage +
  pre-registered ONE-HOT accumulate enables (the 35.75 ns path), rot90
  folded into stage-B with the mc SPRAM sampled one phase early (the
  DSP-input cone), xm/ym as REGISTERED full-width products (+1 encode
  state, 68 cyc/pt) — the form that infers MAC16 deterministically;
  **the build flow now asserts SB_MAC16 == 8** (the v4 6-DSP silent
  regression cannot recur). Sim caught two of my own pipeline bugs
  before silicon (drain one cycle short; an extra enable-delay stage
  misaligned with the pd register — sacc[k] briefly got ring k+1's sum).
- Placement: 30.49 MHz @ the 24 constraint (1.27× margin), 5155/5280
  LC, 8/8 DSP, 30/30 EBR, 1/4 SPRAM. The limiter MOVED to the
  SPRAM-out→negate→rot90-mux stage-B cone (~30.6) — 36 MHz needs the
  stage-B split or the reviewer's mc-prefetch cis-address fold (filed).
- **On device: everything bit-exact** — legacy 0x07 path unchanged
  (hw-match 22c), batch full mode 22c + 441c, batch totals 22c + 441c
  vs `match_int`; encode re-acceptance spot 414/414 + school-p5 150/150
  zero mismatches. Throughput (441-cand sweep, ZERO pacing): **totals
  152 µs/cand = 6 596 poses/s; full 263 µs/cand = 3 803/s** (vs 279 µs
  paced / 16 ms naive). Deploy-shaped: ~500 evals/kf ≈ 76 ms through
  the bench UART; wire floor now dominated by the 20 µs match + reply
  bytes, as designed.
- live.py switched to batch totals mode (locate / wide re-search / map
  probe / direct viewpoint queries); selftest on hardware: enc 24/24,
  fx err 3.2 cm, direct-vs-cached visibility agreement 0.15 m — all
  gates unchanged.

## 2026-07-11 — VSA-translation experiments: the resonator IS line sweeps of the existing silicon primitive (agent-run, ledger-verified)

User directive ("move as much of the system as possible into VSA
operations"); ledger `scratch_vsa.log` spot-verified (bit-exact pin,
deploy-chain medians, ops counts, determinism re-runs).

**Design insight that makes it practical:** this polar lattice is
ALREADY elementwise-separable in x/y — T(Δ)_j = exp(i·k_jx·Δx)·
exp(i·k_jy·Δy) — so resonator translation factors are native axis
phasors, and the HARD-cleanup resonator (RES-H) collapses to
ALTERNATING 1-D LINE SWEEPS of bit-exact `match_int` evaluations, i.e.
the existing fabric primitive (composable with cmd 0x08 line batches
today). Bonus algebra pinned: rot(θ+π) = conj(rot(θ)) — the π branch is
an i32 sign flip, one codebook covers the full rotation group.

- **RES-H (adopt-shaped):** in-window same-argmax 93–98%
  (classroom/spot/stata), ~2.4 iterations, 245 → ~45 candidate
  evaluations (5×); disagreements score-neutral-or-better (ratio
  1.00–1.04). Deploy-shaped tracking chains: median 0.0613 m BOTH arms
  at 71 vs 348 CE/kf (4.9×). Prior-free rotation: 100% heading recovery
  (±3°) at 408 vs 5 880 CE (14.4×). Ops crossover ≈ 5×5×5 window;
  advantage grows with volume (53× at wide-search scale). Basin
  failures are score-DETECTABLE traps (ratio ~0.55 → gate+escalate);
  RES-H4 multi-start = negative (traps are true local maxima). School8
  twin check (fenced): twin/genuine ratio 0.976 — NOT relocalization,
  the §5 wall stands.
- **RES-S soft cleanup: mechanism-note** (≤1-cell ~97% but blurs exact
  argmax and needs a phasor-normalization primitive it doesn't earn).
- **Analytic Newton refine (adopt-shaped; translation only, the E1
  rotation fence kept):** gradient/Hessian = k-weighted inner products
  of the matcher's own integrand (ssp_flow-equivalent, FD-pinned).
  Surface-peak fidelity 0.01 mm vs 2.3 mm parabolic-on-int; buys a 2×
  coarser sweep: sweep24+Newton2 tracks identically (med 0.0613) at
  HALF the frontend CE/kf (175 vs 348). Fabric cost: one (k_jx, k_jy)
  coefficient ROM into the existing accumulator tree.
- **Pose bind-chain probe (mechanism-note):** a pure phasor pose chain
  (permute + d/dθ tangent + bind) tracks SE2 to <1 mm over 1000 steps
  at float64; 8b phase registers drift 1–3 cm; ≤6b slips catastrophically
  (sub-bin deadband) — confirms the i32-accumulator/2b-frozen-store
  split from the other side.
- **Fixture-level bound worth keeping:** the 2b partial-overlap segment
  peak sits med 5.9 cm from geometric alignment (content pull) — this
  floors what ANY argmax engine or refiner can do vs GT on this store
  format; deploy-chain medians (0.0613) sit on it.

Follow-ups filed: RES-H as 0x08 line-batches in the live tracker (host
change only), the (k_jx,k_jy) ROM for on-fabric Newton steps, and the
wide-search escalation gate using the score-detectability of traps.

## 2026-07-11 — map-readout refinement: first-return visibility 0.80 → 0.07 m; the 2 m ghost-comb mechanism; min3 ridge reader (agent-run, ledger-verified)

Fabric-exact emulation (every probe through `match_int` semantics,
batch path asserted bit-exact per run); the classroom fixture
reproduces the LIVE baseline bit-for-bit (image row = scratch_livegt
numbers; the three picks 1.865/0.526/0.664) — results transfer 1:1 to
silicon. Ledger `scratch_readout.log`.

1. **Image readers (host combos of the per-ring partials the fabric
   already returns; zero fabric cost).** The representation AUC ceiling
   is CONFIRMED at silicon level (nothing beats ~0.63 at honest
   coverage) — but readers move RIDGE quality: **min3** (min of rings
   1–3) lifts ridge coverage 56→71% (classroom), 65→96% (school8),
   79→97% (stata) at p50 ≤ 0.12 m, and wins outright on
   heavily-bundled mosaic reads. Fine-ring-alone ≈ chance (the drift-
   decoherence signature, reproduced live); stencils hurt the live map
   (tight-map-only, as the capacity study said). One flagged non-result:
   a stata AUCgt 0.718 row is a 46%-coverage selection, not a ceiling
   break.
2. **First-return visibility: 0.798 → 0.072 m p50 (9% > 25 cm), held-out
   school8 0.037–0.047.** Mechanism ISOLATED: the matched octave ladder
   (λ = 0.25..2, ×2) is all-ring coherently aliased at exactly **2 m** —
   long walls cast interior ghost ridges rivaling the true wall (ghost
   1.01 vs wall 0.62 at wall−2 m). No profile-shape rule fixes it.
   The fix: a **self-certified free-space mask** ray-carved at freeze
   from the system's OWN pass-1 scans at its OWN poses (GT never
   enters; 0.30 m cells, stop 0.45 m short of hits — 121 B classroom /
   1.0 KB school8, O(area)) vetoes ghost peaks; the fabric correlation
   read then contributes the cm placement (range-error early-bias
   84% → 1%). Attribution clean via a mask-only control (0.143 —
   the mask kills ghosts, the fabric read doubles the sharpness).
   **Negative banked:** the 2b relo-ring field is a content-centroid
   dome for in-room queries (float ≡ 2b) — no wall contrast; relo rings
   stay anchor/L1-decode only.
3. **Coverage recipe:** IMG_RAD 3.6 + input-fold bounds → wall coverage
   0.58→1.00 (classroom) / 0.72→0.98 (school8) at 1.6× probes; the (B)
   scoring confound disappears (AUCall 0.390→0.581). Query radius must
   STAY 2.4 (3.6 lets hits land late on exterior teeth).
4. Offline loops: exact-int batch imaging 753k probes/s (34×); float
   separable GEMM ≈ 9.5M pixel-evals/s at 1.1% rel-l2 (corr 0.99994).

Adopting into live.py: IMG_RAD 3.6 for the map probe, per-ring image
replies (sum for display + min3 layer for ray-marching), free-mask
carving at freeze, and the peak-walk + mask-veto + fabric-refine
first-return rule (constants frozen on classroom, school8-verified).

## 2026-07-11 — position⊗time trajectory memory: NEGATIVE for adoption, four mechanism notes (agent-run, ledger-verified)

User-directed VSA thread ("bind positions with time, reconstruct time
samples, perform loop closure"); ledger scratch_postime.log (fixtures
reproduce fr101 1.881 and school8 0.954/93 exactly; deterministic;
time-leak audit structural). Design: P(x)⊙T(t) bundles over
consecutive-K windows, D=360 (point decode needs the incommensurate
relo rings — the MAIN octave band is jointly 2 m-periodic), uniform
time-rung ladder permuted across components (unpermuted 2–3× worse).

- Time→position reconstruction: float p50 1–5 cm through K=32, breaks
  by K=64 — the superposition knee again; 2b bundles break at K=16–32
  (point-ATOM reads cost 1–2 octaves, unlike segment-content reads —
  mechanism: readout-template richness; no contradiction with the
  capacity entry).
- Position→time revisit candidates LOSE to the pose-array proximity
  search everywhere (classroom F1 0.68 vs 0.98; school8 0.11–0.35 vs
  0.49–0.87). Measured bound worth keeping: on fr101, same-place pairs
  sit 2.3–5.8 m apart in est/fin frames — ANY position-derived
  candidate generator is bounded by pairwise pose consistency (the
  verification wall seen from the pose side).
- Footprint: the int16 pose list dominates the fidelity-footprint plane
  at every usable operating point (the 2b vector wins bytes only where
  its decode is broken).
- Mechanism notes: nmag=1 magnitude flattening replicates as crosstalk
  whitening on the TIME spectrum (third independent surface); one-sided
  vs centered time rungs = a group-delay trade with no universal
  winner; the K=32 detection optimum has a SECOND cause (window-flood
  multiple comparisons punish small K); hierarchical coarse-first time
  decode fails exactly like the banked spatial version — binding time
  changes the readout axis, not the selection statistic.

Verdict: do not adopt; the pose array stays. The thread's value is the
mechanism notes + the pose-side wall bound.

## 2026-07-11 — hybrid point/integral sampler (user formulation): NEGATIVE as a regime-split remover, with a new cascade-density mechanism (agent-run, ledger-verified)

User spec: chords that pass the bridging gate become exact sinc-integral
segments at mass (w_i+w_{i+1})/2; undrawn chord sides return half-weight
r·dθ points at the endpoints — mass-conserving by construction, point-
formulation in the no-chord limit, integral in the all-chord limit.
Implementation `scratch_hybrid.py:pack_hybrid` (one seg-pack; degenerate
segments = points); gate imported verbatim (63.4°, no retuning). ALL
consistency selftests pass: gate-never → BIT-EXACT the E2 point pack
(array-identical, no ULP caveat); gate-always → segint + boundary
halves (≤3e-16); intensity: the hybrid uniquely restores FULL scan mass
(1.0× vs bridge2/segint's 0.79–0.95×). Banked anchors reproduce
bit-for-bit (stata 0.196/74, E2 1.659/9, fr101-E2 1.569).

Verdict — NOT a win (criteria: stata≈bridged AND fr079-band≈E2 AND spot
tie): spot ties, fr079-band matches E2's class (med 2.60 vs 2.77), but
**stata inherits the point collapse (1.664/9 vs bridged 0.196/74)** —
the sensor-regime split stays. Drawn-chord fraction 0.59–0.87 across
datasets (mostly-integral operation, as intended).

**The mechanism (control ladder, each arm audited):** not the
half-weight endpoints (drawn-only control collapses too, 4.930/9, and
is strictly worse on fr079/corridor — the endpoint restoration is the
CORRECT part of the design); not the mass convention (gapmass deeper,
1.320/1); not the harness class (bridge2-through-SegIntSLAM reproduces
0.196/74 with identical counters). Per-scan encodes are
cosine-1.000 IDENTICAL to bridge2 on all 6 rings — yet sessions
bifurcate (hybrid 9 accepted/15 veto vs bridge2 74/3 at equal coh_ref).
**The discriminator is bridging DENSITY: 2 sub-points per chord
survives stata's closure cascade; every 1-term-per-chord pack starves
it** — a per-scan-invisible, session-level property, echoing the banked
recurrent-frontend sensitivity (BFC). Prediction filed: the banked
"GROUP/8 run-length decimation is lossless" (synthetic-bench) should
FAIL stata e2e for the same reason — dispatched as a test.

Honest positives: the integral family narrows fr079's WORST draw ~2×
(segint band [2.15..2.54] vs E2 [2.21..4.86]) at equal medians; the
hybrid's fr101 median 1.174 ≈ E2's 1.026. No compute win (segment terms
cost ~2× point MACs).

## 2026-07-11 — per-ring bit allocation: NEGATIVE as a recipe; stata-lean is BAND-dominated (banked-premise amendment); the asymmetry residue (agent-run, independently audited)

RingStoreSLAM (subclass; per-ring nph over the matched band, relo
pinned; lean pipeline). All harness gates passed (stata/fr101/spot lean
+ school8 bench rows reproduced; neutral arm max|dpose|=0 after fixing
a REAL implementation trap worth keeping: a broadcast float64 nph
vector breaks bit-faithfulness on the c64 store via NEP-50 promotion —
5.9e-6 phase divergence, ABOVE the 1e-6 cascade-chaos threshold; rows
must be assembled from scalar quantizer calls). Independent read-only
audit: CLEAN.

**AMENDMENT to the banked story:** stata's lean-2b "failure" (2.122/11)
is a bad ε-draw — uniform-2b's own band is [0.17..2.99] med 1.48 with
loops 11–56, a30's band overlaps it almost entirely, and **uniform-6b
itself collapses at ε=1e-3** (1.18/17, 2.15/13). The banked "stata
needs the 6b tier" is an ε≤1e-6 basin property, not a store-fidelity
tier. stata-lean joins the band-dominated logs; no fixed allocation
stabilizes the cascade. (fr079 additionally pays median +1.3–3.4 for
ANY reallocation; fr101 all arms in-band; spot all tie.)

**Mechanism (reproducible-ε ordering + the attribution arm):** at
ε∈{0,1e-6} loop counts order strictly u2b {11,11} < a30[16,16,4,4]
{24,51} < u6b {71,71} — but **x40 [64,64,4,4] STARVES in every draw
(≤14 loops, dies at the cmatcher pose gates before the veto)**. So the
gap is NOT fine-ring code noise alone: fine-ring fidelity has an
in-family optimum (~16 phases) that only exists while mid/coarse keep
the nph=4 deadband; overshooting fine bits kills the loop match. More
store fidelity is not monotone. The inverted control [4,4,16,16]
collapsed everywhere (stata 3.730/7; school8 precision 0.59, admits
wrong closures) — as predicted.

**The adopt-flavored residue:** on the deploy-shaped low-drift aliased
env (school8-p0), the asymmetric fine-favoring family raises closure
PRECISION 0.78 → 0.80–0.84 at ~1.5× loops (a30/a35, 180–210 B/seg)
while uniform-6b DROPS it to 0.74 — the asymmetry (fine fidelity +
mid/coarse deadband), not bits per se, carries it. Filed as the lever
if the spot/school class ever shows precision pressure; stata-class
multi-pass drift is not fixable by any fixed allocation.

## 2026-07-11 — exact branch-and-bound coarse sweep: PROVEN, then correctly NOT adopted (agent-run, ledger-verified)

An admissible per-block bound tighter than Lipschitz (exact
single-sinusoid interval max per feature off one elementwise product;
proof in scratch_bnb.log): identity gate 4033/4033 recorded real
matches argmax-identical with 0 fallbacks; full fr101/stata/spot runs
BIT-IDENTICAL (max|Δpose| = 0, no ε-band needed). Two reusable
measured facts: (1) Accelerate/AMX zgemv bits DEPEND ON THE CALL'S ROW
COUNT (~1.5e-15 relative between subset paths — "same BLAS shape" is
unattainable for true subsets; exactness must come from guard-band +
verbatim-fallback construction, guard 7 orders above measured
deviation); (2) pruning power lives in the λ≥0.5 rings — the λ=0.25
ring is Nyquist-matched to the 0.06 m grid by design and cannot prune
above cell scale.

Bottom line: evaluates 2–6.5× fewer cells (best on the target platform,
spot 0.154×) yet runs 2.6–5× SLOWER wall — AMX makes the dense
289×240 gemv nearly free (6.7 µs) while bound math is memory-bound
numpy; the coarse sweep is ~1% of the encode-dominated host match (the
59%-of-MACs framing was MAC count, not wall). NOT adopted host-side.
The candidate-count result matters exactly where per-candidate cost is
wire/fabric-bound — and there the per-anchor bound needs 240-element
access (wire-dead over 0x08; the per-ring-partial bound variant prunes
poorly, 0.60–0.81 — honest negative). Fabric candidate reduction stays
with RES-H line sweeps / an on-fabric argmax unit, converging with the
VSA-translation verdict.

## 2026-07-11 — cascade-density prediction test: GROUP transfer-failure CONFIRMED, mechanism REPLACED (supersedes today's hybrid entry's density story; agent-run, independently audited)

Harness gates passed (banked synth-1024 rows reproduce to the digit —
the GROUP/8 claim STANDS ON ITS BENCH; probe layer bit-exact,
max|dpose| 0). Ledger scratch_density.log.

1. **Prediction confirmed:** GROUP/2/4/8 all collapse stata e2e
   (3.4/1.8/3.4 ATE, 8–13 loops vs bridge2 0.196/74), and the GROUP/8
   collapse is ε-band STABLE (identical at 1e-6 and 1e-3 × 2 seeds) —
   a deep basin, not a draw. The banked "mass-exact GROUP decimation,
   lossless" is bench-valid but MUST NOT transfer to stata-class
   sensors.
2. **Mechanism refuted and replaced.** n1 (ONE term per chord)
   SURVIVES at 0.194/74 — the predicted collapse limit doesn't
   collapse. The ladder knee is n3→n4 with collapse on the MORE-points
   side; the half-chords control (same term count as n1) collapses. The
   variable that orders the cosine-1.000 family is **r0-encode distance
   from the exact-integral pack**: ≥ ~3e-3 survives (n1 3.5e-2, b2
   7.6e-3, segint-lone-arc 4.5e-3 — whose "counterexample" dissolves:
   rerun, its accept sequence is IDENTICAL to bridge2's), ≤ ~2e-3
   collapses (n4, n5, exact segint-lone-pt = 0). The exact-integral
   limit itself sits in stata's bad basin; sampling discreteness is
   what rescues. n4/n5 are ε-boundary arms (rescued in 3/4 flip draws
   — PROTOCOL §6 framing); the good basin and the e2/GROUP collapses
   are deep.
3. **Gate localization:** not the named gates — collapsed arms starve
   at the cmatcher displacement PRE-gate (56–68/75 attempts vs 10/75;
   coh-veto nearly silent); ignition-then-flameout for the boundary
   arms. The bifurcation lives in chain-bundle matcher basin quality —
   session-level, recurrent-frontend (BFC-class), invisible to per-scan
   cosine/energy. Today's earlier "2-sub-points-per-chord is the
   discriminator" story is SUPERSEDED by this entry.
4. **Adopt-shaped incidental:** n1 = bridge2-class accuracy at HALF the
   encode terms (911 vs 1734, band-stable on stata) — multi-log suite
   validation dispatched before any adoption talk.

**Addendum — tracker-gate calibration: local surface observability does
NOT separate lost from healthy (measured).** With per-kf diagnostics
logged on the certified-revert runs (corridor/orbit + office/reverse
healthy, corridor/reverse lost): best single statistic (per-ρ margin
< 0.77·spread) catches 83% of lost keyframes at 11% false-flags on
healthy ones — a false odometry-hold every ~2 s, exactly the v2
regression mechanism; curvature thresholds are worse (den_x p50 on
HEALTHY corridor tracking is 0.010·spread — the v2 blind constant 0.04
flagged nearly everything). Healthy-but-degenerate and lost-and-
degenerate keyframes present the SAME local surface; correctness is not
a per-keyframe surface property. Commit-policy gating from local
observability joins the banked walls; recovery needs temporal/structural
evidence instead (the RES-H study's remit).

**Addendum — readout recipe live-integration verified on silicon (with
two integration lessons).** The study recipe needed two fixes found by
GT-scored hardware selftests: (1) the free-space carve BLEEDS — rays
passing tangentially near walls mark wall-adjacent cells free (measured:
22/38 true-wall peaks wrongly vetoed) — fixed by un-freeing cells within
0.35 m of observed hit endpoints; (2) marching the min3 layer with a
naive first-crossing threshold re-admits the 2 m combs the mask exists
to kill (2.1 m p50 vs GT) — the cached path now marches the sum layer
WITH the mask veto. Also ported the study's exact peak rule (scipy
prominence on a smoothed, below-padded profile — my hand-rolled ±3-
sample window rejected broad ridges) and its maskfb boundary-fallback
tier (rays whose wall response is comb-inverted or beyond query
coverage). Verified on hardware, GT-scored: direct-from-phasor hits
p50 0.150 / p90 0.472 m (baseline 0.80/2.03), cached 0.224/0.715;
60/60 rays produce hits (42 peak-tier + 18 boundary-tier). Live server
relaunched with the verified stack.

## 2026-07-11 — cascade scheduling: veto-retry REJECTED (recall-collapse feedback), shipped 4/25 cadence CONFIRMED (agent-run, reconstructed-and-verified ledger)

Resumed post-quota-kill; the ledger was reconstructed from surviving
per-arm outputs and VERIFIED (independent rescore identity, loop
recount from slam.edges, read-only code audit: the retry harness is
verbatim try_constraint + rank selection, bit-exact parity when
inert; all baseline gates reproduce banked numbers incl. fhw 0.981/559
and school8 both seeds).

**H1 second-chain veto-retry: REJECT.** Neutral/no-op where vetoes are
rare (fr101 +2 edges neutral; spot zero vetoes; school8 precision guard
unchanged; fr079 ε=0 bit-identical) — but on fhw, the one log with
hard-veto pressure, the 24 admitted second-chain edges (longer-lever,
non-proximity chains) perturb anchors and TRIPLE primary vetoes
(36→96): loops 559→385 (−31%), ATE 0.981→1.028. A recall/stability
positive-feedback failure, not a precision one — the do-no-harm
asymmetry again: the retry only ever fires where the cascade is already
stressed, and its edges make the stress worse.

**H2 attempt/relax cadence: shipped 4/25 is the flat ridge.**
attempt_every dominates: a8 halves recall (fr101 3.15); a2 outruns
relaxation at r≥25 (2.24/3.00). a2/r10 is nominally best on both logs
(fr101 1.695, stata 0.192) but is the aggressive-relo profile the gates
forbid — flagged for a possible full-suite band study, NOT adopted.

## 2026-07-11 — quota-killed studies harvested from completed worker runs (n1 suite; FHRR caps) — banked from verified ledgers

Both agents died at the session limit AFTER their runs completed; rows
verified (n1: predecessor row re-derived byte-identical + gate anchors;
caps: interim rows already cross-checked in-ledger).

**n1 sampler suite validation: NOT suite-safe — the regime split is
ABSOLUTE ENCODE DENSITY.** Gates passed (all banked anchors). Dense
scans (n1 ≥ ~800 terms/scan): EQUAL or better — stata 0.194/74 ≡ b2,
spot identical counters, synth mixed+corridor identical, school8-s11
BETTER incl. precision .53→.72. Sparse 180° logs (156–321 terms): belg
band-SEPARATED regression (n1 [4.85..5.42] vs b2 [2.33..2.68], one draw
loses ALL loops), fr101 band-median worse AND loses its hallmark
ε-robustness (the point-citable log becomes banded), fr079 worse
median. The knee-distance observable (encode-collapse axis) is SAFE
everywhere (min 7× above the ~3e-3 knee) — necessary-not-sufficient;
the sparse failures are chain-bundle basin scatter from too few terms
per scan. Verdict: REJECT for the suite; FILED as a dense-head deploy
option (the 1024-beam target regime: equal-or-better at half the
encode terms — needs school8-s12 + fhw band + an audit before any
adoption). Missing when killed: fhw band, school-s12.

**FHRR caps: cap=32 (the theory-adjacent value) binds nowhere = free
no-op; cap=16 harmful-ish (fhw-lean eps0 3.03); cap=8 MIXED with one
striking regime win — stata-lean band [0.17..2.99] med 1.48 →
[0.13..0.30] med 0.29 (n=5), the first knob observed to STABILIZE
stata-lean's cascade** (mechanism-consistent: fewer drifted far chain
members → cleaner chain-bundle basin at the density study's
pre-gate). Not a default: fr079-shipped band median 8.37→10.35 at
cap=8; fr101-lean eps0 2.14 vs 1.13 (unbanded). FILED: "chain-cap 8 as
the stata-lean stabilizer" needs fr101-lean/spot-lean bands + audit.
Theory section (already banked interim): naive iid FHRR capacity
REFUTED; spatially-ordered members + landscape variance explain the
measured knee. CELL_CAP/recent_aids theory comparison: not completed.

Threads left open by the limit (next session): RES-H temporal-statistic
tracker study (nothing built), bit-exact frozen-map caching (nothing
built), n1's two missing cells, caps' two missing bands.

## 2026-07-11 — day consolidation (the autonomous burst, closed out)

One session took the iCE40 track from "encoder accepted at 12 MHz" to a
LIVE, GT-verified, self-recovering system, and ran seventeen
agent-executed studies around it. Deploy-integrated today: the 36 MHz
encoder corner + 24 MHz encoder+matcher silicon (both bit-exact,
DSP-count asserted), the 0x08 batched-sweep transport (6.6k poses/s, no
pacing invariants, crash class eliminated), the live endless-replay
system (looping perturbed feeds, interpolated bridges, fabric
localization 3–6 cm through 8+ passes, encode cross-checks 100%
bit-exact throughout), the verified map-readout stack (free-space mask
+ peak/veto/fallback first-return: 0.15 m direct visibility vs GT), and
the GT-wall overlay + trajectory-battery instrumentation
(scene-change/handoff/observability per keyframe).

Verdict ledger of the day: adopted (batched transport, readout stack,
RES-H/Newton as filed fabric candidates, wobble/reverse stressors);
rejected with mechanism (per-ring bits, veto-retry, hybrid sampler as
split-remover, B&B host-side, pos⊗time memory, observability commit
gates, oct36/45, GROUP-on-real-sensors, RES-S, ring-subset
early-termination); amended banked claims (stata-lean band dominance,
"2b costs one octave" → superposition-only, GROUP bench-only,
6b-tier = basin property); superseded (cascade-density → integral-
proximity basins); new hard bounds (2b content-pull 5.9 cm floor,
pose-frame consistency 2.3–5.8 m, 2 m ghost combs, AMX row-count
nondeterminism, NEP-50 quantizer trap). FINDINGS.md carries the
four section-reshaping findings; SotA/fpga_design.md carries the
deploy addenda + in-place retractions. Open next: RES-H temporal
tracker study, frozen-map caching, n1's two missing cells, chain-cap-8
validation, 36 MHz match build, on-fabric argmax.

## 2026-07-11 — v6 RTL: rot90-as-cis-address fold + ON-FABRIC ARGMAX, accepted on silicon

`encoder_match3.v` / `enc_top_m3.v` / `top_match3.v` (v5 files frozen):

- **The fold**: i^mc·cis[a] ≡ cis[(a+64·mc)&255] — verified EXACT on the
  shipped ROM for all codes × all 256 addresses before any RTL — moves
  rot90 into the cis ROM address (8-bit add). The mc SPRAM retimes to a
  4-phase PREFETCH (codes land in registers one angle ahead; 5-cycle
  prologue, 488 cyc/cand) and stage-B becomes register-to-register (the
  conj negate folded into stage-A; the v5 limiter cone — SPRAM→negate→
  mux→DSP — is structurally gone). Both sim paths bit-exact unchanged.
- **On-fabric argmax** (0x08 header mode bit1): the fabric keeps the
  running best ring-summed total + candidate index (strict >, first-max
  = np.argmax over send order) and replies SIX BYTES per batch. On
  silicon: 442-candidate sweep with a deliberate duplicate-winner tie —
  index, total, and tie-break all EXACT; **144 µs/cand = 6 951 poses/s**
  through the bench UART, reply cost eliminated.
- Deploy-build slimming to fit (the argmax + prefetch additions hit the
  5280-LC ceiling; LC count also proved FREQ-dependent — 5280@36 vs
  5297@24 packing): legacy acked-point path retired (0x02≡0x05), bare
  0x07 safely discarded, single-best argmax (margin telemetry deferred),
  est back to 4 bits. Final: 5148/5280 LC, 8/8 DSP (asserted), 30/30
  EBR, 1/4 SPRAM; placed 31.6 MHz @ the 24 constraint (1.32×). 36 MHz
  remains placement-pressure-bound (~150 LC liberation filed: streamer
  merge, one-hot est experiment).
- **Acceptance**: batch full + totals modes bit-exact on the folded
  datapath (22c); hw-argmax exact incl. tie; spot 414/414 + school-p5
  150/150 replays zero-glitch; live selftest green.
- **live.py tracker on argmax mode**: locate() = one argmax batch + a
  3×3 refine batch (sparse tot grid; downstream unchanged) — selftest
  IDENTICAL accuracy (fx 3.2 cm, same recenters) at a 6-byte tracking
  downlink. The deploy localization loop is now: scan up, argmax down.

## 2026-07-12 — anneal-by-time-binding (negate→rematch→reencode): the CYCLE works on explicit frozen-code history at low drift; the TIME-BOUND MEMORY as its store is DEAD (scratch_anneal.py, self-run)

User purpose behind pos⊗time (clarified after the banked NEGATIVE): bind
measurements to time keys so past scans can be READ OUT, NEGATED from the
map, RE-MATCHED against the rest-map, and RE-ENCODED at the corrected
position — continually on random past scans, and backwards along a loop
at closure. The banked study tested retrieval/detection, not this cycle;
this study tests the cycle itself under increasingly honest memory
assumptions. `python3 scratch_anneal.py` → `scratch_anneal.log`.

Setup: 8×6 m room + interior wall with two doorways; K∈{16,32,64} scans
(60 beams, 2π) on a closed loop; drift = RMS-scaled random walk ∈
{0.4, 0.8, 1.2} m; matcher-tier lattice (4 matched rings × 60, D=240);
scan 0 pinned as gauge anchor; 8 anneal passes, random order; 3 seeds.
Metrics: cos(M, M*) raw and gauge-aligned (cosA, mirrors align_se2
convention), per-scan pose error, gauge-free dispersion, ghost count
(err grew >0.5 m). Selftests: sweep recovers an injected offset exactly;
FHRR unbind identity exact; A1 oracle arm converges cos→1.000;
deterministic bit-equal. Anti-oracle: GT touches sensor physics, scoring,
and the fenced A1 diagnostic only. Favorability caveats (all lean TOWARD
the proposal): translation-only drift, full-visibility single room (no
aliasing), iid time keys (no rung-ladder decode noise), no odometry links.

Arms and results (3 seeds each; representative numbers):
- A1 oracle-erase + oracle-repos (DIAGNOSTIC): converges cos 0.51→1.000.
  Harness sanity only.
- A2 oracle float history + match-repos: at drift 0.4 the anneal WORKS —
  cosA 0.990–0.999, dispersion 0.36→0.01–0.04 m, 0 ghosts: the loop's
  internal warp is fully annealed into a rigid gauge offset. At drift 0.8
  it BREAKS (ghosts 11–14/15 at K=16, 26–29/31 at K=32; dispersion ~2 m;
  scans comb-lock at ±ring offsets). Basin edge between 0.4 and 0.8 RMS —
  the banked fine-lattice basin (~0.25–0.5 m) + 2 m octave combs, again.
- A2h frozen 2b per-scan codes (60 B/scan) + float code-sum map + per-scan
  offsets (erase exact; correction = phase op on frozen code — the
  project's own frozen-content principle applied per scan): at 0.4 the
  anneal works with consensus limited by 2b phase noise — cosA 0.68–0.76
  (= the one-time quantization ceiling), dispersion 0.07–0.15 m, 0–1
  ghosts. At 0.8–1.2 bistable: mostly comb-locks, occasionally recovers
  fully (K=16 s13 @1.2: err 1.01→0.28 m). THE deployable form of the idea.
- A2q 2b map REQUANTIZED per touch: dead even at drift 0.4 (errors grow
  0.36→0.5–0.8, ghosts at every K). Mechanism isolated vs A2h: per-touch
  whole-map requantization is the killer, not the 2b code storage.
- A3 the actual proposal (spatial map + time-bound bundle Mt=Σ T_k⊙V_k;
  decode conj(T_k)⊙Mt, spatial-domain cleanup, erase, rematch, re-add,
  patch Mt): DEAD at every cell. Cleanup reconstructions p50 0.6–1.7 m;
  pose errors explode to 4–15 m; ghosts ≈ all annealed scans. Capacity
  arithmetic: K scans × 60 points = 960–3840 atoms per bundle vs the
  banked superposition knee (~32–48) → 20–80× over capacity. The banked
  knee, scaled by constellation size.
- A3r naive raw-decode subtraction: cos 0.60→0.15 in 8 passes. FHRR
  unbinding is LOSSLESS — the decode carries the entire memory re-phased,
  so subtracting it injects noise at ‖noise‖/‖signal‖=√(K−1) (measured
  3.78 vs theory 3.87 at K=16). Negate-after-unbind without cleanup is
  self-destruction, mechanically.

Side answer (user question, same thread): matching the current measurement
NEVER unbinds the map by an integral over time. The deployed matcher (and
every arm here) matches against the pure spatial bundle (on-fabric: the
frozen 2b codes; sweep 0x08). Σ_k conj(T_k)⊙Mt would reconstruct that same
spatial map PLUS K(K−1) re-phased cross-terms (‖noise‖/‖signal‖≈√K) — a
strictly worse copy at identical footprint. Windowed ladder integrals
(banked postime "in-span read", 1.43 vs 0.16 member-units) act as a time
FILTER, but for matching you want the whole map anyway. Time-binding
therefore forces a DUAL STORE (spatial for matching + time-keyed for
retrieval) — 2× footprint before its capacity problem even starts.

VERDICT: the annealing PURPOSE is partially achievable, but time binding
is the wrong memory for it. (a) The negate→rematch→reencode cycle is
mechanically sound and fully anneals in-basin warp (≲0.4 m RMS) given
EXPLICIT history — deployable as frozen 2b per-scan codes + offsets,
consensus 7–15 cm. (b) The time-bound superposed store fails at 20–80×
over the capacity knee, and its naive form self-destructs (√(K−1) noise
injection). (c) The regime where annealing is NEEDED (post-closure,
meters) is exactly where the cycle comb-locks; gross correction must stay
rigid-per-segment (shipped O(D) phase ops along the loop) — the cycle
could at most POLISH post-closure residuals if they are already ≲0.4 m.
(d) Explicit history beats superposed history AGAIN (third instance:
pose list vs pos⊗time; imaging cache vs re-decode; now anneal codes vs
time-bound bundle). Adoption: nothing changes in deploy; if sub-segment
annealing is ever wanted, the A2h recipe (O(T) at 60 B/scan + 8 B offset)
is the measured starting point, gated on drift ≲0.4 m RMS.

## 2026-07-12 — fidelity-space lattice scaling ("increase both"): angles cap at 60, the LADDER is the lever; coarse extension kills the combs; fine floor = sensor coherence length; REPLICATED-AND-LARGER on stata (scratch_biglat{,2,3}.py, self-run)

Context (user directives): the standalone architecture splits VSA spaces —
the on-chip SLAM/matcher space stays oct60 D=240; a SECOND encode-only
fidelity space is folded per segment on-fabric, streamed out (2b codes),
and decoded on the laptop (position display + map readout). That lifts the
D=240 fabric-matcher constraint from the fidelity store, so: does a bigger
polar lattice buy readout fidelity? Sweep "both" axes (scales × angles),
then "go smaller — 1 cm and up, 16+ scales".

Method: school8 fixture (real-pipeline drift-warped store, ATE 0.954, 301
segs, 1935 kf; exact GT walls as scoring labels) re-folded on each lattice
with poses FROZEN (pure representation comparison); paired probes (n=40);
metrics = local-read K-curve, prior-free global joint decode (generalized
cumulative coarse→fine over any ladder + fine matcher), raster extraction
AUC (mosaic imaging, GT scoring-only), analytic PSF (fwhm / worst sidelobe
/ 2 m comb amplitude), bytes/segment (2b, vec+der). GATE: shipped lattice
reproduced the banked capacity2b rows exactly (curve float 0.029/0.123,
2b 0.030/0.129; extract AUCin 0.6088 / AUCgt 0.5524) through the
generalized matcher/imager before any new cell was read. Harness code
imported from the validated capacity2b/capacity2 stack; lattices patched
only via ssp_lattice.set_polar; shipped globals restored after each run.

School8 sweep (AUCgt float/2b-mem/2b-SUM-K32; local32 f/2b med; joint32
med/succ/L1; PSF fwhm/side/comb; B/seg 2b):

  oct4x60      240  152 B | .260/.21/.18 | .029/.030 | .029/.82/1.00 | .5524/.5283/.5186
  oct4x120     480  272 B | .260/.21/.18 | .029/.029 | .029/.82/1.00 | .5282/.5009/.5108
  oct4x180     720  392 B | .260/.21/.18 | .031/.029 | .030/.82/1.00 | .5186/.4973/.5080
  hoct7x60     420  266 B | .280/.18/.09 | .027/.028 | .030/.80/1.00 | .5886/.5520/.5398
  hoct7x120    840  476 B | .280/.18/.09 | .026/.025 | .028/.80/1.00 | .5682/.5238/.5349
  span9x120   1080  612 B | .180/.11/.08 | .026/.025 | .028/.80/1.00 | .5644/.5218/.5348
  span11x120  1320  748 B | .280/.25/.01 | .026/.025 | .026/.82/1.00 | .5817/.5481/.5565
  span11x180  1980 1078 B | .280/.25/.01 | .028/.029 | .030/.80/1.00 | .5762/.5447/.5508
  span9x60     540  342 B | .170/.12/.08 | .027/.026 | .030/.80/1.00 | .5858/.5482/.5389
  hoct9x60     540  342 B | .340/.35/-.00| .032/.032 | .028/.80/1.00 | .6005/.5695/.5600
  span11x60    660  418 B | .280/.27/.01 | .030/.026 | .029/.80/1.00 | .5984/.5662/.5591
  span14x60    840  532 B | .145/.21/.03 | .030/.027 | .030/.80/1.00 | .5996/.5669/.5603
  span18x60   1080  684 B | .075/.18/.05 | .028/.028 | .031/.78/1.00 | .5974/.5682/.5583

Ladders: oct4 = shipped matched band 0.25–2 m; hoct7 = half-octave 0.25–2;
hoct9 = hoct7 + {2.83, 4}; span9/11 = ±fine {0.125, 0.177} / +coarse;
span14/18 = fine end pushed to 4.4 cm / 1.1 cm (user cell).

Verdicts (school8):
1. ANGLES CAP AT 60. ×120/×180 hurt extraction monotonically on EVERY
   ladder (oct4: .5524→.5282→.5186; hoct7 .5886→.5682; span11
   .5984→.5817→.5762) and buy nothing on reads/joint/PSF. Content sits
   within a few m of each anchor — 60 half-circle angles already cover the
   annuli; extra angles add correlated crosstalk to the imaging sum. (The
   e2e matcher angle findings — even-count wall normals, 113-refuted —
   are a different mechanism; nothing anywhere pays for >60.)
2. THE LADDER IS THE LEVER. Half-octave densification +.036 AUCgt;
   +coarse {2.83, 4 m} another +.012 and the 2 m octave ghost comb is
   ANNIHILATED (PSF comb .18→−.00; the banked interior-ghost mechanism
   attacked at its root). Winner school8: hoct9x60 .6005/.5695/.5600.
3. FINE FLOOR = SENSOR COHERENCE LENGTH (registered prediction CONFIRMED).
   With σ_r = 2 cm, ring coherence e^{−(2πσ/λ)²/2} predicts λ ≤ 4.4 cm
   dead: span14/span18 tie hoct9 on every store metric while only the
   NOISE-FREE analytic PSF sharpens (fwhm .340→.075). Sub-coherence rings
   are dead weight (+342 B/seg for nothing). λ_min ≈ 2πσ_r ≈ 12.6 cm is
   the floor — our 0.125 m ring sits exactly on it.
4. The banked "aperture-limited" extraction mechanism is AMENDED: it is
   RADIAL ALIAS STRUCTURE, not raw aperture — D grew 8× (240→1980) with
   zero-to-negative gain whenever growth was angular or sub-coherence.
5. Local reads and prior-free joint decode are FLAT in the lattice on
   school8 (.026–.032 med; L1 1.00 everywhere; succ ~0.80 = the twin-room
   alias ceiling, a store-independent env property — banked). Also note:
   the no-relo generalized joint decode hits L1 1.00 at K=32 WITHOUT the
   5.3/12.8 anchor rings — a rigid 2D shift only realigns axis-aligned
   components, so the 1D "×2 ladder joint-aliases at 2 m" intuition does
   not produce a global degeneracy in 2D.

STATA PHASE 2 (the 113-precedent discipline: real data, floorplan GT,
before banking; fixture ATE 0.196, 229 segs):

  oct4x60      240  152 B | loc32 .022/.020 | joint .025/.85/.93 | .5714/.5674/.5853
  hoct9x60     540  342 B | loc32 .028/.027 | joint .027/.95/.95 | .6685/.6566/.6723
  span11x60    660  418 B | loc32 .022/.022 | joint .023/.90/.95 | .6677/.6553/.6712

REPLICATES AND IS 2–3× LARGER ON REAL DATA: extraction +.096–.097 AUCgt
(float .5714→.6685), 2b +.088–.089, SUM32 +.086–.087; prior-free global
decode success .85→.95 (hoct9) / .90 (span11), L1 .93→.95. AND the
capacity tail extends: local-read p90 at K=96/128 collapses from
.506/.475 (float) and .300/.742 (2b) at oct4x60 to .104–.150 on span11x60
— the whitening-across-rings effect scales the usable K band on real
data. span11x60 recovers hoct9's local-read dilution entirely (.022/.022
= baseline) at an extraction tie; hoct9 keeps the joint-succ edge.

RECIPE (fidelity space; matcher space untouched): span11x60 PRIMARY
(D=660, 418 B/seg 2b vec+der; 11 rings 0.125–4 m half-octave × 60 angles)
— extraction +.096, joint succ +.05, K-tail extended, local reads intact;
hoct9x60 LEAN (D=540, 342 B/seg) — max extraction/joint-succ, −6 mm local
read. Fine end stops at 0.125 m (coherence floor); do not add angles.
On-fabric encode notes: half-octave = ONE extra projection pass (√2-scaled
angle LUTs) then octave bit-slices as today; open-segment accumulator at
D=660 = 10.6 KB of SPRAM (int32 I/Q, vec+der); der costs NO new projection
(cross_a = ±u_{a∓30}, the index-shift identity — per-ring 2π/λ scale
applied at laptop decode); bit-slice envelope caps λ_min ≈ range/128
(~8 cm at 10 m) — moot given the noise floor. Streams at ~130 B/s at
0.3 seg/s. Ledgers: scratch_biglat.log / biglat2.log / biglat3.log;
stores cached per-lattice in the session scratchpad.

## 2026-07-12 — v7 standalone ("dumb-cable") golden model: INTEGER TRACKER ≡ LIVE HOST TRACKER to 2 mm (ice40/host/solo.py)

User requirement: the chip must run SLAM standalone (host = sensor cable;
pose streamed out — it is tracked on-chip anyway; map codes dumped on
request). v7 golden model built on the accepted v3–v6 goldens (imported,
never edited): SoloTracker = FabricLoc ported to pure ints — anchor table
(U-unit xy, TAU/960 heading, Q15 cos/sin ROM mapping), rel/pose via
round-half-up Q15 rotates, 7×7×5 sweep around the int-composed odometry
prediction, first-max argmax, one edge-recenter retry, INTEGER parabolic
sub-grid refine (guarded trunc-toward-zero divide, RTL: small serial
divider off the critical path), EMA gate as ema += (score−ema)>>6 with
commit at 29·(ema>>6) (≈0.45) and relock at 35·(ema>>6) (≈0.55),
re-search every 12 lost kf on 15×15×7. Fold path (v7b): encode_int_at =
integer SE2 point transform (Q15 rotate + U translate) INSERTED between
the az→xy and u-projection stages — fold-at-pose lives in POINT space,
reusing the encoder verbatim; derivative vector costs NO new projection
(cross_a = ±u_{a∓30} index-shift identity; per-ring 2π/λ scale applied at
laptop decode); freeze = mcode_int comparator (|I|≥|Q| + signs), proven
≡ G.mcode_from_vec including the tie. Selftests: mcode equivalence (4096
random + forced ties), identity-transform encode bit-equal, fold→match
convention (peak at −t, 4 mm wander at 1500 pts = content noise), Q15
round-trip.

PARITY BENCH (the acceptance): live.FabricLoc (float, the deployed host
logic) run OFFLINE through a fabric shim over the same golden match_int,
BOTH trackers fed the identical item stream (looped Feed, per-pass noise,
3 passes, GT scoring only):
  classroom/orbit (26 segs, 555 kf): solo med 0.053 p90 0.077 | live
    0.053/0.077 | TRACE DIFF med 0.002 p90 0.004 max 0.099
  mixed/orbit (67 segs, 1278 kf):    solo 0.082/0.131 | live 0.082/0.128
    | TRACE DIFF med 0.002 p90 0.007 max 0.424
The integer contract reproduces the deployed tracker to quantization
noise (2 mm median). The earlier 5.3-vs-3.2 cm concern was PROTOCOL (the
banked live selftest scores a different window/reference), not a defect:
live itself scores 0.053 on this protocol. Note: without the integer
parabolic refine the med is unchanged and only p90 moves 0.079→0.077 —
the refine is nearly free on accuracy here but kept (matches live
semantics; the rho-snap bias hypothesis for the gap was WRONG — the gap
was protocol).

v7 RTL scope validated by this model: resident map store (2b codes +
anchor table in the free SPRAMs; 35 KB school8-scale), on-chip candidate
grid generator feeding the existing 0x08/argmax machinery, EMA/commit
FSM (shift-add only + one small divider), point-space fold pass (4 Q15
mults/point + existing encode datapath), sign-comparator freeze. Next:
RTL (top_solo variant), sim gate vs solo.py bit-exact, then hw-replay.

## 2026-07-12 — BSC vs FHRR at EQUAL BITS (dimension-compensated, user fairness conditions): phasor group vectors dominate by ~4× bundle capacity; BSC knee K*≈8–16 vs phasor 32–48 (scratch_bsc.py)

User challenge to the banked binary verdict: it compared bit-widths at
fixed D — a fair spatter-code test needs DIMENSION COMPENSATION, FINE
codebook resolution, and a MULTI-SCALE kernel stack. This cell meets all
three: global relocalization on the school8 fixture (same paired probes
as capacity2b/biglat), one 480-BIT vector per group on both sides.
Phasor side = the banked 2b-SUM K=32 group vector (240 comps × 2b; joint
med 0.035, succ 0.70, L1 0.90). BSC side = majority bundle of
XOR-bound (position ⊗ content) pairs at 480 bits; position codes =
binarized multi-scale random-Fourier (9 kernel scales = the winning
0.25–4 m ladder), codebook decode at 2 cm atoms over the same ±8 m
domain/locality prior as the joint decoder; content codes = 480-bit SRP
of the translation-invariant MAIN-band magnitude profile (measured
descriptor quality: 0.852 bit-agreement query-vs-stored — NOT the
bottleneck; B1 ≈ B2).

  BSC K=  1 (301 vecs, 17.6 KB): med 0.000  succ@0.10 1.00   <- pipeline valid
  BSC K=  4 ( 76 vecs,  4.5 KB): med 0.014  succ 0.78
  BSC K=  8 ( 38 vecs,  2.2 KB): med 0.020  succ 0.75
  BSC K= 16 ( 19 vecs,  1.1 KB): med 0.105  succ 0.50
  BSC K= 32 ( 10 vecs,  0.6 KB): med 3.336  succ 0.07        <- collapse
  PHASOR 2b-SUM K=32 (10 vecs, 0.74 KB): med 0.035, succ 0.70 (banked)

VERDICT: at equal bits and matched K=32, phasor 0.70 vs BSC 0.07 succ —
the binary bundle needs ~4× the vectors (K≤8) to function, and even then
carries meter-scale p90 tails (K=8 p90 1.02). MECHANISM: majority-bundle
per-bit agreement decays as 0.5 + √(2/πK)/2 (~0.57 at K=32 → ~34-bit
signal gap), which drowns under the extreme-value tail of the fine
codebook (~4k effective independent atoms); the phasor bundle instead
accumulates crosstalk coherently in the correlation sum (superposition
SNR with full complex geometry, further whitened by nmag flattening).
Fairness notes: binarized-RFF is the strongest standard BSC position
code family (smooth multi-scale Hamming kernels); exhaustive codebook
argmin is the optimal decode for bundle-vs-codebook; same locality prior
both sides; K=1 control proves the pipeline. Combined with the banked
arithmetic-knee result (8-bit ops required) and the structural argument
(no continuous translation/derivative/frozen-content transforms in XOR
algebra), the full picture: BSC-style density is already harvested where
it wins (2b store, i^mc multiply-free map operand) and loses everywhere
else it was given a fair shot. Ledger: scratch_bsc.log.

## 2026-07-12 — v7b self-mapping golden bench: fully standalone SLAM works; beats odometry 3.9× on revisit-rich worlds, odometry-grade on long tours; gated folding REJECTED (solo.py selfmap)

The user's 5-min scenario end-to-end at golden level: NO python SLAM
anywhere — odometry boot (5 kf), then the integer tracker localizes
against segments the integer mapper itself froze (fold at committed
poses, 5-kf segments, ungated), map growth stops after pass 1, replay
tracks the self-built map. GT scoring only; raw dead-reckoning chain
reported as the honest baseline.

  classroom/orbit: 37 self-frozen segs | mapping med 0.072 p90 0.093 |
    replay med 0.073 p90 0.099 max 0.175 | RAW ODO med 0.281 p90 0.468
    -> 3.9× better than dead reckoning, fully standalone; ~2 cm worse
    than tracking a python-SLAM-built map (0.053) — the frontend gap.
  mixed/orbit: 85 segs | mapping 0.853/1.636 | replay 0.887/1.630 |
    RAW ODO 0.900/1.886 -> odometry-grade: the self-map is odometry-
    anchored; on a long single loop the tracker cannot beat the gauge
    it inherited (internally consistent, globally warped — the anneal
    study's gauge lesson in system form). Stable, never diverges
    (replay ≈ mapping err; 1 relock, no runaways).

GATED FOLDING (fold only on committed kf) REJECTED: starvation feedback
— segments freeze rarely -> coverage gaps -> more holds -> fewer folds
(mixed: 32 segs, 1149 holds, err 0.85->1.31 mapping / 2.15 replay).
Same do-no-harm feedback class as the banked cascade veto-retry.

RTL scope implications: v7a (standalone localization, preloaded map) is
fully golden-proven at host parity; v7b (standalone mapping) is stable
and demo-grade on revisit-rich floors as-is; closing the gap to python
pass-1 quality (classroom 0.008) needs the sliding local-map frontend
port (per-kf fine registration against RECENT/open content + innovation
gating) — filed as v7.1, not required for the 5-min demo. GATE_FOLD
constant documents the rejected branch in solo.py.

## 2026-07-12 — spare-SPRAM sample reservoir (random-overwrite history + opportunistic re-match-and-refold): NEGATIVE on this system, with the mechanism pinned (solo.py bench_reservoir)

User proposal: use free SPRAM as a reservoir of past samples — per kf,
with chance p, overwrite a random slot (age mix decays exponentially,
tau ~ N/p kf: the user's mem/chance lever — 16 slots @ p=0.25 -> tau≈64
kf); opportunistically re-match a stored sample against the matured map
and re-fold at the improved pose (correction segments, quantized once —
A2h-compatible, no negate/requant; matcher space untouched).

Implementation (golden, solo.py): 16 slots of raw scan ints + seed pose
+ state tag; every 4th kf re-match one random slot — two-stage (coarse
re-search -> fine 7x7x5 + integer parabolic) + DO-NO-HARM gate (accept
only if matched score strictly beats the stored capture pose re-scored
against the same segment — GT-free). Scoring = fold registration error
(capture vs refold, paired; GT scoring-only).

  classroom/orbit: 62 acc/122 rej | cap 0.073 -> refold 0.086 med
    (improved 19%); hold-captured 0.023 -> 0.081 (degraded)
  mixed/orbit:    153/272 | 0.914 -> 0.939 (improved 55% = coin flip)
  mixed/reverse (hold-rich, 960 holds): 203/218 | 1.596 -> 1.645;
    hold-captured 2.887 -> 2.930 (no repair — map gauge itself warped)
  BLACKOUT DIAGNOSTIC (forced 20-kf blind windows every 150 in replay,
  labeled): classroom cap med 0.053 -> refold 0.086 (blind-window
  captures only 3-5 cm off — BELOW the refold floor); mixed unchanged.

VERDICT: NEGATIVE for adoption on this system. Mechanism (three parts):
(1) REFOLD FLOOR ~8 cm (content-pull + rho-snap + grid) sits ABOVE
capture errors wherever the map is good — tracked captures 5-7 cm, even
20-kf-blind captures 3-5 cm (locally excellent odometry); (2) where the
map is bad (long-tour gauge warp), re-matching reproduces the map's own
gauge — cannot recover truth; (3) the do-no-harm SCORE gate does not
select GT improvement (accepted refolds GT-neutral-to-worse; 2/3
rejection rate yet no quality lift) — the banked COMMIT-POLICY WALL
extends to re-registration: score-vs-map cannot distinguish "better
aligned to truth" from "regressed to the map's bias", because capture
and map are the SAME estimator (common-mode errors). The anneal study's
in-basin win (A2/A2h) required externally-injected drift between content
and gauge — the live loop has none.

Surviving design notes: (a) tau = N/p confirmed as the reservoir age
lever; (b) state-tagging slots (hold/relock capture) is the right
GT-free selector shape IF a platform ever exhibits the required window
(bad odometry + good map — e.g. heavy wheel slip); the SPOT hardware
check is the only path to revisit. Default: spend the spare SPRAM on
map capacity / the fidelity space instead. Ledger: solo.py constants
RES_N/RES_P/RES_Q/BLACKOUT document the arms.

### Webvis integration (2026-07-12, uncommitted): recipe ladders in the interactive simulator

demo/index.html sandbox gains a LADDER select (Encoder group): oct6
(existing), span11 √2 (RECIPE), hoct9 √2 (recipe lean), phi — sandbox
scale = real/2.5 (fine ring 0.05 ≙ 0.125 m), so span11 doubles density
on the existing oct endpoints exactly. Selecting a recipe ladder snaps
angles to 60 (the measured cap); the lattice chip now shows D (+ B/vec
in binary mode). Intel/real-data lattice stays PINNED (oct60) per the
6daccc1 lesson. Validation: jsc parse clean (SyntaxError control
verified), 46/46 $() ids defined, makeLattice smoke through the real
code path (span11 D=660, hoct9 D=540, finite k-vectors); rotation=
permutation is ring-independent (phi ladder = the standing self-test).

Addendum (same session): span15g ladder added per user ("bump largest
size >16 m"): span11 + octave coarse extension {3.2, 6.4, 12.8, 25.6}
(15 rings, D=900@60). lam_max 25.6 m > the 16x10 m sandbox worlds ->
+-12.8 m single-ring unambiguous window covers the whole floor = global
anchor with ZERO coarse ambiguity (real-scale analog x2.5: 8-64 m
anchors on a ~40 m floor). Thermometer gate verified safe: MIN_SCALES=3
coarsest rings never fade (the gate anti-aliases FINE rings on sparse
far returns — opposite direction). Smoke: D=900/15 rings/lamMax 25.6
through the real makeLattice; parse + 46 ids still green.

### Webvis top-room glitch (user report): corr surface AUDITED CONSISTENT; root cause = deploy-UNFAITHFUL global-bundle 2b store in binary mode; FIXED with per-segment freeze semantics

User: tracking glitches entering the top room of the sandbox "room"
world, worst in FPGA/binary mode. Audit: the correlation surface IS
matched properly — score(δ)=Re Σ conj(M)·S·e^{iW(guess+δ)} (peak-at-
alignment convention verified), the odometry-prior penalty is applied
in place to the same array the argmax reads, bestSurf keeps the
winning-θ slice, and binary mode passes the identical quantized map to
match and panels. No display/optimizer mismatch.

Root cause (headless jsc repro, fold-at-GT diagnostic, oct60, room
world, patrol orbit + top-room pass, 5 paired probe offsets): binary
mode quantized the WHOLE global accumulation (quantMapView on hundreds
of superposed scans — far past the banked bundle-quantization knee);
the dominant main-room phase rounds away weak/new-room contributions.
  probe (top room)   float 0.018 | global-2b 0.024-0.033 | seg-2b 0.009-0.024
  probe (main ctrl)  float 0.015-0.020 | global-2b 0.043-0.045 | seg-2b 0.021-0.030
  first entry (no top content): float 0.024 | global-2b 0.076 (at the
  ±0.10 search-box edge = the visible glitch)
The real fabric freezes 2b codes per ~6-kf SEGMENT and the map is
their sum. FIX (demo/index.html, uncommitted): binary mode now keeps
an open float segment accumulator + a frozen sum of per-segment 2b
codes (freeze every SEG_KF_SB=6 kf; decay applied to both; sbQ := 
frozen+open so matcher and panels stay identical). Old single-caller
quantMapView(global) removed; parse + 46 ids green.

Float-mode residual glitchiness at entry is the KNOWN corridor-regime
wall (the top strip is 12x2.5 m, two parallel walls -> along-axis
slide) + doorway-turn error under the constant-velocity guess — which
is FAITHFUL to the shipped lidar-only pipeline (odometry fusion exists
behind the prior-γ slider for the FPGA-live analogue). Not patched:
blind gates are the tracker-v2 regression class.

## 2026-07-12 — golden-ratio ladders (user: "phi recipe, 1 cm .. >2 m"): NEGATIVE on real data — the school8 2b edge was a synth artifact; √2 recipe stands (scratch_biglat4.py)

Cells: phi13x60 = 0.01->3.22 m (13 rings, D=780, 494 B/seg; 6 rings below
the 12.6 cm coherence floor — registered dead) and phi8x60 = 0.125->3.63 m
(8 rings, D=480, 304 B/seg; phi spacing per se, no dead rings). phi =
maximally incommensurate ratios (slowest CF convergence).

school8: phi13 .5902/.5784/.5538, phi8 .5818/.5721/.5572 (AUCgt f/2b/SUM),
comb annihilated both (.03/-.03), local/joint flat. The 2b arms BEAT every
sqrt2 ladder's 2b (best .5695) by +.003-.009 — a candidate "quantization-
error decorrelation" mechanism, flagged borderline (an order below the
banked ladder effects) and sent to phase 2 per the 113 discipline.

STATA (phase 2): the edge INVERTS —
  phi13x60: .5793/.5590/.6011 | joint .030/.85/.93 | local .026/.026
  phi8x60 : .6063/.5943/.6287 | joint .029/.88/.88 | local .028/.023
  vs hoct9x60 .6685/.6566/.6723 (joint .95/.95), span11x60 .6677/.6553/
  .6712 (joint .90/.95): phi loses float by -.06, 2b by -.06, joint succ/
  L1 by .05-.07. Only local reads tie.

VERDICT: phi ladders NEGATIVE for the fidelity recipe; the sqrt2 half-
octave family stands (span11x60 primary / hoct9x60 lean). The school8
phi-2b edge joins the phi-prime angle recipe (2026-07-10) as the second
golden-ratio synth-sandbox artifact refuted on real data — mid-band ring
PLACEMENT luck on school8 geometry, not a store mechanism. phi's comb
annihilation replicates (incommensurability does kill the alias) but the
sparser mid-band (ratio 1.618 vs 1.414) costs more extraction than the
comb kill buys — density in the content band is the binding constraint.
Webvis: phi8/phi13 stay in the ladder selector labeled REFUTED (negative-
showcase convention). Ledgers: scratch_biglat4.log, biglat4_stata.log.

## 2026-07-12 — v7.1 best-of-K segment matching (solo): STRICT WIN for localization (beats the live host tracker), REGRESSION while self-mapping a drifting map — mode rule adopted (solo.py K_TRY)

The mixed-world self-mapping gap analysis pointed at segment handoffs
(banked doorway battery: segment-switch = 1.6-3x glitch lift; the live-v2
raw-score handoff REGRESSED and was reverted). v7.1 insight: in the 2b
code domain every segment vector has norm sqrt(D) (phase-only codes), so
raw totals ARE comparable across segments — the float-domain thrash
mechanism does not apply; add incumbent HYSTERESIS (challenger must beat
the last-used segment by 66/64 ~ 3%) against residual flapping.
Implementation: SoloTracker._sweep_seg factored out; K_TRY nearest
segments swept per kf, best effective score wins; K_TRY=1 preserves v7.0
bit-parity (gate re-verified: classroom 0.053/0.077, trace diff 2 mm).

LOCALIZATION vs a coherent (python-built) map, K=3 — strict win on BOTH
worlds and better than live FabricLoc on the identical stream:
  classroom: med 0.049 p90 0.063 max 0.234, holds 0
             (K=1: 0.053/0.077/0.264, holds 34; live: 0.053/0.077/0.266)
  mixed:     med 0.080 p90 0.104 max 0.466, holds 6
             (K=1: 0.082/0.131/0.876, holds 55; live: 0.082/0.128/0.876)
SELF-MAPPING, K=3: classroom IMPROVES (med 0.060 p90 0.079 max 0.092,
holds 113->0) — the handoff mechanism exactly as predicted; mixed
REGRESSES (0.853->1.436 mapping, 155->434 holds): on a gauge-warped
self-built map, adjacent segments carry INCONSISTENT gauges and best-of-K
mixes them (gauge mixing, not thrash).

ADOPTED RULE (v7 contract): K=3 whenever the map is gauge-coherent
(preloaded/frozen — the 5-min demo readout mode), K=1 while self-mapping
a drifting map. RTL cost: K sweeps/kf (3 x 245 cands x 488 cyc ~ 0.36 M
cyc ~ 15 ms @24 MHz — real-time at kf rate) + one 32x6 multiply for the
hysteresis bonus. last_aid tracked incl. relock path.

## 2026-07-12 — line-prior map cleanup (matching pursuit with exact line atoms): reads the store at 5 cm; GT-frame precision is WARP-bound; per-segment decode is mandatory (scratch_lineprior.py)

User directive: "try cleanup methods (e.g. a line prior)". Method: peak-
guided matching pursuit with EXACT line-segment atoms (closed form
A_d = e^{i W_d.c} sinc(W_d.h)); greedy subtract of the LS-projected atom;
stop at tau x first peak. Scored: line precision (sampled line points ->
nearest GT wall) + recall (GT wall length within 0.15 m), school8
fixture, span11x60 recipe store (+oct4x60 control), float + 2b arms.

1. GLOBAL-bundle pursuit (301 segments superposed — past the knee) is the
   WRONG object: prec p50 0.42-0.54, recall 0.05-0.20. The peaks are the
   banked low-AUC raster's crosstalk.
2. PER-SEGMENT pursuit (deploy-shaped: each streamed segment decoded in
   its own +-4 m domain, 5-6 scans ≪ knee): recall 0.68-0.72, prec p50
   0.283-0.393 (float/2b). +CROSS-SEGMENT CONSENSUS (keep lines
   corroborated by another segment within 0.20 m/15deg): prec p50 0.226
   float / 0.295 2b at recall 0.59-0.60.
3. Fine atom refine (perp offset x orientation micro-search) moved p50
   only 0.296->0.283 — placement quantization is NOT the limiter.
4. DECOMPOSITION (the decisive check): the same consensus lines scored
   against the map's OWN gauge (input-scan cells): p50 0.048 m. The
   0.226 GT-frame figure is the fixture's store WARP (ATE 0.954), which
   bounds every readout equally. The pursuit reads the store at 5 cm.
   The p90 tail (~1.4 m, BOTH frames) = genuine junk atoms (secondary/
   clutter fits) — amp-threshold/refit filters are the filed follow-up.

VERDICT: line-prior cleanup WORKS as a parametric readout — ~2.9k lines
≈ 50 KB replace the raster view at 5 cm store-gauge precision and 0.6
recall, and it composes with the fidelity stream (per-segment decode on
the laptop). It does NOT beat the banked visibility readout's 0.150
GT-frame p50 on a clean live map — that comparison needs the same map
gauge (live classroom ATE 0.008 vs fixture 0.954); on equal footing the
methods are complementary (visibility = first-return points; pursuit =
parametric lines + compaction). Adoption: laptop display-layer option;
webvis/live integration + junk-tail filters filed. Ledgers:
scratch_lineprior*.log (3 runs).

## 2026-07-12 — sample-replay-augmented frontend on REAL data: median wins are real and transfer; closure-graph variance and the aliasing wall forbid default adoption; per-environment OPT-IN banked (scratch_replayfront.py)

User observation (webvis): the per-frame sample replay made the sandbox
system more stable. Ported to the pipeline as ReplaySLAM (subclass;
shipped core untouched): per kf, replay one random reservoir slot —
re-match against local_bundle with the frontend matcher + frontend gate,
refine the stored pose in place on strict correlation improvement, fold
at the stored pose (weight 0.3) into the nearest segment; save snapshots
p=0.25/kf with the refined slot protected from overwrite. Arms: OFF
(gate), R0 (match-only control), R1 (folds restricted to the frontend
recency window — the shipped philosophy), R2 (unrestricted cross-pass).
N=16, one fixed config, rseed 11 (+12/13 band on fr101).

GATES: OFF reproduced the banked suite EXACTLY on all four logs (fr101
1.881/53, fhw 0.981/559, belg 2.644/1, stata 0.202/99). R0 ≡ OFF on all
four — the extra matching + slot refinement alone is a true no-op; folds
are the only active ingredient.

  fr101: OFF 1.881 | R1 band {1.396, 1.648, 4.412} (med 1.115/1.303/
         1.492 — BETTER than 1.551 in EVERY draw; loops 28/55/52)
         | R2 2.461 (loops 57)
  fhw:   OFF 0.981 | R1 0.241 (med 0.841->0.218 — 4x) | R2 0.701
  belg:  OFF 2.644 | R1 4.283 | R2 2.324 (med 1.859->1.274; 1-9 loop
         noise log — single draws)
  stata: OFF 0.202 | R1 0.246 (loops 99->83) | R2 14.949 (loops 99->8
         — CATASTROPHE)

VERDICT: NOT a default — the multi-log gate fails (flagship stata loses
mildly at R1; fr101 carries a 4.4 tail draw; belg single-draw loss).
BANKED AS PER-ENVIRONMENT OPT-IN (same status as hex-for-belgioioso):
R1 on open, dense-closure, low-aliasing halls — fhw-class, exactly the
webvis sandbox regime — is a 4x lever.

Mechanism (three parts): (1) The median wins are REAL and transfer —
replay folds improve typical frontend registration on fhw and on every
fr101 draw (the user's webvis observation, confirmed on real data).
(2) The cost channel is CLOSURE-GRAPH INTERFERENCE: replay-folded
content enters segments at replay-time gauge; graph corrections assume
per-segment gauge purity, and the mixed content shifts candidate/veto
statistics — closure counts destabilize (fr101: 28-55 across replay
seeds; the tail draws are closure-pattern draws). fhw tolerates it
because 400+ dense closures re-pin the gauge continuously. (3) R2's
stata catastrophe is the VERIFICATION WALL in system form: unrestricted
cross-pass re-folding is un-vetoed cross-pass alignment; on the aliased
flagship it fuses wrong-pass content and collapses the closure graph
(99->8). The frontend-recency law (old passes reach the pose only
through gated loop edges) gains its strongest confirmation — 74x.

Follow-up in flight: w-dose probe (RES_W 0.3->0.1 globally on fhw+fr101)
— does the median win survive with the closure-variance suppressed?
Ledger: scratch_replayfront.log (all arms, gates, bands).

### n1 suite — the two open cells were already in the resumed ledger; open item CLOSED (scratch_n1suite_school.out, _band_fhw.out)

The banked n1 verdict ("dense-head deploy option, missing school-s12 +
fhw band") is completed by cells the resumed run had already produced:
school8-p0-s12: b2 0.749/115 loops/prec 0.62 vs n1 0.682/113/0.62 — n1
equal-or-better on the second seed (dense-regime safety confirmed;
s11's precision lift 0.53→0.72 stands as the bigger effect). fhw band
(point/interp family): n1 {1.669 eps0, 2.022, 1.200, 2.811} vs b2
{2.837, 2.921, 2.543, 2.801} — n1 sits below b2's ENTIRE band in 3 of 4
draws, the one sparse-family log where n1 helps as a band. AMENDED
regime read: dense (≥800 t/s) = safe/better incl. both school seeds;
sparse = worse (belg band-separated, fr079, fr101) EXCEPT fhw (156 t/s,
largest knee margin AND largest basin swing — knee-distance stays
necessary-not-sufficient). NOT-suite-safe verdict unchanged; the
dense-head option is now fully evidenced.

### v7 RTL sim-gate fixture: generated and round-trip VALIDATED (ice40/host/solo_vectors.py)

The top_solo acceptance fixture now exists in the v3-v6 shape: from the
golden on a deterministic classroom feed — build/solo_map.hex (26 segs,
SPRAM image, 4 codes/byte LSB-first), solo_anchors.hex (16 B/entry:
x_u i32, y_u i32, ah_q i16, cq i16, sq i16 — cs values proven equal to
cs_of(ah)), solo_feed_k{1,3}.bin (220 kf: n_pts + (az,r_mm,w) points +
pred pose) and solo_expect_k{1,3}.bin (pose + ai + score + state,
17 B/record). VALIDATION: a fresh SoloTracker fed ONLY from the file
images reproduces both expect streams BIT-EXACTLY (220/220 kf, K=1 and
K=3). The RTL testbench consumes these directly; any deviation is a DUT
bug by construction.

### v7 RTL stage 1 — resident-map segment addressing: SIM GATE PASS (encoder_solo.v, tb_solo_bridge.v, solo_bridge.py)

encoder_solo.v = the silicon-accepted v6 core + exactly 2 lines (mc_seg
port; mc_a = {mc_seg, ...} for both write and prefetch paths) — SPRAM
layout {seg[5:0], ring[1:0], j[5:0]}, one code/word, 64 segments/SPRAM
(4-codes/word packing filed). Gate: encode a fixture scan (from the
validated solo vector set), preload segments 8 AND 9 at their bases via
the mc_we path, run 8 candidates interleaving the two segments:
encode readback 240/240 bit-exact vs golden Q; all 32 per-ring partials
bit-exact vs ssp_ice40.match_int with the respective segment's codes —
the base switch is exact. Two tb lessons (documented in the tb): (1) mc
preload must use the {ring, j} 64-stride layout, not linear 0..239 (the
x-at-j>=48 signature); (2) after `clear`, wait !busy — start is only
honored in S_IDLE (the dropped-point-0 signature: every component off
by ~one point's mass). Also characterized: ring 3's final accumulate
lands via the S_MD drain ON the m_done edge; single-shot matches with a
halted engine are fine for reads AFTER m_done, and the tb runs
candidates hot (2x back-to-back) matching deploy batch semantics.

## 2026-07-12 — chain-cap-8 validation bands: the stata-lean STABILIZER is REAL (med 1.48→0.26 on fresh draws); spot no-op; fr101-lean mild cost — per-config option, not a lean default (scratch_postsuite.log)

Open item closed with fresh eps-bands ({0, 1e-6, 1e-3 s1, 1e-3 s2}) via
the CapSLAM harness (selftest: cap=None ≡ parent BIT-EXACT, shipped +
lean, re-proven):
  fr101-lean: none {1.132, 2.748, 1.670, 3.080} (med 2.21) vs cap8
    {2.144, 2.808, 2.188, 2.921} (med 2.50) — band NARROWS (spread 1.95
    →0.78) but loses the good draws; no benefit.
  stata-lean: none {2.122, 2.988, 0.835, 0.170} (med 1.48) vs cap8
    {0.288, 0.225, 0.133, 0.305} (med 0.26, max 0.31 vs 2.99) — the
    harvested stabilizer claim REPRODUCES on fresh runs; first knob that
    tames the stata-lean cascade.
  spot-lean: {0.036-0.037} both arms, counters near-identical — chains
    never reach 8 on the target platform; do-no-harm.
ADOPTION: cap=8 = a per-config stabilizer for aliasing-heavy LEAN
deployments (stata-class); NOT a global lean default (fr101-lean band
floor 1.13→2.14). Matches the banked capacity story (chain sums past
K≈8-16 at 2b dilute the loop-attempt matched filter on aliased maps).

Addendum to the replay-frontend entry — w-DOSE INVERTS BY LOG:
  fhw  R1 w=0.1: 1.032 (≈OFF 0.981; the 4x win at w=0.3 EVAPORATES)
  fr101 R1 w=0.1: 1.231, med 0.942 (BETTER than OFF 1.881/1.551 and
    better than every w=0.3 draw {1.40-4.41})
No global dose wins both logs — the per-environment opt-in verdict
stands, now with (environment, dose) as the option surface. Mechanism-
consistent: fhw's win is content-mass-driven (needs full folds), fr101's
is registration-refinement-driven (small nudges suffice; large folds
destabilize its closure graph).

### v7 RTL stage 2 — solo_tracker.v: FULL STEP LOOP 220/220 BIT-EXACT (sim gate PASS)

The on-fabric tracker FSM (rtl/solo_tracker.v, ~500 lines: nearest-anchor
serial d² scan over the anchor EBR; Q15 rel/pose transforms; 7x7x5
candidate grid through the encoder_solo core with one edge-recenter
retry; tot EBR + first-max argmax in golden candidate order; integer
parabolic refine via a UNIT-PROVEN serial trunc divider; EMA commit gate
29/64 shift-add; wrap960 heading branch select; hold path) replays the
entire 220-kf classroom fixture BIT-EXACTLY against solo_expect_k1 —
poses, segment ids, scores, and states, 11 holds included
(tb_solo_step.v + solo_step.py; per-kf sh golden-fed in this pass).

Three RTL defects found and fixed by unit isolation + golden-internal
reconstruction (all banked as classes): (1) serial multiplier OR'd
overlapping partial products (unit test caught; replaced with a
registered combinational multiply for the semantics gate — any LC-lean
serial substitute must pass the standalone MUL test first); (2) unsigned
part-select poisoning — $signed() required on mul_p[31:0] in
expressions or >>> degrades to a logical shift (two sites; additions
are immune, shifts/compares are not); (3) registered-RAM read pipeline
off-by-one — address at phase n is readable at n+2, and the pc capture
one cycle early silently returned pa (making every parabolic offset 0;
found by reconstructing the golden y-triplet). Plus one tb-only lesson:
$readmemh silently truncates an undersized memory (feed array), turning
later keyframes into x — size arrays to the fixture, not the estimate.

Stage 3a (on-chip shift_for) implemented behind SH_ONCHIP: OR of |Q|
over a 240-entry readback scan has the same bit-length as max|Q|, so a
priority encode reproduces shift_for exactly (~245 cyc/kf); gate rerun
in flight. Remaining for top_solo: re-search FSM (parametric grid reuse),
K_TRY=3 loop, UART top + preload/pose-stream commands, then build + hw.

Addendum: stage 3a (on-chip shift_for) VALIDATED — 220/220 kf bit-exact
with SH_ONCHIP=1 (scratch_solostep2.log). The v7 chip-side step loop is
now fully self-contained: raw points + odometry pred in, pose out, with
encode (v6 silicon-accepted), resident-map matching (stage 1), the
tracker FSM (stage 2), and the shift scan (3a) all golden-gated.

## 2026-07-12 — 10h autonomous block: consolidation

User directives: replay stabilized the webvis — real-data experiments;
cleanup methods (line prior); rework docs to the deploy corner goals;
various experiments; 10h, no agents. All threads ledger-verified; OFF/
control gates reproduced the banked suite exactly wherever applicable
(9 gates total, all green).

BANKED THIS BLOCK: (1) replay-frontend suite — median wins real and
transfer, fhw 4x at w=0.3, fr101 prefers w=0.1 (1.231/0.942), stata-R2
catastrophe = the verification wall in system form; per-(environment,
dose) OPT-IN, not default. (2) line-prior pursuit — per-segment decode
mandatory; 5 cm own-gauge readout; GT-frame precision is store-warp-
bound; webvis integration (cLines). (3) v7.1 best-of-K handoffs —
strict localization win (beats the live host tracker), gauge-mixing
regression while self-mapping; K=3 = coherent-map mode. (4) chain-cap-8
— stata-lean stabilizer REAL (med 1.48→0.26), spot no-op, per-config
option. (5) n1 open cells closed from the resumed ledger. (6) docs
reworked: FINDINGS A5-A9, corner goals (dual-space + SOLO corner),
README silicon status, EXPERIMENTS catalogue, no-gos extended. (7) v7
RTL: stage-1 resident map, stage-2 full tracker FSM, stage-3a on-chip
shift scan — ALL 220/220 bit-exact in sim; fixtures round-trip
validated; three RTL defect classes banked (OR'd partials, unsigned
part-select >>> poisoning, registered-RAM read off-by-one) + the
$readmemh truncation tb trap. REMAINING top_solo: re-search FSM, K=3
loop, UART top, build vs LC budget, hw-replay — spec pinned, fixtures
and gates in place. Shipped core untouched throughout; nothing
committed (user commits).

### v7 RTL stage 3b — re-search + RELOCK: 160/160 BIT-EXACT on the kidnap fixture

The grid FSM parametrized (fine 7x7x5/T=12 vs wide 15x15x7/T=48/rho-step
2, argmax-only — one machinery, mode flag), lost12 counter with the
every-12th-hold trigger, relock gate 35/64, grid-commit relock pose.
Fixture: a 320 U diagonal kidnap kick at kf 60 on the classroom feed —
the golden absorbs 0.25 m kicks through edge-retry + basin re-registration
(robustness note: fine tracking swallows kidnaps below ~0.3 m), 320 U
lands in the (absorption, re-search-reach ±336 U) window and produces
116 tracking / 42 holds / 2 RELOCKS. Gate: 160/160 kf bit-exact incl.
both relocks (solo_feed_rl / solo_expect_rl; scratch_solorelock.log).
Two width/parametrization defects found by the gate: bix/biy 3-bit
truncation at wide indexes (compile-silent), and the relock HEADING
recompose using the fine grid's bir-2 instead of bir*rstep-Roff — the
failure fingerprint (relock-only, heading-only, exactly +4 rho with
poses and scores exact) identified the site immediately. K1 220-kf
regression rerun after the edits: in flight (fine-path constants
identical by construction).

The v7 chip-side tracker contract is now FULLY sim-gated: encode (v6
silicon-accepted) + resident map + tracker FSM (all states, both grids)
+ on-chip shift scan. Remaining for top_solo: K_TRY=3 outer loop, UART
top (preload/pose-stream/map-dump), yosys/nextpnr build vs LC budget,
hw-replay.

### Cleanup pass + web-demo rework (2026-07-12, post-block; gate-verified behavior-neutral)

k1 regression after stage-3b: 220/220 PASS (the relock edits left the
classroom gate untouched). CODE CLEANUP: solo_tracker.v header rewritten
to the gated reality (stages 2/3a/3b, parametric grid, wide-aware relock
heading) + dead regs removed (pb unused, committing write-only);
tb_solo_bridge's stale ring-3-timing note corrected (the x was the
preload-addressing bug; ring 3 lands on the m_done edge — double-run
kept as a behaviour-neutral hot-batch exerciser); solo.py header carries
the gate status + the stale no-refine line fixed; solo_step.py usage
documented (suffix arg). POST-CLEANUP GATES: solo selftest ok, k1
prefix 8/8, relock fixture 160/160 — all bit-exact. WEBVIS REWORK:
sandbox controls reorganized into "Encoder" (thermometer/point/hex/
angles) + "Deploy recipes (banked 2026-07-12)" (ladder incl. refuted-phi
showcases, segment-faithful binary FPGA store, sample replay) with
verdict-carrying tooltips; replay-fold legend chip added to the world
caption; parse + 46 ids green throughout. live.html deliberately
untouched (hardware-facing UI, validated in use — churn without a
hardware retest is risk for no gain).

## 2026-07-12 — cleanup formulations program, part 1 (P1/P2/P4): two both-log WINS (3b dead-zero store; coherence-weighted read), one refuted prediction (amplitude pruning), one dial (CLEAN gain) — scratch_cleanups.py

Program from the cleanup-formulations synthesis (taxonomy + 5 laws).
GATE: the pursuit baseline reproduces banked EXACTLY (raw 3449 ln p50
0.283 rec 0.68; +consensus 2942/0.226/0.60; own-gauge 0.048) — after
catching a PARAMETER-SHADOWING defect my gain edit introduced (a local
`gain = amp²·na²` energy variable shadowed the new loop-gain parameter,
turning deflation into res -= energy·amp·atom; all arms bit-identical =
the fingerprint; banked defect class: parameter shadowing in hot loops,
caught by the reproduction gate doing exactly its job).

P1 (CLEAN loop gain + OMP refit) — PREDICTION REFUTED, DIAL BANKED:
  gain=.25 (k=48): +cons 9892 ln, p50 0.308, rec 0.71 — a recall/
    precision DIAL (+0.11 recall, −0.08 precision, 3.4x components),
    not a win; own-gauge unchanged (0.049).
  OMP joint refit + prune 0.15·median: ZERO atoms pruned — bit-identical
    to baseline. The junk tail carries HEALTHY joint amplitudes (clutter
    energy): with micro-refine already nil, the tail is now proven
    MODEL-limited — no amplitude or placement criterion can remove it;
    competing atom types (point/corner, P5) are the only lever left.

P4 (dead-zero store) — BOTH-LOG WIN; 3b BEATS FLOAT ON BOTH LOGS:
  (full-map imaging protocol; local32 med/p90 alongside)
  school8: 2b .5662 -> mask@.10 .5705 -> mask@.20 .5846 -> 3b(nmag2,
    dead) .6065 — ABOVE float-full .5984; local p90 .157->.112 at 3b.
  stata:   2b .6553 -> .6566 -> .6593 -> 3b .6691 — ABOVE float .6677;
    local med improves at mask@.20 (.020).
  Mechanism: nmag=1 flattening has NO zero level (dead=True divides by
  zero there — the deploy-shaped fix is a D-bit liveness mask at
  +D/8 B/vec, or one true-zero magnitude bit = 3b/comp). Zeroing
  sub-noise components removes junk phasors that even the FLOAT store
  carries — quantized-with-liveness > float, both fixtures.

P2 (coherence-weighted Wiener read) — BOTH-LOG WIN; the banked static-
reweighting negative is SUPERSEDED by its adaptive form:
  GT-free observable: per-ring mean |cos| over overlapping world-frame
  segment pairs (<4 m). school8 profile has the warp signature (fine
  rings lowest: .47/.41); stata profile lower overall (less pairwise
  content overlap in corridors — the observable conflates overlap and
  coherence; weights still land correctly).
  school8 AUCgt: raw .5984 -> w=Cr .6068 -> w=Cr² .6144 (+.016)
  stata:         raw .6677 -> .6716 -> .6730 (+.005)
  Regime-consistent: strong on the warped store, small-positive clean.

STACK (P4+P2, mechanism independence confirmed ~additive):
  school8 .5984 float -> .6212 (3b-dead + Cr²; F1 .166->.187, +13%)
  stata   .6677 float -> .6751
READ-SIDE RECIPE for the fidelity stream: 3b dead-zero freeze +
Cr²-coherence-weighted imaging (+ per-segment pursuit for parametric
views). P3 (primitive-space gauge relaxation) in flight; P5 (corner/
point atoms) is the tail lever. Ledgers: scratch_cleanups*.log.

### P3 (primitive-space gauge relaxation): NEUTRAL, mechanism banked

1646 line correspondences over 301 sequence-adjacent segment pairs;
damped LS corrections come out TINY (|t| med 0.037 m, p90 0.088; |dth|
p90 3.0 deg) and GT-frame precision is unchanged (0.226->0.233, recall
0.60->0.61; own-gauge preserved). MECHANISM: adjacent segments already
share their gauge (built consecutively against the same local map) —
their line-to-line residuals are pursuit noise, not warp. The store's
warp is SLOW (accumulates across passes); only long-range/cross-pass
constraints see it, and those are exactly what the frontend-recency law
forbids without verification (the wall, from the primitive side). Warp
correction remains the closure backend's job. The machinery survives as
a MERGE tool (aligned-line fusion for display), not a corrector.

## 2026-07-12 — cleanup program part 2 (P5/P7): dictionary expansion NEGATIVE (energy-greedy can't model-select); the DRIVEN-PATH VETO is the tail-slayer — integrated into decode.py and the webvis (scratch_cleanups.py p5/p7)

P5 (point + 90-deg corner atoms competing in the energy selection) —
NEGATIVE as formulated: points claimed ~nothing (7/301 segments —
energy selection structurally favors extended atoms; a point can never
out-score a line on amp²·‖a‖² against extended residual, so greedy
energy CANNOT do model selection without a complexity penalty); corners
claimed 640 but degraded everything slightly (cons 0.234/1.508/0.58 vs
0.226/1.413/0.60). Survivors: complexity-penalized selection filed;
corner atoms remain analytic and available.

P7 (driven-path veto — the physics-prior family, cleanup law 2): a line
crossing the anchor trail cannot be a wall (the robot swept that
space). Post-filter on the baseline pursuit:
  school8: +cons p50 0.226 -> 0.087, p90 1.413 -> 0.731, recall
    0.60 -> 0.59 (764 junk lines died; own-gauge 0.048 -> 0.042) —
    the tail that survived amplitude pruning, micro-refine, and
    dictionary expansion collapses under the physics prior.
  stata: p50 0.110 -> 0.101, rec 0.56 -> 0.54 (caveat: stata line-p90
    is GT-COVERAGE-limited — floorplan GT covers ~24% of kf — so the
    school full-coverage numbers carry the evidence).
The cleanup-law reading holds exactly: the tail was never a FITTING
problem; it needed information the correlation doesn't have, and the
cheapest independent source was in the stream all along (the trail).

INTEGRATIONS (user directive):
- ice40/host/decode.py: path_veto(anchor trail) before consensus —
  v7 classroom chip-dump decode improves p50 0.232 -> 0.105, p90
  1.879 -> 0.615, recall held (10 cm walls from a PHASE-ONLY oct60
  dump).
- demo/index.html: the line-prior overlay now draws the CONSENSUS +
  PATH-VETO view (replay trail = IN.poses, dirty-flagged recompute;
  tooltip carries the banked numbers; parse + ids green).
Read-side recipe final form: per-segment pursuit -> path veto ->
consensus (+ Cr² Wiener imaging + 3b dead-zero store from part 1).

### v7 build session part 1 — corner-tuned silicon: the lean core, the scale-tuned freeze, and the LUT wall (2026-07-12)

User directive: "time for the v7 build session — tune towards the three
corners of the opt triangle to necessitate solutions across the spectrum
and then consolidate into our final demo", plus "follow up on ... the
3-bit freeze" and "make sure to tune scales".

FREEZE STORE TUNED (scratch_liveness.py, school8+stata x span11/oct4,
gate: banked P4 rows reproduced to the digit). The dump-faithful arms
(unit phasors, no der — exactly what decode.py consumes) separate the
banked 3b-dead winner into its three ingredients:
  - liveness alone (int rule: M = max|I|,|Q| + min>>1; alive iff
    2M >= Mmax_ring): AUCgt school-oct4 .5097 -> .5579; theta sweep
    1/4..5/8: 5/8 wins school but LOSES stata both lattices -> theta
    = 1/2 by the multi-log rule.
  - ring scales alone (Mmax_r amplitude): ~nil (.5097 -> .5142).
  - liveness + scales: BEATS THE FLOAT STORE on extraction on both
    fixtures (school span11 .6138 vs .5984 float; stata .6714 vs
    .6677) — entirely integer, from the same freeze scan, 30 B + 8 B
    per segment on the wire (scales as 5-bit exp + 8-bit mantissa,
    serial normalize).
  - caveat banked: the mask degrades LOCAL matcher reads (p90 .120 ->
    .183 at theta=1/2, .373 at 5/8) — liveness/scales are DUMP-side
    only; the on-chip matcher keeps raw 2b codes.
Golden: solo.liveness_int (+ mapper freeze plane); RTL: freeze pass A
(mcode + per-ring Mmax via the T_SH-style scan) + pass B (compare +
bit-pack) + pass S (serial exp/mant) in top_solo.v.

REFINE A/B (scratch_v7refine.log, classroom): localization med
identical, p90 +2 mm without the divider; SELF-MAPPING IS BETTER
WITHOUT IT (pass-1 0.072->0.051 med, replay 0.073->0.055) — the
sub-grid offsets appear to inject gauge jitter into the fold. The lean
build drops the divider + tot RAM (parameter REFINE=0; single-env
evidence, classroom).

CS_OF TABLE-DEFINED (defect found by the exhaustive fold check in
gen_luts): the float form round(32767*cos) is NOT quarter-symmetric at
exact-half points (np.sin(pi/6) = 0.4999999999999999 vs np.cos(pi/3) =
0.5000000000000001 -> 16383 vs 16384). Golden cs_of is now DEFINED as
the folded 241-entry table (== the RTL ROM structurally); deltas vs the
float definition: 5/960 headings, +-1 LSB. Fixture regen + benches
re-run under the new contract.

THE LEAN CORE (encoder_lean.v — the S-corner solution the corner
tension forced): the v6 4-ring parallel pipeline is a T-corner design
(488 cyc/cand, 6.6k poses/s) where the solo task needs ~250 cand/kf at
5 Hz — 500x less. Serializing the rings (ONE 256x32 acc memory pair,
ONE weight tree, 2 shared MAC16 for r*az / fold rotate / u-projection /
match products, az ROMs quarter-wave folded 8 EBR -> 1 with EXHAUSTIVE
fold equality checks in gen_luts.py) gives:
  encoder: 3361 -> 2034 LUT, 28 -> 6 EBR, 8 -> 2 MAC16;
  ~11 cyc/angle encode (~668/pt), ~23 cyc/angle match (~1385/cand) —
  budget at 24 MHz / 5 Hz = 4.8M cyc/kf >> ~2.1M used (encode x2 +
  sweep + freeze).
Tracker reworked to match (16x16 saturating mult == golden clip, ONE
MAC16; grid offsets and *12 by shift-add; REFINE param; quarter-wave cs
ROM + SE2 service CS/ROT/IROT for the top). GATES: old fixtures k1
220/220 and rl 160/160 (regression, old core + edited tracker);
lean-core stage-1 bridge 32/32 + encode readback 240/240; lean l1
(REFINE=0 fixtures, table cs_of) 220/220 BIT-EXACT. Kidnap (lr) gate +
top-level gate in flight at time of writing.

CORNER BUILDS (yosys/nextpnr/icetime, UP5K):
  T (v6 deploy, top_match3): 5142/5280 LC (97%), 30/30 EBR (100%),
    8/8 DSP — maxed on EVERY axis; icetime 30.22 MHz (24 closes);
    encoder-only build 40.32 MHz (36 closes). 488 cyc/cand.
  S (lean core probe): 2694 LC, 6/30 EBR, 2/8 DSP, 1 SPRAM; 26.06 MHz
    (24 closes). The datapath inverts the v6 resource profile.
  F (full standalone SLAM top_solo: ingest+track+fold+freeze+dump):
    DSP 3/8, EBR 9/30, SPRAM 3/4 — comfortable; LUT 7452/5280 = 1.41x
    OVER. Attribution: encoder 2032 + tracker 2955 + top 2659 + uart
    54. Hand-CSE of the inlined helpers bought ZERO (abc9 already
    shares) — the mass is genuinely the ~26+33-state control FSMs'
    scalar arithmetic and register-enable fabric.
THE FINDING: the VSA datapath is NOT the wall — the scalar SE2/control
plumbing is (tracker+top = 5614 LUT vs 2090 for encoder+uart+pll).
Even localization-only (tracker+encoder = 4987 + any top) exceeds the
device as parallel-word FSMs. The closure path writes itself: the lean
core freed EXACTLY the resource (EBR 9/30, SPRAM 3/4) that a
microcoded scalar engine spends (~650 LC engine + ucode in EBR, est
~2.9k LC total system) — control must move from LUT fabric into the
abundant memory. Filed as the v7.2 fit plan; the full-function RTL is
sim-proven meanwhile (the gates above), so the demo runs bit-exact in
sim and the v6 deploy build remains the flashable hardware today.

### P5 successor — complexity-penalized atom selection: the mechanism confirmed, the trade measured (2026-07-12)

User follow-up on the banked P5 negative ("energy-greedy can't
model-select without a complexity penalty"). scratch_p5pen.py, same
dictionary (line / 90-deg corner / point) and grids as P5, penalty in
the SELECTION only (deflation keeps raw amps); school8 + stata; the
en arm reproduces banked P5 to the digit (gate).

  arm      school +veto p50/p90/rec     stata +veto p50/p90/rec   types (school)
  en       .106 / .762 / .56            .103 / 4.899 / .53        7 pt / 640 co
  en/df    .080 / .759 / .53            .101 / 4.803 / .47        762 pt / 474 co
  en/sup   DEGENERATE (0 lines)         DEGENERATE                3612 pt / 0 co
  aicc     .104 / .761 / .56            .104 / 4.790 / .52        182 pt / 615 co
  (banked line-only + veto: school .087 / .731 / .59; stata .101 / .54)

VERDICT: the P5 mechanism claim is CONFIRMED — dividing energy by df
activates real model selection (points 7 -> 762, corner spam halves)
and buys the best school precision seen (.080). But the P5-predicted
risk is also now measured: points siphon wall amplitude — recall drops
~6 points on BOTH logs. No dominance -> LINE-ONLY + PATH-VETO REMAINS
THE DEPLOY RECIPE; en/df goes in the ledger as the honest
"selection-sanity dial" (use when precision >> recall). en/sup
(support-normalized energy) is a NO-GO FORM — energy density always
favors the smallest atom (all-points collapse, both logs). aicc at
lambda = 2*df*sigma^2 is numerically inert (penalty dwarfed by real
atom energies).

### v7 build session part 2 — the top gate closes; webvis inits on the deploy recipe (2026-07-12, push marker)

TOP-LEVEL END-TO-END GATE (solo_top.py, 8-kf smoke): after two defect
classes were run down, **8/8 pose frames AND the 116-byte map dump
BIT-EXACT** through the full UART protocol (set-pose, mode, keyframes
with points+d_q, pose frames, dump with codes+liveness+scales+anchors).
The two defects (both now banked classes, isolated with tb-level SPRAM
peeks after the pose-trace localized the divergence to "first tracked
kf after the first freeze"):
  1. PHANTOM KEYFRAME: K_TXP returned to K_IDLE on the same edge it
     pulsed kf_ack; the parser clears kf_hdr a cycle later, so K_IDLE
     consumed the STALE flag and started a phantom keyframe each time —
     mid-run it self-synchronized into the next real keyframe (masking
     itself), after the last one it hung the dump. Fix: K_KFEND
     handshake-completion state.
  2. DRAIN-RESET RACE: drain_done was evaluable while drain_rst was
     still in flight, so rd_pt still held the TRACKING pass's count and
     the FOLD replay "completed" instantly — every segment froze an
     all-zero accumulator, i.e. all-code-0 (the +1-phasor "DC map"),
     which PSEUDO-TRACKS (kf5 came out only ~6 cm off!). The honest
     tell was the dump: real anchors, 0xFFFF liveness, zero codes.
     Fix: !drain_rst in drain_done.
Regression after both: smoke PASS bit-exact incl. dump; decode.py
consumes the chip-written planes (liveness ON / scales ON autodetect;
old-dump path regression-clean at the banked 0.105/0.615/0.42).
lr (kidnap) LEAN GATE: **160/160 keyframes bit-exact -> PASS** (landed
just before the push; an earlier partial 75/75 was my stray PID kill —
lesson: check command args before kill). The lean-stack gate suite is
now COMPLETE: bridge 32/32 + readback 240/240, l1 220/220, lr 160/160,
top end-to-end incl. dump. IN FLIGHT at this push: the 220-kf full
classroom tour + its dump -> decode.py numbers (part-3 entry when it
lands).

WEBVIS: now INITS ON THE DEPLOY RECIPE (user directive) — span11
ladder @ 60 angles, segment-faithful binary FPGA store (int8 tables
baked at start via mkTables(cfg.binary), same order as the cBin
handler), sample replay ON; cfg defaults == DOM attributes == start
sequence (asserted in the validation gate); the stale nAng=113 default
(the refuted phi-prime count) is gone. Tooltips carry the freeze-store
tuning, corner-build numbers, and the P5+ verdict. Parse + 52 ids
green.

Cleanup-pass note: last session's archive move of scratch_capacity2.py
broke 12 importers (make_wvs/rebuild_store); restored to the root this
session — archiving load-bearing scratch modules is now a known
cleanup-pass defect class.

### v7 build session part 3 — the 220-kf standalone demo closes bit-exact; chip-built map decodes at ~11 cm (2026-07-12/13)

THE FULL GATE: 220-kf classroom tour through the UART top — **220/220
pose frames AND the 5018-byte 44-segment map dump BIT-EXACT** vs the
golden. En route it caught one more defect (now banked as a class):
live_base decomposed seg*15 as 16s-4s (=12s) instead of 16s-s —
segment liveness regions overlapped by 3 words, each freeze clobbering
its predecessor's tail. The fingerprint was surgical: poses 220/220,
codes/scales/anchors clean, ONLY liveness words 12..14 wrong in every
segment. Single-segment smokes structurally cannot see stride bugs —
multi-segment (>=2 freezes) is now the minimum dump gate (15-kf
3-segment gate added, bit-exact). Address-helper audit after the fix:
c60 (64k-4k), mul3 (2n+n), gmul (8v+4v), th29/th35, div*12 all correct
— live_base was the only miss.

THE LAPTOP SIDE (decode.py on the CHIP-SELF-BUILT map — no python SLAM
anywhere in the loop; 44 segments, self-mapped poses, chip-frozen
codes/liveness/scales):
  raw 2b dump:      p50 0.112  p90 0.909  recall 0.45   C_r .20/.31/.46/.55
  +liveness+scales: p50 0.200  p90 0.953  recall 0.56   C_r .31/.42/.58/.49
Headline: the chip's OWN map decodes to ~11 cm median walls —
essentially matching the python-SLAM-built 26-seg fixture (banked
0.105) — the full standalone story holds end to end. The planes on the
LINE-PURSUIT self-score act as a recall/precision dial (+11 recall,
p50 0.112->0.200; per-ring C_r visibly cleaner .20->.31 fine rings) —
consistent with the banked framing: the liveness+scales win was
measured on the EXTRACTION/imaging metric (AUCgt, both fixtures);
pursuit line-centers prefer the raw store. Both readings ship in the
dump (planes are additive; the consumer chooses).

v7 acceptance status vs the pinned spec: (1) sim gate COMPLETE (all
stages incl. the full top + dump); (2) hw-replay — blocked on the
v7.2 LC closure (7452 LUT vs 5280, microcode plan filed); (3) the
5-min standalone demo runs bit-exact in sim with silicon semantics.

### Webvis: sample replay visualized on REAL DATA (2026-07-13)

User directive. The sandbox reservoir mechanism now also runs on the
real-data replay tab, on the JS-refolded panel map (RMR, own seeded
rng so it never perturbs the sandbox stream): per replayed keyframe,
one buffered snapshot is re-matched against the local segment bundle
(rmBundle + the panel's viz matcher) and re-folded — score-improved
refinements update the snapshot's stored pose; rehearsals fold at the
stored pose. Rehearsal mass goes into the ANCHOR-FRAME segment store
(rmReplayFold: nearest live anchor, rel = anchor^-1 o pose) so graph
snaps re-place it exactly like normal folds, and sv.nf++ lets the
line-prior overlay see rehearsed content. The replayed trajectory
stays authoritative (poses are recorded; only map content anneals).
Overlay: teal rings = refined, gray = rehearsal (fading 12-deep trail);
the replay counter shows improved/total for whichever mode is active.
Reset on cRes toggle, resetIntel/rmReset, and cBin rebuilds.

VALIDATOR DEFECT FOUND AND FIXED: the parse/id gate selected the
LARGEST <script> — which is the 25 MB replay DATA pack, not the 150 KB
core; this session's earlier "parse green" checks were validating the
data blob. The gate now keys on the __CORE_START__ marker and parses
BOTH scripts (core + data) with the SyntaxError-injection control; the
core (with all session edits) parses clean, 52 ids resolved. Defect
class: max-by-size selection of embedded scripts.

### Live demo reworked onto SPOT data; hw session notes (2026-07-13)

User directive ("rework the demo to run with spot data anyway"). live.py
gains SpotFeed (additive; the hardware pipeline untouched): the SPOT
Telluride tour (414 kf, 1024 beams on the fabric's exact az grid, 5 Hz)
replaces the synthetic raycast feed under env="spot". LIDAR-ONLY
posture preserved (anti-oracle): item['odom'] is a ZERO-MOTION chain
anchored at the reference's first pose, so pass-1 guesses degenerate to
the previous estimate; the withheld odometry rides along as item['gt']
for display/eval ONLY (gt_ok hygiene mask from the banked sorted-parquet
fix). Passes replay the same scans verbatim; no synthetic bridge (the
spot loop revisits its start). ssp_spot.DIR made cwd-independent
(live.py runs from ice40/host — the relative path broke the first
launch). Boot verified on hardware: fabric encode cross-checks green
from kf 0 (enc n/n), pass-1 tracking ~0.01 m vs reference early on.
Run: `python3 live.py serve 8642 1 spot`.

HW session note (user observation, classroom synth on the v6 board,
pre-rework): after a long run the live trajectory had drifted upward by
~half the orbit ellipse before the USB cable was pulled. Not diagnosed
(the session moved to spot data); filed — candidate mechanisms: map
staleness under the frozen classroom map across many replay passes, or
EMA-hold drift accumulation. The spot demo (real data, one tour +
replay passes) is the deployment-shaped configuration anyway.

### SPOT live loop semantics + the v7 RESET-MAP command (2026-07-13)

User directives ("are you continually running dataset (with an
interpolation at the end and perturbance of lidar from second loop
on)?" — it wasn't — and "interp to close loop properly"). SpotFeed now
mirrors the synthetic Feed's loop law on real data:
  - BRIDGE: the measured 0.381 m + 24.1 deg end->start snap is closed
    by 9 interpolated keyframes at the tour's own stride (0.038 m /
    2.4 deg per step — verified continuous). Real data cannot
    re-raycast, so bridge scans replay the nearest ENDPOINT scan
    CIRCULARLY ROLLED to the interpolated heading — rotation-exact on
    the 360x1024 head (the 24 deg discontinuity dies); residual
    parallax <= ~0.19 m on 9 of 423 loop keyframes.
  - PERTURBANCE from the second loop on: range noise (sigma =
    S.RANGE_NOISE) + dropout per (pass, kf) with the Feed's stream law
    (verified: max |dr| 0.067 m ~ 3 sigma, fresh dropouts) — the
    localization passes never see the bytes that built the map; pass 0
    maps the pristine recording.
  - carve_freemask generalized via feed.scan_of(i) (synthetic:
    verbatim rng regeneration; real: the recording) — the spot run had
    crashed at the freemask after a CLEAN pass 1 + freeze +
    calibration (0.2 cm / 0.14 deg residuals on real data).

v7 RESET-MAP (user: "so we can switch envs"): 0x29 -> n_seg=0, open
segment discarded, pose kept, ack 0xA7. GATE: 15-kf run with the reset
injected at kf 7 — golden restarts with a fresh tracker+mapper —
**15/15 pose frames + dump BIT-EXACT (dump contains ONLY the
post-reset segment)**.

### Fabric tracking on real SPOT data: the stop-and-pivot cascade, run down (2026-07-13)

The reworked live demo (SpotFeed) put the v6 board's frozen-map tracker
on real data for the first time and it DIVERGED on pass 2 (fx 0.34 ->
6.9 m) while the python SLAM held 5 cm on the same perturbed scans and
every encode cross-checked (enc n/n — silicon exact; the failure was
the tracking loop). The cascade, each step measured from the bench npz:

1. ZERO-VELOCITY STALL: lidar-only feeds have odom deltas == 0, so
   pred = last pose; the robot moves 0.043 m/kf; holds stall the chain.
   Fix: constant-velocity pred from the tracker's own history (clamped
   0.13 m — the banked spot posture). Result: locked 2-3 cm... until
2. CV RUNAWAY THROUGH HOLDS (kf-504 event): the robot BRAKES (gtstep
   0.13 -> 0.02) and the scene score halves (124 -> 46 AT 3 cm ERROR —
   scene-driven, not pose-driven); holds fire; hold-poses feed the
   velocity back to itself and the pred runs away at EXACTLY the clamp
   rate (+0.13 m/kf), then the robot PIVOTS ~55 deg in place; at kf 512
   a CONFIDENT WRONG LOCK 1.8 m off (EMA decayed; the corridor-
   degeneracy wall in tracking form, on real data).
3. Decay (0.5x/hold trans, 0.9x heading): rides the brake (err 0.02 ->
   0.13 -> 0.04) but the pred had slid 0.6 m before decay caught up —
   outside the re-search reach (+-0.35 m) — and the pivot exceeded its
   heading window (+-18 deg): drift resumed UNDER state=tracking, so
   relock never armed (fx-th settled ~167 deg).
4. GLOBAL-IN-HEADING RE-SEARCH (all 60 rho steps, stride 2 — position
   stays local, heading can pivot arbitrarily while stopped): necessary
   but not sufficient alone (the slid center still missed).
5. FREEZE-ON-HOLD (CV zeroed the moment a hold fires — score collapses
   coincide with stops) + wider re-search box (+-0.47 m): **fx RMSE
   5.31 -> 0.136 m, med 0.055, p90 0.224, max 0.489; heading p90
   19.7 deg; states 319 track / 105 hold / 5 relock — the tracker now
   rides the stop-and-pivot (err 0.02 -> 0.12 -> 0.03 through the
   event).** py SLAM same pass: RMSE 0.048 — the frozen-map 2b fabric
   tracker lands within ~3x of the full float SLAM on real data,
   LIDAR-ONLY.

Also added (inert safety net): a host-side pi-BRANCH AUDIT on commits
(conjugated-query score of the antipodal interpretation — the
half-circle lattice folds heading mod pi and the branch bookkeeping
can in principle slip during pivots; the conjugate-wrap physics
distinguishes pi). Never fires in the healthy configuration; costs one
python encode+match per committed kf. The final tracker posture: CV
while tracking, FREEZE on holds, every-12th-hold re-search global in
heading, local (+-0.47 m) in position.

### Live UI rework: env switching + reset map + stable readouts (2026-07-13)

User directives. live.py: the run loop gains a restart shell — handler
threads set _req_reset/_req_env only; the loop applies them between
keyframes (serlock held) via Live.reset_for (fresh feed/python SLAM/
frozen map/tracker state; the BOARD stays — per-scan encode clears it).
Endpoints /reset and /env?name={classroom,mixed,corridor,office,spot};
/state carries the active env. VERIFIED LIVE: /reset restarted mapping
kf 100 -> kf 0 with the encode cross-check counter continuing (26/26).
live.html reworked: env select + reset-map button (select auto-syncs to
the server env, guarded against fighting a pending switch); numeric
readouts monospace + tabular-nums with fixed chip widths and fixed
decimals (the top bar no longer jitters — the old font stack resolved
-apple-system BEFORE the mono fonts); pick deselection (Esc or click
outside the map — previously a pick was permanent); reference-walls
label now honest about real data (empty overlay). SSE reset event
clears the client view on env switches.

### Live demo: WASD driving for synthetic envs; map/view robustness (2026-07-13)

User directives. DrivenFeed (live.py): browser-held key set (/drive
endpoint sends the full pressed set on every change — robust to missed
keyups), motion integrated at keyframe rate (full speed = the scripted
tour's stride; A/D turn 1.6 rad/s), scans raycast from the driven pose,
Feed's odometry noise model, pass-1 scans recorded for the freemask
carve. The map freezes on the UI 'freeze map' action (/freeze pins
n_tour to the driven count; a calibration sample is captured at the
freeze keyframe so early freezes calibrate). env switch carries
drive=1 for synthetic envs (spot stays replay). VERIFIED HEADLESS via
endpoints: W = +2 m at cruise, W+A arcs with wrap-correct heading,
freeze -> calib 2/2 (~1 cm on the driven map) -> fabric image ->
localization tracking at 0.005 m.

Client robustness (user-reported): (a) reset/env-switch left the map
blank until a single SSE map_ready event — now the reset branch starts
loadMap's pending-poll (recovers regardless of event races) and the
canvas keeps drawing trails during the remap; (b) the view was fit
from whatever partial data existed at page load (different position
per reload, robot could leave the box) — replaced fitView/provisional
with a unified content-derived view (union of map/world/trails/robot,
padded, centered) that only refits when something leaves the box:
deterministic across reloads once the map is up, and the robot can
never wander outside the drawn area.

### Live demo hardening: early-freeze calibration crash (2026-07-13)

User-triggered crash: freezing a DRIVEN map at kf 5 left ONE calib
sample; it passed at 1 cm but the assert demanded >= 2 successes ->
AssertionError killed the server. Fixes: (a) threshold is now
sample-count-aware (1 sample -> 1 must pass; >= 2 samples keep the
original at-most-one-failure rule), (b) driven calib sampling densified
(every 60 kf from kf 30 + the freeze keyframe), (c) the demo server
SURVIVES a true calibration failure — loud [calib] FAILED line + auto
remap (board session and UI stay alive) instead of a dead process.
Verified: freeze at kf 24 -> calib OK 1/1 (0.9 cm) -> fabric tracking
at 1.5 cm on the driven map.

### Reservoir save cadence: every 20th frame (user directive, 2026-07-13)

"instead of rand 10% go for every 20 frames instead. still rand
overwrite and sample though. both on fpga and in websim." Applied in
all three reservoirs: websim sandbox (RESV) + websim real-data (RMR) —
RES_EVERY=20 deterministic save cadence with per-mechanism frame
counters (reset with their buffers) — and the chip-side golden
(solo.py bench_reservoir: kk % RES_EVERY == 0, was rand 0.25/kf). The
randomness that matters is retained exactly where it matters: WHICH
slot dies (random overwrite, refined-slot exclusion kept) and WHICH
slot replays (random draw) — the fixed cadence still yields the
exponential age mix, tau ~ N*RES_EVERY = 320 frames at N=16. solo
selftest green; webvis core parses, no RES_P remnants.

## 2026-07-13 — Repo restructure: flat root → role-named packages, bit-identical acceptance gate (docs/RESTRUCTURE-MAP.md)

User-directed accessibility restructure; one mechanical move commit, no
algorithmic change. Layout now: sspslam/ (shipped library, frozen),
runners/ (CLIs, `python3 -m runners.datasets run <name>`), baselines/,
experiments/ (catalogue = experiments/README.md, ex-EXPERIMENTS.md),
hw/ice40/ (ex ice40/ + ssp_ice40.py→hw/ice40/golden.py), docs/
(FINDINGS/RESULTS/PROTOCOL + sota/), scratch/ (ALL transient files,
gitignored; the 323 root scratch_* files moved there unmodified),
archive/ (+ superseded experiments.py). Full old→new table:
docs/RESTRUCTURE-MAP.md.

Mechanics: imports rewritten alias-preserving (`import sspslam.lattice
as ssp_slam_loop` — call sites untouched, module-global patching still
hits the single module instance; package __init__ files are empty so
module-init order is unchanged). 267 import lines in 57 files; the one
dynamic import (`__import__("ssp_slam_carmen")` in runners/synth.py)
converted to a normal import. Path-relative fixes: runners/spot.py DIR
parent→parents[1]; hw host tools repo-root refs parents[2]→parents[3];
experiments/viewpoint.py scratch paths → scratch/. This restructure
commit is the sanctioned exception to rule 1 (shipped files moved +
import lines only), gated as below.

GATE (this ledger's reason to trust everything above): the full
acceptance suite was captured on the pre-move tree and re-run post-move
— outputs BYTE-IDENTICAL except wall-clock ms/kf on every one of:
fr101 1.881 / fhw 0.981 / fr079 5.523 / belg 2.644 / stata 0.202 /
spot 0.034 (all `run <name>` lines incl. med/loops/mem/ref-poses),
synth bench (full table), quantized selftest (bit-exact line). 49/49
modules import; compileall clean. hw/ tools compile-checked only
(board unplugged — runtime untouched by the move beyond import lines +
parents[] indices). New: tests/test_smoke.py (imports + bit-exact
selftest), tests/test_acceptance.py (exact-line suite regression,
expectations = today's outputs), pyproject.toml (deps stay pinned in
requirements.txt).

Convention going forward: scratch lives in scratch/ (run with
`PYTHONPATH=. python3 scratch/scratch_<topic>.py`); ledger/scratch/
archive keep historical flat filenames — resolve via
docs/RESTRUCTURE-MAP.md. Historical scratch files were NOT import-
rewritten (they are evidence of what ran, not runnable code).

## 2026-07-13 — reservoir save-cadence study on REAL data (user: "every 20, every 64 — we only tried the denser random 10%"): cadence does NOT unlock adoption; ANY map-mutating rehearsal makes closure-cascade logs BANDED — collapse is a per-draw event, not a dose response; the 2026-07-12 fhw opt-in is AMENDED to a band (scratch/scratch_reservoir_real.py)

Harness: ReservoirSLAM = verbatim port of scratch_replayfront.ReplaySLAM
(2026-07-13 package paths) + save_every knob (None = banked p=0.25/kf
random; else deterministic k%20 / k%64 — the websim/solo RES_EVERY
convention, random slot overwrite + random replay draw KEPT per user) +
res_w knob. GATES: OFF ≡ acceptance on all 5 logs (incl. spot
0.034/22); ALL 8 p25 rseed-11 control cells reproduce
scratch_replayfront.log BIT-EXACT incl. rep counters. One fixed config
(N=16, w=0.3, 1 replay/kf, RECENT_KF=60); rseeds 11/12/13 = replay-draw
bands; do-no-harm gate = matcher correlation (GT scores only).

ATE (med) per arm; OFF: fr101 1.881(1.551)/53, fhw 0.981(0.841)/559,
stata 0.202(0.123)/99, belg 2.644(1.859)/1, spot 0.034(0.028)/22.

  fr101 R1: p25 {1.396,1.648,4.412 banked} | e20 1.915(1.621) |
            e64 {1.707(1.429), 1.741(1.450), 3.530(1.053)}
  fr101 R2: p25 2.461(1.596) | e20 {1.565(1.184), 1.753(1.281),
            2.530(1.575)} | e64 1.808(1.487)
  fhw   R1: p25 {0.220(0.182), 0.241(0.218), 4.045(3.174)} ← the banked
            opt-in, now banded | e20 {0.223(0.193), 1.009(0.857),
            2.217(0.693)} | e64 0.494(0.402)
  fhw   R2: p25 0.701(0.640) | e20 0.534(0.302) | e64 {0.247(0.216),
            0.248(0.216), 1.016(0.917)}
  stata R1: p25 0.246(0.130) | e20 {0.201(0.127), 0.253(0.165),
            4.743(3.988)} | e64 {0.201(0.125), 0.205(0.125),
            10.921(9.200)} | diag e21 10.711 / e25 0.258 (s11)
  stata R2: p25 14.949 (banked catastrophe) | e20 4.743 | e64
            {0.209(0.125), 0.213(0.130), 1.040(1.043)}
  belg  R1: e20 3.883 | e64 {2.087, 2.405, 5.853}  (loop-noise log)
  belg  R2: e20 2.356 | e64 {2.528, 5.857, 7.103}
  spot  R1+R2, all cadences: 0.033–0.034, 22 loops — NO-OP at the
            reference noise floor (target platform tolerates all arms).

VERDICTS.
(1) CADENCE IS NOT THE UNLOCK: no (arm × cadence) survives the
multi-seed multi-log gate; the user's sparser cadences reshape tails
(fr101's 4.41 p25 tail → 2.53/3.53) but do not remove bimodality.
(2) THE LAW (new): map-mutating rehearsal at ANY dose/cadence converts
closure-cascade logs into BANDED logs. Collapse (loops 99→4-14 on
stata, 559→354 on fhw) is a discrete per-draw event hitting ~1/3 of
arm×seed cells at EVERY cadence — ~70 folds over ~1400 kf re-roll
stata's basin just like 825 folds do; e25 (maximally segment-locked)
is fine while co-prime e21 collapses — NOT resonance, NOT dose. The
replay fold stream is an ONLINE perturbation sequence: the
band-probe rule (PROTOCOL §6) extends to any feature that mutates the
map, and single-seed results on such features are basin draws.
(diag invocation: e21/e25 via inline CADENCE additions to the harness
module — see scratch_reservoir_stata_diag.log.)
(3) AMENDMENT to 2026-07-12 "per-environment OPT-IN": the fhw R1 p25
lever is itself a band {0.220, 0.241, 4.045} — a ~2/3-probability 4×
win, 1/3 collapse. The opt-in stands only with band-aware framing;
even 559 dense closures do not guarantee gauge re-pinning.
(4) MEDIAN WINS CONFIRMED AND BROADENED: median beats OFF in 15/18
non-collapsed replay draws across fr101/fhw/stata (e.g. stata R1 e64
s13 = 0.205 ATE / 0.125 med / 101 loops — better than OFF on all
three); the banked mechanism (registration improves; closure graph
carries the risk) holds at every cadence.
(5) R2 cross-pass at HIGH dose is reliably catastrophic (p25 14.9);
at LOW dose it is banded like everything else (e64 {0.21,0.21,1.04})
— the frontend-recency law's protection matters in proportion to fold
rate.
Follow-up filed: consensus-over-replay-draws (K-seed trajectory
consensus, stablegate-class, offline only) — nothing GT-free selects
the good draw online (commit-policy wall applies to draw selection).

## 2026-07-13 — SPOT Telluride drop 2 (SCHOOL building, runs 1+2): adapter + distillation + webvis; kinematic odometry BROKEN upstream; lidar-only holds ~1.06 m vs gated LIO-SAM on run2's flat window; reservoir arms benign 10/10 there; run1 = band-dominated stress set (runners/spot_school.py)

Dataset (user: "bigger dataset, let's validate findings"): the HF repo
reorganized — classroom/run1 = the banked session (shards byte-identical
to local) + NEW school/run1 (117 pointcloud shards, 8298 clouds, 455 s,
2075 kf @stride 4) and school/run2 (47 shards, 3343 clouds, 167 s,
836 kf); shard format changed pcd_bytes -> npz_bytes (xyz/intensity/
ring). Downloaded pointclouds+odometry+static only (~8.6 GB), DISTILLED
per run into scans.npz + ref_lio.npz (all the pipeline and webvis
consume; parquets re-fetchable — ~8 GB reclaimable at will).

DATA QUALITY (measured; undocumented upstream): the school
odometry_imu is a single ~1 kHz stream that OSCILLATES +-20..300 m for
extended intervals in BOTH runs — kinematic odometry UNUSABLE as
reference (the classroom drop's was fine; same converter change that
renamed the blobs). LIO-SAM: run1 is 99% z=-1000 sentinels (DEAD);
run2 healthy ONLY t<~80 s (z flat +-0.1 m, ~5 Hz), then it AND the
robot localization break simultaneously (z +-19 m, then frozen).
Gating (sentinels + |z|<1 m flat window + 0.4 s windowed velocity
<=2.5 m/s + 1 s dilation): run2 reference = 342/836 kf over a 23x16 m
sweep (35 m path); run1 = ZERO reference. Diagnosis details in
runners/spot_school.py docstring; not resonance-checked upstream.

Infra (additive; the acceptance path re-gated): runners/spot_school.py
(npz-shard parse mirroring runners/spot.py semantics, gated-LIO ref
cache, LIO-frame bundle, protocol stays lidar-only CV); registry
entries school_run1/school_run2 (kind=spot_school); DS.evaluate exact
branch gained an empty-mask guard (nan/0 ref — additive);
demo/export_replay.py honors bundle gt_name. GATE: `run spot`
(classroom) BYTE-IDENTICAL after all edits (0.034/22 loops/366 KB).

school_run2 (canonical): OFF ATE 1.071 med 1.061, 59 loops, 821 KB,
79 ms/kf over the 342-kf flat window — lidar-only holds ~1 m in real
school corridors vs LIO-SAM (the reference's own error there is
unknown; no classroom-style noise-floor claim available). RESERVOIR
VALIDATION: {R1,R2} x {p25,e20,e64} s11 = ATE 1.055–1.068 (ALL <=
OFF), loops 58–60; seed probes R1-e20 {1.061,1.067,1.067} / R2-e64
{1.067,1.068,1.070} — 10/10 replay cells BENIGN, no collapse drawn;
the median-win pattern replicates (small, -0.3..-1.5%). Read with the
2026-07-13 law: collapse probability is per-environment — run2's
dense small-area closure structure (59 loops / 23x16 m) is
fhw-tolerant-class and its rate is evidently low (0/10), vs stata's
~1/3 and fhw-p25's 1/3.

school_run1 (2075 kf, 455 s, real multi-room traverse, NO reference):
pipeline runs BOUNDED and stable — canonical OFF 174 loops, 968 KB,
89 ms/kf. THE WORLD-FRAME DRAW ALONE moves loops 65 (old
kinematic-anchored frame) -> 174 (canonical zeros-anchored) on
identical input: the long aliased school traverse is BAND-DOMINATED
(the perturbation-band law in whole-run form). Reservoir arms (loops
151–191 across p25/e20/e64 x R1/R2) sit INSIDE that frame-draw
variance — no attribution possible without a reference. STATUS:
loop/stability stress set (Deutsches-Museum class); its registry ATE
prints nan by construction.

WEBVIS: replay_school_run2_shipped.json exported (667 KB) and embedded
— 15-replay pack now incl. school_run2 @ 1.071 with honest reference
label ("LIO-SAM, gated flat window; kinematic odometry broken in this
drop"); pack parses as JSON; jsc syntax check PARSE OK. run1
deliberately NOT exported (refless: ate=nan is invalid JSON and there
is no honest reference trace to draw).

VERDICT vs the user's ask ("validate findings"): (1) bounded-memory +
runtime claims HOLD at 2.5x stata scale on the target platform
(<=1.7 MB, ~80-90 ms/kf, 2075 kf); (2) the reservoir median-win
replicates on real school data and NO cadence harms it there (10/10
benign) — consistent with the cadence study's law once collapse
probability is understood as per-environment; (3) the aliasing/band
wall shows up exactly where predicted (run1: frame-draw dominates the
closure graph); (4) NEW hygiene law for the platform: reference
streams must be velocity/z-gated BEFORE use — this drop's kinematic
odometry would silently poison any eval that trusts it.

### school_run2 webvis glitch (user report 2026-07-14): reference-trace placeholders, FIXED

"Glitches back and forth then stays at origin" = the replay's
odometry/reference overlay, not the estimate: the packed Python
trajectory is smooth (step p95 <=0.2 m, a legitimate out-and-back tour
ending at the start). Cause: ref_lio.npz filled the 494 uncovered kf
with a CONSTANT first-valid-pose placeholder — the drawn trace
teleported pose<->placeholder through coverage gaps and parked at the
start pose after t~80 s. Fix (runners/spot_school.py build_ref):
hold-last-valid forward-fill (leading gap back-fills from the first
valid pose = the world-frame seed, so pipeline output is UNCHANGED —
re-gated: ATE 1.071/59 loops identical). Re-exported + re-embedded;
odom trace now step p95 0.137 m with one honest 2.3 m re-acquisition
step; tail parks at the last valid reference pose. jsc PARSE OK.

## 2026-07-14 — FPGA recipe across ALL webvis demos (user directive): every real-data replay now carries its regime-correct FPGA arm; two honest negatives labeled by their numbers (demo/export_replay.py lean/lean-i2)

Config: "lean" = the banked binary winner (point encoding + 2-bit
phase-only ring store + per-ring scales + int8 matcher). NEW "lean-i2"
= the dense-head FPGA recipe (deploy sampler bridged@63.4° carrying
the same 2b+int8 store/arithmetic) — raw points are the WRONG recipe
on stata-class heads (encoder study). Exporter gained the interp flag;
manifest reordered dataset-grouped; 19 replays, jsc PARSE OK.

  FPGA arms (vs their float siblings):
  spot        lean    0.036 @ 14 KB   (float interp2 0.039 @ 354 KB)
  fr101       lean    1.132 @ 75 KB   (float shipped 1.881 — BEATS float)
  fr079       lean    3.149           (float e2 2.210; banded log)
  fhw         lean    1.108 @ 206 KB  (float shipped 0.981 @ 5203 KB — 25x
                                       less map at +13%)
  school_run2 lean    1.120 @ 34 KB   (float shipped 1.071 @ 821 KB — 24x
                                       less map at +5% on the REAL school,
                                       NEW target-platform datapoint)
  belg        lean    5.232 @ 63 KB   (float 2.644 / hex63 2.071 — the
                                       non-Manhattan log rejects the lean
                                       point recipe; negative, labeled)
  stata       lean-i2 6.379 @ 58 KB   (float interp2 0.196 — CLOSURE
                                       STARVATION: 99->2 loops, 0 relax;
                                       the aliasing-heavy flagship's
                                       closure cascade does not survive
                                       the 2b store even with the deploy
                                       sampler; consistent with the banked
                                       stata-lean gap + u2b band. Negative,
                                       kept visible per house style.)

Dynenv arms unchanged (pinned expect gates are the point of those
entries). Read: the FPGA-lean recipe is DEPLOYABLE on the target
platform (spot 0.036/14 KB, school_run2 1.120/34 KB) and on
dense-closure halls at a large memory win (fhw); it is NOT a universal
default — belg-class non-Manhattan structure and the stata-class
aliased flagship reject it (per-environment, as banked).

### Webvis menu trimmed to FPGA-only (user directive 2026-07-14)

manifest.json := the 7 FPGA arms only (spot/stata/fhw/fr101/fr079/
belg/school_run2 — one per dataset, incl. the two labeled negatives);
float/hex/dynenv replay blobs remain on disk and re-embed by adding
them back to the manifest. Page 35.8 -> 14.9 MB; jsc PARSE OK.

## 2026-07-14 — ECP5 TRACK OPENED (Icepi Zero): camera pipeline @ lidar point-parity, FAST-9 RTL bit-exact + 72 MHz, and the 2D-vs-3D lattice study — az×elev KEEPS the exact-rotation algebra in 3D; lidar⊕cam vector fusion AUC 0.92 (hw/ecp5/, runners/spot_cam.py, experiments/lattice3d.py)

User directives: "ECP5 (icepi zero) version taking full lidar + camera
(OV5640 direct later); for now scale dataset cam data + simple feature
detection so lidar and cam have approx the same amount of data" and
"try both a 2D and a 3D version — e.g. Fibonacci sphere (multiple
scales) or other structured layouts". Board = Icepi Zero v1.3
(LFE5U-25F CABGA256, 24k LUT, ~112 KiB EBR, 28 MULT18, 32 MB SDRAM,
50 MHz osc; LPF vendored verbatim). nextpnr-ecp5/ecppack verified
present in the local oss-cad-suite; board arriving — everything below
is sim/PnR-gated (no hardware yet), mirroring the iCE40 staging.

CAMERA PIPELINE (runners/spot_cam.py + hw/ecp5/host/golden_cam.py):
D455 640x480 JPEG -> integer luma (77R+150G+29B)>>8 -> 2x2 box bin
(320x240) -> integer FAST-9 (radius-3 circle, >=9 contiguous,
comparator-only score, 3x3 NMS) with the threshold SERVOED per
keyframe to the lidar valid-beam count. POINT PARITY ACHIEVED:
school_run2 742/836 kf aligned (dt med 7.6 ms), lidar med 948 pts vs
cam med 944 feats, ratio med 1.00 [p10 0.97, p90 1.03]; classroom
414/414, 901 vs 904, ratio 1.00. Bytes/kf: lidar slice 2048 | cam
feats ~5.6 KB | binned gray 76.8 KB | full cloud 128 KB. Caches:
cam_features.npz per run (CSR features + intrinsics).

FAST-9 RTL (hw/ecp5/rtl/fast9.v, ~200 lines): 6 line buffers + 7x7
window, per-tap comparators, 16-rotation contiguous-9 AND-tree, arm
adder trees, ZERO multipliers; one pipeline stage on the cone (the
iCE40 pd-stage pattern — single-cycle missed 50 MHz at 45.9).
GATE: tb vs golden vectors from a REAL school frame at its served
threshold — 3364/3364 centres BIT-EXACT (t=12). Build on LFE5U-25F:
4011/24288 COMB (16.5%), 735 FF, 200 RAMW, 1 DP16KD, 0 MULT18;
timing 72.2 MHz vs the 50 MHz constraint. Defect classes hit + fixed:
tb reset-release race at posedge (X-propagation — declaration initials
+ negedge release), valid-vs-data pipeline skew twice (one register
per added data stage — mismatches appeared exactly one column
shifted).

2D-vs-3D LATTICE STUDY (experiments/lattice3d.py; equal D=240 = the
matcher budget; scales = the shipped ladder; real school_run2 full
clouds via npz shards, classroom via PCD; anti-oracle: references
label pairs only):
  disp (SE(3) displacement decode, grid step 0.15 m):
    |d|=0.8 m: az2d err med 0.409 (cannot see dz + planar aliasing)
    vs fib3d 0.075 / azel3d 0.074 / rand3d 0.075 — all 3D layouts at
    the grid floor; |d|=0.3: 0.138 vs 0.080-0.082.
  rot (yaw 15 deg, permutation vs re-encode):
    az2d cos 1.0000 (exact, the house algebra) | azel3d cos 1.0000
    EXACT — azimuth-rings x elevation-bands keeps rotation = index
    permutation band-wise | fib3d 0.870 med / 0.762 min | rand3d
    0.875/0.696 — the Fibonacci sphere SACRIFICES the exact-rotation
    algebra and buys nothing measurable elsewhere.
  place (classroom, 110 frames, withheld-odometry labels, 179 same /
  4144 diff pairs, yaw-agnostic |cos| proxy):
    az2d AUC 0.744 -> fib3d 0.833 / azel3d 0.822 / rand3d 0.842 —
    3D content lifts same-place separability ~+0.08-0.10.
    (school_run2's gated window has NO revisit pairs — outbound sweep;
    stated, not fudged.)
  cam (camera bearings K^-1[u,v,1] on the 3D lattices, score-weighted):
    cam-only AUC 0.73-0.75; **lidar(+)cam bundled (plain vector
    addition, 50/50): AUC 0.922-0.935** vs lidar-alone 0.82-0.84 —
    the first fusion datapoint on real target-platform data, and it is
    POSITIVE by ~+0.10 AUC.

RECIPE: the ECP5 3D lattice is **azel3d** (az rings x elevation bands
x scale ladder) — matches Fibonacci on displacement/place metrics AND
keeps the exact yaw-permutation matcher primitive. Fibonacci sphere =
labeled negative for this system (algebra loss, no gain). Fusion via
vector addition is live and measured. Next (filed in hw/ecp5/README):
OV5640 DVP front, full-cloud SDRAM ingest, encoder port
(SPRAM->EBR/SDRAM), python fusion experiment at the aliasing wall.

### School-data addendum (user: "did you run over the school data?") — place/cam re-run on school_run2; ALL revisits are reverse-heading, and the exact-permutation layouts are the only ones that survive it

disp/rot already ran on school_run2 clouds (banked above). place/cam
originally fell back to the classroom (the school's gated-LIO window
has no revisit pairs); now re-run on the FULL school_run2 with
OWN-ESTIMATE labels (diagnostic per PROTOCOL: ~1 m label error, pair
margins widened to same<1.0 m / diff>4 m; labels never enter an
encoder). Heading split exposes the structure: the school's 85
same-place pairs are 0 forward / 85 REVERSE (out-and-back tour).

  school_run2 (all-reverse revisits), raw |cos| vs rotation-searched:
    az2d    raw 0.651 -> ROT-SEARCHED 0.767   (60-step exact perm)
    azel3d  raw 0.542 -> ROT-SEARCHED 0.700   (12-step exact perm)
    fib3d   raw 0.531 -> no exact permutation, CANNOT rot-search
    rand3d  raw 0.566 -> likewise stuck
  classroom control (all-forward pairs): rot-search ~= raw
    (0.744->0.705, 0.822->0.807 — max-over-perms lifts the diff tail
    slightly; 3D > 2D unchanged).
  camera on school: cam-only 0.37-0.40, lidar+cam 0.42-0.51 — BELOW
    chance on reverse revisits and not rescued by fusion: the D455
    looks FORWARD; a reverse pass sees a different scene. The banked
    viewpoint wall in fusion form — camera helps only where views
    overlap (classroom 0.92-0.94).

READ: (1) the algebra argument is now empirical — on aliased
corridors with reversed revisits, rotation-as-exact-permutation buys
+0.12..+0.16 AUC that Fibonacci/random layouts structurally cannot
reach; azel3d's gap to az2d (0.700 vs 0.767) is its coarser 15-deg
rotation search (12 az/band at equal D), not the 3D content — fine-
rotation refinement per band (the d/dtheta derivative trick
generalizes band-wise) is the filed follow-up. (2) Forward cameras do
not cross the reverse-revisit wall; fusion gains are view-overlap-
gated. Both consistent with FINDINGS section 5; neither closes it.

## 2026-07-14 — detector-formulation zoo (user: "continue experimenting, same opt corners, various feature detector formulations"): shi-tomasi-L1 wins the F-corner (+0.07 cam-AUC over FAST-9), extrema/edge win the S-corner at 3 line-buffer rows; depth-augmented landmarks NEGATIVE pending real extrinsics (experiments/detzoo.py)

Seven integer, line-buffer-streamable formulations, each servoed to
lidar point parity per frame and judged on the house corners (S =
window rows / mults/px; F = cam-only + lidar(+)cam place-AUC on azel3d
+ adjacent-frame repeatability; T = all 1 px/clk). Venues: classroom
(honest withheld-odometry labels) + school_run2 (own-estimate
DIAGNOSTIC labels; all-reverse revisits, the known wall).

  classroom (lidar-only AUC 0.822):
    det        rows mult  cam-AUC fused  adj-rep
    fast9       7    0    0.742   0.921  0.973   (shipped RTL v0)
    fast12      7    0    0.698   0.915  0.970   (same silicon, worse)
    susan16     7    0    0.733   0.912  0.971
    extrema     3    0    0.723   0.904  0.962   <- S-corner champion
    harris      8    5    0.706   0.923  0.922   (servo saturates 1..4095
                                                  — >>16 rescale too
                                                  coarse; not tuned,
                                                  superseded below)
    shitomasi   8    3    0.809   0.935  0.947   <- F-corner WINNER
    edge        3    0    0.763   0.910  0.972   (edges > corners for
                                                  place signal here)
  school_run2 (lidar-only 0.446; all cam channels at the reverse-view
  wall 0.36-0.41 as banked): fused best = edge 0.522; adj-rep
  0.80-0.84 everywhere — the camera stays a solid frame-to-frame
  (odometry-class) signal even where place recognition is walled.

VERDICTS: (1) shi-tomasi L1 (separable Sobel -> 5x5 structure tensor
-> (Sxx+Syy) - (|Sxx-Syy|+2|Sxy|); 3 MULT18 at pixel rate, response =
abs/add only) is the RTL v1 candidate: +0.067 cam-AUC and +0.014
fused over FAST-9 for 3 of the ECP5's 28 multipliers. (2) The cheap
tier is real: extrema (8 comparators, 3 rows) and edge (Sobel adds, 3
rows) give up <=0.02 fused AUC vs FAST-9 at ~40% of its line buffers
— the right corner when the detector shares the die with the encoder.
(3) fast12/susan16: no wins. (4) DEPTH-AUGMENTED 3D landmarks
(bearing x nearest-lidar-range, 2-deg cone): NEGATIVE as implemented —
the geometry-picked camera->body axis mapping is UNSTABLE across
venues (z->+x @52% hits vs z->+y @76%), and 3D points score BELOW
bearings both venues (classroom 0.566 vs 0.651 cam) — radial noise at
lattice scale without calibrated extrinsics. Filed: real extrinsic
calibration (checkerboard or lidar-camera mutual-information), then
re-run; ALSO noted: bearing-cone orientation vs the azel3d equatorial
band matters (camera-frame vs body-frame encodes differ — pin the
convention when extrinsics are real). FAST-9 RTL v0 stays the shipped
core; shitomasi v1 RTL + OV5640 DVP front are the next hardware steps.

## 2026-07-14 — 4h block part 1 (user: "lidar needs to be scaled, vision needs to be explored; two maps; QVGA@120"): lidar scales with D not ingest (0.70->0.86 @ D960, 3 rings == 64); vision's best channel is DETECTOR-FREE dense intensity; two-map beats shared-map at the wall (experiments/twomap.py, experiments/lidarscale.py)

Operating point locked (user): OV5640 at its max rate = QVGA 320x240 @
120 fps — natively the binned working res (no bin stage), ~9-12 MHz
pixel clock; T-corner: comparator detectors + shi-tomasi FREE; 3-layer
int8 CNN ~60% of the MULT18 budget (quality-reference tier only);
YOLO-nano-class OUT (hw/ecp5/README section 5).

TWO-MAP ARCHITECTURE (school est-labels diagnostic / classroom honest):
lidar map = az2d rot-searched; vision map = own bearing lattice (fib or
FOV-cone); late z-normed score fusion. school: lidar-map 0.720 vs
SHARED-map control 0.516 — the split protects lidar at the wall.
classroom: shared 0.921 vs best late-fusion 0.877 — shared wins under
view overlap. Vision arms (school | classroom, vis-only):
  fast9-pk 0.38|0.75  shito-pk 0.37|0.84(cone 0.81)  dog-pk 0.39|0.79
  dog-raw 0.38|0.76  gridint (8x8 mean-intensity, NO detector)
  0.62|0.949 <- the STRONGEST vision channel in both venues; on school
  fused(a=.5) 0.759 BEATS lidar-alone 0.720 — dense appearance carries
  reverse-view signal sparse corners cannot. FOV-cone lattice ~= fib
  (no win from concentrating directions). ucnn arm was DEGENERATE run 1
  (relu chain after zero-mean random filters -> all-zero heat; all rows
  identical — NOT banked; abs-activation fix in, rerun pending).

LIDAR SCALING (school, azel3d rot-searched, est-labels):
  input:  slice1(2D shipped input) 0.256 | rings3 {16,33,50} 0.701 |
          full 64-ring 0.700  -> THREE RINGS CARRY THE FULL-CLOUD PLACE
          SIGNAL (ingest can drop to 3/64 bandwidth)
  D:      240 0.700 | 480 0.814 | 960 (24az x 5el x 8lam sqrt2 0.25-2.83)
          0.858  -> separability SCALES WITH D on the aliasing venue
          (finer az = finer rot search + more scales; the span-ladder
          law generalizes to 3D)
  SUB:    32/16/8 (2k/4k/8k pts) 0.690/0.700/0.699 -> FLAT; 2k pts
          suffice
  disp:   flat 0.067-0.075 across D (already at grid floor at D=240)
  RECIPE: spend on the lattice (D960-class), not the ingest (3 rings,
  ~2k pts) — quality +0.16 while bandwidth drops ~40x.

### 4h block part 2 — 6-DoF capability, fusion policy, fixed CNN arm

SO(3) capability on FROZEN vectors (school clouds, permutation search,
+-15deg grid @5deg): azel3d decodes unknown pitch/yaw/roll at 3.9 deg
med / 6.1 p90 (grid floor ~3.5) — with z-translation already at floor,
FULL 6-DoF transform capability is demonstrated without re-encoding.
fib3d 4.7/9.3 (loses even its isotropy argument); az2d 19.2 deg =
orientation-blind beyond heading (the 2D+heading system, quantified).
Per-axis quality: azel3d y15 EXACT (1.0000*), p15 0.94, r15 0.92.

FUSION POLICY (lidar rot-searched x gridint, z-normed): school
MAX-RULE 0.798 beats every alpha-sum (best 0.759) and lidar-alone
0.720; classroom prefers vision-heavy sums (a=0.1 -> 0.954). Policy is
venue-dependent; max = the robust default at the wall;
confidence-adaptive alpha filed.

UCNN FIXED (abs activations replace the relu chain that zeroed the
heat): school ucnn-pk 0.426 (sparse, walled like all corners);
ucnn-RAW dense heatmap 0.652 vis / 0.756 fused — beats gridint's 0.622
vis with RANDOM weights: dense appearance >> sparse corners for place,
and a trained tiny head has headroom — but at ~60% of the MULT18
budget @QVGA120 vs gridint ~free. Deploy = gridint; ucnn-raw = quality
reference tier.

### 4h block part 3 — appearance-IN-PHASE encoder family (architecture: bind what with where): a precision/ego-motion channel, NOT a cross-view channel

Three integer cis-ROM-addressable encoders binding appearance into the
phase (vs gridint = appearance as weights): intphase (cell intensity,
3 wavelengths), gradhog (8-dir quantized gradient orientation,
pi-periodic harmonics, magnitude weights), census (cell-level 8-bit
neighbour code, binary phase decomposition). Predictions on record:
census to win school via illumination invariance; intphase fragile.
MEASURED (vis-AUC school | classroom):
  gridint(weights) 0.622 | 0.949     intphase 0.448 | 0.999 (!)
  gradhog          0.425 | 0.829     census   0.425 | 0.852
  adj-rep: intphase 0.897/0.997 — the strongest ego-motion signal yet.
Predictions REFUTED: phase-binding does not rescue cross-view matching
(cell codes/orientations differ across views regardless of lighting);
it SHARPENS view-specificity. LAW: appearance-in-phase = the
PRECISION + EGO-MOTION channel (view-overlap verification, 120 fps
tracking; classroom 0.999/0.997); appearance-as-weights (gridint) =
the coarse CROSS-VIEW place channel (school 0.622). The two-tier
vision similarity drops straight into the two-service architecture
(hw/ecp5/README "Architecture").

### 4h block part 4 — TUM RGB-D 6D probe: DEPTH-LANDMARK REDEMPTION — 3D visual landmarks CROSS the reverse-view wall (rev-AUC 0.941); real-motion 6D rotation decode works (runners/tum.py, experiments/vision6d.py)

Dataset delivered per user ask: TUM RGB-D fr3_long_office_household
(415 kf parsed to the QVGA integer pipeline; 6-DoF mocap labels;
REGISTERED depth = no extrinsic guessing) + fr2_pioneer_slam (robot,
parsing). EuRoC moved upstream (ETH Research Collection; pointer in
data/README, deferred).

PLACE (mocap labels; 1021 same pairs incl 78 REVERSE / 58311 diff):
  gridint-bearings  0.797  (fwd 0.846 / rev 0.198 — dense bearings
                            CRASH across viewpoint reversal)
  fast9-bearings    0.805  (rev 0.494 = chance)
  gridint-3D        0.794  (rev 0.791 — depth restores reverse)
  fast9-3D          0.850  (rev 0.941 — sparse corners x registered
                            depth = TRUE 3D landmarks; view-invariant)
VERDICT: the SPOT depth-augmentation NEGATIVE was an extrinsics
artifact, as filed — with registered depth the camera+range channel
crosses the reverse-view wall that bearings physically cannot. On the
platform, lidar provides the range: CALIBRATED lidar-camera extrinsics
is now the highest-value single step for the fusion track.

REAL-MOTION SO(3) decode (40 pairs, |dt|<0.15 m, true rot 5-20 deg,
frozen gridint-3D vectors): fib3d 6.7 deg med / azel3d 8.4 / az2d 21.7
(heading-blind, as designed). Mixed-axis real rotation favors fib's
isotropy (synthetic single-axis favored azel) — both usable; grid
floor 3.5 not reached (translation contamination |dt| up to 0.15 m).

### 4h block part 5 — fr2_pioneer replicate: the depth-COVERAGE boundary condition

fr2_pioneer_slam (373 kf, robot in a big industrial hall, Kinect):
fast9-3D 0.560 (rev 0.540), gridint-bear 0.543, others 0.36-0.49;
rotation decode fib 10.4 / azel 11.3 / az2d 21.7 deg. NOT a refutation
of the fr3 redemption: the Kinect's ~4 m depth covers almost none of
the hall (landmarks starve) and robot vibration blurs corners — a
depth-COVERAGE boundary condition. The target platform's range source
is 60 m lidar, so fr3-class coverage is the relevant regime; fr2 marks
where the channel degrades. Real-motion rotation decode stays ~2x
better than heading-only even here.

## 2026-07-14 — individual-optimization round 2 (user: "continue optimizing the two systems individually"): lidar scale axis = the LADDER; vision D is FLAT — the two maps want opposite budgets (experiments/lidarscale.py axis, experiments/twomap.py vissweep)

LIDAR (school est-labels, full cloud unless noted, rot-searched):
  axis decomposition at EQUAL D=960: az24xL8 0.858 > az48xL4 0.842 >
  az24xel10xL4 0.819 — the SCALE LADDER buys most, azimuth second,
  elevation bands last (5 suffice). D1920 = 0.876 (+0.018 — diminishing
  but real). Parabolic rotation-peak refinement = exact no-op for place
  ranking (0.858; sub-step rotation matters for pose, not ranking).
  rings3 x D960 = 0.854 @ ~700 pts med — the ingest recipe HOLDS at
  scale. OPERATING POINT: 3 rings, ~0.7-2k pts, azel3d 24az x 5el x
  8lam sqrt2 (D960) -> 0.854; D1920 headroom if EBR allows.

VISION (school | classroom):
  cell size 4/8/16 IDENTICAL AUCs (coarsest angular wavelength 0.3 rad
  oversamples even 300 cells) -> c16 = 300 cells/frame, vision encode
  ~8.6 M cis-MAC/s @ 120 fps (trivial). Weight transform: intensity
  best cross-view (0.622|0.949); sqrt tiny classroom gain (0.957);
  gradmag WORST for place (0.317|0.728) but BEST adjacency
  (0.872|0.974) — ego-motion tier confirmed again. Vision D FLAT
  (0.622/0.622/0.623 @ D120/240/480 school) — the vision map does NOT
  scale with D; D120 suffices. intphase-DC (gain-invariance fix):
  school 0.451 (no rescue — fragility is view-specificity, not gain;
  law holds), classroom 0.989/adj 0.994.

THE ASYMMETRY LAW: the two maps want OPPOSITE budgets — lidar quality
scales with lattice D (spend EBR there: D960-1920) and not with
ingest; vision saturates at D120/300 cells (spend nothing) and splits
into intensity-weights (place) + gradmag/intphase (ego-motion) tiers.
Individual optimization is not just permitted by the two-map split —
it is required.

## 2026-07-14 — 6-DoF refinement, both systems (user: vision first, lidar after, then cohere): the decode stack is near-EXACT (0.29 deg / 3 mm GN on school clouds); real-pair error is CONTENT-OVERLAP, not machinery; visual + cloud gyros at 1.3-1.8 deg (experiments/vision6d.py se3/gyro/depthrob, experiments/lidar6d.py)

VISION (TUM fr3, mocap): joint separable SE(3) decode (343 rotations x
one-matmul translation each): coarse 6.3 deg / 0.216 m; local refine
6.7 / 0.172. VISUAL GYRO (analytic derivative vectors — the segder/Cx
pattern lifted to SO(3)xR3, 6x6 least squares, inner products only):
1.82 deg med on 2.6-deg adjacent-frame motion (rotation-only 3x3 was
2.02; coarse-rings-only variant REGRESSED med 2.31/p90 12.7 — too few
constraints; fine-ring linearization error filed -> iterate on points
for the frame-to-frame service). DEPTH-COVERAGE ladder (the
lidar-projected-depth readiness curve): rot 5.6/6.3/7.2/11.4 deg at
100/50/25/10% cell coverage — GRACEFUL TO 25%, breaks at 10% -> sparse
lidar depth on SPOT is sufficient.

LIDAR (TUM depth clouds as the 6-DoF range surrogate + school lidar):
  school clouds + synthetic SE(3): coarse 2.31 deg/0.051 m ->
  **GNx3 0.29 deg / 0.003 m** — the correlation+GN decode stack is
  near-exact on real lidar geometry.
  TUM real pairs (0.1-0.5 m, 5-20 deg): ~6 deg / 0.17 m at EVERY stage
  (coarse 6.35 / refine 5.72 / GN 6.14) — NOT the optimizer: at these
  baselines the two clouds are not rigid transforms of each other
  (occlusion + surface turnover + depth noise). Correlation 6-DoF
  decode is CONTENT-OVERLAP-limited; GN pays only inside the
  content-consistent regime.
  CLOUD GYRO (GNx2, adjacent frames): 1.34 deg med on 2.5-deg motion —
  beats the visual gyro; its true operating point (20 Hz sensor rate,
  10x smaller steps, ~total overlap) is still better-conditioned.
READ for the deploy variant: 6-DoF verification at loop closures should
use SHORT-baseline decodes (or snapshot-vs-snapshot at the same
anchor), where the stack is near-exact; gyros run at sensor rate on
both modalities with the same 6 derivative vectors per frame.

### fr1 aggressive-motion replicate + DEPLOY VARIANT v1 coherence

fr1_rpy (the purpose-built roll/pitch/yaw sequence): visual gyro 1.95
deg / cloud gyro 1.45 deg med on 3.0-deg adjacent steps — the gyros
REPLICATE on aggressive real rotation. fr1_desk wide-baseline SE(3):
7.4-9.3 deg / 0.18-0.22 m, GN diverges on low-overlap pairs (p90 21) —
the content-overlap limit holds across all three sequences.

DEPLOY VARIANT v1 written to hw/ecp5/README (all numbers banked):
lidar = 3-ring/2k-pt ingest + azel3d D960 per-anchor place layer
(0.854) + cloud gyro; vision = gridint-c16 place + intphase precision
+ visual gyro @120 fps + fast9 RTL sparse path, snapshot library
D120-240 (<=0.5 KB/anchor); fusion = max-rule candidates + intphase
precision verify + SHORT-BASELINE 6-DoF consistency checks (the
near-exact regime: 0.29 deg/3 mm); wide-baseline SE(3) = coarse gate
only; depth-lifted landmarks (the wall-crosser, rev 0.941, graceful to
25% coverage) gated on lidar-camera extrinsic calibration — the
standing unlock. Both maps bounded per anchor; QVGA120 budgets hold
(dense vision encode ~free at 300 cells; CNN tier excluded at 120 fps).

## 2026-07-14 — ECP5 ceiling tuning (user: use the full part, all corners, tune scales/vec-distribution; headroom -> map/history/detection): the "D-scaling law" was a LADDER law — coarse rings 0.5-90 m at D1920 = 0.928; curve saturates ~0.93 (experiments/lidarscale.py tune)

Sweep (school est-labels, full clouds, rot-searched; banked refs D960
0.858 / D1920 0.876 / D3840az 0.880 / D3840lam 0.915):
  LADDER (24az x 5el x 16lam, D1920): coarse-shifted (0.5..90.5 sqrt2)
  **0.928 / adj 0.967** > full-span 0.906 > octave-wide 0.902 >
  fine-shifted 0.896 — sub-0.5 m rings buy NOTHING for place (they are
  registration rings); building-to-campus-scale rings (11-90 m) are
  the lever (the 2D span15g global-anchor lesson, in 3D).
  ELEVATION (x 8lam): +-20 / +-22(sensor-matched) / 3-band / 9-band
  all 0.865-0.872 vs +-40 5-band 0.876 — FLAT (free knob; use
  sensor-matched +-22).
  CLOSE-OUT: 48az x coarse16 (D3840) 0.934; 20-ring 0.354-256 m
  (D2400) 0.933 — the curve SATURATES ~0.93.
CEILING RECIPE (the "better map" headroom sink, quantified): azel3d
24az x 5el x lam16-coarse(0.5-90.5 sqrt2) = D1920 -> **0.928** place
(vs v1's D960 0.854, +0.074), 480 B/anchor @ 2b, encode ~77 M
cis-MAC/s @ 2k pts x 20 Hz, rot-search 24-perm. Deploy v1.1: map
layer upgraded to this; D3840 not worth 2x bytes (+0.006).
Headroom allocation banked in hw/ecp5/README: (1) map/objects (this
recipe + the fidelity/pursuit layer), (2) history reservoir in SDRAM
(~16k raw scans; safe-cadence rehearsal + RAW-history closure
re-verification — new memory the UP5K never had), (3) trained tiny-CNN
detector head.

## 2026-07-14 — 3h formulation block part 1: fidelity/granularity at D1920-coarse (experiments/lidarscale.py fidelity)

Place-snapshot QUANT ladder (phase-only, NO per-ring scales — note):
float 0.928 (15.4 KB/anchor f32) | 4b 0.862 (1.9 KB) | 3b 0.862
(1.4 KB) | 2b 0.834 (0.48 KB). Unlike the matcher-store deadband law,
place-similarity RANKING pays for quantization at D1920-coarse; 3b =
the knee (2x cheaper than 4b, equal AUC). FILED: per-ring-scale quant
(the house 2b recipe) may recover — the flat quant here lacks ring
structure. S-corner options: 3b snapshots 1.4 KB/anchor, or float at
~15 KB (SDRAM: still 2k+ anchors).
MATCHING SPACE: mid-rings 0.5-2.83 alone 0.907 (D960) | coarse 4-90
alone 0.865 | all-16 0.928 — the mid scales are the single-handed
workhorse (reconciles the ladder finding: "fine" junk was sub-0.5 m,
not 0.5-2.8); query can drop to D960 mid-rings at -0.021 (T halves).
ROT-SEARCH GRANULARITY: 24 perms 0.928 | 12 perms 0.905 | 6 perms
0.823 — 12-perm is the budget point, 24 standard.

### 3h block part 2 — extraction variants: no new winner; TIERS MUST STAY SEPARATE CHANNELS

multigrid8+32 == gridint (scale mixing adds nothing; cell-size flatness
law); hog8bin = ego-motion-tier citizen (adj 0.894/0.973, place-weak
0.556/0.843); gridint(+)intphase COMBINED SNAPSHOT: school vis 0.410
(vs gridint 0.622) — summing the precision tier into the cross-view
vector DILUTES it (classroom 0.990 = intphase alone). LAW SHARPENED:
the two vision tiers must remain SEPARATE similarity channels fused at
score level (max-rule); vector addition across tiers is destructive.

## 2026-07-14 — ICEPI ZERO FIRST SILICON: fast9 detector BIT-EXACT on hardware, day-of-arrival (hw/ecp5/rtl/top_fast9_uart.v, host/hw_fast9.py)

Board attached (FT231X, /dev/cu.usbserial-DK0GEIG0; openFPGALoader
-b icepi-zero programs via bitbang). Sequence: detect -> flash
top_fast9 (LED smoke) -> UART gate top (2 Mbaud exact DIV=25, magic-
armed frame protocol, 2 KB TX FIFO) -> full-top SIM gate (uart cores
as host BFM) -> silicon: **HW PASS, 3364/3364 centres bit-exact vs
the golden model on a real school_run2 camera crop (t=12)** — the
ECP5 track's first hardware gate, green on arrival day.

Defect classes caught (both protocol-level, neither in the core):
(1) TX FIFO OVERFLOW at back-to-back input (reply = 2 bytes/pixel vs
1-byte drain at equal baud; missing bytes = exactly one FIFO depth —
host pacing 2.2x masks it; proper flow control filed for the
streaming front). (2) FT231X FIRST-BYTE CORRUPTION after port open:
the raw first byte (= threshold) arrived as garbage -> ZERO corners
with PERFECT framing (large t kills all arms) — hw signature:
mismatches == corner count, all-zero reads. Fix = MAGIC arm byte
(0xA5) + open-settle + dual-buffer flush. The sim-first discipline
held: core sim, full-top sim, then silicon — which agreed with sim on
the first properly-armed attempt.

## 2026-07-14 — 3h block part 3 + consolidation: FROZEN RAW-SAMPLE ANCHOR BANK — the mechanism is real (fhw 3.2x, fr101 -19%, stata celld-win), admission is the remaining work (experiments/frozenbank.py)

User design: frozen samples that never get overwritten (scan #0 + salient
picks) to anchor loop closure / remove drift. Implementation: cell-capped
(O(area)) frozen tier of raw scans; every 4 kf the nearest old frozen
sample re-encodes at its LIVE pose and the current scan matches
raw-vs-raw; accepted matches append standard loop edges (try_constraint
Z-form). DETERMINISTIC (no rng). OFF gates reproduced acceptance on all
6 venues.

v1.0 (naive gate 1.5 m/20 deg, fixed weights):
  fr101 1.881->1.518 (-19%, POINT-STABLE log, citable) loops 53->111
  fhw   0.981->0.305 (3.2x) loops 776    school med 1.061->1.050 (+37
        loops; scored window cannot see the return-leg anchoring)
  stata cell 0.225 (mild loss) but **celld 0.194 BEATS OFF 0.202 — the
        DISTINCTIVENESS gate binds exactly on the aliased flagship and
        flips it** (anti-aliasing selection working as designed)
  spot  no-op | belg CRASH 4.9 (aliased raw matches at full weight)
  first-only (scan #0): neutral-to-negative — a single over-trusted
  anchor distorts; the SPATIALLY-DISTRIBUTED cell bank is the design.
v1.5 (+coherence floor 0.45 + 1/coh^2 inflation):
  belg FIXED (== OFF exactly; all raw matches rejected) | fhw 0.357
  (still 2.7x) | fr101 1.880 (gain gone — floor too strict there) |
  stata 0.291 (WORSE than v1.0 — inflated weak edges still perturb).
VERDICT: the frozen-bank mechanism is REAL and large where revisits
exist, and the distinctiveness selector is the first thing to ever
flip stata; but a 2-constant gate cannot serve all regimes — the SAME
admission problem the shipped cascade solved. FILED v2: route frozen
closures through the SHIPPED admission (aperture classifier, EMA
coherence reference, PCM) instead of a private gate. STATUS: banked as
per-environment OPT-IN (fhw-class halls; v1.0 numbers) pending v2;
graph-only (no map-content mutation), deterministic, and the natural
consumer of the SDRAM frozen tier in the headroom plan.

### frozen-bank v2/v2.1 — the admission surface fully mapped; fhw record 0.242 (deterministic)

v2 (SHIPPED admission replicated verbatim on raw-raw matches: per-ring
coherence vs coh_ref EMA, analytic Hessian, ridge probes,
_coh_response soft inflation): belg FLIPS TO A WIN 2.042 (28 loops
from 1), stata celld 0.193 holds, fhw 0.610, but fr101 REGRESSES
2.132 — soft-inflating GOOD raw matches weakens true edges on the
point-stable log. v2.1 STRICT (same gates, accept-or-reject, full
weights): fr101 recovers 1.827, **fhw 0.242 = the best fhw in project
history** (beats replay-R1 0.241 — and deterministic, graph-only, no
band), stata 0.209 / belg 2.729 small losses (their inflated-weak
edges were the helpful ones). VARIANT TABLE: v1.0 wins {fr101, fhw},
v2 wins {belg, stata, fhw}, v2.1 wins {fr101, fhw!}; spot no-op
always; fhw wins under EVERY variant. The axis is WEIGHT SEMANTICS:
clean-match venues want full weights, ambiguous venues want the soft
ramp — same per-environment surface as every admission thread.
STATUS: per-environment OPT-IN (fhw-class: v2.1, 4.05x; aliased
flagship: v2-celld) — suite-default filed pending a weight-semantics
reconciliation (e.g. WELL-tier -> full weight, else soft — v2.2).
Mechanism verdict unchanged: REAL, large where revisits exist,
deterministic, and the SDRAM frozen tier's consumer.

## 2026-07-14 — camera eve: the full OV5640 DVP front SIM-GATED + two flash-ready tops; frame-rate turned into a measured servo, not a recall claim (hw/ecp5 rtl/{sccb,ov5640_init,dvp_capture,top_ov5640_id,top_ov5640_snap}.v, host/{gen_ov5640_rom,hw_snap}.py)

Sensor lands tomorrow; tonight every layer between the header pins and
the fast9 core was written and machine-gated in sim, so the physical
bring-up ladder (README) is wiring + four gates.

- **SCCB master** (`rtl/sccb.v`, quarter-phase open-drain, write +
  repeated-start read): `SCCB PASS: write ACKed, chip-ID reads 0x5640`
  against a bit-level behavioral slave (`make sim-sccb`).
- **Init ROM** (`host/gen_ov5640_rom.py`): parses the VENDORED
  esp32-camera tables (host/vendor/, Apache-2.0 — silicon-proven
  defaults) and emits 178 entries = defaults + **Y8 grayscale format**
  (luma direct, one byte/pixel — halves DVP bandwidth vs YUYV; the
  pipeline is luma-only) + QVGA 2×2-binned windowing computed by
  replicating ov5640.c set_framesize/set_image_options (the esp32
  driver computes these in code — they are in NO vendor table) + a rate
  section (PLL profile re-derived for XCLK 25 MHz). Selftest asserts
  symbol resolution, section boundaries, and the exact QVGA window
  registers (0x3808..0x380F = 320/240/2060/984).
- **ROM walker** (`rtl/ov5640_init.v`): `INIT-ROM PASS: 172 writes in
  order, all ACKed` — the tb replays the GENERATED hex against the real
  sccb master + slave model (`make sim-init`). (Tb lesson re-learned:
  drive stimulus at negedge; a posedge-blocking `go` pulse raced the
  DUT's always block and was invisible.)
- **DVP capture** (`rtl/dvp_capture.v`): all-sysclk 2FF-oversampled
  PCLK-edge capture (no CDC at ≤12.5 MHz PCLK = 4× oversample),
  VSYNC-rise→fall clean-frame arming, Y8/YUYV byte select with
  FSM-local line parity, snapshot to EBR, free-running
  frames/lines/bytes-line counters. `DVP PASS: Y8 + YUYV snapshots
  bit-exact, counters OK` — per-frame-seeded LFSR patterns prove WHICH
  frame landed (armed mid-frame-1 must capture frame 2), both formats
  (`make sim-dvp`). Tb bug worth remembering: verification is slower
  than a frame, so verify-while-streaming slipped the vsync sync by one
  frame — the pass decodes showed the DUT had correctly captured frame
  5's luma; restructured to phase-sequential fork/join.
- **Tops built** (combined LPF `build/cam.lpf`): `top_ov5640_id.bit`
  (chip-ID hello, 186 MHz) and `top_ov5640_snap.bit` (power seq → auto
  init ROM → armed snapshot → UART dump; 82.4 MHz, 41/56 EBR for the
  76.8 KB frame buffer, 1.1k LUT, 0 DSP). The snap top carries a
  UART→SCCB register passthrough ('W'/'r') and a 13-byte measured
  report ('R': frame count, lines/frame, bytes/line, init status).
- **Frame-rate policy**: the OV5640 PLL/divider chain is only
  half-documented and the esp32 driver's exact set_pll encoding is not
  reproducible from the vendored headers alone — so the ROM's rate
  section is labeled PREDICTED (~25 fps, PCLK ~12.5 MHz @ XCLK 25 MHz)
  and 60/120 fps is reached by stepping 0x3036 (PLL mult) / 0x380E-F
  (VTS) LIVE over the passthrough against the report's measured frame
  counter (`host/hw_snap.py` prints the fps estimate). Recall-risk off
  the critical path; the gate is silicon-measured numbers.
- **Host gate** (`host/hw_snap.py`): init-nerr==0 → chip-ID 0x5640 →
  geometry 240×320 → fps → snapshot std-floor (dead-bus detector) +
  golden fast9 servo smoke → `SNAP GATE PASS`. Writes build/snap.pgm
  (file only, per protocol).

Deferred, with reasons: fast9 EBR-ify (line buffers as LUTRAM cost
~2-3k LUT at W=320 — the step-3 streaming top has 23k free, so
EBR-ify is an optimization, not a blocker); async-FIFO/PLL capture
domain (only needed at the 120 fps rung). Status table + bring-up
ladder updated in hw/ecp5/README.md.

## 2026-07-15 — DEPLOY VERIFIED, sim + silicon, no camera needed (hw/ecp5 rtl/tb_top_snap.v; board re-checked after a brief USB disconnect)

User: "cam is not attached yet — just test in simulation and also verify
the deploy." Two layers, both machine-gated:

- **Full-system sim** (`rtl/tb_top_snap.v`, uart BFM host + the tb_sccb
  bit-level slave + a synthetic DVP source; top parameterized W/H/boot/
  SCCB-div/ms-cycles so the whole run is ~21 ms sim time):
  `TOP-SNAP PASS: init + chip-ID + write + snapshot + report
  end-to-end` — boot sequencer → auto init ROM (688 slave ACKs = exactly
  172 writes × 4 bytes through the real SCCB engine) → 'r' chip-ID
  passthrough → 'W' register write → 'S' armed snapshot bit-exact
  against exactly ONE per-frame-seeded LFSR source frame → 'R' report
  (n_bytes/lines/bytes-per-line all correct). Rebuilt after
  parameterization: 83.3 MHz, same EBR/LUT footprint.
  (Two tb lessons: `matches` is a SystemVerilog-2012 reserved word under
  iverilog -g2012; mixed initialized/plain `integer` declaration lists
  don't parse.)
- **Silicon, no camera**: fast9 gate reflashed after the USB blip —
  `HW PASS: 3364 centres bit-exact on silicon (t=12)` (board healthy,
  full build→flash→UART→compute path re-verified). Then
  `top_ov5640_snap.bit` flashed and probed: **NO-CAM DEPLOY PROBE
  PASS** — init_done with nerr=172 (every ROM write walked the real
  pins and NACKed against the pulled-up bus: the exact no-slave
  signature, and the count proves the entire ROM executed on silicon),
  chip-ID read 0xFF/err=1 (floating bus), frame counter 0 over 1 s (no
  PCLK). Tomorrow's bring-up is now wiring + the ladder gates only; the
  board is left running top_ov5640_snap.

## 2026-07-15 — vision snapshot-library quantization law (S-corner; experiments/visquant.py): intphase is 2b-FREE, gridint wants 3b where views overlap — the ≤0.5 KB/anchor deploy claim is now MEASURED at 300 B

school_run2 (est labels, diagnostic margins; lidar float rot 0.720) and
spot classroom (withheld odometry; lidar 0.705), W_vis_fib D240,
phase-only quant (house semantics), max-rule fusion vs the float lidar
channel:

- **intphase (precision/verify channel): 2b costs NOTHING anywhere** —
  classroom vis 0.999→0.996, adj-rep 0.997→0.994; school vis/fused
  actually IMPROVE (0.448→0.498 / 0.748→0.775). 120 B/anchor.
- **gridint (cross-view place channel): 3b is the knee on view-overlap
  venues** — classroom vis 0.949→0.935@3b (−0.014) but 0.889@2b
  (−0.060), fused 0.862→0.852@3b vs 0.792@2b. On the aliased school
  every quant level is neutral-to-positive (0.622→0.669 vis at 4b/3b/2b
  — quantization as regularization; fused −0.016). 180 B/anchor @3b.
- Adjacent-frame repeatability (the ego-motion proxy) holds 0.81–0.99
  at every level on both venues — the 120 fps service is
  quantization-insensitive.

RECIPE for deploy v1.2: gridint@3b (180 B) + intphase@2b (120 B) =
**300 B/anchor** for the two vision channels (vs the 480 B float
single-channel budget line and the ≤0.5 KB claim). [CORRECTION, same
session: this entry first cited the lidar 2b law as "0.925 vs 0.928" —
wrong recall; the banked lidar numbers at D1920-coarse are float 0.928,
3b 0.862 (the knee), 2b 0.834 — the VISION channels quantize far more
gracefully than the lidar channel.] Per-ring-scale quantization (finer
phase on coarse rings) remains filed.

### joint lidar deploy point (scratch_deploypoint.py): quant x rot-search costs compose ADDITIVELY; 3b x 12-perm ≡ 2b x 24-perm

D1920-coarse on school_run2 (est labels; banked singles reproduced
exactly: float/24 0.928, 3b/24 0.862): float/12 0.905, 3b/12 0.835,
2b/24 0.834, 2b/12 0.805. Deltas stack linearly (−0.066 for 3b, −0.094
for 2b, −0.023..−0.029 for 12-perm). ACTIONABLE EQUIVALENCE: **3b×12
(0.835, 720 B/anchor, half the rotation-search compute) ≡ 2b×24 (0.834,
480 B/anchor, full compute)** — store-vs-T-corner trade at equal
fidelity; pick per platform pressure. Deploy v1.2 default: 3b store ×
12-perm search (best fidelity-per-compute; the 24-perm search can be
enabled per-query for verification-grade matches).

## 2026-07-15 — object-in-map first characterization (experiments/objmap.py, TUM fr3, poses DIAGNOSTIC): mechanism VERIFIED, real-data utility = coarse room-region cue (AUC 0.735 / ~0.8 m) — code distinctiveness is the wall, not the algebra

Setup: world-frame appearance map (c16 cells → 3D via registered depth,
bipolar-spatter/FPE amplitude binding, per-segment bundles over 215 map
kf), 5×5-cell template queries from held-out blocks (≥4 s away),
room-wide coarse→fine decode, foils from fr1_desk. Synthetic selftest:
exact query 3.6 cm, 1-flip kernel 0.85 (theory 0.75), deterministic.

Real data (D960, seg8, scene-DC centered | uncentered census baseline
self 0.674 / cross 0.982 / AUC 0.474):
  census  self 0.917 / cross 1.327 / detect-AUC 0.609
  grad    self 1.184 / cross 0.880 / detect-AUC 0.667
  int     self 1.150 / cross 0.970 / detect-AUC **0.735**
- **Scene-DC centering = a detection↔localization TRADE** (+0.14..0.26
  AUC, −0.2..0.35 m position): the scene-common code amplitude carries
  true matched signal AND correlates the foils.
- **Detection ordering int > grad > census INVERTS the place-encoder
  prediction**: in amplitude-space matched filtering, graded FPE codes
  give partial credit for near-values while census bits flip wholesale
  across views (illumination invariance buys nothing indoors at fixed
  exposure).
- S-corner: D FLAT 480→1920 (0.77/0.80/0.85 — noise-limited, more rows
  don't help); segment scope does NOT rescue detection (seg32 0.749 /
  seg8 0.803 / seg4 0.905 — max-over-segments extreme value eats the
  per-segment SNR gain); **2b quant FREE** (0.741 vs 0.803 float).
- Anchor-bag cheap tier: 5/24 top1 even mean-subtracted — c16 census
  bags alias scene-wide; stage-1 segment selection must come from the
  EXISTING gridint snapshot channel instead.
- T-corner: encode ~0.2M cis-MAC/kf (place-encode class, ~free); query
  ~38M cis-MAC × segments, on-demand only.

VERDICT: the map CAN be queried for objects from camera data — O(D)
algebra end-to-end, bounded storage, mechanically exact — and today's
codes make it a coarse detector/region cue (0.735 AUC, ~0.8 m), not a
precise finder. The wall is cross-view code distinctiveness +
room-wide extreme-value competition (foil max 0.7–0.9× true even with
exact codes — measured synthetically too). FILED, in expected-value
order: (1) two-stage with gridint-snapshot stage-1 (segment shortlist)
+ in-segment decode (small box kills the EV pool); (2) multi-view query
bundles (the robot sees the object across frames — Nq×3 signal); (3)
richer/multi-scale codes (census16, c8+c16); (4) SPOT venue via
lidar-projected depth once extrinsics land (the standing unlock).

## 2026-07-15 — deploy6d: the lidar+cam 6-DoF deploy stack measured at its stored/structured/fused operating point (experiments/deploy6d.py, TUM fr3, TWO vector spaces per the two-map law)

User: "continue experimenting for deploy of lidar + cam 6DoF" + "(run
as two distinct vector spaces)". Lidar channel = house azel3d ladder;
vision channel = its OWN azel lattice on a camera-range ladder
(0.35-2.8 m); fusion only at the estimate level. Four deploy questions:

**1. Quantized-anchor 6-DoF verify** (stored {v0, 6 derivative vectors}
at house phase-only quant, fresh query; 80 small-motion pairs):
  lidar-cloud float 1.44° → 3b 2.33 / 2b 2.20
  vis-grid3D  float 1.32° → 3b 2.23 / 2b 2.68
The stored-anchor verify survives the 2b/3b store at 2.2-2.7° med on
0.3-6° motions — a usable consistency GATE (wrong anchors are gross).
**2b ≈ 3b (lidar even inverted): the ~+1° cost is MAGNITUDE FLATTENING,
not phase resolution** — FILED: per-ring magnitude scales on the stored
derivative vectors (16 scalars × 6 vectors) should recover most of it.
Rotation-search verify (5-20°, frozen permutations on the stored
vector): float 6.9° / 3b 6.9 / 2b 7.1 — quantization FREE in the
content-overlap-limited regime, as everywhere.

**2. Lidar-structured depth** (elevation band + N-beam row masks vs the
banked random-dropout curve; 15 wide-baseline SE(3) pairs): full
registered 6.3°/0.167 m → band+64-beam 9.7°/0.223 → 16-beam 10.6 →
8-beam 7.9 → 4-beam 9.5°/0.266. The hit is the BAND restriction
(~+3°/+0.06 m, vertical-extent conditioning), and the curve is FLAT
from 64 down to 4 beams — **lidar-projected depth suffices for the
coarse-gate/landmark tier at ANY realistic beam count**; the extrinsics
unlock does not need dense beams.

**3. Ego-motion service** (80 adjacent pairs): visual linear 1.32° →
**visual GN×2 1.13°** (the filed fine-ring-linearization fix pays);
cloud GN×2 1.16°; **FUSED (fixed precision weights, rotation-vector
level) 1.08° med / 2.03° p90 — beats both singles on both stats**;
label-free residual selection 1.15° (fusion > selection). Deploy: run
both gyros in their own spaces, fuse ω with fixed weights.

**4. Rotation-grid geometry** (the quaternion question, measured;
uniform rotation-ball vs Euler box, frozen-permutation decode, 5-18°
pairs): euler-729 med 7.3/p90 20.1; ball-729 8.4/17.0; ball-364
8.4/18.5; ball-182 7.8/18.5. **No accuracy lever** — content overlap
dominates grid geometry; a uniform ball at QUARTER the candidates
matches the Euler box (7.8 vs 7.3, within noise), so ~2-4× T-corner
savings are available if ever needed, and the analytic answer to
"would quaternions help" stands: bookkeeping yes (future integrator),
representation no.

### deploy6d tuning round 2 (dtune/fuse2 + scratch_lidstore): ring-magnitude store REPLICATES both channels → verify recipe 3b+ring-mag ≈ 1.7°/35-38 mm; ladder and solver-fusion negatives

- **Per-ring magnitude scales (the filed fix) pay on BOTH channels**:
  vision 2b flat 2.68° → 2b+rs 2.05 → **3b+rs 1.73** (float 1.32);
  lidar 2.20 → 1.85 → **1.72** (float 1.44). Translation 44→35 mm /
  47→38 mm. Cost ~4 scalars/vector (≈50-100 B/anchor across the 7
  stored vectors). DEPLOY VERIFY RECIPE: 3b phase + per-ring magnitude,
  ~1.7° rot / ~37 mm transl small-motion consistency on both channels.
- **Vision 6-DoF ladder: the v1 judgment pick (0.35-2.8 m) stands** —
  finer/coarser/6-ring alternatives all equal-or-worse (1.32 vs
  1.36-1.57). Space is robust to the ladder in this regime.
- **gradmag weights do NOT transfer to derivative solves** (1.38 vs
  1.32 int) — the banked "gradmag = ego-motion weights" law was about
  place-vector repeatability, not the gyro; intensity weights stay.
- **Solver-stacked two-space GN LOSES to post-hoc ω-averaging** (1.13
  vs 1.08 rot; transl equal 26 mm) — independent per-channel solves +
  fixed-weight averaging is optimal here (stacking couples the
  channels' linearization errors); deploy keeps post-hoc fusion.
- First TRANSLATION numbers for the fused ego-motion service: **26 mm
  med / 42 mm p90** on ≤6 cm steps (both fusion forms).
- **Verify space is INDIFFERENT on the lidar side** (scratch_verifyspace,
  float, 40 pairs): D240 house 1.13°/31.5 mm ≡ D1920-coarse
  1.14°/31.6 mm — so the verify set lives in D240 by BYTES (6 D-vectors
  × 240 × 3b+rs ≈ 0.6 KB/anchor vs 4.3 KB at D1920); the anchor's
  D1920 place snapshot stays place-only.

## 2026-07-15 — object-in-map round 2 (experiments/objmap2.py): the two filed levers WORK — stage-1 shortlist perfect via the existing snapshot channel, multi-view queries localize at 0.17 m, combined detection 0.805

TUM fr3, per-segment maps, foils through the SAME pipeline; n=16
queries (48 drawn, depth-spread + multi-view filters); poses DIAGNOSTIC
(deploy: gyro-chained).

- **Stage-1 (whole-query-frame gridint place sim vs the per-map-kf
  snapshot library — ZERO new storage): 25/25 shortlist hit rate**
  across all runs (top-3 of 27 segments), and the frame-sim ALONE
  detects foreign-scene foils at AUC 0.732.
- **Multi-view queries (3 frames, world-gated cells; the robot's own
  poses) are the LOCALIZATION lever**: census 2stage-3view err med
  **0.165 m** (n=16; 0.124 at n=9) vs ~1.3 m banked single-view; int
  0.544 m (p90 1.05).
- **Detection = COMBINED evidence**: stage-2 template score alone is
  weak (0.47-0.69 — box-scoped EV cancels in the real/foil ratio), but
  z-combined stage-1 + stage-2 reaches **0.805 (int, 3-view)** / 0.704
  (census).
- **Complementarity law: int-FPE detects, census localizes** (sharp
  binary codes need multi-view signal but then pinpoint; graded FPE
  codes separate scenes). DEPLOY RECIPE: two-family segment maps
  (~2×240 B @2b), detect via int-combined, localize via census
  multi-view; stage-1 shortlist from the snapshot library.
- Residual wall: p90 localization 1-1.5 m (a hard tail of queries
  never decodes — texture-poor or view-unstable patches).

## 2026-07-15 — ISM330DHCX IMU IO/pinout prepared, SPI rung 1 sim-gated + flash-ready (user: baseline or fusion; hw/ecp5/imu-pins.lpf, rtl/{spi_reg,tb_spi,top_ism330_id}.v)

Role in the architecture (three hats): (a) gyro = high-rate drift
BASELINE against the fused visual+cloud service (1.08°/step); (b)
accel gravity = ABSOLUTE pitch/roll anchor — the drift-free attitude
reference the 6-DoF layer lacks; (c) the FINDINGS §6
independent-absolute-cue class (IMU residual) for wall-crossing.
Pinout: SPI mode-3 4-wire on the Pi-header's native SPI0 positions
(SCLK h23/gpio11, MOSI h19/gpio10, MISO h21/gpio9, CS h32/gpio12,
INT1/2 h38/h40) — zero collisions with the camera map;
`make build/full.lpf` composes board+cam+IMU (unused LOCATEs are safe
for any top, like the vendored board LPF itself). `spi_reg.v` (16-bit
{RW,addr7,data8} register master, SCLK 3.125 MHz of 10 max): `SPI
PASS: WHO_AM_I 0x6B read, write captured, reread OK` vs a bit-level
ST-style slave (`make sim-spi`). `top_ism330_id.bit` built at 224 MHz:
any UART byte → {WHO_AM_I, INT status}; expect 0x6B. Ladder filed in
README: config+polled reads → FIFO stream with FPGA timestamps on the
SAME 50 MHz counter as DVP/lidar (the delay-design single-clock rule;
the IMU becomes the temporal anchor stream) → gravity-anchor and
gyro-baseline fusion experiments.

---

## 2026-07-15 — sspax: JAX efficient core + the approximate-permutable ring-stagger sphere

**Directive:** "build a more efficient version we can run tests in (JAX) …
focus on 6DoF, the fib sphere should be formulatable in an approximate version
as permutable like the other structured variant (approx same distance points on
sphere)" → clarified: "structure a sphere out of rings with half offset between
layers to have approx equal distance between points" + "provide both quantized
and float option" + a 50% GPU-memory-headroom courtesy for a co-running agent.

New top-level package **`sspax/`** (efficient impl, imports the frozen core
only — rule 1). JAX float64, `jit`/`vmap`, GPU footprint capped at ≤45% VRAM.
`python3 -m sspax.bench <parity|uniform|rot|resolution|so3|disp|speed>`,
`--quant nph,nmag` for the FPGA-quantized column.

### parity (the port is bit-faithful)
JAX encode / `perm_of` / `q_polar` reproduce `experiments/lattice3d.py` and
`sspslam.quantized.q_polar` to **4.3e-16** (index+sign permutation exact). The
efficient core is a faithful re-encoding of the shipped algebra, not a new one.

### ringstag3d — rings-with-half-offset sphere (the new formulation)
Latitude rings, full 2π azimuth/ring (antipode = conjugate), per-ring count ∝
`cos(elev)` apportioned to an exact budget, consecutive rings staggered by half
an azimuth step. Each ring equal-azimuth over a FULL circle ⇒ yaw by a multiple
of that ring's step is an exact cyclic shift (a true permutation, **no** conj —
cleaner than the half-circle 2D lattice); different per-ring counts ⇒ a common
yaw is *approximately* a permutation (per-ring integer snap, `yaw_perm`).

- **uniform (its design goal — WIN):** nearest-neighbour arc over the full
  antipodal set (N=60 dirs/scale): ringstag3d **cv 0.081 / min-arc 12.7°** beats
  fib3d (0.102 / 11.1°) and azel3d (0.111 / 11.5°). The staggered ring layout is
  the most equidistant of all layouts tested. (5 rings, az counts
  [18,16,13,9,4].)
- **rot (the cost — resolution-bound):** at the shipped D=240 the approximate
  yaw is coarse (cos ≈ 0.82 @ off-lattice angles) — spreading only 60 directions
  over a 2-sphere leaves each ring a 20–36° azimuth step. az2d/azel3d stay exact
  ONLY at lattice angles (15°/30°); at off-lattice 5° ringstag3d (0.889) beats
  azel3d (0.798).
- **resolution (when it becomes good):** ringstag3d yaw fidelity vs
  directions/scale at off-lattice 7°: **0.816 (D240) → 0.944 (D960) → 0.984
  (D1920) → 0.970 (D3840)**. The stored `encode_dvda` first-order d/dα correction
  CROSSES from hurting to helping at finest-step ≈6° (n_dir≥480); a **gated**
  correction (apply only where per-ring residual <3.5°, known at query time)
  never hurts and reaches 0.987 @ D1920. Verified the derivative independently by
  finite differences (‖fd+dv‖/‖dv‖ = 2.8e-3; dv/dθ = −dv/dα, sign −δ correct);
  it is a valid first-order tool but useless past δ≈2°, which is why the
  coarsest polar ring (4 az, 90° step) pins the worst residual at low D.
- **so3 (honest negative for ringstag):** SE(3)-rotation decode over a ±15°
  Euler grid (5° steps, floor ~3.5°): az2d **16.7°** (the 2D+heading system is
  blind to pitch/roll — confirms out-of-plane needs a 3D lattice), fib3d 6.0,
  azel3d **4.7**, ringstag3d 7.4, rand3d 4.2. ringstag3d's win is uniformity,
  NOT decode accuracy — its non-commensurate rings give noisier approximate
  permutations off the yaw axis.
- **disp:** all 3D lattices recover z-inclusive translation at the grid limit
  (med 0.079 m @ |d|=0.3); az2d searches dz=0 and is blind to the z-component
  (0.150–0.293 m). 6-DoF translation needs the 3D lattice; ringstag3d matches
  the others exactly.

### quantized vs float (both provided, per directive)
FPGA polar quant (16 phase bins / 4 mag levels) on the STORED vectors costs
**~1–2°** on SO(3) decode (azel3d 4.7→6.8, ringstag3d 7.4→9.5, fib3d 6.0→5.9)
and **~0.05** on yaw cos (ringstag3d 0.889→0.835 @5°) — the storage model
survives the sphere layouts, consistent with the 2D quantization ledger.

### speed (the point of the port)
Batched SO(3) decode (64 clouds × 343 candidates, D=240): **JAX 1.0 ms/batch**
(0.43 s warm+compile) vs **numpy 695 ms/cloud** (perm rebuilt per candidate).

**Verdict.** The directive's hypothesis holds: the rings-with-half-offset sphere
is simultaneously the most isotropic layout AND per-ring exactly / globally
approximately yaw-permutable — the reconciliation of fib3d's coverage with
azel3d's algebra. Quantified cost: the approximation is resolution-bound (coarse
at the shipped D=240, accurate >0.98 only at D≈2k or with the gated derivative
correction) and it does not beat azel3d on SO(3) decode. It is the layout of
choice when isotropic coverage matters and a larger direction budget is
affordable; azel3d stays preferable for exact common-yaw at a tiny D. Full
writeup + run recipes in `sspax/README.md`.

---

## 2026-07-15 (cont.) — sspax large formulation sweep (6 batches, cam + lidar + combined)

**Directive:** "run a large sweep across various formulations… both cam and
lidar formulation as well as the combined… split into 10-min sweeps so you can
integrate insights into the next batch." Six insight-driven batches on the JAX
core, float **and** quantized, deterministic synthetic geometry (anti-oracle),
GPU ≤45% VRAM. Runners `sspax/sweep*.py`; raw CSVs `scratch/sweep_b{1..6}_*.csv`;
full writeup `sspax/SWEEP_RESULTS.md`.

Generalized the ring family to knobs (elevation spacing arc|area × azimuth
apportion cos|const × stagger 0|½ × n_rings) + synthetic pinhole/omni camera
(bearings on S²) + VSA fusion (shared-lattice bundle vs concat) + a place-
separability metric (same-place vs different-place AUC under yaw-rotation search).

**Winner:** the permutable staggered-ring sphere **`ring_arc_const_s0.5_r6`**
(equal-azimuth-per-ring, arc elevations, ½ stagger, ~6 rings) on a **wide oct6
ladder (0.25–8 m)** is the cross-modality winner: SO(3) decode **3.0°** (tightest
p90 6.0), place-rec **0.910 [0.861, 0.956]**, quant-robust. Beats the more-
isotropic cos-ring (wins pure uniformity, cv 0.043, but decode outliers p90 11.5)
and azel3d.

Key findings:
- **Uniformity does NOT predict SO(3) decode** — decode is nearly geometry-flat
  (3.0–4.5°); `rand3d` (worst cv 0.43) decodes best (3.0°). Isotropy ↔
  permutability is a genuine trade: cos apportion → uniformity, const apportion
  (more azimuths on polar rings) → yaw permutability (yawRing 0.919 vs 0.886).
- **Translation decode is 0.074 for every 3D layout** (grid-limited, not a
  discriminator); az2d blind to dz (0.15–0.29) — confirms 6-DoF needs the 3D
  lattice, again.
- **Camera rewards permutability** (const-ring wins so3omni 3.1 + placeOmni
  0.857 + placeNarrow 0.662), unlike lidar's decode-flatness. Omni (360°) place-
  rec strong (0.85, rotation-equivariant); narrow FOV hard (0.64).
- **Geometric cam+lidar fusion is REDUNDANT** — naive bundle dilutes below
  lidar-alone; concat worse than bundle. Best geometry + narrow FOV gives a
  small real lift (0.892→0.909). Fusion pays only with an *independent* cue
  (appearance) — the same "independent absolute cue" wall as FINDINGS §5–6.
  Bundling is memory-free (same D), so the system win stands regardless.
- **Resolution** lifts yaw (0.82→0.99 at D=1920); place-rec **saturates ~0.86**
  (content-limited). **Wider ladder helps place-rec** (oct6 best everywhere,
  const-ring 0.901); camera-tight high-freq ladder hurts.
- **Deploy:** place-rec near quant-invariant (**94 B/anchor** @ 4ph2mag holds
  0.844); yaw registration is the bit-sensitive metric (0.864→0.649). Recommend
  4ph2mag/94 B for place-rec, 16ph4mag/184 B for registration.

Honest caveat: place-rec geometry CIs overlap (±~0.05) — the *ladder* effect and
the const-ring's decode-tail are the robust signals; the geometry deltas are
modest. Synthetic-geometry study; a texture/appearance camera channel (to make
fusion non-redundant) is the filed next step.

---

## 2026-07-15 (cont.) — sspax learned front-end + semantic (queryable) SSP map

**Directive:** learn FPGA-fittable pre-processing for BOTH vision and lidar,
end-to-end for SLAM; pretrain vision on external data; make the map queryable by
object ("highlight where the chairs are"). Unifying architecture (user's spec):
each CNN emits a BINARY descriptor; bind each set bit's role hypervector with the
feature POSITION (SSP phasor) and bundle; **significance = bit count**.
Modules `sspax/learn_lidar.py`, `sspax/semantic.py`, `sspax/vision/tinycnn.py`;
writeup `sspax/LEARNED_FRONTEND.md`. Runs on CPU (box cuDNN mismatches JAX GPU
conv → also frees the GPU). Anti-oracle: GT poses only align contrastive targets.

- **Learned lidar saliency (826 params):** BEV-rasterize scan → 3-conv CNN scores
  per-cell saliency on a 2×-downsampled 24×24 map → top-k salient cells → SSP.
  The differentiable SSP encode lets a contrastive place-rec loss backprop into
  the CNN. At a tight k=32-feature budget with interior clutter to reject,
  rotation-invariant matching: learned **0.940** vs uniform-random **0.707**
  place-rec AUC (**+0.232**, GPU-converged; im2col convs sidestep the box's
  cuDNN mismatch so training runs on GPU at 400 steps in ~10 s). 826 params = 0.8 KB int8 / 103 B BNN. Learns to
  spend a sparse budget on repeatable structure, drop moving clutter.
- **Semantic binary-binding map:** `map += Σ_{i∈bits} ROLES[i] ⊗ encode(xyz)`;
  query a class by unbinding its roles and decoding spatial density (peak ∝
  |query ∩ feature bits|). Query 'chair' (16 objects, D=360, k=12): precision
  **1.00**, recall **1.00**, pos err **0.04 m**, **4.0×** contrast vs non-chairs.
  Significance ∝ committed bits (readout 5.4/9.7/12.9/14.9 for 3/6/9/12 bits).
  Capacity (one bounded D=360 vector): recall 0.75/0.38/0.03 @ 10/20/40 objects
  — saturates ~15–20 objects; more bits/object raises SNR; query cost O(D)
  regardless of object count.
- **Learned vision (34 k params, 4.2 KB BNN):** tiny CNN + 64-bit binary
  descriptor head, pretrained on CIFAR-100 (has chair/couch/table/bed/wardrobe →
  a furniture detector). Built + FPGA-costed; pretraining auto-runs on CIFAR
  download completion (slow external mirror).
- **FPGA cost:** lidar 103 B (BNN) / vision 4.2 KB (BNN) / bind O(D), D=360; both
  fit the ECP5 fusion target, lidar fits ice40. Map = one bounded vector, object
  query = O(D) phase ops — history-free, consistent with the project thesis.

**Verdict.** The learned front-ends give real gains at FPGA scale (sparse
trackable lidar features +0.10 AUC; a queryable semantic map at 4 cm / P=R=1.0
within capacity), and the binary-descriptor binding cleanly unifies FPGA
efficiency with VSA object binding. The sweep's fusion caveat resolves here: the
learned *appearance* descriptor is the independent cue that makes cam+lidar
binding non-redundant — the reason to pretrain vision on real images. Open:
vision pretraining accuracy (pending download); scaling capacity with D; an
end-to-end joint (lidar+vision) SLAM+semantics objective.

## 2026-07-15 — sspax TRANSFER GATE (sspax/realbench.py + scratch_attrib240): the sweep's ladder finding is REAL and venue-scaled, the ring geometry is niche; a D240 recipe jump falls out (azel-oct6 0.892)

The incoming sspax sweep (ad6b368, synthetic JAX surface) put up: ring-
stagger-const geometry + oct6 (0.25-8 m) ladder as cross-modality winner.
Real-data gate (pure numpy; sspax/__init__ now degrades gracefully
without jax so this runs on the deploy box):

**school place (est-labels diagnostic, rot-searched 24):**
  D240:  azel-house 0.678 (banked-class) | azel-OCT6 **0.892** |
         ring-house 0.841 | ring-oct6 0.871
  → BOTH levers are individually large (+0.21 ladder, +0.16 geometry)
    but DO NOT COMPOSE — azel-oct6 alone is the best D240 point,
    EXCEEDING the banked D960 (0.858) at 1/4 the budget. LADDER IS THE
    LEVER, again — now at the matcher budget.
  D1920: ring-coarse16 **0.940** (beats the banked ceiling 0.928; same
         ladder → geometry effect isolated at +0.012) | ring-oct6 0.928
  → the VENUE-SCALE ladder law holds on their geometry too (coarse16 >
    oct6 on the building-scale run — their synthetic room capped at 8 m).
**TUM vision landmarks:** ring-oct6 0.766 with rev 0.433 vs azel3d 0.794
  / rev **0.791** — the ring geometry COLLAPSES reverse-view place;
  ring-yaw search does not help real narrow FOV (0.755 ≤ raw; the omni
  "camera rewards permutability" claim does not transfer). Real-motion
  SO(3): ring med 7.3 / p90 **15.4** vs azel 6.9 / 23.8 — the tighter-
  tail claim REPLICATES.
**verify (small motion):** ring dirs × the tuned camera-range ladder =
  **1.12° / 27.6 mm — new best** (W_vis3d 1.32/31.8); ring × oct6 1.40.

VERDICTS: (1) adopt-candidate: azel-oct6 D240 for the matcher-budget
place layer and ring-coarse16 D1920 for the anchor layer — BOTH pending
the rule-5 multi-venue gate (classroom + a second school run; single
venue, est labels so far). (2) vision place stays azel3d (reverse-view
collapse disqualifies ring). (3) verify space adopt-candidate: ring ×
VIS_LAMS. (4) The sweep's synthetic→real transfer record: ladder law
GENERALIZED, geometry claims SPLIT (decode-tail yes, place niche,
camera no).

### deploy-rate chaining (deploy6d chain, fr3 @15 Hz stride-2 parse): chaining small steps BEATS the single-shot keyframe decode — the high-rate service is the odometry prior, not just latency

60 keyframe windows of 3×15 Hz steps, stacked-GN two-space gyro per
step: **chained 1.02° / 27.5 mm** vs single-shot-over-the-window 1.09°
/ 27.2 mm. Composition error does not accumulate faster than the
big-step linearization it replaces (and rotation improves). At the
deploy rates (20 Hz cloud / 120 fps visual) the per-step motion is
4-8× smaller still — trend favorable. The temporal design (interval-
matched buffered fusion + delayed-state re-anchoring) stands on this:
the 120 Hz chain is the master integrator. IMU delay-bench redo filed
for when the ISM330DHCX lands.

## 2026-07-15 — full-board CNN feasibility envelope (hw/ecp5/host/cnn_budget.py): BNN lanes are the regime; every TRAINING_PROGRAM candidate class fits

Full Icepi-Zero allocation model with explicit SLAM reserves (10 DSP,
10k LUT, 57 KB EBR kept; SDRAM 32 MB @ ~170 MB/s practical, CNN share
60 MB/s + 8 MB weight capacity; standing sensor traffic ~13 MB/s):
- **BNN XNOR-popcount lanes: ~40 lanes from 7k LUT = 394 Gbop/s** →
  3.3 Gbop/frame @120 fps — even a yolo-nano-CLASS net fits at full
  camera rate as a BNN (the banked int8 "OUT" verdict stands for int8
  only). int8 is DSP-scarce: 18 DSP → 11-22 MMAC/frame @120, 270-540
  @keyframe (SDRAM-streamed weights).
- Candidates: cellweight-A (regime A, 16 MMAC) fits @120 int8-packed;
  tinycnn-34k @60 as BNN; seg-mnet/mnetv2-class (regime B, 36-87 MMAC,
  17-40k params) @5 Hz; lidar saliency trivial @20.
- Search prior for the architecture search: BNN-first, first-layer
  int8; early stride-2 (line buffers are the EBR cost); keyframe tier
  is where capacity lives. Ceilings written into TRAINING_PROGRAM.md;
  README headroom item 3 rewritten around the program.

---

## 2026-07-15 (cont.) — sspax P1: ladder-vs-world-extent curve (explains the venue-scale artifact)

**Directive (msg.txt P1):** explain WHY the sweep's oct6 (0.25–8 m) ladder won
the synthetic rooms but the deploy transfer gate found coarse16 (0.5–90.5 m)
wins building-scale school_run2. `sspax/ladder_extent.py`: fixed geometry
(ring_arc_const_s0.5_r6, D=360), rotation-searched lidar place AUC, sweep world
EXTENT × ladder λ_max. Synthetic, anti-oracle (same-place = jittered view pairs).

| extent \ λ_max | 8 m | 22.6 m | 90.5 m |
|---|---|---|---|
| **8 m** (room) | **0.909** | 0.905 | 0.876 |
| **30 m** (hall) | 0.835 | **0.890** | 0.864 |
| **80 m** (building) | 0.776 | 0.798 | **0.814** |

The best λ_max tracks the venue extent **exactly** (8→8, 30→22.6, 80→90.5). The
mechanism: place discriminability peaks when the coarsest ring ≈ scene scale;
below that the coarse rings can't resolve global layout, and the fine rings
alias across a large room. So the sweep's "oct6 best" was REAL but venue-scaled —
oct6 matched the 8 m rooms; coarse16 matches buildings. Confirms the deploy
agent's rule-5 transfer critique at the mechanism level: **a fixed ladder is
venue-suboptimal; λ_max should track venue extent.** This motivates the learned
scale-modulation (TRAINING_PROGRAM P4): let the net discover the allocation
instead of hand-picking λ_max per venue.

---

## 2026-07-15 (cont.) — sspax P4 scale-modulation: mechanism built, HONEST synthetic negative

**Directive (TRAINING_PROGRAM.md P4):** learn the ladder allocation via a
THERMOMETER blanking head — per-cell finest-useful-scale cutoff, rings finer
than cutoff blanked (on-fabric: per-ring comparator, zero multipliers).
`sspax/learn_scale.py`: ScaleNet (1778 params, cuDNN-free convs) + differentiable
per-ring-masked SSP encode, contrastive place loss across mixed extents {8,30,80 m},
full ladder 0.25–90.5 m (D=360). Straight-through hard blanking at eval.

**Result — honest negative on the synthetic surface.** No aliasing regime where
blanking helps: aligned views make the full ladder trivially perfect (1.000 at
every extent), so the learned head correctly converges to **cutoff≈0 (keep all
rings)** and lands at 0.925 mean (the learned saliency head slightly hurts a
saturated task, −0.075 vs full ladder); full ±180° rotation drops every ladder
to chance (~0.51) — box-rooms are too self-similar under arbitrary heading. The
mechanism is sound (the learned cutoff is the *correct* policy for a no-aliasing
task), but the synthetic world cannot reward blanking. This reinforces the
deploy agent's rule-5 discipline: **synthetic means don't decide ladder policy;
the learned-blanking advantage needs real building-scale aliasing (school_run2).**
The module is the transfer-gate candidate; `ladder_extent.py` (best λ_max tracks
venue extent) is the transferable synthetic result that motivates it.

Budget (vs the Fable-agent CNN envelope, this push): ScaleNet 1778 params ≈ 222 B
BNN — trivially inside the @120 fps BNN tier (3.3 Gbop/frame); the deploy-scale
k=2000 learned front-end (P5) has vast headroom. Mechanism is BNN-first ready.

---

## 2026-07-15 (cont.) — sspax: CReLU param efficiency for the learned front-ends

**Directive (user):** try CReLU for more param efficiency. Added `crelu` to
`sspax/nnconv.py` (concat[relu(x), relu(-x)] — both activation phases from the
same filters; on-fabric a sign-flip + relu, no extra MACs) and a `crelu` flag to
SaliencyNet. Comparison on the learned lidar saliency (k=32 place-rec, GPU, 400
steps, uniform-random ref 0.707):

| config | params | place-AUC |
|---|---|---|
| ReLU  ch8 | 826 | 0.940 |
| **CReLU ch4** | **422** | **0.962** |
| ReLU  ch4 | 270 | 0.848 |
| CReLU ch8 | 1418 | 0.887 |

**CReLU ch4 beats ReLU ch8 at HALF the parameters (+0.022 AUC).** ch4+CReLU
gives an 8-wide activation from 4 filters, outperforming 8 ReLU filters; the
half-filter ReLU control (ch4) collapses to 0.848, so the gain is CReLU's
phase-preservation, not just width. AUC/param nearly doubles (10.9e-4 vs
5.3e-4). CReLU ch8 (1418) overfits the small synthetic task — the sweet spot is
CReLU at LOW channels, which is exactly what the FPGA weight budget wants.
Adopt: CReLU-low-channel as the default for the deploy-scale front-ends.

---

## 2026-07-15 (cont.) — sspax: REAL-DATA transfer gate reproduced on this box (school_run2)

Fetched the SPOT school_run2 pointclouds (47 shards, 2.3 GB, HF
lorinachey/spot-telluride-workshop-dataset) + parsed to scans.npz; run2 lacks
odometry_lio_sam on HF, so a placeholder zero-pose ref_lio.npz was dropped in
(anti-oracle: est-labels use the pipeline's OWN lidar-only estimate `fin`, never
the reference). Ran `sspax/realbench.py school` (est-labels DIAGNOSTIC,
rot-searched 24):

| arm | this box | deploy agent (msg §3) |
|---|---|---|
| ring-oct6 D240 | 0.870 | 0.871 |
| ring-coarse16 D1920 | **0.939** | 0.940 |
| ring-oct6 D1920 | 0.927 | 0.928 |
| azel-oct6 D720 | 0.881 | — |

The sspax ring lattice **reproduces the deploy transfer numbers on real data**
(±0.001). Confirms on REAL building-scale data what `ladder_extent.py` predicted
synthetically: **ring-coarse16 (0.939) > ring-oct6 (0.927)** at D1920 — coarse
rings win at building scale (+0.012), the venue-scale law. The learned front-end
program (P5) can now be tested on this venue directly.
## 2026-07-15 — MULTI-VENUE rule-5 gate (sspax.realbench mv): azel-oct6 D240 + ring-coarse16 D1920 PASS; the ladder lever holds on HONEST labels; ring-at-D240 is venue-dependent (not adopted)

The transfer-gate winners were single-venue (school_run2) with
OWN-ESTIMATE diagnostic labels. Gate across three venues, six arms
(house azel D240 baseline, the candidates, + controls; rot-searched
24 both families; anti-oracle: labels score pairs only):

**classroom 'spot' (HONEST withheld kinematic odometry, 179/4144
same/diff pairs):**
  D240:  azel-house 0.817 | azel-OCT6 **0.947** | ring-oct6 0.904
  D1920: azel-coarse16 0.904 | ring-coarse16 **0.976** | ring-oct6 0.976
**school_run1 (est DIAGNOSTIC — LIO diverged, ok=0/2075, so no honest
window; distinct multi-room traverse; THIN 21 same-pairs — direction
only):**
  D240:  azel-house 0.593 | azel-oct6 0.745 | ring-oct6 0.822
  D1920: azel-coarse16 0.787 | ring-coarse16 0.816 | ring-oct6 0.815
**school_run2 (prior transfer numbers, est DIAGNOSTIC):** house 0.700 /
  azel-oct6 0.892 / ring-coarse16 0.940 / azel-coarse16 0.928.

VENUE FACT banked: school_run2's honest gated-LIO window (342/836 kf)
contains NO >=40-gap revisits within 1.0 m (first same-pairs at 1.5 m:
21) — honest-label place separability is structurally unmeasurable
there; classroom carries the honest verdict.

VERDICTS: (1) **azel-oct6 D240 ADOPTED for the matcher-budget 3D place
layer** (+0.130 honest, +0.152/+0.192 diagnostic over house — every
venue, both label kinds). Sharpest form of the ladder law: 8 azimuths
x 6-octave ladder beats 60 directions x house ladder at equal D —
azimuth-snap coarseness is cheap, scale coverage is not. (2)
**ring-coarse16 D1920 ADOPTED for the anchor layer** (0.976 honest /
0.940 / 0.816 — best-or-tied everywhere; ring-oct6 ties on classroom =
ladder saturates at high D). (3) ring geometry at D240 NOT adopted
(0.904 vs azel 0.947 honest classroom; wins run1; splits run2 — venue-
dependent, and vision already disqualified ring for place). Runnable:
`python3 -m sspax.realbench mv` (scratch_mvgate.py = the run artifact).

## 2026-07-15 — DELAY bench (experiments/delayfuse.py): interval-matched buffered fusion is latency-INVARIANT; timestamp-ignorant mixing loses the whole slow-tier gain by one keyframe interval of delay

The temporal-design question measured (fr3 @15 Hz stride-2 proxy, 40
gated horizons of 12 steps = 0.8 s; fast tier = vision-only per-step
GN chain, slow tier = cloud-only GN per 3-step keyframe interval
arriving d fast-steps late; identical omega-average fusion math both
policies — window bookkeeping is the ONLY difference; selftest gates
d=0 equality + algebra identities + synthetic-reversal ordering):

  fast-only         rot med 4.21 p90 6.87 | transl med 125.6 mm
  d=0 (both)        3.76 / 6.70 | 111.7 mm      (fusion gain baseline)
  naive   d=1/2/3   3.88 / 4.09 / **4.25** (== no fusion at 200 ms)
  matched d=1/2/3   **3.81 / 6.53 | 115.2 mm — constant in d**

The buffered policy re-anchors each measurement against exactly its
own interval, so arrival time cannot matter (only the last window's
measurement missing the horizon differs from d=0: 3.76→3.81). The
naive policy fuses against the newest window; its error grows
monotonically with window mismatch and the fusion value is fully
destroyed at d = one keyframe interval. Deploy reading: at 5 Hz
keyframes the expected 50-100 ms cloud latency = d~0.25-0.5 intervals
— already in the naive-degradation regime; the ring buffer (one
keyframe interval of per-step increments, delayed-state re-anchoring)
is REQUIRED design, not nicety. Honest tail note: naive shows slightly
better p90 transl (171-186 vs 182 mm) — noise-level, all medians and
rot tails favor matched. IMU tier redo filed for when the ISM330DHCX
lands (same harness, 3-tier).

## 2026-07-15 — ISM330DHCX rung-3 streaming top pre-built and sim-gated (IMU day = wiring only, camera-eve pattern)

`hw/ecp5/rtl/top_ism330_stream.v` + `tb_ism330_stream.v` (bit-level
ST-style slave with register file + DRDY generator + UART BFMs):
boot config {CTRL3_C 0x44, CTRL1_XL 0x60, CTRL2_G 0x60, INT1_CTRL
0x02} then one 21-byte frame per INT1 rise — `AA 55 | ts48 (SHARED
free-running 50 MHz counter, 20 ns units — the cross-sensor time base
the delay design needs) | 12 B gyro+accel | XOR`; 'G'/'g'/'I'/'R'/
'W'/'r' command set (passthrough = rate/FS tuning without rebuild);
no-slave-safe. `make sim-imu-stream` = **IMU-STREAM PASS** (cfg 4/4
captured exact, hello 6B, 3 frames sync+xor+ts-spacing+data exact,
'g' silent, passthrough OK); build clean at **170.2 MHz** vs 50
constraint on build/full.lpf. Host gate `host/hw_imu.py` (hello →
config readback → rate/drop/monotone gates → gravity + gyro-bias rest
physics → scratch/imu_log.npz for the rung-4 fusion baselines).
Defect banked: time-zero `always @*` X — a case-mux fed by an
initialized-but-never-changed select shifts X out the SPI for the
entire first transaction (sim-only, silicon unaffected); fixed
structurally with a function-ROM (evaluates at call time). Second
sighting class of "tb stimulus/initialization races" — prefer
function-ROMs over always@* muxes for boot-path constants.

## 2026-07-15 — network geometry PINNED (user directive b): unified dual-objective nets at full/half sensor res; budget verified

User: both CNNs train for full or half res; the vision CNN optimized
for BOTH tracking and classification — SEGMENTED classification, so
features are spatially encodable into the map. Encoded in
TRAINING_PROGRAM.md ("Network geometry") + cnn_budget.py ARCHS:
ONE net per modality, shared trunk, tracking head (weights +
thermometer cutoff) @frame rate + per-cell seg/label head @keyframe
rate; vision Y8 320x240/160x120, lidar ring raster 3x1024/3x512
(rings-as-channels; BEV demoted to mechanism-study surface).
Budget verdicts (cnn_budget.py): uni-trunk+track FULL 11.3 MMAC =
fits @120 int8-packed 2x headroom (HALF 8x); 40-class seg head on the
40x30x64 trunk 16.7 MMAC @5 Hz (32x headroom, 14k params EBR);
lidar-track FULL 1024x3 2.5 MMAC @20 trivial; label head @kf trivial.
CONSTRAINT surfaced: concurrent full+full line buffers (13.1 vision +
22 KB lidar) + weights = 54.5 of 55 KB CNN-EBR — edge-exact int8;
half-res lidar (11 KB) or BNN weights restores margin. msg.txt §0
addendum re-shapes the other agent's P3 (unified lidar net at ring
raster, two heads) and P4 (NYUv2 = seg head of the SAME vision net,
resized to deploy res).

### verify-space second venue (sspax.realbench verify, fr1_desk + fr1_rpy): ring×VIS_LAMS NOT adopted — the fr3 both-axes win does not replicate on translation

The last open v1.4 adopt-candidate gets its rule-5 second venue (fr1
npz already local; 14/15 small-motion pairs each — thin, medians
indicative). fr3 banked: ring-vislams 1.12°/27.6 mm BEAT W_vis3d
1.32/31.8 on both axes. fr1_desk: rot 1.94 vs 2.19 (ring better) but
transl 28.7 vs **20.2 mm** (ring +42% worse); fr1_rpy: rot 1.91 vs
1.90 (tie; ring p90 3.28 vs 4.59 better) but transl 37.2 vs
**30.8 mm** (+21%). Pattern across all three venues: the ring space is
equal-or-better on ROTATION (and consistently tighter rot p90), the
tuned azel W_vis3d is better on TRANSLATION everywhere except fr3.
VERDICT: NOT adopted — translation feeds the 26–42 mm fused ego-motion
service, so the vision verify space stays W_vis3d; ring×VIS_LAMS is
filed as a rotation-tail specialist (revisit only if a rotation-only
verify tier appears). Closes the v1.4 candidate list: 2 adopted
(azel-oct6 D240, ring-coarse16 D1920), 2 not (ring-geometry D240,
ring×VIS_LAMS verify).

## 2026-07-15 — venue-adaptive ladder preset (sspslam/lattice_presets.ladder_of_extent + sspax.realbench preset): ONE static rule reproduces every hand-picked winner and fixes the scale-mismatched cell (+0.075)

The msg-P5 deliverable, built from sspax/ladder_extent's 3×3 diagonal
without waiting for the densified fit: both ADOPTED ladders are
instances of ladder = geomspace(lam_min, EXTENT_C·extent, n_rings)
with EXTENT_C=1.0 (oct6 = geomspace(0.25, 8, 6) at extent 8; LAMC16 =
geomspace(0.5, 90.5, 16) — half-octave steps — at extent ~90). New
lattice_presets machinery (add-only; selftest extended, smoke PASS):
ladder_of_extent(extent, n_rings, lam_min), extent_of_points (robust
percentile bbox). Extent input = the OWN-estimate-registered map bbox
(deployable, anti-oracle-safe: reference labels score pairs only).

Gate (realbench preset; mv pair protocol; run2 omitted per its banked
venue fact): classroom 'spot' HONEST — measured extent 10.3 m;
preset-azel D240 0.939 vs adopted azel-oct6 0.947 (−0.008, noise
band); preset-ring D1920 0.975 vs adopted coarse16 0.976. school_run1
est-DIAGNOSTIC (21 same-pairs, direction-only) — extent 19.5 m;
preset-azel D240 **0.820 vs adopted azel-oct6 0.745 (+0.075)** — the
rule scales the ladder to the building extent where the fixed D240
recipe was room-scaled; preset-ring D1920 0.819 vs coarse16 0.816.
VERDICT: ADOPTED as the venue-adaptive fallback — matches the
per-venue hand-picked winner at every venue×budget cell and beats it
where the hand pick was scale-mismatched; one formula replaces the
per-venue ladder choice. EXTENT_C refines when the other agent's
densified lam_max*(extent) fit lands (msg P5); the learned thermometer
head (P1) must now beat THIS rule, not just the fixed recipes.

## 2026-07-15 — frozen-bank v2.2 (experiments/frozenbank2.py): the weight-semantics knob measured end-to-end; mult=8 reconciles the historical conflict table (fr101 1.725 = best stable ever) — and fr079 closes the suite-default door at the mechanism level

The FILED reconciliation ("WELL-tier -> full weight, else soft"),
implemented as ONE knob: WELL/target-tier matches (shipped
_coh_response == 1.0) at full weight; marginal matches accepted at
weak_mult x the shipped inflation. mult=1 IS v2; mult -> inf
approaches v2.1. REPLICATION GATE PASSED: mult=1 reproduces banked v2
bit-exactly (fr101 2.132, belg 2.042). Sweep (ATE; OFF = acceptance
1.881 / 2.64 / 0.981 / 0.202):

  mult     1(=v2)   2      3      4       5      6      8      16    v2.1
  fr101    2.132  2.754  1.842  1.129*  1.856  1.841  1.725  2.078  1.827
  belg     2.042  2.135  2.363  2.934   2.372    —    2.041  2.038  2.729
  fhw      0.610    —      —    0.895     —      —    0.655  0.250  0.242
  stata'   0.193    —      —    0.210     —      —    0.210  0.210  0.209
  (stata' = celld policy)

- *fr101 mult=4 ATE 1.129 audited per rule 4: KNIFE-EDGE basin
  (neighbors 3/5/6 give 1.84/1.86/1.84) — not a recipe. The STABLE
  band mult 3-8 is 1.73-1.86, ALL better than OFF 1.881 and the banked
  best 1.827; best stable fr101 in project history = **1.725 @ mult=8**
  (med 1.378 vs v2's 1.823).
- **mult=8 is the first single weight semantics to reconcile the
  fr101/belg conflict** (both win) + fhw 0.655 (1.5x) + stata 0.210
  (inside the 0.193-0.225 variant band). Acceptance sweep at mult=8:
  spot 0.036 (OFF 0.039), school_run2 med 1.062 (~OFF; scored window
  return-leg-blind as banked) — and **fr079 10.635 vs OFF 5.523
  (reproduces acceptance exactly): a 2x CRASH**.
- fr079 attribution (never gated in the earlier frozen-bank rounds):
  v2-soft 12.124 / v2.1-STRICT 9.024 / v2.2@8 10.635 — EVERY weight
  semantics crashes it (73-151 aliased raw-raw closures pass even
  strict admission). celld is INERT on fr079 (identical run, 0 freezes
  rejected): corridor aliasing passes the 0.75 place-distinctiveness
  screen while still poisoning the metric matches. This is the
  project's loop-closure wall (FINDINGS §5-6) surfacing inside the
  frozen-bank thread: admission, not weighting, is what fails on
  aliased corridors, and no appearance-level screen fixes it.
- celld ≡ cell bit-exactly on fr101/fhw/belg too — the distinctiveness
  tau binds ONLY on stata in the entire suite (its anchor twins are the
  only sub-0.75 aliases at the place-vector level).
- The mult response is NON-SMOOTH everywhere (fr101 spike at 4, belg
  dip at 4-5, fhw dip at 4): the IRLS relax basin-hops with edge
  weight. Per-venue mult tuning is NOT sanctioned (rule 5).

VERDICT: suite-default REMAINS CLOSED — now for a measured reason
(fr079-class aliased corridors crash under every variant of the
mechanism). Frozen-bank stays per-environment OPT-IN with **v2.2@8 as
the recommended opt-in form** (revisit-rich venues: fr101 1.725 / belg
2.041 / fhw 0.655; fhw-class halls may opt UP to strict or mult=16 for
0.242-0.250); stata-class keeps celld (0.193-0.210 across semantics);
fr079-class EXCLUDED. The weight-semantics question the v2/v2.1 round
filed is now ANSWERED, not open: the reconciliation exists (mult=8)
but cannot buy a suite-default because the failure that blocks it
lives in admission, not weights.

## 2026-07-15 — P5: densified ladder-vs-extent DESIGN RULE (refines the clean 3-point trend; monotone but NOT linear)

`sspax/ladder_rule.py` — 6 extents (8-120 m) × 9 lam_max candidates,
ring_arc_const, 32 places/extent, rotation-searched lidar place AUC;
n_rings 6 and 8 for ring-count invariance. This densifies the 3-point
`ladder_extent.py` result (8→8, 30→22.6, 80→90.5) that looked like a
clean linear law. The dense sweep tells a MORE HONEST, more nuanced
story:

  best lam_max per extent (n_rings=6): 4, 7, 17, 46, 123, 46 m
  best lam_max per extent (n_rings=8): 4, 11, 75, 75, 123, 123 m
  Spearman(extent, best_lam):  0.90 (n6)  /  0.97 (n8)   <- MONOTONE, robust
  linear-through-origin fit:   0.75×ext R²0.39 (n6) / 1.25×ext R²0.78 (n8)

Verdicts:
- The **directional law is robust** — bigger venue wants a coarser
  coarsest wavelength (Spearman 0.9-0.97). This confirms the mechanism
  and the multi-venue real-data verdict (rooms→oct6, buildings→coarse16)
  from two directions.
- The **precise linear coefficient is NOT robust**: 0.75 vs 1.25
  depending on ring count, and R² is weak at n6. A single through-origin
  slope is the wrong model — the relationship is monotone but flat-then-
  steep, not linear. Ring count confounds because rescaling lam_max at
  fixed n rescales EVERY rung.
- **Effect size is the real story**: AUC spread across lam_max is
  0.03-0.09 at 8-50 m (nearly flat — ladder choice barely matters in
  rooms/halls) but **0.21-0.22 at 80-120 m** (0.60→0.92 at 120 m). The
  law only has teeth at building scale — precisely the regime where a
  wrong (too-fine) ladder aliases catastrophically.

Deploy reading for the venue-adaptive preset (the static cousin of the
learned head): do NOT ship a precise `c×extent` formula (overfits the
noisy mid-range). Ship a COARSE bucket — small venue: ladder choice is
free; large venue (bbox ≳ 60 m): ensure lam_max reaches scene scale
(~extent) or place recognition collapses. Anti-oracle: deterministic
synthetic, no reference labels (jittered-view pairs vs distinct layouts),
fixed seeds, fit reported with R² so a weak rule can't masquerade.

## 2026-07-15 — read-only audit of the sspax JAX surface (rule 4): synthetic scale-negative is VACUOUS, not disproven

Second-opinion audit (dispatched agent, no code changes) of
ladder_extent / learn_scale / learn_lidar / nnconv / realbench. Two
substantive findings, both in `learn_scale.py`, both CONSERVATIVE (they
cannot manufacture a false win, so the stated negative stands) but they
invalidate the *reasoning*:

- **The "honest negative" never runs the aliasing regime it claims to
  test.** `_view(align=True)` is the only path exercised by `batch` and
  `evaluate`; the align=True branch undoes BOTH the GT rotation and
  translation, so the two views of a place collapse to the same
  canonical frame ± 0.02 m noise. The 24-yaw search is redundant
  (identity perm always wins) and the aliasing regime (align=False, full
  rotation) described in the docstring is DEAD CODE. So the synthetic
  "learned ≈ fixed" is a SATURATED task (every ladder ≈ perfect on near-
  duplicate positives), not evidence that blanking can't help.
- **Train/eval scene overlap**: `evaluate(seed0=7000)` layouts are all
  within the training seed range (24/24). `learn_lidar` is clean
  (seed0=9000 disjoint from train max 6400) — the discipline slip is
  isolated to learn_scale.
- Verified SOUND: no anti-oracle leak anywhere (pose only aligns
  targets + labels pairs, never enters an encoder/CNN input); the CReLU
  param-efficiency comparison is fair (ReLU-ch8 826p vs CReLU-ch4 422p,
  same steps/eval); all einsum/mask/pool/AUC shapes correct. Doc nit:
  CReLU DOUBLES the next layer's input width (the comment says halves) —
  numbers unaffected.

Consequence: the synthetic scale-modulation negative is retained only as
"the surface is too easy to probe blanking", NOT as "blanking can't
help". The load-bearing test is the REAL building-scale gate with
genuine rotated + re-rasterised positives and a TIME split
(`sspax/learn_scale_real.py`, P1) — which is exactly the regime the
audit says the synthetic version failed to exercise.

## 2026-07-15 — P1: thermometer scale head on REAL school_run2 — mechanism CONFIRMED (large), learned head LOSES to a fixed coarse cutoff

`sspax/learn_scale_real.py` — the real-data transfer gate the synthetic
`learn_scale.py` negative could not be (its box-rooms have no aliasing
regime; the audit above showed its "negative" was VACUOUS/saturated).
Real building-scale lidar (school_run2 full SUB16 clouds), BEV raster
extent 120 m / 64×64 (3.75 m cells), TIME split (train frames [0,144),
eval [144,240)), rot-NCE over the 24-yaw per-ring search. Positives =
temporally-adjacent kf + synthetic-yaw RE-RASTERISED same cloud;
negatives = own-estimate >4 m (est labels DIAGNOSTIC, anti-oracle stated
in-log; run2 has no honest revisits — banked). Two eval metrics:
selfrot-AUC (frame vs its rotated-and-rerastered self, rotation-searched
— the ALIASING regime) and adjacent-AUC.

**Mechanism — CONFIRMED and LARGE.** selfrot-AUC is monotone in ring
blanking (fixed global thermometer cutoff, keep rings ≥ c):

  c0(all) 0.718  c1 0.748  c2 0.772  c3 0.808  c4 0.859  c5 0.901
  adjacent:  0.879  0.875  0.874  0.871  0.872  0.849

Blanking the fine rings improves rotation-invariant place recognition by
**+0.183 AUC** (c0→c5); keeping only the coarsest ring (λ≈90.5 m ≈
building scale) is best. The fine rings are pure aliasing liability under
rotation — exactly the wall the synthetic surface could not produce. This
independently confirms the coarse16-wins-buildings law and P5's
venue-scale finding from the blanking direction. (adjacent slightly
PREFERS fine rings, 0.879→0.849 — the two objectives tension, quantified.)

**Learned head — does NOT beat the fixed global optimum (negative).**
- yaw_frac=0.5 (mixed positives): head converges to cutoff≈0 (mean 0.02,
  keep all) — the easy adjacent positives dominate the gradient; selfrot
  LEARNED 0.768 ≈ fixed c0. It never learns to blank.
- yaw_frac=0.9 (rotation-dominated): head DOES learn aggressive blanking
  (mean cutoff 5.12, per-cell spread) — selfrot LEARNED 0.836, beats
  c0..c3 — BUT a fixed GLOBAL c5 (0.901) beats it by 0.065. The earlier
  "learned beats c3" reading was an artifact of testing only c0–c3; the
  rigor extension to c4/c5 (gate-only, no retrain) caught the
  false-positive-in-the-making. The per-cell spread HURTS vs a uniform
  coarse cutoff.

**Verdict.** The mechanism transfers to real data and is worth a large
AUC; the LEARNED NET is not — a single fixed coarse-only cutoff (blank
the fine rings) is both simpler and better per venue. This matches the
synthetic "constant cutoff is the correct policy", except the constant is
now c5 (aggressive blank), because real building-scale HAS the aliasing
regime box-rooms lack. Deployable win = a zero-cost fixed coarse ladder
for the rotation-invariant PLACE layer (no net, no per-cell LUT); keep the
fine rings only for short-baseline tracking/registration. The learned
head's only remaining value is CROSS-venue (one policy for mixed extents)
— the P2 multi-venue arm is the real test of whether the net earns its
keep; single-venue it does not. Recipe reference bars (raw-cloud, not
raster): ring-coarse16 D1920 0.994 / azel-oct6 D240 0.995 adjacent — the
BEV raster path costs ~0.12 adjacent AUC vs raw clouds (raster is lossy;
all learned-vs-fixed numbers are within the raster path).

## 2026-07-15 — P1 CORRECTION / RETRACTION (rule 4 audit): the "+0.183 blanking improves place recognition" claim is a rotation-triviality ARTIFACT — RETRACTED

A read-only audit (dispatched agent, reproduced the numbers exactly from
`scratch/learned_scale_real_yaw90.pkl`) REFUTES the previous P1 entry's
central claim. I banked that entry BEFORE the audit returned — the
correction supersedes it. Honest findings:

- The +0.183 gain lives ENTIRELY in the `selfrot` metric (a frame vs its
  OWN 90°-rotated + re-rasterised self — ZERO place variation). The
  honest same-place control already in the gate — `adjAUC` (temporally
  adjacent frames, genuine small-viewpoint same-place, same 24-perm
  search, same far-negatives) — does NOT improve with blanking:
    cut:   c0     c3     c5
    self:  0.718  0.808  0.901   (+0.183 — the retracted number)
    adj :  0.879  0.871  0.849   (FLAT, then DROPS at c5)
- Root cause: as rings coarsen, the coarsest ring (λ=90.5 m spans <1
  phase cycle over ±60 m) makes a 90° rotation a near-trivial permutation
  and re-rasterisation near-lossless, so pos(self) rises 0.68→0.97 while
  neg(far) rises slower — a rotation-ROBUSTNESS effect, not place
  separation. Turn the rotation search OFF and selfAUC collapses to
  0.525→0.606 with pos≈neg throughout: essentially NO place
  discrimination without the self-identity + perm-search crutch.
- "Fixed c5 (0.901) beats the learned head (0.836)" is therefore
  near-tautological: the metric monotonically rewards coarseness, so
  maximal blanking wins by construction. Not evidence about the head.
- CLEAN on audit: anti-oracle (est poses only mask negatives/adjacency,
  never encoded), time-split (eval gate touches only [144,240), _YCACHE
  is train-only), normalization not the driver. Caveat: the fixed-cutoff
  baselines still route through the TRAINED saliency weights (encode_mod
  reads wlog even under force_cut) — entangled, not a pure occupancy
  ladder.

REVISED VERDICT: blanking the fine rings does NOT improve genuine
place recognition on real building-scale lidar — the one honest metric
(adjacent same-place) is flat and slightly regresses under aggressive
blanking. This is CONSISTENT with the project thesis (loop-closure
detection is the irreducible wall — you cannot blank your way to
separability) and with the synthetic honest-negative. The only surviving
physical statement: coarse rings ARE more rotation-robust (fewer phase
wraps) — a real property, but it does not buy place discrimination.
Methodological lesson banked: a self-rotation positive (zero place
variation) is an INVALID place-recognition metric; use adjacent-frame
same-place as the control. `learn_scale_real.py`'s primary gate metric
should be adjAUC; selfrot is a rotation-robustness diagnostic only.
Process lesson: do NOT bank a positive before its rule-4 audit returns.

## 2026-07-15 — P3: unified LIDAR net at the pinned ring-raster geometry (deploy net + budget; honest saturated place gate)

`sspax/lidar_ring.py` — the deploy lidar front-end at the PINNED ingest
geometry (commit 34989b1): ring-range raster, RINGS-AS-CHANNELS
(3×1024 full), 1D-along-azimuth convs, shared trunk + two heads
(tracking: per-bin saliency weight + descriptor @20 Hz; distilled
per-bin label @keyframe — placeholder, school has no lidar semantic
labels). NOT a BEV net (BEV demoted to mechanism-study). Azimuth is
cyclic, so yaw = an exact circular ROLL of the raster (no
re-rasterisation, no permutation search for integer-bin yaw) — the
cleanest rotation model in the project.

BUDGET (the deployable deliverable): 15.9k params, **4.12 MMAC/frame
(82 MMAC/s @20 Hz), 1.9 KB BNN weights** — trivial for the ECP5 lidar
tier, leaves the EBR margin the vision net needs.

HONEST place gate (learned vs uniform saliency, TIME split, rotation =
azimuth-roll searched): the metric is the HONEST adjacent-vs-far
separability (NOT a self-rotation positive — that was retracted as a
rotation-triviality artifact, same-date P1 correction). Result:
  uniform 0.987  learned 0.986  (delta -0.001)
The adjacent-vs-far task is SATURATED at uniform (0.987): adjacent ring
rasters are near-identical, trivially separable from >4 m frames, so
there is no headroom for a learned saliency to show a place gain. This
is NOT evidence the learned head is useless — it is evidence run2 CANNOT
test it: the venue has no honest revisits (banked), so the only
discriminating place pairs (same place, different heading/time) do not
exist. Consistent with P1 and the project thesis (loop-closure detection
is the irreducible wall; the front-end cannot manufacture separability).
Deliverable = the deploy-geometry net + budget + a verified end-to-end
pipeline on real ring rasters; the learned-saliency place value is
UNSCOREABLE on run2 and awaits a revisit-bearing venue (classroom, P2).
Anti-oracle: est poses label pairs only, never rasterised/encoded.

## 2026-07-15 — bounded semantic-map CAPACITY law: cap ~ sqrt(D) — tiling beats one big superposition

`sspax/semantic_capacity.py` — how many features fit in ONE fixed-size SSP
vector before a class query stops resolving them, vs the vector dimension
D (sphere.dirs_ring × oct6 ladder; chair-recall>=0.5; 0-chair scenes
EXCLUDED not scored 0; 24 seeds). Mean chair-recall at rising map load:

  D=120  cap@0.5  7.4 objects
  D=240          12.8
  D=360          16.3
  D=720          23.1
  D=1440         25.7
  FIT: capacity ~ 0.76 * D^0.50   (log-log R^2 0.937)

Capacity scales as **sqrt(D)** — the textbook superposition-retrieval law
(cross-talk noise ~ sqrt(load/D), so recall crosses threshold at
load ~ sqrt(D)). Consequence: to hold N objects at recall>=0.5 a SINGLE
superposition map needs D ~ N^2 — it scales POORLY. The bounded-memory
win therefore comes NOT from one giant vector but from TILING: per-segment
bounded maps (exactly the shipped "per rigid 5-keyframe trajectory
segment" architecture), each holding O(10-20) objects at D=360-720. The
capacity law independently MOTIVATES the segmented-map design. k (bits per
class) has minor effect (k=6-12 best ~16 obj at D=360; k=24 slightly worse
14.8 — denser codes add cross-talk). Anti-oracle: deterministic synthetic,
GT positions score recall only. Pure numpy, fixed seeds.

## 2026-07-15 — P2: classroom fetched + HONEST place gate — reproduces the deploy mv numbers, confirms the venue-scale law on real revisits

Classroom ("spot") session fetched from HF (lorinachey/spot-telluride,
22 pointcloud shards 2.2 GB + kinematic odometry), parsed to scans.npz
(1655 clouds), placed at data/spot_telluride/ (the flat "spot" run).
This is the HONEST venue run2 lacked: withheld kinematic odometry gives
110 keyframes with **179 genuine same-place revisit pairs** (gap>=40,
<0.7 m) — real place recognition is scoreable here.

Honest revisit-AUC (rot-searched, full clouds, ~10 m room):
  ring-oct6      D240   0.904   (adj 0.991)
  azel-oct6      D240   0.947   (adj 0.988)   <- best at D240
  azel-house     D240   0.821   (adj 0.920)
  ring-coarse16  D1920  0.976   (adj 0.993)
  ring-oct6      D1920  0.976   (adj 0.994)   <- best overall (tie)

Verdicts:
- CROSS-BOX REPRODUCTION: matches the deploy agent's 08dc2a4 mv numbers
  (azel-oct6 D240 0.947 honest classroom; ring-vs-azel D240 venue-split
  0.904 vs 0.947) — rule-5 reproduction PASS across boxes/tooling.
- The venue-scale law HOLDS on real revisits: oct6 family (0.90-0.95)
  beats the old house ladder (0.821) at ~10 m — the LADDER is the lever.
- Geometry at D240 is venue-dependent (azel 0.947 > ring 0.904); at
  D1920 they converge (0.976) — consistent with the banked "adopt the
  ladder, not the D240 geometry" verdict.
- Positive, HONEST place result: the bounded SSP encoding separates REAL
  revisits at 0.90-0.98 AUC. This does not contradict the loop-closure
  wall (which is about ALIASED/ambiguous cross-session revisits, not
  clean small-room revisits) — it shows the representation handles clean
  revisits well; the wall bites only where appearance is genuinely
  ambiguous. Anti-oracle: odometry labels pairs only, never encoded.

## 2026-07-15 — P4 + Regime C: unified vision net on REAL NYUv2 — strong TRACKING head, weak deploy-budget SEG head, working queryable map at object level

`sspax/vision/segnet.py` trained on REAL NYUv2 (1449 imgs, resized to the
pinned Y8-luma 160×120, /4 trunk 30×40×64; 20.9k params, 2.5 KB BNN,
22 MMAC/frame). Joint per-pixel seg CE + cell-InfoNCE descriptor.

VISION NET (real, held-out 20%):
- TRACKING descriptor head — STRONG: per-cell retrieval **0.950**, bit
  stability 0.930 under luma jitter. The deploy-budget net produces
  excellent appearance-invariant, cell-discriminative tracking keys on
  real texture.
- SEG head — WEAK at deploy budget. Naive CE collapsed to 3 dominant
  classes (7/40 used, pixacc 0.11). Class-balanced CE (inverse-sqrt-freq,
  void excluded) recovered breadth: **36/40 classes used, pixacc 0.329 on
  non-void (13× the 1/40 chance)** but **mIoU 0.041** — the tiny luma-only
  net gets the GIST, not precise masks. Honest verdict: fine 40-class seg
  from luma at 20k params is infeasible; the per-cell LABEL head needs
  cross-modal DISTILLATION (as the pinned spec intends) or coarser
  classes. Tracking is the head that carries at deploy budget.

REGIME C — the queryable map (`sspax/vision/objmap_nyu.py`): segnet
per-cell class predictions → back-projected to 3D via NYUv2 depth →
bound into the bounded SSP map (`sspax/semantic.py`) → query a class.
The result is a clean demonstration that the CAPACITY LAW dictates the
map design:
- CELL-level binding (~1200 features into D=360, 75× over capacity):
  round-trip AUC **0.576** (near chance — cross-talk destroys it).
- OBJECT-level binding (one feature per predicted class-region centroid,
  **13.1 objects/frame** ≈ the D=360 capacity of ~14): round-trip AUC
  **0.799** (n=523 class-queries), 1.5× median contrast — a WORKING
  queryable map on real predictions.
The 0.576→0.799 jump by respecting the sqrt(D) capacity law is the
capacity finding biting on real data: bind OBJECTS, not cells (and/or
TILE). Anti-oracle: GT labels only SCORE seg; the map is built entirely
from the net's own predictions bound at depth-projected positions.

## 2026-07-15 — segnet quantization robustness: the tracking descriptor survives int4 weights (BNN-regime confirmed on the trained net)

Fake-quant (symmetric per-tensor) of all conv/dense kernels of the trained
NYUv2 segnet, descriptor-head eval (one held-out frame, 1200 cells):
  float32  retrieval 0.730  bit-stability 0.915
  int8     retrieval 0.733  bit-stability 0.916
  int6     retrieval 0.726  bit-stability 0.914
  int4     retrieval 0.720  bit-stability 0.926
Weight precision from fp32 down to int4 costs ~0.01 retrieval and nothing
on stability — the per-cell tracking descriptor (itself sign-binarized to
1 bit, the deploy path) is fundamentally low-precision-friendly. Combined
with the 2.5 KB BNN weight budget, this confirms the "BNN XNOR-popcount
lanes are the regime" deploy claim on the actual trained net, not just the
paper envelope. (Absolute retrieval here 0.73 is single-frame; the
all-frames number is 0.95 — the ablation reads the int4-vs-fp32 DELTA.)

## 2026-07-15 — rule-4 review CORRECTIONS to the capacity law + Regime-C map (one strengthened, one reframed)

Read-only audit (dispatched agent) + my own controls refine the two
2026-07-15 positives:

- **Regime-C queryable map — STRONGER than banked, and confirmed SEMANTIC.**
  The banked object-level round-trip AUC 0.799 was UNDERSTATED: it used
  `semantic.query` which places every query point at a single median z
  (z-flattening). Querying at each object's ACTUAL 3D centroid gives
  round-trip AUC **~1.000** (D=360), and the capacity lever is visible on
  real data: D=120→0.988, D=240→0.998, D>=360→1.000 (13 objects/frame).
  DECISIVE control (mine + the auditor's, agreeing): querying the same map
  with a RANDOM class code collapses to chance (0.49), and a WRONG-class
  code to 0.46 — so the discrimination is carried by the class-code
  BINDING, not by spatial separation of centroids. The queryable map is
  genuinely semantic. Cell-level over-capacity binding stays near chance
  (0.557-0.576) either way — the 0.576->~1.0 gap IS the capacity law.
- **Capacity exponent D^0.50 is CONFOUNDED — reframe, keep the conclusion.**
  D was swept only via n_dirs (D=6*n_dirs), so larger D BOTH lowers
  superposition cross-talk AND sharpens the spatial decode kernel; the
  low-D rows show a spatial-resolution failure (D=120 already <1 recall at
  load 4) independent of bundling. So `0.76*D^0.50` is an empirical,
  resolution-confounded capacity-vs-D trend, NOT a measured
  superposition-physics constant (cap/sqrt(D) is non-flat 0.68→0.86→0.68).
  The load-bearing takeaway is unaffected and conservative: capacity is
  SUBLINEAR in D, so one giant superposition vector scales poorly and the
  bounded-memory win comes from TILING (per-segment maps). Anti-oracle and
  0-chair-exclusion both confirmed sound by the audit.

## 2026-07-15 — classroom learned-vs-fixed PLACE test (the honest venue's definitive wall test): NO learned gain — and an anti-oracle catch

The classroom (withheld odometry, 244 real revisit pairs) finally lets us ask
the question run2 could not: does a LEARNED front-end beat a FIXED encoding for
place recognition on a venue with real revisits? Trained the lidar ring net's
per-bin saliency self-supervised (temporal-adjacency + azimuth-roll positives),
encoded eval frames through `encode_bins` with LEARNED vs UNIFORM bin weights,
gated on honest revisit-vs-far AUC (rotation = azimuth roll).

ANTI-ORACLE CATCH (rule 2, self-caught): a first version used the withheld
ODOMETRY distances to define training far-negatives (d>4 m) and then tested on
odometry revisit labels — GT-contaminated. It showed learned 0.930→0.976
(+0.046). Removing ALL pose labels from training (every off-diagonal batch frame
a negative, positives temporal-only) FLIPS it:

  uniform (fixed)  0.930   (deterministic, no training)
  learned  seed0   0.908   (-0.022)
  learned  seed1   0.882   (-0.048)
  learned  seed2   0.929   (-0.001)

Label-free learned saliency does NOT beat the uniform fixed encoding for revisit
place (−0.001 to −0.048; never a clean win). The +0.046 was ENTIRELY the GT
leak. VERDICT: on the honest venue with real revisits, the learned front-end
provides no place gain over a good fixed encoding — the DIRECT confirmation of
the loop-closure wall / do-no-harm gap that run2 (no revisits) and P1 (selfrot
artifact) could only approach indirectly. The front-end improves FEATURE quality
(tracking descriptors, seg gist — see P4) but cannot manufacture place
separability. Lesson banked: pose-derived training negatives are an anti-oracle
leak when the gate scores on pose — use temporal/label-free negatives, GT for
scoring ONLY.

## 2026-07-15 — tiling REFINEMENT: splitting a fixed D into smaller-D tiles does NOT help (refutes a naive √T intuition)

Constructive follow-up to the capacity trend (`sspax/semantic_capacity.py
tiling`): the naive intuition was "T tiles of D/T each = same total storage
but √(TD) capacity = √T gain." The measurement REFUTES it — at fixed total
D=720, chair recall@load for T=1/2/4 tiles (objects routed by x-position):
  load 16:  T1 0.65   T2 0.53   T4 0.51
  load 32:  T1 0.25   T2 0.12   T4 0.14
More tiles HURT. Reason (the same D-vs-resolution confound the rule-4 audit
flagged for the capacity exponent): a D/T tile has COARSER spatial-decode
resolution, and that loss offsets the reduced per-tile load. So sub-dividing
one fixed storage budget is NOT the lever.

CORRECTED bounded-memory framing: the win is a FULL-D map per bounded-AREA
segment — total storage grows with area (one D-map per segment, each covering
~10-20 objects within its region), never one giant superposition and never
smaller-D sub-tiles. This is exactly the shipped per-5-keyframe-segment
architecture; the capacity study now supports it for the RIGHT reason (each
segment covers few objects at full D), not the wrong one (sub-dividing D).
Supersedes the "bind OBJECTS and TILE" phrasing in the earlier capacity entries
where "tile" could be misread as splitting D.

## 2026-07-15 — the QUERYABLE semantic map obeys the SSP TRANSFORM algebra (unifies the two contributions)

`sspax/semantic_transform.py` — the bounded semantic map (map = Σ_o
roles[bits_o] ⊙ pos(x_o)) is not only queryable-by-class, it is
ALGEBRAICALLY TRANSFORMABLE on frozen content, exactly like the metric
SSP map:
  TRANSLATION t: map ⊙ exp(iW·t) == rebuild@{x+t}   max|err| 1.3e-13 EXACT (any t)
  ROTATION 36° (lattice angle): map[perm_R] == rebuild@{Rx}  2.7e-13 EXACT
  ROTATION 30° (off-lattice):   residual 41.9  approx (ring-sphere snap)
The class query still localises objects in the NEW frame after the
transform (translated 0.18 m, rotated-36° 0.05 m; off-lattice-30° degrades
to 0.37 m but does not break). So a graph correction / loop closure MOVES
the entire queryable map — every object AND its class binding — with ONE
O(D) phase op (translation) or index permutation (on-lattice rotation),
never re-encoding any object. This unifies the project's two contributions
on a single bounded vector: it is simultaneously a QUERYABLE semantic map
and an ALGEBRAICALLY-TRANSFORMABLE (correctable, movable) map. Rotation
inherits the ring-stagger sphere's exact-on-lattice / approximate-off
property (residual shrinks with D or the d/dθ correction — sspax/bench).
Bit-exact identity (not a statistical result); anti-oracle: synthetic,
positions score localisation only.

## 2026-07-15 — significance = committed-bit count, on REAL predictions: mechanism CONFIRMED, quality-signal blocked by the weak seg head

`sspax/vision/objmap_nyu.py sig` — closes the last piece of the original
user directive ("modulate a feature's significance by setting bits") on
REAL data. Each object commits round(confidence*k) of its class's k bits
(segnet softmax confidence), so the bounded-map query READOUT should track
committed bits. On real NYUv2 (816 objects):
- MECHANISM CONFIRMED: Spearman(confidence, query-readout) = **0.607** —
  readout rises with committed bits ∝ confidence (cross-talk/position
  decode add noise, so <1 but clearly positive). The map grades a
  feature's importance by its bit count, exactly as designed.
- QUALITY SIGNAL BLOCKED: high-readout vs low-readout objects are BOTH
  ~0.00 GT-correct — the deploy-budget seg head (mIoU 0.041) is too weak
  and miscalibrated (confidently wrong), so confidence does not track
  correctness and the significance cannot grade quality HERE. The
  mechanism is sound; the seg head is the limiter (same bottleneck as P4
  — usable per-cell labels need distillation/coarser classes, not
  from-scratch deploy-budget learning). Anti-oracle: GT only scores the
  correctness split; the map uses the net's own predictions + confidence.

## 2026-07-15 — seg-head bottleneck characterized: class-count helps but does not fix deploy-budget segmentation

Addendum to P4 (the recurring seg-head limiter). Retrained the same
deploy-budget vision net with FEWER classes to test whether class
imbalance alone caps the seg head:
  40 classes: mIoU 0.041, pixacc(non-void) 0.329
  10 classes: mIoU 0.100, pixacc(non-void) 0.429   (tracking unchanged 0.948)
Fewer classes help (mIoU 2.4x, pixacc +0.10) but the seg head stays weak
— usable per-cell segmentation is NOT reachable at 20k params / Y8-luma /
half-res even at 10 classes. The bottleneck is fundamental (capacity +
mono input + resolution), not merely imbalance. Confirms the P4 verdict:
the deploy per-cell LABEL head needs cross-modal DISTILLATION (a teacher
provides labels, as the pinned spec intends) or a larger net; the
TRACKING head (retrieval 0.95, int4-robust) is the deploy-carrying head.
This closes the seg-head thread — the fix is distillation, out of scope
for a from-scratch deploy-budget net.

## 2026-07-15 — the tracking descriptor is a translation-equivariant CONTENT signature (works for the right reason)

Probe of the deploy-carrying head: it was trained on PHOTOMETRIC jitter
only (luma brightness/contrast/gamma), so does it survive the GEOMETRIC
shifts a real tracking key must? Per-cell retrieval (band-limited
candidate set) on real NYUv2:
  trained photometric jitter (shift 0):     0.840
  untrained geometric 1-cell shift:         0.926
  untrained geometric 2-cell shift:         0.929
The descriptor is MORE robust to (untrained) geometric shifts than to
(trained) photometric jitter — because it is a translation-EQUIVARIANT
content signature: when a feature moves, its descriptor moves with it
(CNN translation equivariance), so cell(i,j) in the shifted image matches
cell(i,j+shift) in the original. That is exactly the property a tracking
key needs (follow the content, not the pixel). Confirms the tracking head
is a genuine tracking descriptor, robust to both appearance jitter and
viewpoint shift — not an artifact of the training augmentation. With the
int4 weight-robustness and 2.5 KB BNN budget, this is the strong,
deploy-ready head of the unified vision net (the seg head remains the
distillation-blocked one).

## 2026-07-15 — queryable-map tolerance to loop-correction ERROR: graceful to ~0.5 m (sub-metre bound)

`sspax/semantic_transform.py robustness` — the transform algebra is exact,
but a real graph correction is APPROXIMATE. Translating the map by the true
t plus an error e, then querying at the true shifted positions, chair recall
vs |e|:
  err 0.0/0.1/0.25 m : 1.00 / 1.00 / 0.94
  err 0.5 m          : 1.00
  err 1.0 m          : 0.44
  err 2.0 m          : 0.00
The queryable map degrades GRACEFULLY: recall holds to ~0.5 m correction
error, decays past ~1 m, gone by 2 m. The tolerance (~sub-metre) is set by
the query's spatial decode/match resolution (~0.8 m match window), NOT the
finest ladder wavelength (lambda_min 0.25 m) — the map is MORE forgiving
than lambda_min. Deploy reading: a loop correction accurate to ~0.5 m keeps
the semantic map queryable — a comfortably achievable bound for real
closures. So the queryable+transformable map is robust to realistic
graph-correction residuals, not just the exact algebra. Anti-oracle:
synthetic, positions score recall only.

## 2026-07-15 — "one bounded vector, DUAL USE" tested directly: place descriptor + queryable semantic map coexist in one D-vector

`sspax/dual_use.py` — the strong form of the core contribution: can ONE
bounded D=360 vector be BOTH a metric place descriptor (revisit matching)
AND a queryable semantic map, or do the roles destroy each other?
  combined = alpha*place_vec + (1-alpha)*sem_vec   (16 scenes, shared W_POS)
  alpha:      1.0   0.9   0.7   0.5   0.3   0.1   0.0
  place-AUC:  1.00  1.00  1.00  1.00  1.00  1.00  1.00
  chair-recall: 0.00 0.00 0.77 0.86 0.86 0.86 0.86
FEASIBLE: at alpha=0.5 the SAME vector gives place-AUC 1.00 AND
chair-recall 0.86 — both roles usable at once. Interference is mild and
ASYMMETRIC: place matching is robust across all alpha (bundling semantic
content never hurts it), while the semantic query needs enough weight
(alpha <= ~0.7; at alpha>=0.9 the place term swamps the bindings ->
recall 0). HONEST CAVEATS: (1) place-AUC is 1.0 even at alpha=0 because
distinct scenes have distinct OBJECTS, so semantic content also
discriminates places (realistic — a revisit sees the same geometry AND
objects — but it means this easy, non-aliased place task does not isolate
geometric matching or stress the interference); the aliased-place wall
regime is a separate, harder question. (2) The clean alternative is
separate/concatenated vectors (2x storage, zero interference); the
shared-vector result shows the SINGLE-vector form is feasible at a small
capacity cost. Anti-oracle: synthetic, GT scores only.

## 2026-07-15 — rule-4 CORRECTIONS to two recent semantic-map positives (transform algebra stands; robustness + dual-use framing overstated)

Read-only audit of the three most recent positives. RESULT A (the transform
algebra: translation = phase mult bit-exact, rotation = permutation exact
on-lattice, query localizes class-specifically with random-code control →0)
is SOUND — keep as banked. Two framing corrections:

- CORRECTION to "queryable-map tolerates ~0.5 m correction error / graceful
  degradation": the audit is right that this OVERSELLS. Translation is
  LOSSLESS — the SSP peak is exact at ANY t (Result A), so NOTHING in the
  representation degrades. The recall-vs-error curve merely measures the
  QUERY's spatial match window (`semantic._match` tol=0.8 m): translating by
  t+e puts the peak exactly at x+t+e, and recall = fraction with |e|<0.8 m.
  Honest restatement: a loop correction must land within the query's ~0.8 m
  match window; the representation itself adds no error at any correction
  magnitude. The number is a harness property, not representational
  robustness. (The non-monotone 0.25→0.5 dip is small-count noise around
  the tol, confirming this.)

- CORRECTION to "one vector = place + semantic, both coexist at alpha=0.5":
  the place-AUC=1.0 column at alpha<1 is a CONTAMINATED metric — the revisit
  vector reuses the SAME sem_vec verbatim (not re-encoded from the jittered
  revisit), so at alpha<1 the match is an identical-vector match, not
  geometric place matching (control: alpha=0 diag sim ==1, off-diag mean
  0.072). So the sweep NEVER stresses the place side, and the claim "place
  survives semantic interference" is NOT supported — RETRACTED. What HOLDS:
  (i) pure place at alpha=1 gives AUC 1.0 on the real jittered revisit (the
  geometric place role works on its own); (ii) the SEMANTIC query survives
  bundling with a place background down to alpha~0.5 (recall 0.86, random-code
  control → 0.00 confirms it is class-specific, not spatial). So the honest
  dual-use statement is one-directional: the semantic map survives an added
  place term to alpha~0.5; whether place survives an added semantic term is
  UNTESTED here (the place task was saturated). Anti-oracle held throughout
  (GT scores only); the fixes are framing, not code.

## 2026-07-15 — regression gate PASS after the session's additions (non-breaking, verified)

Pre-push verification that this session's work (7 new sspax modules —
ladder_rule, learn_scale_real, lidar_ring, semantic_capacity,
semantic_transform, dual_use, vision/segnet, vision/objmap_nyu — all
IMPORT-ONLY on the frozen core, rule 1; + docs) did not regress anything:
- `tests/test_smoke.py`: SMOKE PASS — 49 modules import, bit-exact selftest OK.
- spot flagship acceptance (lidar-only vs withheld odometry, TARGET PLATFORM):
  ATE **0.039** med 0.033 — reproduces the banked 0.039 EXACTLY, and confirms
  the classroom/"spot" data fetched this session (HF, parsed to scans.npz)
  drives the shipped pipeline correctly.
The frozen core is untouched; every new artifact subclasses or imports only.

## 2026-07-15 — lattice choice for the QUERYABLE map: geometry-INSENSITIVE (unlike place) — one lattice serves both roles

`sspax/semantic_lattice.py` — connects the place-lattice thread to the
semantic-map thread. All lattices at matched D=240, chair query, 24 scenes,
8-object load; clean metrics recall + localisation precision:
  ring-oct6   recall 0.95  precision 0.08 m
  azel-oct6   recall 0.93  precision 0.09 m   (the place-ADOPTED lattice)
  azel-house  recall 0.87  precision 0.07 m
  fib3d-oct6  recall 0.93  precision 0.09 m
  ringstag    recall 0.96  precision 0.09 m
Query quality is nearly IDENTICAL across geometries (recall 0.87-0.96,
precision ~0.08 m) — the semantic query is geometry-INSENSITIVE. This
contrasts with PLACE, which IS geometry-sensitive (azel-oct6 0.947 > ring
0.904 at D240, honest classroom). Mechanism: the semantic query is a LOCAL
position decode (one object at a known-ish spot), which any reasonable
lattice resolves; PLACE is a GLOBAL appearance comparison where the
direction distribution matters. Deploy implication: the place-adopted
azel-oct6 serves the semantic query well too (0.93 / 0.09 m) — ONE lattice
for BOTH the metric-place and semantic-query roles, no place/semantic
tension. (Anti-oracle: synthetic, GT scores only. A contrast metric was
dropped as numerically unstable — off-object readout ≈ 0 blows up the
ratio; recall+precision are the clean measures.)

## 2026-07-15 — the semantic-map LADDER is the lever (and has a place/semantic TENSION)

`sspax/semantic_lattice.py run_ladder` — companion to the geometry result.
Fixed ring-oct6 geometry (60 dirs), 6-rung ladders of different wavelength
RANGE (D=360 fixed), chair query, ~12 m room:
  fine   0.1-1.6 m : recall 0.98  precision 0.09 m   <- best (fine object scale)
  oct6   0.25-8 m  : recall 0.91  precision 0.07 m
  wide   0.25-90 m : recall 0.59  precision 0.07 m   <- diluted (rungs too sparse)
  coarse 2-90 m    : recall 0.24  precision 0.11 m   <- worst (no fine resolution)
The LADDER matters enormously (recall 0.24-0.98) while GEOMETRY did not
(0.87-0.96, prior entry) — geometry-insensitive + ladder-sensitive, the
SAME shape of law as place. BUT a genuine place/semantic TENSION emerges in
the ladder: the semantic QUERY wants FINE rungs (sub-metre, object-
localisation scale — fine 0.98), whereas PLACE wants COARSE-reaching rungs
(venue scale — banked coarse16-wins-buildings). A shared 6-rung WIDE ladder
that tries to span both dilutes each (query 0.59). Resolution options: more
rungs (larger D to span object+venue scales), or per-role ladders, or accept
the tradeoff. Corrects the earlier optimistic "one lattice, no tension"
reading: geometry is shared freely, but the LADDER is the contested knob.
Anti-oracle: synthetic, GT scores only.

## 2026-07-15 — the place/semantic ladder tension is FUNDAMENTAL (more rungs do NOT resolve it) — dual-use needs per-role ladders

`sspax/semantic_lattice.py run_ladder_resolve` — tested my proposed
resolution (span object-fine + venue-coarse in one many-rung ladder).
It FAILS. Chair query recall, ring-oct6 geometry, 12 m room:
  6-rung  wide 0.25-90 (D360): 0.59
  12-rung wide 0.1-90  (D720): 0.63   <- barely better; NOT recovered
  12-rung fine 0.1-8   (D720): 0.95   <- only fine ladders work
Adding rungs does not help because the COARSE rungs place needs actively
DILUTE the semantic query: a 90 m wavelength is near-constant across a 12 m
room, so it contributes a DC-like phase that adds cross-object cross-talk
without localising. So the place/semantic ladder tension (place wants
coarse-reach, query wants fine) is FUNDAMENTAL, not a resolution-budget
issue — a shared WIDE ladder compromises the query regardless of D. Deploy
consequence for a dual-use (place + semantic) bounded map: use SEPARATE
ladders per role (a fine object-scale ladder for the semantic bindings + a
coarse venue-scale ladder for the place descriptor, e.g. bundled as two
sub-bands / two vectors), OR accept a compromised role. Honest negative on
my own proposed fix; the geometry-shared / ladder-contested split (prior
two entries) stands, now with the ladder tension shown irreducible.

## 2026-07-15 — significance grading BUYS effective capacity for important objects (connects significance + capacity)

`sspax/semantic_capacity.py significance_capacity` — the two banked
mechanisms combine: capacity is bounded (~sqrt-ish D), but SIGNIFICANCE
(committed bit count) lets you ALLOCATE it. 3 important chairs commit all
k=12 bits; background objects commit b bits; chair recall vs background load
(D=360):
  bg-objects   graded b=3   equal b=12
     4           0.72         0.61
     8           0.69         0.56
     16          0.64         0.25
     32          0.47         0.14
     48          0.36         0.04
Graded significance beats equal-weight at EVERY load and the gap WIDENS
with it — at 48 background objects, 0.36 vs 0.04 (9x). De-prioritised
(low-bit) background objects contribute less to the bounded bundle -> less
cross-talk on the important-object query, so what matters survives far more
load. So significance is not just a readout-strength knob (banked earlier)
— it is an effective CAPACITY lever: the bounded map spends its finite
capacity on what matters. Refines the "capacity is limited -> tile"
conclusion: within a tile, grade features by importance to fit more of what
counts. Clean (chair-recall metric, same scenes, only b varies, large
monotone effect). Anti-oracle: synthetic, GT scores only.

## 2026-07-15 — rule-4 audit of the semantic-map lattice/ladder/significance ablations: all 4 SOUND (mechanisms confirmed), one minor metric caveat

Batched read-only audit of the four recent ablations (geometry-insensitive,
ladder-is-lever, tension-fundamental, significance-buys-capacity). All
reproduced bit-closely; anti-oracle clean throughout. Verdicts:
- SIGNIFICANCE BUYS CAPACITY — SOUND, mechanism NAILED. The chair signal is
  INVARIANT to b (~12-13 both graded and equal); the entire effect is the
  NOISE FLOOR scaling with committed bits (SNR = sig/noise tracks recall).
  A disjoint-bit control (background bits forced to never collide with a
  chair query bit) keeps the graded advantage nearly intact (0.44 vs 0.21
  at 48 bg) — so it is generic superposition cross-talk reduction, not mere
  bit-collision. Precise: de-prioritised few-bit objects inject
  proportionally less cross-talk; the fixed-height important signal survives
  more load. Confirmed real.
- LADDER IS THE LEVER + TENSION FUNDAMENTAL — SOUND, confirmed quantitatively
  by the pre-detection decode kernel: FWHM tracks recall (fine 0.15 m →0.98,
  coarse 5.0 m →0.24), and the far-field DC pedestal (set by the PRESENCE of
  coarse rungs, not their count) explains why more rungs don't resolve it
  (12-rung wide pedestal 0.34 ≈ 6-rung wide 0.36; fine drops to 0.086). Not
  detect() artifacts; D genuinely matched.
- GEOMETRY-INSENSITIVE — SOUND across the load curve (all geometries collapse
  TOGETHER at load 16 → ~0.4, load 32 → ~0.1; not a saturation ceiling).
  CAVEAT (adopted): the PRECISION column is non-discriminating — all
  geometries report ~grid-floor 0.07-0.09 m (grid step 0.2 m), so precision
  cannot separate lattices; RECALL is the real evidence (and it also shows
  insensitivity). The load-8 spread (0.87-0.96) scrambles with load = seed
  noise, not a geometry ranking. Conclusion (geometry free, ladder contested)
  stands; read it off recall, not precision.
No retractions — the careful-design pass produced defensible results; the
only fix is to lean on recall over the grid-floored precision metric.

## 2026-07-15 — the bounded SEMANTIC MAP survives low-precision (FPGA polar-quant): ~4 bits/cell keeps queries alive

`sspax/semantic_quant.py` — tests the deploy-substrate claim for the SEMANTIC
map (the vision descriptor net already quantizes to int4; this is the MAP's
stored phasors, the bounded EBR content) under the FPGA polar-quant model
(`core.q_polar_np`: nph phase bins x nmag magnitude levels). Chair-query
recall (D=360, float baseline 1.00):
  nph  nmag  map-only  map+roles  bits/cell
  64    8     1.00      1.00        9
  32    8     0.97      0.97        8
  16    4     1.00      0.97        6
   8    4     0.97      0.97        5
   8    2     0.97      0.97        4
   4    2     0.93      0.93        3
The map survives aggressive quantization: at 4 bits/cell (nph=8 phase x
nmag=2 mag) recall is 0.97, and 3 bits/cell still holds 0.93 — quantizing
the ROLES too barely adds cost (map+roles ~ map-only). Footprint: a D=360
map at 4 bits/cell is ~180 bytes — trivially EBR-resident. So the queryable
map fits the low-precision (BNN-regime) FPGA substrate the thesis claims, at
a handful of bits per D-cell, with negligible query loss. Standard VSA
phase-noise robustness (well-understood mechanism); clean float-baseline +
sweep + roles-too control. Anti-oracle: synthetic, GT scores only.

## 2026-07-15 — the bounded semantic map is DYNAMICALLY UPDATABLE (add/remove churn leaves no drift)

`sspax/semantic_dynamic.py` — real scenes change (objects move/appear/
disappear). The VSA binding is additive, so an object is ADDED by bundling
its bound feature and REMOVED by SUBTRACTING it, both O(D), no re-encode.
Churn test (D=360, 8 live objects, 40 add/remove steps, 16 seeds):
  (c) DRIFT: max|churned - rebuilt-from-live-set| = 2.1e-14 — BIT-EXACT, no
      residue accumulates over arbitrary edit histories.
  (a) LIVE chair recall after churn: 0.86 = the STATIC recall at 8 objects —
      query quality depends only on the CURRENT live set, not edit count.
  (b) REMOVED chair readout: -1.04 below background — cleanly gone (removal
      subtracts the binding exactly, no ghost).
So the bounded map is a DYNAMIC map: add=bundle, remove=subtract, both O(D)
phase ops on frozen content, and any edit history leaves the map identical
to a fresh build of the live set (linear additive binding). Moving and
disappearing objects (dynamic environments, session-to-session edits) are
handled with no re-encode and no drift — completing the map's capability
set alongside query, transform, capacity, significance, and quantization.
Bit-exact identity + clean query metrics; anti-oracle: synthetic, GT scores
only.

## 2026-07-15 — deploy-side rule-4 review of the 3bfee0b/ca33364 round: everything stands; two audit closures + a registry caveat + repo nits

Read the full 27-entry round + all 13 new modules on the deploy box
(pull clean, SMOKE PASS, spot acceptance untouched). The round's four
self-retractions (P1 selfrot, capacity exponent, transform-robustness
framing, dual-use place side) are exactly rule 4 working; nothing to
re-open. Closures for the two positives that had NOT had a second
opinion:

- `semantic_quant` — SOUND, and deploy-AUTHORITATIVE: `core.q_polar_np`
  is line-for-line identical to `sspslam.quantized.q_polar` (verified
  textually here; `sspax.bench parity` covers JAX-vs-np at 16x4), so the
  4-bits/cell recall-0.97 table runs on the real FPGA arithmetic model.
  Map quantized once on the final bundle (= EBR storage semantics);
  query at float is fine for the storage claim (query precision is a
  separate, compute-side question).
- `semantic_dynamic` — mechanism SOUND (linear additive binding makes
  churn==rebuild trivially exact), but the deploy-honest form needs one
  addition the entry leaves implicit: EXACT removal subtracts the
  ORIGINAL binding, which requires a per-segment OBJECT REGISTRY
  (class bits + stored position per object; ~13 obj x ~8 B ~ 100 B/seg
  — bounded, derived state, not sensor history). Registry-free removal
  by re-observation subtracts an ESTIMATED binding: fine rungs ghost
  (phase error 2*pi*eps/lam — at lam_min 0.25 m an eps of 0.1 m is
  ~2.5 rad on the finest ring, no cancellation). VERDICT: dynamic map =
  D-vector + tiny registry; bank the pair as the deployable unit.

Nits (none change a verdict): (a) the classroom learned-vs-fixed wall
test — the DEFINITIVE direct result of the round — lives only in scratch
on the training box; it deserves a committed module (and the revisit-pair
count differs between entries: 179 in P2 vs 244 in the wall test — state
the threshold that changed). (b) `segnet.forward_np` wraps jnp calls —
not actually numpy-only; the deploy path is the headio v2 forward (below).
(c) `lidar_ring.ring_raster` docstring promises range+intensity channels;
the code builds 3 range channels only (the nets and budgets are built on
3 — fix the docstring, not the code). (d) The semantic suite imports JAX
unconditionally via sspax.semantic/core, so cross-box reproduction on the
deploy box is blocked — acceptable for synthetic mechanism studies, but a
golden-vector npz (map, roles, quant grid, expected recalls) would make
the FPGA-relevant numbers verifiable here.

## 2026-07-15 — headio contract v2: reshaped to the FIRST TRAINED nets; a silent lidar-scaling bug caught before it burned an artifact; random baselines for BOTH gates

Auditing the trained nets against the v1 contract found three mismatches
that would have failed or (worse) silently corrupted their export:
(1) v1 asserted track cout==2 — the trained vision net has separate w/cut
1x1 convs (mergeable exactly) and the trained LIDAR net has NO cut head;
(2) the 32-bit tracking DESCRIPTOR — the head that actually carries
(NYUv2 retrieval 0.950, int4-robust, translation-equivariant) — had no
slot in v1 at all; (3) v1 forward divided ALL inputs by 255 — correct for
Y8, WRONG for lidar ring rasters trained on raw METERS (a trained lidar
head would have seen 60 m -> 0.235 at deploy and produced garbage with no
error raised). `sspax/headio.py` is now contract v2 (v1 files still
load): optional `desc` stack (prefix D, bits = act > 0), track cout 1|2
(cutoff channel retired per the P1 retraction), explicit `in_div`
(vision 255, lidar 1), conv1d-aware budget accounting, and fixtures that
MIRROR the trained nets exactly. Cross-implementation check: my numpy MAC
count on the mirrors reproduces their JAX-side budget prints exactly —
vision-half trunk+track+desc 22.3 MMAC (their "22 MMAC"), lidar 3.47 +
0.655 for their 40-class lab head = 4.12 ("4.12 MMAC/frame"). Selftest:
v1 back-compat + both v2 mirrors, deterministic, roundtrip exact.
New RANDOM-weights baselines on TUM fr3 (the calibration a trained
artifact reads against): desc-bit stability 0.954 adj / 0.906 far
(gap 0.048 — CReLU random convs are even smoother than the seg-bit
0.892/0.828 baseline; the GAP is the only signal); desc-key objmap2 gate
stage-2 AUC 0.522/0.548 (chance), err med 1.08-1.24 m (vs banked census
0.165 m), stage-1 16/16 (head-independent shortlist, as designed).
Fixtures: scratch/head_fixture{,_half,_lidar}.npz.

## 2026-07-15 — deploy-side integration verdict of the round (ECP5 grounding): the place path is CLOSED-FORM, the semantic map is a two-band overlay, and semantic accuracy inherits the metric wall

What this round settles for the target platform:

1. **The @20 Hz place/matcher tier ships with NO net.** Three
   independent results converge — the P1 selfrot retraction (blanking
   buys no genuine place separation), the run2 saturated gate (uniform
   0.987 == learned), and the classroom DIRECT test (label-free learned
   saliency -0.001..-0.048 vs fixed across seeds, the +0.046 "win" being
   a GT leak). The place layer is therefore a fixed encoder + the static
   `ladder_of_extent` preset (P5 densified fit: constant unidentifiable,
   only the reach constraint binds — comment updated in
   lattice_presets.py; EXTENT_C stays 1.0). msg P6 (freeze learned
   cutoffs to a LUT) is DEAD — there is nothing to freeze. Consequence:
   the FPGA CNN block belongs entirely to the keyframe/feature tier; no
   learned parameters exist anywhere in the frame-rate place path. The
   EBR pressure from "both nets resident" (54.5/55 KB edge-exact)
   relaxes: only the vision trunk+track runs at frame rate; the lidar
   net's remaining roles (distilled labels; descriptor keys) are
   keyframe-rate.
2. **The bounded semantic map is deploy-ready as a REPRESENTATION; its
   data source is the one blocker.** Per-segment layout, grounded in
   their measured laws: a SECOND fine-ladder band per segment (per-role
   ladders — the place/semantic ladder tension is fundamental; geometry
   is shared, so the SAME rotation-permutation network and phase-multiply
   datapath serve both bands with different W ROM constants), D=360 at
   4 bits/cell = ~184 B/segment incl. the f32 scale (noise next to the
   metric segment vector), capacity ~13 objects/segment (= the measured
   per-frame object count; matches the shipped 5-keyframe segment scope),
   + the ~100 B object registry (above) for exact dynamic edits. Bundle =
   the existing accumulator; corrections = the existing O(D) phase ops;
   class query = host-side decode. CAVEAT on their fine-ladder recipe:
   the winning 0.1-1.6 m ladder starts BELOW the lidar coherence floor
   (COH_LAM_MIN 0.25 ~ 2*pi*sig_r) — synthetic-clean only; the deploy
   fine band is geomspace(0.25, ~2, 6) pending a 1-line re-run of their
   ablation at lam_min 0.25 (msg round 3). The blocker: per-cell LABELS
   at deploy budget (from-scratch seg mIoU 0.041-0.10) — the distillation
   round. The DESCRIPTOR bits need no labels and can bind NOW as
   appearance keys (headio gate key=desc, random baseline banked above).
3. **Semantic queries inherit the metric map's global accuracy — the
   wall propagates, quantified.** The query match window is ~0.8 m
   (their audit's honest reframing: the representation adds no error;
   the correction/pose must land within the window). In-map queries are
   self-consistent (same frame as the metric map). GLOBAL-frame answers
   inherit segment pose error, and the acceptance medians straddle the
   window: spot 0.029 / stata ~0.15-0.21 / fhw 0.39 INSIDE; fr101 1.38 /
   belg 1.86 OUTSIDE. "Highlight the chairs" is trustworthy exactly
   where the metric map is good. And semantics does NOT cross the
   loop-closure wall: the bits are derived from the SAME sensors with
   the same aliasing — an appearance-derived semantic cue is not the
   independent absolute cue FINDINGS §5-6 requires (their dual_use
   place-side is explicitly untested). No one should re-litigate the
   wall via the semantic layer.
4. **The BNN-lane deploy claim is now TRAINED-net-verified end to end:**
   int4 weights cost ~0.01 retrieval on the trained descriptor; the
   semantic map holds 0.97 recall at 4 bits/cell on the REAL FPGA
   arithmetic (q_polar parity verified); budgets cross-check across two
   implementations. The envelope numbers were not paper-only.

## 2026-07-15 — the seg-accuracy BAR for a useful queryable map: <~30% error (quantifies why distillation is REQUIRED)

`sspax/semantic_segnoise.py` — bridges the P4 seg-head bottleneck to map
usefulness. The objmap round-trip only measures SELF-consistency (query
returns what the net predicted, right or wrong). This asks the deploy
question: with seg error rate p (each object mislabeled w.p. p), can a query
for the TRUE class still find the TRUE objects? D=360, 8 objects:
  p(mislabel)  recall  precision
    0.0         0.82     0.63     (ceiling = capacity/detect at 8 obj)
    0.1         0.69     0.60
    0.2         0.55     0.56
    0.3         0.55     0.53
    0.5         0.43     0.39
    0.7         0.15     0.13
Recall/precision degrade ~linearly and cross ~0.5 at p≈0.3 — so the seg head
must hold error BELOW ~30% (object accuracy above ~70%) for a USEFUL
queryable map. Connecting to P4: the deploy-budget luma head is at ~67%
error (pixacc 0.33) — FAR above the bar, so its map would be useless for
finding the RIGHT objects (recall ~0.15). This QUANTIFIES the P4 verdict:
distillation (or a larger net / RGB-D) is REQUIRED, not optional — the
deploy label head or its teacher must reach ~70% per-cell accuracy to cross
into a useful queryable map. Anti-oracle: TRUE classes score only; the map
is built from the corrupted labels (standing in for noisy predictions).

## 2026-07-15 — the seg bottleneck is NOT capacity (8x params barely helps); color lever untested (RGB path numerically unstable)

Follow-up to the seg-accuracy bar (~70% needed) + P4 bottleneck: is the weak
deploy seg head capacity-limited, or input/task-limited? Trained the SAME net
at 8x capacity (NYUv2, 10 classes, balanced CE):
  luma ch16 (19k params):  seg mIoU 0.100  pixacc(non-void) 0.429
  luma ch48 (152k params): seg mIoU 0.105  pixacc(non-void) 0.450
8x more parameters buys +0.02 pixacc — CAPACITY IS NOT THE BOTTLENECK. So
distillation from a bigger SAME-INPUT (luma) teacher would NOT help: the
teacher cannot cross the ~70% bar from luma either. The limit is the INPUT
(Y8 luma, half-res 160x120) or the intrinsic difficulty of 40/10-class
indoor seg, not model size. (Tracking head unaffected: retrieval 0.95,
stability 0.93 at both sizes.) The COLOR lever (does RGB input cross the
bar?) could NOT be measured here: the rgb=True path diverges to NaN within
~100 steps (3x input energy overflows the un-normalized CReLU stack; fixing
it needs input standardization, which would change the luma net's input
scale and invalidate the banked luma numbers — deferred). Honest open
question: whether RGB(-D) input crosses the seg-accuracy bar is the next
lever to test (with input normalization), not capacity. Consequence: a
useful queryable-map label head needs a DIFFERENT INPUT (color/depth) or a
cross-modal teacher with that input — a bigger luma net is not the answer.

## 2026-07-15 — COLOR is a genuine seg lever (stable 1-NN probe, sidesteps the RGB-training NaN)

Resolves the deferred RGB question without the unstable deep training
(`scratch/scratch_color_probe.py`): a nonparametric 1-NN on RAW /4-cell mean
features, RGB (3) vs luma (1), NYUv2 10-class:
  majority-class chance : 0.532
  luma (1-ch) 1-NN      : 0.347
  RGB  (3-ch) 1-NN      : 0.455
RGB beats luma by +0.108 (~31% relative) — COLOUR carries class information
the luma seg head structurally cannot see, so RGB(-D) input IS a real seg
lever (confirming last entry's "input, not capacity"). HONEST CAVEAT: both
sit BELOW the majority-class baseline (0.53) because raw mean-cell-colour
1-NN ignores texture/context — so colour is necessary but not sufficient
ALONE; the fix is an RGB-input CNN (colour + the net's spatial/texture
features), not colour by itself. Net design consequence: a useful queryable-
map label head needs a COLOUR(/depth) CNN (or a cross-modal teacher with
that input) — a bigger luma net cannot cross the ~70% seg-accuracy bar. This
closes the seg-bottleneck characterization: bar quantified (<30% error),
capacity ruled out, and the input lever (colour) confirmed via a stable
probe. Anti-oracle: GT labels score the 1-NN only.

## 2026-07-15 — rule-4 CORRECTIONS to the recent seg-bottleneck + dynamic results (color SOUND; two softenings; distillation-conclusion survives an object-level re-measure)

Batched audit of 4 recent positives. Verdicts + fixes:

- COLOUR LEVER (color probe) — SOUND, decisive. Dimensionality artifact
  RULED OUT by the auditor's controls: luma×3 REDUNDANT = 0.347 (= luma),
  luma+2 NOISE channels = 0.346, RGB = 0.455; chroma-only (brightness
  removed) = 0.362. Only genuine chroma content lifts the score — the +0.11
  is colour information, not extra dimensions. The "seg fix needs colour"
  conclusion is solid.
- CAPACITY-NOT-BOTTLENECK — SOUND. ch48 reaches LOWER training loss than
  ch16 but the same eval pixacc (0.429->0.450) = fits better, doesn't
  generalise better = capacity genuinely not the lever, NOT an underfit.
  Caveat: short/noisy 10-class training, +0.02 delta -> directional, not a
  tight bound.
- DYNAMIC MAP — mechanism SOUND, headline over-read. The drift 2.1e-14 is
  real but TAUTOLOGICAL (add-A-then-subtract-A cancels the bit-identical
  bind() vector by linearity — a construction, not an empirical robustness
  finding; the interesting APPROXIMATE-removal case never arises). And the
  banked "removed chair reads -1.04 BELOW background" is a SMALL-N NOISE
  artifact (9 seeds/11 samples, SE~0.8, ~1.3sigma from 0; flips to +0.62 in
  a variant). CORRECTED statement: a live chair reads +13.5, a removed
  location collapses into the background band (+-1-2) -> cleanly gone, no
  ghost (the entry's own "(~0 => cleanly gone)" parenthetical was the right
  reading; the -1.04 headline over-read noise).
- SEG-ACCURACY BAR — mechanism sound, threshold SOFTENED, conclusion
  re-grounded. (i) The corruption model (label flips to a uniform-random
  other class) is a first-order proxy (real seg error is structured +
  spatially correlated). (ii) The bar crosses 0.5 nearer p~0.35 than the
  stated 0.3, and is an operating-point number (capacity/detect ceiling 0.82
  at p=0, n_obj=8/D=360), not a hard threshold. (iii) The auditor flagged a
  UNIT MISMATCH: 67% PIXEL error vs a 30% OBJECT-level bar. RE-MEASURED the
  deploy head at OBJECT level (predicted-class region vs GT-majority): 0.232
  object-acc vs 0.324 pixel-acc — object-level is actually LOWER, not higher.
  So the head is far below the ~70% bar at BOTH levels; the auditor's
  hypothesised escape (object-level near the bar) is empirically REFUTED, and
  the "a bigger LUMA net can't cross the bar -> need colour(/depth) or a
  cross-modal teacher" conclusion SURVIVES the correct-units check. Softened:
  "distillation" is the likely fix (colour-input CNN), phrased as needed for
  a USEFUL label map, not a proven hard mandate. No anti-oracle issues in any
  of the four.

## 2026-07-15 — DEPTH adds beyond colour for seg: the label head wants RGB-D (a cross-modal argument)

Extends the validated colour probe with the depth channel (stable z-scored
1-NN, `scratch/scratch_depth_probe.py`), to fix the distillation-teacher
INPUT design. NYUv2 10-class, /4-cell means:
  majority chance : 0.532
  depth only (1)  : 0.350   (~ luma-only 0.347 — depth ~ as informative as brightness)
  RGB   (3)       : 0.454
  RGB+D (4)       : 0.521   (+0.067 over RGB; nearly reaches majority baseline)
DEPTH adds a genuine, substantial lever on TOP of colour (RGB 0.454 ->
RGB+D 0.521). By the colour-probe's validated dimensionality control
(uninformative extra dims add nothing), the 4th channel helps only because
depth carries independent class information — geometry separates furniture
from walls/floor where colour alone cannot. So the deploy seg/label head (or
its distillation teacher) should be RGB-D, not RGB alone. This is a
CROSS-MODAL argument: the pinned geometry assigns depth to the lidar net,
but SEG benefits from FUSING vision-colour with lidar-depth — the teacher
that crosses the ~70% seg bar is an RGB-D net (colour + projected
lidar-depth). Closes the seg-input question: capacity ruled out, colour
confirmed, depth confirmed additive -> teacher input = RGB-D. Anti-oracle:
GT labels score the 1-NN only; same stable-probe methodology the auditor
validated for colour.

## 2026-07-15 — RGB-D CNN seg (NaN fixed via input standardisation): DEPTH is the CNN lever, COLOUR is NOT — corrects the 1-NN colour claim

The 1-NN probes said colour (and depth) are informative; the deep-net test
(the RGB-NaN blocker, now fixed by per-channel input standardisation;
`scratch/scratch_rgbd_seg.py`, seg-only, ch32, 10-class, standardised input)
shows what a CNN can actually USE:
  luma  (1ch, 69k): pixacc 0.480  mIoU 0.120
  RGB   (3ch, 70k): pixacc 0.464  mIoU 0.122   <- ~= luma, colour barely helps
  RGB-D (4ch, 70k): pixacc 0.597  mIoU 0.163   <- +0.12 over luma, DEPTH is the lever
CORRECTION to the 1-NN "colour is a genuine seg lever": for a CNN colour
does NOT translate to an advantage (RGB 0.46 ~= luma 0.48) — the CNN already
extracts colour-equivalent structure from luma texture/context, which the
raw mean-colour 1-NN could not. The real lever is DEPTH (+0.12 pixacc):
geometry gives class information appearance (luma OR colour) cannot. So the
1-NN colour result was a probe artifact for CNN purposes; the deep-net test
is the deployment-faithful one and it says DEPTH, not colour. Bar check:
RGB-D reaches 0.60 pixacc — a large jump but STILL BELOW the ~0.70 useful-map
bar, so depth is NECESSARY but not sufficient at half-res/deploy budget;
crossing the bar likely needs RGB-D + higher resolution (or more capacity).
Distillation recipe REVISED: the teacher input is RGB-D (depth-dominant, via
projected lidar range), and reaching ~0.70 needs full-res, not just the
input change. NaN root cause + fix confirmed: 3x-input-energy logit overflow
from un-normalised input; per-channel standardisation trains stably. This
supersedes the "colour is the lever" reading in the two prior entries.

## 2026-07-15 — full-res does NOT help: deploy-budget RGB-D seg caps ~0.60, below the bar (resolution ruled out)

Testing the "full-res closes the gap to 0.70" hypothesis from the RGB-D
entry (`scratch/scratch_rgbd_fullres.py`, 320x240, seg-only, standardised):
  luma  FULL-RES (1ch): pixacc 0.439  (< half-res luma 0.48 — full-res HURTS)
  RGB-D FULL-RES (4ch): pixacc 0.566  (< half-res RGB-D 0.60 — full-res HURTS)
Full resolution does NOT help — it slightly HURTS at deploy budget (more
cells to classify, same net capacity spread thinner). Combined with the
earlier capacity result (8x luma params -> +0.02) and the RGB-D input result
(the one lever, +0.12), the picture is: at deploy budget (~70k params) the
RGB-D seg ceiling is ~0.60 regardless of resolution or capacity — BELOW the
~0.70 useful-map bar. So RESOLUTION is ruled out alongside capacity; INPUT
(depth) is the only lever and it caps at 0.60. Open question this raises:
is the ~0.70 bar reachable by ANY net (task achievable, big RGB-D teacher
exists -> distillation frame valid), or is 10-class indoor seg
fundamentally ~0.60 on this half-res data (bar unreachable, map limited to
~0.4-0.5 recall or coarser classes)? -> big-RGB-D test next.

## 2026-07-15 — seg ceiling ~0.60 is ARCHITECTURE-limited, not input/capacity/resolution — the fix is a seg DECODER, not distillation-to-the-deploy-trunk

Decisive test (`scratch/scratch_rgbd_big.py`): a BIG RGB-D net (ch96, 592k
params, ~8x deploy) reaches pixacc 0.599 — IDENTICAL to the deploy-budget
RGB-D (0.597). Full ledger of RGB-D seg on NYUv2 10-class (half-res unless
noted), all in the SAME architecture family (UnifiedVisionNet: early-stride
trunk, NO decoder/skips):
  deploy RGB-D (70k) : 0.597
  full-res RGB-D     : 0.566
  BIG RGB-D (592k)   : 0.599
The ~0.60 ceiling holds across CAPACITY (8x), RESOLUTION (full), and INPUT
(RGB-D is the best input) — none reach the ~0.70 bar. Therefore the ceiling
is ARCHITECTURE-limited: this deploy-trunk shape (stride-early, 1x1-mix, no
upsampling decoder / skip connections) is structurally weak for DENSE seg —
consistent with its mIoU ~0.18 vs SOTA NYUv2 U-Nets at ~0.5. CORRECTS the
"distillation with an RGB-D teacher" recipe: there is NO teacher IN THIS
ARCHITECTURE that crosses the bar (the big net also caps 0.60), so
distillation-to-the-deploy-trunk cannot cross it either. The real seg lever
is ARCHITECTURE — a proper encoder-DECODER with skips (U-Net style) at RGB-D
input — not more params/res/input to the early-stride trunk. Practical
consequence for the deploy queryable map: with this trunk the label head caps
~0.60 accuracy -> ~0.4-0.5 query recall (marginal); to get a USEFUL label map
either (a) add a seg decoder (departs the pinned minimal-trunk budget), or
(b) coarsen to fewer super-classes (40->10 already helped; 10->~5 may reach
the bar), or (c) accept a marginal map. This DEFINITIVELY closes the
seg-bottleneck thread: input=depth is the lever within the trunk, but the
0.60 ceiling is the trunk architecture, and crossing 0.70 needs a decoder.

## 2026-07-15 — decoder CONFIRMATION: architecture is a real lever (+0.04) but no single lever crosses the bar (U-Net RGB-D 0.638 < 0.70)

Tested the "architecture is the lever" inference with a proper U-Net
(encoder-DECODER + SKIP connections; `scratch/scratch_unet_seg.py`), same
RGB-D input / half-res / seg-only / class-balanced as the trunk tests:
  deploy-trunk RGB-D (70k)      : pixacc 0.597  mIoU 0.163
  U-NET RGB-D (980k, dec+skip)  : pixacc 0.638  mIoU 0.213
The decoder+skips CONFIRMS architecture is a genuine lever (+0.04 pixacc,
+0.05 mIoU over the early-stride trunk — the inference direction was right).
BUT it is STILL below the ~0.70 bar. So the honest complete picture across
ALL levers (half-res, 4000-step, no-pretrain setup): INPUT (RGB-D) +0.12 is
the biggest lever; ARCHITECTURE (decoder+skips) +0.04 is real; CAPACITY (8x)
and RESOLUTION (full) do not help. Best combo (U-Net RGB-D) = 0.638 — NO
single lever crosses 0.70. Crossing the bar needs the FULL SOTA seg stack
TOGETHER (RGB-D + decoder + full-res + pretraining/heavy aug; my mIoU 0.21 vs
SOTA NYUv2 ~0.5 is largely the missing training regime), which is beyond both
deploy budget AND this quick from-scratch training. Consequence, definitive:
the deploy label head caps ~0.60-0.64 regardless; a USEFUL queryable-map
label (>=0.70 -> good recall) requires a HEAVY off-line RGB-D U-Net teacher
(full pipeline) then distillation, OR coarser super-classes, OR accepting a
marginal (~0.5-recall) map. Corrects the prior "architecture-limited" framing
to "architecture is A lever, not THE lever; the bar needs the whole stack."
This closes the seg-input/architecture investigation.

## 2026-07-15 — coarser classes cross the PIXEL bar (0.858) but NOT the per-object bar (mIoU 0.23) — surfaces reliable, objects still weak

Tested the "coarser super-classes" deploy option (`scratch/scratch_unet5.py`,
U-Net RGB-D, 5 classes = top-4 + void, vs the 10-class 0.638):
  10-class U-Net RGB-D : pixacc 0.638  mIoU 0.213
   5-class U-Net RGB-D : pixacc 0.858  mIoU 0.231
Coarsening to 5 classes CROSSES the ~0.70 PIXEL bar (0.858) — BUT mIoU
barely moves (0.213 -> 0.231). The large pixacc/mIoU gap means the gain is
dominated by the easy MAJORITY classes (wall/floor); the minority OBJECT
super-classes stay weak (low per-class IoU). Crucially, the segnoise
usefulness bar is a PER-QUERIED-CLASS metric (recall/precision of a class's
objects), which tracks mIoU, NOT pixacc. So coarsening crosses the AGGREGATE
(pixel) bar but NOT the per-object bar: the deploy queryable map would
reliably label DOMINANT SURFACES (wall/floor -> "highlight the floor" works)
but still poorly label OBJECTS (chairs -> marginal). HONEST CLOSE of the
"practical options": (a) decoder helps but doesn't cross (0.638); (b)
coarsening crosses pixacc but not the per-object metric (mIoU 0.23) — good
for surfaces, weak for objects; (c) a reliable OBJECT-query map ("highlight
the chairs") at deploy budget remains HARD — it needs the full SOTA seg
stack (heavy RGB-D U-Net + pretraining), off-line, then distillation. Net:
the queryable MAP MECHANISM is sound (validated ~1.0 given good labels); the
LABEL QUALITY for fine objects is the real deploy limiter, and no
deploy-budget lever (input/arch/capacity/res/coarsening) fixes the OBJECT
case. This definitively closes the seg-label investigation.

## 2026-07-15 — rule-4 audit of the seg-investigation conclusions: SOUND (coarse-class claim STRENGTHENED), one prescription softened, one anti-oracle wrinkle logged

Independent audit of the seg-bottleneck conclusions. Verdicts:
- COARSE-CLASS (the key claim) — SOUND, and STRONGER than stated by direct
  measurement. The 5-class remap resolves to wall(53.1%)/floor(22.2%)/
  cabinet(15.3%)/bed(9.3%) of the scored non-void set; wall+floor = 75.3%.
  Critically, 51.1% of the original OBJECT cells (chair/sofa/table/window/…)
  are folded into class-0=void and are EXCLUDED from pixacc(non-void)
  entirely. So the 0.858 is not merely majority-weighted — the queryable
  objects are STRUCTURALLY ABSENT from the metric (the net only needs
  wall+floor near-perfect). mIoU (which the segnoise usefulness bar tracks)
  barely moves (0.213->0.231). The conclusion "coarsening crosses the
  aggregate pixel bar but does NOT fix the object-query map" is confirmed
  and understated. (Wording nit: the scored set includes cabinet/bed, not
  only wall/floor.)
- ARCHITECTURE LEVER — SOUND. The U-Net-vs-trunk +0.04 is NOT a param
  artifact: the same trunk family is flat in capacity (70k->592k = +0.002),
  so the gain is the decoder+skips. (Caveat: 592k not param-matched to the
  U-Net's 980k, and the U-Net swaps crelu->relu — "architecture" bundles
  minor factors — but the flat capacity curve makes the attribution fair.)
- SOTA-TEACHER PRESCRIPTION — SOFTENED. "A useful object-query map needs a
  full off-line SOTA RGB-D U-Net teacher + distillation" is an INFORMED
  EXTRAPOLATION from literature (SOTA NYUv2 mIoU ~0.5), NOT demonstrated
  here — no experiment in this thread reached the object bar. Corrected
  reading: the LABEL-QUALITY LIMITER is demonstrated (no deploy-budget lever
  moves object mIoU off ~0.21-0.23, even the over-budget 980k U-Net); the
  SOTA-teacher FIX is plausible but UNSHOWN on this half-res data.
- ANTI-ORACLE WRINKLE (log, no numeric impact): the top-K frequency remap in
  `segnet.load_nyu` runs over the FULL 1449-image set BEFORE the 80/20
  split, so the class TAXONOMY (which classes are top-K) is chosen using
  test-split labels. Not a per-sample leak (selects the label space, not
  predictions; the wall/floor/... ranking is split-invariant -> zero numeric
  impact), but strictly GT touches pipeline setup beyond scoring; now fenced
  with a code note. The map-mechanism-sound + label-quality-limiter
  DIAGNOSIS stands; only the SOTA-teacher fix is downgraded to inference.

## 2026-07-16 — the bounded map supports COMPOSITIONAL (conjunctive multi-attribute) queries — "red chair", graded by matched-attribute count

`sspax/semantic_compose.py` — the multi-attribute form of the user's "add
bits individually" directive. Each object binds the UNION of several
attribute codes (CLASS + COLOUR, 4x4, k=12 bits each); a conjunctive query
"red chair" = class_code[chair] ∪ colour_code[red]. Readout by number of
matching attributes (D=360, 10 objects/scene, 32 seeds):
  2-match (red chair)        : 28.7
  1-match (red !chair etc.)  : 18.3
  0-match (neither)          :  6.8
The map GRADES 2 > 1 > 0 — a conjunctive query reads out proportional to
matched-attribute count, because the readout weight is |query_bits ∩
object_bits| (2k for a double match, k for single, ~0 for none). DECISIVE
CONTROL: a RANDOM 2k-bit query gives FLAT readout (3.6/3.9/4.4 across
2/1/0-match) — so the grading is from bit OVERLAP (semantic), not object
count/spatial structure. Red-chair retrieval (threshold @ 90th pct of
non-target): recall 0.94, precision 0.54 (strong single-attribute matches
leak in — the 2:1 readout ratio is clear but not clean separation). So the
bounded map answers conjunctive object queries ("red chairs, not blue chairs
or red tables") from ONE vector with no extra structure — the 11th map
capability, directly realising "binary descriptor bits = composable
attributes". Anti-oracle: synthetic, GT (class,colour) score only.

## 2026-07-16 — compositional-query operator: product-of-queries marginally beats union-bits (recall 1.00 vs 0.95), precision caps ~0.55

Follow-up to the compositional-query result: two conjunctive operators for
"red chair" — (a) UNION-BITS: one query with class∪colour bits; (b)
PRODUCT-OF-QUERIES: query class and colour separately, multiply the density
maps (a red table has high colour-density but low class-density -> product
low). D=360, 10 objects, 40 seeds, threshold @ 90th pct of non-target:
  union-bits         : recall 0.95  precision 0.54
  product-of-queries : recall 1.00  precision 0.56
Product-of-queries is marginally better (perfect recall) — but the predicted
big precision jump does NOT materialise (0.54->0.56): the position-decode
background noise blurs the single-match suppression, so strong 1-attribute
matches still leak past the threshold. Honest read: the map GRADES
attributes clearly (2>1>0, banked) and both conjunctive operators work
(product slightly better), but high-PRECISION retrieval of ONLY the
double-match is limited by decode cross-talk at this D/load (~0.55 ceiling),
not by the operator choice. So compositional QUERYING is a real capability;
clean compositional RETRIEVAL would want larger D or a sharper decode.
Anti-oracle: synthetic, GT (class,colour) score only.

## 2026-07-16 — CORRECTION: compositional-retrieval precision is FLAT in D (larger D does NOT help) — the ~0.55 ceiling is operator/threshold, not cross-talk

Tested the prior entry's inference ("clean compositional retrieval would want
larger D"). It is WRONG. Product-of-queries "red chair" retrieval vs D:
  D=240: recall 0.95  precision 0.54
  D=360: recall 1.00  precision 0.56
  D=720: recall 1.00  precision 0.56
  D=1440: recall 1.00 precision 0.56
Precision is FLAT across a 6x D range — larger D does NOT improve
compositional precision. So the ~0.55 ceiling is NOT a D/cross-talk
limitation: it is inherent to the product operator + 90th-pct threshold
given the readout structure (a 2-match reads ~2x a strong 1-match, and BOTH
scale with D, so their RATIO — hence the separability — is D-invariant).
Self-corrects the prior "wants larger D or sharper decode" inference: the
compositional GRADING (2>1>0) is real, robust, and D-invariant; but
high-PRECISION retrieval of ONLY the double-match is bounded by the
2-match/1-match readout RATIO (~2:1) plus decode background, which larger D
cannot change. To sharpen it you'd need a different operator (e.g. a
margin/ratio test between the two attribute densities, or requiring BOTH to
exceed per-attribute thresholds), not more dimensions. Grading capability:
sound and D-robust; clean double-match retrieval: an operator-design problem.

## 2026-07-16 — deploy-side review of ddd0fdb (seg-lever ledger + compositional queries): accepted, with ONE budget-lane correction that reopens the label head, two compose nits, and a convergence note

The seg-bottleneck characterization is exemplary rule-5/rule-4 work — five
levers measured (input/capacity/resolution/architecture/class-coarsening),
each with a control, the colour claim CORRECTED by the deployment-faithful
deep-net test, and the audit catching the units mismatch + taxonomy wrinkle.
All verdicts accepted. My corrections/notes on top:

- **BUDGET-LANE CORRECTION (reopens the label head).** The thread's closing
  frame — "a seg decoder departs the pinned minimal-trunk budget; fine-object
  labels are beyond deploy budget" — mis-reads the PINNED envelope. The seg
  head runs @KEYFRAME rate, and the envelope (TRAINING_PROGRAM.md hard
  ceilings, from hw/ecp5/host/cnn_budget.py) explicitly provides the
  keyframe lane: **540 MMAC @5 Hz and <=8 MB int8 weights via SDRAM
  STREAMING** — EBR-residency (55 KB) binds only the @frame-rate trunk. A
  ~1 MB int8 U-Net label head at 5 Hz is inside BOTH ceilings. So the
  demonstrated blocker is NOT architecture-at-budget: their own U-Net
  RGB-D (980k, quick 4000-step from-scratch) already reaches 0.638 vs the
  ~0.70 bar, and their audit attributes the remaining gap "largely to the
  missing training regime". REVISED endgame: do not distill INTO the
  trunk (architecture-capped ~0.60) — train the streaming-lane U-Net AS
  the keyframe label head and close the 0.06 with the training regime
  (long schedule + heavy aug + self-sup pretraining on their GPU; no
  external weights required a priori). Platform wrinkle unchanged: the
  student is RGB-D, deploy depth = projected lidar -> the lidar-camera
  EXTRINSICS unlock (see convergence below); NYUv2 dataset depth gates it
  until then.
- **Compose nit 1 (repro):** the ledger's decisive random-query control
  (flat 3.6/3.9/4.4) is NOT in the committed module — semantic_compose.py
  only prints a bracketed note; the control ran in scratch. Same for the
  product-of-queries operator and the D-sweep (both scratch variants).
  The module should absorb its own controls (~15 lines).
- **Compose nit 2 (framing):** retrieval recall/precision are computed
  with GT OBJECT CENTROIDS as the candidate set (readout separability at
  known positions), not the grid-detect pass the other semantic entries
  use — the ~0.55 precision ceiling is not comparable to e.g. segnoise
  precision. The grading claim (2>1>0) is unaffected (that is a readout
  study by design).
- **Compose capacity interaction (untested, cheap):** attribute-union
  binding COMMITS 2k bits/object — by the significance-capacity law
  (cross-talk tracks committed bits) a 2-attribute object loads the map
  like a double-significance one, so composition SPENDS capacity;
  n_obj=10 at D=360 with 24-bit objects is already near the measured
  ceiling. A load sweep (n_obj x attrs/object) would pin the price.
- **Scratch-pattern escalation:** FINDINGS pt3 now rests on seven
  scratch-only scripts (colour/depth probes, rgbd_seg/fullres/big,
  unet_seg, unet5) plus the still-uncommitted classroom wall test.
  Load-bearing verdicts cited by FINDINGS should have committed recipes
  (msg round 3b asks for a consolidated seg_levers module).
- **Convergence note:** THREE threads now block on lidar-camera
  extrinsics — depth-lifted landmarks (rev-AUC 0.941 banked), the verify
  fusion, and now the RGB-D label head's deploy depth. It was already
  FINDINGS' "single highest-value step"; the seg thread makes it three
  for three.

segnet.py changes reviewed: taxonomy anti-oracle fence (split-invariant,
zero numeric impact — correctly logged, correctly fenced), rgb flag with
the NaN root cause documented, ch param for the capacity arm. All fine.

## 2026-07-16 — platform-reality clarification on the label-head plan (user-pinned): no RGB-D sensor exists; the depth channel is PROJECTED 64-ring lidar, and the banked +0.12 lever is dense-Kinect-optimistic

Amends the 2026-07-16 review entry so "the student is RGB-D" cannot be
over-read: the platform camera is an OV5640 (mono Y8 ingest; colour
available but worthless for CNN seg per the corrected deep-net test — Y8
validated). ALL deploy depth is the Ouster-class 64x1024 lidar projected
through the pending extrinsics (all 64 rings in the camera FOV; the
3x1024 net ingest is an unrelated budget choice). Projected-lidar depth
is ring-sparse (~0.6-0.7 deg vertical), FOV-clipped, and
parallax-occluded — every banked RGB-D seg number (trunk 0.597, U-Net
0.638, lever +0.12) used DENSE Kinect depth. msg P1 now REQUIRES a
lidar-like-depth degradation arm FIRST (ring-subsample + aperture crop +
occlusion dropout on NYUv2 depth), with a decision rule: lever survives
>= ~+0.07 -> streaming-lane U-Net plan stands; collapses toward
luma-only -> the label head demotes and surfaces+QBE is the deploy
endgame.

## 2026-07-16 — P4 (msg round 3): the DEPLOY semantic fine-band (lam_min = 0.25 m lidar coherence floor) HOLDS — pins geomspace(0.25, 2, 6)

The other agent flagged that the winning semantic fine ladder (0.1-1.6 m,
recall 0.98) starts BELOW the 0.25 m lidar coherence floor — synthetic-clean
only, since lidar cannot provide sub-0.25 m wavelength content. Re-run
respecting the floor (chair query, D=360, 24 seeds):
  fine  .1-1.6 (BELOW floor)  : recall 0.98  precision 0.93
  floor .25-2  (DEPLOY band)  : recall 1.00  precision 0.86
  oct6  .25-8                 : recall 0.91  precision 0.82
The floor-respecting band geomspace(0.25, 2, 6) HOLDS (recall 1.00,
precision 0.86) — actually higher recall than the below-floor ladder, only
slightly lower precision. So the deploy SEMANTIC fine-band is CONFIRMED at
lam_min = the lidar coherence floor (0.25 m); it does NOT need vision-depth
sub-floor precision. Pins the design: the semantic second-band per segment =
geomspace(0.25, ~2, 6), sharing the place band's datapath. Corrects the
earlier fine-ladder result's synthetic-clean caveat (the 0.1-1.6 win was
real but un-deployable; 0.25-2 is the deployable equivalent). Anti-oracle:
synthetic, GT scores only.

## 2026-07-16 — P5 (msg round 3): dual_use retraction CLOSED — single-vector place survives fully re-encoded semantics (but the aliased case stays two-band)

Closes the dual_use place-side retraction (the audit caught the revisit
reusing sem_vec VERBATIM = identical-vector match). Now the revisit is FULLY
re-encoded: same place (jittered cloud) AND same objects at JITTERED
positions (re-encoded semantics, not verbatim). combined = alpha*place +
(1-alpha)*sem, place-AUC (16 scenes, revisit vs different):
  alpha 1.0/0.9/0.7/0.5/0.3 : place-AUC 1.000 at every alpha
Single-vector place SURVIVES the added (independently re-encoded) semantic
term — place-AUC holds 1.0 down to alpha=0.3 (mostly-semantic). So single-
vector dual-use is SAFE in the realistic revisit (same place + same objects:
both cues match and REINFORCE). HONEST CAVEATS (unchanged from the audit):
(i) the place task is easy — 16 DISTINCT scenes are trivially separable, so
this shows no interference COST; (ii) the semantics MATCH on the revisit
(help, not interfere) — the aliased-place + interfering-semantics stress case
is still untested and not realistic (a revisit sees the same objects). So
"single-vector place is safe" is supported for the realistic case; the
TWO-BAND design (semantic as a separate fine band per segment, no alpha —
the msg-round-3 deploy default) remains the conservative choice for aliased
venues. Retraction closed: the verbatim flaw is fixed, the direction tested,
the result honest (works realistically; two-band is the safe default).
Anti-oracle: synthetic, GT scores only.

## 2026-07-16 — compose module absorbs its controls (nits a/b) + composition SPENDS capacity (nit c)

Addressing the msg-round-3b compose nits: `sspax/semantic_compose.py` now
CONTAINS (not just cites from scratch) its decisive controls —
- (a) the RANDOM-query control runs in-module: REAL 28.7/18.3/6.8 (2/1/0
  match) vs RANDOM 3.6/3.9/4.4 (flat) — grading is bit-OVERLAP, not object
  count; plus the product-of-queries operator (recall 1.00 vs union-bits
  0.94, precision ~0.55).
- (b) the retrieval metric is now LABELLED: candidate set = GT object
  CENTROIDS (readout separability at known positions), NOT the grid-detect
  pass — so the ~0.55 precision is not comparable to segnoise precision.
- (c) NEW `load_sweep`: composition SPENDS capacity. A 2-attribute object
  commits 2k=24 bits, so by the significance-capacity law it loads the
  bounded map like a double-significance one. Chair recall (union-bits) vs
  #objects, 1-attr (k bits) vs 2-attr (2k bits), D=360:
    #obj    1-attr    2-attr
      6      0.88      0.71
     10      0.72      0.45
     16      0.26      0.16
     24      0.04      0.02
  2-attr objects exhaust capacity ~2x faster (n_obj=10: 0.72 vs 0.45) —
  composition is not free; each extra attribute is committed bits that
  count against the sqrt-ish capacity budget. So the deploy semantic map
  trades attribute richness against object count at fixed D — the compose
  price, now measured. Anti-oracle: synthetic, GT (class,colour) score only.

## 2026-07-16 — P1 regime increment (long schedule + heavy aug): does NOT close 0.638->0.70 — overfitting-limited; self-sup pretraining is the last lever

P1 (msg 3b): the label head is budget-legal @keyframe, so the gap should be
the TRAINING REGIME. Increment 1+2 = long schedule (5x = 20k steps) + HEAVY
aug (flip + photometric + depth-noise), ch48 RGB-D U-Net (2.2M params, on the
keyframe lane), warmup-cosine LR (`scratch/scratch_unet_regime.py`):
  quick baseline (4000 steps, light noise): pixacc 0.638  mIoU 0.213
  regime (20k steps + heavy aug)          : pixacc 0.642  mIoU 0.205  object-acc 0.502
FLAT — schedule + aug do NOT close the gap. Train loss fell to 0.090 while
eval stayed 0.642 => the net OVERFITS the small 1449-image NYUv2-labeled set
despite heavy aug; the regime's first two components are NOT the lever
(counters the "gap is the missing training regime" hypothesis for these two).
POSITIVE nuance: the U-Net decoder lifts OBJECT-level acc 0.23 (deploy trunk)
-> 0.50 (U-Net) — a MARGINAL map (~0.43 query recall by the segnoise curve),
below the ~0.70 useful bar but well above the trunk's ~0.15. PROVISIONAL
NEGATIVE (2/3 of the regime): fine-object labels from-scratch on 1449 images
cap ~0.64 pixel / 0.50 object with schedule+aug. UNTESTED last lever: SELF-SUP
ENCODER PRETRAINING (the reviewer's biggest-yield bet) on the UNLABELED NYUv2
raw distribution — the fix for the overfitting/small-labeled-set limit. If
self-sup also caps <0.70, the from-scratch object-label negative is FULL and
the surfaces+QBE fork ships alone (labels wait on external pretraining or a
bigger labeled corpus). Budgets: 2.2M params ~2.2 MB int8 < 8 MB lane;
~keyframe-legal. Anti-oracle: TIME-independent (NYUv2 frames independent),
GT scores only.

## 2026-07-16 — P0 (msg 3b): VISION net exported via headio v2 + validated; LIDAR export blocked on a geometry discrepancy

`sspax/artifacts/vision_head.npz` — the trained vision net (segnet_nyu.pkl)
exported through headio v2 (int8 per-channel quant): trunk = Conv_0-3
(crelu, cell 4), track = [w] (Conv_4 cout 1; the cut Conv_5 DROPPED per the
P1 thermometer retraction), desc = Conv_6 (32-bit), seg OMITTED (the 40-class
head is not the k-bit label latent — that arrives with distillation). in_div
255, in (120,160,1). load_head passes _validate; headio `stability` gate on
held-out TUM (cross-dataset): descriptor bit-agreement 0.750 adj / 0.520 far
-> GAP 0.23, vs the random-weights baseline gap 0.06 (0.892/0.828) — the
trained descriptor discriminates same- vs different-place 3.6x better than
random and TRANSFERS across datasets. Artifact ~KB int8, committable.
LIDAR export BLOCKED (flag for deploy side): lidar_ring.ring_raster produces
(1, 1024, 3) [in_h=1, rings-as-CHANNELS in_ch=3], but headio v2's pinned
lidar geometry is (3, 1024, 3) [in_h=3, in_ch=3]. The "3x1024 rings-as-
channels" spec is realised two different ways — my net binds rings to the
CHANNEL axis (1D-azimuth conv), headio expects rings on the HEIGHT axis.
Needs reconciliation (either the ring net re-shapes to (3,1024,C) or headio
admits (1,1024,3)); the vision head — the carrying one — is unaffected and
lands now.

## 2026-07-16 — REQUIRED lidar-like-depth arm (msg efc5f0e, decision-gating): the depth lever HALVES on sparse projected-lidar depth (+0.059 < +0.07) -> label head DEMOTES

Platform reality (user-pinned): no RGB-D sensor; deploy depth = the 64-ring
lidar PROJECTED into the Y8 camera FOV -> sparse (~0.7 deg/ring), FOV-clipped,
occlusion-dropped. My banked "depth is the lever +0.12" used DENSE Kinect
depth (optimistic). Degraded NYUv2 depth to lidar-like geometry (vertical
ring-subsample ~0.7 deg, aperture crop top/bottom 15% -> 0, occlusion dropout
at depth discontinuities, nearest-fill inside the aperture; 68% pixel
coverage) and re-measured at the deploy trunk (pixacc, `scratch/
scratch_lidarlike_depth.py`):
  luma only         : 0.480
  luma + DENSE depth: 0.599   lever +0.119  (reproduces the +0.12 reference)
  luma + LIDAR-LIKE : 0.539   lever +0.059
The depth lever HALVES (+0.119 -> +0.059) on realistic sparse lidar depth,
landing just BELOW the +0.07 decision gate. DECISION (per msg efc5f0e): the
fine-object LABEL HEAD DEMOTES; the deploy endgame is the SURFACES tier
(coarse-class 0.858 pixacc) + label-free QUERY-BY-EXAMPLE on the
tracking-descriptor bits (0.95 retrieval, compositional grading applies to
desc bits + coarse-class bits). Honest nuance: the lever did not fully
COLLAPSE to luma-only (it halved, still +0.059), so a DENSER lidar or a
better projector (more rings-in-FOV, tighter extrinsics) could recover it —
but at the current 64-ring projection it is below the gate. Combined with the
P1 regime negative (from-scratch caps 0.64/0.50 even on DENSE depth), the
fine-object queryable-map LABEL is not viable at deploy on this platform;
surfaces+QBE ships alone, and the object-label map waits on a real RGB-D
sensor OR denser lidar+extrinsics. This RESOLVES the label-head deploy
question. Anti-oracle: GT scores only; depth degraded geometrically, no GT.

## 2026-07-16 — deploy-side review of 491477f/af36073: demotion ACCEPTED as the platform verdict; vision head gated cross-box; lidar export unblocked (geometry misread)

Their round closes the label-head question exactly as the decision rule
demanded — accepted in full: P4 deploy fine-band geomspace(0.25, 2, 6)
PINNED (recall 1.00/precision 0.86 at the coherence floor); P5 dual_use
closed honestly (single-vector place survives re-encoded matching
semantics; two-band stays the aliased-venue default); compose module now
carries its own controls + the capacity price (2-attr objects exhaust
~2x faster — attribute richness trades against object count at fixed D);
P1 regime increment honestly FLAT (schedule+aug 0.642, overfitting-
limited at 1449 labeled images; U-Net object-acc 0.23->0.50 = marginal
map); and the REQUIRED lidar-like-depth arm fired the demote rule
(+0.119 dense -> +0.059 lidar-like < +0.07): fine-object CLASS labels
are OFF this platform's roadmap; surfaces tier + query-by-example desc
bits ship alone. The self-sup increment is hereby RELEASED (moot for
deploy — the input channel, not the regime, is the binding constraint;
bank the provisional negative as final for this platform).

Notes: (a) their stability comparison cited the seg-bit random baseline
(gap 0.06); the matched desc-bit baseline is 0.954/0.906 (gap 0.048) —
the trained gap 0.230 is ~5x random either way. (b) LIDAR export
unblock: the contract was misread, not mismatched. headio's lidar meta
(in_h=3, in_w=1024, in_ch=3) describes the RAW deploy array (rings,
beams) = (3, 1024); forward() transposes to (1024, rings-as-channels)
internally — the SAME layout their (1,1024,3) NHWC net consumes after
squeezing the dummy height axis. Export recipe: squeeze H, emit each
(1,k) conv as conv1d k (weights (cout,cin,1,k) -> (cout,cin,k)), meta
in_h=3/in_w=1024/in_ch=3/in_div=1.0/cell=4 — the headio "lidar" fixture
mirrors the result exactly (selftest passes on it).

## 2026-07-16 — vision_head.npz gated on the deploy box: stability reproduces THEIR numbers exactly; objmap2 desc-key = real but partial cross-dataset transfer

`sspax/artifacts/vision_head.npz` (their P0 export) through both deploy
gates on this box:
- stability (TUM fr3): 0.750 adj / 0.520 far — EXACTLY their box's
  numbers (cross-box bit-faithful artifact transfer through the v2
  contract, the contract's purpose demonstrated). Gap 0.230 vs the
  desc random baseline gap 0.048: ~5x random, cross-dataset.
- objmap2 desc-key gate (16 queries, top-3): stage-1 16/16; stage-2 AUC
  0.592 (1-view) / 0.597 (3-view); err med 0.608 / 0.494 m, p90 1.84 /
  2.22. Read against the brackets: random-weights 0.522/0.548 AUC and
  ~1.1-1.2 m err; banked NATIVE-TUM references int 0.805 / census
  0.165 m. VERDICT: the NYUv2-trained descriptor transfers real
  map-binding signal to an unseen dataset (err halves vs random, AUC
  +0.05-0.07) but does not reach native-simple-feature quality — the
  cross-DOMAIN gap is the price. Deploy reading: QBE on the platform
  will run on OV5640 imagery (also out-of-domain for the head);
  platform-domain adaptation (finetune on OV5640 frames once hw_snap
  produces them, self-supervised — no labels needed) is the identified
  lever, NOT more NYUv2 training.

## 2026-07-16 — FIRST SILICON (Icepi Zero live): FAST-9 HW gate PASS bit-exact; all five bitstreams configure; no-sensor baselines recorded

The board arrived and is plugged in. ECP5 LFE5U-25 detected on JTAG
(idcode 0x41111043); toolchain + UART adapter live.
- `top_fast9_uart` (pre-built 2026-07-14, timing 65.6 MHz vs 50 MHz
  constraint) flashed; `hw_fast9.py` streams the school_run2 golden
  fixture at 2 Mbaud: **HW PASS — 3364 centres bit-exact on silicon**
  (sim == golden == silicon; t=12). The ECP5 track's first hardware
  gate, and the FAST-9 detector is now silicon-proven.
- Sensor-less PRE-FLIGHT (scratch_preflight.py): all four sensor tops
  configure the physical board and answer their UART protocols with
  the documented no-sensor baselines — ism330_id/stream WHO_AM_I=0x00
  (SPI no-hang confirmed on silicon; wired => 0x6b), ov5640_id
  ID=0xffff ack=0x03 (both SCCB phases NACK on the floating bus;
  wired => 0x5640 ack=0x00), ov5640_snap status all-zero (init ROM
  parked without the sensor). Wiring day is now flash-only with
  unambiguous flips. Board restored to top_fast9_uart; gate re-PASS.
## 2026-07-16 — P2 (msg round 3): the classroom wall test committed as a module (sspax/place_wall.py) with the GT-leak control INSIDE

Promoted the scratch classroom learned-vs-fixed place test to a committed
module — "the single most citable negative of the project" — with the
GT-LEAK CONTROL built in (both training-negative definitions run + printed).
Honest classroom venue (withheld odometry, 130 frames, 244 revisit pairs),
learned lidar saliency vs uniform, revisit-vs-far AUC:
  train-negatives          uniform  learned  delta
  temporal (clean)          0.930    0.908   -0.022
  pose-derived (GT-LEAK)    0.930    0.976   +0.046
CLEAN (label-free temporal negatives): learned <= uniform -> NO learned place
gain; the front-end cannot manufacture place separability on the honest
venue = the loop-closure wall, DIRECT (what run2's no-revisits and P1's
selfrot artifact could only approach indirectly). The +0.046 "gain" appears
ONLY with pose-derived training negatives, which leak the same withheld
odometry the gate scores on (rule-2 violation) — the trap is now
self-documented in the module. Pair-count reconciliation (folds the two
banked classroom entries): 110 keyframes -> 179 pairs (recipe gate),
130 keyframes -> 244 (this module's cache) — same venue, more frames = more
pairs. P2 complete; FINDINGS pt3's/pt5's classroom citation now has a
committed recipe. Anti-oracle: temporal negatives are label-free; withheld
odometry scores only.

## 2026-07-16 — P3 (msg round 3): golden vectors + numpy-ONLY checker for the semantic map (deploy-box acceptance, no JAX)

`sspax/semantic_golden.py` + `sspax/artifacts/semantic_golden.npz` (2.7 KB):
freezes a fixed scene's map inputs (W, roles seed, class codes) and the
FPGA-relevant QUANTIZATION-recall sweep (chair recall under the real
polar-quant model, q_polar_np bit-identical to sspslam.quantized.q_polar):
  float      : 0.947
  6-bit/cell : 0.877   (nph16 nmag4)
  4-bit/cell : 0.873   (nph8  nmag2)
  3-bit/cell : 0.773   (nph4  nmag2)
The `check` path re-implements binding/query/polar-quant in PURE NUMPY (the
module imports only numpy; sspax/__init__ guards jax via HAS_JAX, so the
no-JAX deploy box runs it) and recomputes the recalls — GOLDEN CHECK PASS,
all |d| = 0.000 (deterministic). So the deploy box can VERIFY the
FPGA-relevant semantic numbers without JAX, as a map acceptance gate. The
4-bit/cell recall (0.873, vs float 0.947) reconfirms the map survives
low-precision (this fixture's scene is harder than the semantic_quant entry's
0.97 — the golden is self-contained; the deploy check verifies CONSISTENCY
against the frozen expected, not cross-entry absolutes). Anti-oracle:
synthetic, GT scores only. P3 complete.

## 2026-07-16 — USB virtual-sensor streaming (STREAM.md): protocol pinned, v0 SILICON-GATED — 3-ring lidar streams LIVE at 1.5x the 20 Hz target; OV5640 path kept

Shipment-delay pivot (user directive): the robot's Linux PC will stream
lidar + QVGA camera over USB; the FPGA ingests them as VIRTUAL SENSORS
(no PC preprocessing — subsets/formats declared in stream headers; the
OV5640 DVP path is KEPT: the cam assembler's output contract ==
dvp_capture's pixel stream, SRC MUX selectable, tops/#31 untouched).

Board transport facts (measured): the USB-UART is an FT231X (3 Mbaud
ceiling; only TXD/RXD/RTS/DTR reach the FPGA — no FT245 mode), and TWO
USB-C ports have D+/D- wired directly to FPGA pins (LPF usb_dp/dn[0/1])
-> a soft USB-FS CDC device (~1 MB/s) is the pure-RTL v1 transport.

Protocol (hw/ecp5/STREAM.md): transport-agnostic framed packets
(A5 5A | type flags len seq | payload<=4096 | CRC16-CCITT), column-block
lidar (ring_ids declared) + row-block camera + reserved IMU type +
CTRL/STATUS/CREDIT; sensor timestamps travel IN the stream and are the
temporal authority downstream (interval-matched delayfuse law holds at
any link speed; sub-real-time replay is semantically identical).
RTL v0 (rtl/stream_ingest.v + top_stream.v): byte-at-a-time parser with
per-packet SHADOW-COMMIT (digest/meta snapshot at header, rollback on
CRC fail — zero EBR, no store-and-forward); TX echoes per-frame CRC16
digests + status. 54.2 MHz timing vs 50 constraint.

Gates: G0 sim PASS (resync + 3 digests + corrupt-packet ROLLBACK with
clean-resend recovery + counters). G1 SILICON PASS first run: 18/18
frame digests bit-exact (6 cam QVGA + 12 lidar 3x1024 interleaved),
583 pkts, 0 crc_drops, 0 seq_gaps, 197 KB/s goodput @ 2 Mbaud; soak =
241 lidar frames at 30.0 fps sustained (1.5x the 20 Hz live tier),
7956 pkts, 0 drops. Negative banked: 2.4 Mbaud (FTDI-exact, DIV=21)
FAILED on this host — the macOS VCP driver silently kept ~2 Mbaud
(+19% line mismatch, zero packets parsed); reverted to 2 Mbaud, retest
with Linux ftdi_sio on the robot box before enabling.

Live matrix (v0 @ 197 KB/s): 3-ring lidar @ 20 Hz LIVE with 50%
headroom; cam QVGA rides at ~1-2 fps alongside; the deploy-live combo
(3-ring @ 20 Hz + cam @ 5 Hz keyframes = 507 KB/s) needs the v1
soft-USB transport; full-fidelity 64-ring + 30 fps (4.9 MB/s) exceeds
every link this board has -> sub-real-time (deterministic-identical) +
SDRAM-paced at-rate bursts (G4, needs #44). Next gates: G2
fast9-through-ingest (64x64 fixture as CAM packets -> SRC MUX -> fast9
vs golden), G3 encoder-on-streamed-lidar (#43), G4 paced bursts (#44).
host/hw_stream.py is the robot-side reference implementation (pure
pyserial, runs unchanged on Linux).

## 2026-07-16 — their P2/P3 verified on the deploy box: golden checker PASS cross-box; round-3 queue fully closed

ffe1f36 reviewed + verified here: (a) place_wall.py — the wall test is
now a committed module with the GT-leak trap self-documented, and the
179-vs-244 pair count is reconciled (110 vs 130 keyframes, same venue);
nit closed. (b) semantic_golden check runs PURE NUMPY on this box:
all four quant recalls reproduce |d|=0.000 — GOLDEN CHECK PASS
cross-box. The FPGA-relevant semantic numbers are now deploy-box-
verifiable without JAX. Every item from msg rounds 3/3b is closed;
the round-4 queue (lidar head export, finetune harness, surfaces+QBE
reference) is what remains on their side.
## 2026-07-16 — P0 round 4: LIDAR head exported via headio v2 (geometry was a misread, not a mismatch); vision head gated cross-box (P0 fully DONE)

LIDAR export unblocked (deploy-side clarification): headio's lidar meta
(in_h=3, in_w=1024, in_ch=3) is the RAW deploy array (rings, beams)=(3,1024);
forward() transposes to (1024, rings-as-channels) = exactly the layout the
(1,1024,3) NHWC ring net consumes. No reconciliation needed. Exported
lidar_ring.pkl -> `sspax/artifacts/lidar_head.npz` (16.9 KB int8): trunk =
Conv_0-3 as kind=conv1d (k=7,7,5,1 squeezing the H axis), track=[w] (Conv_4,
cout 1), desc=Conv_5 (32-bit), seg omitted; in_h=3 in_w=1024 in_ch=3
in_div=1.0 (METERS, NOT /255), cell=4, act=crelu. load_head passes _validate;
forward on a raw (3,1024) ring array -> track (256,1) + desc (256,32) per
azimuth bin (256 = 1024/cell). Deploy side gates stability with ring
fixtures. Its desc bits are the LIDAR QBE channel.
VISION head (491477f) GATED CROSS-BOX on the deploy box: stability 0.750 adj
/ 0.520 far — EXACTLY my numbers (v2 transfers artifacts bit-faithfully);
objmap2 desc-key stage-1 16/16, stage-2 AUC 0.592/0.597, err 0.61/0.49 m
(random 0.522/~1.2 m; native-TUM 0.805/0.165 m) = real-but-PARTIAL
cross-dataset transfer (err halves vs random; native features win on home
domain). CORRECTION to my stability entry's baseline: cite the MATCHED
desc-bit random baseline 0.954/0.906 (gap 0.048), not the seg-bit 0.06 — the
trained gap 0.230 is ~5x random either way. The QBE quality lever is
PLATFORM-DOMAIN finetune (OV5640 frames), not more NYUv2 — prep as round-4
P1. P0 (both heads) DONE.

## 2026-07-16 — P3 round 4 (the demotion's constructive half): SURFACES-TIER + QBE per-segment band — the end-to-end recipe of what SHIPS

`sspax/surfaces_tier.py` — the deployable semantic stack after the fine-object
label head demoted: ONE bounded D=360 vector per 5-keyframe segment carrying
TWO bit sources bound at cell 3D positions, on the P4 deploy fine-band
geomspace(0.25, 2, 6), stored at 4-bit polar quant (q_polar_np):
  1. SURFACE class bits (5-class U-Net output; pixacc 0.858 banked)
  2. DESC bits (tracking descriptor's 32 sign bits; 0.95 retrieval, gated
     cross-box) -> label-FREE query-by-example.
Results (24 synthetic room segments; surface labels + desc clusters are the
controlled stand-in for the exported vision head's seg-argmax + desc bits):
  surface-query recall (readout AUC vs other cells): floor 0.878, wall 0.955
  QBE retrieval (example cluster's desc bits -> its cells): AUC 0.986
  bytes/segment: 180 B map (D=360 @ 4 bits) + ~448 B registry (64 cells) = ~0.6 KB
So surfaces + label-free QBE ship in ONE ~0.6 KB bounded per-segment vector at
4 bits/cell — the constructive endgame the whole semantic thread now points
at. Fine-object CLASS labels wait on a real RGB-D sensor (demotion, efc5f0e).
The registry (cell positions+class) dominates the map bytes and scales with
bound-cell count -> aggregating cells to per-surface regions/objects (per the
capacity+compose laws) shrinks it. Anti-oracle: synthetic, GT surface class +
cluster id score only.

## 2026-07-16 — P1 round 4 (PREP, harness ready): platform-domain desc-head finetune, dataless-runnable

`sspax/platform_finetune.py` — the QBE quality lever for the camera day (the
gated vision head transfers real-but-PARTIAL cross-dataset; the lever is
PLATFORM-DOMAIN adaptation, not more NYUv2). Label-free luma-jitter InfoNCE on
the DESC head, two arms (trunk FROZEN / desc-only vs trunk LOW-LR 0.1x, both
printed), re-export via headio v2. DATALESS-RUNNABLE: uses build/snap.npz
(hw_snap.py 'S', key "img", 320x240 Y8) if present, else synthetic luma smoke.
Smoke dry-run (200 synthetic frames, NOT representative): both arms run,
retrieval ~0.10 (synthetic = large uniform regions, non-discriminative; NOT a
real yield), stability ~0.85-0.89, re-export VALIDATES through headio load_head.
Harness READY; on the camera day point at build/snap.npz and pick the arm that
improves the deploy-side objmap2 gate (pre-finetune ref: stability 0.750/0.520,
objmap2 ~0.59). Round-4 P1 prep DONE (fires when OV5640 lands). Anti-oracle:
label-free InfoNCE, no GT.

## 2026-07-16 — seg_levers consolidation (msg round 3b nit d, the LAST carried item): FINDINGS-pt3 verdicts now have a committed recipe

`sspax/seg_levers.py` — folds the ~9 scratch seg-lever scripts into ONE
parameterized harness. `train_seg(channels, arch, ch, hw, n_class)` reproduces
any banked lever (channels luma|rgb|rgbd|rgbd_ll incl. the lidar-like
degradation; arch trunk|unet; capacity ch; resolution hw; class count), and
`LEDGER`/`table` records the verdicts inline (luma 0.480 / RGB 0.464 / RGB-D
dense 0.597 / RGB-D lidar-like 0.539 / full-res 0.566 / ch96 0.599 / U-Net
0.638 / regime 0.642 / 5-class 0.858). Verified: table prints; train_seg runs
(luma 200-step smoke 0.388 en route to the banked 0.480). So every FINDINGS-pt3
seg-lever verdict now has a committed, reproducible recipe (no more
scratch-only load-bearing citations). ALL msg round 3/3b/4 items are now
CLOSED: P0 both heads exported, P1 label head demoted + finetune harness ready,
P2 place_wall + golden(+compose), P3 golden + surfaces-tier, P4 deploy band,
P5 dual_use, compose nits, seg_levers. Anti-oracle: as per each lever's entry.

## 2026-07-16 — surfaces-tier SHIPPING optimization: SPARSE cell binding is both smaller AND sharper (the capacity law optimizes the recipe)

The surfaces-tier registry (P3) dominated the segment bytes; the fix follows
directly from the sqrt(D) capacity law. Binding every Nth cell into the D=360
band (4-bit quant), surface + QBE readout AUC vs cells/segment:
  cell-frac  floor  wall   QBE   cells  bytes/seg
    1.00     0.878  0.955  0.986   64     628 B
    0.50     0.975  0.974  0.991   32     404 B
    0.25     0.996  0.987  1.000   16     292 B
FEWER cells -> HIGHER recall AND fewer bytes. At 64 cells the map is far over
its ~sqrt(D) capacity (cross-talk blurs queries); a SPARSE ~16-cell
representative set stays within capacity -> floor 0.996 / wall 0.987 / QBE
1.000 at 292 B (2.2x smaller than the naive 628 B). SHIPPING RECOMMENDATION:
the surfaces-tier binds a SPARSE representative cell set per segment (not every
cell), which the capacity law makes free — cheaper AND more accurate. So the
deployable per-segment semantic band is ~0.3 KB (180 B map + ~112 B registry
for ~16 entities), surfaces + label-free QBE, at 4 bits/cell. The capacity law
(banked earlier) is not just a limit — it prescribes the sparse binding that
optimizes the actual ship recipe. Anti-oracle: synthetic, GT scores only.

## 2026-07-16 — CORRECTION/RETRACTION: the surfaces-tier "sparse cell binding is smaller AND sharper" claim above is WRONG (rule-4 self-audit)

The entry immediately above banked a false win. It measured surface/QBE AUC
ONLY over the cells that remained after subsampling — so dropping cells shrank
the map AND the eval set together, and the "recall goes UP as cells drop" was
an artifact of scoring on a smaller, easier held set (the hard-to-separate
cells were removed from the negatives too, not just from the map).

The honest test — bind a SPARSE subset but EVALUATE on the FULL cell set,
splitting separability (of bound content) from coverage (fraction of ALL
surface cells retrievable):
  bind-frac  sep-floor  cov-floor  sep-wall  cov-wall  bytes
    1.00       0.878      0.542      0.955     0.784    628 B
    0.50       0.721      0.481      0.791     0.503    404 B
    0.25       0.596      0.288      0.695     0.365    292 B
Sparse binding HURTS both separability AND coverage — surface classes are
shared codes bundled across many cell positions; removing positions just
removes retrievable content, it does not relieve a capacity ceiling (D=360 is
not saturated by ~64 shared-code cells the way M distinct objects would be).
The naive full-density surfaces-tier (64 cells, ~628 B, floor 0.878 / wall
0.955 / QBE 0.986) STANDS as the ship recipe; there is no free byte win here.
Lesson (again): never score a subsampled map on the subsampled set — evaluate
on a fixed full probe. Banked pre-push, caught pre-push; nothing shipped.

## 2026-07-16 — semantic map ROLE-PHASE quantization: roles quantize to 3-bit loss-free, 1-bit sign still works (real-only role arithmetic)

Prior quant work coarsened the STORED map (4 bits/cell). This coarsens the ROLE
codes themselves — the bind/query arithmetic backbone — since on FPGA role
phases come from a small LUT/CORDIC, not full-precision. Chair-recall gate
(D=360, map full-precision, roles quantized identically in bind AND unbind =
the correct deploy model), 24 scenes x 3 role-seeds:
  roles full        0.956 +/- 0.014
  roles 3-bit/8ph   0.949 +/- 0.004    (loss-free)
  roles 2-bit/4ph   0.934 +/- 0.036    (-0.02)
  roles 1-bit/sign  0.888 +/- 0.014    (-0.07, but WORKS)
Monotone, stable. 3-bit (8-phase) role LUT is loss-free; 1-bit SIGN roles
(roles in {+1,-1}, real) cost ~0.07 recall but make the role arithmetic
REAL-only — the bind roles[bits].sum ⊙ exp(iW·x) needs no complex multiply for
the role factor, halving that multiplier cost on the FPGA. Combined with the
4-bit/cell stored map, the entire semantic map is a low-precision fixed-point
object end to end: 3-bit role LUT + 4-bit map storage, no recall loss. No leak
(bind/unbind self-consistent; synthetic GT scores only). Rule-4 audit dispatched.

## 2026-07-16 — CORRECTION to the role-phase quant entry above (rule-4 audit + end-to-end run)

The banked role-phase numbers reproduce EXACTLY (audit confirmed: full 0.956 /
8ph 0.949 / 4ph 0.934 / 2ph 0.888; 1-bit is genuinely real {+-1}; a random-code
query scores 0.000 so the metric is honest). But three framings were over-reads:

1. STRUCTURAL CAVEAT (the important one). Bind and unbind use the SAME quantized
   roles, so the matched-code readout carries the factor qr·conj(qr)=|qr|^2=1
   for ANY magnitude-preserving phase quantization — the signal PEAK is
   phase-quant-INVARIANT by identity (verified: peak density barely moves full
   vs 1-bit). Only the CROSS-TALK floor rises (raising the MAD detect threshold,
   burying weak peaks). So "roles quantize gracefully" is a statement about
   cross-talk robustness on SPARSE scenes, NOT about role signal-path precision
   — the signal path cannot be stressed by this test by construction.

2. "3-bit role + 4-bit map, no loss end to end" was NEVER RUN — it multiplied
   two single-factor results. Now run (roles AND map quantized together, 24x3):
     roles full  + map 4-bit : 0.901
     roles 3-bit + map 4-bit : 0.889
     roles 2-bit + map 4-bit : 0.881
     roles 1-bit + map 4-bit : 0.880
   Combined 3-bit-role+4-bit-map = 0.889, ~0.07 BELOW full 0.956 — NOT loss-free.
   The map quant dominates the cost; once the map is 4-bit, role precision is
   irrelevant (1/2/3-bit all 0.880-0.889, within noise). Corrected deploy read:
   at the shipped 4-bit map, roles can drop to 1-bit sign essentially for free.

3. "real-only (1-bit) is the FPGA enabler" over-narrows: 2-bit {+-1,+-i}
   (Gaussian-integer) is ALSO multiplier-free (sign flip + real/imag swap, no
   CORDIC) and scores HIGHER in isolation (0.934 vs 0.888). 1-bit buys no extra
   hardware saving over {+-1,+-i}. (At 4-bit map they tie, so it is moot there.)

Also: the +-0.004/0.014 bars are std of 3 role-seed MEANS; per-scene recall is
bimodal (many 1-chair scenes), true uncertainty is wider. Net: role phases DO
tolerate coarse LUTs, but the honest end-to-end low-precision map costs ~0.07
recall (0.956->0.889), driven by MAP quant, not roles. Nothing pushed.

## 2026-07-16 — deploy-side review of their round-4 close (401e34c..bacc435): all accepted; both artifacts verified cross-box

Their five commits close every msg item: P0 lidar head exported per the
§3 recipe; P3 surfaces_tier (the ship recipe: ONE D=360/segment, surface
bits + desc bits on the deploy fine band, 4-bit quant — floor 0.878 /
wall 0.955 / QBE 0.986 @ ~628 B); P1 platform_finetune harness
(dataless-runnable, fires on camera day); seg_levers consolidation (no
more scratch-only FINDINGS citations); and two same-day rule-4
self-corrections done right — the sparse-binding "smaller AND sharper"
false win (eval-set artifact; full-density recipe STANDS) and the
role-quant over-read (matched-code readout is phase-quant-invariant by
construction; honest end-to-end 4-bit map + any-precision roles = 0.889
vs 0.956 full — map quant dominates, roles drop to 1-2 bit free).
Verified on this box: semantic_golden check PASS incl. the compose
grading triple (|d|max = 0.00, pure numpy); lidar_head.npz loads
through headio v2 (_validate ok: lidar (3,1024,3), in_div 1.0 METERS,
crelu), forward on a (3,1024) ring array -> track (256,1) + desc
(256,32), deterministic. Real-raster desc stability on school data =
deploy-side follow-up.

## 2026-07-16 — WEBVIS: real-time FPGA-in-the-loop browser demo on REAL dataset data (hw/ecp5/host/webvis.py) — full tour 414/414 scans silicon-verified at 5 Hz, zero overruns

User-directed demo, running live at http://localhost:8790. One process:
the REAL SPOT classroom tour (data/spot_telluride, 1024-beam ring-33
slice — the shipped 2D input) at its native 5 Hz keyframe rate; EVERY
scan streams THROUGH the plugged Icepi Zero (top_stream, STREAM.md
protocol, ring subset declared in-header) and is DIGEST-VERIFIED
bit-exact on silicon before the pipeline consumes it — a live
acceptance counter on the deploy ingest path; the SHIPPED lidar-only
recipe (BandSLAM + matcher + CV-guess chain, replicated-with-cite from
runners/spot.run_cv) builds the bounded map in real time; browser UI
(SSE + canvas) renders registered world points, the estimate trail,
the WITHHELD-odometry ghost (display/eval only, anti-oracle stated
in-page), bounded-memory/segment counters, and the FPGA link panel.
FULL-TOUR NUMBERS (first complete pass): 414/414 scan digests verified
on silicon, 0 CRC drops, 0 real-time overruns; FPGA roundtrip 27 ms/kf
+ SLAM 45 ms/kf inside the 200 ms keyframe budget; final map 354 KB /
63 segments; selftest arm (40 kf): med err vs withheld ref 0.010 m.
Serve mode loops the tour (fresh SLAM per pass) so the browser always
shows a live build. Fix banked en route: pyserial blocking-read
granularity made roundtrips ~511 ms (a 256-byte read against a 0.5 s
port timeout for a 21 B echo) -> 20 ms port timeout + per-frame echo
polling = 27 ms; stale echoes from prior sessions are skipped by
frame-id match. Headless: `python3 hw/ecp5/host/webvis.py selftest` =
WEBVIS SELFTEST PASS (numbers only, rule 3). This is the stream v0's
first live APPLICATION (G1 was its gate); G2 (fast9-through-ingest
camera lane) and G3 (encoder-on-streamed-lidar, #43) extend the same
loop with on-chip compute.

## 2026-07-16 — SAME-CLASS multiplicity: shared-code instances are cross-talk-limited, NOT merge-limited (the _detect cliff is a fixed-threshold artifact)

New regime (deploy-realistic: a room of many chairs). Bind K objects sharing
ONE class code at separated positions (+6 filler), query the class, count
distinct instances recovered. Raw _detect recall (D=360, 4-MAD, 20 scenes):
  K:      1     2     4     6     8    12    16    20
  recall 1.00  1.00  0.71  0.22  0.15  0.07  0.01  0.01
Looks like a cliff at K~6 — but the DIAGNOSTIC (self-audit before banking)
shows the signal is preserved and the cliff is the DETECTOR, not the map:
  K   peak@true  bg-MAD  thr(4MAD)  true-chairs>thr  z-score
  1     12.52    1.485     5.82         1.00           8.5
  4     12.21    2.308     8.99         0.97           5.4
  8     11.72    3.127    12.28         0.45           3.8
  16    11.05    4.081    16.22         0.07           2.7
The peak AMPLITUDE at true chair positions is ~constant (12.5 -> 11.0); what
grows is the cross-talk BACKGROUND (MAD 1.49 -> 4.08) as K shared-code deltas
sum their sidelobes. The true-position z-score degrades GRACEFULLY (8.5 -> 2.7
over K=1..16), ~1/sqrt(K)-like — NOT a merge. The recall cliff is the FIXED
4-MAD threshold crossing the falling z-score around K~6-8 (z~4). Honest reads:
  - Same-class multiplicity IS more limited than distinct-class capacity
    (~sqrt(D)~19 distinct vs the shared-code regime), because every instance
    excites the SAME query, so their sidelobes stack into common background.
  - But the limit is CROSS-TALK, not peak merging: the map still holds each
    instance's peak (amplitude preserved); a fixed 4-MAD detector recovers ~4-5
    clean same-class instances, and a tuned (lower/matched) threshold recovers
    more since the peaks stay well above the median out to K~16 (z~2.7).
  - Deploy implication: same-class-dense scenes (chairs) are a DETECTOR
    problem, not a representation problem — lower the threshold or read out
    per-instance at known candidate positions; the bounded vector keeps the
    peaks. Anti-oracle: synthetic, GT positions score only.
## 2026-07-16 — THE VSA ENCODER + COMPRESSED MAP ARE ON SILICON (ECP5): bit-exact on real scans; the map lives ON CHIP and the laptop decodes its fetched codes

Task #43's core rung + the user-directed architecture ("keep the map on
chip, exchange the compressed representation, decode on laptop"), built
and gated in one day. `top_stream_enc` = stream ingest + the UP5K-proven
serial cis-ROM encoder core (hw/ice40/rtl/encoder.v, REUSED unmodified —
one datapath across both FPGA tracks) + a feeder (packet->point replay
with commit queue) + the on-chip MAP BANK (64 segments x 240 x 2-bit
QPSK codes, quantized as the vector streams out). New protocol types
(STREAM.md v1.1): 0x92 VEC (full int32 vector, the golden-crosscheck
channel, ~10 KB/s of headroom at 5 Hz) and 0x0F/0x93 MAP_READ/MAP_SEG
(60 B compressed segments fetched on demand).

GATES: sim (tb_enc) — 480/480 vector words == golden.encode_int and
60/60 packed map bytes == golden.mcode_from_vec through the FULL packet
path. SILICON (hw_enc.py, 25 REAL spot scans): digests 25/25, on-chip
VSA vectors 25/25 BIT-EXACT vs golden, on-chip map codes 25/25, 85 ms/kf
roundtrip; timing 53.4 MHz vs 50. Synthetic worlds also encode bit-exact
through the same path (webvis classroom arm, 39/39).

Three real bugs found and banked (each with its detection story):
(1) pl_idx SKEW — payload bytes were event-tagged one index late
(registered byte vs live-wire index); killed the HDR n_rings capture, so
nothing encoded; found by sim probe. (2) EBR 2-EDGE READ LAW — two
read-pipeline stages consumed one edge early (feeder point reads; map
readback shifted exactly one component — the shift signature identified
it). (3) COMMIT-MISS UNDER REPLAY — the serial core takes ~840
cycles/point, so a 16-col replay (270 us) outlasts packet arrival
(245 us) and the feeder in WBUSY missed the next commit: on real 32-col
frames HALF THE POINTS silently vanished (silicon |d| scaled exactly as
n/2 x 2x127^2 — the arithmetic fingerprint that localized it). Fix =
one-deep commit queue + ping-pong banks + the sender column cap
(n_cols <= 8 at 2 Mbaud) + an on-chip overflow counter (never silent).
Also: hw_stream.read_pkts now takes a CALLER-OWNED buffer — the local
buffer dropped partially-read packets between calls and silently lost
the first silicon VEC (wire dump proved the chip was sending).

## 2026-07-16 — WEBVIS v2: encode-on-chip demo with BOTH maps, chip-map readout decoded on the laptop, QBE map querying, reset + dataset selection

hw/ecp5/host/webvis.py v2 (live at localhost:8790), all user-directed
features in one loop:
- LIDAR lane: every keyframe digest-verified AND encoded ON CHIP
  (bit-exact counter live); the on-chip map bank's compressed segments
  fetched every 8th kf and DECODED LAPTOP-SIDE into the purple chip-map
  layer (matched-filter image of the 2-bit codes at estimated poses,
  sensor-frame rotation per segment) — the previous demo's map readout,
  now from the ECP5's own memory. Python SLAM (shipped run_cv recipe)
  builds trails/points as before.
- CAMERA lane + QBE: aligned D455 frames -> the exported vision head's
  desc bits (headio, numpy, every 2nd kf in a worker thread) = the
  appearance map anchored on the trajectory. Click a cell in the camera
  panel -> hamming QBE against every stored grid -> match strength
  highlights along the trail + top-k list. Honest framing in-UI: the
  CNN's map query IS appearance QBE (labels demoted 2026-07-16).
- Controls: reset button, dataset selector (spot real / classroom +
  school SYNTHETIC via worlds_dyn.make — synthetic scans stream through
  the SAME silicon encode path, 39/39 bit-exact on the classroom arm).
SELFTEST (headless): 40 kf — digests 40/40, on-chip vectors 40/40
bit-exact, 5 chip segments fetched+decoded, 20 cam desc grids, med err
vs withheld ref 0.010 m. Live: ~55 ms fpga + ~50 ms slam per kf at 5 Hz
real time. Honest status: encode + map STORE are on-chip; the on-chip
TRACKER (localization against the chip map — the iCE40 solo design) is
the next #43 rung, after which SLAM poses also come from silicon.
STREAM.md v1.1 = the full interface spec incl. the ROS-bridge
implementer section (robot side codes against it; hw_stream.py is the
reference sender).

## 2026-07-16 — CLASS COUNT is not a map cost: the bounded vector holds many classes free; "demote to 5" was a framing error (user correction)

User correction: "you only need to create the bottleneck according to spec; the
output vec of encoding could theoretically hold many classes... more is better
(not on lidar — lidar only optimized for tracking acc)." This separates two axes
I had conflated:
  - MAP class-capacity: set by the number of BOUND OBJECTS (positions), NOT the
    codebook size. Role codes are ~orthogonal random keys regardless of K.
  - CNN seg ACCURACY: the only thing that fell at 40-way — and it is a SOFT,
    significance-weighted degradation (low-confidence cells contribute fewer
    bits), not a reason to cap the class count.
Validation (per-class query recall vs codebook size K, FIXED 8 objects/scene,
D=360, 24 seeds):
  K classes:   5     10    20    40    80
  recall     0.969 0.958 1.000 1.000 1.000     map bytes = 45 B (D/8), UNCHANGED in K
Recall is flat-to-IMPROVING as K grows, at fixed map cost — and the improvement
is mechanistic: more classes -> fewer same-class instances per query -> less of
the cross-talk that actually limits recovery (ties to the same-class
multiplicity entry above). The codebook (role storage) is seed-generated on the
FPGA, so K costs no storage either.

REFRAME of the seg demotion: the 5-class SURFACES tier is the HIGH-CONFIDENCE
FLOOR (where the CNN is accurate), NOT a cap on what the map holds. The deploy
recipe binds the FULL vision head output — 40-class (or more) seg-argmax +
32-bit desc — into the same bounded vector, per-cell significance = confidence,
because more classes are strictly richer at ~zero extra map cost and degrade
gracefully. The only real limiter is CNN per-class accuracy (a soft weighting),
not the representation. LIDAR stays tracking-only (no seg head; seg=[], k_bits=0)
— confirmed in the export contract. Anti-oracle: synthetic, GT scores only.

## 2026-07-16 — the VSA object encoder as a PRETRAIN-then-TRUNCATE bottleneck seg net (user architecture): VSA-agnostic seg training yields a bind-ready code

User architecture: the CNN is trained on ORDINARY dense segmentation, fully
VSA-agnostic, with ONE structural constraint — a narrow PER-OUTPUT-PIXEL
bottleneck code followed by a SINGLE FC (a 1x1 conv = per-pixel linear
classifier, NO further convs) that translates each cell's code vector into its
class. At deploy the FC head is CUT and the binarized per-cell code is bound into
the SSP vector at each cell's position; the map's fixed bit-width IS the
bottleneck. Nothing is trained against the binding/query objective (the SSP
encode is a fixed algebraic consumer). New module sspax/vision/bottleneck_seg.py
(BottleneckSegNet; STE binarization: fwd sign, bwd tanh grad).

Trained on NYUv2 (40-class luma, per-pixel bottleneck bits=32, single-FC head,
4000 steps, ~47 s):
  seg pixacc (binarized-code path, non-void)      : 0.338
  cut-code class separability (cosine of +-1 code): same-class 0.381 vs
    diff-class 0.144  ->  AUC 0.723
  UNTRAINED-net CONTROL (rule-4, rules out class imbalance): AUC 0.496 (chance),
    same 0.555 ~= diff 0.553 — the trained separability is genuine class
    structure, not frequent-class (wall/floor) pair inflation.
VERDICT: the architecture WORKS end-to-end. A single LINEAR FC head forces the
per-cell +-1 bottleneck code to be COSINE-separable by class — and cosine IS the
VSA query operation — so a seg-only, VSA-agnostic pretrain produces a bind-ready
descriptor once the head is cut. The code QUALITY (AUC 0.723) is bounded by the
seg ACCURACY (pixacc 0.338 = the known luma-40-class ceiling; RESULTS 2026-07-15
"seg bottleneck is NOT capacity"), NOT by the encoder architecture — more classes
/ a stronger input (RGB-D, at fewer classes) lift both together. Anti-oracle:
NYUv2 GT labels train + score the seg task only; no GT touches the binding.

## 2026-07-16 — bottleneck encoder: rule-4 audit corrections + FULL QAT is free (int8 weights+acts)

AUDIT CORRECTIONS to the bottleneck-encoder entry above (read-only agent, all
reproduced): (1) "VSA-agnostic pretrain PRODUCES a bind-ready code" was an
over-read — the single-linear-head + binary-bottleneck IS the VSA-readiness
inductive bias; the cosine-separability (AUC ~0.70) is a DESIGNED consequence of
constraining the readout to one linear layer, NOT an emergent property (a soft
tautology of linear-classifier training). The untrained control (0.496) proves it
requires training, but does not make it "emergent." (2) per-class-BALANCED AUC =
0.687 (~= pooled 0.700), so frequent-class (wall/floor) inflation is NOT the
driver — separability is broad across classes. (3) clean-code AUC UPPER-BOUNDS
in-map retrieval (superposition crosstalk lowers the real query fidelity) — it is
not the deploy number. (4) low effective bit-usage: untrained same~=diff~=0.55
DC component => effective code entropy < 32 bits. (5) rule-2: mu/sd were computed
over full X (train+test) — FIXED to train-only (bottleneck_seg.py). Corrected
claim: a per-pixel binary bottleneck + single linear head trained on seg CE gives
a +-1 code cosine-separable by the seg classes (pixacc 0.338, AUC ~0.70 balanced,
vs 0.50 untrained) — a valid end-to-end demo that the binary bottleneck can HOST
the linear class boundary, quality bounded by seg accuracy; the cosine-readiness
is designed, and clean AUC upper-bounds in-map retrieval.

FULL QAT (new): int8 weights (per-cout) + int8/uint8 activations (per-tensor,
STE), 1-bit code — the whole trunk trained against its int8 deploy arithmetic
(headio models weight-int8 only; the activation-int8 is the part it does not yet
model). NYUv2 40-class, same seed/data:
  arm        seg pixacc   code AUC
  float        0.340        0.728
  QAT-int8     0.339        0.717     (dpixacc -0.000, dAUC -0.011)
=> full int8 QAT is essentially FREE here — no accuracy cost over float. int8 is
not the limiter; the seg accuracy is. Sets up the arch x quant-level sweep.

## 2026-07-16 — ARCH x QUANT-LEVEL sweep for the FPGA bottleneck encoder (int8/int4/int2/binary + staggered mixed precision + arch variants)

Generalized QAT (per-cout weights, per-tensor activations, STE, at arbitrary
bits; 1-bit VSA code fixed) swept over quant levels AND architectures. NYUv2
40-class, 2500 steps, single seed. pixacc = deployable binary-code path;
W-KB = FPGA weight footprint (sum params*wbits/8). (sspax/vision/bottleneck_seg.py
QNet + sweep().)
  config              pixacc  codeAUC   W-KB   params
  uniform int8         0.325   0.706    18.9   19512
  uniform int4         0.311   0.689    11.1   19512
  uniform int2         0.269   0.563     7.2   19512
  uniform binary       0.250   0.694     5.2   19512
  stagger 8-4-2-2      0.257   0.480     8.4   19512
  stagger 8-4-4-2      0.313   0.668    10.6   19512
  stagger 8-2-2-1      0.258   0.705     7.0   19512
  narrow ch8 int4      0.299   0.670     4.2    6464
  wide ch24 int4       0.326   0.707    21.8   40496
  shallow int4         0.275   0.642     6.1    9240
  deep int4            0.341   0.722    20.1   37976
CAVEAT: single-seed; code-AUC carries ~+-0.03 noise (binary 0.694 > int2 0.563
is within noise), so read pixacc as primary and only COARSE orderings as robust.
All numbers sit in the luma-40-class band (0.25-0.34) — the sweep ranks FOOTPRINT
at fixed task difficulty, NOT the accuracy ceiling.

ACTIONABLE FPGA RULES (coarse, robust):
1. int4 weights+acts is NEAR-FREE (0.311 vs int8 0.325) at ~60% the bytes ->
   the DEFAULT deploy precision. int2 is the CLIFF (0.269); binary lower on
   pixacc (0.250). Below int4, accuracy degrades.
2. MIXED PRECISION has a rule: the dense-3x3 (main receptive-field) layer is
   PRECISION-SENSITIVE — keep it >= int4; the final 1x1 TAIL tolerates int2/
   binary. stagger 8-4-4-2 = int4-quality at 10.6 KB; 8-4-2-2 (int2 in the RF
   layer) COLLAPSES (AUC 0.480, below chance). So stagger DOWN toward the head,
   never through the RF layer.
3. ARCHITECTURE: DEPTH/receptive-field helps most (deep int4 0.341 — best of
   the sweep, beats int8-base); width helps modestly (wide 0.326); dropping the
   dense 3x3 HURTS (shallow 0.275 — RF matters). narrow ch8 int4 is the
   PARETO-EFFICIENCY corner: 0.299 at 4.2 KB / 6.5K params (~92% of int8-base
   accuracy at ~22% of the bytes).
PARETO: tiny = narrow-ch8-int4 (0.299 @ 4.2 KB); balanced = int4 / stagger-8-4-4-2
(~0.31 @ ~11 KB); max = deep-int4 (0.341 @ 20 KB). Anti-oracle: NYUv2 GT trains/
scores the seg task only; no GT touches the code binarization or binding.

## 2026-07-16 — multi-seed (3) firming of the arch x quant Pareto corners (+ correction to a single-seed over-read)

Firming the single-seed sweep above. NYUv2 40-class, 2500 steps, seeds {0,1,2}
(seed varies BOTH init and batch order), pixacc mean +- std:
  config             pixacc          W-KB
  deep int4          0.345 +- 0.012  20.1
  uniform int8       0.337 +- 0.009  18.9
  uniform int4       0.328 +- 0.014  11.1
  narrow ch8 int4    0.307 +- 0.012   4.2
  uniform int2       0.234 +- 0.052   7.2
ROBUST (non-overlapping by >1 std):
  - int4 is NEAR-FREE: 0.328 overlaps int8 0.337 — confirmed at ~60% the bytes.
  - int2 is a genuine CLIFF: 0.234, robustly below every other corner AND
    unstable (+-0.052 — a seed dipped to 0.160). Do not deploy int2 uniform.
  - narrow ch8 int4 is a robust efficient corner: 0.307 @ 4.2 KB (~0.03 below
    int8 for 4.5x fewer bytes / 6.5K params).
CORRECTION to the single-seed sweep: "deep int4 0.341 is the BEST, beats
int8-base" was PARTLY NOISE — over 3 seeds deep (0.345 +- 0.012) OVERLAPS int8
(0.337 +- 0.009), so DEPTH is NOT a robust lever at this scale; deep ~= int8 ~=
int4 (all ~0.33-0.34, the luma-40 ceiling). The robust levers are only:
int4-is-free, int2-is-a-cliff, and narrow-is-Pareto-efficient. Single-seed
fine orderings (~+-0.02) were noise, as flagged; the coarse story holds.

## 2026-07-16 — deploy-side review of 676fdc6 + 2a71c5f..4cc78de: all accepted; the bottleneck code exports through the EXISTING desc slot; ship-recipe update queued

Five commits reviewed (ECP5 unplugged today — no hardware gates; review
is code/ledger-level). Verdicts:

- SAME-CLASS MULTIPLICITY — accepted, exemplary diagnostic: the K~6
  recall cliff is the fixed 4-MAD detector crossing a gracefully
  degrading z-score (~1/sqrt(K); peaks preserved to K=16), not peak
  merging. Deploy read: dense same-class scenes are a DETECTOR-tuning
  problem (matched thresholds / candidate-position readout).
- CLASS COUNT IS NOT A MAP COST (user correction) — accepted, and it
  usefully REVISES the demotion narrative: the label-head demotion was
  always about CNN accuracy, never map capacity (recall flat-to-better
  at 5->80 classes, fixed 45 B; mechanistic via fewer same-class
  collisions). The ship recipe becomes: surfaces-5 = high-confidence
  FLOOR; bind the FULL class output soft (significance = confidence) +
  desc bits; lidar stays tracking-only. Their msg-r5 P0 (surfaces_tier
  with real bits) should build THIS form — queued in msg round 6.
- BOTTLENECK SEG NET (user architecture) — accepted with their audit's
  honest frame (cosine-separability is DESIGNED by the single-linear-
  head constraint; clean AUC upper-bounds in-map retrieval; a rule-2
  mu/sd split leak self-caught and fixed). KEY CONVERGENCE (deploy
  side): the truncated bottleneck code is bits = sign(act) per cell —
  EXACTLY headio v2's `desc` stack semantics. The trained bottleneck
  net exports through the EXISTING contract with zero changes (desc =
  the code head; seg=[] stays legal). The separate desc+seg-argmax
  pair collapses into one code head whose bits are simultaneously
  class-separable and bindable.
- FULL QAT FREE + ARCH x QUANT PARETO (multi-seed-firmed) — accepted:
  int4 weights+acts near-free (robust), int2 a genuine unstable cliff,
  narrow-ch8-int4 the Pareto corner (0.307 @ 4.2 KB), mixed precision
  staggers DOWN toward the head but never through the receptive-field
  layer, and the "deep beats int8" single-seed over-read was corrected
  to noise. CONTRACT NOTE: their QAT trains against int8 ACTIVATIONS,
  which headio's numpy forward does not model (float acts — the safe
  direction for gating, deploy-more-precise-than-trained). When the
  CNN moves on-chip, headio needs an activation-quant mode for
  bit-faithful golden parity — filed as the headio v2.1 item.

## 2026-07-16 — WEBVIS v2.2: the camera VSA OBJECT MAP with class-select -> laptop UNBIND -> objects on the map (the full query loop, live)

User-directed: "select class, unbind from the cam VSA vec, see objects
on the map; unbinding on laptop; FPGA does SLAM + map integration and
transfers the map vec, laptop decodes." Implemented as the laptop-side
prototype of EXACTLY that decode path (the cam band moves on-chip when
the camera hangs off the FPGA; the lidar band already runs the full
chip path, gated bit-exact):

- CamMap (webvis.py): per keyframe, the exported head's 32-bit cell
  codes are ONLINE-CLUSTERED (leader clustering, hamming<=8, cap 12 —
  honest post-demotion classes = APPEARANCE CLUSTERS; the bottleneck
  artifact's real class bits drop in with zero glue) and BOUND at
  world positions lifted from the LIDAR: cell bearing (D455 HFOV 69
  deg, NOMINAL yaw-aligned extrinsics — demo-grade, labeled in-code)
  -> ring-33 range at that bearing -> world point. Capacity law
  respected: cells aggregate to per-cluster centroids, 3-6 bindings
  per keyframe, ONE bounded D=240 vector per kf on W_MAIN (the
  contract band; spatter keys = headio._head_keys — the SAME
  conventions a chip-built cam band will use).
- Class query: UI chips (count + example thumbnail per cluster) ->
  POST /query_class -> UNBIND (conj-amplitude elementwise product) on
  the laptop against every kf vector -> matched-filter density on the
  world grid -> orange objects layer + peak markers on the map,
  alongside the lidar map layers.
- SELFTEST (laptop-lanes mode, board unplugged): 60 kf -> 30 kf
  vectors, 12 appearance classes, class-0 unbind -> 5 object marks
  (peak-normalized), SLAM med err 0.016 m. Graceful no-FPGA fallback
  added (board-out days keep the demo alive; FPGA panel shows 0s
  honestly). Anti-oracle: est poses/lidar only; no reference enters
  any estimate; extrinsics assumption stated.

msg round 6b sends the requested arch menu (dw-separable RF, dilated
pyramid, scene-context branch, bit decorrelation for the <32-bit
effective entropy, free confidence head) + dataset ladder (SUN RGB-D
first — 7x labeled data, supersets NYU, dedup required; Hypersim
pretrain arm; self-sup on OUR platform imagery; ScanNet last).

## 2026-07-16 — WEBVIS v2.3: real school runs in the selector (synth dropped, user directive); the map query system rebuilt — calibrated z-scores, structured region queries, reverse readout

Dataset selector is REAL-only now: spot (classroom) + school_run1 +
school_run2, built from each run's scans.npz (ring-33 1024-beam slices,
stride-4 keyframes) with the honest reference semantics per venue —
run1 est-only (ghost hidden, gt_ok all-False), run2 shows the ghost
only inside its gated-LIO window (342/836 kf). Both school runs get the
FULL experience incl. the cam VSA map (rgb_d455 shards align by
timestamp). Synthetic worlds removed.

QUERY SYSTEM v2 (user: "come up with a better way to query map") —
three upgrades, one shared calibrated sweep:
1. CALIBRATED SCORES: every query density is z-scored against a
   seeded RANDOM-CODE control sweep (the semantic-thread's decisive-
   control discipline, now in the live loop); marks require z > 4.
   A garbage query now answers NO MATCH instead of a normalized fake
   peak — measured live: a diffuse cluster query on a 12-grid map
   reads z_max 3.8 -> 0 marks.
2. STRUCTURED REGION QUERIES (the objmap patch form): drag a box on
   the camera panel -> q = sum_c A(bits_c) * exp(iW.(p_c - x0)) over
   the cells' lidar-lifted positions — object-shaped, position-
   structured matching. Control shares the SPATIAL structure (random
   codes at the same offsets) so the envelope calibrates away.
   Measured: on the same thin school_run2 map where the centroid query
   fails, the patch query clears the bar (z 4.3, 2 marks) — structure
   adds real discrimination.
3. REVERSE READOUT: click the MAP -> project the local readout onto
   the spatter keys -> recover the code bits at that spot -> nearest
   class + hamming + support count, then auto-highlight its other
   locations. Query both directions: "where is X?" and "what is here?".
Single-cell QBE click retained. All laptop-side on the same per-kf
bounded vectors (the chip-band decode path unchanged). Anti-oracle:
est poses + lidar only; nominal yaw extrinsics stated.

## 2026-07-16 — webvis fix: school datasets hung on load — npz members decompress the WHOLE array on every access

"Doesn't seem to work" diagnosis: selecting a school run froze the demo.
load_bundle indexed `z["ranges"][i]` inside the keyframe loop — a lazy
npz member decompresses the FULL (8298, 1024) array on EVERY access, so
~2000 keyframes took minutes while the run loop (and every SSE client)
blocked. Materialize once -> school_run1 loads 2075 kf in 0.1 s. Repo
gotcha worth remembering anywhere scans.npz is consumed. Also surfaced:
school_run1 has no rgb_d455 shards on this box -> its cam lane degrades
gracefully to lidar-only (run2 + spot keep the full camera/QBE lanes).
All three datasets verified switching live at rate.

## 2026-07-16 — TARGET-OBJECT map recall OPTIMIZED on synthetic data (tune_recall.py): recall 0.94 -> 1.00, robust to 25% bit flips; boost knob shipped to the UI

User directive: optimize map recall of certain objects with synthetic
data. hw/ecp5/host/tune_recall.py drives the REAL CamMap code (webvis)
with synthetic scenes — 8 classes x 3 objects, wandering 69-deg-FOV
trajectory, per-observation BIT-FLIP noise (matching the measured head
stability) + position noise (bearing-cell + nominal extrinsics), GT
scores only. Metric fixed early: recall over OBSERVED objects (the
first pass counted never-seen objects — an observability ceiling
masquerading as map failure; recall jumped 0.33 -> 0.94 base once
corrected).

Knob ladder (8 seeds, bit-flip 0.15, sig_p 0.20 m; targets = 2 classes):
  base (centroid, max-fuse, z4)      recall 0.940  prec 0.245
  max + instance-split + boost x2    recall 1.000  prec 0.336
  max + instance-split + boost x3    recall 1.000  prec 0.410
  sum-fusion arms                    recall ~0.50  (HARMFUL)
Winner robustness (max+split+boost x3): recall 1.000/1.000/0.976/0.952
at (bit,pos) = (.10,.15)/(.25,.20)/(.15,.35)/(.25,.35); control with
boost OFF: 0.929. VERDICTS:
- SIGNIFICANCE BOOST works live: binding target classes at 2-3x
  amplitude lifts recall to 1.0 AND precision (+0.09..+0.17) — the
  boosted peaks clear the background-class cross-talk (the
  significance-buys-capacity law, now in the deploy query chain).
- INSTANCE SPLIT (same-class obs >1.2 m apart bind separately) replaces
  the blind per-class centroid, which averaged multi-instance classes
  into a phantom between them (multiplicity law, live).
- SUM-FUSION REFUTED — my own prior, committed as the default an hour
  earlier, reversed by the sweep: summing kf vectors integrates
  cross-talk faster than signal (recall halves); MAX-fusion (best
  single view) stands. Support-count gating buys nothing on top.
  Default reverted same-session; the support image stays display-only.
SHIPPED to webvis: FUSE=max, instance-split in ingest, boost dict at
bind time, UI = shift-click a class chip to star/boost it x3 (applies
to future bindings). Marks now report raw z (selftest: z 4.6 marks on
20-kf spot map). Anti-oracle: synthetic GT scores only; the harness
imports the code under test (no reimplementation drift).
