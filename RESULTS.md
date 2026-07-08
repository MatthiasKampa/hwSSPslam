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
odometry 189.3 — a 3.3x reduction on a genuinely unseen log at 12 ms/kf and
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

*under shared-CPU load. **high seed variance (1.26-2.74 across seeds).
Honest verdict: RBPF is GMapping-class on the revisit-dense logs (sub-15 cm,
near the eval's timestamp-slop floor) — a 20x accuracy gap to ours there; ICP
beats ours everywhere at 2-4x our memory with full scan retention; CSM beats
ours on fr079/ACES and loses on Intel rmse (wins median). Ours holds the
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
or false. The gain R5 books is not accuracy (it matches R4 bit-for-bit) but
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
