# FINDINGS — Bounded-Memory VSA/SSP 2D-Lidar SLAM

*A distilled synthesis of the research in this repository. The chronological
lab notebook — every run, retraction, and independent audit — is
[`RESULTS.md`](RESULTS.md); section names cited below (e.g. "R3 refinement",
"Tikhonov prior sweep") point there for depth. This document reports only the
CURRENT / shipped numbers; RESULTS.md keeps the superseded ones as
stratigraphy. Read time: ~15 minutes.*

---

## 1. Thesis

We built a bounded-memory 2D-lidar SLAM stack whose map is a fixed-size complex
Fourier-feature vector (a Spatial Semantic Pointer / VSA hypervector on a polar
frequency lattice) per rigid 5-keyframe trajectory segment — **no sensor
history is stored anywhere**. The algebra carries the load: translation of map
content is elementwise phase multiplication, rotation is an *exact* index
permutation of the lattice plus a stored d/dθ derivative vector for sub-grid
angles, matching is an inner product, and graph corrections are O(D) phase ops
on frozen content that is never re-encoded. The honest contribution is a
*representation*, not an accuracy record: the map is **O(area)-bounded (not
O(time)), history-free, and algebraically transformable, at a usable accuracy**.
It is explicitly **not** state-of-the-art accurate and **not** bytes/m²-compact
(per m² the SSP map is actually 3–12× *denser* than an occupancy grid; the
memory win holds only against history-storing baselines like ICP's retained
scans or RBPF's particle×grid). The system pays off in proportion to
dead-reckoning drift and revisit density, and degrades gracefully (locally
rigid, globally bent) rather than diverging when revisits are sparse.

### Shipped numbers

ATE rmse vs RBPF-corrected references; `ssp_bounded_carmen.py <log>`,
deterministic, reproduced bit-exact by an independent auditor.

| log | ours | raw odometry | map memory | speed |
|---|---|---|---|---|
| Intel Research Lab (full) | **2.440 m** (med 1.553) | 24.2 m | 5–8 MB | 17 ms/kf |
| Freiburg 079 (zero-shot) | 5.523 m | 14.35 m | 3.5 MB | 30 ms/kf |
| ACES3 Austin (zero-shot) | 6.212 m | **5.41 m** | 5.3 MB | 13 ms/kf |
| MIT Infinite Corridor, 1.9 km (held-out) | 38–58 m | 189.3 m | 27 MB | 12–21 ms/kf |

In-repo reference baselines (identical parsing / keyframing / eval,
`baseline_*.py`; see "Reference baselines"): ICP + pose graph **1.70 m** (15–35
MB, retains all scans), correlative grids 3.27 m, RBPF/GMapping-lite **0.12 m**
(39–56 MB peak, particle grids). The positioning is deliberate: RBPF is a ~20×
accuracy gap on revisit-dense logs; ICP beats us everywhere at 2–4× our memory
with full scan retention. We hold the smallest, boundedly-sized, history-free
state. **ACES is a frank negative** (its odometry is already excellent and only
17 closures fired on one large sparse loop). **MIT is the honest held-out test**
— 4–6× longer than any tuning log, zero code/parameter changes, a 3.3× drift
reduction that demonstrates the revisit-density limit at full scale.

> **Transfer caveat (referee).** Intel/fr079/ACES are no longer held out: the
> shipped config was selected by minimizing the worst-of-three ATE ratio. MIT
> is the only genuinely unseen log. The drift-proportional-payoff pattern held
> there in relative form exactly as the ledger predicted.

---

## 2. The encoder result

The encoder study (`ssp_slam.py`; "Headline table", "Findings", "World
dependence", "Six-world battery") set the frequency lattice used by everything
downstream. Scans become uniformly-spaced samples along hit-to-hit segments
(an occlusion filter kills radial phantom-wall bridges across depth
discontinuities), each sample encoded as φ(p) = exp(iWp); registration is a
coarse-to-fine search of Re⟨Map, exp(iWd)·Scanθ⟩. The finding is a coherent
story about *matching a deterministic lattice to the environment's angular
spectrum*:

- **Structure beats randomness at small D, for two separable reasons.** Matching
  the spectral band alone (banded-random control) buys ~2.4× over Gaussian RFF;
  making the grid *deterministic* buys another ~10× on top in flat-wall worlds
  where grid angles hit wall normals exactly. Net: a D=48–96 polar-lattice
  encoder reaches RFF-2048 accuracy at 1/20th the dimensions and ~25× the speed.

- **The ladder is octaves.** λ = 0.25/0.5/1/2 m is as good as 6–72 tuned
  geometric scales, because the constant-velocity/odometry prediction keeps the
  search window below the coarsest wavelength (~±0.36 m); big wavelengths buy
  nothing *inside* the matched band. λ_min tracks the noise scale; gaps wider
  than ~2 octaves reintroduce phase-unwrapping sidelobe ambiguity.

- **The golden-ratio catastrophe** ("Findings" #7; falsification history in
  RESULTS). Swept ring-ratios show a flat safe plateau r≈1.4–2.8 with one sharp
  exception: the golden ratio φ is *uniquely* bad (12.1 cm on all 6 seeds vs
  0.9–2.2 for its neighbors), and its Fibonacci convergent 8/5 fails
  catastrophically-but-stochastically (one seed 50 m: exact rational
  coincidence → occasional false lock). The mechanism was chased and corrected
  twice. It is **not** the collinear three-wave story (angular dithering does
  not rescue it — falsified) but a **magnitude-only additive resonance**: φ²=φ+1
  makes each frequency the exact sum of the two below, so |wᵢ|−|wⱼ|=|wₘ|
  difference-beats live *inside* ring k's own radial kernel, angle-independent
  and undithererable. It is also **environment-coupled**: in curved/blob worlds
  the extra scale density of the golden ladder *wins* (1.17 vs 2.83 cm); flat
  walls expose the radial beats along wall normals. sqrt(e)=1.6487 (2% from φ)
  is fine — the culprit is the additive identity, not the neighborhood. The
  "most irrational is best" intuition that makes golden *angles* good is
  backwards for *scales*. The reconciliation (SotA/golden_dithering.md):
  Schretter–Kobbelt's golden sampling is *additive* (Weyl mod 1), where φ's
  continued fraction is a virtue; used *multiplicatively* as a ladder ratio the
  same identity becomes additive closure — same constant, opposite side of the
  exponential map. No registration/interferometry source warns against φ
  ladders; the design rule appears publishable.

- **≥6 well-spread axes for a point-like peak; angle density is
  environment-coupled** ("Findings" #3, #6 — the one real fragility). Three
  axes give a 3-ridge kernel (heading superb, translation slides >1 m along
  ridges). A long straight wall only excites frequencies within ~λ/L of its
  normal, so exact grids win when a grid angle *hits* a normal (axis-aligned
  world: any even angle count works down to 4 angles; odd counts miss 90° and
  lose an entire translation axis) and lose by luck-of-alignment otherwise. In a
  37°-rotated world, 24 angles (7.5° spacing) is the knee. Curved worlds spread
  energy over all angles, so any well-spread dense set works. Staggering angles
  across scales to densify the union *breaks the grid symmetry and makes things
  worse*; area-uniform frequency sampling also loses to the plain uniform grid.
  **Shipped choice: 4 octaves × 60 angles (D=240) matched band + 2 incommensurate
  relocalization rings (5.3/12.8 m)** → D=360 total; random RFF remains the
  flat-response fallback when environment orientation is unknown.

---

## 3. The architecture

The deliverable is `ssp_bounded.py` (continual, bounded); `ssp_hier.py`'s HY4
split is the shipped map refinement.

- **Rigid anchor-frame segment vectors + d/dθ derivative (novel in mechanism).**
  Each keyframe folds *at frame time* — points still in hand — into its
  5-keyframe segment's vector, encoded rigidly in the anchor's frame, together
  with the vector's d/dθ (rotation derivative about the anchor origin).
  Query-time anchor rotation = nearest-3° lattice permutation + δ·derivative:
  an analytic **first-order Lie correction** of the sub-grid residual, not
  matcher slack. The ablation ("Derivative-vector novelty ablation") confirms it
  behaves *exactly* as a genuine first-order term — 2.8× reconstruction gain at
  δ=0.5°, tapering to 1.1× at the 1.5° lattice edge, turning *harmful* beyond ~2°
  (accurate near the expansion point, wrong far from it). The operating range
  [0, 1.5°] sits entirely in its beneficial regime; removing it costs ~4× on the
  bench. Nothing is ever re-encoded, so between relaxations the representation
  cannot drift.

- **Translation = phase, rotation = exact permutation.** Translation by d is
  elementwise ×exp(iWd); rotation by a grid step m·π/60 is an exact circular
  shift of the angle index (conjugate on sector wrap). This is what makes graph
  corrections O(D) phase ops and coarse rotation candidates free lattice
  permutations (one encode per match instead of ~22).

- **Pass-segregated, gated, robustified backend.** An anchor pose graph;
  sequential edges quadratic at the frontend's true 5-frame accuracy (0.03 m /
  0.3°, inflated to 0.10 m / 1.5° over odometry-fallback spans). The frontend
  matches only *recent* segments; old passes act exclusively through loop
  constraints that are pass-segregated (one contiguous chain at a time),
  lever-inflated in σ, drift-scaled in innovation gating, Cauchy-IRLS
  reweighted (loops only), and leave-one-out chi-square pruned. Injected-outlier
  studies: 10% i.i.d. wrong closures absorbed by either protection layer alone;
  correlated large aliases (1.05 m) rejected 100% pre-insertion — the unprobed
  hard case is *small* correlated aliases (0.2–0.35 m, repetitive-bay scale).

- **Early stopping IS the regularizer (Hanke semiconvergence).** The shipped
  solver is analytic-Jacobian trust-region TRF with **max_nfev=30 as
  load-bearing early stopping** — iterative regularization / semiconvergence
  (Hanke 1997). This was hard-won and thrice-corrected: "GN is a strict
  upgrade", "the soft veto costs accuracy", "TRF step control fixes flat
  valleys" were all *artifacts*. A read-only bisection showed fully-converged
  solvers slide anchors up to 2.7 m along flat-valley directions the data does
  not constrain (fr079 12.7 m at 300 evals); the shipped solver essentially
  never converges (68/76 Intel solves hit the cap) and that is *why* it holds.
  The designed principled successor — an explicit Tikhonov prior toward
  pre-relax anchors — was **built, swept, and rejected** ("Tikhonov prior
  sweep"): every λ converges cleanly yet loses (worst-of-three ratio ≥3.39 vs
  1.97 shipped), because a static quadratic prior stiffens each direction
  independently while 30-eval truncation limits the *total* correction path
  path-dependently, applying data-consistent step components first and quitting
  before the sloppy ones. The prior ships default-off; **every future
  acceptance gate must be multi-log**.

- **complex64 is a free 2×** ("Memory levers"). Segment storage at 8 B/component
  gives round-trip error ~2.4e-8 and a match-peak shift of 0.000 mm / 0.00
  millideg; Intel/fr079/ACES ATE bit-identical. It is opt-in (default off on
  chaos-sensitive long runs like MIT, where a 0.26 mm perturbation moves the
  result 12 m). Net map story: HY4 matched-band split (0.67×) × complex64
  (0.5×) ≈ 0.33× of the original 8 MB at bit-identical Intel accuracy (~2.6 MB).

---

## 4. The three architectural laws

Each was learned from a *measured* failure, not designed in.

**Law R1 — the loop/matched band must live in graph-consistent memory.**
World-frame decay is never slow enough. Tiered world-frame region maps
("Hierarchical multi-scale maps") that put the loop band in fast-decaying
regional cells make loop matches hit the current pass's own last-35-kf writes
and *confirm* the drifted estimate — confident wrong edges (Z-error p90 2.15 m
vs 0.25 m) that every gate is blind to (coherence high, innovation ~0). Rigid
anchor-relative segments keep map geometry exactly consistent with their anchor
by construction; that invariant is what the hierarchy gives up (net: ~37× less
map memory for 1.5–2.8× worse ATE, never beating the control).

**Law R2 — independent recency/gap knobs; single-burst drift-consistent
content.** The frontend recency window and the loop-candidate age gap are
*independent* knobs: keep recency **small** and gap **large** (a GT-grounded
bench isolates the mechanism). Wide recency lets the frontend snap onto drifted
geometry and bends the sequential chain until honest closures get
innovation-gated; a large gap forces sub-lap closures onto high-innovation old
geometry — and accepting sub-lap closures retro-corrupts even early anchors
(lap-1 RMS 0.45 vs 0.13 without). The dead band between the two knobs is benign.
The **scale-arrays study** ("Spatially-anchored per-scale submap arrays") sharpens
R2 into a content law: a primitive used for *closure* must hold
**drift-consistent single-burst content**. Persistent per-area cells sum
observations made at different drift states into one smeared, broad correlation
peak; the biased Z passes every gate (a smeared match is broad but still
correlated) and drags the graph off the good frontend trajectory. Frontend-only
SA is a fine 4.34 m on Intel; enabling closures *worsens* it to 11–15 m —
graph-anchoring fixes the frame's drift-riding but not the intra-cell write-time
smear. Ephemeral 5-kf segments guarantee single-burst content; persistent cells
violate it by construction. HY4 stands as the memory-efficient matched-band
representation.

**Law R3 — map content must be viewpoint-neutral** ("Encode-side gating study").
Directional encode weighting (soft von-Mises rolloff + anisotropic
per-frequency weights) wins the same-heading bench decisively (corridor false
edges 5.2→0.8) and **regresses the real logs 2.2–3.3×** (Intel 2.440→7.98),
because directional weights bake the writing pass's ray geometry into rigid map
content that later passes must match from *other* headings. The matched band
must be graph-consistent (R1) AND viewpoint-neutral.

---

## 5. The corridor information-limit (the centerpiece)

Five independent lines of attack on the MIT corridor's residual error all
converge on one conclusion: **place-recognition information is starved at the
coarse relocalization wavelengths (λ 5.3/12.8 m), and no verification,
consensus, belief, or multi-scale trick recovers it** — the corridor blur
washes out door/alcove-scale detail, leaving retrieval that works only at
distinctive junctions.

1. **Drought relocalization** ("R3 refinement"). A map-anchored coarse search
   against the global relocalization vector, fired only after a genuine-closure
   drought, breaks *engineered* droughts on the bench cleanly (329→5 cm, 4/4
   seeds) and re-anchors the MIT run at two distinctive junctions (45.3→38.0 m,
   final-third closures 15→27). But it never breaks the deep final-hour corridor
   drought. The lesson: *aggressiveness is the enemy* — every harmful snap came
   from faster pairing in self-similar corridor stretches, and a wrong
   pre-aligned building-scale snap is unrecoverable. The 25-kf cadence is the
   load-bearing independence guarantee.

2. **The numeric capacity/aliasing study** ("Aliasing and capacity"). Diagnoses
   *why*: under MIT drift the matched band is **dead** (peak attenuated to
   0.03–0.09 of clean); only the relo band survives, and its corridor recall@40
   plateaus at 1.00 at 125 m → 0.50 at 1 km → 0.25 at 1.9 km. MIT (2402 places)
   sits ~6× over the single-vector 75% capacity. Sharding the coarse vector by
   travel distance is the capacity fix; per-ring whitening + NMS remove the
   red-dominance alias trap.

3. **R4 (whitening + NMS + sharded store)** ("R4 refinement"). Lands the study's
   recommendations cleanly (bench 4/4, Intel/fr079 bit-identical, +4–15 KB) — but
   the study's projected recall 0.25→0.75 **does not transfer**: in-pipeline
   corridor recall moves only 0.00→0.04. All 104 R4 failures at true revisits are
   *wrong-peak* (confident z≥3 hypotheses at the wrong corridor twin), zero
   sub-gate misses. Sharper retrieval converts misses into *consistent wrong
   snaps* — the one wrong pair agreed **tighter** (0.54 m / 0.6°) than genuine
   junction snaps (0.72 m / 2.6°). Only a search-breadth escalation (demand a
   third fire when the sweep exceeded 6000 cells) stops it.

4. **R5 (PCM consensus admission)** ("R5 refinement"). Replaces R4's three
   heuristics with Mangelson et al.'s Pairwise-Consistent-Measurement-Set
   maximization (SE(2) cycle-consistency graph + max clique), reproducing every
   shipped number bit-identically. Its verdict is the sharp one: **there is no
   consensus to find**. Retrieval scores **0/106 stage-1 hits** at deep true
   revisits, so no true clique can exist; and the deep-corridor wrong fires are
   mostly *pairwise-inconsistent scatter* (clique_max=1 even with 3–6 co-located
   candidates) with only sporadic 28-kf twin pairs the consensus-size rule holds.
   The limit is **not** a correlated-alias family fooling the backend (the case
   consensus clustering is built to catch) — it is upstream information
   starvation no backend consensus can undo, true or false.

5. **Scale-cascade** ("Iterative slow-to-fast scale-cascade") + **belief
   frontend** ("Belief-carrying frontend") close the pincer. The cascade is a
   materially better twin *discriminator* on labelled synthetic corridors (AUC
   +0.11, false edges halved at equal recall) but inherits the coarse rings'
   place-recognition limit: it can *reject* a twin the coarse ring locked onto
   (precision) but cannot *recall* a closure the coarse ring missed. The belief
   frontend shows the aperture worlds have the **tightest** frontend surfaces
   (corridor posterior spread p90 8.8 cm, 0.0% multimodal — tighter than the
   well-conditioned room): the drift that makes closures necessary is *not*
   represented as frontend ambiguity (the frontend is confidently-and-
   consistently slightly-wrong), so carrying belief regresses every log wherever
   it fires.

**Synthesis:** retrieval was never the MIT bottleneck; verification *information*
is, and it is absent at the coarse wavelengths. A finer relocalization band
cannot be world-frame-summed (Law R1), so within this architecture the
deep-corridor residual is a genuine **place-recognition information limit**, not
a tuning gap.

---

## 6. The negatives (honest table)

Each was designed, built, tested, and rejected. Fuller accounts under the cited
RESULTS section.

| idea | what it was | why rejected |
|---|---|---|
| Hierarchical world-frame tiers | tiered region maps replacing rigid segments (37× less map) | Law R1: loop band in world-frame decay confirms drift; 1.5–2.8× worse ATE, never wins ("Hierarchical multi-scale maps") |
| Scale-arrays (O(area) submaps) | persistent per-scale graph-anchored cells | good frontend, catastrophic closure: intra-cell write-time smear biases every loop Z (4.34→11–15 m Intel) ("Spatially-anchored per-scale submap arrays") |
| Spin exclusion (CSM transfer) | don't fold rotation-triggered keyframes | Intel 2.440→8.716 m: the VSA frontend map *is* the recent folds; starving 37% breaks matching ("R2 refinement" #4) |
| Linear loss (CSM transfer) | drop IRLS, keep LOO + pruning | Intel 2.440→4.320 m: without Cauchy downweighting, surviving mid-weight wrong edges pull the graph ("R2 refinement" #5) |
| Tikhonov prior | explicit quadratic prior toward pre-relax poses | loses to accidental early-stopping on every λ (ratio ≥3.39 vs 1.97); static prior can't reproduce path-dependent truncation ("Tikhonov prior sweep") |
| Session-relative profile veto | cascade fine/coarse coherence ratio as a general veto | catastrophic over-veto; the discriminative *direction* is geometry-specific (bench false closures have the opposite profile to corridor twins) ("Session-relative profile veto … REJECTED") |
| Belief frontend | carried pose belief on the matcher's coarse surface | aperture surfaces are tightest; no ambiguity to catch, firing regresses every log ("Belief-carrying frontend") |
| Heterodyne virtual rings | synthesize coarse rings vᵢ⊗conj(vⱼ) instead of storing | algebra exact but 1.5–3.6× noisier and never cheaper: free differences never reach the relo band; on-band pairs cost full price ("VSA binding experiments" #2) |
| Nonlinear per-ring fusion | product/geomean/min of per-ring coherences | golden hazard is a *conjunctive* alias, not per-ring-separable; fusion rescues nothing and lets one weak ring veto the true peak ("VSA binding experiments" #1) |

---

## 7. Novelty & related work

Claims corrected after a deep literature scout (SotA/vsa_ssp_theory.md,
SotA/spectral_registration.md; "Related work" in RESULTS).

- **Rotation-as-permutation — partially anticipated.** Krausse/Neftci/Sommer/
  Renner (NICE 2025, arXiv:2503.08608) use approximate rotation-as-permutation
  in a grid-cell VSA (cognitive maps only). Ours remains distinct: an *exact*
  group-closed lattice applied to raw lidar registration.

- **The d/dθ derivative vector — novel in mechanism.** They solve sub-grid
  rotation by bump-vector convolution; we use an analytic first-order Lie
  correction. The ablation (§3) is the clean proof; a head-to-head against a
  120-angle-no-derivative lattice at equal storage is the open comparison.

- **The golden-ratio-ladder rule — apparently new to registration.** The
  phenomenon is known in grid-cell theory (Vago & Ujfalussy 2018) and its
  mechanism class in turbulence ("Fibonacci Turbulence" PRX 2021, which *builds*
  shell models on kₙ₊₁=kₙ+kₙ₋₁ precisely to maximize three-wave resonance), but
  no registration/interferometry source warns against φ scale-ladders.

- **Residue-HDC relocalization rings.** The incommensurate 5.3/12.8 m relo rings
  are a residue-number-system trick (Kymn et al., Residue HDC, 2025); the octave
  ladder is classic multi-wavelength phase unwrapping and the ">2-octave gaps
  reintroduce sidelobes" finding is the standard unwrapping ambiguity condition.

- **Nearest prior systems.** The closest "SSP SLAM" (Dumont/Furlong/Orchard/
  Eliasmith, Frontiers 2023) does *landmark*-level SLAM with given data
  association; VSA-OGM (npj 2026) assumes given poses. Dense scan-to-map
  registration in SSP space appears to be new. Highest-leverage unexploited
  borrowings: Frady/Kleyko/Sommer FHRR capacity theory (s=√(N/M)) to *derive*
  window/cell-cap sizes instead of hand-tuning; Olson-style correlation-peak
  curvature as edge information matrices.

---

## 8. Limits, honest framing, and open questions

**What this is not.** Not SotA-accurate (RBPF is ~20× better on revisit-dense
logs; ICP beats us everywhere). Not bytes/m²-compact — the SSP map is 3–12×
*denser* per m² than a 5 cm occupancy grid; the memory win holds only against
history-storing baselines. Trajectory bookkeeping still grows O(time) (as any
trajectory output must); full-graph relaxation is the one remaining O(t) compute
term, with windowed relax + seq-edge marginalization the designed-but-optional
fix (merged, verified, opt-in via `windowed=True`).

**What is defensible.** The three properties are the bound, the absence of
history, and the algebra — each verified: Intel plateaus at 698 segments (84% of
the cell-cap ceiling); MIT grows dead-linear at 1.26 seg/m as new corridor is
exposed; the map is transformed only by phase ops and permutations, never
re-encoded. ACES is a frank negative; MIT demonstrates the revisit-density limit
at scale rather than hiding it.

**Open questions.**

1. **Derivative vs. angles.** Does 60-angle+derivative or
   120-angle-no-derivative match better at equal storage? They differ in compute
   (120 angles doubles matcher cost; the derivative only adds storage), so it is
   a tradeoff, not a free swap. Needs a lattice rebuild — flagged, not chased.

2. **A richer descriptor for corridor place recognition.** The information limit
   (§5) is at the coarse wavelengths; a finer relocalization band cannot be
   world-frame-summed under Law R1. What kind of graph-consistent,
   viewpoint-neutral, single-burst descriptor could carry enough
   place-recognition information along corridors — without reintroducing the
   O(time) growth that ephemeral segments were meant to avoid — is the central
   open question.

3. **The small-correlated-alias case.** Outlier robustness is proven against
   i.i.d. and large (1.05 m) correlated aliases; small repetitive-bay-scale
   correlated aliases (0.2–0.35 m) remain the unprobed hard case, and are exactly
   the regime the corridor information limit lives in.
