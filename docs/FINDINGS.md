# FINDINGS — Bounded-Memory VSA/SSP 2D-Lidar SLAM

*A distilled synthesis of the research in this repository. The chronological
lab notebook — every run, retraction, and independent audit — is
[`RESULTS.md`](RESULTS.md); section names cited below (e.g. "R3 refinement",
"Tikhonov prior sweep") point there for depth. This document reports only the
CURRENT / shipped numbers; RESULTS.md keeps the superseded ones as
stratigraphy. Read time: ~15 minutes. Module paths in this document reflect
the 2026-07-13 restructure; ledger sections and scratch files cite the
historical flat filenames — the mapping is [`RESTRUCTURE-MAP.md`](RESTRUCTURE-MAP.md).*

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

ATE rmse vs RBPF-corrected references; `runners/carmen.py <log>`,
deterministic, reproduced bit-exact by an independent auditor.

| log | ours | raw odometry | map memory | speed |
|---|---|---|---|---|
| Freiburg 101 (held-out, dense-revisit building) | **1.883 m** | 8.56 m | 1.9 MB | — |
| Intel Research Lab (full) | 2.440 m (med 1.553) | 24.2 m | 5–8 MB | 17 ms/kf |
| Belgioioso Castle (held-out, non-Manhattan) | 2.640 m (range-identity eval) | **1.72 m** | — | — |
| Freiburg 079 (zero-shot) | 5.523 m | 14.35 m | 3.5 MB | 30 ms/kf |
| ACES3 Austin (zero-shot) | 6.212 m | **5.41 m** | 5.3 MB | 13 ms/kf |
| MIT Infinite Corridor, 1.9 km (held-out) | 42.66 m (38–58 band) | 189.3 m | 27 MB | 12–21 ms/kf |

*(These values are deterministic and reproduce bit-exact; note however that
Intel/fr079/ACES are perturbation-sensitive — a ≥1e-4 relative map perturbation
moves them into multi-meter bands — so config-to-config deltas below the band
width are not attributable. See §9 measurement caveat 3 and the §3
stability-boundary bullet; fr101 is the one point-stable log.)*

In-repo reference baselines (identical parsing / keyframing / eval,
`baseline_*.py`; see "Reference baselines"): ICP + pose graph **1.70 m** (15–35
MB, retains all scans), correlative grids 3.27 m, RBPF/GMapping-lite **0.12 m**
(39–56 MB peak, particle grids). *Caveat:* the ATE reference (`*.gfs.log`) is
itself GMapping (RBPF) output, so the RBPF baseline's 0.12 m is scored against
its own algorithm family — partly self-referential, and the only one of the
four numbers not cross-family comparable; read it as "reproduces the reference,"
not a 14× accuracy win. The comparable numbers are ours 2.44 / ICP 1.70 / CSM
3.27. The positioning is deliberate: ICP beats us everywhere at 2–4× our memory
with full scan retention. We hold the smallest, boundedly-sized, history-free
state. The stack pays off **in proportion to dead-reckoning drift**, and three
held-out logs (fr101, belgioioso, MIT — zero code/parameter changes) now nail
that down from both ends:

- **fr101 is the best transfer result in the project (1.88 m).** A dense-revisit
  loopy building — exactly the regime loop closure exists for — where everything
  stacks: odometry 8.56 → frontend 3.16 → +loops 1.88 m on 53 accepted closures.
- **belgioioso (2.64 m) is the honest non-Manhattan probe.** Castle stone-wall /
  courtyard geometry, precisely the octave-ring matcher's one documented
  environment fragility; it lands close but does not beat its own excellent
  odometry (1.72 m).
- **ACES (6.21) and belgioioso (2.64) are the two frank negatives, and they are
  the SAME failure mode:** where dead-reckoning is already excellent, the
  scan-matching frontend replaces good odometry deltas with slightly worse ones
  (see "The frontend do-no-harm gap", §6). Not a loop-backend failure — on ACES
  the closure backend is net-positive.
- **MIT is the deep held-out test** — 4–6× longer than any tuning log, a ~4.4×
  drift reduction (42.66 m vs raw 189.3) that demonstrates the revisit-density
  limit at full scale, and the site of the now-closed corridor limit (§5).

> **Transfer caveat (referee).** Intel/fr079/ACES are no longer held out: the
> shipped config was selected by minimizing the worst-of-three ATE ratio.
> fr101, belgioioso, and MIT are the genuinely unseen logs, all run zero-retune.
> The drift-proportional-payoff pattern held on every one exactly as the ledger
> predicted: fr101 (high drift, revisit-dense) is the best result; belgioioso and
> ACES (low drift) are the honest ceilings; MIT (high drift, revisit-sparse) is
> the graceful-degradation case.

---

## 2. The encoder result

The encoder study (`sspslam/encoder.py`; "Headline table", "Findings", "World
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
  backwards for *scales*. The reconciliation (sota/golden_dithering.md):
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

The deliverable is `sspslam/bounded.py` (continual, bounded); `experiments/hier.py`'s HY4
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
  bench. The equal-storage question is now **settled** ("Derivative vs angular
  resolution at equal storage"): the derivative doubles matched-band storage
  (segvec 240 + segder 240), and spending that same budget on angles instead
  (120 angles, no derivative, identical 480 components) loses decisively —
  **Intel 2.44 vs 7.93 m, fr079 5.52 vs 14.8 m**, at 2× the matcher compute. The
  derivative is load-bearing, not a wash: 60 angles + derivative dominates raw
  angular density on both accuracy-per-component and accuracy-per-flop, because
  angular resolution is not the bottleneck — rotational *fidelity* is, and the
  analytic first-order term delivers it in half the dimensions. Nothing is ever
  re-encoded, so between relaxations the representation cannot drift.

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

- **The stability boundary of that early-stopped pipeline is now measured — and
  Intel's 2.440 is a knife-edge** ("FPGA track opened", 2026-07-10). A
  freeze-time noise control (i.i.d. relative Gaussian on each frozen segment
  vector, deterministic) shows Intel is *bit-identical* through ε=1e-5 and
  **bifurcates between 1e-5 and 1e-4 relative map noise into a ~[3.5, 5.5] m
  band that does not widen with ε** (1e-4 → 4.00; three seeds at 1e-3 → 4.17/
  3.60/3.87; 1e-2 → 5.23; 3e-2 → 3.46) — not progressive damage but a
  solver-path bifurcation of the early-stopped relax + closure-admission
  cascade (even float32-rounding the frontend guess lands in the same band,
  3.97). fr101 shrugs off 1e-2 (1.884 ≈ 1.881). Two consequences. (1) Honest
  framing: the shipped Intel figure is a knife-edge configuration; under *any*
  implementation perturbation ≥1e-4 (storage quantization, fixed-point
  arithmetic, reordered accumulation) the robust Intel figure is the band —
  still ~5× under raw odometry. (2) Design spec: per-log map-noise tolerance
  varies by 3+ orders of magnitude (fr101 ≥1e-2, Intel ≤1e-5), so quantized /
  hardware deployments must be judged per regime, not on the tuning log's
  point value.

- **complex64 is a free 2×** ("Memory levers"). Segment storage at 8 B/component
  gives round-trip error ~2.4e-8 and a match-peak shift of 0.000 mm / 0.00
  millideg; Intel/fr079/ACES ATE bit-identical. It is opt-in (default off on
  chaos-sensitive long runs like MIT, where a 0.26 mm perturbation moves the
  result 12 m). Net map story: HY4 matched-band split (0.67×) × complex64
  (0.5×) ≈ 0.33× of the original 8 MB at bit-identical Intel accuracy (~2.6 MB).

- **Map COMPOSITION works — the thesis realized** ("VSA map composition",
  `experiments/multisession.py`, audit-verified). Two *independently*-built Intel
  sub-maps compose by shift/permute/**add**: the merge is a genuine per-cell A+B
  vector addition (bit-exact — standalone `place()` == `world_vec_seg`, and
  folding a rigid T into an anchor pose == transforming the world-placed
  bundle), so nothing is re-encoded from scans. Overlapping 2 m cells BUNDLE
  rather than duplicate (37 % memory saving, O(area of the *union*)); adding one
  session does not destroy the other's signal (bundled-cell cosine 0.940 median,
  a small ~9 % RMSE crosstalk cost, not zero); and superposition tolerates
  sub-wavelength (~0.25 m) misalignment. This is the property grids and point
  clouds structurally lack. **Scope caveat** (from the audit): the localization
  test is a gt-seeded local *refinement* of a session's own build scans, so it
  demonstrates *composition-without-corruption*, not from-scratch cross-session
  relocalization; and aligning two independently-drifted maps to sub-wavelength
  (a ~1.2 m non-rigid residual to any single rigid T_AB) is a *general*
  multi-session-SLAM problem — cross-session constraints + joint relax — not a
  property of the representation. After alignment, the algebra applies unchanged.

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

## 5. The corridor information-limit (the centerpiece) — CLOSED

The MIT infinite corridor's residual error (42.66 m; med-error band 38–58 m) is
the deepest thread in the project, and it is now **closed**. The conclusion is
sharper — and more general — than the earlier framing: the corridor is
**fundamentally ambiguous from 2D-lidar appearance at every level available to a
bounded, appearance-only SLAM** — per-frame, per-place-descriptor,
per-geometric-consensus, and per-temporal-sequence. This is an information limit
of the sensor–environment pair, triangulated from three *independent* directions,
not a tuning or representation gap. (The earlier framing — "place-recognition
information is starved at the coarse relocalization wavelengths" — was the
correct *local* diagnosis but the wrong *root cause*; §5.3 below overturns it.)

### 5.1 The first pincer (drought / R4 / R5 / cascade / belief)

Five refinements established that no verification, consensus, belief, or
multi-scale trick recovers the corridor within the shipped coarse-band retrieval.
**Drought relocalization** ("R3 refinement") breaks *engineered* droughts on the
bench (329→5 cm, 4/4 seeds) and re-anchors MIT at two distinctive junctions
(45.3→38.0 m) but never the deep corridor — *aggressiveness is the enemy*, a
wrong pre-aligned building-scale snap is unrecoverable. **R4** (whitening + NMS +
sharded store) sharpened retrieval but converted misses into *consistent wrong
snaps* (the wrong twin agreed tighter — 0.54 m / 0.6° — than genuine junctions).
**R5** replaced R4's heuristics with Mangelson et al.'s Pairwise-Consistent-
Measurement-Set max-clique consensus, bit-identical, and returned the verdict
"there is no consensus to find" — the deep-corridor wrong fires are mostly
pairwise-inconsistent scatter. The **scale-cascade** is a better twin
*discriminator* (AUC +0.11) but can only *reject*, not *recall*. The **belief
frontend** showed the aperture worlds have the *tightest* frontend surfaces
(corridor posterior p90 8.8 cm, 0.0% multimodal): the drift is not represented as
frontend ambiguity, so carrying belief regresses everywhere it fires. The pincer
localized the wall to retrieval at the coarse relo wavelengths (λ 5.3/12.8 m) —
but could not say whether that was an *environment* limit or a *summary* artifact.

### 5.2 Scan-Context control: the signal IS in the raw scan (representation, not environment)

The decisive control. A standalone Scan-Context radial ring-key computed on the
*raw* MIT scan retrieves true revisits at **recall@40 0.674** (deep-corridor
third 0.717) where the coarse SSP relo band scores **0.000** (0/106 at the
deepest revisits). Wired into the drought pipeline as a shortlister (§5.3) the
same descriptor reaches **0.808** (coarse-band 0.317 on the identical attempts) —
the canonical 0.32→0.81 lift. **The coarse-band "starvation" was a
representation/summary artifact, not an environment limit:** the radial-occupancy
signal that distinguishes corridor places is present and retrievable; the
λ 5.3/12.8 m summary throws it away. This overturns the §5.1 root-cause reading.

### 5.3 Ring-key shortlister: retrieval solved — and the wall moves to CONSENSUS

Wiring that ring-key retrieval into the drought path (`experiments/ringkey.py`, feeding
the unchanged verify + PCM admission) recovers 84/104 true revisits vs the coarse
band's 33/104 — retrieval is *solved in-pipeline*. Yet the ATE does **not**
improve: verified drought fires jump **70 → 252** while admitted PCM cliques go
**2 → 0**, and ATE nudges *worse*, **42.66 → 45.24 m**. The 252 fires include
many corridor **twins** that each pass fine geometric verification (they align)
but point to *different* wrong places, so PCM finds no pairwise-consistent
clique — and the extra genuine revisits arrive diluted by even more twin scatter.
**Better retrieval made geometric consensus HARDER, not easier.** The wall was
never retrieval; it is twin disambiguation / consensus.

### 5.4 SeqSLAM: the corridor is SEQUENCE-AMBIGUOUS (the last independent axis)

PCM checks *geometric* pairwise consistency, not *temporal*. SeqSLAM (Milford &
Wyeth 2012) is the SotA answer to geometric-consensus failure: a genuine revisit
yields a temporally-consistent *run* of ring-key matches (a velocity-swept
diagonal); a twin is an isolated coincidence. `experiments/seqslam.py` scores the diagonal
at each drought candidate. It **also fails**: over 34 true-revisit attempts,
sequence picks genuine over the best twin **17/34 = 0.500 (chance)**; at every
gate threshold twins pass at ≥ the genuine rate. Consequently **seq-only snapped
2 TWINS** (ATE **56.23 m**, max error 200 m — a catastrophic building-scale wrong
snap); seq+PCM's clique requirement safely blocks the twins (0 snaps) but
sequence gives PCM nothing new to consense on → 45.24 m, identical to
drought-off. **Corridor twins produce diagonals as temporally consistent as
genuine revisits — the geometry is identical over the whole sequence window.**

### 5.5 Conclusion: an information limit, triangulated three ways

| axis of place recognition | probe | verdict |
|---|---|---|
| retrieval (per-place descriptor) | Scan-Context / ring-key | signal IS recoverable (0.317 → 0.808) — **not** the bottleneck |
| geometric consensus | PCM max-clique | twins verify geometrically and scatter; no clique isolates the genuine set |
| temporal sequence | SeqSLAM diagonal | twin diagonals as consistent as genuine (0.500, chance) |

The MIT infinite corridor is ambiguous from 2D-lidar appearance at *every* level
a bounded appearance-only SLAM can reach — per-frame, place-descriptor,
geometric-consensus, and temporal-sequence. Correct relocalization there requires
an **independent absolute cue** (global positioning, surveyed landmarks, active
perception, or a wider-aperture/3D sensor) that lidar appearance alone cannot
supply. This is a property of the *sensor–environment pair*, established from
three independent directions — not tuning, and not (per §5.2) representation.
**The shipped conservative drought (2 genuine snaps, 42.66 m) is at the
achievable frontier; more aggressive relocalization (seq-only) snaps twins and is
strictly worse (56.23 m).** The right move under an irreducible ambiguity is to
decline it, which is what ships.

**This is a fundamental limit for one case and a strongly-triangulated,
constraint-conditioned one for the other — not an engineering gap in either**
(confirmed by a SotA exhaustion scout + an adversarial code review, 2026-07;
epistemic status made precise per the latter). Distinguish two cases:
*local degeneracy* — a smooth featureless corridor has a **singular range-finder
Fisher Information Matrix**, its kernel the corridor axis, so along-axis position
is unobservable and the Cramér-Rao bound blocks *every* unbiased estimator (Censi,
ICRA 2007; operationalized as the degeneracy factor in Zhang, Kaess & Singh, ICRA
2016; "geometric constraints along the axis of symmetry are indistinguishable from
noise," X-ICP, Tuna et al. T-RO 2024) — and *discrete perceptual aliasing* —
repeated identical bays give a **multimodal** global likelihood (a data-
association ambiguity). The MIT twin is the second. The decisive structural fact:
a genuine twin's false closures are **mutually consistent by construction**
(Lajoie, Hu, Beltrame & Carlone, RA-L 2019: "perceptual aliasing creates a large
number of *mutually-consistent* outliers"), so every robust back-end that rejects
outliers *via* inconsistency — max-mixtures (Olson & Agarwal 2013), switchable
constraints (Sünderhauf 2012), DCS (Agarwal 2013), GNC (Yang 2020), and PCM/max-
clique (Mangelson 2018), the ones we tested — cannot prefer the true branch, and
multi-hypothesis / topological SLAM (Ranganathan & Dellaert, T-RO 2006; lazy data
association, Hähnel 2003) correctly *defers* rather than resolving a twin that
never becomes inconsistent. Range-only itself resolves geometry only up to a
reflection/translation (Boots & Gordon, ICML 2013), and for any bounded local
match a false positive is constructible — only a **globally-unique** place rescues
it (Kuipers & Byun, 1991). Breaking the symmetry needs a bit from *outside* it: a
non-range modality ("two rooms may look identical for a 2-D laser scanner, while a
camera may discern them," Cadena et al. T-RO 2016), a globally-unique landmark, or
physically exiting the corridor. No constraint-respecting (2D-range-only, no
learning, no external cue, bounded-memory) method escapes this — the "oct mode
wins" frontend is doing geometric *sequence* disambiguation, the best such a method
can, and it too fails on a genuine twin. (A clean impossibility theorem for range-
only verification of internally-consistent aliased places is an open citation gap
this work sits in.)

*Epistemic precision (code review).* The smooth-degeneracy half is genuinely
fundamental (the CRB is a proof). The discrete-aliasing half is a **strong,
triangulated empirical conclusion conditioned on the stated constraint set** — the
"twins are mutually consistent" fact was *measured with this system's lossy 5-kf
descriptors, this matcher, and four classical robust backends*. A scan-retaining
verifier (ICP-class, holding raw scans) or a higher-resolution geometric verifier
with explicit along-corridor covariance has strictly more information and is *not*
excluded by proof — it could pin partial lateral/heading constraints where the SSP
matcher "slides" (coherence ~0.40). So the honest label is: the *decision* (decline
the ambiguous closure) is correct and well-defended, but "information-theoretic
limit" applies without qualification only to the FIM case; for discrete aliasing it
is a limit *of this representation + these verifiers under these constraints*, and
the general theorem remains the open gap above.

### 5.6 The wall bounds the refinement layer and the cross-session case too

Three further paradigms — each a detection-free or refinement-based attempt to
*sidestep* verification rather than perform it — independently reach the same wall,
extending the triangulation from single-session place recognition to the
gradient-refinement layer and the cross-session axis.

- **Continual gradient-flow closure** (`experiments/flow.py`). A detection-free energy in
  which spatially-near / temporally-distant segments attract via the *analytic*
  SSP-correlation gradient (translation = phase gradient iW, rotation = stored
  d/dθ vector; FD-exact to 8e-8), fixed anchor for gauge. Valid, well-conditioned,
  non-folding — but from a GT-perfect init the flow **drifts away** (the true
  alignment is not an attractor), because genuine co-observed cross-pass cosine
  (0.097) is indistinguishable from noise-pair cosine (p90 0.29). It refines
  *within* the coarse-ring basin; it cannot recover drift or establish
  cross-session association. The gradient-L2 normalization (divide the force by
  |uᵢ||uⱼ|, keep stored vectors raw so additivity survives) and soft/hex
  partition-of-unity binning are confirmed **conditioning/robustness wins**
  (lr-robust; smoother, wider-basin descent) — but neither moves the cosine wall.
- **Hybrid: detected anchors + overlap-flow** (`experiments/hybrid.py`). Sparse verified
  closures (topology) + dense overlap refinement (deformation), the classic
  pose-graph intuition. Single-session: **net-negative** (overlap adds only bounded
  noise to an already-solved map; λ_ov=0 = shipped is best). Multi-session (a fenced
  GT-anchor diagnostic — GT only selects co-visible pairs, supplies the closure Z,
  and scores; the flow force is GT-free): the overlap force **always tightens**
  genuinely-overlapping anchors (~8–16%, a real local signal) but **always worsens**
  global ATE (twin/aliased near-pairs it cannot discriminate); sparse correct
  anchors do all the useful work and the hybrid **does not beat anchors-alone**.
  Independently audited SOUND. This is the most precise statement of the limit: the
  overlap force is *signal not separable from twin-noise*, not *no signal* —
  verification bounds even the gradient-flow refinement layer.
- **Viewpoint dual-channel** (`experiments/viewpoint.py`). A second channel binding the
  robot *poses* a cell's points were sampled from (distinct symbol; rides the graph
  rigidly, channel-off bit-exact to shipped). A genuine **win for map
  composition / redundancy-dedup** (corr +0.93 with true viewpoint proximity, 48/53
  revisits flagged vs blind bundling double-counting all 53) — valuable for the
  bounded-map storage question. But **not a genuine-vs-twin discriminator**: on
  low-drift Intel it is confounded by spatial proximity (viewpoint AUC 0.896 < the
  0.909 proximity confound — it merely re-encodes "near in the estimate," which the
  pose already gives for free); on the deep MIT corridor it is *below chance*. Where
  drift is low the pose already discriminates; where drift is high — the case that
  needs it — appearance (content **or** viewpoint) cannot.

**Extended verdict.** Every mechanism that could substitute for verified detection
— detection-free flow, anchored overlap refinement, viewpoint appearance — converges
on the same genuine-vs-twin indistinguishability. The backend (pose-graph +
overlap-flow + VSA composition) is provably sound *given* closures; detection /
verification is the single irreducible limit, on both the corridor (§5.5) and the
cross-session axis. (Depth: RESULTS.md "Continual gradient-flow", "Hybrid …
single-session", "Hybrid multi-session", "Viewpoint-tagged dual-channel".)

### 5.7 Extended-overlap consensus — and the wall quantified on real appearance

A fourth verification paradigm reaches the same wall, and en route the wall is
measured directly on real Intel appearance.

- **Extended-overlap consensus** (`scratch_overlapslam.py`). The user's "close only
  after a significant *overlap* of the two ends, match them n-to-n, joint-solve, then
  lerp." In isolation the n-to-n consensus is a *perfect* point-twin discriminator: a
  genuine revisit and a point-twin that alias at a single spot have IDENTICAL single-
  point coherence (0.962) yet consensus separates them 1.00 vs 0.00. But no full-SLAM
  integration helps (sanity: the null gate reproduces shipped bit-exact). A brief
  genuine revisit has too little overlap to corroborate — indistinguishable from a
  point-twin — so dense-revisit fr101 regresses; and in a real aliased environment the
  closures that *do* corroborate are the extended-symmetry aliases, so corroboration
  becomes **anti-correlated with correctness** (the soft variant up-weights the twins).
  Temporal holding also fights the bounded-memory architecture — a held closure can
  never be re-measured, so it enters as a stale constraint. The bundle-to-bundle
  registration primitive it required (a stored anchor-frame segment plugs straight into
  the matcher's rotate/translate path) is sound and reusable; the closure paradigm is
  not.
- **The wall on real appearance** (`scratch_components.py`, verification rung; a
  dependency-ordered component ladder that climbs each component from easy-synthetic to
  a real benchmark subset). As a synthetic twin is morphed toward a geometric copy the
  coherence discriminator does not merely go blind — its AUC *inverts below 0.5*: a
  geometric twin out-coheres a genuine revisit, because the real revisit carries the
  viewpoint change the twin does not. Viewpoint-invariance and twin-rejection are in
  direct tension. On a real Intel subset this is concrete: genuine revisits (204 pairs,
  median coherence 0.238) sit above different-place pairs (median 0.093) on average
  (AUC 0.82), but a **4.6% aliased tail** of different-place pairs out-coheres the
  median genuine revisit, and those twins **outnumber the genuine revisits ~7:1** — no
  coherence threshold admits the revisits without admitting the twins. The corridor /
  cross-session wall, measured directly from the appearance side.

---

## 6. The frontend do-no-harm gap (a second irreducible limit)

The transfer suite exposed a second, symmetric information limit — this one in
the scan-matching *frontend* rather than the loop backend. **The frontend helps
iff odometry drifts, and hurts when odometry is already excellent:**

| log | raw odo | frontend-only | +loops (shipped) | frontend verdict |
|---|---|---|---|---|
| Intel | 24.2 | (frontend essential) | 2.44 | hero (odo drifts) |
| fr101 | 8.56 | 3.16 | 1.88 | hero (odo drifts) |
| ACES | 5.41 | 6.38 | 6.21 | harmful (odo excellent) |
| belgioioso | 1.72 | 2.45 | 2.64 | harmful (odo excellent) |

On ACES and belgioioso the damage is **not** in loop-closure admission (the
backend is net-positive on ACES; belgioioso has one revisit) — it is the frontend
replacing already-good odometry deltas with slightly worse scan-matched ones. The
obvious fix is a *do-no-harm guard* that shrinks the frontend correction toward
the odometry guess when odometry is already self-consistent. It cannot be built
from anything this sensor suite observes — a **triple negative**:

1. **Per-frame coherence/magnitude** (`experiments/frontguard.py`). The session-relative
   coherence-ratio distribution is *identical* across regimes (median ~1.0,
   ~21–26% of frames below 0.85); magnitude *inverts* the naive expectation
   (ACES's corrections are larger — med 0.036 — than Intel's 0.017). Every sweep
   rule that helps ACES/belgioioso catastrophically regresses Intel/fr079
   (+200–420%). A **coherence-blind control** (constant α=0.25) reproduces the
   same frontier (ACES 3.11 but Intel +478%), proving the regression is intrinsic
   to *damping*, not a gate artifact.
2. **Windowed translation-systematicness** (`experiments/frontsys.py`). ρ = ‖Σδᵢ‖/Σ‖δᵢ‖
   over trailing corrections — meant to separate "systematic drift-fix" from
   "random map-noise." The regimes **interleave backwards**: fr101 (drift log,
   frontend essential) has the *lowest* xy ρ of all five logs; scan-match jitter
   dominates ρ everywhere.
3. **Windowed heading-systematicness.** Heading corrections *do* median-order
   correctly (drift 0.15–0.38 vs excellent 0.067–0.085), but the tails overlap;
   the sweep regresses the drift logs +62–135% while only partially helping ACES.
   No separable threshold.

"Odometry is already accurate" is a property of the odometry *versus unavailable
ground truth* — a large frontend correction is per-frame indistinguishable
between "odometry drifted, trust the match" (Intel) and "odometry fine, match is
map-noise" (ACES). A do-no-harm guard therefore requires an **independent
odometry-quality estimate** (wheel-slip / IMU residual) that this 2D-lidar suite
does not provide. The ACES/belgioioso regression is documented, understood, and
irreducible; **shipped behavior stands** (declining a guard that cannot be built
from the available signal). The soft-veto column concentrates its win on Intel;
hard-veto edges it on the odo-excellent logs, exactly as the coh-target sweep
predicted — a small, known cost.

**2026-07-10 addendum — the gap's *size* is conditional on sampling quality
(the guard's impossibility is unchanged).** Swapping the segment-resampling /
occlusion-filter preprocessing for per-beam point encoding (`sspslam/quantized.py` "E2",
one phasor per hit, weight = r·dθ) improves the *frontend* on every log tested
— frontend-only ATE: fr079 11.62 → 2.75, ACES 6.38 → 4.33, Intel 5.07 → 4.39,
fr101 3.16 → 3.00, belgioioso 2.45 → 2.17. On ACES the E2 frontend now **beats
the excellent raw odometry** (4.33 vs 5.41) that the shipped frontend degraded
— so a large share of the documented "frontend hurts when odometry is
excellent" was *sampling-quality damage* (hallucinated chord interpolation
feeding an outlier-blind correlation), not an intrinsic scan-matching limit;
belgioioso narrows but does not flip (2.17 vs odo 1.72). The *detection*
question — knowing when to leave good odometry alone — is untouched and still
needs the independent signal above. (Audited; mechanism localized by
frontend-only decomposition; the viewpoint-neutrality and beam-count/Nyquist
explanations were both tested and refuted en route — see RESULTS 2026-07-10.)

**2026-07-10 second addendum — E2's win is a 180°-FOV/sparse-beam REGIME
result, and the general law is per-regime sampling (RESULTS "encoder
sampling study").** The synth FOV ladder flips the E2-vs-shipped ranking
with aperture (180°: tie/lose; 260–360°: 2× win), and on dense wide-FOV
heads raw per-beam points COLLAPSE the closure layer (stata 1040 beams:
1.659, 9 loops) while bridged pair-interpolation at the shipped 63.4° gate
restores and edges past shipped (0.196 vs 0.202, 74 loops); the gate value
is a genuine optimum there (45° under-bridges → 1.66; 75° over-bridges →
8.25). The EXACT segment integral (sinc(k·d/2)·phasor — the principled form
of the range-thermometer blanking) matches the sub-point implementation
(stata 0.196 either way) and doubles as a mass-exact decimation operator
(GROUP/8: 1.6 cm at 2.5× fewer terms on the 1024-beam bench). Blanking
compensation (α) is moot at target density and a consistent
half-energy-restoration effect on sparse 180° logs (α=0.5 best 3/3, never
beating plain E2 there). Deploy recipe for the SPOT head (360°×1024): bridge
at 63.4°, arc mass, sub-points or integrals; shipped fixed-0.12 m resampling
FAILS at that density (1024-beam corridor: 1.76 m, 1 loop).

---

## 7. The negatives (honest table)

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
| Ring-key drought shortlister | Scan-Context kNN feeding drought verify+PCM | retrieval solved (recall 0.317→0.808) but verified fires 70→252 with **0** admitted cliques (baseline 2); better retrieval makes geometric consensus *harder* (twin scatter), MIT 42.66→45.24 (§5.3, "Ring-key shortlister") |
| SeqSLAM sequence consistency | velocity-swept diagonal of ring-key matches | corridor twins produce diagonals as temporally consistent as genuine (genuine-vs-twin 0.500 = chance); seq-only snaps 2 twins (56.23 m, catastrophic), seq+PCM adds nothing (§5.4, "SeqSLAM sequence consistency") |
| Frontend do-no-harm guard | shrink scan-match correction on low-confidence frames | triple negative — per-frame coherence/magnitude, windowed translation-ρ, and heading-ρ all fail; "odometry is already good" is not scan-match-observable (§6, "The frontend do-no-harm gap is CLOSED") |
| Continual gradient-flow closure | detection-free analytic SSP-correlation-gradient flow (no detect/gate) | valid/non-folding but from GT-perfect init drifts *away* — genuine cross-pass cosine 0.097 ≈ noise 0.29; refines within-basin only, no drift recovery or cross-session assoc (§5.6, "Continual gradient-flow") |
| Hybrid overlap-refinement | detected anchors + dense overlap-flow relaxation | single-session net-negative; multi-session tightens true overlaps (~8–16%) but worsens ATE via twin contamination — does not beat anchors-alone; "signal not separable from twin-noise" (§5.6, audited SOUND, "Hybrid multi-session") |
| Viewpoint dual-channel (discrimination) | second channel binding sampled-from robot poses | compose/dedup WIN (corr +0.93) but not a discriminator: Intel viewpoint AUC 0.896 < spatial-proximity confound 0.909; MIT below chance (§5.6, "Viewpoint-tagged dual-channel") |

**Two of these negatives were saved from being false by a read-only audit before
the result was trusted.** The ring-key run's deep-consensus escalation was
silently disabled (`noff`=pool-size never reached the `deep_noff` threshold); the
SeqSLAM run was a lifecycle no-op (the current descriptor was stored *after* the
drought attempt, so the seq drought early-returned every time — a degenerate
356/0/0/0 that would have read as "sequence cleanly avoids twins"). Both bugs
would have inverted the conclusion. Proactive read-only audits ahead of trusting
a number were consistently high-value across the ledger.

---

## 8. Novelty & related work

Claims corrected after a deep literature scout (sota/vsa_ssp_theory.md,
sota/spectral_registration.md; "Related work" in RESULTS).

- **Rotation-as-permutation — partially anticipated.** Krausse/Neftci/Sommer/
  Renner (NICE 2025, arXiv:2503.08608) use approximate rotation-as-permutation
  in a grid-cell VSA (cognitive maps only). Ours remains distinct: an *exact*
  group-closed lattice applied to raw lidar registration.

- **The d/dθ derivative vector — novel in mechanism.** They solve sub-grid
  rotation by bump-vector convolution; we use an analytic first-order Lie
  correction. The ablation (§3) is the clean proof, and the equal-storage
  head-to-head is now settled: 60 angles + derivative beats 120 angles-no-
  derivative decisively (Intel 2.44 vs 7.93 m, fr079 5.52 vs 14.8 m) at half the
  matcher compute — angular density is not a substitute for rotational fidelity.

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

- **External SotA independently validates the shipped design's key choices**
  (targeted literature scouts, 2026-07). Two design questions that were pushed
  hard this cycle both resolved *toward* the shipped architecture: (i) *submap vs
  temporal-segment closure* — Cartographer (Hess 2016) and Kimera never co-mingle
  passes; they freeze a short single-burst unit before drift smears it and link
  revisits with prunable edges, and VSA superposition theory (Frady/Kleyko/Sommer
  2018) gives the mechanistic reason (bundling is lossy, a summed-in closure is
  only approximately prunable) — exactly the shipped frozen-5-kf-segment +
  prunable-loop-edge design, reached from three independent directions. (ii)
  *per-environment lattice resolution* — the coarse-helps-dense-revisit /
  fine-needed-for-aliased effect is the generalization-vs-discrimination /
  matched-filter-bandwidth tradeoff (BoW quantization FAB-MAP 2008; Scan-Context
  resolution; SSP kernel-width Komer 2019 / Frady-Kanerva-Sommer 2021; radar
  ambiguity theory), and the shipped fixed lattice sits at a robust operating
  point (the one coarser-is-better datapoint, fr101 0.67 m, audited to a fragile
  knife-edge, §7). The convergence is corroboration, not novelty — but it is the
  strongest evidence that the design is principled rather than over-fit.

---

## 9. Limits, honest framing, and open questions

**What this is not.** Not SotA-accurate (ICP beats us everywhere; RBPF's ~14×
edge on revisit-dense logs is partly self-referential — the ATE reference is
itself GMapping/RBPF output, see the circularity caveat under §1). Not
bytes/m²-compact — the SSP map is 3–12×
*denser* per m² than a 5 cm occupancy grid; the memory win holds only against
history-storing baselines. Trajectory bookkeeping still grows O(time) (as any
trajectory output must); full-graph relaxation is the one remaining O(t) compute
term, with windowed relax + seq-edge marginalization the designed-but-optional
fix (merged, verified, opt-in via `windowed=True`).

**Three measurement caveats, stated plainly** (1–2 surfaced by an adversarial
full-code review, 2026-07; 3 by the perturbation-band study, 2026-07-10).
(1) **"Bounded-memory" is really bounded-*map*-memory.** The
O(area) bound and every reported map-MB figure (1.9–27 MB) are the segment
vectors + derivatives *only* (`memory_kb()` counts nothing else); the anchor
poses, per-keyframe relative poses, and edge list all grow O(time), and in the
shipped real-log config (`windowed=False`) the relax solves over all anchors, so
compute is O(t) too. The honest headline is *bounded-map-memory, history-free*.
(2) **All accuracy is measured against an RBPF/GMapping reference, not surveyed
ground truth.** Every ATE in this repo — ours *and* the ICP/CSM baselines — is a
distance to GMapping's own trajectory estimate (`*.gfs.log`); no dataset here has
surveyed GT. The eval itself is fair (nearest-timestamp ≤0.3 s, Horn `align_se2`,
RMSE; the anti-oracle discipline keeps GT out of the shipped pipeline), but the
entire numeric ledger measures *agreement-with-GMapping*, not truth. The single
highest-value future validation is an independent, non-RBPF-derived reference
(one log with surveyed/fiducial GT) to decouple "2.44 m" from "2.44 m from
GMapping".
(3) **Single-trajectory ATEs on the sensitive logs are exact but fragile —
config comparisons need perturbation bands.** Every published point value is
deterministic and reproduces bit-exact, but freeze-time map perturbations of
1e-4–1e-3 relative move Intel 2.44 → a non-monotone [3.5, 5.5] band, fr079
5.52 → [5.5, 12.4], ACES 6.21 → [6.2, 8.2] (closure-cascade decision flips;
§3 stability-boundary bullet). fr101 is the one point-stable log (1.88 at 1%
noise). Consequently a config delta smaller than the log's band width is a
basin draw, not an effect; PROTOCOL §6 now requires the band probe
(`sspslam.quantized.BandSLAM`) for config-level claims on the sensitive logs.
The 2026-07-13 reservoir-cadence study extended this law to *online* map
mutation: any rehearsal/replay feature that folds content into the live map
acts as a perturbation sequence — it converts even point-quotable logs into
banded ones (stata: ~1/3 of replay-seed draws collapse the closure graph at
every save cadence and dose tested, from ~70 folds to ~4300), so single-seed
results on map-mutating features are basin draws, and the fhw replay opt-in
is itself a band ({0.22, 0.24, 4.05}; RESULTS "reservoir save-cadence
study").

**What is defensible.** The three properties are the bound, the absence of
history, and the algebra — each verified: Intel plateaus at 698 segments (84% of
the cell-cap ceiling); MIT grows dead-linear at 1.26 seg/m as new corridor is
exposed; the map is transformed only by phase ops and permutations, never
re-encoded. ACES and belgioioso are frank negatives (odometry already excellent,
frontend do-no-harm gap §6); fr101 is the best held-out transfer (1.88 m); MIT
demonstrates the revisit-density limit at scale rather than hiding it.

**Two former open questions are now closed.** *Derivative vs. angles* is settled
— 60-angle+derivative beats 120-angle-no-derivative at equal storage on both real
logs and half the compute (§3, §8). *A richer corridor descriptor* was the
central open question under the old §5 framing; the ring-key / SeqSLAM
triangulation (§5.2–5.5) answers it in the negative — a richer *appearance*
descriptor does recover retrieval, but the corridor is ambiguous at consensus and
sequence too, so no appearance-only descriptor closes the limit. The next lever
is not a better descriptor but an **independent absolute cue** (IMU/wheel-slip
for the frontend do-no-harm guard §6; global positioning / surveyed landmarks /
3D or wider-aperture sensing for the corridor §5.5) — outside this bounded,
appearance-only, 2D-lidar problem statement.

**The unifying finding — verification, not retrieval, is the recurring wall.**
The strongest cross-cutting result of the whole campaign is that every hard limit
lands on the *same* structural bottleneck: appearance-only place **verification /
discrimination** — separating a genuine revisit from a consistent alias — not
retrieval, and not the VSA algebra. The MIT corridor (§5): ring-key *retrieves*
the true revisits (recall 0.32→0.81), but geometric consensus and temporal
sequence cannot *verify* them against self-similar corridor twins. Multi-session
map merging (§3, the composition capstone): superposition, algebraic merge, and
ring-key retrieval all *work*, but establishing cross-session correspondences hits
the identical wall — absolute coherence cannot separate viewpoint-changed genuine
ties from wrong-nearby places (correct/false coherence medians 0.171 vs 0.172),
and PCM consensus helps (best oracle-free two-session ATE 3.35 m, over B-alone
4.37) but cannot close it because the aliased twins are themselves geometrically
consistent. Both are the same limit approached from opposite environments —
self-similar corridors vs cross-session viewpoint change — and both confirm that
what the SSP/VSA representation contributes (drift-independent retrieval,
algebraic composition, bounded history-free maps) is *sound and demonstrated*,
while the residual limit is a property of appearance-only place recognition that
no representation removes without an independent cue. The positive corollary: the
VSA-native pieces — composition, algebraic merge, ring-key appearance retrieval —
are each independently validated; a system that pairs them with an external
verification cue (loop-closure ground truth, GPS, or a discriminative learned
place model) inherits all three for free.

Two GT-oracle diagnostics quantify that corollary — and reveal the corridor wall
is DEEPER than multi-session. Multi-session, given correct correspondences: B
→ 2.49 m (place verification was the *only* missing piece). MIT corridor, given a
perfect PLACE oracle: 42.66 → only 41.04 m (~4 %). The asymmetry is the finding:
on aperture-degenerate corridors, perfect place recognition is **necessary but
not sufficient** — self-similarity also corrupts the relative-pose GEOMETRY (at a
genuine revisit the matcher slides along the corridor, coherence ~0.40), so of
1.9 km only ~3 genuine revisits have geometry a lidar matcher can pin down at all
(dropping the geometry gate admits genuine-but-misaligned slides that make ATE
*worse*, 73 m). So the external cue the corollary needs is **two-fold on
corridors — place AND pose** — not place alone; on distinctive geometry
(multi-session) place alone suffices. The residual limit is verification in the
full sense (which place, and where exactly), and the shipped conservative drought
is near the achievable frontier even against a place oracle.

**Open questions.**

1. **The small-correlated-alias case — now CLOSED (robust).** A controlled sweep
   of a fixed correlated alias (0.15–1.0 m, 10 % injection, room/corridor/sparse)
   shows the backend is robust at *every* magnitude — ATE lift ≤ +0.26 cm across
   the whole range, including the 0.2–0.35 m regime. Two-layer defense: the
   innovation gate rejects only large aliases (>~0.9 m ≈ 3× the drift cap), and
   IRLS Cauchy downweighting neutralizes the small ones that pass the gate (they
   survive as edges but carry negligible weight — a fixed alias disagrees with the
   genuine closures it corrupts). Crucially this is a DISTINCT failure mode from
   the corridor limit: outlier injection places the alias *alongside* a genuine
   competitor (so IRLS can downweight it), whereas a corridor twin has *no*
   genuine competitor to weigh against — so this result sharpens, not softens, the
   corridor verification finding. The two are orthogonal.

---

## 10. The FPGA track (opened and banded 2026-07-10)

The deployment thesis: the per-keyframe hot path (encode, frontend match,
segment fold, loop verify) is O(D) phase arithmetic + small dense MACs and maps
onto FPGA fabric; the sparse pose-graph relax stays host-side. `sspslam/quantized.py`
(subclass-only; neutralised = bit-exact) carries the track. All config claims
below are perturbation-banded per PROTOCOL §6 (RESULTS "2026-07-10
consolidation" has the full table).

- **Compute is trivial; memory and perturbation-tolerance are the design
  pressures.** The whole hot path is ~2.6 M MAC-equiv/keyframe → 0.05 GMAC/s
  at 20 Hz; at the 8-bit arithmetic knee these are LUT-adds (no DSPs needed).
  The map is the resource, and it compresses hard (below).
- **Per-beam point encoding ("E2") is the session's algorithmic win — a
  frontend registration improvement on 5/5 logs** (frontend-only: fr079
  11.62→2.75, aces 6.38→4.33 — now BEATING its excellent raw odometry 5.41 —
  intel 5.07→4.39, fr101 3.16→3.00, belgioioso 2.45→2.17; audited). The
  mechanism was pinned by a discriminator ladder (E3/E4/E4a, audited): the
  win is **the arc-length r·dθ weighting + retaining isolated depth-jump
  SILHOUETTE hits** (0.45% of hits carrying a 2× frontend factor on fr079);
  the dr/tang occlusion masking itself is nearly free under point sampling
  (a first E4 arm that suggested "half is the filter" was audit-corrected —
  it had silently deleted silhouette hits shipped always keeps). Separately,
  in the CHORD-sampling regime the occlusion filter is a per-environment
  tradeoff (frontend-only occ-off: fr079 11.6→4.8, aces 6.4→2.4 — beating
  E2 and raw odometry there — while intel needs it, 5.1→7.0).
  (Interpolated-position mass, beam-count/Nyquist, and viewpoint-neutrality
  were each tested and refuted as explanations; sample positions affect the
  median, not the rmse-dominating failure.)
  Full-system, band-vs-band: **dominates shipped on fr079** (whole band
  [2.21..4.86] clears shipped's best 5.52), **wins aces and belgioioso**,
  median-improves fr101 at some variance cost, and is band-worse on intel
  (keep shipped sampling there). It also deletes the one irregular-compute
  preprocessing stage from the fabric path.
- **Write-time polar quantization needs PER-RING scales** (the relo rings
  otherwise crush the fine rings that the closure veto reads: fidelity
  0.84→0.97 at the same 6 bits). With them, the STORE tolerates astonishing
  compression on band-robust logs — down to **2 bits/phasor phase-only**
  (fr101: 1.41 single-draw, [1.13..3.08] banded with int8 arithmetic at
  75 KB vs shipped's flat 1.88 at 1.9 MB).
- **The arithmetic knee is 8-bit** (256-entry cis ROM, 7-bit values): free on
  fr101, in-band on intel; below it there is real damage beyond any band
  (intel addr6 10.5 vs band ≤6.7). The QPSK/binary-spatter extreme holds
  MEDIANS (fr101 med 1.57 = shipped) but not tails — **the binary bottleneck
  is arithmetic, not storage**.
- **The deployable "FPGA-lean" configuration** — point encoding + 2-bit
  phase-only per-ring store + 8-bit integer arithmetic — runs an integer
  datapath with a ~25× smaller map at comparable-to-better banded accuracy on
  the transfer logs (fr101 [1.13..3.08] @ 75 KB; fr079 {3.15, 6.25} @ 110 KB
  vs shipped's [5.5..12.4] @ 2.6 MB), and holds at scale: **MIT 1.9 km =
  58.1 vs shipped 57.4 (both in the documented 38–58 chaos band) at 625 KB
  vs 13.5 MB — 22× less map**. Configuration is per-regime: aces/belg want
  E2-float (the lean store starves their sparse closures), intel-class
  sparse-beam clutter keeps shipped sampling and lives in its perturbation
  band regardless.
- **Fine rotation via the scan's d/dθ derivative (11→3 encodes/match): clean
  negative** (fr101 2.50, Intel 4.67) — the first-order term is a storage-side
  tool (bundle-averaged, ≤1.5°), not a matcher-side one. The re-encodes stay
  (they are cheap).
- **The knife-edge is not recoverable by local rule-hardening** (decision-
  stability thread, `experiments/stablegate.py` + divergence trace). A perturbed run
  tracks the unperturbed one at sub-mm for thousands of keyframes, then ONE
  boundary decision flips (first flip in the traced pair: the closure edge's
  chain-median attribution) and the early-stopped relax's path-dependence
  amplifies it. Four targeted interventions — loop-gate jitter-consensus
  (no-op), coh_ref pinning (no collapse), frontend jitter-consensus (band
  collapses 14× onto its median, losing the fragile upside), nearest-anchor
  attribution (relocates the draw) — all fail to keep Intel's 2.44: it is an
  unstable fixed point with a ~measure-zero basin. Where determinism matters
  (hardware), frontend jitter-consensus gives a tight predictable band
  ([3.84..4.15] on Intel) at band-median accuracy.
- The webvis real-data path is now **Python-fed**: `demo/export_replay.py`
  runs the actual deliverable and embeds a self-contained replay (poses,
  per-relax anchor snapshots, loop edges); the browser replays it exactly
  (jsc parity harness), so the demo can no longer drift from the system —
  and the real-data tab now DEFAULTS to the replay (the live-JS port stacks
  an LM/PCG relax, a no-odometry guess chain, and knife-edge admission —
  the observed "irrecoverable divergence at loop closures").

## Addendum (2026-07-11, hardware + mechanism session) — four findings that reshape sections above

1. **Integral-proximity cascade basins (sampling).** On stata-class
   dense sensors, session outcome is selected not by per-scan encode
   fidelity (cosine-1.000 families bifurcate) but by the pack's
   fine-ring distance from the EXACT segment integral: packs within
   ~2e-3 (r0 rel) — including the exact integral itself — fall into a
   deep bad basin of the closure cascade (starving at the cmatcher
   displacement pre-gate); sampling discreteness rescues (n1–n3
   sub-points survive; n4+, GROUP folds, and half-chord packs collapse).
   Bench losslessness does not transfer. The deployed bridge2 (n2,
   7.6e-3) sits safely inside the good basin.
2. **The band owns more than we knew (store).** stata-lean's "2b
   failure / 6b rescue" is an ε≤1e-6 basin property, not a fidelity
   tier: uniform-2b's own band spans [0.17..2.99] and uniform-6b
   collapses at ε=1e-3. No fixed per-ring bit allocation stabilizes it;
   a chain-bundle cap of 8 members is the first observed stabilizer
   (band med 1.48→0.29; validation incomplete). Fine-ring bit
   over-provisioning ([64,64,4,4]) STARVES the loop matcher — store
   fidelity is not monotone in bits.
3. **The wall extends to commit policy (tracking).** In fabric-tracker
   form, corridor-class degeneracy produces confident wrong peaks; the
   local score surface is IDENTICAL between healthy-but-degenerate and
   lost keyframes (best gate: 83% catch at 11% false-flag = worse than
   no gate). Correctness is not a per-keyframe surface property —
   §5's information wall reaches every local commit rule.
4. **Bounded map readout has a sharp shape (readout).** The matched
   octave ladder is coherently aliased at exactly 2 m (ghost combs);
   raster occupancy tops at AUC ~0.63–0.69 at ANY bits/K
   (representation-limited) — but correlation reads are cm-sharp, and
   a ~0.1–1 KB self-certified free-space mask (from the system's own
   pass-1 scans) plus a fabric refine turns first-return visibility
   from 0.80 m to 0.07–0.15 m vs GT on silicon. The map's fidelity
   lives in reads, not rasters; O(area) auxiliary masks are cheap and
   GT-free.

## Addendum (2026-07-12, standalone + fidelity-space session) — the dual-space architecture and five laws

**A5. The dual-space architecture (supersedes the single-lattice framing
in §3/§10 for deployment).** The chip runs SLAM standalone in the MATCHER
space (oct60, D=240, 2b codes, per-segment matching) — golden-proven to
2 mm parity with the host tracker, self-mapping at 7.3 cm on revisit-rich
floors (3.9× better than dead reckoning) and odometry-grade on long
tours; pose is streamed (it is tracked on-chip anyway). A SECOND
encode-only FIDELITY space is folded per segment on-fabric, frozen to 2b,
streamed out (~130 B/s), and decoded on the laptop for display/readout —
and, in future, fused with other sensors by vector addition. Encode-only
scales to D≈2k today (one SPRAM accumulator; octave rings are bit-slice
free, half-octave costs one extra projection pass; the derivative vector
costs NO extra projection — cross_a = ±u_{a∓30}).

**A6. The fidelity-lattice laws (scratch_biglat family, school8 + stata
phase-2).** (i) ANGLES CAP AT 60 — more angles hurt extraction on every
ladder tested (both stores, both fixtures; nothing anywhere has ever paid
for >60). (ii) THE LADDER IS THE LEVER: half-octave densification of the
content band + coarse extension (2.83/4 m) lifts real-data extraction
+.096 AUCgt, pushes prior-free global-decode success .85→.95, extends the
2b capacity tail (K=128 p90 .74→.15), and ANNIHILATES the 2 m octave
ghost comb. Recipe: span11x60 (0.125–4 m, D=660, 418 B/seg) primary;
hoct9x60 lean. (iii) THE FINE FLOOR IS THE SENSOR COHERENCE LENGTH
λ_min ≈ 2πσ_r (≈12.6 cm at σ=2 cm): sub-coherence rings are dead bytes
(confirmed to 1.1 cm; only the noise-free analytic PSF sharpens). (iv)
The old "aperture-limited" extraction mechanism is AMENDED to RADIAL
ALIAS STRUCTURE — D grew 8× with zero gain when growth was angular or
sub-coherence. (v) Golden-ratio ladders REFUTED on real data (second
φ synth-sandbox artifact); √2 stands — mid-band DENSITY is the binding
constraint, and φ's sparser band costs more than its superior
incommensurability buys.

**A7. Explicit history beats superposed history (third and fourth
instances) + the commit-wall extends to re-registration.** The anneal
cycle (negate→rematch→reencode) works on explicit frozen-code history
in-basin (≲0.4 m) and is DEAD via time-bound superposition (20–80× over
the knee; naive unbind-subtract injects √(K−1) noise). The spare-SPRAM
sample reservoir is GT-neutral-to-worse on this system at every arm
incl. forced-blackout diagnostics: capture and map are the SAME estimator
(common-mode errors), so score-vs-map cannot select GT improvement —
the commit-policy wall (§6) seen from the re-registration side. BSC at
equal bits (dimension-compensated, fine multi-scale codes): bundle knee
K*≈8–16 vs phasor 32–48 — binary-grade STORAGE is already harvested
(2b codes, i^mc multiply-free matching); binary-grade OPERATIONS lose.

**A8. Readout is gauge-bound, not method-bound.** The line-prior pursuit
(exact line atoms, per-segment decode + cross-segment consensus) reads
the store at 5 cm median in the map's own gauge; scored against GT it
inherits the store warp exactly like every readout must. Per-segment
(sub-knee) decode is mandatory — global-bundle readout is the low-AUC
raster. Parametric lines (~50 KB) are a display/compaction option
composing with the fidelity stream.

**A9. Handoffs are a solved sub-problem in the code domain.** Best-of-3
nearest-segment matching with incumbent hysteresis is a STRICT win for
localization on coherent maps (mixed p90 .128→.104, max .876→.466, holds
55→6; beats the live host tracker) because 2b phase-only segment vectors
are norm-equal — raw totals compare across segments, unlike the float
domain where the same idea thrashed and was reverted. On gauge-warped
self-built maps it mixes gauges and regresses: K=3 is a MODE (coherent
map), not a default.

**A10. On a memory-rich small FPGA, the VSA representation is cheap —
CONTROL is the expensive part (v7 build session, measured).** The full
standalone-SLAM function on iCE40 UP5K: the entire VSA datapath (encode,
resident-map match, fold-at-pose, all ROMs) fits in 2034 LUT / 6 EBR /
2 DSP after serializing the rings (the throughput the parallel v6 core
sells — 488 vs 1385 cyc/cand — is 500x more than the 5 Hz task needs).
What does NOT fit is the scalar SE(2)/gating/protocol machinery written
as parallel-word FSMs: tracker + top = 5614 LUT of the 7452 total vs a
5280 budget, and hand-CSE recovers nothing (synthesis already shares).
The corner exercise inverted the resource profile (v6: 97% LC + 100%
EBR; lean: 51% LC + 20% EBR) and thereby exposed the closure: move
control into the memory the lean core freed (microcoded scalar engine,
~650 LC + EBR ucode). The representation thesis survives contact with
silicon; the engineering lesson is that its bounded-memory property
extends to control — the natural home of a SLAM chip's brain is a tiny
interpreter over the same BRAM that holds the map. Freeze-store corollary
(banked, both fixtures): 2b phase + 1 liveness bit + 4 ring scales
(38 B/seg, all integer, from one freeze scan) reads out BETTER than the
float store it quantizes — quantization noise is below, and dead-lattice
junk above, the extraction's noise floor.

## A10 — Sensor-fusion architecture laws (ECP5 track, 2026-07-14; RESULTS "4h block" parts 1–5)

(1) **Two maps of different kinds.** The phase algebra transforms range
points but not camera appearance (parallax): the lidar map is
transformable world-frame content; the vision map is a bounded
per-anchor appearance-snapshot library queried by max-similarity. Late
score-level fusion; max-rule at the aliasing wall (school 0.798 >
lidar-alone 0.720), vision-weighted sums under view overlap (0.954).
(2) **Lidar scales with D, not ingest.** School place-AUC 0.700 → 0.858
from D240 → D960 (finer azimuth = finer exact-rotation search + more
scales); 3 of 64 rings and ~2k points carry the full signal — spend on
the lattice, not the bandwidth.
(3) **Full 6-DoF on frozen vectors.** azel3d decodes unknown
pitch/yaw/roll at ~4° (grid floor) + translation incl. z at grid floor —
2D+heading (az2d) is orientation-blind beyond yaw (19–22°). Real-motion
(TUM): 7–8° med; mixed-axis motion favors fib3d's isotropy slightly.
(4) **Vision: dense beats sparse for cross-view place; appearance-in-
phase is a precision/ego-motion tier, not a cross-view tier.** Weights-
only intensity grid = the coarse cross-view channel (school 0.622);
phase-bound appearance sharpens view-specificity (classroom 0.999 /
adj-rep 0.997 — the ego-motion service) and hurts across viewpoint
change. Untrained int8 CNN heatmaps ≈ +0.03 over raw intensity (trained
headroom exists at ~60% of the MULT18 budget @QVGA120).
(5) **3D visual landmarks cross the reverse-view wall — given depth
coverage.** TUM fr3 (registered depth): corners × depth rev-AUC 0.941
where bearings collapse to 0.198; fr2's 4 m Kinect in a hall = the
coverage boundary. On the platform the range source is 60 m lidar →
**calibrated lidar-camera extrinsics is the single highest-value step**
(the SPOT depth negative was an extrinsics artifact, proven).
