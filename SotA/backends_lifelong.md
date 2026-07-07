# SotA scout: robust backends, sparsification, early-stopping theory, and lifelong bounded-memory mapping

Literature scout 2026-07-07, cross-referenced against RESULTS.md (SSP bounded SLAM:
anchor pose graph, IRLS+LOO+innovation gating, max_nfev=30 early-stopping-as-regularizer,
windowed relax + eviction marginalization, drought relocalization in progress vs the MIT
Infinite Corridor sparse-loop failure, O(area) cell-cap map).

---

## Cluster 1 — Backends

### 1.1 Robust pose-graph optimization (outlier-tolerant back-ends)

**Switchable Constraints — Sünderhauf & Protzel, IROS 2012.**
[Paper (PDF)](https://nikosuenderhauf.github.io/assets/papers/IROS12-switchableConstraints.pdf) · [Project page](https://nikosuenderhauf.github.io/projects/switchableConstraints/)
Adds a continuous switch variable s_ij in [0,1] per loop edge that scales the edge's
information matrix, jointly optimized with the poses under a prior pulling s toward 1 —
the optimizer itself can turn off edges that fight the consensus. This is the ancestor of
our soft sigma-inflation veto, but *inside* the solve rather than as a pre-insertion gate;
it lets edges be resurrected if later evidence supports them, which our hard LOO pruning
cannot. RESULTS cross-ref: our "soft veto = sigma inflation" (coh_soft) is functionally a
hand-scheduled switchable constraint with the switch driven by coherence instead of residual.
*Actionable:* keep vetoed/pruned edges in the graph as switched edges (tiny weight, switch
prior from the coherence ratio) instead of deleting them — gives the corridor perceptual-alias
leaks a path to self-correct after more evidence arrives.

**Dynamic Covariance Scaling (DCS) — Agarwal, Tipaldi, Spinello, Stachniss, Burgard, ICRA 2013.**
[Paper (PDF)](http://www2.informatik.uni-freiburg.de/~agarwal/docs/agarwal13_icra_ws.pdf) · [Experimental analysis](https://www.researchgate.net/publication/286680057_Experimental_analysis_of_dynamic_covariance_scaling_for_robust_map_optimization_under_bad_initial_estimates)
Closed-form solution of the switchable-constraints inner problem: each loop residual is
scaled by s = min(1, 2*Phi/(Phi + chi2)), which is exactly a Geman-McClure robust kernel —
one extra line in the residual evaluation, no added variables, much faster convergence than SC.
Follow-up analysis shows DCS is sensitive to bad initialization on floppy graphs — precisely
the fr079 regime where our early-stopped TRF is load-bearing.
*Actionable:* DCS is the cheapest upgrade path from our Cauchy IRLS — same code shape,
known equivalence to switchable constraints, one Phi knob; A/B it against Cauchy on the
corridor false-closure world before building anything heavier.

**Max-Mixtures — Olson & Agarwal, RSS 2012 (via Vertigo).**
[Vertigo repo/README](https://github.com/OpenSLAM-org/openslam_vertigo/blob/master/README) · [TU Chemnitz project](https://www.tu-chemnitz.de/etit/proaut/en/research/robustslam.html)
Each loop edge is a max-of-Gaussians: an inlier mode plus a broad null-hypothesis mode; the
max operator picks the active mode per iteration, so a wrong edge degrades to a near-zero-
information constraint without leaving the graph. Handles *multi-modal* aliases natively —
one edge can carry "bay k or bay k+1" as two modes, which no per-edge chi-square test can.
*Actionable:* for the repetitive-bay small-alias case RESULTS flags as unprobed (0.2–0.35 m
correlated aliases under the gate allowance), a two-mode max-mixture edge (candidate transform
+ null) is the literature-standard representation; Vertigo also ships the standard datasets +
a spoiler script for benchmarking our stack against SC/DCS/max-mix directly.

**Graduated Non-Convexity (GNC) — Yang, Antonante, Tzoumas, Carlone, RA-L 2020.**
[arXiv 1909.08605](https://arxiv.org/pdf/1909.08605) · [MIT-SPARK code](https://github.com/MIT-SPARK/GNC-and-ADAPT) · [Efficient GNC scheduling](https://arxiv.org/pdf/2310.06765) · [Adaptive GNC](https://arxiv.org/pdf/2308.11444)
Optimizes a robust cost by homotopy: start from a convexified surrogate (mu large — everything
inlier), alternate weight/variable updates (Black-Rangarajan duality, like IRLS), and anneal mu
until the true truncated-least-squares cost is reached. No initial guess needed, ~70-90% outlier
tolerance on PGO benchmarks; the annealing schedule is the compute cost (see the 2023-24
scheduling papers for speedups). Warning for us: GNC's guarantees assume the *inlier* set is
well-conditioned — on fr079-style floppy graphs GNC converges confidently to the same flat-valley
slide our early stopping avoids; GNC replaces IRLS, not the regularization.
*Actionable:* use GNC as the *batch verifier* inside consensus clustering (run GNC on each
candidate alias-family subset; the subset whose GNC solve keeps the most edges at full weight
wins), rather than as the online solver.

**PCM max-clique — Mangelson, Dominic, Eustice, Vasudevan, ICRA 2018.**
[Paper (PDF)](http://robots.engin.umich.edu/publications/jmangelson-2018a.pdf) · [IEEE](https://ieeexplore.ieee.org/document/8460217/) · [GTSAM example impl](https://github.com/U-AMC/PCM_gtsam) · [Group-k extension, IJRR 2024](https://journals.sagepub.com/doi/10.1177/02783649241256970)
Implementation recipe for our prescribed consensus-cluster test: (1) for each pair of loop
edges z_ij, z_kl compute the cycle error C = z_ij ∘ (odom j→l) ∘ z_kl^-1 ∘ (odom k→i);
(2) accept the pair as consistent if the Mahalanobis norm of C (covariance = composed edge +
odometry covariances) passes a chi2 threshold; (3) build the consistency graph and take the
maximum clique as the inlier set. Key insight vs our per-edge gates: a family of correlated
aliases (fixed wrong SE(2), RESULTS' unprobed hard case) is *internally* consistent but
inconsistent with the odometry cycles connecting it to true closures — pairwise cycle checks
see exactly what per-edge innovation cannot. Exact max-clique is real-time at our edge counts
(28–288 edges); the Group-k 2024 extension handles cases where pairwise checks are too weak
(pairs of aliased edges that happen to agree).
*Actionable:* implement PCM as a periodic batch filter over the accepted-loop-edge set (our
seq-chain covariances at 0.03 m/0.3 deg give tight cycle gates); the max clique = the trusted
family, second-largest clique = the alias family — this IS the consensus-clustering design,
with 8 years of validation behind it.

**DC-GM — Lajoie, Hu, Beltrame, Carlone, RA-L 2019.**
[arXiv 1810.11692](https://arxiv.org/abs/1810.11692) · [MIT SPARKlab page](http://web.mit.edu/sparklab/research/dcgm/)
The only work that *models outlier correlation explicitly*: a discrete-continuous graphical
model where discrete outlier-selection variables carry a Markov coupling (aliased measurements
are selected/rejected together), solved by semidefinite relaxation with suboptimality
certificates. Directly formalizes RESULTS' observation that "correlated aliases have a
signature per-edge tests cannot see" — their perceptual-aliasing experiments are the same
repetitive-structure failure as our corridor world.
*Actionable:* borrow the modeling idea, not the SDP: add a correlation prior in our consensus
clustering — edges whose relative transforms agree within tolerance form one family and are
accepted/rejected as a unit (family-level LOO instead of edge-level).

**riSAM — McGann, Rogers, Kaess, ICRA 2023.**
[arXiv 2209.14359](https://arxiv.org/pdf/2209.14359) · [Code (GTSAM-based)](https://github.com/rpl-cmu/risam)
The state of the art for *online* robust PGO: GNC made incremental by riding on iSAM2's Bayes
tree, with a graduated kernel that only re-anneals the parts of the graph a new measurement
touches. Matches offline GNC accuracy at online cost — the reference point our
"windowed relax + gates" architecture should be benchmarked against, and the proof that
robustness and incrementality compose.
*Actionable:* their per-edge convexity-graduation state (each edge remembers its mu) is the
right pattern for our windowed backend: robustness state must persist across windows or
re-solved windows re-admit previously downweighted aliases.

### 1.2 Incremental smoothing vs our windowed relax

**iSAM2 / Bayes tree — Kaess, Johannsson, Roberts, Ila, Leonard, Dellaert, IJRR 2012.**
[Paper (PDF)](https://www.cs.cmu.edu/~kaess/pub/Kaess12ijrr.pdf) · [IJRR](https://journals.sagepub.com/doi/10.1177/0278364911430419)
Incremental factorization as editing a Bayes tree: a new factor only invalidates the path from
its variables to the root; fluid relinearization re-linearizes only variables whose delta
exceeds a threshold. Exact (up to relinearization thresholds), with update cost proportional
to the affected subtree — which for exploration is O(1) and for big loop closures is O(n),
the same worst case our escalation ladder hits. Two lessons for our windowed design: (a) the
*graph structure itself* (tree + wildfire threshold) selects the update region, rather than a
BFS radius heuristic — a closure's true influence region is its Bayes-tree path, not a metric
ball; (b) iSAM2 has no robustness or forgetting — it solves a different axis than our
gates/eviction, confirming the two designs are complementary (riSAM is the merge).
RESULTS cross-ref: our finding that windowed relax "costs accuracy for no speed at Intel scale
because analytic GN made full solves cheap" mirrors the iSAM2 literature's own observation
that batch is competitive below ~10^4 poses; windowing pays only on genuinely long runs —
exactly what RESULTS measured (-21% wall at n=3000).
*Actionable:* replace the BFS cycle-window with an influence-propagation criterion — expand
the window while boundary-pose deltas from the inner solve exceed the fluid-relinearization
threshold (0.01–0.1 rad/m in iSAM2 practice); this makes the escalation ladder principled and
self-terminating.

### 1.3 Sparsification / marginalization with fidelity guarantees

**Generic Linear Constraints (GLC) — Carlevaris-Bianco, Kaess, Eustice, T-RO 2014.**
[Semantic Scholar](https://www.semanticscholar.org/paper/0b9e7c5d54f4f88810ab6666e1b58124eab72c3f) · [Long-term GLC node removal](https://www.researchgate.net/publication/261352950_Long-term_simultaneous_localization_and_mapping_with_generic_linear_constraint_node_removal)
Node removal by exact marginalization onto the Markov blanket, then optional sparsification of
the dense result via a Chow-Liu tree (keep the maximum-mutual-information spanning tree of the
blanket, KLD-optimal among trees). Demonstrated multi-session graphs whose size tracks
*environment area, not operation time* — the published precedent for our O(area) claim, at the
graph level. Known weakness: linearization at the removal-time estimate bakes in errors ("world-
frame sums never move with the graph" is our hierarchical-map cousin of this problem).
*Actionable:* our eviction marginalization currently composes seq edges (a chain = a tree with
no fill-in); when an evicted anchor has loop edges, GLC says the correct target is the Chow-Liu
tree over the full Markov blanket, not just the chain — measure the KLD gap on Intel evictions
to know whether the shortcut costs anything.

**Nonlinear Factor Recovery (NFR) — Mazuran, Burgard, Tipaldi, IJRR 2016 (NGS, RSS 2014).**
[IJRR](https://journals.sagepub.com/doi/10.1177/0278364915581629) · [NGS (PDF)](https://www.roboticsproceedings.org/rss10/p40.pdf)
The fidelity-guarantee upgrade over GLC: choose any sparse *nonlinear* factor topology, then
solve a convex problem for the factor information matrices that minimize KLD to the dense
marginal — recovered factors are relative (relinearizable), so the approximation survives
later graph corrections. This addresses exactly the invariant RESULTS identified ("rigid-
segment design keeps map geometry consistent with its anchor by construction"): NFR keeps
marginalized *constraints* graph-consistent the same way.
*Actionable:* when eviction removes an anchor carrying loop edges, emit 2-3 relative SE(2)
factors among its neighbors with NFR-optimized information matrices instead of inflating a
composed seq edge; the KLD number it produces is the "fidelity guarantee" our marginalization
currently lacks.

**Information-theoretic pose-graph compression — Kretzschmar & Stachniss, IJRR 2012 (IROS 2011 with Grisetti).**
[IJRR](https://journals.sagepub.com/doi/abs/10.1177/0278364912455072) · [IROS 2011 (PDF)](http://www2.informatik.uni-freiburg.de/~stachnis/pdf/kretzschmar11iros.pdf) · [Lifelong map learning](https://link.springer.com/article/10.1007/s13218-010-0034-2)
Discard the *laser scans* (nodes) whose expected mutual information with the occupancy map is
lowest, then marginalize their poses with Chow-Liu tree approximation. Result: graph grows only
when the robot gains new information about the environment — scales with environment size, not
trajectory length; the canonical statement of the O(area)-not-O(t) goal for laser SLAM.
RESULTS cross-ref: our cell-cap eviction is the *map-side* twin of this (we evict segment
vectors, they evict scans); our graph bookkeeping still O(t) is the half they solved.
*Actionable:* their selection criterion (expected information gain of a node w.r.t. the map)
transfers to our eviction policy: evict the segment whose removal least changes the SSP map
bundle (smallest incremental coherence contribution), instead of pure age/cap order.

**Factor descent sparsification — Vallvé, Solà, Andrade-Cetto, RAS 2019.**
[Paper (PDF)](http://www.iri.upc.edu/files/scidoc/2193-Pose-graph-SLAM-sparsification-using-factor-descent.pdf) · [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0921889018303002)
Iterative coordinate-descent on the KLD objective, one factor at a time, for *populated* (non-
tree) replacement topologies — cheaper than NFR's joint convex solve at equal KLD, and studies
which topologies are worth populating in pose-graph-specific settings.
*Actionable:* if NFR's convex solve is too heavy per-eviction at our rates, factor descent is
the budget implementation of the same guarantee.

### 1.4 Early stopping as regularization — the theory behind max_nfev=30

**Verdict up front: YES, this is a known, named, and theorem-backed phenomenon** — the
optimization/inverse-problems literature calls it *iterative regularization* (the general
principle), *semiconvergence* (the failure mode of not stopping), and the stopped methods are
"regularizing Newton-type iterations." Our finding is a clean instance, citable as follows.

**Semiconvergence + discrepancy principle — Engl, Hanke, Neubauer, *Regularization of Inverse Problems*, Kluwer 1996; survey: Benning & Burger 2018.**
[Modern regularization methods survey (arXiv 1801.09922)](https://arxiv.org/pdf/1801.09922) · [Stopping rules for Landweber-type iteration](https://www.researchgate.net/publication/231057995_Stopping_rules_for_Landweber-type_iteration)
For ill-posed least squares, iterative solvers (Landweber, CG, Gauss-Newton variants) first
reduce error in well-conditioned (data-determined) directions, then start fitting noise/null-
space directions — error vs iteration is U-shaped ("semiconvergence"), and the iteration index
is *literally the regularization parameter*. The principled stopping rule is Morozov's
discrepancy principle: stop at the first iterate whose residual falls to ~tau x noise level
(tau slightly > 1). This is our fr079 story exactly: fully-converged TRF (300 evals) slides to
12.7 m down data-unconstrained valley directions; truncation at 30 evals stops before the
noise-fitting phase.
*Actionable:* replace the fixed max_nfev=30 with a discrepancy stop — terminate the TRF solve
when the weighted residual RMS reaches the known sensor noise floor (our seq-edge sigmas give
the noise level directly). This converts a magic constant into a self-tuning rule and removes
the scipy-pin fragility RESULTS flags as load-bearing.

**Regularizing Levenberg-Marquardt — Hanke, *Inverse Problems* 13:79-95, 1997.**
[IOPscience](https://iopscience.iop.org/article/10.1088/0266-5611/13/1/007) · [KLUEDO record](https://kluedo.ub.rptu.de/frontdoor/index/index/docId/4861)
Proves that Levenberg-Marquardt / trust-region-style damped Gauss-Newton, with damping chosen
by an inexact-Newton criterion and iteration stopped by the discrepancy principle, is an
optimal-order *regularization method* for nonlinear ill-posed problems. This is the closest
published theorem to our empirical finding: trust-region step control + truncation = implicit
regularizer, no explicit prior needed. It also explains the TRF-vs-plain-GN bisection result
(RESULTS: TRF stops 0.5% higher in cost, 3.7x better in ATE): the trust region *is* part of
the regularization, per-iteration.

**Truncated/inexact Newton-CG — Hanke 1997/2010; inexact Newton regularization — Rieder et al.**
[Inexact Newton regularization in Hilbert scales (arXiv 1009.3868)](https://arxiv.org/pdf/1009.3868) · [General convergence analysis of inexact Newton for nonlinear inverse problems (arXiv 1010.3435)](https://arxiv.org/pdf/1010.3435)
The *inner*-iteration analogue: truncating the CG solve of each Gauss-Newton step acts as a
spectral filter on that step — the well-conditioned components of the step are applied first
and the sloppy ones never computed. This is verbatim the mechanism RESULTS deduced ("the trust
region applies the data-consistent components of the step first and quits before the sloppy
ones") — it has a name (regularizing inexact Newton / Newton-CG) and convergence-rate theorems.

**Iteratively regularized Gauss-Newton (IRGNM) — Bakushinskii 1992; Kaltenbacher, Neubauer, Scherzer, *Iterative Regularization Methods for Nonlinear Ill-Posed Problems*, de Gruyter 2008.**
[Book record](https://books.google.com/books/about/Iterative_Regularization_Methods_for_Non.html?id=_uNUAAAAYAAJ) · [IRGNM in Banach spaces (arXiv 1306.2191)](https://arxiv.org/pdf/1306.2191)
IRGNM adds a *decaying* Tikhonov term alpha_n(x_0 - x_n) to each Gauss-Newton step — the
explicit-prior route done right: the prior weight must shrink along the iteration path, and
the anchor is x_0 (a fixed reference), not the current linearization point. RESULTS cross-ref:
our Tikhonov sweep failed with *static* lambda toward pre-relax poses; the theory says static
lambda is the wrong object — a schedule alpha_n -> 0 with discrepancy stopping is the theorem-
carrying version, and RESULTS' conclusion that "a regularizer reproducing early stopping would
need to be trajectory-aware" is precisely the known distinction between Tikhonov (static) and
iterative (path-dependent) regularization.
*Actionable:* if an explicit regularizer is ever wanted again (e.g., for solver portability),
implement IRGNM's decaying schedule, not fixed-lambda Tikhonov — the sweep already run does
not falsify IRGNM.

**Early-stopped gradient descent ~ ridge path (statistics/ML branch) — e.g., Ali, Kolter, Tibshirani AISTATS 2019; risk bounds arXiv 2503.03426; momentum variant arXiv 2201.05405.**
[Sharp risk bounds for early stopping (arXiv 2503.03426)](https://arxiv.org/pdf/2503.03426) · [Momentum GD with early stopping (arXiv 2201.05405)](https://arxiv.org/abs/2201.05405) · [Cross-validation for early-stopped GD (Tibshirani)](https://www.stat.berkeley.edu/~ryantibs/papers/gradcv.pdf)
For linear least squares, iterate k of gradient descent is a spectral filter ~ ridge with
lambda ∝ 1/(step*k), with per-eigendirection shrinkage (1-(1-eta*s_i)^k) vs ridge's
s_i/(s_i+lambda) — similar but *not identical* filters; the equivalence is direction-wise
approximate, and breaks under nonlinearity/IRLS coupling. This is the modern citation for why
our Tikhonov-vs-early-stop discrepancy is expected even in the linear picture, before the
IRLS/LOO cascade path-dependence RESULTS documents.

---

## Cluster 2 — Lifelong / bounded-memory mapping

**RatSLAM persistent operation — Milford & Wyeth, "Persistent Navigation and Mapping using a Biologically Inspired SLAM System," IJRR 2010.**
[Semantic Scholar](https://www.semanticscholar.org/paper/49c9a64c47b20426356e560f992e7afa9171dd34) · [Experience mapping origin (ACRA 2005)](https://www.araa.asn.au/acra/acra2005/papers/milford.pdf) · [Overview slides](https://www.cs.ubc.ca/labs/lci/robuds/slides/2014-09-16-ratslam.pdf)
The closest biological cousin to our approach (continuous attractor pose network + graph of
"experiences" relaxed by a spring-like update — their experience map relaxation is a gradient-
descent pose-graph solve, incidentally also run truncated, a fixed small number of iterations
per cycle). Their two-week autonomous office-delivery run is the landmark lifelong result; its
map-maintenance lesson: cap experience *density* per unit area and delete at random when over
cap — crude, but sufficient for weeks of operation, and the direct precedent for our cell-cap
eviction (theirs is a node cap, ours a map-vector cap). Reported weakness: pruning interacts
badly with cyclic appearance change (day/night) — distinct-condition experiences get evicted
as "duplicates."
*Actionable:* cite RatSLAM's density cap as precedent for O(area) eviction, and adopt their
negative result: eviction policy must not key on spatial density alone once the map must hold
multiple conditions/passes — our pass-segregated segments already avoid this trap; keep it.

**Experience-Based Navigation — Churchill & Newman, IJRR 2013.**
[IJRR](https://journals.sagepub.com/doi/abs/10.1177/0278364913499193) · [ORI project page](https://ori.ox.ac.uk/news/experience-based-navigation)
"Plastic maps": store multiple parallel *experiences* of the same place; add a new experience
exactly when localization against existing ones fails. Failure-driven growth means storage
grows with environmental *diversity*, not time — their bounded-growth argument (saturation as
conditions are covered) is a different, complementary boundedness claim from our O(area) one.
*Actionable:* their growth trigger inverts our eviction question — a principled *admission*
policy (only fold a new segment where frontend match quality is degraded) would slow cell-cap
pressure before eviction is ever needed.

**Work Smart, Not Hard — Linegar, Churchill, Newman, ICRA 2015.**
[Paper (PDF)](https://www.robots.ox.ac.uk/~mobile/Papers/2015ICRA_Linegar.pdf)
EBN follow-up: fixed compute budget per frame, learned prediction of *which* stored experiences
to load ("recall relevant experiences") — memory can be unbounded on disk while the working set
is bounded; performance degrades gracefully with budget.
*Actionable:* the pattern "bounded working set + relevance-triggered recall" is the missing
piece of our drought story: coarse-ring relocalization should act as the recall trigger that
temporarily pulls old co-located segments into the frontend recency set (keeping RESULTS' law:
recency stays small, recall is event-driven, not a widened window).

**RTAB-Map memory management — Labbé & Michaud, JFR 2019 (memory mgmt IROS 2011/2013).**
[JFR paper (HTML)](https://arxiv.org/html/2403.06341v1) · [Memory management paper](https://arxiv.org/html/2407.15890v1) · [Project](http://introlab.github.io/rtabmap/)
The production-grade bounded-compute SLAM: Working Memory (graph nodes eligible for loop
closure, bounded so per-frame time meets a real-time cap) vs Long-Term Memory (evicted nodes
in a database); eviction by a weighted recency/importance policy, and *retrieval* brings LTM
neighborhoods back into WM when a WM loop closure lands nearby. Compute-bounded rather than
memory-bounded — the map still grows on disk, so it does not make our O(area) claim; its
lesson is that eviction is safe only if paired with retrieval.
*Actionable:* our eviction is currently irreversible (vectors superposed/discarded); RESULTS'
MIT failure (no closures in the final hour) argues for the RTAB-Map split — evicted-but-
archived anchor summaries (even just poses + coarse ring signatures, a few hundred bytes each)
enable drought recall at negligible memory cost while the O(area) claim continues to hold for
the *map* proper.

**Dynamic pose-graph SLAM in low-dynamic environments — Walcott-Bryant, Kaess, Johannsson, Leonard, IROS 2012.**
[Referenced in lifelong-SLAM survey results](https://www.researchgate.net/publication/220634182_Lifelong_Map_Learning_for_Graph-based_SLAM_in_Static_Environments)
Maintains active vs inactive node sets and removes nodes invalidated by environmental change;
explicitly notes graph size is *not* bounded during exploration of new areas — an honest
statement of the same boundary condition our O(area) claim has (new area = new cells).
*Actionable:* when writing up the O(area) claim, adopt their framing: bounded *revisit* cost,
linear-in-area exploration cost — it preempts the obvious referee objection.

**ElasticFusion — Whelan, Salas-Moreno, Glocker, Davison, Leutenegger, RSS 2015 / IJRR 2016.**
[RSS paper (PDF)](https://www.roboticsproceedings.org/rss11/p01.pdf) · [IJRR (PDF)](https://www.thomaswhelan.ie/Whelan16ijrr.pdf)
Map-centric SLAM with *no pose graph*: loop closures apply non-rigid corrections directly to
the surfel map through a sparse deformation graph; the map itself is the state being bent.
The instructive contrast for us: RESULTS' hierarchical-map experiment failed precisely because
world-frame sums "never move with the graph" — ElasticFusion's answer is to make the map
deformable rather than the trajectory. Our SSP algebra actually supports a cheap version:
phase multiplication can *re-transform* a segment vector under a pose correction exactly.
*Actionable:* a "deformation" pass that re-phases evicted/merged world-frame content after
large graph corrections is the SSP-native analogue of their deformation graph — worth a bench
probe before abandoning world-frame tiers entirely (would attack the ~18 cm structural floor
RESULTS identified).

**MS-SLAM: sliding-window map sparsification — Zhang et al., JFR 2025.**
[Wiley](https://onlinelibrary.wiley.com/doi/10.1002/rob.22431)
Recent visual-SLAM evidence that online map-point eviction under a sliding-window redundancy
test preserves tracking accuracy at large memory savings; useful as a contemporary datapoint
that bounded-map claims are publishable when paired with accuracy-parity evidence — the same
G1-vs-HY4 parity methodology RESULTS already uses.
*Actionable:* mirror their evaluation template (memory curve vs run length, ATE parity) for
the eviction-soak experiment that RESULTS says the O(area) claim still awaits.

**Cartographer — Hess, Kohler, Rapp, Andor, ICRA 2016.**
[Google Research (PDF)](https://research.google.com/pubs/archive/45466.pdf) · [Algorithm walkthrough](https://github.com/cartographer-project/cartographer_ros/blob/master/docs/source/algo_walkthrough.rst)
2D lidar SLAM whose loop closure is a depth-first branch-and-bound search over a precomputed
multi-resolution stack of the submap — global (wide-window) scan-to-map matching cheap enough
to run on *every* keyframe against nearby submaps, which is why Cartographer does not have a
closure-drought problem: candidates are generated exhaustively, then accepted only above a
match-score floor, and the robust (Huber) pose graph absorbs the rest. Already queued in
RESULTS as a borrowing; the MIT failure raises its priority to top.
*Actionable:* our coarse octave rings + exact lattice rotations give a native B&B structure
(coarse ring = low-resolution level; permutations enumerate rotation branches with one encode)
— implement drought-triggered branch-and-bound over (theta-permutation x coarse-cell) with the
fine rings as the leaf-level verifier.

**Scan Context — Kim & Kim, IROS 2018 (Scan Context++, T-RO 2022).**
[Code/overview](https://github.com/chenchiWHU/scancontext) · [Survey context](https://arxiv.org/pdf/2405.04812)
The standard lidar place-recognition descriptor (polar occupancy signature, rotation handled
by column shift — structurally a cousin of our ring/angle lattice, which supports rotation by
*exact permutation*). Their experience: descriptor retrieval proposes, geometric verification
disposes; retrieval recall is what prevents closure droughts in revisit-sparse runs.
*Actionable:* our coarse incommensurate rings already are a compact place descriptor — build a
small kd/brute index of per-anchor coarse-ring signatures and query it during droughts as the
candidate generator feeding the B&B verifier (retrieval + verification split, per the field's
consensus architecture).

---

## Top-5 actionables

**1. MIT sparse-loop fix (specific recommendation).** Three-stage drought pipeline:
(i) *generate* — index per-anchor coarse-ring signatures (Scan-Context-style retrieval; our
rings are already the descriptor) and query when closure-drought or drift-budget triggers;
(ii) *verify* — Cartographer-style branch-and-bound over rotation permutations x coarse cells,
descending to fine rings only for surviving branches (one encode per rotation branch — our
exact-permutation trick makes B&B unusually cheap here); (iii) *admit* — hold candidates in a
buffer and run PCM pairwise cycle-consistency (Mangelson ICRA'18) so a drought-ending closure
is only inserted as part of a mutually consistent pair/clique, protecting against the high-
drift-alias risk that makes single drought closures dangerous. Keep evicted-anchor coarse
signatures archived RTAB-Map-style (~bytes each) so the final-hour drought can still close
against hour-one geometry.

**2. Early-stopping verdict: solidly grounded — cite it, then upgrade it.** The phenomenon is
*iterative regularization / semiconvergence* (Engl-Hanke-Neubauer 1996); truncated trust-
region Gauss-Newton specifically is proven an optimal-order regularization method (Hanke,
Inverse Problems 1997; inexact Newton-CG analyses), and the "well-conditioned step components
applied first" mechanism RESULTS deduced is the textbook spectral-filter explanation. The
Tikhonov-sweep failure is also predicted: static Tikhonov ≠ the iteration-path filter; the
theory-matching explicit alternative is IRGNM's *decaying* alpha_n schedule, and the principled
replacement for max_nfev=30 is Morozov's discrepancy stop (terminate when weighted residual
hits the noise floor given by our edge sigmas) — removes the magic constant and the scipy pin.

**3. Consensus clustering of alias families = PCM + DC-GM's correlation insight.** Implement
pairwise cycle-consistency (chi2 on composed edge-odometry-edge-odometry cycles, seq-chain
covariances) -> consistency graph -> max clique as trusted family; treat near-identical-
transform edge groups as single units (DC-GM correlation modeling) so a whole alias family is
accepted/rejected atomically; use Group-k (Forsgren IJRR 2024) if pairwise proves too weak on
the repetitive-bay case; benchmark on Vertigo's spoiled standard datasets.

**4. Eviction marginalization with a fidelity number.** On evicting an anchor with loop edges,
emit Chow-Liu-tree (GLC) or NFR-recovered relative factors over its Markov blanket instead of
plain seq-edge composition, and log the KLD to the dense marginal — this supplies the "fidelity
guarantee" for the marginalization half of the O(area) claim, and GLC/Kretzschmar are the
citations that graph growth tracking environment-area-not-time is achievable and publishable.

**5. Position the cell cap in the lifelong lineage.** RatSLAM's experience-density cap
(random eviction, two-week run) is the direct precedent for cell-cap eviction; RTAB-Map's
WM/LTM shows eviction should be paired with cheap recall; EBN shows admission control
(only add where matching degrades) delays eviction pressure; Kretzschmar's info-gain criterion
is the smarter-than-age eviction policy. Frame our claim as: bounded map memory O(area) with
*information-aware* eviction — stronger than RatSLAM's random deletion, memory-bounded where
RTAB-Map is only compute-bounded — and run the MS-SLAM-style memory-vs-accuracy soak to
certify it.
