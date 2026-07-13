# VSA / HDC / SSP spatial representation — state of the art vs. this codebase

Literature scout for the SSP-Fourier-feature 2D lidar SLAM stack (polar frequency
lattice, rotation = lattice permutation + d/dθ companion vectors, bounded segment-vector
maps, range-gated encoding, hierarchical tiers, IRLS/LOO pose graph). Cross-references to
`RESULTS.md` findings are by number. Written 2026-07-07.

Legend for cross-refs: F1–F8 = the numbered "Findings" section; R1/R2 = hierarchical-map
refinements; "ring-ratio" = F7 golden-ratio result.

---

## A. Capacity / bundling theory (sizing the hand-tuned knobs)

### 1. Plate — Holographic Reduced Representations (HRR / FHRR)
- **Citation.** T. A. Plate, *Holographic Reduced Representations: Distributed Representation
  for Cognitive Structures*, CSLI/Univ. Chicago Press, 2003 (and *IEEE TNN* 6(3), 1995).
  Book: https://press.uchicago.edu/ucp/books/book/distributed/H/bo3643252.html
- **Relevance.** Our map vector IS an FHRR object: unit-magnitude complex phasors, translation =
  phase multiplication (fractional binding), unbinding = conjugate. Plate's original theory says
  binding capacity grows ~linearly with dimension d for the *projected/unitary* variant — which is
  exactly our unitary complex features — so our D=360 has a principled capacity ceiling rather than
  a purely empirical one.
- **Actionable.** Treat each segment vector as an FHRR bundle of K phasor-encoded samples and use
  Plate's linear-in-d capacity to set a per-segment sample cap: stop folding samples into one
  rigid segment once K approaches the FHRR fidelity knee for d=720 real dims, instead of the
  current fixed 5-frame segment length.

### 2. Frady, Kleyko, Sommer — superposition/bundling capacity theory
- **Citation.** E. P. Frady, D. Kleyko, F. T. Sommer, "A theory of sequence indexing and working
  memory in recurrent neural networks," *Neural Computation* 30(6):1449–1513, 2018.
  arXiv:1803.00412. Companion: "Theory of the superposition principle for randomly generated
  vectors," arXiv:1707.01429.
- **Relevance.** Gives the closed-form retrieval SNR for pulling one item out of a bundle of M
  vectors in dimension N: **s = √(N/M)**, with accuracy p_corr = ∫ φ(h)[Φ(h+s)]^(D−1) dh and the
  high-fidelity form **s² = 4[ln(D−1) − ln(2ε)]**; practical ceiling ≈ ½ bit/dimension. This is
  the theory the RESULTS "related work" flagged as the way to *derive* window/cell-cap sizes
  instead of hand-tuning them (F: bounded-memory soak; R1 cell caps).
- **Actionable.** Invert s=√(N/M) with target ε: for the local-map bundle (N=720 real dims), the
  max reliably-decodable number of superposed segments before crosstalk swamps a match is
  M ≈ N/s². Use it to set the frontend recency window (`recent_aids`) and the tier-cell EMA
  half-life analytically — directly attacks the F "recency small / gap large" tuning and the R1
  "loop band must live in slow memory" dwell-ratio hand-set (eps_global=0.002).

### 3. Kleyko et al. — VSA survey (context / vocabulary anchor)
- **Citation.** D. Kleyko, D. Rachkovskij, E. Osipov, A. Rahimi, "A Survey on Hyperdimensional
  Computing aka Vector Symbolic Architectures, Parts I–II," *ACM Computing Surveys*, 2023.
  arXiv:2111.06077.
- **Relevance.** Canonical map of binding/bundling/permutation operators; confirms our
  "rotation = permutation" is the *permutation* primitive of VSA applied to a *structured*
  (non-random) basis — an unusual but not unheard-of combination. Useful for framing claims in
  VSA-native language for any writeup.
- **Actionable.** Adopt the survey's operator taxonomy in the paper's method section so reviewers
  place the contribution correctly (structured-basis permutation + FPE translation), and cite it
  as the baseline against which the group-closed lattice is the novelty.

---

## B. Search-free decoding / factorization (relocalization backend)

### 4. Frady, Kent, Olshausen, Sommer — Resonator Networks
- **Citation.** E. P. Frady, S. J. Kent, B. A. Olshausen, F. T. Sommer, "Resonator Networks 1 & 2,"
  *Neural Computation* 32(12):2311–2419, 2020. arXiv:1906.11684; rctn.org/bruno/papers/resonator1.pdf.
- **Relevance.** Solves c = x⁽¹⁾ ⊙ … ⊙ x⁽ᶠ⁾ (composite = Hadamard product of factors) by
  iterating x̂⁽ᶠ⁾[t+1] = sgn(Xf Xf† (ô⁽ᶠ⁾[t] ⊙ c)) — "search in superposition," ~100× faster than
  optimization, operational capacity M_max ∝ N². Our relocalization currently does a coarse-to-fine
  grid search over (θ, d). A resonator factors (rotation-index) × (x-phase) × (y-phase)
  simultaneously without the sweep.
- **Actionable.** Prototype a resonator for wide-mode relocalization: codebooks = {rotation
  permutations} × {x translation phasors} × {y translation phasors}; recover pose factors from one
  query⊙map product. Targets the F/R "does not yet snap large sparse loops" and corridor
  false-closure holes, and would replace the fixed prior window with N²-capacity factorization.

### 5. Kymn et al. — Residue Hyperdimensional Computing (CRT moduli)
- **Citation.** C. J. Kymn, D. Kleyko, E. P. Frady, C. Bybee, P. Kanerva, F. T. Sommer, B. Olshausen,
  "Computing with Residue Numbers in High-Dimensional Representation," *Neural Computation*, 2024/2025.
  arXiv:2311.04872; rctn.org/bruno/papers/kymn_residue_NECOreprint.pdf.
- **Relevance.** Formalizes exactly our "incommensurate relocalization rings" hunch (RESULTS related
  work already cites this). Integer x → z(x)=z^x with per-modulus phases multiples of 2π/m; **moduli
  must be pairwise coprime**, unambiguous range = product ∏mₖ, memory ∑mₖ not ∏mₖ, decode by
  resonator. Their kernel is 1 at Δx≡0 (mod M), ≈0 elsewhere — the extended-range unwrapping our
  coarse rings need (F7: "incommensurability matters only for wide-window relocalization rings").
  Also builds a **hexagonal residue** giving 3m²−3m+1 states from 3m codebook vectors vs m² from 2m
  for square — a resolution argument distinct from our registration-peak findings.
- **Actionable.** Choose the coarse-ring wavelengths as a pairwise-coprime (residue) set sized so
  ∏mₖ covers the max relocalization drought distance (the MIT-scale ~1.9 km / final-hour dead zone),
  and decode ring phases with a small resonator — turns the ad-hoc "wide mode on coarse incommensurate
  rings" into a CRT with a provable unambiguous range. Note: our F7 says coprimality buys nothing
  *inside* the matched band; apply this ONLY to the relocalization ring set.

---

## C. SSP spatial-representation line (the direct predecessors)

### 6. Komer, Stewart, Voelker, Eliasmith — SSP / fractional binding
- **Citation.** B. Komer, T. C. Stewart, A. Voelker, C. Eliasmith, "A neural representation of
  continuous space using fractional binding," *CogSci* 2019.
  compneuro.uwaterloo.ca (Lu et al. 2019 companion "Representing spatial relations with fractional
  binding").
- **Relevance.** Defines S(x,y)=X^x⊗Y^y with unitary Fourier coefficients — the encoder family our
  polar-lattice encoders specialize. Establishes that SSP dot product ≈ a kernel (kernel-density
  view), which is the theoretical frame for our correlation-peak registration objective.
- **Actionable.** Frame the registration score Re⟨Map, e^{iWd}·Scan_θ⟩ explicitly as an SSP kernel
  evaluation; then peak *curvature* is the kernel Hessian and gives edge information matrices for
  free (the queued Olson-ICRA'09 borrowing). One code hook: return the analytic Hessian of the
  matched kernel as the pose-graph edge weight.

### 7. Voelker — dot product between rotated Fourier transforms (the kernel/rotation link)
- **Citation.** A. R. Voelker, "A short letter on the dot product between rotated Fourier transforms,"
  arXiv:2007.13462, 2020.
- **Relevance.** Proves E[A·B] = ∏ₖ sinc(aₖ) for displacement a (correcting the earlier "Gaussian"
  conjecture), with the exact form A·B = (1/d)Σⱼ cos(Σₖ Θ_{k,j} aₖ). Rotation of an SSP = applying a
  2D rotation matrix to the columns of the phase matrix K. **Crucially for us: this kernel analysis
  predicts the *displacement* similarity envelope but says nothing about *angular sampling* sharpness**
  — so the SSP line does NOT predict our F3/F4/F6 (need ≥6 well-spread axes; hex triples add nothing;
  angle density is environment-coupled). Those are registration-specific and appear genuinely outside
  the published SSP kernel theory.
- **Actionable.** Use the sinc-product kernel to set lam_min analytically (first sinc null at
  a=1 ⇒ resolution ≈ λ) — corroborates F5's "lam_min ~ noise scale, 0.5 costs 3×" from theory rather
  than sweep, and predicts the F5 "gaps > ~2 octaves reintroduce sidelobes" as sinc sidelobe overlap.

### 8. Dumont & Eliasmith — hexagonal SSP "optimality" (the claim our data contradicts)
- **Citation.** N. Dumont, C. Eliasmith, "Accurate representation for spatial cognition using grid
  cells," *CogSci* 2020. compneuro.uwaterloo.ca/files/publications/dumont.2020.pdf.
- **Relevance.** Claims hexagonal (3-axis, 60°-spaced) SSP bases *optimize* accuracy — "grid cells
  form the optimal basis for decoding" place cells. **Our F3/F4 directly contradict this for
  registration**: 3 axes give a 3-ridge kernel (heading great, translation slides >1 m), and hex
  triples add nothing over a polar grid at equal D. Reconciliation (important, and defensible): their
  optimality is for *place-cell readout / path-integration decoding accuracy* under a fixed neuron
  budget, a different objective than *scan-to-map correlation-peak localizability*. Their kernel
  analysis (sinc products, per #7) does **not** contain an angular-localizability term, so it does not
  predict our 3/5/7-axis findings — the contradiction is between two different optimality criteria,
  not a factual clash.
- **Actionable.** State this explicitly in the writeup: hex is optimal for single-module decoding
  density (Dumont/Eliasmith; and the residue hex of #5), NOT for correlation-peak registration, where
  ≥6 axes matched to the environment's angular spectrum win (F6). Optionally test their exact hex
  basis at *matched D* on our registration bench to quantify the ridge pathology as a figure.

---

## D. Grid-cell scale theory (the golden-ratio finding, cross-referenced)

### 9. Wei, Prentice, Balasubramanian — optimal grid scale ratio √e
- **Citation.** X.-X. Wei, J. Prentice, V. Balasubramanian, "The sense of place: grid cells and the
  transcendental number e," arXiv:1304.0031; *eLife* 4:e08362, 2015 ("A principle of economy predicts
  the functional architecture of grid cells").
- **Relevance.** Derives optimal adjacent-scale ratio = **√e ≈ 1.649** (robustly 1.4–1.7 for realistic
  noise), grid scales in geometric progression, triangular lattice. This is a striking corroboration of
  our F7: our best ring ratio is a flat plateau r≈1.4–2.8 with octaves (r=2) winning, and we explicitly
  note **√e = 1.6487 is fine (1.35 cm)** — i.e. our empirical optimum brackets the grid-cell theoretical
  optimum. Different objective (neuron economy vs registration), same number.
- **Actionable.** Cite √e as independent theoretical support for the F7 octave/near-2 plateau, and add
  a bench point at exactly r=√e to show it sits on the plateau (predicted ~1.3 cm) — cheap, strong
  figure connecting our result to normative grid theory.

### 10. Vágó & Ujfalussy — golden-ratio interference in grid codes (the F7 twin)
- **Citation.** L. Vágó, B. B. Ujfalussy, "Robust and efficient coding with grid cells," *PLOS
  Computational Biology* 14(1):e1005922, 2018. PMC5774847; biorxiv 107060.
- **Relevance.** **This is the closest prior art to our F7 golden-ratio result.** They give a
  number-theoretic analysis of multi-scale grid coding range and find interference (decoding ambiguity)
  when the scale ratio α = σ ≈ **1.618 = golden ratio**, tied to how well the ratio is approximated by
  rationals with small denominators (Fibonacci convergents). So **the phenomenon "golden ratio is bad
  for multi-scale spatial codes" is KNOWN.** BUT their stated mechanism is Diophantine (rational
  approximability), whereas our F7 mechanism is *additive resonance* φ²=φ+1 (each frequency = sum of the
  two below, aligning three-wave interference in the correlation), with the Fibonacci convergent 8/5
  failing *separately* as a rational coincidence. These are two mechanisms both landing on φ.
- **Actionable.** Reframe F7's novelty carefully (see novelty assessment (c) below): claim the
  *additive-resonance* explanation and the *registration-correlation* setting, and cite Vágó–Ujfalussy
  as independent confirmation of the empirical φ-is-bad phenomenon in the grid-cell setting. Run the
  probe that separates the two mechanisms: test √e (irrational, well-approximable near 1.649 by 5/3?) vs
  φ — if only φ (additive) fails and other equally-approximable ratios don't, that isolates additive
  resonance from Diophantine, which would be a genuinely new sub-claim.

---

## E. Rotation-equivariant / transformation VSA (novelty falsification targets)

### 11. Krausse, Neftci, Sommer, Renner — Grid-Cell Structured Vector Algebra ★ CONCURRENT PRIOR ART
- **Citation.** S. Krausse, E. Neftci, F. T. Sommer, A. Renner, "A Grid Cell-Inspired Structured Vector
  Algebra for Cognitive Maps," NICE 2025. arXiv:2503.08608.
- **Relevance.** **The single most important find for our novelty claims.** They implement 2D rotation
  in a grid-cell VSA as a **permutation of module activity** ("shifting the activity in the first column
  of cube modules into the second column") — i.e. the discrete-rotation-as-permutation idea we believed
  novel exists in concurrent 2025 work. HOWEVER, key differences preserve most of our claim: (i) their
  lattice is **NOT proven closed** under a discrete rotation group — rotation is *approximate*, needs
  dense generator sampling (nθ=23) and is *limited to points near the rotation center* ("a common
  limitation of rotations in VSA space"); (ii) their sub-step handling is a **circular convolution with
  a localized bump vector V_R + fractional power encoding**, NOT an analytic d/dθ derivative vector;
  (iii) application is cognitive maps / family-tree reasoning, **not lidar scan registration or SLAM**.
- **Actionable.** Must-do: revise RESULTS "Related work" — the flat claim that rotation-as-permutation
  is "found nowhere in the VSA/registration literature" is now false (concurrent, cognitive-maps
  domain). Re-scope our novelty to (a) *exact group-closed* lattice with *no near-center limitation* for
  *rigid segment vectors*, and (b) the *analytic derivative* companion vs their bump-convolution. Directly
  compare against their VR-bump interpolation as an ablation arm (our d/dθ vs their convolution) — the
  RESULTS ablation already shows d/dθ is load-bearing (4× on the bench); benchmarking it against VR is the
  clean novelty-isolating experiment.

### 12. Kleyko, Rachkovskij et al. — shift-equivariant similarity-preserving hypervectors
- **Citation.** D. Kleyko et al., "Shift-Equivariant Similarity-Preserving Hypervector Representations of
  Sequences," *Cognitive Computation*, 2024. arXiv:2112.15475; and "Representation of spatial objects by
  shift-equivariant similarity-preserving hypervectors," *Neural Comput. & Appl.*, 2022.
- **Relevance.** Prior art on *translation*-equivariant (shift) structured hypervectors via permutation,
  the 1D/2D cousin of our translation handling. Establishes that permutation-based equivariance for
  *translation* is well-trodden — which sharpens where our novelty actually lies: *rotation* closure, not
  translation.
- **Actionable.** Cite as the translation-equivariance baseline and explicitly contrast: our contribution
  is the *rotation*-closed lattice, since translation-equivariant permutation VSA already exists.

### 13. Fourier–Mellin scan-matching (the image-domain cousin, already noted in RESULTS)
- **Citation.** P. Checchin et al., "Radar Scan Matching SLAM Using the Fourier-Mellin Transform,"
  *FSR* 2010; Reddy & Chatterji, "An FFT-based technique for translation, rotation, and scale-invariant
  image registration," *IEEE TIP* 5(8), 1996.
- **Relevance.** Fourier–Mellin decouples rotation/scale/translation via log-polar phase correlation —
  the image-domain analogue of our polar-lattice permutation rotation (RESULTS already flags this). It is
  a dense-image method, not a fixed-size algebraic vector map with no sensor history, which is our
  distinction.
- **Actionable.** Use Fourier–Mellin as the "known cousin" framing in the writeup and, if a reviewer
  pushes, run a log-polar phase-correlation baseline to show the VSA-vector formulation matches its
  registration accuracy at bounded memory / no image retention.

---

## F. Neuromorphic / spiking SSP-SLAM (deployment target)

### 14. Dumont, Furlong, Orchard, Eliasmith — spiking neural SSP-SLAM
- **Citation.** N. S.-Y. Dumont, P. M. Furlong, J. Orchard, C. Eliasmith, "Exploiting semantic
  information in a spiking neural SLAM system," *Frontiers in Neuroscience* 17:1190515, 2023.
  Code: github.com/nsdumont/Semantic-Spiking-Neural-SLAM-2023 (Nengo + Nengo-Loihi).
- **Relevance.** The closest prior "SSP SLAM" (RESULTS related work). It does **landmark-level** SLAM with
  given data association + a path integrator on Loihi; it does **not** register raw lidar scans. Our dense
  scan-to-map registration in SSP space is the gap it leaves open. Confirms the neuromorphic viability of
  the SSP substrate our map already lives in.
- **Actionable.** Position our frontend as the "missing raw-sensor registration front-end" for this line,
  and scope a Loihi/Nengo port: our permutation-rotation + phase-multiply translation are both
  neuromorphic-friendly (index shuffle + phasor multiply), unlike ICP/CSM baselines.

### 15. Frady, Sommer — spiking-phasor / resonator neuromorphic substrate
- **Citation.** E. P. Frady et al., "Hyperdimensional Computing with Spiking-Phasor Neurons," NICE/
  neuromorphic venues, 2022–2023.
- **Relevance.** Shows FHRR phasors and resonator dynamics run on spiking hardware — the enabling substrate
  for a neuromorphic version of both our map (phasors) and the proposed relocalization resonator (#4).
- **Actionable.** If the resonator relocalizer (#4) is built, target spiking-phasor implementation so the
  whole stack (encode + register + relocalize) is neuromorphic end-to-end.

---

## G. Multi-wavelength interferometry (the octave-ladder cousin)

### 16. Multi-wavelength / synthetic-wavelength phase unwrapping
- **Citation.** Representative: e.g. *Light: Advanced Manufacturing* review "A review of the dual-wavelength
  technique for phase imaging and 3D topography," 2022; general theory arXiv:1706.04039.
- **Relevance.** Our octave ladder = classic multi-wavelength phase unwrapping (RESULTS related work). The
  synthetic wavelength Λ = λ₁λ₂/|λ₁−λ₂| and the rule "longest synthetic wavelength ≥ 2× max deformation,
  too-few wavelengths risk ambiguity" is exactly F5's "gaps > ~2 octaves reintroduce sidelobes" unwrapping
  condition. **But**: this literature does **not** report a golden-ratio-specific failure — wavelength
  selection there is about coverage/count, not additive resonance. So the interferometry field does NOT
  pre-empt our F7 φ finding.
- **Actionable.** Cite the synthetic-wavelength ambiguity condition as the textbook basis for F5, and note
  explicitly that the F7 additive-resonance failure is *absent* from the interferometry selection rules —
  supporting F7's novelty in that domain.

---

## Top-5 prioritized actionables

1. **Fix the RESULTS novelty claim now (correctness).** Krausse, Neftci, Sommer & Renner (NICE 2025,
   arXiv:2503.08608) already do rotation-as-permutation in a grid-cell VSA. Rewrite the "found nowhere"
   sentence and re-scope our novelty to *exact group-closure + no near-center limit + analytic derivative*
   (see #11 and novelty (a)/(b)). This is a factual error in the current ledger.
2. **Size the tuned windows from Frady superposition capacity** (#2): invert s=√(N/M), ε-target, to set
   `recent_aids`, tier-cell EMA half-life, and per-segment sample cap — replaces the F/R1 hand-tuning with
   a theorem. Highest leverage on the open "recency small / gap large" and eps_global guesswork.
3. **Build the residue/resonator relocalizer** (#4 + #5): coprime coarse rings sized by ∏mₖ ≥ drought
   distance, decoded by a resonator to factor (θ, x, y) without the grid sweep. Targets the standing
   "does not snap large sparse loops" limit and the corridor false-closure hole.
4. **Add the two decisive figures connecting to normative theory**: (i) bench r=√e on the ring-ratio study
   to land it on the F7 plateau (Wei–Balasubramanian #9); (ii) a probe separating additive-resonance from
   Diophantine approximability at φ (Vágó–Ujfalussy #10) to defend F7's mechanism claim.
5. **Kernel-Hessian edge information matrices** (#6/#7): return the analytic curvature of the matched SSP
   kernel as pose-graph edge weights (Olson-ICRA'09 borrowing, now theoretically grounded via the
   sinc-product kernel) — cheap, and attacks the floppy-graph / IRLS-tuning fragility the SHIPPED block
   is built around.

---

## Novelty assessment

**(a) Rotation-closed frequency lattice + permutation rotations — PARTIALLY ANTICIPATED (novelty
reduced, not eliminated).** Concurrent prior art now exists: Krausse et al. (NICE 2025, #11) implement 2D
rotation as a permutation of grid-cell module activity, and shift-equivariant permutation VSA for
*translation* is older still (Kleyko/Rachkovskij, #12); Fourier–Mellin is the classical image-domain cousin
(#13). What still appears distinct: (i) building the lattice to be **exactly closed under a discrete rotation
group so rotation is a bit-exact index permutation** (Krausse's is approximate, dense-sampling-dependent, and
**limited to points near the rotation center** — ours rotates *rigid segment vectors* with no such limit); (ii)
the SLAM/raw-lidar-registration application. Verdict: reframe from "novel" to "an exact, unrestricted
group-closed strengthening of a concurrently-published approximate idea, applied to registration." Honest and
still publishable, but the RESULTS "found nowhere" claim is falsified and must be corrected.

**(b) d/dθ companion (derivative/tangent) vectors for sub-grid rotation — LIKELY NOVEL in mechanism.** The
problem (sub-grid-angle rotation interpolation) is shared with Krausse et al., but their solution is
**circular convolution with a localized bump vector V_R + fractional power encoding**, a fundamentally
different construction from storing the **analytic first-order derivative of the encoding** as a companion
vector and doing a first-order Taylor correction. No VSA/SSP work found stores an explicit d/dθ tangent vector;
the nearest conceptual neighbor (Voelker's rotated-Fourier-transform dot product, #7) does not take the angle
derivative at all. Verdict: the derivative-vector *mechanism* looks genuinely new; de-risk the claim by an
explicit ablation vs the V_R-bump interpolation (our RESULTS already shows d/dθ is load-bearing at 4× — the
head-to-head against Krausse's method is the clean novelty proof).

**(c) Golden-ratio additive-resonance failure — PHENOMENON KNOWN, MECHANISM/SETTING LIKELY NOVEL.** The bare
fact "golden ratio is bad for multi-scale spatial codes" is **already in grid-cell theory**: Vágó & Ujfalussy
(PLOS Comp Bio 2018, #10) find grid-code interference exactly at scale ratio ≈1.618, and Wei–Balasubramanian's
√e≈1.649 optimum (#9) brackets our plateau. **However** their stated mechanism is Diophantine (rational
approximability by Fibonacci convergents), whereas our F7 identifies **additive resonance φ²=φ+1** (three-wave
sum-frequency alignment in the *correlation/registration* objective), with the 8/5 convergent failing as a
*separate* rational-coincidence effect. Multi-wavelength interferometry (#16) does **not** report any
golden-ratio rule at all. Verdict: do not claim the phenomenon as new (grid-cell theory has it); claim (i) the
*additive-resonance* explanation as distinct from the Diophantine one and (ii) its appearance in *scan-to-map
correlation registration* rather than neural population decoding. The separating experiment in actionable #4
would upgrade this from "plausibly novel mechanism" to "demonstrated."
