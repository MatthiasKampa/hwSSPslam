# Golden-ratio dithering / low-discrepancy sampling — the RWTH paper, and what it does (and does not) say about radial golden spacing

Literature retrieval for the F7 ring-ratio result and the in-flight golden-angle dither
sweep. Cross-refs: F7 = RESULTS.md finding 7 (golden RATIO radial ladder uniquely bad,
additive resonance phi^2 = phi + 1); F6 = angle-staggering finding. Written 2026-07-07.

---

## 1. The paper: Schretter & Kobbelt (RWTH Aachen), JGT 2012

- **Citation.** Colas Schretter, Leif Kobbelt (RWTH Aachen University), Paul-Olivier Dehaye
  (ETH Zurich), "Golden Ratio Sequences for Low-Discrepancy Sampling," *Journal of Graphics
  Tools* 16(2):95–104, 2012. DOI: 10.1080/2165347X.2012.679555.
  - RWTH publication page: https://www.graphics.rwth-aachen.de/publication/032/
  - PDF (RWTH graphics group): https://www.graphics.rwth-aachen.de/media/papers/jgt.pdf
  - Publisher: https://www.tandfonline.com/doi/abs/10.1080/2165347X.2012.679555

This is almost certainly "the RWTH golden-ratio dithering paper": it is the only
golden-ratio sampling paper out of RWTH Aachen (Kobbelt's computer-graphics group), and it
is the standard citation behind the phrase "golden-ratio dithering" via Bart Wronski's
widely-read 2016 dithering series (below), which uses exactly this sequence as a dither
generator and cites this paper by name.

### The construction (verified against the PDF)

- **1D golden ratio sequence (a Weyl / additive sequence):**
  `G_s(i) = frac(s + i*phi)`, phi = (1+sqrt(5))/2; equivalently the conjugate
  tau = phi − 1 = 0.6180339887. Implementation is one integer add mod 2^32
  (`+ 2654435769`) per sample.
- **2D "golden point set" on the unit square:** first coordinates are `G_s(i)`, second
  coordinates the SAME sequence under the permutation `sigma_s` that sorts it; `sigma_s`
  is generated incrementally with two consecutive Fibonacci numbers a = F_{2i+1},
  b = F_{2i} (a+b >= N): `sigma(i+1) = sigma(i)+a if sigma(i) <= b else sigma(i)−b`.
- **Unit disc / polar version:** one coordinate interpreted as *normalized radius*, the
  other multiplied by 2*pi as *angle* — the phyllotaxis picture (they cite Coxeter 1953:
  sunflower heads, pineapples). I.e., angles advance by the golden angle; radii are the
  low-discrepancy sequence values in [0,1).

### The precise property it optimizes

- **Discrepancy (even coverage / stratification), NOT a blue-noise spectrum.** The paper's
  claim is that the fractional-parts orbit of phi covers the unit interval "more evenly
  ... for any number of elements" than van der Corput: smaller largest gap, larger minimum
  separation, at every prefix length N. The stated mechanism is number-theoretic: tau is
  "the most irrational" number (continued fraction [0;1,1,1,...], worst rational
  approximations), so "the variability of pairwise distances between points is minimized"
  (three-distance-theorem behavior: gaps take at most three values with bounded ratio).
- The paper explicitly *contrasts* golden sets with blue noise: blue noise is better than
  Hammersley for anti-aliasing, but golden sets beat blue noise for Monte-Carlo
  integration (ambient occlusion) because blue noise loses stratification under density
  warping. Golden sets are called "quasi-regular"/"quasi-lattice": aliasing is avoided by
  irregularity of the additive orbit, not by spectral shaping.
- Wronski's spectral analysis of the same sequence (1D, used as a temporal dither) finds
  it is *not* blue noise at all: the DFT shows spikes at frequencies spaced by factors of
  phi with energy growing by phi per spike — a structured, aliasing-prone spectrum that
  merely happens to be low-energy at low frequencies.

### Domain: spatial (sample positions), additive mod 1 — never a frequency ladder

Everything in the paper lives in the **sample-position domain**: unit interval, unit
square, unit disc (points for a ray tracer / dither offsets). The golden ratio enters
**only additively, modulo 1** — as a rotation number. When mapped to the disc it becomes
the golden **angle** increment plus a low-discrepancy (additively stratified) **radius
value in [0,1)**. Nowhere in the paper — nor in the QMC literature it builds on
(Niederreiter '92; Fibonacci rank-1 lattices, Sloan & Reztsov '02) — is phi used as a
**multiplicative ratio between successive radii or frequencies**. The rank-1 Fibonacci
lattice it discusses uses the *rational* ratio F_{k−1}/F_k as an additive generator, again
mod 1. The paper therefore contains no claim, positive or negative, about geometric
phi-spaced radial ladders; its optimality argument (poor rational approximability of the
rotation number) is logically about additive orbits on a compact group.

## 2. The classics it sits on (the radial-vs-angular ledger)

- **Vogel 1979 (the spiral).** H. Vogel, "A better way to construct the sunflower head,"
  *Mathematical Biosciences* 44:179–189, 1979.
  https://doi.org/10.1016/0025-5564(79)90080-4
  Point n at angle `n * 137.508 deg` (golden angle) and radius `c*sqrt(n)`. Note the
  radius law is **sqrt(n) (area-uniform), not phi^n**: even the canonical "Fibonacci
  spiral" has no geometric golden ladder in radius.
- **Winkelmann et al. 2007 (golden-angle radial MRI — a frequency-domain use).**
  S. Winkelmann, T. Schaeffter, T. Koehler, H. Eggers, O. Doessel, "An Optimal Radial
  Profile Order Based on the Golden Ratio for Time-Resolved MRI," *IEEE Trans. Medical
  Imaging* 26(1):68–76, 2007. https://doi.org/10.1109/TMI.2006.885337
  k-space spokes at azimuth increments of 111.246 deg = pi/phi^2: golden ratio **additive
  in angle**, radius sampled *uniformly along each spoke*. This is the one classic that
  operates in the frequency domain, and it too is purely angular — supporting evidence
  that the community's golden-ratio optimality results are all about angular/additive
  increments, never radial/multiplicative spacing.
- **Keinert et al. 2015 (spherical Fibonacci — Erlangen, not Aachen).** B. Keinert,
  M. Innmann, M. Sänger, M. Stamminger, "Spherical Fibonacci Mapping," *ACM TOG* 34(6),
  SIGGRAPH Asia 2015. https://doi.org/10.1145/2816795.2818131
  Inverse mapping for spherical Fibonacci point sets (Fibonacci lattice on the cylinder,
  area-preserving map to the sphere). Again an additive lattice; included here mainly to
  disambiguate — if the user's memory says "Fibonacci sphere paper," it is this Erlangen
  group, not RWTH.
- **Wronski 2016 (the "golden ratio dithering" meme-source).** B. Wronski, "Dithering
  part two — golden ratio sequence, blue noise and highpass-and-remap," blog, Oct 2016.
  https://bartwronski.com/2016/10/30/dithering-part-two-golden-ratio-sequence-blue-noise-and-highpass-and-remap/
  Uses `frac(i*phi)` as a 1D (temporal/per-frame) dither sequence, cites Schretter &
  Kobbelt, ranks it *above* blue noise for that scenario, but warns it "could alias with
  certain signal frequencies" — i.e., the structured phi-spiked spectrum bites when the
  signal has matching periodicities.
- (Long-shot alternate: an RWTH TCS *student seminar* slide deck on dithering, M. Tirdatov,
  WS 2022/23, https://tcs.rwth-aachen.de/lehre/postscript/WS2022/talk-dithering.pdf — a
  survey talk, not original research. The Schretter–Kobbelt paper is the substantive match.)

## 3. Relation to finding 7 and predictions for the in-flight dither sweep

There is no contradiction between this literature and F7 — the two golden-ratio uses live
on opposite sides of the exp map, and the same continued fraction [1;1,1,...] is the hero
of one and the villain of the other. In Schretter–Kobbelt (and Vogel/Winkelmann), phi acts
**additively mod 1** (a rotation number): all-1s continued-fraction coefficients mean the
orbit's convergents are Fibonacci ratios, which is precisely what makes prefix coverage
maximally even and immune to *rational* resonances — the correct sense in which golden
ANGLES are good. In F7, phi acts **multiplicatively on |k|**: the Fibonacci recurrence
behind those convergents *is* the additive closure phi^2 = phi + 1, so f_{k+2} =
f_{k+1} + f_k exactly, aligning three-wave products in the correlation. The QMC
literature's optimality theorems simply do not transfer — they say nothing protective
about geometric phi ladders, and F7 shows the transfer actually inverts. Note also that a
true Fibonacci-spiral (Vogel) frequency layout has sqrt(n) radii, hence *no* geometric
phi ladder and no systematic sum-closure triples; F7's pathology is specific to the
phi^k ring ladder, not to "Fibonacci layouts" broadly.

**What each theory predicts for the sweep** (golden-angle dithering of the per-ring angle
grids on the golden radial ladder): (a) F7's own three-wave mechanism requires *vector*
closure k_a + k_b = k_c, i.e., collinear triples across rings with |k| ratios phi; rotating
successive rings by golden-angle offsets destroys collinearity, so it predicts a
ratio-*specific* rescue — the golden ladder should collapse from 12.1 cm back toward the
0.9–2.2 cm plateau of its neighbors, while non-resonant ratios barely move. (b) The
Schretter–Kobbelt discrepancy view predicts only a mild, ratio-*independent* gain from
better angular union coverage (largest in curved worlds), plus Wronski's caveat that the
phi-spiked spectrum can itself alias against environment periodicities. (c) F6's empirics
warn of a dither tax in flat-wall worlds: staggering angles across scales previously
*lost* accuracy by forfeiting exact wall-normal hits and rotation-group closure — so the
expected net outcome is rescue-with-a-tax: golden ladder much better than 12.1 cm yet
still behind the aligned octave grid (r = 2, 0.68 cm), and slight degradation of
non-resonant ratios in axis-aligned worlds. Discriminating observable: if golden-angle
dithering does NOT rescue the golden ladder, the resonance is not (only) angular-collinear
three-wave alignment, and a magnitude-only mechanism (e.g., envelope beats in the
per-scale correlation sum) should be suspected instead.
