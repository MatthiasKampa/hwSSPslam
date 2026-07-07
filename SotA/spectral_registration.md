# SotA scout: spectral registration, ladder design, range-dependent reliability, spectral maps

Literature scout, 2026-07-07. Companion to `RESULTS.md` (finding numbers below
refer to its **Findings** section; named sections cited where relevant).
Four areas: (1) Fourier-Mellin / phase-correlation registration for
lidar/radar/sonar, (2) multi-wavelength ladder design under phase noise —
including the PRIORITY question of whether the golden-ratio additive-resonance
hazard (finding 7) has precedent, (3) lidar noise models grounding the
range-gated encoding weight sigma_phi = 2*pi*r*sigma_theta/lam (hier section),
(4) spectral / frequency-domain map representations.

---

## Area 1 — Fourier-Mellin & phase-correlation registration (lidar/radar/sonar)

### 1.1 Checchin, Gerossier, Blanc, Chapuis, Trassoudaine — "Radar Scan Matching SLAM Using the Fourier-Mellin Transform," FSR 2009 (Springer STAR 62, 2010)
<https://link.springer.com/chapter/10.1007/978-3-642-13408-1_14>
First full-scan FMT registration for radar: rotation/scale from the log-polar
magnitude spectrum, translation from phase correlation, no landmarks, feeding
EKF-SLAM. The closest image-domain cousin of our "rotation = exact lattice
permutation" construction — FMT makes rotation a shift in resampled log-polar
coordinates; our lattice makes it a permutation by construction, with no
resampling error. Their rotation/scale-decoupling problem is finding 2's
symmetric-grid requirement seen from the image domain.
**Actionable:** cite as the FMT baseline the lattice-permutation design
supersedes; reuse their whole-scan (no-feature) robustness argument in the
write-up.

### 1.2 Park, Shin, Kim — "PhaRaO: Direct Radar Odometry using Phase Correlation," ICRA 2020
<https://arxiv.org/abs/2001.11384>
Sequential FMT: coarse rotation from down-sampled log-polar phase correlation,
then coarse-to-fine translation phase correlation; 2.34% trans / 2.93 deg
error over 4 km in real time. Validates the theta-then-d decoupling our
permutation matcher uses, at scale on real FMCW data.
**Actionable:** run the coarse rotation sweep on coarse rings only (lam 1–2 m)
before engaging fine rings — cheaper and less alias-prone, complementary to
the existing 2-encode half-step scheme.

### 1.3 Hurtos, Ribas, Cufi, Petillot, Salvi — "Fourier-based Registration for Robust Forward-looking Sonar Mosaicing...," J. Field Robotics 32(1), 2015
<https://researchportal.hw.ac.uk/en/publications/fourier-based-registration-for-robust-forward-looking-sonar-mosai/>
Phase correlation adapted to forward-looking sonar's fan-shaped LIMITED FOV:
they window/mask the fan region before FFT and feed registrations to a pose
graph. Closest published sensor geometry to a 180-deg-FOV lidar doing
frequency-domain registration; the fan boundary dominates the spectrum unless
windowed.
**Actionable:** check whether our hit-to-hit segment sampling leaks an
FOV-edge component at the 180-deg scan boundary (Intel is exactly 180 deg); a
soft weight taper on the first/last few beams is a one-line experiment.

### 1.4 Olson — "Real-Time Correlative Scan Matching," ICRA 2009
<https://april.eecs.umich.edu/pdfs/olson2009icra.pdf>
Exhaustive multi-resolution correlation over (x, y, theta); match covariance =
likelihood-weighted SAMPLE COVARIANCE of the correlation surface. His
canonical failure example is the long hallway producing an elongated
covariance — exactly the corridor aperture problem (finding 3; six-world
battery; corridor false-closure section). Already queued in RESULTS Related
Work; this is the exact mechanism to borrow.
**Actionable:** evaluate the SSP score on the final local (theta, d) grid
around the accepted peak, form the likelihood-weighted second moment, and use
its inverse as the loop-edge information matrix — corridor closures then
self-report rank deficiency along the ridge instead of relying on the brittle
0.55 coherence-ratio veto.

### 1.5 Foroosh, Zerubia, Berthod — "Extension of Phase Correlation to Subpixel Registration," IEEE TIP 11(3), 2002
<https://www.cs.ucf.edu/~foroosh/subreg.pdf>
Closed-form sub-pixel shift from the ratio of the main correlation peak to its
neighbors (polyphase-sinc peak model), with noise error analysis. We already
do the rotation-side analog with the stored d/dtheta derivative vector; this
is the principled translation-side counterpart.
**Actionable:** replace the fine translation grid-minimum with the two-sample
analytic sub-cell estimate along each axis of the (theta, d) landscape —
nearly free, decouples accuracy from search-grid resolution.

### 1.6 Guizar-Sicairos, Thurman, Fienup — "Efficient Subpixel Image Registration Algorithms," Opt. Lett. 33(2), 2008
<https://opg.optica.org/ol/abstract.cfm?uri=ol-33-2-156>
The standard upsampled-DFT refinement: evaluate the cross-correlation only in
a small neighborhood of the coarse peak via matrix-multiply DFT — zero-padding
accuracy at a fraction of the cost. Applies verbatim to the SSP inner product
since translation is phase multiplication.
**Actionable:** evaluate Re<Map, exp(iWd)*Scan> on a locally upsampled d-grid
around the coarse peak (a few extra phase multiplies) instead of a full
coarse-to-fine level — could shave ms/kf and improve the 0.03 m seq-edge
floor.

### 1.7 Padfield — "Masked Object Registration in the Fourier Domain," IEEE TIP 21(5), 2012
<https://www.researchgate.net/publication/51895870_Masked_Object_Registration_in_the_Fourier_Domain> (impl: scikit-image `masked_register_translation`)
Exact masked normalized cross-correlation via FFTs: the correlation at each
shift is normalized by the actual overlap of the validity masks, removing the
bias where partial overlap drags the peak toward high-overlap shifts. Directly
relevant to 180-deg scan-to-map matching: the SSP inner product is
UNNORMALIZED, so map content outside the current FOV dilutes both the score
and the coherence statistic the veto depends on.
**Actionable:** add overlap normalization — bundle a cheap mask vector (FOV
footprint encoding, or per-ring energy of the scan-visible region) and divide
the correlation by the map-side energy within the overlap; an overlap-ratio
floor gives a principled minimum-overlap gate for loop candidates.

### 1.8 Buelow, Birk — "Spectral 6DOF Registration of Noisy 3D Range Data with Partial Overlap," IEEE TPAMI 35(7), 2013 (sonar precursor: Auton. Robots 30(3), 2011)
<https://link.springer.com/article/10.1007/s10514-011-9221-8>
Phase-Only Matched Filtering (POMF) registration for very noisy sonar with
PARTIAL OVERLAP: phase-only whitening keeps the peak sharp when the two views'
amplitude spectra disagree. Contrast with us: the SSP inner product weights
frequencies by map amplitude (matched filter); POMF whitens — more robust
under partial overlap, at the cost of noise amplification at weak frequencies.
**Actionable:** test a per-ring phase-only or SNR-weighted Wiener variant of
the match score; a Wiener weight subsumes both this and the von-Mises
range-gating weight under evaluation (hier section).

### 1.9 Pfingsthorn, Birk, Buelow — "Uncertainty Estimation for a 6-DoF Spectral Registration Method...," ICRA 2012
<https://www.researchgate.net/publication/239763466_Uncertainty_estimation_for_a_6DoF_spectral_registration_method_as_basis_for_sonar-based_underwater_3D_SLAM>
Treats each POMF correlation output as a probability mass function (1D over
yaw, separate over translation), fits Gaussians, and uses the covariances as
SLAM edge information — the spectral-registration counterpart of Olson's
covariance, validated in a pose graph.
**Actionable:** when implementing 1.4, use their factored form — independent
1D theta-PMF and 2D d-PMF fits matching our decoupled search — and consider
their top-k multimodal extension (hypothesis edges) for the corridor
perceptual-alias leak that "no available statistic separates."

### 1.10 Bernreiter, Ott, Nieto, Siegwart, Cadena — "PHASER: A Robust and Correspondence-free Global Pointcloud Registration," IEEE RA-L 6(2), 2021
<https://arxiv.org/abs/2102.02767> (code: <https://github.com/ethz-asl/phaser>)
Modern lidar-native spectral registration: spherical-Fourier correlation for
rotation, spatial FFT for translation, with Bingham/Gaussian fits to the
correlation surfaces for uncertainty; robust to noise, sparsity, partial
overlap.
**Actionable:** borrow their evaluation protocol — sweep controlled overlap
fractions, report success vs overlap — to characterize where the SSP matcher's
basin collapses; feeds the drift-aware innovation allowance and relocalization
ring design.

### 1.11 Barnes, Weston, Posner — "Masking by Moving: Learning Distraction-Free Radar Odometry from Pose Information," CoRL 2019
<https://arxiv.org/abs/1909.03752>
Learns a reliability mask over radar scans for a differentiable
correlation-based matcher (68% error reduction) with uncertainty from a
softmax-normalized correlation volume. The learned analog of our
physics-derived range-gating weights — evidence that per-region reliability
weighting on top of a fixed correlator is where accuracy comes from.
**Actionable:** copy the correlation-volume softmax normalization so the
veto's coherence statistic is comparable across scenes — directly addresses
the "absolute thresholds do not transfer" lesson from the corridor fix.

**Area-1 gap note:** three independent lines (Olson, Pfingsthorn, PHASER)
converge on covariance-from-correlation-surface feeding a pose graph — the
best-supported borrowing available. Partial overlap has real precedent
(Padfield exact masking, Hurtos fan windowing, Buelow whitening) but nothing
specifically on 180-deg planar-lidar phase correlation: that niche is open.

---

## Area 2 — Multi-wavelength ladder design under phase noise (PRIORITY: finding 7 precedent)

### 2.1 Towers, Towers, Jones — "Optimum frequency selection in multifrequency interferometry," Opt. Lett. 28(11), 2003; "Generalized frequency selection...," Opt. Lett. 29(12), 2004
<https://opg.optica.org/ol/abstract.cfm?uri=ol-28-11-887> / <https://opg.optica.org/ol/abstract.cfm?uri=ol-29-12-1348>
The classic optimum-ladder result: for N phase measurements, the
reliability-optimal choice is a GEOMETRIC series of synthetic wavelengths,
with usable range bounded jointly by phase noise and N. The literature's
version of finding 5 ("octaves suffice; plateau r~1.4–2.8"), with a formal
fringe-order-error probability replacing our empirical plateau.
**Actionable:** adopt per-ring fringe-order error probability as the
principled way to derive the ring ratio from measured sigma_phi instead of
sweeping.

### 2.2 Zuo, Huang, Zhang, Chen, Asundi — "Temporal phase unwrapping algorithms for fringe projection profilometry: A comparative review," Opt. Lasers Eng. 85, 2016
<https://www.sciencedirect.com/science/article/abs/pii/S0143816616300653>
Hierarchical coarse-to-fine multi-frequency unwrapping (= our octave cascade)
is the most noise-reliable strategy; heterodyne/synthetic-wavelength schemes
are the most fragile because phase noise is amplified by the wavelength
magnification at every beat. The standard criterion — fringe order fails when
the ratio-scaled combined phase error exceeds pi — yields a
sigma_phi-dependent adjacent-ring ratio cap: exactly the shape of finding 5's
"gaps > ~2 octaves reintroduce sidelobes."
**Actionable:** plug our own sigma_phi = 2*pi*r*sigma_theta/lam into the
half-fringe (pi) criterion and verify r=2 sits inside the safe region at the
worst gated range — turning the octave choice into a derived quantity.

### 2.3 Zhang, Zuo et al. — "Robust and efficient multi-frequency temporal phase unwrapping: optimal fringe frequency and pattern sequence selection," Opt. Express 25(17), 2017
<https://opg.optica.org/oe/fulltext.cfm?uri=oe-25-17-20381>
Explicit judgment criterion for whether a given adjacent-frequency ratio
unwraps stably under a measured noise model, plus the optimal highest
frequency — the closest thing in print to a recipe for our ladder +
range-gating co-design.
**Actionable:** their "optimal highest frequency" maps to finding 5's
lam_min ~ noise scale (0.25 m); re-derive lam_min from the range-gated
sigma_phi envelope rather than the sweep.

### 2.4 Cheng & Wyant — "Two-wavelength phase shifting interferometry," Appl. Opt. 23, 1984
Origin of the synthetic wavelength Lambda = lam1*lam2/|lam1-lam2| and the
error-magnification rule (phase error scaled by Lambda/lam when cascading).
Grounds why the incommensurate relocalization rings (5.3/12.8) must budget
beat-scale noise amplification.
**Actionable:** before trusting a wide-mode coarse lock, budget the
amplification factor ~ Lambda_beat/lam_ring of the ring pair used.

### 2.5 de Groot — "Extending the unambiguous range of two-color interferometers," Appl. Opt. 33(25), 1994
<https://opg.optica.org/ao/abstract.cfm?uri=ao-33-25-5948>
Unambiguous range can exceed the synthetic wavelength by a range multiplier
whose cost is proportionally stricter phase-precision requirements — a clean
quantitative statement of finding 5's "range can stop at the search window"
trade.
**Actionable:** justify capping lam_max ~ search window via de Groot's
multiplier argument instead of empirically.

### 2.6 Xia & Wang — "Phase unwrapping and a robust Chinese remainder theorem," IEEE SPL 14(4), 2007; Xiao, Xia, Wang — "Multi-stage robust CRT," IEEE TSP 62(18), 2014
<https://www.eecis.udel.edu/~xxia/RobustCRT.pdf>
Rigorous theory of multi-frequency ranging with pairwise-coprime moduli robust
to bounded remainder errors — the formal home of our residue-number-system
relocalization rings (alongside Kymn et al. Residue HDC, already in RESULTS).
Quantifies exactly how much phase error each modulus pair tolerates before the
reconstruction jumps.
**Actionable:** replace the two-consistent-relocalizations acceptance
heuristic with the robust-CRT error bound as the accept/reject test for
wide-mode relocalization.

### 2.7 Li, Qi et al. — "Closed-form, robust and accurate multi-frequency phase unwrapping: frequency design and algorithm," arXiv:1604.08845, 2016
<https://arxiv.org/abs/1604.08845>
Jointly designed frequency pattern + closed-form estimator asymptotically
achieving the Cramer-Rao bound for phase-based ranging; frequency selection
and estimator are co-designed rather than CRT-then-hope.
**Actionable:** their CRB analysis is the template for proving the
ladder+matcher combination is information-efficient (or finding the ratio
where it stops being).

### 2.8 Babcock — "Intermodulation Interference in Radio Systems," Bell Syst. Tech. J. 32(1), 1953
<https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1953.tb01422.x>
Founding paper of additive-coincidence-free frequency-set design: choose
channels so third-order products f_i + f_j - f_k never land on an occupied
channel ("Babcock spacing"; kin: Golomb rulers for pairwise differences,
sum-free sets for f_i + f_j = f_k). The 70-year-old engineering discipline
whose hazard class finding 7 rediscovered in the scale domain.
**Actionable:** screen every future ladder candidate (including jittered and
relocalization rings) with a Babcock-style collision test — flag any triple
with |f_i + f_j - f_k| below the correlation peak bandwidth.

### 2.9 Vladimirova, Shavit, Falkovich — "Fibonacci Turbulence," Phys. Rev. X 11:021063, 2021
<https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.021063> (arXiv:2101.10418)
THE direct precedent for finding 7's mechanism: they build wave systems on a
mode ladder with omega_{n+1} = omega_n + omega_{n-1} — ratio converging to phi
— PRECISELY BECAUSE this makes neighboring triplets exactly resonant for
three-wave interaction (conservation laws come out in Fibonacci numbers).
Conversely, standard turbulence shell models (GOY/Sabra) conventionally use
intershell ratio lambda = 2 — octaves — where triads are non-resonant. The
turbulence community independently identifies phi as the unique
geometric-ladder ratio maximizing additive three-wave coupling, and octaves as
the benign default.
**Actionable:** cite this plus the shell-model convention as the physics
precedent when publishing finding 7: "known resonance structure, new
consequence for correlation registration."

### 2.10 Dobrowiecki, Schoukens, Pintelon — "Design of multisine excitations to characterize the nonlinear distortions during FRF measurements," IEEE Trans. Instrum. Meas., 2001 (odd / sparse-odd multisine literature)
<https://www.researchgate.net/publication/3090129_Design_of_multisine_excitations_to_characterize_the_nonlinear_distortions_during_FRF-measurements>
System-identification practice: excite a tone subset chosen so intermodulation
products of the excited set fall on NON-excited detection lines, making
additive-mixing contamination measurable instead of corrupting.
**Actionable:** a live self-test borrowed wholesale — evaluate correlation
energy at f_i + f_j "detection" frequencies NOT in the lattice to detect
additive-resonance contamination in any candidate ladder.

### DIRECT ANSWER — does the golden-ratio ladder hazard have precedent?

**Partial precedent: the mechanism is known in adjacent fields; the design
rule is absent from ranging/metrology.** The exact mathematical fact —
a geometric ladder at ratio phi is the canonical construction for exact
three-wave (triadic/additive) resonance because k_{n+1} = k_n + k_{n-1} holds
identically — is established in turbulence shell-model theory (Fibonacci
Turbulence, PRX 2021, builds a model family ON this closure; GOY/Sabra use
octave ratio 2 as the generic non-resonant choice). Independently, RF channel
allocation (Babcock 1953) and multisine excitation design treat additive
coincidences f_i + f_j = f_k as the primary frequency-set hazard and construct
sum-collision-free sets. However, NO interferometry, synthetic-wavelength,
fringe-projection, FMCW, or multi-tone-ranging paper found warns against
phi-ratio ladders — those literatures use near-coincident pairs, geometric
series, or coprime-integer (CRT) sets where the question never arises.
Finding 7 as stated — "phi is the additive-resonance ratio; avoid it in
correlation-registration scale ladders; the 'most irrational' intuition is
backwards for scales" — appears NOVEL in the registration/ranging context and
is publishable, citing shell-model triad theory and Babcock as mechanism
precedent.

---

## Area 3 — Lidar noise models for range-dependent spectral reliability

### 3.1 Ye & Borenstein — "Characterization of a 2-D laser scanner for mobile robot obstacle negotiation," ICRA 2002
<https://www.researchgate.net/publication/3955333_Characterization_of_a_2D_laser_scanner_for_mobile_robot_obstaclenegotiation>
The canonical SICK LMS-200 error anatomy: warm-up drift, incidence-angle
effects, mixed pixels at depth discontinuities, and a probabilistic range
model. Key support: range noise is roughly range-INDEPENDENT for SICK-class
sensors (cm-level, surface/incidence dominated) — the premise of the
anisotropic tangential/radial split (tangential grows as r*sigma_theta, radial
stays flat).
**Actionable:** cite as the empirical basis for range-flat sigma_r in the
anisotropic variant; our occlusion filter (finding 8) IS a mixed-pixel filter
and should reference their analysis.

### 3.2 Okubo, Ye, Borenstein — "Characterization of the Hokuyo URG-04LX laser rangefinder...," SPIE 7332, 2009
<https://www.spiedigitallibrary.org/conference-proceedings-of-spie/7332/1/Characterization-of-the-Hokuyo-URG-04LX-laser-rangefinder-for-mobile/10.1117/12.818332.full>
Same methodology on the low-cost Hokuyo class: strong surface-brightness
dependence and NO reliable returns beyond ~40 deg incidence at distance — a
dropout failure mode, not just noise inflation, matching our occlusion
filter's grazing-angle side effect.
**Actionable:** if per-point weights are added, model incidence-driven
unreliability as dropout probability, not only variance inflation.

### 3.3 Soudarissanane, Lindenbergh, Menenti, Teunissen — "Scanning geometry: influencing factors on the quality of terrestrial laser scanning points," ISPRS J. 2011
<https://gnss.curtin.edu.au/wp-content/uploads/sites/21/2016/04/Soudarissanane2009Incidence.pdf>
Models noise as a function of incidence angle and range via SNR degradation of
the elongated elliptical footprint (noise ~ 1/cos(incidence) plus range term),
experimentally validated. The RADIAL-noise counterpart to our purely
tangential sigma_phi: radial is not flat at grazing incidence.
**Actionable:** scale sigma_radial by 1/cos(alpha_hat) with incidence
estimated from local hit geometry — the occlusion filter's tangential-gap
neighbors already give the needed local slope.

### 3.4 Lichti & Gordon — "Error propagation in directly georeferenced TLS point clouds...," FIG 2004; Lichti — "A resolution measure for terrestrial laser scanners" (EMR), ISPRS 2004
<https://www.fig.net/resources/proceedings/fig_proceedings/athens/papers/wsa2/WSA2_6_Lichti_Gordon.pdf> / <https://www.isprs.org/proceedings/xxxv/congress/comm5/papers/552.pdf>
Two load-bearing results: (a) finite beamwidth is itself angular uncertainty —
for divergence delta, sigma_theta_beam ~ delta/4, added in quadrature to
encoder jitter; (b) EMR formalizes that resolvable spatial detail is limited
jointly by angular sample spacing (r*dtheta) and footprint size — precisely
the Nyquist argument behind the hard cutoff lam >= beta*r*dtheta.
**Actionable:** replace bare sigma_theta in the gating weight with
sigma_theta^2 = sigma_jitter^2 + (delta/4)^2 (+ quantization, see 3.7); cite
EMR to justify beta >= 2 (Nyquist) with the excess covering footprint blur.

### 3.5 Pfister, Kriechbaum, Roumeliotis, Burdick — "Weighted range sensor matching algorithms for mobile robot displacement estimation," ICRA 2002 / journal
<http://robotics.caltech.edu/~sam/JournalPapers/WLSM_pfister.pdf>
Closest spatial-domain precedent for our gating weight: ML scan matching where
each point is weighted by modeled uncertainty (range noise, incidence,
correspondence error) and the displacement covariance is propagated
analytically. Our von-Mises weight is the frequency-domain transcription of
this per-measurement-model -> per-contribution-weight -> output-covariance
pipeline.
**Actionable:** derive the registration covariance from our weighted
correlation (per-ring weights already known) — converges with the Olson
peak-curvature item (1.4).

### 3.6 Censi — "An accurate closed-form estimate of ICP's covariance," ICRA 2007
<https://censi.science/research/robot-perception/icpcov/>
Registration covariance from the curvature of the error function with
measurement covariance propagated through; correctly captures the aperture
problem (corridor ridge -> near-singular information). Directly applicable
since our correlation objective is analytic in (theta, d).
**Actionable:** compute the second derivative of the weighted correlation at
the optimum (cheap in SSP space — second derivatives are frequency-squared
moments) as the loop-edge information matrix; corridor false closures (soak:
5.2 false/run) arrive pre-flagged as rank-deficient.

### 3.7 Diosi & Kleeman — "Fast laser scan matching using polar coordinates," IJRR 26(10), 2007
<https://research.monash.edu/en/publications/fast-laser-scan-matching-using-polar-coordinates>
Scan matching natively in (r, theta), where sensor noise is DIAGONAL: range
noise on r, bearing quantization on theta. Confirms the coordinate system in
which our noise model is separable — the tangential/radial split is diagonal
in polar sensor coordinates and only becomes anisotropic after Cartesian
projection.
**Actionable:** include the half-beam-spacing quantization term
dtheta/sqrt(12) in sigma_theta alongside jitter and beam divergence (3.4).

### 3.8 Segal, Haehnel, Thrun — "Generalized-ICP," RSS 2009
<https://www.researchgate.net/publication/221344436_Generalized-ICP>
The standard reference for per-point anisotropic covariances letting the
optimizer weight residual directions (plane-to-plane). Precedent that
anisotropic per-point models robustly beat isotropic; ours is the
sensor-geometry (not surface-geometry) version.
**Actionable:** the SSP analog is direction-dependent weights per (point,
ring-angle) pair: a lattice frequency at angle psi sampled from a point at
bearing phi is tangentially corrupted by sin(psi - phi) * 2*pi*r*sigma_theta/
lam — refine the isotropic sigma_phi at ~zero cost since the encoder already
loops over lattice angles.

### 3.9 Rodriguez & Martin — "Theory and design of interferometric synthetic aperture radars," IEE Proc.-F 139(2), 1992 (coherence–phase-variance CRB; textbook: Hanssen 2001)
Formula review (open): <https://eartharxiv.org/repository/object/750/download/1651/>
InSAR's foundational result: sigma_phi^2 = (1 - gamma^2)/(2L gamma^2) — phase
variance is a closed-form function of measured coherence, and the whole
downstream pipeline weights phases by coherence-derived variance. The mature,
field-proven version of our von-Mises weight: they ESTIMATE reliability from
data (coherence), we PREDICT it from geometry (r*sigma_theta/lam). InSAR
practice strongly supports soft variance weighting over hard masking (hard
thresholds only for unwrapping-graph pruning).
**Actionable:** invert the relation — convert our measured per-ring coherence
(already computed for the veto EMA) to an empirical sigma_phi via the CRB and
cross-check against the geometric prediction; agreement validates the model,
disagreement calibrates beta per-sensor online.

### 3.10 Tong et al. — "Robust fine registration ... enhanced subpixel phase correlation," Sensors 20(15):4338, 2020; Stone et al. — "Subpixel accuracy analysis of phase correlation ... aliased imagery," 2001
<https://doi.org/10.3390/s20154338> / <https://www.researchgate.net/publication/228879039_Subpixel_accuracy_analysis_of_phase_correlation_registration_methods_applied_to_aliased_imagery>
The phase-correlation community's answer to unreliable spectral bins: adaptive
masking/weighting of the cross-power spectrum, because phase at low-SNR or
ALIASED frequencies is uniform junk that corrupts the peak. Stone's aliasing
analysis is the hard cutoff's cousin: frequencies beyond the (in our case
spatially varying) Nyquist limit must be masked before correlating.
**Actionable:** the two candidate gating schemes are LAYERS, not rivals — keep
the geometric hard cutoff as the outer bound (aliasing destroys information;
matches G2's gating-off divergence in the hier bench) and apply the soft
von-Mises weight inside it (noise only degrades information).

**Area-3 synthesis.** The literature cleanly separates the regimes our three
candidate schemes conflate: ALIASING (r*dtheta > lam/2 — information
destroyed; hard cutoff correct: Lichti EMR, Stone) vs PHASE NOISE (jitter +
beam divergence — information degraded; soft variance weight correct:
Rodriguez-Martin CRB, Pfister, GICP). Recommended combined per-(point r,
ring lam) form: hard-zero if lam < 2*r*dtheta_beam, else
w = exp(-sigma_phi^2/2) with sigma_phi = 2*pi*r*sigma_theta_eff/lam and
sigma_theta_eff^2 = sigma_jitter^2 + (delta_beam/4)^2 + dtheta_beam^2/12;
optionally anisotropic in ring angle vs point bearing (3.8) and 1/cos
(incidence) radial (3.3). The observed beta in {2,3,4} flatness (hier finding
2) is consistent: all three sit at/above Nyquist, and the soft-noise regime
they differ in is second-order on the bench worlds.

---

## Area 4 — Spectral / frequency-domain map representations

### 4.1 Ramos, Ott — "Hilbert maps: Scalable continuous occupancy mapping with stochastic gradient descent," RSS 2015; IJRR 35(14), 2016
<https://fabioramos.github.io/Publications_files/HilbertMapsIJRR16Final.pdf>
Closest occupancy-mapping cousin: lidar hits projected through random Fourier
features, occupancy as a logistic readout trained online — a learned
discriminative spectral summary, vs our raw superposition. Uses isotropic
Gaussian RFF, exactly the recipe our deterministic polar lattice beats 10-20x
in dimensions (finding 1).
**Actionable:** train a small logistic "Hilbert-map head" over the existing
lattice features to get calibrated occupancy probabilities from the same map
vector — free occupancy queries for planning without touching the
registration path; cite as the direct RFF-occupancy precedent for finding 1.

### 4.2 Guizilini, Ramos — "Towards real-time 3D continuous occupancy mapping using Hilbert maps," IJRR 37(6), 2018
<https://fabioramos.github.io/Publications_files/0278364918771476.pdf>
Scales Hilbert maps to 3D with LOCALLY-supported feature clusters and
incremental updates; their global-to-local move mirrors the G1
rigid-segment vs G2 world-frame-cell tension (hier section).
**Actionable:** cite as the published analogue of HY4's "rigid anchor-frame
segments beat world-frame cells" result (R2 finding 3) — they too found local
anchoring beats one global model.

### 4.3 Schaefer, Luft, Burgard — "DCT Maps: Compact Differentiable Lidar Maps Based on the Cosine Transform," IEEE RA-L 3(2), 2018
<https://arxiv.org/abs/1910.11147>
The most literal spectral-map precedent: the map IS a set of DCT coefficients
whose continuous inverse yields a differentiable lidar decay-rate field;
beats grid maps, GP occupancy, and Hilbert maps at equal memory. They store a
real cosine basis on a Cartesian lattice — which CANNOT do
rotation-as-permutation; our rotation-closed complex polar lattice is the
delta.
**Actionable:** their decay-rate (ray permeability) field flags what our
hit-only encoding discards — free-space evidence; a second "miss" vector per
segment (bundled ray-interior samples) is the SSP analogue (see also 4.4);
key citation for the rotation-closure argument.

### 4.4 Snyder, Capodieci, Gorsich, Parsa — "Brain-inspired probabilistic occupancy grid mapping with vector symbolic architectures" (VSA-OGM), npj Unconventional Computing, 2026
<https://www.nature.com/articles/s44335-026-00052-w>
Very recent independent confirmation that SSP/fractional-power-encoded lidar
points bundled into memory vectors work as probabilistic occupancy maps, with
tile decomposition and separate occupied/empty class vectors. Assumes
externally provided poses — no registration — so our correlation-matching
registration in SSP space remains unclaimed; they hit the same
bundle-saturation wall our cell-cap/window sizing dances around.
**Actionable:** adopt the dual-class idea — bundle a free-space vector per
segment from sampled ray interiors, giving an occupied-vs-empty contrast
statistic to corroborate the brittle 0.55 coherence veto (the prescribed
robustification in the SHIPPED block); cite in Related Work as nearest
published system.

### 4.5 Lu, Xu et al. — "One RING to Rule Them All: Radon Sinogram for Place Recognition, Orientation and Translation Estimation," IROS 2022; "RING++," IEEE T-RO 2023
<https://arxiv.org/abs/2204.07992> / <https://arxiv.org/abs/2210.05984> (code: <https://github.com/lus6-Jenny/RING>)
Learning-free global localization on sparse scan maps: Radon sinogram
(rotation = circular shift), row-wise DFT magnitude (translation invariance),
circular cross-correlation for orientation, then correlation translation
solve — the published system closest in spirit to our wide-window
relocalization.
**Actionable:** a RING-style EXHAUSTIVE rotation sweep on the coarse
incommensurate rings (rotation candidates are exact permutations — already
cheap via the perm matcher) could replace the prior-windowed relocalization's
rotation search and supply the large-sparse-loop snapping that ACES and the
corridor lack (zero-shot transfer verdict).

### 4.6 Xu, Yin et al. — "DiSCO: Differentiable Scan Context with Orientation," IEEE RA-L 6(2) / ICRA 2021
<https://arxiv.org/abs/2010.10949> (code: <https://github.com/MaverickPeter/DiSCO-pytorch>)
Polar-BEV frequency-domain scan representation: spectrum MAGNITUDE as a
rotation-invariant place signature, differentiable phase correlation for
relative orientation — the retrieval-vs-metric information split our
coherence statistics and permutation matcher exploit implicitly.
**Actionable:** cheap relocalization pre-filter — per-ring magnitude profiles
of segment vectors are rotation-invariant by construction on our lattice; use
them to rank loop candidates before running full permutation correlation,
cutting drought-mode cost and possibly raising the ~0.5% closure density seen
on MIT-corridor-class runs.

### 4.7 Fan et al. — "FreSCo: Frequency-Domain Scan Context for LiDAR-based Place Recognition with Translation and Rotation Invariance," 2022
<https://arxiv.org/abs/2206.12628> (code: <https://github.com/soytony/fresco>)
Fixes Scan Context's viewpoint dependence via Fourier transform + circular
shift for simultaneous translation and rotation invariance, plus a fast
two-stage relative pose estimate after retrieval — another datapoint that
polar-Fourier is the standard answer to LATERALLY-OFFSET revisits, the failure
mode behind "only 77 closures fired" on MIT.
**Actionable:** bench loop recall vs cross-track offset (0.5-3 m) to quantify
how much of the revisit-density limit is translation-sensitivity of the fine
rings — decides whether mid-ring-only candidate matching should widen its
window.

### 4.8 Biber, Strasser — "The Normal Distributions Transform," IROS 2003; Magnusson et al. — "Automatic appearance-based loop detection from 3D laser data using NDT," JFR 26, 2009; Saarinen et al. — "NDT-MCL," IROS 2013
<https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.20314> / <https://ieeexplore.ieee.org/document/6696380/>
The established compact-parametric map family: ~10 floats/cell Gaussians,
smooth registration cost, NDT-histogram descriptors for loop detection, MCL
directly on the map. NDT is the memory-boundedness comparator reviewers will
reach for — also O(area) and history-free. Notably: NO Fourier/spectral-
compressed NDT variant surfaced; that combination appears open.
**Actionable:** add NDT bytes/m^2 to the positioning table (Reference
baselines section) — "smallest history-free state" needs NDT as the honest
comparator, not just grids/RBPF; Magnusson's range-interval histograms are a
template for a cheap per-ring-energy loop-candidate descriptor.

### 4.9 Kymn, Kleyko, Frady, Bybee, Kanerva, Sommer, Olshausen — "Computing with Residue Numbers in High-Dimensional Representation," 2023/2025; Krausse et al. — "A Grid Cell-Inspired Structured Vector Algebra for Cognitive Maps," 2025
<https://arxiv.org/abs/2311.04872> / <https://arxiv.org/abs/2503.08608>
Already cited in RESULTS for the incommensurate rings; the fuller value is the
DECODING THEORY: residue-HDC gives explicit capacity and noise-tolerance
bounds for recovering position from coprime-modulus phase codes. GC-VSA's
module structure (discrete scales x orientations) is literally our octave x
angle lattice with a hexagonal twist — which finding 4 says adds nothing, a
point of disagreement worth stating.
**Actionable:** use residue-decoding error bounds to size the 5.3/12.8 m
relocalization rings principledly (max unambiguous range vs phase-noise
level) — the queued bundle-capacity borrowing, now with a second source
(and see robust CRT, 2.6).

### 4.10 Komer, Stewart, Voelker, Eliasmith — "A neural representation of continuous space using fractional binding," CogSci 2019
<https://compneuro.uwaterloo.ca/files/publications/komer.2019.pdf>
The origin of the SSP encoding: fractional binding = exp(iWx) with W fixed by
a random unitary base vector. All their spatial-map operations (region
queries, shifts, object memory) apply unchanged to our deterministic lattice;
none of the SSP literature chooses W for rotation-group closure.
**Actionable:** frame the encoder in write-ups as "fractional binding with a
designed, rotation-closed base vector" — one sentence connecting findings 1-2
and 6-7 to VSA vocabulary and making the delta (designed W, C_n closure,
d/dtheta companion) crisp.

**Also noted:** CURL-SLAM (Zhang et al. 2025, <https://arxiv.org/abs/2506.21077>)
— spherical-harmonics implicit 3D lidar map with pose estimation on the
continuous representation, 10 Hz CPU: the 3D analogue of "register against
the spectral map, not the points." DeepRING (<https://arxiv.org/abs/2210.11029>)
— learned RING variant.

---

## Top-5 actionables

1. **Correlation-surface covariance as loop-edge information matrices**
   (Olson 1.4 + Pfingsthorn 1.9 + Censi 3.6 + Pfister 3.5 — four independent
   lines converge). Evaluate the SSP score on the local (theta, d) grid around
   each accepted peak (or take analytic second derivatives — frequency-squared
   moments), fit factored 1D-theta / 2D-d Gaussians, use the inverse as the
   edge information matrix. Corridor/aperture closures (finding 3; soak's 5.2
   false edges/run) then self-report rank deficiency instead of leaning on the
   brittle 0.55 coherence veto; keeps the veto's prescribed robustification
   honest.

2. **Layer the gating schemes — hard Nyquist cutoff outside, soft von-Mises
   inside** (Stone/Tong 3.10, Lichti EMR 3.4, InSAR CRB 3.9). Hard-zero when
   lam < 2*r*dtheta_beam (aliasing destroys information — matches G2
   gating-off divergence), soft w = exp(-sigma_phi^2/2) within, with
   sigma_theta_eff^2 = sigma_jitter^2 + (delta_beam/4)^2 + dtheta_beam^2/12.
   Cross-check the geometric sigma_phi against the empirical one implied by
   measured ring coherence via the InSAR CRB — a free online calibration of
   beta.

3. **Overlap-normalized matching for the 180-deg FOV** (Padfield 1.7, Hurtos
   1.3, Buelow 1.8). Normalize the correlation by map-side energy within the
   scan's FOV footprint (cheap mask/energy vector per ring) and add an
   overlap-ratio floor as a loop-candidate gate; test a per-ring
   Wiener/phase-only score variant. Nothing published does 180-deg planar
   lidar phase correlation — this is both a fix and a claimable niche.

4. **Derive the ladder from unwrapping theory instead of sweeping** (Towers
   2.1, Zuo 2.2, Zhang 2.3, de Groot 2.5). Plug the range-gated sigma_phi
   envelope into the half-fringe criterion to derive the safe ring ratio
   (verify r=2), lam_min, and lam_max ~ search window as computed quantities;
   size the 5.3/12.8 relocalization rings and their acceptance test with
   robust CRT (Xia & Wang 2.6) / residue-HDC bounds (4.9). Screen every
   candidate ladder with a Babcock triple-collision test (2.8) and adopt the
   multisine detection-line diagnostic (2.10) as a live self-test.

5. **Sub-cell peak refinement + magnitude-profile pre-filter** (Foroosh 1.5,
   Guizar-Sicairos 1.6, DiSCO 4.6, RING 4.5). Analytic two-sample sub-cell
   peak interpolation and locally-upsampled-d evaluation around the coarse
   peak (near-free accuracy below the search grid step); rotation-invariant
   per-ring magnitude profiles to rank loop candidates before full
   permutation correlation, plus a RING-style exhaustive rotation sweep on
   coarse rings for large sparse loops (the ACES/MIT closure-density gap).

## Verdict: golden-ratio ladder hazard precedent

**Partially precedented — mechanism yes, design rule no.** The additive
closure phi^2 = phi + 1 making a phi-ratio geometric ladder exactly
three-wave-resonant is established physics: Fibonacci-Turbulence shell models
(Vladimirova/Shavit/Falkovich, PRX 2021) are built on precisely
k_{n+1} = k_n + k_{n-1}, and standard shell models use octave ratio 2 as the
canonical NON-resonant spacing — independently mirroring finding 7's "phi
uniquely bad, r=2 wins." The hazard CLASS (additive coincidences
f_i + f_j = f_k in a frequency set) is a 70-year-old design discipline in RF
channel allocation (Babcock 1953, sum-free sets, Golomb-ruler kin) and
multisine system identification. But no interferometry, synthetic-wavelength,
fringe-projection, FMCW, or multi-tone-ranging source found states the rule
"avoid phi-ratio wavelength ladders" — their frequency-selection methods
(near-coincident pairs, geometric series, coprime CRT sets) never encounter
it. Finding 7 is therefore publishable as a new design rule for
correlation-based registration lattices, with shell-model triad theory and
Babcock spacing as the mechanism precedents to cite — including the
counter-intuitive corollary that "most irrational" (good for golden ANGLES)
is exactly wrong for SCALES.
