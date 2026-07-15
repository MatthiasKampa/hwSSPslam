# ENCODINGS — every suggested/measured formulation (as of 2026-07-14)

All encodings share one algebra: a point/feature set becomes
`φ(P) = Σᵢ wᵢ · exp(i(W·pᵢ + θ_app,ᵢ))` on a frequency lattice `W`;
similarity = inner product; translation = phase multiply; rotation =
(exact or approximate) index permutation. Numbers cite the ledger
(`RESULTS.md` sections named). Verdicts: **SHIPPED** (in the deliverable),
**RECIPE** (measured winner, adopt for the ECP5 track), **OPTION**
(per-regime), **REF** (quality reference, not deploy), **NEG** (banked
negative), **FILED** (suggested, not yet run).

## A. Lattice layouts (the W matrix)

| layout | construction | rotation algebra | key numbers | verdict |
|---|---|---|---|---|
| az2d / oct60 | n_az half-circle dirs × octave λ ladder (+ d/dθ derivative vector) | yaw = EXACT permutation | the whole 2D suite; school place 0.767 rot-searched | **SHIPPED** |
| hex 2D | full-circle hex, plain-shift permutation | yaw exact (full circle) | belg 2.071 vs 2.644 | OPTION (non-Manhattan) |
| RFF random 2D | random directions | none | loses 10–20× at equal D | NEG (control) |
| **azel3d** | azimuth rings × elevation bands × λ ladder | yaw EXACT per band; SO(3) decode 3.9° (grid floor) | school place 0.700→**0.858** @ D240→960; disp incl. z at grid floor | **RECIPE (3D)** |
| fib3d | Fibonacci half-sphere × λ ladder | no exact perm (yaw cos 0.87); real mixed-axis SO(3) 6.7° (best) | place ≈ azel3d | NEG for matcher; OPTION for isotropic SO(3) |
| rand3d | random 3D dirs | none | ≈ fib on place | control |
| scale ladders | octave (shipped) · √2 half-oct (span11/hoct9 = fidelity recipe) · φ golden | — | φ = NEG both venues; √2 stands; 3D D960 uses √2 0.25–2.83 | per row |
| W_vis_fib | bearing lattice, angular scales 0.3–2.4 | n/a (vision) | vision benches below | default vision map |
| W_vis_cone | directions concentrated in camera FOV | n/a | ≈ fib (no gain) | NEG (no win) |

## B. Lidar point encodings (what enters φ)

| encoding | idea | numbers | verdict |
|---|---|---|---|
| seg resample | 0.12 m segment resampling | the shipped baseline | **SHIPPED** |
| E2 points | per-beam points, w = r·dθ | fr079 5.52→2.21; frontend win sparse heads | OPTION (sparse 180°) |
| interp2 bridged | pair sub-points @ 63.4° occlusion gate | stata 0.196 vs points 1.659 | **RECIPE (dense heads)** |
| SegInt exact integrals | sinc(k·d/2)·phasor; GROUP/8 mass-exact decimation | ≡ sub-points on stata; 2.5× fewer terms | OPTION |
| n1 single sub-point | 1 point/chord | dense-head equal-or-better; sparse regression | OPTION (dense only) |
| slice1 → rings3 → full | 1 ring / 3 rings / 64 rings into azel3d | place 0.256 / **0.701** / 0.700 | **RECIPE: 3 rings** |
| SUB ladder | 2k/4k/8k pts | 0.690/0.700/0.699 (flat) | **RECIPE: ~2k pts** |
| store: 2b+scales (lean), 6b, QPSK, 3b dead-zero, int8 matcher | write-time quantized map | lean = deploy winner per-regime (webvis pack) | OPTION per-env |

## C. Vision encodings

### C1. Sparse bearing features (detector peaks → weighted bearings)
Cost = (line-buffer rows, mult/px). All servo to lidar point parity.

| detector | cost | cam-AUC classrm/school | verdict |
|---|---|---|---|
| fast9 | 7, 0 | 0.742 / 0.38 | **SHIPPED RTL v0** (bit-exact core @72 MHz) |
| fast12 | same silicon | 0.698 / 0.40 | NEG (no win) |
| susan16 | 7, 0 | 0.733 / 0.36 | NEG (no win) |
| extrema | **3, 0** | 0.723 / 0.37 | **S-corner champion** |
| edge (Sobel+NMS) | 3, 0 | 0.763 / 0.38 | cheap tier; edges>corners for place |
| shi-tomasi L1 | 8, **3** | **0.809** / 0.375 | **F-corner winner → RTL v1** |
| harris | 8, 5 | 0.706 / 0.41 | superseded (servo saturates) |
| dog peaks | 9, 0 | 0.789 / 0.39 | mid |
| ucnn peaks | ~60% DSP @120fps | 0.665 / 0.426 | REF only |

### C2. Dense appearance as WEIGHTS (the cross-view tier)

| encoding | idea | school / classroom | verdict |
|---|---|---|---|
| **gridint** | 8×8 mean-intensity cells, ~free | **0.622** / 0.949; max-fused 0.798 | **RECIPE (cross-view place)** |
| dog-raw | dense integer DoG heat | 0.38 / 0.76 | NEG vs gridint |
| ucnn-raw | untrained int8 3-layer CNN heat | **0.652** / — | REF (trained headroom; 60% DSP) |

### C3. Appearance IN PHASE (what⊗where binding — the precision/ego-motion tier)

| encoding | binding | school / classroom / adj-rep | verdict |
|---|---|---|---|
| intphase | cell intensity → 3 phase wavelengths | 0.448 / **0.999** / **0.997** | **RECIPE (verification + 120 fps ego-motion)** |
| gradhog | 8-dir quantized gradient orientation (π-periodic 2θ harmonics), magnitude weights | 0.425 / 0.829 / 0.90 | OPTION |
| census-phase | 8-neighbour cell code → binary phase decomposition | 0.425 / 0.852 / 0.81 | NEG for cross-view (illumination-invariance prediction refuted) |

Law: phase-binding SHARPENS view-specificity — precision channel under
view overlap, harmful across viewpoint change; weights-only stays the
cross-view channel.

### C4. 3D visual landmarks (geometry lift)

| encoding | numbers | verdict |
|---|---|---|
| bearings × lidar range (guessed extrinsics, SPOT) | 3D < bearings both venues | NEG — extrinsics artifact (proven) |
| bearings × REGISTERED depth (TUM fr3) | **rev-AUC 0.941** vs bearings 0.198 | **THE wall-crossing channel** — needs calibrated lidar↔cam extrinsics on platform |
| (fr2 pioneer: 4 m Kinect in a hall) | 0.560 | depth-COVERAGE boundary condition |

### C5. Suggested, not yet run (FILED)

- **Trained tiny-YOLO-class int8 head** (boxes → weighted bearings, or raw
  logit heat): fits ≤60% MULT18 @QVGA120 only at ~3-layer scale; train
  offline, deploy weights via SPI flash.
- **BRIEF/ORB descriptor binding**: descriptor bits → sign-flip pattern on
  sub-lattices; tension with the bounded-map philosophy (descriptor
  stores), so gated on a capacity budget.
- **Full HOG histogram per cell** (all 8 bins as separate bound phasors,
  not just dominant) — natural gradhog extension.
- **Multi-scale intensity pyramid** weights; DoG scale-space peaks with
  scale bound in phase.
- **Lidar-depth-on-camera-cells at deploy** (the calibrated-extrinsics
  successor to C4 on SPOT).
- Temporal/sequence binding: banked NEG on lidar (SeqSLAM); not pursued.

## D. Fusion/composition

| scheme | numbers | verdict |
|---|---|---|
| shared map, vector addition (azel3d) | classroom 0.92–0.94; school 0.516 | OPTION (view-overlap venues) |
| two-map late fusion, z-normed α-sum | classroom 0.954 @ α=0.1 | OPTION |
| two-map **max-rule** | school **0.798** > lidar 0.720 | **RECIPE at the wall** |
| per-anchor snapshot library + max-sim | architecture (FINDINGS A10) | **RECIPE (vision map lifecycle)** |
| confidence-adaptive α | — | FILED |

## E. Object/appearance world-map encodings (experiments/objmap.py, 2026-07-15)

The "find objects in the map from cam data" family: camera cells lifted
to 3D world points, appearance bound as a complex amplitude, bundled
into bounded per-segment vectors; query = template patch → matched
filter → translation-correlation decode (O(D)/candidate, the map's own
algebra).

**Appearance-binding law (measured in objmap selftests — applies to ANY
decode-against-the-map use of appearance, unlike the place-similarity
use in §C where harmonics are fine):**

| binding | cross-code kernel | verdict |
|---|---|---|
| 3-harmonic cyclic (the §C place scheme) | ~1/√3-coherent | ✗ bump forest, clusters AT the true answer (difference-density peaks at 0) |
| uniform per-bit phase keys | 0 at ≥1 flip | ✗ no grace — census flips bits across views routinely |
| concentrated per-bit keys (±2π/3) | sinc-graceful | ✗ forest returns via near-Hamming code pairs |
| per-(bit,value) random phasors | matching_bits/8 ≈ **0.5 for random pairs** | ✗ half-coherent clutter (measured, subtle) |
| **bipolar spatter** A[c]=Σₖ±e^{iKₖ}/√8 | (8−2·Hamming)/8, zero-mean random pairs | ✓ linear grace + pseudo-random cross-talk; kernel law verified 0.85 (theory 0.75 + clutter) |

Scalar codes (orientation, intensity): FPE with random per-row
exponents — graceful and forest-free by construction.

**Decode/search laws:** scale-split decode (coarse rings for the wide
search — fine-ring lobes fall between coarse grid samples), iterated
re-centred refine (single-box refine clips edge peaks); ladder must be
SMEAR-MATCHED to the data (real depth+cell world-point noise ~5-10 cm
decoheres sub-0.5 m rings — they only add map noise; synthetic clean
data wants them). Detection margin against room-wide extreme-value
noise is thin for single 25-cell templates (foil max ≈ 0.7-0.9× true
even with exact codes at 400 items) → architectural direction:
segment-scoped two-stage retrieval (which-segment first, metric decode
inside), bigger/multi-view templates. Real-data verdicts (RESULTS
2026-07-15): int-FPE > grad > census for detection (0.735/0.667/0.609
centered — graded kernels beat binary codes in amplitude matched
filtering); centering = detection↔localization trade; D flat
(noise-limited); seg scoping no rescue (max-over-segments EV); 2b
free. Status: coarse room-region cue; two-stage + multi-view filed.
