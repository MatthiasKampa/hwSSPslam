# FPGA design note — SSP SLAM datapath (from the 2026-07-10 measurements)

A concrete architecture sketch for taking the bounded-memory SSP SLAM stack
to fabric, grounded entirely in tonight's audited numbers (RESULTS
"2026-07-10" sections; `ssp_fpga.py` is the bit-accurate reference model).
This is the hand-off document for an RTL session.

## Target sensor (user spec, 2026-07-10)

SPOT-mounted custom head: **360° × 1024 beams @ 20 rotations/s**
(0.35° spacing, 20.5k returns/s). Consequences, measured:

- **Encode budget is a non-issue**: full 1024-term encode = 6.31 M
  MAC-equiv/keyframe → 0.13 GMAC/s at 20 Hz keyframes ≈ **1 DSP48 @
  200 MHz** (`ops_report(1024)`). The coarse/fine correlation stages
  (point-count-independent) dominate. ~~Group-integral decimation~~
  **RETRACTED as an option (2026-07-11)**: GROUP/2/4/8 collapse stata
  e2e in deep basins despite bench losslessness (see Deploy addenda);
  the term-budget option is interp-n1 (dense heads only, pending
  validation).
- **Fine-ring aliasing from beam spacing vanishes**: wall-sample spacing
  r·dθ stays under λ_min/2 = 0.125 m out to r ≈ 20 m at normal incidence —
  the thermometer/blanking question reduces to occlusion gaps and grazing
  incidence (encoder study, synth-360: renorm_alpha inert).
- **Sampling structure matters more than density**: on the 1040-beam
  stata proxy, raw per-beam points COLLAPSE the closure layer (1.659, 9
  loops) while bridged pair-interpolation at the shipped 63.4° gate
  restores it (0.196, 74 loops ≈ shipped 0.202) — the deploy sampler must
  bridge, not just weight. (Full verdict in the encoder-study RESULTS
  entry.)
- **Motion skew** at 20 rot/s on a walking base (~1.5 m/s → up to ~7.5 cm
  smear/rev): de-skew each revolution against the odometry twist before
  encoding (host-side, cheap). Untested here — flag for the integration
  session.
- **FIRST CONTACT (Telluride workshop set, 2026-07-10)**: lidar-only (CV
  guesses, odometry withheld as reference), 78 s / 36.5 m / 7×7 m room:
  **float 0.039 ≡ lean 2-bit+int8 0.039 ATE (med 3.3–3.5 cm) at 14 KB
  map** — the full binary datapath reproduces float on the platform's own
  data at the reference's noise floor. Frontend window can shrink
  0.48→0.12 m with bit-identical outcomes at this motion regime (compute
  knob, not accuracy). Adapter lesson for the integration session: sort
  and hygiene-check the odometry stream (the workshop parquet carried a
  ~62 s out-of-order block).

## System split

- **Fabric**: per-keyframe hot path — scan sampling, phase encode, frontend
  match (translation banks × rotation permutations), segment fold, loop-
  candidate match + per-ring coherence sums. All integer.
- **Host (CPU/soft-core)**: pose bookkeeping (SE2), gates, edge list, the
  sparse early-stopped TRF relax (runs every ~25 kf over ≤ ~1k anchors;
  milliseconds on any embedded core), and the band-probe telemetry.

## Numeric contract (audited operating points)

| block | format | evidence |
|---|---|---|
| phase arithmetic | 8-bit cis ROM (256 entries × 2 × 7-bit signed) | int ladder knee: fr101 2.10 free-ish; below addr8 real damage beyond bands (intel addr6 10.5) |
| sample weights | 7-bit fixed point (or unit weights only on robust logs) | w7 in the knee config; QPSK/unit = medians only |
| map store | per-ring polar codes; 6 b/phasor (16ph×4mag) general, 2 b (4ph×1mag, one-level clamp) on closure-redundant deployments | 6 b: algebra intact (0.986/0.925); 2 b: fr101 e2e 1.820 @ 76 KB (true one-level verified) |
| per-ring scales | 6 × f32 (or shared exponent) per vector — REQUIRED | per-vector scales crush fine rings 0.84 vs 0.97 fidelity (the veto's own statistic) |
| accumulators | int32 (fold), int32–40 (correlation MAC trees) | value model exact in f64; magnitudes tracked in ssp_fpga |
| segder | same store format as segvec; keep at N_ANG=60 | sub-grid path adds no extra quant damage (0.986 at 6 b) |

## Blocks

1. **Sampler** (per-SENSOR-REGIME mux — settled by the 2026-07-10 encoder
   sampling study; the earlier per-log mux was the 180°-FOV shadow of it):
   - **TARGET (wide-FOV dense head, SPOT 360°×1024 / stata-class): `bridge`
     mode** — connect consecutive returns gated at the shipped 63.4°
     occlusion angle (dr ≤ 2·tang, a compare + two mults) with arc mass
     r̄·dθ_gap; emit n2/n3 midpoint sub-points (zero new hardware) or exact
     integral terms (one extra D-dot for k·d + a sinc LUT). stata: 0.196 vs
     raw-points 1.659 (closure layer collapses without bridging); the gate
     value is a real optimum (45°→1.66, 75°→8.25) — bake it.
     GROUP/k integral folds are BENCH-ONLY (2026-07-11: they collapse
     stata e2e — do not deploy; see Deploy addenda). If the stream must
     shrink on a dense head, interp-n1 is the candidate (half terms,
     equal on every dense log; sparse logs regress — pending final
     validation).
   - `point` mode (sparse 180° heads: fr079/aces/belg/fr101-class):
     one phasor per hit, w = r·dθ. No geometry preprocessing at all —
     a pure stream. Band-dominant on fr079/aces/belg.
   - `chord` (the shipped fixed-0.12 m resampler): retired for the target —
     it FAILS at 1024-beam density (synth corridor 1.76 m, 1 loop) and
     up-samples the stream (1636 terms from 1024 beams).
2. **Encoder**: phase = W·p as fixed-point MAC (W rows are constants),
   top-8-bits address the cis ROM, weighted accumulate. ~200 pts × 240 dims
   per encode; 11 encodes per frontend match (the derivative shortcut is a
   measured clean negative — keep the re-encodes, they are cheap).
3. **Matcher**: 14 coarse rotation banks (index permutations of 2 encodes)
   × 289-offset translation ROM MAC + 9-θ fine stage × 49 offsets.
   ~2.6 M MAC-equiv/keyframe → 0.05 GMAC/s at 20 Hz: LUT-adds at these
   widths, no DSP pressure. Optional `front-consensus` mode (2 extra
   matches vs jittered bundle, decline unstable) where a guaranteed-tight
   spec matters: intel band collapses [2.44..6.72] → [3.84..4.15].
4. **Segment store**: BRAM. Active segment accumulates in full precision
   (1-deep buffer); freeze-on-anchor-advance quantizes to polar codes with
   per-ring scales. Budgets (audited): fr101 75 KB, fr079 110 KB,
   Intel 368–400 KB (6 b), MIT-1.9 km 625 KB (2 b lean) — an Artix-7-100T
   class part holds building-scale maps on-chip.
4b. **Readout capacity rule (measured 2026-07-10, "global readout"
   RESULTS entry)**: any bundle read must stay **≤ 32 segments at 2 bits**
   (≤ 64 at 6 b) — the superposition-capacity knee on the stata store; a
   building = ~8 grouped readout vectors. Do NOT widen the matched band
   for bigger reads (score dilution, measured monotone negative), and do
   not budget coarse-ring hierarchical descent for global relocalization
   (fails at K ≥ 128: the place-SNR wall at the readout layer). Full-map
   frontend reads are an ENVIRONMENT OPTION only (dense-revisit
   fr101-class: global 2-bit read scored 0.885 e2e; sparse-revisit logs
   break, law 2) — ship radius-grouped reads.

5. **Loop path**: candidate chain bundle (sum of ≤ ~6 stored vectors,
   dequant + permute + phase-add), one cmatcher match, per-ring coherence
   sums to the host. The loop layer is measured perturbation-ROBUST
   (1e-3 bundle noise → decisions unchanged); no consensus hardware needed.

## Accuracy contract (what to promise)

Per-regime bands, not points (PROTOCOL §6; the knife-edge points are
unstable fixed points with ~measure-zero basins — verified by divergence
trace to a single 0.4 mm boundary flip):

| deployment | config | promise |
|---|---|---|
| dense/furnished buildings | point + 6 b store + int8 | fr079-class [3.2..6.7], fr101-class [1.1..3.1] at ~10–25× less map than float |
| ditto, memory-critical | point + 2 b one-level store + int8 | fr101 1.82–3.1 @ 75 KB; MIT-scale 625 KB |
| sparse-beam cluttered (intel-class) | chord + 6 b + int8 | band [3.7..5.6] ≈ float band; FPGA8 is band-indistinguishable from float shipped |
| determinism-critical | + front-consensus | band width ~0.3 m — but the guaranteed LEVEL is log-dependent: intel lands at band-median, fr079 at the band TOP ([12.2..14.8] vs shipped [5.5..12.4], suite sweep 2026-07-11). Opt-in only, NOT default; 2.5× frontend compute |
| uncertainty-aware | + twin pipeline w/ 1e-3 dither, report divergence | online band/health estimate (the band is frontend-recurrence noise — BFC negative — so an envelope must be measured, not formulated away) |

## Fallback primitive (tested, not needed here)

Multi-bin phase-shifted binary (vernier) + dithered fractional encoding
(user's scheme, RESULTS 2026-07-10 later): at matched bits a flat k-bit
phase code + small cis ROM dominates on fidelity and e2e bands; the
vernier's per-bin sign-op datapath only pays on LUT-free substrates, and
dither only de-biases a 1-bin QPSK encode path. If a future target cannot
afford even a 256-entry ROM, this is the primitive to reach for.

## Binary-version spec sheet (bill of materials, computed 2026-07-10)

UPDATE (no-relo ablation, bit-identical on 3 logs): the lam 5.3/12.8 relo
rings are dead weight in the bounded deliverable — drop them for deployment
(D 360->240, every store/compute figure below improves ~33%: binary segment
204->136 B, Intel-scale map ~93 KB at 2 b / ~187 KB at 6 b, MIT ~417 KB).
Keep them only if the drought/global-relocalization extension ships. Scale
ladder stays {0.25..2}: lambda_min x store-bits is a per-sensor co-design
(fr101 finer-lambda wins float but breaks 2 b; stata the reverse; see
RESULTS "scale ladder x store bits").

Constants: W ROM 360x2 in turns/m Q3.21 = 2.2 KB; cis ROM 256x2x8b = 512 B
(64-entry quarter-wave folded: 128 B). Encoder: points (x,y) Q7.11 + 8-bit
weights; per point-component 2 phase MACs (24x18, fractional-turn top 8 bits
address the ROM) + 2 weighted MACs (8x8); accumulators D x 2 x int32 =
2.9 KB; active-segment buffer 5.8 KB; ~13 encodes/keyframe -> ~52 MMAC/s at
20 Hz = 1-2 DSP48 @ 100 MHz or LUT adders. Store: 2 b/phasor (4-PSK,
one-level clamp) + 6 per-ring 16-bit scales -> 204 B/segment; budgets fr101
68-76 KB / fr079 110 KB / Intel-scale 140 KB / MIT 480-625 KB. Matcher:
E-banks recomputed through the encoder datapath (0 BRAM; stored variant
139 KB coarse + 300 KB loop-coarse); rotations = index permutation; scores =
240-term complex dots 8x8->int32. Whole fabric (recompute-E): ~155 KB BRAM +
<=6 DSPs; Artix-7 100T fits with 4x headroom. Intel-class deployments: chord
sampling + 6-bit store (16-PSK x 4 mag ~= 560 B/segment). Accuracy contract:
the band table; arithmetic floor = 8-bit cis ROM (QPSK is median-only).

## Open items for the RTL session

- Fixed-point W·p MAC width vs. range (ranges ≤ 40 m, W ≤ 25.1 rad/m →
  phase ≤ ~1005 rad; 12.20 fixed point comfortably exceeds the 8-bit ROM
  address precision after mod-2π).
- The one-level 2 b clamp is verified e2e on fr101 only; run the band table
  for it if it becomes the default.
- Host relax: port the analytic-Jacobian TRF with max_nfev=30 semantics
  EXACTLY — the LM/PCG solver class measurably slides flat valleys (the
  live-JS demo's failure); early stopping is load-bearing.
  UPDATE (2026-07-11 session): "exactly" is not portable — the regularizer
  is scipy-TRF's *path* (radius schedule, x_scale='jac', eval order). Two
  routes: (a) vendor/freeze scipy's TRF (BSD) as a dependency-free f64
  reference, accepted band-vs-band per PROTOCOL §6; (b) RESEARCH: replace
  with fixed-budget truncated Krylov inside GN (LSQR/CG, hard iteration
  cap) — the canonical semiconvergent iterative regularizer (same Hanke
  lineage the audit cites), path-dependent truncation like the audited
  mechanism and unlike the REJECTED static Tikhonov prior. ~30 exactly-
  specifiable lines, no scipy pin. Subclass `_gn`, sweep cap × suite ×
  bands.
- Scale storage: f32 per ring is lazy; a shared 5-bit exponent + 8-bit
  mantissa per ring is plenty (quantization noise ≪ store noise).
  Harder variant worth one band run: power-of-two-only per-ring scales →
  the read-path scale multiply becomes a shift. (Per-RING stays required.)

## Bit-exact hardware restructurings (2026-07-11 session; no accuracy gate
## needed — assert equivalence in the model and take them)

DSP count was never binding; the value here is power, RTL surface area
(one multiplier family + index arithmetic), and headroom for all-frames
20 Hz. All three are bit-identical to the current model by construction.

- **A1 — octave bit-slice phases.** λ = 0.25/0.5/1/2 is exactly ×2 and W
  is stored in turns, so per point compute ONE projection per angle
  u_j = x·cosθ_j + y·sinθ_j at full precision; the four rings' cis-ROM
  addresses are BIT-SLICES of the same fixed-point word (ring k = top 8
  fractional bits of u_j/λ_min >> k; mod-2π = binary wrap, free). Cuts
  the wide phase MACs 4× (240 → 60/point). The spectrally-optimal ladder
  is also the only binary-hardware-native one. (Relo rings 5.3/12.8 don't
  share the base — one more reason no-relo is the deployment lattice.
  If Q-format ROM rounding breaks exact ×2 ratios, snap the entries.)
- **A2 — everything in r·LUT(az) form.** On a uniform-az head: x,y from a
  1024-entry az LUT (beams exactly on grid); the d/dθ fold term is
  −ω·r·sin(az−θ) — same quarter-wave ROM; the occlusion gate integerizes
  exactly (dr ≤ 2·tang ⇔ 5·dr² ≤ 4·chord², chord² = r0²+r1²−2r0r1·cosΔaz
  with cosΔaz ∈ {2 consts}); and since projection is LINEAR, bridged
  sub-points are AVERAGED projections (½, ¼/¾ shift-adds) — no new
  multiplies; sinc-integral argument = projection difference × per-ring
  shift → sinc LUT.
- **A3 — phase-domain map reads, generalized.** The stored code IS a
  phase index: rotate = address remap, translate/world-place = 8-bit
  phase-index ADD mod 256, conjugate = negate. One cis lookup per
  component at the end of the operator chain; only bundling and the
  δ·segder MAC stay complex. Every SE(2) op on frozen content becomes
  integer index/phase arithmetic.

Gated restructurings (bit-DIFFERENT → model-first, suite + band per
PROTOCOL §6): rotation candidates as az-bin index offsets end-to-end
(coarse every 4 bins = 1.406°, fine every bin = 0.3516°; the sampler is
az-shift-equivariant, so all trig and per-point rotation disappears);
streaming fold during the revolution + phase-ramp translation de-skew
(W·v once per rev, one add/component/beam) with nearest-bin az remap for
rotation de-skew; N_ANG × bits co-design on the wide-FOV suite (spot
lattice sweep: oct36 tied oct60 in the easy room — if 36–45 angles holds
on stata+synth-360+spot it is −25..40% on every store/matcher figure);
config registers in PHYSICAL units (seconds/meters — the spot stride-8
arm silently disabled closures because gap_kf/recent_aids are counts).

## iCE40 track (2026-07-11): target = iCEbreaker v1.0e (iCE40UP5K)

Board on the dev machine (FT2232H: iceprog ch A, UART ch B). Resources:
5280 LUT4, 8× MAC16 DSP, 30× 4 Kbit EBR (15 KB, dual-port — cis/az ROMs,
hit buffer, FIFOs), 4× 32 KB SPRAM (128 KB — the map). Slow fabric
(~20–48 MHz realistic). Toolchain: oss-cad-suite (yosys + nextpnr-ice40 +
icestorm), hw-in-the-loop over UART golden vectors.

Why the lean datapath fits: 2 b no-relo ≈ 136 B/segment → 128 KB SPRAM
holds ~900 segments ≈ an Intel-scale building (~95 KB); spot classroom =
14 KB. **Measured 2026-07-11: the 10× deployment env (school8 tour, 483
m²) is 301 segments = 35 KB of 2b phase codes at oct60 — 27% of SPRAM.
The oct36 co-design is NOT needed for fit and is REJECTED on accuracy
(N_ANG study, RESULTS 2026-07-11 second burst: stata proxy 0.202 → 0.975
shipped-float, loops 99 → 53; spot doesn't bind the knob).** Compute at
5 Hz keyframes: encode ~8 M mult/s with A1/A2 (one MAC16 at 24 MHz =
24 M/s); matcher at the spot-validated shrunk window (t_half 0.48→0.12
bit-identical there) is trivial; full 17×17 window ≈ one busy MAC16 —
corner-dependent.

v1 integer spec (golden model `ssp_ice40.py`, RTL under `ice40/`):
position unit = λ_min/256 ≈ 0.977 mm with the mm→unit factor FOLDED INTO
the az-LUT constants (no extra multiply); x,y from az LUT (exact beam
grid) as i16 → hits beyond ±32.7 m masked at encode (classroom/school ok;
measure the mask's cost on long-corridor synth before trusting it at
range); u_j via 2-MAC dots against 60 (cosθ_j, sinθ_j) ROM pairs on
MAC16; ring addresses by A1 bit-slice; cis ROM 256×2×7 b (EBR); int32
accumulators; per-ring shift-only scales if the band run allows (else
5-bit-exp + 8-bit-mantissa). The single-mult r·cos(az−θ_j) variant (4096
LUT, ≤0.044° angle snap) is a corner-S option — band-check the snap.

Three optimization corners (S and T guide experimentation; F is the
deliverable). **Status 2026-07-12 EVENING (build session — every corner
now carries MEASURED yosys/nextpnr/icetime numbers on UP5K; the corner
tension forced two architectural solutions: the LEAN CORE and the
control-as-microcode plan):**
- **S — smallest footprint that solves the task.** Matcher space stays
  CLOSED at oct60 (oct36/45 rejected). NEW: `encoder_lean.v` — the
  serial-ring core the corner forced: ONE 256×32 acc memory pair, ONE
  weight tree, TWO shared MAC16 (r·az / fold rotate / u-projection /
  match products), az ROMs quarter-wave folded 8 EBR → 1 (exhaustive
  fold-equality checks in gen_luts.py). **Measured: 2034 LUT / 6 EBR /
  2 MAC16 (v6 core: 3361/28/8); placed probe 2694 LC, 26.06 MHz — 24
  closes.** Cost: ~1385 cyc/cand vs 488 (the solo task needs ~250
  cand/kf at 5 Hz = 500× less than v6 provides — throughput was the
  right thing to sell). Stage-1/2 gates bit-exact (bridge 32/32,
  l1 220/220).
- **F — max fidelity. The fidelity space + laptop decode.** span11x60
  recipe unchanged (banked). NEW (scratch_liveness, "tune scales"):
  the FREEZE STORE contract — 2b phase + liveness bit (theta=1/2:
  2·M ≥ Mmax_ring, M = max|I|,|Q| + min>>1) + per-ring scales (Mmax_r
  as 5-bit exp + 8-bit mantissa) **beats the float store on extraction
  on both fixtures** (school .6138 vs .5984; stata .6714 vs .6677);
  38 B/segment on the wire; DUMP-side only (masked local reads
  measurably worse — the on-chip matcher keeps raw 2b codes).
  Implemented in top_solo.v freeze passes A/B/S + solo.liveness_int +
  decode.py planes.
- **T — max throughput at max usage.** REBUILT for the record: v6
  deploy (top_match3) = **5142/5280 LC (97%), 30/30 EBR (100%), 8/8
  DSP, icetime 30.22 MHz** — maxed on every axis, 488 cyc/cand, 6.95k
  poses/s; encoder-only build **40.32 MHz** (36 closes). The T corner
  is the EXISTING silicon-accepted architecture; its parallel-ring
  pipeline is exactly what the S corner strips.
- **SOLO — standalone SLAM on fabric (v7): RTL COMPLETE, SIM-GATED;
  UP5K fit = the measured wall.** Full function (ingest + track + fold
  + freeze incl. liveness/scales + dump + preload, top_solo.v):
  **DSP 3/8, EBR 9/30, SPRAM 3/4 — comfortable; LUT 7452/5280 = 1.41×
  over.** Attribution: encoder 2032 + tracker 2955 + top 2659 + uart
  54 — the VSA datapath is NOT the wall; the scalar SE2/control fabric
  is (5614 LUT). Even localization-only exceeds the device as
  parallel-word FSMs; hand-CSE bought zero (abc9 already shares).
  **v7.2 fit plan (filed): microcoded scalar engine (~650 LC + ucode
  in EBR — the exact resource the lean core freed), est ~2.9k LC
  total.** Gates: k1/rl regressions 220/220+160/160 (edited tracker,
  old core), lean l1 220/220 bit-exact; kidnap + top-level end-to-end
  gates were in flight at write time (see RESULTS build-session
  entries for the closure). The demo runs bit-exact in sim; the v6
  deploy build remains the flashable hardware today.

Synth-env rework for the 10×/dynamic deployment (`ssp_dynenv.py`, new
module — imports worlds/ssp_synth, no shipped edits): classroom
(spot-proxy 7×7), school10x (hallway + N IDENTICAL connected classrooms —
deliberate aliasing stressor at the twin-wall finding's scale), people as
circle obstacles r≈0.2 m, static ("standing") or waypoint-walking
~1.2 m/s, deterministic seeded paths, frozen per frame (motion-during-
sweep = the de-skew ablation, later). Paired-seed controls (identical run
± people) isolate the dynamic-actor cost; knobs: people count {0,2,5,10},
static/walking, per-room furniture jitter (identical vs distinguishable
rooms). Metrics: ATE + loop precision/recall (GT edge labels) + veto/
inflation fire counts + map growth.

No-gos (measured; do not re-litigate in RTL): sub-8-bit cis ROM, QPSK
arithmetic, BPSK store, per-vector scales, matcher-side d/dθ, >32-segment
reads @2 b, global reads on sparse revisit, chord resampler at 1024
beams, golden-ratio ladders (re-confirmed 2026-07-12 on the scale axis:
school8 2b edge inverted to −.06 on stata). Added 2026-07-12: angle
counts >60 in ANY space; sub-coherence fine rings (λ < 2πσ_r);
global-bundle readout (per-segment decode only); BSC-style binary
bundles at equal bits (knee K*≈8–16 vs phasor 32–48); spare-SPRAM sample
reservoir re-folds (GT-neutral — common-mode estimator; revisit only on
bad-odometry platforms); fold-only-on-commit gating (starvation
feedback); K_TRY>1 while self-mapping a gauge-warped map (gauge mixing);
time-bound superposed history for anneal/replay (explicit frozen codes
or nothing).

## Deploy addenda (2026-07-11 late session — reintegrated verdicts)

**Sampling (the deployed bridge2 n=2 is load-bearing, twice over).**
GROUP run-length decimation is bench-only: GROUP/2/4/8 all collapse
stata end-to-end in DEEP basins (ε-stable) despite per-scan
losslessness — do not transfer it to real sensors; the encode-budget
option it promised is instead available as interp-n1 (ONE sub-point per
chord: equal-or-better on every DENSE-head log at half the terms, but
band-separated regressions on sparse 180° logs — dense-only option,
pending school-s12/fhw-band/audit). Mechanism (supersedes any
per-chord-density intuition): packs whose fine-ring encode sits within
~2e-3 (r0 rel) of the EXACT segment integral fall into stata's bad
cascade basin — the exact-integral limit itself is IN the bad basin;
sampling discreteness rescues. Deployed n2 sits at 7.6e-3 with margin.

**Store.** Uniform 2b stands. Per-ring bit allocation: rejected as a
recipe (stata-lean is BAND-dominated — even 6b collapses at ε=1e-3);
two REGIME LEVERS filed: asymmetric [16,16,4,4]/[16,16,8,8] raises
school8-class closure PRECISION (0.78→0.82–0.84) where uniform-6b
lowers it; chain-cap=8 tightens stata-lean's band 1.48→0.29 med (the
only stata-lean stabilizer observed; incomplete validation). NEP-50
trap: quantizer vectors must be assembled from scalar calls (broadcast
f64 breaks c64 bit-faithfulness above the 1e-6 cascade threshold).

**Map readout (deployed live stack, GT-verified on silicon).** The
matched octave ladder is coherently aliased at exactly 2 m → interior
ghost combs; profile-shape rules cannot fix it. Deployed fix: a
self-certified FREE-SPACE MASK ray-carved at freeze from the system's
own pass-1 scans (0.30 m cells, stop 0.45 m short, then UN-free cells
within 0.35 m of hit endpoints — tangential rays otherwise bleed free
into wall cells), O(area) ~0.1–1 KB. First-return rule: smoothed-
profile prominence peaks → mask+trajectory veto → first survivor ≥0.55
of surviving max → fabric refine; mask-boundary fallback tier for
comb-inverted/uncovered rays. Verified: direct hits p50 0.150 m,
cached 0.224 m vs GT (baseline 0.80). min3 ring reader = display layer
only (marching it naked re-admits the combs). Raster AUC ceiling ~0.63
is representation-level: correlation reads are the map's fidelity.

**Localization tracker posture.** Grid sweep (9×9×5 batched 0x08) +
score-EMA hold + wide re-search stands. Per-keyframe surface-
observability commit gating is a MEASURED dead end (best statistic:
83% catch at 11% false-flag = drift-inducing); corridor-class
degeneracy is the wall in tracking form (confident wrong peaks).
Open: temporal statistics, RES-H line sweeps (5× fewer evals,
bit-exact same argmax 93–98%, prior-free heading 14×) + analytic
Newton refine (2× coarser grids) — both fabric-shaped, filed.

**Wire/protocol.** cmd 0x08 batched sweeps (FIFO-buffered, totals or
per-ring replies) is the deployed transport: no host pacing invariant,
6.6k poses/s at 3 Mbaud; the OS-write-coalescing crash class is dead.
36 MHz for the match build needs the stage-B split or mc-prefetch
cis-address fold (encoder-only build already at 36). DSP count is
asserted ==8 in the build flow (inference regressed silently twice).

**New hard bounds for any future claim.** 2b partial-overlap segment
peaks sit med 5.9 cm from geometric alignment (content pull) — floors
every matcher/refiner vs GT on this store. Bundle capacity knee is
superposition-only (2b member noise is nil; naive iid theory refuted —
spatially-ordered members crosstalk less). Same-place pose pairs sit
2.3–5.8 m apart in est frames on fr101 — bounds any position-derived
candidate generation.

## v7 "top_solo" — standalone SLAM on fabric (spec pinned 2026-07-12; golden contract = ice40/host/solo.py)

Requirement (user): chip runs SLAM alone; host = sensor cable (points
in, pose out) + map dump. Golden model VALIDATED: integer tracker ≡
live FabricLoc to 2 mm trace median (classroom/mixed); self-mapping
works (classroom 0.073 vs odo 0.281; odometry-grade on long tours —
v7.1 frontend port filed separately). Build as a SEPARATE top
(frozen-versions pattern); v6 deploy top unchanged.

Memory map (UP5K, 4×32 KB SPRAM):
  SPRAM0: mc working buffer (as v6, prefetch path unchanged)
  SPRAM1: RESIDENT MAP — 2b segment codes, 60 B/seg @oct60 matcher
          space -> ~540 segs/SPRAM; addressed by seg index (the v6
          4-phase prefetch generalizes: base = seg*240*2b)
  SPRAM2: anchor table (x_u i32, y_u i32, ah_q i16, pad -> 16 B/seg;
          540 segs = 8.6 KB) + pose/state regs spillover + (option)
          fidelity-space open-segment accumulator D<=660: 10.6 KB
          int32 I/Q vec+der — fits alongside
  SPRAM3: free (reservoir verdict: NEGATIVE on bench odometry — spend
          on more map or bigger fidelity D; revisit only on SPOT hw)

New blocks (all integer, constants from solo.py):
  - nearest-anchor scan: linear pass over anchor table, min (dx²+dy²)
    i64 compare; 540 segs ≈ 1.1k cycles — negligible at kf rate
  - rel/pose Q15 rotate: 4 MAC per transform; heading ROM = 240-entry
    quarter-wave (cs_of contract: round(32767·cos))
  - candidate grid gen FSM: 7×7×5 @ SWEEP_T=12 around -t_u seed
    (edge-recenter retry ×1); re-search 15×15×7 @48, rho step 2,
    every 12 lost kf
  - EMA/commit: ema += (score-ema)>>6; commit iff score > 29·(ema>>6);
    relock iff > 35·(ema>>6) — shift-add + one 32×6 multiply
  - parabolic refine: (T·(a-c))/(2·den) trunc-toward-zero, clamp ±T/2
    — small serial divider, off critical path (bench: p90-only gain)
  - fold-at-pose: Q15 rotate+translate INSERTED between az->xy and
    u-projection (encode_int_at contract; envelope asserts i16);
    der-accumulate FREE of new projections (cross_a = ±u_{a∓30} index
    shift; ring scale 2π/λ applied at laptop decode; operand cross>>2)
  - freeze: mcode_int comparator (|I|>=|Q| + signs; tie -> I-axis),
    ≡ mcode_from_vec proven; write codes+anchor, clear accumulators,
    5-kf cadence; fold UNGATED (gated folding REJECTED — starvation)
  - UART: pose frame out per kf (x_u i32, y_u i32, h_q i16 + state
    byte ≈ 11 B -> ~22-55 B/s); map dump cmd (codes+liveness+scales+
    anchors); preload cmd (v7a mode); RESET-MAP cmd 0x29 (env switch:
    n_seg=0 + open segment discarded, pose kept — gate: reset mid-tour,
    dump contains only post-reset segments); fidelity segment stream

LC risk: v6 deploy = 5148/5280 with host-streaming paths that solo
mode retires (bare-0x07 path, FIFO cand ingest can shrink — grid gen
replaces it). Mitigations if over: drop parabolic divider (p90-only),
est 4-bit trims (precedent), single-segment sweep only (no handoff —
HANDOFF_M=-1 in live anyway).

Acceptance (per protocol): (1) sim gate — icarus tb replays recorded
(q_ints, pred) sequences, asserts pose/state/score BIT-EXACT vs
solo.py step() and fold/freeze vs SoloMapper (incl. mcode ties);
(2) hw-replay — same sequences through UART, zero mismatches;
(3) live selfmap demo — classroom/orbit expects ~0.07 med (banked
golden bench), pose stream + map dump decoded laptop-side.

### v7 stream-format + consumer addendum (2026-07-12 cleanup program)

READ-SIDE RECIPE (banked, both fixtures): the laptop consumer is
`ice40/host/decode.py` — loads the chip dump (map hex + anchor hex),
computes the GT-free per-ring COHERENCE profile C_r (map-health metric;
low fine-ring C_r = warp signature), images with Cr^2 Wiener weights
(+.016 warped / +.005 clean AUCgt), and extracts parametric walls via
per-segment CLEAN pursuit + cross-segment consensus (gamma = the banked
recall/precision dial). Validated on the v7 classroom fixture: 26 segs
-> 117 consensus lines, p50 0.232 from the PHASE-ONLY oct60 dump.

STREAM-FORMAT UPGRADE (filed for the fidelity space): 3b/component
freeze — 2b phase + ONE true-zero magnitude/liveness bit — BEATS THE
FLOAT STORE on extraction on both fixtures (school .6065 vs .5984
float; stata .6691 vs .6677) and improves local-read tails (p90 .157->
.112). On-fabric cost: liveness = |acc| >= theta * per-ring scale — a
per-ring max scan (same machinery as the T_SH shift scan, per-ring
granularity) + one compare at freeze; +D/8 B/segment on the wire.
nmag=1 has NO zero level (dead=True divides by zero) — the liveness bit
IS the zero level. Per-ring scales (5-bit exp + 8-bit mantissa, already
spec'd) ride along for the fidelity stream.

Cleanup no-gos (measured, do not re-litigate): amplitude-based pursuit
pruning (junk atoms carry healthy joint amps — tail is MODEL-limited;
point/corner atoms are the lever); primitive-space gauge relaxation on
sequence-adjacent pairs (neutral — neighbors already share gauge; slow
warp needs closure-backend corrections, and cross-pass constraints are
the wall); parameter shadowing in hot loops (defect class — the
reproduction gate catches it).

### v7.2 fit plan — control-as-microcode (filed 2026-07-12 build session)

The measured wall: tracker+top = 5614 LUT of scalar SE(2)/gating/
protocol FSMs vs 2090 for the whole VSA datapath (encoder+uart+pll);
UP5K = 5280. The closure moves control into the memory the lean core
freed (EBR 9/30 used, SPRAM 3/4):

  - nano-engine: 32-bit accumulator machine, ~16 ops (add/sub/cmp/
    shifts/and/or, load/store to a small regfile, IO get/put, branch),
    ucode in 1024x32 EBR (8 EBR, $readmemh — SPRAM is NOT bitstream-
    initializable), regfile 32x32 in 2 EBR. Est ~650 LC.
  - IO ports = the existing module interfaces (encoder start/clear/
    m_start/rd_idx, tracker step/svc, SPRAM2/3, uart) — the FSMs'
    register writes become put-ops.
  - The MATCH/ENCODE inner loops stay hardware (the lean core is
    untouched); the grid-issue/argmax loop stays a small hardware
    sequencer (perf: 245 cands x ~30 ucode cycles would still be fine,
    but hardware is already written and gated).
  - ucode assembled by a python script that shares constants with
    solo.py; acceptance = the SAME fixtures (l1/lr/solo_top) bit-exact.
  - Est system: encoder 2032 + engine ~650 + grid sequencer ~300 +
    uart/pll ~150 ≈ 3.1k LC, EBR ~19/30 — fits with margin; SPRAM3
    still free for a bigger resident map or the span11 fidelity
    accumulator.
