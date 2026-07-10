# FPGA design note — SSP SLAM datapath (from the 2026-07-10 measurements)

A concrete architecture sketch for taking the bounded-memory SSP SLAM stack
to fabric, grounded entirely in tonight's audited numbers (RESULTS
"2026-07-10" sections; `ssp_fpga.py` is the bit-accurate reference model).
This is the hand-off document for an RTL session.

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

1. **Sampler** (per-environment mux — measured, per-log):
   - `point` mode (default for dense/furnished: fr079/aces/belg/fr101):
     one phasor per hit, w = r·dθ. No geometry preprocessing at all —
     a pure stream. Band-dominant on fr079/aces/belg.
   - `chord` mode (intel-class sparse-beam clutter): the shipped
     resampler. Irregular but small; silhouette emphasis (double-write of
     depth-jump hits at r·dθ) is a frontend-only lever, not worth fabric.
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
| determinism-critical | + front-consensus | band width ~0.3 m at band-median accuracy |
| uncertainty-aware | + twin pipeline w/ 1e-3 dither, report divergence | online band/health estimate (the band is frontend-recurrence noise — BFC negative — so an envelope must be measured, not formulated away) |

## Fallback primitive (tested, not needed here)

Multi-bin phase-shifted binary (vernier) + dithered fractional encoding
(user's scheme, RESULTS 2026-07-10 later): at matched bits a flat k-bit
phase code + small cis ROM dominates on fidelity and e2e bands; the
vernier's per-bin sign-op datapath only pays on LUT-free substrates, and
dither only de-biases a 1-bin QPSK encode path. If a future target cannot
afford even a 256-entry ROM, this is the primitive to reach for.

## Binary-version spec sheet (bill of materials, computed 2026-07-10)

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
- Scale storage: f32 per ring is lazy; a shared 5-bit exponent + 8-bit
  mantissa per ring is plenty (quantization noise ≪ store noise).
