# EXPERIMENTS — operational guide + module catalogue

How to run things, the standard harness patterns, and an index of every
experiment with its one-line verdict and where to read the detail. Findings live
in `docs/FINDINGS.md` (synthesis) and `docs/RESULTS.md` (chronological, cites section
names). Discipline lives in `docs/PROTOCOL.md`.

Modules in this directory are frozen once their verdict lands. They run from
the repo root as `python3 -m experiments.<name>`. Before the 2026-07-13
restructure they lived at the repo root as `ssp_<name>.py` — ledger entries
cite those historical names; the mapping is `docs/RESTRUCTURE-MAP.md`.

---

## Datasets

CARMEN `.log` (odometry + laser); `.gfs.log` is the GMapping/RBPF-corrected
reference used **only** for scoring. Under `data/` (gitignored, fetched).

| key | file | role |
|---|---|---|
| stata | `data/stata/2012-01-27-07-37-01.bag` | **primary** — floorplan-anchored GT (independent-class reference); 260° FOV, nearest to the SPOT/360° target |
| fr101 | `data/fr101.log` | **primary** — dense-revisit building, point-stable log |
| fhw | `data/fhw.log` | **primary** — exhibition hall, 559 closures |
| fr079 | `data/fr079.log` | **primary** — zero-shot; perturbation-banded (read as bands) |
| belg | `data/belgioioso.log` | **primary** — non-Manhattan castle; range-identity eval |
| aces | `data/aces_publicb.log` | secondary — banded, odo already good |
| mit | `data/MIT_Infinite_Corridor_2002_09_11_same_floor.log` | stress-only — corridor capstone; range-identity eval |

**intel was REMOVED from the suite (user call, 2026-07-10)**: diverging
behavior class, GMapping-referenced GT, 180° FOV vs the deployment target
(SPOT, 360° lidar). The `ssp_datasets` loader keeps the entry only so
historical `docs/RESULTS.md` numbers stay reproducible. The synthetic bench
(`experiments/sampling.py synth`, 360° FOV, exact GT) joins the acceptance suite as
the target-matched environment.

**One registry for all of it: `runners/datasets.py`** — `load(name)`,
`evaluate(bundle, fin)`, `run(name, cls, sample=..., **kw)`. The run loop is
a bit-exact transcription of `ssp_fpga.run_log` (asserted in
`python3 -m runners.datasets selftest`); eval dispatches per dataset: `gfs`
nearest-timestamp <0.3 s, `ident` range-array identity (belg/mit — their gfs
timestamps are corrupt), `stata` floorplan GT <30 ms. Quick single runs:
`python3 -m runners.datasets run belg [shipped|e2|lean]`.

## Run recipes

```bash
python3 -m runners.carmen data/intel.log          # shipped deliverable + ATE
python3 -m runners.carmen data/fr101.log 1893      # optional keyframe cap
python3 -m baselines.icp data/intel.log                 # ICP+posegraph baseline
python3 -m baselines.csm  data/intel.log                # correlative-grid baseline
python3 -m baselines.rbpf data/intel.log                # RBPF/GMapping-lite baseline
python3 -m sspslam.bench                                   # synthetic GT-labelled loop bench
```

## Harness patterns

**Subclass the backend** (new solver/admission behaviour), prove bit-exact when
neutralised:
```python
import sspslam.bounded as B
class MySLAM(B.BoundedSLAM):
    def _gn(self, ...):
        if not self.use_feature:      # neutralised path MUST be bit-exact to parent
            return super()._gn(...)
        ...
```

**Run the shipped loop with a parameterised encoder** (sweep a lattice constant
without editing the module) — `sspslam/lattice_presets.py` is the one committed home:
```python
import sspslam.lattice_presets as LP, runners.datasets as DS
LP.set_polar((0.125, 0.25, 0.5, 1.0), n_ang=60)   # any ladder/count
LP.set_hex(63)                                     # full-circle hex
LP.shipped()                                       # exact restore
r = DS.run("fr101", spec=None, nph=0)              # then run
```
(`python3 -m sspslam.lattice_presets` asserts set_polar()==shipped globals exactly and
hex rotation==permutation.) Validate any such harness reproduces a known
shipped number before trusting a sweep (fr101 → 1.881 m at the baseline
lattice).

**Reuse the flow field** (`experiments.flow.FlowField`): pose-pose stiff edges +
analytic SSP-correlation overlap force, grad-L2 normalised. `experiments/hybrid.py` shows
building it from a shipped `BoundedSLAM` and from a two-session split.

## Module catalogue (verdicts + pointers)

**Shipped core** — `sspslam/bounded.py` (BoundedSLAM: rigid 5-kf segment vectors +
d/dθ, continual constraints, IRLS + LOO pruning, PCM admission),
`runners/carmen.py` (driver+eval), `sspslam/lattice.py` (polar lattice globals +
rot-permute + relo), `sspslam/encoder.py` (encoder/matcher), `sspslam/frontend.py`
(frontend). `sspslam/worlds.py`/`archive/experiments.py`/`experiments/angles.py` = encoder-design studies
(oct grid beats RFF ~10–20×; derivative vector load-bearing).

**Baselines / classical controls** — `baselines/icp.py` 1.70 m, `baselines/csm.py`
3.27, `baselines/rbpf.py` 0.12* (*ref is itself GMapping — circular, not a win);
`baselines/scancontext.py` (Scan-Context control: the revisit signal IS in the raw scan,
so the SSP corridor limit is representational not environmental).

**Corridor / verification wall** (all CLOSED — `docs/FINDINGS.md` §5) —
`experiments/ringkey.py` retrieval solved (recall 0.32→0.81) but consensus harder;
`experiments/seqslam.py` sequence ambiguous (0.500 = chance); `experiments/cascade.py`
coarse→fine profile veto (precision not recall); `experiments/mit_gtverify.py` perfect
place oracle only 42.66→41.04 (place necessary-not-sufficient; wall is two-fold
place AND pose); `experiments/aniso.py`/`experiments/aniso_mit.py` observability-weighted
anisotropy — real but fragile/regime-specific, non-shippable (MIT oracle 41.04→
39.12 diagnostic only).

**Multi-session / composition** (`docs/FINDINGS.md` §5.6 + multi-session sections) —
`experiments/multisession.py` VSA composition works (bundling=addition, 37% overlap
saving); `experiments/multisession_align.py` naive win was a shared-index leak (audit);
`experiments/multisession_ringkey.py` appearance retrieval works, verification is the wall;
`experiments/multisession_verify.py` oracle-free verifier — PCM clique≥2 best (5.54→3.35 m)
but doesn't reach 2.49/2.44, wall narrows not closes; `experiments/percluster.py`,
`experiments/adaptmap.py`, `experiments/iteralign.py` = map-handling probes (per-cluster split,
adaptive-map memory win, iterative-align negative).

**Memory / hierarchy** — `experiments/hier.py` HY4 adopted (0.67× memory, bit-identical);
`experiments/hiergraph.py` coarse-to-fine relax = no gain (flat relax already optimal);
`experiments/scale_arrays.py` **spatial O(area) submaps FAIL for closure** — persistent
per-area cells smear write-time content across drift states, biasing every loop Z
(4.34→11–15 m); law: a closure primitive must hold single-burst drift-consistent
content (why shipped uses ephemeral *temporal* segments).

**Frontend do-no-harm** (`docs/FINDINGS.md` §6, CLOSED triple-negative) —
`experiments/frontguard.py`, `experiments/frontsys.py`, `experiments/belief.py`, `experiments/posefilter.py`:
"odometry is already good" is not scan-match-observable; needs an independent
odo-quality signal this suite lacks.

**This session's paradigms** (`docs/FINDINGS.md` §5.6) — `experiments/flow.py` continual
detection-free gradient-flow (valid/non-folding, but drifts from GT-perfect init;
refines within-basin only; grad-L2 + soft/hex = conditioning wins);
`experiments/hybrid.py` detected anchors + overlap-flow (single-session net-negative;
multi-session tightens true overlaps ~8–16% but worsens ATE via twin
contamination — does not beat anchors-alone; audited SOUND);
`experiments/viewpoint.py` viewpoint dual-channel (compose/dedup WIN corr +0.93; NOT a
discriminator — Intel AUC 0.896 < 0.909 proximity confound, MIT below chance).

**Bench** — `sspslam/bench.py` synthetic multiloop world with GT edge labels
(precision/recall on closures without a real dataset).

**FPGA track (2026-07-10)** — `sspslam/quantized.py` (subclass-only; neutralised paths
asserted bit-exact): `QuantStoreSLAM` write-time per-ring polar-quantized
store; `NoiseStoreSLAM` freeze-noise chaos control; `DerFineMatcher` E1
(negative); `points_from_scan` E2 per-beam point encoding (frontend win on
every log, audited); `IntSpec`/`IntSLAM` integer front-path arithmetic model;
`BandSLAM` + `band_table` the perturbation-band acceptance harness (PROTOCOL
§6). Run recipes:
```bash
python3 -m sspslam.quantized selftest          # neutralised == parent, bit-exact
python3 -m sspslam.quantized ops               # hot-path MAC / map-bit sizing
python3 -m sspslam.quantized sweep --ring intel  # write-time quant ladder
python3 -m sspslam.quantized front fr101 intel   # E1/E2 frontend variants
python3 -m sspslam.quantized int fr101           # integer arithmetic ladder
python3 -m sspslam.quantized chaos intel         # freeze-noise response curve
python3 -m sspslam.quantized band fr079          # config x perturbation band table
python3 demo/export_replay.py stata --config=shipped   # Python-fed webvis
python3 demo/export_replay.py --embed   # splice manifest into index.html
```
Verdicts in docs/RESULTS.md "2026-07-10 — FPGA track opened" (+ the consolidation
section that follows it): quantized store needs PER-RING scales; ≥6-bit quant
deltas on Intel are not attributable (perturbation band); E2 = frontend
registration win 5/5 logs, closure-layer per-log; arithmetic knee = 8-bit ROM
cis; 2-bit phase-only store viable on robust logs (fr101 band [1.1–2.7] at
75 KB); QPSK arithmetic = median-only.

**Stability / decision-flip thread (2026-07-10)** — `experiments/stablegate.py`
(JitterMixin loop-layer dither-consensus: bands narrow but medians drift;
ConsensusMatcher FRONTEND jitter-consensus: collapses the intel band 14×
onto the median — the mitigation candidate); `experiments/bfc.py` (bundle-frame
smooth closure factors: clean negative — band is a recurrent-estimator
property, not a factor-shape artifact). Both carry the verbatim copied
`try_constraint` body pattern (neutralised → bit-exact, asserted).

**Datasets / lattice infra (2026-07-10)** — `runners/datasets.py` (registry +
the one shared run/eval harness — see Datasets above); `runners/stata.py` (bag
adapter, floorplan GT; shipped 0.202 m); `experiments/hexreal.py` (full-circle hex
lattice e2e on real logs: belg 2.071 vs polar 2.644, intel control loses —
per-environment option); `sspslam/lattice_presets.py` (set_polar/set_hex/shipped —
consolidated lattice patching, replaces per-scratch copies);
`runners/synth.py` (SPOT-proxy synthetic 360° bench over sspslam/worlds.py — exact GT,
emits ssp_datasets bundles, deterministic local rng; `bench`/`fov` CLIs).

**Encoding/sampling family (2026-07-10, user thread)** — `experiments/sampling.py`:
`sample_interp(n_sub, cut_deg, w_mode)` multi-sub-point bridging with the
occlusion gate angle-parameterized (shipped OCC_RATIO 2.0 == 63.4°;
cut<0 == E2 points exactly); `pack_segint`/`pack_arcint` + `SegIntEncoder`
EXACT line-integral encoding (sinc(k·d/2)·phasor — the principled form of
the thermometer blanking; endpoint packing keeps matcher transforms
linear); `renorm_alpha` restores blanked-ring energy on surviving rings
(the blanking-vs-weighting question); `SegCore` copied-body override at the
BoundedSLAM MRO level (Quant/Band layers compose above). Selftest:
quadrature vs 400-pt sub-sampling, analytic dθ derivative vs numeric,
degenerate-pack primitives bit-exact, e2e ATE-equivalent (ULP-amplified,
documented). Run: `python3 -m experiments.sampling [selftest|interp|segint|renorm
<logs>|synth]`.

**Target platform (2026-07-10)** — `runners/spot.py`: SPOT Telluride adapter
(ring-34 slice of 1024×64 Ouster clouds @ 20 Hz → 1024-beam scans;
lidar-only protocol, CV guesses, 528-Hz odometry withheld as GT — SORTED
by timestamp; the parquet carries a ~62 s out-of-order block that
teleported 3 keyframes' reference pre-fix; hygiene mask guards eval).
First contact: **float 0.039 ≡ FPGA-lean 2b+int8 0.039 (med 3.3–3.5 cm)
at 14 KB vs 354 KB map** — the reference's own noise floor (with-odom
diagnostic 0.041); all samplers/lattices/windows tie in the 7×7 m room
(t_half 0.48→0.12 bit-identical outcomes — shrink freely for compute).
Registry name `spot` (guess_mode="cv" in the shared runner). Webvis: spot
float + lean replays lead the pack.
Capacity/global-readout scripts (`scratch_capacity*.py`,
`scratch_fullread*.py`, kept in archive/scratch after banking): bundle-K
knee 32@2b/64@float, deadband mechanism, grouped global decode.

**iCE40 hw-in-the-loop track (2026-07-11)** — `hw/ice40/golden.py` (golden
integer model: az-LUT + A1 octave bit-slice phases + cis ROM + i16/i18
envelope; bit-exact spec for the RTL; fidelity vs float ≥0.99999 on
spot/synth) and `hw/ice40/` (RTL + build + host: `rtl/encoder.v` v1 serial,
`rtl/encoder_par.v` v2 ring-parallel 63 cyc/pt — both verified BIT-EXACT
on the iCEbreaker; `top_encoder.v` UART protocol with always-listening
parser + FIFO + bulk mode at 3 Mbaud). Corner-S encode acceptance PASS:
684 scans (spot 414 + dynenv 270) through the fabric, 0 mismatches. Run:
```bash
python3 -m hw.ice40.golden selftest              # golden-model invariants
python3 hw/ice40/host/gen_luts.py             # ROM images (shared w/ RTL)
python3 hw/ice40/host/vectors.py gen 64       # test vector + golden
python3 hw/ice40/host/vectors.py sim rtl/encoder_par.v   # iverilog bit-exact
cd ice40 && make TOP=top_encoder_par \
  RTL="rtl/uart.v rtl/encoder_par.v rtl/top_encoder.v" build prog
python3 hw/ice40/host/vectors.py hw-replay '' spot        # zero-glitch gate
```
Verdicts in docs/RESULTS.md "2026-07-11 — iCE40 hardware-in-the-loop track".

**iCE40 second burst (2026-07-11): corner T (36 MHz) + corner F
(matcher on silicon)** — `rtl/encoder_pipe.v` v3 (3 pipe stages, cis ROM
forced to 4 EBR replicas, bit-sliced readback; 39.7 MHz icetime, 2933
LC) + `rtl/enc_top.v`/`top_direct.v`/`top_pll.v` (32-bit FIFO wrapper,
36 MHz PLL — acceptance PASS 684 scans, 0 mismatches at 36 MHz);
`rtl/encoder_match.v` v4 + `enc_top_m.v`/`top_match.v` (24 MHz): the
candidate-pose matcher — golden `ssp_ice40.match_int` (per-ring partial
sums; H = conj(Q)·i^mc sign/swap; rotation = permute+conj-wrap; per-scan
shift), M codes in SPRAM, 483 cyc/candidate, cmds 0x06/0x07. hw-match
BIT-EXACT (22 candidates × 4 rings). Run:
```bash
python3 -m hw.ice40.golden match                 # golden matcher geometry pins
python3 hw/ice40/host/vectors.py sim rtl/encoder_pipe.v rtl/tb_encoder2.v
python3 hw/ice40/host/vectors.py sim-match    # matcher RTL vs golden
cd ice40 && make TOP=top_pll FREQ=36 \
  RTL="rtl/uart.v rtl/enc_top.v rtl/top_pll.v rtl/encoder_pipe.v" build prog
cd ice40 && make TOP=top_match FREQ=24 \
  RTL="rtl/uart.v rtl/enc_top_m.v rtl/top_match.v rtl/encoder_match.v" build prog
python3 hw/ice40/host/vectors.py hw-match     # silicon vs match_int
python3 hw/ice40/host/vectors.py hw-match-sweep   # 441-cand paced sweep, 3.6k/s
# v5 (encoder_match2/enc_top_m2/top_match2, cmd 0x08 batched via FIFO):
cd ice40 && make TOP=top_match2 FREQ=24 \
  RTL="rtl/uart.v rtl/enc_top_m2.v rtl/top_match2.v rtl/encoder_match2.v" build prog
python3 hw/ice40/host/vectors.py hw-batch     # both modes bit-exact; 6.6k/s totals
```
N_ANG×bits study (`scratch_nang.py`, self-validated vs banked registry
numbers): **oct36/45 rejected** — stata 0.202→0.975/1.203, loops 99→53/33;
spot indifferent; fr101@36 = knife-edge draw. oct60 stands; school8 map
= 35 KB @2b/no-relo/oct60 (27% of SPRAM). RESULTS "second burst" entry.

**Live FPGA-in-the-loop vis (`hw/ice40/host/live.py` + `live.html`)** —
endless perturbed classroom replay (interpolated bridge closes the tour;
per-pass noise redraw; nothing resets), python SLAM forever + fabric
encode cross-check + fabric localization vs the frozen 2b map + fabric
matched-filter map image (delta probe) + pickable-viewpoint visibility
(cached image ray-march, or every ray sample probed live from silicon).
Geometry matcher↔SE2 pinned empirically at startup (hard-fail).
`python3 hw/ice40/host/live.py selftest` (headless numbers; enc 24/24,
fx err ~3 cm, direct-vs-cached walls 0.15 m median) then
`python3 hw/ice40/host/live.py serve 8642 2` → http://127.0.0.1:8642/
(needs the top_match bitstream flashed; owns the serial port).

**Dynamic multi-room environments (2026-07-11)** — `sspslam/worlds_dyn.py`:
classroom (spot-proxy) + school8 (hallway + 8 identical rooms, 9.9×
area, the aliasing stressor) with people as standing/walking circle
obstacles; noise draws people-independent (±people at fixed seed =
exactly paired); fenced diag_gt loop-precision labels. Key verdicts:
school aliasing admits wrong closures at prec 0.53–0.62 with zero people
(net-negative vs raw odometry in 7/10 arms — wrong > missed); standing
people de-alias monotonically (→0.85), walkers don't; classroom loop
on/off is a basin draw (read as bands); 10× map = 301 segments = 40 KB
@2b (24 KB oct36) vs the UP5K's 128 KB SPRAM. Run:
`python3 -m sspslam.worlds_dyn [check|quick|bench|tenx]`.

**Fidelity-space lattice scaling (2026-07-12)** — `scratch_biglat{,2,3}.py`
(gitignored scratch; harness imports the validated capacity2b stack):
frozen-pose store re-encode across 13 lattices (D 240→1980) on school8 +
stata fixtures. Verdicts: angles cap at 60 (×120/×180 hurt extraction on
every ladder); the LADDER is the lever — half-octave densification +
coarse {2.83, 4 m} extension lifts extraction AUCgt +.048 school8 /
+.096 stata and annihilates the 2 m ghost comb; fine floor = sensor
coherence length (λ_min ≈ 2πσ_r ≈ 12.6 cm at σ=2 cm; sub-coherence rings
= dead bytes, confirmed to 1.1 cm); capacity K-tail extends on real data
(2b K=128 p90 .742→.150). Recipe: fidelity space span11x60 (D=660,
418 B/seg) primary, hoct9x60 (D=540) lean; matcher space stays oct60
D=240. Run: `python3 scratch_biglat.py {gate|sweep} school`, then
`scratch_biglat2.py` (x60 + fine cells), `scratch_biglat3.py` (stata
phase 2). Gate reproduces banked capacity2b rows before new cells.

**Anneal-by-time-binding (2026-07-12)** — `scratch_anneal.py`: the
negate→rematch→reencode cycle under 5 memory assumptions. Verdict: works
on EXPLICIT frozen-code history in-basin (≲0.4 m; A2h = deployable form),
DEAD via time-bound superposition (20–80× knee; naive unbind-subtract =
√(K−1) noise injection); gated folding & 2b-requant-per-touch rejected.
Run: `python3 scratch_anneal.py`.

**BSC vs FHRR equal-bit shootout (2026-07-12)** — `scratch_bsc.py`:
dimension-compensated 480-bit majority bundles (multi-scale binarized-RFF
position codes, 2 cm atoms) vs the banked 2b-SUM phasor group vector.
Verdict: BSC knee K*≈8–16 vs phasor 32–48; at K=32 succ 0.07 vs 0.70;
K=1 control validates pipeline. Run: `python3 scratch_bsc.py` (+ ksweep).

**Golden-ratio ladders (2026-07-12)** — `scratch_biglat4.py`: phi8/phi13
(1 cm..3.6 m). Verdict: REFUTED on stata (−.06 both stores; 2nd φ synth
artifact); comb-kill replicates but mid-band density is the binding
constraint. Run: `python3 scratch_biglat4.py {school|stata}`.

**Replay-augmented frontend (2026-07-12)** — `scratch_replayfront.py`:
the webvis sample-replay mechanism as a pipeline subclass (reservoir,
refine-in-place, protected overwrite, do-no-harm, R0/R1/R2 arms + gates).
Verdict: see RESULTS (log-dependent; fhw 4× win, fr101/belg regress —
banked as opt-in, not default). Run: `python3 scratch_replayfront.py
[logs]`.

**Line-prior cleanup (2026-07-12)** — `scratch_lineprior.py`: matching
pursuit with exact line atoms; per-segment decode + cross-segment
consensus + micro-refine. Verdict: reads the store at 5 cm own-gauge
(GT-frame bound by store warp); global-bundle pursuit is the past-the-
knee control; display/compaction option. Run: `python3
scratch_lineprior.py school` (global arms) or `run_perseg()`.

**v7 standalone golden + v7.1 handoffs (2026-07-12)** —
`hw/ice40/host/solo.py`: integer SoloTracker/SoloMapper (parity 2 mm vs
live), selfmap/reservoir/blackout benches, K_TRY best-of-K rule.
Run: `python3 hw/ice40/host/solo.py {bench|selfmap|reservoir} [env] [traj]`.

**v7 top_solo RTL track (2026-07-12)** — `hw/ice40/rtl/encoder_solo.v` (v6
core + 2-line resident-map mc_seg addressing), `hw/ice40/rtl/solo_tracker.v`
(on-fabric tracker FSM: nearest/rel/parametric grid+retry/parabolic/EMA/
pose/re-search+relock, on-chip shift_for behind SH_ONCHIP),
`hw/ice40/rtl/tb_solo_bridge.v` + `tb_solo_step.v`, harnesses
`hw/ice40/host/solo_bridge.py`, `solo_step.py [n_kf] [suffix]`, fixtures via
`solo_vectors.py`. Gates: bridge 32/32 partials; step 220/220 kf
bit-exact (golden-sh AND on-chip-sh); relock fixture (320U kidnap kick,
2 relocks) gate = the stage-3b acceptance. Defect classes banked in
RESULTS (OR'd partials, $signed part-selects, RAM read off-by-one,
$readmemh truncation).

**Cleanup formulations program (2026-07-12)** — `scratch_cleanups.py`
(+ pursuit upgrades in `scratch_lineprior.py`): P1 CLEAN gain/OMP
(gain = recall/precision dial; amplitude pruning NO-OP — tail is
model-limited), P4 dead-zero store (3b = 2b phase + liveness bit BEATS
FLOAT both fixtures), P2 coherence-weighted Wiener read (GT-free C_r
observable; +.016/+.005), P4+P2 stack ~additive (school .6212, stata
.6751), P3 gauge relax NEUTRAL (adjacent segments share gauge; slow
warp = backend's job), P5 dictionary NEGATIVE (energy-greedy can't
model-select), P7 DRIVEN-PATH VETO = the tail-slayer (school p50
.226->.087, p90 halved, recall free; in decode.py + the webvis). Run: `python3 scratch_cleanups.py {p1|p4|p2}
{school|stata}`; stack/relax via `python3 -c "import scratch_cleanups
as CU; CU.p42_stack(...); CU.p3_gauge_relax(...)"`.
**v7 stream consumer** — `hw/ice40/host/decode.py [map.hex] [anchors.hex]
[--gamma G] [--out npz]`: the deploy read-side recipe on chip dumps
(coherence profile + Wiener imaging + pursuit/consensus lines).

**v7 build session (2026-07-12 evening)** — corner-tuned silicon:
`scratch_liveness.py {school|stata}` (freeze-store tuning: theta=1/2
liveness + per-ring Mmax scales BEAT the float store on extraction both
fixtures; theta=5/8 refuted by stata; scales-only ~nil);
`scratch_v7refine.py` (REFINE A/B: divider is p90-neutral for
localization, selfmap BETTER without it — lean build sets REFINE=0);
`scratch_p5pen.py {school|stata}` (P5 successor: complexity-penalized
atom selection — en/df activates model selection (points get chosen)
and improves +veto p50, at a recall cost; en/sup DEGENERATE all-points;
aicc too weak — see RESULTS for the verdict). RTL: `encoder_lean.v`
(serial-ring S/F core, 2034 LUT / 6 EBR / 2 MAC16, gates bit-exact),
`top_solo.v` + `tb_solo_top.v` + `hw/ice40/host/solo_top.py [n_kf]` (the
full standalone-SLAM top + UART-level end-to-end gate; phantom-keyframe
handshake race banked as a defect class), corner builds via `make
TOP=... RTL=... FREQ=...` (v6 deploy 5142 LC / 30 EBR / 30.2 MHz;
lean probe 2694 LC / 26.1 MHz; full solo 7452 LUT = the UP5K wall;
microcode closure filed). Lean fixtures: `python3 solo_vectors.py lean`
then `python3 solo_step.py 220 l1 lean` / `160 lr lean`.

**Truncated-Krylov GN backend (OPEN — unbanked)** — `experiments/krylov.py`:
portable exactly-specifiable replacement candidate for the load-bearing
scipy-TRF max_nfev=30 truncation (fixed-budget CG on the normal
equations, hard iteration cap). No banked verdict yet — the module is
committed for provenance; run + bank before any deploy claim.
