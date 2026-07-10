# EXPERIMENTS — operational guide + module catalogue

How to run things, the standard harness patterns, and an index of every
experiment with its one-line verdict and where to read the detail. Findings live
in `FINDINGS.md` (synthesis) and `RESULTS.md` (chronological, cites section
names). Discipline lives in `PROTOCOL.md`.

---

## Datasets

CARMEN `.log` (odometry + laser); `.gfs.log` is the GMapping/RBPF-corrected
reference used **only** for scoring. Under `data/` (gitignored, fetched).

| key | file | notes |
|---|---|---|
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
historical `RESULTS.md` numbers stay reproducible. The synthetic bench
(`ssp_sampling.py synth`, 360° FOV, exact GT) joins the acceptance suite as
the target-matched environment.

**One registry for all of it: `ssp_datasets.py`** — `load(name)`,
`evaluate(bundle, fin)`, `run(name, cls, sample=..., **kw)`. The run loop is
a bit-exact transcription of `ssp_fpga.run_log` (asserted in
`python3 ssp_datasets.py selftest`); eval dispatches per dataset: `gfs`
nearest-timestamp <0.3 s, `ident` range-array identity (belg/mit — their gfs
timestamps are corrupt), `stata` floorplan GT <30 ms. Quick single runs:
`python3 ssp_datasets.py run belg [shipped|e2|lean]`.

## Run recipes

```bash
python3 ssp_bounded_carmen.py data/intel.log          # shipped deliverable + ATE
python3 ssp_bounded_carmen.py data/fr101.log 1893      # optional keyframe cap
python3 baseline_icp.py data/intel.log                 # ICP+posegraph baseline
python3 baseline_csm.py  data/intel.log                # correlative-grid baseline
python3 baseline_rbpf.py data/intel.log                # RBPF/GMapping-lite baseline
python3 bench_loop.py                                   # synthetic GT-labelled loop bench
```

## Harness patterns

**Subclass the backend** (new solver/admission behaviour), prove bit-exact when
neutralised:
```python
import ssp_bounded as B
class MySLAM(B.BoundedSLAM):
    def _gn(self, ...):
        if not self.use_feature:      # neutralised path MUST be bit-exact to parent
            return super()._gn(...)
        ...
```

**Run the shipped loop with a parameterised encoder** (sweep a lattice constant
without editing the module) — `ssp_lattice.py` is the one committed home:
```python
import ssp_lattice, ssp_datasets as DS
ssp_lattice.set_polar((0.125, 0.25, 0.5, 1.0), n_ang=60)   # any ladder/count
ssp_lattice.set_hex(63)                                     # full-circle hex
ssp_lattice.shipped()                                       # exact restore
r = DS.run("fr101", spec=None, nph=0)                       # then run
```
(`python3 ssp_lattice.py` asserts set_polar()==shipped globals exactly and
hex rotation==permutation.) Validate any such harness reproduces a known
shipped number before trusting a sweep (fr101 → 1.881 m at the baseline
lattice).

**Reuse the flow field** (`ssp_flow.FlowField`): pose-pose stiff edges +
analytic SSP-correlation overlap force, grad-L2 normalised. `ssp_hybrid.py` shows
building it from a shipped `BoundedSLAM` and from a two-session split.

## Module catalogue (verdicts + pointers)

**Shipped core** — `ssp_bounded.py` (BoundedSLAM: rigid 5-kf segment vectors +
d/dθ, continual constraints, IRLS + LOO pruning, PCM admission),
`ssp_bounded_carmen.py` (driver+eval), `ssp_slam_loop.py` (polar lattice globals +
rot-permute + relo), `ssp_slam.py` (encoder/matcher), `ssp_slam_carmen.py`
(frontend). `worlds.py`/`experiments.py`/`ssp_angles.py` = encoder-design studies
(oct grid beats RFF ~10–20×; derivative vector load-bearing).

**Baselines / classical controls** — `baseline_icp.py` 1.70 m, `baseline_csm.py`
3.27, `baseline_rbpf.py` 0.12* (*ref is itself GMapping — circular, not a win);
`ssp_scancontext.py` (Scan-Context control: the revisit signal IS in the raw scan,
so the SSP corridor limit is representational not environmental).

**Corridor / verification wall** (all CLOSED — `FINDINGS.md` §5) —
`ssp_ringkey.py` retrieval solved (recall 0.32→0.81) but consensus harder;
`ssp_seqslam.py` sequence ambiguous (0.500 = chance); `ssp_cascade.py`
coarse→fine profile veto (precision not recall); `ssp_mit_gtverify.py` perfect
place oracle only 42.66→41.04 (place necessary-not-sufficient; wall is two-fold
place AND pose); `ssp_aniso.py`/`ssp_aniso_mit.py` observability-weighted
anisotropy — real but fragile/regime-specific, non-shippable (MIT oracle 41.04→
39.12 diagnostic only).

**Multi-session / composition** (`FINDINGS.md` §5.6 + multi-session sections) —
`ssp_multisession.py` VSA composition works (bundling=addition, 37% overlap
saving); `ssp_multisession_align.py` naive win was a shared-index leak (audit);
`ssp_multisession_ringkey.py` appearance retrieval works, verification is the wall;
`ssp_multisession_verify.py` oracle-free verifier — PCM clique≥2 best (5.54→3.35 m)
but doesn't reach 2.49/2.44, wall narrows not closes; `ssp_percluster.py`,
`ssp_adaptmap.py`, `ssp_iteralign.py` = map-handling probes (per-cluster split,
adaptive-map memory win, iterative-align negative).

**Memory / hierarchy** — `ssp_hier.py` HY4 adopted (0.67× memory, bit-identical);
`ssp_hiergraph.py` coarse-to-fine relax = no gain (flat relax already optimal);
`ssp_scale_arrays.py` **spatial O(area) submaps FAIL for closure** — persistent
per-area cells smear write-time content across drift states, biasing every loop Z
(4.34→11–15 m); law: a closure primitive must hold single-burst drift-consistent
content (why shipped uses ephemeral *temporal* segments).

**Frontend do-no-harm** (`FINDINGS.md` §6, CLOSED triple-negative) —
`ssp_frontguard.py`, `ssp_frontsys.py`, `ssp_belief.py`, `ssp_posefilter.py`:
"odometry is already good" is not scan-match-observable; needs an independent
odo-quality signal this suite lacks.

**This session's paradigms** (`FINDINGS.md` §5.6) — `ssp_flow.py` continual
detection-free gradient-flow (valid/non-folding, but drifts from GT-perfect init;
refines within-basin only; grad-L2 + soft/hex = conditioning wins);
`ssp_hybrid.py` detected anchors + overlap-flow (single-session net-negative;
multi-session tightens true overlaps ~8–16% but worsens ATE via twin
contamination — does not beat anchors-alone; audited SOUND);
`ssp_viewpoint.py` viewpoint dual-channel (compose/dedup WIN corr +0.93; NOT a
discriminator — Intel AUC 0.896 < 0.909 proximity confound, MIT below chance).

**Bench** — `bench_loop.py` synthetic multiloop world with GT edge labels
(precision/recall on closures without a real dataset).

**FPGA track (2026-07-10)** — `ssp_fpga.py` (subclass-only; neutralised paths
asserted bit-exact): `QuantStoreSLAM` write-time per-ring polar-quantized
store; `NoiseStoreSLAM` freeze-noise chaos control; `DerFineMatcher` E1
(negative); `points_from_scan` E2 per-beam point encoding (frontend win on
every log, audited); `IntSpec`/`IntSLAM` integer front-path arithmetic model;
`BandSLAM` + `band_table` the perturbation-band acceptance harness (PROTOCOL
§6). Run recipes:
```bash
python3 ssp_fpga.py selftest          # neutralised == parent, bit-exact
python3 ssp_fpga.py ops               # hot-path MAC / map-bit sizing
python3 ssp_fpga.py sweep --ring intel  # write-time quant ladder
python3 ssp_fpga.py front fr101 intel   # E1/E2 frontend variants
python3 ssp_fpga.py int fr101           # integer arithmetic ladder
python3 ssp_fpga.py chaos intel         # freeze-noise response curve
python3 ssp_fpga.py band fr079          # config x perturbation band table
python3 demo/export_replay.py stata --config=shipped   # Python-fed webvis
python3 demo/export_replay.py --embed   # splice manifest into index.html
```
Verdicts in RESULTS.md "2026-07-10 — FPGA track opened" (+ the consolidation
section that follows it): quantized store needs PER-RING scales; ≥6-bit quant
deltas on Intel are not attributable (perturbation band); E2 = frontend
registration win 5/5 logs, closure-layer per-log; arithmetic knee = 8-bit ROM
cis; 2-bit phase-only store viable on robust logs (fr101 band [1.1–2.7] at
75 KB); QPSK arithmetic = median-only.

**Stability / decision-flip thread (2026-07-10)** — `ssp_stablegate.py`
(JitterMixin loop-layer dither-consensus: bands narrow but medians drift;
ConsensusMatcher FRONTEND jitter-consensus: collapses the intel band 14×
onto the median — the mitigation candidate); `ssp_bfc.py` (bundle-frame
smooth closure factors: clean negative — band is a recurrent-estimator
property, not a factor-shape artifact). Both carry the verbatim copied
`try_constraint` body pattern (neutralised → bit-exact, asserted).

**Datasets / lattice infra (2026-07-10)** — `ssp_datasets.py` (registry +
the one shared run/eval harness — see Datasets above); `ssp_stata.py` (bag
adapter, floorplan GT; shipped 0.202 m); `ssp_hexreal.py` (full-circle hex
lattice e2e on real logs: belg 2.071 vs polar 2.644, intel control loses —
per-environment option); `ssp_lattice.py` (set_polar/set_hex/shipped —
consolidated lattice patching, replaces per-scratch copies);
`ssp_synth.py` (SPOT-proxy synthetic 360° bench over worlds.py — exact GT,
emits ssp_datasets bundles, deterministic local rng; `bench`/`fov` CLIs).

**Encoding/sampling family (2026-07-10, user thread)** — `ssp_sampling.py`:
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
documented). Run: `python3 ssp_sampling.py [selftest|interp|segint|renorm
<logs>|synth]`.
