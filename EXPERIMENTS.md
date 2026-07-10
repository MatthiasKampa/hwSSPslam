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
| intel | `data/intel.log` | Intel Research Lab, the tuning log (full 6205 kf) |
| fr079 | `data/fr079.log` | Freiburg 079, zero-shot |
| fr101 | `data/fr101.log` | Freiburg 101, held-out dense-revisit building |
| aces | `data/aces_publicb.log` | ACES3 Austin, zero-shot, odo already good |
| belgioioso | `data/belgioioso.log` | non-Manhattan castle; range-identity eval |
| mit | `data/MIT_Infinite_Corridor_2002_09_11_same_floor.log` | 1.9 km held-out capstone |

**Eval recipe** (identical everywhere): parse `.gfs.log`, match each reference
pose to the nearest keyframe timestamp within 0.3 s, `C.align_se2` the matched xy,
report RMSE / median / max. Belgioioso/MIT have timestamp quirks — check the
relevant `RESULTS.md` section before trusting an ATE there.

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
without editing the module) — see `scratch_lattice_sweep.py`:
```python
import ssp_slam_loop as L
def set_lattice(lams_matched, n_ang):          # 4 matched rings + 2 relo fixed
    L.LAMS = np.array(list(lams_matched) + [5.3, 12.8]); L.N_ANG = n_ang
    L.N_RING = len(L.LAMS); L.MAIN = slice(0, 4 * n_ang)
    L.W = L.build_W(); L.ENC.W = L.W; L.ENC_MAIN.W = L.W[L.MAIN]
    L._RINGS = np.repeat(np.arange(L.N_RING), n_ang)
    L.WIDE = L._RINGS >= 2; L.WIDER = L._RINGS >= 3
# then replicate ssp_bounded_carmen.main's loop + eval verbatim
```
Validate any such harness reproduces a known shipped number before trusting a
sweep (fr101 → 1.881 m at the baseline lattice).

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
python3 demo/export_replay.py data/intel.log --embed   # Python-fed webvis
```
Verdicts in RESULTS.md "2026-07-10 — FPGA track opened" (+ the consolidation
section that follows it): quantized store needs PER-RING scales; ≥6-bit quant
deltas on Intel are not attributable (perturbation band); E2 = frontend
registration win 5/5 logs, closure-layer per-log; arithmetic knee = 8-bit ROM
cis; 2-bit phase-only store viable on robust logs (fr101 band [1.1–2.7] at
75 KB); QPSK arithmetic = median-only.
