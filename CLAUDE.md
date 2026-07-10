# CLAUDE.md — working guide for this repository

Bounded-memory **VSA/SSP 2D-lidar SLAM**. The map is a fixed-size complex
Fourier-feature vector (a Spatial Semantic Pointer on a polar frequency lattice)
per rigid 5-keyframe trajectory segment — **no sensor history is stored**.
Translation of map content = elementwise phase multiply; rotation = exact index
permutation of the lattice (+ a stored d/dθ derivative vector for sub-grid
angles); matching = inner product; bundling/composition = addition; graph
corrections = O(D) phase ops on frozen content (never re-encoded).

The contribution is a **representation** (bounded O(area), history-free,
algebraically transformable, at usable accuracy), **not** a state-of-the-art
accuracy or bytes/m² record. See `FINDINGS.md` for the distilled thesis and
`RESULTS.md` for the chronological lab notebook.

## The one finding that shapes everything

Loop-closure **detection/verification is the irreducible wall**: on aliased
environments (corridors, cross-session revisits) a genuine match is
cosine-indistinguishable from an aliased twin, at *every* level tested (per-frame,
place-descriptor, geometric-consensus/PCM, temporal-sequence/SeqSLAM,
gradient-flow force, hybrid anchor+overlap, viewpoint appearance). The **backend
is sound given closures; the frontend/verifier cannot manufacture them** from
2D-lidar appearance alone. Crossing the wall needs an *independent absolute cue*
(IMU/wheel-slip residual, surveyed landmarks, GPS, 3D/wider aperture). Do not
re-litigate this without new information — see `FINDINGS.md` §5–§6.

## Rules (non-negotiable — full detail in `PROTOCOL.md`)

1. **Never edit a shipped/prior module in an experiment.** Subclass or import
   only. Shipped core: `ssp_bounded.py`, `ssp_bounded_carmen.py`,
   `ssp_slam.py`, `ssp_slam_loop.py`, `ssp_slam_carmen.py`.
2. **Anti-oracle.** Ground truth (`*.gfs.log`) is for **scoring only** — never to
   seed, select, gate, or discriminate. If GT touches a pipeline for a diagnostic
   upper bound, fence it explicitly in code + log and label the result a
   diagnostic, not a deployable number.
3. **Numbers only.** No GUI, no screenshots, no `open`, no windows. If you must
   use matplotlib, `matplotlib.use("Agg")` and write a file — never display.
4. **Audit positive results before trusting them.** Read-only re-derivation has
   caught false-wins/leaks/bugs repeatedly this project. Agents may be used
   **only** for such second opinions — never to write code or run experiments.
5. **Determinism + multi-log gates.** Fixed seeds; no wall-clock/random in logic.
   Accept a change only if it holds across logs (single-log tuning hid real
   regressions here). Do not tune the encoder per dataset unless explicitly
   running a per-dataset study — the held-out numbers depend on one fixed config.
6. **Scratch hygiene.** Throwaway files are `scratch_*.{py,log,out}` (gitignored).
   Commit only when asked; end messages with the Co-Authored-By + session trailer.

## Run the shipped deliverable

```bash
python3 ssp_bounded_carmen.py data/fr101.log        # build + ATE vs .gfs.log
python3 ssp_datasets.py run stata                   # flagship (floorplan GT)
```
**Acceptance suite (2026-07-10 redirect; see PROTOCOL):** stata 0.202
(floorplan GT — the target proxy; platform = SPOT, 360° × 1024-beam head @
20 Hz), fr101 1.88, fhw 0.98, fr079 5.52 (read as a band), belg 2.64, plus
the synthetic 360° bench (`ssp_synth.py`). **intel was REMOVED from the
suite** (knife-edge band, GMapping-referenced GT, 180° FOV) — do not tune
or accept against it; its historical numbers stay in the ledger. Baselines
(same harness): ICP 1.70 / CSM 3.27 / RBPF 0.12* (intel-era). Encoder
lattice lives as module globals in `ssp_slam_loop.py` (`LAMS`, `N_ANG`,
`W`, `ENC`, `ENC_MAIN`); the frontend matcher uses the 4 matched rings × 60
angles (D=240). Patch lattices only via `ssp_lattice.py`; run datasets only
via `ssp_datasets.py`.

## Map of the repo

- **Shipped**: `ssp_bounded*.py`, `ssp_slam*.py` (see above).
- **Baselines/controls**: `baseline_{icp,csm,rbpf}.py`, `ssp_scancontext.py`.
- **Experiments** (subclass/import only): everything else `ssp_*.py`. Catalogue
  with verdicts + how-to-run in `EXPERIMENTS.md`.
- **Docs**: `FINDINGS.md` (synthesis), `RESULTS.md` (chronological ledger, cites
  section names), `PROTOCOL.md` (experimental protocol), `README.md` (front door).
- **Data**: `data/*.log` (+ `*.gfs.log` GMapping-corrected references); gitignored,
  fetched separately. Eval = range-nearest match within 0.3 s → `align_se2` → RMSE.

When you finish a thread, update `RESULTS.md` (what ran, the numbers, the verdict)
and, if it changes the synthesis, `FINDINGS.md`. Keep both honest about negatives.
