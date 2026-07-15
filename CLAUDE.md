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
accuracy or bytes/m² record. See `docs/FINDINGS.md` for the distilled thesis
and `docs/RESULTS.md` for the chronological lab notebook.

## The one finding that shapes everything

Loop-closure **detection/verification is the irreducible wall**: on aliased
environments (corridors, cross-session revisits) a genuine match is
cosine-indistinguishable from an aliased twin, at *every* level tested (per-frame,
place-descriptor, geometric-consensus/PCM, temporal-sequence/SeqSLAM,
gradient-flow force, hybrid anchor+overlap, viewpoint appearance). The **backend
is sound given closures; the frontend/verifier cannot manufacture them** from
2D-lidar appearance alone. Crossing the wall needs an *independent absolute cue*
(IMU/wheel-slip residual, surveyed landmarks, GPS, 3D/wider aperture). Do not
re-litigate this without new information — see `docs/FINDINGS.md` §5–§6.

## Layout (2026-07-13 restructure; old→new table in `docs/RESTRUCTURE-MAP.md`)

- `sspslam/` — **shipped library, frozen** (rule 1): `encoder.py` (scan→SSP +
  matcher), `lattice.py` (polar lattice globals `LAMS`/`N_ANG`/`W`/`ENC`/
  `ENC_MAIN`, rotation permutation, relocalization), `frontend.py` (CARMEN
  frontend), `bounded.py` (BoundedSLAM deliverable). Load-bearing infra under
  the same subclass/import-only discipline: `quantized.py` (FPGA arithmetic
  models + BandSLAM), `lattice_presets.py` (the ONE lattice-patching home),
  `worlds.py`/`worlds_dyn.py`/`bench.py`.
- `runners/` — CLI entry points, run from the repo root:
  `python3 -m runners.datasets run stata`, `python3 -m runners.carmen
  data/fr101.log`, plus `spot.py`, `stata.py`, `synth.py`, `rpe.py`.
- `baselines/` — `icp.py`, `csm.py`, `rbpf.py`, `scancontext.py` (controls).
- `experiments/` — one module per catalogued experiment, frozen post-verdict;
  **`experiments/README.md` is the catalogue with verdicts + run recipes**.
  Run as `python3 -m experiments.<name>`.
- `sspax/` — **JAX efficient experimentation surface** (2026-07-15; imports the
  frozen core only, rule 1). Bit-faithful JAX port of the encode/rotate/decode
  algebra (`core.py`, verified `python3 -m sspax.bench parity`); the
  approximate-permutable ring-stagger sphere (`sphere.py`); the 6-DoF formulation
  sweep (`sweep*.py` → `SWEEP_RESULTS.md`); learned FPGA front-ends — lidar
  saliency (`learn_lidar.py`) + vision CNN (`vision/`) with cuDNN-free convs
  (`nnconv.py`); and the binary-descriptor **semantic/queryable map**
  (`semantic.py`, "highlight the chairs"). Overview in `sspax/README.md`,
  synthesis in `sspax/LEARNED_FRONTEND.md`.
- `hw/ice40/` — FPGA track (`rtl/`, `host/` incl. live demo server,
  `golden.py`). `docs/` — FINDINGS, RESULTS (ledger), PROTOCOL,
  RESTRUCTURE-MAP, `sota/` notes. `demo/` — browser replay demo.
  `data/` — datasets (gitignored, fetched). `archive/` — superseded code.
- `scratch/` — ALL transient session files, gitignored wholesale.
- Everything executes **from the repo root** via `python3 -m` (no install
  required; `pyproject.toml` enables optional `pip install -e .`).

## Rules (non-negotiable — full detail in `docs/PROTOCOL.md`)

1. **Never edit a shipped/prior module in an experiment.** Subclass or import
   only. Shipped core: `sspslam/bounded.py`, `sspslam/encoder.py`,
   `sspslam/lattice.py`, `sspslam/frontend.py`, `runners/carmen.py`.
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
6. **Scratch hygiene.** Throwaway files are `scratch/scratch_<topic>.{py,log,out}`
   (the `scratch/` dir is gitignored wholesale); run them with
   `PYTHONPATH=. python3 scratch/scratch_<topic>.py`. Commit only when asked;
   end messages with the Co-Authored-By + session trailer.

## Run the shipped deliverable

```bash
python3 -m runners.carmen data/fr101.log        # build + ATE vs .gfs.log
python3 -m runners.datasets run stata           # flagship (floorplan GT)
python3 tests/test_smoke.py                     # fast: imports + bit-exact selftest
```
**Acceptance suite (2026-07-10 redirect; see PROTOCOL):** spot 0.039
(TARGET PLATFORM, lidar-only vs withheld odometry — `runners/spot.py`), stata
0.202 (floorplan GT — the public proxy; platform = SPOT, 360° ×
1024-beam head @ 20 Hz), fr101 1.88, fhw 0.98, fr079 5.52 (read as a
band), belg 2.64, plus the synthetic 360° bench (`python3 -m runners.synth
bench`). **intel was REMOVED from the suite** (knife-edge band,
GMapping-referenced GT, 180° FOV) — do not tune or accept against it; its
historical numbers stay in the ledger. Baselines (same harness): ICP 1.70 /
CSM 3.27 / RBPF 0.12* (intel-era). Encoder lattice lives as module globals
in `sspslam/lattice.py` (`LAMS`, `N_ANG`, `W`, `ENC`, `ENC_MAIN`); the
frontend matcher uses the 4 matched rings × 60 angles (D=240). Patch
lattices only via `sspslam/lattice_presets.py`; run datasets only via
`runners/datasets.py`. (2026-07-13 restructure verification: whole suite
reproduced bit-identically across the move; `tests/test_acceptance.py`
re-checks it.)

## Docs

`README.md` (front door) → `docs/FINDINGS.md` (synthesis) →
`experiments/README.md` (catalogue: what was tried, verdicts, recipes) →
`docs/RESULTS.md` (append-only ledger; cites section names and historical
flat filenames — resolve via `docs/RESTRUCTURE-MAP.md`) → `docs/PROTOCOL.md`
(the discipline). When you finish a thread, update `docs/RESULTS.md` (what
ran, the numbers, the verdict) and, if it changes the synthesis,
`docs/FINDINGS.md`. Keep both honest about negatives.
