# RESTRUCTURE-MAP — 2026-07-13 repo reorganisation

One-commit mechanical move from the historical flat layout into role-named
packages. **No algorithmic change**: imports were rewritten alias-preserving
(`import sspslam.lattice as ssp_slam_loop`-style, so call sites are untouched
and module-global patching still hits the single module instance), and the
full acceptance suite reproduced the pre-move outputs bit-identically
(only wall-clock `ms/kf` fields differ). `docs/RESULTS.md` (the append-only
ledger) and everything in `scratch/` and `archive/` deliberately keep the
historical names — use this table to resolve them.

## Modules

| old (flat) | new |
|---|---|
| `ssp_slam.py` | `sspslam/encoder.py` |
| `ssp_slam_loop.py` | `sspslam/lattice.py` |
| `ssp_slam_carmen.py` | `sspslam/frontend.py` |
| `ssp_bounded.py` | `sspslam/bounded.py` |
| `ssp_lattice.py` | `sspslam/lattice_presets.py` |
| `ssp_fpga.py` | `sspslam/quantized.py` |
| `worlds.py` | `sspslam/worlds.py` |
| `ssp_dynenv.py` | `sspslam/worlds_dyn.py` |
| `bench_loop.py` | `sspslam/bench.py` |
| `ssp_bounded_carmen.py` | `runners/carmen.py` |
| `ssp_datasets.py` | `runners/datasets.py` |
| `ssp_spot.py` | `runners/spot.py` |
| `ssp_stata.py` | `runners/stata.py` |
| `ssp_synth.py` | `runners/synth.py` |
| `rpe.py` | `runners/rpe.py` |
| `baseline_icp.py` | `baselines/icp.py` |
| `baseline_csm.py` | `baselines/csm.py` |
| `baseline_rbpf.py` | `baselines/rbpf.py` |
| `ssp_scancontext.py` | `baselines/scancontext.py` |
| `ssp_ice40.py` | `hw/ice40/golden.py` |
| `ssp_<experiment>.py` | `experiments/<experiment>.py` (prefix stripped; all 29) |
| `experiments.py` (encoder study) | `archive/experiments.py` |

## Directories / docs

| old | new |
|---|---|
| `FINDINGS.md`, `RESULTS.md`, `PROTOCOL.md` | `docs/` (same names) |
| `EXPERIMENTS.md` | `experiments/README.md` |
| `SotA/` | `docs/sota/` |
| `ice40/` | `hw/ice40/` |
| root `scratch_*.{py,log,out,md,sh}` | `scratch/` (unmodified — old imports inside them are historical) |

## Command equivalents

| old | new |
|---|---|
| `python3 ssp_bounded_carmen.py data/fr101.log` | `python3 -m runners.carmen data/fr101.log` |
| `python3 ssp_datasets.py run stata` | `python3 -m runners.datasets run stata` |
| `python3 ssp_fpga.py selftest` | `python3 -m sspslam.quantized selftest` |
| `python3 ssp_synth.py bench` | `python3 -m runners.synth bench` |
| `python3 ssp_spot.py sweep` | `python3 -m runners.spot sweep` |
| `python3 ssp_sampling.py selftest` | `python3 -m experiments.sampling selftest` |

Everything runs from the repo root; `python3 -m` puts the root on `sys.path`
(no install needed — `pip install -e .` additionally makes `sspslam`
importable from anywhere).

Alias-preserving import style inside the codebase: old call sites like
`ssp_slam_loop.LAMS` still work because moved modules are imported as
`import sspslam.lattice as ssp_slam_loop`. New code may import under any
name; there is exactly one module instance either way.
