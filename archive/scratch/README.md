# archive/scratch — retired throwaway scripts

Transient `scratch_*` experiment scripts + their `.out`/`.log` captures,
moved out of the repo root by the 2026-07-10 cleanup. Numbers they produced
are banked in `RESULTS.md` (sections cite the file names). Most are
untracked (per PROTOCOL 6); the two tracked ones (`scratch_belgioioso.py`,
`scratch_aces_diag.py`) predate the rule.

To rerun one, use the repo root as the working directory:

```bash
PYTHONPATH=. python3 archive/scratch/scratch_<name>.py
```

Cross-scratch imports (`scratch_binscale` → `scratch_lattice_sweep`) resolve
within this directory. The reusable pieces were promoted to committed
modules: lattice patching → `ssp_lattice.py`, dataset loading/eval →
`ssp_datasets.py` (range-identity eval from `scratch_belgioioso.py`).
