#!/usr/bin/env python3
"""Fast repo health check: every module imports through the package layout,
and the quantized-store selftest is bit-exact to the parent (needs
data/fr101.log; skipped with a notice if datasets are not fetched).
Plain python (no pytest dependency): python3 tests/test_smoke.py
Run from the repo root.
"""
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODULES = (
    [f"sspslam.{m}" for m in "encoder lattice frontend bounded lattice_presets"
     " quantized worlds bench worlds_dyn".split()]
    + [f"runners.{m}" for m in "carmen datasets spot stata synth rpe".split()]
    + [f"baselines.{m}" for m in "icp csm rbpf scancontext".split()]
    + [f"experiments.{m}" for m in
       "adaptmap angles aniso aniso_mit belief bfc cascade flow frontguard"
       " frontsys hexreal hier hiergraph hybrid iteralign krylov mit_gtverify"
       " multisession multisession_align multisession_ringkey"
       " multisession_verify percluster posefilter ringkey sampling"
       " scale_arrays seqslam stablegate viewpoint".split()]
    + ["hw.ice40.golden"]
)


def test_imports():
    sys.path.insert(0, str(ROOT))
    for m in MODULES:
        importlib.import_module(m)
    print(f"ok: {len(MODULES)} modules import")


def test_quantized_selftest():
    if not (ROOT / "data" / "fr101.log").exists():
        print("SKIP: quantized selftest (data/fr101.log not fetched)")
        return
    r = subprocess.run([sys.executable, "-m", "sspslam.quantized", "selftest"],
                       cwd=ROOT, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "selftest ok: neutralised == parent bit-exact" in r.stdout, r.stdout
    print("ok: quantized selftest bit-exact")


if __name__ == "__main__":
    test_imports()
    test_quantized_selftest()
    print("SMOKE PASS")
