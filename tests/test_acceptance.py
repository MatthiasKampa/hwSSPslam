#!/usr/bin/env python3
"""Acceptance suite regression check (deterministic — exact-line match).

Expected lines were captured 2026-07-13 on the restructure commit, where the
whole suite reproduced the pre-restructure flat-layout outputs bit-identically
(docs/RESTRUCTURE-MAP.md). The wall-clock `ms/kf` field is ignored.

    python3 tests/test_acceptance.py            # fr101 only (fast)
    python3 tests/test_acceptance.py all        # the full six-dataset suite

Missing datasets are reported as SKIP (fetch via data/fetch_datasets.sh;
spot additionally needs the parsed cache: python3 -m runners.spot parse).
Plain python; run from the repo root.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "fr101": ("data/fr101.log",
              "fr101: ATE 1.881  med 1.551  loops 53  mem 1896 KB  (290 ref poses"),
    "fhw":   ("data/fhw.log",
              "fhw: ATE 0.981  med 0.841  loops 559  mem 5203 KB  (483 ref poses"),
    "fr079": ("data/fr079.log",
              "fr079: ATE 5.523  med 3.533  loops 32  mem 2610 KB  (4286 ref poses"),
    "belg":  ("data/belgioioso.log",
              "belg: ATE 2.644  med 1.859  loops 1  mem 1642 KB  (1698 ref poses"),
    "stata": ("data/stata/2012-01-27-07-37-01.bag",
              "stata: ATE 0.202  med 0.123  loops 99  mem 1238 KB  (819 ref poses"),
    "spot":  ("data/spot_telluride/scans.npz",
              "spot: ATE 0.034  med 0.028  loops 22  mem 366 KB  (414 ref poses"),
}
NOTE = ("fr079 and belg sit in perturbation bands (PROTOCOL band-probe rule); "
        "an exact-match failure there after an intentional change means "
        "re-capture, not necessarily regression.")


def run_one(name):
    path, want = EXPECTED[name]
    if not (ROOT / path).exists():
        print(f"SKIP {name}: {path} not present")
        return True
    r = subprocess.run([sys.executable, "-m", "runners.datasets", "run", name],
                       cwd=ROOT, capture_output=True, text=True, timeout=7200)
    got = next((ln for ln in r.stdout.splitlines()
                if ln.startswith(f"{name}: ATE")), "")
    got_cmp = re.sub(r",\s*\d+ ms/kf\)$", "", got)
    if r.returncode == 0 and got_cmp.strip() == want.strip():
        print(f"PASS {name}: {got}")
        return True
    print(f"FAIL {name}:\n  want {want}...\n  got  {got or r.stdout + r.stderr}")
    return False


if __name__ == "__main__":
    names = list(EXPECTED) if "all" in sys.argv[1:] else ["fr101"]
    ok = all([run_one(n) for n in names])
    print(NOTE)
    print("ACCEPTANCE PASS" if ok else "ACCEPTANCE FAIL")
    sys.exit(0 if ok else 1)
