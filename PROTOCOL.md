# PROTOCOL — experimental discipline for this repository

This project's credibility rests on a small set of rules that were *earned* — each
one traces to a specific false-win, leak, or regression that it prevents.
`CLAUDE.md` has the short list; this is the full protocol with the reasoning.

---

## 1. Never edit a shipped/prior module in an experiment

Experiments **subclass or import**; they do not modify the deliverable or any
earlier experiment's file. Shipped core is `ssp_bounded.py`,
`ssp_bounded_carmen.py`, `ssp_slam.py`, `ssp_slam_loop.py`, `ssp_slam_carmen.py`.

- New backend behaviour → subclass `BoundedSLAM` and override the method
  (e.g. `ssp_aniso.py: AnisoBoundedSLAM(AnisoMixin, B.BoundedSLAM)`).
- Reusing machinery → `import` it (e.g. `ssp_hybrid.py` imports
  `ssp_flow.FlowField`; `ssp_multisession_verify.py` imports the PCM engine).
- Sweeping a shipped **constant** (e.g. the lattice) → patch the module global at
  runtime in a `scratch_*` harness, rebuild everything derived from it, and never
  touch the source. Pattern: `scratch_lattice_sweep.py` (`set_lattice`).

**Why.** A subclass that neutralises to the parent must be **bit-exact** to it
(prove it: `use_aniso=False` → `max|dpose| = 0`). This guarantees any measured
change is the *idea*, not an accidental solver regression, and it keeps the
shipped numbers reproducible forever.

## 2. Anti-oracle: ground truth is for scoring only

`*.gfs.log` (the GMapping-corrected reference) may **only** score a result. It
must never seed a matcher, select/gate candidates, set a threshold, or
discriminate genuine from twin.

- If GT is used to establish a **diagnostic upper bound** (e.g. "what would a
  perfect place oracle buy?"), fence it explicitly: GT confined to
  candidate-selection + scoring, the actual optimiser/force GT-free, stated in
  code comments *and* the log, and the number labelled a **diagnostic, not
  deployable**. Examples done right: `ssp_mit_gtverify.py`, `ssp_hybrid.run_multi`
  (GT only picks co-visible pairs + the closure Z + scores; the flow force is
  GT-free — provable at the zero-anchor row).

**Why.** The project's sharpest near-miss: an early cross-session "win" was a
false result — the A/B split shared keyframe indices, and the bootstrap seeded
the matcher *at the answer* via that shared index. Stripping the oracle collapsed
it. Any GT leak into selection manufactures the result you were trying to measure.

## 3. Numbers only — no GUI, no screenshots

No `open`, no windows, no displayed plots, no screenshots. If a figure is
genuinely needed, `matplotlib.use("Agg")` and write a PNG; never display. Prefer
printing numbers. (The shipped `ssp_bounded_carmen.py` writes one Agg PNG by
design; experiments should print tables.)

**Why.** Screenshots and interactive windows are slow, non-reproducible, and
were explicitly ruled out. A number in a log is auditable; a picture is not.

## 4. Audit positive results before trusting them

A positive/surprising result is not trusted until an **independent read-only
re-derivation** confirms it — trace every input, especially any GT contact and
every metric definition. Agents may be used for this (and *only* this — see §8).

**Why (this is the highest-value rule).** Read-only audits have repeatedly
inverted conclusions here: a ring-key run whose deep-consensus escalation was
silently disabled; a SeqSLAM run that was a lifecycle no-op (descriptor stored
*after* the drought attempt, so the seq path early-returned every time —
a degenerate 356/0/0/0 that would have read as "sequence cleanly avoids twins");
the shared-index leak in §2. Each would have shipped a false finding. Negatives
are lower-risk (audit lightly); positives get the full treatment.

## 5. Determinism

No `Date.now()`/`random` in logic; fixed seeds. Same inputs → same numbers, so
an auditor reproduces bit-exact. Report reproduction explicitly ("reproduces
2.440 bit-exact").

## 6. Multi-log acceptance gates; held-out discipline

- Accept a change only if it holds **across logs**. Single-log (Intel-only)
  acceptance hid a real fr079 regression (analytic-GN over-convergence on floppy
  graphs) that only surfaced on the transfer suite.
- **Band-probe rule (2026-07-10).** A single-trajectory ATE on a
  perturbation-sensitive log is a *basin draw*, not a measurement: the closure
  cascade flips discrete decisions under freeze-time map perturbations as small
  as 1e-4 relative (Intel bifurcates 2.44 → a non-monotone 3.5–5.5 m band;
  fr079 5.5 → 13.5 at 1e-3; a combo config scored 2.15 on ACES at ε=0 and 8.65
  at ε=1e-6). Therefore any config-level claim must report its **perturbation
  band** — `ssp_fpga.BandSLAM` over ε ∈ {0, 1e-6, 1e-3 × 2 seeds} — and the
  accepted quantity is the band (min/median/max), not the point. Exceptions:
  logs demonstrated robust at 1e-2 (fr101) may cite points, with the band run
  once as evidence of robustness.
- **Do not tune the encoder/params per dataset** unless explicitly running a
  per-dataset study. The held-out claims (fr101, belgioioso, MIT: *zero* code or
  parameter changes) are load-bearing and depend on one fixed config. A
  per-dataset-tuned number is a *separate*, clearly-labelled quantity — never
  merged into the zero-shot headline.

## 7. Report honestly

State negatives plainly; keep superseded numbers in `RESULTS.md` as stratigraphy
(retractions visible, not erased). Distinguish "diagnostic upper bound" from
"deployable". Name the metric precisely (a metric that rewards the optimiser's own
objective — e.g. co-observed residual under an attractive force — is weaker
evidence than an independent aggregate like ATE; say so).

## 8. Agents, scratch, and commits

- **Agents**: second opinions / audits **only**. Never delegate coding or
  experiment-running to an agent — do that hands-on. When you do dispatch an
  audit, carry every guardrail into its prompt (no shipped edits, numbers-only,
  anti-oracle, read-only).
- **Scratch**: throwaway code/logs are `scratch_*.{py,log,out}` (gitignored).
  Verify nothing un-ignored would be swept before any `git add`.
- **Commits**: only when asked. Multi-line message stating what ran + the verdict;
  end with the `Co-Authored-By` + `Claude-Session` trailer. Push when asked.

---

*Rule of thumb: if a result would change what we ship or claim, it must survive
(a) neutralisation to bit-exact parent, (b) an anti-oracle trace, (c) a
multi-log check, and (d) an independent audit. Most "wins" that skip these turn
out to be one of those four failures.*
