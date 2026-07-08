# 2D-lidar SLAM datasets — survey & selection (2026-07-07)

Goal: find the TWO datasets that best extend the current real-log suite
(Intel / fr079 / ACES / MIT — classic CARMEN FLASER logs with RBPF-corrected
`.gfs.log` references). Selection is driven by the transfer findings in
`RESULTS.md`:

- payoff scales with **dead-reckoning drift × revisit density**;
- the stack **does not snap large sparse loops** (MIT's 57.8 m held-out
  failure = corridor self-similarity / place-recognition starvation);
- ACES is an **honest negative** (odometry already excellent, one big loop);
- the encoder's **one real fragility is environment-coupled angular spectrum**
  (RESULTS finding 6 / world-dependence table: flat-wall grids win, but
  curved/organic "blob" geometry needs angular density — only ever tested in
  *synthetic* worlds).

Two gaps in the current suite:
1. **No dense-revisit loopy multi-room building** — the regime where loop
   closure should EXCEL (the complement to MIT's sparse corridors and ACES's
   single loop).
2. **No genuinely different environment type** — everything so far is a
   rectilinear office/lab building.

## Zero-adapter criterion (measured against the actual driver)

`ssp_slam_carmen.py::parse_flaser` reads **only `FLASER` lines** and takes the
`x y theta` pose; `ssp_bounded_carmen.py` builds beams as
`-90° + i·(180°/n_beams)` — i.e. it **assumes a 180° front-facing FOV** and
adapts to any beam count. Evaluation loads `<log>`→`<log>.gfs.log` and matches
poses. Therefore **zero adapter cost** requires ALL of:

- a **raw** CARMEN log whose scans are `FLASER` messages (NOT `RAWLASER` /
  `ROBOTLASER` — the "new CARMEN" format the driver does not parse), whose
  `FLASER` poses are the **uncorrected odometry** poses;
- a **180° FOV** SICK-style front laser (any beam count; 360@0.5° like fr079
  and 180@1° like Intel/ACES both work);
- a **`.gfs.log`** corrected reference (FLASER lines, corrected poses);
- **logger-relative timestamps** sharing a base between log and `.gfs.log`
  (else eval needs the MIT-style range-array-identity matcher).

All URLs below live under the StachnissLab mirror; the `www2.informatik…`
base used by `data/fetch_datasets.sh` now 302-redirects to
`ais.informatik.uni-freiburg.de/staff/stachnis/…` — `curl -L` (already used)
follows it transparently. Canonical host verified resolving 2026-07-07.

`BASE = http://www2.informatik.uni-freiburg.de/~stachnis/datasets/datasets`

## Survey — CARMEN / Radish / StachnissLab (verified 2026-07-07)

Verified = `curl -IL` returned `200 application/x-gzip`; scan counts / beams /
timestamp class = streamed and inspected first-hand unless marked "(catalog)".

| dataset | dir | raw log (FLASER?) | `.gfs.log` ref | beams / FOV | raw scans / ref poses | gz size (raw) | environment | zero-adapter? | gap it fills |
|---|---|---|---|---|---|---|---|---|---|
| **Intel** *(in suite)* | `intel-lab/` | `intel.log` ✓ | `intel.gfs.log` | 180 / 180° | 13,631 / ~890 | | Intel Lab, 28.5 m, loops+rooms | yes | primary tuning |
| **fr079** *(in suite)* | `fr079/` | `fr079-complete.log` ✓ | `…gfs.log` | 360 / 180° | 4,934 / — | | Freiburg bldg 079 | yes | transfer |
| **ACES** *(in suite)* | `aces/` | `aces_publicb.log` ✓ | `…gfs.log` | 180 / 180° | 7,374 / — | Austin ACES3, one big loop | yes | honest negative |
| **MIT Inf. Corridor** *(in suite)* | `MIT/` | `MIT_Infinite_Corridor_2002_09_11_same_floor.log` ✓ | `…gfs.log` | 180 / 180° | 17,480 / — | 1.9 km corridors | yes* | held-out sparse-loop stress |
| **fr101** ← **PICK 1** | `fr101/` | `fr101.carmen.log` ✓ | `fr101.carmen.gfs.log` ✓ | **360 / 180°** | **4,758 / 292** | 2.1 MB | Freiburg CS bldg 101 — large multi-room+corridor, several loops | **yes** (fr079-identical setup, clean logger-rel ts) | **dense-revisit loopy building** |
| **Belgioioso** ← **PICK 2** | `belgioioso/` | `belgioioso.log` ✓ | `belgioioso.gfs.log` ✓ | **361 / 180°** | **4,047 / 395** | 1.5 MB | **Belgioioso Castle** — non-Manhattan stone walls, courtyards, semi-outdoor (has NMEA-GGA GPS) | **yes**, w/ eval caveat† | **genuinely different structure** |
| FHW | `fhw/` | `fhw-rec-001.log` ✓ | `…gfs.log` ✓ | 180 / 180° | 38,613 / 488 | 6.2 MB | large open exhibition hall | yes, clean ts | different SCALE (open hall); 2nd sparse stress — but 2× MIT length |
| MIT CSAIL | `csail/` | `csail-newcarmen.log` **✗ (RAWLASER/ROBOTLASER)**; `csail.corrected.log` is FLASER but already-corrected | `csail.corrected.gfs.gz` (native gfs, not `.gfs.log`) | 361 / 180° | 383+ / — | 1.2 MB | dense multi-room research floor — archetypal loopy office | **no** — needs RAWLASER/ROBOTLASER→FLASER converter to get a *raw-odometry* input | would fill gap 1, but adapter cost |
| Orebro | `orebro/` | `orebro.raw.log` ✓ | `orebro.gfs.log` ✓ | (catalog) | small (0.23 MB) | | Örebro Univ. indoor | likely yes | generic indoor, low novelty |
| Freiburg Campus | `freiburg-campus/` | `fr-campus-20040714.carmen.log` ✓ | `…gfs.log` ✓ | (catalog) | | | **outdoor** campus, GPS | probably (verify FOV) | outdoor — but sparse structure, hard for 2D scan-match |
| Seattle (UW, 4-floor) | `seattle-4-floor/` | **none** (only `seattle-r.gfs.log`) | `seattle-r.gfs.log` | (catalog) | | | 4-floor building | **no** — no raw log; would read `odom_*` fields (adapter) | multi-floor, but no raw input |
| Edmonton | `edmonton/` | **none** (only `edmonton.gfs.log`) | `edmonton.gfs.log` | (catalog) | | | convention centre | **no** — gfs-only | — |
| Acapulco (mexico) | `mexico/` | **none** (only `mexico.gfs.log`) | `mexico.gfs.log` | (catalog) | | | Acapulco convention centre | **no** — gfs-only | — |

\* MIT eval already uses range-array-identity matching (gfs timestamps corrupt
upstream). † see caveat under Pick 2.

Non-usable-as-raw note: Seattle/Edmonton/Acapulco ship **only** the corrected
`.gfs.log`. Their FLASER lines still carry raw odometry in the trailing
`odom_x odom_y odom_theta` fields, so they *could* be driven — but only by
adding a "read odom fields as the motion prior" path, i.e. an adapter. Skipped.

## Survey — non-CARMEN standards (for completeness; all need conversion)

| dataset | source | format | sensor / FOV | ground truth | conversion cost for this driver |
|---|---|---|---|---|---|
| **Deutsches Museum** (Cartographer) | google-cartographer-ros datasets | ROS `.bag` | Hokuyo UTM-30LX, **270° FOV** | Cartographer SLAM result (not independent GT) | **High**: bag→FLASER extraction **and** the 180°-FOV beam model must be generalized to 270° (breaks the `180/n` assumption). |
| **Rawseeds** Bicocca (indoor) / Bovisa (mixed in/outdoor) | rawseeds.org | custom multi-sensor CSV/CLF | SICK LMS 180° + Hokuyo 270° | GPS + vision + laser reference ("extended GT") | **Med-high**: pick the 180° SICK stream, convert CSV→FLASER, remap GT to `.gfs.log` pose stream. |
| **MIT Stata Center** | projects.csail.mit.edu/stata | ROS `.bag` | Hokuyo | ground truth from survey/odometry fusion | **High**: bag extraction + FOV/format conversion. |

These buy nothing the Radish set doesn't, at real adapter cost, so they are
not selected. Deutsches Museum is the natural future pick if a **270° FOV**
generalization of the beam model is ever built (worth a one-line
`fov_deg` parameter someday).

---

## THE TWO PICKS

### Pick 1 — Freiburg Building 101 (`fr101`) — the dense-revisit loopy building

**Fills gap 1** (loop-closure-excels regime): a large university CS building
with multiple rooms, connected corridors and **several revisited loops** —
the structural complement to MIT's single 1.9 km sparse corridor and ACES's
one big square loop. Recorded with the **same SICK 180°/0.5° / 360-beam**
front laser as fr079, which the driver already runs zero-shot.

- **Zero adapter, verified:** raw `fr101.carmen.log` = 4,758 `FLASER` scans,
  360 beams (180° FOV — driver's `180/360 = 0.5°` spacing is exact, and 360 is
  even so it hits the 0/90° wall-normal grid the encoder likes). Corrected
  reference `fr101.carmen.gfs.log` = 292 FLASER poses. **Timestamps are
  logger-relative and share a base** (raw first ts 156.3, gfs 158.4) exactly
  like Intel → the existing timestamp matcher works; **no MIT-style
  workaround, no code change**. (Raw poses start at x≈11.4 in the odometry
  frame, gfs near origin in the corrected frame — same as Intel; `rpe.py`'s
  rigid alignment already handles the frame offset.)
- **Hypothesis it tests:** on a building denser in revisits than any tuning
  log, does the graph's benefit *grow* with revisit density as the ledger
  predicts — i.e. does fr101's loop-closed ATE beat raw odometry by a *larger*
  drift-proportional margin than fr079, and approach the in-repo ICP/RBPF
  baselines? This is the positive control that MIT (sparse) and ACES (single
  loop) cannot provide. Secondary: the 292-pose sparse reference makes it an
  ACES-class ATE check (coarse but valid).
- **Adapter work: none.** Fetch snippet:
  ```sh
  get "$BASE/fr101/fr101.carmen.log.gz"     fr101.log.gz
  get "$BASE/fr101/fr101.carmen.gfs.log.gz" fr101.gfs.log.gz
  ```
  (the `get()` helper in `fetch_datasets.sh` gunzips; `.log`/`.gfs.log`
  basenames satisfy the driver's `path.replace(".log",".gfs.log")`.)

### Pick 2 — Belgioioso Castle (`belgioioso`) — the genuinely different environment

**Fills gap 2** (genuinely different structure) and **directly probes the
encoder's one documented real fragility.** Belgioioso is a **historic castle**:
thick irregular **non-Manhattan** stone walls, courtyards, arches, and
semi-outdoor traversal (the log carries NMEA-GGA GPS). This is the closest
real-world analogue to RESULTS finding 6 / the synthetic **"curved blob
world"**, where flat-wall octave grids lose their exact-wall-normal advantage
and angular *density* dominates — a regime the project has only ever tested in
simulation. Every current real log is a rectilinear office building; this is
the first that isn't.

- **Zero driver adapter, verified:** raw `belgioioso.log` = 4,047 `FLASER`
  scans, 361 beams; corrected `belgioioso.gfs.log` = 395 FLASER poses. 180°
  FOV SICK. parse_flaser skips the interleaved GPS/ODOM lines cleanly.
- **Eval caveat † (flag, workaround exists):** the `.gfs.log` timestamps are
  written in **low-precision scientific notation** (`1.01893e+09` vs the raw
  log's full `1018931783.431252`), so second-level precision is lost — the
  **same class of problem as MIT's corrupt gfs**. Use the existing MIT
  **range-array-identity** matcher for ATE (the gfs re-emits the same scans
  with corrected poses, so range arrays match byte-for-byte). This is
  eval-side only; the SLAM run itself is untouched.
- **Minor encoder caveat:** 361 beams is endpoint-inclusive (−90…+90 at 0.5°),
  but the driver uses `180/361 = 0.4986°` spacing → a cumulative ~0.14° angular
  compression at the array ends (the "endpoint-inclusive stretch" RESULTS
  already saw cost a few points of fallback rate on Intel). Harmless in
  practice; an *optional* one-line fix is `linspace(-90,90,n)` for inclusive
  odd-count arrays. Not required to run.
- **Hypothesis it tests:** does the octave-grid encoder's flat-wall-tuned
  angular sampling degrade on real non-rectilinear geometry the way the
  synthetic blob world predicts — and does loop closure still recover
  drift-proportionally when wall normals no longer align to the grid? A clean
  positive would be strong evidence the encoder generalizes past Manhattan
  worlds; a degradation localizes the fragility to real data and motivates the
  jittered/36-angle variant. Either outcome is publishable signal the current
  suite cannot produce.
- **Adapter work: none on the driver** (one eval-script reuse of the MIT
  range-identity matcher). Fetch snippet:
  ```sh
  get "$BASE/belgioioso/belgioioso.log.gz"     belgioioso.log.gz
  get "$BASE/belgioioso/belgioioso.gfs.log.gz" belgioioso.gfs.log.gz
  ```

**If a fully clean eval is preferred over maximal environmental contrast,**
`fhw` (large open exhibition hall, 180 beams, **clean logger-relative
timestamps**, 38,613 scans) is the fallback for gap 2 — but it is a *scale*
difference (open hall) more than a *structural* one, and at 2× MIT's length it
behaves as a second sparse-stress log rather than a clean environment contrast.
Belgioioso is the better gap-2 answer; fhw is the safe alternate.

## Summary

| pick | dataset | gap filled | adapter cost | key caveat |
|---|---|---|---|---|
| 1 | **fr101** (Freiburg bldg 101) | dense-revisit loopy multi-room building — loop closure should EXCEL | **none** (fr079-identical 360-beam setup, clean timestamps) | sparse 292-pose reference (ACES-class ATE) |
| 2 | **belgioioso** (Belgioioso Castle) | genuinely different, non-Manhattan structure — probes the encoder's angular-spectrum fragility on real data | **none on driver**; eval reuses MIT range-identity matcher | gfs timestamps low-precision (MIT-class); 361-beam endpoint stretch (minor) |
