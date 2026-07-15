# Datasets

Classic CARMEN-format 2D lidar logs from the Radish repository, mirrored by
StachnissLab (pre-2014 robotics 2D-laser datasets). Run `./fetch_datasets.sh`
in this directory to download everything the experiments and drivers expect
(~120 MB unpacked). Please credit the original dataset authors (Dirk Haehnel
for Intel; see the StachnissLab page for the others).

| log | building | scans | beams | used as |
|---|---|---|---|---|
| intel.log | Intel Research Lab, Seattle (28.5 m) | 13,631 | 180 x 1 deg | primary tuning log |
| fr079.log | Freiburg building 079 | 4,934 | 360 x 0.5 deg | transfer log |
| aces.log | ACES3 Austin | 7,374 | 180 x 1 deg | transfer log (honest negative) |
| mit.log | MIT Infinite Corridor (1.9 km) | 17,480 | 180 x 1 deg | held-out stress log |
| fr101.log | Freiburg building 101 | 4,758 | 360 x 0.5 deg | held-out transfer (dense-revisit building) |
| belgioioso.log | Belgioioso Castle | 4,047 | 361 x 0.5 deg | held-out transfer (non-Manhattan structure) |
| fhw.log | exhibition hall (large open space) | 38,613 | 180 x 1 deg | suite growth 2026-07-10 (dense 483-pose ref; E2's best absolute result 0.35 m) |
| stata/*.bag | MIT Stata Center (PR2, Hokuyo 1040 x 0.25 deg, 260 deg FOV) | — | rosbag | **independent-class reference**: floorplan-anchored GT (~2-3 cm, per scan) — decouples ATE from the GMapping reference family; adapter = `runners/stata.py` |

fr101 and belgioioso were added 2026-07 to fill two gaps in the transfer suite
(a dense-revisit loopy building where loop closure should excel, and a
genuinely non-Manhattan environment); the selection rationale and the full
survey of candidate 2D-lidar SLAM datasets are in
[`../docs/sota/datasets.md`](../docs/sota/datasets.md).

The matching `.gfs.log` files are RBPF-corrected reference trajectories used
for ATE/RPE evaluation only (note: mit.gfs.log AND belgioioso.gfs.log have
corrupt/low-precision timestamps upstream; evaluation matches scans by exact
range-array identity instead — see docs/RESULTS.md. Intel/fr079/aces/fr101 have
clean shared-base timestamps.).

## Deutsches Museum (Cartographer, 270-deg backpack — SPOT-adjacent regime)

```bash
mkdir -p data/museum && curl -L -C - -o data/museum/cartographer_paper_deutsches_museum.bag \
  https://storage.googleapis.com/cartographer-public-data/bags/backpack_2d/cartographer_paper_deutsches_museum.bag
```
493 MB; topics: `horizontal_laser_2d` (MultiEchoLaserScan, 270°),
`vertical_laser_2d`, `imu`. NO wheel odometry (walking backpack) and no
published ground-truth relations (bucket probed 2026-07-10) — usable as a
frontend/loop stress set; absolute eval TBD.

## SPOT Telluride workshop (TARGET PLATFORM — first collected dataset)

```bash
# HF dataset lorinachey/spot-telluride-workshop-dataset (apache-2.0)
# pointclouds (22 shards, 2.4 GB) + odometry_imu + odometry_lio_sam
# -> data/spot_telluride/{pointclouds,odometry_imu,odometry_lio_sam}/
python3 -m runners.spot parse    # -> data/spot_telluride/scans.npz (ring-34 slice)
```
Ouster-class 1024×64 @ 20 Hz (the custom-head spec), SPOT odometry 528 Hz,
78 s / 36.5 m / ~7×7 m room. Protocol: LIDAR-ONLY runs (odometry withheld
as ground truth) — see `runners/spot.py`.

### SPOT Telluride drop 2 — SCHOOL building (runs 1+2, 2026-07-13)

Same HF dataset, reorganized: `school/run1` (117 pointcloud shards, 455 s,
8298 clouds) and `school/run2` (47 shards, 167 s, 3343 clouds); shard rows
carry `npz_bytes` (xyz/intensity/ring) instead of ASCII PCD. Fetch
pointclouds + odometry_imu + odometry_lio_sam + static into
`data/spot_telluride/school_run{1,2}/` (≈8.6 GB), then distill:

```bash
python3 -m runners.spot_school parse school_run1   # -> scans.npz
python3 -m runners.spot_school refs  school_run1   # -> ref_lio.npz
python3 -m runners.datasets run school_run2        # canonical run
```

**Reference warning (measured 2026-07-13, undocumented upstream):** this
drop's kinematic `odometry_imu` oscillates ±20–300 m for extended intervals
in both runs — unusable as reference. LIO-SAM is the only reference and only
run2's is healthy, for t<~80 s (the gated flat window: 342/836 kf). run1 has
ZERO usable reference → loop/stability stress set only (ATE prints nan).
Everything downstream consumes only the two npz caches; the parquet shards
can be deleted and re-fetched at will.

## TUM RGB-D + EuRoC (3D/6D vision SLAM references, 2026-07-14)

`data/tum/`: rgbd_dataset_freiburg3_long_office_household (415 kf) and
rgbd_dataset_freiburg2_pioneer_slam (373 kf) — RGB 640×480 @30 Hz,
REGISTERED 16-bit depth, 100 Hz 6-DoF mocap ground truth. Parse to the
QVGA integer-pipeline cache with `python3 -m runners.tum parse <seq>`
(→ `<seq>.npz`: gray/depth_mm/gt/K, bin-2 frame). Benches:
`python3 -m experiments.vision6d {place,rot} <seq>`.
EuRoC MAV moved upstream to the ETH Research Collection
(https://projects.asl.ethz.ch/datasets/euroc-mav/) — deferred.
