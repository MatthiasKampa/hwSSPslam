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
| stata/*.bag | MIT Stata Center (PR2, Hokuyo 1040 x 0.25 deg, 260 deg FOV) | — | rosbag | **independent-class reference**: floorplan-anchored GT (~2-3 cm, per scan) — decouples ATE from the GMapping reference family; adapter = `ssp_stata.py` |

fr101 and belgioioso were added 2026-07 to fill two gaps in the transfer suite
(a dense-revisit loopy building where loop closure should excel, and a
genuinely non-Manhattan environment); the selection rationale and the full
survey of candidate 2D-lidar SLAM datasets are in
[`../SotA/datasets.md`](../SotA/datasets.md).

The matching `.gfs.log` files are RBPF-corrected reference trajectories used
for ATE/RPE evaluation only (note: mit.gfs.log AND belgioioso.gfs.log have
corrupt/low-precision timestamps upstream; evaluation matches scans by exact
range-array identity instead — see RESULTS.md. Intel/fr079/aces/fr101 have
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
python3 ssp_spot.py parse    # -> data/spot_telluride/scans.npz (ring-34 slice)
```
Ouster-class 1024×64 @ 20 Hz (the custom-head spec), SPOT odometry 528 Hz,
78 s / 36.5 m / ~7×7 m room. Protocol: LIDAR-ONLY runs (odometry withheld
as ground truth) — see `ssp_spot.py`.
