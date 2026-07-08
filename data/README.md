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
