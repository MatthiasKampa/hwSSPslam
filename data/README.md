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

The matching `.gfs.log` files are RBPF-corrected reference trajectories used
for ATE/RPE evaluation only (note: mit.gfs.log has corrupt timestamps
upstream; evaluation matches scans by exact range-array identity instead —
see RESULTS.md).
