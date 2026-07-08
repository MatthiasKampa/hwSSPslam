#!/bin/sh
# Fetch the CARMEN 2D lidar logs used by this project (StachnissLab mirrors).
set -e
BASE="http://www2.informatik.uni-freiburg.de/~stachnis/datasets/datasets"
get() { echo "fetching $2"; curl -sL "$1" -o "$2"; gunzip -kf "$2"; }

get "$BASE/intel-lab/intel.log.gz"        intel.log.gz
get "$BASE/intel-lab/intel.gfs.log.gz"    intel.gfs.log.gz
get "$BASE/fr079/fr079-complete.log.gz"   fr079.log.gz
get "$BASE/fr079/fr079-complete.gfs.log.gz" fr079.gfs.log.gz
get "$BASE/aces/aces_publicb.log.gz"      aces_publicb.log.gz
get "$BASE/aces/aces_publicb.gfs.log.gz"  aces_publicb.gfs.log.gz
ln -sf aces_publicb.log aces.log
ln -sf aces_publicb.gfs.log aces.gfs.log
get "$BASE/MIT/MIT_Infinite_Corridor_2002_09_11_same_floor.log.gz"     mit_raw.log.gz
get "$BASE/MIT/MIT_Infinite_Corridor_2002_09_11_same_floor.gfs.log.gz" mit_raw.gfs.log.gz
ln -sf mit_raw.log mit.log
ln -sf mit_raw.gfs.log mit.gfs.log
# held-out transfer logs added 2026-07 (see SotA/datasets.md for the selection):
# fr101 = dense-revisit loopy building; belgioioso = non-Manhattan castle.
get "$BASE/fr101/fr101.carmen.log.gz"       fr101.log.gz
get "$BASE/fr101/fr101.carmen.gfs.log.gz"   fr101.gfs.log.gz
get "$BASE/belgioioso/belgioioso.log.gz"     belgioioso.log.gz
get "$BASE/belgioioso/belgioioso.gfs.log.gz" belgioioso.gfs.log.gz
echo "done."
