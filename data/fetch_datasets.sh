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
# added 2026-07-10 (suite growth; see RESULTS "better benchmarking datasets"):
# fhw = large open exhibition hall (clean timestamps, dense 483-pose ref).
get "$BASE/fhw/fhw-rec-001.log.gz"      fhw.log.gz
get "$BASE/fhw/fhw-rec-001.gfs.log.gz"  fhw.gfs.log.gz
# NOTE orebro was evaluated and dropped (frontend never engages — beam/FOV
# mismatch with the FLASER driver assumptions; identical ATE across configs).
#
# MIT Stata Center (INDEPENDENT-class reference: floorplan-anchored GT,
# ~2-3 cm, per scan) — Google Drive; needs gdown (pip install gdown):
#   mkdir -p stata && cd stata
#   python3 -m gdown 14i9n9Y9HRulX-U8ae6DMnfzZlydwlEWn -O 2012-01-27-07-37-01.bag
#   python3 -m gdown 10ncCNbw0pnsQ6PQVBZdIwNW96pxmMdEk -O 2012-01-27_part1_floor2.gt.poses
#   python3 -m gdown 1ljiyCKPo7JTGcdLj74NqvHKZTSPMh9ui -O 2012-01-27_part3_floor2.gt.poses
# then: python3 ssp_stata.py   (needs: pip install rosbags)
#
# RAWSEEDS Bicocca (independent camera-network GT + published GMapping
# baselines vs that GT — ATE 2.04+-1.87 on Bicocca_2009-02-25b): files live
# in AIRLab's Dropbox folder (link at
# https://airlab.deib.polimi.it/datasets-and-tools/); per-file headless
# addressing defeated us — one manual download of
# Bicocca_2009-02-25b-{SICK_FRONT,ODOMETRY_XYT,GROUNDTRUTH}.csv.bz2 suffices.

# Deutsches Museum (Cartographer 270-deg backpack; SPOT-adjacent walking
# regime; no odometry topic, no published GT relations — loop-stress set)
mkdir -p data/museum
curl -L -C - -o data/museum/cartographer_paper_deutsches_museum.bag \
  https://storage.googleapis.com/cartographer-public-data/bags/backpack_2d/cartographer_paper_deutsches_museum.bag
