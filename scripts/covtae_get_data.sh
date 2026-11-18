#!/bin/bash
# cov-TAE: download TUM RGB-D (8 sequences: static and dynamic) and 7-Scenes (one test sequence per scene) into
# benchmark_data/{tum,sevenscenes}. Public data; about 20 GB of downloads.
#   bash scripts/covtae_get_data.sh
set -u
SELF=$(cd "$(dirname "$0")/.." && pwd)
C=$(dirname "$SELF")
RAW=$C/covtae_raw
mkdir -p "$RAW" "$SELF/benchmark_data/tum" "$SELF/benchmark_data/sevenscenes"
LOG=$C/covtae/STATUS_data.txt
mkdir -p "$(dirname "$LOG")"
say() { echo "[$(date +%H:%M)] $*" | tee -a "$LOG"; }

for s in freiburg1_desk freiburg1_room freiburg2_desk freiburg3_long_office_household \
         freiburg3_walking_xyz freiburg3_walking_halfsphere freiburg3_walking_static freiburg3_sitting_xyz; do
  d=$SELF/benchmark_data/tum/rgbd_dataset_$s
  [ -f "$d/groundtruth.txt" ] && { say "SKIP tum $s"; continue; }
  fr=${s%%_*}
  wget -q -c -O "$RAW/$s.tgz" "https://cvg.cit.tum.de/rgbd/dataset/$fr/rgbd_dataset_$s.tgz" \
    && tar xzf "$RAW/$s.tgz" -C "$SELF/benchmark_data/tum" && say "OK tum $s ($(ls "$d/depth" | wc -l) depth frames)" \
    || say "FAILED tum $s"
done

URL=https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8
for pair in chess:seq-03 fire:seq-03 heads:seq-01 office:seq-02 pumpkin:seq-01 redkitchen:seq-03 stairs:seq-01; do
  sc=${pair%%:*}; seq=${pair##*:}
  d=$SELF/benchmark_data/sevenscenes/${sc}_$seq
  [ -d "$d" ] && [ "$(ls "$d" | grep -c pose)" -gt 0 ] && { say "SKIP 7scenes $sc $seq"; continue; }
  (wget -q -c -O "$RAW/$sc.zip" "$URL/$sc.zip" && cd "$RAW" && unzip -o -q "$sc.zip" "$sc/$seq.zip" \
     && unzip -o -q "$sc/$seq.zip" -d "$RAW/$sc" && mkdir -p "$d" && mv "$RAW/$sc/$seq/"* "$d/") \
    && say "OK 7scenes $sc $seq ($(ls "$d" | grep -c depth) frames)" || say "FAILED 7scenes $sc $seq"
done
du -sh "$SELF/benchmark_data/tum" "$SELF/benchmark_data/sevenscenes" | sed 's/^/disk: /' | tee -a "$LOG"
say "ALL FINISHED"
