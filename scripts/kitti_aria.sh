#!/bin/bash
# KITTI for the StreamVGGT video-depth benchmark, downloaded with aria2c (16 connections
# per file, 4 files at once) instead of wget, then MonST3R's prepare_kitti.py. Only what the
# preparation reads: the annotated depth (val split) and the 13 raw validation drives.
# Afterwards, when GPU_A is free (STAC finished), runs the KITTI benchmark for our caches.
#
#   bash scripts/kitti_aria.sh        (GPU_A=2)
set -u
SELF=$(cd "$(dirname "$0")/.." && pwd)
C=$(dirname "$SELF")
source "$C/svg_env.sh"
export ROOT=${ROOT:-$C/svg_tcc} VDAQ=$SELF CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
SVG=$ROOT/StreamVGGT
PY=$ROOT/venv/bin/python
MON=$ROOT/monst3r
K=$(readlink -f "$MON/data/kitti")
OUT=$C/svg_kitti
GPU_A=${GPU_A:-2}
mkdir -p "$OUT" "$K"
say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT/STATUS.txt"; }

command -v aria2c >/dev/null || { say "aria2c not found"; exit 1; }
pkill -f "avg-kitti" 2>/dev/null; true      # stop the slow wget from queue 6

S3=https://s3.eu-central-1.amazonaws.com/avg-kitti
{
  echo "$S3/data_depth_annotated.zip"
  for d in 2011_09_26_drive_0002 2011_09_26_drive_0005 2011_09_26_drive_0013 2011_09_26_drive_0020 \
           2011_09_26_drive_0023 2011_09_26_drive_0036 2011_09_26_drive_0079 2011_09_26_drive_0095 \
           2011_09_26_drive_0113 2011_09_28_drive_0037 2011_09_29_drive_0026 2011_09_30_drive_0016 \
           2011_10_03_drive_0047; do
    echo "$S3/raw_data/$d/${d}_sync.zip"
  done
} > "$OUT/urls.txt"

say "download into $K"
aria2c -x16 -s16 -j4 -c --file-allocation=none --summary-interval=60 -d "$K" -i "$OUT/urls.txt" \
  > "$OUT/aria2c.log" 2>&1 || { say "DOWNLOAD FAILED, see $OUT/aria2c.log"; exit 1; }
say "unzip"
(cd "$K" && for z in data_depth_annotated.zip 2011_*_sync.zip; do unzip -o -q "$z" || exit 1; done) \
  || { say "UNZIP FAILED"; exit 1; }
say "prepare"
(cd "$MON/datasets_preprocess" && "$PY" prepare_kitti.py) > "$OUT/prepare.log" 2>&1 || { say "PREPARE FAILED"; exit 1; }
# data/eval may itself be $MON/data: only link when it is a different folder
  [ "$(readlink -f "$SVG/data/eval")" = "$(readlink -f "$MON/data")" ] || ln -sfn "$MON/data/kitti" "$SVG/data/eval/kitti"
say "sequences: $(ls "$SVG/data/eval/kitti/depth_selection/val_selection_cropped/image_gathered" | wc -l)"

say "waiting for STAC to finish (GPU $GPU_A)"
while ! grep -q "ALL FINISHED" "$C/svg_stac/STATUS.txt" 2>/dev/null; do sleep 120; done
say "START kitti benchmark"
cd "$SVG/src" && export STREAMVGGT_SRC=$SVG/src
CUDA_VISIBLE_DEVICES=$GPU_A KVS="none plain:a2g64 kivi:2:64 delta:int4:bin:8 delta:int4:a2g64:8 xstream:2048:4:4" \
  DATASETS="kitti" bash "$VDAQ/scripts/vast_streamvggt.sh" eval > "$OUT/eval.log" 2>&1
DATASETS="kitti" bash "$VDAQ/scripts/vast_streamvggt.sh" fid > "$OUT/fid.log" 2>&1
say "ALL FINISHED"
