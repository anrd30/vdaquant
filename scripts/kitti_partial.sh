#!/bin/bash
# KITTI for the StreamVGGT video-depth benchmark: only the frames the benchmark reads, fetched
# with HTTP range requests into the official zips (scripts/kitti_partial.py), then MonST3R's
# prepare_kitti.py.
# Afterwards, when GPU_A is free (STAC finished), runs the KITTI benchmark for our caches.
#
#   bash scripts/kitti_partial.sh     (GPU_A=2)
set -u
SELF=$(cd "$(dirname "$0")/.." && pwd)
C=$(dirname "$SELF")
source "$C/svg_env.sh"
export ROOT=${ROOT:-$C/svg_tcc} VDAQ=$SELF CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
SVG=$ROOT/StreamVGGT
PY=$ROOT/venv/bin/python
MON=$ROOT/monst3r
K=$(readlink -f "$MON/data/kitti")
OUT=$C/svg_kitti2
GPU_A=${GPU_A:-2}
mkdir -p "$OUT" "$K"
say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT/STATUS.txt"; }

PIP_CONFIG_FILE=/dev/null "$PY" -m pip install -q remotezip || { say "pip remotezip failed"; exit 1; }
say "fetch the benchmark frames only, into $K"
"$PY" -u "$VDAQ/scripts/kitti_partial.py" --dst "$K" --threads 8 > "$OUT/partial.log" 2>&1   || { say "FETCH FAILED, see $OUT/partial.log"; exit 1; }
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
