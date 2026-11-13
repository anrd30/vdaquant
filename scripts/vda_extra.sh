#!/bin/bash
# VDA on more data (supplement): all five standard Bonn sequences (balloon2, crowd2, crowd3,
# person_tracking2, synchronous; full length) and the 13 KITTI validation drives, ViT-S/B/L,
# with the configurations of the main VDA table, short-window baselines (last 16 / 8 frames at INT4),
# and ground-truth metrics under per-frame and per-sequence alignment. Sintel and the 3-scene Bonn set
# are rerun so every dataset has the same measurements.
#   lane A (GPU_A): Bonn, 5 sequences      lane B (GPU_B): KITTI drives
#
#   bash scripts/vda_extra.sh          (GPU_A=2 GPU_B=0)
set -u
SELF=$(cd "$(dirname "$0")/.." && pwd)
C=$(dirname "$SELF")
source "$C/svg_env.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
PY=$C/svg_tcc/venv/bin/python
BD=$SELF/benchmark_data
OUT=$C/vda_extra2
GPU_A=${GPU_A:-2}
GPU_B=${GPU_B:-0}
CFGS=fp32,zero,int2-g16,int2-g64-asym,int4-g16::d8:bin-g16,int4-g16::d4:bin-g16,int2-g32-asym,int4-g16::d8:int2-g64-asym,int4-g16@win16,int4-g16@win8
mkdir -p "$OUT"
say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT/STATUS.txt"; }

# Bonn: the five sequences of the StreamVGGT/MonST3R benchmark, full length
mkdir -p "$BD/bonn5"
for s in balloon2 crowd2 crowd3 person_tracking2 synchronous; do
  ln -sfn "$C/svg_tcc/StreamVGGT/data/eval/bonn/rgbd_bonn_dataset/rgbd_bonn_$s" "$BD/bonn5/rgbd_bonn_$s"
done
ls "$BD/bonn5" | tee -a "$OUT/STATUS.txt"
ls "$BD/kitti/depth_selection/val_selection_cropped/image_gathered" | wc -l | sed 's/^/kitti drives: /' | tee -a "$OUT/STATUS.txt"

run() {   # run <dataset> <gpu>
  local ds=$1 gpu=$2 enc
  for enc in vits vitb vitl; do
    say "START $ds $enc"
    if (cd "$SELF" && CUDA_VISIBLE_DEVICES=$gpu "$PY" -u -X utf8 scripts/stream_eval.py --dataset "$ds" \
          --encoder "$enc" --configs "$CFGS" --no-tae --out "$OUT/${ds}_${enc}.json" > "$OUT/${ds}_${enc}.log" 2>&1); then
      say "OK $ds $enc"
    else
      say "FAILED $ds $enc"
    fi
  done
}

DATASETS_A=${DATASETS_A:-"sintel bonn5"}
DATASETS_B=${DATASETS_B:-"bonn kittiseq"}
( for d in $DATASETS_A; do run $d "$GPU_A"; done; say "LANE A FINISHED" ) &
( for d in $DATASETS_B; do run $d "$GPU_B"; done; say "LANE B FINISHED" ) &
wait
(cd "$OUT" && tar czf "$C/vda_extra2.tgz" STATUS.txt *.json *.log)
say "ALL FINISHED (results in $C/vda_extra2.tgz)"
