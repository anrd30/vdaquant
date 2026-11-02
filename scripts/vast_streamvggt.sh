#!/usr/bin/env bash
# One-shot setup + run for StreamVGGT KV-cache experiments on a rented GPU.
# Usage (inside tmux!):   bash vast_streamvggt.sh setup     # ~30-60 min
#                         bash vast_streamvggt.sh eval      # official video depth, all KV configs
#                         bash vast_streamvggt.sh oom       # frames-until-OOM benchmark
set -eu
ROOT=${ROOT:-$HOME/work}
SVG=$ROOT/StreamVGGT
VDAQ=$ROOT/vdaquant
MON=$ROOT/monst3r
KVS=${KVS:-"none fp16 plain:int4 plain:int2 delta:int3:bin:8 delta:int4:bin:8"}
DATASETS=${DATASETS:-"sintel bonn kitti"}

setup() {
  mkdir -p "$ROOT" && cd "$ROOT"
  [ -d "$SVG" ]  || git clone --depth 1 https://github.com/wzzheng/StreamVGGT.git "$SVG"
  [ -d "$VDAQ" ] || git clone -b verify https://github.com/anrd30/vdaquant.git "$VDAQ"
  [ -d "$MON" ]  || git clone --depth 1 https://github.com/Junyi42/monst3r.git "$MON"
  cd "$SVG" && pip install -q -r requirements.txt && pip install -q accelerate huggingface_hub
  mkdir -p "$SVG/ckpt"
  [ -f "$SVG/ckpt/checkpoints.pth" ] || \
    huggingface-cli download lch01/StreamVGGT checkpoints.pth --local-dir "$SVG/ckpt"
  # Evaluation data via MonST3R's scripts (same protocol StreamVGGT reports).
  cd "$MON/data"
  for d in sintel bonn kitti; do [ -d "$d" ] || bash download_$d.sh; done
  cd "$MON/datasets_preprocess" && python prepare_bonn.py && python prepare_kitti.py
  mkdir -p "$SVG/data" && ln -sfn "$MON/data" "$SVG/data/eval"
  echo "== check these paths exist =="
  ls -d "$SVG/data/eval/sintel/training/final" "$SVG/data/eval/bonn/rgbd_bonn_dataset" \
        "$SVG/data/eval/kitti/depth_selection/val_selection_cropped/image_gathered" || true
  nvidia-smi --query-gpu=name,memory.total --format=csv
}

run_eval() {
  cd "$SVG/src"
  export STREAMVGGT_SRC="$SVG/src"
  for kv in $KVS; do
    tag=${kv//:/_}
    for data in $DATASETS; do
      out="../eval_results/video_depth/${data}_${tag}"
      echo "[$(date +%H:%M)] $kv $data"
      python -u "$VDAQ/scripts/streamvggt_kv_launch.py" --kv "$kv" -- \
        --weights ../ckpt/checkpoints.pth --output_dir "$out" --eval_dataset "$data" --size 518 \
        > "$out.log" 2>&1 || { echo "  launch failed: $out.log"; continue; }
      python ../src/eval/video_depth/eval_depth.py --output_dir "$out" --eval_dataset "$data" \
        --align scale >> "$out.log" 2>&1 || echo "  scoring failed: $out.log"
      tail -3 "$out.log"
    done
  done
}

run_oom() {
  cd "$SVG/src"
  export STREAMVGGT_SRC="$SVG/src"
  python -u "$VDAQ/scripts/streamvggt_oom.py" --weights ../ckpt/checkpoints.pth \
    --frames-dir ../data/eval/bonn/rgbd_bonn_dataset/rgbd_bonn_crowd/rgb \
    --kv "${KVS// /,}" --max-frames ${MAXF:-3000} --out ../eval_results/oom.json
}

case "${1:-}" in
  setup) setup ;;
  eval)  run_eval ;;
  oom)   run_oom ;;
  *) echo "usage: $0 setup|eval|oom" ;;
esac
