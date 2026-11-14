#!/bin/bash
# cov-TAE: rank different models (VDA video, VDA per-frame, VDA 2-bit, StreamVGGT) under every temporal
# metric. Needs the VDA predictions of covtae_queue.sh (Sintel max_samples 2000, Bonn max_samples 6000).
#   StreamVGGT: Sintel all frames, Bonn first 110 frames per sequence (its cache grows with every frame);
#   Bonn is analysed on those 110 frames for every model.
#   GPU=0 bash scripts/covtae_models.sh
set -u
REQ_GPU=${GPU:-0}
SELF=$(cd "$(dirname "$0")/.." && pwd)
C=$(dirname "$SELF")
source "$C/svg_env.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
PY=$C/svg_tcc/venv/bin/python
SVG=${SVG_ROOT:-$C/svg_tcc/StreamVGGT}
export STREAMVGGT_SRC=$SVG/src
OUT=$C/covtae
DUMPS=$OUT/dumps
mkdir -p "$OUT/logs"
say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT/STATUS_models.txt"; }
say "settings: gpu=$REQ_GPU streamvggt=$SVG"
ls "$SVG/ckpt/checkpoints.pth" > /dev/null || { say "FAILED: no StreamVGGT checkpoint at $SVG/ckpt/checkpoints.pth"; exit 1; }

wait_gpu() {   # until the GPU has 22 GB free (other jobs may still be running)
  local free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$REQ_GPU" | tr -d " ")
    [ "$free" -ge 22000 ] && return
    say "waiting for GPU $REQ_GPU (free ${free} MiB)"; sleep 300
  done
}

wait_dumps() {   # until the full-Bonn VDA predictions of <encoder> exist (max_samples 6000)
  until grep -q '"max_samples": 6000' "$DUMPS/bonn_$1_w1_fp32.json" 2>/dev/null; do
    say "waiting for Bonn $1 predictions"; sleep 300
  done
}

dump() {   # dump <dataset> <max-samples> [max-per-scene]
  local extra=""
  [ -n "${3:-}" ] && extra="--max-per-scene $3"
  wait_gpu
  say "START StreamVGGT $1"
  (cd "$SELF" && CUDA_VISIBLE_DEVICES=$REQ_GPU "$PY" -u -X utf8 scripts/covtae_dump_streamvggt.py --dataset "$1" \
      --weights "$SVG/ckpt/checkpoints.pth" --bf16 --max-samples "$2" $extra --dumps "$DUMPS" \
      > "$OUT/logs/streamvggt_$1.log" 2>&1) && say "OK StreamVGGT $1" || say "FAILED StreamVGGT $1"
  for ext in npz json; do          # the same predictions also join the ViT-L analysis group
    ln -sf "$DUMPS/${1}_vits_wS_streamvggt-none.$ext" "$DUMPS/${1}_vitl_wS_streamvggt-none.$ext"
  done
}

analyse() {   # analyse <dataset> <encoder> <extra args...>
  local ds=$1 enc=$2; shift 2
  wait_gpu
  say "START analysis models $ds $enc"
  (cd "$SELF" && CUDA_VISIBLE_DEVICES=$REQ_GPU "$PY" -u -X utf8 scripts/covtae_analysis.py --dataset "$ds" \
      --encoder "$enc" --dumps "$DUMPS" --tags w32_fp32,w1_fp32,w32_2bit,wS_streamvggt-none "$@" \
      --out "$OUT/analysis_models_${ds}_${enc}.json" > "$OUT/logs/analysis_models_${ds}_${enc}.log" 2>&1) \
      && say "OK analysis models $ds $enc" || say "FAILED analysis models $ds $enc"
}

dump sintel 2000
dump bonn 6000 110
for enc in vits vitl; do analyse sintel $enc --flow-root benchmark_data/sintel/training; done
for enc in vits vitl; do wait_dumps $enc; analyse bonn $enc --max-per-scene 110; done
(cd "$OUT" && tar czf "$C/covtae_models.tgz" STATUS_models.txt analysis_models_*.json logs/streamvggt_*.log logs/analysis_models_*.log)
say "ALL FINISHED (results in $C/covtae_models.tgz)"
