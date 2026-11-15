#!/bin/bash
# TCC on ScanNet (VDA's TAE benchmark: 20 test scenes, frames 10..179), with the configurations of the
# main VDA table (vda_extra.sh), ViT-S/B/L. Needs benchmark_data/scannet from scannet_prepare.sh.
#   lane A (GPU_A): ViT-S then ViT-B     lane B (GPU_B): ViT-L
#   GPU_A=2 GPU_B=0 bash scripts/scannet_tcc.sh
set -u
REQ_GPU_A=${GPU_A:-2}
REQ_GPU_B=${GPU_B:-0}
SELF=$(cd "$(dirname "$0")/.." && pwd)
C=$(dirname "$SELF")
source "$C/svg_env.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
PY=$C/svg_tcc/venv/bin/python
OUT=$C/scannet_tcc
CFGS=fp32,zero,int2-g16,int2-g64-asym,int4-g16::d8:bin-g16,int4-g16::d4:bin-g16,int2-g32-asym,int4-g16::d8:int2-g64-asym,int4-g16@win16,int4-g16@win8
mkdir -p "$OUT"
say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT/STATUS.txt"; }
say "settings: gpu_a=$REQ_GPU_A gpu_b=$REQ_GPU_B commit=$(cd "$SELF" && git log -1 --format=%h)"
n=$(ls -d "$SELF/benchmark_data/scannet"/scene* 2>/dev/null | wc -l)
say "scannet scenes: $n"
[ "$n" -ge 20 ] || { say "FAILED: need 20 scenes (run scannet_prepare.sh)"; exit 1; }

run() {   # run <encoder> <gpu>
  say "START scannet $1"
  if (cd "$SELF" && CUDA_VISIBLE_DEVICES=$2 "$PY" -u -X utf8 scripts/stream_eval.py --dataset scannet \
        --encoder "$1" --configs "$CFGS" --no-tae --out "$OUT/scannet_$1.json" > "$OUT/scannet_$1.log" 2>&1); then
    say "OK scannet $1"
  else
    say "FAILED scannet $1"
  fi
}

( run vits "$REQ_GPU_A"; run vitb "$REQ_GPU_A"; say "LANE A FINISHED" ) &
( run vitl "$REQ_GPU_B"; say "LANE B FINISHED" ) &
wait
(cd "$OUT" && tar czf "$C/scannet_tcc.tgz" STATUS.txt *.json *.log)
say "ALL FINISHED (results in $C/scannet_tcc.tgz)"
