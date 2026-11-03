#!/usr/bin/env bash
# VDA ViT-B / ViT-L streaming: memory (synthetic video) then Sintel accuracy.
set -u
cd "$(dirname "$0")/.."
MEM="fp32,fp16,int2-g16,int3-g16::d8:bin-g16"
ACC="fp32,zero,int2-g16,int3-g16::d8:bin-g16"
for enc in vitb vitl; do
  echo "[$(date +%H:%M:%S)] memory $enc"
  python -u -X utf8 scripts/stream_memory.py --encoder $enc --frames 50 --configs "$MEM" \
    --out outputs/stream/mem_${enc}_518.json > outputs/stream/mem_${enc}.log 2>&1 || echo "  FAILED mem $enc"
  tail -6 outputs/stream/mem_${enc}.log
done
for enc in vitb vitl; do
  echo "[$(date +%H:%M:%S)] sintel $enc"
  python -u -X utf8 scripts/stream_eval.py --dataset sintel --encoder $enc --configs "$ACC" \
    --out outputs/stream/sintel_${enc}.json > outputs/stream/sintel_${enc}.log 2>&1 || echo "  FAILED sintel $enc"
  tail -7 outputs/stream/sintel_${enc}.log
done
echo "[$(date +%H:%M:%S)] ALL DONE"
