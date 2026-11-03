#!/usr/bin/env bash
# Fading-memory tests (VDA ViT-S streaming), after the night queue frees the GPU:
#  1. attention mass per cached frame by age (Bonn, Sintel)
#  2. causal age sensitivity (zero / 1-bit one age bucket at a time) and
#     equal-memory fading vs reversed vs shuffled vs uniform, with fidelity metrics
set -u
cd "$(dirname "$0")/.."
O=outputs/stream
while ! grep -q "ALL DONE" $O/night_queue.log 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] attention probes"
for ds in bonn sintel; do
  python -u -X utf8 scripts/stream_age_probe.py --dataset $ds --frames 150 --out $O/age_attention_$ds.json \
    > $O/age_attention_$ds.log 2>&1 || echo "  FAILED probe $ds"
  grep "^layer" $O/age_attention_$ds.log
done
CFG=$(cat scripts/fading_configs.txt)
for ds in sintel bonn; do
  echo "[$(date +%H:%M:%S)] fading eval $ds"
  python -u -X utf8 scripts/stream_eval.py --dataset $ds --encoder vits --configs "$CFG" \
    --out $O/${ds}_vits_fading.json > $O/${ds}_vits_fading.log 2>&1 || echo "  FAILED eval $ds"
done
echo "[$(date +%H:%M:%S)] ALL DONE"
