#!/usr/bin/env bash
# Mixed-precision delta coding of the streaming temporal cache (no rotation).
# Waits for the lattice-vs-INT phase-3 sweep to finish (shared GPU), then runs
# Bonn and Sintel.  Config: <keyframe mode>::d<K>:<residual mode>
set -u
cd "$(dirname "$0")/.."
LOG3=outputs/lattice_vs_int_s8_phase3.log
while [ -f "$LOG3" ] && ! grep -q "ALL DONE" "$LOG3"; do sleep 30; done
CFGS="bin-g16,int3-g16::d8:bin-g16,int3-g16::d16:bin-g16,int4-g16::d8:bin-g16,int3-g16::d8:int2-g16"
for ds in bonn sintel; do
  echo "[$(date +%H:%M:%S)] $ds"
  python -u -X utf8 scripts/stream_eval.py --dataset $ds --encoder vits --configs "$CFGS" \
    --out outputs/stream/${ds}_vits_delta_mixed.json > outputs/stream/${ds}_vits_delta_mixed.log 2>&1 \
    || echo "  FAILED $ds"
done
echo "[$(date +%H:%M:%S)] ALL DONE"
