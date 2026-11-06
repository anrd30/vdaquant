#!/usr/bin/env bash
# Queue 2 (2026-11-03 evening): stronger INT2 baselines, group-size and K sweeps.
# Resumable like run_overnight.sh: a stage is skipped when its JSON has all results.
#   start detached (PowerShell):
#   Start-Process 'C:\Program Files\Git\bin\bash.exe' -ArgumentList '-lc','"cd /c/Users/aniru/vdaquant && bash scripts/run_queue2.sh >> outputs/stream/queue2_stdout.log 2>&1"' -WindowStyle Hidden
set -u
cd "$(dirname "$0")/.."
O=outputs/stream
LOG=$O/queue2.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }

done_n() {
  python -I -X utf8 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if len(d.get('results', [])) >= int(sys.argv[2]) else 1)" "$1" "$2" </dev/null 2>/dev/null
}

eval_stage() {  # name dataset encoder n_expected configs
  local name=$1 ds=$2 enc=$3 n=$4 cfg=$5
  if done_n $O/$name.json $n; then log "skip $name (done)"; return; fi
  log "run  $name"
  python -u -X utf8 scripts/stream_eval.py --dataset $ds --encoder $enc --configs "$cfg" \
    --out $O/$name.json </dev/null > $O/$name.log 2>&1 && log "done $name" || log "FAILED $name (see $O/$name.log)"
}

log "=== queue2 start ==="
# 1. KIVI-style asymmetric INT2 at equal size (g64, 2.5 b) and at g32 (3.0 b): a stronger 2-bit baseline
KIVI="fp32,int2-g64-asym,int2-g32-asym"
eval_stage sintel_vits_int2asym sintel vits 3 "$KIVI"
eval_stage sintel_vitb_int2asym sintel vitb 3 "$KIVI"
eval_stage sintel_vitl_int2asym sintel vitl 3 "$KIVI"
# 2. residual group size (8-bit scale index for every g) and keyframe interval K=2, 64
SWEEP="fp32,int4-g16::d8:bin-g8-s8,int4-g16::d8:bin-g32-s8,int4-g16::d8:bin-g64-s8,int4-g16::d2:bin-g16,int4-g16::d64:bin-g16"
eval_stage sintel_vits_sweep sintel vits 6 "$SWEEP"
# 3. Bonn for the stronger INT2 baseline
eval_stage bonn_vits_int2asym bonn vits 3 "$KIVI"
eval_stage bonn_vitb_int2asym bonn vitb 3 "$KIVI"
eval_stage bonn_vitl_int2asym bonn vitl 3 "$KIVI"
# 4. Bonn sweep
eval_stage bonn_vits_sweep bonn vits 6 "$SWEEP"
log "=== queue2 ALL DONE ==="
