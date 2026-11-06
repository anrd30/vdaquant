#!/usr/bin/env bash
# Queue 4: TCC vs the KIVI-style INT2 baseline at matched rate.  K=4 stays under 2.5 b;
# a real 4-level residual (int2-g64-asym, ~2.75 b) compares with int2-g32-asym at 3.0 b.
# Starts after the second "queue3 ALL DONE" line (the first one is from the failed 11-03 run).
#   start detached (PowerShell):
#   Start-Process 'C:\Program Files\Git\bin\bash.exe' -ArgumentList '-lc','"cd /c/Users/aniru/vdaquant && bash scripts/run_queue4.sh >> outputs/stream/queue4_stdout.log 2>&1"' -WindowStyle Hidden
set -u
cd "$(dirname "$0")/.."
O=outputs/stream
LOG=$O/queue4.log
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

log "waiting for queue3 to finish"
until [ "$(grep -c 'queue3 ALL DONE' $O/queue3.log 2>/dev/null)" -ge 2 ]; do sleep 120; done
log "=== queue4 start ==="
MATCH="fp32,int4-g16::d4:bin-g16,int4-g16::d8:int2-g64-asym"
eval_stage sintel_vits_match sintel vits 3 "$MATCH"
eval_stage sintel_vitb_match sintel vitb 3 "$MATCH"
eval_stage sintel_vitl_match sintel vitl 3 "$MATCH"
eval_stage bonn_vits_match bonn vits 3 "$MATCH"
eval_stage bonn_vitb_match bonn vitb 3 "$MATCH"
eval_stage bonn_vitl_match bonn vitl 3 "$MATCH"
log "=== queue4 ALL DONE ==="
