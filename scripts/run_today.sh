#!/usr/bin/env bash
# Sintel ViT-B/L with fidelity metrics and the INT4-keyframe default, after the
# resumed overnight queue finishes. Resumable like run_overnight.sh.
set -u
cd "$(dirname "$0")/.."
O=outputs/stream
LOG=$O/overnight.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
# wait until the overnight queue has logged ALL DONE after its 10:00 restart
until [ "$(grep -c 'ALL DONE' $LOG)" -ge 2 ]; do sleep 60; done
done_n() { python -I -X utf8 -c "import json,sys; sys.exit(0 if len(json.load(open(sys.argv[1]))['results'])>=int(sys.argv[2]) else 1)" "$1" "$2" 2>/dev/null; }
CFG="fp32,zero,int2-g16,int4-g16::d8:bin-g16"
for enc in vitb vitl; do
  n=sintel_${enc}_fidelity
  if done_n $O/$n.json 4; then log "skip $n (done)"; continue; fi
  log "run  $n"
  python -u -X utf8 scripts/stream_eval.py --dataset sintel --encoder $enc --configs "$CFG" \
    --out $O/$n.json > $O/$n.log 2>&1 && log "done $n" || log "FAILED $n"
done
log "=== today queue ALL DONE ==="
