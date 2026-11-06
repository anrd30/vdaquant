#!/usr/bin/env bash
# Queue 3: timing on an otherwise idle GPU.  Waits until queue 2 has finished.
#   start detached (PowerShell):
#   Start-Process 'C:\Program Files\Git\bin\bash.exe' -ArgumentList '-lc','"cd /c/Users/aniru/vdaquant && bash scripts/run_queue3.sh >> outputs/stream/queue3_stdout.log 2>&1"' -WindowStyle Hidden
set -u
cd "$(dirname "$0")/.."
O=outputs/stream
LOG=$O/queue3.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }

log "waiting for queue2 to finish"
until grep -q "queue2 ALL DONE" $O/queue2.log 2>/dev/null; do sleep 120; done
sleep 30
CFG="fp32,int2-g16,int2-g64-asym,int4-g16::d8:bin-g16"
if [ ! -s $O/timing_kernel.json ]; then
  log "run  timing_kernel"
  python -u -X utf8 scripts/stream_timing.py --encoders vits,vitb,vitl --configs "$CFG" --repeats 2 \
    --decode kernel --out $O/timing_kernel.json </dev/null > $O/timing_kernel.log 2>&1 && log "done timing_kernel" || log "FAILED timing_kernel"
fi
if [ ! -s $O/timing_reference.json ]; then
  log "run  timing_reference (PyTorch decoders, ViT-S)"
  python -u -X utf8 scripts/stream_timing.py --encoders vits --configs "int2-g16,int4-g16::d8:bin-g16" --repeats 2 \
    --decode per-entry --out $O/timing_reference.json </dev/null > $O/timing_reference.log 2>&1 && log "done timing_reference" || log "FAILED timing_reference"
fi
log "=== queue3 ALL DONE ==="
