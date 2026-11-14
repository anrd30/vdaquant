#!/bin/bash
# ScanNet v2 for the TAE / cov-TAE evaluation: downloads the .sens files of the first N test scenes
# (VDA's eval_tae.sh scores the first 20: scene0707_00 ... scene0726_00) and exports their first FRAMES
# frames to benchmark_data/scannet/<scene>/ with scripts/scannet_sens_export.py.
# Needs ScanNet's download-scannet.py (sent after the Terms of Use are signed; not part of this repo).
# The download script asks to confirm the Terms of Use once per call; this script answers it.
#   DL=/path/to/download-scannet.py N=20 bash scripts/scannet_prepare.sh
set -u
REQ_DL=${DL:?set DL=/path/to/download-scannet.py}
REQ_N=${N:-20}
REQ_FRAMES=${FRAMES:-510}
SELF=$(cd "$(dirname "$0")/.." && pwd)
C=$(dirname "$SELF")
source "$C/svg_env.sh"
PY=$C/svg_tcc/venv/bin/python
RAW=$C/scannet_sens
OUT=$SELF/benchmark_data/scannet
mkdir -p "$RAW" "$OUT"
LOG=$C/covtae/STATUS_scannet.txt
mkdir -p "$(dirname "$LOG")"
say() { echo "[$(date +%H:%M)] $*" | tee -a "$LOG"; }
say "settings: n=$REQ_N frames=$REQ_FRAMES raw=$RAW out=$OUT"

ok=0
for k in $(seq 0 $((REQ_N - 1))); do
  id=$(printf 'scene%04d_00' $((707 + k)))
  if [ "$(ls "$OUT/$id/depth" 2>/dev/null | wc -l)" -ge "$REQ_FRAMES" ]; then
    say "SKIP $id (already exported)"; ok=$((ok + 1)); continue
  fi
  sens=$(find "$RAW" -name "$id.sens" 2>/dev/null | head -1)
  if [ -z "$sens" ]; then
    say "DOWNLOAD $id"
    printf '\n\n\n' | python3 "$REQ_DL" -o "$RAW" --id "$id" --type .sens > "$RAW/download_$id.log" 2>&1
    sens=$(find "$RAW" -name "$id.sens" 2>/dev/null | head -1)
    [ -z "$sens" ] && { say "FAILED download $id (see $RAW/download_$id.log)"; continue; }
  fi
  if "$PY" -u "$SELF/scripts/scannet_sens_export.py" --sens "$sens" --out "$OUT/$id" --max-frames "$REQ_FRAMES" \
       >> "$LOG" 2>&1; then ok=$((ok + 1)); else say "FAILED export $id"; fi
done
du -sh "$RAW" "$OUT" | sed 's/^/disk: /' | tee -a "$LOG"
say "ALL FINISHED: $ok of $REQ_N scenes in $OUT"
