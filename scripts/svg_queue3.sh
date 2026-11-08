#!/usr/bin/env bash
# Third StreamVGGT queue: lower frame rates (every 2nd / 4th frame), INT3
# baselines on the official benchmark, and channel-outlier statistics.
#
#   tmux new -d -s tcc_queue3 "bash <clone>/scripts/svg_queue3.sh 2>&1 | tee <log>"
#
#   GPU_A  every 2nd frame: 240 frames of each Bonn sequence, exact reference
#   GPU_B  INT3 official benchmark, every 4th frame, outlier statistics
set -u
SELF=$(cd "$(dirname "$0")/.." && pwd)
C=$(dirname "$SELF")
# shellcheck disable=SC1091
source "$C/svg_env.sh"
export ROOT=${ROOT:-$C/svg_tcc} VDAQ=$SELF CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
SVG=$ROOT/StreamVGGT
RES=$SVG/eval_results/video_depth
PY=$ROOT/venv/bin/python
OUT=$C/svg_queue3
W=$SVG/ckpt/checkpoints.pth
BONN=$SVG/data/eval/bonn/rgbd_bonn_dataset
SINTEL=$SVG/data/eval/sintel/training/final
GPU_A=${GPU_A:-2}
GPU_B=${GPU_B:-0}
SCENES="balloon2 crowd2 crowd3 person_tracking2 synchronous"
FPSKV=offload,delta:int4:bin:8,delta:int4:a2g64:8,drop:2:int4,drop:4:int4,plain:a2g64,kivi:2:64
EVALKV="plain:int3 plain:a3g64"
mkdir -p "$OUT"
export STREAMVGGT_SRC=$SVG/src
cd "$SVG/src" || exit 1

say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT/STATUS.txt"; }

pack() {
  mkdir -p "$OUT/res"
  (cd "$RES" && find . -maxdepth 2 -type f \( -name '*.json' -o -name '*.txt' \) | tar cf - -T -) | tar xf - -C "$OUT/res"
  local tmp="$C/.svg_queue3.$BASHPID.tgz"
  (cd "$OUT" && find . -type f \( -name '*.json' -o -name '*.txt' -o -name '*.log' \) -size -30M \
      -not -path './*/ref/*' -not -path './*/depth/*' | tar czf "$tmp" -T -) && mv "$tmp" "$C/svg_queue3.tgz"
}

step() {
  local name=$1
  shift
  if [ -f "$OUT/$name.done" ]; then say "skip $name"; return 0; fi
  say "START $name"
  if "$@" > "$OUT/$name.log" 2>&1; then touch "$OUT/$name.done"; say "OK $name"
  else say "FAILED $name, see $OUT/$name.log"; fi
  pack
}

long() {
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u "$VDAQ/scripts/streamvggt_long.py" \
    --weights "$W" --bf16 "$@"
}

# Folders holding every n-th frame of each Bonn sequence (symbolic links).
subsampled() {
  local n=$1 s dir i f
  for s in $SCENES; do
    dir=$OUT/fps_$n/rgbd_bonn_${s}_every$n/rgb
    if [ ! -d "$dir" ]; then
      mkdir -p "$dir"
      i=0
      for f in $(ls "$BONN/rgbd_bonn_$s/rgb"/*.png | awk -v n="$n" 'NR % n == 1'); do
        ln -s "$f" "$dir/$(printf '%05d' $i).png"
        i=$((i + 1))
      done
    fi
    echo "$dir"
  done
}

preflight() {
  # shellcheck disable=SC2046
  CUDA_VISIBLE_DEVICES=$GPU_A long --frames-dirs $(subsampled 4 | head -1) --max-frames 6 --log-every 3 \
    --out "$OUT/check" --kv offload,delta:int4:bin:8,drop:2:int4 || return 1
  CUDA_VISIBLE_DEVICES=$GPU_A long --frames-dirs "$BONN/rgbd_bonn_crowd2/rgb" --max-frames 4 --log-every 2 \
    --no-compare --trace --kv offload --out "$OUT/check_outliers" || return 1
}

fps() {
  local n=$1 gpu=$2
  # shellcheck disable=SC2046
  CUDA_VISIBLE_DEVICES=$gpu long --frames-dirs $(subsampled "$n") --max-frames 240 --kv "$FPSKV" \
    --out "$OUT/every$n"
}

int3() {
  CUDA_VISIBLE_DEVICES=$GPU_B KVS=$EVALKV DATASETS="sintel bonn" bash "$VDAQ/scripts/vast_streamvggt.sh" eval
  DATASETS="sintel bonn" bash "$VDAQ/scripts/vast_streamvggt.sh" fid
  local kv data
  for kv in $EVALKV; do
    for data in sintel bonn; do
      [ -f "$RES/${data}_${kv//:/_}.done" ] || { echo "missing $data $kv"; return 1; }
    done
  done
}

outliers() {
  local s dirs=""
  for s in $SCENES; do dirs="$dirs $BONN/rgbd_bonn_$s/rgb"; done
  # shellcheck disable=SC2086,SC2046
  CUDA_VISIBLE_DEVICES=$GPU_B long --frames-dirs $dirs $(ls -d "$SINTEL"/*) --max-frames 110 \
    --no-compare --trace --kv offload --out "$OUT/outliers"
}

say "queue3 started from $(git -C "$VDAQ" log -1 --format=%h)"
step preflight preflight
if [ ! -f "$OUT/preflight.done" ]; then
  say "CHECK FAILED, nothing else will run. See $OUT/preflight.log"
  tail -15 "$OUT/preflight.log"
  exit 1
fi
say "CHECK PASSED"
( step every2 fps 2 "$GPU_A"; say "LANE A FINISHED" ) &
( step int3 int3; step every4 fps 4 "$GPU_B"; step outliers outliers; say "LANE B FINISHED" ) &
wait
pack
say "ALL FINISHED"
