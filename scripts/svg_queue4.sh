#!/usr/bin/env bash
# Open-loop control: the same TCC settings, but residuals taken against the true
# previous frame instead of the decoder's reconstruction (':o').  Shows end to end
# whether closed-loop coding matters.
#
#   tmux new -d -s tcc_queue4 "bash <clone>/scripts/svg_queue4.sh 2>&1 | tee <log>"
#
#   GPU_B  official benchmark (Sintel + Bonn), then fidelity
#   GPU_A  240-frame Bonn and 495-frame crowd2 against the existing references
set -u
SELF=$(cd "$(dirname "$0")/.." && pwd)
C=$(dirname "$SELF")
# shellcheck disable=SC1091
source "$C/svg_env.sh"
export ROOT=${ROOT:-$C/svg_tcc} VDAQ=$SELF CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
SVG=$ROOT/StreamVGGT
RES=$SVG/eval_results/video_depth
PY=$ROOT/venv/bin/python
OUT=$C/svg_queue4
W=$SVG/ckpt/checkpoints.pth
BONN=$SVG/data/eval/bonn/rgbd_bonn_dataset
GPU_A=${GPU_A:-2}
GPU_B=${GPU_B:-0}
KV="delta:int4:bin:8:o delta:int4:a2g64:8:o"
mkdir -p "$OUT"
export STREAMVGGT_SRC=$SVG/src
cd "$SVG/src" || exit 1

say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT/STATUS.txt"; }
pack() {
  mkdir -p "$OUT/res"
  (cd "$RES" && find . -maxdepth 2 -type f \( -name '*.json' -o -name '*.txt' \) | tar cf - -T -) | tar xf - -C "$OUT/res"
  (cd "$OUT" && find . -type f \( -name '*.json' -o -name '*.txt' -o -name '*.log' \) -size -30M \
      -not -path './*/ref/*' | tar czf "$C/.q4.$BASHPID.tgz" -T -) && mv "$C/.q4.$BASHPID.tgz" "$C/svg_queue4.tgz"
}
step() {
  local name=$1; shift
  if [ -f "$OUT/$name.done" ]; then say "skip $name"; return 0; fi
  say "START $name"
  if "$@" > "$OUT/$name.log" 2>&1; then touch "$OUT/$name.done"; say "OK $name"; else say "FAILED $name"; fi
  pack
}
long() {
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u "$VDAQ/scripts/streamvggt_long.py" --weights "$W" --bf16 "$@"
}

preflight() {
  CUDA_VISIBLE_DEVICES=$GPU_A long --frames-dirs "$BONN/rgbd_bonn_crowd2/rgb" --max-frames 10 --log-every 5 \
    --out "$OUT/check" --kv offload,delta:int4:bin:8,delta:int4:bin:8:o || return 1
}
official() {
  CUDA_VISIBLE_DEVICES=$GPU_B KVS=$KV DATASETS="sintel bonn" bash "$VDAQ/scripts/vast_streamvggt.sh" eval
  DATASETS="sintel bonn" bash "$VDAQ/scripts/vast_streamvggt.sh" fid
}
long240() {
  local s dirs=""
  for s in balloon2 crowd2 crowd3 person_tracking2 synchronous; do dirs="$dirs $BONN/rgbd_bonn_$s/rgb"; done
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$GPU_A long --frames-dirs $dirs --max-frames 240 --kv "${KV// /,}" \
    --ref-dir "$C/svg_queue/long_bonn/ref" --out "$OUT/long240"
}
long495() {
  CUDA_VISIBLE_DEVICES=$GPU_A long --frames-dirs "$BONN/rgbd_bonn_crowd2/rgb" --max-frames 495 --kv "${KV// /,}" \
    --ref-dir "$C/svg_queue/long_far/ref" --out "$OUT/long495"
}

say "queue4 started from $(git -C "$VDAQ" log -1 --format=%h)"
step preflight preflight
[ -f "$OUT/preflight.done" ] || { say "CHECK FAILED, see $OUT/preflight.log"; tail -15 "$OUT/preflight.log"; exit 1; }
say "CHECK PASSED"
( step official official; say "LANE B FINISHED" ) &
( step long240 long240; step long495 long495; say "LANE A FINISHED" ) &
wait
pack
say "ALL FINISHED"
