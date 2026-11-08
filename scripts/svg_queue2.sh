#!/usr/bin/env bash
# Second StreamVGGT queue: frame dropping and drop + TCC, the zero-cache control,
# and a scene-cut test with adaptive keyframes.  Same layout as svg_queue.sh:
#
#   tmux new -d -s tcc_queue2 "bash <clone>/scripts/svg_queue2.sh 2>&1 | tee <log>"
#
#   GPU_B  official benchmark (Sintel + Bonn) for the new configs, then fidelity
#   GPU_A  240-frame Bonn runs for the drop configs, then the scene-cut stream
set -u
SELF=$(cd "$(dirname "$0")/.." && pwd)
C=$(dirname "$SELF")
# shellcheck disable=SC1091
source "$C/svg_env.sh"
export ROOT=${ROOT:-$C/svg_tcc} VDAQ=$SELF CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
SVG=$ROOT/StreamVGGT
RES=$SVG/eval_results/video_depth
PY=$ROOT/venv/bin/python
OUT=$C/svg_queue2
W=$SVG/ckpt/checkpoints.pth
BONN=$SVG/data/eval/bonn/rgbd_bonn_dataset
REF1=$C/svg_queue/long_bonn/ref
GPU_A=${GPU_A:-2}
GPU_B=${GPU_B:-0}
SCENES="balloon2 crowd2 crowd3 person_tracking2 synchronous"
# drop:<n>:<kind> keeps every n-th frame; ':s2' runs TCC on every 2nd frame
EVALKV="zero drop:3:int4 drop:4:int4 drop:2:a2g64 delta:int4:bin:8:s2 delta:int4:bin:4:s2 delta:int4:a2g64:8:s2"
LONGKV=drop:2:int4,drop:4:int4,delta:int4:bin:8:s2,delta:int4:bin:4:s2,delta:int4:a2g64:8:s2,zero
CUTKV=offload,delta:int4:bin:8,delta:int4:bin:8:a0.5,delta:int4:bin:8:a0.3,plain:a2g64,drop:2:int4
mkdir -p "$OUT"
export STREAMVGGT_SRC=$SVG/src
cd "$SVG/src" || exit 1

say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT/STATUS.txt"; }

pack() {
  mkdir -p "$OUT/res"
  (cd "$RES" && find . -maxdepth 2 -type f \( -name '*.json' -o -name '*.txt' \) | tar cf - -T -) | tar xf - -C "$OUT/res"
  local tmp="$C/.svg_queue2.$BASHPID.tgz"
  (cd "$OUT" && find . -type f \( -name '*.json' -o -name '*.txt' -o -name '*.log' \) -size -30M \
      -not -path './*/ref/*' -not -path './*/depth/*' | tar czf "$tmp" -T -) && mv "$tmp" "$C/svg_queue2.tgz"
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

bonn_dirs() { for s in $SCENES; do echo "$BONN/rgbd_bonn_$s/rgb"; done; }

preflight() {
  [ -f "$REF1/crowd2/pose.npy" ] || { echo "missing 240-frame reference in $REF1"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU_A long --frames-dirs "$BONN/rgbd_bonn_crowd2/rgb" --max-frames 9 --log-every 3 \
    --out "$OUT/check" --kv offload,zero,drop:3:int4,delta:int4:bin:8:s2,delta:int4:bin:8:a0.5 || return 1
}

# ---- GPU B: official benchmark -------------------------------------------
official() {
  CUDA_VISIBLE_DEVICES=$GPU_B KVS=$EVALKV DATASETS="sintel bonn" bash "$VDAQ/scripts/vast_streamvggt.sh" eval
  DATASETS="sintel bonn" bash "$VDAQ/scripts/vast_streamvggt.sh" fid
  local kv data
  for kv in $EVALKV; do
    for data in sintel bonn; do
      [ -f "$RES/${data}_${kv//:/_}.done" ] || { echo "missing $data $kv"; return 1; }
    done
  done
}

# ---- GPU A: long Bonn and scene cut ----------------------------------------
long_drop() {
  # shellcheck disable=SC2046
  CUDA_VISIBLE_DEVICES=$GPU_A long --frames-dirs $(bonn_dirs) --max-frames 240 --kv "$LONGKV" \
    --ref-dir "$REF1" --out "$OUT/long_drop"
}

scene_cut() {
  # 120 frames of crowd2, then 120 of balloon2: a hard cut at frame 121
  local dir=$OUT/cut_frames/rgbd_bonn_cut/rgb i=0 f
  rm -rf "$OUT/cut_frames"
  mkdir -p "$dir"
  for f in $(ls "$BONN/rgbd_bonn_crowd2/rgb"/*.png | head -120) $(ls "$BONN/rgbd_bonn_balloon2/rgb"/*.png | head -120); do
    ln -s "$f" "$dir/$(printf '%05d' $i).png"
    i=$((i + 1))
  done
  CUDA_VISIBLE_DEVICES=$GPU_A long --frames-dirs "$dir" --max-frames 240 --kv "$CUTKV" --out "$OUT/scene_cut"
}

say "queue2 started from $(git -C "$VDAQ" log -1 --format=%h)"
step preflight preflight
if [ ! -f "$OUT/preflight.done" ]; then
  say "CHECK FAILED, nothing else will run. See $OUT/preflight.log"
  tail -15 "$OUT/preflight.log"
  exit 1
fi
say "CHECK PASSED"
( step official official; say "LANE B FINISHED" ) &
( step long_drop long_drop; step scene_cut scene_cut; say "LANE A FINISHED" ) &
wait
pack
say "ALL FINISHED"
