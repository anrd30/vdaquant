#!/usr/bin/env bash
# Every remaining StreamVGGT experiment, queued on two GPUs, from one command.
# Run it from a separate clone so running jobs keep their code:
#
#   tmux new -d -s tcc_queue "bash <clone>/scripts/svg_queue.sh 2>&1 | tee <log>"
#
# Expects svg_env.sh next to the clone (ROOT and dataset paths).  A short check
# runs first and stops everything if it fails.  Each lane then waits for the job
# holding its GPU, and a failed step does not stop the steps after it.  Finished
# steps are skipped on a restart.  After every step the small result files are
# packed into svg_queue.tgz next to the clone, so results can be copied at any
# time; STATUS.txt in the output folder shows progress.
#
#   GPU_A (after tcc_base)  long-video quality, then residual statistics
#   GPU_B (after tcc_lean)  timing, new baselines, authors' XStreamVGGT, second OOM
set -u
SELF=$(cd "$(dirname "$0")/.." && pwd)
C=$(dirname "$SELF")
# shellcheck disable=SC1091
source "$C/svg_env.sh"
export ROOT=${ROOT:-$C/svg_tcc} VDAQ=$SELF CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
SVG=$ROOT/StreamVGGT
RES=$SVG/eval_results/video_depth
PY=$ROOT/venv/bin/python
OUT=$C/svg_queue
W=$SVG/ckpt/checkpoints.pth
BONN=$SVG/data/eval/bonn/rgbd_bonn_dataset
SINTEL=$SVG/data/eval/sintel/training/final
GPU_A=${GPU_A:-2}
GPU_B=${GPU_B:-0}
WAIT_A=${WAIT_A:-tcc_base}
WAIT_B=${WAIT_B:-tcc_lean}
SCENES="balloon2 crowd2 crowd3 person_tracking2 synchronous"
LONGKV=offload,plain:a2g64,plain:int2,plain:int4,kivi:2:64,delta:int4:bin:8,delta:int4:bin:4,delta:int4:a2g64:8,xstream:2048:4:4
FARKV=offload,delta:int4:bin:8,delta:int4:a2g64:8,plain:a2g64,plain:int4,xstream:2048:4:4
NEWKV="kivi:2:64 kivi:2:64:w4 plain:a2g64:w4 drop:2:int4"
mkdir -p "$OUT"
export STREAMVGGT_SRC=$SVG/src
cd "$SVG/src" || exit 1

say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT/STATUS.txt"; }

pack() {
  mkdir -p "$OUT/res"
  (cd "$RES" && find . -maxdepth 2 -type f \( -name '*.json' -o -name '*.txt' \) | tar cf - -T -) | tar xf - -C "$OUT/res"
  local tmp="$C/.svg_queue.$BASHPID.tgz"
  (cd "$OUT" && find . -type f \( -name '*.json' -o -name '*.txt' -o -name '*.log' \) -size -30M \
      -not -path './*/ref/*' -not -path './*/depth/*' | tar czf "$tmp" -T -) && mv "$tmp" "$C/svg_queue.tgz"
}

step() {
  local name=$1
  shift
  if [ -f "$OUT/$name.done" ]; then say "skip $name"; return 0; fi
  say "START $name"
  if "$@" > "$OUT/$name.log" 2>&1; then
    touch "$OUT/$name.done"
    say "OK $name"
  else
    say "FAILED $name, see $OUT/$name.log"
  fi
  pack
}

wait_for() {
  say "waiting for $1"
  while tmux has-session -t "$1" 2>/dev/null; do sleep 60; done
  say "$1 is finished"
}

long() {
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u "$VDAQ/scripts/streamvggt_long.py" \
    --weights "$W" --bf16 "$@"
}

bonn_dirs() { for s in $SCENES; do echo "$BONN/rgbd_bonn_$s/rgb"; done; }

# ---- check: every code path on a few real frames, before any waiting ---------
preflight() {
  free -g
  df -h "$C"
  for s in $SCENES; do echo "$s $(ls "$BONN/rgbd_bonn_$s/rgb" | wc -l) frames"; done
  CUDA_VISIBLE_DEVICES=$GPU_A long --frames-dirs "$BONN/rgbd_bonn_crowd2/rgb" --max-frames 9 --log-every 3 \
    --out "$OUT/check" --trace \
    --kv offload,none,plain:a2g64,kivi:2:64,kivi:2:64:w4,plain:a2g64:w4,drop:2:int4,delta:int4:bin:8,delta:int4:a2g64:8,xstream:2048:4:4     || return 1
  "$PY" - "$OUT/check" <<'PY'
import json, sys
import numpy as np
d = json.load(open(sys.argv[1] + '/none.json'))['scenes']['crowd2']['series']
gap = max(d['dev'])
print('offload vs none, max deviation %:', gap)
assert gap < 0.05, 'offload does not reproduce the uncompressed model'
PY
  [ $? -eq 0 ] || return 1
  CUDA_VISIBLE_DEVICES=$GPU_A PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u "$VDAQ/scripts/streamvggt_oom.py" --weights "$W" --bf16 \
    --frames-dir "$BONN/rgbd_bonn_crowd2/rgb" --kv delta:int4:bin:8 --max-frames 3 --log-every 1 \
    --out "$OUT/check/oom.json" || return 1
  test -s "$OUT/check/oom.json"
}

# ---- GPU A -------------------------------------------------------------------
long_bonn() {
  # shellcheck disable=SC2046
  CUDA_VISIBLE_DEVICES=$GPU_A long --frames-dirs $(bonn_dirs) --max-frames 240 --kv "$LONGKV" \
    --keep crowd2 --out "$OUT/long_bonn"
}

long_far() {
  # About 0.1 GB of host memory per frame for the reference; use at most half of what is free.
  local free_gb frames
  free_gb=$(free -g | awk '/^Mem:/{print $7}')
  frames=$((free_gb * 5))
  [ "$frames" -gt 600 ] && frames=600
  echo "free ${free_gb} GB, frames ${frames}"
  [ "$frames" -ge 300 ] || { echo "not enough host memory"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU_A long --frames-dirs "$BONN/rgbd_bonn_crowd2/rgb" --max-frames "$frames" \
    --kv "$FARKV" --out "$OUT/long_far"
}

long_sintel() {
  # shellcheck disable=SC2046
  CUDA_VISIBLE_DEVICES=$GPU_A long --frames-dirs $(ls -d "$SINTEL"/*) --max-frames 60 \
    --kv offload,plain:a2g64,plain:int2,delta:int4:bin:8,delta:int4:a2g64:8 --out "$OUT/long_sintel"
}

residual_stats() {
  # shellcheck disable=SC2046
  CUDA_VISIBLE_DEVICES=$GPU_A long --frames-dirs $(bonn_dirs) $(ls -d "$SINTEL"/*) --max-frames 110 \
    --no-compare --trace --kv delta:int4:bin:8,delta:int4:a2g64:8 --out "$OUT/residual_stats"
}

# ---- GPU B -------------------------------------------------------------------
timing() {
  nvidia-smi
  for run in a b; do
    CUDA_VISIBLE_DEVICES=$GPU_B long --frames-dirs "$BONN/rgbd_bonn_crowd2/rgb" --max-frames 200 --no-compare \
      --kv none,plain:a2g64,delta:int4:bin:8,delta:int4:a2g64:8 --out "$OUT/timing_$run" || return 1
  done
}

new_baselines() {
  CUDA_VISIBLE_DEVICES=$GPU_B KVS=$NEWKV DATASETS="sintel bonn" bash "$VDAQ/scripts/vast_streamvggt.sh" eval
  DATASETS="sintel bonn" bash "$VDAQ/scripts/vast_streamvggt.sh" fid
  local kv data
  for kv in $NEWKV; do
    for data in sintel bonn; do
      [ -f "$RES/${data}_${kv//:/_}.done" ] || { echo "missing $data $kv"; return 1; }
    done
  done
}

xstream_authors() {
  set -o pipefail
  local x=$ROOT/XStreamVGGT out=$OUT/xstream_authors data budget
  [ -d "$x" ] || git clone -b prune_and_quantize --depth 1 https://github.com/ywh187/XStreamVGGT.git "$x" || return 1
  mkdir -p "$x/data" "$x/ckpt" "$out"
  ln -sfn "$SVG/data/eval" "$x/data/eval"
  ln -sfn "$W" "$x/ckpt/checkpoints.pth"
  git -C "$x" log -1 --format=%h
  for spec in "bonn 2048" "bonn 47000" "sintel 2048" "sintel 12000"; do
    read -r data budget <<< "$spec"
    echo "== $data $budget"
    (cd "$x/src" && unset STREAMVGGT_SRC && CUDA_VISIBLE_DEVICES=$GPU_B KV_PRUNE_MODE=XStreamVGGT KV_QUANT_MODE=KCVT \
       KV_POOL_SIZE=16 KV_CACHE_SIZE=$budget "$PY" eval/video_depth/launch.py --weights ../ckpt/checkpoints.pth \
       --output_dir "$out/${data}_$budget" --eval_dataset "$data" --size 518 2>&1 | tail -20) || return 1
    (cd "$x/src" && "$PY" eval/video_depth/eval_depth.py --output_dir "$out/${data}_$budget" \
       --eval_dataset "$data" --align scale 2>&1 | tail -5) || return 1
  done
  for data in bonn sintel; do
    "$PY" "$VDAQ/scripts/streamvggt_fidelity.py" --ref "$RES/${data}_none" --runs "$out/${data}_"*/ \
      --out "$out/fidelity_$data.json"
  done
  find "$out" -name 'frame_*' -delete
}

oom_second() {
  CUDA_VISIBLE_DEVICES=$GPU_B PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u "$VDAQ/scripts/streamvggt_oom.py" --weights "$W" --bf16 \
    --frames-dir "$BONN/rgbd_bonn_synchronous/rgb" \
    --kv none,delta:int4:bin:8,plain:a2g64,delta:int4:a2g64:8 --max-frames 3000 --out "$OUT/oom_second.json"
}

lane_a() {
  wait_for "$WAIT_A"
  step long_bonn long_bonn
  step long_far long_far
  step long_sintel long_sintel
  step residual_stats residual_stats
  say "LANE A FINISHED"
}

lane_b() {
  wait_for "$WAIT_B"
  step timing timing
  step new_baselines new_baselines
  step xstream_authors xstream_authors
  step oom_second oom_second
  say "LANE B FINISHED"
}

say "queue started from $(git -C "$VDAQ" log -1 --format=%h)"
step preflight preflight
if [ ! -f "$OUT/preflight.done" ]; then
  say "CHECK FAILED, nothing else will run. See $OUT/preflight.log"
  tail -15 "$OUT/preflight.log"
  exit 1
fi
say "CHECK PASSED"
lane_a &
lane_b &
wait
pack
say "ALL FINISHED"
