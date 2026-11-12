#!/bin/bash
# Queue 7 (overnight 2026-11-12/13), supplement runs.
#   lane A (GPU_A): STAC (own venv, CUDA extensions already built by stac_eval2.sh) on KITTI:
#                   full / default / recommended, scored against the official uncompressed run.
#   lane B (GPU_B): crowd3 long stream (offload reference + TCC-1, TCC-2, INT2, KIVI), sized to
#                   free RAM; then TCC-2 on crowd2 again (it ran out of GPU memory at 697 frames
#                   on a shared GPU), reusing queue 5's stored reference.
#
#   bash scripts/svg_queue7.sh        (GPU_A=2 GPU_B=0)
set -u
SELF=$(cd "$(dirname "$0")/.." && pwd)
C=$(dirname "$SELF")
source "$C/svg_env.sh"
export ROOT=${ROOT:-$C/svg_tcc} VDAQ=$SELF CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
SVG=$ROOT/StreamVGGT
RES=$SVG/eval_results/video_depth
PY=$ROOT/venv/bin/python
SPY=$ROOT/stac_venv/bin/python
STAC=$ROOT/STAC
W=$SVG/ckpt/checkpoints.pth
BONN=$SVG/data/eval/bonn/rgbd_bonn_dataset
OUT=$C/svg_queue7
GPU_A=${GPU_A:-2}
GPU_B=${GPU_B:-0}
mkdir -p "$OUT"
export STREAMVGGT_SRC=$SVG/src

say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT/STATUS.txt"; }
step() {
  local name=$1; shift
  if [ -f "$OUT/$name.done" ]; then say "skip $name"; return 0; fi
  say "START $name"
  if "$@" > "$OUT/$name.log" 2>&1; then touch "$OUT/$name.done"; say "OK $name"; else say "FAILED $name"; fi
}
long() {
  (cd "$SVG/src" && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u "$VDAQ/scripts/streamvggt_long.py" \
    --weights "$W" --bf16 "$@")
}

stac_run() {   # stac_run <name> <data> <launch.py args...>
  local name=$1 data=$2; shift 2
  local out=$OUT/stac/${data}_$name
  mkdir -p "$out"
  (cd "$STAC" && CUDA_VISIBLE_DEVICES=$GPU_A "$SPY" - "$out" "$data" "$@" <<'PY'
import json, os, runpy, sys, time, torch
out, data, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, os.getcwd())
sys.argv = ['eval/video_depth/launch.py', '--output_dir', out, '--eval_dataset', data, '--size', '518',
            '--model_name', 'causalvggt', '--base_model', 'streamvggt'] + extra
torch.cuda.reset_peak_memory_stats()
t = time.time()
runpy.run_path('eval/video_depth/launch.py', run_name='__main__')
json.dump({'args': extra, 'peak_mb': torch.cuda.max_memory_allocated() / 2**20, 'seconds': time.time() - t},
          open(os.path.join(out, 'stac_stats.json'), 'w'), indent=1)
PY
  )
}
stac_kitti() {
  stac_run full kitti --mode full || return 1
  stac_run default kitti --mode stac || return 1
  stac_run recommended kitti --mode stac -win 4 -ck 4 -hh 2 -ret_sz 2 -ret_buf || return 1
  mkdir -p "$OUT/stac/runs"
  local n
  for n in full default recommended; do ln -sfn "$OUT/stac/kitti_$n/streamvggt" "$OUT/stac/runs/stac_kitti_$n"; done
  "$PY" "$VDAQ/scripts/streamvggt_fidelity.py" --ref "$RES/kitti_none" --runs "$OUT/stac/runs/stac_kitti_"* \
    --out "$OUT/stac/fid_official_kitti.json"
  "$PY" "$VDAQ/scripts/streamvggt_fidelity.py" --ref "$OUT/stac/runs/stac_kitti_full" \
    --runs "$OUT/stac/runs/stac_kitti_default" "$OUT/stac/runs/stac_kitti_recommended" --out "$OUT/stac/fid_stacfull_kitti.json"
  find "$OUT/stac" -name 'frame_*.png' -delete
}

long_crowd3() {
  local avail n
  avail=$(free -g | awk '/^Mem/{print $7}')
  n=$(( (avail - 20) * 1000 / 98 ))
  [ "$n" -gt 895 ] && n=895
  echo "available RAM ${avail} GB -> ${n} frames"
  [ "$n" -ge 500 ] || { echo "not enough RAM"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU_B long --frames-dirs "$BONN/rgbd_bonn_crowd3/rgb" --max-frames "$n" --log-every 100 \
    --kv offload,delta:int4:bin:8,delta:int4:a2g64:8,plain:a2g64,kivi:2:64 --ref offload \
    --ref-dir "$OUT/long_crowd3/ref" --out "$OUT/long_crowd3"
}
tcc2_crowd2() {
  local n
  n=$(ls "$C/svg_queue5/longn/ref/crowd2" | grep -c '^d_')
  echo "reference frames: $n"
  CUDA_VISIBLE_DEVICES=$GPU_B long --frames-dirs "$BONN/rgbd_bonn_crowd2/rgb" --max-frames "$n" --log-every 100 \
    --kv delta:int4:a2g64:8 --ref offload --ref-dir "$C/svg_queue5/longn/ref" --out "$OUT/tcc2_crowd2"
}

say "queue7 started from $(git -C "$VDAQ" log -1 --format=%h 2>/dev/null)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "$OUT/STATUS.txt"
( step stac_kitti stac_kitti; say "LANE A FINISHED" ) &
( step long_crowd3 long_crowd3; step tcc2_crowd2 tcc2_crowd2; say "LANE B FINISHED" ) &
wait
find "$OUT" -path '*/ref' -prune -o -type f \( -name '*.json' -o -name '*.log' -o -name 'STATUS.txt' \) -print \
  | tar czf "$C/q7.tgz" -T - 2>/dev/null
say "ALL FINISHED (results packed in $C/q7.tgz)"
