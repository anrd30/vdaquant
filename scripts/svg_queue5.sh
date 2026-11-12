#!/bin/bash
# Queue 5: experiments requested by reviews (2026-11-12).
#   lane A (GPU_A): XStreamVGGT at larger budgets (equal-memory points) and XStreamVGGT + TCC
#                   (complementarity), official benchmark on Sintel + Bonn; then residual
#                   statistics (contraction) for TCC-1 and TCC-2.
#   lane B (GPU_B): KITTI official video depth (new dataset, fast driving motion), if the data
#                   is there; then a long crowd2 stream (up to 895 frames, sized to free RAM).
#
#   bash scripts/svg_queue5.sh          (GPU_A=2 GPU_B=0 by default)
set -u
SELF=$(cd "$(dirname "$0")/.." && pwd)
C=$(dirname "$SELF")
source "$C/svg_env.sh"
export ROOT=${ROOT:-$C/svg_tcc} VDAQ=$SELF CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
SVG=$ROOT/StreamVGGT
PY=$ROOT/venv/bin/python
OUT=$C/svg_queue5
W=$SVG/ckpt/checkpoints.pth
BONN=$SVG/data/eval/bonn/rgbd_bonn_dataset
KITTI=$SVG/data/eval/kitti/depth_selection/val_selection_cropped/image_gathered
GPU_A=${GPU_A:-2}
GPU_B=${GPU_B:-0}
mkdir -p "$OUT"
export STREAMVGGT_SRC=$SVG/src
cd "$SVG/src" || exit 1

say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT/STATUS.txt"; }
step() {
  local name=$1; shift
  if [ -f "$OUT/$name.done" ]; then say "skip $name"; return 0; fi
  say "START $name"
  if "$@" > "$OUT/$name.log" 2>&1; then touch "$OUT/$name.done"; say "OK $name"; else say "FAILED $name"; fi
}
long() {
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u "$VDAQ/scripts/streamvggt_long.py" --weights "$W" --bf16 "$@"
}

preflight() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  free -g | head -2
  echo "kitti: $(ls "$KITTI" 2>/dev/null | head -3 | tr '\n' ' ')"
  CUDA_VISIBLE_DEVICES=$GPU_A long --frames-dirs "$BONN/rgbd_bonn_crowd2/rgb" --max-frames 6 --log-every 3 \
    --out "$OUT/check" --no-compare --kv xstream+delta:2048:int4:bin:8,xstream:8192:4:4 || return 1
}
xs_official() {
  CUDA_VISIBLE_DEVICES=$GPU_A KVS="xstream:8192:4:4 xstream:24576:4:4 xstream+delta:2048:int4:bin:8 xstream+delta:8192:int4:bin:8" \
    DATASETS="sintel bonn" bash "$VDAQ/scripts/vast_streamvggt.sh" eval
  DATASETS="sintel bonn" bash "$VDAQ/scripts/vast_streamvggt.sh" fid
}
contraction() {
  CUDA_VISIBLE_DEVICES=$GPU_A long --frames-dirs "$BONN/rgbd_bonn_crowd2/rgb" "$BONN/rgbd_bonn_balloon2/rgb" \
    --max-frames 240 --no-compare --trace --kv delta:int4:bin:8,delta:int4:a2g64:8 --out "$OUT/contraction"
}
kitti() {
  [ -d "$KITTI" ] || { echo "KITTI data missing at $KITTI"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU_B KVS="none plain:a2g64 kivi:2:64 delta:int4:bin:8 delta:int4:a2g64:8 xstream:2048:4:4" \
    DATASETS="kitti" bash "$VDAQ/scripts/vast_streamvggt.sh" eval
  DATASETS="kitti" bash "$VDAQ/scripts/vast_streamvggt.sh" fid
}
longn() {
  # the host-memory reference holds ~98 MB per frame: size the run to the free RAM (20 GB margin)
  local avail n
  avail=$(free -g | awk '/^Mem/{print $7}')
  n=$(( (avail - 20) * 1000 / 98 ))
  [ "$n" -gt 895 ] && n=895
  echo "available RAM ${avail} GB -> ${n} frames"
  [ "$n" -ge 500 ] || { echo "not enough RAM for a run longer than the existing 495 frames"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU_B long --frames-dirs "$BONN/rgbd_bonn_crowd2/rgb" --max-frames "$n" --log-every 100 \
    --kv offload,delta:int4:bin:8,delta:int4:a2g64:8,plain:a2g64,kivi:2:64 --ref offload \
    --ref-dir "$OUT/longn/ref" --out "$OUT/longn"
}

say "queue5 started from $(git -C "$VDAQ" log -1 --format=%h 2>/dev/null)"
step preflight preflight
[ -f "$OUT/preflight.done" ] || { say "CHECK FAILED, see $OUT/preflight.log"; tail -15 "$OUT/preflight.log"; exit 1; }
say "CHECK PASSED"
( step xs_official xs_official; step contraction contraction; say "LANE A FINISHED" ) &
( step kitti kitti; step longn longn; say "LANE B FINISHED" ) &
wait
say "ALL FINISHED"
