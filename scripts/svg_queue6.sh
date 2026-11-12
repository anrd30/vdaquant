#!/bin/bash
# Queue 6 (2026-11-12): STAC (CVPR 2026, released code) on the official StreamVGGT
# video-depth benchmark, and KITTI for our caches.
#   now (CPU/network): download + prepare KITTI (MonST3R scripts), set up STAC
#   then on GPU_A, after queue 5 lane A: STAC full / default / recommended on Sintel+Bonn,
#                                         KITTI official benchmark for our caches.
# STAC is used read-only from a pinned commit; its outputs (frame_*.npy) are scored with
# our fidelity script against the official uncompressed run and against STAC's own
# full-cache mode. Peak GPU memory is recorded around each run.
#
#   bash scripts/svg_queue6.sh        (GPU_A=2)
set -u
SELF=$(cd "$(dirname "$0")/.." && pwd)
C=$(dirname "$SELF")
source "$C/svg_env.sh"
export ROOT=${ROOT:-$C/svg_tcc} VDAQ=$SELF CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
SVG=$ROOT/StreamVGGT
RES=$SVG/eval_results/video_depth
PY=$ROOT/venv/bin/python
W=$SVG/ckpt/checkpoints.pth
OUT=$C/svg_queue6
STAC=$ROOT/STAC
MON=$ROOT/monst3r
KITTI=$SVG/data/eval/kitti/depth_selection/val_selection_cropped/image_gathered
GPU_A=${GPU_A:-2}
mkdir -p "$OUT"

say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT/STATUS.txt"; }
step() {
  local name=$1; shift
  if [ -f "$OUT/$name.done" ]; then say "skip $name"; return 0; fi
  say "START $name"
  if "$@" > "$OUT/$name.log" 2>&1; then touch "$OUT/$name.done"; say "OK $name"; else say "FAILED $name"; fi
}

kitti_data() {
  [ -d "$KITTI" ] && { echo "already there"; return 0; }
  df -h "$ROOT" | tail -1
  [ -d "$MON" ] || git clone -q --depth 1 https://github.com/Junyi42/monst3r.git "$MON" || return 1
  (cd "$MON/data" && sed 's/^wget /wget -c -q /' download_kitti.sh > dl_kitti_c.sh && bash dl_kitti_c.sh) || return 1
  (cd "$MON/datasets_preprocess" && "$PY" prepare_kitti.py) || return 1
  ln -sfn "$MON/data/kitti" "$SVG/data/eval/kitti"
  ls "$KITTI" | head -3; ls "$KITTI" | wc -l
}

stac_setup() {
  [ -d "$STAC" ] || git clone -q https://github.com/Rainzor/STAC.git "$STAC" || return 1
  git -C "$STAC" checkout -q fd7e718 && git -C "$STAC" log -1 --format=%h
  ln -sfn "$SVG/data/eval" "$STAC/data"
  mkdir -p "$STAC/ckpt/streamvggt" && ln -sfn "$W" "$STAC/ckpt/streamvggt/model.pth"
  PIP_CONFIG_FILE=/dev/null "$PY" -m pip install -q rich roma imageio
  cd "$STAC" && "$PY" -c "import eval.video_depth.launch" 2>&1 | tail -3; true
}

# run STAC's own launcher with peak-memory recording; args after the name go to launch.py
stac_run() {
  local name=$1 data=$2; shift 2
  local out=$OUT/stac/${data}_$name
  mkdir -p "$out"
  (cd "$STAC" && CUDA_VISIBLE_DEVICES=$GPU_A "$PY" - "$out" "$data" "$@" <<'PY'
import json, os, runpy, sys, time, torch
out, data, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, os.getcwd())
sys.argv = ['eval/video_depth/launch.py', '--output_dir', out, '--eval_dataset', data, '--size', '518',
            '--model_name', 'causalvggt', '--base_model', 'streamvggt', '--streaming'] + extra
torch.cuda.reset_peak_memory_stats()
t = time.time()
runpy.run_path('eval/video_depth/launch.py', run_name='__main__')
json.dump({'args': extra, 'peak_mb': torch.cuda.max_memory_allocated() / 2**20, 'seconds': time.time() - t},
          open(os.path.join(out, 'stac_stats.json'), 'w'), indent=1)
print('peak MB', torch.cuda.max_memory_allocated() / 2**20)
PY
  )
}

stac_eval() {
  local data
  for data in bonn sintel; do
    stac_run full "$data" --mode full || return 1
    stac_run default "$data" --mode stac --attn_backend triton --voxel_backend python || return 1
    stac_run recommended "$data" --mode stac --attn_backend triton --voxel_backend python \
      -win 4 -ck 4 -hh 2 -ret_sz 2 -ret_buf || return 1
    "$PY" "$VDAQ/scripts/streamvggt_fidelity.py" --ref "$RES/${data}_none" --runs "$OUT/stac/${data}_"*/ \
      --out "$OUT/stac/fidelity_vs_official_$data.json" || true
    "$PY" "$VDAQ/scripts/streamvggt_fidelity.py" --ref "$OUT/stac/${data}_full" --runs "$OUT/stac/${data}_"*/ \
      --out "$OUT/stac/fidelity_vs_stacfull_$data.json" || true
  done
  find "$OUT/stac" -name 'frame_*.png' -delete
}

kitti_eval() {
  [ -d "$KITTI" ] || { echo "KITTI data missing"; return 1; }
  cd "$SVG/src" && export STREAMVGGT_SRC=$SVG/src
  CUDA_VISIBLE_DEVICES=$GPU_A KVS="none plain:a2g64 kivi:2:64 delta:int4:bin:8 delta:int4:a2g64:8 xstream:2048:4:4" \
    DATASETS="kitti" bash "$VDAQ/scripts/vast_streamvggt.sh" eval
  DATASETS="kitti" bash "$VDAQ/scripts/vast_streamvggt.sh" fid
}

say "queue6 started from $(git -C "$VDAQ" log -1 --format=%h 2>/dev/null)"
( step kitti_data kitti_data ) &
step stac_setup stac_setup
wait
say "waiting for queue 5 lane A (GPU $GPU_A)"
while ! grep -q "LANE A FINISHED" "$C/svg_queue5/STATUS.txt" 2>/dev/null; do sleep 120; done
step stac_eval stac_eval
step kitti_eval kitti_eval
say "ALL FINISHED"
