#!/bin/bash
# STAC (CVPR 2026, github.com/Rainzor/STAC @ fd7e718) on the official StreamVGGT video-depth
# benchmark, in its own venv: STAC needs torch.nn.attention.flex_attention (PyTorch >= 2.5),
# our evaluation venv stays at 2.3.1. Builds the venv now, then waits for queue 6 to finish
# (GPU_A free) -- v2: builds STAC's CUDA extensions, no --streaming for full mode, waits for KITTI -- then runs STAC full / default / recommended on Bonn and Sintel and scores the
# depth maps with our fidelity script (our venv).
#
#   bash scripts/stac_eval2.sh       (GPU_A=2)
set -u
SELF=$(cd "$(dirname "$0")/.." && pwd)
C=$(dirname "$SELF")
source "$C/svg_env.sh"
export ROOT=${ROOT:-$C/svg_tcc} VDAQ=$SELF CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
SVG=$ROOT/StreamVGGT
RES=$SVG/eval_results/video_depth
PY=$ROOT/venv/bin/python
STAC=$ROOT/STAC
SVENV=$ROOT/stac_venv
SPY=$SVENV/bin/python
OUT=$C/svg_stac2
GPU_A=${GPU_A:-2}
mkdir -p "$OUT"
say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT/STATUS.txt"; }

venv() {
  [ -x "$SPY" ] || python3 -m venv "$SVENV" || return 1
  export PIP_CONFIG_FILE=/dev/null
  "$SPY" -m pip install -q --upgrade pip
  "$SPY" -m pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 || return 1
  "$SPY" -m pip install -q numpy==1.26.4 pillow==10.3.0 accelerate safetensors huggingface_hub rich roma imageio \
    einops tqdm scipy opencv-python-headless matplotlib || return 1
  # STAC's CUDA extensions (voxel merger, sparse attention), built for the A100 only
  export CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH STAC_CUDA_ARCHS=8.0 TORCH_CUDA_ARCH_LIST=8.0
  "$SPY" -m pip install -q ninja setuptools wheel || return 1
  (cd "$STAC" && "$SPY" -m pip install --no-build-isolation ./merger-cuda ./attn-cuda) || return 1
  (cd "$STAC" && CUDA_VISIBLE_DEVICES=$GPU_A "$SPY" -c "import torch, merger_cuda, attn_cuda, eval.video_depth.launch; print('STAC import ok', torch.__version__, torch.cuda.is_available())")
}

run() {   # run <name> <data> <launch.py args...>
  local name=$1 data=$2; shift 2
  local out=$OUT/${data}_$name
  mkdir -p "$out"
  (cd "$STAC" && CUDA_VISIBLE_DEVICES=$GPU_A "$SPY" - "$out" "$data" "$@" <<'PY'
import json, os, runpy, sys, time, torch
out, data, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, os.getcwd())
sys.argv = ['eval/video_depth/launch.py', '--output_dir', out, '--eval_dataset', data, '--size', '518',
            '--model_name', 'causalvggt', '--base_model', 'streamvggt'] + extra   # 'stac' mode streams by itself
torch.cuda.reset_peak_memory_stats()
t = time.time()
runpy.run_path('eval/video_depth/launch.py', run_name='__main__')
json.dump({'args': extra, 'peak_mb': torch.cuda.max_memory_allocated() / 2**20, 'seconds': time.time() - t},
          open(os.path.join(out, 'stac_stats.json'), 'w'), indent=1)
print('peak MB', torch.cuda.max_memory_allocated() / 2**20)
PY
  ) > "$out.log" 2>&1
}

say "venv"
venv > "$OUT/venv.log" 2>&1 || { say "VENV FAILED, see $OUT/venv.log"; tail -5 "$OUT/venv.log"; exit 1; }
tail -1 "$OUT/venv.log" | tee -a "$OUT/STATUS.txt"
say "waiting for the KITTI benchmark (GPU $GPU_A)"
while ! grep -q "KITTI DONE" "$C/svg_kitti2/STATUS.txt" 2>/dev/null; do sleep 120; done
for data in bonn sintel; do
  for cfg in "full --mode full" \
             "default --mode stac --attn_backend triton --voxel_backend python" \
             "recommended --mode stac --attn_backend triton --voxel_backend python -win 4 -ck 4 -hh 2 -ret_sz 2 -ret_buf"; do
    set -- $cfg; name=$1; shift
    say "START $data $name"
    if run "$name" "$data" "$@"; then say "OK $data $name"; else say "FAILED $data $name (see $OUT/${data}_$name.log)"; fi
  done
  "$PY" "$VDAQ/scripts/streamvggt_fidelity.py" --ref "$RES/${data}_none" --runs "$OUT/${data}_"*/ \
    --out "$OUT/fidelity_vs_official_$data.json" > "$OUT/fid_official_$data.log" 2>&1
  "$PY" "$VDAQ/scripts/streamvggt_fidelity.py" --ref "$OUT/${data}_full" --runs "$OUT/${data}_"*/ \
    --out "$OUT/fidelity_vs_stacfull_$data.json" > "$OUT/fid_stacfull_$data.log" 2>&1
done
find "$OUT" -name 'frame_*.png' -delete
say "ALL FINISHED"
