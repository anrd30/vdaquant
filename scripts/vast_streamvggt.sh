#!/usr/bin/env bash
# StreamVGGT + TCC experiments on a GPU server or a rented GPU. Run inside tmux.
#
#   bash vast_streamvggt.sh setup     # clone, private venv, checkpoint, datasets
#   bash vast_streamvggt.sh smoke     # a few minutes: real model, 20 frames
#   bash vast_streamvggt.sh eval      # official video depth, all KVS x DATASETS
#   bash vast_streamvggt.sh fid       # fidelity of every run against the 'none' run
#   bash vast_streamvggt.sh oom       # frames-until-OOM (needs an otherwise idle GPU)
#   bash vast_streamvggt.sh xstream   # XStreamVGGT baselines (XKVS)
#   bash vast_streamvggt.sh all       # eval, fid, oom
#
# Environment:
#   ROOT   where StreamVGGT, the venv and results go (default $HOME/work)
#   VDAQ   this repo, if already checked out (default $ROOT/vdaquant)
#   DATA_SINTEL  folder holding training/final and training/depth
#   DATA_BONN    folder holding rgbd_bonn_<seq>/
#   DATA_KITTI   folder holding depth_selection/
#                Any of these unset: downloaded with MonST3R's scripts.
#   CUDA_VISIBLE_DEVICES  which GPU to use; set it on a shared machine.
#
# Finished runs are skipped, so any command can be restarted. The uncompressed
# cache is 16-bit here: on Ampere the evaluator runs under bf16 autocast.
set -eu
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # GPU numbers match nvidia-smi
ROOT=${ROOT:-$HOME/work}
SVG=$ROOT/StreamVGGT
VDAQ=${VDAQ:-$ROOT/vdaquant}
MON=$ROOT/monst3r
VENV=$ROOT/venv
RES=$SVG/eval_results/video_depth
# none = reference; fp16 = 16-bit store; plain:* = baselines; delta:* = TCC-1 (K=8, K=4), TCC-2
KVS=${KVS:-"none fp16 plain:int2 plain:a2g64 plain:a2g32 plain:int4 delta:int4:bin:8 delta:int4:bin:4 delta:int4:a2g64:8"}
XKVS=${XKVS:-"xstream:2048:4:4 xstream+delta:2048:int4:bin:8"}
OOMKVS=${OOMKVS:-"none plain:a2g64 delta:int4:bin:8 delta:int4:a2g64:8"}
DATASETS=${DATASETS:-"sintel bonn kitti"}

activate() { [ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"; return 0; }

setup() {
  mkdir -p "$ROOT" && cd "$ROOT"
  [ -d "$SVG" ]  || git clone --depth 1 https://github.com/wzzheng/StreamVGGT.git "$SVG"
  [ -d "$VDAQ" ] || git clone -b verify https://github.com/anrd30/vdaquant.git "$VDAQ"
  [ -d "$MON" ]  || git clone --depth 1 https://github.com/Junyi42/monst3r.git "$MON"
  [ -d "$VENV" ] || python3 -m venv "$VENV"
  activate
  pip install -q --upgrade pip
  # Exactly what the evaluator imports, at fixed versions. StreamVGGT's own
  # requirements.txt adds training and viewer packages and pins an old numpy,
  # which leaves pip searching old versions for a long time.
  echo "== installing torch"
  pip install --no-input --progress-bar off torch==2.3.1 torchvision==0.18.1
  echo "== installing evaluation packages"
  pip install --no-input --progress-bar off numpy==1.26.4 pillow==10.3.0 \
    opencv-python-headless==4.10.0.84 scipy==1.13.1 matplotlib==3.9.2 \
    accelerate==0.33.0 transformers==4.44.2 huggingface_hub==0.24.6 \
    tqdm einops roma imageio safetensors
  echo "== downloading checkpoint"
  mkdir -p "$SVG/ckpt"
  [ -f "$SVG/ckpt/checkpoints.pth" ] || python - "$SVG/ckpt" <<'PY'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download('lch01/StreamVGGT', 'checkpoints.pth', local_dir=sys.argv[1]))
PY
  # Datasets: reuse local copies where given, then MonST3R's preparation.
  mkdir -p "$MON/data"
  if [ -n "${DATA_SINTEL:-}" ]; then ln -sfn "$DATA_SINTEL" "$MON/data/sintel"; fi
  if [ -n "${DATA_KITTI:-}" ];  then ln -sfn "$DATA_KITTI"  "$MON/data/kitti"; fi
  if [ -n "${DATA_BONN:-}" ];   then
    mkdir -p "$MON/data/bonn" && ln -sfn "$DATA_BONN" "$MON/data/bonn/rgbd_bonn_dataset"
  fi
  cd "$MON/data"
  for d in sintel bonn kitti; do [ -e "$d" ] || bash "download_$d.sh"; done
  cd "$MON/datasets_preprocess"
  # A failure here only affects that dataset; the checks below report it.
  ls -d "$MON"/data/bonn/rgbd_bonn_dataset/*/rgb_110 >/dev/null 2>&1 || python prepare_bonn.py \
    || echo "WARNING: Bonn preparation failed"
  [ -d "$MON/data/kitti/depth_selection/val_selection_cropped/image_gathered" ] || python prepare_kitti.py \
    || echo "WARNING: KITTI preparation failed"
  mkdir -p "$SVG/data" && ln -sfn "$MON/data" "$SVG/data/eval"
  echo "== check these paths exist =="
  ls -d "$SVG/data/eval/sintel/training/final" "$SVG/data/eval/sintel/training/depth" \
        "$SVG/data/eval/bonn/rgbd_bonn_dataset" \
        "$SVG/data/eval/kitti/depth_selection/val_selection_cropped/image_gathered" || true
  for s in balloon2 crowd2 crowd3 person_tracking2 synchronous; do
    [ -d "$SVG/data/eval/bonn/rgbd_bonn_dataset/rgbd_bonn_$s/rgb_110" ] || echo "MISSING Bonn sequence: $s"
  done
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
}

smoke() {
  activate
  cd "$SVG/src"
  export STREAMVGGT_SRC="$SVG/src"
  frames=$(ls -d ../data/eval/sintel/training/final/* | head -1)
  python -u "$VDAQ/scripts/streamvggt_oom.py" --weights ../ckpt/checkpoints.pth --bf16 \
    --frames-dir "$frames" --kv "none,plain:a2g64,delta:int4:bin:8,delta:int4:a2g64:8" \
    --max-frames 20 --log-every 10 --out "$ROOT/smoke.json"
  python - "$ROOT/smoke.json" <<'PY'
import json, sys
for r in json.load(open(sys.argv[1])):
    last = (r.get('log') or [{}])[-1]
    print(r['kv'], 'frames', r.get('frames_reached'), 'cache_mb', last.get('cache_mb'),
          'peak_mb', last.get('peak_mb'), 'ERROR ' + r['error'][-300:] if 'error' in r else '')
PY
}

run_list() {
  activate
  mkdir -p "$RES"
  cd "$SVG/src"
  export STREAMVGGT_SRC="$SVG/src"
  for kv in $1; do
    tag=${kv//:/_}
    for data in $DATASETS; do
      out="$RES/${data}_${tag}"
      if [ -f "$out.done" ]; then echo "skip $kv $data (done)"; continue; fi
      echo "[$(date +%H:%M)] $kv $data"
      rm -rf "$out"
      python -u "$VDAQ/scripts/streamvggt_kv_launch.py" --kv "$kv" -- \
        --weights ../ckpt/checkpoints.pth --output_dir "$out" --eval_dataset "$data" --size 518 \
        > "$out.log" 2>&1 || { echo "  launch failed: $out.log"; continue; }
      python eval/video_depth/eval_depth.py --output_dir "$out" --eval_dataset "$data" \
        --align scale >> "$out.log" 2>&1 || { echo "  scoring failed: $out.log"; continue; }
      touch "$out.done"
      tail -3 "$out.log"
    done
  done
}

run_fid() {
  activate
  for data in $DATASETS; do
    [ -d "$RES/${data}_none" ] || { echo "no reference run for $data"; continue; }
    python "$VDAQ/scripts/streamvggt_fidelity.py" --ref "$RES/${data}_none" \
      --runs "$RES/${data}_"*/ --out "$RES/fidelity_${data}.json"
  done
}

run_oom() {
  activate
  cd "$SVG/src"
  export STREAMVGGT_SRC="$SVG/src"
  frames=${OOM_FRAMES:-../data/eval/bonn/rgbd_bonn_dataset/rgbd_bonn_crowd2/rgb}
  python -u "$VDAQ/scripts/streamvggt_oom.py" --weights ../ckpt/checkpoints.pth --bf16 \
    --frames-dir "$frames" --kv "$(echo $OOMKVS | tr ' ' ,)" \
    --max-frames "${MAXF:-3000}" --out "$RES/oom.json"
}

case "${1:-}" in
  setup)   setup ;;
  smoke)   smoke ;;
  eval)    run_list "$KVS" ;;
  xstream) run_list "$XKVS" ;;
  fid)     run_fid ;;
  oom)     run_oom ;;
  all)     run_list "$KVS"; run_fid; run_oom ;;
  *) echo "usage: $0 setup|smoke|eval|fid|oom|xstream|all" ;;
esac
