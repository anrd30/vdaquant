#!/bin/bash
# StreamVGGT 3D reconstruction (7-Scenes, NRGBD) with compressed caches, on the
# official mv_recon evaluator: accuracy, completeness, normal consistency.
#   sparse: official frame strides (7-Scenes every 200th, NRGBD every 500th frame)
#   dense:  every 20th frame, so each scene is a stream of ~50-80 frames
#
#   C=<code_run dir> G=<gpu> [WAIT=<file>] [KVS="spec ..."] [SETTINGS="sparse dense"] bash scripts/svg_recon.sh
set -u
: "${C:?set C}" "${G:?set G}"
SVG=$C/svg_tcc/StreamVGGT
PY=$C/svg_tcc/venv/bin/python
CODE=$C/code_run2
OUT=$C/svg_recon
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G STREAMVGGT_SRC=$SVG/src TCC_FUSED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -n "${WAIT:-}" ]; then
  echo "waiting for $WAIT"; while [ ! -e "$WAIT" ]; do sleep 60; done
fi

cd "$SVG/src" || exit 1
{ "$PY" -c "import open3d, accelerate; print('imports ok')" \
  && ls -d ../data/eval/7scenes ../data/eval/neural_rgbd \
  && nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i $G; } \
  > "$OUT/preflight.log" 2>&1 || { echo "PREFLIGHT FAILED, see $OUT/preflight.log"; exit 1; }

KVS=${KVS:-"none plain:a2g64 kivi:2:64 delta:int4:bin:8 delta:int4:a2g64:8"}
SETTINGS=${SETTINGS:-"sparse dense"}
for name in $SETTINGS; do
  case $name in sparse) a=0; b=0 ;; dense) a=20; b=20 ;; esac
  for kv in $KVS; do
    o=$OUT/${name}_${kv//:/_}
    [ -e "$o/done" ] && continue
    "$PY" -u "$CODE/scripts/streamvggt_kv_launch.py" --kv "$kv" --task mv_recon --kf $a $b -- \
      --weights ../ckpt/checkpoints.pth --output_dir "$o" --model_name StreamVGGT --size 518 \
      > "$o.log" 2>&1 && touch "$o/done" || echo "FAILED $name $kv" >> "$OUT/STATUS.txt"
    echo "finished $name $kv $(date +%H:%M)" >> "$OUT/STATUS.txt"
  done
done
echo ALL DONE >> "$OUT/STATUS.txt"
