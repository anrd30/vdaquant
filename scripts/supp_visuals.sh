#!/bin/bash
# Supplement figures: saved depth (every 40th frame, 3 scenes) for the paper's VDA formats on the paper's datasets.
# Run BEFORE the paper deadline; the figures are rendered from these files later.
#   GPU=2 bash scripts/supp_visuals.sh   ->  $C/supp_vis.tgz
set -u
REQ_GPU=${GPU:-2}
SELF=$(cd "$(dirname "$0")/.." && pwd)
C=$(dirname "$SELF")
source "$C/svg_env.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
PY=$C/svg_tcc/venv/bin/python
OUT=$C/supp_vis
mkdir -p "$OUT/logs"
say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT/STATUS.txt"; }
say "settings: gpu=$REQ_GPU commit=$(cd "$SELF" && git log -1 --format=%h)"
CFGS=fp32,int2-g16,int2-g64-asym,int4-g16::d8:bin-g16,int4-g16::d8:int2-g64-asym
for ds in sintel bonn kittiseq scannet; do
  say "START $ds"
  if (cd "$SELF" && CUDA_VISIBLE_DEVICES=$REQ_GPU "$PY" -u -X utf8 scripts/stream_eval.py --dataset $ds --encoder vits \
        --max-scenes 3 --no-tae --configs "$CFGS" --save-preds "$OUT/preds" --save-every 40 \
        --out "$OUT/$ds.json" > "$OUT/logs/$ds.log" 2>&1); then say "OK $ds"; else say "FAILED $ds"; fi
done
(cd "$OUT" && tar czf "$C/supp_vis.tgz" STATUS.txt *.json preds logs)
say "ALL FINISHED ($C/supp_vis.tgz)"
