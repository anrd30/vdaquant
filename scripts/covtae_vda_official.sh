#!/bin/bash
# Reproduce VDA's published ScanNet TAE with VDA's own pipeline (benchmark/: extract, infer.py offline model,
# eval_tae.py with the settings of eval_tae.sh), then score the ground truth with the same code.
#   GPU=2 bash scripts/covtae_vda_official.sh   ->  $C/vda_official.tgz
set -u
REQ_GPU=${GPU:-2}
SELF=$(cd "$(dirname "$0")/.." && pwd)
C=$(dirname "$SELF")
source "$C/svg_env.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$REQ_GPU
PY=$C/svg_tcc/venv/bin/python
VDA=$SELF/Video-Depth-Anything
OUT=$C/vda_official
BENCH=$OUT/benchmark
PRED=$OUT/pred
mkdir -p "$OUT/logs" "$PRED"
say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT/STATUS.txt"; }
say "settings: gpu=$REQ_GPU commit=$(cd "$SELF" && git log -1 --format=%h)"
[ -e "$VDA/checkpoints" ] || ln -s "$SELF/checkpoints" "$VDA/checkpoints"
"$PY" -c "import natsort" 2>/dev/null || "$PY" -m pip install -q natsort   # used by VDA's extractor

if [ ! -f "$BENCH/scannet/scannet_video_tae.json" ]; then
  say "START extract (VDA extract_scannet on benchmark_data/scannet)"
  (cd "$SELF" && "$PY" -u scripts/covtae_official_helpers.py extract "$SELF/benchmark_data/scannet" "$BENCH" \
     > "$OUT/logs/extract.log" 2>&1) && say "OK extract" || { say "FAILED extract"; exit 1; }
fi
(cd "$SELF" && "$PY" scripts/covtae_official_helpers.py swapjson "$BENCH" | tee -a "$OUT/STATUS.txt")

for enc in vits vitl; do
  say "START infer $enc (VDA offline model, benchmark/infer/infer.py)"
  (cd "$VDA" && PYTHONPATH=. "$PY" -u benchmark/infer/infer.py --infer_path "$PRED/$enc" \
     --json_file "$BENCH/scannet/scannet_video_tae.json" --datasets scannet --encoder $enc \
     > "$OUT/logs/infer_$enc.log" 2>&1) && say "OK infer $enc" || say "FAILED infer $enc"
done
(cd "$SELF" && "$PY" -u scripts/covtae_official_helpers.py gtpred "$BENCH" "$PRED/gt" > "$OUT/logs/gtpred.log" 2>&1) \
  && say "OK gtpred" || say "FAILED gtpred"

for m in vits vitl gt; do
  say "START eval_tae $m"
  (cd "$VDA" && PYTHONPATH=. "$PY" -u benchmark/eval/eval_tae.py --infer_path "$PRED/$m" --benchmark_path "$BENCH" \
     --datasets scannet --start_idx 10 --end_idx 180 --eval_scenes_num 20 --hard_crop \
     > "$OUT/logs/eval_$m.log" 2>&1) && say "OK eval_tae $m: $(grep -h 'scannet:' "$PRED/$m/results.txt" | tail -1)" \
     || say "FAILED eval_tae $m"
done
(cd "$OUT" && tar czf "$C/vda_official.tgz" STATUS.txt logs pred/*/results.txt)
say "ALL FINISHED ($C/vda_official.tgz)"
