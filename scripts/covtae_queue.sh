#!/bin/bash
# cov-TAE v2 experiments. Saves raw predictions once per model configuration, then computes every
# TAE variant and control offline (scripts/covtae_analysis.py).
#   configurations: VDA ViT-S and ViT-L, offline 32-frame windows (as in the workshop paper),
#                   FP32 and E8 cache quantization at 8/4/3/2 bits, plus FP32 with 1-frame windows
#                   (a per-frame model: the positive control with real flicker)
#   datasets:       Sintel (23 scenes) on lane A, Bonn (all sequences, first 2000 frames) on lane B
#   disk:           about 26 GB of float16 predictions under $C/covtae/dumps
#
#   GPU_A=2 GPU_B=0 bash scripts/covtae_queue.sh
set -u
SELF=$(cd "$(dirname "$0")/.." && pwd)
C=$(dirname "$SELF")
source "$C/svg_env.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
PY=$C/svg_tcc/venv/bin/python
OUT=$C/covtae
DUMPS=$OUT/dumps
GPU_A=${GPU_A:-2}
GPU_B=${GPU_B:-0}
ENCS=${ENCS:-"vits vitl"}
mkdir -p "$OUT/logs" "$DUMPS"
say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT/STATUS.txt"; }

predict() {   # predict <dataset> <encoder> <gpu> <max-scenes>
  local ds=$1 enc=$2 gpu=$3 ns=$4
  local common="--dataset $ds --encoder $enc --eval-mode temporal --quantizer lattice_e8 --scale-bits 8
                --no-qjl --rht-seed 0 --max-samples ${MAXS:-2000} --max-scenes $ns --tae-covis-tau 0.05 --dump-preds $DUMPS"
  say "START predict $ds $enc (FP32 + 8/4/3/2 bit, window 32)"
  (cd "$SELF" && CUDA_VISIBLE_DEVICES=$gpu "$PY" -u -X utf8 scripts/run_pareto_benchmark_suite.py $common \
      --bits 8 4 3 2 --temporal-window 32 --output-dir "$OUT/live_${ds}_${enc}_w32" > "$OUT/logs/${ds}_${enc}_w32.log" 2>&1) \
      && say "OK predict $ds $enc w32" || say "FAILED predict $ds $enc w32"
  say "START predict $ds $enc (FP32, window 1)"
  (cd "$SELF" && CUDA_VISIBLE_DEVICES=$gpu "$PY" -u -X utf8 scripts/run_pareto_benchmark_suite.py $common \
      --bits --temporal-window 1 --output-dir "$OUT/live_${ds}_${enc}_w1" > "$OUT/logs/${ds}_${enc}_w1.log" 2>&1) \
      && say "OK predict $ds $enc w1" || say "FAILED predict $ds $enc w1"
}

analyse() {   # analyse <dataset> <encoder> <gpu>
  local ds=$1 enc=$2 gpu=$3
  say "START analysis $ds $enc"
  (cd "$SELF" && CUDA_VISIBLE_DEVICES=$gpu "$PY" -u -X utf8 scripts/covtae_analysis.py --dataset "$ds" --encoder "$enc" \
      --dumps "$DUMPS" --out "$OUT/analysis_${ds}_${enc}.json" > "$OUT/logs/analysis_${ds}_${enc}.log" 2>&1) \
      && say "OK analysis $ds $enc" || say "FAILED analysis $ds $enc"
}

lane() {   # lane <dataset> <gpu> <max-scenes>
  for enc in $ENCS; do predict "$1" "$enc" "$2" "$3"; done
  for enc in $ENCS; do analyse "$1" "$enc" "$2"; done
  say "LANE $1 FINISHED"
}

df -h "$OUT" | tail -1 | sed 's/^/disk: /' | tee -a "$OUT/STATUS.txt"
DATASETS=${DATASETS:-"sintel bonn"}      # e.g. DATASETS=bonn MAXS=6000 for all eight Bonn sequences
case " $DATASETS " in *" sintel "*) ( lane sintel "$GPU_A" 23 ) & ;; esac
case " $DATASETS " in *" bonn "*) ( lane bonn "$GPU_B" 8 ) & ;; esac
wait
(cd "$OUT" && tar czf "$C/covtae_results.tgz" STATUS.txt analysis_*.json logs live_*)
say "ALL FINISHED (results in $C/covtae_results.tgz; predictions stay in $DUMPS)"
