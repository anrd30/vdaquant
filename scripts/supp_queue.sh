#!/bin/bash
# CVPR supplement: every experiment for the reviewer-style questions, in one queue.
#   lane A (GPU_A): frame-stride sweep (Sintel, Bonn); abrupt scene cuts (ScanNet splices, per-frame deviation)
#   lane B (GPU_B): rate-fidelity of fixed K vs adaptive keyframes; fitted temporal predictor (calibrated on another
#                   dataset); error after each temporal module; cache redundancy per layer/dataset/stride
#   then, alone on GPU_A: decoder latency vs keyframe interval K
# Results: $C/supp/*.json, packed into $C/supp.tgz.   GPU_A=2 GPU_B=0 bash scripts/supp_queue.sh
set -u
REQ_GPU_A=${GPU_A:-2}
REQ_GPU_B=${GPU_B:-0}
REQ_ENCS=${ENCS:-"vits vitl"}
SELF=$(cd "$(dirname "$0")/.." && pwd)
C=$(dirname "$SELF")
source "$C/svg_env.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
PY=$C/svg_tcc/venv/bin/python
OUT=$C/supp
mkdir -p "$OUT/logs"
say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT/STATUS.txt"; }
say "settings: gpu_a=$REQ_GPU_A gpu_b=$REQ_GPU_B encoders=[$REQ_ENCS] commit=$(cd "$SELF" && git log -1 --format=%h)"

TCC1=int4-g16::d8:bin-g16
TCC1K4=int4-g16::d4:bin-g16
TCC2=int4-g16::d8:int2-g64-asym
BASE="fp32,int2-g64-asym,int4-g16@win16,$TCC1,$TCC1K4,$TCC2"

ev() {   # ev <gpu> <tag> <stream_eval args...>
  local gpu=$1 tag=$2; shift 2
  say "START $tag"
  if (cd "$SELF" && CUDA_VISIBLE_DEVICES=$gpu "$PY" -u -X utf8 scripts/stream_eval.py --no-tae "$@" \
        --out "$OUT/$tag.json" > "$OUT/logs/$tag.log" 2>&1); then say "OK $tag"; else say "FAILED $tag"; fi
}

lane_a() {
  local g=$REQ_GPU_A enc st
  for enc in $REQ_ENCS; do
    for st in 2 4; do ev $g "stride_sintel_${enc}_s$st" --dataset sintel --encoder $enc --stride $st --configs "$BASE"; done
    for st in 2 4 8; do ev $g "stride_bonn_${enc}_s$st" --dataset bonn --encoder $enc --stride $st --configs "$BASE"; done
    ev $g "splice_scannet_$enc" --dataset scannet --encoder $enc --splice 85 --per-frame \
       --configs "fp32,int2-g64-asym,$TCC1,$TCC1:a0.35,$TCC1:a0.5,$TCC2"
  done
  say "LANE A FINISHED"
}

lane_b() {
  local g=$REQ_GPU_B enc ds cal
  local RD="fp32"
  for k in 2 4 8 16 32; do RD="$RD,int4-g16::d$k:bin-g16"; done
  for a in 0.25 0.35 0.5 0.75; do RD="$RD,int4-g16::d32:bin-g16:a$a"; done
  for ds in sintel bonn scannet; do ev $g "rate_${ds}_vits" --dataset $ds --encoder vits --configs "$RD"; done
  for enc in $REQ_ENCS; do
    for ds in sintel bonn scannet; do
      cal=sintel; [ "$ds" = sintel ] && cal=bonn
      ev $g "linfit_${ds}_$enc" --dataset $ds --encoder $enc --linfit-calib $cal --linfit-frames 150 \
         --configs "fp32,$TCC1,$TCC1:linfit,$TCC1:extrap2,$TCC2,$TCC2:linfit"
    done
    for ds in sintel bonn; do
      ev $g "moderr_${ds}_$enc" --dataset $ds --encoder $enc --module-err 4 \
         --configs "fp32,int2-g16,int2-g64-asym,$TCC1,$TCC2"
    done
  done
  say "START redundancy"
  if (cd "$SELF" && CUDA_VISIBLE_DEVICES=$g "$PY" -u -X utf8 scripts/supp_redundancy.py --datasets sintel bonn scannet \
        --encoders $REQ_ENCS --strides 1 2 4 8 --out "$OUT/redundancy.json" > "$OUT/logs/redundancy.log" 2>&1); then
    say "OK redundancy"; else say "FAILED redundancy"; fi
  say "LANE B FINISHED"
}

n=$(ls -d "$SELF/benchmark_data/scannet"/scene* 2>/dev/null | wc -l)
say "scannet scenes: $n"
lane_a &
lane_b &
wait
say "START timing vs K (alone on GPU $REQ_GPU_A)"
CFG="fp32,int2-g64-asym"
for k in 2 4 8 16 32; do CFG="$CFG,int4-g16::d$k:bin-g16"; done
if (cd "$SELF" && CUDA_VISIBLE_DEVICES=$REQ_GPU_A "$PY" -u -X utf8 scripts/stream_timing.py --encoders vits,vitl \
      --configs "$CFG" --repeats 3 --out "$OUT/timing_k.json" > "$OUT/logs/timing_k.log" 2>&1); then
  say "OK timing"; else say "FAILED timing"; fi
(cd "$OUT" && tar czf "$C/supp.tgz" STATUS.txt *.json logs)
say "ALL FINISHED (results in $C/supp.tgz)"
