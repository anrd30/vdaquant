#!/usr/bin/env bash
# Overnight, resumable, one-GPU-job-at-a-time queue for the laptop.
# Every stage is skipped if its output already contains the expected number of
# results, so the script can be killed and restarted safely.
#   start detached (survives the Claude session):  see docs/overnight.md
set -u
cd "$(dirname "$0")/.."
O=outputs/stream
LOG=$O/overnight.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }

done_n() {  # json expected_results -> 0 if complete
  python -I -X utf8 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if len(d.get('results', d.get('per_layer_slot_mass', []))) >= int(sys.argv[2]) else 1)" "$1" "$2" 2>/dev/null
}

eval_stage() {  # name dataset encoder n_expected configs
  local name=$1 ds=$2 enc=$3 n=$4 cfg=$5
  if done_n $O/$name.json $n; then log "skip $name (done)"; return; fi
  log "run  $name"
  python -u -X utf8 scripts/stream_eval.py --dataset $ds --encoder $enc --configs "$cfg" \
    --out $O/$name.json > $O/$name.log 2>&1 && log "done $name" || log "FAILED $name (see $O/$name.log)"
}

probe_stage() {  # name dataset
  local name=$1 ds=$2
  if [ -f $O/$name.json ]; then log "skip $name (done)"; return; fi
  log "run  $name"
  python -u -X utf8 scripts/stream_age_probe.py --dataset $ds --frames 150 --out $O/$name.json \
    > $O/$name.log 2>&1 && log "done $name" || log "FAILED $name"
}

log "=== overnight queue start ==="
KEY="fp32,zero,int2-g16,int3-g16::d8:bin-g16,int4-g16::d8:bin-g16"
FADE=$(cat scripts/fading_configs.txt)
ABL="fp32,int4-g16::d4:bin-g16,int4-g16::d8:bin-g16,int4-g16::d16:bin-g16,int4-g16::d32:bin-g16,int8-g16::d8:bin-g16,int4-g16::d8:int2-g16,int4-g16::d8:bin-g16:none,int4-g16::d8:bin-g16:extrap2"

# 1. fidelity-to-FP32 reruns of the key configs (ViT-S)
eval_stage sintel_vits_fidelity sintel vits 5 "$KEY"
eval_stage bonn_vits_fidelity   bonn   vits 5 "$KEY"
# 2. fading memory: causal age sensitivity + equal-memory allocations
eval_stage sintel_vits_fading sintel vits 17 "$FADE"
probe_stage age_attention_sintel sintel
probe_stage age_attention_bonn   bonn
eval_stage bonn_vits_fading   bonn   vits 17 "$FADE"
# 3. delta-coding ablations: keyframe interval K, keyframe bits, residual bits, predictor
eval_stage sintel_vits_ablation sintel vits 9 "$ABL"
eval_stage bonn_vits_ablation   bonn   vits 9 "$ABL"
# 4. Bonn for the larger encoders
eval_stage bonn_vitb_key bonn vitb 4 "fp32,zero,int2-g16,int4-g16::d8:bin-g16"
eval_stage bonn_vitl_key bonn vitl 4 "fp32,zero,int2-g16,int4-g16::d8:bin-g16"
log "=== overnight queue ALL DONE ==="
