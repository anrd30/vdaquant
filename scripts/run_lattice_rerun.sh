#!/usr/bin/env bash
# =============================================================================
# F28 RE-RUN — lattice experiments only, with the decoder range fix.
#
# WHY. Both lattice quantizers reserved part of the representable range before
# decoding so a greedy parity flip could never go out of bounds, while the
# grouped-scalar baseline used the full range. That handicapped E8 by ~0.87 dB
# and D4 by ~2.02 dB at the 3-bit operating point (ledger F28). Every published
# lattice number was produced by a handicapped quantizer, so the E8-vs-scalar
# and D4-vs-E8 tables have to be re-measured. scalar_g8 is UNAFFECTED and is not
# re-run here -- the existing scalar finals remain the comparison point.
#
# Results go to outputs/rerun_f28 so outputs/finals is preserved for a
# before/after diff. Do NOT delete the old results.
#
# Usage (fire and forget):
#   cd vdaquant && git pull && nohup bash scripts/run_lattice_rerun.sh > rerun.log 2>&1 &
#   tail -f rerun.log
#
# When done:
#   cd vdaquant && zip -r rerun_f28_results.zip outputs/rerun_f28 && (send the zip)
#
# Rough time: ~2-4 h on one modern GPU. Fully resumable -- a disconnect just
# means re-running the same command. Delete a result dir to force a re-run.
#
# Conventions identical to scripts/run_finals.sh so the comparison is
# like-for-like: --no-qjl everywhere (F15), --scale-bits 8 (F18), same splits,
# same sample counts, same rotation seeds.
# =============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-$(command -v python || command -v python3)}"
if [ -z "$PY" ]; then
  echo "ERROR: no python interpreter found (tried 'python' and 'python3'). Set PY=/path/to/python and re-run."
  exit 1
fi
echo "Using python: $PY"
SUITE="$PY scripts/run_pareto_benchmark_suite.py"
OUT="outputs/rerun_f28"
mkdir -p "$OUT"
PASS=0; SKIP=0; FAIL=0; FAILED=""

run() {  # run <name> <args...>
  local name="$1"; shift
  local d="$OUT/$name"
  if [ -f "$d/pareto_benchmark_results.json" ]; then
    echo "[skip] $name"; SKIP=$((SKIP+1)); return 0
  fi
  echo ""; echo "════════ [run] $name  $(date '+%H:%M:%S') ════════"; echo "   $*"
  local t0=$SECONDS
  if $SUITE "$@" --output-dir "$d" 2>&1 | tee "$d.log" | grep -E "delta1|TAE|covis|skipped|Surgery|Error|Traceback"; then :; fi
  if [ -f "$d/pareto_benchmark_results.json" ]; then
    echo "[done] $name in $(( (SECONDS-t0)/60 )) min"; PASS=$((PASS+1))
  else
    echo "[FAIL] $name — see $d.log"; FAIL=$((FAIL+1)); FAILED="$FAILED $name"
  fi
}

echo "########## STAGE 0: F27 identity control (cheap, unblocks scale ladder) ##########"
# Full surgery with a no-op quantizer: differs from FP32 ONLY by the attention
# implementation. Decides whether the vitb/vitl anomaly is a surgery-fidelity
# bug or a real bit-width effect. ~20 min. Independent of F28 but still open.
for enc in vits vitl; do
  run "identity_${enc}_nyu" --dataset nyuv2 --eval-mode groundtruth --encoder "$enc" \
      --quantizer identity --bits 8 --no-qjl --rht-seed 0 --max-samples 200
done

echo "########## STAGE 1: lattice finals, full splits (tab:fair, tab:d4) ##########"
# These two tables are the ones F28 invalidates. scalar_g8 is NOT re-run.
run e8_nyu   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
    --scale-bits 8 --bits 8 4 3 2 --no-qjl --rht-seed 0 --max-samples 654
run e8_kitti --dataset kitti --eval-mode groundtruth --quantizer lattice_e8 \
    --scale-bits 8 --bits 8 4 3 2 --no-qjl --rht-seed 0 --max-samples 1000
run d4_nyu   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_d4 \
    --scale-bits 8 --bits 4 3 2 --no-qjl --rht-seed 0 --max-samples 654
run d4_kitti --dataset kitti --eval-mode groundtruth --quantizer lattice_d4 \
    --scale-bits 8 --bits 4 3 2 --no-qjl --rht-seed 0 --max-samples 1000

echo "########## STAGE 2: seed sweeps (seed-as-random-effect analysis) ##########"
# The headline CIs treat the rotation seed as a random effect, so every seed
# that fed that analysis has to be re-measured on the fixed decoder.
for s in 1 2; do
  run "e8_nyu_seed$s"   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits 3 --no-qjl --rht-seed "$s" --max-samples 654
  run "e8_kitti_seed$s" --dataset kitti --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits 3 --no-qjl --rht-seed "$s" --max-samples 1000
done

echo "########## STAGE 3: Sintel temporal with co-visibility TAE ##########"
# The TAE/co-visibility finding itself is quantizer-independent, but the E8 rows
# in that table were produced by the handicapped decoder.
run e8_sintel_covis --dataset sintel --eval-mode temporal --quantizer lattice_e8 \
    --scale-bits 8 --bits 8 4 3 2 --no-qjl --rht-seed 0 --temporal-window 32 \
    --max-samples 2000 --max-scenes 23 --tae-covis-tau 0.05

echo "########## STAGE 4: scale ladder, lattice rows only ##########"
# Only meaningful if STAGE 0 comes back clean. Run it anyway -- it is cheap and
# banking it now avoids a second round trip.
for enc in vitb vitl; do
  for ds in nyuv2 kitti; do
    run "ladder_${enc}_e8_${ds}" --dataset "$ds" --eval-mode groundtruth --encoder "$enc" \
        --quantizer lattice_e8 --scale-bits 8 --bits 4 3 --no-qjl --rht-seed 0 --max-samples 200
  done
done
run ladder_vitl_sintel --dataset sintel --eval-mode temporal --encoder vitl \
    --quantizer lattice_e8 --scale-bits 8 --bits 4 3 2 --no-qjl --rht-seed 0 \
    --temporal-window 32 --max-samples 2000 --max-scenes 10 --tae-covis-tau 0.05

echo "########## STAGE 5: blur degradation (TAE-audit generalisation) ##########"
# Reviewer objection to close off: "maybe the TAE failure is a quantisation
# artefact." Answer with a degradation that has NOTHING to do with quantisation
# -- Gaussian blur on the FP32 predictions themselves -- and show unmasked TAE
# still falls while masked TAE correctly rises. One FP32 pass + cheap CPU blurs
# per sigma. ~20 min on one GPU (dominated by the FP32 pass).
BLUR_DIR="$OUT/blur_degradation"
if [ -f "$BLUR_DIR/blur_degradation_results.json" ]; then
  echo "[skip] blur_degradation"; SKIP=$((SKIP+1))
else
  echo ""; echo "════════ [run] blur_degradation  $(date '+%H:%M:%S') ════════"
  t0=$SECONDS
  mkdir -p "$BLUR_DIR"
  if $PY scripts/run_blur_degradation.py --dataset sintel \
       --sigmas 0 0.5 1 2 4 8 --temporal-window 32 --max-scenes 23 \
       --tae-covis-tau 0.05 --output-dir "$BLUR_DIR" 2>&1 \
       | tee "$BLUR_DIR.log" | grep -E "sigma|TAE|covis|delta1|Error|Traceback"; then :; fi
  if [ -f "$BLUR_DIR/blur_degradation_results.json" ]; then
    echo "[done] blur_degradation in $(( (SECONDS-t0)/60 )) min"; PASS=$((PASS+1))
  else
    echo "[FAIL] blur_degradation — see $BLUR_DIR.log"; FAIL=$((FAIL+1))
    FAILED="$FAILED blur_degradation"
  fi
fi

echo "########## STAGE 5b: P1 group-size sweep (lattice-direction decisive) ##########"
# HYPOTHESIS (synthetic, ledger F28/F29): E8 ties grouped scalar at g=8 and the
# gain grows monotonically with g (+0.32/+0.49/+0.57 dB at g=16/32/64 on
# Gaussian input). NEVER tested on real KV cache. If the trend holds, the
# lattice claim is alive; if E8 tracks scalar_g8 at all g or gets WORSE as g
# grows, the lattice direction is closed outright.
# NOTE: coarser g LOWERS effective bits (3b: g=8 -> 4.00, g=64 -> 3.125), so
# E8 must be read against scalar_g8 AT THE SAME g, not against g=8.
for g in 16 32 64; do
  run "p1_e8_nyu_g$g"   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
      --group-size "$g" --scale-bits 8 --bits 4 3 --no-qjl --rht-seed 0 --max-samples 654
  run "p1_sg_nyu_g$g"   --dataset nyuv2 --eval-mode groundtruth --quantizer scalar_g8 \
      --group-size "$g" --scale-bits 8 --bits 4 3 --no-qjl --rht-seed 0 --max-samples 654
  run "p1_e8_kitti_g$g" --dataset kitti --eval-mode groundtruth --quantizer lattice_e8 \
      --group-size "$g" --scale-bits 8 --bits 4 3 --no-qjl --rht-seed 0 --max-samples 1000
  run "p1_sg_kitti_g$g" --dataset kitti --eval-mode groundtruth --quantizer scalar_g8 \
      --group-size "$g" --scale-bits 8 --bits 4 3 --no-qjl --rht-seed 0 --max-samples 1000
done

echo "########## STAGE 6: measured GPU peak (analytic table cross-check) ##########"
# The pareto suite already records fps + peak-memory per config in Stages 1-4,
# so the hardware-benchmark table in the paper is assembled from those JSONs.
# This extra call is the one measured FP16-vs-quant-sim peak that report_kv_memory
# prints for transparency about the simulation, on both encoders.
HW_DIR="$OUT/hardware"
if [ -f "$HW_DIR/kv_memory_measured.log" ]; then
  echo "[skip] kv_memory_measured"; SKIP=$((SKIP+1))
else
  echo ""; echo "════════ [run] kv_memory_measured  $(date '+%H:%M:%S') ════════"
  t0=$SECONDS
  mkdir -p "$HW_DIR"
  if $PY scripts/report_kv_memory.py --measure 2>&1 | tee "$HW_DIR/kv_memory_measured.log" \
       | grep -E "encoder|window|FP16|quant|peak|MB|GB|Error"; then :; fi
  if [ -s "$HW_DIR/kv_memory_measured.log" ]; then
    echo "[done] kv_memory_measured in $(( (SECONDS-t0)/60 )) min"; PASS=$((PASS+1))
  else
    echo "[FAIL] kv_memory_measured"; FAIL=$((FAIL+1))
    FAILED="$FAILED kv_memory_measured"
  fi
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "F28 RE-RUN COMPLETE   pass=$PASS  skip=$SKIP  fail=$FAIL"
[ -n "$FAILED" ] && echo "FAILED:$FAILED"
echo "Zip for download:  cd $(pwd) && zip -r rerun_f28_results.zip $OUT"
echo "Old results are still in outputs/finals for the before/after diff."
echo "════════════════════════════════════════════════════════════"
