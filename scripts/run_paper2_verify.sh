#!/usr/bin/env bash
# =============================================================================
# PAPER 2 (LATTICE LADDER) — VERIFICATION RUN on the 3 already-downloaded
# datasets (NYU-654, KITTI-1000, Sintel). If numbers look good we extend to
# TartanAir/Bonn/DIODE in a follow-up run.
#
# What this adds beyond the F28 rerun (which was Paper 1 focus):
#   S1  Sintel K/V asymmetry (temporal version of the NYU/KITTI K/V we already have)
#   S2  Fine-grained bit-rate sweep (5, 6, 7-bit payloads) for a smooth Pareto
#   S3  Extra rotation seeds (3, 4) for tighter statistical intervals
#   S4  QJL ablation on full splits (was only N=200 before F15)
#   S5  Symmetric-vs-asymmetric K/V verification on NYU/KITTI (re-checks
#       what we ran under asym_kv/ with a consistent config for the paper table)
#
# All results land in outputs/paper2_verify/ so nothing overwrites existing runs.
#
# Usage:
#   cd <repo>/code_run1
#   git fetch && git checkout verify && git pull
#   tmux new -s paper2
#   bash scripts/run_paper2_verify.sh
#   # Ctrl+B then D to detach. Reattach with: tmux attach -t paper2
#
# Rough time: ~6-8 h on one A100. Resumable — re-running skips finished experiments.
# =============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-$(command -v python || command -v python3)}"
if [ -z "$PY" ]; then
  echo "ERROR: no python interpreter found. Set PY=/path/to/python and re-run."
  exit 1
fi
echo "Using python: $PY"
SUITE="$PY scripts/run_pareto_benchmark_suite.py"
OUT="outputs/paper2_verify"
mkdir -p "$OUT"
PASS=0; SKIP=0; FAIL=0; FAILED=""
T0=$SECONDS

run() {  # run <name> <args...>
  local name="$1"; shift
  local d="$OUT/$name"
  if [ -f "$d/pareto_benchmark_results.json" ]; then
    echo "[skip] $name (already done)"; SKIP=$((SKIP+1)); return 0
  fi
  echo ""; echo "════════ [run] $name  $(date '+%H:%M:%S') ════════"; echo "   $*"
  local t0=$SECONDS
  if $SUITE "$@" --output-dir "$d" 2>&1 | tee "$d.log" \
       | grep -E "delta1|TAE|covis|skipped|Surgery|Error|Traceback"; then :; fi
  if [ -f "$d/pareto_benchmark_results.json" ]; then
    echo "[done] $name in $(( (SECONDS-t0)/60 )) min"; PASS=$((PASS+1))
  else
    echo "[FAIL] $name — see $d.log"; FAIL=$((FAIL+1)); FAILED="$FAILED $name"
  fi
}

# =============================================================================
echo "########## STAGE 0: BW16 & Golay24-A lattice sweeps (NEW LATTICES) ##########"
# Two new lattice families beyond E8/D4/scalar:
#   * lattice_bw16    -- Construction A over RM(1,4). 16-dim lattice, min-norm 4.
#                       group_size 16 (minimum) or a multiple. head_dim=64 fits
#                       (4 groups per head).
#   * lattice_golay24 -- Construction A over the extended Golay code G_24.
#                       24-dim lattice, min-norm 4. head_dim=64 is NOT
#                       divisible by 24, so we set group_size=48 (uses
#                       tensor flattening across the trailing dim so pairs
#                       of heads are quantised together).
# Both quantisers correspond to well-defined lattices verified by unit tests
# in tests/test_bw16_golay24.py (kissing weights, mod-2 membership, etc.).
# Reported precisely as "L16_A / L24_A (Construction A)" in the paper.
for bits in 4 3; do
  run "bw16_nyu_b${bits}"   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_bw16 \
      --scale-bits 8 --bits $bits --no-qjl --rht-seed 0 --max-samples 654 --group-size 16
  run "bw16_kitti_b${bits}" --dataset kitti --eval-mode groundtruth --quantizer lattice_bw16 \
      --scale-bits 8 --bits $bits --no-qjl --rht-seed 0 --max-samples 1000 --group-size 16
done
for bits in 4 3; do
  run "g24_nyu_b${bits}"   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_golay24 \
      --scale-bits 8 --bits $bits --no-qjl --rht-seed 0 --max-samples 654 --group-size 48
  run "g24_kitti_b${bits}" --dataset kitti --eval-mode groundtruth --quantizer lattice_golay24 \
      --scale-bits 8 --bits $bits --no-qjl --rht-seed 0 --max-samples 1000 --group-size 48
done
# Sintel temporal + co-visibility TAE for BW16 and Golay24 at 3-bit only.
run "bw16_sintel_b3" --dataset sintel --eval-mode temporal --quantizer lattice_bw16 \
    --scale-bits 8 --bits 3 --no-qjl --rht-seed 0 --temporal-window 32 \
    --max-samples 2000 --max-scenes 23 --tae-covis-tau 0.05 --group-size 16
run "g24_sintel_b3"  --dataset sintel --eval-mode temporal --quantizer lattice_golay24 \
    --scale-bits 8 --bits 3 --no-qjl --rht-seed 0 --temporal-window 32 \
    --max-samples 2000 --max-scenes 23 --tae-covis-tau 0.05 --group-size 48

# =============================================================================
echo "########## STAGE 1: Sintel K/V asymmetry (temporal — NEW) ##########"
# Complements the NYU/KITTI K/V we already ran under outputs_new/asym_kv/.
# Uses the corrected TAE (co-visibility mask) from Paper 1's finding.
for kv in "3 3" "4 2" "2 4"; do
  K=$(echo $kv | awk '{print $1}'); V=$(echo $kv | awk '{print $2}')
  run "kv_${K}_${V}_sintel" --dataset sintel --eval-mode temporal --quantizer lattice_e8 \
      --scale-bits 8 --bits $K --v-bits $V --no-qjl --rht-seed 0 \
      --temporal-window 32 --max-samples 2000 --max-scenes 23 --tae-covis-tau 0.05
done

# =============================================================================
echo "########## STAGE 2: Fine-grained bit-rate sweep (NEW) ##########"
# Fills in 5/6/7-bit payloads for a smooth Pareto curve. Paper had only
# 2/3/4/8 which shows a cliff at 4.0 eff bits — this smooths it.
for bits in 5 6 7; do
  run "e8_nyu_b${bits}"   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits $bits --no-qjl --rht-seed 0 --max-samples 654
  run "e8_kitti_b${bits}" --dataset kitti --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits $bits --no-qjl --rht-seed 0 --max-samples 1000
done

# =============================================================================
echo "########## STAGE 3: Extra rotation seeds for tighter CIs (NEW) ##########"
# Extends the 3-seed analysis to 5 seeds at the headline configuration
# (3-bit E8, 4.0 eff bits). Tighter confidence intervals for the paper table.
for s in 3 4; do
  run "e8_nyu_seed${s}"   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits 3 --no-qjl --rht-seed "$s" --max-samples 654
  run "e8_kitti_seed${s}" --dataset kitti --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits 3 --no-qjl --rht-seed "$s" --max-samples 1000
done

# =============================================================================
echo "########## STAGE 4: QJL ablation on FULL splits (upgrade from N=200) ##########"
# F15 said "QJL is dominated" from N=200 data. Confirm on full splits so the
# ablation table cites full-split numbers.
run "qjl_on_nyu"    --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
    --scale-bits 8 --bits 3      --qjl --rht-seed 0 --max-samples 654
run "qjl_on_kitti"  --dataset kitti --eval-mode groundtruth --quantizer lattice_e8 \
    --scale-bits 8 --bits 3      --qjl --rht-seed 0 --max-samples 1000

# Symmetric K=V=3b baseline is NOT re-run here — it's already in
# outputs/rerun_f28/e8_nyu/ and e8_kitti/ at the 3-bit rows. Reuse those.
# (Removed a redundant stage to save ~30 min of GPU.)

# =============================================================================
echo ""
echo "════════════════════════════════════════════════════════════════════════"
printf "PAPER 2 VERIFY COMPLETE   pass=%d  skip=%d  fail=%d   total=%d min\n" \
  "$PASS" "$SKIP" "$FAIL" $(( (SECONDS-T0)/60 ))
[ -n "$FAILED" ] && echo "FAILED:$FAILED"
echo ""
echo "Zip for download:"
echo "  cd $(pwd) && zip -r paper2_verify_results.zip $OUT"
echo ""
echo "Quick summary:"
echo "  grep -E 'delta1|TAE' $OUT/*.log | tail -80"
echo "════════════════════════════════════════════════════════════════════════"
