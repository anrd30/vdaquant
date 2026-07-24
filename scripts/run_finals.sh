#!/usr/bin/env bash
# =============================================================================
# FINALS — the complete remaining GPU run for the workshop paper, in one
# resumable script. Fire once and walk away; everything that finishes is
# banked, and re-running skips what's done (per-experiment results-file check).
#
# Usage (fire and forget):
#   !cd vdaquant && git pull && nohup bash scripts/run_finals.sh > finals.log 2>&1 &
#   !tail -f finals.log
#
# When done:
#   cd vdaquant && zip -r finals_results.zip outputs/finals && (send finals_results.zip)
#
# Rough time: ~4–6 h on one modern GPU (vitl is ~3x vits). Fully resumable, so
# a disconnect just means re-run the same command. Delete a result dir to force
# that one experiment to re-run.
#
# What this produces (everything the paper's tables/figures need that we do NOT
# already have in hand):
#   1. Fair-baseline finals: scalar_g8 on FULL splits (we only had N=200).
#   2. D4 + E8 finals on full splits, current code (per-image dumps -> CIs).
#   3. E8@3b KITTI seed sweep (NYU seeds already exist in phase4/g2b).
#   4. E7: Sintel temporal WITH co-visibility TAE — the gameability centerpiece
#      (our earlier Sintel run predates the S6 covis mask).
#   5. Scale ladder: vitb + vitl on NYU/KITTI (+ vitl Sintel temporal).
#   6. Analytic KV-memory table (CPU) + one measured vitl peak.
#
# Conventions (match Phase 3/4): --no-qjl on every row (F15: QJL dominated),
# --rht-seed 0 unless a seed sweep, scale-bits 8 (F18: 8 ≡ 16, 4.0 eff bits).
# =============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
SUITE="python scripts/run_pareto_benchmark_suite.py"
OUT="outputs/finals"
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

echo "########## STAGE 1: vits finals (full splits, groundtruth) ##########"
# E8 hero + D4 + fair scalar_g8, full splits, current code (per-image -> CIs).
run e8_nyu       --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
    --scale-bits 8 --bits 8 4 3 2 --no-qjl --rht-seed 0 --max-samples 654
run e8_kitti     --dataset kitti --eval-mode groundtruth --quantizer lattice_e8 \
    --scale-bits 8 --bits 8 4 3 2 --no-qjl --rht-seed 0 --max-samples 1000
run d4_nyu       --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_d4 \
    --scale-bits 8 --bits 4 3 2 --no-qjl --rht-seed 0 --max-samples 654
run d4_kitti     --dataset kitti --eval-mode groundtruth --quantizer lattice_d4 \
    --scale-bits 8 --bits 4 3 2 --no-qjl --rht-seed 0 --max-samples 1000
run scalarg8_nyu --dataset nyuv2 --eval-mode groundtruth --quantizer scalar_g8 \
    --scale-bits 8 --bits 4 3 2 --no-qjl --rht-seed 0 --max-samples 654
run scalarg8_kitti --dataset kitti --eval-mode groundtruth --quantizer scalar_g8 \
    --scale-bits 8 --bits 4 3 2 --no-qjl --rht-seed 0 --max-samples 1000

# E8@3b KITTI seed sweep for symmetric CIs vs NYU (NYU seeds live in phase4/g2b).
for s in 1 2; do
  run "e8_kitti_seed$s" --dataset kitti --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits 3 --no-qjl --rht-seed "$s" --max-samples 1000
done

echo "########## STAGE 2: Sintel temporal WITH co-visibility TAE (E7) ##########"
# The gameability centerpiece: raw TAE, median, AND covis-masked TAE together.
run e8_sintel_covis     --dataset sintel --eval-mode temporal --quantizer lattice_e8 \
    --scale-bits 8 --bits 8 4 3 2 --no-qjl --rht-seed 0 --temporal-window 32 \
    --max-samples 2000 --max-scenes 23 --tae-covis-tau 0.05
run scalarg8_sintel_covis --dataset sintel --eval-mode temporal --quantizer scalar_g8 \
    --scale-bits 8 --bits 4 3 --no-qjl --rht-seed 0 --temporal-window 32 \
    --max-samples 2000 --max-scenes 23 --tae-covis-tau 0.05

echo "########## STAGE 3: scale ladder (vitb, vitl) ##########"
# Generalization across model scale. N=200 is enough for a transfer check.
for enc in vitb vitl; do
  for ds in nyuv2 kitti; do
    run "ladder_${enc}_e8_${ds}"  --dataset "$ds" --eval-mode groundtruth --encoder "$enc" \
        --quantizer lattice_e8 --scale-bits 8 --bits 4 3 --no-qjl --rht-seed 0 --max-samples 200
    run "ladder_${enc}_sg8_${ds}" --dataset "$ds" --eval-mode groundtruth --encoder "$enc" \
        --quantizer scalar_g8  --scale-bits 8 --bits 3   --no-qjl --rht-seed 0 --max-samples 200
  done
done
# vitl temporal covis — does the TAE-gameability finding hold at scale?
run ladder_vitl_sintel --dataset sintel --eval-mode temporal --encoder vitl \
    --quantizer lattice_e8 --scale-bits 8 --bits 4 3 2 --no-qjl --rht-seed 0 \
    --temporal-window 32 --max-samples 2000 --max-scenes 10 --tae-covis-tau 0.05

echo "########## STAGE 4: analytic KV-memory table (CPU) + measured vitl peak ##########"
if [ ! -f "$OUT/memory/kv_memory_table.md" ]; then
  mkdir -p "$OUT/memory"
  if python scripts/report_kv_memory.py --measure --encoder vitl --output-dir "$OUT/memory" \
       > "$OUT/memory/report.log" 2>&1; then
    echo "[done] kv_memory_table"; PASS=$((PASS+1))
  else
    echo "[FAIL] kv_memory_table — see $OUT/memory/report.log"; FAIL=$((FAIL+1)); FAILED="$FAILED memory"
  fi
else
  echo "[skip] kv_memory_table"; SKIP=$((SKIP+1))
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "FINALS COMPLETE   pass=$PASS  skip=$SKIP  fail=$FAIL"
[ -n "$FAILED" ] && echo "FAILED:$FAILED"
echo "Zip for download:  cd $(pwd) && zip -r finals_results.zip $OUT"
echo "════════════════════════════════════════════════════════════"
