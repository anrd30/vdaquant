#!/usr/bin/env bash
# =============================================================================
# Phase 3 — the complete remaining GPU experiment set for the paper, in one
# sequential, RESUMABLE script (docs/optimization_ledger.md, "Phase 3").
#
# Colab usage (fire and walk away):
#   !cd /content/vdaquant && nohup bash scripts/run_phase3_experiments.sh > /content/phase3.log 2>&1 &
#   !tail -f /content/phase3.log
#
# Stages:   STAGE=core    -> E1-E4 (baselines + ablations, N=200, ~3 h)
#           STAGE=finals  -> E5-E6 (full-split finals + figures,   ~3 h)
#           STAGE=all     -> everything (default)
#   e.g.  STAGE=core bash scripts/run_phase3_experiments.sh
#
# RESUMABLE: each experiment writes to its own outputs/phase3/<name>/ dir and
# is SKIPPED if its pareto_benchmark_results.json already exists — so after a
# Colab disconnect, just re-run the same command and it continues where it
# stopped. Delete a result dir to force that experiment to re-run.
#
# Conventions (deliberate, do not change casually):
#   --rht-seed 0 everywhere except E3, whose whole point is seeds 1 and 2.
#   --no-qjl on EVERY comparison row (headline config is QJL-off; baselines
#     must not be charged QJL side-channel bits the headline doesn't pay).
#     The ONE exception is E4b, whose whole point is measuring QJL-on.
# =============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
SUITE="python scripts/run_pareto_benchmark_suite.py"
ROOT_OUT="outputs/phase3"
STAGE="${STAGE:-all}"
mkdir -p "$ROOT_OUT"

PASS=0; SKIP=0; FAIL=0; FAILED_NAMES=""

run() {  # run <name> <args...>
  local name="$1"; shift
  local out="$ROOT_OUT/$name"
  if [ -f "$out/pareto_benchmark_results.json" ]; then
    echo "[skip] $name (already done)"; SKIP=$((SKIP+1)); return 0
  fi
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "[run ] $name    $(date '+%H:%M:%S')"
  echo "       $*"
  echo "════════════════════════════════════════════════════════════"
  local t0=$SECONDS
  if $SUITE "$@" --output-dir "$out" 2>&1 | tee "$out.log" | grep -E "delta1|TAE|skipped|Surgery|Error|Traceback" ; then :; fi
  if [ -f "$out/pareto_benchmark_results.json" ]; then
    echo "[done] $name in $(( (SECONDS-t0)/60 )) min"
    PASS=$((PASS+1))
  else
    echo "[FAIL] $name — no results json produced; full log: $out.log"
    FAIL=$((FAIL+1)); FAILED_NAMES="$FAILED_NAMES $name"
  fi
}

if [ "$STAGE" = "core" ] || [ "$STAGE" = "all" ]; then
  echo "########## STAGE: core (E1–E4) ##########"

  # ---- E1a: scalar RTN baseline (eff 4.25 / 3.25 / 2.25 b/scalar) ----------
  run e1a_scalar_nyu   --dataset nyuv2 --eval-mode groundtruth --quantizer scalar \
      --bits 4 3 2 --no-qjl --rht-seed 0 --max-samples 200
  run e1a_scalar_kitti --dataset kitti --eval-mode groundtruth --quantizer scalar \
      --bits 4 3 2 --no-qjl --rht-seed 0 --max-samples 200

  # ---- E1b: D4 lattice baseline, 8-bit scales (eff 6 / 5 / 4 b/scalar) -----
  # Matched-rate cell for the paper: D4@2-bit = 4.0 eff == E8@3-bit's 4.0 eff.
  run e1b_d4_nyu   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_d4 \
      --scale-bits 8 --bits 4 3 2 --no-qjl --rht-seed 0 --max-samples 200
  run e1b_d4_kitti --dataset kitti --eval-mode groundtruth --quantizer lattice_d4 \
      --scale-bits 8 --bits 4 3 2 --no-qjl --rht-seed 0 --max-samples 200

  # ---- E2: rotation ablation (quantize RAW activations) --------------------
  run e2_norot_nyu   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits 4 3 --no-qjl --no-rotation --max-samples 200
  run e2_norot_kitti --dataset kitti --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits 4 3 --no-qjl --no-rotation --max-samples 200

  # ---- E3: RHT seed robustness (seed 0 exists from the headline run) -------
  run e3_seed1_nyu --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits 3 --no-qjl --rht-seed 1 --max-samples 200
  run e3_seed2_nyu --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits 3 --no-qjl --rht-seed 2 --max-samples 200

  # ---- E4: metadata & QJL cost at the operating point ----------------------
  run e4a_scalebits16_nyu --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 16 --bits 3 --no-qjl --rht-seed 0 --max-samples 200
  run e4b_qjl_on_nyu --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits 3 --qjl --rht-seed 0 --max-samples 200
fi

if [ "$STAGE" = "finals" ] || [ "$STAGE" = "all" ]; then
  echo "########## STAGE: finals (E5–E6) ##########"

  # ---- E5: full-split finals (the paper's tables) ---------------------------
  run e5a_nyu_full   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits 8 4 3 2 --no-qjl --rht-seed 0 --max-samples 654
  run e5b_kitti_full --dataset kitti --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits 8 4 3 2 --no-qjl --rht-seed 0 --max-samples 1000
  run e5c_sintel_temporal_full --dataset sintel --eval-mode temporal --quantizer lattice_e8 \
      --scale-bits 8 --bits 8 4 3 2 --no-qjl --rht-seed 0 --max-samples 2000 --max-scenes 23

  # ---- E6: qualitative figures ----------------------------------------------
  if [ ! -f "$ROOT_OUT/e6_figures/.done" ]; then
    echo "[run ] e6_figures"
    mkdir -p "$ROOT_OUT/e6_figures"
    python scripts/dump_depth_samples.py --dataset nyuv2  --num-frames 6 \
      --quantizer lattice_e8 --scale-bits 8 --no-qjl --output-dir "$ROOT_OUT/e6_figures/nyuv2" \
      && \
    python scripts/dump_depth_samples.py --dataset kitti  --num-frames 6 \
      --quantizer lattice_e8 --scale-bits 8 --no-qjl --output-dir "$ROOT_OUT/e6_figures/kitti" \
      && \
    python scripts/dump_depth_samples.py --dataset sintel --num-frames 12 --make-video \
      --quantizer lattice_e8 --scale-bits 8 --no-qjl --output-dir "$ROOT_OUT/e6_figures/sintel" \
      && touch "$ROOT_OUT/e6_figures/.done" && PASS=$((PASS+1)) \
      || { echo "[FAIL] e6_figures"; FAIL=$((FAIL+1)); FAILED_NAMES="$FAILED_NAMES e6_figures"; }
  else
    echo "[skip] e6_figures (already done)"; SKIP=$((SKIP+1))
  fi
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "PHASE 3 COMPLETE   pass=$PASS  skip=$SKIP  fail=$FAIL"
[ -n "$FAILED_NAMES" ] && echo "FAILED:$FAILED_NAMES"
echo "All results under $ROOT_OUT/<experiment>/pareto_benchmark_results.json"
echo "Zip everything for download:"
echo "  cd $(pwd) && zip -r phase3_results.zip $ROOT_OUT"
echo "════════════════════════════════════════════════════════════"
