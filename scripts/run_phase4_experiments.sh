#!/usr/bin/env bash
# =============================================================================
# Phase 4 -- the gate-closing GPU experiment set for the paper (docs/
# optimization_ledger.md Phase 3 results + docs/phase4_a_star_plan.md
# section 3). Mirrors scripts/run_phase3_experiments.sh's structure exactly:
# resumable, staged, skip-if-results-exist.
#
# Colab/GPU-box usage (fire and walk away):
#   !cd vdaquant && nohup bash scripts/run_phase4_experiments.sh > phase4.log 2>&1 &
#   !tail -f phase4.log
#
# Stages:   STAGE=gates   -> G1, G2b, G3a/b/d, G10, G11 (decision-gate runs
#                            + F16/F17 evidence; requires only S1/S2/S3/S10,
#                            all landed)                             ~2 h
#           STAGE=extend  -> G4, G5, E7, G8 (requires S4-S7 -- vitl
#                            support, K/V target flag, TAE covis mask, KV
#                            memory table -- NOT YET IMPLEMENTED as of this
#                            script's creation). Each block below AUTO-
#                            DETECTS whether its required CLI flag/script
#                            exists and skips with a clear message if not --
#                            so once a future task lands S4-S7, this same
#                            script starts running extend with no edits.
#           STAGE=all     -> both (default)
#   e.g.  STAGE=gates bash scripts/run_phase4_experiments.sh
#
# RESUMABLE: each experiment writes to its own outputs/phase4/<name>/ dir and
# is SKIPPED if its results file already exists -- after a disconnect, just
# re-run the same command. Delete a result dir to force a re-run.
#
# Conventions (same as Phase 3, deliberate, do not change casually):
#   --rht-seed 0 everywhere except G2b/G3 seed sweeps, whose point IS the seed.
#   --no-qjl on every comparison row (F15: QJL is strictly dominated -- costs
#     +2.25 eff bits for WORSE accuracy; settled, not re-litigated here).
#
# NOT included here (see docs/phase4_a_star_plan.md sec 3 for why):
#   G2a (scale-bits {6,12} sweep) -- LatticeE8Quantizer/ScalarGroupQuantizer
#     only accept scale_bits in {8,16} (research/quantizers/lattice_vq.py
#     asserts this; the --scale-bits CLI choices are locked to [8,16]).
#     Adding 6/12-bit scale storage is a real quantizer change, out of scope
#     for the S1-S10 tasks this runner was built alongside. G2b below
#     (scale-bits {8,16} x 3 seeds, full split) already answers DG-1
#     without it.
#   G9 -- CANCELLED. F13: E8@2-bit collapses on every dataset: there is no
#     3.0-effective-bit story to chase, so the E8@2b-vs-scalar_g8@2b
#     comparison G9 would have run is moot.
# =============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
SUITE="python scripts/run_pareto_benchmark_suite.py"
ROOT_OUT="outputs/phase4"
STAGE="${STAGE:-all}"
mkdir -p "$ROOT_OUT"

PASS=0; SKIP=0; FAIL=0; FAILED_NAMES=""

run() {  # run <name> <args...>   -- for run_pareto_benchmark_suite.py
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

run_custom() {  # run_custom <name> <sentinel_filename> <cmd...>  -- for the
                # dump_activation_stats.py / dump_structure_stats.py scripts,
                # whose output filename differs from the benchmark suite's.
  local name="$1"; shift
  local sentinel="$1"; shift
  local out="$ROOT_OUT/$name"
  if [ -f "$out/$sentinel" ]; then
    echo "[skip] $name (already done)"; SKIP=$((SKIP+1)); return 0
  fi
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "[run ] $name    $(date '+%H:%M:%S')"
  echo "       $*"
  echo "════════════════════════════════════════════════════════════"
  local t0=$SECONDS
  mkdir -p "$out"
  if "$@" --output-dir "$out" 2>&1 | tee "$out.log" ; then :; fi
  if [ -f "$out/$sentinel" ]; then
    echo "[done] $name in $(( (SECONDS-t0)/60 )) min"
    PASS=$((PASS+1))
  else
    echo "[FAIL] $name — sentinel $sentinel not produced; full log: $out.log"
    FAIL=$((FAIL+1)); FAILED_NAMES="$FAILED_NAMES $name"
  fi
}

has_flag() {  # has_flag <flag-substring> -- capability-detect a not-yet-
              # -guaranteed CLI flag so STAGE=extend can auto-activate once
              # S4-S7 land, with no edits to this script.
  $SUITE --help 2>&1 | grep -q -- "$1"
}

if [ "$STAGE" = "gates" ] || [ "$STAGE" = "all" ]; then
  echo "########## STAGE: gates (G1, G2b, G3a/b/d, G10, G11) ##########"

  # ---- G1: fair scalar_g8 baseline (S1) -- resolves F11, decides DG-3 -------
  run g1_scalar_g8_nyu   --dataset nyuv2 --eval-mode groundtruth --quantizer scalar_g8 \
      --scale-bits 8 --bits 4 3 2 --no-qjl --rht-seed 0 --max-samples 200
  run g1_scalar_g8_kitti --dataset kitti  --eval-mode groundtruth --quantizer scalar_g8 \
      --scale-bits 8 --bits 4 3 2 --no-qjl --rht-seed 0 --max-samples 200

  # ---- G2b: DECISIVE for DG-1 -- full-654 NYU, E8@3b, scale-bits x seeds ----
  # F12 already showed the N=200 "+0.01 over FP32" was subset noise; this
  # sweep supplies the confidence intervals (via scripts/compute_stats.py)
  # to state "indistinguishable from FP32" rigorously, and settles the
  # residual scale-bits question left open in F12 (different seeds confound
  # the E4a-vs-E3 comparison there).
  for sb in 8 16; do
    for seed in 0 1 2; do
      run "g2b_sb${sb}_seed${seed}" --dataset nyuv2 --eval-mode groundtruth \
          --quantizer lattice_e8 --scale-bits "$sb" --bits 3 --no-qjl \
          --rht-seed "$seed" --max-samples 654
    done
  done

  # ---- G3a: rotation control (with-rotation, same N/config as E2) -----------
  # E2's no-rotation run showed 0.9138 NYU / 0.9244 KITTI at 4.0 eff bits,
  # inside the with-rotation seed spread (F14) -- but no seed-0 with-rotation
  # run at this EXACT N=200/config exists in the Phase 3 zip. This is that
  # control; without it we can't rule out the flag silently not doing anything.
  run g3a_rot_control_nyu   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits 4 3 --no-qjl --rht-seed 0 --max-samples 200
  run g3a_rot_control_kitti --dataset kitti  --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits 4 3 --no-qjl --rht-seed 0 --max-samples 200

  # ---- G3b: rotation ablation at 2-bit -- the one regime RHT could still matter ----
  # E8@2b collapses on accuracy (F13) regardless, but this is the last place
  # rotation could plausibly still be earning its keep before DG-2 concludes
  # "rotation is inert on this model class at every rate we tested."
  run g3b_rot_on_2bit_nyu    --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits 2 --no-qjl --rht-seed 0 --max-samples 200
  run g3b_rot_off_2bit_nyu   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits 2 --no-qjl --no-rotation --max-samples 200
  run g3b_rot_on_2bit_kitti  --dataset kitti  --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits 2 --no-qjl --rht-seed 0 --max-samples 200
  run g3b_rot_off_2bit_kitti --dataset kitti  --eval-mode groundtruth --quantizer lattice_e8 \
      --scale-bits 8 --bits 2 --no-qjl --no-rotation --max-samples 200

  # ---- G3d: activation statistics (S3) -- the mechanism behind DG-2 ---------
  run_custom g3d_activation_stats_nyu "activation_stats.json" \
      python scripts/dump_activation_stats.py --dataset nyuv2 --num-images 8

  # ---- G10: temporal-window sweep (F17) --------------------------------------
  # Tests whether the Sintel-vs-static accuracy gap at 4.0 eff bits grows
  # with window length (quantization error accumulating across the cache)
  # or is just Sintel being a harder dataset (flat with window length).
  for w in 8 16 32; do
    run "g10_tempwindow${w}_sintel" --dataset sintel --eval-mode temporal \
        --quantizer lattice_e8 --scale-bits 8 --bits 3 --no-qjl --rht-seed 0 \
        --temporal-window "$w" --max-samples 200 --max-scenes 5
  done

  # ---- G11: structure/degeneracy diagnostic (S10) -- quantifies F16 ---------
  run_custom g11_structure_sintel "structure_stats.json" \
      python scripts/dump_structure_stats.py --dataset sintel --quantizer lattice_e8 \
      --scale-bits 8 --bits 8 4 3 2 --no-qjl --max-samples 200
  run_custom g11_structure_nyu "structure_stats.json" \
      python scripts/dump_structure_stats.py --dataset nyuv2 --quantizer lattice_e8 \
      --scale-bits 8 --bits 8 4 3 2 --no-qjl --max-samples 200
fi

if [ "$STAGE" = "extend" ] || [ "$STAGE" = "all" ]; then
  echo "########## STAGE: extend (G4, G5, E7, G8) ##########"
  echo "  (each block below requires S4-S7; auto-skips with a message if the"
  echo "   flag/script hasn't landed yet -- no edits needed once it does)"

  # ---- G4: VDA-Large transfer (needs S4: --encoder vitl) ---------------------
  if has_flag "--encoder"; then
    run g4_vitl_e8_nyu       --dataset nyuv2 --eval-mode groundtruth --encoder vitl \
        --quantizer lattice_e8  --scale-bits 8 --bits 4 3 --no-qjl --rht-seed 0 --max-samples 200
    run g4_vitl_scalarg8_nyu --dataset nyuv2 --eval-mode groundtruth --encoder vitl \
        --quantizer scalar_g8   --scale-bits 8 --bits 3   --no-qjl --rht-seed 0 --max-samples 200
    run g4_vitl_e8_kitti     --dataset kitti  --eval-mode groundtruth --encoder vitl \
        --quantizer lattice_e8  --scale-bits 8 --bits 4 3 --no-qjl --rht-seed 0 --max-samples 200
  else
    echo "[skip] G4 (VDA-Large transfer) -- S4 (--encoder vitl) not yet implemented"
  fi

  # ---- G5: K/V target ablation (needs S5: --quantize-target) -----------------
  if has_flag "--quantize-target"; then
    run g5_k_only_nyu --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
        --scale-bits 8 --bits 3 --no-qjl --rht-seed 0 --quantize-target k --max-samples 200
    run g5_v_only_nyu --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
        --scale-bits 8 --bits 3 --no-qjl --rht-seed 0 --quantize-target v --max-samples 200
  else
    echo "[skip] G5 (K/V target ablation) -- S5 (--quantize-target) not yet implemented"
  fi

  # ---- E7: Sintel temporal rerun w/ GT co-visibility mask (needs S6) --------
  if has_flag "--tae-covis-tau"; then
    run e7_sintel_covis --dataset sintel --eval-mode temporal --quantizer lattice_e8 \
        --scale-bits 8 --bits 8 4 3 2 --no-qjl --rht-seed 0 \
        --max-samples 2000 --max-scenes 23 --tae-covis-tau 0.05
  else
    echo "[skip] E7 (GT co-visibility TAE) -- S6 (--tae-covis-tau) not yet implemented"
  fi

  # ---- G8: KV memory table (needs S7: scripts/report_kv_memory.py) ----------
  if [ -f scripts/report_kv_memory.py ]; then
    out="$ROOT_OUT/g8_kv_memory"
    if [ -f "$out/.done" ]; then
      echo "[skip] g8_kv_memory (already done)"; SKIP=$((SKIP+1))
    else
      mkdir -p "$out"
      if python scripts/report_kv_memory.py --measure > "$out/kv_memory_table.log" 2>&1; then
        touch "$out/.done"; PASS=$((PASS+1)); echo "[done] g8_kv_memory"
      else
        echo "[FAIL] g8_kv_memory — see $out/kv_memory_table.log"
        FAIL=$((FAIL+1)); FAILED_NAMES="$FAILED_NAMES g8_kv_memory"
      fi
    fi
  else
    echo "[skip] G8 (KV memory table) -- S7 (scripts/report_kv_memory.py) not yet implemented"
  fi
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "PHASE 4 COMPLETE   pass=$PASS  skip=$SKIP  fail=$FAIL"
[ -n "$FAILED_NAMES" ] && echo "FAILED:$FAILED_NAMES"
echo "All results under $ROOT_OUT/<experiment>/{pareto_benchmark_results.json,"
echo "  activation_stats.json, structure_stats.json}"
echo ""
echo "Next: compute confidence intervals on the decisive comparison, e.g."
echo "  python scripts/compute_stats.py $ROOT_OUT/g2b_sb8_seed0/pareto_benchmark_results.json \\"
echo "    --config-a FP32_Baseline --config-b 3bit --metric delta1"
echo ""
echo "Zip everything for download:"
echo "  cd $(pwd) && zip -r phase4_results.zip $ROOT_OUT"
echo "════════════════════════════════════════════════════════════"
