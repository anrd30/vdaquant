#!/usr/bin/env bash
# =============================================================================
# PAPER 2 (LATTICE LADDER) — FINALISATION RUN
#
# What's here (everything left on the current A100 to make Paper 2's tables
# fully populated without new datasets or CUDA kernels):
#
#   SEED  BW16 5-seed CI on the headline 3-bit config, ViT-S,
#         NYU / KITTI / Sintel, seeds {1,2,3,4}.  seed 0 was already done
#         in paper2_verify / paper2_extend, so we only need four more.
#         Closes §5.8 (currently promises seed CIs, does not deliver).
#         ~2h,  12 runs
#
#   BASE  Missing table-1 baselines on ViT-S:
#           - Uniform vector quantiser (dimension 4, K-means) at b_eff 3.5, 4.5
#           - E8 at raw 3 and 4 bits (raw b_eff 3.5, 4.5)
#         These are the natural non-lattice and single-lattice comparators
#         for BW16.  Closes the "you say BW16 dominates E8 but do not show
#         E8 in the same regime" reviewer point.
#         ~1.3h,  8 runs
#
# Skipped intentionally:
#   - T=64 temporal window.  VDA's motion-module positional-encoding buffer
#     is baked into the checkpoint at max_len=32; running at T=64 crashes
#     in pos_encoder with a shape mismatch (not an OOM, not our decoder).
#     Any T>32 result would require PE extrapolation, which is orthogonal
#     to KV cache quantisation.  Paper 2 reports T in {8,16,32} and says
#     so explicitly.
#   - KIVI empirical baseline.  Paper 2 will position KIVI as "targets LLM
#     autoregressive decode; VDA has neither, comparison is out of scope"
#     in related work and drop the K/V asymmetry section entirely.
#   - Fused CUDA/Triton kernel work — engineering, not benchmarking; lives
#     on the personal A100 after these runs land.
#   - TartanAir / Bonn / DIODE — deferred to personal A100.
#
# Everything is resumable in outputs/paper2_final/.  Total ~3.3h.
#
# Usage:
#   cd <repo>/code_run1
#   git fetch && git checkout verify && git pull
#   tmux new -d -s p2fin "bash scripts/run_paper2_finalize.sh > p2fin.log 2>&1"
# =============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-$(command -v python || command -v python3)}"
if [ -z "$PY" ]; then echo "no python"; exit 1; fi
SUITE="$PY scripts/run_pareto_benchmark_suite.py"
OUT="outputs/paper2_final"
mkdir -p "$OUT"
PASS=0; SKIP=0; FAIL=0; FAILED=""; T0=$SECONDS

run() {
  local name="$1"; shift
  local d="$OUT/$name"
  if [ -f "$d/pareto_benchmark_results.json" ]; then
    echo "[skip] $name"; SKIP=$((SKIP+1)); return 0
  fi
  echo ""; echo "════════ [run] $name  $(date '+%H:%M:%S') ════════"; echo "   $*"
  local t0=$SECONDS
  if $SUITE "$@" --output-dir "$d" 2>&1 | tee "$d.log" \
       | grep -E "delta1|TAE|covis|Surgery|Error|Traceback"; then :; fi
  if [ -f "$d/pareto_benchmark_results.json" ]; then
    echo "[done] $name in $(( (SECONDS-t0)/60 )) min"; PASS=$((PASS+1))
  else
    echo "[FAIL] $name — see $d.log"; FAIL=$((FAIL+1)); FAILED="$FAILED $name"
  fi
}

# =============================================================================
echo "########## STAGE SEED: BW16 3-bit ViT-S, seeds 1-4 (12 runs) ##########"
# Same headline config as bw16_{nyu,kitti,sintel}_b3 from paper2_verify but
# with different Hadamard random-sign draws.  Standard error on d1 across
# seeds gives the CI we quote in §5.8.
for s in 1 2 3 4; do
  run "seed${s}_nyu_b3"   --dataset nyuv2 --eval-mode groundtruth \
      --quantizer lattice_bw16 --scale-bits 8 --bits 3 --no-qjl \
      --rht-seed $s --max-samples 654  --group-size 16
  run "seed${s}_kitti_b3" --dataset kitti --eval-mode groundtruth \
      --quantizer lattice_bw16 --scale-bits 8 --bits 3 --no-qjl \
      --rht-seed $s --max-samples 1000 --group-size 16
  run "seed${s}_sintel_b3" --dataset sintel --eval-mode temporal \
      --quantizer lattice_bw16 --scale-bits 8 --bits 3 --no-qjl \
      --rht-seed $s --temporal-window 32 --max-samples 2000 --max-scenes 23 \
      --tae-covis-tau 0.05 --group-size 16
done

# =============================================================================
echo "########## STAGE BASE: Missing table-1 baselines (8 runs) ##########"
# Uniform vector quantiser (dimension 4, K-means codebook) at matched b_eff.
# --quantizer uniform_vector uses the same rotation and scale infrastructure
# as the lattices, so the comparison isolates the codebook.
for bits in 3 4; do
  run "uvq_nyu_b${bits}"   --dataset nyuv2 --eval-mode groundtruth \
      --quantizer uniform_vector --scale-bits 8 --bits $bits --no-qjl \
      --rht-seed 0 --max-samples 654  --group-size 16
  run "uvq_kitti_b${bits}" --dataset kitti --eval-mode groundtruth \
      --quantizer uniform_vector --scale-bits 8 --bits $bits --no-qjl \
      --rht-seed 0 --max-samples 1000 --group-size 16
done

# E8 at raw 3- and 4-bit on ViT-S.  paper2_verify only ran E8 at 5/6/7 bits
# (b_eff 6/7/8, the "high-precision half" of the ladder); we need the 3/4
# bit rows so table 1's "E8 at matched b_eff" column is honest.
for bits in 3 4; do
  run "e8_nyu_b${bits}"   --dataset nyuv2 --eval-mode groundtruth \
      --quantizer lattice_e8 --scale-bits 8 --bits $bits --no-qjl \
      --rht-seed 0 --max-samples 654  --group-size 8
  run "e8_kitti_b${bits}" --dataset kitti --eval-mode groundtruth \
      --quantizer lattice_e8 --scale-bits 8 --bits $bits --no-qjl \
      --rht-seed 0 --max-samples 1000 --group-size 8
done

# =============================================================================
# STAGE TW64 removed — VDA's pos_encoder is baked at max_len=32, cannot
# accept T=64 without checkpoint-level PE extrapolation.  See header.
# =============================================================================
echo ""
echo "════════════════════════════════════════════════════════════════════════"
printf "PAPER 2 FINAL COMPLETE   pass=%d  skip=%d  fail=%d   total=%d min\n" \
  "$PASS" "$SKIP" "$FAIL" $(( (SECONDS-T0)/60 ))
[ -n "$FAILED" ] && echo "FAILED:$FAILED"
echo ""
echo "Zip for download:"
echo "  cd $(pwd) && zip -r paper2_final_results.zip $OUT p2fin.log"
echo "════════════════════════════════════════════════════════════════════════"
