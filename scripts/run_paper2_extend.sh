#!/usr/bin/env bash
# =============================================================================
# PAPER 2 (LATTICE LADDER) — EXTENSION RUN
# Everything runnable on NYU / KITTI / Sintel to close the A* review gap
# without touching new datasets or writing real CUDA kernels. That's what
# the personal A100 + Leech/BW16 hardware-kernel push after CVPR handle.
#
# What's here:
#   S1  BW16 cross-encoder (ViT-B, ViT-L)   — 8 runs, ~2 h
#   S3  BW16 rotation ablation              — 4 runs, ~1 h
#   S5  BW16 group-size sweep               — 6 runs, ~1.5 h
#   S6  BW16 temporal window sweep on Sintel — 3 runs, ~1 h
#   HW  Hardware benchmark: real GPU peak memory + wall-clock latency /
#       FPS at every headline configuration. Uses the existing simulated
#       quantiser but reports MEASURED runtime alongside the analytic
#       memory savings, matching how LLVQ (ICML 2026) presents its numbers
#       before it had integer kernels.
#   QUAL Qualitative depth-map panels (Fig-3-style side-by-sides) on
#       NYU / KITTI / Sintel + a rotation-ablation panel. Produces PNG
#       strips (RGB | FP32 | 4b | 3b | 2b) and an MP4 for the Sintel
#       ladder so reviewers can see the 2-bit temporal collapse. ~20 min.
#
# Skipped intentionally (do them later, on the personal A100 or a follow-up):
#   #2  K/V asymmetry seed sweep         — dropped, current results already ship
#   #4  BW16 seed sweep                  — dropped, main-seed BW16 is the story
#   #7  K/V cross-encoder                — nice-to-have, run on personal A100
#   #8  BW16 Sintel 4-bit                — nice-to-have, single point
#
# Results land in outputs/paper2_extend/, resumable.
#
# Usage:
#   cd <repo>/code_run1
#   git fetch && git checkout verify && git pull
#   tmux new -d -s p2ext "bash scripts/run_paper2_extend.sh > p2ext.log 2>&1"
# =============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-$(command -v python || command -v python3)}"
if [ -z "$PY" ]; then echo "no python"; exit 1; fi
SUITE="$PY scripts/run_pareto_benchmark_suite.py"
OUT="outputs/paper2_extend"
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
echo "########## STAGE 1: BW16 cross-encoder validation (8 runs) ##########"
for enc in vitb vitl; do
  for bits in 4 3; do
    run "bw16_${enc}_nyu_b${bits}"   --dataset nyuv2 --eval-mode groundtruth \
        --encoder "$enc" --quantizer lattice_bw16 --scale-bits 8 --bits $bits \
        --no-qjl --rht-seed 0 --max-samples 654 --group-size 16
    run "bw16_${enc}_kitti_b${bits}" --dataset kitti --eval-mode groundtruth \
        --encoder "$enc" --quantizer lattice_bw16 --scale-bits 8 --bits $bits \
        --no-qjl --rht-seed 0 --max-samples 1000 --group-size 16
  done
done

# =============================================================================
echo "########## STAGE 3: BW16 rotation ablation (4 runs) ##########"
for bits in 4 3; do
  run "bw16_norot_nyu_b${bits}"   --dataset nyuv2 --eval-mode groundtruth \
      --quantizer lattice_bw16 --scale-bits 8 --bits $bits --no-qjl --no-rotation \
      --rht-seed 0 --max-samples 654 --group-size 16
  run "bw16_norot_kitti_b${bits}" --dataset kitti --eval-mode groundtruth \
      --quantizer lattice_bw16 --scale-bits 8 --bits $bits --no-qjl --no-rotation \
      --rht-seed 0 --max-samples 1000 --group-size 16
done

# =============================================================================
echo "########## STAGE 5: BW16 group-size sweep (6 runs) ##########"
for g in 32 48 64; do
  run "bw16_nyu_b3_g${g}"   --dataset nyuv2 --eval-mode groundtruth \
      --quantizer lattice_bw16 --scale-bits 8 --bits 3 --no-qjl --rht-seed 0 \
      --max-samples 654 --group-size $g
  run "bw16_kitti_b3_g${g}" --dataset kitti --eval-mode groundtruth \
      --quantizer lattice_bw16 --scale-bits 8 --bits 3 --no-qjl --rht-seed 0 \
      --max-samples 1000 --group-size $g
done

# =============================================================================
echo "########## STAGE 6: BW16 temporal window sweep on Sintel (3 runs) ##########"
for w in 8 16 64; do
  run "bw16_sintel_w${w}" --dataset sintel --eval-mode temporal \
      --quantizer lattice_bw16 --scale-bits 8 --bits 3 --no-qjl --rht-seed 0 \
      --temporal-window $w --max-samples 2000 --max-scenes 23 --tae-covis-tau 0.05 \
      --group-size 16
done

# =============================================================================
echo "########## STAGE HW: Hardware benchmark (real memory + latency) ##########"
# Runs report_kv_memory.py --measure across all headline configs so we can
# quote a MEASURED wall-clock latency and MEASURED GPU peak memory alongside
# the analytic KV-cache footprint. This is the same style LLVQ (ICML 2026)
# uses in its Table 4 before it had integer kernels. Custom int3/int4 decode
# kernels are follow-up work.
HW_DIR="$OUT/hardware"
if [ ! -f "$HW_DIR/vitl_bw16_b3.log" ]; then
  mkdir -p "$HW_DIR"
  echo ""; echo "════════ [run] hardware_benchmark  $(date '+%H:%M:%S') ════════"
  t0=$SECONDS

  # Baseline FP16
  $PY scripts/report_kv_memory.py --measure --encoder vitl --window 32 \
      2>&1 | tee "$HW_DIR/fp16_baseline.log" | grep -E "encoder|peak|MB|GB|latency|FPS|Error" || true

  # E8 at 3-bit (headline for Paper 2 baseline)
  $PY scripts/report_kv_memory.py --measure --encoder vitl --window 32 \
      --quantizer lattice_e8 --bits 3 --scale-bits 8 \
      2>&1 | tee "$HW_DIR/vitl_e8_b3.log" | grep -E "encoder|peak|MB|GB|latency|FPS|Error" || true

  # BW16 at 3-bit (Paper 2 headline)
  $PY scripts/report_kv_memory.py --measure --encoder vitl --window 32 \
      --quantizer lattice_bw16 --bits 3 --scale-bits 8 --group-size 16 \
      2>&1 | tee "$HW_DIR/vitl_bw16_b3.log" | grep -E "encoder|peak|MB|GB|latency|FPS|Error" || true

  echo "[done] hardware_benchmark in $(( (SECONDS-t0)/60 )) min"
  PASS=$((PASS+1))
else
  echo "[skip] hardware_benchmark"; SKIP=$((SKIP+1))
fi

# =============================================================================
echo "########## STAGE QUAL: Qualitative depth-map figures (4 panels) ##########"
# Fig-3-style side-by-side visualisations: RGB | FP32 | 4b | 3b | 2b for the
# headline quantiser (BW16), on all three benchmarks; plus a rotation-ablation
# panel showing why the Hadamard rotation is not optional. These are the
# qualitative equivalents of the numerical Pareto — reviewers who don't read
# TAE tables can eyeball the collapse at 2-bit and the rotation lift at 3-bit.
QUAL_DIR="$OUT/qualitative"
DUMP="$PY scripts/dump_depth_samples.py"
mkdir -p "$QUAL_DIR"

qual_run() {
  local name="$1"; shift
  local d="$QUAL_DIR/$name"
  # Skip if any strip already produced for this panel.
  if compgen -G "$d/strip_*.png" > /dev/null; then
    echo "[skip] qual/$name"; SKIP=$((SKIP+1)); return 0
  fi
  echo ""; echo "════════ [qual] $name  $(date '+%H:%M:%S') ════════"; echo "   $*"
  local t0=$SECONDS
  if $DUMP --output-dir "$d" "$@" 2>&1 | tee "$d.log" \
       | grep -E "Model|Surgery|Saved|Video|Error|Traceback"; then :; fi
  if compgen -G "$d/strip_*.png" > /dev/null; then
    echo "[done] qual/$name in $(( (SECONDS-t0)/60 )) min"; PASS=$((PASS+1))
  else
    echo "[FAIL] qual/$name — see $d.log"; FAIL=$((FAIL+1)); FAILED="$FAILED qual_$name"
  fi
}

# Panel 1: NYU indoor — headline BW16 bit-ladder. Static frames, no video.
#          Bits 8 (~FP16 storage), 4 (lossless), 3 (headline), 2 (collapse).
qual_run "nyu_bw16_ladder" --dataset nyuv2 --encoder vits \
    --quantizer lattice_bw16 --scale-bits 8 --group-size 16 \
    --num-frames 6 --bits 8 4 3 2 --no-qjl --rht-seed 0 --tag bw16

# Panel 2: KITTI outdoor — same ladder, single scene.
qual_run "kitti_bw16_ladder" --dataset kitti --encoder vits \
    --quantizer lattice_bw16 --scale-bits 8 --group-size 16 \
    --num-frames 6 --bits 8 4 3 2 --no-qjl --rht-seed 0 --tag bw16

# Panel 3: Sintel video — the temporal-collapse showcase.
#          --make-video so we get an MP4 that visually shows flicker under 2-bit.
qual_run "sintel_bw16_ladder" --dataset sintel --encoder vits \
    --quantizer lattice_bw16 --scale-bits 8 --group-size 16 \
    --num-frames 24 --bits 8 4 3 2 --no-qjl --rht-seed 0 --tag bw16 \
    --make-video

# Panel 4: Rotation ablation on Sintel — BW16 3-bit WITH vs WITHOUT rotation.
#          Same scene, same seed, so the difference is purely the RHT.
qual_run "sintel_bw16_b3_norot" --dataset sintel --encoder vits \
    --quantizer lattice_bw16 --scale-bits 8 --group-size 16 \
    --num-frames 12 --bits 3 --no-qjl --no-rotation --rht-seed 0 --tag norot

# =============================================================================
echo ""
echo "════════════════════════════════════════════════════════════════════════"
printf "PAPER 2 EXTEND COMPLETE   pass=%d  skip=%d  fail=%d   total=%d min\n" \
  "$PASS" "$SKIP" "$FAIL" $(( (SECONDS-T0)/60 ))
[ -n "$FAILED" ] && echo "FAILED:$FAILED"
echo ""
echo "Zip for download:"
echo "  cd $(pwd) && zip -r paper2_extend_results.zip $OUT"
echo "════════════════════════════════════════════════════════════════════════"
