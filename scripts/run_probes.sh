#!/usr/bin/env bash
# =============================================================================
# PROBES — 13 experiments that each try to FIND something, run serially.
#
# These exist because the F28 fix opened a knob that did not previously exist:
# the scale group is now decoupled from the lattice dimension (--group-size),
# and the K/V budget can be split (--v-bits). Synthetic measurements say the
# E8 coding gain is zero at group 8 and rises monotonically to its theoretical
# +0.65 dB as the scale is amortized further, and that rotation and fine
# grouping are substitutes. NONE of that has been tested on real KV cache.
#
# Every probe below states a HYPOTHESIS and a FALSIFIER. A probe that comes
# back against the hypothesis is a result, not a failure -- record it either
# way. Probes are ordered cheapest-and-most-decisive first, so a partial run
# still answers the important questions.
#
# Results go to outputs/probes/<probe>, kept separate from outputs/finals and
# outputs/rerun_f28 so nothing is overwritten and every comparison is auditable.
#
# Usage (standalone):
#   cd vdaquant && nohup bash scripts/run_probes.sh > probes.log 2>&1 &
# Usually you want the delta run first -- use scripts/run_everything.sh instead.
#
# Rough time: ~8-12 h on one modern GPU. Fully resumable: re-running skips
# anything that already produced a results file. Delete a dir to force a redo.
#
# Conventions match run_finals.sh so everything is comparable: --no-qjl (F15),
# --scale-bits 8 unless the probe varies it (F18), --rht-seed 0 unless the
# probe sweeps seeds, vits unless the probe varies encoder.
# =============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
SUITE="python scripts/run_pareto_benchmark_suite.py"
OUT="outputs/probes"
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

banner() { echo ""; echo "##########################################################"; echo "# $*"; echo "##########################################################"; }

# =============================================================================
banner "P1 — Does the E8 coding gain appear on real KV cache as the scale group coarsens?"
# HYPOTHESIS: at g=8 E8 ties grouped scalar; the gain grows with g (+0.32 dB at
#   16, +0.49 at 32, +0.57 at 64 on Gaussian input) because per-group
#   max-normalization stops substituting for lattice shaping (ledger F28).
# FALSIFIER: delta1 for E8 tracks scalar_g8 at ALL group sizes, or E8 gets
#   WORSE as g grows -> the synthetic mechanism does not transfer to real KV
#   cache, and the lattice has no operating point anywhere. That kills the
#   lattice direction outright, which is worth knowing definitively.
# NOTE: coarser g also LOWERS effective bits (3b: g=8 -> 4.00, g=64 -> 3.125),
#   so E8 must be read against scalar_g8 AT THE SAME g, not against g=8.
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

# =============================================================================
banner "P2 — Are rotation and fine grouping substitutes on real data?"
# HYPOTHESIS: they solve the same problem (outlier containment), so doing both
#   is paying twice. On synthetic heavy-tailed input, rotation HURT by 2.9 dB at
#   g=8 and HELPED by 2.8 dB at g=512. Predict: no-rotation wins at g=8,
#   rotation wins at g=64. This reframes DG-2's "rotation is rate-dependent" as
#   granularity-dependent.
# FALSIFIER: rotation helps (or hurts) uniformly across g -> the two are
#   independent knobs and the substitution story is wrong.
# NOTE: this is the probe most likely to yield a citable, practical claim, and
#   also the one whose territory is most occupied (GyRot arXiv:2607.27694
#   establishes the interaction for LLM weights). Frame any result as
#   verification in a new domain, never as discovery.
for g in 8 64; do
  run "p2_rot_nyu_g$g"   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
      --group-size "$g" --scale-bits 8 --bits 3 --no-qjl --rotation --rht-seed 0 --max-samples 654
  run "p2_norot_nyu_g$g" --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
      --group-size "$g" --scale-bits 8 --bits 3 --no-qjl --no-rotation --max-samples 654
  run "p2_rot_kitti_g$g"   --dataset kitti --eval-mode groundtruth --quantizer lattice_e8 \
      --group-size "$g" --scale-bits 8 --bits 3 --no-qjl --rotation --rht-seed 0 --max-samples 1000
  run "p2_norot_kitti_g$g" --dataset kitti --eval-mode groundtruth --quantizer lattice_e8 \
      --group-size "$g" --scale-bits 8 --bits 3 --no-qjl --no-rotation --max-samples 1000
done

# =============================================================================
banner "P3 — At a fixed bit budget, is it better to spend bits on PAYLOAD or on SCALES?"
# This is the probe with the clearest practical payoff. The field's default
#   (group 8, 8-bit scales) spends a FULL effective bit on metadata -- 25% of a
#   4-bit budget. Synthetically, moving that bit into payload bought 1.9 dB.
# HYPOTHESIS: 4-bit payload at g=64 (4.125 eff bits) beats 3-bit at g=8
#   (4.00 eff bits) despite costing only 0.125 more effective bits.
# FALSIFIER: the g=8 config wins -> fine grouping is genuinely worth its
#   metadata on real KV cache, and the all-inclusive framing loses its teeth.
run p3_isorate_nyu_4b_g64   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
    --group-size 64 --scale-bits 8 --bits 4 --no-qjl --rht-seed 0 --max-samples 654
run p3_isorate_kitti_4b_g64 --dataset kitti --eval-mode groundtruth --quantizer lattice_e8 \
    --group-size 64 --scale-bits 8 --bits 4 --no-qjl --rht-seed 0 --max-samples 1000
run p3_isorate_nyu_sg_4b_g64   --dataset nyuv2 --eval-mode groundtruth --quantizer scalar_g8 \
    --group-size 64 --scale-bits 8 --bits 4 --no-qjl --rht-seed 0 --max-samples 654
run p3_isorate_kitti_sg_4b_g64 --dataset kitti --eval-mode groundtruth --quantizer scalar_g8 \
    --group-size 64 --scale-bits 8 --bits 4 --no-qjl --rht-seed 0 --max-samples 1000

# =============================================================================
banner "P4 — Should keys and values get the SAME number of bits?"
# Keys perturb the softmax weights (an error there is renormalized and can
#   partially cancel); values perturb the output directly (no such damping).
#   Nobody has asked this for video depth.
# All three configs below cost EXACTLY 4.0 effective bits, so this is a clean
#   three-way at matched rate: K4/V2, K2/V4, and the K3/V3 baseline already run.
# HYPOTHESIS: the split matters, and K4/V2 differs measurably from K2/V4.
# FALSIFIER: all three land within noise -> the K/V budget split is irrelevant
#   here, which is itself worth one sentence and saves future work.
run p4_kv_k4v2_nyu   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
    --bits 4 --v-bits 2 --scale-bits 8 --no-qjl --rht-seed 0 --max-samples 654
run p4_kv_k2v4_nyu   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
    --bits 2 --v-bits 4 --scale-bits 8 --no-qjl --rht-seed 0 --max-samples 654
run p4_kv_k4v2_kitti --dataset kitti --eval-mode groundtruth --quantizer lattice_e8 \
    --bits 4 --v-bits 2 --scale-bits 8 --no-qjl --rht-seed 0 --max-samples 1000
run p4_kv_k2v4_kitti --dataset kitti --eval-mode groundtruth --quantizer lattice_e8 \
    --bits 2 --v-bits 4 --scale-bits 8 --no-qjl --rht-seed 0 --max-samples 1000

# =============================================================================
banner "P5 — Does 2-bit become viable once scales stop eating the budget?"
# 2-bit is where the field wants to be and where our 2-bit runs collapsed. At
#   g=8 a 2-bit payload really costs 3.0 effective bits; at g=64 it costs 2.125.
# HYPOTHESIS: the collapse at 2 bits is partly a metadata problem, so 2b/g=64
#   degrades more gracefully than 2b/g=8 despite the LOWER effective rate.
# FALSIFIER: 2b/g=64 is worse or equally collapsed -> the 2-bit failure is a
#   genuine resolution floor, not a rate-allocation artifact.
for g in 8 64; do
  run "p5_2bit_nyu_g$g"   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
      --group-size "$g" --scale-bits 8 --bits 2 --no-qjl --rht-seed 0 --max-samples 654
  run "p5_2bit_kitti_g$g" --dataset kitti --eval-mode groundtruth --quantizer lattice_e8 \
      --group-size "$g" --scale-bits 8 --bits 2 --no-qjl --rht-seed 0 --max-samples 1000
done

# =============================================================================
banner "P6 — Are fp16 scales worth it once they are amortized 8x further?"
# F18 found scale_bits 8 == 16 at g=8, where fp16 scales cost a full 2.0
#   effective bits. At g=64 they cost only 0.25, so the trade changes entirely.
# HYPOTHESIS: at g=64, scale_bits=16 is worth its 0.125-bit premium because a
#   coarse group's scale must cover a much wider spread and quantizing that
#   scale to uint8 starts to bite.
# FALSIFIER: still no difference -> scale precision is a non-issue at every
#   granularity, and F18 generalizes.
run p6_sb16_nyu_g64   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
    --group-size 64 --scale-bits 16 --bits 3 --no-qjl --rht-seed 0 --max-samples 654
run p6_sb16_kitti_g64 --dataset kitti --eval-mode groundtruth --quantizer lattice_e8 \
    --group-size 64 --scale-bits 16 --bits 3 --no-qjl --rht-seed 0 --max-samples 1000

# =============================================================================
banner "P7 — Does the rotation seed matter MORE when the scale is coarse?"
# The headline CIs treat the rotation seed as a random effect. If rotation is
#   load-bearing only at coarse g (P2), then seed variance should GROW with g.
# HYPOTHESIS: seed spread at g=64 is materially wider than at g=8.
# FALSIFIER: spread is the same -> rotation is not doing the work the
#   substitution story assigns it, and P2's interpretation needs revisiting.
# This is the internal-consistency check on P2; run them together.
for s in 1 2; do
  run "p7_seed${s}_nyu_g64" --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_e8 \
      --group-size 64 --scale-bits 8 --bits 3 --no-qjl --rht-seed "$s" --max-samples 654
  run "p7_seed${s}_kitti_g64" --dataset kitti --eval-mode groundtruth --quantizer lattice_e8 \
      --group-size 64 --scale-bits 8 --bits 3 --no-qjl --rht-seed "$s" --max-samples 1000
done

# =============================================================================
banner "P8 — D4 vs E8 at TRULY matched effective bits"
# The paper compares D4 and E8 at matched NOMINAL payload, which silently hands
#   D4 two extra effective bits (its group of 4 doubles the scale metadata).
#   At scale_bits=8: D4 3b -> 5.0 eff; E8 4b at g=8 -> 5.0 eff. Matched.
# HYPOTHESIS: D4 loses, but on RATE ALLOCATION, not lattice geometry -- so the
#   paper's conclusion direction survives while its stated mechanism does not.
# FALSIFIER: D4 wins at matched rate -> lattice dimension really does matter and
#   the metadata explanation is incomplete.
run p8_d4_matched_nyu   --dataset nyuv2 --eval-mode groundtruth --quantizer lattice_d4 \
    --scale-bits 8 --bits 3 --no-qjl --rht-seed 0 --max-samples 654
run p8_d4_matched_kitti --dataset kitti --eval-mode groundtruth --quantizer lattice_d4 \
    --scale-bits 8 --bits 3 --no-qjl --rht-seed 0 --max-samples 1000

# =============================================================================
banner "P9 — Does quantization error COMPOUND over longer temporal windows?"
# Unique to video, and unasked for depth. A KV cache is reused across frames, so
#   error may accumulate with window length rather than staying constant.
# HYPOTHESIS: the FP32-to-quantized gap WIDENS with window length.
# FALSIFIER: the gap is flat in window length -> error does not accumulate, the
#   cache is effectively refreshed, and long-video deployment is safe. Either
#   answer is publishable and neither is currently known.
for w in 8 16 32; do
  run "p9_window${w}" --dataset sintel --eval-mode temporal --quantizer lattice_e8 \
      --scale-bits 8 --bits 4 3 --no-qjl --rht-seed 0 --temporal-window "$w" \
      --max-samples 2000 --max-scenes 12 --tae-covis-tau 0.05
done

# =============================================================================
banner "P10 — Does the co-visibility TAE finding survive at coarse granularity?"
# The masked-TAE result is the most defensible thing in the project. It was
#   established at g=8 only. If it is granularity-dependent, that is a
#   limitation the paper must state itself before a reviewer finds it.
# HYPOTHESIS: masked TAE stays monotone in bit-rate at g=64 (the finding is
#   about the METRIC, so it should not care about the quantizer's grouping).
# FALSIFIER: monotonicity breaks at g=64 -> the centrepiece claim is narrower
#   than stated and must be scoped explicitly.
run p10_sintel_g64 --dataset sintel --eval-mode temporal --quantizer lattice_e8 \
    --group-size 64 --scale-bits 8 --bits 4 3 2 --no-qjl --rht-seed 0 \
    --temporal-window 32 --max-samples 2000 --max-scenes 23 --tae-covis-tau 0.05

# =============================================================================
banner "P11 — Does the granularity result transfer across model scale?"
# Only meaningful once the F27 identity control is clean (run by the delta
#   script). If surgery fidelity is broken at vitl, treat this as UNINTERPRETABLE
#   rather than as evidence -- a plausible number from an unfaithful pipeline is
#   still not evidence (the mistake F26 made).
# HYPOTHESIS: the g=8 -> g=64 improvement holds at vitl.
# FALSIFIER: it reverses at vitl -> the effect is encoder-specific and no
#   general recommendation can be made.
for g in 8 64; do
  run "p11_vitl_nyu_g$g" --dataset nyuv2 --eval-mode groundtruth --encoder vitl \
      --quantizer lattice_e8 --group-size "$g" --scale-bits 8 --bits 3 --no-qjl \
      --rht-seed 0 --max-samples 200
done

# =============================================================================
banner "P12 — Do the best settings found on NYU transfer to KITTI?"
# Indoor 0.1-10 m vs outdoor 0.1-80 m. If the optimal granularity differs by
#   dataset, no single recommendation is defensible and the paper must say so.
# HYPOTHESIS: the granularity trend is monotone on both, so the recommendation
#   transfers even if absolute numbers differ.
# FALSIFIER: the optimal g differs between datasets -> report per-domain, and
#   drop any universal claim.
# NOTE: this probe needs NO new runs -- it is an ANALYSIS of P1's outputs across
#   both datasets. Left here as a named deliverable so it is not forgotten.
echo "[analysis-only] P12 is computed from P1 outputs; no GPU run required."

# =============================================================================
banner "P13 — Is the TAE effect about quantization, or about losing detail in general?"
# Gaussian blur on FP32 predictions -- a degradation with nothing to do with
#   quantization. Built long ago (F16/F25 motivation), never run.
# HYPOTHESIS: unmasked TAE FALLS as blur rises (the metric "improves" as the
#   prediction gets less detailed) while masked TAE rises. That would establish
#   the gameability as a general law, not a quantization artifact -- and it is
#   the strongest possible support for the co-visibility contribution.
# FALSIFIER: blur does not move unmasked TAE -> the mechanism is specific to
#   quantization noise and the general claim must be withdrawn.
if [ ! -f "$OUT/p13_blur/blur_degradation_results.json" ]; then
  mkdir -p "$OUT/p13_blur"
  if python scripts/run_blur_degradation.py --dataset sintel \
       --sigmas 0 1 2 4 8 --temporal-window 32 --max-scenes 12 \
       --max-samples 2000 --tae-covis-tau 0.05 --output-dir "$OUT/p13_blur" \
       > "$OUT/p13_blur/run.log" 2>&1; then
    echo "[done] p13_blur"; PASS=$((PASS+1))
  else
    echo "[FAIL] p13_blur — see $OUT/p13_blur/run.log"; FAIL=$((FAIL+1)); FAILED="$FAILED p13_blur"
  fi
else
  echo "[skip] p13_blur"; SKIP=$((SKIP+1))
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "PROBES COMPLETE   pass=$PASS  skip=$SKIP  fail=$FAIL"
[ -n "$FAILED" ] && echo "FAILED:$FAILED"
echo ""
echo "Zip for download:  cd $(pwd) && zip -r probes_results.zip $OUT"
echo ""
echo "Read the results against each probe's stated FALSIFIER, not for a"
echo "favourable number. P1 and P2 are the decisive ones: if P1 shows no"
echo "granularity trend on real KV cache, the lattice direction is closed."
echo "════════════════════════════════════════════════════════════"
