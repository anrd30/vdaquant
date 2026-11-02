#!/usr/bin/env bash
# Fair lattice-vs-scalar comparison in the HARD setting: backbone + temporal
# attention quantized, offline T=32 windows, Sintel, ViT-S, 8-bit scales,
# RHT seed 0.  Usage: bash scripts/run_lattice_vs_int.sh [max_scenes]
set -u
cd "$(dirname "$0")/.."
SC=${1:-8}
OUT=outputs/lattice_vs_int_s${SC}
mkdir -p "$OUT"
COMMON="--dataset sintel --eval-mode temporal --scale-bits 8 --no-qjl --rht-seed 0 --temporal-window 32 --max-samples 2000 --max-scenes $SC --tae-covis-tau 0.05"
run() {  # tag quantizer group bits...
  local tag=$1 q=$2 g=$3; shift 3
  echo "[$(date +%H:%M:%S)] $tag ($q g=$g bits=$*)"
  python -u -X utf8 scripts/run_pareto_benchmark_suite.py $COMMON --quantizer "$q" --group-size "$g" --bits "$@" \
    --output-dir "$OUT/$tag" > "$OUT/$tag.log" 2>&1 || echo "  FAILED $tag (see $OUT/$tag.log)"
  grep -E "delta1" "$OUT/$tag.log" | grep -v "N=" | sed 's/^/    /'
}
PHASE=${2:-all}
if [ "$PHASE" = all ] || [ "$PHASE" = 1 ]; then
run bw16_g16  lattice_bw16 16 4 3 2
run e8_g16    lattice_e8   16 3 2
run int_g16   scalar_g8    16 3 2
run bw16_g64  lattice_bw16 64 4 3
run int_g64   scalar_g8    64 4 3
run int_tensor scalar      64 4 3
fi
if [ "$PHASE" = all ] || [ "$PHASE" = 2 ]; then
run bw16_g64  lattice_bw16 64 3
run e8_g64    lattice_e8   64 3
run bw16_g16  lattice_bw16 16 4 3 2
run int_g64b2 scalar_g8    64 2
run bw16_g64b4 lattice_bw16 64 4
run e8_g64b4  lattice_e8   64 4
fi
if [ "$PHASE" = 3 ]; then          # E8 vs INT around 2.1-3.1 bits (packed)
run int_g64b2 scalar_g8    64 2
run e8_g64b2  lattice_e8   64 2
run e8_g32    lattice_e8   32 3
fi
echo "[$(date +%H:%M:%S)] ALL DONE"
