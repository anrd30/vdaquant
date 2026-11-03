#!/usr/bin/env bash
# Night queue (sequential, one GPU job at a time):
#  1. regression: refactored delta code must reproduce the stored Bonn balloon numbers
#  2. codec probes (skip blocks, motion compensation) on Bonn and Sintel
#  3. fidelity-to-FP32 reruns of the key configs on Bonn and Sintel, ViT-S
set -u
cd "$(dirname "$0")/.."
O=outputs/stream
echo "[$(date +%H:%M:%S)] 1. regression"
python -u -X utf8 scripts/stream_eval.py --dataset bonn --max-scenes 1 --configs int3-g16::d8:bin-g16 \
  --out $O/regression_balloon.json > $O/regression_balloon.log 2>&1 || echo "  FAILED regression"
python -I -X utf8 - <<'EOF'
import json
new = json.load(open('outputs/stream/regression_balloon.json'))['results'][0]['per_scene']['balloon']
old = [r for r in json.load(open('outputs/stream/bonn_vits_delta_mixed.json'))['results']
       if r.get('config') == 'int3-g16::d8:bin-g16'][0]['per_scene']['balloon']
for k in ('delta1', 'tae', 'tae_covis'):
    print(f"  regression {k}: old={old[k]:.6f} new={new[k]:.6f} diff={new[k]-old[k]:+.2e}")
EOF
echo "[$(date +%H:%M:%S)] 2. codec probes"
for ds in bonn sintel; do
  python -u -X utf8 scripts/stream_codec_probe.py --dataset $ds --frames 150 \
    --out $O/codec_probe_$ds.json > $O/codec_probe_$ds.log 2>&1 || echo "  FAILED probe $ds"
  grep "^layer" $O/codec_probe_$ds.log
done
echo "[$(date +%H:%M:%S)] 3. fidelity reruns"
CFG="fp32,zero,int2-g16,int3-g16::d8:bin-g16,int4-g16::d8:bin-g16"
for ds in bonn sintel; do
  python -u -X utf8 scripts/stream_eval.py --dataset $ds --encoder vits --configs "$CFG" \
    --out $O/${ds}_vits_fidelity.json > $O/${ds}_vits_fidelity.log 2>&1 || echo "  FAILED fidelity $ds"
  tail -6 $O/${ds}_vits_fidelity.log
done
echo "[$(date +%H:%M:%S)] ALL DONE"
