#!/usr/bin/env bash
# Wait for the delta-coding runs to finish (shared GPU), then render the grids.
set -u
cd "$(dirname "$0")/.."
DRV=outputs/stream/delta_mixed_driver.log
while ! grep -q "ALL DONE" "$DRV" 2>/dev/null; do sleep 30; done
mkdir -p outputs/visuals
python -u -X utf8 scripts/visual_grid.py --mode stream  --out outputs/visuals/stream_grid.png  > outputs/visuals/stream.log 2>&1
python -u -X utf8 scripts/visual_grid.py --mode offline --out outputs/visuals/offline_grid.png > outputs/visuals/offline.log 2>&1
echo "VISUALS DONE"
