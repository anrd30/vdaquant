#!/bin/bash
# Unpack 7-Scenes and Neural RGBD (already downloaded to $C/recon_data) into the
# layout of StreamVGGT's mv_recon evaluator, as Spann3R's data guide describes:
#   7-Scenes: unzip, then SimpleRecon's 7scenes_preprocessing.py writes
#             frame-XXXXXX.depth.proj.png (depth registered to the colour camera)
#   NRGBD:    unzip
# SimpleRecon's script is fetched, not copied; we only point it at our folder,
# skip the train split, and keep every 20th test frame (all the benchmark reads
# at strides 20 and 200).  Runs at low CPU/IO priority.
#
#   C=<code_run dir> bash scripts/prep_recon_data.sh
set -eu
: "${C:?set C}"
D=$C/recon_data
SVG=$C/svg_tcc/StreamVGGT
PY=$C/svg_tcc/venv/bin/python
cd "$D"

"$PY" -c "import skimage, joblib" 2>/dev/null || PIP_CONFIG_FILE=/dev/null "$PY" -m pip install -q scikit-image joblib

echo "[$(date +%H:%M)] unzip 7-Scenes"
mkdir -p 7scenes
for s in chess fire heads office pumpkin redkitchen stairs; do
  [ -d 7scenes/$s ] || nice -n 19 unzip -q $s.zip -d 7scenes
  for z in 7scenes/$s/seq-*.zip; do
    [ -e "$z" ] || continue
    nice -n 19 unzip -q -o "$z" -d 7scenes/$s && rm "$z"
  done
done

echo "[$(date +%H:%M)] unzip NRGBD"
[ -d neural_rgbd_raw ] || nice -n 19 unzip -q neural_rgbd_data.zip -d neural_rgbd_raw
NR=$(dirname "$(dirname "$(find neural_rgbd_raw -maxdepth 4 -name poses.txt | head -n 1)")")
echo "NRGBD root: $NR"

echo "[$(date +%H:%M)] register 7-Scenes depth"
wget -q -O sr7_orig.py https://raw.githubusercontent.com/nianticlabs/simplerecon/main/data_scripts/7scenes_preprocessing.py
"$PY" - "$D/7scenes" <<'EOF'
import sys
s = open('sr7_orig.py').read()
def sub(a, b):
    global s
    assert s.count(a) == 1, a
    s = s.replace(a, b)
sub("src_folder = '/mnt/res_nas/shared/datasets/7scenes'", "src_folder = %r" % sys.argv[1])
sub("process_frames('TrainSplit.txt')", "pass")
sub("depth_files = [f for f in files if f.endswith('depth.png')]",
    "depth_files = [f for f in files if f.endswith('depth.png') and int(f[6:12]) % 20 == 0]")
sub("Parallel(n_jobs=7", "Parallel(n_jobs=2")
open('sr7.py', 'w').write(s)
print('patched')
EOF
nice -n 19 ionice -c3 "$PY" sr7.py

mkdir -p "$SVG/data/eval"
ln -sfn "$D/7scenes" "$SVG/data/eval/7scenes"
ln -sfn "$D/$NR" "$SVG/data/eval/neural_rgbd"
echo "7-Scenes registered depth files: $(find "$D/7scenes" -name '*.depth.proj.png' | wc -l)"
echo "NRGBD scenes: $(ls "$SVG/data/eval/neural_rgbd" | tr '\n' ' ')"
echo "[$(date +%H:%M)] PREP DONE"
