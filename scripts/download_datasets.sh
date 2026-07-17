#!/usr/bin/env bash
# =============================================================================
# Dataset downloader for vdaquant GT evaluation.
#
# Every URL here was HEAD-verified (HTTP 200 + Content-Length) and every host
# was confirmed to serve HTTP 206 partial content, which is what makes the
# aria2c parallel-chunk trick below work.
#
# Usage:
#   bash scripts/download_datasets.sh nyuv2 kitti davis sintel   # ~9 GB total
#   bash scripts/download_datasets.sh kitti-eigen                # +71 GB (see note)
#   bash scripts/download_datasets.sh all
#
# Colab: run inside a `nohup ... &` or tmux so a dropped tab doesn't kill it,
# and download to /content (local SSD), NOT to mounted Drive — Drive writes are
# slow and will bottleneck the transfer far below network speed.
# =============================================================================
set -uo pipefail

DATA_DIR="${DATA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/benchmark_data}"
mkdir -p "$DATA_DIR"

# --- The download trick -------------------------------------------------------
# aria2c with 16 parallel connections per file (-x16) split into 16 chunks
# (-s16). On these hosts this is typically 5-10x faster than wget/curl, which
# use a single connection and leave most of the pipe idle.
#   -c                  resume a partial file instead of restarting (timeout insurance)
#   --max-tries=0       retry forever rather than dying on a transient 5xx
#   --retry-wait=5      back off 5s between retries
#   --file-allocation=none   don't pre-allocate (important on Colab/ext4 — otherwise
#                            aria2 stalls for minutes zeroing a 14GB file first)
#   --timeout=60 --connect-timeout=30   fail a stuck socket fast so retry kicks in
# Falls back to `wget -c` if aria2c isn't installed (still resumable, just slower).
ARIA_OPTS=(-x16 -s16 -k1M -c --max-tries=0 --retry-wait=5 \
           --file-allocation=none --timeout=60 --connect-timeout=30 \
           --summary-interval=10 --console-log-level=warn)

have_aria() { command -v aria2c >/dev/null 2>&1; }

fetch() {  # fetch <url> <dest_dir>
  local url="$1" dest="$2"
  mkdir -p "$dest"
  local fname; fname="$(basename "${url%%\?*}")"
  if [ -f "$dest/$fname" ] && [ ! -f "$dest/$fname.aria2" ]; then
    echo "  [skip] $fname already present"
    return 0
  fi
  echo "  [get ] $fname"
  if have_aria; then
    aria2c "${ARIA_OPTS[@]}" -d "$dest" -o "$fname" "$url"
  else
    echo "  [warn] aria2c not found -> falling back to wget (much slower)."
    echo "         Install it for the parallel-chunk speedup:  apt-get install -y aria2"
    wget -c --tries=0 --timeout=60 -O "$dest/$fname" "$url"
  fi
}

unzip_once() {  # unzip_once <zip> <dest> <sentinel_path>
  local zip="$1" dest="$2" sentinel="$3"
  if [ -e "$sentinel" ]; then echo "  [skip] already extracted -> $sentinel"; return 0; fi
  echo "  [unzip] $(basename "$zip")"
  unzip -q -o "$zip" -d "$dest"
}

# --- NYUv2 (labeled test split, 654 imgs, RGB + GT depth) ---------------------
# This is the one scripts/datasets_gt.py already downloads automatically; this
# target just pre-fetches it faster than the script's serial urllib download.
# NOTE: the .mat is MATLAB v7.3 (HDF5) -> needs h5py, not scipy.io.loadmat.
get_nyuv2() {
  echo "== NYUv2 labeled (2.8 GB) =="
  local d="$DATA_DIR/nyuv2_gt"
  fetch "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat" "$d"
  fetch "http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat" "$d"
  echo "   -> loader: scripts/datasets_gt.py::load_nyuv2_gt_test_split (IMPLEMENTED)"
}

# --- KITTI depth (RECOMMENDED, self-contained) --------------------------------
# data_depth_selection.zip is the smart pick: 1.9 GB and it contains BOTH the
# RGB images AND the GT depth for 1000 val images, verified by reading the zip's
# central directory:
#     depth_selection/val_selection_cropped/image/            (1000 RGB)
#     depth_selection/val_selection_cropped/groundtruth_depth/(1000 GT depth)
#     depth_selection/val_selection_cropped/intrinsics/
# The 14 GB data_depth_annotated.zip contains depth maps ONLY (no RGB) and is
# NOT needed unless you are training. No KITTI login required — the S3 mirror
# below is public, unlike the cvlibs.net download page.
get_kitti() {
  echo "== KITTI depth_selection (1.9 GB, self-contained RGB+GT, 1000 val imgs) =="
  local d="$DATA_DIR/kitti"
  fetch "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip" "$d"
  unzip_once "$d/data_depth_selection.zip" "$d" "$d/depth_selection/val_selection_cropped/image"
  echo "   -> loader: NOT YET WRITTEN (see note at bottom)"
}

# --- KITTI Eigen split (ONLY if you need to match published numbers) ----------
# 71 GB, because each raw sync zip bundles velodyne point clouds, all 4 cameras
# and IMU just to get you the 697 image_02 frames the Eigen split actually uses.
# Awful bytes-per-useful-image ratio. Use get_kitti above unless a reviewer
# specifically needs Eigen-split comparability with published depth papers.
get_kitti_eigen() {
  echo "== KITTI Eigen test split raw (71 GB — see comment, prefer 'kitti') =="
  local d="$DATA_DIR/kitti_raw"
  local list; list="$(dirname "${BASH_SOURCE[0]}")/kitti_eigen_test_urls.txt"
  [ -f "$list" ] || { echo "  [err] missing $list"; return 1; }
  if have_aria; then
    # -j4: 4 files at once, each with 16 connections. Do not raise much higher;
    # S3 will start throttling and you net out slower.
    aria2c "${ARIA_OPTS[@]}" -j4 -d "$d" -i "$list"
  else
    while read -r u; do fetch "$u" "$d"; done < "$list"
  fi
  echo "  [unzip] extracting $(ls "$d"/*.zip 2>/dev/null | wc -l) archives..."
  for z in "$d"/*.zip; do unzip -q -n "$z" -d "$d"; done
  echo "   -> loader: NOT YET WRITTEN"
}

# --- DAVIS 2017 (480p) --------------------------------------------------------
# IMPORTANT: DAVIS has NO ground-truth depth — it is a video *segmentation*
# dataset. You cannot compute AbsRel/delta1 on it. It is only useful here as
# real video for the TAE temporal-consistency metric (which is exactly what the
# VDA paper uses it for). Do not run --eval-mode groundtruth against it.
get_davis() {
  echo "== DAVIS 2017 480p trainval (0.78 GB, video only — NO depth GT) =="
  local d="$DATA_DIR/davis"
  fetch "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip" "$d"
  unzip_once "$d/DAVIS-2017-trainval-480p.zip" "$d" "$d/DAVIS/JPEGImages"
  echo "   -> use for: TAE (temporal flicker) only. No AbsRel/delta1 possible."
}

# --- MPI Sintel (depth) -------------------------------------------------------
# Real GT depth in metres + camera intrinsics/extrinsics, and it is genuine
# video — so it is the one dataset that can exercise BOTH depth accuracy AND
# the TAE temporal metric. Best value for the paper's temporal claim.
get_sintel() {
  echo "== MPI-Sintel depth training (1.5 GB, GT depth + video) =="
  local d="$DATA_DIR/sintel"
  fetch "https://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip" "$d"
  unzip_once "$d/MPI-Sintel-depth-training-20150305.zip" "$d" "$d/training"
  echo "   -> loader: NOT YET WRITTEN"
}

# --- ScanNet ------------------------------------------------------------------
# Cannot be scripted: ScanNet requires emailing a signed Terms-of-Use agreement
# from an institutional address to scannet@googlegroups.com, after which they
# send you download-scannet.py. There is no public mirror to point at, and
# working around the gate would violate their ToU. Budget ~1 week for the reply.
get_scannet() {
  cat <<'EOF'
== ScanNet — MANUAL STEP REQUIRED, cannot be automated ==
   ScanNet is gated behind a signed Terms-of-Use agreement:
     1. Fill the ToU form linked at http://www.scan-net.org/  (institutional email required)
     2. Email it to scannet@googlegroups.com
     3. They reply with download-scannet.py (allow ~1 week; check your mail isn't bouncing)
   There is no public mirror. Do not use a scraped copy — it breaks their ToU
   and a reviewer can ask how you obtained it.
EOF
}

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(nyuv2 kitti davis sintel)
if [ "${targets[0]}" = "all" ]; then targets=(nyuv2 kitti davis sintel kitti-eigen scannet); fi

have_aria || echo "!! aria2c not installed — run: apt-get update && apt-get install -y aria2 (Colab) / brew install aria2"

for t in "${targets[@]}"; do
  case "$t" in
    nyuv2)        get_nyuv2 ;;
    kitti)        get_kitti ;;
    kitti-eigen)  get_kitti_eigen ;;
    davis)        get_davis ;;
    sintel)       get_sintel ;;
    scannet)      get_scannet ;;
    *) echo "unknown target: $t (valid: nyuv2 kitti kitti-eigen davis sintel scannet all)" ;;
  esac
  echo ""
done

cat <<'EOF'
=============================================================================
DOWNLOADING IS NOT ENOUGH — loaders still need writing.

scripts/datasets_gt.py currently implements ONLY load_nyuv2_gt_test_split.
run_pareto_benchmark_suite.py deliberately raises on
`--eval-mode groundtruth --dataset {kitti,sintel,davis,scannet}` rather than
silently evaluating against fake/proxy data (docs/optimization_ledger.md T2).
So having the bytes on disk does not yet give you numbers — each dataset needs
its own loader returning {rgb, depth, valid_mask} plus a --dataset wiring.

Per-dataset notes:
  KITTI  — depth PNGs are uint16, divide by 256.0 for metres; 0 = invalid.
           Standard eval uses the Garg/Eigen crop and caps depth at 80m.
  Sintel — depth is a custom .dpt float format; the zip ships an SDK with a
           reader. Real video => can exercise TAE as well as accuracy.
  DAVIS  — no depth GT at all. TAE only.
=============================================================================
EOF
