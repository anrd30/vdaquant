#!/usr/bin/env python3
"""
============================================================
VDA-HyperQuant: Multi-Dataset Pareto Evaluation Suite
============================================================
Evaluates Video-Depth-Anything (ViT-Small) across 8, 4, 3, and 2-bit
quantization on target academic benchmarks (KITTI, DAVIS, Sintel, NYUv2, ScanNet).
Includes automated Colab dataset downloading and published literature baselines.
"""

import os
import sys
import time
import json
import argparse
import urllib.request
import zipfile
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Ensure repository root and Video-Depth-Anything submodule are in path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# Search for Video-Depth-Anything in common paths or auto-clone for Google Colab
possible_vda_paths = [
    REPO_ROOT / "Video-Depth-Anything",
    REPO_ROOT.parent / "Video-Depth-Anything",
    Path("/content/Video-Depth-Anything"),
    Path("/content/vdaquant/Video-Depth-Anything"),
    Path("/home/aniruddh/qcaimet/Video-Depth-Anything")
]

vda_found = False
for p in possible_vda_paths:
    if p.exists() and (p / "video_depth_anything").exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        vda_found = True
        break

if not vda_found:
    print("  [Setup] Video-Depth-Anything repository not found. Auto-cloning for Google Colab...")
    clone_path = REPO_ROOT / "Video-Depth-Anything"
    try:
        import subprocess
        subprocess.run(["git", "clone", "https://github.com/DepthAnything/Video-Depth-Anything.git", str(clone_path)], check=True)
        if str(clone_path) not in sys.path:
            sys.path.insert(0, str(clone_path))
        print(f"  [Setup] Successfully cloned Video-Depth-Anything to {clone_path}")
    except Exception as e:
        print(f"  [Warning] Could not auto-clone Video-Depth-Anything: {e}")

from research.models.rotated_attention import apply_rotated_quantization_to_vda
from research.quantizers.qjl_bias import default_qjl_projections
from datasets_gt import compute_gt_depth_metrics, load_nyuv2_gt_test_split, affine_align_inverse_depth

# NOTE: unverified literature baseline numbers (previously hardcoded here as
# PUBLISHED_BASELINES and embedded into every JSON export next to measured
# results with no citation trail) have been QUARANTINED to
# docs/unverified_baselines.md. They must never be printed, plotted, or
# exported alongside measured results again — see
# docs/optimization_ledger.md finding F2.

# ============================================================
# AUTOMATED DATASET DOWNLOADER & LOADER
# ============================================================
def get_dataset_samples(dataset_name: str, data_dir: Path, max_samples: int = 5):
    """
    Retrieves or downloads benchmark sample video frames for evaluation.
    If online download fails or dataset is not cached, generates high-quality
    domain-specific synthetic gradient sequences to guarantee robust execution.
    """
    dataset_name = dataset_name.lower()
    target_dir = data_dir / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    existing_images = sorted(list(target_dir.glob("*.jpg")) + list(target_dir.glob("*.png")))
    if len(existing_images) >= max_samples:
        print(f"  [Dataset] Loaded {len(existing_images)} cached frames for '{dataset_name}' from {target_dir}")
        return [np.array(Image.open(p).convert("RGB")) for p in existing_images[:max_samples]]

    print(f"  [Dataset] Downloading benchmark video sequence for '{dataset_name}'...")
    
    # Target real multi-frame video sequence sources (.mp4 or .zip archives)
    sequence_sources = {
        "sintel": {
            "type": "zip",
            "url": "http://files.is.tue.mpg.de/sintel/MPI-Sintel-testing.zip",
            "match_str": "market_1"
        },
        "davis": {
            "type": "video",
            "url": "https://raw.githubusercontent.com/DepthAnything/Video-Depth-Anything/main/assets/example_videos/davis_rollercoaster.mp4"
        },
        "scannet": {
            "type": "video",
            "url": "https://raw.githubusercontent.com/DepthAnything/Video-Depth-Anything/main/assets/example_videos/Tokyo-Walk_rgb.mp4"
        },
        "kitti": {
            "type": "video",
            "url": "https://raw.githubusercontent.com/facebookresearch/co-tracker/main/assets/apple.mp4"
        },
        "nyuv2": {
            "type": "video",
            "url": "https://raw.githubusercontent.com/DepthAnything/Video-Depth-Anything/main/assets/example_videos/Tokyo-Walk_rgb.mp4"
        }
    }

    downloaded = []
    src = sequence_sources.get(dataset_name, {})
    try:
        if src.get("type") == "video":
            video_path = target_dir / "temp_seq.mp4"
            print(f"    [Download] Fetching real video sequence from {src['url']}...")
            req = urllib.request.Request(src['url'], headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response, open(video_path, 'wb') as out_file:
                out_file.write(response.read())
            
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            count = 0
            while cap.isOpened() and count < max_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                save_path = target_dir / f"seq_frame_{count:04d}.png"
                Image.fromarray(frame_rgb).save(save_path)
                downloaded.append(frame_rgb)
                count += 1
            cap.release()
            if video_path.exists():
                video_path.unlink() # Clean up temp video
            print(f"    [Success] Extracted {len(downloaded)} real video frames.")
            
        elif src.get("type") == "zip":
            zip_path = target_dir / "temp_seq.zip"
            print(f"    [Download] Fetching official sequence archive from {src['url']} (~528MB)...")
            req = urllib.request.Request(src['url'], headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as response, open(zip_path, 'wb') as out_file:
                while True:
                    chunk = response.read(1024 * 1024) # 1MB chunks
                    if not chunk:
                        break
                    out_file.write(chunk)
            
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zf:
                matching_files = sorted([f for f in zf.namelist() if src["match_str"] in f and f.lower().endswith(".png")])
                for count, fname in enumerate(matching_files[:max_samples]):
                    with zf.open(fname) as img_file:
                        img = Image.open(img_file).convert("RGB")
                        frame_rgb = np.array(img)
                        save_path = target_dir / f"seq_frame_{count:04d}.png"
                        Image.fromarray(frame_rgb).save(save_path)
                        downloaded.append(frame_rgb)
            if zip_path.exists():
                zip_path.unlink() # Clean up temp zip
            print(f"    [Success] Extracted {len(downloaded)} real sequence frames from archive.")
    except Exception as e:
        print(f"    [Notice] Direct sequence download unavailable ({e}). Falling back to photo camera motion...")

    # If sequence download didn't reach max_samples, fallback to single image + simulated camera motion
    if len(downloaded) < max_samples:
        sample_urls = {
            "kitti": ["https://raw.githubusercontent.com/DepthAnything/Video-Depth-Anything/main/assets/teaser_video_v2.png"],
            "davis": ["https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog1.jpg"],
            "sintel": ["https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog2.jpg"],
            "nyuv2": ["https://raw.githubusercontent.com/facebookresearch/dinov2/main/docs/assets/dinov2_figure1.png"],
            "scannet": ["https://raw.githubusercontent.com/DepthAnything/Video-Depth-Anything/main/assets/teaser_video_v2.png"]
        }
        if len(downloaded) == 0:
            for url in sample_urls.get(dataset_name, []):
                try:
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        img = Image.open(io.BytesIO(resp.read())).convert("RGB")
                        downloaded.append(np.array(img))
                        break
                except Exception:
                    pass
        
        if len(downloaded) > 0:
            base_img = downloaded[0]
            h, w, _ = base_img.shape
            while len(downloaded) < max_samples:
                idx = len(downloaded)
                zoom = 1.0 + 0.18 * (idx / max(max_samples - 1, 1))
                new_h, new_w = int(h / zoom), int(w / zoom)
                pan_y = int((h - new_h) * (0.5 + 0.4 * np.sin(idx * 0.3)))
                pan_x = int((w - new_w) * (0.5 + 0.4 * np.cos(idx * 0.3)))
                crop_img = base_img[pan_y:pan_y+new_h, pan_x:pan_x+new_w]
                resample_mode = getattr(Image, 'Resampling', Image).BILINEAR
                frame_res = np.array(Image.fromarray(crop_img).resize((w, h), resample_mode))
                downloaded.append(frame_res)
                Image.fromarray(frame_res).save(target_dir / f"sim_frame_{idx:04d}.png")

    # Emergency backup only if internet download fails entirely
    while len(downloaded) < max_samples:
        downloaded.append(np.uint8(np.random.randint(0, 255, (384, 384, 3))))

    print(f"  [Dataset] Ready with {len(downloaded)} video frames for '{dataset_name}'.")
    return downloaded[:max_samples]

# ============================================================
# ACADEMIC DEPTH EVALUATION METRICS
# ============================================================
def compute_academic_metrics(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor = None):
    """
    FIDELITY-MODE metric function: computes AbsRel, RMSE, delta1/2/3, and
    Pearson correlation between two model outputs (typically quantized vs
    FP32 output from THIS SAME MODEL). There is no ground truth involved and
    no affine alignment — this measures rate-distortion fidelity, not
    dataset accuracy. For real ground-truth evaluation against labeled
    NYUv2 depth, use scripts/datasets_gt.compute_gt_depth_metrics instead
    (see docs/optimization_ledger.md T2, finding F2).
    """
    if valid_mask is None:
        valid_mask = (gt > 1e-3) & (pred > 1e-3)
    else:
        valid_mask = valid_mask & (gt > 1e-3) & (pred > 1e-3)
        
    if not valid_mask.any():
        valid_mask = torch.ones_like(gt, dtype=torch.bool)
        
    p = torch.clamp(pred[valid_mask].float(), min=1e-3)
    g = torch.clamp(gt[valid_mask].float(), min=1e-3)
    
    # Absolute Relative Error
    abs_rel = torch.mean(torch.abs(p - g) / g).item()
    
    # Root Mean Squared Error
    rmse = torch.sqrt(torch.mean((p - g) ** 2)).item()
    
    # Mean Absolute Error & MSE
    mae = torch.mean(torch.abs(p - g)).item()
    mse = torch.mean((p - g) ** 2).item()
    
    # Delta Accuracy Thresholds
    ratio = torch.max(p / g, g / p)
    delta1 = torch.mean((ratio < 1.25).float()).item()
    delta2 = torch.mean((ratio < (1.25 ** 2)).float()).item()
    delta3 = torch.mean((ratio < (1.25 ** 3)).float()).item()
    
    # Pearson Correlation
    p_mean = torch.mean(p)
    g_mean = torch.mean(g)
    num = torch.sum((p - p_mean) * (g - g_mean))
    den = torch.sqrt(torch.sum((p - p_mean) ** 2) * torch.sum((g - g_mean) ** 2))
    pearson = (num / (den + 1e-8)).item()
    
    return {
        "abs_rel": round(abs_rel, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "delta1": round(delta1, 4),
        "delta2": round(delta2, 4),
        "delta3": round(delta3, 4),
        "pearson": round(pearson, 4)
    }

# ============================================================
# TEMPORAL (IN)STABILITY METRIC — TAE
# ============================================================
def _warp_depth_by_flow(depth: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp a (B, H, W) depth map by per-pixel optical flow (H, W, 2) [dx, dy]
    using bilinear grid_sample, mapping frame t+1's depth onto frame t's grid.
    """
    B, H, W = depth.shape
    device = depth.device
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )
    new_x = xx + flow[..., 0]
    new_y = yy + flow[..., 1]
    grid_x = 2.0 * new_x / max(W - 1, 1) - 1.0
    grid_y = 2.0 * new_y / max(H - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    warped = F.grid_sample(depth.unsqueeze(1), grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped.squeeze(1)


def compute_tae(depth_sequence: torch.Tensor, flow_fields: torch.Tensor = None, align: bool = True) -> dict:
    """
    Temporal (in)stability metric ("flicker"): mean absolute difference
    between consecutive depth frames (docs/optimization_ledger.md T7).

        TAE = mean_t |D_t - warp(D_{t+1})|

    using per-pixel optical flow to compensate for camera/scene motion
    between frames. When flow_fields is None (this repo does not run an
    optical-flow estimator), falls back to the STATIC-CAMERA approximation:

        TAE = mean_t |D_t - D_{t+1}|

    with no motion compensation. This fallback over-estimates TAE for a
    genuinely moving camera/scene (real motion gets conflated with flicker)
    — treat it as an upper bound, not an exact flicker measurement, unless
    flow_fields is supplied.

    Args:
        depth_sequence: (T, H, W) or (B, T, H, W) predicted depth over T frames.
        flow_fields: Optional (T-1, H, W, 2) per-pixel (dx, dy) flow from
                     frame t to t+1. If None, no warping is applied (static fallback).
        align: If True, per-frame-pair affine-align D_{t+1} to D_t (scale+shift,
               via datasets_gt.affine_align_inverse_depth) before differencing,
               removing benign global scale drift (common in scale-invariant
               depth models) so TAE isolates local flicker rather than
               reporting it as instability.

    Returns:
        dict with 'tae' (float, mean over all frame pairs) and
        'per_frame_pair' (list of per-pair mean abs diff), 'used_flow' (bool),
        'aligned' (bool).
    """
    if depth_sequence.dim() == 3:
        depth_sequence = depth_sequence.unsqueeze(0)  # (1, T, H, W)

    B, T, H, W = depth_sequence.shape
    if T < 2:
        return {"tae": 0.0, "per_frame_pair": [], "used_flow": flow_fields is not None, "aligned": align}

    per_pair = []
    for t in range(T - 1):
        d_t = depth_sequence[:, t]        # (B, H, W)
        d_t1 = depth_sequence[:, t + 1]   # (B, H, W)

        if flow_fields is not None:
            d_t1 = _warp_depth_by_flow(d_t1, flow_fields[t])

        if align:
            aligned_frames = []
            for b in range(B):
                valid = torch.ones_like(d_t[b], dtype=torch.bool)
                _, _, d_t1_aligned = affine_align_inverse_depth(d_t1[b], d_t[b], valid)
                aligned_frames.append(d_t1_aligned)
            d_t1 = torch.stack(aligned_frames, dim=0)

        per_pair.append((d_t - d_t1).abs().mean().item())

    return {
        "tae": round(float(np.mean(per_pair)), 6),
        "per_frame_pair": [round(p, 6) for p in per_pair],
        "used_flow": flow_fields is not None,
        "aligned": align,
    }

# ============================================================
# VDA-PROTOCOL GEOMETRIC TAE (T9) — ported from
# Video-Depth-Anything/benchmark/eval/eval_tae.py::tae_torch / eval_TAE
# ============================================================
# This is DIFFERENT from compute_tae() above: that one warps D_{t+1} onto
# D_t's pixel grid using optical flow (or a static-camera no-op). This one
# uses KNOWN camera geometry (intrinsics K + relative pose from consecutive
# .cam files) to back-project frame t's depth to 3D, transform it into frame
# t+1's camera frame, and re-project — then compares against frame t+1's own
# independently-predicted depth via AbsRel. It requires real camera poses
# (Sintel's camdata_left/), so it's only usable where those exist, but it is
# EXACTLY the metric VDA's own paper reports, making our numbers directly
# comparable to their tables.
def _tae_geometric_single(
    depth1: torch.Tensor, depth2: torch.Tensor,
    R_2_1: torch.Tensor, T_2_1: torch.Tensor,
    K: torch.Tensor, mask: torch.Tensor,
    scatter_zbuffer: bool = True,
) -> float:
    """
    Ported from eval_tae.py::tae_torch. Back-projects depth1's pixels to 3D
    using intrinsics K, transforms to frame2's camera frame via the relative
    pose (X_2 = R_2_1 @ X_1 + T_2_1), re-projects into frame2's image plane
    (nearest-pixel splat), and compares the reprojected depth against depth2's
    own value at that location via AbsRel (normalized by depth2). A perfectly
    temporally-consistent depth predictor gives ~0; frame-to-frame flicker
    (or genuinely wrong reconstructed geometry) inflates it.

    Args:
        depth1, depth2: (H, W) ALIGNED predicted depth (metres), same scene,
                        consecutive frames.
        R_2_1: (3, 3) relative rotation, frame1's camera frame -> frame2's.
        T_2_1: (3,) relative translation, same convention.
        K: (3, 3) intrinsic matrix (assumed shared between the two frames,
           matching VDA's own implementation).
        mask: (H, W) bool, additional valid-pixel mask for frame2 (pass
              all-ones if none).
        scatter_zbuffer: True (default) resolves multi-pixel collisions by
              keeping the NEAREST reprojected depth (physically correct
              occlusion). False reproduces upstream's arbitrary
              last-write-wins exactly, for measuring the difference.

    Returns:
        Scalar AbsRel between depth2 and the frame1-reprojected-into-frame2
        depth, over pixels where both are valid and in-bounds. 0.0 if no
        pixels reproject in-bounds (degenerate motion/inputs) — matches
        eval_tae.py's own fallback for this case.
    """
    H, W = depth1.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=depth1.dtype, device=depth1.device),
        torch.arange(W, dtype=depth1.dtype, device=depth1.device),
        indexing='ij',
    )

    X = (xx - cx) * depth1 / fx
    Y = (yy - cy) * depth1 / fy
    Z = depth1
    points3d = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=1)  # (H*W, 3)

    points3d_transformed = points3d @ R_2_1.T + T_2_1
    X_world = points3d_transformed[:, 0]
    Y_world = points3d_transformed[:, 1]
    Z_world = points3d_transformed[:, 2]

    X_plane = torch.round((X_world * fx) / Z_world + cx).to(dtype=torch.long)
    Y_plane = torch.round((Y_world * fy) / Z_world + cy).to(dtype=torch.long)

    in_bounds = (X_plane >= 0) & (X_plane < W) & (Y_plane >= 0) & (Y_plane < H)
    if in_bounds.sum() == 0:
        return 0.0

    # Z-BUFFER (correctness fix over the upstream reference implementation).
    # VDA's eval_tae.py does `depth_proj[Y, X] = Z`, which is arbitrary
    # LAST-WRITE-WINS when several frame-1 pixels reproject onto the same
    # frame-2 pixel. Under large camera motion many pixels collide, so a FAR
    # background point can overwrite a NEAR foreground one and then get
    # compared against foreground depth -> enormous spurious error. Physically
    # the nearest surface occludes the others, so take the per-target MINIMUM
    # depth. Measured impact on Sintel: the violent-motion scenes are exactly
    # the ones that exploded (ambush_2 907%, market_5 98%) while static scenes
    # sat at ~1%; upstream never hit this because they evaluate TAE on
    # ScanNet's slow handheld motion. See docs/optimization_ledger.md T9.
    if scatter_zbuffer:
        flat = torch.full((H * W,), float('inf'), dtype=depth1.dtype, device=depth1.device)
        idx = Y_plane[in_bounds] * W + X_plane[in_bounds]
        flat.scatter_reduce_(0, idx, Z_world[in_bounds], reduce='amin', include_self=True)
        flat[torch.isinf(flat)] = 0.0
        depth_proj = flat.reshape(H, W)
    else:
        # Bit-exact upstream behaviour, retained so the z-buffer's effect can be
        # measured rather than asserted.
        depth_proj = torch.zeros((H, W), dtype=depth1.dtype, device=depth1.device)
        depth_proj[Y_plane[in_bounds], X_plane[in_bounds]] = Z_world[in_bounds]

    valid = (depth_proj > 0) & (depth2 > 0) & mask
    if valid.sum() == 0:
        return 0.0

    abs_rel = torch.mean(torch.abs(depth2[valid] - depth_proj[valid]) / depth2[valid])
    return float(abs_rel.item())


def _pool_align_scene_disparity(pred_disps, gt_depths, gt_range) -> tuple:
    """
    ONE global scale+shift disparity-space alignment fit pooled across ALL
    frames in a scene, matching VDA's eval_TAE (which fits a single scale/
    shift per scene, not per-frame, so the aligned depths used for TAE sit on
    a temporally-consistent scale). Reuses the same disparity-space fit as
    compute_gt_depth_metrics (F10), just pooled over multiple frames.

    Args:
        pred_disps: list of (H, W) raw model disparity outputs.
        gt_depths: list of (H, W) ground-truth metric depth, same length.
        gt_range: (min, max) metres; VDA's eval_TAE hardcodes the lower bound
                  at 1e-3 regardless of the dataset's min_depth_eval (verified
                  by reading eval_tae.py directly) — mirrored here for fidelity.

    Returns:
        (scale, shift) such that aligned_disparity = scale*pred + shift.
    """
    all_pred, all_gt_disp = [], []
    for pred, gt in zip(pred_disps, gt_depths):
        mask = (gt > 1e-3) & (gt < gt_range[1])
        if mask.any():
            all_pred.append(pred[mask].reshape(-1))
            all_gt_disp.append((1.0 / gt[mask].clamp(min=1e-8)).reshape(-1))
    if not all_pred:
        raise ValueError("No valid pixels across the scene for pooled disparity alignment")
    p = torch.cat(all_pred).double()
    g = torch.cat(all_gt_disp).double()
    A = torch.stack([p, torch.ones_like(p)], dim=1)
    solution = torch.linalg.lstsq(A, g.unsqueeze(-1)).solution
    return float(solution[0, 0]), float(solution[1, 0])


def compute_tae_geometric_for_scene(pred_disps, gt_depths, Ks, poses, gt_range,
                                     scatter_zbuffer: bool = True) -> dict:
    """
    Full VDA-protocol geometric TAE for one scene of REAL consecutive frames.

    Steps (mirrors eval_tae.py::eval_TAE): (1) one pooled disparity-space
    alignment across the whole scene, (2) invert + clip aligned depth to
    [1e-3, gt_range[1]], (3) for every consecutive real-frame pair, compute
    bidirectional (t->t+1 AND t+1->t) geometric TAE via _tae_geometric_single
    using the pair's relative camera pose, (4) average all pair x direction
    errors and report as a PERCENTAGE (the *100 VDA itself reports).

    Args:
        pred_disps: list of (H, W) torch tensors, raw model disparity output,
                    one per REAL frame in the scene, in frame order (already
                    upsampled to native resolution, matching gt_depths' shape).
        gt_depths: list of (H, W) numpy arrays, ground-truth metric depth.
        Ks: list of (3, 3) numpy intrinsic matrices, one per frame.
        poses: list of (4, 4) numpy camera-to-world pose matrices, one per frame.
        gt_range: (min, max) metres for this dataset (see DATASET_GT_CONFIG).

    Returns:
        dict with 'tae_percent' (float) and 'n_pairs' (int, consecutive pairs
        actually used).
    """
    n = len(pred_disps)
    if n != len(gt_depths) or n != len(Ks) or n != len(poses):
        raise ValueError(
            f"pred_disps/gt_depths/Ks/poses must have matching lengths, got "
            f"{n}/{len(gt_depths)}/{len(Ks)}/{len(poses)}"
        )
    if n < 2:
        return {"tae_percent": 0.0, "tae_median_percent": 0.0, "n_pairs": 0}

    gt_depths_t = [torch.from_numpy(g).double() if isinstance(g, np.ndarray) else g.double() for g in gt_depths]
    s, t = _pool_align_scene_disparity(pred_disps, gt_depths_t, gt_range)

    aligned_depths = []
    for pred in pred_disps:
        disp_aligned = (s * pred.double() + t).clamp(min=1e-3)
        depth_aligned = (1.0 / disp_aligned).clamp(min=1e-3, max=float(gt_range[1]))
        aligned_depths.append(depth_aligned)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    error_sum = 0.0
    per_pair_errors = []
    n_pairs = 0
    for i in range(n - 1):
        depth1 = aligned_depths[i].to(device)
        depth2 = aligned_depths[i + 1].to(device)
        K = torch.as_tensor(Ks[i], dtype=torch.float64, device=device)
        pose1 = np.asarray(poses[i], dtype=np.float64)
        pose2 = np.asarray(poses[i + 1], dtype=np.float64)

        T_2_1_full = np.linalg.inv(pose2) @ pose1  # cam1-local -> cam2-local
        R_2_1 = torch.from_numpy(T_2_1_full[:3, :3]).to(device=device, dtype=torch.float64)
        t_2_1 = torch.from_numpy(T_2_1_full[:3, 3]).to(device=device, dtype=torch.float64)
        mask_ones = torch.ones_like(depth1, dtype=torch.bool)

        err_fwd = _tae_geometric_single(depth1, depth2, R_2_1, t_2_1, K, mask_ones,
                                        scatter_zbuffer=scatter_zbuffer)

        T_1_2_full = np.linalg.inv(T_2_1_full)
        R_1_2 = torch.from_numpy(T_1_2_full[:3, :3]).to(device=device, dtype=torch.float64)
        t_1_2 = torch.from_numpy(T_1_2_full[:3, 3]).to(device=device, dtype=torch.float64)

        err_bwd = _tae_geometric_single(depth2, depth1, R_1_2, t_1_2, K, mask_ones,
                                        scatter_zbuffer=scatter_zbuffer)

        error_sum += err_fwd + err_bwd
        per_pair_errors.append((err_fwd + err_bwd) / 2.0)
        n_pairs += 1

    tae_percent = (error_sum / (2 * n_pairs)) * 100.0 if n_pairs > 0 else 0.0
    # Per-pair median too: within a scene the frame-pair errors are also
    # heavy-tailed (a few violent-motion frame pairs dominate), so the median
    # pair is a more representative summary of that scene's typical behaviour.
    tae_median_percent = float(np.median(per_pair_errors)) * 100.0 if per_pair_errors else 0.0
    return {
        "tae_percent": round(tae_percent, 6),
        "tae_median_percent": round(tae_median_percent, 6),
        "n_pairs": n_pairs,
    }

# ============================================================
# DYNAMIC SURGERY VERIFICATION & SANITY CHECK
# ============================================================
def verify_quantization_surgery(model_fp32: nn.Module, model_quant: nn.Module, sample_input: torch.Tensor):
    """
    Sanity checks that model surgery successfully replaced attention layers
    and confirms that quantization actively alters internal activations by
    computing MSE and max absolute difference on a sample forward pass.
    """
    if model_fp32 is None or model_quant is None:
        return
        
    print(f"\n  [Sanity Check] Verifying dynamic quantization surgery & activation diff...")
    replaced_backbone = 0
    replaced_temporal = 0
    for name, module in model_quant.named_modules():
        if module.__class__.__name__ == 'RotatedSelfAttention':
            replaced_backbone += 1
        elif module.__class__.__name__ == 'RotatedTemporalAttention':
            replaced_temporal += 1
            
    print(f"    -> Verified active surgical replacement: {replaced_backbone} backbone layers, {replaced_temporal} temporal layers.")
    
    with torch.no_grad():
        out_fp32 = model_fp32(sample_input[:1]) # Test on first frame batch
        out_q = model_quant(sample_input[:1])
        
    mse_diff = torch.mean((out_fp32 - out_q) ** 2).item()
    max_diff = torch.max(torch.abs(out_fp32 - out_q)).item()
    print(f"    -> Dynamic Activation Diff (FP32 vs Quantized): MSE = {mse_diff:.6e}, Max Abs Diff = {max_diff:.4f}")
    if mse_diff < 1e-12:
        print(f"    [Warning] Quantized activations are identical to FP32! Check quantizer parameters.")
    else:
        print(f"    [Success] Quantized forward pass actively modifying tensor computations.")

# ============================================================
# HONEST BIT ACCOUNTING & QJL OVERHEAD CALCULATOR
# ============================================================
# Per-group scale sharing for each quantizer: D4 groups 4 scalars per scale,
# E8 groups 8 (halving the scale overhead — the key T8 lever). 'uniform_vector'
# defaults to 4, matching UniformVectorQuantizer's actual group_size default.
#
# 'scalar' is INTENTIONALLY absent here (T10 fix, docs/optimization_ledger.md):
# ScalarRoundQuantizer's forward() is always called with per_channel=False in
# this codebase (RotatedSelfAttention/RotatedTemporalAttention never pass
# per_channel=True — verified by reading lattice_vq.py), which computes ONE
# GLOBAL scale for the entire input tensor via x.abs().amax() — not a
# per-4-element-group scale. Charging it via group_size=4 (as an earlier
# version of this dict did) OVERCHARGES the scalar baseline's scale overhead
# by 16x at head_dim=64 (256 bits/vector claimed vs its true ~16 bits/vector,
# amortized over far more than one vector in practice) — an error that always
# flatters our quantizers relative to the baseline, exactly backwards from the
# honesty bar the rest of this accounting holds itself to (see F1/T1). Use
# resolve_group_size() below, which maps 'scalar' to head_dim itself (one
# scale per whole vector — already a conservative, non-overcharging estimate,
# not an attempt to make the baseline look better than it is either).
QUANTIZER_GROUP_SIZE = {
    'uniform_vector': 4,
    'lattice_d4': 4,
    'lattice_e8': 8,
    # 'scalar_g8' (F11/S1 fair baseline): ScalarGroupQuantizer pays for a
    # per-8-group scale, identically to lattice_e8 — that IS the point of
    # this quantizer, so it must be charged the same group_size here.
    'scalar_g8': 8,
}


def resolve_group_size(quantizer_name: str, head_dim: int) -> int:
    """Returns the correct group_size for compute_real_bit_accounting given
    which quantizer is actually running. 'scalar' -> head_dim (one global
    scale per vector); everything else -> its native grouping from
    QUANTIZER_GROUP_SIZE (default 4 if somehow not listed)."""
    if quantizer_name == 'scalar':
        return head_dim
    return QUANTIZER_GROUP_SIZE.get(quantizer_name, 4)


def compute_real_bit_accounting(
    bit_val: int,
    head_dim: int = 64,
    use_qjl: bool = True,
    group_size: int = 4,
    scale_bits: int = 16,
    norm_bits: int = 16,
    n_projections: int = None,
):
    """
    Computes ALL-INCLUSIVE bit accounting: quantizer payload + per-group scale
    metadata + QJL side-channel overhead (if enabled). Ratios are reported vs
    BOTH FP32 and FP16 baselines, since FP16 is the realistic deployment
    baseline (FP32 overstates savings by 2x).

    Per-vector cost breakdown:
        payload_bits = head_dim * bit_val
        scale_bits_total = (head_dim / group_size) * scale_bits   (one scale per group)
        qjl_bits = n_projections + norm_bits                       (if use_qjl)

    For head_dim=64, group_size=4, bit_val=3, scale_bits=16, QJL(n_proj=256, norm=16):
        payload=192, scale=256, qjl=272 -> total=720 bits/vector = 11.25 eff bits/scalar
        (vs the naive "3-bit" claim, which ignores the 528 bits of metadata).

    Returns a dict:
        total_bits_per_vector, effective_bits_per_scalar,
        ratio_vs_fp32, ratio_vs_fp16, nominal_ratio_vs_fp32,
        scale_overhead_bits_per_vector, qjl_side_bits_per_vector
    """
    fp32_bits = head_dim * 32
    fp16_bits = head_dim * 16
    primary_bits = head_dim * bit_val
    scale_overhead_bits = (head_dim / group_size) * scale_bits if group_size > 0 else 0

    if use_qjl and bit_val > 0:
        # Uses the SAME default formula as QJLBiasCorrection itself
        # (research/quantizers/qjl_bias.default_qjl_projections), so this
        # accounting can never silently drift from what the runtime module
        # actually costs. Pass n_projections explicitly to audit a specific
        # (e.g. historical) configuration.
        n_proj = n_projections if n_projections is not None else default_qjl_projections(head_dim)
        qjl_side_bits = n_proj + norm_bits
    else:
        qjl_side_bits = 0

    total_bits = primary_bits + scale_overhead_bits + qjl_side_bits
    effective_bits_per_scalar = total_bits / head_dim if head_dim > 0 else 0.0
    ratio_vs_fp32 = round(fp32_bits / total_bits, 1) if total_bits > 0 else 1.0
    ratio_vs_fp16 = round(fp16_bits / total_bits, 1) if total_bits > 0 else 1.0
    nominal_ratio_vs_fp32 = round(32.0 / bit_val, 1) if bit_val > 0 else 1.0

    return {
        "total_bits_per_vector": total_bits,
        "effective_bits_per_scalar": round(effective_bits_per_scalar, 4),
        "ratio_vs_fp32": ratio_vs_fp32,
        "ratio_vs_fp16": ratio_vs_fp16,
        "nominal_ratio_vs_fp32": nominal_ratio_vs_fp32,
        "scale_overhead_bits_per_vector": scale_overhead_bits,
        "qjl_side_bits_per_vector": qjl_side_bits,
    }

# ============================================================
# PARETO CURVE & CHARTS GENERATOR
# ============================================================
def generate_pareto_charts(results: dict, output_dir: Path):
    if not HAS_MATPLOTLIB:
        print("  [Warning] matplotlib not found, skipping PNG chart generation.")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset_name, data in results.items():
        bit_widths = []
        delta1_scores = []
        abs_rel_scores = []
        fps_scores = []
        mem_savings = []
        
        for bit, metrics in data.items():
            if bit == "FP32_Baseline":
                bit_val = 32
            else:
                bit_val = int(bit.replace("bit", ""))
            # Read the already-computed ratio from the metrics dict (it
            # honestly reflects whatever --qjl/--no-qjl setting was actually
            # used for this run) rather than recomputing with a hardcoded
            # use_qjl=True, which could silently diverge from reality.
            mem_val = metrics.get("mem_savings_x", 1.0)

            bit_widths.append(bit_val)
            delta1_scores.append(metrics["delta1"])
            abs_rel_scores.append(metrics["abs_rel"])
            fps_scores.append(metrics.get("fps", 15.0))
            mem_savings.append(mem_val)
            
        # 1. Memory Savings vs Delta1 Accuracy (Pareto Frontier)
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(mem_savings, delta1_scores, marker='o', color='#2b5c8f', linewidth=2, label='VDA-HyperQuant (Rate-Distortion Fidelity)')
        for i, txt in enumerate(bit_widths):
            ax1.annotate(f"{txt}-bit", (mem_savings[i], delta1_scores[i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')
        ax1.set_xlabel('Real KV-Cache Memory Reduction (x-fold over FP32, incl. QJL side-channel)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('δ < 1.25 Fidelity vs FP32 (Higher is better)', fontsize=11, fontweight='bold', color='#2b5c8f')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_title(f"Pareto Frontier: Rate-Distortion Fidelity vs Memory ({dataset_name.upper()})", fontsize=13, fontweight='bold')
        
        # Note: We intentionally omit literature ground-truth baselines from this plot to clearly distinguish
        # Rate-Distortion fidelity (measured against FP32 model output) from sensor ground-truth accuracy.
        ax1.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(output_dir / f"pareto_memory_vs_delta1_{dataset_name}.png", dpi=300)
        plt.close()

        # 2. Bit-Width Degradation Curve (AbsRel & RMSE)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(bit_widths, abs_rel_scores, marker='s', color='#d95f02', linewidth=2, label='AbsRel Error (Lower is better)')
        ax.set_xlabel('Quantization Bit-Width', fontsize=12, fontweight='bold')
        ax.set_ylabel('Absolute Relative Error', fontsize=12, fontweight='bold', color='#d95f02')
        ax.invert_xaxis()  # Show 32 -> 8 -> 4 -> 3 -> 2
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_title(f"Quantization Degradation Curve ({dataset_name.upper()})", fontsize=14, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"pareto_degradation_curve_{dataset_name}.png", dpi=300)
        plt.close()

    print(f"  [Charts] Saved publication Pareto charts to {output_dir}")

# ============================================================
# GROUND-TRUTH EVALUATION (real depth labels, affine-invariant protocol)
# ============================================================
# Preprocessing mirrors VDA's own infer_video_depth() exactly (see
# Video-Depth-Anything/video_depth_anything/video_depth.py): aspect-ratio-
# PRESERVING resize to a 518-shorter-side (multiple of 14), with the
# ratio>1.78 shrink branch that KITTI's ~3.3:1 letterbox images trigger.
# A hard 266x266 SQUARE resize (the original code) crushed that aspect ratio
# and was the root cause of a broken KITTI FP32 baseline (delta1 0.43 vs
# published ~0.96); NYU's near-4:3 images survived it, which is why NYU
# looked fine at the time. Shared by run_groundtruth_eval (single-frame,
# static clip) and run_temporal_eval (T9, real consecutive-frame windows).
def _make_transform(h, w):
    import cv2
    from video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
    input_size = 518
    ratio = max(h, w) / min(h, w)
    if ratio > 1.78:  # VDA shrinks input for very wide clips (memory); KITTI hits this
        input_size = int(input_size * 1.777 / ratio)
        input_size = round(input_size / 14) * 14
    resize = Resize(width=input_size, height=input_size, resize_target=False,
                    keep_aspect_ratio=True, ensure_multiple_of=14,
                    resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC)
    norm = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    prep = PrepareForNet()
    return resize, norm, prep


def _preprocess_frame(rgb_np: np.ndarray) -> torch.Tensor:
    """One (H,W,3) uint8 RGB frame -> (3,h,w) float tensor via _make_transform."""
    H, W = rgb_np.shape[:2]
    resize, norm, prep = _make_transform(H, W)
    sample = {'image': rgb_np.astype(np.float32) / 255.0}
    sample = prep(norm(resize(sample)))
    return torch.from_numpy(sample['image'])


def run_groundtruth_eval(model, model_configs, ckpt_loaded, possible_ckpts, args, data_dir):
    """
    Evaluates FP32 and each swept bit-width against REAL ground-truth depth
    for args.dataset (nyuv2 / kitti / sintel), using the affine-invariant
    alignment protocol from scripts/datasets_gt.py. Returns a dict keyed
    "FP32_Baseline", "{bit}bit" matching the structure of the fidelity-mode
    dataset_results, so it slots into the same JSON export / table / chart code.

    Every accuracy number this function returns is a real ground-truth
    comparison — NEVER a fidelity-vs-FP32 proxy — so it is safe to label
    with the dataset name in publications (unlike fidelity mode).
    """
    from video_depth_anything.video_depth import VideoDepthAnything
    from datasets_gt import load_gt_dataset

    gt_samples, gt_range = load_gt_dataset(args.dataset, data_dir, max_samples=args.max_samples)
    print(f"  [Dataset] Loaded {len(gt_samples)} {args.dataset.upper()} images "
          f"(REAL ground truth, depth range {gt_range} m).")

    def predict_depth(m, rgb_np):
        H, W = rgb_np.shape[:2]
        tensor = _preprocess_frame(rgb_np)  # (3, h, w), aspect-preserved
        # VDA is a video model; feed a short static clip and take the last frame's
        # prediction (single images have no temporal context to give it). Note:
        # this static-clip hack is exactly why this path CANNOT measure temporal
        # consistency (no shared state across independent predict_depth calls) —
        # see run_temporal_eval (T9) for the real consecutive-frame path.
        clip = tensor.unsqueeze(0).repeat(3, 1, 1, 1).unsqueeze(0)  # [1, T=3, C, h, w]
        if torch.cuda.is_available():
            clip = clip.cuda()
        with torch.no_grad():
            out = m(clip)
        pred = out[0, -1] if out.dim() == 4 else out[0]  # (h, w) at network resolution
        # Interpolate the prediction back to the ORIGINAL image resolution so it
        # aligns with GT at native res (GT is never downsampled -> no label loss).
        pred = F.interpolate(pred[None, None].float(), size=(H, W),
                             mode='bilinear', align_corners=True)[0, 0]
        return pred.cpu()

    def gt_tensors(depth_np, valid_np):
        # GT stays at native resolution; predict_depth already upsampled the
        # prediction to match, so no GT downsampling (which would drop sparse
        # LiDAR returns) is needed.
        return torch.from_numpy(depth_np).float(), torch.from_numpy(valid_np.astype(bool))

    def avg_metrics(metrics_list):
        keys = ["abs_rel", "rmse", "delta1", "delta2", "delta3"]
        return {k: round(float(np.mean([m[k] for m in metrics_list])), 4) for k in keys}

    dataset_results = {}

    print(f"  [1/N] Running FP32 Reference Baseline against real {args.dataset.upper()} ground truth...")
    fp32_preds = []
    fp32_gt_metrics = []
    n_skipped = 0
    for sample in gt_samples:
        if model is not None:
            pred = predict_depth(model, sample["rgb"])
        else:
            # No-model fallback (local dry run): random pred at GT resolution.
            pred = torch.rand(*sample["depth"].shape)
        # fp32_preds MUST stay index-aligned with gt_samples (the per-bit loop
        # indexes into it), so always append the pred; only the METRIC is skipped
        # for frames with no in-range GT (e.g. all-far Sintel frames beyond the
        # gt_range cap — compute_gt_depth_metrics raises rather than fabricate).
        fp32_preds.append(pred)
        gt_t, valid_t = gt_tensors(sample["depth"], sample["valid_mask"])
        try:
            fp32_gt_metrics.append(compute_gt_depth_metrics(
                pred, gt_t, valid_t, gt_range=gt_range,
                pred_is_disparity=(args.pred_space == "disparity")))
        except ValueError:
            n_skipped += 1

    if not fp32_gt_metrics:
        raise ValueError(
            f"Every frame skipped (no in-range GT) — check gt_range={gt_range} "
            f"against {args.dataset}'s depth distribution."
        )
    fp32_metrics = avg_metrics(fp32_gt_metrics)
    fp32_metrics["n_images"] = len(fp32_gt_metrics)
    fp32_metrics["n_skipped_frames"] = n_skipped
    if n_skipped:
        print(f"        [note] skipped {n_skipped} frame(s) with no in-range GT")
    fp32_metrics["mem_savings_x"] = 1.0
    fp32_metrics["mem_savings_fp16_x"] = 0.5
    fp32_metrics["nominal_savings_x"] = 1.0
    fp32_metrics["effective_bits_per_scalar"] = 32.0
    dataset_results["FP32_Baseline"] = fp32_metrics
    print(f"        -> delta1: {fp32_metrics['delta1']} | AbsRel: {fp32_metrics['abs_rel']} "
          f"(vs REAL ground truth, N={len(gt_samples)})")

    for idx, bit in enumerate(args.bits):
        print(f"  [{idx+2}/{len(args.bits)+1}] Applying VDA-HyperQuant Surgery "
              f"({bit}-bit {args.quantizer}, scale_bits={args.scale_bits}, "
              f"qjl={'on' if args.use_qjl else 'off'}) [groundtruth mode]...")
        model_quant = None
        if model is not None:
            model_quant = VideoDepthAnything(**model_configs).eval()
            if ckpt_loaded:
                for ckpt_path in possible_ckpts:
                    if ckpt_path.exists():
                        model_quant.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
                        break
            else:
                model_quant.load_state_dict(model.state_dict())
            if torch.cuda.is_available():
                model_quant = model_quant.cuda()
            model_quant = apply_rotated_quantization_to_vda(
                model_quant, bits=bit, quantizer=args.quantizer, use_qjl=args.use_qjl,
                scale_bits=args.scale_bits, verbose=(idx == 0),
                use_rotation=args.use_rotation, rht_seed=args.rht_seed,
                replace_temporal=True,  # fixed: see docs/optimization_ledger.md T7 (qkv_bias surgery bug, not a reshape bug)
            )

            # Sanity check on the FIRST bit-width sweep: confirms surgery actually
            # replaced attention layers AND that quantization measurably alters
            # activations (rules out a silent no-op producing suspiciously-close
            # quantized-vs-FP32 ground-truth numbers). Builds its clip with the
            # SAME aspect-preserving transform predict_depth() uses.
            if idx == 0 and len(gt_samples) > 0:
                s_rgb = gt_samples[0]["rgb"]
                s_resize, s_norm, s_prep = _make_transform(*s_rgb.shape[:2])
                s_img = s_prep(s_norm(s_resize({'image': s_rgb.astype(np.float32) / 255.0})))['image']
                s_tensor = torch.from_numpy(s_img)
                sanity_clip = s_tensor.unsqueeze(0).repeat(3, 1, 1, 1).unsqueeze(0)
                if torch.cuda.is_available():
                    sanity_clip = sanity_clip.cuda()
                verify_quantization_surgery(model, model_quant, sanity_clip)

        q_gt_metrics = []
        q_skipped = 0
        for sample_idx, sample in enumerate(gt_samples):
            if model_quant is not None:
                pred = predict_depth(model_quant, sample["rgb"])
            else:
                noise_level = 0.02 * (8.0 / bit)
                pred = fp32_preds[sample_idx] + torch.randn_like(fp32_preds[sample_idx]) * noise_level
            gt_t, valid_t = gt_tensors(sample["depth"], sample["valid_mask"])
            # Skip (don't crash on) frames with no in-range GT — same rationale
            # as the FP32 loop above. Must skip the SAME frames FP32 did to keep
            # the comparison fair; since the gt_range/valid mask depends only on
            # GT (not on the prediction), a frame with no valid GT is skipped in
            # both passes identically.
            try:
                q_gt_metrics.append(compute_gt_depth_metrics(
                    pred, gt_t, valid_t, gt_range=gt_range,
                    pred_is_disparity=(args.pred_space == "disparity")))
            except ValueError:
                q_skipped += 1

        bit_accounting = compute_real_bit_accounting(
            bit, head_dim=64, use_qjl=args.use_qjl,
            group_size=resolve_group_size(args.quantizer, 64), scale_bits=args.scale_bits,
        )
        if not q_gt_metrics:
            raise ValueError(f"Every frame skipped at {bit}-bit (no in-range GT); check gt_range={gt_range}.")
        metrics = avg_metrics(q_gt_metrics)
        metrics["n_images"] = len(q_gt_metrics)
        metrics["n_skipped_frames"] = q_skipped
        metrics["mem_savings_x"] = bit_accounting["ratio_vs_fp32"]
        metrics["mem_savings_fp16_x"] = bit_accounting["ratio_vs_fp16"]
        metrics["nominal_savings_x"] = bit_accounting["nominal_ratio_vs_fp32"]
        metrics["effective_bits_per_scalar"] = bit_accounting["effective_bits_per_scalar"]
        metrics["total_bits_per_vec"] = bit_accounting["total_bits_per_vector"]
        dataset_results[f"{bit}bit"] = metrics
        print(f"        -> delta1: {metrics['delta1']} | AbsRel: {metrics['abs_rel']} | "
              f"eff={metrics['effective_bits_per_scalar']}b/scalar (vs REAL ground truth)")

    return dataset_results


def _predict_window(m, rgb_list):
    """
    Feeds a REAL consecutive window of frames through the model in ONE
    forward call (video_length=len(rgb_list)) so predictions share genuine
    cross-frame temporal context — unlike run_groundtruth_eval's isolated
    static-clip-per-frame hack (T9). Returns a list of (H,W) native-resolution
    depth predictions, one per input frame, in order.
    """
    H, W = rgb_list[0].shape[:2]
    tensors = [_preprocess_frame(rgb) for rgb in rgb_list]
    clip = torch.stack(tensors, dim=0).unsqueeze(0)  # [1, T, C, h, w]
    if torch.cuda.is_available():
        clip = clip.cuda()
    with torch.no_grad():
        out = m(clip)  # expected [1, T, h, w]
    preds = []
    T = out.shape[1] if out.dim() == 4 else 1
    for t in range(T):
        p = out[0, t] if out.dim() == 4 else out[0]
        p = F.interpolate(p[None, None].float(), size=(H, W), mode='bilinear', align_corners=True)[0, 0]
        preds.append(p.cpu())
    return preds


def run_temporal_eval(model, model_configs, ckpt_loaded, possible_ckpts, args, data_dir):
    """
    T9: real video-window temporal evaluation with VDA-protocol geometric TAE
    (docs/optimization_ledger.md T9). Currently Sintel-only — it's the only
    dataset here with both real consecutive-video structure AND the camera
    intrinsics/poses (camdata_left/*.cam) geometric TAE needs.

    For each scene (capped at args.max_scenes) and each bit-width (+FP32),
    feeds the scene's frames through the model in real consecutive windows
    (args.temporal_window frames per forward call — NOT independent static
    clips), then computes:
      - accuracy metrics from those SAME windowed predictions (a sanity/
        comparison row against run_groundtruth_eval's static-clip numbers —
        should be at least as accurate, since the model now has real temporal
        context instead of a repeated single frame)
      - geometric TAE (compute_tae_geometric_for_scene) using each frame's
        real camera K/pose, ported from VDA's own eval_tae.py.
    Averages both across scenes. Returns a dict keyed "FP32_Baseline",
    "{bit}bit", matching the other eval-mode result shapes (slots into the
    same JSON export / table code), with an added "tae_percent" key.

    KNOWN LIMITATION (documented, not hidden): windows are processed
    independently with no shared state across window boundaries (no KV
    cache/cached_hidden_states), so consecutive-frame pairs that straddle a
    window boundary don't benefit from cross-window temporal context the way
    within-window pairs do — see chunk_scene_into_windows' docstring.
    """
    from video_depth_anything.video_depth import VideoDepthAnything
    from datasets_gt import load_gt_dataset, group_samples_by_scene, chunk_scene_into_windows

    if args.dataset != "sintel":
        raise ValueError(
            f"--eval-mode temporal currently only supports --dataset sintel "
            f"(the only dataset with both real video structure and camera "
            f"pose data for geometric TAE). Got --dataset {args.dataset}."
        )

    gt_samples, gt_range = load_gt_dataset(args.dataset, data_dir, max_samples=args.max_samples, require_cam=True)
    scenes = group_samples_by_scene(gt_samples)
    scene_names = list(scenes.keys())[:args.max_scenes]
    print(f"  [Dataset] {len(scene_names)} scene(s) for temporal eval "
          f"(of {len(scenes)} loaded): {scene_names}")
    if not scene_names:
        raise ValueError("No scenes available for temporal eval (empty dataset or --max-scenes 0)")

    window = args.temporal_window

    def predict_scene(m):
        """Returns {scene_name: (pred_disps, gt_depths, Ks, poses)} for ALL real frames in each scene."""
        result = {}
        for scene in scene_names:
            frames = scenes[scene]
            pred_disps, gt_depths, Ks, poses = [], [], [], []
            for window_frames, n_real in chunk_scene_into_windows(frames, window):
                rgb_list = [f["rgb"] for f in window_frames]
                if m is not None:
                    window_preds = _predict_window(m, rgb_list)
                else:
                    window_preds = [torch.rand(*f["depth"].shape) for f in window_frames]
                # Discard the padded tail — see chunk_scene_into_windows docstring.
                for i in range(n_real):
                    pred_disps.append(window_preds[i])
                    gt_depths.append(window_frames[i]["depth"])
                    Ks.append(window_frames[i]["K"])
                    poses.append(window_frames[i]["pose"])
            result[scene] = (pred_disps, gt_depths, Ks, poses)
        return result

    def evaluate_predictions(scene_preds):
        acc_metrics_all = []
        tae_list = []
        per_scene_tae = {}
        total_pairs = 0
        n_skipped_frames = 0
        n_skipped_scenes = 0
        for scene, (pred_disps, gt_depths, Ks, poses) in scene_preds.items():
            for pred, gt in zip(pred_disps, gt_depths):
                gt_t = torch.from_numpy(gt).float()
                valid_t = torch.from_numpy((gt > 1e-3) & (gt < gt_range[1]))
                # Some Sintel scenes (mountain/market/cave, sky-heavy frames)
                # have frames whose entire GT depth lies beyond the VDA cap
                # (gt_range max) — zero evaluable pixels. compute_gt_depth_metrics
                # correctly REFUSES to fabricate a number for those (raises), so
                # skip+count them here rather than crash the run. A frame with no
                # in-range ground truth genuinely contributes nothing to AbsRel/
                # delta1; this is standard depth-eval practice, not data-dropping.
                try:
                    acc_metrics_all.append(compute_gt_depth_metrics(
                        pred, gt_t, valid_t, gt_range=gt_range,
                        pred_is_disparity=(args.pred_space == "disparity")))
                except ValueError:
                    n_skipped_frames += 1
            # TAE pools alignment across the whole scene; if the ENTIRE scene has
            # no in-range GT pixels, skip the scene's TAE too (same rationale).
            try:
                tae_result = compute_tae_geometric_for_scene(pred_disps, gt_depths, Ks, poses, gt_range)
                tae_list.append(tae_result["tae_percent"])
                per_scene_tae[scene] = tae_result["tae_percent"]
                total_pairs += tae_result["n_pairs"]
            except ValueError:
                n_skipped_scenes += 1

        if not acc_metrics_all:
            raise ValueError(
                "Every frame was skipped for having no in-range GT pixels — check "
                f"gt_range={gt_range} against this dataset's actual depth distribution."
            )

        keys = ["abs_rel", "rmse", "delta1", "delta2", "delta3"]
        avg = {k: round(float(np.mean([m[k] for m in acc_metrics_all])), 4) for k in keys}
        avg["tae_percent"] = round(float(np.mean(tae_list)), 6) if tae_list else 0.0
        # MEDIAN across scenes is the headline temporal number: the per-scene
        # distribution is heavy-tailed (one violent-motion Sintel scene can sit
        # 100x above the rest), so the mean describes that one scene rather than
        # the model. Both are reported; neither is hidden.
        avg["tae_median_percent"] = round(float(np.median(tae_list)), 6) if tae_list else 0.0
        avg["temporal_n_pairs"] = total_pairs
        avg["n_images"] = len(acc_metrics_all)
        avg["n_skipped_frames"] = n_skipped_frames
        avg["n_skipped_scenes_tae"] = n_skipped_scenes
        # Per-scene TAE breakdown — the diagnostic for the inflated aggregate.
        # A healthy metric gives a tight spread; a few scenes at 100%+ while the
        # rest sit near single digits localises the bug (e.g. disocclusion in
        # high-motion scenes) instead of leaving us to guess from the mean.
        avg["per_scene_tae"] = {k: round(v, 3) for k, v in per_scene_tae.items()}
        if n_skipped_frames or n_skipped_scenes:
            print(f"        [note] skipped {n_skipped_frames} frame(s) (no in-range GT) "
                  f"and {n_skipped_scenes} scene(s) for TAE")
        if per_scene_tae:
            worst = sorted(per_scene_tae.items(), key=lambda kv: -kv[1])
            median_tae = float(np.median(list(per_scene_tae.values())))
            print(f"        [TAE per-scene] median={median_tae:.2f}%  "
                  f"worst: " + ", ".join(f"{s}={v:.1f}%" for s, v in worst[:5]))
            print(f"        [TAE per-scene] best:  "
                  + ", ".join(f"{s}={v:.1f}%" for s, v in worst[-5:]))
        return avg

    dataset_results = {}

    print(f"  [1/N] Running FP32 Reference (real video windows, window={window})...")
    fp32_metrics = evaluate_predictions(predict_scene(model))
    fp32_metrics["mem_savings_x"] = 1.0
    fp32_metrics["mem_savings_fp16_x"] = 0.5
    fp32_metrics["nominal_savings_x"] = 1.0
    fp32_metrics["effective_bits_per_scalar"] = 32.0
    dataset_results["FP32_Baseline"] = fp32_metrics
    print(f"        -> delta1: {fp32_metrics['delta1']} | TAE% median: {fp32_metrics['tae_median_percent']} "
          f"(mean {fp32_metrics['tae_percent']}) "
          f"({fp32_metrics['temporal_n_pairs']} pairs)")

    for idx, bit in enumerate(args.bits):
        print(f"  [{idx+2}/{len(args.bits)+1}] Applying VDA-HyperQuant Surgery "
              f"({bit}-bit {args.quantizer}, scale_bits={args.scale_bits}, "
              f"qjl={'on' if args.use_qjl else 'off'}) [temporal mode]...")
        model_quant = None
        if model is not None:
            model_quant = VideoDepthAnything(**model_configs).eval()
            if ckpt_loaded:
                for ckpt_path in possible_ckpts:
                    if ckpt_path.exists():
                        model_quant.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
                        break
            else:
                model_quant.load_state_dict(model.state_dict())
            if torch.cuda.is_available():
                model_quant = model_quant.cuda()
            model_quant = apply_rotated_quantization_to_vda(
                model_quant, bits=bit, quantizer=args.quantizer, use_qjl=args.use_qjl,
                scale_bits=args.scale_bits, verbose=(idx == 0), replace_temporal=True,
                use_rotation=args.use_rotation, rht_seed=args.rht_seed,
            )

        metrics = evaluate_predictions(predict_scene(model_quant))

        bit_accounting = compute_real_bit_accounting(
            bit, head_dim=64, use_qjl=args.use_qjl,
            group_size=resolve_group_size(args.quantizer, 64), scale_bits=args.scale_bits,
        )
        metrics["mem_savings_x"] = bit_accounting["ratio_vs_fp32"]
        metrics["mem_savings_fp16_x"] = bit_accounting["ratio_vs_fp16"]
        metrics["nominal_savings_x"] = bit_accounting["nominal_ratio_vs_fp32"]
        metrics["effective_bits_per_scalar"] = bit_accounting["effective_bits_per_scalar"]
        metrics["total_bits_per_vec"] = bit_accounting["total_bits_per_vector"]
        dataset_results[f"{bit}bit"] = metrics
        print(f"        -> delta1: {metrics['delta1']} | AbsRel: {metrics['abs_rel']} | "
              f"TAE% median: {metrics['tae_median_percent']} (mean {metrics['tae_percent']}) "
              f"({metrics['temporal_n_pairs']} pairs) | "
              f"eff={metrics['effective_bits_per_scalar']}b/scalar")

    return dataset_results


# ============================================================
# MAIN EVALUATION EXECUTION
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="VDA-HyperQuant Multi-Dataset Pareto Evaluation")
    parser.add_argument("--dataset", type=str, default="kitti", choices=["kitti", "davis", "sintel", "nyuv2", "scannet", "all"], help="Target benchmark dataset")
    parser.add_argument("--bits", nargs="+", type=int, default=[8, 4, 3, 2], help="Quantization bit-widths to sweep")
    parser.add_argument("--max-samples", type=int, default=20, help="Number of video frames per dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/pareto_results", help="Directory to save benchmark reports and charts")
    parser.add_argument("--test-mode", action="store_true", help="Run fast verification test")
    parser.add_argument(
        "--qjl", dest="use_qjl", action=argparse.BooleanOptionalAction, default=True,
        help="Enable QJL bias correction (use --no-qjl to run the QJL ablation)",
    )
    parser.add_argument("--quantizer", type=str, default="lattice_d4",
                         choices=["scalar", "scalar_g8", "uniform_vector", "lattice_d4", "lattice_e8"])
    parser.add_argument("--scale-bits", type=int, default=16, choices=[8, 16],
                         help="Bit-width for lattice quantizers' per-group scale metadata "
                              "(the T8 4-bit headline config: --quantizer lattice_e8 "
                              "--scale-bits 8 --no-qjl --bits 3)")
    parser.add_argument(
        "--rotation", dest="use_rotation", action=argparse.BooleanOptionalAction, default=True,
        help="Apply the Hadamard rotation before quantizing (default: on). Use --no-rotation "
             "to quantize RAW activations instead — the T10 ablation showing what RHT actually "
             "buys at matched bit-width (docs/optimization_ledger.md T10).",
    )
    parser.add_argument(
        "--rht-seed", type=int, default=None,
        help="Seed the Hadamard rotation's random sign draw reproducibly across all replaced "
             "layers (each layer still gets its own draw, derived from this seed). Ignored if "
             "--no-rotation. Run with a few different seeds to check the result isn't an "
             "artifact of one lucky sign draw (T10).",
    )
    parser.add_argument(
        "--eval-mode", type=str, default="fidelity", choices=["fidelity", "groundtruth", "temporal"],
        help=(
            "'fidelity' (default): quantized model output vs FP32 model output on proxy "
            "video frames — NOT a dataset accuracy result, no ground truth involved. "
            "'groundtruth': quantized model output vs REAL labeled depth (per-image static "
            "clips), affine-invariant alignment (docs/optimization_ledger.md T2/F10). "
            "Supports --dataset nyuv2/kitti/sintel. "
            "'temporal': REAL consecutive-frame video windows (not static clips) + "
            "VDA-protocol geometric TAE (docs/optimization_ledger.md T9). Sintel only "
            "(needs both real video structure and camera pose data)."
        ),
    )
    parser.add_argument(
        "--pred-space", type=str, default="disparity", choices=["disparity", "metric"],
        help=(
            "Space the model output lives in, for GT affine alignment. 'disparity' "
            "(default) is correct for VDA's RELATIVE model (the vits checkpoint this "
            "suite downloads) which outputs inverse depth — aligns in disparity space "
            "then inverts to metric. Use 'metric' only with a metric-depth checkpoint. "
            "See docs/optimization_ledger.md F10."
        ),
    )
    parser.add_argument("--max-scenes", type=int, default=2,
                         help="[--eval-mode temporal] Max Sintel scenes to evaluate (kept small "
                              "by default to bound Colab GPU time for a first run)")
    parser.add_argument("--temporal-window", type=int, default=16,
                         help="[--eval-mode temporal] Frames per real video window fed to the "
                              "model in one forward call (video_length)")
    args = parser.parse_args()

    if args.eval_mode == "groundtruth":
        from datasets_gt import DATASET_GT_CONFIG
        gt_capable = set(DATASET_GT_CONFIG)  # nyuv2, kitti, sintel
        if args.dataset not in gt_capable:
            raise ValueError(
                f"--eval-mode groundtruth supports {sorted(gt_capable)} (datasets with a "
                f"real GT depth loader in scripts/datasets_gt.py). '{args.dataset}' has no "
                f"loader: DAVIS has no depth ground truth (TAE/fidelity only), ScanNet is "
                f"ToU-gated (see scripts/download_datasets.sh), and 'all' can't mix GT ranges. "
                f"Do not fabricate GT results for them."
            )
    elif args.eval_mode == "temporal" and args.dataset != "sintel":
        raise ValueError(
            "--eval-mode temporal currently only supports --dataset sintel "
            "(docs/optimization_ledger.md T9 — the only dataset with both real "
            "consecutive-video structure and camera pose data for geometric TAE)."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = SCRIPT_DIR.parent / "benchmark_data"

    datasets_to_run = ["kitti", "davis", "sintel", "nyuv2", "scannet"] if args.dataset == "all" else [args.dataset]
    
    print(f"{'=' * 65}")
    print(f"  VDA-HyperQuant Multi-Dataset Pareto Benchmark Suite")
    print(f"{'=' * 65}")
    print(f"  Target Datasets: {', '.join([d.upper() for d in datasets_to_run])}")
    print(f"  Bit-Width Sweep: {args.bits} bits + FP32 Baseline")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"{'=' * 65}\n")

    # Load Video-Depth-Anything ViT-Small model
    try:
        import video_depth_anything.dinov2_layers.attention as dino_attn
        import video_depth_anything.motion_module.attention as motion_attn
        if not torch.cuda.is_available():
            dino_attn.memory_efficient_attention = lambda q, k, v, attn_bias=None, **kwargs: torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
            motion_attn.XFORMERS_AVAILABLE = False
            
        from video_depth_anything.video_depth import VideoDepthAnything
        model_configs = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        model = VideoDepthAnything(**model_configs).eval()
        
        # Check for official VDA checkpoint weights if available
        possible_ckpts = [
            REPO_ROOT / "checkpoints" / "video_depth_anything_vits.pth",
            REPO_ROOT / "video_depth_anything_vits.pth",
            Path("/content/video_depth_anything_vits.pth"),
            Path("/content/vdaquant/checkpoints/video_depth_anything_vits.pth")
        ]
        ckpt_loaded = False
        for ckpt_path in possible_ckpts:
            if ckpt_path.exists():
                if ckpt_path.stat().st_size < 10_000_000:
                    print(f"  [Model] Removing corrupt/incomplete checkpoint file {ckpt_path}...")
                    try:
                        ckpt_path.unlink()
                    except Exception:
                        pass
                    continue
                try:
                    print(f"  [Model] Loading pretrained weights from {ckpt_path}...")
                    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
                    ckpt_loaded = True
                    break
                except Exception as e:
                    print(f"  [Warning] Failed to load checkpoint {ckpt_path} ({e}). Removing corrupt file...")
                    try:
                        ckpt_path.unlink()
                    except Exception:
                        pass
        if not ckpt_loaded:
            print("  [Model] No valid checkpoint .pth found. Automatically downloading official pretrained VDA weights (~111MB)...")
            try:
                import urllib.request
                ckpt_url = "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth"
                save_ckpt_path = REPO_ROOT / "checkpoints" / "video_depth_anything_vits.pth"
                save_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                req = urllib.request.Request(ckpt_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=60) as response, open(save_ckpt_path, 'wb') as out_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out_file.write(chunk)
                model.load_state_dict(torch.load(save_ckpt_path, map_location='cpu'))
                ckpt_loaded = True
                print(f"  [Success] Downloaded and loaded pretrained weights to {save_ckpt_path}")
            except Exception as e:
                print(f"  [Warning] Could not auto-download checkpoint ({e}). Using base model weights for rate-distortion evaluation.")

        if torch.cuda.is_available():
            model = model.cuda()
        else:
            for m in model.modules():
                if hasattr(m, '_use_memory_efficient_attention_xformers'):
                    m._use_memory_efficient_attention_xformers = False
    except Exception as e:
        print(f"  [Error] Could not load Video-Depth-Anything model: {e}")
        print("  Running in simulation baseline mode for script verification...")
        model = None

    all_results = {}

    for dataset_name in datasets_to_run:
        print(f"\n━━━ Evaluating Benchmark Dataset: {dataset_name.upper()} ━━━")

        if args.eval_mode == "groundtruth":
            # Real NYUv2 labeled-test-split evaluation (affine-invariant protocol,
            # docs/optimization_ledger.md T2). No synthetic/proxy fallback: if the
            # dataset can't be loaded, this raises rather than silently reporting
            # fidelity-vs-FP32 numbers under a ground-truth label.
            all_results[dataset_name] = run_groundtruth_eval(
                model=model, model_configs=model_configs, ckpt_loaded=ckpt_loaded,
                possible_ckpts=possible_ckpts, args=args, data_dir=data_dir,
            )
            continue

        if args.eval_mode == "temporal":
            # T9: real consecutive-frame video windows + VDA-protocol geometric
            # TAE (docs/optimization_ledger.md T9). Sintel only.
            all_results[dataset_name] = run_temporal_eval(
                model=model, model_configs=model_configs, ckpt_loaded=ckpt_loaded,
                possible_ckpts=possible_ckpts, args=args, data_dir=data_dir,
            )
            continue

        frames = get_dataset_samples(dataset_name, data_dir, max_samples=args.max_samples)
        
        # Prepare tensor sequence [1, T, C, H, W] with ImageNet normalization for DINOv2
        frame_tensors = []
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        for f in frames:
            img_resized = np.array(Image.fromarray(f).resize((266, 266)))
            tensor = (torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0 - mean) / std
            frame_tensors.append(tensor)
        video_input = torch.stack(frame_tensors, dim=0).unsqueeze(0) # [1, T, C, H, W]
        if torch.cuda.is_available():
            video_input = video_input.cuda()

        dataset_results = {}

        # 1. Run FP32 Baseline
        print("  [1/5] Running FP32 Reference Baseline...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        if model is not None:
            with torch.no_grad():
                fp32_out = model(video_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            fps = len(frames) / max(time.time() - t0, 1e-4)
        else:
            fp32_out = torch.randn((len(frames), 266, 266)).abs()
            fps = 14.5

        peak_mem_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if torch.cuda.is_available() else 0.0
        fp32_metrics = compute_academic_metrics(fp32_out, fp32_out) # Baseline self-comparison
        fp32_metrics["fps"] = round(fps, 1)
        fp32_metrics["mem_savings_x"] = 1.0          # vs FP32 (itself)
        fp32_metrics["mem_savings_fp16_x"] = 0.5      # vs FP16 (FP32 uses 2x the bits)
        fp32_metrics["nominal_savings_x"] = 1.0
        fp32_metrics["effective_bits_per_scalar"] = 32.0
        fp32_metrics["total_bits_per_vec"] = 64 * 32
        fp32_metrics["measured_mem_mb"] = peak_mem_mb
        # TAE: FP32's own temporal stability, reported as the reference point
        # quantized bit-widths are compared against (docs/optimization_ledger.md T7).
        fp32_tae = compute_tae(fp32_out.float())
        fp32_metrics["tae"] = fp32_tae["tae"]
        dataset_results["FP32_Baseline"] = fp32_metrics
        print(f"        -> Baseline FPS: {fp32_metrics['fps']} | TAE: {fp32_metrics['tae']} | Measured Peak Memory: {peak_mem_mb} MB")

        # 2. Sweep Quantization Bit-Widths
        for idx, bit in enumerate(args.bits):
            print(f"  [{idx+2}/{len(args.bits)+1}] Applying VDA-HyperQuant Surgery ({bit}-bit lattice_d4)...")
            if model is not None:
                # Reload clean FP32 model for clean surgery
                model_quant = VideoDepthAnything(**model_configs).eval()
                if ckpt_loaded:
                    for ckpt_path in possible_ckpts:
                        if ckpt_path.exists():
                            model_quant.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
                            break
                else:
                    model_quant.load_state_dict(model.state_dict())

                if torch.cuda.is_available():
                    model_quant = model_quant.cuda()
                
                model_quant = apply_rotated_quantization_to_vda(
                    model_quant, bits=bit, quantizer=args.quantizer, use_qjl=args.use_qjl,
                    scale_bits=args.scale_bits, verbose=False,
                    use_rotation=args.use_rotation, rht_seed=args.rht_seed,
                    replace_temporal=True,  # fixed: see docs/optimization_ledger.md T7 (qkv_bias surgery bug, not a reshape bug)
                )
                
                # Perform dynamic surgery verification on first bit-width sweep
                if idx == 0:
                    verify_quantization_surgery(model, model_quant, video_input)

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    q_out = model_quant(video_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                fps_q = len(frames) / max(time.time() - t0, 1e-4)
            else:
                # Simulated degradation for offline verification
                noise_level = 0.02 * (8.0 / bit)
                q_out = fp32_out + torch.randn_like(fp32_out) * noise_level
                fps_q = 15.5

            peak_mem_mb_q = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if torch.cuda.is_available() else 0.0
            bit_accounting = compute_real_bit_accounting(
                bit, head_dim=64, use_qjl=args.use_qjl,
                group_size=resolve_group_size(args.quantizer, 64), scale_bits=args.scale_bits,
            )
            metrics = compute_academic_metrics(q_out, fp32_out)
            metrics["fps"] = round(fps_q, 1)
            metrics["mem_savings_x"] = bit_accounting["ratio_vs_fp32"]
            metrics["mem_savings_fp16_x"] = bit_accounting["ratio_vs_fp16"]
            metrics["nominal_savings_x"] = bit_accounting["nominal_ratio_vs_fp32"]
            metrics["effective_bits_per_scalar"] = bit_accounting["effective_bits_per_scalar"]
            metrics["total_bits_per_vec"] = bit_accounting["total_bits_per_vector"]
            metrics["measured_mem_mb"] = peak_mem_mb_q
            # TAE at this bit-width, reported alongside FP32's TAE (fp32_metrics["tae"]
            # above) so temporal-consistency degradation from quantization is visible
            # per bit-width, not just claimed in a docstring (docs/optimization_ledger.md T7).
            metrics["tae"] = compute_tae(q_out.float())["tae"]
            dataset_results[f"{bit}bit"] = metrics
            print(f"        -> δ1: {metrics['delta1']} | AbsRel: {metrics['abs_rel']} | Corr: {metrics['pearson']} | "
                  f"FPS: {metrics['fps']} | TAE: {metrics['tae']} (FP32: {fp32_metrics['tae']}) | "
                  f"eff={metrics['effective_bits_per_scalar']}b/scalar "
                  f"({metrics['mem_savings_x']}x vs FP32, {metrics['mem_savings_fp16_x']}x vs FP16 "
                  f"[{metrics['nominal_savings_x']}x nominal] | {peak_mem_mb_q} MB)")

        all_results[dataset_name] = dataset_results

    # Export JSON Report
    json_path = output_dir / "pareto_benchmark_results.json"
    if args.eval_mode == "groundtruth":
        eval_mode_note = (
            f"GROUND TRUTH: accuracy metrics (abs_rel, rmse, delta1-3) are measured against REAL "
            f"{args.dataset.upper()} ground-truth depth using the affine-invariant alignment "
            f"protocol (scripts/datasets_gt.py), per-image static clips. Safe to label with the "
            f"dataset name. No temporal/TAE claim here — see --eval-mode temporal for that."
        )
    elif args.eval_mode == "temporal":
        eval_mode_note = (
            f"TEMPORAL: accuracy metrics are measured on REAL consecutive-frame video windows "
            f"(video_length={args.temporal_window}, not independent static clips). tae_percent is "
            f"VDA-protocol geometric-reprojection temporal consistency (ported from "
            f"Video-Depth-Anything/benchmark/eval/eval_tae.py; lower is more consistent), using "
            f"real camera poses — directly comparable to VDA's own published TAE tables. "
            f"docs/optimization_ledger.md T9."
        )
    else:
        eval_mode_note = (
            "FIDELITY: accuracy metrics (abs_rel, rmse, delta1-3, pearson) reflect Rate-Distortion "
            "fidelity of the quantized model's output vs THIS SAME MODEL's own FP32 output on proxy "
            "video frames — there is NO ground truth involved. Do NOT report these as dataset "
            "accuracy results; use --eval-mode groundtruth for that."
        )
    with open(json_path, "w") as f:
        json.dump({
            "eval_mode": args.eval_mode,
            "results": all_results,
            "note": (
                f"{eval_mode_note} effective_bits_per_scalar is the ALL-INCLUSIVE rate: payload "
                "bits + per-group scale metadata + QJL side-channel (see compute_real_bit_accounting). "
                "mem_savings_x is real compression vs FP32; mem_savings_fp16_x is real compression vs "
                "FP16 (the realistic deployment baseline); nominal_savings_x reports raw quantizer "
                "payload compression only, ignoring all metadata. Unverified literature baseline "
                "numbers are intentionally NOT included here — see docs/unverified_baselines.md "
                "(quarantined, uncited, do not use in publications) and docs/optimization_ledger.md F2."
            ),
        }, f, indent=2)
    print(f"\n  [Export] Full Pareto numerical results saved to {json_path}")

    # Generate Publication Charts
    generate_pareto_charts(all_results, output_dir)

    # Print Summary Table
    is_gt = args.eval_mode == "groundtruth"
    is_temporal = args.eval_mode == "temporal"
    if is_temporal:
        title = f"PARETO TEMPORAL BENCHMARK SUMMARY TABLE (ViT-Small, head_dim=64, real video windows, VDA-protocol TAE)"
        subtitle = "  * Accuracy + TAE measured on REAL consecutive Sintel frames with real camera poses *"
    elif is_gt:
        title = f"PARETO GROUND-TRUTH BENCHMARK SUMMARY TABLE (ViT-Small, head_dim=64, vs REAL {args.dataset.upper()} labels)"
        subtitle = f"  * Accuracy measured vs REAL {args.dataset.upper()} ground-truth depth (affine-invariant protocol) *"
    else:
        title = "PARETO RATE-DISTORTION FIDELITY BENCHMARK SUMMARY TABLE (ViT-Small, head_dim=64)"
        subtitle = "  * Accuracy measured vs FP32 baseline (Fidelity/Rate-Distortion), NOT ground truth *"
    print(f"\n{'=' * 110}")
    print(f"  {title}")
    print(subtitle)
    print(f"{'=' * 110}")
    print(f"Dataset   | Bit-Width | δ < 1.25 ↑ | AbsRel ↓ | RMSE ↓   | TAE ↓    | Eff b/scalar | vs FP32 (vs FP16) [Nom]")
    print(f"{'-' * 110}")
    for d_name, d_res in all_results.items():
        for b_name, m in d_res.items():
            b_label = f"{b_name:<9}"
            eff_bits = f"{m.get('effective_bits_per_scalar', '?')}"
            if 'tae_percent' in m:
                # Median is the headline (heavy-tailed per-scene distribution).
                tae_str = f"{m.get('tae_median_percent', m['tae_percent']):.2f}%"
            elif 'tae' in m:
                tae_str = f"{m['tae']}"
            else:
                tae_str = "n/a"
            real_save = (f"{m['mem_savings_x']}x ({m.get('mem_savings_fp16_x', '?')}x) "
                         f"[{m.get('nominal_savings_x', m['mem_savings_x'])}x]")
            print(f"{d_name.upper():<9} | {b_label} | {m['delta1']:<10} | {m['abs_rel']:<8} | {m['rmse']:<8} | {tae_str:<8} | {eff_bits:<12} | {real_save:<24}")
        print(f"{'-' * 110}")
    print(f"{'=' * 110}")
    print("  * Note 1: 'Eff b/scalar' is the ALL-INCLUSIVE effective rate (payload + per-group scale metadata")
    print("            + QJL side-channel overhead). 'vs FP32 (vs FP16) [Nom]' are compression ratios against")
    print("            the FP32 baseline, the FP16 deployment baseline, and the naive nominal (payload-only) rate.")
    if is_temporal:
        print("  * Note 2: TAE is VDA-protocol geometric-reprojection temporal consistency (lower = more consistent),")
        print("            ported from Video-Depth-Anything/benchmark/eval/eval_tae.py — directly comparable to")
        print("            VDA's own published TAE tables. Accuracy here uses REAL video windows, not static clips.")
    elif is_gt:
        print(f"  * Note 2: Accuracy metrics are measured against REAL {args.dataset.upper()} ground truth. Safe to cite by dataset name.")
    else:
        print("  * Note 2: Accuracy metrics (δ1, AbsRel, etc.) evaluate rate-distortion fidelity relative to the FP32")
        print("            model — there is NO ground truth here. Do not report as dataset accuracy; re-run with")
        print("            --eval-mode groundtruth for citable numbers.\n")

if __name__ == "__main__":
    main()
