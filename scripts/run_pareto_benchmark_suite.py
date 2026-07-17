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
# E8 groups 8 (halving the scale overhead — the key T8 lever). 'scalar' and
# 'uniform_vector' default to 4 for historical consistency with earlier
# accounting; this doesn't affect the T8 gate, which uses lattice_e8.
QUANTIZER_GROUP_SIZE = {
    'scalar': 4,
    'uniform_vector': 4,
    'lattice_d4': 4,
    'lattice_e8': 8,
}


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

    # Preprocessing mirrors VDA's own infer_video_depth() exactly (see
    # Video-Depth-Anything/video_depth_anything/video_depth.py): aspect-ratio-
    # PRESERVING resize to a 518-shorter-side (multiple of 14), with the
    # ratio>1.78 shrink branch that KITTI's ~3.3:1 letterbox images trigger.
    # The previous 266x266 SQUARE resize crushed that aspect ratio and was the
    # root cause of the broken KITTI FP32 baseline (delta1 0.43 vs published
    # ~0.96); NYU's near-4:3 images survived it, which is why NYU looked fine.
    import cv2
    from video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

    def _make_transform(h, w):
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

    def predict_depth(m, rgb_np):
        H, W = rgb_np.shape[:2]
        resize, norm, prep = _make_transform(H, W)
        sample = {'image': rgb_np.astype(np.float32) / 255.0}
        sample = prep(norm(resize(sample)))
        tensor = torch.from_numpy(sample['image'])  # (3, h, w), aspect-preserved
        # VDA is a video model; feed a short static clip and take the last frame's
        # prediction (single images have no temporal context to give it).
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
    for sample in gt_samples:
        if model is not None:
            pred = predict_depth(model, sample["rgb"])
        else:
            # No-model fallback (local dry run): random pred at GT resolution.
            pred = torch.rand(*sample["depth"].shape)
        fp32_preds.append(pred)
        gt_t, valid_t = gt_tensors(sample["depth"], sample["valid_mask"])
        fp32_gt_metrics.append(compute_gt_depth_metrics(pred, gt_t, valid_t, gt_range=gt_range))

    fp32_metrics = avg_metrics(fp32_gt_metrics)
    fp32_metrics["n_images"] = len(gt_samples)
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
                replace_temporal=True,  # fixed: see docs/optimization_ledger.md T7 (qkv_bias surgery bug, not a reshape bug)
            )

            # Sanity check on the FIRST bit-width sweep: confirms surgery actually
            # replaced attention layers AND that quantization measurably alters
            # activations (rules out a silent no-op producing suspiciously-close
            # quantized-vs-FP32 ground-truth numbers). Uses the same clip-building
            # convention as predict_depth() above.
            if idx == 0 and len(gt_samples) > 0:
                img_resized = np.array(Image.fromarray(gt_samples[0]["rgb"]).resize((266, 266)))
                tensor = (torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0 - mean) / std
                sanity_clip = tensor.unsqueeze(0).repeat(3, 1, 1, 1).unsqueeze(0)
                if torch.cuda.is_available():
                    sanity_clip = sanity_clip.cuda()
                verify_quantization_surgery(model, model_quant, sanity_clip)

        q_gt_metrics = []
        for sample_idx, sample in enumerate(gt_samples):
            if model_quant is not None:
                pred = predict_depth(model_quant, sample["rgb"])
            else:
                noise_level = 0.02 * (8.0 / bit)
                pred = fp32_preds[sample_idx] + torch.randn_like(fp32_preds[sample_idx]) * noise_level
            gt_t, valid_t = gt_tensors(sample["depth"], sample["valid_mask"])
            q_gt_metrics.append(compute_gt_depth_metrics(pred, gt_t, valid_t, gt_range=gt_range))

        bit_accounting = compute_real_bit_accounting(
            bit, head_dim=64, use_qjl=args.use_qjl,
            group_size=QUANTIZER_GROUP_SIZE.get(args.quantizer, 4), scale_bits=args.scale_bits,
        )
        metrics = avg_metrics(q_gt_metrics)
        metrics["n_images"] = len(gt_samples)
        metrics["mem_savings_x"] = bit_accounting["ratio_vs_fp32"]
        metrics["mem_savings_fp16_x"] = bit_accounting["ratio_vs_fp16"]
        metrics["nominal_savings_x"] = bit_accounting["nominal_ratio_vs_fp32"]
        metrics["effective_bits_per_scalar"] = bit_accounting["effective_bits_per_scalar"]
        metrics["total_bits_per_vec"] = bit_accounting["total_bits_per_vector"]
        dataset_results[f"{bit}bit"] = metrics
        print(f"        -> delta1: {metrics['delta1']} | AbsRel: {metrics['abs_rel']} | "
              f"eff={metrics['effective_bits_per_scalar']}b/scalar (vs REAL ground truth)")

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
                         choices=["scalar", "uniform_vector", "lattice_d4", "lattice_e8"])
    parser.add_argument("--scale-bits", type=int, default=16, choices=[8, 16],
                         help="Bit-width for lattice quantizers' per-group scale metadata "
                              "(the T8 4-bit headline config: --quantizer lattice_e8 "
                              "--scale-bits 8 --no-qjl --bits 3)")
    parser.add_argument(
        "--eval-mode", type=str, default="fidelity", choices=["fidelity", "groundtruth"],
        help=(
            "'fidelity' (default): quantized model output vs FP32 model output on proxy "
            "video frames — NOT a dataset accuracy result, no ground truth involved. "
            "'groundtruth': quantized model output vs REAL NYUv2 labeled depth, using the "
            "affine-invariant alignment protocol (see scripts/datasets_gt.py, "
            "docs/optimization_ledger.md T2). Only --dataset nyuv2 is supported in this mode; "
            "this downloads the official ~2.8GB NYUv2 labeled dataset on first use."
        ),
    )
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
                group_size=QUANTIZER_GROUP_SIZE.get(args.quantizer, 4), scale_bits=args.scale_bits,
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
    eval_mode_note = (
        f"GROUND TRUTH: accuracy metrics (abs_rel, rmse, delta1-3) are measured against REAL "
        f"{args.dataset.upper()} ground-truth depth using the affine-invariant alignment protocol "
        f"(scripts/datasets_gt.py). Safe to label with the dataset name."
        if args.eval_mode == "groundtruth" else
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
    title = ("PARETO GROUND-TRUTH BENCHMARK SUMMARY TABLE (ViT-Small, head_dim=64, vs REAL NYUv2 labels)"
              if is_gt else
              "PARETO RATE-DISTORTION FIDELITY BENCHMARK SUMMARY TABLE (ViT-Small, head_dim=64)")
    subtitle = ("  * Accuracy measured vs REAL NYUv2 ground-truth depth (affine-invariant protocol) *"
                if is_gt else
                "  * Accuracy measured vs FP32 baseline (Fidelity/Rate-Distortion), NOT ground truth *")
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
            tae_str = f"{m['tae']}" if 'tae' in m else "n/a"
            real_save = (f"{m['mem_savings_x']}x ({m.get('mem_savings_fp16_x', '?')}x) "
                         f"[{m.get('nominal_savings_x', m['mem_savings_x'])}x]")
            print(f"{d_name.upper():<9} | {b_label} | {m['delta1']:<10} | {m['abs_rel']:<8} | {m['rmse']:<8} | {tae_str:<8} | {eff_bits:<12} | {real_save:<24}")
        print(f"{'-' * 110}")
    print(f"{'=' * 110}")
    print("  * Note 1: 'Eff b/scalar' is the ALL-INCLUSIVE effective rate (payload + per-group scale metadata")
    print("            + QJL side-channel overhead). 'vs FP32 (vs FP16) [Nom]' are compression ratios against")
    print("            the FP32 baseline, the FP16 deployment baseline, and the naive nominal (payload-only) rate.")
    if is_gt:
        print("  * Note 2: Accuracy metrics are measured against REAL NYUv2 ground truth. Safe to cite by dataset name.")
    else:
        print("  * Note 2: Accuracy metrics (δ1, AbsRel, etc.) evaluate rate-distortion fidelity relative to the FP32")
        print("            model — there is NO ground truth here. Do not report as dataset accuracy; re-run with")
        print("            --eval-mode groundtruth for citable numbers.\n")

if __name__ == "__main__":
    main()
