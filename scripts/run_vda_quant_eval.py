"""
VDA-HyperQuant: End-to-End Evaluation on Video-Depth-Anything

This script:
1. Loads the real Video-Depth-Anything (VDA) model.
2. Applies model surgery (RHT + D4 Lattice VQ + QJL bias correction).
3. Compares the original FP32 model against the quantized model:
   - Numerical differences (MAE, MSE, Pearson Correlation)
   - Inference speed (FPS)
   - Visual output comparisons (saved as PNGs)
"""

import sys
import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Setup paths to include VDA and local modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
VDA_DIR = os.path.join(REPO_DIR, "Video-Depth-Anything")

sys.path.insert(0, REPO_DIR)
sys.path.insert(0, VDA_DIR)

try:
    from video_depth_anything.video_depth import VideoDepthAnything
    from utils.dc_utils import read_video_frames
    VDA_AVAILABLE = True
except ImportError:
    VDA_AVAILABLE = False


def download_sample_resources():
    """Download a sample video and model checkpoint if not present."""
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Download ViT-Small checkpoint if missing
    ckpt_path = "checkpoints/video_depth_anything_vits.pth"
    if not os.path.exists(ckpt_path):
        print("Downloading Video-Depth-Anything checkpoint (ViT-S)...")
        # Direct download from Hugging Face
        url = "https://huggingface.co/depth-anything/Video-Depth-Anything-ViT-Small/resolve/main/video_depth_anything_vits.pth"
        os.system(f"wget -O {ckpt_path} {url}")
        
    # 2. Download sample video if missing
    video_path = "outputs/sample_video.mp4"
    if not os.path.exists(video_path):
        print("Downloading sample video...")
        url = "https://raw.githubusercontent.com/intel-isl/DensePredictionTransformer/master/dpt/assets/dog.jpg"
        # We can also download a small video or use an image repeated as frames
        os.system(f"wget -O outputs/sample_frame.jpg {url}")


def run_eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if not VDA_AVAILABLE:
        print("\n[!] Video-Depth-Anything is not installed or cloned in the current path.")
        print(f"Please clone it into: {VDA_DIR}")
        print("Run: git clone https://github.com/DepthAnything/Video-Depth-Anything.git")
        return

    # Download checkpoints and sample video/images
    download_sample_resources()
    
    # Configuration for ViT-Small VDA
    configs = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    
    # 1. Load baseline FP32 model
    print("\n[1/4] Loading original FP32 model...")
    model_fp32 = VideoDepthAnything(**configs)
    ckpt_path = "checkpoints/video_depth_anything_vits.pth"
    if os.path.exists(ckpt_path):
        model_fp32.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model_fp32 = model_fp32.to(device).eval()
    
    # 2. Prepare sample frames (simulate video sequence)
    print("\n[2/4] Preparing input video frames...")
    # Load frame and repeat to simulate a 10-frame video
    frame_path = "outputs/sample_frame.jpg"
    if os.path.exists(frame_path):
        img = Image.open(frame_path).convert('RGB')
        frame_np = np.array(img)
        frames = [frame_np] * args.num_frames
    else:
        # Fallback to random noise
        print("No sample frame found. Using random noise.")
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)] * args.num_frames
        
    print(f"Running inference on {len(frames)} frames at resolution {args.input_size}x{args.input_size}...")
    
    # Convert frames to tensor [B, T, C, H, W]
    input_frames = []
    for f in frames:
        img_tensor = torch.tensor(f).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = F.interpolate(img_tensor, size=(args.input_size, args.input_size), mode='bilinear', align_corners=False)
        input_frames.append(img_tensor)
        
    input_batch = torch.cat(input_frames, dim=0).unsqueeze(0).to(device)  # [1, T, C, H, W]

    # Run original model inference
    with torch.no_grad():
        start = time.time()
        depths_fp32 = model_fp32(input_batch)
        time_fp32 = (time.time() - start) / len(frames)
    print(f"FP32 Baseline: {1/time_fp32:.1f} FPS ({time_fp32*1000:.1f} ms/frame)")
    
    # 3. Apply VDA-HyperQuant Model Surgery
    print(f"\n[3/4] Rebuilding model with VDA-HyperQuant ({args.bits}-bit {args.quantizer})...")
    from research.models import apply_rotated_quantization_to_vda
    
    model_quant = VideoDepthAnything(**configs)
    if os.path.exists(ckpt_path):
        model_quant.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        
    model_quant = apply_rotated_quantization_to_vda(
        model_quant,
        bits=args.bits,
        quantizer=args.quantizer,
        use_qjl=args.use_qjl,
        verbose=True
    )
    model_quant = model_quant.to(device).eval()
    
    # Run quantized model inference
    with torch.no_grad():
        start = time.time()
        depths_quant = model_quant(input_batch)
        time_quant = (time.time() - start) / len(frames)
    print(f"Quantized Model: {1/time_quant:.1f} FPS ({time_quant*1000:.1f} ms/frame)")
    
    # 4. Compute Metrics & Comparison
    print("\n[4/4] Evaluating depth fidelity...")
    
    pred = depths_quant.flatten().cpu()
    target = depths_fp32.flatten().cpu()
    
    # Scale-invariant / Normalized metrics for depth
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    target_norm = (target - target.min()) / (target.max() - target.min() + 1e-8)
    
    mae = (pred_norm - target_norm).abs().mean().item()
    mse = ((pred_norm - target_norm) ** 2).mean().item()
    
    # Pearson Correlation
    p_centered = pred_norm - pred_norm.mean()
    t_centered = target_norm - target_norm.mean()
    correlation = (p_centered * t_centered).sum() / (p_centered.norm() * t_centered.norm() + 1e-8)
    correlation = correlation.item()
    
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS (vs FP32 Baseline)")
    print("=" * 60)
    print(f"  Mean Absolute Error (MAE):     {mae:.5f}")
    print(f"  Mean Squared Error (MSE):      {mse:.6f}")
    print(f"  Pearson Correlation:           {correlation:.5f}")
    print(f"  Memory Savings (KV-Cache):     {32 / args.bits:.1f}x")
    print(f"  FPS (FP32 vs Quantized):       {1/time_fp32:.1f} vs {1/time_quant:.1f}")
    print("=" * 60)
    
    # Save a comparison image
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # First frame depth maps
    d_fp32 = depths_fp32[0, 0].cpu().numpy()
    d_quant = depths_quant[0, 0].cpu().numpy()
    diff = np.abs(d_fp32 - d_quant)
    
    axes[0].imshow(d_fp32, cmap='viridis')
    axes[0].set_title("FP32 Depth Map")
    axes[0].axis('off')
    
    axes[1].imshow(d_quant, cmap='viridis')
    axes[1].set_title(f"Quantized Depth ({args.bits}-bit)")
    axes[1].axis('off')
    
    im = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f"Absolute Difference (Max: {diff.max():.4f})")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig("outputs/depth_comparison.png", dpi=150)
    print("Comparison image saved to outputs/depth_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VDA-HyperQuant")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits (3 or 4)")
    parser.add_argument("--quantizer", type=str, default="lattice_d4", choices=["scalar", "uniform_vector", "lattice_d4"])
    parser.add_argument("--use_qjl", action="store_true", default=True, help="Use QJL bias correction")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of frames to test")
    parser.add_argument("--input_size", type=int, default=266, help="Inference resolution")
    
    args = parser.parse_args()
    run_eval(args)
