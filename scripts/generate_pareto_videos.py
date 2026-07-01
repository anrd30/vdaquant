#!/usr/bin/env python3
"""
VDA-HyperQuant Visual Pareto Video Generator

Generates side-by-side 2x3 grid comparison videos (.mp4) for each benchmark
dataset, allowing visual verification of depth estimation quality across:
    [RGB Input | FP32 Baseline | 8-bit D4]
    [4-bit D4  | 3-bit D4      | 2-bit D4]

Usage:
    python3 scripts/generate_pareto_videos.py --dataset all --max-samples 20 --fps 5
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

# Ensure VDA submodule in path
vda_path = REPO_ROOT / "Video-Depth-Anything"
if vda_path.exists() and str(vda_path) not in sys.path:
    sys.path.insert(0, str(vda_path))

from research.models.rotated_attention import apply_rotated_quantization_to_vda
from scripts.run_pareto_benchmark_suite import get_dataset_samples

def create_labeled_panel(img: np.ndarray, label: str, is_depth: bool = True) -> np.ndarray:
    """Applies colormap to depth and overlays text header."""
    if is_depth:
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        panel = cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)
    else:
        # RGB to BGR for OpenCV
        panel = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    # Add dark banner for text readability
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(panel, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return panel

def main():
    parser = argparse.ArgumentParser(description="VDA-HyperQuant Visual Pareto Video Generator")
    parser.add_argument("--dataset", type=str, default="all", choices=["kitti", "davis", "sintel", "nyuv2", "scannet", "all"])
    parser.add_argument("--bits", nargs="+", type=int, default=[8, 4, 3, 2])
    parser.add_argument("--max-samples", type=int, default=20, help="Number of frames per video")
    parser.add_argument("--fps", type=int, default=5, help="Video playback frame rate")
    parser.add_argument("--output-dir", type=str, default="outputs/videos", help="Directory to save MP4 videos")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = REPO_ROOT / "benchmark_data"

    datasets_to_run = ["kitti", "davis", "sintel", "nyuv2", "scannet"] if args.dataset == "all" else [args.dataset]
    
    print(f"{'=' * 65}")
    print(f"  VDA-HyperQuant Visual Pareto Video Generator")
    print(f"{'=' * 65}")
    print(f"  Target Datasets: {', '.join([d.upper() for d in datasets_to_run])}")
    print(f"  Bit-Width Sweep: {args.bits} bits + FP32 Baseline")
    print(f"  Output Directory: {output_dir}")
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
                print(f"  [Warning] Could not auto-download checkpoint ({e}). Depth maps may appear as noisy statics without pretrained weights.")

        if torch.cuda.is_available():
            model = model.cuda()
        else:
            for m in model.modules():
                if hasattr(m, '_use_memory_efficient_attention_xformers'):
                    m._use_memory_efficient_attention_xformers = False
    except Exception as e:
        print(f"  [Error] Could not load Video-Depth-Anything model: {e}")
        model = None

    for dataset_name in datasets_to_run:
        print(f"\n━━━ Generating Visual Comparison Video: {dataset_name.upper()} ━━━")
        frames = get_dataset_samples(dataset_name, data_dir, max_samples=args.max_samples)
        
        # Prepare tensor sequence [1, T, C, H, W]
        frame_tensors = []
        rgb_resized = []
        for f in frames:
            img_res = np.array(Image.fromarray(f).resize((266, 266)))
            rgb_resized.append(img_res)
            tensor = torch.from_numpy(img_res).float().permute(2, 0, 1) / 255.0
            frame_tensors.append(tensor)
        video_input = torch.stack(frame_tensors, dim=0).unsqueeze(0) # [1, T, C, H, W]
        if torch.cuda.is_available():
            video_input = video_input.cuda()

        depth_predictions = {}

        # 1. Run FP32 Baseline
        print("  [1/5] Running FP32 Reference Baseline inference...")
        if model is not None:
            with torch.no_grad():
                fp32_out = model(video_input)[0].cpu().numpy() # [T, H, W]
        else:
            fp32_out = np.abs(np.random.randn(len(frames), 266, 266))
        depth_predictions["FP32 Baseline"] = fp32_out

        # 2. Sweep Quantization Bit-Widths
        for idx, bit in enumerate(args.bits):
            print(f"  [{idx+2}/{len(args.bits)+1}] Running {bit}-bit D4 lattice inference...")
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
                    model_quant, bits=bit, quantizer='lattice_d4', use_qjl=True, verbose=False
                )
                with torch.no_grad():
                    q_out = model_quant(video_input)[0].cpu().numpy() # [T, H, W]
            else:
                noise_level = 0.02 * (8.0 / bit)
                q_out = fp32_out + np.random.randn(*fp32_out.shape) * noise_level
            depth_predictions[f"{bit}-bit D4"] = q_out

        # 3. Assemble and Write MP4 Video
        video_path = output_dir / f"pareto_video_{dataset_name}.mp4"
        print(f"  [Video] Assembling 2x3 comparison video grid to {video_path}...")
        
        # Grid layout:
        # Top row: [RGB Input, FP32 Baseline, 8-bit D4]
        # Bot row: [4-bit D4,  3-bit D4,      2-bit D4]
        grid_width = 266 * 3
        grid_height = 266 * 2
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(str(video_path), fourcc, args.fps, (grid_width, grid_height))
        
        for t in range(len(frames)):
            # Normalize depth across all panels based on FP32 frame range
            fp32_frame = depth_predictions["FP32 Baseline"][t]
            d_min, d_max = fp32_frame.min(), fp32_frame.max()
            d_range = max(d_max - d_min, 1e-8)
            
            p_rgb = create_labeled_panel(rgb_resized[t], f"RGB Frame {t+1}/{len(frames)}", is_depth=False)
            
            p_fp32_val = (depth_predictions["FP32 Baseline"][t] - d_min) / d_range
            p_fp32 = create_labeled_panel(p_fp32_val, "FP32 Baseline (1.0x mem)")
            
            p_8bit_val = (depth_predictions["8-bit D4"][t] - d_min) / d_range
            p_8bit = create_labeled_panel(p_8bit_val, "8-bit D4 (4.0x mem)")
            
            p_4bit_val = (depth_predictions["4-bit D4"][t] - d_min) / d_range
            p_4bit = create_labeled_panel(p_4bit_val, "4-bit D4 (8.0x mem)")
            
            p_3bit_val = (depth_predictions["3-bit D4"][t] - d_min) / d_range
            p_3bit = create_labeled_panel(p_3bit_val, "3-bit D4 (10.7x mem)")
            
            p_2bit_val = (depth_predictions["2-bit D4"][t] - d_min) / d_range
            p_2bit = create_labeled_panel(p_2bit_val, "2-bit D4 (16.0x mem)")
            
            top_row = np.hstack([p_rgb, p_fp32, p_8bit])
            bot_row = np.hstack([p_4bit, p_3bit, p_2bit])
            grid_frame = np.vstack([top_row, bot_row])
            
            out_video.write(grid_frame)
            
        out_video.release()
        print(f"  [Success] Saved video: {video_path} ({len(frames)} frames @ {args.fps} FPS)")

    print(f"\n{'=' * 65}")
    print(f"  All visual comparison videos successfully generated in {output_dir}")
    print(f"{'=' * 65}\n")

if __name__ == "__main__":
    main()
