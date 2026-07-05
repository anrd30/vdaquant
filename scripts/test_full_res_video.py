#!/usr/bin/env python3
"""
C
Full Resolution PTQ Custom Video Evaluation Script
Runs the VDA PTQ sweep (FP32, 8-bit, 4-bit) on a single video at its full original resolution.
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

vda_path = REPO_ROOT / "Video-Depth-Anything"
if vda_path.exists() and str(vda_path) not in sys.path:
    sys.path.insert(0, str(vda_path))

from research.models.rotated_attention import apply_rotated_quantization_to_vda

def create_labeled_panel(img: np.ndarray, label: str, is_depth: bool = True) -> np.ndarray:
    if is_depth:
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        panel = cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)
    else:
        panel = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(panel, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return panel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="videos/video2.mp4", help="Path to input video")
    parser.add_argument("--bits", nargs="+", type=int, default=[8, 4])
    parser.add_argument("--max-frames", type=int, default=30, help="Max frames to process")
    parser.add_argument("--output-dir", type=str, default="outputs/custom_videos", help="Output directory")
    args = parser.parse_args()

    video_path = REPO_ROOT / args.video
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not video_path.exists():
        print(f"Error: {video_path} not found.")
        return

    print(f"\nProcessing {video_path.name} at full resolution...")
    
    # Read frames and determine resolution
    cap = cv2.VideoCapture(str(video_path))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    
    # ViT models require dimensions to be a multiple of 14 (patch size)
    new_w = max(14, round(orig_w / 14) * 14)
    new_h = max(14, round(orig_h / 14) * 14)
    print(f"  Original Resolution: {orig_w}x{orig_h}")
    print(f"  Padded for ViT Model: {new_w}x{new_h}")

    frames = []
    while len(frames) < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    if not frames:
        print("No frames read from video.")
        return

    # Load Model
    try:
        import video_depth_anything.dinov2_layers.attention as dino_attn
        import video_depth_anything.motion_module.attention as motion_attn
        if not torch.cuda.is_available():
            dino_attn.memory_efficient_attention = lambda q, k, v, attn_bias=None, **kwargs: torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
            motion_attn.XFORMERS_AVAILABLE = False
            
        from video_depth_anything.video_depth import VideoDepthAnything
        model_configs = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        model = VideoDepthAnything(**model_configs).eval()
        
        ckpt_path = REPO_ROOT / "checkpoints" / "video_depth_anything_vits.pth"
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
            ckpt_loaded = True
        else:
            print("Checkpoint not found!")
            return

        if torch.cuda.is_available():
            model = model.cuda()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    frame_tensors = []
    rgb_resized = []
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for f in frames:
        img_res = np.array(Image.fromarray(f).resize((new_w, new_h), Image.Resampling.LANCZOS))
        rgb_resized.append(img_res)
        tensor = (torch.from_numpy(img_res).float().permute(2, 0, 1) / 255.0 - mean) / std
        frame_tensors.append(tensor)
        
    video_input = torch.stack(frame_tensors, dim=0).unsqueeze(0)
    if torch.cuda.is_available():
        video_input = video_input.cuda()

    depth_predictions = {}
    
    print("  Running FP32 Baseline (Full Res)...")
    with torch.no_grad():
        depth_predictions["FP32 Baseline"] = model(video_input)[0].cpu().numpy()
        
    for bit in args.bits:
        print(f"  Running {bit}-bit PTQ (Full Res)...")
        model_quant = VideoDepthAnything(**model_configs).eval()
        model_quant.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        if torch.cuda.is_available():
            model_quant = model_quant.cuda()
            
        model_quant = apply_rotated_quantization_to_vda(
            model_quant, bits=bit, quantizer='lattice_d4', use_qjl=True, verbose=False, replace_temporal=False
        )
        with torch.no_grad():
            depth_predictions[f"{bit}-bit D4"] = model_quant(video_input)[0].cpu().numpy()

    # Layout: Top [RGB, FP32], Bottom [8-bit, 4-bit]
    out_path = output_dir / f"ptq_fullres_{video_path.name}"
    grid_width, grid_height = new_w * 2, new_h * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(out_path), fourcc, fps, (grid_width, grid_height))
    
    print(f"  Assembling 2x2 Full Resolution Grid: {grid_width}x{grid_height}")
    for t in range(len(frames)):
        fp32_frame = depth_predictions["FP32 Baseline"][t]
        d_min, d_max = fp32_frame.min(), fp32_frame.max()
        d_range = max(d_max - d_min, 1e-8)
        
        p_rgb = create_labeled_panel(rgb_resized[t], f"Input Frame {t+1}/{len(frames)}", is_depth=False)
        p_fp32 = create_labeled_panel((depth_predictions["FP32 Baseline"][t] - d_min) / d_range, "FP32 Baseline")
        p_8bit = create_labeled_panel((depth_predictions["8-bit D4"][t] - d_min) / d_range, "8-bit PTQ")
        p_4bit = create_labeled_panel((depth_predictions["4-bit D4"][t] - d_min) / d_range, "4-bit PTQ")
        
        top_row = np.hstack([p_rgb, p_fp32])
        bot_row = np.hstack([p_8bit, p_4bit])
        grid_frame = np.vstack([top_row, bot_row])
        out_video.write(grid_frame)
        
    out_video.release()
    print(f"  Saved full resolution comparison grid to {out_path}")

if __name__ == "__main__":
    main()
