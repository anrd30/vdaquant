#!/usr/bin/env python3
"""
PTQ Custom Video Evaluation Script
Runs the VDA PTQ sweep (FP32, 8-bit, 4-bit, 3-bit, 2-bit) on custom videos.
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
    cv2.putText(panel, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return panel

def load_video_frames(video_path: Path, max_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", type=str, default="videos", help="Directory containing custom mp4s")
    parser.add_argument("--bits", nargs="+", type=int, default=[8, 4, 3, 2])
    parser.add_argument("--max-frames", type=int, default=30, help="Max frames per video to process")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS")
    parser.add_argument("--output-dir", type=str, default="outputs/custom_videos", help="Output directory")
    args = parser.parse_args()

    videos_dir = REPO_ROOT / args.videos_dir
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = list(videos_dir.glob("*.mp4"))
    if not video_files:
        print(f"No .mp4 files found in {videos_dir}")
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

    for v_path in video_files:
        print(f"\nProcessing {v_path.name}...")
        frames = load_video_frames(v_path, args.max_frames)
        if not frames:
            continue
            
        frame_tensors = []
        rgb_resized = []
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        for f in frames:
            # Resize to exactly 266x266 to match the grid layout requirement
            img_res = np.array(Image.fromarray(f).resize((266, 266)))
            rgb_resized.append(img_res)
            tensor = (torch.from_numpy(img_res).float().permute(2, 0, 1) / 255.0 - mean) / std
            frame_tensors.append(tensor)
            
        video_input = torch.stack(frame_tensors, dim=0).unsqueeze(0)
        if torch.cuda.is_available():
            video_input = video_input.cuda()

        depth_predictions = {}
        
        print("  Running FP32 Baseline...")
        with torch.no_grad():
            depth_predictions["FP32 Baseline"] = model(video_input)[0].cpu().numpy()
            
        for bit in args.bits:
            print(f"  Running {bit}-bit PTQ...")
            model_quant = VideoDepthAnything(**model_configs).eval()
            model_quant.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
            if torch.cuda.is_available():
                model_quant = model_quant.cuda()
                
            model_quant = apply_rotated_quantization_to_vda(
                model_quant, bits=bit, quantizer='lattice_d4', use_qjl=True, verbose=False, replace_temporal=False
            )
            with torch.no_grad():
                depth_predictions[f"{bit}-bit D4"] = model_quant(video_input)[0].cpu().numpy()

        out_path = output_dir / f"ptq_{v_path.name}"
        grid_width, grid_height = 266 * 3, 266 * 2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(str(out_path), fourcc, args.fps, (grid_width, grid_height))
        
        for t in range(len(frames)):
            fp32_frame = depth_predictions["FP32 Baseline"][t]
            d_min, d_max = fp32_frame.min(), fp32_frame.max()
            d_range = max(d_max - d_min, 1e-8)
            
            p_rgb = create_labeled_panel(rgb_resized[t], f"Input Frame {t+1}/{len(frames)}", is_depth=False)
            p_fp32 = create_labeled_panel((depth_predictions["FP32 Baseline"][t] - d_min) / d_range, "FP32 Baseline")
            p_8bit = create_labeled_panel((depth_predictions["8-bit D4"][t] - d_min) / d_range, "8-bit PTQ")
            p_4bit = create_labeled_panel((depth_predictions["4-bit D4"][t] - d_min) / d_range, "4-bit PTQ")
            p_3bit = create_labeled_panel((depth_predictions["3-bit D4"][t] - d_min) / d_range, "3-bit PTQ")
            p_2bit = create_labeled_panel((depth_predictions["2-bit D4"][t] - d_min) / d_range, "2-bit PTQ")
            
            grid_frame = np.vstack([np.hstack([p_rgb, p_fp32, p_8bit]), np.hstack([p_4bit, p_3bit, p_2bit])])
            out_video.write(grid_frame)
            
        out_video.release()
        print(f"  Saved comparison grid to {out_path}")

if __name__ == "__main__":
    main()
