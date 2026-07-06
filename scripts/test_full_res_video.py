#!/usr/bin/env python3
"""
Full Resolution PTQ Custom Video Evaluation Script
Runs the VDA PTQ sweep (FP32, 8-bit, 4-bit, etc.) on multiple videos at full original resolution.
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
    parser.add_argument("--videos-dir", type=str, default="videos", help="Directory containing mp4s")
    parser.add_argument("--bits", nargs="+", type=int, default=[8, 4, 3])
    parser.add_argument("--max-frames", type=int, default=5, help="Max frames to process")
    parser.add_argument("--output-dir", type=str, default="outputs/new", help="Output directory")
    args = parser.parse_args()

    input_path = REPO_ROOT / args.videos_dir
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file() and input_path.suffix == '.mp4':
        video_files = [input_path]
    else:
        video_files = list(input_path.glob("*.mp4"))
    if not video_files:
        print(f"Error: No .mp4 files found in {input_path}")
        return

    # Load Model Once
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

    # Process each video
    for video_path in video_files:
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
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        if not frames:
            print("  No frames read from video, skipping.")
            continue
            
        print(f"  Total frames extracted: {len(frames)}")

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Precompute resized frames to save time
        rgb_resized = []
        frame_tensors_all = []
        for f in frames:
            img_res = np.array(Image.fromarray(f).resize((new_w, new_h), Image.Resampling.LANCZOS))
            rgb_resized.append(img_res)
            tensor = (torch.from_numpy(img_res).float().permute(2, 0, 1) / 255.0 - mean) / std
            frame_tensors_all.append(tensor)

        depth_predictions = {}
        chunk_size = args.max_frames
        
        print(f"  Running FP32 Baseline in chunks of {chunk_size}...")
        fp32_preds = []
        for start_idx in range(0, len(frames), chunk_size):
            end_idx = min(start_idx + chunk_size, len(frames))
            chunk_tensors = frame_tensors_all[start_idx:end_idx]
            video_input = torch.stack(chunk_tensors, dim=0).unsqueeze(0)
            if torch.cuda.is_available():
                video_input = video_input.cuda()
            with torch.no_grad():
                fp32_preds.append(model(video_input)[0].cpu().numpy())
        depth_predictions["FP32 Baseline"] = np.concatenate(fp32_preds, axis=0)
            
        for bit in args.bits:
            print(f"  Running {bit}-bit PTQ in chunks of {chunk_size}...")
            model_quant = VideoDepthAnything(**model_configs).eval()
            model_quant.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
            if torch.cuda.is_available():
                model_quant = model_quant.cuda()
                
            model_quant = apply_rotated_quantization_to_vda(
                model_quant, bits=bit, quantizer='lattice_d4', use_qjl=True, verbose=False, replace_temporal=False
            )
            
            q_preds = []
            for start_idx in range(0, len(frames), chunk_size):
                end_idx = min(start_idx + chunk_size, len(frames))
                chunk_tensors = frame_tensors_all[start_idx:end_idx]
                video_input = torch.stack(chunk_tensors, dim=0).unsqueeze(0)
                if torch.cuda.is_available():
                    video_input = video_input.cuda()
                with torch.no_grad():
                    q_preds.append(model_quant(video_input)[0].cpu().numpy())
            depth_predictions[f"{bit}-bit D4"] = np.concatenate(q_preds, axis=0)
            
            del model_quant
            torch.cuda.empty_cache()

        # Layout: Dynamic based on args.bits
        out_path = output_dir / f"ptq_fullres_{video_path.name}"
        
        num_cols = max(2, (len(args.bits) + 2 + 1) // 2)
        grid_width, grid_height = new_w * num_cols, new_h * 2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(str(out_path), fourcc, fps, (grid_width, grid_height))
        
        print(f"  Assembling Full Resolution Grid: {grid_width}x{grid_height}")
        for t in range(len(frames)):
            fp32_frame = depth_predictions["FP32 Baseline"][t]
            d_min, d_max = fp32_frame.min(), fp32_frame.max()
            d_range = max(d_max - d_min, 1e-8)
            
            p_rgb = create_labeled_panel(rgb_resized[t], f"Input Frame {t+1}/{len(frames)}", is_depth=False)
            p_fp32 = create_labeled_panel((depth_predictions["FP32 Baseline"][t] - d_min) / d_range, "FP32 Baseline")
            
            top_panels = [p_rgb, p_fp32]
            bot_panels = []
            
            for bit in args.bits:
                p = create_labeled_panel((depth_predictions[f"{bit}-bit D4"][t] - d_min) / d_range, f"{bit}-bit PTQ")
                if len(top_panels) < num_cols:
                    top_panels.append(p)
                else:
                    bot_panels.append(p)
                    
            # Pad with black panels to complete the grid
            while len(top_panels) < num_cols:
                top_panels.append(np.zeros_like(p_rgb))
            while len(bot_panels) < num_cols:
                bot_panels.append(np.zeros_like(p_rgb))
                
            top_row = np.hstack(top_panels)
            bot_row = np.hstack(bot_panels)
            grid_frame = np.vstack([top_row, bot_row])
            out_video.write(grid_frame)
            
        out_video.release()
        print(f"  Saved full resolution comparison grid to {out_path}")

if __name__ == "__main__":
    main()
