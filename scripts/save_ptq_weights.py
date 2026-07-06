#!/usr/bin/env python3
"""
Save PTQ Model Weights
Loads the FP32 VDA model, applies PTQ (8-bit, 4-bit, 3-bit), and saves the quantized weights to disk.
"""
import os
import sys
import argparse
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

vda_path = REPO_ROOT / "Video-Depth-Anything"
if vda_path.exists() and str(vda_path) not in sys.path:
    sys.path.insert(0, str(vda_path))

from research.models.rotated_attention import apply_rotated_quantization_to_vda

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", nargs="+", type=int, default=[8, 4, 3])
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory for weights")
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Base FP32 Model...")
    try:
        import video_depth_anything.dinov2_layers.attention as dino_attn
        import video_depth_anything.motion_module.attention as motion_attn
        if not torch.cuda.is_available():
            dino_attn.memory_efficient_attention = lambda q, k, v, attn_bias=None, **kwargs: torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
            motion_attn.XFORMERS_AVAILABLE = False
            
        from video_depth_anything.video_depth import VideoDepthAnything
        model_configs = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        
        ckpt_path = REPO_ROOT / "checkpoints" / "video_depth_anything_vits.pth"
        if not ckpt_path.exists():
            print(f"Error: Checkpoint not found at {ckpt_path}")
            return
            
    except Exception as e:
        print(f"Failed to load model architecture: {e}")
        return

    # Process each bit-width
    for bit in args.bits:
        print(f"\n--- Processing {bit}-bit PTQ ---")
        model_quant = VideoDepthAnything(**model_configs).eval()
        model_quant.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        
        # We don't necessarily need CUDA just to quantize and save, but doing it on CPU is fine and avoids OOM.
        
        print(f"Applying Rotated Quantization ({bit}-bit)...")
        model_quant = apply_rotated_quantization_to_vda(
            model_quant, bits=bit, quantizer='lattice_d4', use_qjl=True, verbose=False, replace_temporal=False
        )
        
        out_path = output_dir / f"video_depth_anything_vits_{bit}bit_ptq.pth"
        print(f"Saving quantized weights to {out_path}...")
        torch.save(model_quant.state_dict(), out_path)
        print("Done!")

    print("\nAll PTQ encodings/weights have been successfully saved!")

if __name__ == "__main__":
    main()
