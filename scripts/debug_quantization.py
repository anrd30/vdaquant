#!/usr/bin/env python3
"""
Diagnostic script to pinpoint why quantized VDA models produce noise.
Run on Colab: python3 scripts/debug_quantization.py
"""
import os, sys, torch, numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
vda_path = REPO_ROOT / "Video-Depth-Anything"
if vda_path.exists():
    sys.path.insert(0, str(vda_path))

# Patch xformers
try:
    import video_depth_anything.dinov2_layers.attention as dino_attn
    import video_depth_anything.motion_module.attention as motion_attn
    if not torch.cuda.is_available():
        dino_attn.memory_efficient_attention = lambda q, k, v, attn_bias=None, **kwargs: torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        motion_attn.XFORMERS_AVAILABLE = False
except:
    pass

from video_depth_anything.video_depth import VideoDepthAnything
from research.models.rotated_attention import apply_rotated_quantization_to_vda

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 1. Build model and load checkpoint
model_configs = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
model = VideoDepthAnything(**model_configs).eval()

ckpt_path = None
for p in [REPO_ROOT / "checkpoints" / "video_depth_anything_vits.pth",
          Path("/content/vdaquant/checkpoints/video_depth_anything_vits.pth"),
          Path("/content/vdaquant/vdaquant/checkpoints/video_depth_anything_vits.pth")]:
    if p.exists() and p.stat().st_size > 10_000_000:
        ckpt_path = p
        break
if ckpt_path:
    print(f"Loading checkpoint: {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
else:
    print("ERROR: No valid checkpoint found!")
    sys.exit(1)

model = model.to(device)

# 2. Create a simple test input (1 frame, 266x266)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
fake_img = torch.rand(3, 266, 266)
normalized = (fake_img - mean) / std
video_input = normalized.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, C, H, W]

print(f"\nInput shape: {video_input.shape}")
print(f"Input range: [{video_input.min():.3f}, {video_input.max():.3f}]")

# 3. FP32 baseline
with torch.no_grad():
    fp32_out = model(video_input)
    if isinstance(fp32_out, tuple):
        fp32_depth = fp32_out[0]
    else:
        fp32_depth = fp32_out
    print(f"\n=== FP32 Output ===")
    print(f"  Shape: {fp32_depth.shape}")
    print(f"  Range: [{fp32_depth.min():.6f}, {fp32_depth.max():.6f}]")
    print(f"  Mean:  {fp32_depth.mean():.6f}")
    print(f"  NaN:   {torch.isnan(fp32_depth).any()}")
    print(f"  Inf:   {torch.isinf(fp32_depth).any()}")

# 4. Check what modules exist before surgery
print(f"\n=== Model Attention Modules (before surgery) ===")
attn_count = 0
for name, module in model.named_modules():
    if module.__class__.__name__ in ('MemEffAttention', 'Attention', 'CrossAttention', 'TemporalAttention'):
        print(f"  {name}: {module.__class__.__name__}")
        attn_count += 1
print(f"  Total attention layers found: {attn_count}")

# 5. Build quantized model 
print(f"\n=== Building 4-bit Quantized Model ===")
model_quant = VideoDepthAnything(**model_configs).eval()
model_quant.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
model_quant = model_quant.to(device)

# Check weight stats BEFORE surgery
print(f"\n--- Weight stats BEFORE surgery ---")
for name, p in list(model_quant.named_parameters())[:5]:
    print(f"  {name}: min={p.min():.4f}, max={p.max():.4f}, mean={p.mean():.6f}, nan={torch.isnan(p).any()}")

model_quant = apply_rotated_quantization_to_vda(
    model_quant, bits=4, quantizer='lattice_d4', use_qjl=True, verbose=True
)

# Check weight stats AFTER surgery
print(f"\n--- Weight stats AFTER surgery ---")
for name, p in list(model_quant.named_parameters())[:5]:
    print(f"  {name}: min={p.min():.4f}, max={p.max():.4f}, mean={p.mean():.6f}, nan={torch.isnan(p).any()}")

# 6. Run quantized inference
with torch.no_grad():
    q_out = model_quant(video_input)
    if isinstance(q_out, tuple):
        q_depth = q_out[0]
    else:
        q_depth = q_out
    print(f"\n=== 4-bit Quantized Output ===")
    print(f"  Shape: {q_depth.shape}")
    print(f"  Range: [{q_depth.min():.6f}, {q_depth.max():.6f}]")
    print(f"  Mean:  {q_depth.mean():.6f}")
    print(f"  NaN:   {torch.isnan(q_depth).any()}")
    print(f"  Inf:   {torch.isinf(q_depth).any()}")

# 7. Compare
if not torch.isnan(q_depth).any() and not torch.isnan(fp32_depth).any():
    diff = (fp32_depth - q_depth).abs()
    print(f"\n=== FP32 vs 4-bit Difference ===")
    print(f"  Max diff:  {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  Correlation: {torch.corrcoef(torch.stack([fp32_depth.flatten(), q_depth.flatten()]))[0,1]:.6f}")

# 8. Test individual components
print(f"\n=== Testing Hadamard rotation round-trip ===")
from research.transforms.hadamard import HadamardRotation
rot = HadamardRotation(64).to(device)
test_vec = torch.randn(1, 6, 100, 64, device=device)
roundtrip = rot.inverse(rot(test_vec))
rt_error = (test_vec - roundtrip).abs().max()
print(f"  Max round-trip error: {rt_error:.10f}")

print(f"\n=== Testing D4 quantizer ===")
from research.quantizers.lattice_vq import LatticeD4Quantizer
quant = LatticeD4Quantizer(bits=4)
test_data = torch.randn(1, 6, 100, 64, device=device)
q_data, info = quant(test_data)
q_error = (test_data - q_data).abs().mean()
print(f"  Mean quantization error: {q_error:.6f}")
print(f"  NaN in quantized: {torch.isnan(q_data).any()}")

print(f"\n=== Done ===")
