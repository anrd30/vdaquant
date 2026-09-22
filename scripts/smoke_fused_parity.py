"""
Smoke test: does the fused-kernel path produce output close to the
simulator path on a real VDA temporal-attention layer shape?

Prints per-layer max diff. We expect small differences (fp32 kernel
accumulator vs fp16 SDPA + QJL disabled in fused mode). "Close enough"
for the wall-clock story is max diff < ~0.05 relative to output std;
larger than that means fusion is wrong, don't run the bench.

Usage:
  python scripts/smoke_fused_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from research.models.rotated_attention import RotatedTemporalAttention


VDA_LAYER_SHAPES = [
    # (label,   B,  h,  M,  d)
    ("mm0",     8,  8, 1369, 192 // 8),
    ("mm1",     8,  8,  361, 384 // 8),
    ("mm2",     8,  8, 1369,  64 // 8),
    ("mm3",     8,  8, 5476,  64 // 8),
]


def build_layer(dim: int, num_heads: int) -> RotatedTemporalAttention:
    return RotatedTemporalAttention(
        dim=dim, num_heads=num_heads, qkv_bias=True,
        bits=3, quantizer='lattice_bw16', use_qjl=True,
        scale_bits=16, use_rotation=True, rht_seed=0,
    ).cuda().eval()


def main():
    torch.manual_seed(0)
    device = "cuda"
    print(f"{'layer':<6} {'B':>3} {'h':>3} {'M':>6} {'d':>4} "
          f"{'sim_std':>9} {'fus_std':>9} {'max_diff':>10} {'rel_diff':>10}")
    for label, B, h, M, d in VDA_LAYER_SHAPES:
        dim = h * d
        layer = build_layer(dim=dim, num_heads=h)
        # Fake token stream shaped like VDA's flatten (batch*tokens, frames, feat).
        # Skip the reshape gymnastics: call the internal path with pre-shaped
        # Q/K/V by passing hidden_states of shape (B, M, C).
        x = torch.randn(B, M, dim, device=device, dtype=torch.float16)

        with torch.no_grad():
            layer.use_fused_kernel = False
            out_sim = layer(hidden_states=x, encoder_hidden_states=x).float()
            layer.use_fused_kernel = True
            out_fus = layer(hidden_states=x, encoder_hidden_states=x).float()
        diff = (out_sim - out_fus).abs()
        sim_std = out_sim.std().item()
        fus_std = out_fus.std().item()
        max_d = diff.max().item()
        rel = max_d / max(sim_std, 1e-6)
        print(f"{label:<6} {B:>3} {h:>3} {M:>6} {d:>4} "
              f"{sim_std:>9.3f} {fus_std:>9.3f} {max_d:>10.4f} {rel:>10.3f}")
        del layer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
