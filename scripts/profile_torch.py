"""
torch.profiler alternative to nsys for when Nsight Systems isn't
available on the server. Runs one RotatedTemporalAttention forward
through the profiler and prints:

  - top ops sorted by CUDA time
  - top ops sorted by CPU time
  - kernel launch count per op

Trace is also exported as Chrome-tracing JSON, viewable at
chrome://tracing or ui.perfetto.dev after scp-ing it back.

Usage:
  python scripts/profile_torch.py --layer mm1
  python scripts/profile_torch.py --layer mm0 --repeats 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.profiler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from research.models.rotated_attention import RotatedTemporalAttention


LAYER_SHAPES = {
    "mm0": (8, 8, 1369, 24),
    "mm1": (8, 8,  361, 48),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="mm1", choices=list(LAYER_SHAPES.keys()))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "torch_profile"))
    args = ap.parse_args()

    B, h, M, d = LAYER_SHAPES[args.layer]
    dim = h * d
    torch.manual_seed(0)
    x = torch.randn(B, M, dim, device="cuda", dtype=torch.float32)

    layer = RotatedTemporalAttention(
        dim=dim, num_heads=h, qkv_bias=True,
        bits=3, quantizer='lattice_bw16', use_qjl=False,
        scale_bits=16, use_rotation=True, rht_seed=0,
    ).cuda().eval()
    layer.use_fused_kernel = True

    # Warm up (so CUDA graph capture happens outside the profile).
    with torch.no_grad():
        for _ in range(3):
            _ = layer(hidden_states=x, encoder_hidden_states=x)
    torch.cuda.synchronize()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[torch.profiler] tracing {args.repeats} forwards of layer {args.layer} "
          f"(B={B} h={h} M={M} d={d})\n")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            for _ in range(args.repeats):
                with torch.profiler.record_function(f"forward_{args.layer}"):
                    _ = layer(hidden_states=x, encoder_hidden_states=x)
        torch.cuda.synchronize()

    print("========= TOP 30 by CUDA time =========")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    print("\n\n========= TOP 20 by CPU time =========")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    trace_path = out_dir / f"torch_profile_{args.layer}.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"\n[trace saved] {trace_path}")
    print("(scp back and open at ui.perfetto.dev or chrome://tracing)")


if __name__ == "__main__":
    main()
