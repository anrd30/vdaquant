"""
Measure the KV-cache memory of the packed BW16 layout on real VDA-shaped
tiles and report the ratio versus fp16 storage.

This is the "measured memory savings" claim for Paper 2 Table 6.

We do NOT integrate into VDA surgery for this benchmark -- we just
measure what a packed storage container occupies for the SAME tensors
that VDA's temporal attention produces at each layer, and compare
against the fp16 cache size.  Correctness of the packed decode is
covered by kernels/tests/test_bw16_roundtrip.py at the tensor level.

Output: prints a Markdown-friendly table.
"""
from __future__ import annotations

import argparse
import gc
import time

import torch

from kernels.reference.packed_bw16 import pack_bw16, unpack_bw16
from kernels.reference.bw16_codebook import bw16_cosets

try:
    from kernels.triton_kernels.decode_bw16 import decode_bw16_triton
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# VDA temporal-attention KV shapes measured from report_kv_memory.py.
# Each row is (encoder, motion-module, tokens, head_dim).  We generate
# a synthetic KV tile of shape (batch=1, T, tokens, head_dim) matching
# what VDA actually caches per forward.
VDA_LAYERS = [
    ("ViT-S", "mm0_layer3", 1369, 192),
    ("ViT-S", "mm1_layer4",  361, 384),
    ("ViT-S", "mm2_path4",  1369,  64),
    ("ViT-S", "mm3_path3",  5476,  64),
    ("ViT-L", "mm0_layer3", 1369, 1024),
    ("ViT-L", "mm1_layer4",  361, 1024),
    ("ViT-L", "mm2_path4",  1369,  256),
    ("ViT-L", "mm3_path3",  5476,  256),
]


def benchmark_one(encoder: str, layer: str, tokens: int, head_dim: int,
                  T: int = 32, bits: int = 3, device: str = "cuda"):
    """Return a dict of the per-tile memory + accuracy numbers."""
    shape = (1, T, tokens, head_dim)
    x = torch.randn(*shape, device=device, dtype=torch.float16)
    fp16_bytes = x.numel() * 2

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    p = pack_bw16(x.float(), bits=bits, group_size=16)
    if device == "cuda":
        torch.cuda.synchronize()
    t_pack = time.perf_counter() - t0

    t0 = time.perf_counter()
    y = unpack_bw16(p, dtype=torch.float32).to(torch.float16)
    if device == "cuda":
        torch.cuda.synchronize()
    t_unpack = time.perf_counter() - t0

    # Triton fast path timing.
    t_tri = -1.0
    if _TRITON_AVAILABLE and device == "cuda":
        codebook = bw16_cosets(dtype=torch.float32, device=device)
        # Warmup
        _ = decode_bw16_triton(p.codeword_bytes, p.group_scale, codebook,
                               bits=bits); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            _ = decode_bw16_triton(p.codeword_bytes, p.group_scale, codebook,
                                   bits=bits)
        torch.cuda.synchronize()
        t_tri = (time.perf_counter() - t0) / 5 * 1000

    err = (y - x).abs()
    peak = (torch.cuda.max_memory_allocated() / 1024**2) if device == "cuda" else -1.0

    return {
        "encoder": encoder,
        "layer": layer,
        "shape": shape,
        "fp16_bytes": fp16_bytes,
        "packed_bytes": p.nbytes(),
        "ratio": fp16_bytes / max(p.nbytes(), 1),
        "err_max": err.max().item(),
        "err_mean": err.mean().item(),
        "pack_ms": t_pack * 1000,
        "unpack_ms": t_unpack * 1000,
        "triton_ms": t_tri,
        "peak_mb": peak,
    }


def print_table(rows):
    print()
    print("| Encoder | Layer          | Shape                   | fp16 MB | Packed MB | Ratio |"
          " Pack ms | Unpack (PyT) ms | Unpack (Triton) ms | Speedup |")
    print("|---------|----------------|-------------------------|--------:|----------:|------:|"
          "--------:|----------------:|-------------------:|--------:|")
    total_fp16 = 0
    total_packed = 0
    for r in rows:
        total_fp16 += r["fp16_bytes"]
        total_packed += r["packed_bytes"]
        speedup = (r["unpack_ms"] / r["triton_ms"]) if r["triton_ms"] > 0 else 0.0
        tri_str = f"{r['triton_ms']:7.2f}" if r["triton_ms"] > 0 else "   n/a"
        sp_str = f"{speedup:6.1f}x" if speedup > 0 else "   n/a"
        print(f"| {r['encoder']}  | {r['layer']:<14s} | "
              f"{str(r['shape']):<23s} | "
              f"{r['fp16_bytes']/1024**2:7.2f} | "
              f"{r['packed_bytes']/1024**2:9.2f} | "
              f"{r['ratio']:5.2f}x | "
              f"{r['pack_ms']:7.1f} | "
              f"{r['unpack_ms']:15.2f} | "
              f"{tri_str:>18s} | "
              f"{sp_str:>7s} |")
    print(f"|         | TOTAL          |                         | "
          f"{total_fp16/1024**2:7.2f} | "
          f"{total_packed/1024**2:9.2f} | "
          f"{total_fp16/max(total_packed,1):5.2f}x |         |                 |                    |         |")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=32, help="temporal window")
    ap.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--layers", nargs="+", default=None,
                    help="filter by encoder substring (e.g. ViT-S)")
    args = ap.parse_args()

    if args.device == "cuda":
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"CUDA device: {name}  SM {cap}  {total:.1f} GB")

    rows = []
    for enc, layer, tokens, head_dim in VDA_LAYERS:
        if args.layers and not any(t in enc for t in args.layers):
            continue
        try:
            r = benchmark_one(enc, layer, tokens, head_dim,
                              T=args.T, bits=args.bits, device=args.device)
            rows.append(r)
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] {enc} {layer}  {(1, args.T, tokens, head_dim)}")
            torch.cuda.empty_cache()

    print_table(rows)


if __name__ == "__main__":
    main()
