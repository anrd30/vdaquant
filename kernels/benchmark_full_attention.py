"""
Full-attention benchmark for Paper 2 Table 6.

Compares three paths on every VDA temporal-attention layer:

  (A) FP16 baseline:
        Q, K, V are fp16 tensors on the GPU.  Standard
        F.scaled_dot_product_attention.  Peak-memory and latency
        baseline.

  (B) Simulator quantiser (current paper Table 6 methodology):
        K, V pass through LatticeBW16Quantizer at 3-bit; the fp16
        reconstructions live in the KV cache and standard attention
        runs on them.  Zero memory savings on the cache; the
        quantisation acts only as accuracy noise.

  (C) Fused packed BW16 attention (new):
        K, V are packed via pack_bw16 and remain in packed int form
        on the GPU.  fused_attention_bw16 decodes on the fly inside
        the two Triton kernels; K and V never materialise as fp16.

We report:
  - KV cache footprint per layer (fp16 vs packed int bytes)
  - Attention forward latency per layer (all three paths)
  - Speedup Fused / FP16 baseline
  - Speedup Fused / Simulator
  - Output tensor max diff (Fused vs Simulator, after Hadamard rotation)

Total row aggregates the KV footprint and pretends every layer is run
independently -- there is no cross-layer amortisation.  The final
number is the "measured memory + latency" for Paper 2 Table 6.
"""
from __future__ import annotations

import argparse
import gc
import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from kernels.reference.packed_bw16 import pack_bw16, unpack_bw16
from kernels.reference.bw16_codebook import bw16_cosets
from kernels.triton_kernels.fused_attention import fused_attention_bw16


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


@dataclass
class BenchResult:
    encoder: str
    layer:   str
    N: int
    D: int
    fp16_kv_bytes: int
    packed_kv_bytes: int
    t_fp16_ms: float
    t_sim_ms: float
    t_fused_ms: float
    max_diff_fused_vs_sim: float


def _bench_shape(encoder, layer, N, D, bits=3, device="cuda", warmup=3, iters=5):
    Q = torch.randn(N, D, device=device, dtype=torch.float16)
    K = torch.randn(N, D, device=device)
    V = torch.randn(N, D, device=device)
    scale = 1.0 / math.sqrt(D)

    # --- fp16 baseline ---
    K16 = K.half(); V16 = V.half()
    fp16_kv_bytes = K16.numel() * 2 + V16.numel() * 2
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(Q, K16, V16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out_fp16 = F.scaled_dot_product_attention(Q, K16, V16)
    torch.cuda.synchronize()
    t_fp16 = (time.perf_counter() - t0) / iters * 1000

    # --- simulator path ---
    from research.quantizers.lattice_vq import LatticeBW16Quantizer
    q_bw16 = LatticeBW16Quantizer(bits=bits, group_size=16, scale_bits=8)
    for _ in range(warmup):
        K_q, _ = q_bw16(K); V_q, _ = q_bw16(V)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        K_q, _ = q_bw16(K); V_q, _ = q_bw16(V)
        s = Q.float() @ K_q.t() * scale
        p = F.softmax(s, dim=-1)
        out_sim = p @ V_q
    torch.cuda.synchronize()
    t_sim = (time.perf_counter() - t0) / iters * 1000

    # --- fused path (cache pre-packed) ---
    codebook = bw16_cosets(dtype=torch.float32, device=device)
    pK = pack_bw16(K, bits=bits, group_size=16, scale_bits=8)
    pV = pack_bw16(V, bits=bits, group_size=16, scale_bits=8)
    packed_kv_bytes = pK.nbytes() + pV.nbytes()
    for _ in range(warmup):
        _ = fused_attention_bw16(Q, pK, pV, codebook, bits=bits)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out_fused = fused_attention_bw16(Q, pK, pV, codebook, bits=bits)
    torch.cuda.synchronize()
    t_fused = (time.perf_counter() - t0) / iters * 1000

    max_diff = (out_sim - out_fused).abs().max().item()

    return BenchResult(
        encoder=encoder, layer=layer, N=N, D=D,
        fp16_kv_bytes=fp16_kv_bytes, packed_kv_bytes=packed_kv_bytes,
        t_fp16_ms=t_fp16, t_sim_ms=t_sim, t_fused_ms=t_fused,
        max_diff_fused_vs_sim=max_diff,
    )


def print_table(rows):
    print()
    print("| Encoder | Layer          | Shape          | FP16 KV MB | Packed KV MB | Ratio |"
          " FP16 attn ms | Sim attn ms | Fused attn ms | vs FP16 | vs Sim | max diff |")
    print("|---------|----------------|----------------|-----------:|-------------:|------:|"
          "-------------:|------------:|--------------:|--------:|-------:|---------:|")
    tot_fp16_kv = tot_pk_kv = 0
    tot_fp16 = tot_sim = tot_fused = 0.0
    for r in rows:
        tot_fp16_kv += r.fp16_kv_bytes
        tot_pk_kv += r.packed_kv_bytes
        tot_fp16 += r.t_fp16_ms
        tot_sim += r.t_sim_ms
        tot_fused += r.t_fused_ms
        ratio = r.fp16_kv_bytes / max(r.packed_kv_bytes, 1)
        sp_fp16 = r.t_fp16_ms / max(r.t_fused_ms, 1e-6)
        sp_sim = r.t_sim_ms / max(r.t_fused_ms, 1e-6)
        print(f"| {r.encoder}  | {r.layer:<14s} | {r.N}x{r.D:<8d} | "
              f"{r.fp16_kv_bytes/1024**2:10.2f} | "
              f"{r.packed_kv_bytes/1024**2:12.2f} | "
              f"{ratio:5.2f}x | "
              f"{r.t_fp16_ms:12.2f} | "
              f"{r.t_sim_ms:11.2f} | "
              f"{r.t_fused_ms:13.2f} | "
              f"{sp_fp16:6.2f}x | "
              f"{sp_sim:5.2f}x | "
              f"{r.max_diff_fused_vs_sim:.2e} |")
    print(f"|         | TOTAL          |                | "
          f"{tot_fp16_kv/1024**2:10.2f} | "
          f"{tot_pk_kv/1024**2:12.2f} | "
          f"{tot_fp16_kv/max(tot_pk_kv,1):5.2f}x | "
          f"{tot_fp16:12.2f} | "
          f"{tot_sim:11.2f} | "
          f"{tot_fused:13.2f} | "
          f"{tot_fp16/max(tot_fused,1e-6):6.2f}x | "
          f"{tot_sim/max(tot_fused,1e-6):5.2f}x |          |")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    ap.add_argument("--filter", default=None, help="substring on encoder name")
    args = ap.parse_args()

    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}  "
              f"SM {torch.cuda.get_device_capability(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    rows = []
    for enc, layer, N, D in VDA_LAYERS:
        if args.filter and args.filter not in enc:
            continue
        try:
            print(f"  benchmarking {enc} {layer}...", flush=True)
            r = _bench_shape(enc, layer, N, D, bits=args.bits)
            rows.append(r)
            gc.collect(); torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] {enc} {layer}")
            torch.cuda.empty_cache()

    print_table(rows)


if __name__ == "__main__":
    main()
