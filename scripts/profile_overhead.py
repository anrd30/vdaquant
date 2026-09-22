"""
Per-step profiler for RotatedTemporalAttention.

Answers before we refactor:
  1. Where does time go in the fused forward?  (rotation vs pack vs
     Python (B,h) loop vs kernel vs rebuild).
  2. How much peak-memory overhead comes from K + K_rot + K_q +
     packed_K coexisting, vs the theoretical minimum (packed_K alone)?

Patches RotatedTemporalAttention.forward with CUDA event timings and
torch.cuda.memory_allocated() sampling at every substep, on the
biggest VDA temporal-attention layer (mm3 / 5476 tokens / head_dim
padded to 16-multiple).

Usage:
  python scripts/profile_overhead.py --layer mm1
  python scripts/profile_overhead.py --layer mm0 --repeats 20

Output: printed per-step table with mean latency, per-step memory
delta, and peak / theoretical-min memory ratio.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from research.models.rotated_attention import RotatedTemporalAttention


# Only the shapes where padded_dim % 16 == 0 (fused kernel prereq).
LAYER_SHAPES = {
    # label: (B, h, M, head_dim)  -- padded_dim = next power of 2
    "mm0": (8, 8, 1369, 24),   # padded 32, 2 groups of 16
    "mm1": (8, 8,  361, 48),   # padded 64, 4 groups of 16
}


def _sync_ms(start_evt, end_evt):
    end_evt.synchronize()
    return start_evt.elapsed_time(end_evt)


def _mb(bytes_val):
    return bytes_val / (1024 ** 2)


def _make_layer(dim: int, num_heads: int, use_fused: bool) -> RotatedTemporalAttention:
    layer = RotatedTemporalAttention(
        dim=dim, num_heads=num_heads, qkv_bias=True,
        bits=3, quantizer='lattice_bw16',
        use_qjl=False,     # apples-to-apples; QJL disabled on both paths
        scale_bits=16, use_rotation=True, rht_seed=0,
    ).cuda().eval()
    layer.use_fused_kernel = use_fused
    return layer


def profile_layer(label: str, B: int, h: int, M: int, d: int, repeats: int = 10):
    dim = h * d
    torch.manual_seed(0)
    x = torch.randn(B, M, dim, device="cuda", dtype=torch.float32)

    print(f"\n{'=' * 78}")
    print(f"Layer {label}: B={B}  h={h}  M={M}  head_dim={d}  dim={dim}")
    print(f"{'=' * 78}")

    for path_name, use_fused in [("SIMULATOR", False), ("FUSED", True)]:
        layer = _make_layer(dim=dim, num_heads=h, use_fused=use_fused)
        # Warm-up (2 runs) so autotune settles.
        with torch.no_grad():
            for _ in range(2):
                _ = layer(hidden_states=x, encoder_hidden_states=x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Timed pass (whole forward).
        forward_ms = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                _ = layer(hidden_states=x, encoder_hidden_states=x)
            end.record()
            forward_ms.append(_sync_ms(start, end))

        peak_alloc_mb = _mb(torch.cuda.max_memory_allocated())
        alloc_at_end_mb = _mb(torch.cuda.memory_allocated())
        forward_mean = sum(forward_ms) / len(forward_ms)
        forward_std = (sum((t - forward_mean) ** 2 for t in forward_ms) / len(forward_ms)) ** 0.5

        # Theoretical-min memory: packed K + packed V only.
        # At 3-bit BW16: 5 bytes / codeword, one codeword per 16 scalars.
        padded_d = 2 ** ((d - 1).bit_length())
        n_codewords = B * h * M * padded_d // 16
        packed_kv_bytes = 2 * n_codewords * 5    # K + V
        theoretical_min_mb = _mb(packed_kv_bytes)

        print(f"\n[{path_name}]")
        print(f"  forward wall-clock : {forward_mean:.3f} +- {forward_std:.3f} ms")
        print(f"  peak alloc         : {peak_alloc_mb:.1f} MB")
        print(f"  alloc at end       : {alloc_at_end_mb:.1f} MB")
        print(f"  theoretical min KV : {theoretical_min_mb:.1f} MB   (packed K + packed V only)")
        print(f"  overhead ratio     : {peak_alloc_mb / max(theoretical_min_mb, 0.001):.1f}x "
                f"over theoretical KV min")

        del layer
        torch.cuda.empty_cache()


def profile_substeps(label: str, B: int, h: int, M: int, d: int):
    """
    Instrument each substep of the FUSED forward path by monkey-patching:
      rotation, pack, per-head kernel loop, rotation.inverse

    Prints per-substep time and per-substep memory delta.
    """
    from kernels.reference.packed_bw16 import pack_bw16, PackedBW16Bits
    from kernels.reference.bw16_codebook import bw16_cosets
    from kernels.triton_kernels.fused_attention import fused_attention_bw16

    dim = h * d
    torch.manual_seed(0)
    x = torch.randn(B, M, dim, device="cuda", dtype=torch.float32)
    layer = _make_layer(dim=dim, num_heads=h, use_fused=True)

    print(f"\n{'-' * 78}")
    print(f"[{label}] FUSED-path substep profile  (B={B} h={h} M={M} d={d})")
    print(f"{'-' * 78}")
    print(f"{'step':<26} {'time_ms':>10} {'mem_delta_MB':>14} {'mem_after_MB':>14}")

    def _step(name, fn):
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end)
        mem_after = torch.cuda.memory_allocated()
        delta = _mb(mem_after - mem_before)
        print(f"{name:<26} {ms:>10.3f} {delta:>+14.1f} {_mb(mem_after):>14.1f}")
        return result

    # Manually replay the fused branch so we can time each stage.
    query_input = context_input = x
    B_, N, C = query_input.shape
    M_ = context_input.shape[1]
    with torch.no_grad():
        # Warm up
        for _ in range(2):
            _ = layer(hidden_states=x, encoder_hidden_states=x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        Q = _step("q_proj + reshape", lambda:
            layer.q_proj(query_input).reshape(B_, N, h, d).transpose(1, 2))
        K = _step("k_proj + reshape", lambda:
            layer.k_proj(context_input).reshape(B_, M_, h, d).transpose(1, 2))
        V = _step("v_proj + reshape", lambda:
            layer.v_proj(context_input).reshape(B_, M_, h, d).transpose(1, 2))
        K_rot = _step("rotation(K)", lambda: layer.rotation(K))
        V_rot = _step("rotation(V)", lambda: layer.rotation(V))
        Q_rot = _step("rotation(Q)", lambda: layer.rotation(Q))
        packed_K = _step("pack_bw16(K_rot)", lambda: pack_bw16(K_rot.contiguous(), bits=3))
        packed_V = _step("pack_bw16(V_rot)", lambda: pack_bw16(V_rot.contiguous(), bits=3))
        codebook = _step("bw16_cosets", lambda: bw16_cosets(dtype=torch.float32, device="cuda"))

        def run_kernel_loop():
            padded_d = K_rot.shape[-1]
            out = torch.empty(B_, h, N, padded_d, dtype=torch.float32, device="cuda")
            for b in range(B_):
                for hd in range(h):
                    pk = PackedBW16Bits(
                        codeword_bytes=packed_K.codeword_bytes[b, hd].contiguous(),
                        group_scale=packed_K.group_scale[b, hd].contiguous(),
                        original_shape=K_rot[b, hd].shape,
                        bits=packed_K.bits,
                        n_bytes_per_codeword=packed_K.n_bytes_per_codeword,
                    )
                    pv = PackedBW16Bits(
                        codeword_bytes=packed_V.codeword_bytes[b, hd].contiguous(),
                        group_scale=packed_V.group_scale[b, hd].contiguous(),
                        original_shape=V_rot[b, hd].shape,
                        bits=packed_V.bits,
                        n_bytes_per_codeword=packed_V.n_bytes_per_codeword,
                    )
                    Q_bh = Q_rot[b, hd].contiguous().to(torch.float16)
                    out[b, hd] = fused_attention_bw16(Q_bh, pk, pv, codebook,
                                                     scale=layer.scale, bits=3).to(out.dtype)
            return out

        out_rot = _step(f"kernel loop ({B_}x{h}={B_*h} launches)", run_kernel_loop)
        out = _step("rotation.inverse", lambda: layer.rotation.inverse(out_rot))
        _step("out_proj", lambda: layer.out_proj(out.transpose(1, 2).reshape(B_, N, C)))

        print(f"\n  peak memory during fused forward: "
              f"{_mb(torch.cuda.max_memory_allocated()):.1f} MB")
        # Theoretical min = packed_K + packed_V only.
        print(f"  theoretical min (packed KV only) : "
              f"{_mb(packed_K.codeword_bytes.numel() + packed_V.codeword_bytes.numel()):.1f} MB")

    del layer
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="mm1", choices=list(LAYER_SHAPES.keys()))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--substeps-only", action="store_true",
                    help="Skip whole-forward timing; only run the substep breakdown.")
    args = ap.parse_args()

    B, h, M, d = LAYER_SHAPES[args.layer]

    if not args.substeps_only:
        profile_layer(args.layer, B, h, M, d, repeats=args.repeats)
    profile_substeps(args.layer, B, h, M, d)


if __name__ == "__main__":
    main()
