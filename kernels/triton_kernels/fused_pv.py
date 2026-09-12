"""
Fused P @ V kernel with on-the-fly BW16 decode.

Given attention weights P (M, N) and packed V (N, D), compute
    O = P @ V         (M, D)
with V decoded from packed BW16 storage inside the kernel.  V is
never materialised as fp16.

Combined with fused_qk.py this gives full attention with the KV cache
in packed integer form on the GPU end to end -- the fp16 K and V
intermediates from decoding never exist.

Kernel structure (one program per output tile):
  for m_block, d_block in grid:
    load P[m_block:, :]  fp32                        (BLOCK_M, N)
    acc = zeros (BLOCK_M, BLOCK_D)
    for group in [0, D_GROUPS):
        for n_block in [0, N, BLOCK_N):
            for i in [0, 16):
                decode V[n_block:, group*16+i]
                if group*16 + i in [d_block, d_block + BLOCK_D):
                    acc[:, i] += P[:, n_block:] @ v_col
    store O
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


LATTICE_DIM = 16
_LAYOUT = {
    2: dict(bytes_per_codeword=3, offset_bits=1, offset_mask=0x1, offset_bias=1),
    3: dict(bytes_per_codeword=5, offset_bits=2, offset_mask=0x3, offset_bias=2),
    4: dict(bytes_per_codeword=7, offset_bits=3, offset_mask=0x7, offset_bias=4),
}


if TRITON_AVAILABLE:

    _AUTOTUNE_CONFIGS = [
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8),
    ]

    @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=['M', 'N', 'D_GROUPS'])
    @triton.jit
    def _fused_pv_bw16_kernel(
        P_ptr,                 # fp32 (M, N)
        V_pack_ptr,            # uint8 (N, D_GROUPS, BYTES_PER_CODEWORD)
        V_scale_ptr,           # fp32  (N, D_GROUPS)
        codebook_ptr,          # fp32  (32, 16)
        out_ptr,               # fp32  (M, D)
        M, N, D_GROUPS,
        stride_pm, stride_pn,
        stride_vn, stride_vg, stride_vb,
        stride_sn, stride_sg,
        stride_om, stride_od,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BYTES_PER_CODEWORD: tl.constexpr,
        OFFSET_BITS: tl.constexpr,
        OFFSET_MASK: tl.constexpr,
        OFFSET_BIAS: tl.constexpr,
    ):
        # One program computes one 16-scalar D-group for BLOCK_M output rows.
        pid_m = tl.program_id(0)
        pid_g = tl.program_id(1)                          # which D-group
        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
        m_mask = m_offs < M

        # Accumulate all 16 output columns for this D-group in registers.
        acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        # Use a single (BLOCK_M, 16) accumulator via broadcast + reduce.
        # Simpler: loop 16 times, keep one column at a time?  We want a
        # (BLOCK_M, 16) tile.  Triton needs a 2D tile of static shape.
        acc = tl.zeros((BLOCK_M, 16), dtype=tl.float32)

        for n_start in range(0, N, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < N

            # Load P block (BLOCK_M, BLOCK_N) fp32.
            p_tile = tl.load(
                P_ptr + m_offs[:, None] * stride_pm + n_offs[None, :] * stride_pn,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )                                              # (BLOCK_M, BLOCK_N)

            # Decode V block for this group: (BLOCK_N, 16).
            v_pack_base = V_pack_ptr + n_offs * stride_vn + pid_g * stride_vg
            packed = tl.zeros((BLOCK_N,), dtype=tl.int64)
            for byte_i in tl.static_range(0, BYTES_PER_CODEWORD):
                b = tl.load(
                    v_pack_base + byte_i * stride_vb,
                    mask=n_mask, other=0,
                ).to(tl.int64)
                packed |= (b << (byte_i * 8))
            coset_idx = (packed & 0x1F).to(tl.int32)
            scale = tl.load(
                V_scale_ptr + n_offs * stride_sn + pid_g * stride_sg,
                mask=n_mask, other=0.0,
            )                                              # (BLOCK_N,)

            # For each of 16 output coords in the group, decode + accumulate.
            col_range = tl.arange(0, 16)
            for i in tl.static_range(0, 16):
                u = ((packed >> (5 + i * OFFSET_BITS)) & OFFSET_MASK).to(tl.int32)
                offset_signed = u - OFFSET_BIAS
                c = tl.load(codebook_ptr + coset_idx * 16 + i,
                            mask=n_mask, other=0.0)
                x_scaled = 2.0 * offset_signed.to(tl.float32) + c
                v_val = x_scaled * scale                    # (BLOCK_N,)
                # Contribution: sum_n P[m, n] * v_val[n]
                contribution = tl.sum(p_tile * v_val[None, :], axis=1)  # (BLOCK_M,)
                # Scatter into acc[:, i] using mask trick.
                is_i = (col_range == i).to(tl.float32)      # (16,)
                acc += contribution[:, None] * is_i[None, :]

        # Store the 16 output columns for this D-group.
        d_offs = pid_g * 16 + tl.arange(0, 16)
        tl.store(
            out_ptr + m_offs[:, None] * stride_om + d_offs[None, :] * stride_od,
            acc,
            mask=m_mask[:, None],
        )

    def fused_pv_bw16(
        P: torch.Tensor,                    # fp32 (M, N)
        V_pack: torch.Tensor,               # uint8 (N, D_GROUPS, bytes)
        V_scale: torch.Tensor,              # fp32  (N, D_GROUPS)
        codebook: torch.Tensor,             # fp32  (32, 16)
        bits: int = 3,
        BLOCK_M: int = 32,
        BLOCK_N: int = 32,
    ) -> torch.Tensor:
        assert bits in _LAYOUT
        L = _LAYOUT[bits]
        assert P.dtype == torch.float32
        assert V_pack.dtype == torch.uint8
        assert V_scale.dtype == torch.float32
        assert codebook.shape == (32, 16) and codebook.dtype == torch.float32

        M, N = P.shape
        N_v, D_GROUPS, bpcw = V_pack.shape
        assert N_v == N
        assert bpcw == L['bytes_per_codeword']
        D = D_GROUPS * 16

        P = P.contiguous()
        V_pack = V_pack.contiguous()
        V_scale = V_scale.contiguous()

        out = torch.empty(M, D, dtype=torch.float32, device=P.device)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), D_GROUPS)
        _fused_pv_bw16_kernel[grid](
            P, V_pack, V_scale, codebook.contiguous(), out,
            M, N, D_GROUPS,
            P.stride(0), P.stride(1),
            V_pack.stride(0), V_pack.stride(1), V_pack.stride(2),
            V_scale.stride(0), V_scale.stride(1),
            out.stride(0), out.stride(1),
            BYTES_PER_CODEWORD=L['bytes_per_codeword'],
            OFFSET_BITS=L['offset_bits'],
            OFFSET_MASK=L['offset_mask'],
            OFFSET_BIAS=L['offset_bias'],
        )
        return out

else:
    def fused_pv_bw16(*a, **kw):
        raise RuntimeError("triton is not installed")


if __name__ == "__main__":
    if not TRITON_AVAILABLE:
        raise SystemExit("triton missing")
    from kernels.reference.packed_bw16 import pack_bw16, unpack_bw16
    from kernels.reference.bw16_codebook import bw16_cosets

    torch.manual_seed(0)
    device = "cuda"
    M, N, D = 64, 128, 64
    P = torch.randn(M, N, device=device)
    V = torch.randn(N, D, device=device)
    packed = pack_bw16(V, bits=3, group_size=16, scale_bits=8)
    codebook = bw16_cosets(dtype=torch.float32, device=device)

    V_ref = unpack_bw16(packed, dtype=torch.float32)
    out_ref = P @ V_ref
    out_fused = fused_pv_bw16(P, packed.codeword_bytes, packed.group_scale, codebook)

    diff = (out_ref - out_fused).abs()
    print(f"fused P @ V   shape={tuple(out_ref.shape)}")
    print(f"  max diff  = {diff.max().item():.4e}")
    print(f"  mean diff = {diff.mean().item():.4e}")
    print(f"  ref range = [{out_ref.min().item():.3f}, {out_ref.max().item():.3f}]")
    assert diff.max().item() < 1e-1
    print("  OK")
