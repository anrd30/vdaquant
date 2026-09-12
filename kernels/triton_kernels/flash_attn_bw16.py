"""
FlashAttention-style fused BW16 attention kernel.

Everything in one kernel:
    - Load Q block from fp16
    - For each K/V block:
        * Decode K block from packed BW16
        * Compute S = Q @ K^T * scale
        * Online softmax update (max + normalization tracked across blocks)
        * Decode V block from packed BW16
        * Accumulate P @ V into the output block
    - Divide accumulator by softmax normaliser, write fp16 output

Compared to the two-kernel fused_attention_bw16 wrapper:
    - The fp32 (M, N) score matrix never exists
    - Softmax is fused into the loop, no separate PyTorch call
    - K and V decoding stays in registers, never spills to global mem

Correctness gate is against fused_attention_bw16 (which is itself
gated against standard PyTorch attention on decoded K, V).  Expect fp
round-off diff, not bit-parity.

Only bits=3 supported here.  bits in {2, 4} follow the same pattern
by swapping the _LAYOUT constants.

Assumes D is a multiple of 16 (BW16 lattice_dim) and fits in one
program's register file.  For D > ~128 we'd need an inner D-tiling
loop; that's follow-up.
"""
from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


LATTICE_DIM = 16
_BITS = 3
_LAYOUT = {
    2: dict(bytes_per_codeword=3, offset_bits=1, offset_mask=0x1, offset_bias=1),
    3: dict(bytes_per_codeword=5, offset_bits=2, offset_mask=0x3, offset_bias=2),
    4: dict(bytes_per_codeword=7, offset_bits=3, offset_mask=0x7, offset_bias=4),
}


if TRITON_AVAILABLE:

    @triton.jit
    def _flash_attn_bw16_kernel(
        Q_ptr,                 # fp16 (M, D)
        K_pack_ptr,            # uint8 (N, D_GROUPS, 5)
        K_scale_ptr,           # fp32  (N, D_GROUPS)
        V_pack_ptr,            # uint8 (N, D_GROUPS, 5)
        V_scale_ptr,           # fp32  (N, D_GROUPS)
        codebook_ptr,          # fp32  (32, 16)
        out_ptr,               # fp32  (M, D)
        M, N, D, D_GROUPS,
        softmax_scale,
        stride_qm, stride_qd,
        stride_kn, stride_kg, stride_kb,
        stride_ksn, stride_ksg,
        stride_vn, stride_vg, stride_vb,
        stride_vsn, stride_vsg,
        stride_om, stride_od,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BYTES_PER_CODEWORD: tl.constexpr,
        OFFSET_BITS: tl.constexpr,
        OFFSET_MASK: tl.constexpr,
        OFFSET_BIAS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_g = tl.program_id(1)                          # which D group
        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
        m_mask = m_offs < M
        col_range = tl.arange(0, 16)

        # Softmax state persistent across n-blocks.
        m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float('inf')
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, 16), dtype=tl.float32)

        # Load Q columns for THIS D group.  (BLOCK_M, 16).
        q_d_offs = pid_g * 16 + tl.arange(0, 16)
        q_tile = tl.load(
            Q_ptr + m_offs[:, None] * stride_qm + q_d_offs[None, :] * stride_qd,
            mask=m_mask[:, None] & (q_d_offs[None, :] < D),
            other=0.0,
        ).to(tl.float32)                                  # (BLOCK_M, 16)

        # Iterate over K/V blocks.
        for n_start in range(0, N, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < N

            # ---------- score contribution from THIS D group ----------
            # Decode K for this group and n_offs range: (BLOCK_N, 16).
            k_pack_base = K_pack_ptr + n_offs * stride_kn + pid_g * stride_kg
            packed_k = tl.zeros((BLOCK_N,), dtype=tl.int64)
            for byte_i in tl.static_range(0, BYTES_PER_CODEWORD):
                b = tl.load(k_pack_base + byte_i * stride_kb, mask=n_mask, other=0
                            ).to(tl.int64)
                packed_k |= (b << (byte_i * 8))
            coset_k = (packed_k & 0x1F).to(tl.int32)
            scale_k = tl.load(K_scale_ptr + n_offs * stride_ksn + pid_g * stride_ksg,
                              mask=n_mask, other=0.0)

            k_tile = tl.zeros((BLOCK_N, 16), dtype=tl.float32)
            for i in tl.static_range(0, 16):
                u = ((packed_k >> (5 + i * OFFSET_BITS)) & OFFSET_MASK).to(tl.int32)
                offset_signed = u - OFFSET_BIAS
                c = tl.load(codebook_ptr + coset_k * 16 + i, mask=n_mask, other=0.0)
                x_scaled = 2.0 * offset_signed.to(tl.float32) + c
                v = x_scaled * scale_k
                is_i = (col_range == i).to(tl.float32)
                k_tile = k_tile + v[:, None] * is_i[None, :]

            # Score contribution from THIS group only.
            # NOTE: For proper multi-group attention, we'd need to
            # accumulate scores over all groups BEFORE softmax.  This
            # single-group kernel is correct only for D == 16.
            s_group = tl.dot(q_tile, tl.trans(k_tile), allow_tf32=False)
            s_group = s_group * softmax_scale
            # Mask out invalid n positions with -inf.
            s_group = tl.where(n_mask[None, :], s_group, -float('inf'))

            # ---------- online softmax update ----------
            m_new = tl.maximum(m_i, tl.max(s_group, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s_group - m_new[:, None])
            # Renormalise the accumulator.
            acc = acc * alpha[:, None]
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

            # ---------- V decode + P @ V ----------
            v_pack_base = V_pack_ptr + n_offs * stride_vn + pid_g * stride_vg
            packed_v = tl.zeros((BLOCK_N,), dtype=tl.int64)
            for byte_i in tl.static_range(0, BYTES_PER_CODEWORD):
                b = tl.load(v_pack_base + byte_i * stride_vb, mask=n_mask, other=0
                            ).to(tl.int64)
                packed_v |= (b << (byte_i * 8))
            coset_v = (packed_v & 0x1F).to(tl.int32)
            scale_v = tl.load(V_scale_ptr + n_offs * stride_vsn + pid_g * stride_vsg,
                              mask=n_mask, other=0.0)

            v_tile = tl.zeros((BLOCK_N, 16), dtype=tl.float32)
            for i in tl.static_range(0, 16):
                u = ((packed_v >> (5 + i * OFFSET_BITS)) & OFFSET_MASK).to(tl.int32)
                offset_signed = u - OFFSET_BIAS
                c = tl.load(codebook_ptr + coset_v * 16 + i, mask=n_mask, other=0.0)
                x_scaled = 2.0 * offset_signed.to(tl.float32) + c
                v = x_scaled * scale_v
                is_i = (col_range == i).to(tl.float32)
                v_tile = v_tile + v[:, None] * is_i[None, :]

            acc = acc + tl.dot(p, v_tile, allow_tf32=False)

        # Normalise and write.
        acc = acc / l_i[:, None]
        out_d_offs = pid_g * 16 + tl.arange(0, 16)
        tl.store(
            out_ptr + m_offs[:, None] * stride_om + out_d_offs[None, :] * stride_od,
            acc,
            mask=m_mask[:, None] & (out_d_offs[None, :] < D),
        )


    def flash_attn_bw16(
        Q: torch.Tensor,                    # fp16 (M, D)
        packed_K,                           # PackedBW16Bits
        packed_V,                           # PackedBW16Bits
        codebook: torch.Tensor,             # fp32 (32, 16)
        softmax_scale: float = None,
        bits: int = 3,
        BLOCK_M: int = 32,
        BLOCK_N: int = 64,
    ) -> torch.Tensor:
        """
        Single-kernel FlashAttention-style attention with packed BW16 K/V.

        WARNING: This implementation is per-D-group.  It computes softmax
        over the N axis using only ONE 16-scalar slice of D at a time and
        is therefore only mathematically correct for D == 16.  For larger
        D the softmax needs to see the full-D score matrix.

        This is checked in as a scaffold: correctness on D=16 verifies
        the online-softmax mechanics; a multi-D-group extension folds
        the per-group score contributions before softmax.
        """
        assert bits in _LAYOUT
        L = _LAYOUT[bits]
        M, D = Q.shape
        N = packed_K.codeword_bytes.shape[0]
        D_GROUPS = packed_K.codeword_bytes.shape[1]

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(D)

        out = torch.empty(M, D, dtype=torch.float32, device=Q.device)
        grid = (triton.cdiv(M, BLOCK_M), D_GROUPS)

        _flash_attn_bw16_kernel[grid](
            Q, packed_K.codeword_bytes, packed_K.group_scale,
            packed_V.codeword_bytes, packed_V.group_scale,
            codebook.contiguous(), out,
            M, N, D, D_GROUPS,
            softmax_scale,
            Q.stride(0), Q.stride(1),
            packed_K.codeword_bytes.stride(0),
            packed_K.codeword_bytes.stride(1),
            packed_K.codeword_bytes.stride(2),
            packed_K.group_scale.stride(0),
            packed_K.group_scale.stride(1),
            packed_V.codeword_bytes.stride(0),
            packed_V.codeword_bytes.stride(1),
            packed_V.codeword_bytes.stride(2),
            packed_V.group_scale.stride(0),
            packed_V.group_scale.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            BYTES_PER_CODEWORD=L['bytes_per_codeword'],
            OFFSET_BITS=L['offset_bits'],
            OFFSET_MASK=L['offset_mask'],
            OFFSET_BIAS=L['offset_bias'],
        )
        return out


if __name__ == "__main__":
    if not TRITON_AVAILABLE:
        raise SystemExit("triton missing")

    from kernels.reference.packed_bw16 import pack_bw16
    from kernels.reference.bw16_codebook import bw16_cosets

    # D = 16 (single group) -- the mathematically-correct case for this
    # scaffold kernel.
    torch.manual_seed(0)
    device = "cuda"
    M, N, D = 64, 128, 16
    Q = torch.randn(M, D, device=device, dtype=torch.float16)
    K = torch.randn(N, D, device=device)
    V = torch.randn(N, D, device=device)
    pK = pack_bw16(K, bits=3, group_size=16, scale_bits=8)
    pV = pack_bw16(V, bits=3, group_size=16, scale_bits=8)
    codebook = bw16_cosets(dtype=torch.float32, device=device)

    # Reference via decoded K, V + PyTorch attention.
    from kernels.reference.packed_bw16 import unpack_bw16
    K_d = unpack_bw16(pK, dtype=torch.float32)
    V_d = unpack_bw16(pV, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)
    scores = Q.float() @ K_d.t() * scale
    probs = torch.softmax(scores, dim=-1)
    ref = probs @ V_d

    flash = flash_attn_bw16(Q, pK, pV, codebook, softmax_scale=scale, bits=3)

    diff = (ref - flash).abs()
    print(f"D=16 (scaffold-correct case):")
    print(f"  max diff = {diff.max().item():.3e}")
    print(f"  mean diff = {diff.mean().item():.3e}")
    print(f"  ref range = [{ref.min().item():.3f}, {ref.max().item():.3f}]")
