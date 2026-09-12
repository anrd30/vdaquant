"""
Fused Q * K^T kernel with on-the-fly BW16 decode.

Standard attention:
    scores  = Q @ K^T                       (K materialised as fp16)
    probs   = softmax(scores / sqrt(D))
    output  = probs @ V

Our fused version replaces the first step with a kernel that reads the
PACKED bytes of K, decodes each 16-scalar group inline, and accumulates
Q @ K^T directly.  K is never materialised as fp16; the fp16 K tensor
that would live in the KV cache disappears entirely.

Kernel structure (one program per output tile of the score matrix):

  for m_block in [0, M, BLOCK_M):
      for n_block in [0, N, BLOCK_N):
          load Q[m_block:m_block+BLOCK_M, :]                   fp16
          for group in [0, D_GROUPS):
              load packed_K bytes for K[n_block:n_block+BLOCK_N,
                                       group*16:(group+1)*16]  uint8
              decode into register tile: (BLOCK_N, 16) fp32
              accumulate scores += Q_tile[:, group*16:group*16+16] @ decoded^T
          store scores fp32

Correctness gate: the decoded K used inside the kernel is bit-exact
with unpack_bw16 on the same packed input (see the same
_LAYOUT/decode logic).  The fused kernel therefore matches Q @ K^T
computed on the unpacked K, up to fp accumulation order round-off.

Bit width: 3 for now (5-byte codeword); trivial to extend to 2 and 4.
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

    # BLOCK sweep on RTX 4050 found:
    #   small M/N shapes  -> BM=32, BN=128
    #   large M/N shapes  -> BM=64, BN=256
    #   BM=64, BN=64 is a shared-memory trap and 30x slower than either.
    # Autotune picks per-GPU without our hardcoding.  Configs kept small
    # (5) so autotune compilation stays cheap; add more if a new GPU
    # regresses.
    _AUTOTUNE_CONFIGS = [
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32},  num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8),
    ]

    @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=['M', 'N', 'D'])
    @triton.jit
    def _fused_qk_bw16_kernel(
        Q_ptr,                 # fp16  (M, D)
        K_pack_ptr,            # uint8 (N, D_GROUPS, BYTES_PER_CODEWORD)
        K_scale_ptr,           # fp32  (N, D_GROUPS)
        codebook_ptr,          # fp32  (32, 16)
        out_ptr,               # fp32  (M, N)
        M, N, D,
        D_GROUPS,              # D // 16
        stride_qm, stride_qd,
        stride_kn, stride_kg, stride_kb,
        stride_sn, stride_sg,
        stride_om, stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BYTES_PER_CODEWORD: tl.constexpr,
        OFFSET_BITS: tl.constexpr,
        OFFSET_MASK: tl.constexpr,
        OFFSET_BIAS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)     # (BLOCK_M,)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)     # (BLOCK_N,)
        m_mask = m_offs < M
        n_mask = n_offs < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Iterate over the D_GROUPS groups; each contributes a 16-wide
        # inner product to the score tile.
        for g in range(0, D_GROUPS):
            # 1. Load Q tile columns for this group: (BLOCK_M, 16) fp16.
            d_offs = g * 16 + tl.arange(0, 16)
            q_tile = tl.load(
                Q_ptr + m_offs[:, None] * stride_qm + d_offs[None, :] * stride_qd,
                mask=m_mask[:, None] & (d_offs[None, :] < D),
                other=0.0,
            ).to(tl.float32)                                    # (BLOCK_M, 16)

            # 2. Decode K tile rows for this group: (BLOCK_N, 16) fp32.
            #    Each K row at this group is stored as one codeword +
            #    one fp32 scale.  Compute a 1D base pointer for the
            #    codeword bytes and add byte offsets.
            k_pack_base = K_pack_ptr + n_offs * stride_kn + g * stride_kg
            packed = tl.zeros((BLOCK_N,), dtype=tl.int64)
            for byte_i in tl.static_range(0, BYTES_PER_CODEWORD):
                b = tl.load(
                    k_pack_base + byte_i * stride_kb,
                    mask=n_mask, other=0,
                ).to(tl.int64)
                packed |= (b << (byte_i * 8))
            coset_idx = (packed & 0x1F).to(tl.int32)
            scale = tl.load(
                K_scale_ptr + n_offs * stride_sn + g * stride_sg,
                mask=n_mask, other=0.0,
            )                                                   # (BLOCK_N,)

            # 3. Build the decoded K tile (BLOCK_N, 16) column-by-column
            #    via mask-and-add (same trick as fused_pv).  Then use
            #    tl.dot for the (BLOCK_M, 16) @ (16, BLOCK_N) matmul.
            col_range = tl.arange(0, 16)                        # (16,)
            k_tile = tl.zeros((BLOCK_N, 16), dtype=tl.float32)
            for i in tl.static_range(0, 16):
                u = ((packed >> (5 + i * OFFSET_BITS)) & OFFSET_MASK).to(tl.int32)
                offset_signed = u - OFFSET_BIAS
                c = tl.load(codebook_ptr + coset_idx * 16 + i,
                            mask=n_mask, other=0.0)
                x_scaled = 2.0 * offset_signed.to(tl.float32) + c
                k_val = x_scaled * scale                        # (BLOCK_N,) fp32
                is_i = (col_range == i).to(tl.float32)          # (16,)
                k_tile = k_tile + k_val[:, None] * is_i[None, :]
            # tl.dot: (BLOCK_M, 16) @ (16, BLOCK_N) -> (BLOCK_M, BLOCK_N)
            acc += tl.dot(q_tile, tl.trans(k_tile), allow_tf32=False)

        # Write the score tile.
        tl.store(
            out_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )

    def fused_qk_bw16(
        Q: torch.Tensor,                    # fp16 (M, D)
        K_pack: torch.Tensor,               # uint8 (N, D_GROUPS, bytes)
        K_scale: torch.Tensor,              # fp32 (N, D_GROUPS)
        codebook: torch.Tensor,             # fp32 (32, 16)
        bits: int = 3,
        BLOCK_M: int = None,                # ignored (kernel is autotuned)
        BLOCK_N: int = None,                # ignored (kernel is autotuned)
    ) -> torch.Tensor:
        """
        Compute Q @ K^T where K is stored in packed BW16 form and is
        decoded on the fly inside the kernel.  Returns fp32 scores.
        """
        assert bits in _LAYOUT
        L = _LAYOUT[bits]
        assert Q.is_contiguous() and K_pack.is_contiguous() and K_scale.is_contiguous()
        assert Q.dtype == torch.float16
        assert K_pack.dtype == torch.uint8
        assert K_scale.dtype == torch.float32
        assert codebook.shape == (32, 16) and codebook.dtype == torch.float32
        assert K_pack.shape[-1] == L['bytes_per_codeword']

        M, D = Q.shape
        N = K_pack.shape[0]
        D_GROUPS = K_pack.shape[1]
        assert D_GROUPS * 16 == D or D_GROUPS * 16 >= D, f"D_GROUPS*16={D_GROUPS*16} vs D={D}"
        assert K_scale.shape == (N, D_GROUPS)

        out = torch.empty(M, N, dtype=torch.float32, device=Q.device)

        # Autotune picks BLOCK_M / BLOCK_N; grid is computed lazily.
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),
                             triton.cdiv(N, META['BLOCK_N']))
        _fused_qk_bw16_kernel[grid](
            Q, K_pack, K_scale, codebook.contiguous(), out,
            M, N, D, D_GROUPS,
            Q.stride(0), Q.stride(1),
            K_pack.stride(0), K_pack.stride(1), K_pack.stride(2),
            K_scale.stride(0), K_scale.stride(1),
            out.stride(0), out.stride(1),
            BYTES_PER_CODEWORD=L['bytes_per_codeword'],
            OFFSET_BITS=L['offset_bits'],
            OFFSET_MASK=L['offset_mask'],
            OFFSET_BIAS=L['offset_bias'],
        )
        return out

else:
    def fused_qk_bw16(*a, **kw):
        raise RuntimeError("triton is not installed")


if __name__ == "__main__":
    # Smoke test: fused Q @ K^T decode-in-kernel vs Q @ decoded_K^T.
    if not TRITON_AVAILABLE:
        raise SystemExit("triton missing")

    from kernels.reference.packed_bw16 import pack_bw16, unpack_bw16
    from kernels.reference.bw16_codebook import bw16_cosets

    torch.manual_seed(0)
    device = "cuda"
    M, N, D = 64, 128, 64
    Q = torch.randn(M, D, device=device, dtype=torch.float16)
    K = torch.randn(N, D, device=device)

    packed = pack_bw16(K, bits=3, group_size=16, scale_bits=8)
    codebook = bw16_cosets(dtype=torch.float32, device=device)

    # Reference: decode K first, then Q @ K^T.
    K_ref = unpack_bw16(packed, dtype=torch.float32)          # (N, D)
    scores_ref = Q.float() @ K_ref.t()                        # (M, N)

    # Fused: Q @ K^T with K decoded inline.
    scores_fused = fused_qk_bw16(Q, packed.codeword_bytes,
                                 packed.group_scale, codebook)

    diff = (scores_ref - scores_fused).abs()
    print(f"fused Q @ K^T   shape={tuple(scores_ref.shape)}")
    print(f"  max diff = {diff.max().item():.4e}")
    print(f"  mean diff = {diff.mean().item():.4e}")
    print(f"  ref range = [{scores_ref.min().item():.3f}, {scores_ref.max().item():.3f}]")
    # Bounded by fp16 -> fp32 -> fp32 accumulation order; expect ~1e-2 max.
    assert diff.max().item() < 1e-1, "fused Q @ K^T diverges from reference"
    print("  OK")
