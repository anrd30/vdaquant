"""
Triton decode kernel for packed BW16 KV storage.

For every 5-byte codeword this kernel:
  1. loads the 5 bytes and reconstructs a 40-bit integer
  2. extracts the 5-bit coset index (bits 0..4) and 16 * 2-bit offsets
     (bits 5..5+16*2-1)
  3. looks up the 32-coset codebook
  4. computes 2 * (offset - bias) + coset  (this is x_scaled)
  5. multiplies by the fp32 group scale

The output is a fp16 tensor of shape (n_codewords, 16) that matches
bit-for-bit with unpack_bw16_ref on the same packed input.

Only implemented for bits=3 for now; bits=2 and bits=4 are natural
generalisations with different byte counts (3 and 7 respectively).
Add them when needed.

Triton on Windows: as of triton 3.7 the runtime works but you need
CUDA-capable device.  Correctness reference: kernels/reference/packed_bw16.py.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# Codeword layout table -- per raw bits/scalar.
#   bytes_per_codeword = ceil((5 + 16*(bits-1)) / 8)
#   offset_bits        = bits - 1
#   offset_mask        = (1 << offset_bits) - 1
#   offset_bias        = 1 << (offset_bits - 1) if offset_bits > 0 else 0
_LAYOUT = {
    2: dict(bytes_per_codeword=3, offset_bits=1, offset_mask=0x1, offset_bias=1),
    3: dict(bytes_per_codeword=5, offset_bits=2, offset_mask=0x3, offset_bias=2),
    4: dict(bytes_per_codeword=7, offset_bits=3, offset_mask=0x7, offset_bias=4),
}
LATTICE_DIM = 16


if TRITON_AVAILABLE:

    @triton.jit
    def _decode_bw16_kernel(
        codeword_ptr,          # uint8*  (n_codewords, BYTES_PER_CODEWORD)
        scale_ptr,             # fp32*   (n_codewords,)
        codebook_ptr,          # fp32*   (32, LATTICE_DIM)
        out_ptr,               # fp16*   (n_codewords, LATTICE_DIM)
        n_codewords,
        BLOCK: tl.constexpr,
        BYTES_PER_CODEWORD: tl.constexpr,
        LATTICE_DIM: tl.constexpr,
        OFFSET_BITS: tl.constexpr,
        OFFSET_MASK: tl.constexpr,
        OFFSET_BIAS: tl.constexpr,
    ):
        # One program handles BLOCK codewords (LATTICE_DIM output vals each).
        pid = tl.program_id(0)
        start = pid * BLOCK
        offs_cw = start + tl.arange(0, BLOCK)          # (BLOCK,)
        mask = offs_cw < n_codewords

        # Load bytes, assemble packed int.  int64 covers up to 8 bytes /
        # codeword; the biggest supported here is bits=4 -> 7 bytes.
        packed = tl.zeros((BLOCK,), dtype=tl.int64)
        for byte_i in tl.static_range(0, BYTES_PER_CODEWORD):
            b = tl.load(codeword_ptr + offs_cw * BYTES_PER_CODEWORD + byte_i,
                        mask=mask, other=0).to(tl.int64)
            packed |= (b << (byte_i * 8))

        # 5-bit coset index (bits 0..4).
        coset_idx = (packed & 0x1F).to(tl.int32)
        scale = tl.load(scale_ptr + offs_cw, mask=mask, other=0.0)

        # Per-coordinate: OFFSET_BITS-bit offset, look up codebook,
        # combine to x_scaled = 2 * (offset - bias) + coset, scale.
        for i in tl.static_range(0, LATTICE_DIM):
            u = ((packed >> (5 + i * OFFSET_BITS)) & OFFSET_MASK).to(tl.int32)
            offset_signed = u - OFFSET_BIAS
            c = tl.load(codebook_ptr + coset_idx * LATTICE_DIM + i,
                        mask=mask, other=0.0)
            x_scaled = 2.0 * offset_signed.to(tl.float32) + c
            y = x_scaled * scale
            tl.store(out_ptr + offs_cw * LATTICE_DIM + i,
                     y.to(tl.float16),
                     mask=mask)

    def decode_bw16_triton(
        codeword_bytes: torch.Tensor,    # uint8  (..., G, bytes_per_cw)
        group_scale:    torch.Tensor,    # fp32   (..., G)
        codebook:       torch.Tensor,    # fp32   (32, 16)
        bits: int = 3,
        BLOCK: int = 128,
    ) -> torch.Tensor:
        """
        Decode a packed BW16 tile to fp16, for bits in {2, 3, 4}.
        Returns fp16, shape (..., G, 16) -- reshape at call site.
        """
        assert bits in _LAYOUT, f"unsupported bits={bits}"
        L = _LAYOUT[bits]
        assert codeword_bytes.dtype == torch.uint8
        assert codeword_bytes.is_contiguous()
        assert group_scale.dtype == torch.float32
        assert codebook.shape == (32, 16)
        assert codebook.dtype == torch.float32

        prefix = codeword_bytes.shape[:-2]
        G = codeword_bytes.shape[-2]
        bpcw = L['bytes_per_codeword']
        assert codeword_bytes.shape[-1] == bpcw, \
            f"expected {bpcw} bytes/codeword at bits={bits}, got {codeword_bytes.shape[-1]}"
        assert group_scale.shape == (*prefix, G), \
            f"scale shape mismatch: {group_scale.shape} vs prefix+G {(*prefix, G)}"

        n_cw = 1
        for s in prefix:
            n_cw *= s
        n_cw *= G

        codewords_flat = codeword_bytes.reshape(n_cw, bpcw).contiguous()
        scale_flat = group_scale.reshape(n_cw).contiguous()

        out = torch.empty(n_cw, LATTICE_DIM,
                          dtype=torch.float16,
                          device=codeword_bytes.device)

        grid = (triton.cdiv(n_cw, BLOCK),)
        _decode_bw16_kernel[grid](
            codewords_flat, scale_flat,
            codebook.contiguous(),
            out,
            n_cw,
            BLOCK=BLOCK,
            BYTES_PER_CODEWORD=bpcw,
            LATTICE_DIM=LATTICE_DIM,
            OFFSET_BITS=L['offset_bits'],
            OFFSET_MASK=L['offset_mask'],
            OFFSET_BIAS=L['offset_bias'],
        )
        return out.reshape(*prefix, G, LATTICE_DIM)

else:
    def decode_bw16_triton(*a, **kw):
        raise RuntimeError("triton is not installed; import failed.")


if __name__ == "__main__":
    # Cheap smoke test — run and compare against the reference unpack.
    if not TRITON_AVAILABLE:
        print("triton not available; skipping.")
        raise SystemExit(0)
    from kernels.reference.packed_bw16 import (
        pack_bw16_ref, unpack_bw16_ref, bitpack_bw16, bitunpack_bw16,
    )
    from kernels.reference.bw16_codebook import bw16_cosets

    device = "cuda"
    torch.manual_seed(0)
    x = torch.randn(2, 4, 32, device=device)
    codebook = bw16_cosets(dtype=torch.float32, device=device)
    for bits in [2, 3, 4]:
        ref = pack_bw16_ref(x, bits=bits, group_size=16, scale_bits=8)
        packed_bits = bitpack_bw16(ref)
        y_ref = unpack_bw16_ref(bitunpack_bw16(packed_bits),
                                dtype=torch.float32).to(torch.float16)
        y_tri = decode_bw16_triton(packed_bits.codeword_bytes,
                                   packed_bits.group_scale, codebook,
                                   bits=bits)
        y_tri = y_tri.reshape(x.shape).to(torch.float16)
        diff = (y_ref - y_tri).abs()
        print(f"  bits={bits}: max diff = {diff.max().item():.2e}   "
              f"mean = {diff.mean().item():.2e}   "
              f"bytes/codeword = {_LAYOUT[bits]['bytes_per_codeword']}")
