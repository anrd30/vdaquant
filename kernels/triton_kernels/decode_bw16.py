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


BITS = 3
BYTES_PER_CODEWORD = 5      # 5 + 16 * 2 = 37 bits -> 5 bytes
LATTICE_DIM = 16
OFFSET_BITS = 2             # per-coord offset width at bits=3
OFFSET_BIAS = 2             # (1 << OFFSET_BITS) // 2


if TRITON_AVAILABLE:

    @triton.jit
    def _decode_bw16_kernel(
        codeword_ptr,          # uint8*  (n_codewords, 5)
        scale_ptr,             # fp32*   (n_codewords,)
        codebook_ptr,          # fp32*   (32, 16)
        out_ptr,               # fp16*   (n_codewords, 16)
        n_codewords,           # int
        BLOCK: tl.constexpr,
        BYTES_PER_CODEWORD: tl.constexpr,
        LATTICE_DIM: tl.constexpr,
        OFFSET_BITS: tl.constexpr,
        OFFSET_BIAS: tl.constexpr,
    ):
        # One program handles BLOCK codewords (16 output values each).
        pid = tl.program_id(0)
        start = pid * BLOCK
        offs_cw = start + tl.arange(0, BLOCK)          # (BLOCK,)
        mask = offs_cw < n_codewords

        # Load the 5 bytes per codeword and assemble a 64-bit packed int.
        packed = tl.zeros((BLOCK,), dtype=tl.int64)
        for byte_i in tl.static_range(0, BYTES_PER_CODEWORD):
            b = tl.load(codeword_ptr + offs_cw * BYTES_PER_CODEWORD + byte_i,
                        mask=mask, other=0).to(tl.int64)
            packed |= (b << (byte_i * 8))

        # Extract coset index (bits 0..4) and load per-group scale.
        coset_idx = (packed & 0x1F).to(tl.int32)            # (BLOCK,)
        scale = tl.load(scale_ptr + offs_cw, mask=mask, other=0.0)

        # For each of 16 coordinates, extract the 2-bit offset, look up
        # the coset value at that coordinate, combine, scale, and store.
        for i in tl.static_range(0, LATTICE_DIM):
            u = ((packed >> (5 + i * OFFSET_BITS)) & 0x3).to(tl.int32)
            offset_signed = u - OFFSET_BIAS                # {-2, -1, 0, 1}
            c = tl.load(codebook_ptr + coset_idx * LATTICE_DIM + i,
                        mask=mask, other=0.0)
            x_scaled = 2.0 * offset_signed.to(tl.float32) + c
            y = x_scaled * scale
            tl.store(out_ptr + offs_cw * LATTICE_DIM + i,
                     y.to(tl.float16),
                     mask=mask)

    def decode_bw16_triton(
        codeword_bytes: torch.Tensor,    # uint8  (..., G, BYTES_PER_CODEWORD)
        group_scale:    torch.Tensor,    # fp32   (..., G)
        codebook:       torch.Tensor,    # fp32   (32, 16)
        BLOCK: int = 128,
    ) -> torch.Tensor:
        """
        Decode a packed BW16 tile to fp16.

        Args:
            codeword_bytes: uint8, shape (..., G, 5)
            group_scale:    fp32,  shape (..., G)
            codebook:       fp32,  shape (32, 16)
            BLOCK:          codewords per program

        Returns:
            out: fp16, shape (..., G, 16) -- reshape at call site to
                 recover the original tile shape.
        """
        assert codeword_bytes.dtype == torch.uint8
        assert codeword_bytes.is_contiguous()
        assert group_scale.dtype == torch.float32
        assert codebook.shape == (32, 16)
        assert codebook.dtype == torch.float32

        prefix = codeword_bytes.shape[:-2]
        G = codeword_bytes.shape[-2]
        assert codeword_bytes.shape[-1] == BYTES_PER_CODEWORD, \
            f"expected {BYTES_PER_CODEWORD} bytes/codeword, got {codeword_bytes.shape[-1]}"
        assert group_scale.shape == (*prefix, G), \
            f"scale shape mismatch: {group_scale.shape} vs prefix+G {(*prefix, G)}"

        n_cw = int(torch.tensor(prefix).prod().item()) * G if prefix else G
        codewords_flat = codeword_bytes.reshape(n_cw, BYTES_PER_CODEWORD).contiguous()
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
            BYTES_PER_CODEWORD=BYTES_PER_CODEWORD,
            LATTICE_DIM=LATTICE_DIM,
            OFFSET_BITS=OFFSET_BITS,
            OFFSET_BIAS=OFFSET_BIAS,
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
    ref = pack_bw16_ref(x, bits=3, group_size=16, scale_bits=8)
    bits = bitpack_bw16(ref)

    # Reference decode.
    y_ref = unpack_bw16_ref(bitunpack_bw16(bits), dtype=torch.float32).to(torch.float16)

    # Triton decode.
    codebook = bw16_cosets(dtype=torch.float32, device=device)
    y_tri = decode_bw16_triton(bits.codeword_bytes, bits.group_scale, codebook)
    # y_tri comes back as (..., G, 16); reshape to match ref's original shape.
    y_tri = y_tri.reshape(x.shape).to(torch.float16)

    diff = (y_ref - y_tri).abs()
    print(f"Reference vs Triton: max diff = {diff.max().item():.2e}   "
          f"mean = {diff.mean().item():.2e}")
    print(f"Reference dtype: {y_ref.dtype}   Triton dtype: {y_tri.dtype}")
