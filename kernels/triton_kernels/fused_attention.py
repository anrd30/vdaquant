"""
fused_attention_bw16: end-to-end attention with packed BW16 K and V.

Combines fused_qk (Q @ K^T with K decoded inline) + softmax (unfused,
PyTorch) + fused_pv (P @ V with V decoded inline) into one call.  The
fp16 K and V intermediates never exist -- only the attention weights
matrix P lives in fp32 memory during the forward.

This is the two-kernel version.  A full FlashAttention-style single-
kernel version with online softmax is next up (fold the softmax into
the fused_qk pass so P also never materialises, then stream V decode
into the accumulator).

Correctness: fp accumulation-order-bounded diff vs standard attention
on the same (decoded) K, V.  For bit-parity with a SIMULATOR path
(fp16 quantised K, V through PyTorch attention), see
kernels/tests/test_fused_attention.py.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .fused_qk import fused_qk_bw16
from .fused_pv import fused_pv_bw16


def fused_attention_bw16(
    Q: torch.Tensor,                       # fp16 (M, D)
    packed_K,                              # PackedBW16Bits
    packed_V,                              # PackedBW16Bits
    codebook: torch.Tensor,                # fp32 (32, 16)
    scale: Optional[float] = None,
    bits: int = 3,
    BLOCK_M: int = 32,
    BLOCK_N: int = 32,
) -> torch.Tensor:
    """
    Full attention output = softmax(Q @ K^T * scale) @ V, with K and V
    packed and decoded inside their fused kernels.

    Args:
        Q:          fp16 (M, D)
        packed_K:   PackedBW16Bits produced by pack_bw16 on a (M', D)
                    tensor with M' == M (assumes self-attention shape;
                    for cross-attention pass separate M and M').
        packed_V:   PackedBW16Bits produced by pack_bw16 on the V tensor.
        codebook:   the BW16 codebook returned by bw16_cosets on the
                    same device.
        scale:      attention scale factor, default 1 / sqrt(D).

    Returns:
        out: fp32 (M, D)
    """
    assert Q.dtype == torch.float16
    M, D = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # 1. Q @ K^T with K decoded inline -> fp32 (M, N).
    scores = fused_qk_bw16(Q, packed_K.codeword_bytes, packed_K.group_scale,
                            codebook, bits=bits, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    scores = scores * scale

    # 2. Softmax over the N axis.  PyTorch is fine here -- the fused-
    # softmax variant (FlashAttention-style) is the next step.
    probs = F.softmax(scores, dim=-1)

    # 3. P @ V with V decoded inline -> fp32 (M, D).
    out = fused_pv_bw16(probs, packed_V.codeword_bytes, packed_V.group_scale,
                         codebook, bits=bits, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out
