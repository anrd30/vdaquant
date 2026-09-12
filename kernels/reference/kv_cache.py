"""
PackedKVCache: drop-in replacement for a fp16 KV buffer in VDA temporal
cross-attention.

The current VDA path stores fp16 K and V tensors of shape
(B, T, tokens, head_dim) and reads them back during the temporal-window
attention.  This module replaces that fp16 buffer with a packed BW16
representation while keeping the read/write API identical, so the
integration point in RotatedTemporalAttention is a one-line swap:

    -   self.k_cache = torch.zeros(B, T, N, D, dtype=torch.float16, device=dev)
    +   self.k_cache = PackedKVCache(B, T, N, D, bits=3, device=dev)
    ...
    -   self.k_cache[:, t] = k                       # write
    -   attn_k = self.k_cache[:, :t+1]               # read
    +   self.k_cache.write(t, k)                     # write
    +   attn_k = self.k_cache.read(t + 1)            # read, returns fp16

Reads decode the packed storage back to fp16 on the fly; the packed
bytes live on the GPU for the duration of the temporal window and are
the only per-frame KV state that persists.

Correctness note.  Writes use `pack_bw16(..., scale_bits=8)` which is
BIT-EXACT with LatticeBW16Quantizer(scale_bits=8) (see
kernels/tests/test_simulator_parity.py), so this cache produces the
same fp16 output tensor a simulator-based cache would produce.

Memory note.  For B=1, T=32 and every VDA temporal-attention layer,
the sum of packed sizes is 5.33x smaller than the fp16 baseline
(kernels/benchmark_memory.py).  Rea-only allocations spike briefly
during the decode step but the persistent cache is int8.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from .packed_bw16 import (
    PackedBW16Bits,
    bitpack_bw16, bitunpack_bw16,
    pack_bw16_ref, unpack_bw16_ref,
)


class PackedKVCache:
    """Per-temporal-frame packed BW16 cache slot store.

    The cache is a list of `T` packed BW16 tiles, one per temporal
    frame.  Each tile stores shape (B, tokens, head_dim) after a
    write; reads return fp16 tensors of shape (B, k, tokens, head_dim)
    for the requested prefix length `k`.

    Args:
        B, T, tokens, head_dim: cache shape.
        bits:                   raw BW16 bits per scalar (2, 3, or 4)
        scale_bits:             8 for simulator-parity, 16 for max
                                accuracy (default 8)
        group_size:             scale-group size, multiple of 16
        device:                 target CUDA / CPU device
        rotate_fn:              optional callable applied to k/v BEFORE
                                packing; use to swap in the Hadamard
                                rotation if the caller has not already
                                applied one.  Signature (x -> x).
    """

    def __init__(self,
                 B: int, T: int, tokens: int, head_dim: int,
                 bits: int = 3,
                 scale_bits: int = 8,
                 group_size: int = 16,
                 device: str | torch.device = "cuda",
                 rotate_fn: Optional[callable] = None):
        assert bits in (2, 3, 4), f"bits must be 2, 3, or 4, got {bits}"
        assert scale_bits in (8, 16)
        self.B = B
        self.T = T
        self.tokens = tokens
        self.head_dim = head_dim
        self.bits = bits
        self.scale_bits = scale_bits
        self.group_size = group_size
        self.device = torch.device(device)
        self.rotate_fn = rotate_fn
        # Lazy allocation: only fill slots that are actually written.
        # A None slot is "not yet written".  Reads before write raise.
        self._slots: List[Optional[PackedBW16Bits]] = [None] * T
        # Cached rotation buffer shared with the write path to avoid
        # per-frame allocation churn.
        self._buf: Optional[torch.Tensor] = None

    # -------------------------------------------------------------- write
    def write(self, t: int, x: torch.Tensor) -> None:
        """Store the frame at temporal index t.

        Args:
            t: 0 <= t < T
            x: (B, tokens, head_dim) fp tensor, on self.device.
        """
        assert 0 <= t < self.T, f"t={t} out of [0, {self.T})"
        assert x.shape == (self.B, self.tokens, self.head_dim), \
            f"expected shape {(self.B, self.tokens, self.head_dim)}, got {tuple(x.shape)}"
        if self.rotate_fn is not None:
            x = self.rotate_fn(x)
        # pack_bw16_ref accepts (..., D) with D a multiple of group_size;
        # our head_dim may be smaller than group_size (VDA has head_dim=8
        # in the DPT cross-attention).  Absorb the tokens axis if so.
        if self.head_dim < self.group_size or self.head_dim % self.group_size != 0:
            merged = x.reshape(self.B, self.tokens * self.head_dim)
            pad = (-merged.shape[-1]) % self.group_size
            if pad > 0:
                merged = torch.nn.functional.pad(merged, (0, pad))
            ref = pack_bw16_ref(merged, bits=self.bits,
                                group_size=self.group_size,
                                scale_bits=self.scale_bits)
        else:
            ref = pack_bw16_ref(x, bits=self.bits,
                                group_size=self.group_size,
                                scale_bits=self.scale_bits)
        self._slots[t] = bitpack_bw16(ref)

    # -------------------------------------------------------------- read
    def read(self, k: int,
             dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Return the first k frames as an fp tensor of shape
        (B, k, tokens, head_dim).  Slots that have not been written are
        an error."""
        assert 0 < k <= self.T, f"k={k} out of (0, {self.T}]"
        frames = []
        for t in range(k):
            slot = self._slots[t]
            if slot is None:
                raise RuntimeError(f"cache slot {t} not yet written")
            ref = bitunpack_bw16(slot)
            x = unpack_bw16_ref(ref, dtype=torch.float32)
            # If write absorbed the tokens axis into head_dim, restore.
            if self.head_dim < self.group_size or self.head_dim % self.group_size != 0:
                merged = self.tokens * self.head_dim
                x = x[..., :merged].reshape(self.B, self.tokens, self.head_dim)
            frames.append(x.unsqueeze(1).to(dtype))
        return torch.cat(frames, dim=1)                       # (B, k, tokens, head_dim)

    # -------------------------------------------------------------- utils
    def nbytes(self) -> int:
        """Total bytes on the GPU for the packed cache."""
        return sum(s.nbytes() for s in self._slots if s is not None)

    def fp16_reference_bytes(self) -> int:
        """Bytes a fp16 buffer of shape (B, T, tokens, head_dim) would take."""
        return self.B * self.T * self.tokens * self.head_dim * 2

    def compression_ratio(self) -> float:
        return self.fp16_reference_bytes() / max(self.nbytes(), 1)

    def n_filled(self) -> int:
        return sum(1 for s in self._slots if s is not None)

    def __repr__(self) -> str:
        return (f"PackedKVCache(B={self.B}, T={self.T}, tokens={self.tokens}, "
                f"head_dim={self.head_dim}, bits={self.bits}, "
                f"filled={self.n_filled()}/{self.T}, "
                f"nbytes={self.nbytes()/1024**2:.2f} MB)")
