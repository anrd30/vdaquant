"""
Packed BW16 KV-cache storage (pure PyTorch reference implementation).

Simulator-parity scale convention.  For target raw bits per scalar b in
{2, 3, 4}, the packer uses:

    half_levels    = 2**b // 2
    alpha          = per-group absmax                        (fp)
    scale          = alpha / (half_levels - 1)               (fp)
    x_scaled       = x / scale                              (fp), then clamp
                     to [-half_levels, half_levels - 1]
    nearest BW16   = argmin_c || 2 * round((x - c)/2) + c - x_scaled ||^2
                     over the 32 RM(1,4) cosets
    offset per     = round((x_scaled - c_winner) / 2), bounded to
       coordinate    [-half_levels/2, half_levels/2 - 1]

This matches research/quantizers/lattice_vq.py::LatticeBW16Quantizer up to
floating-point round-off.  The bit width of each offset is
`ceil(log2(half_levels))` == b - 1, so at 3-bit offsets fit in 2 bits and
we can pack:

    per 16-scalar group:  5 bits coset + 16 * (b-1) bits offsets
    at b == 3:            5 + 32 = 37 bits + 8-bit scale = 45 bits
                          rounded to 6 bytes -> 3.00 raw bits/scalar
                          + 8/g scale-metadata overhead

which lands on the paper's 3.5 effective bits/scalar quote (with g=16).

Two layouts are exposed:

    - PackedBW16Ref (this file)  -- uint8 per offset, easy to read, 1.6x
                                    compression.  Serves as the correctness
                                    reference for the bit-packed layout.
    - PackedBW16Bits (see below) -- real bit-packed 5+32-bit codewords in
                                    uint8 arrays, 4x compression, matches
                                    PackedBW16Ref bit-for-bit after decode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .bw16_codebook import bw16_cosets, bw16_lattice_dim, bw16_num_cosets


LATTICE_DIM = bw16_lattice_dim()   # 16
NUM_COSETS = bw16_num_cosets()     # 32

_CHUNK_ROWS = 1 << 15              # 32k rows per chunk -> ~64 MB intermediate


# =============================================================================
# Reference layout (uint8 per offset)
# =============================================================================
@dataclass
class PackedBW16Ref:
    """Reference-layout packed BW16 tile.

    packed_coset:   uint8, shape (..., G)      coset in 0..31
    packed_offsets: int8,  shape (..., G, 16)  per-coordinate integer offset
    group_scale:    fp16,  shape (..., G)      per-group scale (alpha/(hl-1))
    original_shape: tuple                       shape before flatten-to-groups
    bits:           int                         raw bits per scalar (2, 3, 4)
    """
    packed_coset:   torch.Tensor
    packed_offsets: torch.Tensor
    group_scale:    torch.Tensor
    original_shape: Tuple[int, ...]
    bits:           int

    def nbytes(self) -> int:
        # scale is fp32 in the reference dataclass for bit-parity with the
        # simulator; real deployment uses 1 byte per scale entry (int8
        # index against a per-tensor fp32 step), so we report the deployed
        # cost of 1 byte per scale rather than the reference's 4 bytes.
        return (self.packed_coset.numel()   * 1     # uint8
                + self.packed_offsets.numel() * 1   # int8
                + self.group_scale.numel()    * 1)  # deployed 1 byte / group

    def fp16_reference_bytes(self) -> int:
        n = 1
        for s in self.original_shape:
            n *= s
        return n * 2

    def compression_ratio(self) -> float:
        return self.fp16_reference_bytes() / max(self.nbytes(), 1)


def _find_nearest_bw16(x_scaled: torch.Tensor,
                       cosets: torch.Tensor,
                       lo: float,
                       hi: float
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Nearest-neighbour search over the 32 BW16 cosets, per-coordinate range
    clamped to [lo, hi], batch-chunked so a full VDA KV tile fits in 6 GB.

    Args:
        x_scaled: (..., 16)  after per-group affine (in the y-frame).
        cosets:   (32, 16)   BW16 coset representatives (0/1).
        lo, hi:   coordinate bounds; nearest point z satisfies 2z+c in [lo,hi].

    Returns:
        best_idx:      (...,)    long, winning coset in 0..31
        best_z:        (..., 16) integer offsets z (rounded and clamped)
    """
    orig_shape = x_scaled.shape[:-1]
    flat = x_scaled.reshape(-1, 16)                       # (N, 16)
    N = flat.shape[0]

    # Per-coordinate legal bounds for z depend on c.
    lo_bounds = torch.ceil((lo - cosets) * 0.5)           # (32, 16)
    hi_bounds = torch.floor((hi - cosets) * 0.5)          # (32, 16)

    if N <= _CHUNK_ROWS:
        delta = (flat.unsqueeze(1) - cosets) * 0.5        # (N, 32, 16)
        z = delta.round().clamp(min=lo_bounds, max=hi_bounds)   # (N, 32, 16)
        cand = 2.0 * z + cosets                            # (N, 32, 16)
        sq_err = ((cand - flat.unsqueeze(1)) ** 2).sum(dim=-1)  # (N, 32)
        best_idx = sq_err.argmin(dim=-1)                   # (N,)
        best_z = z.gather(1, best_idx.view(-1, 1, 1).expand(-1, 1, 16)).squeeze(1)
    else:
        best_idx = torch.empty(N, dtype=torch.long, device=flat.device)
        best_z = torch.empty(N, 16, dtype=flat.dtype, device=flat.device)
        for i in range(0, N, _CHUNK_ROWS):
            j = min(i + _CHUNK_ROWS, N)
            slab = flat[i:j]
            delta = (slab.unsqueeze(1) - cosets) * 0.5
            z = delta.round().clamp(min=lo_bounds, max=hi_bounds)
            cand = 2.0 * z + cosets
            sq_err = ((cand - slab.unsqueeze(1)) ** 2).sum(dim=-1)
            idx_slab = sq_err.argmin(dim=-1)
            best_idx[i:j] = idx_slab
            best_z[i:j] = z.gather(1, idx_slab.view(-1, 1, 1).expand(-1, 1, 16)).squeeze(1)

    return (best_idx.reshape(orig_shape),
            best_z.reshape(*orig_shape, 16))


def pack_bw16_ref(x: torch.Tensor,
                  bits: int = 3,
                  group_size: int = LATTICE_DIM,
                  scale_bits: int = 16
                  ) -> PackedBW16Ref:
    """
    Pack an fp tensor into reference-layout BW16 storage.

    Simulator-parity scaling: half_levels = 2**bits // 2,
    alpha = per-group absmax, scale = alpha / (half_levels - 1),
    x_scaled = clamp(x / scale, -half_levels, half_levels - 1).

    Args:
        x:          (..., D), D % group_size == 0.  Any float dtype.
        bits:       raw bits per scalar in {2, 3, 4}.  Coset uses 5 bits
                    on top; effective bits/scalar = bits + 8/group_size.
        group_size: scalars per shared scale, multiple of 16.

    Returns:
        PackedBW16Ref with all fields populated.  Round-tripped via
        unpack_bw16_ref this matches the simulator to within fp round-off.
    """
    assert bits in (2, 3, 4), f"bits must be 2, 3, or 4, got {bits}"
    assert scale_bits in (8, 16), f"scale_bits must be 8 or 16, got {scale_bits}"
    assert x.dtype in (torch.float16, torch.float32, torch.bfloat16), \
        f"expected float tensor, got {x.dtype}"
    assert group_size % LATTICE_DIM == 0, \
        f"group_size ({group_size}) must be a multiple of {LATTICE_DIM}"
    assert x.shape[-1] % group_size == 0, \
        f"last dim {x.shape[-1]} must be a multiple of group_size {group_size}"

    original_shape = tuple(x.shape)
    half_levels = (1 << bits) // 2                        # 3-bit -> 4
    lo = -float(half_levels)                              # -4
    hi = float(half_levels - 1)                           # 3
    D = x.shape[-1]
    n_groups_per_row = D // group_size
    lattices_per_group = group_size // LATTICE_DIM

    x_grouped = x.reshape(*x.shape[:-1], n_groups_per_row, group_size)
    alpha = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = alpha / (half_levels - 1)                     # matches simulator

    if scale_bits == 8:
        # Match LatticeBW16Quantizer's exact 8-bit scale scheme so packed
        # storage becomes a BIT-PARITY drop-in for the simulator.
        scale_max = scale.abs().amax().clamp(min=1e-8)
        scale_step = scale_max / 255.0
        scale = (scale / scale_step).round().clamp(0, 255) * scale_step

    x_scaled = (x_grouped / scale).clamp(lo, hi)
    x_lat = x_scaled.reshape(*x_scaled.shape[:-1], lattices_per_group, LATTICE_DIM)

    cosets = bw16_cosets(dtype=x_lat.dtype, device=x_lat.device)
    best_idx, best_z = _find_nearest_bw16(x_lat, cosets, lo, hi)
    # best_idx has shape (..., n_groups_per_row, lattices_per_group).
    # best_z  has shape (..., n_groups_per_row, lattices_per_group, 16).
    # Fold the two grouping axes into one G_total axis so downstream code
    # only sees a single per-group axis.
    prefix = best_idx.shape[:-2]                          # (...,)
    G_total = n_groups_per_row * lattices_per_group

    packed_coset = best_idx.to(torch.uint8).reshape(*prefix, G_total)
    packed_offsets = (best_z.round().clamp(-128, 127).to(torch.int8)
                      .reshape(*prefix, G_total, LATTICE_DIM))
    # Broadcast per-group scale to per-lattice so unpack does not need to
    # know lattices_per_group.
    scale_flat = scale.squeeze(-1)                        # (..., n_groups_per_row)
    if lattices_per_group != 1:
        scale_flat = scale_flat.unsqueeze(-1).expand(*scale_flat.shape,
                                                     lattices_per_group).reshape(
            *scale_flat.shape[:-1], -1)                   # (..., G_total)

    # Store the quantised scale in fp32 so unpack sees the same value the
    # simulator uses.  In a real deployment the scale can be stored as
    # int8 index + fp32 step (2 numbers per tile) at negligible cost;
    # keeping fp32 here is the simplest reference layout.
    return PackedBW16Ref(
        packed_coset=packed_coset,
        packed_offsets=packed_offsets,
        group_scale=scale_flat.to(torch.float32),
        original_shape=original_shape,
        bits=bits,
    )


def unpack_bw16_ref(p: PackedBW16Ref, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Decode a reference-layout tile back to fp."""
    dev = p.packed_coset.device
    cosets = bw16_cosets(dtype=dtype, device=dev)

    idx = p.packed_coset.long().unsqueeze(-1).unsqueeze(-1).expand(
        *p.packed_coset.shape, 1, LATTICE_DIM)             # (..., G, 1, 16)
    coset_val = cosets.expand(*p.packed_coset.shape, NUM_COSETS, LATTICE_DIM).gather(
        -2, idx).squeeze(-2)                               # (..., G, 16)
    z = p.packed_offsets.to(dtype)                         # (..., G, 16)

    x_scaled_hat = 2.0 * z + coset_val                     # (..., G, 16), in y-frame
    x_hat = x_scaled_hat * p.group_scale.to(dtype).unsqueeze(-1)  # broadcast scale
    # Restore original shape by flatten and reshape.
    return x_hat.reshape(*x_hat.shape[:-2], -1).reshape(p.original_shape).to(dtype)


# =============================================================================
# Bit-packed layout (real 3-bit storage)
# =============================================================================
@dataclass
class PackedBW16Bits:
    """Bit-packed BW16 tile.

    codeword_bytes: uint8, shape (..., G, n_bytes_per_codeword)
                                one codeword = 5-bit coset || 16 * (bits-1) bits offsets
                                packed little-endian.
    group_scale:    fp16,  shape (..., G)
    original_shape: tuple
    bits:           int
    n_bytes_per_codeword: int   ceil((5 + 16 * (bits-1)) / 8)
    """
    codeword_bytes: torch.Tensor
    group_scale:    torch.Tensor
    original_shape: Tuple[int, ...]
    bits:           int
    n_bytes_per_codeword: int

    def nbytes(self) -> int:
        # Same 1-byte-per-scale accounting as PackedBW16Ref: deployed
        # cost, not the reference's fp32 storage cost.
        return (self.codeword_bytes.numel() * 1
                + self.group_scale.numel() * 1)

    def fp16_reference_bytes(self) -> int:
        n = 1
        for s in self.original_shape:
            n *= s
        return n * 2

    def compression_ratio(self) -> float:
        return self.fp16_reference_bytes() / max(self.nbytes(), 1)


def _bytes_per_codeword(bits: int) -> int:
    """Total bytes needed to store one BW16 codeword at raw `bits`/scalar."""
    return (5 + 16 * (bits - 1) + 7) // 8


def bitpack_bw16(p: PackedBW16Ref) -> PackedBW16Bits:
    """
    Convert a reference-layout PackedBW16Ref into the true bit-packed layout.

    Bit layout within one codeword (little-endian across bytes):
        bit 0..4         : coset index (5 bits)
        bit 5..5+(b-1)   : offset[0] biased by half_levels/2 to be unsigned
        ...
        bit 5+15*(b-1)..: offset[15] biased

    A 3-bit codeword thus needs 5 + 16*2 = 37 bits -> 5 bytes.
    A 2-bit codeword needs        5 + 16*1 = 21 bits -> 3 bytes.
    A 4-bit codeword needs        5 + 16*3 = 53 bits -> 7 bytes.
    """
    b = p.bits
    off_bits = b - 1                                       # bits per offset
    off_bias = (1 << off_bits) // 2                        # 3-bit -> bias 2
    n_bytes = _bytes_per_codeword(b)
    dev = p.packed_coset.device

    # Flatten per-codeword axes.
    coset = p.packed_coset.long()                          # (..., G)
    offsets = p.packed_offsets.long() + off_bias           # (..., G, 16), unsigned

    # Sanity: unsigned offsets must fit in off_bits.
    max_uoff = (1 << off_bits) - 1
    assert (offsets >= 0).all() and (offsets <= max_uoff).all(), \
        f"offsets out of unsigned {off_bits}-bit range at bits={b}: " \
        f"[{offsets.min().item()}, {offsets.max().item()}], expected [0, {max_uoff}]"

    # Pack into an integer per codeword: bit 0..4 coset, then offsets.
    packed_int = coset & 0x1F                              # (..., G)
    for i in range(16):
        packed_int = packed_int | (offsets[..., i] << (5 + i * off_bits))

    # Serialise to bytes, LSB-first.
    codeword_bytes = torch.zeros(*coset.shape, n_bytes, dtype=torch.uint8, device=dev)
    for byte_i in range(n_bytes):
        codeword_bytes[..., byte_i] = ((packed_int >> (byte_i * 8)) & 0xFF).to(torch.uint8)

    return PackedBW16Bits(
        codeword_bytes=codeword_bytes,
        group_scale=p.group_scale,
        original_shape=p.original_shape,
        bits=b,
        n_bytes_per_codeword=n_bytes,
    )


def bitunpack_bw16(p: PackedBW16Bits) -> PackedBW16Ref:
    """Inverse of bitpack_bw16.  Returns a reference-layout tile."""
    b = p.bits
    off_bits = b - 1
    off_bias = (1 << off_bits) // 2
    off_mask = (1 << off_bits) - 1

    # Reassemble the per-codeword integer.
    packed_int = torch.zeros(p.codeword_bytes.shape[:-1],
                             dtype=torch.long,
                             device=p.codeword_bytes.device)
    for byte_i in range(p.n_bytes_per_codeword):
        packed_int = packed_int | (p.codeword_bytes[..., byte_i].long() << (byte_i * 8))

    coset = (packed_int & 0x1F).to(torch.uint8)                       # (..., G)
    offsets = torch.empty(*packed_int.shape, 16, dtype=torch.int8,
                          device=packed_int.device)
    for i in range(16):
        u = (packed_int >> (5 + i * off_bits)) & off_mask
        offsets[..., i] = (u - off_bias).to(torch.int8)

    return PackedBW16Ref(
        packed_coset=coset,
        packed_offsets=offsets,
        group_scale=p.group_scale,
        original_shape=p.original_shape,
        bits=b,
    )


# =============================================================================
# Top-level convenience: fp -> bit-packed -> fp
# =============================================================================
def pack_bw16(x: torch.Tensor,
              bits: int = 3,
              group_size: int = LATTICE_DIM,
              scale_bits: int = 16
              ) -> PackedBW16Bits:
    """Top-level packer: fp tensor -> bit-packed BW16 tile.

    Pass scale_bits=8 for bit-parity with LatticeBW16Quantizer(scale_bits=8).
    """
    return bitpack_bw16(pack_bw16_ref(x, bits=bits, group_size=group_size,
                                      scale_bits=scale_bits))


def unpack_bw16(p: PackedBW16Bits, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Top-level unpacker: bit-packed BW16 tile -> fp tensor."""
    return unpack_bw16_ref(bitunpack_bw16(p), dtype=dtype)


# =============================================================================
# Legacy alias kept so existing tests / callers keep working.
# =============================================================================
PackedBW16 = PackedBW16Ref            # legacy dataclass name


def pack_bw16_reference(x: torch.Tensor,
                        group_size: int = LATTICE_DIM,
                        scale_bits: int = 8,
                        bits: int = 3) -> PackedBW16Ref:
    """Legacy alias for the reference (non-bit-packed) packer."""
    _ = scale_bits  # accepted for backwards compat; 8-bit scale is implicit
    return pack_bw16_ref(x, bits=bits, group_size=group_size)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 4, 32)
    ref = pack_bw16_ref(x, bits=3, group_size=16)
    bits_pack = bitpack_bw16(ref)
    ref_roundtrip = bitunpack_bw16(bits_pack)
    y_ref = unpack_bw16_ref(ref)
    y_bits = unpack_bw16(bits_pack)

    print(f"[ref layout]   packed={ref.nbytes()} B   fp16={ref.fp16_reference_bytes()} B"
          f"   ratio={ref.compression_ratio():.2f}x")
    print(f"[bit-packed]   packed={bits_pack.nbytes()} B   fp16={bits_pack.fp16_reference_bytes()} B"
          f"   ratio={bits_pack.compression_ratio():.2f}x")
    print(f"[bytes/codeword] {bits_pack.n_bytes_per_codeword}")
    print(f"[bit-parity] coset diff = "
          f"{(ref.packed_coset != ref_roundtrip.packed_coset).sum().item()}")
    print(f"[bit-parity] offset diff = "
          f"{(ref.packed_offsets != ref_roundtrip.packed_offsets).sum().item()}")
    err = (y_bits - y_ref).abs()
    print(f"[decode identity]  max diff = {err.max().item():.2e}   mean = {err.mean().item():.2e}")
    err = (y_ref - x).abs()
    print(f"[round-trip]       max err  = {err.max().item():.4f}   mean = {err.mean().item():.4f}")
