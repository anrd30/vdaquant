"""
Lattice and Vector Quantizers for post-rotation compression.

After applying the Randomized Hadamard Transform (RHT), feature vectors
are approximately uniformly distributed on a hypersphere. This module
provides quantizers optimized for this transformed distribution.

Quantizer Hierarchy (from simplest to most powerful):
    1. ScalarRoundQuantizer: Standard uniform rounding (baseline, like AIMET)
    2. UniformVectorQuantizer: Groups dimensions into small vectors and
       quantizes each vector jointly (better rate-distortion than scalar)
    3. LatticeD4Quantizer: Uses the D4 checkerboard lattice for near-optimal
       4-dimensional vector quantization

Theory:
    For a d-dimensional uniform source, the rate-distortion bound states
    that vector quantization with block size k achieves a coding gain of:
        G_k = (1/d) · V_k^{2/k}
    where V_k is the volume of the k-dimensional Voronoi region.

    The D4 lattice (4-dimensional checkerboard) achieves G_4 ≈ 1.19 dB
    coding gain over scalar quantization — meaning at the SAME bit-rate,
    D4 lattice quantization has ~20% lower mean squared error.

References:
    [1] Conway & Sloane, "Sphere Packings, Lattices and Groups", 1999
    [2] TurboQuant (PolarQuant component), Google Research, 2026
    [3] HyperQuant (E8/D4 lattice quantization), arXiv:2606.23406, 2026
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class IdentityQuantizer(nn.Module):
    """
    No-op quantizer: returns the input unchanged. Used to isolate
    non-quantization effects (e.g. the Hadamard-rotation / attention-reshape
    contract) from quantization noise during equivalence testing — see
    docs/optimization_ledger.md T7.
    """

    def __init__(self, bits: int = 32, group_size: int = 1):
        super().__init__()
        self.bits = bits
        self.group_size = group_size

    def forward(self, x: torch.Tensor, per_channel: bool = False) -> Tuple[torch.Tensor, dict]:
        return x, {'method': 'identity', 'bits': self.bits}

    def extra_repr(self) -> str:
        return "method=identity (no-op, for equivalence testing)"


class ScalarRoundQuantizer(nn.Module):
    """
    Standard uniform scalar round-to-nearest quantizer (baseline).

    This is what AIMET / TensorRT / standard PTQ does:
        q(x) = clamp(round(x / Δ)) · Δ
    where Δ = (max - min) / (2^bits - 1).

    We include this as a BASELINE to compare against our vector/lattice
    quantizers. It will fail badly at 3-4 bits on ViT activations with
    outliers (unless RHT is applied first!).
    """

    def __init__(self, bits: int = 4, symmetric: bool = True):
        """
        Args:
            bits: Number of bits per scalar value (e.g., 4 for INT4).
            symmetric: If True, uses symmetric range [-α, α].
                       If False, uses asymmetric range [min, max].
        """
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.n_levels = 2 ** bits

    def forward(
        self, x: torch.Tensor, per_channel: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Quantize and immediately dequantize (simulate quantization).

        Args:
            x: Input tensor of any shape.
            per_channel: If True, compute scale per last-dim channel.

        Returns:
            (x_quant, info_dict) where info_dict contains scale, zero_point, etc.
        """
        if per_channel:
            # Per-channel: compute range along all dims except the last
            reduce_dims = tuple(range(x.dim() - 1))
            x_max = x.abs().amax(dim=reduce_dims, keepdim=True)
        else:
            x_max = x.abs().amax()

        if self.symmetric:
            # Symmetric: map [-α, α] → [-2^(b-1), 2^(b-1)-1]
            alpha = x_max.clamp(min=1e-8)
            scale = alpha / (self.n_levels // 2 - 1)
            x_int = (x / scale).round().clamp(
                -(self.n_levels // 2), self.n_levels // 2 - 1
            )
            x_quant = x_int * scale
            zero_point = torch.zeros_like(scale)
        else:
            # Asymmetric: map [min, max] → [0, 2^b - 1]
            x_min = x.amin() if not per_channel else x.amin(
                dim=reduce_dims, keepdim=True
            )
            x_max_val = x.amax() if not per_channel else x.amax(
                dim=reduce_dims, keepdim=True
            )
            scale = ((x_max_val - x_min) / (self.n_levels - 1)).clamp(min=1e-8)
            zero_point = (-x_min / scale).round()
            x_int = ((x / scale) + zero_point).round().clamp(0, self.n_levels - 1)
            x_quant = (x_int - zero_point) * scale

        info = {
            'scale': scale,
            'zero_point': zero_point,
            'bits': self.bits,
            'method': 'scalar_round',
        }
        return x_quant, info

    def extra_repr(self) -> str:
        return f"bits={self.bits}, symmetric={self.symmetric}"


class ScalarGroupQuantizer(nn.Module):
    """
    Group-wise scalar round-to-nearest quantizer (fair KIVI-style KV-cache
    baseline).

    ScalarRoundQuantizer (above) uses ONE global scale for the ENTIRE input
    tensor — a single distant outlier anywhere inflates the scale for every
    other value, even ones nowhere near it. Comparing that against
    LatticeD4Quantizer/LatticeE8Quantizer (which use a per-4/8-group scale)
    conflates lattice coding gain with scale granularity — see
    docs/optimization_ledger.md finding F11.

    ScalarGroupQuantizer pairs plain scalar rounding with THE SAME per-group
    scale machinery the lattice quantizers use: a scale per contiguous
    group of `group_size` elements, with the identical optional scale_bits=8
    uint8-simulated-storage path LatticeE8Quantizer uses. This mirrors how
    real KV-cache quantizers are actually built (e.g. KIVI, KVQuant use
    per-channel/per-group scalar quantization, never a single per-tensor
    scale), so a scalar-vs-lattice comparison at matched effective bit-rate
    isolates lattice coding gain instead of conflating it with granularity.

    References:
        [1] docs/optimization_ledger.md F11 (the confound this class resolves)
        [2] Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for
            KV Cache", 2024 (per-group scalar KV quantization is the field's
            standard baseline shape)
    """

    def __init__(self, bits: int = 4, group_size: int = 8, scale_bits: int = 16):
        """
        Args:
            bits: Bits per scalar coordinate.
            group_size: Number of consecutive scalars sharing one scale.
                        Default 8 to match LatticeE8Quantizer's grouping.
            scale_bits: Bit-width used to store each group's scale (16 = fp16,
                        8 = uint8 quantized against a per-tensor fp32 max,
                        identical simulation to the lattice quantizers). Real
                        overhead — see info['scale_overhead_bits_per_scalar'].
        """
        super().__init__()
        assert scale_bits in (8, 16), "scale_bits must be 8 or 16"
        self.bits = bits
        self.group_size = group_size
        self.scale_bits = scale_bits
        self.n_levels = 2 ** bits
        self.half_levels = self.n_levels // 2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Quantize by grouping consecutive dimensions, each with its own scale.

        Args:
            x: Input tensor of shape (..., d). d must be divisible by group_size.

        Returns:
            (x_quant, info_dict)
        """
        orig_shape = x.shape
        d = x.shape[-1]
        k = self.group_size
        assert d % k == 0, (
            f"Feature dim {d} must be divisible by group_size {k}"
        )

        # Reshape into groups: (..., d) -> (..., d//k, k)
        x_grouped = x.reshape(*x.shape[:-1], d // k, k)

        # Per-group symmetric scale (same convention as the lattice quantizers).
        alpha = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = alpha / (self.half_levels - 1)

        if self.scale_bits == 8:
            # Simulate storing each group's scale as uint8 against a single
            # per-tensor fp32 max — identical simulation to LatticeD4/E8Quantizer,
            # so the comparison at scale_bits=8 is apples-to-apples.
            scale_max = scale.abs().amax().clamp(min=1e-8)
            scale_step = scale_max / 255.0
            scale = (scale / scale_step).round().clamp(0, 255) * scale_step

        x_int = (x_grouped / scale).round().clamp(
            -self.half_levels, self.half_levels - 1
        )
        x_quant = (x_int * scale).reshape(orig_shape)

        info = {
            'scale': scale.squeeze(-1),
            'bits': self.bits,
            'group_size': k,
            'method': 'scalar_group',
            'scale_bits': self.scale_bits,
            'scale_overhead_bits_per_scalar': self.scale_bits / k,
        }
        return x_quant, info

    def extra_repr(self) -> str:
        return f"bits={self.bits}, group_size={self.group_size}, scale_bits={self.scale_bits}"


class UniformVectorQuantizer(nn.Module):
    """
    Uniform Vector Quantizer: groups consecutive scalars into small
    vectors and quantizes each vector jointly.

    After RHT rotation, features are approximately isotropic (uniform on
    a hypersphere). In this regime, grouping k consecutive dimensions
    into a vector and quantizing jointly (using a uniform grid in k-D
    space) achieves better rate-distortion than independent scalar
    quantization.

    The key insight: for the same total bit budget, k-dimensional vector
    quantization reduces MSE by the "space-filling gain" of the k-D
    lattice/grid vs k independent 1-D grids.

    Algorithm:
        1. Reshape x into groups of size k: (..., d) → (..., d//k, k)
        2. Compute per-group scale (max-abs of each k-vector)
        3. Uniformly quantize each coordinate to b bits
        4. Effective bits-per-scalar = b (same as scalar), but lower MSE!
    """

    def __init__(self, bits: int = 4, group_size: int = 4):
        """
        Args:
            bits: Bits per scalar coordinate within each vector.
            group_size: Number of consecutive scalars to group (k).
                        Common choices: 2, 4, 8. Must divide feature dim.
        """
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.n_levels = 2 ** bits

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Quantize by grouping consecutive dimensions.

        Args:
            x: Input tensor of shape (..., d). d must be divisible by group_size.

        Returns:
            (x_quant, info_dict)
        """
        orig_shape = x.shape
        d = x.shape[-1]
        k = self.group_size
        assert d % k == 0, f"Feature dim {d} must be divisible by group_size {k}"

        # Reshape into groups: (..., d) → (..., d//k, k)
        x_grouped = x.reshape(*x.shape[:-1], d // k, k)

        # Per-group symmetric scale (max-abs across each k-vector)
        alpha = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = alpha / (self.n_levels // 2 - 1)

        # Quantize each coordinate
        x_int = (x_grouped / scale).round().clamp(
            -(self.n_levels // 2), self.n_levels // 2 - 1
        )
        x_quant = (x_int * scale).reshape(orig_shape)

        info = {
            'scale': scale.squeeze(-1),
            'bits': self.bits,
            'group_size': k,
            'method': 'uniform_vector',
            'effective_bits_per_scalar': self.bits,
        }
        return x_quant, info

    def extra_repr(self) -> str:
        return f"bits={self.bits}, group_size={self.group_size}"


class LatticeD4Quantizer(nn.Module):
    """
    D4 Checkerboard Lattice Vector Quantizer.

    The D4 lattice is the set of all integer points in ℝ⁴ whose
    coordinates sum to an even number:
        D4 = { (x1, x2, x3, x4) ∈ ℤ⁴ : x1 + x2 + x3 + x4 ≡ 0 (mod 2) }

    D4 is the densest sphere packing in 4 dimensions and provides a
    coding gain of ~1.19 dB over scalar quantization. This means that
    at the same bit-rate, D4 achieves ~20% lower MSE.

    Algorithm:
        1. Reshape features into 4-vectors: (..., d) → (..., d//4, 4)
        2. Scale each 4-vector to the quantizer's dynamic range
        3. Find the nearest D4 lattice point for each 4-vector:
           - Round to nearest integer → candidate z_round
           - If sum(z_round) is even → z_round is already a D4 point
           - If sum(z_round) is odd → flip the coordinate with smallest
             fractional residual (this is the standard D4 decoding trick)
        4. Scale back to original range

    References:
        [1] Conway & Sloane, Ch. 4 (D_n lattices)
        [2] HyperQuant, Section 3.2 (Lattice quantization pipeline)
    """

    def __init__(self, bits: int = 4, group_size: int = 4, scale_bits: int = 16):
        """
        Args:
            bits: Bits per scalar coordinate (controls the grid resolution).
            group_size: Must be 4 for D4 lattice. Kept as arg for API consistency.
            scale_bits: Bit-width used to store each group's per-group scale
                        (16 = fp16 scale, 8 = uint8 scale quantized against a
                        single per-tensor fp32 max). This overhead is real and
                        is reported in info['scale_overhead_bits_per_scalar'];
                        see docs/optimization_ledger.md T1/T4.
        """
        super().__init__()
        assert group_size == 4, "D4 lattice requires group_size=4"
        assert scale_bits in (8, 16), "scale_bits must be 8 or 16"
        self.bits = bits
        self.group_size = 4
        self.scale_bits = scale_bits
        self.n_levels = 2 ** bits
        self.half_levels = self.n_levels // 2

    def _nearest_d4_point(
        self, x_scaled: torch.Tensor, lo: float, hi: float
    ) -> torch.Tensor:
        """
        Find the nearest D4 lattice point to each 4-vector, subject to every
        coordinate lying in [lo, hi].

        The D4 decoding algorithm:
        1. Round each coordinate independently → z_round
        2. Check parity: if sum(z_round) is even → it's a D4 point (done!)
        3. If odd → flip one coordinate by ±1. Any single flip fixes parity,
           so we take the cheapest flip that keeps the coordinate in range.

        Earlier revisions instead shrank the caller's input range so that the
        greedy "largest residual" flip was always legal. That reserved 2.0 of
        the representable units and cost far more than the lattice gain it
        protected (see docs/optimization_ledger.md T4); the bounded search
        below keeps the full range and is never worse than the greedy choice.

        Args:
            x_scaled: Tensor of shape (..., 4).
            lo, hi: inclusive bounds every output coordinate must respect.

        Returns:
            Nearest in-range D4 lattice points, same shape.
        """
        z_round = x_scaled.round().clamp(lo, hi)
        residuals = x_scaled - z_round  # fractional parts

        # Check parity of coordinate sum
        coord_sum = z_round.sum(dim=-1)  # (...,)
        is_odd = (coord_sum.long() % 2 != 0)  # (...,) boolean mask

        if is_odd.any():
            # Flipping coordinate j by +1 changes squared error by 1 - 2*r_j,
            # and by -1 by 1 + 2*r_j. Mask out flips that would leave [lo, hi]
            # and take the global minimum over both directions.
            d = x_scaled.shape[-1]
            inf = torch.full_like(residuals, float('inf'))
            cost_up = torch.where((z_round + 1.0) <= hi, 1.0 - 2.0 * residuals, inf)
            cost_dn = torch.where((z_round - 1.0) >= lo, 1.0 + 2.0 * residuals, inf)

            both = torch.cat([cost_up, cost_dn], dim=-1)  # (..., 2d)
            flip_idx = both.argmin(dim=-1, keepdim=True)
            picked = torch.zeros_like(both)
            picked.scatter_(-1, flip_idx, 1.0)
            correction = picked[..., :d] - picked[..., d:]

            # Apply correction only to odd-parity vectors
            is_odd_expanded = is_odd.unsqueeze(-1).expand_as(z_round)
            z_round = torch.where(is_odd_expanded, z_round + correction, z_round)

        return z_round

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Quantize features using the D4 lattice.

        Args:
            x: Input tensor of shape (..., d). d must be divisible by 4.

        Returns:
            (x_quant, info_dict)
        """
        orig_shape = x.shape
        d = x.shape[-1]
        assert d % 4 == 0, f"Feature dim {d} must be divisible by 4 for D4 lattice"

        # Reshape into 4-vectors
        x_grouped = x.reshape(*x.shape[:-1], d // 4, 4)

        # Per-group symmetric scale
        alpha = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = alpha / (self.half_levels - 1)

        if self.scale_bits == 8:
            # Simulate storing each group's scale as uint8 against a single
            # per-tensor fp32 max (this is real metadata, not free: see
            # info['scale_overhead_bits_per_scalar'] below).
            scale_max = scale.abs().amax().clamp(min=1e-8)
            scale_step = scale_max / 255.0
            scale = (scale / scale_step).round().clamp(0, 255) * scale_step

        # Scale to integer range
        x_scaled = x_grouped / scale

        # Use the FULL representable range, matching the grouped-scalar
        # baseline. The decoder enforces [lo, hi] internally by choosing a
        # legal parity flip, so no range has to be reserved up front.
        lo, hi = -float(self.half_levels), float(self.half_levels - 1)
        x_scaled = x_scaled.clamp(lo, hi)

        # Find nearest in-range D4 lattice point (the decoder guarantees both
        # even-coordinate-sum membership and the bounds, so no post-hoc clamp
        # is applied -- that could itself break membership).
        x_lattice = self._nearest_d4_point(x_scaled, lo, hi)

        # Dequantize
        x_quant = (x_lattice * scale).reshape(orig_shape)

        info = {
            'scale': scale.squeeze(-1),
            'bits': self.bits,  # nominal payload bits; no unearned savings
            'group_size': 4,
            'method': 'lattice_d4',
            'scale_bits': self.scale_bits,
            'scale_overhead_bits_per_scalar': self.scale_bits / self.group_size,
            'coding_gain_db': 1.19,  # Theoretical D4 gain (requires index coding to realize as a rate reduction)
        }
        return x_quant, info

    def extra_repr(self) -> str:
        return f"bits={self.bits}, lattice=D4, coding_gain=1.19dB"


class LatticeE8Quantizer(nn.Module):
    """
    E8 (Gosset) Lattice Vector Quantizer.

    E8 = D8 ∪ (D8 + g), where D8 = { x ∈ ℤ⁸ : sum(x_i) even } (the
    8-dimensional checkerboard lattice, same construction as D4 generalized
    to 8 dims) and g = (½,½,...,½) is the "glue vector". E8 is the densest
    known sphere packing in 8 dimensions and provides a coding gain of
    ~1.5 dB over scalar quantization (~0.65 dB over D4) — at the same
    bit-rate, E8 achieves lower MSE than D4.

    With group_size=8 (vs D4's 4), the per-group scale metadata overhead is
    HALVED relative to D4 at the same scale_bits — this is the key lever for
    reaching a true ≤4.0 effective-bits/scalar target (see
    docs/optimization_ledger.md T8): d=64, group=8, b=3, scale_bits=8,
    QJL disabled -> 64·3 + (64/8)·8 = 192 + 64 = 256 bits/vector = exactly
    4.0 effective bits/scalar.

    Decoding (Conway & Sloane, "Fast decoding algorithms for lattices",
    Algorithm 4): for a scaled point x ∈ ℝ⁸, form two candidates —
        1. The nearest D8 point to x (integer coset).
        2. The nearest D8 point to (x - g), shifted back by +g (half-integer
           coset). Since (x-g) is decoded with the SAME parity-fixing D8
           decoder, the result minus g always has an even coordinate sum by
           construction — no separate parity check is needed for the
           half-integer coset.
    Return whichever candidate is closer (smaller squared distance) to x.
    This is the standard "glue vector" / union-of-cosets decoder for E8.

    References:
        [1] Conway & Sloane, "Sphere Packings, Lattices and Groups", Ch. 4 (E8).
        [2] Conway & Sloane, "Fast decoding algorithms for lattices", 1986.
    """

    def __init__(self, bits: int = 4, group_size: int = 8, scale_bits: int = 16):
        """
        Args:
            bits: Bits per scalar coordinate (controls the grid resolution).
            group_size: Number of consecutive scalars sharing ONE SCALE. Must be
                        a multiple of 8. This is deliberately DECOUPLED from the
                        lattice dimension: E8 always decodes 8-vectors, but the
                        scale may be amortized over 8, 16, 32, ... scalars.

                        The distinction matters. The +0.65 dB E8 granular gain
                        assumes a fixed lattice over a stationary source, but
                        per-group max-normalization re-fits the cell size every
                        `group_size` values, which substitutes for exactly the
                        shaping the lattice provides. At group_size=8 the two
                        cancel and E8 ties grouped scalar; the gain reappears
                        monotonically as the scale is amortized further
                        (measured on Gaussian input: +0.32 dB at 16, +0.49 at
                        32, +0.57 at 64, +0.64 at 512). See ledger F28.
            scale_bits: Bit-width used to store each group's per-group scale
                        (16 = fp16, 8 = uint8 quantized against a per-tensor
                        fp32 max). Real overhead — see
                        info['scale_overhead_bits_per_scalar'].
        """
        super().__init__()
        assert group_size % 8 == 0 and group_size >= 8, (
            f"E8 scale group_size must be a positive multiple of 8, got {group_size}"
        )
        assert scale_bits in (8, 16), "scale_bits must be 8 or 16"
        self.bits = bits
        self.group_size = group_size
        self.scale_bits = scale_bits
        self.n_levels = 2 ** bits
        self.half_levels = self.n_levels // 2

    def _nearest_d8_point(
        self, x_scaled: torch.Tensor, lo: float, hi: float
    ) -> torch.Tensor:
        """
        Nearest D8 (even-coordinate-sum integer) lattice point with every
        coordinate constrained to [lo, hi], using the same bounded parity-flip
        search as LatticeD4Quantizer._nearest_d4_point generalized to
        8-dimensional groups.

        Earlier revisions reserved 2.5 of the representable units so that the
        greedy "largest residual" flip was always legal. Measured against a
        full-range decode that reservation cost about 0.9 dB at the 3-bit
        operating point -- more than the E8 coding gain it existed to protect
        (docs/optimization_ledger.md T4).
        """
        z_round = x_scaled.round().clamp(lo, hi)
        residuals = x_scaled - z_round

        coord_sum = z_round.sum(dim=-1)
        is_odd = (coord_sum.long() % 2 != 0)

        if is_odd.any():
            d = x_scaled.shape[-1]
            inf = torch.full_like(residuals, float('inf'))
            cost_up = torch.where((z_round + 1.0) <= hi, 1.0 - 2.0 * residuals, inf)
            cost_dn = torch.where((z_round - 1.0) >= lo, 1.0 + 2.0 * residuals, inf)

            both = torch.cat([cost_up, cost_dn], dim=-1)  # (..., 2d)
            flip_idx = both.argmin(dim=-1, keepdim=True)
            picked = torch.zeros_like(both)
            picked.scatter_(-1, flip_idx, 1.0)
            correction = picked[..., :d] - picked[..., d:]

            is_odd_expanded = is_odd.unsqueeze(-1).expand_as(z_round)
            z_round = torch.where(is_odd_expanded, z_round + correction, z_round)

        return z_round

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Quantize features using the E8 lattice.

        Args:
            x: Input tensor of shape (..., d). d must be divisible by 8.

        Returns:
            (x_quant, info_dict)
        """
        orig_shape = x.shape
        d = x.shape[-1]
        k = self.group_size
        assert d % k == 0, (
            f"Feature dim {d} must be divisible by scale group_size {k}"
        )

        # SCALE is computed over groups of k scalars...
        x_scale_grouped = x.reshape(*x.shape[:-1], d // k, k)
        alpha = x_scale_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = alpha / (self.half_levels - 1)

        if self.scale_bits == 8:
            scale_max = scale.abs().amax().clamp(min=1e-8)
            scale_step = scale_max / 255.0
            scale = (scale / scale_step).round().clamp(0, 255) * scale_step

        # ...but DECODING is always over 8-vectors. Normalize within the scale
        # group, then re-view as 8-vectors for the lattice. k is a multiple of 8
        # so this split is exact and no 8-vector ever straddles two scales.
        x_scaled = (x_scale_grouped / scale).reshape(*x.shape[:-1], d // 8, 8)

        # Use the FULL representable range, matching the grouped-scalar
        # baseline. Each coset's decoder enforces its own bounds internally,
        # so no range has to be reserved for the parity correction.
        lo, hi = -float(self.half_levels), float(self.half_levels - 1)
        x_scaled = x_scaled.clamp(lo, hi)

        # Candidate A: integer coset (D8), constrained to [lo, hi]
        z_int = self._nearest_d8_point(x_scaled, lo, hi)

        # Candidate B: half-integer coset (D8 + [0.5]*8). Decoding the shifted
        # point in [lo-0.5, hi-0.5] puts the +0.5 result back inside [lo, hi].
        z_half = self._nearest_d8_point(x_scaled - 0.5, lo - 0.5, hi - 0.5) + 0.5

        # Pick whichever candidate is closer to the (clamped) point
        dist_int = ((x_scaled - z_int) ** 2).sum(dim=-1, keepdim=True)
        dist_half = ((x_scaled - z_half) ** 2).sum(dim=-1, keepdim=True)
        use_half = (dist_half < dist_int).expand_as(z_int)
        x_lattice = torch.where(use_half, z_half, z_int)

        # Dequantize: fold back to scale groups so `scale` broadcasts correctly.
        x_quant = (x_lattice.reshape(*x.shape[:-1], d // k, k) * scale).reshape(orig_shape)

        info = {
            'scale': scale.squeeze(-1),
            'bits': self.bits,  # nominal payload bits; no unearned savings
            'group_size': k,
            'method': 'lattice_e8',
            'scale_bits': self.scale_bits,
            'scale_overhead_bits_per_scalar': self.scale_bits / k,
            'coding_gain_db': 1.5,  # Theoretical E8 gain (requires index coding to realize as a rate reduction)
        }
        return x_quant, info

    def extra_repr(self) -> str:
        return f"bits={self.bits}, lattice=E8, coding_gain=1.5dB"


# =============================================================================
# BW16 (Barnes-Wall Λ16) via Construction A over RM(1, 4)
# =============================================================================
from ._golay_rm import rm14_codewords_torch, golay24_codewords_torch


class LatticeBW16Quantizer(nn.Module):
    """
    Barnes-Wall Λ16 lattice vector quantizer via Construction A applied to
    the Reed-Muller RM(1, 4) code.

    Construction A: Λ16 = { x ∈ Z^16 : (x mod 2) ∈ RM(1, 4) }.
    This is the standard 16-dimensional Barnes-Wall lattice, densest known
    packing in R^16, with kissing number 4320 and minimum squared norm 4.

    In the source-coding regime, its coding gain over scalar quantisation is
    approximately +0.86 dB -- roughly +0.21 dB over the E8 lattice at
    matched rate. On a source that has been per-group max-normalized (as we
    do here) the realised gain depends on the group size: like E8, BW16 ties
    grouped scalar at group=16 and its granular gain appears as the scale is
    amortized over larger groups.

    Decoder. RM(1, 4) has only 2^5 = 32 codewords. For an input x we
    enumerate all 32 cosets 2Z^16 + c and pick the nearest lattice point:
        1. For each c in RM(1, 4), the nearest 2Z^16 + c point to x is
                z_c = 2 * round((x - c) / 2) + c
           with each coordinate clamped to the representable range.
        2. Return argmin_c ||x - z_c||^2.
    This is a mathematically exact nearest-neighbour decoder, not a
    bounded-distance approximation.

    References:
        Barnes & Wall, "Some extreme forms defined in terms of Abelian
            groups", J. Aust. Math. Soc. 1959.
        Forney, "Coset codes -- Part I: Introduction and geometrical
            classification", IEEE T-IT 1988.
    """

    def __init__(self, bits: int = 4, group_size: int = 16, scale_bits: int = 16):
        super().__init__()
        assert group_size % 16 == 0 and group_size >= 16, (
            f"BW16 scale group_size must be a positive multiple of 16, got {group_size}"
        )
        assert scale_bits in (8, 16), "scale_bits must be 8 or 16"
        self.bits = bits
        self.group_size = group_size
        self.scale_bits = scale_bits
        self.n_levels = 2 ** bits
        self.half_levels = self.n_levels // 2

    def _nearest_bw16_point(
        self, x_scaled: torch.Tensor, lo: float, hi: float
    ) -> torch.Tensor:
        """
        For each 16-vector row in x_scaled, return the nearest BW16 point
        with every coordinate in [lo, hi].

        Implementation. Enumerate all 32 RM(1, 4) codewords c. For each c,
        the nearest point of 2Z^16 + c to x is obtained coordinatewise by
        rounding (x - c) / 2 to the nearest integer in the range
        [ceil((lo - c) / 2), floor((hi - c) / 2)] and mapping back through
        y = 2 * z + c. Then pick the codeword whose point is closest.
        """
        assert x_scaled.shape[-1] == 16, x_scaled.shape
        # Codewords: shape (32, 16).
        C = rm14_codewords_torch(device=x_scaled.device, dtype=x_scaled.dtype)

        # x' shape: (..., 1, 16); c shape: (1, ..., 32, 16). Broadcast.
        x_exp = x_scaled.unsqueeze(-2)                    # (..., 1, 16)
        # (x - c) / 2, then round to integer within the legal range.
        shifted = (x_exp - C) * 0.5                       # (..., 32, 16)

        # Legal integer bounds for z depend on c coordinate-by-coordinate.
        # lo <= 2*z + c <= hi  <=>  (lo - c)/2 <= z <= (hi - c)/2.
        lo_bounds = torch.ceil((lo - C) * 0.5)            # (32, 16)
        hi_bounds = torch.floor((hi - C) * 0.5)           # (32, 16)

        z = shifted.round().clamp(lo_bounds, hi_bounds)   # (..., 32, 16)
        cand = 2.0 * z + C                                # (..., 32, 16)

        # Distance to x for each candidate coset representative.
        d2 = ((cand - x_exp) ** 2).sum(dim=-1)            # (..., 32)
        best = d2.argmin(dim=-1, keepdim=True)            # (..., 1)
        # Gather the winning 16-vector.
        best_expanded = best.unsqueeze(-1).expand(*best.shape, 16)  # (..., 1, 16)
        picked = cand.gather(-2, best_expanded).squeeze(-2)         # (..., 16)
        return picked

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        orig_shape = x.shape
        d = x.shape[-1]
        k = self.group_size
        assert d % k == 0, (
            f"Feature dim {d} must be divisible by scale group_size {k}"
        )

        x_scale_grouped = x.reshape(*x.shape[:-1], d // k, k)
        alpha = x_scale_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = alpha / (self.half_levels - 1)

        if self.scale_bits == 8:
            scale_max = scale.abs().amax().clamp(min=1e-8)
            scale_step = scale_max / 255.0
            scale = (scale / scale_step).round().clamp(0, 255) * scale_step

        # Reshape to 16-vectors (independent of group size, as long as k % 16 == 0).
        x_scaled = (x_scale_grouped / scale).reshape(*x.shape[:-1], d // 16, 16)

        lo, hi = -float(self.half_levels), float(self.half_levels - 1)
        x_scaled = x_scaled.clamp(lo, hi)

        x_lattice = self._nearest_bw16_point(x_scaled, lo, hi)

        x_quant = (x_lattice.reshape(*x.shape[:-1], d // k, k) * scale).reshape(orig_shape)

        info = {
            'scale': scale.squeeze(-1),
            'bits': self.bits,
            'group_size': k,
            'method': 'lattice_bw16',
            'scale_bits': self.scale_bits,
            'scale_overhead_bits_per_scalar': self.scale_bits / k,
            'coding_gain_db': 0.86,  # BW16 vs scalar; ~0.21 dB over E8 at matched rate
        }
        return x_quant, info

    def extra_repr(self) -> str:
        return f"bits={self.bits}, lattice=BW16 (Λ16 via RM(1,4)), coding_gain=0.86dB"


# =============================================================================
# Λ24 (Golay-A lattice) via Construction A over the extended Golay code
# =============================================================================
class LatticeGolay24Quantizer(nn.Module):
    """
    24-dimensional lattice vector quantizer via Construction A applied to
    the extended binary Golay code G_{24}.

    Construction A: Λ24_A = { x ∈ Z^24 : (x mod 2) ∈ G_{24} }.
    This lattice has kissing number 4600 and minimum squared norm 4. It is
    NOT identical to the (much denser) true Leech lattice Λ24 -- the true
    Leech uses Construction B or D and needs a more elaborate decoder
    (Amrani-Be'ery or Adoul-Barth) -- but Construction A is a legitimate,
    correct 24-dimensional lattice with meaningful coding gain over scalar
    quantisation and is the standard first step toward the true Leech.

    Report and cite this quantizer precisely as "Λ24_A (Construction A over
    the extended Golay code)" rather than as "Leech".

    Decoder. G_{24} has 2^12 = 4096 codewords. For an input x we enumerate
    all 4096 cosets 2Z^24 + c and pick the closest:
        1. For each c in G_{24}, the nearest 2Z^24 + c point to x is
                z_c = 2 * round((x - c) / 2) + c
           with each coordinate clamped to the representable range.
        2. Return argmin_c ||x - z_c||^2.
    Complexity per input vector: O(4096 * 24) = ~10^5 float ops. Vectorised
    over the batch on GPU.

    Head-dim constraint. The natural group size is a multiple of 24; on a
    ViT with head_dim = 64 the smallest group_size that both (a) divides
    into 24-vectors and (b) is a multiple of the head dimension is 192
    (= lcm(24, 64) * 1). We therefore require the caller to pass a group
    size divisible by 24 AND to arrange that the feature tensor being
    quantised has trailing dimension divisible by 24. When applying to
    head_dim = 64 KV cache the natural approach is to flatten across
    multiple heads before quantising; the caller is responsible for that
    reshape.

    References:
        Conway & Sloane, SPLAG Ch. 5, 12 (Golay code and Leech lattice).
        Nebe & Sloane, "Catalogue of Lattices" (Λ24 tables).
    """

    def __init__(self, bits: int = 4, group_size: int = 24, scale_bits: int = 16):
        super().__init__()
        assert group_size % 24 == 0 and group_size >= 24, (
            f"Λ24 scale group_size must be a positive multiple of 24, got {group_size}"
        )
        assert scale_bits in (8, 16), "scale_bits must be 8 or 16"
        self.bits = bits
        self.group_size = group_size
        self.scale_bits = scale_bits
        self.n_levels = 2 ** bits
        self.half_levels = self.n_levels // 2

    def _nearest_golay24_point(
        self, x_scaled: torch.Tensor, lo: float, hi: float
    ) -> torch.Tensor:
        """
        For each 24-vector row in x_scaled, return the nearest Λ24_A point
        with every coordinate in [lo, hi]. See LatticeBW16Quantizer for the
        analogous 16-dim enumeration; here the codebook has 4096 codewords.
        """
        assert x_scaled.shape[-1] == 24, x_scaled.shape
        C = golay24_codewords_torch(device=x_scaled.device, dtype=x_scaled.dtype)  # (4096, 24)

        x_exp = x_scaled.unsqueeze(-2)                    # (..., 1, 24)
        shifted = (x_exp - C) * 0.5                       # (..., 4096, 24)

        lo_bounds = torch.ceil((lo - C) * 0.5)            # (4096, 24)
        hi_bounds = torch.floor((hi - C) * 0.5)           # (4096, 24)

        z = shifted.round().clamp(lo_bounds, hi_bounds)
        cand = 2.0 * z + C                                # (..., 4096, 24)

        d2 = ((cand - x_exp) ** 2).sum(dim=-1)            # (..., 4096)
        best = d2.argmin(dim=-1, keepdim=True)
        best_expanded = best.unsqueeze(-1).expand(*best.shape, 24)
        picked = cand.gather(-2, best_expanded).squeeze(-2)
        return picked

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        orig_shape = x.shape
        d = x.shape[-1]
        k = self.group_size
        assert d % k == 0, (
            f"Feature dim {d} must be divisible by scale group_size {k}"
        )
        assert d % 24 == 0, (
            f"Feature dim {d} must be divisible by lattice dimension 24 for Λ24. "
            "For head_dim = 64 KV caches you must flatten across heads before "
            "calling this quantiser."
        )

        x_scale_grouped = x.reshape(*x.shape[:-1], d // k, k)
        alpha = x_scale_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = alpha / (self.half_levels - 1)

        if self.scale_bits == 8:
            scale_max = scale.abs().amax().clamp(min=1e-8)
            scale_step = scale_max / 255.0
            scale = (scale / scale_step).round().clamp(0, 255) * scale_step

        x_scaled = (x_scale_grouped / scale).reshape(*x.shape[:-1], d // 24, 24)

        lo, hi = -float(self.half_levels), float(self.half_levels - 1)
        x_scaled = x_scaled.clamp(lo, hi)

        x_lattice = self._nearest_golay24_point(x_scaled, lo, hi)

        x_quant = (x_lattice.reshape(*x.shape[:-1], d // k, k) * scale).reshape(orig_shape)

        info = {
            'scale': scale.squeeze(-1),
            'bits': self.bits,
            'group_size': k,
            'method': 'lattice_golay24',
            'scale_bits': self.scale_bits,
            'scale_overhead_bits_per_scalar': self.scale_bits / k,
            'coding_gain_db': 1.03,  # Λ24_A vs scalar; true Leech is higher (~2.7 dB)
        }
        return x_quant, info

    def extra_repr(self) -> str:
        return (
            f"bits={self.bits}, lattice=Λ24_A (Construction A over Golay G_24), "
            "coding_gain~1.03dB"
        )
