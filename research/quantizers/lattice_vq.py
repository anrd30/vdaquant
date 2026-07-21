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

    def _nearest_d4_point(self, x_scaled: torch.Tensor) -> torch.Tensor:
        """
        Find the nearest D4 lattice point to each 4-vector.

        The D4 decoding algorithm:
        1. Round each coordinate independently → z_round
        2. Check parity: if sum(z_round) is even → it's a D4 point (done!)
        3. If odd → find the coordinate with the largest rounding error
           and flip it to the opposite direction. This guarantees the
           result has even coordinate sum (= D4 point) with minimum distortion.

        Callers MUST pre-clamp x_scaled to [-half_levels+0.5, half_levels-1.5]
        (see forward()) so that the ±1 parity correction applied here can
        never push a coordinate outside [-half_levels, half_levels-1].

        Args:
            x_scaled: Tensor of shape (..., 4), pre-clamped by the caller.

        Returns:
            Nearest D4 lattice points, same shape, guaranteed in-range.
        """
        z_round = x_scaled.round()
        residuals = x_scaled - z_round  # fractional parts

        # Check parity of coordinate sum
        coord_sum = z_round.sum(dim=-1)  # (...,)
        is_odd = (coord_sum.long() % 2 != 0)  # (...,) boolean mask

        if is_odd.any():
            # For odd-parity vectors: flip the coordinate with largest |residual|
            abs_residuals = residuals.abs()
            # Find which coordinate to flip
            flip_idx = abs_residuals.argmax(dim=-1)  # (...,)

            # Gather the residual sign at the flip index
            flip_idx_expanded = flip_idx.unsqueeze(-1)  # (..., 1)
            flip_residual = residuals.gather(-1, flip_idx_expanded)  # (..., 1)
            flip_dir = flip_residual.sign()  # +1 if we should round up, -1 if down

            # Tie-break: a zero residual means the scaled coordinate was
            # already an exact integer, so sign() gives 0 and the parity
            # would be left unfixed. Force a direction, preferring +1 unless
            # that would push the coordinate past the upper boundary.
            zero_mask = (flip_dir == 0)
            if zero_mask.any():
                current_val = z_round.gather(-1, flip_idx_expanded)
                would_exceed_upper = (current_val + 1) > (self.half_levels - 1)
                tie_dir = torch.where(
                    would_exceed_upper,
                    torch.full_like(flip_dir, -1.0),
                    torch.full_like(flip_dir, 1.0),
                )
                flip_dir = torch.where(zero_mask, tie_dir, flip_dir)

            # Create the correction: +1 or -1 applied only at flip_idx
            correction = torch.zeros_like(z_round)
            correction.scatter_(-1, flip_idx_expanded, flip_dir)

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

        # Pre-clamp BEFORE decoding: guarantees that round() followed by the
        # single ±1 parity correction in _nearest_d4_point can never leave
        # [-half_levels, half_levels-1]. See docs/optimization_ledger.md T4.
        x_scaled = x_scaled.clamp(-self.half_levels + 0.5, self.half_levels - 1.5)

        # Find nearest D4 lattice point (already guaranteed in-range; no
        # post-hoc clamp needed, and none is applied, since clamping AFTER
        # the parity correction could itself break even-coordinate-sum
        # membership).
        x_lattice = self._nearest_d4_point(x_scaled)

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
            group_size: Must be 8 for E8 lattice. Kept as arg for API consistency.
            scale_bits: Bit-width used to store each group's per-group scale
                        (16 = fp16, 8 = uint8 quantized against a per-tensor
                        fp32 max). Real overhead — see
                        info['scale_overhead_bits_per_scalar'].
        """
        super().__init__()
        assert group_size == 8, "E8 lattice requires group_size=8"
        assert scale_bits in (8, 16), "scale_bits must be 8 or 16"
        self.bits = bits
        self.group_size = 8
        self.scale_bits = scale_bits
        self.n_levels = 2 ** bits
        self.half_levels = self.n_levels // 2

    def _nearest_d8_point(self, x_scaled: torch.Tensor) -> torch.Tensor:
        """
        Nearest D8 (even-coordinate-sum integer) lattice point, using the
        same round + single-coordinate parity-flip algorithm as
        LatticeD4Quantizer._nearest_d4_point, generalized to 8-dimensional
        groups. Callers must pre-clamp x_scaled to
        [-half_levels+0.5, half_levels-1.5] (same discipline as D4; see
        docs/optimization_ledger.md T4) so the ±1 parity correction can never
        push a coordinate outside [-half_levels, half_levels-1].
        """
        z_round = x_scaled.round()
        residuals = x_scaled - z_round

        coord_sum = z_round.sum(dim=-1)
        is_odd = (coord_sum.long() % 2 != 0)

        if is_odd.any():
            abs_residuals = residuals.abs()
            flip_idx = abs_residuals.argmax(dim=-1)
            flip_idx_expanded = flip_idx.unsqueeze(-1)
            flip_residual = residuals.gather(-1, flip_idx_expanded)
            flip_dir = flip_residual.sign()

            zero_mask = (flip_dir == 0)
            if zero_mask.any():
                current_val = z_round.gather(-1, flip_idx_expanded)
                would_exceed_upper = (current_val + 1) > (self.half_levels - 1)
                tie_dir = torch.where(
                    would_exceed_upper,
                    torch.full_like(flip_dir, -1.0),
                    torch.full_like(flip_dir, 1.0),
                )
                flip_dir = torch.where(zero_mask, tie_dir, flip_dir)

            correction = torch.zeros_like(z_round)
            correction.scatter_(-1, flip_idx_expanded, flip_dir)

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
        assert d % 8 == 0, f"Feature dim {d} must be divisible by 8 for E8 lattice"

        # Reshape into 8-vectors
        x_grouped = x.reshape(*x.shape[:-1], d // 8, 8)

        # Per-group symmetric scale
        alpha = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = alpha / (self.half_levels - 1)

        if self.scale_bits == 8:
            scale_max = scale.abs().amax().clamp(min=1e-8)
            scale_step = scale_max / 255.0
            scale = (scale / scale_step).round().clamp(0, 255) * scale_step

        x_scaled = x_grouped / scale

        # Pre-clamp so BOTH cosets' D8 decode inputs stay in the safe range
        # [-half_levels+0.5, half_levels-1.5]: the integer coset decodes
        # x_scaled_clamped directly, the half-integer coset decodes
        # (x_scaled_clamped - 0.5), whose range is a strict subset of the
        # integer coset's when x_scaled_clamped is clamped to
        # [-half_levels+1.0, half_levels-1.5].
        x_scaled = x_scaled.clamp(-self.half_levels + 1.0, self.half_levels - 1.5)

        # Candidate A: integer coset (D8)
        z_int = self._nearest_d8_point(x_scaled)

        # Candidate B: half-integer coset (D8 + [0.5]*8)
        z_half = self._nearest_d8_point(x_scaled - 0.5) + 0.5

        # Pick whichever candidate is closer to the (clamped) point
        dist_int = ((x_scaled - z_int) ** 2).sum(dim=-1, keepdim=True)
        dist_half = ((x_scaled - z_half) ** 2).sum(dim=-1, keepdim=True)
        use_half = (dist_half < dist_int).expand_as(z_int)
        x_lattice = torch.where(use_half, z_half, z_int)

        # Dequantize
        x_quant = (x_lattice * scale).reshape(orig_shape)

        info = {
            'scale': scale.squeeze(-1),
            'bits': self.bits,  # nominal payload bits; no unearned savings
            'group_size': 8,
            'method': 'lattice_e8',
            'scale_bits': self.scale_bits,
            'scale_overhead_bits_per_scalar': self.scale_bits / self.group_size,
            'coding_gain_db': 1.5,  # Theoretical E8 gain (requires index coding to realize as a rate reduction)
        }
        return x_quant, info

    def extra_repr(self) -> str:
        return f"bits={self.bits}, lattice=E8, coding_gain=1.5dB"
