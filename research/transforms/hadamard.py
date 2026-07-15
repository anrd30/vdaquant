"""
Fast Walsh–Hadamard Transform (FWHT) and Randomized Hadamard Transform (RHT).

The RHT is the mathematical foundation of TurboQuant / HyperQuant.
It transforms any high-dimensional vector into a near-uniform spherical
distribution, eliminating the extreme activation outliers that destroy
standard scalar quantization in Vision Transformers.

Mathematical Background:
    Given a vector x ∈ ℝ^d, the RHT computes:
        x' = (1/√d) · H_d · D · x
    where:
        H_d = Hadamard matrix (recursive Kronecker structure, entries ±1)
        D   = random diagonal sign-flip matrix (entries ±1, drawn once)

    Key Property: After RHT, the entries of x' are approximately i.i.d.
    Gaussian with variance ||x||² / d, regardless of the original
    distribution of x. This makes the rotated vector ideal for uniform
    or lattice quantization.

    Computational Cost: O(d log d) using the butterfly decomposition,
    with ONLY additions and subtractions (zero multiplications needed
    for the Hadamard part).

References:
    [1] Ailon & Chazelle, "The Fast Johnson-Lindenstrauss Transform", 2009
    [2] TurboQuant, Google Research, 2026
    [3] HyperQuant, arXiv:2606.23406, 2026
"""

import torch
import torch.nn as nn
import math
from typing import Optional


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length()


def build_hadamard_matrix(d: int, device: torch.device = None) -> torch.Tensor:
    """
    Build the d×d Hadamard matrix using the Sylvester (recursive Kronecker)
    construction. d must be a power of 2.

    The matrix satisfies: H @ H.T = d * I  (i.e., H/√d is orthogonal).

    This is primarily for verification/testing. For actual transforms,
    use fast_hadamard_transform() which is O(d log d) instead of O(d²).

    Args:
        d: Matrix dimension. Must be a power of 2.
        device: Target torch device.

    Returns:
        Tensor of shape (d, d) with entries ±1.
    """
    if d == 1:
        return torch.ones(1, 1, device=device)
    assert d > 0 and (d & (d - 1)) == 0, f"d must be a power of 2, got {d}"

    # Sylvester construction: H_{2n} = [[H_n, H_n], [H_n, -H_n]]
    H_half = build_hadamard_matrix(d // 2, device=device)
    H = torch.cat([
        torch.cat([H_half, H_half], dim=1),
        torch.cat([H_half, -H_half], dim=1),
    ], dim=0)
    return H


def fast_hadamard_transform(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform (FWHT) using the butterfly decomposition.
    Operates on the LAST dimension of x. Purely functional: never mutates
    the input tensor (all butterfly levels build new tensors via stack/reshape
    rather than writing through a view of x).

    Computational complexity: O(d log d) with only additions/subtractions.
    No matrix multiplications are performed.

    The butterfly algorithm:
        For each level k = 0, 1, ..., log2(d)-1:
            stride = 2^k
            For pairs (i, i+stride):
                x[..., i]         = x[..., i] + x[..., i+stride]
                x[..., i+stride]  = x[..., i] - x[..., i+stride]
                (using the pre-update values)

    Args:
        x: Input tensor of shape (..., d). d must be a power of 2.
           If d is not a power of 2, x is zero-padded along the last dim.
        normalize: If True, divide by √d to make the transform orthonormal
                   (i.e., energy-preserving). Default True.

    Returns:
        Transformed tensor of same shape as (possibly padded) input, same
        dtype as the input. If the input is float16/bfloat16, the butterfly
        is computed internally in float32 for numerical stability and cast
        back to the original dtype before returning.
    """
    orig_d = x.shape[-1]
    d = _next_power_of_two(orig_d)
    orig_dtype = x.dtype

    # Low-precision dtypes accumulate error over O(log d) butterfly levels;
    # compute in float32 and cast back.
    compute_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    x = x.to(compute_dtype)

    # Zero-pad if needed (torch.cat always allocates a new tensor)
    if d != orig_d:
        pad = torch.zeros(*x.shape[:-1], d - orig_d, device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad], dim=-1)

    # Butterfly decomposition: log2(d) levels, fully out-of-place at every
    # level (stack + reshape build a new tensor instead of writing through
    # a view of x, so the caller's original tensor is never touched).
    h = 1
    while h < d:
        x_pairs = x.reshape(*x.shape[:-1], d // (2 * h), 2, h)
        a = x_pairs[..., 0, :]
        b = x_pairs[..., 1, :]
        x = torch.stack((a + b, a - b), dim=-2).reshape(*x.shape[:-1], d)
        h *= 2

    if normalize:
        x = x / math.sqrt(d)

    return x.to(orig_dtype)


def randomized_hadamard_transform(
    x: torch.Tensor,
    signs: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Randomized Hadamard Transform (RHT): x' = (1/√d) · H · diag(signs) · x

    This is the core operation that makes TurboQuant work. The random sign
    flips (drawn once per model and frozen) ensure that after the Hadamard
    rotation, the entries of x' are approximately i.i.d. sub-Gaussian,
    regardless of the original distribution of x.

    This means:
    - Massive outlier channels (common in DinoV2/LayerNorm activations)
      are spread uniformly across ALL dimensions.
    - The resulting distribution is ideal for uniform scalar or lattice
      vector quantization.
    - The transform is DATA-OBLIVIOUS: the same frozen signs work for
      indoor scenes (NYUv2), outdoor driving (KITTI), and synthetic
      data (Sintel) without recalibration.

    Args:
        x: Input tensor of shape (..., d).
        signs: Tensor of shape (d,) with entries ±1. Draw once via:
               signs = torch.randint(0, 2, (d,)) * 2 - 1
        normalize: If True, output is energy-preserving (orthonormal).

    Returns:
        Rotated tensor of same shape.
    """
    # Step 1: Random sign flip (diagonal matrix multiplication, O(d))
    x_flipped = x * signs.to(x.device, x.dtype)

    # Step 2: Fast Hadamard Transform, O(d log d)
    return fast_hadamard_transform(x_flipped, normalize=normalize)


def inverse_randomized_hadamard_transform(
    x_rot: torch.Tensor,
    signs: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Inverse RHT: x = diag(signs) · H^T · (√d · x_rot)

    Since H is symmetric (H = H^T) and H·H = d·I, the inverse is simply
    applying the FHT again and then undoing the sign flips.

    Args:
        x_rot: Rotated tensor from randomized_hadamard_transform().
        signs: The SAME sign tensor used in the forward transform.
        normalize: Must match the forward transform's normalize flag.

    Returns:
        Reconstructed tensor (should match original input up to float precision).
    """
    # Step 1: Apply FHT again (H is its own inverse up to scaling)
    x_unrot = fast_hadamard_transform(x_rot, normalize=normalize)

    # Step 2: Undo sign flips
    return x_unrot * signs.to(x_unrot.device, x_unrot.dtype)


class HadamardRotation(nn.Module):
    """
    A reusable nn.Module wrapper around the Randomized Hadamard Transform.

    Usage:
        rot = HadamardRotation(dim=384)  # e.g., ViT-Small hidden dim
        x_rotated = rot(x)               # Forward RHT
        x_back = rot.inverse(x_rotated)  # Inverse RHT

    The random signs are stored as a non-trainable buffer and are frozen
    at initialization. They persist across forward passes and are saved
    with model state_dict.

    For Weight Absorption:
        If you want to absorb the rotation into a linear layer's weights
        offline (so that inference has ZERO overhead), use:
            W_absorbed = rot.absorb_into_weight(W)
        Then: x @ W == rot(x) @ W_absorbed  (mathematically equivalent)
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: Feature dimension to rotate. Will be padded to next power of 2.
        """
        super().__init__()
        self.dim = dim
        self.padded_dim = _next_power_of_two(dim)

        # Draw random signs once and freeze them (size = dim, NOT padded_dim)
        signs = torch.randint(0, 2, (self.dim,)) * 2 - 1
        self.register_buffer('signs', signs.float())

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Zero-pad last dimension from dim to padded_dim."""
        if self.dim == self.padded_dim:
            return x
        pad = torch.zeros(*x.shape[:-1], self.padded_dim - self.dim,
                          device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=-1)

    def _unpad(self, x: torch.Tensor) -> torch.Tensor:
        """Remove padding from last dimension."""
        if self.dim == self.padded_dim:
            return x
        return x[..., :self.dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RHT to the last dimension of x (handles non-power-of-2 dims).
        
        IMPORTANT: If dim is not a power of 2, output has padded_dim as last
        dimension. Use self.inverse() to recover the original dimension.
        For ViT-Small (head_dim=64) and ViT-Base (head_dim=64), dim IS a
        power of 2, so no padding occurs.
        """
        # Sign flip on original dim, then pad, then FHT
        x_flipped = x * self.signs.to(x.device, x.dtype)
        x_padded = self._pad(x_flipped)
        x_transformed = fast_hadamard_transform(x_padded, normalize=True)
        return x_transformed

    def inverse(self, x_rot: torch.Tensor) -> torch.Tensor:
        """Apply inverse RHT and return to original dimension."""
        x_transformed = fast_hadamard_transform(x_rot, normalize=True)
        x_unpadded = self._unpad(x_transformed)
        return x_unpadded * self.signs.to(x_unpadded.device, x_unpadded.dtype)

    def absorb_into_weight(self, W: torch.Tensor) -> torch.Tensor:
        """
        Absorb the Hadamard rotation into a weight matrix offline.

        Given linear layer Y = X @ W, after absorption:
            Y = RHT(X) @ W_absorbed
        where W_absorbed = (1/√d) · diag(signs) · H · W

        This means at inference time, we can:
        1. Apply RHT to activations (O(d log d), very fast)
        2. Quantize the rotated activations (now uniform, easy to quantize)
        3. Multiply by W_absorbed (same cost as original linear layer)
        => Total overhead is negligible.

        Args:
            W: Weight matrix of shape (d_in, d_out) where d_in == self.dim.

        Returns:
            Absorbed weight matrix of same shape.
        """
        d = self.padded_dim
        # Build full Hadamard for offline absorption (only done once)
        H = build_hadamard_matrix(d, device=W.device).float()
        D = torch.diag(self.signs.to(W.device))
        # R = (1/√d) · H · D (the full RHT matrix)
        R = (H @ D) / math.sqrt(d)
        # W_absorbed = R @ W (absorb rotation into weights)
        # Pad W if needed
        if W.shape[0] < d:
            W_padded = torch.zeros(d, W.shape[1], device=W.device, dtype=W.dtype)
            W_padded[:W.shape[0]] = W
            return (R @ W_padded)[:W.shape[0]]
        return R @ W

    def extra_repr(self) -> str:
        return f"dim={self.dim}, padded_dim={self.padded_dim}"
