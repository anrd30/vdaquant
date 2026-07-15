"""
Quantized Johnson-Lindenstrauss (QJL) Bias Correction.

When we quantize Key vectors in the attention mechanism, the quantization
error introduces BIAS into the attention scores:

    True score:  s[i,j] = Q[i] · K[j]
    Quant score: s_q[i,j] = Q[i] · K_q[j] = Q[i] · (K[j] - ε_K[j])
    Bias:        s_q[i,j] - s[i,j] = -Q[i] · ε_K[j]

where ε_K = K - K_q is the quantization error.

This bias causes systematic errors in the softmax distribution, leading
to incorrect attention weights and degraded depth predictions.

QJL Bias Correction estimates and removes this bias using:
1. The ERROR NORMS ||ε_K[j]|| (stored as float16, cheap)
2. The ERROR SIGNS sign(R · ε_K[j]) (stored as 1 bit each)

Corrected score:
    s_corrected[i,j] = s_q[i,j] + ||Q[i]|| · ||ε_K[j]|| · corr(i,j)
where:
    rho(i,j)  = (1/m) · sign(R·Q[i])^T · sign(R·ε_K[j])   [SimHash statistic]
    corr(i,j) = cos( (π/2) · (1 − rho(i,j)) )               ≈ cos(angle between Q[i] and ε_K[j])

Note: rho is NOT itself a cosine estimate (E[rho] = 1 − 2·theta/π, linear in
the angle). It must be mapped through the arccos-linear relation above
before use as corr(i,j) — see correct_scores() and
docs/optimization_ledger.md finding F3.

This makes the correction MAGNITUDE-AWARE: it scales with how big
the query and quantization error actually are.

Memory Cost:
    Per K vector: m bits (signs) + norm_bits (norm) ≈ (m + norm_bits) bits
    For m=32, d=64, norm_bits=16: 48 bits vs 2048 bits (FP32) = 42x savings
    (nominal payload only — see docs/optimization_ledger.md T1 for the
    all-inclusive effective rate once quantizer payload + scale metadata
    are also counted).

References:
    [1] TurboQuant, Section 4 (QJL for unbiased attention), Google 2026
    [2] Johnson & Lindenstrauss, 1984; Achlioptas, 2003
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


def default_qjl_projections(dim: int) -> int:
    """
    Default number of QJL random projections (m) for a given feature dim.
    The sign-correlation estimator has higher variance at low d, so more
    projections are used relative to dim for small dims, capped at 128 to
    bound the side-channel overhead (see docs/optimization_ledger.md T3).
    Shared by QJLBiasCorrection and scripts' honest bit-accounting so the
    two never silently disagree about what the runtime actually costs.
    """
    if dim <= 128:
        return min(2 * dim, 128)
    elif dim <= 256:
        return dim * 2
    else:
        return dim


class QJLBiasCorrection(nn.Module):
    """
    Magnitude-aware 1-bit QJL bias correction for quantized attention.

    Stores two things per quantized K vector:
    1. error_signs: sign(R · ε_K) ∈ {-1, +1}^m  (m bits)
    2. error_norms: ||ε_K|| ∈ ℝ                   (16 bits, float16)

    During attention, estimates the bias Q·ε_K using:
        bias_est[i,j] = ||Q[i]|| · ||ε_K[j]|| · (1/m) · sign(R·Q)^T · sign(R·ε_K)

    Usage:
        qjl = QJLBiasCorrection(dim=64, n_projections=32)

        # Encode: store error metadata alongside quantized K
        error_signs, error_norms = qjl.encode(K_original, K_quantized)

        # Correct: fix attention scores
        attn_corrected = qjl.correct_scores(attn_raw, Q, error_signs, error_norms)
    """

    def __init__(self, dim: int, n_projections: Optional[int] = None, norm_bits: int = 16):
        """
        Args:
            dim: Feature dimension of K/V vectors (e.g., 64 for ViT head_dim).
            n_projections: Number of random projections (m). More = better
                           estimate but more memory. Default: see
                           default_qjl_projections().
            norm_bits: Bit-width used to store each vector's error norm
                       (16 = fp16, 8 = uint8 quantized against a per-call
                       tensor max). This is real side-channel overhead —
                       see docs/optimization_ledger.md T1/T3.
        """
        super().__init__()
        assert norm_bits in (8, 16), "norm_bits must be 8 or 16"
        self.dim = dim
        self.norm_bits = norm_bits
        self.n_proj = n_projections if n_projections is not None else default_qjl_projections(dim)

        # Random projection matrix R ∈ {-1, +1}^{m×d} (Rademacher)
        # NOT scaled by 1/√m here — we handle scaling in correct_scores
        R = torch.randint(0, 2, (self.n_proj, dim)).float() * 2 - 1
        self.register_buffer('R', R)

    def encode(
        self, x_original: torch.Tensor, x_quantized: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the error signature (signs + norms) for quantized vectors.

        Args:
            x_original: Original (unquantized) tensor, shape (..., dim).
            x_quantized: Quantized tensor, same shape.

        Returns:
            error_signs: shape (..., n_projections) with entries in {-1, +1}
            error_norms: shape (..., 1) with ||ε|| per vector
        """
        # Quantization error
        epsilon = x_original - x_quantized  # (..., dim)

        # Store the L2 norm of each error vector
        error_norms = epsilon.norm(dim=-1, keepdim=True)  # (..., 1)

        if self.norm_bits == 8:
            # Simulate uint8 storage of the norms against a single tensor-wide
            # max (this is real metadata cost, not free — see norm_bits docs).
            norm_max = error_norms.amax().clamp(min=1e-8)
            norm_step = norm_max / 255.0
            error_norms = (error_norms / norm_step).round().clamp(0, 255) * norm_step

        # Project error onto random directions: ε @ R^T → (..., n_proj)
        error_proj = epsilon @ self.R.T

        # Store only the sign (1 bit per projection)
        error_signs = error_proj.sign()
        error_signs[error_signs == 0] = 1.0  # Break ties consistently

        return error_signs, error_norms

    def correct_scores(
        self,
        attn_scores: torch.Tensor,
        Q: torch.Tensor,
        K_error_signs: torch.Tensor,
        K_error_norms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Correct quantization bias in attention scores.

        The bias is:  bias[i,j] = -Q[i] · ε_K[j]

        We estimate it as:
            bias_est[i,j] = ||Q[i]|| · ||ε_K[j]|| · cosine_est(i,j)

        where cosine_est is built from the sign-correlation statistic:
            rho(i,j)        = (1/m) · sign(R·Q[i])^T · sign(R·ε_K[j])
            cosine_est(i,j) = cos( (π/2) · (1 − rho(i,j)) )

        IMPORTANT: rho itself is the SimHash collision statistic, with
        E[rho] = 1 − 2·theta/π (linear in the angle between Q[i] and ε_K[j],
        NOT its cosine). Using rho directly as if it were cos(theta) — as an
        earlier version of this function did — introduces a systematic bias
        of up to ≈0.21 near theta=π/4. The arccos-linear map above corrects
        this (see docs/optimization_ledger.md finding F3).

        Then: corrected[i,j] = attn_scores[i,j] + bias_est[i,j]
        (We ADD because the bias was -Q·ε, so we're undoing the subtraction)

        Args:
            attn_scores: Raw Q @ K_q^T scores, shape (..., n_q, n_k).
            Q: Query vectors, shape (..., n_q, dim).
            K_error_signs: From encode(), shape (..., n_k, n_proj).
            K_error_norms: From encode(), shape (..., n_k, 1).

        Returns:
            Corrected attention scores, same shape.
        """
        # Query norms: ||Q[i]||
        Q_norms = Q.norm(dim=-1, keepdim=True)  # (..., n_q, 1)

        # Project queries and take sign
        Q_proj_sign = (Q @ self.R.T).sign()  # (..., n_q, n_proj)
        Q_proj_sign[Q_proj_sign == 0] = 1.0

        # Sign-correlation statistic rho (SimHash collision rate estimator,
        # NOT a cosine estimate on its own — see docstring above).
        # (..., n_q, n_proj) @ (..., n_proj, n_k) → (..., n_q, n_k)
        rho = (Q_proj_sign @ K_error_signs.transpose(-2, -1)) / self.n_proj

        # Map rho -> a true cosine estimate via the arccos-linear relation
        # E[rho] = 1 - 2*theta/pi  =>  theta_hat = (pi/2)*(1 - rho).
        theta_hat = (math.pi / 2) * (1 - rho.clamp(-1.0, 1.0))
        cosine_est = torch.cos(theta_hat)

        # Scale by norms: ||Q[i]|| · ||ε_K[j]||
        # Q_norms: (..., n_q, 1),  K_error_norms: (..., n_k, 1) → need (..., 1, n_k)
        norm_scale = Q_norms * K_error_norms.transpose(-2, -1)  # (..., n_q, n_k)

        # Full bias estimate
        bias_estimate = norm_scale * cosine_est

        # ADD back the estimated bias (because quant removed Q·ε_K)
        corrected = attn_scores + bias_estimate

        return corrected

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, n_projections={self.n_proj}, norm_bits={self.norm_bits}, "
                f"cost={self.n_proj}bits+{self.norm_bits}bits/vector")
