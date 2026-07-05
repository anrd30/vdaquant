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
    corr(i,j) = (1/m) · sign(R·Q[i])^T · sign(R·ε_K[j])
    ≈ cos(angle between Q[i] and ε_K[j])

This makes the correction MAGNITUDE-AWARE: it scales with how big
the query and quantization error actually are.

Memory Cost:
    Per K vector: m bits (signs) + 16 bits (norm) ≈ (m+16) bits
    For m=32, d=64: 48 bits vs 2048 bits (FP32) = 42x savings

References:
    [1] TurboQuant, Section 4 (QJL for unbiased attention), Google 2026
    [2] Johnson & Lindenstrauss, 1984; Achlioptas, 2003
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


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

    def __init__(self, dim: int, n_projections: Optional[int] = None):
        """
        Args:
            dim: Feature dimension of K/V vectors (e.g., 64 for ViT head_dim).
            n_projections: Number of random projections (m). More = better
                           estimate but more memory. Default: dim // 2.
                           Rule of thumb: m ≥ 4·log(n_tokens) for good estimates.
        """
        super().__init__()
        self.dim = dim
        # For small dims (ViT head_dim=64), we need many more projections
        # than for LLM dims (d=1024+) because the sign-correlation estimator
        # has higher variance at low d. Use 4x for d≤128, 2x for d≤256.
        if n_projections is None:
            if dim <= 128:
                self.n_proj = dim * 4
            elif dim <= 256:
                self.n_proj = dim * 2
            else:
                self.n_proj = dim
        else:
            self.n_proj = n_projections

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

        # Store the L2 norm of each error vector (cheap: 1 float per vector)
        error_norms = epsilon.norm(dim=-1, keepdim=True)  # (..., 1)

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

        where cosine_est uses the sign-correlation:
            cosine_est(i,j) = (1/m) · sign(R·Q[i])^T · sign(R·ε_K[j])

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

        # Sign-correlation: approximate cosine similarity between Q and ε_K
        # (..., n_q, n_proj) @ (..., n_proj, n_k) → (..., n_q, n_k)
        cosine_est = (Q_proj_sign @ K_error_signs.transpose(-2, -1)) / self.n_proj

        # Scale by norms: ||Q[i]|| · ||ε_K[j]||
        # Q_norms: (..., n_q, 1),  K_error_norms: (..., n_k, 1) → need (..., 1, n_k)
        norm_scale = Q_norms * K_error_norms.transpose(-2, -1)  # (..., n_q, n_k)

        # Full bias estimate
        bias_estimate = norm_scale * cosine_est

        # ADD back the estimated bias (because quant removed Q·ε_K)
        corrected = attn_scores + bias_estimate

        return corrected

    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_projections={self.n_proj}, cost={self.n_proj}bits+16bits/vector"


class ShrinkageQJLBiasCorrection(QJLBiasCorrection):
    """
    Confidence-weighted (shrinkage) variant of QJL bias correction.

    This is a first-pass heuristic weighting (not a rigorously derived variance formula)
    meant to make the QJL correction fade out smoothly at low bit-widths instead of
    requiring a hard qjl_min_bits cutoff.

    When quantization error is large (low bit-widths), the estimated bias can be noisy.
    We scale the correction by a shrinkage factor w ∈ [0, 1]:
        correction = w · bias_estimate
        corrected_score = raw_score + correction

    where:
        confidence = ||Q|| · ||ε_K||
        noise_floor = ||ε_K|| / sqrt(m)
        w = confidence / (confidence + noise_floor + eps), clamped to [0, 1]

    When ||ε_K|| is tiny (high precision), w ≈ 1.
    When ||ε_K|| is large (coarse precision), w shrinks to attenuate noise.
    """

    def correct_scores(
        self,
        attn_scores: torch.Tensor,
        Q: torch.Tensor,
        K_error_signs: torch.Tensor,
        K_error_norms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Correct quantization bias in attention scores using confidence shrinkage.

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

        # Sign-correlation: approximate cosine similarity between Q and ε_K
        cosine_est = (Q_proj_sign @ K_error_signs.transpose(-2, -1)) / self.n_proj

        # Scale by norms: ||Q[i]|| · ||ε_K[j]||
        norm_scale = Q_norms * K_error_norms.transpose(-2, -1)  # (..., n_q, n_k)

        # Full bias estimate
        bias_estimate = norm_scale * cosine_est

        # Compute confidence and noise floor for shrinkage weight w
        # Using ||ε_K||^2 / sqrt(m) for noise floor so that w shrinks at coarse quantization
        confidence = norm_scale  # ||Q|| * ||ε_K||
        noise_floor = (K_error_norms.transpose(-2, -1) ** 2) / math.sqrt(self.n_proj)  # (..., 1, n_k)

        # When ||ε_K|| is tiny, set w ≈ 1.0 to avoid divide-by-zero or vanishing weights
        eps = 1e-8
        w = torch.where(
            K_error_norms.transpose(-2, -1) < 1e-6,
            torch.ones_like(confidence),
            confidence / (confidence + noise_floor + eps)
        ).clamp(min=0.0, max=1.0)

        # Scale correction by confidence weight w
        correction = w * bias_estimate

        # ADD back the weighted estimated bias
        corrected = attn_scores + correction

        return corrected

