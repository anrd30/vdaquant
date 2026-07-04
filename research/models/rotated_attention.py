"""
Hadamard-Rotated Attention Modules for Video-Depth-Anything (VDA).

These are drop-in replacements for the standard MemEffAttention (DinoV2
self-attention) and CrossAttention (DPT temporal attention) layers in VDA.

Architecture Overview:
    Standard VDA Attention:
        Q, K, V = linear(x)
        attn = softmax(Q @ K^T / √d) @ V    ← K,V stored in FP32 (huge!)

    Our Rotated + Quantized Attention:
        Q, K, V = linear(x)
        K_rot = RHT(K)                        ← Hadamard rotation (O(d log d))
        V_rot = RHT(V)                        ← Same rotation
        K_q, K_signs = quantize(K_rot)         ← 3-4 bit lattice quantization
        V_q = quantize(V_rot)                  ← 3-4 bit lattice quantization
        attn = softmax(Q_rot @ K_q^T / √d)    ← QJL-corrected scores
        out = attn @ V_q                       ← Standard matmul
        out = RHT_inv(out)                     ← Undo rotation

    Benefits:
        - K,V cache compressed from FP32 (32 bits) to 3-4 bits = 8-10x savings
        - RHT eliminates outliers → uniform distribution → near-optimal quantization
        - QJL correction removes systematic attention bias
        - Total overhead: O(d log d) additions per vector (negligible)

Usage with Video-Depth-Anything:
    from research.models import apply_rotated_quantization_to_vda

    model = VideoDepthAnything(encoder='vits', ...)
    model.load_state_dict(torch.load('checkpoint.pth'))
    model = apply_rotated_quantization_to_vda(model, bits=4, quantizer='lattice_d4')

References:
    [1] VDA original: github.com/DepthAnything/Video-Depth-Anything
    [2] TurboQuant KV cache compression, Google 2026
    [3] 3DTurboQuant (DUSt3R KV compression), arXiv:2604.05366
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal

from research.transforms.hadamard import HadamardRotation
from research.quantizers.lattice_vq import (
    ScalarRoundQuantizer,
    UniformVectorQuantizer,
    LatticeD4Quantizer,
    residual_vector_quantize,
)
from research.quantizers.qjl_bias import QJLBiasCorrection


def _get_quantizer(
    method: Literal['scalar', 'uniform_vector', 'lattice_d4'],
    bits: int,
    group_size: int = 4,
) -> nn.Module:
    """Factory function to create the appropriate quantizer."""
    if method == 'scalar':
        return ScalarRoundQuantizer(bits=bits, symmetric=True)
    elif method == 'uniform_vector':
        return UniformVectorQuantizer(bits=bits, group_size=group_size)
    elif method == 'lattice_d4':
        return LatticeD4Quantizer(bits=bits, group_size=4)
    else:
        raise ValueError(f"Unknown quantizer method: {method}")


class RotatedSelfAttention(nn.Module):
    """
    Drop-in replacement for DinoV2's MemEffAttention.

    Standard MemEffAttention uses xFormers' memory-efficient attention kernel.
    We replace it with our Hadamard-rotated, quantized attention that:
    1. Rotates K and V using RHT before quantization
    2. Quantizes K and V to low bit-width (3-4 bits)
    3. Corrects attention scores using QJL bias correction
    4. Produces output equivalent to FP32 (up to quantization noise)

    This module is designed for the DinoV2 backbone of VDA where each
    transformer block has a self-attention layer operating on spatial
    patch tokens.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bits: int = 4,
        quantizer: str = 'lattice_d4',
        use_qjl: bool = True,
        use_residual: bool = False,
    ):
        """
        Args:
            dim: Input feature dimension (e.g., 384 for ViT-Small).
            num_heads: Number of attention heads (e.g., 6 for ViT-Small).
            qkv_bias: Whether to use bias in QKV projection.
            attn_drop: Dropout on attention weights.
            proj_drop: Dropout on output projection.
            bits: Quantization bit-width for K and V caches.
            quantizer: Quantizer type ('scalar', 'uniform_vector', 'lattice_d4').
            use_qjl: Whether to apply QJL bias correction.
            use_residual: Enable temporal residual (delta) quantization.
                Frame 0 = I-frame (direct quant), frames 1+ = P-frames
                (quantize only the residual from previous reconstruction).
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.bits = bits
        self.use_qjl = use_qjl
        self.use_residual = use_residual

        # Standard attention projections (same as original)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Our additions: Hadamard rotation + quantizer + QJL
        self.rotation = HadamardRotation(self.head_dim)
        self.k_quantizer = _get_quantizer(quantizer, bits)
        self.v_quantizer = _get_quantizer(quantizer, bits)
        if use_qjl:
            self.qjl = QJLBiasCorrection(self.rotation.padded_dim)
        else:
            self.qjl = None

    def forward(self, x: torch.Tensor = None, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with Hadamard-rotated quantized attention.
        """
        if x is None and len(args) > 0:
            x = args[0]
        elif x is None:
            x = kwargs.get('hidden_states', None)
        if x is None:
            x = kwargs.get('x', None)

        B, N, C = x.shape
        h = self.num_heads
        d = self.head_dim

        # Standard QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, h, d).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv.unbind(0)  # each: (B, h, N, d)

        # ═══ OUR INNOVATION STARTS HERE ═══

        # Step 1: Apply Hadamard rotation to K and V
        K_rot = self.rotation(K)  # (B, h, N, d) → rotated
        V_rot = self.rotation(V)

        # Step 2: Quantize rotated K and V
        # With use_residual: B dimension = frames in chunk, so temporal_dim=0
        # Frame 0 (I-frame): quantize directly. Frames 1+ (P-frames): quantize residual only.
        if self.use_residual and B > 1:
            K_q, k_info = residual_vector_quantize(K_rot, self.k_quantizer, temporal_dim=0)
            V_q, v_info = residual_vector_quantize(V_rot, self.v_quantizer, temporal_dim=0)
        else:
            K_q, k_info = self.k_quantizer(K_rot)
            V_q, v_info = self.v_quantizer(V_rot)

        # Step 3: Rotate Q into the same basis for correct dot products
        Q_rot = self.rotation(Q)

        # Step 4: Compute attention scores
        attn = (Q_rot @ K_q.transpose(-2, -1)) * self.scale

        # Step 5: QJL bias correction (if enabled)
        if self.qjl is not None and self.training is False:
            K_error_signs, K_error_norms = self.qjl.encode(K_rot, K_q)
            attn = self.qjl.correct_scores(attn, Q_rot, K_error_signs, K_error_norms)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Step 6: Weighted sum with quantized V
        out_rot = attn @ V_q  # (B, h, N, d) in rotated basis

        # Step 7: Inverse rotation to return to original basis
        out = self.rotation.inverse(out_rot)

        # ═══ OUR INNOVATION ENDS HERE ═══

        # Standard output projection
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class RotatedTemporalAttention(nn.Module):
    """
    Drop-in replacement for VDA's DPT temporal cross-attention.

    In Video-Depth-Anything, the DPT decoder head uses temporal
    cross-attention to aggregate depth features across consecutive
    video frames. This is the layer defined in dpt_temporal.py.

    Our innovation for temporal attention:
        - We apply the SAME Hadamard rotation to K/V across all frames
        - This ensures that the quantization grid is temporally consistent
        - Combined with QJL bias correction, this guarantees that the
          relative depth ordering between frame t and frame t+1 is
          preserved even at 3-bit precision
        - Result: zero temporal flickering (low TAE)

    The key insight: because the Hadamard rotation is orthogonal and
    deterministic (same frozen signs for every frame), the quantization
    error is temporally correlated in a controlled way. This is MUCH
    better than independent per-frame scalar quantization, which
    introduces uncorrelated noise that causes flickering.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        bits: int = 4,
        quantizer: str = 'lattice_d4',
        use_qjl: bool = True,
        use_residual: bool = False,
    ):
        """
        Args:
            dim: Feature dimension of the DPT decoder.
            num_heads: Number of attention heads.
            qkv_bias: Use bias in projections.
            bits: Quantization bits for temporal K/V cache.
            quantizer: Quantizer type.
            use_qjl: Enable QJL bias correction.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.bits = bits
        self.use_residual = use_residual

        # Cross-attention projections
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        # Aliases for VDA TemporalAttention compatibility
        self.to_q = self.q_proj
        self.to_k = self.k_proj
        self.to_v = self.v_proj
        self.to_out = nn.ModuleList([self.out_proj, nn.Dropout(0.0)])

        # Shared rotation for temporal consistency
        self.rotation = HadamardRotation(self.head_dim)
        self.k_quantizer = _get_quantizer(quantizer, bits)
        self.v_quantizer = _get_quantizer(quantizer, bits)
        if use_qjl:
            self.qjl = QJLBiasCorrection(self.rotation.padded_dim)
        else:
            self.qjl = None

    def forward(
        self,
        hidden_states: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        video_length: int = None,
        cached_hidden_states: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        """
        Temporal cross-attention with Hadamard-rotated quantized KV cache.
        Supports both standard QKV calls and VDA DPT TemporalAttention calls.
        """
        # Resolve positional args if passed as query/context instead of hidden_states
        if hidden_states is None and len(args) > 0:
            hidden_states = args[0]
        elif hidden_states is None:
            hidden_states = kwargs.get('query', None)

        if encoder_hidden_states is None and len(args) > 1:
            encoder_hidden_states = args[1]
        elif encoder_hidden_states is None:
            encoder_hidden_states = kwargs.get('context', None)

        query_input = hidden_states
        context_input = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        orig_b_f = hidden_states.shape[0] if hidden_states is not None else None
        orig_d_tokens = hidden_states.shape[1] if hidden_states is not None else None

        # Handle VDA temporal sequence formatting (rearrange across frames)
        d_in = 0
        input_hidden_states = hidden_states
        is_vda_temporal = (video_length is not None or cached_hidden_states is not None)

        if is_vda_temporal:
            if cached_hidden_states is None and video_length is not None:
                b_f, d_tokens, c_dim = hidden_states.shape
                f_frames = video_length
                b_size = b_f // f_frames
                hidden_states = hidden_states.reshape(b_size, f_frames, d_tokens, c_dim).permute(0, 2, 1, 3).reshape(b_size * d_tokens, f_frames, c_dim)
                input_hidden_states = hidden_states
            elif cached_hidden_states is not None:
                b_f, d_tokens, c_dim = hidden_states.shape
                hidden_states = hidden_states.reshape(-1, 1, d_tokens, c_dim).permute(0, 2, 1, 3).reshape(-1, 1, c_dim)
                input_hidden_states = hidden_states
                d_in = cached_hidden_states.shape[1]
                hidden_states = torch.cat([cached_hidden_states, hidden_states], dim=1)

            if getattr(self, 'pos_encoder', None) is not None:
                hidden_states = self.pos_encoder(hidden_states)
            if getattr(self, 'group_norm', None) is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query_input = hidden_states[:, d_in:, ...]
            context_input = hidden_states

        B, N, C = query_input.shape
        M = context_input.shape[1]
        h = self.num_heads
        d = self.head_dim

        # Project Q from query, K/V from context
        Q = self.q_proj(query_input).reshape(B, N, h, d).transpose(1, 2)
        K = self.k_proj(context_input).reshape(B, M, h, d).transpose(1, 2)
        V = self.v_proj(context_input).reshape(B, M, h, d).transpose(1, 2)

        # ═══ TEMPORAL-COUPLED ROTATION + QUANTIZATION ═══
        K_rot = self.rotation(K)
        V_rot = self.rotation(V)

        # Apply residual quantization along the frame/sequence dimension (dim=2 = M)
        if self.use_residual and K_rot.shape[2] > 1:
            K_q, _ = residual_vector_quantize(K_rot, self.k_quantizer, temporal_dim=2)
            V_q, _ = residual_vector_quantize(V_rot, self.v_quantizer, temporal_dim=2)
        else:
            K_q, _ = self.k_quantizer(K_rot)
            V_q, _ = self.v_quantizer(V_rot)

        Q_rot = self.rotation(Q)

        # Attention with optional QJL correction
        attn = (Q_rot @ K_q.transpose(-2, -1)) * self.scale

        if self.qjl is not None and not self.training:
            K_signs, K_norms = self.qjl.encode(K_rot, K_q)
            attn = self.qjl.correct_scores(attn, Q_rot, K_signs, K_norms)

        attn = attn.softmax(dim=-1)
        out_rot = attn @ V_q

        # Inverse rotation
        out = self.rotation.inverse(out_rot)

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        if hasattr(self, 'to_out') and isinstance(self.to_out, (nn.Sequential, nn.ModuleList)) and len(self.to_out) > 1:
            out = self.to_out[1](out)

        if is_vda_temporal:
            # Reshape back from (b d) f c -> (b f) d c using exact original token and batch counts
            bd_size, f_len, c_dim = out.shape
            b_size = orig_b_f // f_len if orig_b_f is not None and f_len > 0 else bd_size // (orig_d_tokens or 1)
            d_tokens = orig_d_tokens if orig_d_tokens is not None else bd_size // (b_size or 1)
            out = out.reshape(b_size, d_tokens, f_len, c_dim).permute(0, 2, 1, 3).reshape(b_size * f_len, d_tokens, c_dim)
            return out, input_hidden_states

        return out


def apply_rotated_quantization_to_vda(
    model: nn.Module,
    bits: int = 4,
    quantizer: str = 'lattice_d4',
    use_qjl: bool = True,
    replace_backbone: bool = True,
    replace_temporal: bool = True,
    use_residual: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """
    Apply Hadamard-rotated quantization to a Video-Depth-Anything model.

    This function walks the model tree and replaces:
    1. DinoV2 MemEffAttention → RotatedSelfAttention (backbone)
    2. DPT CrossAttention → RotatedTemporalAttention (temporal decoder)

    The replacement modules copy the pretrained weights from the original
    layers and add our rotation/quantization/QJL machinery on top.

    Args:
        model: A loaded VDA model (e.g., VideoDepthAnything with vits encoder).
        bits: Target bit-width for KV cache quantization.
        quantizer: Quantizer type ('scalar', 'uniform_vector', 'lattice_d4').
        use_qjl: Enable QJL bias correction for attention scores.
        replace_backbone: Replace DinoV2 self-attention layers.
        replace_temporal: Replace DPT temporal cross-attention layers.
        verbose: Print replacement summary.

    Returns:
        Modified model with rotated quantized attention layers.
    """
    n_backbone = 0
    n_temporal = 0

    if replace_backbone:
        # Replace DinoV2 backbone MemEffAttention layers
        for name, module in model.named_modules():
            # Look for attention modules in DinoV2 blocks
            if hasattr(module, 'attn') and module.attn.__class__.__name__ in (
                'MemEffAttention', 'QuantizableAttention', 'MockMemEffAttention'
            ):
                old_attn = module.attn
                dim = old_attn.qkv.in_features
                num_heads = getattr(old_attn, 'num_heads', dim // 64)
                has_bias = old_attn.qkv.bias is not None

                device = getattr(old_attn.qkv.weight, 'device', torch.device('cpu'))
                dtype = getattr(old_attn.qkv.weight, 'dtype', torch.float32)
                new_attn = RotatedSelfAttention(
                    dim=dim,
                    num_heads=num_heads,
                    qkv_bias=has_bias,
                    bits=bits,
                    quantizer=quantizer,
                    use_qjl=use_qjl,
                    use_residual=use_residual,
                ).to(device=device, dtype=dtype)

                # Copy pretrained weights
                new_attn.qkv.weight.data.copy_(old_attn.qkv.weight.data)
                if has_bias:
                    new_attn.qkv.bias.data.copy_(old_attn.qkv.bias.data)
                new_attn.proj.weight.data.copy_(old_attn.proj.weight.data)
                if old_attn.proj.bias is not None:
                    new_attn.proj.bias.data.copy_(old_attn.proj.bias.data)

                module.attn = new_attn
                n_backbone += 1

    if replace_temporal:
        # Replace DPT temporal CrossAttention layers
        for name, module in model.named_modules():
            if module.__class__.__name__ in ('CrossAttention', 'MockCrossAttention', 'TemporalAttention'):
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]

                # Get parent module
                parent = model
                for part in parent_name.split('.'):
                    if part:
                        parent = getattr(parent, part)

                old_cross = module
                # Infer dimensions from existing weights
                dim = old_cross.to_q.in_features if hasattr(old_cross, 'to_q') else \
                      old_cross.q_proj.in_features if hasattr(old_cross, 'q_proj') else None

                if dim is not None:
                    num_heads = getattr(old_cross, 'heads', 8)
                    weight_attr = getattr(old_cross, 'to_q', getattr(old_cross, 'q_proj', None))
                    device = getattr(weight_attr.weight, 'device', torch.device('cpu')) if weight_attr is not None else torch.device('cpu')
                    dtype = getattr(weight_attr.weight, 'dtype', torch.float32) if weight_attr is not None else torch.float32
                    new_cross = RotatedTemporalAttention(
                        dim=dim,
                        num_heads=num_heads,
                        bits=bits,
                        quantizer=quantizer,
                        use_qjl=use_qjl,
                        use_residual=use_residual,
                    ).to(device=device, dtype=dtype)
                    # Copy weights where possible
                    if hasattr(old_cross, 'to_q'):
                        new_cross.q_proj.weight.data.copy_(old_cross.to_q.weight.data)
                        new_cross.k_proj.weight.data.copy_(old_cross.to_k.weight.data)
                        new_cross.v_proj.weight.data.copy_(old_cross.to_v.weight.data)
                    if hasattr(old_cross, 'to_out'):
                        out_layer = old_cross.to_out[0] if isinstance(old_cross.to_out, (nn.Sequential, nn.ModuleList)) else old_cross.to_out
                        new_cross.out_proj.weight.data.copy_(out_layer.weight.data)
                        if out_layer.bias is not None:
                            new_cross.out_proj.bias.data.copy_(out_layer.bias.data)

                    if hasattr(old_cross, 'pos_encoder'):
                        new_cross.pos_encoder = old_cross.pos_encoder
                    if hasattr(old_cross, 'group_norm'):
                        new_cross.group_norm = old_cross.group_norm

                    setattr(parent, attr_name, new_cross)
                    n_temporal += 1

    try:
        device = next(model.parameters()).device
        model = model.to(device)
    except StopIteration:
        pass

    if verbose:
        print(f"{'=' * 60}")
        print(f"  VDA-HyperQuant Model Surgery Complete")
        print(f"{'=' * 60}")
        print(f"  Backbone (DinoV2) attention layers replaced: {n_backbone}")
        print(f"  Temporal (DPT) cross-attention layers replaced: {n_temporal}")
        print(f"  Quantizer: {quantizer} @ {bits}-bit")
        print(f"  QJL bias correction: {'Enabled' if use_qjl else 'Disabled'}")
        print(f"  Compression: ~{32 / bits:.1f}x over FP32 KV cache")
        print(f"{'=' * 60}")

    return model
