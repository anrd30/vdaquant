"""
Equivalence gate for RotatedTemporalAttention vs the REAL VDA/AnimateDiff
TemporalAttention class (docs/optimization_ledger.md T7).

This test imports the actual cloned Video-Depth-Anything source
(Video-Depth-Anything/video_depth_anything/motion_module/motion_module.py)
rather than a hand-built mock, so it validates against ground truth. It is
skipped (not failed) if that source or its `einops` dependency isn't present,
so the rest of the test suite stays portable across environments that don't
have the VDA repo cloned.

Finding (see docs/optimization_ledger.md T7): the previously-disabled
replace_temporal surgery was blamed on a "reshape_heads_to_batch_dim
incompatibility", but tracing the real TemporalAttention.forward line by
line shows the reshape/attention math is mathematically identical to
RotatedTemporalAttention's (B,h,N,d) convention. The actual bug was that the
surgery constructed RotatedTemporalAttention with qkv_bias defaulting to
True while the real TemporalAttention (via CrossAttention's bias=False
default, never overridden anywhere in VDA's source) has NO bias on
to_q/to_k/to_v — so the replacement silently added a random, untrained bias
vector to every Q/K/V projection. With quantization bypassed (identity
quantizer, no QJL) and qkv_bias correctly matched, output error is exactly
0.0 (float-identical); with the old qkv_bias=True default, error is ~0.10 —
easily enough to garble a depth map.

Run: pytest tests/test_temporal_equivalence.py -q
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

VDA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Video-Depth-Anything")
sys.path.insert(0, VDA_DIR)

try:
    from video_depth_anything.motion_module.motion_module import TemporalAttention
    VDA_TEMPORAL_AVAILABLE = True
except ImportError:
    VDA_TEMPORAL_AVAILABLE = False

from research.models.rotated_attention import RotatedTemporalAttention


requires_vda = pytest.mark.skipif(
    not VDA_TEMPORAL_AVAILABLE,
    reason="Video-Depth-Anything source (and/or einops) not available; "
           "clone it to run this equivalence gate against ground truth.",
)


def _build_real_and_replacement(dim=64, heads=8, dim_head=8, qkv_bias_mismatch=False):
    torch.manual_seed(0)
    real = TemporalAttention(
        query_dim=dim, heads=heads, dim_head=dim_head, bias=False,
        norm_num_groups=None, temporal_max_len=32, pos_embedding_type="ape",
    ).eval()

    replacement = RotatedTemporalAttention(
        dim=dim, num_heads=heads, bits=32, quantizer='identity', use_qjl=False,
        qkv_bias=(True if qkv_bias_mismatch else (real.to_q.bias is not None)),
    ).eval()
    replacement.pos_encoder = real.pos_encoder
    replacement.group_norm = real.group_norm
    replacement.q_proj.weight.data.copy_(real.to_q.weight.data)
    replacement.k_proj.weight.data.copy_(real.to_k.weight.data)
    replacement.v_proj.weight.data.copy_(real.to_v.weight.data)
    if real.to_q.bias is not None and not qkv_bias_mismatch:
        replacement.q_proj.bias.data.copy_(real.to_q.bias.data)
        replacement.k_proj.bias.data.copy_(real.to_k.bias.data)
        replacement.v_proj.bias.data.copy_(real.to_v.bias.data)
    replacement.out_proj.weight.data.copy_(real.to_out[0].weight.data)
    if real.to_out[0].bias is not None:
        replacement.out_proj.bias.data.copy_(real.to_out[0].bias.data)

    return real, replacement


@requires_vda
def test_temporal_equivalence_non_cached():
    """
    Standard (non-streaming) call: video_length given, no cached_hidden_states.
    With quantization bypassed and qkv_bias correctly matched, output must be
    within 1e-3 of the REAL TemporalAttention (per T7's spec); in practice
    this is float-identical (~0.0) since both compute the same math.
    """
    dim, heads, dim_head = 64, 8, 8
    real, replacement = _build_real_and_replacement(dim, heads, dim_head)

    b, f, d_tokens = 2, 4, 5
    torch.manual_seed(1)
    hidden_states = torch.randn(b * f, d_tokens, dim)

    with torch.no_grad():
        out_real, cache_real = real(hidden_states, video_length=f)
        out_repl, cache_repl = replacement(hidden_states, video_length=f)

    assert out_real.shape == out_repl.shape, (out_real.shape, out_repl.shape)
    err = (out_real - out_repl).abs().max().item()
    print(f"  Non-cached equivalence: max abs error = {err:.8f}, shape={tuple(out_real.shape)}")
    assert err < 1e-3, f"RotatedTemporalAttention diverges from real TemporalAttention: {err:.6f}"


@requires_vda
def test_temporal_equivalence_cached_streaming():
    """
    Streaming call: cached_hidden_states given (video_length=None), matching
    how VDA processes video frame-by-frame with a growing temporal cache.
    """
    dim, heads, dim_head = 64, 8, 8
    real, replacement = _build_real_and_replacement(dim, heads, dim_head)

    b, d_tokens = 2, 5
    torch.manual_seed(2)
    # Prime a cache by running one frame through in non-cached mode first,
    # then feed a second single frame with that cache (mirrors streaming use).
    first_frame = torch.randn(b, d_tokens, dim)
    with torch.no_grad():
        _, cache_real = real(first_frame, video_length=1)
        _, cache_repl = replacement(first_frame, video_length=1)

    second_frame = torch.randn(b, d_tokens, dim)
    with torch.no_grad():
        out_real, _ = real(second_frame, cached_hidden_states=cache_real)
        out_repl, _ = replacement(second_frame, cached_hidden_states=cache_repl)

    assert out_real.shape == out_repl.shape, (out_real.shape, out_repl.shape)
    err = (out_real - out_repl).abs().max().item()
    print(f"  Cached/streaming equivalence: max abs error = {err:.8f}, shape={tuple(out_real.shape)}")
    assert err < 1e-3, f"RotatedTemporalAttention diverges from real TemporalAttention (cached): {err:.6f}"


@requires_vda
def test_qkv_bias_mismatch_regression_guard():
    """
    Regression guard proving WHY the bias fix matters: reconstructing the
    OLD (buggy) behavior — qkv_bias=True regardless of the source module's
    actual bias — must reproduce a clearly-detectable divergence (>> 1e-3).
    If this test starts failing (error becomes tiny), the surgery's bias
    detection has been silently removed/broken and this guard should catch it.
    """
    dim, heads, dim_head = 64, 8, 8
    real, replacement_buggy = _build_real_and_replacement(dim, heads, dim_head, qkv_bias_mismatch=True)

    b, f, d_tokens = 2, 4, 5
    torch.manual_seed(1)
    hidden_states = torch.randn(b * f, d_tokens, dim)

    with torch.no_grad():
        out_real, _ = real(hidden_states, video_length=f)
        out_buggy, _ = replacement_buggy(hidden_states, video_length=f)

    err = (out_real - out_buggy).abs().max().item()
    print(f"  qkv_bias mismatch regression guard: max abs error = {err:.6f} (expected >> 1e-3)")
    assert err > 1e-3, (
        f"Expected the qkv_bias=True/False mismatch to cause a clear divergence "
        f"(>1e-3), got {err:.6f} — the bug this guards against may no longer reproduce."
    )


def test_surgery_detects_qkv_bias_false_without_real_vda():
    """
    Portable regression test (no VDA repo/einops required) for the surgery
    fix itself: apply_rotated_quantization_to_vda must detect a source
    cross-attention module's bias=False on to_q/to_k/to_v and construct the
    replacement with qkv_bias=False (q_proj.bias is None), not silently use
    the RotatedTemporalAttention class default.
    """
    import torch.nn as nn
    from research.models.rotated_attention import apply_rotated_quantization_to_vda

    class MockCrossAttention(nn.Module):
        """Minimal stand-in matching CrossAttention's bias=False Q/K/V convention."""
        def __init__(self, dim=32, heads=4):
            super().__init__()
            self.heads = heads
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_k = nn.Linear(dim, dim, bias=False)
            self.to_v = nn.Linear(dim, dim, bias=False)
            self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=True), nn.Dropout(0.0)])

        def forward(self, x):
            return x  # never actually called; surgery only inspects the module

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cross = MockCrossAttention()

    model = TinyModel()
    model = apply_rotated_quantization_to_vda(
        model, bits=32, quantizer='identity', use_qjl=False,
        replace_backbone=False, replace_temporal=True, verbose=False,
    )

    from research.models.rotated_attention import RotatedTemporalAttention
    assert isinstance(model.cross, RotatedTemporalAttention), "Surgery did not replace the mock module"
    assert model.cross.q_proj.bias is None, (
        "Surgery should have detected bias=False on the source module's to_q and "
        "constructed q_proj with no bias; found a bias parameter instead."
    )
    assert model.cross.k_proj.bias is None
    assert model.cross.v_proj.bias is None
    print("  [OK] Surgery correctly detected qkv_bias=False from mock source module")


if __name__ == "__main__":
    test_surgery_detects_qkv_bias_false_without_real_vda()
    if not VDA_TEMPORAL_AVAILABLE:
        print("Video-Depth-Anything source not available; skipping real-module equivalence tests.")
    else:
        test_temporal_equivalence_non_cached()
        test_temporal_equivalence_cached_streaming()
        test_qkv_bias_mismatch_regression_guard()
        print("All temporal equivalence tests passed.")
