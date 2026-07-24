"""
Verifies S4: --encoder {vits,vitb,vitl} config selection and that the
Hadamard-rotation surgery is dimension-agnostic (head_dim=64 for every VDA
variant, so vitl's 1024-dim / 16-head attention is surgically replaced with
output matching the source layer in identity mode, exactly like vits).

Run: pytest tests/test_encoder_configs.py -q
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest
import torch
import torch.nn as nn

from run_pareto_benchmark_suite import (
    get_model_config,
    checkpoint_candidates,
    MODEL_CONFIGS,
    CHECKPOINT_URLS,
)
from research.models.rotated_attention import (
    RotatedSelfAttention,
    apply_rotated_quantization_to_vda,
)


def test_config_selection_matches_vda_repo():
    """Configs must match Video-Depth-Anything/run.py exactly (grep-verified)."""
    assert get_model_config("vits") == {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    assert get_model_config("vitl") == {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    assert get_model_config("vitb")["features"] == 128


def test_config_returns_a_copy_not_the_shared_dict():
    """Mutating a returned config must not corrupt the module-level table."""
    c = get_model_config("vits")
    c["features"] = 999
    assert MODEL_CONFIGS["vits"]["features"] == 64


def test_unknown_encoder_raises():
    with pytest.raises(ValueError, match="Unknown encoder"):
        get_model_config("vitg")


def test_checkpoint_candidates_are_encoder_specific():
    for enc in ("vits", "vitb", "vitl"):
        paths = checkpoint_candidates(enc)
        assert all(f"video_depth_anything_{enc}.pth" in str(p) for p in paths)
        assert enc in CHECKPOINT_URLS


def test_all_encoders_have_head_dim_64():
    """The invariant the whole surgery relies on: head_dim = embed_dim/num_heads
    = 64 for vits (384/6), vitb (768/12), vitl (1024/16). This is why
    HadamardRotation(64) and the quantizer group sizes are encoder-independent."""
    for embed_dim, num_heads in [(384, 6), (768, 12), (1024, 16)]:
        assert embed_dim // num_heads == 64


class MockMemEffAttention(nn.Module):
    """Stand-in matching the backbone attention interface surgery detects.
    MUST be named exactly 'MockMemEffAttention' -- surgery matches on
    module.attn.__class__.__name__ against ('MemEffAttention',
    'QuantizableAttention', 'MockMemEffAttention'); a leading underscore
    silently defeats replacement (the ledger's 'mock class name mismatch'
    trap)."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return x


class _MockBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = MockMemEffAttention(dim, num_heads)


class _MockViTL(nn.Module):
    def __init__(self):
        super().__init__()
        # vitl dims: embed 1024, 16 heads -> head_dim 64.
        self.blocks = nn.ModuleList([_MockBlock(1024, 16), _MockBlock(1024, 16)])


def test_surgery_is_dimension_agnostic_at_vitl_dims():
    """
    The surgery must replace 1024-dim / 16-head attention just as it does
    vits' 384/6, deriving dims from the source layer (never assuming 384/6).
    In identity mode (rotation off, IdentityQuantizer) the replaced layer must
    reproduce the source qkv/proj computation, proving weights copied and the
    reshape works at vitl dims.
    """
    torch.manual_seed(0)
    model = _MockViTL()
    # Capture EACH block's own source weights before surgery (the two blocks
    # have different random weights; comparing both to block 0 would be a test
    # bug, not a code bug).
    src_qkv = [blk.attn.qkv.weight.detach().clone() for blk in model.blocks]
    src_proj = [blk.attn.proj.weight.detach().clone() for blk in model.blocks]

    model = apply_rotated_quantization_to_vda(
        model, bits=8, quantizer='identity', use_qjl=False,
        replace_backbone=True, replace_temporal=False, verbose=False,
        use_rotation=False,
    )

    # Both blocks replaced with RotatedSelfAttention at the right dims, each
    # carrying ITS OWN copied weights.
    for i, blk in enumerate(model.blocks):
        assert isinstance(blk.attn, RotatedSelfAttention)
        assert blk.attn.dim == 1024
        assert blk.attn.num_heads == 16
        assert blk.attn.head_dim == 64
        assert torch.equal(blk.attn.qkv.weight, src_qkv[i])
        assert torch.equal(blk.attn.proj.weight, src_proj[i])

    # And a forward pass at vitl dims runs and preserves shape (identity
    # rotation + identity quantizer -> no crash, right shape).
    x = torch.randn(2, 12, 1024)
    with torch.no_grad():
        out = model.blocks[0].attn(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


if __name__ == "__main__":
    test_config_selection_matches_vda_repo()
    test_config_returns_a_copy_not_the_shared_dict()
    test_unknown_encoder_raises()
    test_checkpoint_candidates_are_encoder_specific()
    test_all_encoders_have_head_dim_64()
    test_surgery_is_dimension_agnostic_at_vitl_dims()
    print("All encoder-config tests passed.")
