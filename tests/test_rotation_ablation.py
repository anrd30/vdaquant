"""
Verifies T10: the --rotation/--no-rotation ablation (HadamardRotation
identity mode + --rht-seed reproducibility) and the scalar-baseline bit-
accounting fix (docs/optimization_ledger.md T10). Pure synthetic tensors,
no GPU/dataset/network.

Run: pytest tests/test_rotation_ablation.py -q
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import torch
import torch.nn as nn

from research.transforms.hadamard import HadamardRotation
from research.models.rotated_attention import (
    RotatedSelfAttention,
    apply_rotated_quantization_to_vda,
)
from research.quantizers.lattice_vq import LatticeD4Quantizer
from run_pareto_benchmark_suite import resolve_group_size, compute_real_bit_accounting, QUANTIZER_GROUP_SIZE


# ----------------------------- HadamardRotation identity mode -----------------

def test_hadamard_identity_is_true_noop():
    torch.manual_seed(0)
    rot = HadamardRotation(dim=64, identity=True)
    assert rot.padded_dim == 64, "identity mode needs no power-of-2 padding"
    x = torch.randn(4, 10, 64)
    out = rot(x)
    assert torch.equal(out, x), "identity forward must return the exact same values"
    back = rot.inverse(out)
    assert torch.equal(back, x), "identity inverse must return the exact same values"
    print("  [OK] HadamardRotation(identity=True) is a true no-op (forward and inverse)")


def test_hadamard_identity_has_no_signs_buffer():
    """Identity mode shouldn't even allocate a signs buffer (nothing to draw)."""
    rot = HadamardRotation(dim=64, identity=True)
    assert not hasattr(rot, 'signs') or 'signs' not in dict(rot.named_buffers())


def test_hadamard_seed_reproducibility():
    rot_a = HadamardRotation(dim=64, seed=42)
    rot_b = HadamardRotation(dim=64, seed=42)
    assert torch.equal(rot_a.signs, rot_b.signs), "same seed must give identical signs"

    rot_c = HadamardRotation(dim=64, seed=43)
    assert not torch.equal(rot_a.signs, rot_c.signs), "different seeds should (almost certainly) differ"
    print("  [OK] HadamardRotation(seed=N) is reproducible; different seeds differ")


def test_hadamard_seed_does_not_disturb_global_rng():
    """Seeding via a local generator must not perturb the GLOBAL torch RNG
    state -- otherwise passing --rht-seed would silently change unrelated
    randomness elsewhere in the same run (e.g. weight init, dataset shuffling)."""
    torch.manual_seed(123)
    before = torch.rand(5)

    torch.manual_seed(123)
    _ = HadamardRotation(dim=64, seed=999)  # should NOT touch the global RNG
    after = torch.rand(5)

    assert torch.equal(before, after), "seeding HadamardRotation must not perturb the global RNG stream"


def test_hadamard_extra_repr_reflects_identity():
    rot_on = HadamardRotation(dim=64)
    rot_off = HadamardRotation(dim=64, identity=True)
    assert "IDENTITY" in rot_off.extra_repr()
    assert "IDENTITY" not in rot_on.extra_repr()


# ----------------------------- rotation ablation actually matters -------------

def test_rotation_reduces_quantization_error_on_outlier_data():
    """
    The entire justification for RHT: on activation data with per-group
    outliers (the realistic ViT activation pattern this whole method targets),
    rotating BEFORE lattice-quantizing must give lower MSE than quantizing the
    raw data directly. This is the direction-of-effect the --no-rotation
    ablation is supposed to demonstrate; lock it in as an actual assertion,
    not just "the flag doesn't crash".
    """
    torch.manual_seed(1)
    d = 64
    bits = 3
    x = torch.randn(2000, d) * 0.5
    # Inject a sparse extreme outlier per row (simulates the activation-outlier
    # pattern RHT is designed to suppress).
    outlier_idx = torch.randint(0, d, (x.shape[0],))
    x[torch.arange(x.shape[0]), outlier_idx] += torch.randn(x.shape[0]) * 15.0

    rot_on = HadamardRotation(dim=d)
    rot_off = HadamardRotation(dim=d, identity=True)
    quantizer_on = LatticeD4Quantizer(bits=bits)
    quantizer_off = LatticeD4Quantizer(bits=bits)

    x_rot = rot_on(x)
    x_rot_q, _ = quantizer_on(x_rot)
    x_reconstructed_rotated_path = rot_on.inverse(x_rot_q)
    mse_with_rotation = ((x - x_reconstructed_rotated_path) ** 2).mean().item()

    x_raw = rot_off(x)  # no-op
    x_raw_q, _ = quantizer_off(x_raw)
    x_reconstructed_raw_path = rot_off.inverse(x_raw_q)
    mse_without_rotation = ((x - x_reconstructed_raw_path) ** 2).mean().item()

    print(f"  MSE with rotation: {mse_with_rotation:.6f}, without: {mse_without_rotation:.6f}")
    assert mse_with_rotation < mse_without_rotation, (
        f"Expected RHT to reduce quantization MSE on outlier data "
        f"({mse_with_rotation:.6f}) vs raw quantization ({mse_without_rotation:.6f})"
    )


# ----------------------------- RotatedSelfAttention wiring ---------------------

def test_rotated_self_attention_use_rotation_flag():
    attn_on = RotatedSelfAttention(dim=64, num_heads=1, bits=3, quantizer='lattice_d4', use_rotation=True)
    attn_off = RotatedSelfAttention(dim=64, num_heads=1, bits=3, quantizer='lattice_d4', use_rotation=False)
    assert attn_on.rotation.identity is False
    assert attn_off.rotation.identity is True

    x = torch.randn(2, 10, 64)
    with torch.no_grad():
        out_on = attn_on(x)
        out_off = attn_off(x)
    assert out_on.shape == x.shape
    assert out_off.shape == x.shape
    assert not torch.isnan(out_off).any()


def test_rotated_self_attention_rht_seed_reproducibility():
    attn_a = RotatedSelfAttention(dim=64, num_heads=1, bits=4, quantizer='lattice_d4', rht_seed=7)
    attn_b = RotatedSelfAttention(dim=64, num_heads=1, bits=4, quantizer='lattice_d4', rht_seed=7)
    assert torch.equal(attn_a.rotation.signs, attn_b.rotation.signs)


# ----------------------------- surgery wiring -----------------------------------

class MockCrossAttention(nn.Module):
    """Minimal stand-in matching CrossAttention's bias=False Q/K/V convention
    (same pattern as tests/test_temporal_equivalence.py's mock)."""
    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=True), nn.Dropout(0.0)])

    def forward(self, x):
        return x


class _TinyModelT10(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross1 = MockCrossAttention()
        self.cross2 = MockCrossAttention()


def test_surgery_no_rotation_propagates_to_all_layers():
    model = _TinyModelT10()
    model = apply_rotated_quantization_to_vda(
        model, bits=4, quantizer='lattice_d4', use_qjl=False,
        replace_backbone=False, replace_temporal=True, verbose=False,
        use_rotation=False,
    )
    assert model.cross1.rotation.identity is True
    assert model.cross2.rotation.identity is True


def test_surgery_rht_seed_reproducible_but_distinct_per_layer():
    """Same base seed -> the WHOLE model's rotation state is byte-identical
    across two independent surgery runs (reproducibility). Within ONE run,
    different layers get DIFFERENT (seed-derived) sign vectors -- matching
    the original unseeded behavior's per-layer independence, not literally
    the same rotation repeated at every layer."""
    model1 = _TinyModelT10()
    model1 = apply_rotated_quantization_to_vda(
        model1, bits=4, quantizer='lattice_d4', use_qjl=False,
        replace_backbone=False, replace_temporal=True, verbose=False,
        rht_seed=100,
    )
    model2 = _TinyModelT10()
    model2 = apply_rotated_quantization_to_vda(
        model2, bits=4, quantizer='lattice_d4', use_qjl=False,
        replace_backbone=False, replace_temporal=True, verbose=False,
        rht_seed=100,
    )

    # Reproducible across independent runs with the same base seed.
    assert torch.equal(model1.cross1.rotation.signs, model2.cross1.rotation.signs)
    assert torch.equal(model1.cross2.rotation.signs, model2.cross2.rotation.signs)
    # But distinct WITHIN one run (different layers, different derived seeds).
    assert not torch.equal(model1.cross1.rotation.signs, model1.cross2.rotation.signs)
    print("  [OK] --rht-seed is reproducible across runs, distinct across layers within a run")


def test_surgery_no_seed_still_works_unseeded():
    """rht_seed=None (default) must not crash and must NOT force identical
    signs across layers (preserves original unseeded independent-draw behavior)."""
    model = _TinyModelT10()
    model = apply_rotated_quantization_to_vda(
        model, bits=4, quantizer='lattice_d4', use_qjl=False,
        replace_backbone=False, replace_temporal=True, verbose=False,
    )
    assert model.cross1.rotation.identity is False
    assert not torch.equal(model.cross1.rotation.signs, model.cross2.rotation.signs)


# ----------------------------- scalar accounting fix -----------------------------

def test_resolve_group_size_scalar_uses_head_dim():
    assert resolve_group_size('scalar', 64) == 64
    assert resolve_group_size('scalar', 128) == 128


def test_resolve_group_size_other_quantizers_unchanged():
    assert resolve_group_size('lattice_d4', 64) == 4
    assert resolve_group_size('lattice_e8', 64) == 8
    assert resolve_group_size('uniform_vector', 64) == 4


def test_scalar_not_hardcoded_in_group_size_dict():
    """'scalar' must NOT be a fixed entry in QUANTIZER_GROUP_SIZE -- its
    correct group_size depends on head_dim (one global scale per whole
    vector), which resolve_group_size handles; a fixed dict entry would
    silently reintroduce the T10 overcharge bug for a different head_dim."""
    assert 'scalar' not in QUANTIZER_GROUP_SIZE


def test_scalar_bit_accounting_no_longer_overcharged():
    """
    Before T10: group_size=4 (wrong) -> scale_overhead = (64/4)*16 = 256 bits/vec
        -> effective_bits_per_scalar = (192+256)/64 = 7.0 (a 2.15x overcharge
        vs the true cost, since ScalarRoundQuantizer(per_channel=False) computes
        ONE scale for the whole tensor, not one per 4-element group).
    After T10: group_size=64 (correct, one global scale per vector) ->
        scale_overhead = (64/64)*16 = 16 bits/vec
        -> effective_bits_per_scalar = (192+16)/64 = 3.25 exactly.
    """
    result = compute_real_bit_accounting(
        bit_val=3, head_dim=64, use_qjl=False,
        group_size=resolve_group_size('scalar', 64), scale_bits=16,
    )
    print(f"  Fixed scalar accounting @ 3-bit: {result}")
    assert result["scale_overhead_bits_per_vector"] == 16.0, result
    assert result["total_bits_per_vector"] == 208.0, result  # 192 payload + 16 scale
    assert result["effective_bits_per_scalar"] == 3.25, result

    # Explicitly confirm this is NOT the old (overcharged) number.
    old_wrong_result = compute_real_bit_accounting(
        bit_val=3, head_dim=64, use_qjl=False, group_size=4, scale_bits=16,
    )
    assert old_wrong_result["effective_bits_per_scalar"] == 7.0  # the bug, for contrast
    assert result["effective_bits_per_scalar"] < old_wrong_result["effective_bits_per_scalar"]


if __name__ == "__main__":
    test_hadamard_identity_is_true_noop()
    test_hadamard_identity_has_no_signs_buffer()
    test_hadamard_seed_reproducibility()
    test_hadamard_seed_does_not_disturb_global_rng()
    test_hadamard_extra_repr_reflects_identity()
    test_rotation_reduces_quantization_error_on_outlier_data()
    test_rotated_self_attention_use_rotation_flag()
    test_rotated_self_attention_rht_seed_reproducibility()
    test_surgery_no_rotation_propagates_to_all_layers()
    test_surgery_rht_seed_reproducible_but_distinct_per_layer()
    test_surgery_no_seed_still_works_unseeded()
    test_resolve_group_size_scalar_uses_head_dim()
    test_resolve_group_size_other_quantizers_unchanged()
    test_scalar_not_hardcoded_in_group_size_dict()
    test_scalar_bit_accounting_no_longer_overcharged()
    print("All rotation ablation / scalar accounting tests passed.")
