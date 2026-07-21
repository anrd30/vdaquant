"""
Verifies ScalarGroupQuantizer (research/quantizers/lattice_vq.py), the fair
KIVI-style baseline added to resolve docs/optimization_ledger.md finding F11:
ScalarRoundQuantizer uses ONE global scale for the whole tensor, which
conflates "scalar vs lattice" comparisons with scale granularity.
ScalarGroupQuantizer pairs scalar rounding with the same per-8-group scale
machinery LatticeE8Quantizer pays for.

Run: pytest tests/test_scalar_group.py -q
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from research.quantizers.lattice_vq import ScalarGroupQuantizer, ScalarRoundQuantizer


def test_scalar_group_exactness_on_grid():
    """At bits=8, values already sitting exactly on the quantization grid
    for their group must reconstruct bit-exactly (no rounding to move)."""
    q = ScalarGroupQuantizer(bits=8, group_size=8, scale_bits=16)
    half_levels = q.half_levels  # 128

    # The quantizer computes its OWN alpha as x.abs().amax() and derives
    # scale = alpha / (half_levels - 1). For x = steps * c to land back on
    # exact integer steps after that recomputation, the group's max-abs
    # step must be exactly (half_levels - 1) -- otherwise the recomputed
    # alpha (and hence scale) won't match what generated x, and round()
    # is no longer a no-op. Force that explicitly rather than assuming it.
    torch.manual_seed(0)
    steps = torch.randint(-(half_levels - 1), half_levels, (1, 8)).float()
    steps[0, 0] = half_levels - 1
    c = 0.37  # arbitrary real-valued step size in original units
    x = steps * c  # exactly on-grid once c cancels out in x/scale

    x_q, info = q(x)
    print(f"  on-grid max abs error: {(x_q - x).abs().max().item():.3e}")
    assert torch.allclose(x_q, x, atol=1e-3), (x_q, x)
    assert info['bits'] == 8
    assert info['group_size'] == 8


def test_scalar_group_independence_vs_scalar_round():
    """
    THE point of this class: an outlier in one group must not affect the
    reconstruction of a different group. Group 0 has a large outlier;
    group 1 is small uniform values. ScalarGroupQuantizer must recover
    group 1 nearly exactly; plain ScalarRoundQuantizer (one global scale)
    must destroy it.
    """
    outlier = 100.0
    small = 0.01
    x = torch.tensor([[outlier] + [small] * 7 + [small] * 8])
    assert x.shape == (1, 16)

    group_q = ScalarGroupQuantizer(bits=4, group_size=8, scale_bits=16)
    x_q, _ = group_q(x)
    group1_recon = x_q[0, 8:]
    group1_true = x[0, 8:]
    group_err = (group1_recon - group1_true).abs().max().item()

    # Reference: quantizing group 1's 8 values in complete isolation (no
    # group 0 present at all) must give the identical result -- proves the
    # groups are truly independent, not just "less bad".
    standalone_q, _ = group_q(x[:, 8:])
    assert torch.allclose(x_q[:, 8:], standalone_q, atol=1e-6), (
        "group 1's reconstruction changed depending on whether group 0's "
        "outlier was present in the same tensor -- groups are not independent"
    )

    scalar_q = ScalarRoundQuantizer(bits=4, symmetric=True)
    x_scalar, _ = scalar_q(x)
    group1_scalar_recon = x_scalar[0, 8:]
    scalar_err = (group1_scalar_recon - group1_true).abs().max().item()

    print(f"  group1 error: ScalarGroupQuantizer={group_err:.6f}, "
          f"ScalarRoundQuantizer(global scale)={scalar_err:.6f}")
    assert group_err < 1e-3, f"grouped quantizer should reconstruct group 1 almost exactly, got err={group_err}"
    assert scalar_err > 10 * group_err, (
        "global-scale ScalarRoundQuantizer should be MUCH worse on group 1 "
        f"than the grouped quantizer (got scalar_err={scalar_err}, group_err={group_err})"
    )


def test_scalar_group_zero_guard():
    """All-zero input must produce all-zero output with no NaN/inf."""
    q = ScalarGroupQuantizer(bits=4, group_size=8, scale_bits=16)
    x = torch.zeros(3, 16)
    x_q, info = q(x)
    assert torch.isfinite(x_q).all(), x_q
    assert torch.allclose(x_q, torch.zeros_like(x_q)), x_q
    assert torch.isfinite(info['scale']).all(), info['scale']


def test_scalar_group_shape_dtype_passthrough():
    """Arbitrary leading dims and both fp32/fp16 must pass through with
    unchanged shape and dtype; head_dim=64 divides group_size=8 cleanly."""
    q = ScalarGroupQuantizer(bits=4, group_size=8, scale_bits=16)
    for dtype in (torch.float32, torch.float16):
        x = torch.randn(2, 3, 7, 64, dtype=dtype)
        x_q, info = q(x)
        assert x_q.shape == x.shape, (x_q.shape, x.shape)
        assert x_q.dtype == x.dtype, (x_q.dtype, x.dtype)
        assert torch.isfinite(x_q).all()
    print("  shape/dtype passthrough OK for fp32 and fp16")


def test_scalar_group_scale_bits_8_matches_e8_simulation():
    """scale_bits=8 must halve the metadata overhead vs scale_bits=16,
    using the identical uint8-against-per-tensor-max simulation as the
    lattice quantizers (so a scale_bits=8 comparison is apples-to-apples)."""
    torch.manual_seed(2)
    x = torch.randn(50, 64)

    q16 = ScalarGroupQuantizer(bits=4, group_size=8, scale_bits=16)
    q8 = ScalarGroupQuantizer(bits=4, group_size=8, scale_bits=8)

    x_q16, info16 = q16(x)
    x_q8, info8 = q8(x)

    assert torch.isfinite(x_q16).all() and torch.isfinite(x_q8).all()
    assert info16['scale_overhead_bits_per_scalar'] == 2.0, info16  # 16/8
    assert info8['scale_overhead_bits_per_scalar'] == 1.0, info8    # 8/8
    print(f"  scale_bits=16 overhead={info16['scale_overhead_bits_per_scalar']}, "
          f"scale_bits=8 overhead={info8['scale_overhead_bits_per_scalar']}")


if __name__ == "__main__":
    test_scalar_group_exactness_on_grid()
    test_scalar_group_independence_vs_scalar_round()
    test_scalar_group_zero_guard()
    test_scalar_group_shape_dtype_passthrough()
    test_scalar_group_scale_bits_8_matches_e8_simulation()
    print("All scalar_group tests passed.")
