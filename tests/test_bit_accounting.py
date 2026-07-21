"""
Verifies compute_real_bit_accounting() reports all-inclusive effective bit
rates (payload + per-group scale metadata + QJL side-channel), not just the
nominal quantizer payload. See docs/optimization_ledger.md task T1.

Run: pytest tests/test_bit_accounting.py -q
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from run_pareto_benchmark_suite import compute_real_bit_accounting


def test_bit_accounting_with_qjl_audited_config():
    """
    Pinned audit example (docs/optimization_ledger.md finding F1, task T1):
    d=64, group=4, b=3, scale_bits=16, QJL(n_proj=256 explicit, norm=16)
    -> 720 bits/vec, 11.25 effective bits/scalar. n_projections=256 is
    passed explicitly here because it reflects the ORIGINAL (pre-T3)
    default; T3 lowered the actual runtime default to 128 for d=64 (see
    test_bit_accounting_matches_qjl_default below), which changes this
    total if left implicit.
    """
    result = compute_real_bit_accounting(
        bit_val=3, head_dim=64, use_qjl=True, group_size=4, scale_bits=16, norm_bits=16,
        n_projections=256,
    )
    print(f"  With QJL (n_proj=256, audited config): {result}")
    assert result["total_bits_per_vector"] == 720, result
    assert result["effective_bits_per_scalar"] == 11.25, result
    assert result["ratio_vs_fp32"] == 2.8, result
    assert result["ratio_vs_fp16"] == 1.4, result


def test_bit_accounting_without_qjl():
    """Same config, QJL disabled -> 448 bits/vec, 7.0 effective bits/scalar."""
    result = compute_real_bit_accounting(
        bit_val=3, head_dim=64, use_qjl=False, group_size=4, scale_bits=16, norm_bits=16,
    )
    print(f"  Without QJL: {result}")
    assert result["total_bits_per_vector"] == 448, result
    assert result["effective_bits_per_scalar"] == 7.0, result


def test_bit_accounting_matches_qjl_default():
    """
    compute_real_bit_accounting's implicit QJL n_projections default must
    exactly match QJLBiasCorrection's own default (default_qjl_projections),
    so the "honest" accounting can never silently diverge from what the
    runtime module actually costs (docs/optimization_ledger.md T3).
    For d=64 the new default is min(2*64, 128) = 128.
    """
    from research.quantizers.qjl_bias import default_qjl_projections, QJLBiasCorrection

    head_dim = 64
    expected_n_proj = default_qjl_projections(head_dim)
    assert expected_n_proj == 128, expected_n_proj

    qjl = QJLBiasCorrection(dim=head_dim)
    assert qjl.n_proj == expected_n_proj, (
        f"QJLBiasCorrection default n_proj ({qjl.n_proj}) diverged from "
        f"default_qjl_projections ({expected_n_proj})"
    )

    result_implicit = compute_real_bit_accounting(bit_val=3, head_dim=head_dim, use_qjl=True)
    result_explicit = compute_real_bit_accounting(
        bit_val=3, head_dim=head_dim, use_qjl=True, n_projections=expected_n_proj,
    )
    assert result_implicit == result_explicit, (result_implicit, result_explicit)
    print(f"  Default QJL accounting matches runtime module default: {result_implicit}")


def test_bit_accounting_e8_target():
    """d=64, group=8 (E8), b=3, scale_bits=8, no QJL -> exactly 4.0 effective bits/scalar (T8 target)."""
    result = compute_real_bit_accounting(
        bit_val=3, head_dim=64, use_qjl=False, group_size=8, scale_bits=8,
    )
    print(f"  E8 target config: {result}")
    assert result["total_bits_per_vector"] == 256, result
    assert result["effective_bits_per_scalar"] == 4.0, result


def test_bit_accounting_scalar_g8_matched_to_e8():
    """
    F11/S1: scalar_g8 (group=8, same as E8) at b=3, scale_bits=8, no QJL
    -> exactly 4.0 effective bits/scalar -- MATCHED to E8's rate above, so
    an E8-vs-scalar_g8 comparison at this config isolates lattice coding
    gain instead of scale-granularity. At b=4 -> 5.0 effective bits/scalar.
    """
    from run_pareto_benchmark_suite import resolve_group_size

    assert resolve_group_size("scalar_g8", head_dim=64) == 8

    result_3bit = compute_real_bit_accounting(
        bit_val=3, head_dim=64, use_qjl=False,
        group_size=resolve_group_size("scalar_g8", 64), scale_bits=8,
    )
    print(f"  scalar_g8 3-bit: {result_3bit}")
    assert result_3bit["total_bits_per_vector"] == 256, result_3bit
    assert result_3bit["effective_bits_per_scalar"] == 4.0, result_3bit

    result_4bit = compute_real_bit_accounting(
        bit_val=4, head_dim=64, use_qjl=False,
        group_size=resolve_group_size("scalar_g8", 64), scale_bits=8,
    )
    print(f"  scalar_g8 4-bit: {result_4bit}")
    assert result_4bit["total_bits_per_vector"] == 320, result_4bit
    assert result_4bit["effective_bits_per_scalar"] == 5.0, result_4bit


if __name__ == "__main__":
    test_bit_accounting_with_qjl_audited_config()
    test_bit_accounting_without_qjl()
    test_bit_accounting_matches_qjl_default()
    test_bit_accounting_e8_target()
    test_bit_accounting_scalar_g8_matched_to_e8()
    print("All bit-accounting tests passed.")
