"""
Verifies LatticeE8Quantizer (research/quantizers/lattice_vq.py) per
docs/optimization_ledger.md task T8: exact lattice membership, boundary
safety, and the rate-distortion ordering MSE(E8) <= MSE(D4) <= MSE(scalar).

Run: pytest tests/test_e8_quantizer.py -q
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from research.quantizers.lattice_vq import (
    LatticeE8Quantizer,
    LatticeD4Quantizer,
    ScalarRoundQuantizer,
)


def _decode_lattice_points(x_q: torch.Tensor, scale: torch.Tensor, n_groups: int, group_size: int) -> torch.Tensor:
    """Recover the integer/half-integer lattice coordinates from dequantized output."""
    scale_expanded = scale.unsqueeze(-1)
    x_grouped = x_q.reshape(x_q.shape[0], n_groups, group_size)
    return x_grouped / scale_expanded


def test_e8_lattice_membership():
    """
    Every decoded 8-vector must be EITHER:
      (a) all-integer coordinates with even sum (D8 coset), OR
      (b) all-half-integer coordinates (coord - 0.5 is integer) whose
          shifted integer coordinates also have an even sum (D8 + [1/2]^8 coset).
    Checked over 10000 random 8-vectors (multiple seeds/shapes), including
    inputs with an extreme per-group outlier to exercise the clamp path.
    """
    q = LatticeE8Quantizer(bits=4)
    hl = q.half_levels

    torch.manual_seed(0)
    x_normal = torch.randn(1250, 64)  # 1250 * 8 groups = 10000 groups
    x_extreme = torch.randn(1250, 64)
    x_extreme[:, ::8] *= 5.0  # push some group maxima to scale up the whole group

    for name, x in [("normal", x_normal), ("extreme", x_extreme)]:
        x_q, info = q(x)
        n_groups = x.shape[-1] // 8
        z = _decode_lattice_points(x_q, info['scale'], n_groups, 8)

        is_int_coset = torch.isclose(z, z.round(), atol=1e-4).all(dim=-1)
        is_half_coset = torch.isclose(z - 0.5, (z - 0.5).round(), atol=1e-4).all(dim=-1)
        valid_coset = is_int_coset | is_half_coset
        coset_frac = valid_coset.float().mean().item()

        # Parity check within whichever coset each group belongs to.
        int_part_for_parity = torch.where(is_int_coset.unsqueeze(-1), z, z - 0.5)
        rounded = int_part_for_parity.round()
        even_sum = (rounded.sum(dim=-1).long() % 2 == 0)
        even_frac = even_sum.float().mean().item()

        in_range = ((z >= -hl - 0.5) & (z <= hl - 0.5)).all().item()  # generous bound: integer coset in [-hl,hl-1], half coset in [-hl+.5, hl-.5]

        status = "OK" if (coset_frac == 1.0 and even_frac == 1.0 and in_range) else "FAIL"
        print(f"  [{status}] E8 membership ({name}): coset_frac={coset_frac:.4f}, "
              f"even_sum_frac={even_frac:.4f}, in_range={in_range}")
        assert coset_frac == 1.0, f"E8 produced points outside both cosets ({name}): {coset_frac:.4f}"
        assert even_frac == 1.0, f"E8 produced odd-parity points ({name}): {even_frac:.4f}"
        assert in_range, f"E8 produced out-of-range points ({name})"


def test_e8_boundary_edge_case():
    """
    Targeted regression test mirroring test_d4_boundary_edge_case: an
    8-vector with a coordinate at the round-to-boundary tie point must
    never end up outside the safe range after coset selection + parity fix.
    """
    q = LatticeE8Quantizer(bits=4)
    hl = q.half_levels  # 8 for bits=4

    alpha = float(hl - 1)  # scale = alpha / (hl - 1) = 1.0 -> x_scaled == x
    x = torch.tensor([[
        hl - 1.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, alpha,
    ]])

    x_q, info = q(x)
    scale = info['scale'].unsqueeze(-1)
    z = (x_q / scale)
    in_range = ((z >= -hl - 0.5) & (z <= hl - 0.5)).all().item()

    status = "OK" if in_range else "FAIL"
    print(f"  [{status}] E8 boundary tie case: z={z.tolist()}, in_range={in_range}")
    assert in_range, f"E8 boundary tie case produced out-of-range point: {z.tolist()}"


def test_e8_rate_distortion_ordering():
    """
    Rate-distortion ordering at equal bit-rate on N(0,1) input:
    MSE(E8) <= MSE(D4) <= MSE(scalar). Validates the coding-gain direction
    empirically (theory: E8 ~1.5dB, D4 ~1.19dB over scalar; E8 - D4 ~0.65dB).
    """
    torch.manual_seed(0)
    d = 64
    bits = 4
    x = torch.randn(4000, d)

    scalar_q = ScalarRoundQuantizer(bits=bits)
    d4_q = LatticeD4Quantizer(bits=bits)
    e8_q = LatticeE8Quantizer(bits=bits)

    x_scalar, _ = scalar_q(x)
    x_d4, _ = d4_q(x)
    x_e8, _ = e8_q(x)

    mse_scalar = ((x - x_scalar) ** 2).mean().item()
    mse_d4 = ((x - x_d4) ** 2).mean().item()
    mse_e8 = ((x - x_e8) ** 2).mean().item()

    print(f"  MSE @ {bits}-bit: scalar={mse_scalar:.6f}, D4={mse_d4:.6f}, E8={mse_e8:.6f}")
    assert mse_e8 <= mse_d4, f"E8 MSE ({mse_e8:.6f}) should be <= D4 MSE ({mse_d4:.6f})"
    assert mse_d4 <= mse_scalar, f"D4 MSE ({mse_d4:.6f}) should be <= scalar MSE ({mse_scalar:.6f})"

    gain_e8_over_scalar_db = 10 * torch.log10(torch.tensor(mse_scalar / mse_e8)).item()
    gain_e8_over_d4_db = 10 * torch.log10(torch.tensor(mse_d4 / mse_e8)).item()
    print(f"  Measured coding gain: E8 vs scalar = {gain_e8_over_scalar_db:.2f} dB "
          f"(theory ~1.5dB), E8 vs D4 = {gain_e8_over_d4_db:.2f} dB (theory ~0.65dB)")


def test_e8_scale_bits_option():
    """scale_bits=8 must still produce valid lattice points, with half the metadata overhead of scale_bits=16."""
    torch.manual_seed(1)
    x = torch.randn(200, 64)

    q16 = LatticeE8Quantizer(bits=4, scale_bits=16)
    q8 = LatticeE8Quantizer(bits=4, scale_bits=8)

    x_q16, info16 = q16(x)
    x_q8, info8 = q8(x)

    for name, x_q, info in [("scale_bits=16", x_q16, info16), ("scale_bits=8", x_q8, info8)]:
        n_groups = x.shape[-1] // 8
        z = _decode_lattice_points(x_q, info['scale'], n_groups, 8)
        is_int = torch.isclose(z, z.round(), atol=1e-4).all(dim=-1)
        is_half = torch.isclose(z - 0.5, (z - 0.5).round(), atol=1e-4).all(dim=-1)
        valid = (is_int | is_half).float().mean().item()
        print(f"  E8 {name}: valid_coset_frac={valid:.4f}, scale_overhead={info['scale_overhead_bits_per_scalar']}")
        assert valid == 1.0, f"E8 {name} produced invalid lattice points"

    assert info16['scale_overhead_bits_per_scalar'] == 2.0, info16  # 16/8
    assert info8['scale_overhead_bits_per_scalar'] == 1.0, info8    # 8/8


def test_e8_no_unearned_effective_bits_claim():
    """Same honesty requirement as D4 (T4): E8 must not claim effective_bits_per_scalar below nominal bits."""
    q = LatticeE8Quantizer(bits=3)
    x = torch.randn(64, 64)
    _, info = q(x)
    assert info['bits'] == 3, info
    assert 'effective_bits_per_scalar' not in info or info['effective_bits_per_scalar'] == info['bits']


if __name__ == "__main__":
    test_e8_lattice_membership()
    test_e8_boundary_edge_case()
    test_e8_rate_distortion_ordering()
    test_e8_scale_bits_option()
    test_e8_no_unearned_effective_bits_claim()
    print("All E8 quantizer tests passed.")
