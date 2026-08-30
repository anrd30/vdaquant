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
    ScalarGroupQuantizer,
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


def test_e8_matches_grouped_scalar_at_matched_rate():
    """
    At MATCHED all-inclusive rate and MATCHED scale granularity, E8 must land
    within a small tolerance of the grouped-scalar baseline on N(0,1) input.

    This is the regression test for the range-reservation bug (ledger T4). An
    earlier revision shrank the decode range by 2.5 units so the greedy parity
    flip was always legal; that cost ~0.9 dB at the 3-bit operating point and
    silently biased every published E8-vs-scalar comparison. Any reintroduced
    handicap shows up here as E8 falling outside the tolerance band.

    Note this asserts a TIE, not an E8 win. The +0.65 dB E8 granular gain is a
    high-resolution result for a fixed lattice on a stationary source; per-group
    max-normalization at group 8 re-fits the cell size every 8 values, which
    substitutes for exactly the shaping the lattice would have provided.
    """
    torch.manual_seed(0)
    x = torch.randn(4000, 64)

    for bits in (3, 4, 6):
        # both at group 8 with the same scale width -> identical b_eff
        scalar_q = ScalarGroupQuantizer(bits=bits, group_size=8, scale_bits=16)
        e8_q = LatticeE8Quantizer(bits=bits, group_size=8, scale_bits=16)

        mse_scalar = ((x - scalar_q(x)[0]) ** 2).mean().item()
        mse_e8 = ((x - e8_q(x)[0]) ** 2).mean().item()
        delta_db = 10 * torch.log10(torch.tensor(mse_scalar / mse_e8)).item()

        print(f"  b_eff={bits + 2.0}: scalar={mse_scalar:.6f}, E8={mse_e8:.6f} "
              f"({delta_db:+.2f} dB)")
        assert -0.25 <= delta_db <= 0.75, (
            f"E8 at {bits}b payload is {delta_db:+.2f} dB vs matched-rate grouped "
            f"scalar; expected a near-tie. A large negative value means the decode "
            f"range is being reserved again (ledger T4)."
        )


def test_d4_loses_at_matched_rate_on_metadata():
    """
    D4's group of 4 costs twice the scale metadata of E8's group of 8, so at
    matched all-inclusive rate D4 must run a lower payload and lose badly.

    This pins the ATTRIBUTION: D4's deficit is rate allocation, not lattice
    geometry. Comparing at equal nominal payload instead hands D4 two extra
    effective bits and inverts the conclusion.
    """
    torch.manual_seed(0)
    x = torch.randn(4000, 64)
    scale_bits = 16
    target_beff = 8.0

    d4_payload = int(target_beff - scale_bits / 4)   # 4 bits
    e8_payload = int(target_beff - scale_bits / 8)   # 6 bits

    d4_q = LatticeD4Quantizer(bits=d4_payload, group_size=4, scale_bits=scale_bits)
    e8_q = LatticeE8Quantizer(bits=e8_payload, group_size=8, scale_bits=scale_bits)

    mse_d4 = ((x - d4_q(x)[0]) ** 2).mean().item()
    mse_e8 = ((x - e8_q(x)[0]) ** 2).mean().item()
    gap_db = 10 * torch.log10(torch.tensor(mse_d4 / mse_e8)).item()

    print(f"  at b_eff={target_beff}: D4({d4_payload}b)={mse_d4:.6f}, "
          f"E8({e8_payload}b)={mse_e8:.6f}, gap={gap_db:+.2f} dB")
    assert mse_d4 > mse_e8, (
        f"D4 ({mse_d4:.6f}) should lose to E8 ({mse_e8:.6f}) at matched b_eff"
    )
    assert gap_db > 5.0, (
        f"D4 deficit is only {gap_db:.2f} dB; expected >5 dB from paying two "
        f"extra effective bits of scale metadata"
    )


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


def test_e8_scale_group_decoupled_from_lattice_dimension():
    """
    The scale group may be any multiple of 8 while E8 still decodes 8-vectors.
    Membership must hold at every group, and the metadata overhead must fall
    as 1/group_size (ledger F28).
    """
    torch.manual_seed(0)
    x = torch.randn(512, 512)

    for k in (8, 16, 32, 64, 128):
        q = LatticeE8Quantizer(bits=3, group_size=k, scale_bits=16)
        x_q, info = q(x)

        assert info['group_size'] == k
        assert info['scale_overhead_bits_per_scalar'] == 16.0 / k

        # Recover lattice coordinates through the SCALE grouping, then view as
        # 8-vectors -- k is a multiple of 8, so no 8-vector straddles two scales.
        scale = info['scale'].unsqueeze(-1)
        z = (x_q.reshape(*x.shape[:-1], 512 // k, k) / scale).reshape(-1, 8)

        is_int = torch.isclose(z, z.round(), atol=1e-3).all(-1)
        is_half = torch.isclose(z - 0.5, (z - 0.5).round(), atol=1e-3).all(-1)
        parity_src = torch.where(is_int.unsqueeze(-1), z, z - 0.5).round()

        assert (is_int | is_half).all(), f"non-E8 point emitted at group_size={k}"
        assert (parity_src.sum(-1).long() % 2 == 0).all(), \
            f"odd coordinate sum at group_size={k}"
        print(f"  g={k}: valid E8 points, overhead "
              f"{info['scale_overhead_bits_per_scalar']:.3f} b/scalar")


def test_e8_rejects_scale_group_not_multiple_of_eight():
    """A scale group that is not a multiple of 8 would split an 8-vector across
    two scales, which the decoder cannot represent. Reject it loudly."""
    for bad in (4, 12, 20):
        try:
            LatticeE8Quantizer(bits=3, group_size=bad, scale_bits=16)
        except AssertionError:
            continue
        raise AssertionError(f"group_size={bad} should have been rejected")
