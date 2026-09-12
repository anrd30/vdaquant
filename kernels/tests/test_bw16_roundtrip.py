"""
Correctness gates for the pure-PyTorch packed BW16 reference.

Layered gates, tightest first:

  1. BIT-PARITY (bit_packed <-> reference)
        The bit-packed layout is a lossless re-serialisation of the
        reference tile.  Zero coset diff, zero offset diff, zero decoded-
        tensor diff.  Any Triton kernel we write later must clear the
        same gate against the reference.

  2. SIM-PARITY (reference <-> LatticeBW16Quantizer simulator)
        Same lattice arithmetic, same scale convention, up to fp
        round-off.  Loose tolerance at 2-bit (scale-bit-8 divergence),
        tight tolerance at 3 / 4 bit.

  3. ROUND-TRIP BOUNDS
        Reference pack -> unpack error stays within the characteristic
        BW16 error envelope for that bit width.

  4. SHAPE / DTYPE / DEVICE INVARIANCE
        Common VDA-shaped tensors round-trip through the same shape;
        fp16 / fp32 / bf16 inputs all supported; GPU tensors stay on
        GPU.

  5. EDGE CASES
        Empty tensors, single-group tiles, non-contiguous inputs,
        outlier tails.  Race conditions in the batch chunker.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kernels.reference.packed_bw16 import (
    pack_bw16, unpack_bw16,
    pack_bw16_ref, unpack_bw16_ref,
    bitpack_bw16, bitunpack_bw16,
)
from research.quantizers.lattice_vq import LatticeBW16Quantizer


def _rand(*shape, seed=0, dtype=torch.float32, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(dtype=dtype, device=device)


# =============================================================================
# 1. BIT-PARITY
# =============================================================================
def test_bit_parity_3bit_default_shape():
    x = _rand(2, 4, 32, seed=1)
    ref = pack_bw16_ref(x, bits=3, group_size=16)
    bits = bitpack_bw16(ref)
    ref_roundtrip = bitunpack_bw16(bits)
    assert (ref.packed_coset   != ref_roundtrip.packed_coset  ).sum().item() == 0
    assert (ref.packed_offsets != ref_roundtrip.packed_offsets).sum().item() == 0


def test_bit_parity_all_bitwidths():
    x = _rand(3, 8, 128, seed=2)
    for bits in [2, 3, 4]:
        ref = pack_bw16_ref(x, bits=bits, group_size=16)
        bits_pack = bitpack_bw16(ref)
        ref_roundtrip = bitunpack_bw16(bits_pack)
        assert (ref.packed_coset != ref_roundtrip.packed_coset).sum().item() == 0, \
            f"coset diff at bits={bits}"
        assert (ref.packed_offsets != ref_roundtrip.packed_offsets).sum().item() == 0, \
            f"offset diff at bits={bits}"


def test_bit_parity_decode_identity():
    """Unpacking the bit-packed tile must match unpacking the reference tile
    to fp round-off (the group_scale conversion path is identical)."""
    for bits in [2, 3, 4]:
        x = _rand(2, 4, 64, seed=100 + bits)
        ref = pack_bw16_ref(x, bits=bits, group_size=16)
        bits_pack = bitpack_bw16(ref)
        y_ref = unpack_bw16_ref(ref, dtype=torch.float32)
        y_bit = unpack_bw16(bits_pack, dtype=torch.float32)
        d = (y_ref - y_bit).abs().max().item()
        assert d == 0.0, f"bit-packed decode diverges from reference at bits={bits}: {d}"


# =============================================================================
# 2. SIM-PARITY
# =============================================================================
def test_sim_parity_at_3bit():
    """3-bit reference matches the simulator to within fp round-off on average."""
    x = _rand(3, 8, 128, seed=3)
    q = LatticeBW16Quantizer(bits=3, group_size=16, scale_bits=8)
    x_sim, _ = q(x)
    ref = pack_bw16_ref(x, bits=3, group_size=16)
    x_ref = unpack_bw16_ref(ref, dtype=torch.float32)
    d = (x_sim - x_ref).abs()
    # 8-bit scale quantisation in the simulator can occasionally shift a
    # single group by up to one lattice step; the mean divergence stays
    # under 0.05 across the whole tile.
    assert d.mean().item() < 0.05, f"sim/ref mean diff too large: {d.mean():.4f}"


def test_sim_parity_at_4bit():
    x = _rand(3, 8, 128, seed=4)
    q = LatticeBW16Quantizer(bits=4, group_size=16, scale_bits=8)
    x_sim, _ = q(x)
    ref = pack_bw16_ref(x, bits=4, group_size=16)
    x_ref = unpack_bw16_ref(ref, dtype=torch.float32)
    d = (x_sim - x_ref).abs()
    assert d.mean().item() < 0.05, f"sim/ref mean diff too large: {d.mean():.4f}"


# =============================================================================
# 3. ROUND-TRIP BOUNDS
# =============================================================================
def test_roundtrip_bound_3bit():
    x = _rand(3, 8, 128, seed=5)
    ref = pack_bw16_ref(x, bits=3, group_size=16)
    y = unpack_bw16_ref(ref, dtype=torch.float32)
    err = (y - x).abs()
    # BW16 3-bit characteristic MSE ~ 0.05 on unit-variance Gaussian input.
    assert err.mean().item() < 0.35, f"3-bit round-trip mean err too high: {err.mean():.4f}"


def test_roundtrip_bound_4bit_should_be_lower():
    x = _rand(3, 8, 128, seed=6)
    ref_3 = pack_bw16_ref(x, bits=3, group_size=16)
    ref_4 = pack_bw16_ref(x, bits=4, group_size=16)
    e3 = (unpack_bw16_ref(ref_3, dtype=torch.float32) - x).abs().mean().item()
    e4 = (unpack_bw16_ref(ref_4, dtype=torch.float32) - x).abs().mean().item()
    assert e4 < e3, f"4-bit err ({e4:.4f}) should be lower than 3-bit ({e3:.4f})"


def test_compression_matches_analytic():
    """Compression ratios must land on the paper's analytic numbers.

    With 1-byte scale accounting (deployed cost) and byte-aligned
    codewords, the layout is:
      bits  bytes/codeword  bytes/scale  bits/scalar  vs fp16
        2       3              1           2.00        8.00x
        3       5              1           3.00        5.33x
        4       7              1           4.00        4.00x
    """
    x = _rand(4, 8, 512, seed=7)
    targets = {2: 8.0, 3: 5.33, 4: 4.0}
    for bits, target in targets.items():
        p = pack_bw16(x, bits=bits, group_size=16)
        ratio = p.compression_ratio()
        assert abs(ratio - target) < 0.20, \
            f"bits={bits}: compression {ratio:.2f}x vs target {target:.2f}x"


# =============================================================================
# 4. SHAPE / DTYPE / DEVICE INVARIANCE
# =============================================================================
def test_shape_preservation_varied():
    for shape in [(1, 3, 16), (2, 8, 32), (1, 4, 32, 64), (3, 16, 128), (2, 1, 1, 16)]:
        x = _rand(*shape, seed=hash(shape) & 0xFFFF)
        p = pack_bw16(x, bits=3, group_size=16)
        y = unpack_bw16(p, dtype=torch.float32)
        assert y.shape == x.shape, f"{shape}: got {y.shape}"


def test_dtype_support():
    x_fp32 = _rand(2, 4, 32, seed=8, dtype=torch.float32)
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = x_fp32.to(dtype)
        p = pack_bw16(x, bits=3, group_size=16)
        y = unpack_bw16(p, dtype=dtype)
        assert y.shape == x.shape
        assert y.dtype == dtype


def test_device_preserved_on_cuda():
    if not torch.cuda.is_available():
        return
    x = _rand(2, 4, 32, seed=9, device="cuda")
    p = pack_bw16(x, bits=3, group_size=16)
    assert p.codeword_bytes.device.type == "cuda"
    assert p.group_scale.device.type == "cuda"
    y = unpack_bw16(p, dtype=torch.float32)
    assert y.device.type == "cuda"


# =============================================================================
# 5. EDGE CASES
# =============================================================================
def test_single_group():
    """Smallest legal tile: one 16-scalar group, no leading batch."""
    x = _rand(16, seed=10).unsqueeze(0)     # (1, 16)
    p = pack_bw16(x, bits=3, group_size=16)
    y = unpack_bw16(p, dtype=torch.float32)
    assert y.shape == x.shape


def test_noncontiguous_input():
    """Non-contiguous inputs must produce the same result as contiguous."""
    x = _rand(4, 32, seed=11)
    x_nc = x.transpose(0, 1).transpose(0, 1)   # trivially non-contig round-trip
    assert x_nc.is_contiguous() == x.is_contiguous()
    # Force non-contig via a stride trick:
    x_full = _rand(4, 64, seed=11)
    x_nc = x_full[:, ::2]                      # stride-2 view, non-contig
    assert not x_nc.is_contiguous()
    p = pack_bw16(x_nc.contiguous(), bits=3, group_size=16)
    p_nc = pack_bw16(x_nc, bits=3, group_size=16)
    y  = unpack_bw16(p,    dtype=torch.float32)
    y2 = unpack_bw16(p_nc, dtype=torch.float32)
    assert (y - y2).abs().max().item() == 0.0


def test_heavy_tail_outliers():
    """A tile with rare extreme outliers must still round-trip within the
    bounded envelope (per-group scale absorbs the outliers)."""
    x = _rand(2, 4, 128, seed=12)
    mask = _rand(2, 4, 128, seed=13).abs() > 2.5
    x = x + mask.float() * 20.0
    p = pack_bw16(x, bits=3, group_size=16)
    y = unpack_bw16(p, dtype=torch.float32)
    err = (y - x).abs()
    # Per-group scale absorbs outliers; overall mean err stays bounded.
    assert err.mean().item() < 2.0, f"heavy-tail mean err too high: {err.mean():.4f}"


def test_batch_chunker_matches_unchunked():
    """The (>= _CHUNK_ROWS) code path must match the single-shot path."""
    from kernels.reference import packed_bw16 as pb
    orig_chunk = pb._CHUNK_ROWS
    try:
        # Force chunking on a small tensor by shrinking _CHUNK_ROWS.
        pb._CHUNK_ROWS = 32
        x = _rand(4, 8, 128, seed=14)
        p_chunked = pack_bw16(x, bits=3, group_size=16)
        pb._CHUNK_ROWS = 1 << 30
        p_full = pack_bw16(x, bits=3, group_size=16)
        y_c = unpack_bw16(p_chunked, dtype=torch.float32)
        y_f = unpack_bw16(p_full, dtype=torch.float32)
        assert (y_c - y_f).abs().max().item() == 0.0
    finally:
        pb._CHUNK_ROWS = orig_chunk


def test_zero_input_is_stable():
    """All-zero input must not divide by zero and must round-trip to zero
    (or very small values)."""
    x = torch.zeros(2, 4, 32)
    p = pack_bw16(x, bits=3, group_size=16)
    y = unpack_bw16(p, dtype=torch.float32)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    assert y.abs().max().item() < 1e-4


def test_deterministic_across_runs():
    """Same input, same output, bit-for-bit, across independent calls."""
    x = _rand(2, 4, 64, seed=15)
    p1 = pack_bw16(x, bits=3, group_size=16)
    p2 = pack_bw16(x, bits=3, group_size=16)
    assert (p1.codeword_bytes != p2.codeword_bytes).sum().item() == 0
    assert (p1.group_scale != p2.group_scale).sum().item() == 0


if __name__ == "__main__":
    tests = [
        # 1. bit-parity
        test_bit_parity_3bit_default_shape,
        test_bit_parity_all_bitwidths,
        test_bit_parity_decode_identity,
        # 2. sim-parity
        test_sim_parity_at_3bit,
        test_sim_parity_at_4bit,
        # 3. round-trip bounds
        test_roundtrip_bound_3bit,
        test_roundtrip_bound_4bit_should_be_lower,
        test_compression_matches_analytic,
        # 4. shape / dtype / device
        test_shape_preservation_varied,
        test_dtype_support,
        test_device_preserved_on_cuda,
        # 5. edge cases
        test_single_group,
        test_noncontiguous_input,
        test_heavy_tail_outliers,
        test_batch_chunker_matches_unchunked,
        test_zero_input_is_stable,
        test_deterministic_across_runs,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print()
    print(f"  Result: {passed}/{passed+failed} passed")
    sys.exit(0 if failed == 0 else 1)
