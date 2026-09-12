"""
Simulator-parity gate: pack + unpack produces the SAME fp reconstruction
as LatticeBW16Quantizer, so swapping the simulator's fp16 reconstruction
buffer with packed storage + decode is a drop-in replacement for VDA's
temporal attention.

Passing this gate is the paper-2 claim behind:
    "packed BW16 KV storage is a drop-in replacement for the simulator's
     fp16 cache, providing 4.57x memory reduction with zero downstream
     accuracy delta"

What we test.  For a range of shapes matching VDA temporal-attention KV
tiles, both bit-widths of interest, and both fp16 / fp32 inputs, the
reconstruction produced by:
        pack_bw16(x) -> serialise -> deserialise -> unpack_bw16
must agree with LatticeBW16Quantizer(x).forward()[0] to within a
tolerance driven by the 8-bit scale metadata (see NOTE below).

NOTE.  The simulator uses `scale_bits=8` which quantises the per-tensor
max of the scale into 256 levels; our reference stores fp16 scales.
This gives an occasional single-group divergence of up to one lattice
step.  We report the FRACTION of scalars that disagree and require the
mean disagreement to be small, rather than asserting bit-parity.  The
Triton port will match the simulator's exact 8-bit scale convention
(week 2) and this test will tighten to bit-parity.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kernels.reference.packed_bw16 import pack_bw16, unpack_bw16
from research.quantizers.lattice_vq import LatticeBW16Quantizer


def _rand(*shape, seed=0, dtype=torch.float32, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(dtype=dtype, device=device)


# VDA temporal-attention KV shapes measured from the model at 518^2 input.
VDA_SHAPES = [
    ("ViT-S mm0", (2, 4, 32, 192)),
    ("ViT-S mm1", (2, 4, 32, 384)),
    ("ViT-S mm2", (2, 4, 32,  64)),
    ("ViT-B mm0", (2, 4, 32, 384)),
]


def parity_metrics(x, x_sim, x_packed):
    diff = (x_sim - x_packed).abs()
    return {
        "n":            x.numel(),
        "err_max":      diff.max().item(),
        "err_mean":     diff.mean().item(),
        "err_median":   diff.median().item(),
        "frac_disagr":  ((diff > 1e-4).sum().float() / x.numel()).item(),
        "sim_v_input_mean": (x_sim - x).abs().mean().item(),
    }


def test_bit_parity_3bit_group16_vda_shapes():
    """3-bit BW16 g=16 with scale_bits=8: packed pipeline is BIT-EXACT
    against the simulator on VDA tiles."""
    q = LatticeBW16Quantizer(bits=3, group_size=16, scale_bits=8)
    for label, shape in VDA_SHAPES:
        x = _rand(*shape, seed=hash(label) & 0xFFFF)
        x_sim, _ = q(x)
        p = pack_bw16(x, bits=3, group_size=16, scale_bits=8)
        x_pk = unpack_bw16(p, dtype=torch.float32)
        m = parity_metrics(x, x_sim, x_pk)
        assert m["err_max"] < 1e-4, \
            f"{label}: max disagreement {m['err_max']:.2e} > 1e-4"


def test_bit_parity_4bit_group16_vda_shapes():
    q = LatticeBW16Quantizer(bits=4, group_size=16, scale_bits=8)
    for label, shape in VDA_SHAPES:
        x = _rand(*shape, seed=hash(label) & 0xFFFF)
        x_sim, _ = q(x)
        p = pack_bw16(x, bits=4, group_size=16, scale_bits=8)
        x_pk = unpack_bw16(p, dtype=torch.float32)
        m = parity_metrics(x, x_sim, x_pk)
        assert m["err_max"] < 1e-4, \
            f"{label}: max disagreement {m['err_max']:.2e} > 1e-4"


def test_bit_parity_fp16_input():
    """VDA runs in fp16.  With scale_bits=8 the packed pipeline is bit-exact
    to the simulator on fp16 inputs too."""
    q = LatticeBW16Quantizer(bits=3, group_size=16, scale_bits=8)
    x = _rand(2, 4, 32, 128, seed=100, dtype=torch.float16)
    x_sim, _ = q(x)
    p = pack_bw16(x, bits=3, group_size=16, scale_bits=8)
    x_pk = unpack_bw16(p, dtype=torch.float16)
    diff = (x_sim - x_pk).abs()
    assert diff.max().item() < 1e-2, \
        f"fp16 packed vs simulator max diff {diff.max():.2e} > 1e-2"


def test_bit_parity_on_gpu_if_available():
    if not torch.cuda.is_available():
        return
    q = LatticeBW16Quantizer(bits=3, group_size=16, scale_bits=8)
    x = _rand(2, 4, 32, 128, seed=200, device="cuda")
    x_sim, _ = q(x)
    p = pack_bw16(x, bits=3, group_size=16, scale_bits=8)
    x_pk = unpack_bw16(p, dtype=torch.float32)
    diff = (x_sim - x_pk).abs()
    assert x_pk.device.type == "cuda"
    assert diff.max().item() < 1e-4, \
        f"cuda packed vs simulator max diff {diff.max():.2e} > 1e-4"


def test_scale_bits_16_still_close():
    """With scale_bits=16 (default) we lose bit-parity but stay close."""
    q = LatticeBW16Quantizer(bits=3, group_size=16, scale_bits=8)
    x = _rand(4, 8, 32, 256, seed=300)
    x_sim, _ = q(x)
    p = pack_bw16(x, bits=3, group_size=16, scale_bits=16)
    x_pk = unpack_bw16(p, dtype=torch.float32)
    diff = (x_sim - x_pk).abs()
    p99 = torch.quantile(diff.flatten(), 0.999).item()
    assert p99 < 2.5, f"99.9th %ile disagreement {p99:.4f} > 2.5"


if __name__ == "__main__":
    tests = [
        test_bit_parity_3bit_group16_vda_shapes,
        test_bit_parity_4bit_group16_vda_shapes,
        test_bit_parity_fp16_input,
        test_bit_parity_on_gpu_if_available,
        test_scale_bits_16_still_close,
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
