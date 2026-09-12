"""
VDA integration proof: the fused BW16 attention kernel produces the same
output tensor as VDA's current simulated-quantiser attention path.

We construct the exact operations RotatedTemporalAttention does after
the QKV projections:

    K_rot = rotation(K), V_rot = rotation(V), Q_rot = rotation(Q)
    K_q = simulator(K_rot)                   fp16 reconstruction
    V_q = simulator(V_rot)                   fp16 reconstruction
    scores = Q_rot @ K_q^T * scale
    probs  = softmax(scores)
    out    = probs @ V_q

The fused replacement is:

    K_rot = rotation(K), V_rot = rotation(V), Q_rot = rotation(Q)
    packed_K = pack_bw16(K_rot)               real int bytes
    packed_V = pack_bw16(V_rot)               real int bytes
    out = fused_attention_bw16(Q_rot, packed_K, packed_V)

Correctness gate: the two outputs must agree to within fp round-off.
This is the "the fused kernel is a drop-in for VDA temporal attention"
proof.
"""
from __future__ import annotations

from pathlib import Path
import math
import sys

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kernels.reference.packed_bw16 import pack_bw16, unpack_bw16
from kernels.reference.bw16_codebook import bw16_cosets
from kernels.triton_kernels.fused_attention import fused_attention_bw16
from research.quantizers.lattice_vq import LatticeBW16Quantizer
from research.models.rotated_attention import HadamardRotation


def _rand(*shape, seed=0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=g).to(device)


def _simulator_attention(Q_rot, K_rot, V_rot, bits, scale_bits, group_size):
    """The path RotatedTemporalAttention runs today."""
    q_bw16 = LatticeBW16Quantizer(bits=bits, group_size=group_size,
                                  scale_bits=scale_bits)
    K_q, _ = q_bw16(K_rot)                                     # fp32
    V_q, _ = q_bw16(V_rot)
    D = Q_rot.shape[-1]
    scores = Q_rot @ K_q.transpose(-2, -1) * (1.0 / math.sqrt(D))
    probs = F.softmax(scores, dim=-1)
    return probs @ V_q


def _fused_attention(Q_rot, K_rot, V_rot, codebook, bits, scale_bits, group_size):
    """The fused-kernel path."""
    packed_K = pack_bw16(K_rot, bits=bits, group_size=group_size,
                         scale_bits=scale_bits)
    packed_V = pack_bw16(V_rot, bits=bits, group_size=group_size,
                         scale_bits=scale_bits)
    return fused_attention_bw16(Q_rot.half(), packed_K, packed_V, codebook,
                                bits=bits)


def test_vda_head_dim_64():
    """Standard VDA-S ViT-S mm2 head dim path."""
    if not torch.cuda.is_available():
        return
    device = "cuda"
    N, D = 512, 64                                            # smaller than mm2 for speed
    codebook = bw16_cosets(dtype=torch.float32, device=device)
    Q = _rand(N, D, seed=0, device=device)
    K = _rand(N, D, seed=1, device=device)
    V = _rand(N, D, seed=2, device=device)

    rot = HadamardRotation(dim=D, seed=0).to(device)
    Q_rot = rot(Q); K_rot = rot(K); V_rot = rot(V)

    out_sim = _simulator_attention(Q_rot, K_rot, V_rot, bits=3, scale_bits=8, group_size=16)
    out_fused = _fused_attention(Q_rot, K_rot, V_rot, codebook,
                                 bits=3, scale_bits=8, group_size=16)

    diff = (out_sim - out_fused).abs()
    print(f"    head_dim=64: max={diff.max().item():.2e}  mean={diff.mean().item():.2e}  "
          f"ref_range=[{out_sim.min().item():.3f}, {out_sim.max().item():.3f}]")
    # Attention outputs are bounded; small fp round-off + fp16 cast in
    # the fused path give <1e-2 max diff.
    assert diff.max().item() < 5e-3, f"head_dim=64 max diff {diff.max():.2e}"


def test_vda_head_dim_192():
    """ViT-S mm0 head dim path."""
    if not torch.cuda.is_available():
        return
    device = "cuda"
    N, D = 256, 192
    codebook = bw16_cosets(dtype=torch.float32, device=device)
    Q = _rand(N, D, seed=10, device=device)
    K = _rand(N, D, seed=11, device=device)
    V = _rand(N, D, seed=12, device=device)

    rot = HadamardRotation(dim=D, seed=0).to(device)
    Q_rot = rot(Q); K_rot = rot(K); V_rot = rot(V)

    out_sim = _simulator_attention(Q_rot, K_rot, V_rot, bits=3, scale_bits=8, group_size=16)
    out_fused = _fused_attention(Q_rot, K_rot, V_rot, codebook,
                                 bits=3, scale_bits=8, group_size=16)

    diff = (out_sim - out_fused).abs()
    print(f"    head_dim=192: max={diff.max().item():.2e}  mean={diff.mean().item():.2e}")
    assert diff.max().item() < 5e-3


def test_vda_head_dim_384():
    """ViT-S mm1 head dim path."""
    if not torch.cuda.is_available():
        return
    device = "cuda"
    N, D = 256, 384
    codebook = bw16_cosets(dtype=torch.float32, device=device)
    Q = _rand(N, D, seed=20, device=device)
    K = _rand(N, D, seed=21, device=device)
    V = _rand(N, D, seed=22, device=device)

    rot = HadamardRotation(dim=D, seed=0).to(device)
    Q_rot = rot(Q); K_rot = rot(K); V_rot = rot(V)

    out_sim = _simulator_attention(Q_rot, K_rot, V_rot, bits=3, scale_bits=8, group_size=16)
    out_fused = _fused_attention(Q_rot, K_rot, V_rot, codebook,
                                 bits=3, scale_bits=8, group_size=16)

    diff = (out_sim - out_fused).abs()
    print(f"    head_dim=384: max={diff.max().item():.2e}  mean={diff.mean().item():.2e}")
    assert diff.max().item() < 5e-3


if __name__ == "__main__":
    tests = [
        test_vda_head_dim_64,
        test_vda_head_dim_192,
        test_vda_head_dim_384,
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
