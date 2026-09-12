"""
Correctness gate: fused_attention_bw16 matches standard PyTorch attention
run on the SAME decoded K, V, up to fp accumulation-order round-off.

Not tested here: bit-parity against a fp16-only simulator path.  The
accumulation order between our decode-fused kernels and standard PyTorch
matmul differs, so the outputs agree only to fp round-off in the score
range.  For end-to-end delta_1 matching on VDA, the accumulator-order
divergence is well under the quantiser noise floor.
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


def _rand(*shape, seed=0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=g).to(device)


def _pytorch_attention(Q, K_packed, V_packed, codebook, bits=3, scale=None):
    """Reference: decode K, V then standard PyTorch attention."""
    K = unpack_bw16(K_packed, dtype=torch.float32)
    V = unpack_bw16(V_packed, dtype=torch.float32)
    D = Q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    scores = Q.float() @ K.t() * scale
    probs = F.softmax(scores, dim=-1)
    return probs @ V


def test_fused_attention_small():
    if not torch.cuda.is_available():
        return
    device = "cuda"
    M, N, D = 64, 128, 64
    Q = _rand(M, D, seed=0, device=device).to(torch.float16)
    K = _rand(N, D, seed=1, device=device)
    V = _rand(N, D, seed=2, device=device)
    pK = pack_bw16(K, bits=3, group_size=16, scale_bits=8)
    pV = pack_bw16(V, bits=3, group_size=16, scale_bits=8)
    codebook = bw16_cosets(dtype=torch.float32, device=device)

    ref = _pytorch_attention(Q, pK, pV, codebook, bits=3)
    fused = fused_attention_bw16(Q, pK, pV, codebook, bits=3)
    diff = (ref - fused).abs()
    print(f"    small (64x128x64): max={diff.max().item():.2e} "
          f"mean={diff.mean().item():.2e} "
          f"ref range=[{ref.min().item():.3f}, {ref.max().item():.3f}]")
    assert diff.max().item() < 1e-2


def test_fused_attention_vda_mm2():
    if not torch.cuda.is_available():
        return
    device = "cuda"
    M, N, D = 512, 512, 64                                # smaller than mm2 to be quick
    Q = _rand(M, D, seed=10, device=device).to(torch.float16)
    K = _rand(N, D, seed=11, device=device)
    V = _rand(N, D, seed=12, device=device)
    pK = pack_bw16(K, bits=3, group_size=16, scale_bits=8)
    pV = pack_bw16(V, bits=3, group_size=16, scale_bits=8)
    codebook = bw16_cosets(dtype=torch.float32, device=device)

    ref = _pytorch_attention(Q, pK, pV, codebook, bits=3)
    fused = fused_attention_bw16(Q, pK, pV, codebook, bits=3)
    diff = (ref - fused).abs()
    print(f"    mid   (512x512x64): max={diff.max().item():.2e} "
          f"mean={diff.mean().item():.2e}")
    assert diff.max().item() < 1e-2


def test_fused_attention_varied_D():
    if not torch.cuda.is_available():
        return
    device = "cuda"
    for D in [64, 128, 256, 384]:
        M, N = 32, 64
        Q = _rand(M, D, seed=20, device=device).to(torch.float16)
        K = _rand(N, D, seed=21, device=device)
        V = _rand(N, D, seed=22, device=device)
        pK = pack_bw16(K, bits=3, group_size=16, scale_bits=8)
        pV = pack_bw16(V, bits=3, group_size=16, scale_bits=8)
        codebook = bw16_cosets(dtype=torch.float32, device=device)
        ref = _pytorch_attention(Q, pK, pV, codebook, bits=3)
        fused = fused_attention_bw16(Q, pK, pV, codebook, bits=3)
        diff = (ref - fused).abs()
        print(f"    D={D:3d}: max={diff.max().item():.2e}")
        assert diff.max().item() < 1e-2, f"D={D} max diff {diff.max():.2e}"


if __name__ == "__main__":
    tests = [
        test_fused_attention_small,
        test_fused_attention_vda_mm2,
        test_fused_attention_varied_D,
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
