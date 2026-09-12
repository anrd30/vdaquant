"""
BW16 (Barnes-Wall Lambda_16) codebook as a small lookup table.

BW16 is Construction A over RM(1,4), the first-order Reed-Muller code of
length 16.  RM(1,4) has 32 codewords over GF(2), so Lambda_16 has 32
cosets modulo 2*Z^16.  Nearest-neighbour decoding in Lambda_16 reduces
to enumerating the 32 cosets, and for each coset rounding (x - c)/2 to
the nearest integer per coordinate.

For packed-int KV storage we need the codebook as a plain fp16/fp32
tensor of shape [32, 16].  This module builds it once and caches it.

The 32 codewords are (in the canonical ordering):
  - 1 all-zero:                       weight 0
  - 30 weight-8 codewords:            all 15 "half-and-half" patterns and
                                      their complements (drawn from the
                                      RM(1,4) generator matrix)
  - 1 all-one:                        weight 16

Weight enumerator: 1, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 1
Total: 32 codewords, as required.

This module cross-checks that against the existing
research/quantizers/_golay_rm.py implementation at import time.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

# Locate the repo root so we can borrow the vetted RM(1,4) generator.
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _rm14_codewords() -> np.ndarray:
    """Return the 32 RM(1,4) codewords as an (32, 16) int8 matrix over {0,1}.

    Delegates to the vetted enumeration in research.quantizers._golay_rm,
    which is already used by the accuracy quantiser and whose weight
    distribution is verified at import time there.
    """
    from research.quantizers._golay_rm import rm14_codewords
    C = rm14_codewords()
    assert C.shape == (32, 16), f"RM(1,4) codebook must be 32x16, got {C.shape}"
    return C.astype(np.int8)


def _verify_weight_enumerator(cwds: np.ndarray) -> None:
    """RM(1,4) has weight enumerator 1 + 30 z^8 + z^16."""
    weights = cwds.sum(axis=1)
    counts = np.bincount(weights, minlength=17)
    expected = np.zeros(17, dtype=np.int64)
    expected[0] = 1
    expected[8] = 30
    expected[16] = 1
    assert np.array_equal(counts, expected), \
        f"RM(1,4) weight distribution wrong: got {counts.tolist()}, want {expected.tolist()}"


# Build and verify at import time — a bad codebook silently breaks everything
# downstream, so we fail loudly here rather than during a benchmark run.
_CODEWORDS_INT8 = _rm14_codewords()
_verify_weight_enumerator(_CODEWORDS_INT8)


def bw16_cosets(dtype: torch.dtype = torch.float32,
                device: torch.device | str = "cpu") -> torch.Tensor:
    """Return the 32 BW16 coset representatives as a (32, 16) tensor.

    These are the 32 RM(1,4) codewords cast to the requested dtype.  Each
    row is the {0,1}^16 pattern that identifies one of the 32 cosets of
    Lambda_16 / (2 Z^16).
    """
    t = torch.from_numpy(_CODEWORDS_INT8.astype(np.float32))
    return t.to(dtype=dtype, device=device)


def bw16_num_cosets() -> int:
    """Number of BW16 cosets == 32."""
    return _CODEWORDS_INT8.shape[0]


def bw16_lattice_dim() -> int:
    """BW16 lattice dimension == 16."""
    return _CODEWORDS_INT8.shape[1]


def bw16_coset_index_bits() -> int:
    """Bits needed to store one coset index: ceil(log2(32)) == 5."""
    return 5


if __name__ == "__main__":
    # Manual smoke test — print the codebook and confirm the invariants.
    cw = bw16_cosets(dtype=torch.float32)
    print(f"BW16 codebook shape: {tuple(cw.shape)}")
    print(f"BW16 codebook dtype: {cw.dtype}")
    print(f"Coset weights (should be [0, 8, ..., 8, 16]):")
    weights = cw.sum(dim=1).long().tolist()
    print(f"  {weights}")
    print(f"Weight distribution (should be {{0: 1, 8: 30, 16: 1}}):")
    from collections import Counter
    print(f"  {dict(sorted(Counter(weights).items()))}")
    print()
    print(f"Coset index needs {bw16_coset_index_bits()} bits.")
    print(f"Lattice dimension: {bw16_lattice_dim()}")
    print(f"Number of cosets: {bw16_num_cosets()}")
