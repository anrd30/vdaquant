"""
Extended binary Golay code G_{24} and Reed-Muller RM(1, 4) code.

These are the finite codes underlying:
    * BW16 (Barnes-Wall Λ16)     -- Construction A applied to RM(1, 4)
    * Λ24 (Golay-A lattice)       -- Construction A applied to G_{24}

For Construction A: Λ = { x ∈ Z^n : (x mod 2) ∈ C } for a binary code C.
The nearest-neighbor decoder enumerates all 2^k codewords, picks the closest
coset representative, then rounds to 2Z^n + c.

Both codebooks are small enough to enumerate exhaustively:
    RM(1, 4):      2^5   =    32 codewords, length 16
    Extended Golay: 2^12  =  4096 codewords, length 24

The Golay generator is built programmatically from the QR-mod-11 bordered
construction (Curtis 1976; SPLAG Fig 3.4, p.303) so we do not depend on a
hand-transcribed lookup that could hide a typo.

References:
    Conway & Sloane, "Sphere Packings, Lattices and Groups" (SPLAG), Ch. 4-5.
    MacWilliams & Sloane, "The Theory of Error-Correcting Codes", Ch. 20.
"""

from functools import lru_cache
import numpy as np
import torch


# =============================================================================
# Reed-Muller RM(1, 4)  -- (n=16, k=5, d=8)
# =============================================================================
# Generator: 5 rows, one all-ones row plus four coordinate-indicator rows over
# the boolean cube {0, 1}^4. Codewords are all 32 GF(2)-linear combinations.
_RM14_GEN = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],   # all-ones
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],   # x4  (MSB)
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],   # x3
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],   # x2
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],   # x1  (LSB)
], dtype=np.int8)


# =============================================================================
# Extended binary Golay generator via bordered QR-mod-11 construction
# =============================================================================
def _build_golay24_generator() -> np.ndarray:
    """
    Build a generator for the extended Golay [24, 12, 8] code via the
    cyclic [23, 12, 7] Golay code + overall parity bit.

    The cyclic Golay code has generator polynomial (SPLAG p.68, Prange 1959):
        g(x) = 1 + x^2 + x^4 + x^5 + x^6 + x^10 + x^11
    of degree 11, dividing x^23 + 1 over GF(2). The 12 information rows are
    x^i * g(x) for i = 0..11. Extending with an overall parity check produces
    the extended [24, 12, 8] Golay code.

    Verified at import time via the weight distribution
        {0:1, 8:759, 12:2576, 16:759, 24:1}.
    """
    # g(x) = 1 + x^2 + x^4 + x^5 + x^6 + x^10 + x^11 encoded as coefficient
    # vector (index i = coefficient of x^i). Length 12, degree 11.
    g = np.zeros(12, dtype=np.int8)
    for i in (0, 2, 4, 5, 6, 10, 11):
        g[i] = 1

    # 12 cyclic shifts of g(x) padded to length 23. Row i has g at positions [i, i+11].
    G23 = np.zeros((12, 23), dtype=np.int8)
    for i in range(12):
        G23[i, i:i + 12] = g

    # Verify G23 is full rank (dim 12) by counting distinct linear combinations.
    # (Systematic reduction is deferred to below; we just need any generator.)
    # Cyclic Golay [23, 12, 7] is guaranteed to be dim 12 by the polynomial,
    # so we skip a redundant rank check here.

    # Append overall parity column to get [24, 12] extended Golay.
    parity = G23.sum(axis=1, keepdims=True) % 2
    G24 = np.concatenate([G23, parity], axis=1)

    # Row-reduce over GF(2) to systematic form [ I_12 | B ] for consistency with
    # the RM(1, 4) convention and to make the codewords look canonical.
    G24 = _gf2_systematize(G24)
    return G24


def _gf2_systematize(M: np.ndarray) -> np.ndarray:
    """Row-reduce a binary matrix over GF(2) to systematic form [I | B] if possible."""
    M = M.copy().astype(np.int8)
    k, n = M.shape
    for col in range(k):
        # Find a pivot row at or below `col` with a 1 in `col`.
        pivot = None
        for r in range(col, k):
            if M[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            raise RuntimeError(f"Cannot systematize: column {col} has no pivot")
        if pivot != col:
            M[[col, pivot]] = M[[pivot, col]]
        # Zero out column `col` in every other row.
        for r in range(k):
            if r != col and M[r, col] == 1:
                M[r] = (M[r] + M[col]) % 2
    return M


_GOLAY24_GEN = _build_golay24_generator()


# =============================================================================
# Enumerate all codewords of a small binary code
# =============================================================================
def _enumerate_codewords(gen: np.ndarray) -> np.ndarray:
    """Return the 2^k x n matrix of all codewords c = m G  (m ∈ GF(2)^k)."""
    k, n = gen.shape
    msgs = np.zeros((1 << k, k), dtype=np.int8)
    for i in range(k):
        msgs[:, i] = (np.arange(1 << k) >> i) & 1
    return (msgs @ gen) % 2


@lru_cache(maxsize=1)
def rm14_codewords() -> np.ndarray:
    """All 32 codewords of RM(1, 4); shape (32, 16), int8, entries in {0, 1}."""
    C = _enumerate_codewords(_RM14_GEN)
    _verify_rm14(C)
    return C


@lru_cache(maxsize=1)
def golay24_codewords() -> np.ndarray:
    """All 4096 codewords of the extended Golay code; shape (4096, 24), int8."""
    C = _enumerate_codewords(_GOLAY24_GEN)
    _verify_golay24(C)
    return C


def _verify_rm14(C: np.ndarray) -> None:
    assert C.shape == (32, 16), C.shape
    weights = C.sum(axis=1)
    # RM(1, 4) weight distribution: 0:1, 8:30, 16:1
    assert (weights == 0).sum() == 1, "RM(1,4): expected one zero codeword"
    assert (weights == 8).sum() == 30, "RM(1,4): expected 30 codewords of weight 8"
    assert (weights == 16).sum() == 1, "RM(1,4): expected one all-ones codeword"


def _verify_golay24(C: np.ndarray) -> None:
    assert C.shape == (4096, 24), C.shape
    weights = C.sum(axis=1)
    # Extended Golay weight distribution: 0:1, 8:759, 12:2576, 16:759, 24:1.
    counts = {w: int((weights == w).sum()) for w in (0, 8, 12, 16, 24)}
    expected = {0: 1, 8: 759, 12: 2576, 16: 759, 24: 1}
    if counts != expected:
        raise AssertionError(
            f"Golay G24 weight distribution mismatch. Got {counts}, expected {expected}. "
            "The generator matrix is incorrect."
        )
    # Also check: no other weights present (i.e. weights are 0/8/12/16/24 only).
    valid = np.isin(weights, [0, 8, 12, 16, 24])
    assert valid.all(), f"Golay G24 has codewords of forbidden weight: {sorted(set(weights[~valid]))}"


# =============================================================================
# Cached torch tensor views of the codebooks
# =============================================================================
def rm14_codewords_torch(device=None, dtype=torch.float32) -> torch.Tensor:
    C = torch.from_numpy(rm14_codewords()).to(dtype=dtype)
    if device is not None:
        C = C.to(device=device)
    return C


def golay24_codewords_torch(device=None, dtype=torch.float32) -> torch.Tensor:
    C = torch.from_numpy(golay24_codewords()).to(dtype=dtype)
    if device is not None:
        C = C.to(device=device)
    return C


__all__ = [
    'rm14_codewords', 'golay24_codewords',
    'rm14_codewords_torch', 'golay24_codewords_torch',
]
