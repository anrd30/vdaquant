"""
Verifies scripts/run_blur_degradation.py, and -- more importantly -- tests the
MECHANISM the blur study is designed to demonstrate, on synthetic data with no
model and no GPU: does removing spatial structure lower the unmasked geometric
TAE while co-visibility masking resists that?

If the synthetic test below fails, the real experiment's premise is wrong and
we should find out here rather than from a reviewer.

Run: pytest tests/test_blur_degradation.py -q
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import pytest
import torch

from run_blur_degradation import gaussian_blur2d
from run_pareto_benchmark_suite import _tae_geometric_single, _covisibility_mask


def test_sigma_zero_is_exact_noop():
    """sigma=0 must return the input bit-exactly, so the sigma=0 row of the
    sweep is a true control and not an approximation of one."""
    torch.manual_seed(0)
    x = torch.rand(16, 16)
    assert torch.equal(gaussian_blur2d(x, 0), x)
    assert torch.equal(gaussian_blur2d(x, -1), x)


def test_blur_preserves_shape_and_is_monotone_in_smoothing():
    """Larger sigma must remove more structure (lower gradient energy),
    with shape unchanged."""
    torch.manual_seed(1)
    x = torch.rand(32, 32)

    def grad_energy(t):
        dx = (t[:, 1:] - t[:, :-1]).abs().mean()
        dy = (t[1:, :] - t[:-1, :]).abs().mean()
        return float(dx + dy)

    energies = []
    for s in (0, 1, 2, 4):
        y = gaussian_blur2d(x, s)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        energies.append(grad_energy(y))
    for a, b in zip(energies, energies[1:]):
        assert b <= a + 1e-9, f"gradient energy not monotone under blur: {energies}"
    assert energies[-1] < energies[0] * 0.5, energies


def test_blur_preserves_mean_under_replicate_padding():
    """Replicate padding must not darken the border: a constant map stays
    constant. Zero padding would fail this and inject a fake depth
    discontinuity at the frame edge."""
    x = torch.full((20, 20), 7.5)
    y = gaussian_blur2d(x, 3.0)
    assert torch.allclose(y, x, atol=1e-4), (y.min(), y.max())


def test_blur_lowers_unmasked_tae_but_covisibility_resists():
    """
    THE MECHANISM TEST. Build a scene with real lateral camera motion so that
    part of frame 1 is disoccluded / leaves the frame in frame 2, and give the
    prediction fine spatial structure. Blurring the prediction should:
      (a) LOWER the unmasked geometric TAE  -- the metric "improves" as the
          prediction is destroyed, because there is less structure to
          misalign on the invalid correspondences; and
      (b) lower it by LESS under a co-visibility mask, which excludes exactly
          those invalid pixels.
    All values computed before asserting; the assertion is on the direction
    and on masked-vs-unmasked, not on a hardcoded magnitude.
    """
    H, W = 48, 96
    fx = fy = 60.0
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64)
    T = torch.tensor([0.9, 0.0, 0.0], dtype=torch.float64)  # real lateral motion

    # Ground-truth geometry: a near slab on the left, far background right.
    gt1 = torch.full((H, W), 30.0, dtype=torch.float64)
    gt1[:, : W // 3] = 6.0
    gt2 = torch.full((H, W), 30.0, dtype=torch.float64)
    gt2[:, : W // 4] = 6.0          # slab has shifted -> disocclusion band

    # Prediction: same gross geometry plus fine high-frequency detail, which
    # is what blur will remove.
    torch.manual_seed(0)
    detail = torch.sin(torch.linspace(0, 24 * np.pi, W, dtype=torch.float64))[None, :].repeat(H, 1)
    pred1 = gt1 + 1.5 * detail
    pred2 = gt2 + 1.5 * detail

    ones = torch.ones((H, W), dtype=torch.bool)
    covis = _covisibility_mask(gt1, gt2, R, T, K, tau=0.05)
    assert 0.0 < float(covis.float().mean()) < 1.0, "need a partial mask for this test"

    def tae(p1, p2, mask):
        return _tae_geometric_single(p1, p2, R, T, K, mask)

    sharp_raw = tae(pred1, pred2, ones)
    sharp_cov = tae(pred1, pred2, covis)

    b1 = gaussian_blur2d(pred1.float(), 4.0).double()
    b2 = gaussian_blur2d(pred2.float(), 4.0).double()
    blur_raw = tae(b1, b2, ones)
    blur_cov = tae(b1, b2, covis)

    print(f"  unmasked: sharp={sharp_raw:.5f} -> blurred={blur_raw:.5f}")
    print(f"  masked  : sharp={sharp_cov:.5f} -> blurred={blur_cov:.5f}")

    # (a) blurring "improves" the unmasked metric
    assert blur_raw < sharp_raw, (sharp_raw, blur_raw)

    # (b) the co-visibility mask resists that reward: the unmasked metric is
    # improved by strictly more than the masked one.
    assert (sharp_raw - blur_raw) > (sharp_cov - blur_cov), (
        f"masking should absorb less of the blur reward: "
        f"unmasked gain {sharp_raw - blur_raw:.5f} vs masked {sharp_cov - blur_cov:.5f}")


if __name__ == "__main__":
    test_sigma_zero_is_exact_noop()
    test_blur_preserves_shape_and_is_monotone_in_smoothing()
    test_blur_preserves_mean_under_replicate_padding()
    test_blur_lowers_unmasked_tae_but_covisibility_resists()
    print("All blur-degradation tests passed.")
