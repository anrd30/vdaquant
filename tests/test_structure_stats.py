"""
Verifies scripts/dump_structure_stats.py (S10, docs/optimization_ledger.md
F16): the spatial-structure diagnostic that quantifies the degeneracy
behind the TAE-gameability finding. Pure-function tests only -- no model,
no dataset (per S10 spec).

Run: pytest tests/test_structure_stats.py -q
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dump_structure_stats import (
    _minmax_normalize,
    grad_energy,
    laplacian_var,
    entropy_8bit,
    pred_std,
    compute_structure_stats,
    average_stats,
)


def _gaussian_blur(x: torch.Tensor, kernel_size: int = 9, sigma: float = 3.0) -> torch.Tensor:
    """Simple separable Gaussian blur, torch-only, for building the
    'heavily blurred copy' fixture the spec asks for."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, 1, kernel_size)
    xin = x.unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    xin = F.conv2d(xin, g, padding=(0, pad))
    xin = F.conv2d(xin, g.transpose(-1, -2), padding=(pad, 0))
    return xin[0, 0]


def test_minmax_normalize_range_and_constant_guard():
    torch.manual_seed(0)
    x = torch.rand(20, 20) * 50 - 10  # arbitrary range
    x_norm = _minmax_normalize(x)
    assert abs(x_norm.min().item() - 0.0) < 1e-5
    assert abs(x_norm.max().item() - 1.0) < 1e-5

    const = torch.full((10, 10), 7.5)
    const_norm = _minmax_normalize(const)
    assert torch.isfinite(const_norm).all()
    assert torch.allclose(const_norm, torch.zeros_like(const_norm))


def test_structure_metrics_finite_on_constant_image():
    """A constant image must give grad_energy == 0 and no NaN in any metric."""
    x = torch.full((32, 32), 3.14)
    stats = compute_structure_stats(x)
    for k, v in stats.items():
        assert math.isfinite(v), (k, v)
    assert stats["grad_energy"] == 0.0, stats
    assert stats["laplacian_var"] == 0.0, stats
    assert stats["pred_std"] == 0.0, stats


def test_blurred_image_has_lower_structure_than_sharp_noise():
    """
    Core S10 acceptance test: a sharp random-noise image vs a heavily
    Gaussian-blurred copy of the SAME image. The blurred copy must show
    strictly lower grad_energy, lower laplacian_var, and lower entropy --
    this is the exact direction we expect the degenerate 2-bit prediction
    to move in relative to FP32 (F16). Seeded, deterministic.
    """
    torch.manual_seed(0)
    sharp = torch.rand(64, 64)
    blurred = _gaussian_blur(sharp, kernel_size=9, sigma=3.0)

    sharp_stats = compute_structure_stats(sharp)
    blurred_stats = compute_structure_stats(blurred)

    print(f"  sharp: {sharp_stats}")
    print(f"  blurred: {blurred_stats}")

    assert blurred_stats["grad_energy"] < sharp_stats["grad_energy"]
    assert blurred_stats["laplacian_var"] < sharp_stats["laplacian_var"]
    assert blurred_stats["entropy_8bit"] < sharp_stats["entropy_8bit"]


def test_grad_energy_and_laplacian_var_zero_only_on_flat_input():
    torch.manual_seed(1)
    x = torch.rand(16, 16)
    x_norm = _minmax_normalize(x)
    assert grad_energy(x_norm) > 0.0
    assert laplacian_var(x_norm) > 0.0

    flat = torch.zeros(16, 16)
    assert grad_energy(flat) == 0.0
    assert laplacian_var(flat) == 0.0


def test_entropy_bounds():
    """Entropy of a [0,1]-normalized frame must be in [0, log2(256)]."""
    torch.manual_seed(2)
    x_norm = _minmax_normalize(torch.rand(40, 40))
    h = entropy_8bit(x_norm)
    assert 0.0 <= h <= math.log2(256) + 1e-6, h

    flat = torch.zeros(10, 10)
    assert entropy_8bit(flat) == 0.0  # all mass in one bin -> zero entropy


def test_pred_std_scales_with_spread():
    torch.manual_seed(3)
    narrow = _minmax_normalize(torch.rand(20, 20) * 0.01)
    wide = _minmax_normalize(torch.rand(20, 20))
    # both are min-max normalized to full [0,1] range by construction, so
    # instead compare a genuinely narrow-vs-wide DISTRIBUTION pre-normalization:
    x = torch.zeros(20, 20)
    x[:2, :2] = 1.0  # a few pixels far from the rest -> higher std after norm
    flat = torch.full((20, 20), 0.5)
    assert pred_std(_minmax_normalize(x)) > pred_std(_minmax_normalize(flat))


def test_average_stats_matches_manual_mean():
    frame_stats = [
        {"grad_energy": 1.0, "laplacian_var": 2.0, "entropy_8bit": 3.0, "pred_std": 4.0},
        {"grad_energy": 3.0, "laplacian_var": 4.0, "entropy_8bit": 5.0, "pred_std": 6.0},
    ]
    avg = average_stats(frame_stats)
    assert avg["grad_energy"] == 2.0
    assert avg["laplacian_var"] == 3.0
    assert avg["entropy_8bit"] == 4.0
    assert avg["pred_std"] == 5.0


if __name__ == "__main__":
    test_minmax_normalize_range_and_constant_guard()
    test_structure_metrics_finite_on_constant_image()
    test_blurred_image_has_lower_structure_than_sharp_noise()
    test_grad_energy_and_laplacian_var_zero_only_on_flat_input()
    test_entropy_bounds()
    test_pred_std_scales_with_spread()
    test_average_stats_matches_manual_mean()
    print("All structure-stats tests passed.")
