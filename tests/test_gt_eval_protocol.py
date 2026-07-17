"""
Verifies the affine-invariant ground-truth evaluation protocol
(scripts/datasets_gt.py) with pure synthetic tensors — NO network access,
NO real dataset download. See docs/optimization_ledger.md T2 / F10.

Two alignment spaces are covered:
  * pred_is_disparity=False  -> classic metric-space linear affine (used when
    the model already outputs metric depth, e.g. the VDA metric checkpoint).
  * pred_is_disparity=True   -> DEFAULT; the model outputs disparity (inverse
    depth, VDA relative model), so we align in disparity space and invert to
    metric depth. This is the fix for the broken KITTI baseline (F10).

Run: pytest tests/test_gt_eval_protocol.py -q
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import torch

from datasets_gt import affine_align_inverse_depth, compute_gt_depth_metrics


# ---------------- metric-space alignment (pred_is_disparity=False) ------------

def test_identity_prediction_near_perfect():
    """metric-space: pred == gt (metric) -> AbsRel ~0, delta1 == 1.0."""
    torch.manual_seed(0)
    gt = torch.rand(64, 64) * 5.0 + 1.0  # metres in [1, 6]
    pred = gt.clone()

    metrics = compute_gt_depth_metrics(pred, gt, pred_is_disparity=False)
    print(f"  Identity (metric space): {metrics}")
    assert metrics["abs_rel"] < 1e-6, metrics
    assert metrics["delta1"] == 1.0, metrics


def test_affine_invariance():
    """
    metric-space: pred = 2*gt + 3 -> after affine alignment, AbsRel < 1e-4.
    The fit solves s*pred + t ~ gt, recovering the inverse relationship
    gt = 0.5*pred - 1.5, i.e. s=0.5, t=-1.5.
    """
    torch.manual_seed(1)
    gt = torch.rand(64, 64) * 5.0 + 1.0
    pred = 2.0 * gt + 3.0

    metrics = compute_gt_depth_metrics(pred, gt, pred_is_disparity=False)
    print(f"  Affine invariance (metric space): {metrics}")
    assert metrics["abs_rel"] < 1e-4, metrics
    assert abs(metrics["affine_scale"] - 0.5) < 1e-3, metrics
    assert abs(metrics["affine_shift"] - (-1.5)) < 1e-3, metrics


def test_gt_range_mask_excludes_out_of_range_pixels():
    """Pixels with gt outside (0.1, 10.0) must be excluded from both fit and eval."""
    torch.manual_seed(2)
    gt = torch.rand(10, 10) * 5.0 + 1.0
    pred = gt.clone()

    gt_corrupted = gt.clone()
    gt_corrupted[0:3, 0:3] = 50.0
    gt_corrupted[7:10, 7:10] = 0.01
    pred_corrupted = pred.clone()
    pred_corrupted[0:3, 0:3] = 999.0
    pred_corrupted[7:10, 7:10] = -999.0

    n_expected_valid = 100 - (9 + 9)

    metrics = compute_gt_depth_metrics(pred_corrupted, gt_corrupted, gt_range=(0.1, 10.0),
                                       pred_is_disparity=False)
    print(f"  Range mask: {metrics} (expected n_valid_pixels={n_expected_valid})")
    assert metrics["n_valid_pixels"] == n_expected_valid, metrics
    assert metrics["abs_rel"] < 1e-6, metrics


def test_valid_mask_and_range_mask_combine():
    """An explicit valid_mask (e.g. sensor dropout) further restricts pixels."""
    torch.manual_seed(3)
    gt = torch.rand(10, 10) * 5.0 + 1.0
    pred = gt.clone()
    sensor_valid = torch.ones(10, 10, dtype=torch.bool)
    sensor_valid[0:2, :] = False

    metrics = compute_gt_depth_metrics(pred, gt, valid_mask=sensor_valid, gt_range=(0.1, 10.0),
                                       pred_is_disparity=False)
    print(f"  Combined mask: {metrics}")
    assert metrics["n_valid_pixels"] == 80, metrics
    assert metrics["abs_rel"] < 1e-6, metrics


def test_no_valid_pixels_raises():
    """If every pixel is masked out, raise rather than return garbage."""
    gt = torch.full((5, 5), 50.0)  # all out of default (0.1, 10.0) range
    pred = gt.clone()
    try:
        compute_gt_depth_metrics(pred, gt, pred_is_disparity=False)
        raised = False
    except ValueError:
        raised = True
    print(f"  No valid pixels raises ValueError: {raised}")
    assert raised, "should raise ValueError when no pixels are valid"


# ---------------- disparity-space alignment (default, the F10 fix) ------------

def test_disparity_perfect_prediction():
    """
    DEFAULT path: a perfect disparity prediction (pred = 1/gt, up to arbitrary
    affine) must recover metric depth almost exactly after disparity-space
    alignment + inversion -> AbsRel ~0, delta1 == 1.0.
    """
    torch.manual_seed(4)
    gt = torch.rand(64, 64) * 79.0 + 1.0  # wide KITTI-like range [1, 80] m
    pred_disp = 1.0 / gt                    # perfect disparity
    # scramble by an arbitrary positive affine in disparity space; alignment must undo it
    pred_disp_scrambled = 3.7 * pred_disp + 0.12

    metrics = compute_gt_depth_metrics(pred_disp_scrambled, gt, gt_range=(1e-3, 80.0))
    print(f"  Disparity perfect: {metrics}")
    assert metrics["abs_rel"] < 1e-4, metrics
    assert metrics["delta1"] == 1.0, metrics


def test_disparity_alignment_beats_metric_on_wide_range():
    """
    Regression test for F10: for a genuine disparity prediction over a WIDE
    depth range (KITTI-like 1-80m), aligning in disparity space must be
    dramatically better than the old metric-space linear fit — reproducing
    exactly the NYU-ok / KITTI-broken differential we observed on Colab.
    """
    torch.manual_seed(5)
    gt = torch.rand(80, 80) * 79.0 + 1.0          # [1, 80] m
    pred_disp = 1.0 / gt + torch.randn(80, 80) * 0.002  # near-perfect disparity + tiny noise

    m_disp = compute_gt_depth_metrics(pred_disp, gt, gt_range=(1e-3, 80.0), pred_is_disparity=True)
    m_metric = compute_gt_depth_metrics(pred_disp, gt, gt_range=(1e-3, 80.0), pred_is_disparity=False)
    print(f"  wide-range delta1: disparity={m_disp['delta1']:.3f}  metric={m_metric['delta1']:.3f}")
    # Disparity-space alignment should be near-perfect; metric-space linear fit
    # of a disparity signal over a wide range should be far worse.
    assert m_disp["delta1"] > 0.95, m_disp
    assert m_disp["delta1"] - m_metric["delta1"] > 0.20, (m_disp["delta1"], m_metric["delta1"])


def test_disparity_handles_negative_aligned_values():
    """
    If the affine fit pushes some aligned disparities <= 0, the clamp-before-invert
    must keep depths finite/positive (no NaN/inf) rather than crashing.
    """
    torch.manual_seed(6)
    gt = torch.rand(32, 32) * 20.0 + 0.5
    pred_disp = 1.0 / gt
    # inject a few outliers that could drag the fit to produce non-positive disparity
    pred_disp[0, 0] = -5.0
    pred_disp[1, 1] = 100.0

    metrics = compute_gt_depth_metrics(pred_disp, gt, gt_range=(0.1, 30.0))
    print(f"  Disparity with outliers: {metrics}")
    assert metrics["abs_rel"] == metrics["abs_rel"], "abs_rel is NaN"  # NaN != NaN
    assert metrics["rmse"] == metrics["rmse"], "rmse is NaN"
    assert 0.0 <= metrics["delta1"] <= 1.0, metrics


if __name__ == "__main__":
    test_identity_prediction_near_perfect()
    test_affine_invariance()
    test_gt_range_mask_excludes_out_of_range_pixels()
    test_valid_mask_and_range_mask_combine()
    test_no_valid_pixels_raises()
    test_disparity_perfect_prediction()
    test_disparity_alignment_beats_metric_on_wide_range()
    test_disparity_handles_negative_aligned_values()
    print("All GT eval protocol tests passed.")
