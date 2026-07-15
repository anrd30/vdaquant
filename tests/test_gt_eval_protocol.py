"""
Verifies the affine-invariant ground-truth evaluation protocol
(scripts/datasets_gt.py) with pure synthetic tensors — NO network access,
NO real NYUv2 download. See docs/optimization_ledger.md task T2.

Run: pytest tests/test_gt_eval_protocol.py -q
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import torch

from datasets_gt import affine_align_inverse_depth, compute_gt_depth_metrics


def test_identity_prediction_near_perfect():
    """pred == gt (within the valid range) -> AbsRel ~0, delta1 == 1.0."""
    torch.manual_seed(0)
    gt = torch.rand(64, 64) * 5.0 + 1.0  # uniform in [1, 6], inside (0.1, 10) range
    pred = gt.clone()

    metrics = compute_gt_depth_metrics(pred, gt)
    print(f"  Identity prediction: {metrics}")
    assert metrics["abs_rel"] < 1e-6, metrics
    assert metrics["delta1"] == 1.0, metrics


def test_affine_invariance():
    """
    pred = 2*gt + 3 -> after affine alignment, AbsRel < 1e-4. The fit solves
    for (s, t) such that s*pred + t ~ gt (aligning pred TO gt), so it recovers
    the INVERSE of the pred=2*gt+3 relationship: gt = 0.5*pred - 1.5, i.e.
    s=0.5, t=-1.5 exactly.
    """
    torch.manual_seed(1)
    gt = torch.rand(64, 64) * 5.0 + 1.0
    pred = 2.0 * gt + 3.0

    metrics = compute_gt_depth_metrics(pred, gt)
    print(f"  Affine invariance: {metrics}")
    assert metrics["abs_rel"] < 1e-4, metrics
    assert abs(metrics["affine_scale"] - 0.5) < 1e-3, metrics
    assert abs(metrics["affine_shift"] - (-1.5)) < 1e-3, metrics


def test_gt_range_mask_excludes_out_of_range_pixels():
    """Pixels with gt outside (0.1, 10.0) must be excluded from both fit and eval."""
    torch.manual_seed(2)
    gt = torch.rand(10, 10) * 5.0 + 1.0  # in-range baseline
    pred = gt.clone()

    # Corrupt a block of pixels: push their GT far out of range, and make the
    # prediction wildly wrong there. If the mask fails to exclude them, both
    # abs_rel and n_valid_pixels will reflect the corruption.
    gt_corrupted = gt.clone()
    gt_corrupted[0:3, 0:3] = 50.0     # above the 10.0 max
    gt_corrupted[7:10, 7:10] = 0.01   # below the 0.1 min
    pred_corrupted = pred.clone()
    pred_corrupted[0:3, 0:3] = 999.0
    pred_corrupted[7:10, 7:10] = -999.0

    n_out_of_range = 9 + 9  # two 3x3 blocks
    n_total = 100
    n_expected_valid = n_total - n_out_of_range

    metrics = compute_gt_depth_metrics(pred_corrupted, gt_corrupted, gt_range=(0.1, 10.0))
    print(f"  Range mask: {metrics} (expected n_valid_pixels={n_expected_valid})")
    assert metrics["n_valid_pixels"] == n_expected_valid, metrics
    # Since the in-range pixels are still pred==gt exactly, alignment should
    # still be near-perfect despite the corrupted pixels being present in
    # the tensors (just excluded from fit/eval).
    assert metrics["abs_rel"] < 1e-6, metrics


def test_valid_mask_and_range_mask_combine():
    """An explicit valid_mask (e.g. sensor dropout) further restricts pixels, combined with gt_range."""
    torch.manual_seed(3)
    gt = torch.rand(10, 10) * 5.0 + 1.0
    pred = gt.clone()
    sensor_valid = torch.ones(10, 10, dtype=torch.bool)
    sensor_valid[0:2, :] = False  # simulate a dropout region

    metrics = compute_gt_depth_metrics(pred, gt, valid_mask=sensor_valid, gt_range=(0.1, 10.0))
    print(f"  Combined mask: {metrics}")
    assert metrics["n_valid_pixels"] == 80, metrics  # 100 - 20 dropout rows
    assert metrics["abs_rel"] < 1e-6, metrics


def test_no_valid_pixels_raises():
    """If every pixel is masked out, the function must raise rather than silently return garbage."""
    gt = torch.full((5, 5), 50.0)  # entirely out of the default (0.1, 10.0) range
    pred = gt.clone()
    try:
        compute_gt_depth_metrics(pred, gt)
        raised = False
    except ValueError:
        raised = True
    print(f"  No valid pixels raises ValueError: {raised}")
    assert raised, "compute_gt_depth_metrics should raise ValueError when no pixels are valid"


if __name__ == "__main__":
    test_identity_prediction_near_perfect()
    test_affine_invariance()
    test_gt_range_mask_excludes_out_of_range_pixels()
    test_valid_mask_and_range_mask_combine()
    test_no_valid_pixels_raises()
    print("All GT eval protocol tests passed.")
