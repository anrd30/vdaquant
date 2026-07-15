"""
Verifies the TAE (temporal instability) metric — scripts/run_pareto_benchmark_suite.py
compute_tae() — with pure synthetic tensors. See docs/optimization_ledger.md task T7.

Run: pytest tests/test_tae_metric.py -q
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import torch

from run_pareto_benchmark_suite import compute_tae, _warp_depth_by_flow


def test_tae_zero_for_static_sequence():
    """Identical frames -> TAE == 0.0 exactly."""
    torch.manual_seed(0)
    frame = torch.rand(20, 20) * 5.0 + 1.0
    seq = frame.unsqueeze(0).repeat(6, 1, 1)  # (T=6, H, W), all identical

    result = compute_tae(seq, align=False)
    print(f"  Static sequence: {result}")
    assert result["tae"] == 0.0, result
    assert len(result["per_frame_pair"]) == 5, result


def test_tae_matches_manual_diff_no_align():
    """Without alignment, TAE must equal the manual mean |D_t - D_{t+1}|."""
    torch.manual_seed(1)
    seq = torch.rand(5, 10, 10) * 5.0 + 1.0

    result = compute_tae(seq, align=False)
    manual = torch.stack([
        (seq[t] - seq[t + 1]).abs().mean() for t in range(seq.shape[0] - 1)
    ]).mean().item()
    print(f"  Manual diff: {manual:.6f}, compute_tae: {result['tae']:.6f}")
    assert abs(result["tae"] - manual) < 1e-5, (result["tae"], manual)


def test_tae_alignment_removes_global_scale_drift():
    """
    A pure global affine drift between frames (e.g. scale-invariant depth
    model output drifting frame to frame) should be almost fully removed by
    per-pair affine alignment, giving TAE near 0 even though the raw
    (unaligned) frame-to-frame difference is large.
    """
    torch.manual_seed(2)
    base = torch.rand(15, 15) * 5.0 + 1.0
    # Each frame is an affine transform of the base (simulates benign scale drift,
    # not real flicker/instability).
    seq = torch.stack([2.0 * base + 0.5, 1.5 * base + 1.0, 0.8 * base - 0.2, 3.0 * base])

    result_no_align = compute_tae(seq, align=False)
    result_aligned = compute_tae(seq, align=True)
    print(f"  No align: {result_no_align['tae']:.4f}, aligned: {result_aligned['tae']:.4f}")
    assert result_aligned["tae"] < 1e-3, result_aligned
    assert result_aligned["tae"] < result_no_align["tae"], (result_aligned, result_no_align)


def test_tae_single_frame_is_zero():
    """A single-frame sequence has no frame pairs -> TAE == 0.0, empty per-pair list."""
    seq = torch.rand(1, 8, 8)
    result = compute_tae(seq)
    print(f"  Single frame: {result}")
    assert result["tae"] == 0.0, result
    assert result["per_frame_pair"] == [], result


def test_tae_batched_input():
    """(B, T, H, W) input should average over both batch and frame pairs."""
    torch.manual_seed(3)
    seq = torch.rand(3, 4, 10, 10) * 5.0 + 1.0
    result = compute_tae(seq, align=False)
    print(f"  Batched: {result['tae']:.6f}")
    assert result["tae"] > 0.0
    assert len(result["per_frame_pair"]) == 3  # T-1 = 3 pairs


def test_warp_depth_identity_flow_is_noop():
    """Zero optical flow should leave the depth map unchanged (up to interpolation edge effects)."""
    torch.manual_seed(4)
    depth = torch.rand(2, 16, 16) * 5.0 + 1.0
    zero_flow = torch.zeros(16, 16, 2)
    warped = _warp_depth_by_flow(depth, zero_flow)
    err = (depth - warped).abs().max().item()
    print(f"  Zero-flow warp max error: {err:.6f}")
    assert err < 1e-4, err


def test_tae_with_flow_field():
    """
    A pure horizontal shift between frames should be exactly compensated by
    a matching flow field, giving TAE near 0 (vs. large TAE without flow).
    """
    torch.manual_seed(5)
    H, W = 20, 20
    base = torch.rand(1, H, W) * 5.0 + 1.0
    shift = 3
    shifted = torch.roll(base, shifts=shift, dims=2)  # shift right by `shift` pixels
    seq = torch.cat([base, shifted], dim=0)  # (T=2, H, W)

    # Flow from frame0 -> frame1: content at (x,y) in frame0 moved to (x+shift,y) in
    # frame1, so to WARP frame1 back onto frame0's grid we sample frame1 at (x+shift,y).
    flow = torch.zeros(1, H, W, 2)
    flow[0, :, :, 0] = shift  # dx

    result_no_flow = compute_tae(seq, align=False)
    result_with_flow = compute_tae(seq, flow_fields=flow, align=False)
    print(f"  No flow: {result_no_flow['tae']:.4f}, with flow: {result_with_flow['tae']:.4f}")
    assert result_with_flow["used_flow"] is True
    assert result_with_flow["tae"] < result_no_flow["tae"]


if __name__ == "__main__":
    test_tae_zero_for_static_sequence()
    test_tae_matches_manual_diff_no_align()
    test_tae_alignment_removes_global_scale_drift()
    test_tae_single_frame_is_zero()
    test_tae_batched_input()
    test_warp_depth_identity_flow_is_noop()
    test_tae_with_flow_field()
    print("All TAE metric tests passed.")
