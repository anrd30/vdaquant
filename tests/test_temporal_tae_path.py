"""
Verifies the T9 real video-window path + VDA-protocol geometric TAE
(docs/optimization_ledger.md T9) — pure synthetic tensors/fixtures, NO
network access, NO real dataset, NO GPU. Covers:
  - datasets_gt._read_sintel_cam / _cam_to_world_pose (camera file I/O + math)
  - datasets_gt.group_samples_by_scene / chunk_scene_into_windows
  - load_sintel_gt's scene/frame_idx/K/pose fields
  - run_pareto_benchmark_suite._tae_geometric_single /
    _pool_align_scene_disparity / compute_tae_geometric_for_scene

Run: pytest tests/test_temporal_tae_path.py -q
"""
import sys
import os
import struct
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from datasets_gt import (
    _read_sintel_cam,
    _cam_to_world_pose,
    load_sintel_gt,
    group_samples_by_scene,
    chunk_scene_into_windows,
)
from run_pareto_benchmark_suite import (
    _tae_geometric_single,
    _pool_align_scene_disparity,
    compute_tae_geometric_for_scene,
)

requires_pil = pytest.mark.skipif(not HAS_PIL, reason="Pillow not installed")


def _write_dpt(path: Path, arr: np.ndarray):
    h, w = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<f", 202021.25))
        f.write(struct.pack("<i", w))
        f.write(struct.pack("<i", h))
        arr.astype("<f4").tofile(f)


def _write_cam(path: Path, K: np.ndarray, R: np.ndarray, t: np.ndarray):
    """Writes a Sintel .cam file matching the official sintel_io.cam_read format."""
    N = np.concatenate([R, t.reshape(3, 1)], axis=1)  # (3,4)
    with open(path, "wb") as f:
        f.write(struct.pack("<f", 202021.25))
        K.astype("<f8").tofile(f)
        N.astype("<f8").tofile(f)


# ----------------------------- .cam I/O ---------------------------------------

def test_cam_read_roundtrip():
    tmp = Path(tempfile.mkdtemp())
    try:
        K = np.array([[1000.0, 0, 320], [0, 1000.0, 240], [0, 0, 1]])
        R = np.eye(3)
        R[0, 0], R[0, 1] = np.cos(0.3), -np.sin(0.3)
        R[1, 0], R[1, 1] = np.sin(0.3), np.cos(0.3)  # a real rotation (not identity)
        t = np.array([1.5, -0.2, 3.0])
        p = tmp / "frame_0001.cam"
        _write_cam(p, K, R, t)

        K_read, R_read, t_read = _read_sintel_cam(p)
        assert np.allclose(K_read, K), (K_read, K)
        assert np.allclose(R_read, R), (R_read, R)
        assert np.allclose(t_read, t), (t_read, t)
        print("  [OK] .cam round-trip: K, R, t recovered exactly")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cam_read_rejects_bad_magic():
    tmp = Path(tempfile.mkdtemp())
    try:
        p = tmp / "bad.cam"
        with open(p, "wb") as f:
            f.write(struct.pack("<f", 1.0))
            np.zeros(9, dtype="<f8").tofile(f)
            np.zeros(12, dtype="<f8").tofile(f)
        with pytest.raises(RuntimeError, match="not a valid Sintel .cam"):
            _read_sintel_cam(p)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cam_to_world_pose_is_true_inverse():
    """
    T (camera-to-world) applied to X_cam must recover X_world, i.e. T is the
    exact inverse of the world-to-camera map X_cam = R @ X_world + t.
    """
    torch.manual_seed(0)
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    t = np.array([2.0, -1.0, 0.5])

    T = _cam_to_world_pose(R, t)
    assert T.shape == (4, 4)
    assert np.allclose(T[3], [0, 0, 0, 1]), T[3]

    rng = np.random.RandomState(1)
    for _ in range(5):
        X_world = rng.randn(3) * 5.0
        X_cam = R @ X_world + t
        X_world_recovered = T[:3, :3] @ X_cam + T[:3, 3]
        assert np.allclose(X_world_recovered, X_world, atol=1e-10), (X_world_recovered, X_world)
    print("  [OK] cam_to_world_pose is the exact inverse of the world-to-camera extrinsic")


# ----------------------------- scene grouping / windowing ---------------------

def test_group_samples_by_scene_sorts_and_groups():
    samples = [
        {"scene": "b", "frame_idx": 2, "val": "b2"},
        {"scene": "a", "frame_idx": 1, "val": "a1"},
        {"scene": "b", "frame_idx": 1, "val": "b1"},
        {"scene": "a", "frame_idx": 0, "val": "a0"},
    ]
    groups = group_samples_by_scene(samples)
    assert list(groups.keys()) == ["b", "a"], list(groups.keys())  # first-seen order
    assert [s["val"] for s in groups["a"]] == ["a0", "a1"]
    assert [s["val"] for s in groups["b"]] == ["b1", "b2"]
    print("  [OK] group_samples_by_scene groups correctly and sorts by frame_idx")


def test_group_samples_by_scene_rejects_none_scene():
    samples = [{"scene": None, "frame_idx": 0}]
    with pytest.raises(ValueError, match="Cannot group samples with scene=None"):
        group_samples_by_scene(samples)


def test_chunk_scene_into_windows_exact_multiple():
    frames = list(range(8))
    windows = chunk_scene_into_windows(frames, window=4)
    assert len(windows) == 2
    for chunk, n_real in windows:
        assert len(chunk) == 4
        assert n_real == 4
    assert windows[0][0] == [0, 1, 2, 3]
    assert windows[1][0] == [4, 5, 6, 7]


def test_chunk_scene_into_windows_remainder():
    frames = list(range(9))  # 4+4+1
    windows = chunk_scene_into_windows(frames, window=4)
    assert len(windows) == 3
    assert windows[0] == ([0, 1, 2, 3], 4)
    assert windows[1] == ([4, 5, 6, 7], 4)
    last_chunk, n_real = windows[2]
    assert n_real == 1
    assert len(last_chunk) == 4
    assert last_chunk[0] == 8       # the one real frame
    assert last_chunk[1:] == [8, 8, 8]  # padded by repeating the last real frame
    print("  [OK] remainder window static-pads by repeating the last real frame")


def test_chunk_scene_into_windows_scene_shorter_than_window():
    frames = ["x", "y", "z"]
    windows = chunk_scene_into_windows(frames, window=16)
    assert len(windows) == 1
    chunk, n_real = windows[0]
    assert n_real == 3
    assert len(chunk) == 16
    assert chunk[:3] == ["x", "y", "z"]
    assert all(f == "z" for f in chunk[3:])


def test_chunk_scene_into_windows_empty_and_invalid():
    assert chunk_scene_into_windows([], window=4) == []
    with pytest.raises(ValueError, match="window must be positive"):
        chunk_scene_into_windows([1, 2, 3], window=0)
    with pytest.raises(ValueError, match="window must be positive"):
        chunk_scene_into_windows([1, 2, 3], window=-1)


# ----------------------------- Sintel loader with/without cam -----------------

@requires_pil
def test_sintel_loader_populates_scene_frame_idx_K_pose():
    tmp = Path(tempfile.mkdtemp())
    try:
        depth_root = tmp / "training" / "depth" / "alley_1"
        rgb_root = tmp / "training" / "clean" / "alley_1"
        cam_root = tmp / "training" / "camdata_left" / "alley_1"
        depth_root.mkdir(parents=True)
        rgb_root.mkdir(parents=True)
        cam_root.mkdir(parents=True)

        H, W = 6, 10
        rng = np.random.RandomState(2)
        K = np.array([[500.0, 0, W / 2], [0, 500.0, H / 2], [0, 0, 1]])
        for frame in (1, 2):
            depth = rng.rand(H, W).astype(np.float32) * 10.0 + 0.5
            _write_dpt(depth_root / f"frame_{frame:04d}.dpt", depth)
            rgb = rng.randint(0, 255, size=(H, W, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(rgb_root / f"frame_{frame:04d}.png")
            R = np.eye(3)
            t = np.array([float(frame) * 0.1, 0.0, 0.0])  # camera moves along +x each frame
            _write_cam(cam_root / f"frame_{frame:04d}.cam", K, R, t)

        samples = load_sintel_gt(tmp, max_samples=None, download=False, require_cam=True)
        assert len(samples) == 2
        for i, s in enumerate(samples, start=1):
            assert s["scene"] == "alley_1"
            assert s["frame_idx"] == i
            assert s["K"] is not None and s["K"].shape == (3, 3)
            assert s["pose"] is not None and s["pose"].shape == (4, 4)
            assert np.allclose(s["K"], K)
        print("  [OK] Sintel loader populates scene/frame_idx/K/pose correctly")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@requires_pil
def test_sintel_loader_graceful_without_cam_unless_required():
    """No .cam files present: require_cam=False -> K/pose are None (accuracy eval unaffected);
    require_cam=True -> raises loudly (never silently proceeds without real poses)."""
    tmp = Path(tempfile.mkdtemp())
    try:
        depth_root = tmp / "training" / "depth" / "alley_1"
        rgb_root = tmp / "training" / "clean" / "alley_1"
        depth_root.mkdir(parents=True)
        rgb_root.mkdir(parents=True)
        _write_dpt(depth_root / "frame_0001.dpt", np.ones((4, 4), dtype=np.float32) * 5.0)
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(rgb_root / "frame_0001.png")

        samples = load_sintel_gt(tmp, download=False, require_cam=False)
        assert samples[0]["K"] is None
        assert samples[0]["pose"] is None

        with pytest.raises(RuntimeError, match="camera file .* missing"):
            load_sintel_gt(tmp, download=False, require_cam=True)
        print("  [OK] missing .cam is graceful (require_cam=False) or loud (require_cam=True)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------- geometric TAE math ------------------------------

def test_tae_geometric_identity_pose_zero_motion():
    """Identity relative pose + depth1==depth2 exactly -> reprojection is a
    pure identity map (no motion), so TAE must be exactly 0."""
    H, W = 20, 20
    K = torch.tensor([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]], dtype=torch.float64)
    depth = torch.full((H, W), 15.0, dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64)
    T = torch.zeros(3, dtype=torch.float64)
    mask = torch.ones((H, W), dtype=torch.bool)

    err = _tae_geometric_single(depth, depth.clone(), R, T, K, mask)
    print(f"  Identity pose, zero motion: TAE={err}")
    assert err == pytest.approx(0.0, abs=1e-8)


def test_tae_geometric_lateral_translation_frontoparallel_plane():
    """
    A frontal-parallel plane (constant depth D) viewed by a camera translated
    laterally (pure X shift) is still at depth D everywhere in the new
    camera's frame -- reprojecting frame1 into frame2's grid should land at
    shifted pixel columns but recover the SAME depth D, giving TAE ~0 for the
    pixels that reproject in-bounds. This exercises the full back-project /
    transform / reproject pipeline (fx,fy,cx,cy, matmul, divide, round,
    indexing) with a NON-trivial (non-identity) motion.
    """
    H, W = 40, 200  # wide enough that a 25px shift still leaves plenty in-bounds
    fx = fy = 100.0
    cx, cy = 100.0, 20.0
    D = 20.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float64)
    depth1 = torch.full((H, W), D, dtype=torch.float64)
    depth2 = torch.full((H, W), D, dtype=torch.float64)  # same plane, still fronto-parallel to cam2

    tx = 5.0
    R = torch.eye(3, dtype=torch.float64)
    T = torch.tensor([tx, 0.0, 0.0], dtype=torch.float64)
    mask = torch.ones((H, W), dtype=torch.bool)

    err = _tae_geometric_single(depth1, depth2, R, T, K, mask)
    print(f"  Lateral translation, fronto-parallel plane: TAE={err} "
          f"(expected shift = tx*fx/D = {tx*fx/D:.1f}px)")
    assert err == pytest.approx(0.0, abs=1e-6)


def test_tae_zbuffer_keeps_nearest_surface_on_collision():
    """
    Z-buffer correctness (fix over upstream's arbitrary last-write-wins):
    when two frame-1 pixels reproject onto the SAME frame-2 pixel, the NEAR
    surface must win (it occludes the far one). Construct an exact collision:
    two pixels at different depths whose reprojections land on one target.

    With last-write-wins, whichever pixel is processed later wins regardless
    of depth -- so a FAR background point can overwrite a NEAR foreground one
    and then be compared against foreground depth, producing huge spurious
    error. This is what inflated Sintel's violent-motion scenes (ambush_2 at
    907% vs a ~7% median). See docs/optimization_ledger.md T9.

    Construction: LATERAL translation shifts a pixel by tx*fx/depth, which is
    DEPTH-DEPENDENT, so two pixels at different depths can land on one target.
    Background sits at depth 100 (shift 0.1px -> reprojects onto itself, zero
    error), isolating the collision so it is the ONLY source of error:
      (2,6) depth 2.5 -> x = 6 + 10/2.5 = 10   [near]
      (2,8) depth 5.0 -> x = 8 + 10/5.0 = 10   [far, LATER in row-major order]
    frame2 holds the true near surface (2.5) at the target. Nearest-wins is
    therefore exactly right (error 0); last-write-wins keeps the far value and
    is badly wrong.
    """
    H, W = 4, 16
    fx = fy = 10.0
    cx, cy = 4.0, 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float64)
    depth1 = torch.full((H, W), 100.0, dtype=torch.float64)
    depth1[2, 6] = 2.5
    depth1[2, 8] = 5.0
    depth2 = torch.full((H, W), 100.0, dtype=torch.float64)
    depth2[2, 10] = 2.5
    R = torch.eye(3, dtype=torch.float64)
    T = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    mask = torch.ones((H, W), dtype=torch.bool)

    err_zbuf = _tae_geometric_single(depth1, depth2, R, T, K, mask, scatter_zbuffer=True)
    err_last = _tae_geometric_single(depth1, depth2, R, T, K, mask, scatter_zbuffer=False)
    print(f"  Collision handling: z-buffer={err_zbuf:.6f} (correct), "
          f"last-write-wins={err_last:.6f} (spurious)")
    # The z-buffer resolves the occlusion exactly; last-write-wins does not.
    assert err_zbuf == pytest.approx(0.0, abs=1e-9), err_zbuf
    assert err_last > 0.5, err_last
    assert err_zbuf < err_last, (err_zbuf, err_last)


def test_tae_zbuffer_identical_when_no_collisions():
    """With identity motion every pixel maps to its own target -- no collisions --
    so the z-buffer and upstream last-write-wins must agree EXACTLY (the fix
    changes nothing where nothing collides)."""
    H, W = 12, 12
    K = torch.tensor([[20.0, 0, 6], [0, 20.0, 6], [0, 0, 1]], dtype=torch.float64)
    torch.manual_seed(7)
    depth1 = torch.rand(H, W, dtype=torch.float64) * 5.0 + 5.0
    depth2 = torch.rand(H, W, dtype=torch.float64) * 5.0 + 5.0
    R = torch.eye(3, dtype=torch.float64)
    T = torch.zeros(3, dtype=torch.float64)
    mask = torch.ones((H, W), dtype=torch.bool)

    a = _tae_geometric_single(depth1, depth2, R, T, K, mask, scatter_zbuffer=True)
    b = _tae_geometric_single(depth1, depth2, R, T, K, mask, scatter_zbuffer=False)
    assert a == pytest.approx(b, abs=1e-12), (a, b)


def test_tae_geometric_no_inbounds_returns_zero():
    """Motion that sends every point out of frame -> returns 0.0 (matches eval_tae.py's own fallback)."""
    H, W = 10, 10
    K = torch.tensor([[50.0, 0, 5], [0, 50.0, 5], [0, 0, 1]], dtype=torch.float64)
    depth1 = torch.full((H, W), 10.0, dtype=torch.float64)
    depth2 = torch.full((H, W), 10.0, dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64)
    T = torch.tensor([10000.0, 0.0, 0.0], dtype=torch.float64)  # huge shift, nothing lands in-bounds
    mask = torch.ones((H, W), dtype=torch.bool)
    err = _tae_geometric_single(depth1, depth2, R, T, K, mask)
    assert err == 0.0


def test_tae_geometric_detects_real_inconsistency():
    """If depth2 is deliberately WRONG (doesn't match the geometrically
    reprojected frame1), TAE must be clearly nonzero -- confirms the metric
    actually measures something, not just returning 0 unconditionally."""
    H, W = 20, 20
    K = torch.tensor([[100.0, 0, 10], [0, 100.0, 10], [0, 0, 1]], dtype=torch.float64)
    depth1 = torch.full((H, W), 10.0, dtype=torch.float64)
    depth2 = torch.full((H, W), 25.0, dtype=torch.float64)  # inconsistent with depth1
    R = torch.eye(3, dtype=torch.float64)
    T = torch.zeros(3, dtype=torch.float64)
    mask = torch.ones((H, W), dtype=torch.bool)
    err = _tae_geometric_single(depth1, depth2, R, T, K, mask)
    print(f"  Deliberately inconsistent depths: TAE={err} (expect ~0.6 = |10-25|/25)")
    assert err == pytest.approx(0.6, abs=1e-6)


def test_pool_align_scene_disparity_recovers_known_affine():
    torch.manual_seed(3)
    gt_depths = [torch.rand(16, 16).double() * 20 + 1.0 for _ in range(4)]
    true_s, true_t = 2.5, 0.3
    pred_disps = []
    for gt in gt_depths:
        gt_disp = 1.0 / gt
        pred_disps.append((gt_disp - true_t) / true_s)  # inverse of s*pred+t=gt_disp

    s, t = _pool_align_scene_disparity(pred_disps, gt_depths, gt_range=(0.1, 100.0))
    print(f"  Recovered (s,t)=({s:.4f},{t:.4f}), true=({true_s},{true_t})")
    assert s == pytest.approx(true_s, rel=1e-3)
    assert t == pytest.approx(true_t, abs=1e-3)


def test_compute_tae_geometric_for_scene_perfectly_consistent():
    """
    Full pipeline, TRUE zero-flicker case: perfect (noiseless) disparity
    predictions, IDENTICAL depth every frame, identity pose (zero motion)
    between frames -> reprojection is a pure identity map at every pair, so
    TAE must be exactly 0.
    """
    H, W = 16, 16
    K = np.array([[80.0, 0, 8], [0, 80.0, 8], [0, 0, 1]])
    n_frames = 4
    d = 10.0

    pred_disps, gt_depths, Ks, poses = [], [], [], []
    for _ in range(n_frames):
        gt = np.full((H, W), d, dtype=np.float64)
        pred_disps.append(torch.from_numpy(1.0 / gt))  # perfect disparity prediction
        gt_depths.append(gt)
        Ks.append(K)
        poses.append(np.eye(4))  # identity pose for every frame -> zero motion between frames

    result = compute_tae_geometric_for_scene(pred_disps, gt_depths, Ks, poses, gt_range=(0.1, 80.0))
    print(f"  End-to-end scene TAE (perfectly consistent): {result}")
    assert result["n_pairs"] == n_frames - 1
    assert result["tae_percent"] == pytest.approx(0.0, abs=1e-6)


def test_compute_tae_geometric_for_scene_detects_frame_to_frame_drift():
    """
    Same pipeline, but with GENUINE frame-to-frame depth drift (still zero
    camera motion, so any AbsRel is purely "flicker", not motion-induced
    error) -- TAE must equal the analytically-derived value, computed here
    from first principles (bidirectional AbsRel per consecutive pair,
    averaged, x100), not a hand-picked bound. This is the oracle check that
    the multi-frame pooled-alignment + pairwise-averaging orchestration
    (not just the single-pair math already covered by
    test_tae_geometric_identity_pose_zero_motion) is wired correctly.
    """
    H, W = 16, 16
    K = np.array([[80.0, 0, 8], [0, 80.0, 8], [0, 0, 1]])
    depths_true = [10.0, 10.5, 9.8, 10.2]  # genuine per-frame drift

    pred_disps, gt_depths, Ks, poses = [], [], [], []
    for d in depths_true:
        gt = np.full((H, W), d, dtype=np.float64)
        pred_disps.append(torch.from_numpy(1.0 / gt))
        gt_depths.append(gt)
        Ks.append(K)
        poses.append(np.eye(4))

    result = compute_tae_geometric_for_scene(pred_disps, gt_depths, Ks, poses, gt_range=(0.1, 80.0))

    # Oracle: pred == gt exactly per-frame -> pooled disparity alignment is a
    # perfect fit (s=1, t=0), so aligned depths equal depths_true exactly.
    # Zero motion -> depth_proj == depth1 at every pixel (identity reprojection),
    # so each pair's bidirectional AbsRel is just |d_i - d_{i+1}| / d_{normalizer}.
    expected_sum = 0.0
    for i in range(len(depths_true) - 1):
        d1, d2 = depths_true[i], depths_true[i + 1]
        expected_sum += abs(d2 - d1) / d2  # forward: reproject frame i into i+1, normalize by d2
        expected_sum += abs(d1 - d2) / d1  # backward
    expected_tae_percent = (expected_sum / (2 * (len(depths_true) - 1))) * 100.0

    print(f"  End-to-end scene TAE (drifting depths): {result}, oracle={expected_tae_percent:.6f}")
    assert result["n_pairs"] == len(depths_true) - 1
    assert result["tae_percent"] == pytest.approx(expected_tae_percent, rel=1e-4)


def test_compute_tae_geometric_for_scene_length_mismatch_raises():
    with pytest.raises(ValueError, match="matching lengths"):
        compute_tae_geometric_for_scene(
            [torch.ones(4, 4)], [np.ones((4, 4)), np.ones((4, 4))], [np.eye(3)], [np.eye(4)],
            gt_range=(0.1, 10.0),
        )


def test_compute_tae_geometric_for_scene_single_frame_returns_zero():
    result = compute_tae_geometric_for_scene(
        [torch.rand(4, 4)], [np.ones((4, 4))], [np.eye(3)], [np.eye(4)], gt_range=(0.1, 10.0),
    )
    assert result["n_pairs"] == 0
    assert result["tae_percent"] == 0.0
    assert result["tae_median_percent"] == 0.0


if __name__ == "__main__":
    test_cam_read_roundtrip()
    test_cam_read_rejects_bad_magic()
    test_cam_to_world_pose_is_true_inverse()
    test_group_samples_by_scene_sorts_and_groups()
    test_group_samples_by_scene_rejects_none_scene()
    test_chunk_scene_into_windows_exact_multiple()
    test_chunk_scene_into_windows_remainder()
    test_chunk_scene_into_windows_scene_shorter_than_window()
    test_chunk_scene_into_windows_empty_and_invalid()
    test_sintel_loader_populates_scene_frame_idx_K_pose()
    test_sintel_loader_graceful_without_cam_unless_required()
    test_tae_geometric_identity_pose_zero_motion()
    test_tae_geometric_lateral_translation_frontoparallel_plane()
    test_tae_zbuffer_keeps_nearest_surface_on_collision()
    test_tae_zbuffer_identical_when_no_collisions()
    test_tae_geometric_no_inbounds_returns_zero()
    test_tae_geometric_detects_real_inconsistency()
    test_pool_align_scene_disparity_recovers_known_affine()
    test_compute_tae_geometric_for_scene_perfectly_consistent()
    test_compute_tae_geometric_for_scene_detects_frame_to_frame_drift()
    test_compute_tae_geometric_for_scene_length_mismatch_raises()
    test_compute_tae_geometric_for_scene_single_frame_returns_zero()
    print("All temporal/TAE path tests passed.")
