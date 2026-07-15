"""
Verifies load_nyuv2_gt_test_split's h5py fallback for MATLAB v7.3 (HDF5)
files. The official nyu_depth_v2_labeled.mat (~2.8GB) is saved in v7.3
format, which scipy.io.loadmat cannot read at all (raises NotImplementedError
"Please use HDF reader for matlab v7.3 files") — discovered when this was
first run on Colab against the real file. See docs/optimization_ledger.md T2.

This test builds a small SYNTHETIC v7.3-shaped HDF5 file locally (no network,
no real dataset) to verify the fallback path and axis-transpose math, without
needing the real 2.8GB download.

Run: pytest tests/test_nyuv2_h5py_loader.py -q
"""
import sys
import os
import tempfile
import shutil
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import scipy.io

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from datasets_gt import load_nyuv2_gt_test_split

requires_h5py = pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")


@requires_h5py
def test_h5py_fallback_loads_v73_labeled_file():
    """
    Build a tiny HDF5 file matching the real file's on-disk layout
    (images: (N,3,W,H), depths: (N,W,H), MATLAB-v7.3-style axis order) and
    confirm load_nyuv2_gt_test_split correctly falls back to h5py and
    transposes each sample back to (H,W,3) / (H,W).
    """
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        n, c, w, h = 5, 3, 8, 6
        rng = np.random.RandomState(0)
        images_raw = rng.randint(0, 255, size=(n, c, w, h), dtype=np.uint8)
        depths_raw = rng.rand(n, w, h).astype(np.float32) * 5.0 + 0.5

        mat_path = tmp_dir / "nyu_depth_v2_labeled.mat"
        with h5py.File(str(mat_path), "w") as f:
            f.create_dataset("images", data=images_raw)
            f.create_dataset("depths", data=depths_raw)

        # Confirm scipy genuinely can't read this plain HDF5 file directly,
        # so the test actually exercises the h5py fallback branch (not the
        # scipy success branch). A bare h5py-written file lacks MATLAB's
        # v7.3 header signature, so scipy fails during version-sniffing with
        # ValueError rather than the NotImplementedError a genuine MATLAB
        # v7.3 export raises (confirmed against the real file on Colab) —
        # load_nyuv2_gt_test_split's except clause catches both.
        with pytest.raises((NotImplementedError, ValueError)):
            scipy.io.loadmat(str(mat_path))

        # splits.mat: small, old-format, loadable directly by scipy (matches
        # the real file's behavior observed on Colab).
        splits_path = tmp_dir / "splits.mat"
        test_ndxs = np.array([[1], [2], [4]])  # MATLAB 1-indexed
        scipy.io.savemat(str(splits_path), {"testNdxs": test_ndxs, "trainNdxs": np.array([[3], [5]])})

        samples = load_nyuv2_gt_test_split(tmp_dir, max_samples=None, download=False)

        print(f"  Loaded {len(samples)} samples via h5py fallback")
        assert len(samples) == 3, f"Expected 3 test samples, got {len(samples)}"

        expected_indices_0based = [0, 1, 3]  # test_ndxs - 1
        for sample, idx in zip(samples, expected_indices_0based):
            assert sample["rgb"].shape == (h, w, c), sample["rgb"].shape
            assert sample["depth"].shape == (h, w), sample["depth"].shape
            assert sample["valid_mask"].shape == (h, w)

            # Verify the transpose actually recovers the original per-pixel
            # values (not just the right shape) for a specific index.
            expected_rgb = np.transpose(images_raw[idx], (2, 1, 0))
            expected_depth = np.transpose(depths_raw[idx], (1, 0))
            assert np.array_equal(sample["rgb"], expected_rgb), f"RGB transpose mismatch at idx={idx}"
            assert np.allclose(sample["depth"], expected_depth), f"Depth transpose mismatch at idx={idx}"

        print("  [OK] h5py fallback correctly transposes images/depths to (H,W,C)/(H,W)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@requires_h5py
def test_h5py_fallback_respects_max_samples():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        n, c, w, h = 6, 3, 4, 4
        rng = np.random.RandomState(1)
        images_raw = rng.randint(0, 255, size=(n, c, w, h), dtype=np.uint8)
        depths_raw = rng.rand(n, w, h).astype(np.float32) + 0.1

        mat_path = tmp_dir / "nyu_depth_v2_labeled.mat"
        with h5py.File(str(mat_path), "w") as f:
            f.create_dataset("images", data=images_raw)
            f.create_dataset("depths", data=depths_raw)

        splits_path = tmp_dir / "splits.mat"
        scipy.io.savemat(str(splits_path), {"testNdxs": np.array([[1], [2], [3], [4], [5], [6]])})

        samples = load_nyuv2_gt_test_split(tmp_dir, max_samples=2, download=False)
        print(f"  max_samples=2 -> loaded {len(samples)} samples")
        assert len(samples) == 2
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    if not HAS_H5PY:
        print("h5py not available; skipping.")
    else:
        test_h5py_fallback_loads_v73_labeled_file()
        test_h5py_fallback_respects_max_samples()
        print("All NYUv2 h5py loader tests passed.")
