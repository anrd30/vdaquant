"""
Verifies the KITTI and Sintel ground-truth loaders (scripts/datasets_gt.py)
using tiny SYNTHETIC on-disk fixtures — no network, no real dataset download.
See docs/optimization_ledger.md T2.

Each test builds the exact directory layout the real download+extract produces
(verified against the real zips' central directories), writes a couple of
frames, and checks the loader returns the {rgb, depth(metres), valid_mask}
contract with correct values, matching, masking, and .dpt parsing.

Run: pytest tests/test_kitti_sintel_loaders.py -q
"""
import sys
import os
import struct
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from datasets_gt import (
    load_kitti_gt,
    load_sintel_gt,
    load_gt_dataset,
    _read_sintel_dpt,
    DATASET_GT_CONFIG,
)

requires_pil = pytest.mark.skipif(not HAS_PIL, reason="Pillow not installed")


# ----------------------------- KITTI -----------------------------------------

@requires_pil
def test_kitti_loader_matches_rgb_and_scales_depth():
    tmp = Path(tempfile.mkdtemp())
    try:
        base = tmp / "depth_selection" / "val_selection_cropped"
        img_dir = base / "image"
        depth_dir = base / "groundtruth_depth"
        img_dir.mkdir(parents=True)
        depth_dir.mkdir(parents=True)

        H, W = 8, 12
        # Two frames with the KITTI val_selection_cropped naming convention.
        stems = [
            ("2011_09_26_drive_0013_sync_image_0000000005_image_02",
             "2011_09_26_drive_0013_sync_groundtruth_depth_0000000005_image_02"),
            ("2011_09_26_drive_0013_sync_image_0000000010_image_03",
             "2011_09_26_drive_0013_sync_groundtruth_depth_0000000010_image_03"),
        ]
        rng = np.random.RandomState(0)
        expected = []
        for i, (img_stem, depth_stem) in enumerate(stems):
            rgb = rng.randint(0, 255, size=(H, W, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(img_dir / f"{img_stem}.png")

            # uint16 depth: some zeros (invalid), rest = metres*256
            depth_u16 = np.zeros((H, W), dtype=np.uint16)
            depth_u16[2:6, 3:9] = (rng.rand(4, 6) * 40.0 * 256).astype(np.uint16) + 256
            Image.fromarray(depth_u16, mode="I;16").save(depth_dir / f"{depth_stem}.png")
            expected.append((rgb, depth_u16))

        samples = load_kitti_gt(tmp, max_samples=None, download=False)
        assert len(samples) == 2, len(samples)

        for sample, (exp_rgb, exp_u16) in zip(samples, expected):
            assert sample["rgb"].shape == (H, W, 3)
            assert sample["depth"].shape == (H, W)
            assert np.array_equal(sample["rgb"], exp_rgb), "RGB not matched to correct depth file"
            # depth = raw/256, valid where raw>0
            assert np.allclose(sample["depth"], exp_u16.astype(np.float32) / 256.0, atol=1e-3)
            assert np.array_equal(sample["valid_mask"], exp_u16 > 0)
        print(f"  [OK] KITTI loader: {len(samples)} pairs, depth scaled /256, RGB matched by filename")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_kitti_loader_raises_when_missing():
    tmp = Path(tempfile.mkdtemp())
    try:
        with pytest.raises(RuntimeError, match="val_selection_cropped not found"):
            load_kitti_gt(tmp, download=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@requires_pil
def test_kitti_max_samples():
    tmp = Path(tempfile.mkdtemp())
    try:
        base = tmp / "depth_selection" / "val_selection_cropped"
        (base / "image").mkdir(parents=True)
        (base / "groundtruth_depth").mkdir(parents=True)
        for i in range(5):
            stem_i = f"2011_09_26_drive_0013_sync_image_000000000{i}_image_02"
            stem_d = f"2011_09_26_drive_0013_sync_groundtruth_depth_000000000{i}_image_02"
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(base / "image" / f"{stem_i}.png")
            d = np.ones((4, 4), dtype=np.uint16) * 256
            Image.fromarray(d, mode="I;16").save(base / "groundtruth_depth" / f"{stem_d}.png")
        samples = load_kitti_gt(tmp, max_samples=3, download=False)
        assert len(samples) == 3, len(samples)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------- Sintel ----------------------------------------

def _write_dpt(path: Path, arr: np.ndarray):
    """Write a Sintel .dpt file: magic float, int32 W, int32 H, then W*H float32."""
    h, w = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<f", 202021.25))
        f.write(struct.pack("<i", w))
        f.write(struct.pack("<i", h))
        arr.astype("<f4").tofile(f)


def test_sintel_dpt_reader_roundtrip():
    tmp = Path(tempfile.mkdtemp())
    try:
        arr = np.arange(6 * 4, dtype=np.float32).reshape(6, 4) * 1.5
        p = tmp / "frame_0001.dpt"
        _write_dpt(p, arr)
        read = _read_sintel_dpt(p)
        assert read.shape == (6, 4), read.shape
        assert np.allclose(read, arr), "dpt round-trip mismatch"
        print("  [OK] Sintel .dpt reader round-trips shape and values")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_sintel_dpt_reader_rejects_bad_magic():
    tmp = Path(tempfile.mkdtemp())
    try:
        p = tmp / "bad.dpt"
        with open(p, "wb") as f:
            f.write(struct.pack("<f", 1234.5))  # wrong magic
            f.write(struct.pack("<ii", 2, 2))
            np.zeros(4, dtype="<f4").tofile(f)
        with pytest.raises(RuntimeError, match="not a valid Sintel .dpt"):
            _read_sintel_dpt(p)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@requires_pil
def test_sintel_loader_merges_depth_and_rgb_trees():
    tmp = Path(tempfile.mkdtemp())
    try:
        # Depth tree (from depth-training zip) and RGB tree (from complete zip)
        # both live under training/, as they would after extracting both zips
        # into the same directory.
        depth_root = tmp / "training" / "depth" / "alley_1"
        rgb_root = tmp / "training" / "clean" / "alley_1"
        depth_root.mkdir(parents=True)
        rgb_root.mkdir(parents=True)

        H, W = 6, 10
        rng = np.random.RandomState(1)
        expected = []
        for frame in (1, 2):
            depth = rng.rand(H, W).astype(np.float32) * 20.0 + 0.5
            depth[0, 0] = 1e11  # sky sentinel -> must be masked out by max_depth
            _write_dpt(depth_root / f"frame_{frame:04d}.dpt", depth)
            rgb = rng.randint(0, 255, size=(H, W, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(rgb_root / f"frame_{frame:04d}.png")
            expected.append((rgb, depth))

        samples = load_sintel_gt(tmp, max_samples=None, download=False, max_depth=1000.0)
        assert len(samples) == 2, len(samples)

        for sample, (exp_rgb, exp_depth) in zip(samples, expected):
            assert sample["rgb"].shape == (H, W, 3)
            assert sample["depth"].shape == (H, W)
            assert np.array_equal(sample["rgb"], exp_rgb), "Sintel RGB not matched to correct frame"
            assert np.allclose(sample["depth"], exp_depth, atol=1e-2)
            # The sky-sentinel pixel must be invalid; a normal pixel valid.
            assert sample["valid_mask"][0, 0] == False, "sky sentinel should be masked out"
            assert sample["valid_mask"][3, 5] == True
        print(f"  [OK] Sintel loader: merged depth+RGB trees, {len(samples)} frames, sky masked")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_sintel_loader_raises_without_rgb_tree():
    """Depth present but the complete-zip RGB pass missing -> loud, specific error."""
    tmp = Path(tempfile.mkdtemp())
    try:
        depth_root = tmp / "training" / "depth" / "alley_1"
        depth_root.mkdir(parents=True)
        _write_dpt(depth_root / "frame_0001.dpt", np.ones((4, 4), dtype=np.float32))
        with pytest.raises(RuntimeError, match="RGB tree not found"):
            load_sintel_gt(tmp, download=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------- dispatcher ------------------------------------

def test_load_gt_dataset_rejects_unknown_and_davis():
    tmp = Path(tempfile.mkdtemp())
    try:
        for bad in ("davis", "scannet", "bogus"):
            with pytest.raises(ValueError, match="No ground-truth loader"):
                load_gt_dataset(bad, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_dataset_gt_config_ranges():
    """Sanity: each configured dataset has a sensible (min<max, positive) gt_range."""
    for name, cfg in DATASET_GT_CONFIG.items():
        lo, hi = cfg["gt_range"]
        assert 0 < lo < hi, (name, cfg["gt_range"])
    # NYU indoor is capped tighter than KITTI outdoor.
    assert DATASET_GT_CONFIG["nyuv2"]["gt_range"][1] < DATASET_GT_CONFIG["kitti"]["gt_range"][1]


if __name__ == "__main__":
    test_kitti_loader_matches_rgb_and_scales_depth()
    test_kitti_loader_raises_when_missing()
    test_kitti_max_samples()
    test_sintel_dpt_reader_roundtrip()
    test_sintel_dpt_reader_rejects_bad_magic()
    test_sintel_loader_merges_depth_and_rgb_trees()
    test_sintel_loader_raises_without_rgb_tree()
    test_load_gt_dataset_rejects_unknown_and_davis()
    test_dataset_gt_config_ranges()
    print("All KITTI/Sintel loader tests passed.")
