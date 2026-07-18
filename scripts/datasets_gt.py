"""
Ground-truth depth evaluation: NYUv2 labeled test-split loading and the
affine-invariant evaluation protocol (MiDaS/DepthAnything style).

This module is deliberately split out of run_pareto_benchmark_suite.py so the
GT protocol math (affine alignment, masked metrics) can be unit-tested with
synthetic tensors, with ZERO network access and ZERO dependency on actually
having the dataset downloaded. See docs/optimization_ledger.md task T2.

IMPORTANT: load_nyuv2_gt_test_split() performs a real network download and
is NOT invoked anywhere in this module or by the test suite. It exists so the
Colab environment (which has bandwidth and disk for the ~2.8GB official
NYUv2 labeled .mat file) can call it explicitly; local/CI runs never trigger
a download as a side effect of importing this file.
"""
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch


NYUV2_LABELED_MAT_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
NYUV2_SPLITS_URL = "http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat"


def affine_align_inverse_depth(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[float, float, torch.Tensor]:
    """
    Per-image least-squares affine alignment of predicted (inverse) depth to
    ground-truth (inverse) depth, restricted to valid_mask:

        (s, t) = argmin_{s,t} sum_{valid} (s*pred + t - gt)^2

    This is the standard MiDaS/DepthAnything protocol for scale-and-shift
    invariant monocular depth models, which do not predict metric depth
    directly.

    Args:
        pred: Predicted (inverse) depth, any shape.
        gt: Ground-truth (inverse) depth, same shape as pred.
        valid_mask: Boolean mask, same shape, selecting pixels to fit AND
                    to report alignment error over.

    Returns:
        (s, t, pred_aligned) where pred_aligned = s * pred + t (full shape,
        same as input; alignment is only fit on valid_mask but applied
        everywhere).
    """
    p = pred[valid_mask].reshape(-1).double()
    g = gt[valid_mask].reshape(-1).double()

    if p.numel() < 2:
        raise ValueError(f"Need at least 2 valid pixels to fit an affine alignment, got {p.numel()}")

    A = torch.stack([p, torch.ones_like(p)], dim=1)  # (N, 2)
    solution = torch.linalg.lstsq(A, g.unsqueeze(-1)).solution  # (2, 1)
    s = float(solution[0, 0])
    t = float(solution[1, 0])

    pred_aligned = (s * pred.double() + t).to(pred.dtype)
    return s, t, pred_aligned


def compute_gt_depth_metrics(
    pred: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    gt_range: Tuple[float, float] = (0.1, 10.0),
    pred_is_disparity: bool = True,
) -> Dict[str, float]:
    """
    Affine-invariant ground-truth depth evaluation (MiDaS/DepthAnything protocol).

    CRITICAL — alignment space (see docs/optimization_ledger.md F10): VDA's
    RELATIVE-depth model (the `vits` checkpoint this suite auto-downloads)
    outputs DISPARITY (inverse depth, higher = closer), NOT metric depth —
    the metric model is a separate checkpoint requiring --metric. A scale+shift
    fit is only valid in the space the model actually predicts, so we must
    align in DISPARITY space, then invert to metric depth for the metrics:

        gt_disp        = 1 / gt_depth
        (s, t)         = argmin || s*pred + t - gt_disp ||^2   over valid pixels
        pred_depth     = 1 / clamp(s*pred + t, min=eps)
        metrics(pred_depth, gt_depth)  in METRIC space

    Aligning a disparity prediction LINEARLY against metric depth (the earlier
    bug) can't reconcile the reciprocal relationship; it half-works on a narrow
    range (NYU 0.1-10m: delta1~0.81) and collapses on a wide one (KITTI 1-80m:
    delta1~0.43). Set pred_is_disparity=False only if the model already outputs
    metric depth (e.g. the VDA metric checkpoint), in which case we align
    linearly in metric space as before.

    Steps:
      1. Keep pixels where valid_mask AND gt_depth in gt_range (metres).
      2. Fit one per-image affine (s, t) on those pixels, in the appropriate space.
      3. Report AbsRel/RMSE/delta1-3 on the aligned prediction in METRIC space.

    Args:
        pred: Model output, any shape. Disparity if pred_is_disparity (default).
        gt_depth: Ground-truth METRIC depth (metres), same shape.
        valid_mask: Optional extra boolean mask (e.g. sensor dropout / no LiDAR
                    return). Combined with the gt_range mask.
        gt_range: (min, max) metres; pixels with gt outside are excluded from
                  BOTH fit and evaluation.
        pred_is_disparity: True -> align in disparity space then invert (default,
                  correct for the relative VDA model). False -> align in metric.

    Returns:
        Dict with abs_rel, rmse, delta1, delta2, delta3, affine_scale,
        affine_shift, n_valid_pixels.
    """
    eps = 1e-6
    if valid_mask is None:
        valid_mask = torch.ones_like(gt_depth, dtype=torch.bool)

    range_mask = (gt_depth > gt_range[0]) & (gt_depth < gt_range[1])
    mask = valid_mask & range_mask

    if not mask.any():
        raise ValueError(
            f"No valid pixels after applying valid_mask and gt_range={gt_range}"
        )

    if pred_is_disparity:
        # Fit pred (disparity) to GT disparity, then invert aligned disparity
        # back to metric depth. Disparity must stay positive after the affine,
        # so clamp before inverting.
        gt_disp = 1.0 / torch.clamp(gt_depth, min=eps)
        s, t, pred_disp_aligned = affine_align_inverse_depth(pred, gt_disp, mask)
        pred_depth_full = 1.0 / torch.clamp(pred_disp_aligned, min=eps)
    else:
        s, t, pred_depth_full = affine_align_inverse_depth(pred, gt_depth, mask)

    # Cap predicted depth to the evaluation range (standard KITTI/NYU practice).
    # After inverting disparity, far pixels whose aligned disparity approaches 0
    # explode toward 1/eps (~1e6 m); those outliers don't affect delta1 (a ratio
    # threshold) but would otherwise wreck the metric-space means AbsRel and RMSE.
    # Capping to [gt_range] is the depth-eval convention (e.g. KITTI's 80 m cap).
    p = torch.clamp(pred_depth_full[mask], min=gt_range[0], max=gt_range[1])
    g = torch.clamp(gt_depth[mask], min=eps)

    abs_rel = torch.mean(torch.abs(p - g) / g).item()
    rmse = torch.sqrt(torch.mean((p - g) ** 2)).item()

    ratio = torch.max(p / g, g / p)
    delta1 = torch.mean((ratio < 1.25).float()).item()
    delta2 = torch.mean((ratio < 1.25 ** 2).float()).item()
    delta3 = torch.mean((ratio < 1.25 ** 3).float()).item()

    return {
        "abs_rel": round(abs_rel, 6),
        "rmse": round(rmse, 6),
        "delta1": round(delta1, 6),
        "delta2": round(delta2, 6),
        "delta3": round(delta3, 6),
        "affine_scale": round(s, 6),
        "affine_shift": round(t, 6),
        "n_valid_pixels": int(mask.sum().item()),
    }


def load_nyuv2_gt_test_split(
    cache_dir: Path,
    max_samples: Optional[int] = None,
    download: bool = True,
):
    """
    Loads the official NYUv2 labeled test split (654 RGB-D pairs) for
    ground-truth evaluation.

    NOT invoked by this module or by the test suite — this is a real network
    download (~2.8GB .mat file) intended to be called explicitly from a
    Colab/GPU environment with adequate bandwidth and disk, per
    docs/optimization_ledger.md T2 ("no synthetic fallback in GT mode").

    Args:
        cache_dir: Directory to cache the downloaded .mat files and the
                   extracted per-image RGB/depth arrays.
        max_samples: If set, only load this many test-split images.
        download: If False and the cache is missing, raises instead of
                  downloading (useful for CI environments that pre-seed
                  the cache out-of-band).

    Returns:
        List of dicts: {"rgb": np.ndarray (H,W,3) uint8, "depth": np.ndarray
        (H,W) float32 meters, "valid_mask": np.ndarray (H,W) bool}.

    Raises:
        RuntimeError: if the dataset is not cached and download is False,
                      or if the download/extraction fails. This function
                      NEVER silently substitutes synthetic or proxy data —
                      GT-mode evaluation must fail loudly rather than report
                      numbers against fake ground truth.
    """
    import scipy.io  # local import: heavy optional dependency, only needed here

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mat_path = cache_dir / "nyu_depth_v2_labeled.mat"
    splits_path = cache_dir / "splits.mat"

    if not mat_path.exists() or not splits_path.exists():
        if not download:
            raise RuntimeError(
                f"NYUv2 labeled dataset not found in {cache_dir} and download=False. "
                f"Pre-seed the cache with nyu_depth_v2_labeled.mat and splits.mat, "
                f"or call with download=True on an environment with network access "
                f"(e.g. the Colab evaluation notebook)."
            )
        import urllib.request

        for url, dest in [(NYUV2_LABELED_MAT_URL, mat_path), (NYUV2_SPLITS_URL, splits_path)]:
            print(f"  [Dataset] Downloading {url} -> {dest} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as response, open(dest, "wb") as out_file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out_file.write(chunk)

    splits = scipy.io.loadmat(str(splits_path))
    test_indices = splits["testNdxs"].flatten() - 1  # MATLAB 1-indexed -> 0-indexed

    if max_samples is not None:
        test_indices = test_indices[:max_samples]

    # The official nyu_depth_v2_labeled.mat (~2.8GB) is saved in MATLAB v7.3
    # format, which is HDF5-based and NOT readable by scipy.io.loadmat
    # (raises NotImplementedError: "Please use HDF reader for matlab v7.3
    # files" on the real file; some file variants/scipy versions can also
    # surface this as a ValueError during version sniffing). splits.mat is
    # small and saves in an older format that scipy handles fine (confirmed
    # above), but the labeled file needs h5py.
    try:
        data = scipy.io.loadmat(str(mat_path))
        images = data["images"]  # (H, W, 3, N)
        depths = data["depths"]  # (H, W, N), meters
        samples = []
        for i, idx in enumerate(test_indices):
            rgb = images[:, :, :, idx].astype(np.uint8)
            depth = depths[:, :, idx].astype(np.float32)
            valid_mask = depth > 0
            # NYU test-split images are independent single frames, not video —
            # scene=None marks them as non-groupable for --eval-mode temporal
            # (see group_samples_by_scene).
            samples.append({"rgb": rgb, "depth": depth, "valid_mask": valid_mask,
                             "scene": None, "frame_idx": i})
        return samples
    except (NotImplementedError, ValueError):
        try:
            import h5py
        except ImportError as e:
            raise RuntimeError(
                "nyu_depth_v2_labeled.mat is a MATLAB v7.3 (HDF5) file, which requires "
                "the 'h5py' package (scipy.io.loadmat cannot read it). Install it with "
                "`pip install h5py` and re-run."
            ) from e

        samples = []
        with h5py.File(str(mat_path), "r") as f:
            # HDF5-backed MATLAB arrays are stored axis-reversed relative to
            # the original MATLAB shape: images is (N, 3, W, H), depths is
            # (N, W, H). Transpose each sample back to (H, W, C) / (H, W).
            images = f["images"]
            depths = f["depths"]
            for i, idx in enumerate(test_indices):
                idx = int(idx)
                rgb = np.transpose(images[idx], (2, 1, 0)).astype(np.uint8)   # (3,W,H) -> (H,W,3)
                depth = np.transpose(depths[idx], (1, 0)).astype(np.float32)  # (W,H) -> (H,W)
                valid_mask = depth > 0
                samples.append({"rgb": rgb, "depth": depth, "valid_mask": valid_mask,
                                 "scene": None, "frame_idx": i})
        return samples


def load_kitti_gt(
    cache_dir: Path,
    max_samples: Optional[int] = None,
    download: bool = False,
):
    """
    Loads the KITTI depth val_selection_cropped split (1000 RGB + GT-depth
    pairs) from a pre-extracted data_depth_selection.zip.

    Expects the layout produced by `scripts/download_datasets.sh kitti`:
        <cache_dir>/depth_selection/val_selection_cropped/image/*.png
        <cache_dir>/depth_selection/val_selection_cropped/groundtruth_depth/*.png
    (verified against the real zip's central directory — this split is the
    only self-contained one that ships BOTH RGB and GT depth; see
    docs/optimization_ledger.md T2 / scripts/download_datasets.sh).

    KITTI depth PNGs are uint16 where depth_metres = pixel / 256.0 and a
    pixel value of 0 means "no LiDaR return" (invalid). Returns the SAME
    contract as load_nyuv2_gt_test_split: a list of
    {"rgb": (H,W,3) uint8, "depth": (H,W) float32 metres, "valid_mask": (H,W) bool}.

    Args:
        cache_dir: Directory containing the extracted depth_selection tree.
        max_samples: If set, load only this many pairs (sorted order).
        download: KITTI is NOT auto-downloaded here (the 1.9GB zip belongs in
                  the explicit downloader). If the data is missing this raises
                  rather than silently downloading or faking it.

    Raises:
        RuntimeError: if the expected directories are absent — GT-mode eval
                      must fail loudly, never substitute proxy data.
    """
    from PIL import Image

    cache_dir = Path(cache_dir)
    base = cache_dir / "depth_selection" / "val_selection_cropped"
    img_dir = base / "image"
    depth_dir = base / "groundtruth_depth"

    if not img_dir.is_dir() or not depth_dir.is_dir():
        raise RuntimeError(
            f"KITTI val_selection_cropped not found under {base}. Expected "
            f"image/ and groundtruth_depth/ subdirs. Run "
            f"`bash scripts/download_datasets.sh kitti` first (downloads and "
            f"extracts data_depth_selection.zip). This loader never downloads "
            f"or fabricates data (download={download})."
        )

    # Match each GT-depth file to its RGB image. In this split the two share
    # an identical filename except for the field name token, e.g.
    #   image:            2011_09_26_drive_0013_sync_image_0000000005_image_02.png
    #   groundtruth_depth 2011_09_26_drive_0013_sync_groundtruth_depth_0000000005_image_02.png
    # so we derive the RGB name from each depth name by swapping that token.
    depth_files = sorted(depth_dir.glob("*.png"))
    if not depth_files:
        raise RuntimeError(f"No groundtruth_depth PNGs found in {depth_dir}")
    if max_samples is not None:
        depth_files = depth_files[:max_samples]

    samples = []
    for i, depth_path in enumerate(depth_files):
        img_name = depth_path.name.replace("groundtruth_depth", "image", 1)
        img_path = img_dir / img_name
        if not img_path.exists():
            raise RuntimeError(
                f"KITTI RGB image {img_path} missing for GT depth {depth_path.name}. "
                f"The depth_selection zip should contain a matching image/ file for "
                f"every groundtruth_depth/ file."
            )

        rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        # uint16 PNG; divide by 256 for metres, 0 = invalid (KITTI devkit convention).
        depth_raw = np.array(Image.open(depth_path), dtype=np.float32)
        depth = depth_raw / 256.0
        valid_mask = depth_raw > 0
        # val_selection_cropped draws from multiple independent KITTI drives,
        # not one continuous sequence — scene=None, same as NYU (non-groupable).
        samples.append({"rgb": rgb, "depth": depth, "valid_mask": valid_mask,
                         "scene": None, "frame_idx": i})

    return samples


def _read_sintel_dpt(path: Path) -> np.ndarray:
    """
    Read a Sintel .dpt depth map (the format used by the MPI-Sintel depth
    package). Layout: float32 magic tag 202021.25, int32 width, int32 height,
    then width*height float32 depth values (metres), row-major. This mirrors
    the official Sintel SDK's depth_read().
    """
    with open(path, "rb") as f:
        magic = np.fromfile(f, dtype=np.float32, count=1)
        if magic.size == 0 or abs(float(magic[0]) - 202021.25) > 1e-2:
            raise RuntimeError(
                f"{path} is not a valid Sintel .dpt file (bad magic {magic}); "
                f"expected 202021.25."
            )
        width = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        height = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        depth = np.fromfile(f, dtype=np.float32, count=width * height)
    return depth.reshape(height, width)


def _read_sintel_cam(path: Path):
    """
    Read a Sintel .cam camera file (official Sintel SDK `sintel_io.cam_read`
    format — same TAG_FLOAT magic as .dpt/.flo). Layout: float32 magic
    202021.25, then M (9 float64 = 3x3 intrinsic matrix), then N (12 float64
    = 3x4 extrinsic matrix [R|t]), such that for a homogeneous world point X,
    x = M @ N @ X gives homogeneous image pixel coordinates. N is therefore
    the WORLD-TO-CAMERA extrinsic: X_cam = R @ X_world + t.

    Returns (K, R, t): K is (3,3) intrinsic, R is (3,3) rotation, t is (3,)
    translation, all float64.
    """
    with open(path, "rb") as f:
        magic = np.fromfile(f, dtype=np.float32, count=1)
        if magic.size == 0 or abs(float(magic[0]) - 202021.25) > 1e-2:
            raise RuntimeError(
                f"{path} is not a valid Sintel .cam file (bad magic {magic}); "
                f"expected 202021.25."
            )
        M = np.fromfile(f, dtype=np.float64, count=9).reshape(3, 3)
        N = np.fromfile(f, dtype=np.float64, count=12).reshape(3, 4)
    K = M
    R = N[:, :3]
    t = N[:, 3]
    return K, R, t


def _cam_to_world_pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Converts a Sintel .cam WORLD-TO-CAMERA extrinsic (X_cam = R @ X_world + t)
    into a 4x4 CAMERA-TO-WORLD homogeneous pose matrix T such that
    X_world = T @ [X_cam; 1]. Since R is an orthonormal rotation, R^-1 = R.T,
    so T = [[R.T, -R.T @ t], [0, 0, 0, 1]]. This is the "pose" convention
    Video-Depth-Anything/benchmark/eval/eval_tae.py expects (it composes
    relative pose as T_2_1 = inv(pose_2) @ pose_1 for camera-to-world poses).
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    return T


def load_sintel_gt(
    cache_dir: Path,
    max_samples: Optional[int] = None,
    download: bool = False,
    pass_name: str = "clean",
    max_depth: float = 1000.0,
    require_cam: bool = False,
):
    """
    Loads MPI-Sintel depth ground truth with matching RGB frames (and, when
    available, camera intrinsics/pose for geometric TAE — see T9).

    Sintel splits RGB and depth across TWO archives (verified via each zip's
    central directory — see scripts/download_datasets.sh):
        MPI-Sintel-depth-training-20150305.zip -> training/depth/<scene>/frame_XXXX.dpt
                                                   training/camdata_left/<scene>/frame_XXXX.cam
        MPI-Sintel-complete.zip                -> training/{clean,final}/<scene>/frame_XXXX.png
    Extract BOTH into the same <cache_dir> so they merge under one training/
    tree. The depth-training zip's depth_viz/ PNGs are colourised previews,
    NOT model input — this loader ignores them and reads the real .dpt depth.
    Camera files (camdata_left/) ship in the SAME zip as depth, so no extra
    download is needed for the K/pose fields below.

    Returns a flat list of dicts, sorted by scene then frame:
        "rgb": (H,W,3) uint8, "depth": (H,W) float32 metres,
        "valid_mask": (H,W) bool, "scene": str, "frame_idx": int,
        "K": (3,3) float64 intrinsic or None if the .cam file is missing,
        "pose": (4,4) float64 camera-to-world homogeneous pose or None.
    (Grouping into per-scene sequences for temporal eval is the caller's job
    — see group_samples_by_scene.)

    Args:
        cache_dir: Directory with the merged training/ tree.
        max_samples: If set, cap the number of frames (sorted order).
        download: Never downloads here; raises if data is missing.
        pass_name: 'clean' (default) or 'final' RGB render pass.
        max_depth: Depths above this (metres) are treated as invalid — Sintel
                   sky/background carries enormous sentinel depths that would
                   otherwise dominate the affine fit.
        require_cam: If True, raise when a frame's .cam file is missing
                     instead of leaving K/pose as None. Set True for
                     --eval-mode temporal (which needs real camera poses for
                     geometric TAE); leave False for plain accuracy eval
                     (which never touches K/pose at all).

    Raises:
        RuntimeError: if the depth or RGB trees are absent, or (require_cam)
                      if any frame's camera file is missing.
    """
    from PIL import Image

    cache_dir = Path(cache_dir)
    depth_root = cache_dir / "training" / "depth"
    rgb_root = cache_dir / "training" / pass_name
    cam_root = cache_dir / "training" / "camdata_left"

    if not depth_root.is_dir():
        raise RuntimeError(
            f"Sintel depth tree not found at {depth_root}. Run "
            f"`bash scripts/download_datasets.sh sintel` (fetches and extracts "
            f"BOTH the depth-training and complete zips). This loader never "
            f"downloads or fabricates data (download={download})."
        )
    if not rgb_root.is_dir():
        raise RuntimeError(
            f"Sintel RGB tree not found at {rgb_root}. The depth-training zip "
            f"has NO RGB frames — you also need MPI-Sintel-complete.zip extracted "
            f"into the same directory (the '{pass_name}' pass). "
            f"`bash scripts/download_datasets.sh sintel` fetches both."
        )

    depth_files = sorted(depth_root.glob("*/frame_*.dpt"))
    if not depth_files:
        raise RuntimeError(f"No .dpt depth files found under {depth_root}")
    if max_samples is not None:
        depth_files = depth_files[:max_samples]

    samples = []
    for depth_path in depth_files:
        scene = depth_path.parent.name
        frame = depth_path.stem  # frame_XXXX
        frame_idx = int(frame.split("_")[-1])
        rgb_path = rgb_root / scene / f"{frame}.png"
        if not rgb_path.exists():
            raise RuntimeError(
                f"Sintel RGB {rgb_path} missing for depth {scene}/{frame}. "
                f"Ensure MPI-Sintel-complete.zip's '{pass_name}' pass is extracted "
                f"into the same tree as the depth package."
            )

        rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        depth = _read_sintel_dpt(depth_path).astype(np.float32)
        valid_mask = (depth > 0) & (depth < max_depth) & np.isfinite(depth)

        cam_path = cam_root / scene / f"{frame}.cam"
        K, pose = None, None
        if cam_path.exists():
            K_raw, R_raw, t_raw = _read_sintel_cam(cam_path)
            K = K_raw
            pose = _cam_to_world_pose(R_raw, t_raw)
        elif require_cam:
            raise RuntimeError(
                f"Sintel camera file {cam_path} missing (required: require_cam=True, "
                f"needed for --eval-mode temporal's geometric TAE). It ships in the "
                f"SAME MPI-Sintel-depth-training-20150305.zip as the .dpt depth files "
                f"under training/camdata_left/ — re-extract that zip if missing."
            )

        samples.append({
            "rgb": rgb, "depth": depth, "valid_mask": valid_mask,
            "scene": scene, "frame_idx": frame_idx, "K": K, "pose": pose,
        })

    return samples


# Central registry so run_pareto_benchmark_suite.py can dispatch on --dataset
# without hardcoding NYUv2. Each entry: the loader, the cache subdir under
# benchmark_data/, and the gt_range (metres) passed to compute_gt_depth_metrics.
#
# These EXACTLY mirror the (min_depth_eval, max_depth_eval) constants in the
# cloned Video-Depth-Anything repo's own benchmark protocol
# (Video-Depth-Anything/benchmark/eval/eval.py, per-dataset branches in main()):
#   nyuv2:  min=0.1, max=10.0
#   kitti:  min=0.1, max=80.0
#   sintel: min=0.1, max=70.0   <- NOT 80.0; verified by reading eval.py directly
#           (an earlier imprecise grep for "80.0" false-matched the kitti
#           branch and produced a wrong 80.0 guess for Sintel — corrected here
#           against the actual source, docs/optimization_ledger.md T11).
# Mirroring these makes our numbers directly comparable to VDA's own published
# tables, not just "a reasonable cap we picked."
#
# 'auto_download' marks whether the loader will fetch on its own (only NYU
# does; the big multi-zip datasets go through download_datasets.sh).
DATASET_GT_CONFIG = {
    "nyuv2":  {"loader": load_nyuv2_gt_test_split, "cache_subdir": "nyuv2_gt", "gt_range": (0.1, 10.0), "auto_download": True},
    "kitti":  {"loader": load_kitti_gt,            "cache_subdir": "kitti",    "gt_range": (0.1, 80.0), "auto_download": False},
    "sintel": {"loader": load_sintel_gt,           "cache_subdir": "sintel",   "gt_range": (0.1, 70.0), "auto_download": False},
}


def load_gt_dataset(name: str, data_dir: Path, max_samples: Optional[int] = None, require_cam: bool = False):
    """
    Dispatch to the right GT loader for `name`, returning
    (samples, gt_range). Raises for datasets without a real loader (e.g.
    'davis' has no depth GT; 'scannet' is ToU-gated) rather than faking data.

    Args:
        require_cam: Passed through to loaders that support camera-pose
                     loading (currently only Sintel). Set True for
                     --eval-mode temporal, which needs real K/pose for
                     geometric TAE; ignored (not passed) for loaders that
                     don't accept it, since accuracy-only eval never needs it.
    """
    name = name.lower()
    if name not in DATASET_GT_CONFIG:
        raise ValueError(
            f"No ground-truth loader for dataset '{name}'. Implemented: "
            f"{sorted(DATASET_GT_CONFIG)}. (DAVIS has no depth GT — TAE only; "
            f"ScanNet is ToU-gated, see scripts/download_datasets.sh.)"
        )
    cfg = DATASET_GT_CONFIG[name]
    data_dir = Path(data_dir)
    cache = data_dir / cfg["cache_subdir"]
    kwargs = {"max_samples": max_samples, "download": cfg["auto_download"]}
    if cfg["loader"] is load_sintel_gt:
        kwargs["require_cam"] = require_cam
    samples = cfg["loader"](cache, **kwargs)
    return samples, cfg["gt_range"]


def group_samples_by_scene(samples):
    """
    Groups a flat sample list (each with a 'scene' key) into an ordered dict
    {scene_name: [samples sorted by frame_idx]}, for --eval-mode temporal.

    Raises if any sample has scene=None (NYUv2/KITTI are independent single
    images with no video structure — grouping them into "sequences" would be
    meaningless and could silently fabricate a fake temporal-consistency
    signal from unrelated frames). Only real video datasets (currently
    Sintel) are groupable.
    """
    from collections import OrderedDict

    groups = OrderedDict()
    for s in samples:
        scene = s.get("scene")
        if scene is None:
            raise ValueError(
                "Cannot group samples with scene=None into video sequences. This "
                "dataset has no real scene/temporal structure (e.g. NYUv2/KITTI "
                "are independent single images, not consecutive video frames). "
                "--eval-mode temporal currently only supports Sintel."
            )
        groups.setdefault(scene, []).append(s)
    for scene in groups:
        groups[scene] = sorted(groups[scene], key=lambda s: s["frame_idx"])
    return groups


def chunk_scene_into_windows(frames, window: int = 16):
    """
    Splits a scene's ordered frame list into consecutive windows of length
    `window`, for feeding real consecutive frames through the model's video
    path (video_length=window) in a single forward call — unlike an isolated
    static-clip-per-frame hack, this actually gives the model shared temporal
    context across the window, which is required for TAE to measure anything
    meaningful (T9).

    The final window, if shorter than `window`, is static-padded by repeating
    its last real frame (mirrors VDA's own infer_video_depth chunking
    convention, which pads incomplete trailing chunks the same way) so every
    window handed to the model has uniform length.

    Returns a list of (window_frames, n_real) tuples: window_frames always has
    exactly `window` entries; n_real (<= window) is how many are genuine
    frames. Callers MUST discard the padded tail (window_frames[n_real:]) from
    any downstream temporal-consistency comparison — comparing a frame against
    its own padding copy would trivially and falsely read as perfect
    consistency. Concatenating just the first n_real predictions from each
    window (in order) reconstructs the full per-frame prediction sequence for
    the scene; consecutive-pair TAE over that flat sequence then naturally
    includes window-boundary pairs too (frames that were in different forward
    calls and so did NOT share cross-window model state/cache) — a known,
    documented limitation, not a hidden one.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if not frames:
        return []
    result = []
    n = len(frames)
    for start in range(0, n, window):
        chunk = frames[start:start + window]
        n_real = len(chunk)
        if n_real < window:
            chunk = chunk + [chunk[-1]] * (window - n_real)
        result.append((chunk, n_real))
    return result
