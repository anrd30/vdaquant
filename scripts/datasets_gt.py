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
    pred_inv_depth: torch.Tensor,
    gt_inv_depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    gt_range: Tuple[float, float] = (0.1, 10.0),
) -> Dict[str, float]:
    """
    Affine-invariant ground-truth depth evaluation protocol.

    1. Restrict to pixels where valid_mask is True AND gt is inside gt_range
       (the standard NYUv2 evaluation range in meters for the inverse-depth
       convention used here — swap to your dataset's convention/units as
       needed).
    2. Fit a single per-image affine alignment (s, t) via least squares on
       those pixels (see affine_align_inverse_depth).
    3. Compute AbsRel, RMSE, and delta1/delta2/delta3 on the ALIGNED
       prediction, restricted to the same valid pixels.

    Args:
        pred_inv_depth: Model's predicted (inverse) depth, any shape.
        gt_inv_depth: Ground-truth (inverse) depth, same shape.
        valid_mask: Optional additional boolean mask (e.g. sensor dropout
                    regions in NYUv2's raw depth maps). If None, all pixels
                    are considered valid before the gt_range restriction.
        gt_range: (min, max) — pixels with gt outside this range are
                  excluded from BOTH fitting and evaluation.

    Returns:
        Dict with abs_rel, rmse, delta1, delta2, delta3, affine_scale,
        affine_shift, n_valid_pixels.
    """
    if valid_mask is None:
        valid_mask = torch.ones_like(gt_inv_depth, dtype=torch.bool)

    range_mask = (gt_inv_depth > gt_range[0]) & (gt_inv_depth < gt_range[1])
    mask = valid_mask & range_mask

    if not mask.any():
        raise ValueError(
            f"No valid pixels after applying valid_mask and gt_range={gt_range}"
        )

    s, t, pred_aligned = affine_align_inverse_depth(pred_inv_depth, gt_inv_depth, mask)

    p = torch.clamp(pred_aligned[mask], min=1e-6)
    g = torch.clamp(gt_inv_depth[mask], min=1e-6)

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

    data = scipy.io.loadmat(str(mat_path))
    images = data["images"]  # (H, W, 3, N)
    depths = data["depths"]  # (H, W, N), meters

    if max_samples is not None:
        test_indices = test_indices[:max_samples]

    samples = []
    for idx in test_indices:
        rgb = images[:, :, :, idx].astype(np.uint8)
        depth = depths[:, :, idx].astype(np.float32)
        valid_mask = depth > 0
        samples.append({"rgb": rgb, "depth": depth, "valid_mask": valid_mask})

    return samples
