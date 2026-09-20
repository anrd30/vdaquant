"""
Render a qualitative figure of the co-visibility mask on a Sintel frame pair.

Answers the reviewer question: which 14 percent of pixels does cov-TAE
actually exclude?  Five-panel layout, one row:

    RGB(t+1) | GT depth(t+1) | standard Omega | cov Omega | excluded (Omega \\ Omega_cov)

The excluded panel is heat-coloured so you can immediately see the
excluded pixels concentrate at disocclusion boundaries and out-of-frame
reprojections (the story the paper tells) rather than something
suspicious like whole textured regions.

Usage (on the A100 or any GPU with Sintel accessible):

  python scripts/dump_cov_mask_figure.py --scene ambush_2 --frame 0
  python scripts/dump_cov_mask_figure.py --scene market_5 --frame 8
  python scripts/dump_cov_mask_figure.py --scene temple_3 --frame 4 --tau 0.05

Output:

  outputs/mask_figure/cov_mask_<scene>_f<frame>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Make repo-relative imports work.
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from datasets_gt import load_gt_dataset, group_samples_by_scene
from run_pareto_benchmark_suite import _covisibility_mask


def _colourise_depth(d: np.ndarray) -> np.ndarray:
    """Robust per-frame min-max normalise (H, W) depth -> BGR uint8 heatmap."""
    import cv2
    lo, hi = np.percentile(d[d > 0], 2), np.percentile(d[d > 0], 98)
    dn = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    d8 = (dn * 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)


def _mask_overlay(rgb: np.ndarray, mask: np.ndarray, colour_true=(0, 255, 0),
                  colour_false=(0, 0, 200), alpha: float = 0.55) -> np.ndarray:
    """Green where True, red where False, blended over the RGB frame."""
    out = rgb.astype(np.float32).copy()
    layer = np.zeros_like(out)
    layer[mask]  = colour_true
    layer[~mask] = colour_false
    return np.clip(out * (1 - alpha) + layer * alpha, 0, 255).astype(np.uint8)


def _excluded_heat(rgb: np.ndarray, excluded: np.ndarray) -> np.ndarray:
    """RGB darkened with excluded pixels highlighted in bright yellow."""
    out = (rgb.astype(np.float32) * 0.35).astype(np.uint8)
    out[excluded] = (0, 255, 255)   # BGR yellow
    return out


def _label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    """Small top-left title bar so panels are self-identifying."""
    import cv2
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (max(120, 10 * len(text)), 24), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_figure(scene: str, frame: int, tau: float,
                  data_dir: Path, out_dir: Path,
                  device: str = "cuda") -> Path:
    import cv2

    # Load Sintel scene (all frames), then pick the (frame, frame+1) pair.
    samples, _meta = load_gt_dataset("sintel", data_dir)
    per_scene = group_samples_by_scene(samples)
    if scene not in per_scene:
        raise ValueError(f"scene '{scene}' not found. Available: {sorted(per_scene)[:10]}...")
    scene_samples = per_scene[scene]
    scene_samples = sorted(scene_samples, key=lambda s: int(s.get("frame_idx", 0)))
    if frame + 1 >= len(scene_samples):
        raise ValueError(f"frame {frame} out of range for scene {scene} "
                         f"(has {len(scene_samples)} frames)")

    s0 = scene_samples[frame]
    s1 = scene_samples[frame + 1]

    rgb0 = s0["rgb"]                                     # (H, W, 3) uint8 RGB
    rgb1 = s1["rgb"]
    gt0  = torch.from_numpy(s0["depth"].astype(np.float32)).to(device)
    gt1  = torch.from_numpy(s1["depth"].astype(np.float32)).to(device)
    K0   = torch.from_numpy(s0["K"].astype(np.float32)).to(device)
    K1   = torch.from_numpy(s1["K"].astype(np.float32)).to(device)
    P0   = torch.from_numpy(s0["pose"].astype(np.float32)).to(device)  # (4,4) cam-to-world
    P1   = torch.from_numpy(s1["pose"].astype(np.float32)).to(device)

    # Relative pose src=frame0 -> dst=frame1 (VDA convention).
    P0_inv = torch.linalg.inv(P0)
    T_rel = P1 @ P0_inv                                  # world-inv then to frame1
    R_1_0 = T_rel[:3, :3]
    t_1_0 = T_rel[:3, 3]

    H, W = gt1.shape
    # Standard Omega: destination pixels that receive at least one splat.
    # Reproduce the reprojection scatter directly.
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij',
    )
    fx, fy, cx, cy = K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]
    X = (xx - cx) * gt0 / fx
    Y = (yy - cy) * gt0 / fy
    Z = gt0
    pts = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=1)
    pts_t = pts @ R_1_0.T + t_1_0
    Xp = torch.round((pts_t[:, 0] * fx) / pts_t[:, 2] + cx).long()
    Yp = torch.round((pts_t[:, 1] * fy) / pts_t[:, 2] + cy).long()
    in_bounds = ((Xp >= 0) & (Xp < W) & (Yp >= 0) & (Yp < H) & (pts_t[:, 2] > 0))
    omega_std = torch.zeros(H * W, dtype=torch.bool, device=device)
    omega_std[Yp[in_bounds] * W + Xp[in_bounds]] = True
    omega_std = omega_std.reshape(H, W)
    # Also require destination GT depth valid (excludes out-of-range GT).
    gt1_valid = (gt1 > 0.1) & (gt1 < 70.0)
    omega_std &= gt1_valid

    # cov-visibility mask: same scatter, agree with destination GT to within tau.
    omega_cov = _covisibility_mask(gt0, gt1, R_1_0, t_1_0, K1, tau=tau,
                                   scatter_zbuffer=True)
    omega_cov &= gt1_valid

    # Excluded from standard by the co-visibility test (the 14%).
    excluded = omega_std & ~omega_cov

    frac_std = float(omega_std.float().mean())
    frac_cov = float(omega_cov.float().mean())
    frac_excluded = float(excluded.float().mean())
    print(f"  standard Omega  fraction: {frac_std:.3f}")
    print(f"  cov Omega       fraction: {frac_cov:.3f}")
    print(f"  excluded pixels fraction: {frac_excluded:.3f} "
          f"({frac_excluded / max(frac_std, 1e-6):.1%} of standard)")

    # ---- Compose the 5-panel figure ----
    # BGR panels for OpenCV.
    rgb_bgr = cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR)
    gt_bgr = _colourise_depth(gt1.cpu().numpy())
    omega_std_np = omega_std.cpu().numpy()
    omega_cov_np = omega_cov.cpu().numpy()
    excluded_np  = excluded.cpu().numpy()

    p1 = _label(rgb_bgr,                                             "RGB t+1")
    p2 = _label(gt_bgr,                                              "GT depth t+1")
    p3 = _label(_mask_overlay(rgb_bgr, omega_std_np),                f"Standard Omega ({frac_std:.2f})")
    p4 = _label(_mask_overlay(rgb_bgr, omega_cov_np),                f"Omega_cov (tau={tau}, {frac_cov:.2f})")
    p5 = _label(_excluded_heat(rgb_bgr, excluded_np),                f"Excluded ({frac_excluded:.2f})")

    gap = np.zeros((H, 6, 3), dtype=np.uint8)
    strip = np.concatenate([p1, gap, p2, gap, p3, gap, p4, gap, p5], axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cov_mask_{scene}_f{frame:04d}.png"
    cv2.imwrite(str(out_path), strip)
    print(f"  wrote {out_path}  ({strip.shape[1]} x {strip.shape[0]})")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="ambush_2",
                    help="Sintel scene name (default: ambush_2, a heavy-motion scene "
                         "where excluded pixels concentrate at disocclusions).")
    ap.add_argument("--frame", type=int, default=8,
                    help="Frame index within the scene; pair is (frame, frame+1).")
    ap.add_argument("--tau", type=float, default=0.05,
                    help="Depth agreement tolerance (headline value = 0.05).")
    ap.add_argument("--data-dir", default=None,
                    help="Root of benchmark_data/; defaults to <repo>/benchmark_data.")
    ap.add_argument("--output-dir", default="outputs/mask_figure")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _REPO_ROOT / "benchmark_data"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    render_figure(args.scene, args.frame, args.tau,
                  data_dir=data_dir, out_dir=Path(args.output_dir),
                  device=device)


if __name__ == "__main__":
    main()
