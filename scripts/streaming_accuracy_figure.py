"""
Per-frame accuracy trace figure for Paper 2.

Reviewer complaint we're pre-empting: "Aggregate delta1 hides
frame-level drift; how does BW16 accuracy behave over the course of a
long clip?"

This script runs VDA on ONE Sintel scene end-to-end in two configs
(FP32 baseline, and BW16-3b simulator - the accuracy-claimed path)
and logs per-frame delta1 / AbsRel / RMSE against Sintel's ground-truth
depth. Also logs per-frame-pair TAE for a temporal-consistency curve.

Output: a two-panel figure ('quality over time') suitable for §3 or §5:
    top:    per-frame delta1 (FP32 vs BW16-3b, overlaid)
    bottom: per-frame TAE     (FP32 vs BW16-3b, overlaid)

Also dumps a JSON with all per-frame numbers so we can regenerate the
figure without rerunning inference.

Usage:
  python scripts/streaming_accuracy_figure.py                # bamboo_1
  python scripts/streaming_accuracy_figure.py --scene alley_1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
for _p in [
    REPO_ROOT / "Video-Depth-Anything",
    REPO_ROOT.parent / "Video-Depth-Anything",
    Path("/content/Video-Depth-Anything"),
]:
    if _p.exists() and (_p / "video_depth_anything").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from datasets_gt import (
    load_gt_dataset, group_samples_by_scene,
    compute_gt_depth_metrics,
)


def _load_vda(encoder: str, ckpt: Path, device: str = "cuda"):
    from video_depth_anything.video_depth import VideoDepthAnything
    cfg = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }[encoder]
    model = VideoDepthAnything(**cfg)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
    return model.to(device).eval()


def _apply_bw16(model, bits: int = 3):
    from research.models import apply_rotated_quantization_to_vda
    return apply_rotated_quantization_to_vda(
        model, bits=bits, quantizer='lattice_bw16', use_qjl=True,
        replace_backbone=False, replace_temporal=True, verbose=True,
    )


def _tae_pair(pred_t, pred_tp1, gt_t, gt_tp1, R_1_0, t_1_0, K):
    """Simple TAE for one frame pair: relative disparity change under GT reprojection.

    Uses standard-mask (any pixel that reprojects into the frame with valid GT)
    to keep this figure comparable to the paper's non-cov-TAE baseline.
    """
    device = pred_t.device
    H, W = gt_t.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device), indexing='ij')
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (xx - cx) * gt_t / fx; Y = (yy - cy) * gt_t / fy; Z = gt_t
    pts = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), 1)
    pt = pts @ R_1_0.T + t_1_0
    Xp = torch.round((pt[:, 0] * fx) / pt[:, 2] + cx).long()
    Yp = torch.round((pt[:, 1] * fy) / pt[:, 2] + cy).long()
    ok = (Xp >= 0) & (Xp < W) & (Yp >= 0) & (Yp < H) & (pt[:, 2] > 0)
    if ok.sum() < 100:
        return float('nan')
    # Warp pred_t via forward scatter (last-write-wins), then diff with pred_tp1.
    warped = torch.zeros_like(pred_tp1)
    src_disp = 1.0 / gt_t.clamp(min=0.01).flatten()
    dst_disp = torch.zeros(H * W, device=device)
    valid = ok
    dst_idx = Yp[valid] * W + Xp[valid]
    dst_disp[dst_idx] = src_disp[valid]
    warped = dst_disp.reshape(H, W)
    mask = warped > 0
    if mask.sum() < 100:
        return float('nan')
    pred_disp_tp1 = 1.0 / pred_tp1.clamp(min=1e-3)
    a = warped[mask]; b = pred_disp_tp1[mask]
    return float(((a - b).abs() / a.clamp(min=1e-3)).mean() * 100)


def _run_scene(model, scene_samples, input_size: int, fp32: bool, device: str):
    frames_rgb = np.stack([s["rgb"] for s in scene_samples], axis=0)
    depth_out, _ = model.infer_video_depth(
        frames_rgb, target_fps=24, input_size=input_size, device=device, fp32=fp32,
    )
    return depth_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="bamboo_1")
    ap.add_argument("--encoder", default="vits")
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "checkpoints" / "video_depth_anything_vits.pth"))
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "benchmark_data"))
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "streaming_figure"))
    args = ap.parse_args()

    device = "cuda"
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    samples, gt_range = load_gt_dataset("sintel", Path(args.data_dir), require_cam=True)
    scenes = group_samples_by_scene(samples)
    if args.scene not in scenes:
        raise SystemExit(f"scene '{args.scene}' missing. Have: {sorted(scenes)[:12]}...")
    scene_samples = sorted(scenes[args.scene], key=lambda s: int(s.get("frame_idx", 0)))
    N = len(scene_samples)
    print(f"[Sintel] scene={args.scene}  frames={N}")

    per_config = {}
    for cfg_name, apply_quant, fp32 in [
        ("FP32_Baseline", False, True),
        ("BW16_3b_sim",   True,  False),
    ]:
        print(f"\n--- Config: {cfg_name} ---")
        model = _load_vda(args.encoder, Path(args.ckpt), device)
        if apply_quant:
            model = _apply_bw16(model, bits=3)
        depth = _run_scene(model, scene_samples, args.input_size, fp32, device)
        # depth shape: (N, H_org, W_org) numpy
        # Compute per-frame δ1 / AbsRel etc.
        per_frame = []
        for i, s in enumerate(scene_samples):
            gt_d = torch.from_numpy(s["depth"].astype(np.float32)).to(device)
            valid = torch.from_numpy(s["valid_mask"]).to(device)
            pred = torch.from_numpy(depth[i].astype(np.float32)).to(device)
            m = compute_gt_depth_metrics(pred, gt_d, valid_mask=valid,
                                          gt_range=gt_range, pred_is_disparity=True)
            per_frame.append(m)
        # Per-pair TAE.
        tae_series = [float('nan')]
        for i in range(1, N):
            s0, s1 = scene_samples[i - 1], scene_samples[i]
            gt0 = torch.from_numpy(s0["depth"].astype(np.float32)).to(device)
            gt1 = torch.from_numpy(s1["depth"].astype(np.float32)).to(device)
            K1 = torch.from_numpy(s1["K"].astype(np.float32)).to(device)
            P0 = torch.from_numpy(s0["pose"].astype(np.float32)).to(device)
            P1 = torch.from_numpy(s1["pose"].astype(np.float32)).to(device)
            T_rel = P1 @ torch.linalg.inv(P0)
            R_1_0 = T_rel[:3, :3]; t_1_0 = T_rel[:3, 3]
            p0 = torch.from_numpy(depth[i - 1].astype(np.float32)).to(device)
            p1 = torch.from_numpy(depth[i].astype(np.float32)).to(device)
            tae_series.append(_tae_pair(p0, p1, gt0, gt1, R_1_0, t_1_0, K1))
        per_config[cfg_name] = {"per_frame": per_frame, "tae": tae_series}
        del model; torch.cuda.empty_cache()

    # ---- Save numbers ----
    out_json = out_dir / f"streaming_{args.scene}.json"
    with open(out_json, "w") as f:
        json.dump({"scene": args.scene, "n_frames": N, "configs": per_config}, f, indent=2)
    print(f"\nSaved numbers to {out_json}")

    # ---- Figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    xs = np.arange(N)
    colors = {"FP32_Baseline": "#2E7D32", "BW16_3b_sim": "#C62828"}
    for cfg_name, res in per_config.items():
        d1 = [m["delta1"] for m in res["per_frame"]]
        ax1.plot(xs, d1, label=cfg_name, color=colors[cfg_name], linewidth=1.4)
        ax2.plot(xs, res["tae"], label=cfg_name, color=colors[cfg_name], linewidth=1.4)
    ax1.set_ylabel(r"$\delta_1$")
    ax1.set_ylim(0.0, 1.02)
    ax1.grid(alpha=0.3); ax1.legend(loc="lower left", frameon=False)
    ax1.set_title(f"Per-frame accuracy on Sintel {args.scene} ({N} frames)")
    ax2.set_ylabel("TAE (%)")
    ax2.set_xlabel("Frame index")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / f"streaming_{args.scene}.pdf"
    fig.savefig(fig_path, dpi=150)
    fig.savefig(fig_path.with_suffix(".png"), dpi=150)
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
