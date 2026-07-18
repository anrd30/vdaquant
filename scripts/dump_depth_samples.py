#!/usr/bin/env python3
"""
Qualitative depth dump: for a few sample frames, run FP32 + each swept
bit-width and save side-by-side colourised depth strips (and, for video
datasets, a per-bit MP4) so you can EYEBALL what quantization does.

This is a VISUALIZATION tool, not an eval — it colourises the model's raw
disparity output (brighter = closer, VDA convention), so no GT / alignment /
capping is involved here. For numbers use run_pareto_benchmark_suite.py.

Usage (Colab):
  python scripts/dump_depth_samples.py --dataset sintel --num-frames 8 \
    --quantizer lattice_e8 --scale-bits 8 --no-qjl --make-video
  # then download outputs/depth_samples/*.png (and *.mp4)

Output: outputs/depth_samples/
  strip_XXXX.png   horizontal [RGB | FP32 | 8b | 4b | 3b | 2b], labelled
  compare_<scene>.mp4   (video datasets, --make-video) same strip over time
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# Locate the Video-Depth-Anything package (same search the suite uses).
for p in [REPO_ROOT / "Video-Depth-Anything", REPO_ROOT.parent / "Video-Depth-Anything",
          Path("/content/Video-Depth-Anything"), Path("/content/vdaquant/Video-Depth-Anything")]:
    if p.exists() and (p / "video_depth_anything").exists():
        sys.path.insert(0, str(p))
        break

import cv2
from research.models.rotated_attention import apply_rotated_quantization_to_vda


def colourise(depth: np.ndarray) -> np.ndarray:
    """Per-frame min-max normalise a (H,W) disparity map -> BGR uint8 heatmap."""
    d = depth.astype(np.float32)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)  # robust range, ignore outliers
    d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    d8 = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)  # BGR


def label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (max(90, 9 * len(text)), 22), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_model(model_configs):
    from video_depth_anything.video_depth import VideoDepthAnything
    m = VideoDepthAnything(**model_configs).eval()
    ckpts = [REPO_ROOT / "checkpoints" / "video_depth_anything_vits.pth",
             REPO_ROOT / "video_depth_anything_vits.pth",
             Path("/content/vdaquant/checkpoints/video_depth_anything_vits.pth")]
    loaded = False
    for c in ckpts:
        if c.exists() and c.stat().st_size > 10_000_000:
            m.load_state_dict(torch.load(c, map_location="cpu"))
            loaded = True
            print(f"  [Model] loaded {c}")
            break
    if not loaded:
        print("  [Model] WARNING: no checkpoint found — outputs will be from an untrained model.")
    if torch.cuda.is_available():
        m = m.cuda()
    else:
        import video_depth_anything.dinov2_layers.attention as dino_attn
        import video_depth_anything.motion_module.attention as motion_attn
        dino_attn.memory_efficient_attention = lambda q, k, v, attn_bias=None, **kw: \
            torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        motion_attn.XFORMERS_AVAILABLE = False
    return m


def make_transform(h, w):
    import cv2 as _cv2
    from video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
    input_size = 518
    ratio = max(h, w) / min(h, w)
    if ratio > 1.78:
        input_size = int(input_size * 1.777 / ratio)
        input_size = round(input_size / 14) * 14
    return (Resize(width=input_size, height=input_size, resize_target=False, keep_aspect_ratio=True,
                   ensure_multiple_of=14, resize_method='lower_bound',
                   image_interpolation_method=_cv2.INTER_CUBIC),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet())


def predict(m, rgb_np):
    H, W = rgb_np.shape[:2]
    resize, norm, prep = make_transform(H, W)
    sample = prep(norm(resize({'image': rgb_np.astype(np.float32) / 255.0})))
    t = torch.from_numpy(sample['image'])
    clip = t.unsqueeze(0).repeat(3, 1, 1, 1).unsqueeze(0)
    if torch.cuda.is_available():
        clip = clip.cuda()
    with torch.no_grad():
        out = m(clip)
    pred = out[0, -1] if out.dim() == 4 else out[0]
    pred = F.interpolate(pred[None, None].float(), size=(H, W), mode='bilinear', align_corners=True)[0, 0]
    return pred.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="Dump qualitative depth comparisons across bit-widths")
    ap.add_argument("--dataset", default="sintel", choices=["nyuv2", "kitti", "sintel"])
    ap.add_argument("--bits", nargs="+", type=int, default=[8, 4, 3, 2])
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--quantizer", default="lattice_e8",
                    choices=["scalar", "uniform_vector", "lattice_d4", "lattice_e8"])
    ap.add_argument("--scale-bits", type=int, default=8, choices=[8, 16])
    ap.add_argument("--qjl", dest="use_qjl", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--output-dir", default="outputs/depth_samples")
    ap.add_argument("--make-video", action="store_true",
                    help="Also assemble the strips into an MP4 (best for the video datasets).")
    args = ap.parse_args()

    from datasets_gt import load_gt_dataset
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = REPO_ROOT / "benchmark_data"

    # First N frames — for a video dataset these are consecutive (same scene),
    # so the MP4 shows real temporal behaviour.
    samples, _ = load_gt_dataset(args.dataset, data_dir, max_samples=args.num_frames)
    print(f"  [Data] {len(samples)} {args.dataset} frames")

    model_configs = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    fp32 = build_model(model_configs)

    # Pre-build one quantized model per bit-width (fresh surgery from the FP32 weights each time).
    quant_models = {}
    for bit in args.bits:
        mq = build_model(model_configs)
        mq = apply_rotated_quantization_to_vda(
            mq, bits=bit, quantizer=args.quantizer, use_qjl=args.use_qjl,
            scale_bits=args.scale_bits, verbose=False, replace_temporal=True)
        quant_models[bit] = mq
        print(f"  [Surgery] {bit}-bit {args.quantizer} ready")

    strips = []
    for i, s in enumerate(samples):
        rgb = s["rgb"]
        H, W = rgb.shape[:2]
        cols = [label(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "RGB")]
        cols.append(label(colourise(predict(fp32, rgb)), "FP32"))
        for bit in args.bits:
            cols.append(label(colourise(predict(quant_models[bit], rgb)), f"{bit}-bit"))
        # unify heights (all already HxW) and concat horizontally with 4px gaps
        gap = np.zeros((H, 4, 3), dtype=np.uint8)
        strip = cols[0]
        for c in cols[1:]:
            strip = np.concatenate([strip, gap, c], axis=1)
        path = out_dir / f"strip_{i:04d}.png"
        cv2.imwrite(str(path), strip)
        strips.append(strip)
        print(f"  [Saved] {path.name}  ({strip.shape[1]}x{strip.shape[0]})")

    if args.make_video and len(strips) > 1:
        h, w = strips[0].shape[:2]
        vid_path = out_dir / f"compare_{args.dataset}.mp4"
        vw = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 6, (w, h))
        for st in strips:
            if st.shape[:2] != (h, w):
                st = cv2.resize(st, (w, h))
            vw.write(st)
        vw.release()
        print(f"  [Video] {vid_path}  ({len(strips)} frames @6fps)")

    print(f"\nDone. Download everything under: {out_dir}")


if __name__ == "__main__":
    main()
