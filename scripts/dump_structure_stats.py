#!/usr/bin/env python
"""
S10: spatial-structure diagnostic for the TAE-gameability finding (F16,
docs/optimization_ledger.md). At 2-bit E8 on Sintel, TAE improved 7.5x over
FP32 while delta1 collapsed (0.71 -> 0.50) -- per-scene evidence pointed to
a degenerate, structureless prediction rather than genuine temporal
consistency (hard scenes fell toward a floor, easy scenes got WORSE). This
script quantifies that structure loss directly instead of arguing it.

For FP32 and each swept bit-width, computes four metrics over the predicted
DISPARITY map (the model's native output space -- do NOT invert to depth;
that's a nonlinear transform and would confound the structure measurement):

    grad_energy    mean Sobel-gradient magnitude (edge/detail energy)
    laplacian_var  variance of the Laplacian (Pech-Pacheco et al. 2000
                   blur/focus measure -- standard in image-quality work)
    entropy_8bit   Shannon entropy (bits) of the 256-bin value histogram
    pred_std       spatial standard deviation of the prediction

CRITICAL DESIGN CHOICE (a reviewer WILL ask about this): every metric is
computed on a PER-FRAME MIN-MAX NORMALIZED [0,1] copy of the prediction,
never the raw values. A quantized model's output can sit at a different
overall scale/offset than FP32's for reasons that have nothing to do with
structure (e.g. a shifted disparity range) -- normalizing first means an
affine rescale can never masquerade as "structure loss". We are measuring
whether the SHAPE of the prediction collapsed, not whether its magnitude did.

Usage:
    python scripts/dump_structure_stats.py --dataset sintel --quantizer lattice_e8 \
        --scale-bits 8 --bits 8 4 3 2 --no-qjl --max-samples 200 \
        --output-dir outputs/phase4/g11_structure

INTERPRETATION KEY:
    grad_energy and laplacian_var drop SHARPLY at some bit-width while
    entropy_8bit also drops and pred_std often does too
        -> degeneracy CONFIRMED at that bit-width: the prediction has lost
           spatial structure, and any TAE improvement there is an artifact
           of having nothing left to misalign (F16), not genuine temporal
           consistency. Safe to state F16 as fact in the paper.
    metrics track FP32 closely even where TAE improves
        -> the degeneracy hypothesis is WRONG for that config -- do NOT
           assert F16 there without another explanation. Tell the boss
           before writing anything.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================
# METRICS -- pure functions, no model/dataset required
# ============================================================

def _minmax_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Per-frame min-max normalization to [0,1]. A constant frame (min==max)
    normalizes to all-zeros rather than dividing by zero -- deliberate: a
    perfectly flat prediction has zero gradient/Laplacian-variance/entropy
    regardless of what constant it sits at, so mapping it to an all-zero
    frame doesn't change any downstream metric's correct value.
    """
    x = x.float()
    lo = x.min()
    hi = x.max()
    span = (hi - lo).clamp(min=1e-8)
    return (x - lo) / span


def grad_energy(x_norm: torch.Tensor) -> float:
    """Mean Sobel-gradient magnitude on an already-normalized (H,W) frame."""
    x = x_norm.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    dx = F.conv2d(x, sobel_x, padding=1)
    dy = F.conv2d(x, sobel_y, padding=1)
    # No backward() ever runs on this (pure eval diagnostic), so no epsilon
    # is needed to protect sqrt's gradient at 0 -- and a flat frame must
    # produce an EXACT 0.0, not a rounding-epsilon-sized false positive.
    mag = torch.sqrt(dx ** 2 + dy ** 2)
    return float(mag.mean())


def laplacian_var(x_norm: torch.Tensor) -> float:
    """Variance of the Laplacian response (Pech-Pacheco et al., 2000
    blur/focus-detection measure) on an already-normalized (H,W) frame."""
    x = x_norm.unsqueeze(0).unsqueeze(0)
    kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
    lap = F.conv2d(x, kernel, padding=1)
    return float(lap.var())


def entropy_8bit(x_norm: torch.Tensor) -> float:
    """Shannon entropy (bits) of the 256-bin histogram of an already-
    normalized [0,1] (H,W) frame."""
    hist = torch.histc(x_norm.float(), bins=256, min=0.0, max=1.0)
    total = hist.sum().clamp(min=1e-12)
    p = hist / total
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum())


def pred_std(x_norm: torch.Tensor) -> float:
    """Spatial standard deviation of an already-normalized (H,W) frame."""
    return float(x_norm.std())


def compute_structure_stats(x: torch.Tensor) -> dict:
    """
    x: a single (H,W) prediction frame, ANY numeric range. Normalizes
    min-max to [0,1] ONCE, then computes all four metrics on that same
    normalized copy -- see module docstring for why normalization comes
    first (structure vs scale).
    """
    x_norm = _minmax_normalize(x)
    return {
        "grad_energy": round(grad_energy(x_norm), 6),
        "laplacian_var": round(laplacian_var(x_norm), 6),
        "entropy_8bit": round(entropy_8bit(x_norm), 6),
        "pred_std": round(pred_std(x_norm), 6),
    }


def average_stats(frame_stats: list) -> dict:
    """Averages a list of per-frame compute_structure_stats() dicts."""
    keys = frame_stats[0].keys()
    return {k: round(float(np.mean([fs[k] for fs in frame_stats])), 6) for k in keys}


# ============================================================
# REAL MODE -- actual VDA model + dataset frames
# ============================================================

def _load_real_model():
    import video_depth_anything.dinov2_layers.attention as dino_attn
    import video_depth_anything.motion_module.attention as motion_attn
    if not torch.cuda.is_available():
        dino_attn.memory_efficient_attention = lambda q, k, v, attn_bias=None, **kwargs: \
            torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        motion_attn.XFORMERS_AVAILABLE = False
    from video_depth_anything.video_depth import VideoDepthAnything
    model_configs = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    model = VideoDepthAnything(**model_configs).eval()

    possible_ckpts = [
        REPO_ROOT / "checkpoints" / "video_depth_anything_vits.pth",
        REPO_ROOT / "video_depth_anything_vits.pth",
        Path("/content/video_depth_anything_vits.pth"),
        Path("/content/vdaquant/checkpoints/video_depth_anything_vits.pth"),
    ]
    for ckpt_path in possible_ckpts:
        if ckpt_path.exists() and ckpt_path.stat().st_size >= 10_000_000:
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            if torch.cuda.is_available():
                model = model.cuda()
            else:
                for m in model.modules():
                    if hasattr(m, "_use_memory_efficient_attention_xformers"):
                        m._use_memory_efficient_attention_xformers = False
            return model
    raise RuntimeError(
        "No VDA checkpoint found (checked checkpoints/video_depth_anything_vits.pth and "
        "friends). Run scripts/run_pareto_benchmark_suite.py once first (it auto-downloads "
        "the checkpoint)."
    )


def predict_disparity(m, rgb_np: np.ndarray) -> torch.Tensor:
    """Same static-3-frame-clip convention as run_groundtruth_eval's
    predict_depth in run_pareto_benchmark_suite.py: a video model needs a
    T-dim even for a single still image; take the last frame's prediction,
    upsampled back to native resolution."""
    from run_pareto_benchmark_suite import _preprocess_frame
    H, W = rgb_np.shape[:2]
    tensor = _preprocess_frame(rgb_np)
    clip = tensor.unsqueeze(0).repeat(3, 1, 1, 1).unsqueeze(0)  # [1, T=3, C, h, w]
    if torch.cuda.is_available():
        clip = clip.cuda()
    with torch.no_grad():
        out = m(clip)
    pred = out[0, -1] if out.dim() == 4 else out[0]
    pred = F.interpolate(pred[None, None].float(), size=(H, W), mode="bilinear", align_corners=True)[0, 0]
    return pred.cpu()


def run_structure_eval(args, data_dir: Path) -> dict:
    from datasets_gt import load_gt_dataset
    from run_pareto_benchmark_suite import compute_real_bit_accounting, resolve_group_size
    from research.models.rotated_attention import apply_rotated_quantization_to_vda

    samples, _ = load_gt_dataset(args.dataset, data_dir, max_samples=args.max_samples)
    rgb_frames = [s["rgb"] for s in samples]
    print(f"  [Dataset] Loaded {len(rgb_frames)} {args.dataset.upper()} frames "
          f"(GT ignored -- structure-only diagnostic, no accuracy claim here).")

    configs = {}

    print("  [1/N] FP32 baseline...")
    model = _load_real_model()
    fp32_frame_stats = [compute_structure_stats(predict_disparity(model, rgb)) for rgb in rgb_frames]
    configs["FP32_Baseline"] = average_stats(fp32_frame_stats)
    configs["FP32_Baseline"]["effective_bits_per_scalar"] = 32.0

    for idx, bit in enumerate(args.bits):
        print(f"  [{idx + 2}/{len(args.bits) + 1}] {bit}-bit {args.quantizer}...")
        model_q = _load_real_model()
        model_q = apply_rotated_quantization_to_vda(
            model_q, bits=bit, quantizer=args.quantizer, use_qjl=args.use_qjl,
            scale_bits=args.scale_bits, verbose=(idx == 0),
        )
        frame_stats = [compute_structure_stats(predict_disparity(model_q, rgb)) for rgb in rgb_frames]
        cfg = average_stats(frame_stats)
        bit_accounting = compute_real_bit_accounting(
            bit, head_dim=64, use_qjl=args.use_qjl,
            group_size=resolve_group_size(args.quantizer, 64), scale_bits=args.scale_bits,
        )
        cfg["effective_bits_per_scalar"] = bit_accounting["effective_bits_per_scalar"]
        configs[f"{bit}bit"] = cfg
        print(f"        -> grad_energy={cfg['grad_energy']} laplacian_var={cfg['laplacian_var']} "
              f"entropy_8bit={cfg['entropy_8bit']} pred_std={cfg['pred_std']} "
              f"(eff={cfg['effective_bits_per_scalar']}b/scalar)")

    return configs


# ============================================================
# OUTPUT
# ============================================================

METRIC_NAMES = ["grad_energy", "laplacian_var", "entropy_8bit", "pred_std"]


def _write_markdown(configs: dict, output_dir: Path, dataset: str):
    lines = [f"# Structure diagnostic ({dataset}) -- ledger F16/S10", "",
             "| Config | eff bits | grad_energy | laplacian_var | entropy_8bit | pred_std |",
             "|---|---|---|---|---|---|"]
    for name, cfg in configs.items():
        lines.append(
            f"| {name} | {cfg['effective_bits_per_scalar']} | {cfg['grad_energy']} | "
            f"{cfg['laplacian_var']} | {cfg['entropy_8bit']} | {cfg['pred_std']} |"
        )
    text = "\n".join(lines) + "\n"
    (output_dir / "structure_stats.md").write_text(text)
    print(text)


def _plot_metrics(configs: dict, output_dir: Path, dataset: str):
    fp32 = configs.get("FP32_Baseline")
    quant_names = [n for n in configs if n != "FP32_Baseline"]
    # Sort quantized configs by effective bits, descending, for a sane x-axis.
    quant_names.sort(key=lambda n: -configs[n]["effective_bits_per_scalar"])
    xs = [configs[n]["effective_bits_per_scalar"] for n in quant_names]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, metric in zip(axes.flat, METRIC_NAMES):
        ys = [configs[n][metric] for n in quant_names]
        ax.plot(xs, ys, marker="o", color="#2b5c8f", label=metric)
        if fp32 is not None:
            ax.axhline(fp32[metric], color="gray", linestyle="--", label="FP32")
        ax.set_xlabel("effective bits/scalar")
        ax.set_title(metric)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)
    fig.suptitle(f"Structure diagnostic vs bit-rate ({dataset})")
    plt.tight_layout()
    out_png = output_dir / "structure_stats.png"
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=str, default="sintel")
    ap.add_argument("--quantizer", type=str, default="lattice_e8",
                     choices=["scalar", "scalar_g8", "uniform_vector", "lattice_d4", "lattice_e8"])
    ap.add_argument("--scale-bits", type=int, default=8, choices=[8, 16])
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 4, 3, 2])
    ap.add_argument("--qjl", dest="use_qjl", action="store_true", default=False)
    ap.add_argument("--no-qjl", dest="use_qjl", action="store_false")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--output-dir", type=str, default="outputs/phase4/g11_structure")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = REPO_ROOT / "benchmark_data"

    configs = run_structure_eval(args, data_dir)

    out_json = output_dir / "structure_stats.json"
    with open(out_json, "w") as f:
        json.dump({
            "dataset": args.dataset, "quantizer": args.quantizer,
            "scale_bits": args.scale_bits, "use_qjl": args.use_qjl,
            "configs": configs,
            "note": (
                "All metrics computed on the PER-FRAME MIN-MAX-NORMALIZED predicted "
                "disparity (never raw values, never inverted to depth) -- an affine "
                "rescale of the prediction can never masquerade as structure loss. "
                "See docs/optimization_ledger.md F16 and the module docstring."
            ),
        }, f, indent=2, sort_keys=True)
    print(f"  Wrote {out_json}")

    _write_markdown(configs, output_dir, args.dataset)
    if HAS_MATPLOTLIB:
        _plot_metrics(configs, output_dir, args.dataset)
    else:
        print("  [Warning] matplotlib not found, skipping structure_stats.png.")


if __name__ == "__main__":
    main()
