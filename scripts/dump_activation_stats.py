#!/usr/bin/env python
"""
S3: pre/post-RHT activation outlier statistics (docs/optimization_ledger.md
F14, decision gate DG-2). Answers WHY the rotation ablation (E2/G3) shows
what it shows, by measuring the K/V activation distribution the quantizer
actually sees, with and without the Hadamard rotation.

Runs the model TWICE via the existing surgery machinery, quantizer='identity'
(zero quantization noise -- we are only measuring the input distribution):
    pass 1: rotation ON  -> captures POST-RHT K, V
    pass 2: rotation OFF -> captures RAW K, V
Captured via forward-pre-hooks on k_quantizer/v_quantizer -- i.e. EXACTLY
the tensor a real quantizer would see (post-reshape, per-head, head_dim=64),
never some upstream tensor. See research/models/rotated_attention.py:
K_rot/V_rot are computed, then immediately passed to k_quantizer/v_quantizer.

Usage (real model + dataset, GPU box):
    python scripts/dump_activation_stats.py --dataset nyuv2 --num-images 8 \
        --output-dir outputs/activation_stats

Usage (CPU, no download, no dataset -- what the test suite runs):
    python scripts/dump_activation_stats.py --synthetic --num-images 1 --max-tokens 512

INTERPRETATION KEY (read results with this, no re-derivation needed):
    raw kurtosis ~0 across layers (Gaussian-ish already)
        -> rotation has nothing to fix at 3-4 bits -> supports DG-2 branch (c):
           "video-ViT KV activations don't need rotation at moderate rates."
    raw kurtosis high (>5) but the rotation ablation STILL shows no accuracy
    benefit
        -> suspicious; do not conclude branch (c) -- escalate and re-check
           the rotation ablation plumbing before trusting either result.
    raw kurtosis high AND rotated kurtosis ~0
        -> rotation is doing its job (Ailon-Chazelle gaussianization) and
           SHOULD matter at low bits -> expect G3b (2-bit ablation) to show
           a real gap between rotation on/off.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from research.models.rotated_attention import apply_rotated_quantization_to_vda

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================
# STATISTICS -- computed on exactly what the quantizer sees
# ============================================================

def excess_kurtosis(x: torch.Tensor) -> float:
    """Excess kurtosis (0 for a perfect Gaussian) of all values in x."""
    x = x.flatten().double()
    if x.numel() < 2:
        return 0.0
    diffs = x - x.mean()
    var = (diffs ** 2).mean()
    if var <= 1e-12:
        return 0.0
    m4 = (diffs ** 4).mean()
    return float(m4 / (var ** 2) - 3.0)


def outlier_ratio(x: torch.Tensor) -> float:
    """max|x| / RMS(x): how far the single worst value sits from the bulk."""
    x = x.flatten().double()
    rms = torch.sqrt((x ** 2).mean()).clamp(min=1e-12)
    return float(x.abs().max() / rms)


def channel_spread(x: torch.Tensor) -> float:
    """
    max_c(amax_c) / median_c(amax_c) over the last dimension (channels):
    how much worse the single worst channel's dynamic range is than a
    typical channel's. This is the statistic per-group scale quantizers
    (D4/E8/scalar_g8) are directly sensitive to.
    """
    x2 = x.reshape(-1, x.shape[-1]).double()
    amax_c = x2.abs().amax(dim=0)
    med = amax_c.median().clamp(min=1e-12)
    return float(amax_c.max() / med)


def compute_activation_stats(x: torch.Tensor) -> dict:
    """x: (..., channels), any leading dims. Stats over the flattened set
    of channel-vectors -- exactly what a quantizer's forward() operates on."""
    flat = x.reshape(-1, x.shape[-1])
    return {
        "kurtosis": round(excess_kurtosis(flat), 4),
        "outlier_ratio": round(outlier_ratio(flat), 4),
        "chan_spread": round(channel_spread(flat), 4),
        "n_tokens": int(flat.shape[0]),
    }


# ============================================================
# CAPTURE -- forward-pre-hooks on k_quantizer/v_quantizer
# ============================================================

def register_capture_hooks(model: nn.Module, storage: dict):
    """
    Hooks every k_quantizer/v_quantizer submodule (i.e. every layer the
    surgery replaced) and captures the INPUT to that quantizer's forward --
    K_rot/V_rot, exactly what gets quantized, never an upstream tensor.
    """
    handles = []
    for name, module in model.named_modules():
        if name.endswith("k_quantizer"):
            tensor_type = "K"
        elif name.endswith("v_quantizer"):
            tensor_type = "V"
        else:
            continue
        layer_name = name.rsplit(".", 1)[0]

        def _hook(mod, inputs, layer_name=layer_name, tensor_type=tensor_type):
            x = inputs[0]
            flat = x.detach().float().cpu().reshape(-1, x.shape[-1])
            storage.setdefault((layer_name, tensor_type), []).append(flat)

        handles.append(module.register_forward_pre_hook(_hook))
    return handles


def run_capture_pass(model: nn.Module, frames, forward_fn, max_tokens=None):
    """Runs forward_fn(model, frame) for every frame with capture hooks
    live, then returns (stats_dict, tensors_dict) keyed 'layer::K'/'layer::V'."""
    model.eval()
    storage = {}
    handles = register_capture_hooks(model, storage)
    with torch.no_grad():
        for frame in frames:
            forward_fn(model, frame)
    for h in handles:
        h.remove()

    stats, tensors = {}, {}
    for (layer_name, tensor_type), chunks in storage.items():
        full = torch.cat(chunks, dim=0)
        if max_tokens is not None and full.shape[0] > max_tokens:
            full = full[:max_tokens]
        key = f"{layer_name}::{tensor_type}"
        tensors[key] = full
        stats[key] = compute_activation_stats(full)
    return stats, tensors


# ============================================================
# REAL MODE -- actual VDA model + dataset frames
# ============================================================

def _load_real_vda_model():
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
        "the checkpoint), or pass --synthetic to skip the real model entirely."
    )


def build_model_with_rotation(use_rotation: bool, bits: int = 8):
    model = _load_real_vda_model()
    return apply_rotated_quantization_to_vda(
        model, bits=bits, quantizer="identity", use_qjl=False,
        use_rotation=use_rotation, verbose=False,
    )


def real_mode_frames(dataset: str, num_images: int, data_dir: Path):
    from datasets_gt import load_gt_dataset
    from run_pareto_benchmark_suite import _preprocess_frame

    samples, _ = load_gt_dataset(dataset, data_dir, max_samples=num_images)
    clips = []
    for sample in samples:
        tensor = _preprocess_frame(sample["rgb"])
        # Same static-clip convention as run_groundtruth_eval's predict_depth:
        # a video model needs a T-dim even for a single still image.
        clip = tensor.unsqueeze(0).repeat(3, 1, 1, 1).unsqueeze(0)  # [1, T=3, C, h, w]
        if torch.cuda.is_available():
            clip = clip.cuda()
        clips.append(clip)
    return clips


# ============================================================
# SYNTHETIC MODE -- no download, no dataset, CPU-only
# ============================================================

class MockCrossAttention(nn.Module):
    """
    Minimal CrossAttention stand-in. MUST be named exactly 'MockCrossAttention'
    -- apply_rotated_quantization_to_vda's surgery matches on
    module.__class__.__name__ in ('CrossAttention', 'MockCrossAttention',
    'TemporalAttention'); a differently-named class silently fails to be
    replaced (this exact mismatch bit a previous task -- see
    docs/optimization_ledger.md, "Mock class name mismatch"). Same shape as
    tests/test_rotation_ablation.py's MockCrossAttention.
    """

    def __init__(self, dim=64, heads=1):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=True), nn.Dropout(0.0)])

    def forward(self, x):
        return x


class _TinyModelS3(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.layer1 = MockCrossAttention(dim=dim)
        self.layer2 = MockCrossAttention(dim=dim)


def build_synthetic_model(use_rotation: bool, bits: int = 8, dim: int = 64):
    model = _TinyModelS3(dim=dim)
    return apply_rotated_quantization_to_vda(
        model, bits=bits, quantizer="identity", use_qjl=False,
        replace_backbone=False, replace_temporal=True, verbose=False,
        use_rotation=use_rotation,
    )


def synthetic_frames(num_images: int, max_tokens: int, dim: int = 64, seed: int = 0):
    """
    torch.randn 'frames' exercising the SAME hook/surgery machinery real
    mode uses -- deliberately heavy-tailed (small Gaussian bulk + a rare
    large per-token outlier channel) so it's a meaningful, non-vacuous
    input for the rotation comparison, not just a smoke test. Seeded via a
    local Generator so it never touches global torch RNG state.
    """
    gen = torch.Generator().manual_seed(seed)
    tokens = max(1, min(max_tokens, 64))
    clips = []
    for _ in range(num_images):
        x = torch.randn(1, tokens, dim, generator=gen) * 0.3
        outlier_channel = torch.randint(0, dim, (1,), generator=gen).item()
        x[:, :, outlier_channel] += torch.randn(1, tokens, generator=gen) * 15.0
        clips.append(x)
    return clips


def _synthetic_forward(model, x):
    model.layer1(x)
    model.layer2(x)


# ============================================================
# OUTPUT
# ============================================================

def _dump_histograms(tensors_raw: dict, tensors_rotated: dict, output_dir: Path):
    layer_names = sorted({k.split("::")[0] for k in tensors_raw} |
                          {k.split("::")[0] for k in tensors_rotated})
    if not layer_names:
        return
    picks = [layer_names[0]] if len(layer_names) == 1 else [layer_names[0], layer_names[-1]]

    fig, axes = plt.subplots(len(picks), 4, figsize=(16, 4 * len(picks)), squeeze=False)
    panels = [("K", "raw", tensors_raw), ("V", "raw", tensors_raw),
              ("K", "rotated", tensors_rotated), ("V", "rotated", tensors_rotated)]
    for row, layer in enumerate(picks):
        for col, (tensor_type, source, tensors) in enumerate(panels):
            ax = axes[row][col]
            key = f"{layer}::{tensor_type}"
            if key in tensors:
                vals = tensors[key].flatten().numpy()
                ax.hist(vals, bins=80)
                ax.set_yscale("log")
            ax.set_title(f"{layer}\n{tensor_type} ({source})", fontsize=8)
    plt.tight_layout()
    out_png = output_dir / "activation_histograms.png"
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out_png}")


def _print_interpretation(records):
    print("\n  Interpretation (see module docstring for the full key):")
    for r in records:
        raw_k = r.get("raw_kurtosis")
        if raw_k is None:
            continue
        if raw_k < 1.0:
            verdict = "near-Gaussian raw -> rotation has little to fix"
        elif raw_k >= 5.0:
            verdict = "heavy-tailed raw -> rotation SHOULD matter at low bits"
        else:
            verdict = "moderately heavy-tailed"
        print(f"    {r['layer']}::{r['tensor']}  raw_kurtosis={raw_k:.2f}  ({verdict})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=str, default="nyuv2")
    ap.add_argument("--num-images", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=None,
                     help="Cap captured tokens per (layer, tensor) after concatenation.")
    ap.add_argument("--output-dir", type=str, default="outputs/activation_stats")
    ap.add_argument("--synthetic", action="store_true",
                     help="Skip the real model/dataset; exercise the identical hook/surgery "
                          "machinery on a tiny mock model fed torch.randn frames. CPU-only, "
                          "no download, no dataset.")
    ap.add_argument("--seed", type=int, default=0, help="Synthetic-mode frame seed.")
    ap.add_argument("--bits", type=int, default=8,
                     help="Nominal bits passed to the identity-quantizer surgery call "
                          "(irrelevant to captured stats; identity ignores it).")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        print("[synthetic mode] tiny mock model, no download, no dataset")
        frames = synthetic_frames(args.num_images, args.max_tokens or 512, dim=64, seed=args.seed)
        model_rot = build_synthetic_model(use_rotation=True, bits=args.bits)
        model_raw = build_synthetic_model(use_rotation=False, bits=args.bits)
        forward_fn = _synthetic_forward
    else:
        data_dir = REPO_ROOT / "benchmark_data"
        print(f"[real mode] dataset={args.dataset} num_images={args.num_images}")
        frames = real_mode_frames(args.dataset, args.num_images, data_dir)
        model_rot = build_model_with_rotation(use_rotation=True, bits=args.bits)
        model_raw = build_model_with_rotation(use_rotation=False, bits=args.bits)
        forward_fn = lambda m, clip: m(clip)  # noqa: E731

    print("  Pass 1/2: rotation ON (post-RHT activations)...")
    stats_rotated, tensors_rotated = run_capture_pass(model_rot, frames, forward_fn, args.max_tokens)
    print("  Pass 2/2: rotation OFF (raw activations)...")
    stats_raw, tensors_raw = run_capture_pass(model_raw, frames, forward_fn, args.max_tokens)

    if not stats_rotated or not stats_raw:
        raise RuntimeError(
            "No K/V activations were captured -- the model has no k_quantizer/v_quantizer "
            "submodules (surgery replaced nothing). Check replace_backbone/replace_temporal "
            "and the module class-name whitelist in apply_rotated_quantization_to_vda."
        )

    all_keys = sorted(set(stats_rotated) | set(stats_raw))
    records = []
    for key in all_keys:
        layer_name, tensor_type = key.split("::")
        entry = {"layer": layer_name, "tensor": tensor_type}
        if key in stats_raw:
            entry.update({f"raw_{k}": v for k, v in stats_raw[key].items()})
        if key in stats_rotated:
            entry.update({f"rotated_{k}": v for k, v in stats_rotated[key].items()})
        records.append(entry)

    out_json = output_dir / "activation_stats.json"
    with open(out_json, "w") as f:
        json.dump({
            "mode": "synthetic" if args.synthetic else "real",
            "dataset": None if args.synthetic else args.dataset,
            "num_images": args.num_images,
            "records": records,
        }, f, indent=2, sort_keys=True)
    print(f"  Wrote {out_json}")

    if HAS_MATPLOTLIB and records:
        _dump_histograms(tensors_raw, tensors_rotated, output_dir)
    elif not HAS_MATPLOTLIB:
        print("  [Warning] matplotlib not found, skipping histogram PNGs.")

    _print_interpretation(records)


if __name__ == "__main__":
    main()
