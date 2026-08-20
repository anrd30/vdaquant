#!/usr/bin/env python
"""
Blur-degradation study: is the TAE failure mode about QUANTISATION, or about
LOSS OF SPATIAL STRUCTURE in general?

Motivation. We showed (ledger F16/F25) that a collapsed 2-bit quantised model
attains a far BETTER unmasked TAE than FP32, and that co-visibility masking
reverses that ranking. The proposed mechanism is that TAE averages over
reprojected pixels with no valid correspondence, and those terms are large
exactly when the prediction is detailed -- so ANY degradation that removes
fine spatial detail should lower unmasked TAE. Quantisation is then just one
instance of a general law.

This script tests that directly, with a degradation that has nothing to do
with quantisation: Gaussian blur applied to the FP32 model's own predictions.
Prediction:
    * unmasked TAE should FALL as sigma rises (metric "improves" as the
      prediction is destroyed),
    * accuracy (delta1) should FALL too,
    * co-visibility-masked TAE should RISE (correctly penalising the
      degradation).
If all three hold, the failure mode is structure loss in general, not a
quirk of quantisation, and the paper's claim generalises.

Efficiency. The FP32 forward pass over the dataset is run ONCE and cached in
memory; each sigma is then a cheap CPU blur + re-evaluation. Cost is
therefore one FP32 temporal pass plus a few seconds per sigma.

Everything downstream -- windowing, pooled disparity alignment, the z-buffer,
the geometric TAE, the co-visibility mask -- is the SAME code path used for
every other result in the paper (imported, not reimplemented), so the numbers
are directly comparable to the quantisation sweep.

Usage:
    python scripts/run_blur_degradation.py --dataset sintel \\
        --sigmas 0 0.5 1 2 4 8 --temporal-window 32 --max-scenes 23 \\
        --tae-covis-tau 0.05 --output-dir outputs/finals/blur_degradation
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

from run_pareto_benchmark_suite import (  # noqa: E402
    _predict_window,
    compute_tae_geometric_for_scene,
    get_model_config,
    checkpoint_candidates,
)
from datasets_gt import (  # noqa: E402
    load_gt_dataset,
    group_samples_by_scene,
    chunk_scene_into_windows,
    compute_gt_depth_metrics,
)


# ============================================================
# BLUR — the degradation instrument
# ============================================================

def gaussian_blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Separable Gaussian blur of a single (H, W) map. sigma <= 0 returns the
    input unchanged (so sigma=0 is an exact no-op control row, not an
    approximate one). Kernel is truncated at 3 sigma and forced odd, and
    edges use replicate padding so the border does not darken -- a
    zero-padded blur would introduce an artificial depth discontinuity at
    the frame edge and contaminate exactly the reprojection terms we are
    trying to study.
    """
    if sigma is None or sigma <= 0:
        return x
    radius = max(1, int(round(3.0 * sigma)))
    ksize = 2 * radius + 1
    coords = torch.arange(ksize, dtype=torch.float32) - radius
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = (g / g.sum()).to(dtype=x.dtype)

    inp = x[None, None]
    inp = F.pad(inp, (radius, radius, 0, 0), mode="replicate")
    inp = F.conv2d(inp, g.view(1, 1, 1, ksize))
    inp = F.pad(inp, (0, 0, radius, radius), mode="replicate")
    inp = F.conv2d(inp, g.view(1, 1, ksize, 1))
    return inp[0, 0]


# ============================================================
# MODEL
# ============================================================

def load_fp32_model(encoder: str):
    import video_depth_anything.dinov2_layers.attention as dino_attn
    import video_depth_anything.motion_module.attention as motion_attn
    if not torch.cuda.is_available():
        dino_attn.memory_efficient_attention = lambda q, k, v, attn_bias=None, **kw: \
            torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        motion_attn.XFORMERS_AVAILABLE = False
    from video_depth_anything.video_depth import VideoDepthAnything

    cfg = get_model_config(encoder)
    model = VideoDepthAnything(**cfg).eval()
    for ck in checkpoint_candidates(encoder):
        if ck.exists() and ck.stat().st_size >= 10_000_000:
            model.load_state_dict(torch.load(ck, map_location="cpu"))
            print(f"  [Model] loaded {ck}")
            if torch.cuda.is_available():
                return model.cuda()
            for m in model.modules():
                if hasattr(m, "_use_memory_efficient_attention_xformers"):
                    m._use_memory_efficient_attention_xformers = False
            return model
    raise RuntimeError(
        f"No {encoder} checkpoint found. Run scripts/run_pareto_benchmark_suite.py "
        f"once first (it auto-downloads), or place the .pth in checkpoints/."
    )


# ============================================================
# EVAL
# ============================================================

def evaluate(scene_preds, gt_range, covis_tau, pred_space="disparity"):
    """Same aggregation contract as run_temporal_eval: mean accuracy across
    frames, mean/median TAE across scenes, plus co-visibility columns."""
    acc, tae_list, cov_list, frac_list = [], [], [], []
    n_skipped_frames = n_skipped_scenes = 0
    per_scene = {}

    for scene, (preds, gts, Ks, poses) in scene_preds.items():
        for pred, gt in zip(preds, gts):
            gt_t = torch.from_numpy(gt).float()
            valid = torch.from_numpy((gt > 1e-3) & (gt < gt_range[1]))
            try:
                acc.append(compute_gt_depth_metrics(
                    pred, gt_t, valid, gt_range=gt_range,
                    pred_is_disparity=(pred_space == "disparity")))
            except ValueError:
                n_skipped_frames += 1
        try:
            r = compute_tae_geometric_for_scene(
                preds, gts, Ks, poses, gt_range, covis_tau=covis_tau)
            tae_list.append(r["tae_percent"])
            per_scene[scene] = r["tae_percent"]
            if covis_tau is not None:
                cov_list.append(r["tae_covis_percent"])
                frac_list.append(r["covis_fraction"])
        except ValueError:
            n_skipped_scenes += 1

    if not acc:
        raise ValueError("Every frame skipped -- check gt_range.")

    out = {k: round(float(np.mean([m[k] for m in acc])), 6)
           for k in ("abs_rel", "rmse", "delta1", "delta2", "delta3")}
    out["tae_percent"] = round(float(np.mean(tae_list)), 6) if tae_list else 0.0
    out["tae_median_percent"] = round(float(np.median(tae_list)), 6) if tae_list else 0.0
    if covis_tau is not None:
        out["tae_covis_percent"] = round(float(np.mean(cov_list)), 6) if cov_list else 0.0
        out["covis_fraction"] = round(float(np.mean(frac_list)), 6) if frac_list else 0.0
    out["n_images"] = len(acc)
    out["n_skipped_frames"] = n_skipped_frames
    out["n_skipped_scenes_tae"] = n_skipped_scenes
    out["per_scene_tae"] = {k: round(v, 3) for k, v in per_scene.items()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="sintel")
    ap.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 0.5, 1, 2, 4, 8])
    ap.add_argument("--temporal-window", type=int, default=32)
    ap.add_argument("--max-scenes", type=int, default=23)
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--tae-covis-tau", type=float, default=0.05)
    ap.add_argument("--pred-space", default="disparity", choices=["disparity", "metric"])
    ap.add_argument("--output-dir", default="outputs/finals/blur_degradation")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = REPO_ROOT / "benchmark_data"

    samples, gt_range = load_gt_dataset(args.dataset, data_dir,
                                        max_samples=args.max_samples, require_cam=True)
    scenes = group_samples_by_scene(samples)
    scene_names = list(scenes.keys())[:args.max_scenes]
    print(f"  [Dataset] {len(scene_names)} scene(s), gt_range={gt_range}")

    print("  [1/2] FP32 forward pass (run ONCE, cached for every sigma)...")
    model = load_fp32_model(args.encoder)
    base = {}
    for scene in scene_names:
        frames = scenes[scene]
        preds, gts, Ks, poses = [], [], [], []
        for window_frames, n_real in chunk_scene_into_windows(frames, args.temporal_window):
            wp = _predict_window(model, [f["rgb"] for f in window_frames])
            for i in range(n_real):
                preds.append(wp[i])
                gts.append(window_frames[i]["depth"])
                Ks.append(window_frames[i]["K"])
                poses.append(window_frames[i]["pose"])
        base[scene] = (preds, gts, Ks, poses)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("  [2/2] Blur sweep (CPU)...")
    results = {}
    for sigma in args.sigmas:
        blurred = {s: ([gaussian_blur2d(p, sigma) for p in preds], gts, Ks, poses)
                   for s, (preds, gts, Ks, poses) in base.items()}
        m = evaluate(blurred, gt_range, args.tae_covis_tau, args.pred_space)
        m["blur_sigma"] = sigma
        results[f"sigma_{sigma:g}"] = m
        print(f"    sigma={sigma:<5g} delta1={m['delta1']:.4f}  "
              f"TAE={m['tae_percent']:.3f}  covTAE={m.get('tae_covis_percent', float('nan')):.3f}")

    with open(out_dir / "blur_degradation_results.json", "w") as f:
        json.dump({"dataset": args.dataset, "encoder": args.encoder,
                   "temporal_window": args.temporal_window,
                   "tae_covis_tau": args.tae_covis_tau,
                   "results": results,
                   "note": (
                       "Gaussian blur applied to the FP32 model's own predictions -- a "
                       "degradation with no connection to quantisation. If unmasked TAE "
                       "falls while delta1 falls and co-visibility-masked TAE rises, the "
                       "TAE failure mode is loss of spatial structure in general, not a "
                       "quantisation artefact. FP32 forward pass is shared across all "
                       "sigmas; only the blur differs.")},
                  f, indent=2)

    lines = [f"# Blur degradation ({args.dataset}, {args.encoder})", "",
             "| sigma | delta1 | AbsRel | TAE % | cov-TAE % |", "|---|---|---|---|---|"]
    for k, m in results.items():
        lines.append("| %g | %.4f | %.4f | %.3f | %.3f |" % (
            m["blur_sigma"], m["delta1"], m["abs_rel"],
            m["tae_percent"], m.get("tae_covis_percent", float("nan"))))
    text = "\n".join(lines) + "\n"
    (out_dir / "blur_degradation.md").write_text(text)
    print()
    print(text)
    print(f"  Wrote {out_dir/'blur_degradation_results.json'}")


if __name__ == "__main__":
    main()
