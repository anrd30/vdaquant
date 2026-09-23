"""
End-to-end wall-clock benchmark for Paper 2 §5.9.

Measures REAL end-to-end FPS and peak GPU memory for VDA-S under four
configurations, on synthetic video of user-specified resolution and
length:

    - fp32_baseline           full-precision reference
    - fp16_autocast           the default deployment baseline
    - bw16_3b_simulator       quantise->fp16-reconstruct->SDPA (accuracy path)
    - bw16_3b_fused           packed uint8 KV + fused Triton attention (kernel path)

The two bw16 rows use the SAME accuracy-tested quantiser, so any FPS
delta between them is pure kernel-integration overhead / speed-up. That
is the number Paper 2 §5.9 uses.

Also runs a MAX-SEQUENCE-LENGTH sweep: for each config, find the
largest video length that fits before CUDA OOM at a fixed resolution.
Produces the "enables regime FP16 can't reach" curve.

Usage:
  # Standard 480p + 720p run at video lengths 32 and 128
  python scripts/bench_wall_clock.py --encoder vits

  # Just an OOM sweep at 480p
  python scripts/bench_wall_clock.py --encoder vits --oom-only --resolution 476

Output: outputs/bench_wall_clock/<run-tag>/results.json + printed table.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# Same VDA lookup as run_pareto_benchmark_suite.py so this works
# regardless of whether Video-Depth-Anything sits inside vdaquant/
# or one level up alongside it.
for _p in [
    REPO_ROOT / "Video-Depth-Anything",
    REPO_ROOT.parent / "Video-Depth-Anything",
    Path("/content/Video-Depth-Anything"),
    Path("/content/vdaquant/Video-Depth-Anything"),
]:
    if _p.exists() and (_p / "video_depth_anything").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break


def _synthetic_video(n_frames: int, h: int, w: int) -> np.ndarray:
    """Reproducible random-noise clip in HxWx3 uint8 (FPS is content-independent)."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(n_frames, h, w, 3), dtype=np.uint8)


def _load_vda(encoder: str, ckpt_path: Path, device: str = "cuda"):
    from video_depth_anything.video_depth import VideoDepthAnything
    model_cfg = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }[encoder]
    model = VideoDepthAnything(**model_cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)
    model = model.to(device).eval()
    return model


def _apply_bw16(model, bits: int, use_fused_kernel: bool):
    from research.models import apply_rotated_quantization_to_vda
    return apply_rotated_quantization_to_vda(
        model, bits=bits, quantizer='lattice_bw16', use_qjl=(not use_fused_kernel),
        replace_backbone=False, replace_temporal=True, verbose=True,
        use_fused_kernel=use_fused_kernel,
    )


def _time_infer(model, frames: np.ndarray, input_size: int, fp32: bool,
                warmup: int = 1, trials: int = 3) -> dict:
    """Wall-clock a single infer_video_depth call, averaged over `trials`."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    # Warmup
    for _ in range(warmup):
        _ = model.infer_video_depth(frames, target_fps=30, input_size=input_size, fp32=fp32)
        torch.cuda.synchronize()
    latencies = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.infer_video_depth(frames, target_fps=30, input_size=input_size, fp32=fp32)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    lat_mean = float(np.mean(latencies))
    lat_std = float(np.std(latencies))
    fps = len(frames) / lat_mean
    return {
        "latency_s_mean": lat_mean,
        "latency_s_std": lat_std,
        "fps": fps,
        "peak_mem_mb": peak_mem_mb,
        "n_frames": int(len(frames)),
    }


def _run_config(encoder: str, ckpt: Path, config: str, n_frames: int,
                resolution: int, device: str, warmup: int = 1,
                trials: int = 3) -> dict:
    """Load a fresh model in the requested config and time one inference run."""
    model = _load_vda(encoder, ckpt, device)
    fp32 = (config == "fp32_baseline")
    if config in ("bw16_3b_simulator", "bw16_3b_fused"):
        model = _apply_bw16(model, bits=3, use_fused_kernel=(config == "bw16_3b_fused"))
    # Video content is irrelevant for wall-clock; height ~ resolution (rounded to /14).
    h = resolution
    w = int(resolution * 16 / 9)   # 16:9 aspect
    frames = _synthetic_video(n_frames, h, w)
    metrics = _time_infer(model, frames, input_size=resolution, fp32=fp32,
                          warmup=warmup, trials=trials)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def _find_max_frames(encoder: str, ckpt: Path, config: str, resolution: int,
                     device: str, start: int = 32, cap: int = 4096) -> int:
    """Binary-search the largest n_frames that fits before OOM."""
    def try_run(n: int) -> bool:
        try:
            _run_config(encoder, ckpt, config, n_frames=n,
                        resolution=resolution, device=device)
            return True
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return False
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return False
            raise
    # Exponential grow, then binary
    lo, hi = 0, 0
    n = start
    while n <= cap and try_run(n):
        lo = n
        n *= 2
    hi = min(n, cap)
    while hi - lo > 8:
        mid = (lo + hi) // 2
        if try_run(mid):
            lo = mid
        else:
            hi = mid
    return lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "checkpoints" / "video_depth_anything_vits.pth"))
    ap.add_argument("--resolutions", nargs="+", type=int, default=[476, 714],
                    help="Input sizes in pixels (must be multiple of 14).")
    ap.add_argument("--video-lengths", nargs="+", type=int, default=[32, 128])
    ap.add_argument("--configs", nargs="+", default=[
        "fp32_baseline", "fp16_autocast", "bw16_3b_simulator", "bw16_3b_fused",
    ])
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "bench_wall_clock"))
    ap.add_argument("--oom-only", action="store_true",
                    help="Skip the FPS grid; run the OOM sweep only.")
    ap.add_argument("--oom-resolution", type=int, default=476,
                    help="Resolution to use for the OOM sweep.")
    ap.add_argument("--trials", type=int, default=5,
                    help="Timed trials averaged for FPS. Bumped from 3 -> 5 "
                         "to reduce variance (previous run showed 20% CV).")
    ap.add_argument("--warmup", type=int, default=3,
                    help="Untimed warmup runs before trials. Bumped from 1 -> "
                         "3 so all CUDA graph captures (one per unique "
                         "attention shape) complete before timing starts.")
    args = ap.parse_args()

    device = "cuda"
    ckpt = Path(args.ckpt)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"encoder": args.encoder, "fps_grid": [], "oom_sweep": []}

    if not args.oom_only:
        print(f"\n=== FPS + peak-memory grid ===")
        print(f"{'config':<22} {'res':>4} {'frames':>7} {'FPS':>8} {'lat_ms':>10} "
              f"{'peakMB':>10}")
        for config in args.configs:
            for res in args.resolutions:
                for n in args.video_lengths:
                    try:
                        m = _run_config(args.encoder, ckpt, config, n_frames=n,
                                        resolution=res, device=device,
                                        warmup=args.warmup, trials=args.trials)
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        if "out of memory" in str(e).lower():
                            torch.cuda.empty_cache()
                            print(f"{config:<22} {res:>4} {n:>7} {'OOM':>8}")
                            results["fps_grid"].append({
                                "config": config, "resolution": res, "n_frames": n,
                                "oom": True,
                            })
                            continue
                        raise
                    row = {"config": config, "resolution": res, "n_frames": n, **m}
                    results["fps_grid"].append(row)
                    print(f"{config:<22} {res:>4} {n:>7} "
                          f"{m['fps']:>8.2f} {m['latency_s_mean']*1000:>10.1f} "
                          f"{m['peak_mem_mb']:>10.0f}")

    print(f"\n=== OOM sweep @ {args.oom_resolution}px ===")
    for config in args.configs:
        max_n = _find_max_frames(args.encoder, ckpt, config,
                                  resolution=args.oom_resolution, device=device)
        results["oom_sweep"].append({
            "config": config, "resolution": args.oom_resolution,
            "max_frames": max_n,
        })
        print(f"{config:<22} max frames fitted: {max_n}")

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
