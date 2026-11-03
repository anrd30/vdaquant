"""
Streaming-mode memory/fidelity benchmark for the compressed temporal cache.

Runs VDA's streaming inference (infer_video_depth_one) frame by frame with the
cache stored as fp32 (reference), fp16, or Λ16-packed, and reports:
weights, cache bytes, peak GPU memory per step, step time, and the per-frame
depth deviation from the fp32-cache run (relative L1).

    python scripts/stream_memory.py --encoder vits --frames 60 --size 518
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'Video-Depth-Anything'))

from video_depth_anything.video_depth_stream import VideoDepthAnything  # noqa: E402
from research.models.stream_cache import cache_nbytes, enable_compressed_cache  # noqa: E402

CFGS = {
    'vits': dict(encoder='vits', features=64, out_channels=[48, 96, 192, 384]),
    'vitb': dict(encoder='vitb', features=128, out_channels=[96, 192, 384, 768]),
    'vitl': dict(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]),
}
MB = 1024 ** 2


def make_frames(n, h, w, video=None):
    if video:
        import cv2
        cap, frames = cv2.VideoCapture(video), []
        while len(frames) < n:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        return frames
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w * 2].astype(np.float32)
    base = np.stack([np.sin(xx / 37) + np.cos(yy / 23), np.sin((xx + yy) / 51),
                     np.cos(xx / 17) * np.sin(yy / 29)], -1)
    base = ((base - base.min()) / np.ptp(base) * 200 + rng.random(base.shape) * 55).astype(np.uint8)
    return [np.ascontiguousarray(base[:, 4 * i:4 * i + w]) for i in range(n)]


def run(args, mode, bits, frames, kv=False, delta=0, residual_mode=None):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = VideoDepthAnything(**CFGS[args.encoder])
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'), strict=True)
    model = model.cuda().eval()
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated()
    model = enable_compressed_cache(model, mode=mode, bits=bits, seed=args.seed, kv=kv, delta=delta,
                                    residual_mode=residual_mode)

    depths, peaks, times = [], [], []
    for f in frames:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        d = model.infer_video_depth_one(f, input_size=args.size, device='cuda')
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        peaks.append(torch.cuda.max_memory_allocated())
        depths.append(d)
    res = {
        'mode': mode + ('-kv' if kv else ''), 'bits': bits if mode == 'bw16' else None,
        'weights_mb': weights / MB,
        'cache_mb': cache_nbytes(model) / MB,
        'cache_entries': len(model.frame_cache_list),
        'peak_mb': max(peaks[args.warmup:]) / MB,
        'steady_alloc_mb': torch.cuda.memory_allocated() / MB,
        'step_ms_median': 1000 * float(np.median(times[args.warmup:])),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res, depths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', default='vits', choices=list(CFGS))
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--frames', type=int, default=60)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--size', type=int, default=518)
    ap.add_argument('--height', type=int, default=518)
    ap.add_argument('--width', type=int, default=518)
    ap.add_argument('--video', default=None, help='optional real video file')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--configs', default='fp32,fp16,bw16:4,bw16:3')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    args.ckpt = args.ckpt or os.path.join(REPO, 'checkpoints', f'video_depth_anything_{args.encoder}.pth')

    frames = make_frames(args.frames, args.height, args.width, args.video)
    results, ref = [], None
    for cfg in args.configs.split(','):
        parts = cfg.split(':')
        mode, kv = parts[0], parts[0].endswith('-kv')
        mode = mode[:-3] if kv else mode
        bits = int(parts[1]) if len(parts) > 1 and parts[1] else 3
        dk = int(parts[2][1:]) if len(parts) > 2 and parts[2] else 0
        rmode = parts[3] if len(parts) > 3 else None
        res, depths = run(args, mode, bits, frames, kv=kv, delta=dk, residual_mode=rmode)
        res['config'] = cfg
        if ref is None:
            ref = depths
        rel = [float(np.abs(d - r).mean() / (np.abs(r).mean() + 1e-8)) for d, r in zip(depths, ref)]
        res['depth_rel_l1_mean'] = float(np.mean(rel))
        res['depth_rel_l1_max'] = float(np.max(rel))
        results.append(res)
        print(json.dumps(res), flush=True)

    base = results[0]
    print(f"\nencoder={args.encoder} frames={len(frames)} input={args.height}x{args.width} (size {args.size})")
    print(f"{'config':14s} {'cache MB':>9s} {'x smaller':>9s} {'peak MB':>8s} {'peak cut':>8s} {'ms/frame':>9s} {'depth relL1':>11s}")
    for r in results:
        name = r.get('config', r['mode'])
        print(f"{name:14s} {r['cache_mb']:9.1f} {base['cache_mb'] / r['cache_mb']:9.2f} {r['peak_mb']:8.1f} "
              f"{1 - r['peak_mb'] / base['peak_mb']:8.1%} {r['step_ms_median']:9.1f} {r['depth_rel_l1_mean']:11.4f}")
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump({'args': vars(args), 'results': results}, fh, indent=2)


if __name__ == '__main__':
    main()
