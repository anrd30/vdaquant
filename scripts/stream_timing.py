"""
Per-frame latency, throughput and memory breakdown of streaming VDA with a
compressed temporal cache.  Run alone on the GPU (timings are wall clock).

For each encoder and cache config: stream one Sintel scene, discard warm-up
frames, and report median / mean ms per frame (torch.cuda.synchronize around
each call, so the cost of encoding and decoding the cache is included), FPS,
weights MB (allocated after loading), cache MB (bit-packed storage), and peak MB.

usage: python -X utf8 scripts/stream_timing.py --encoders vits,vitb,vitl
           --configs fp32,int2-g16,int4-g16::d8:bin-g16 --out outputs/stream/timing.json
"""
import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'Video-Depth-Anything'))
sys.path.insert(0, str(REPO))
from video_depth_anything.video_depth_stream import VideoDepthAnything  # noqa: E402
from research.models.stream_cache import cache_nbytes, enable_compressed_cache  # noqa: E402

CFGS = {
    'vits': dict(encoder='vits', features=64, out_channels=[48, 96, 192, 384]),
    'vitb': dict(encoder='vitb', features=128, out_channels=[96, 192, 384, 768]),
    'vitl': dict(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]),
}


def parse(cfg):
    """Same grammar as stream_eval.py: mode[:bits][:dK][:residual_mode][:predictor]."""
    p = cfg.split(':')
    return dict(mode=p[0], bits=int(p[1]) if len(p) > 1 and p[1] else 3,
                delta=int(p[2][1:]) if len(p) > 2 and p[2] else 0,
                residual_mode=p[3] if len(p) > 3 and p[3] else None,
                predictor=p[4] if len(p) > 4 else 'prev')


def run(enc, cfg, frames, warmup):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = VideoDepthAnything(**CFGS[enc])
    m.load_state_dict(torch.load(REPO / 'checkpoints' / f'video_depth_anything_{enc}.pth', map_location='cpu'))
    m = m.cuda().eval()
    torch.cuda.synchronize()
    weights_mb = torch.cuda.memory_allocated() / 2 ** 20
    c = parse(cfg)
    if c['mode'] != 'fp32' or c['delta']:
        m = enable_compressed_cache(m, mode=c['mode'], bits=c['bits'], seed=0, delta=c['delta'],
                                    residual_mode=c['residual_mode'],
                                    **({'predictor': c['predictor']} if c['delta'] else {}))
    times = []
    for i, f in enumerate(frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m.infer_video_depth_one(f, input_size=518, device='cuda')
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(1000 * (time.perf_counter() - t0))
    res = dict(encoder=enc, config=cfg, n_timed=len(times),
               ms_median=statistics.median(times), ms_mean=statistics.mean(times),
               ms_p90=sorted(times)[int(0.9 * (len(times) - 1))],
               fps=1000 / statistics.median(times), weights_mb=weights_mb,
               cache_mb=cache_nbytes(m) / 2 ** 20, peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
               gpu=torch.cuda.get_device_name())
    del m
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoders', default='vits,vitb,vitl')
    ap.add_argument('--configs', default='fp32,int2-g16,int4-g16::d8:bin-g16')
    ap.add_argument('--scene', default='alley_2')
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--decode', default='kernel', choices=['kernel', 'batched', 'per-entry'],
                    help='cache decoder: fused Triton kernel, batched PyTorch, or one entry at a time')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    import research.models.stream_cache as sc
    sc.USE_TCC_KERNEL = args.decode == 'kernel'
    sc.BATCHED_SCALAR_DECODE = args.decode in ('kernel', 'batched')
    d = REPO / 'benchmark_data' / 'sintel' / 'training' / 'clean' / args.scene
    frames = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in sorted(d.glob('*.png'))]
    results = []
    for enc in args.encoders.split(','):
        for rep in range(args.repeats):
            for cfg in args.configs.split(','):
                with torch.no_grad():
                    r = run(enc, cfg, frames, args.warmup)
                r['repeat'], r['decode'] = rep, args.decode
                results.append(r)
                print(json.dumps(r), flush=True)
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps({'args': vars(args), 'results': results}, indent=1))


if __name__ == '__main__':
    main()
