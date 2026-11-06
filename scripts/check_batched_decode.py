"""Batched scalar decode must give bit-identical depth to the per-entry decoder.
Streams one Sintel scene with BATCHED_SCALAR_DECODE off and on, compares every
frame's depth, and reports ms/frame for both.
usage: python -X utf8 scripts/check_batched_decode.py [--frames 50]"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'Video-Depth-Anything'))
sys.path.insert(0, str(REPO))
import research.models.stream_cache as sc  # noqa: E402
from video_depth_anything.video_depth_stream import VideoDepthAnything  # noqa: E402
from stream_timing import CFGS, parse  # noqa: E402


def stream(cfg, frames, batched, enc='vits', kernel=False):
    sc.BATCHED_SCALAR_DECODE = batched
    sc.USE_TCC_KERNEL = kernel
    m = VideoDepthAnything(**CFGS[enc])
    m.load_state_dict(torch.load(REPO / 'checkpoints' / f'video_depth_anything_{enc}.pth', map_location='cpu'))
    m = m.cuda().eval()
    c = parse(cfg)
    m = sc.enable_compressed_cache(m, mode=c['mode'], bits=c['bits'], seed=0, delta=c['delta'],
                                   residual_mode=c['residual_mode'],
                                   **({'predictor': c['predictor']} if c['delta'] else {}))
    out, ts = [], []
    with torch.no_grad():
        for f in frames:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out.append(m.infer_video_depth_one(f, input_size=518, device='cuda'))
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    return out, 1000 * float(np.median(ts[5:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=50)
    ap.add_argument('--scene', default='alley_2')
    ap.add_argument('--configs', default='int4-g16::d8:bin-g16,int2-g16,int2-g64-asym,'
                                         'int4-g16::d8:bin-g16:none,int4-g16::d4:bin-g16,int4-g16::d8:bin-g32-s8')
    args = ap.parse_args()
    d = REPO / 'benchmark_data' / 'sintel' / 'training' / 'clean' / args.scene
    frames = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in sorted(d.glob('*.png'))][:args.frames]
    ok = True
    for cfg in args.configs.split(','):
        a, ta = stream(cfg, frames, False)
        b, tb = stream(cfg, frames, True)
        k, tk = stream(cfg, frames, True, kernel=True)
        diff = max(float(np.abs(x - y).max()) for x, y in zip(a, b))
        dk = max(float(np.abs(x - y).max()) for x, y in zip(a, k))
        ok &= diff == 0.0 and dk == 0.0
        print(f'{cfg:30s} max|diff| batched {diff:.1e} kernel {dk:.1e} | ms/frame per-entry {ta:6.1f}'
              f' batched {tb:6.1f} kernel {tk:6.1f}', flush=True)
    print('BIT-IDENTICAL' if ok else 'MISMATCH')


if __name__ == '__main__':
    main()
