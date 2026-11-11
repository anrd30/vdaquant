"""
Time one cache read (all past frames of one layer, K and V) with the PyTorch decoder
and with the fused Triton decoder, at StreamVGGT's real frame size.

    python scripts/bench_svg_decode.py --frames 100 200 --spec delta:int4:bin:8
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.models import streamvggt_cache as kc  # noqa: E402
from kernels.triton_kernels import decode_svg  # noqa: E402


def build(spec, n, shape):
    torch.manual_seed(0)
    cache = kc.CompressedKV(kc.parse(spec))
    x = torch.randn(shape, device='cuda')
    kc.FUSED = True
    for _ in range(n):
        x = x + 0.2 * torch.randn(shape, device='cuda')
        f = x.to(torch.bfloat16)
        prev = cache.read(f, f)[2] if cache.frames else None
        cache.append(f, f, prev)
    return cache, f


def timed(cache, cur, fused, reps):
    kc.FUSED = fused
    cache.read(cur, cur)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        cache.read(cur, cur)
    torch.cuda.synchronize()
    return 1000 * (time.time() - t) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, nargs='+', default=[50, 100])
    ap.add_argument('--spec', nargs='+', default=['delta:int4:bin:8', 'delta:int4:a2g64:8', 'plain:a2g64'])
    ap.add_argument('--tokens', type=int, default=1041)
    ap.add_argument('--reps', type=int, default=5)
    args = ap.parse_args()
    shape = (1, 16, 1, args.tokens, 64)
    calls = []
    real = decode_svg.svg_decode

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)
    decode_svg.svg_decode = counting
    for spec in args.spec:
        for n in args.frames:
            cache, cur = build(spec, n, shape)
            calls.clear()
            tf = timed(cache, cur, True, args.reps)
            used = len(calls) > 0
            tp = timed(cache, cur, False, args.reps)
            print(f'{spec:22s} frames {n:4d}  pytorch {tp:8.2f} ms  fused {tf:8.2f} ms  '
                  f'speedup {tp / tf:5.2f}x  fused path used: {used}', flush=True)
            del cache
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
