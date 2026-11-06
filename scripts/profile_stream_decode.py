"""Where does a TCC streaming step spend its time?  Times cache decode
(DeltaLazyLayer.materialize / read_layer), cache encode, and the whole step.
usage: python -X utf8 scripts/profile_stream_decode.py [--config int4-g16::d8:bin-g16]"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'Video-Depth-Anything'))
sys.path.insert(0, str(REPO))
import research.models.stream_cache as sc  # noqa: E402
from video_depth_anything.video_depth_stream import VideoDepthAnything  # noqa: E402
from stream_timing import CFGS, parse  # noqa: E402

T = defaultdict(float)


def timed(name, fn):
    def wrap(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        T[name] += time.perf_counter() - t0
        return out
    return wrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='int4-g16::d8:bin-g16')
    ap.add_argument('--encoder', default='vits')
    ap.add_argument('--frames', type=int, default=40)
    args = ap.parse_args()
    sc.DeltaLazyLayer.materialize = timed('decode (delta materialize)', sc.DeltaLazyLayer.materialize)
    sc.LazyLayer.materialize = timed('decode (lazy materialize)', sc.LazyLayer.materialize)
    sc._DeltaEncoder.encode = timed('encode (delta)', sc._DeltaEncoder.encode)
    sc.tcc_decode = timed('  of which tcc_decode (stack + kernel)', sc.tcc_decode)
    sc._read_scalar_batched = timed('  of which batched PyTorch decode', sc._read_scalar_batched)
    d = REPO / 'benchmark_data' / 'sintel' / 'training' / 'clean' / 'alley_2'
    frames = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in sorted(d.glob('*.png'))][:args.frames]
    m = VideoDepthAnything(**CFGS[args.encoder])
    m.load_state_dict(torch.load(REPO / 'checkpoints' / f'video_depth_anything_{args.encoder}.pth', map_location='cpu'))
    m = m.cuda().eval()
    c = parse(args.config)
    if c['mode'] != 'fp32' or c['delta']:
        m = sc.enable_compressed_cache(m, mode=c['mode'], bits=c['bits'], seed=0, delta=c['delta'],
                                       residual_mode=c['residual_mode'],
                                       **({'predictor': c['predictor']} if c['delta'] else {}))
    step = timed('whole step', m.infer_video_depth_one)
    with torch.no_grad():
        for i, f in enumerate(frames):
            if i == 5:
                T.clear()
            step(f, input_size=518, device='cuda')
            if i == 5:
                st0 = torch.cuda.memory_stats()
    n = args.frames - 5
    for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
        print(f'{k:40s} {1000 * v / n:8.1f} ms/frame')
    st = torch.cuda.memory_stats()
    for k in ('num_alloc_retries', 'num_device_alloc', 'num_device_free', 'segment.all.allocated'):
        if k in st:
            print(f'{k:40s} {st[k] - st0.get(k, 0)} during timed frames')
    print(f"reserved {torch.cuda.memory_reserved() / 2**20:.0f} MB, peak allocated "
          f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MB")


if __name__ == '__main__':
    main()
