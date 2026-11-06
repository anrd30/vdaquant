"""
Run StreamVGGT's OFFICIAL video-depth launcher with a compressed KV cache.

    python scripts/streamvggt_kv_launch.py --kv delta:int3:bin:8 -- \
        --weights ../ckpt/checkpoints.pth --output_dir OUT --eval_dataset sintel --size 518

Everything after `--` is passed to StreamVGGT/src/eval/video_depth/launch.py
unchanged.  Run from StreamVGGT/src (its relative data paths assume that).
Writes OUT/kv_stats.json: per-sequence frames, cache bytes, ideal bits/scalar,
peak GPU memory.
"""
import json
import os
import runpy
import sys
import time

VDAQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    argv = sys.argv[1:]
    split = argv.index('--') if '--' in argv else len(argv)
    own, rest = argv[:split], argv[split + 1:]
    kv = own[own.index('--kv') + 1] if '--kv' in own else 'none'
    svg_src = os.environ.get('STREAMVGGT_SRC', os.path.abspath('.'))
    for p in (svg_src, os.path.join(svg_src, 'eval', 'video_depth'), VDAQ):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    from research.models import streamvggt_cache as kc
    kc.install(kv)

    from streamvggt.models import streamvggt as sv
    stats = []
    orig_inference = sv.StreamVGGT.inference

    def inference(self, frames, *a, **kw):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        kc._LIVE.pin = []        # inference frees its caches on return
        try:
            out = orig_inference(self, frames, *a, **kw)
            torch.cuda.synchronize()
            caches = kc._LIVE.pin
            stats.append({
                'frames': len(frames),
                'seconds': time.time() - t0,
                'cache_bytes': sum(c.nbytes() for c in caches),
                'ideal_bits': sum(c.ideal_bits() for c in caches) / max(1, len(caches)),
                'n_caches': len(caches),
                'peak_mb': torch.cuda.max_memory_allocated() / 2 ** 20,
            })
        finally:
            kc._LIVE.pin = None
        return out
    sv.StreamVGGT.inference = inference

    out_dir = rest[rest.index('--output_dir') + 1] if '--output_dir' in rest else '.'
    sys.argv = [os.path.join(svg_src, 'eval', 'video_depth', 'launch.py')] + rest
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'kv_stats.json'), 'w') as fh:
            json.dump({'kv': kv, 'sequences': stats}, fh, indent=2)


if __name__ == '__main__':
    main()
