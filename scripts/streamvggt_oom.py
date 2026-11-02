"""
How many frames can StreamVGGT stream before running out of GPU memory,
with and without the compressed KV cache?

Feeds real frames (a long sequence, cycled if needed) one at a time through
StreamVGGT.inference-style streaming and records peak memory and per-frame
time until OOM or --max-frames.

    python scripts/streamvggt_oom.py --weights ../ckpt/checkpoints.pth \
        --frames-dir ../data/eval/bonn/rgbd_bonn_dataset/rgbd_bonn_crowd/rgb \
        --kv none,fp16,plain:int2,delta:int3:bin:8 --max-frames 2000 --out oom.json
"""
import argparse
import gc
import glob
import json
import os
import subprocess
import sys
import time

VDAQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_one(args, kv):
    import torch
    from research.models import streamvggt_cache as kc
    kc.install(kv)
    from streamvggt.models.streamvggt import StreamVGGT
    from streamvggt.utils.load_fn import load_and_preprocess_images

    model = StreamVGGT()
    model.load_state_dict(torch.load(args.weights, map_location='cpu'), strict=True)
    model = model.cuda().eval()
    paths = sorted(glob.glob(os.path.join(args.frames_dir, '*.png')) +
                   glob.glob(os.path.join(args.frames_dir, '*.jpg')))
    past = [None] * model.aggregator.depth
    log, n, oom = [], 0, False
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        while n < args.max_frames:
            img = load_and_preprocess_images([paths[n % len(paths)]]).cuda()
            t0 = time.time()
            try:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.bf16):
                    _, _, past = model.aggregator(img[None], past_key_values=past,
                                                  use_cache=True, past_frame_idx=n)
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                oom = True
                break
            n += 1
            if n % args.log_every == 0:
                log.append({'frames': n, 'peak_mb': torch.cuda.max_memory_allocated() / 2 ** 20,
                            'cache_mb': kc.total_cache_bytes() / 2 ** 20,
                            'ms_per_frame': 1000 * (time.time() - t0)})
                print(f"  [{kv}] {log[-1]}", flush=True)
    return {'kv': kv, 'frames_reached': n, 'oom': oom, 'log': log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--frames-dir', required=True)
    ap.add_argument('--kv', default='none,fp16,plain:int2,delta:int3:bin:8')
    ap.add_argument('--max-frames', type=int, default=2000)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--single', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--out', default='oom.json')
    args = ap.parse_args()
    svg_src = os.environ.get('STREAMVGGT_SRC', os.path.abspath('.'))
    for p in (svg_src, VDAQ):
        sys.path.insert(0, p)

    if args.single:                       # child process: one config
        print(json.dumps(run_one(args, args.single)), flush=True)
        return
    results = []
    for kv in args.kv.split(','):         # fresh process per config: clean GPU state
        cmd = [sys.executable, os.path.abspath(__file__), '--weights', args.weights,
               '--frames-dir', args.frames_dir, '--max-frames', str(args.max_frames),
               '--log-every', str(args.log_every), '--single', kv] + (['--bf16'] if args.bf16 else [])
        p = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
        last = [l for l in p.stdout.splitlines() if l.startswith('{')]
        res = json.loads(last[-1]) if last else {'kv': kv, 'error': p.stderr[-2000:]}
        results.append(res)
        print(kv, '->', res.get('frames_reached'), 'frames', 'OOM' if res.get('oom') else '', flush=True)
        with open(args.out, 'w') as fh:
            json.dump(results, fh, indent=2)
        gc.collect()


if __name__ == '__main__':
    main()
