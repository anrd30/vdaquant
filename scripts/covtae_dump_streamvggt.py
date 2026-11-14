"""
Save StreamVGGT depth predictions in the format of run_pareto_benchmark_suite.py --dump-preds, so
scripts/covtae_analysis.py can rank StreamVGGT against VDA under every temporal metric.

Frames come from the same loader (and the same --max-samples) as the VDA runs; each scene is streamed
from its first frame with a fresh cache, at most --max-per-scene frames (StreamVGGT's cache grows with
every frame). Each frame goes through StreamVGGT's own preprocessing (load_and_preprocess_images); the
predicted depth is resized to the ground-truth resolution and stored as disparity (1 / depth).

Files: <dumps>/<dataset>_<group>_wS_streamvggt-<kv>.{npz,json}; <group> (default vits) is the
analysis group the predictions join, so one analysis file holds every model.

    STREAMVGGT_SRC=.../StreamVGGT/src python scripts/covtae_dump_streamvggt.py --dataset bonn \
        --weights .../ckpt/checkpoints.pth --max-samples 6000 --max-per-scene 110 --dumps covtae/dumps
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO))
from datasets_gt import group_samples_by_scene, load_gt_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['sintel', 'bonn'])
    ap.add_argument('--weights', required=True)
    ap.add_argument('--kv', default='none', help="StreamVGGT cache spec (tcc streamvggt_cache), 'none' = 16-bit")
    ap.add_argument('--max-samples', type=int, default=2000)
    ap.add_argument('--max-per-scene', type=int, default=None)
    ap.add_argument('--group', default='vits')
    ap.add_argument('--dumps', required=True)
    ap.add_argument('--bf16', action='store_true')
    args = ap.parse_args()
    sys.path.insert(0, os.environ.get('STREAMVGGT_SRC', os.path.abspath('.')))

    from research.models import streamvggt_cache as kc
    kc.install(args.kv)
    from streamvggt.models.streamvggt import StreamVGGT
    from streamvggt.utils.load_fn import load_and_preprocess_images
    model = StreamVGGT()
    model.load_state_dict(torch.load(args.weights, map_location='cpu'), strict=True)
    model = model.cuda().eval()

    samples, _ = load_gt_dataset(args.dataset, REPO / 'benchmark_data', max_samples=args.max_samples, require_cam=True)
    scenes = group_samples_by_scene(samples)
    arrays, meta = {}, {}
    tmp = tempfile.mkdtemp()
    for scene, frames in scenes.items():
        frames = frames[:args.max_per_scene] if args.max_per_scene else frames
        past = [None] * model.aggregator.depth
        past_cam = [None] * model.camera_head.trunk_depth
        kc._LIVE.pin = []
        with torch.no_grad():
            for i, f in enumerate(frames):
                path = os.path.join(tmp, 'frame.png')
                Image.fromarray(f['rgb']).save(path)
                images = load_and_preprocess_images([path]).cuda()[None]
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.bf16):
                    tokens, start, past = model.aggregator(images, past_key_values=past, use_cache=True,
                                                           past_frame_idx=i)
                    with torch.cuda.amp.autocast(enabled=False):
                        _, past_cam = model.camera_head(tokens, past_key_values_camera=past_cam, use_cache=True)
                        depth, _ = model.depth_head(tokens, images=images, patch_start_idx=start)
                d = depth.float()[0, 0, ..., 0]
                H, W = f['depth'].shape
                d = F.interpolate(d[None, None], size=(H, W), mode='bilinear', align_corners=False)[0, 0]
                arrays[f'{scene}|{i:05d}'] = (1.0 / d.clamp(min=1e-6)).cpu().numpy().astype(np.float16)
        meta[scene] = len(frames)
        del past, past_cam
        torch.cuda.empty_cache()
        print(f'  {scene}: {len(frames)} frames, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB', flush=True)

    tag = 'streamvggt-' + args.kv.replace(':', '_')
    stem = f'{args.dataset}_{args.group}_wS_{tag}'
    out = Path(args.dumps)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f'{stem}.npz', **arrays)
    (out / f'{stem}.json').write_text(json.dumps({
        'scenes': meta, 'window': 'S', 'tag': tag, 'dataset': args.dataset, 'encoder': args.group,
        'model': 'StreamVGGT', 'kv': args.kv, 'max_samples': args.max_samples,
        'max_per_scene': args.max_per_scene, 'bf16': args.bf16}))
    print(f'[dump] {stem}: {sum(meta.values())} frames -> {out}', flush=True)


if __name__ == '__main__':
    main()
