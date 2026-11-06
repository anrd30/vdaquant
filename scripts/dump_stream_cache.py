"""
Dump VDA's streaming temporal cache (the entry each frame adds to every layer)
and the depth outputs for one Sintel scene, for the paper's illustration figures.
Runs on CPU in FP32 by default so it does not disturb GPU jobs.

usage: python -X utf8 scripts/dump_stream_cache.py --scene bamboo_1 --out outputs/figdata/bamboo_1.pt
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'Video-Depth-Anything'))
sys.path.insert(0, str(REPO))
from video_depth_anything.video_depth_stream import VideoDepthAnything  # noqa: E402

CFGS = {
    'vits': dict(encoder='vits', features=64, out_channels=[48, 96, 192, 384]),
    'vitb': dict(encoder='vitb', features=128, out_channels=[96, 192, 384, 768]),
    'vitl': dict(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='bamboo_1')
    ap.add_argument('--encoder', default='vits')
    ap.add_argument('--frames', type=int, default=50)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = VideoDepthAnything(**CFGS[args.encoder])
    ckpt = REPO / 'checkpoints' / f'video_depth_anything_{args.encoder}.pth'
    model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
    model = model.to(args.device).eval()

    frame_dir = REPO / 'benchmark_data' / 'sintel' / 'training' / 'clean' / args.scene
    paths = sorted(frame_dir.glob('*.png'))[:args.frames]
    cache, depth, rgb = [], [], []
    for i, p in enumerate(paths):
        frame = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        d = model.infer_video_depth_one(frame, device=args.device, fp32=True)
        # newest cache entry: one tensor per temporal layer, shape (N, 1, C)
        cache.append([h[:, 0].half().cpu() for h in model.frame_cache_list[-1]])
        depth.append(torch.from_numpy(d).half())
        rgb.append(torch.from_numpy(frame))
        print(f'{i + 1}/{len(paths)}', flush=True)

    x_shape = model.transform({'image': frame.astype(np.float32) / 255.0})['image'].shape  # (3, H, W)
    n_layers = len(cache[0])
    out = {
        'scene': args.scene, 'encoder': args.encoder, 'input_hw': tuple(x_shape[1:]),
        'cache': [torch.stack([c[l] for c in cache]) for l in range(n_layers)],   # per layer (T, N, C)
        'depth': torch.stack(depth), 'rgb': torch.stack(rgb),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print('layers', [tuple(t.shape) for t in out['cache']], 'input', out['input_hw'])


if __name__ == '__main__':
    main()
