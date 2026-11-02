"""
Probe: how predictable is each frame's temporal-cache entry from the previous
frame's?  For every temporal-attention layer, compares the energy of the entry
h_t with the energy of the residual h_t - h_{t-1}.  A residual/energy ratio r
means delta coding could save about 0.5*log2(1/r) bits per scalar at equal
distortion (high-rate approximation).

    python scripts/stream_delta_probe.py --dataset bonn --frames 150
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
for p in (REPO, SCRIPT_DIR, REPO / 'Video-Depth-Anything'):
    sys.path.insert(0, str(p))

from datasets_gt import group_samples_by_scene, load_gt_dataset  # noqa: E402
from video_depth_anything.video_depth_stream import VideoDepthAnything  # noqa: E402

CFGS = {'vits': dict(encoder='vits', features=64, out_channels=[48, 96, 192, 384])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='bonn')
    ap.add_argument('--scene', default=None)
    ap.add_argument('--frames', type=int, default=150)
    ap.add_argument('--encoder', default='vits')
    args = ap.parse_args()

    samples, _ = load_gt_dataset(args.dataset, REPO / 'benchmark_data', require_cam=True)
    scenes = group_samples_by_scene(samples)
    names = [args.scene] if args.scene else list(scenes)
    model = VideoDepthAnything(**CFGS[args.encoder])
    model.load_state_dict(torch.load(REPO / 'checkpoints' / f'video_depth_anything_{args.encoder}.pth',
                                     map_location='cpu'))
    model = model.cuda().eval()

    for name in names:
        model.transform, model.frame_id_list, model.frame_cache_list, model.id = None, [], [], -1
        prev, energy, resid, centred = None, None, None, None
        for f in scenes[name][:args.frames]:
            model.infer_video_depth_one(f['rgb'], input_size=518, device='cuda')
            cur = [t.float() for t in model.frame_cache_list[-1]]
            if prev is not None:
                e = [float((c ** 2).mean()) for c in cur]
                r = [float(((c - p) ** 2).mean()) for c, p in zip(cur, prev)]
                v = [float(((c - c.mean(dim=0, keepdim=True)) ** 2).mean()) for c in cur]
                centred = v if centred is None else [a + b for a, b in zip(centred, v)]
                energy = e if energy is None else [a + b for a, b in zip(energy, e)]
                resid = r if resid is None else [a + b for a, b in zip(resid, r)]
            prev = cur
        ratios = [r / v for r, v in zip(resid, centred)]   # vs per-channel-mean-removed variance
        print(f"{args.dataset}/{name}: centred/raw energy per layer = "
              + ", ".join(f"{v / e:.3f}" for v, e in zip(centred, energy)))
        saved = [0.5 * np.log2(1 / max(x, 1e-12)) for x in ratios]
        print(f"{args.dataset}/{name}: residual/centred per layer = "
              + ", ".join(f"{x:.3f}" for x in ratios))
        print(f"    => potential saving per layer (bits/scalar) = "
              + ", ".join(f"{s:.2f}" for s in saved)
              + f"   | weighted residual/centred {sum(resid) / sum(centred):.3f}")


if __name__ == '__main__':
    main()
