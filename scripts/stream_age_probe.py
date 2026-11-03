"""
How much attention does each cached frame get, by age?  Runs VDA streaming in
FP32 and logs, per temporal-attention layer, the mean softmax mass the current
frame's queries put on each of the 31 cache slots (+ the current frame).

    python scripts/stream_age_probe.py --dataset bonn --frames 150 --out age_bonn.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
for p in (REPO, SCRIPT_DIR, REPO / 'Video-Depth-Anything'):
    sys.path.insert(0, str(p))

from datasets_gt import group_samples_by_scene, load_gt_dataset  # noqa: E402
from video_depth_anything.video_depth_stream import VideoDepthAnything  # noqa: E402
from research.models.stream_age import AttentionLog, bucket_slots, enable_age_quant  # noqa: E402

CFGS = {'vits': dict(encoder='vits', features=64, out_channels=[48, 96, 192, 384])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='bonn')
    ap.add_argument('--frames', type=int, default=150)
    ap.add_argument('--skip-warmup', type=int, default=45,
                    help='ignore the first frames, while the window still holds copies of frame 0')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    samples, _ = load_gt_dataset(args.dataset, REPO / 'benchmark_data', require_cam=True)
    scenes = group_samples_by_scene(samples)
    model = VideoDepthAnything(**CFGS['vits'])
    model.load_state_dict(torch.load(REPO / 'checkpoints' / 'video_depth_anything_vits.pth',
                                     map_location='cpu'))
    model = model.cuda().eval()
    log = AttentionLog()
    gate = {'on': False}

    class GatedLog(AttentionLog):
        def add(self, layer, probs):
            if gate['on']:
                log.add(layer, probs)
    enable_age_quant(model, {}, GatedLog())

    for name, frames in scenes.items():
        model.transform, model.frame_id_list, model.frame_cache_list, model.id = None, [], [], -1
        for i, f in enumerate(frames[:args.frames]):
            gate['on'] = i >= args.skip_warmup
            with torch.no_grad():
                model.infer_video_depth_one(f['rgb'], input_size=518, device='cuda')
        print(f"  {name}: {min(len(frames), args.frames)} frames", flush=True)

    summ = log.summary()
    buckets = {b: bucket_slots(b) for b in ('anchor', 'oldest', 'old', 'mid', 'recent')}
    out = {'per_layer_slot_mass': summ, 'per_layer_bucket_mass': {}}
    for layer, mass in summ.items():
        row = {b: sum(mass[s] for s in idx) for b, idx in buckets.items()}
        row['current'] = mass[31]
        row['per_slot_mean'] = {b: row[b] / len(idx) for b, idx in buckets.items()}
        out['per_layer_bucket_mass'][layer] = row
        print(f"layer {layer}: " + "  ".join(f"{b}={row[b]:.3f}" for b in
                                          ('anchor', 'oldest', 'old', 'mid', 'recent', 'current')),
              flush=True)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2)


if __name__ == '__main__':
    main()
