"""
Go/no-go measurements for codec-style cache coding (no compression applied):

1. Skip blocks: per layer, fraction of 16-channel groups whose frame-to-frame
   residual RMS is below tau x (the layer's per-channel-centred RMS), and the
   share of residual energy those groups carry (what skipping would discard).
2. Motion compensation: on the token grid, for blocks of BxB tokens, search
   integer shifts within +-R of the previous frame and keep the best; report
   best-shift residual energy / zero-shift residual energy.

    python scripts/stream_codec_probe.py --dataset bonn --frames 150 --out probe_bonn.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
for p in (REPO, SCRIPT_DIR, REPO / 'Video-Depth-Anything'):
    sys.path.insert(0, str(p))

from datasets_gt import group_samples_by_scene, load_gt_dataset  # noqa: E402
from video_depth_anything.video_depth_stream import VideoDepthAnything  # noqa: E402

CFGS = {'vits': dict(encoder='vits', features=64, out_channels=[48, 96, 192, 384]),
        'vitb': dict(encoder='vitb', features=128, out_channels=[96, 192, 384, 768]),
        'vitl': dict(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])}
TAUS = [0.02, 0.05, 0.1, 0.2]


def grid_shape(n, aspect):
    best = None
    for a in range(1, int(n ** 0.5) + 2):
        if n % a == 0:
            for h, w in ((a, n // a), (n // a, a)):
                err = abs(np.log((h / w) / aspect))
                if best is None or err < best[0]:
                    best = (err, h, w)
    return best[1], best[2]


def motion_ratio(cur, prev, hw, B, R):
    """cur, prev: (N, C).  Returns (best-shift energy, zero-shift energy)."""
    H, W = hw
    c = cur.reshape(H, W, -1).permute(2, 0, 1)[None]
    p = F.pad(prev.reshape(H, W, -1).permute(2, 0, 1)[None], (R, R, R, R), mode='replicate')
    Hb, Wb = H // B * B, W // B * B
    best, zero = None, None
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            sh = p[..., R + dy:R + dy + H, R + dx:R + dx + W]
            e = ((c - sh) ** 2).sum(1, keepdim=True)[..., :Hb, :Wb]
            eb = F.avg_pool2d(e, B) * B * B                     # energy per block
            best = eb if best is None else torch.minimum(best, eb)
            if dy == 0 and dx == 0:
                zero = eb
    return float(best.sum()), float(zero.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='bonn')
    ap.add_argument('--encoder', default='vits')
    ap.add_argument('--frames', type=int, default=150)
    ap.add_argument('--max-scenes', type=int, default=1000)
    ap.add_argument('--block', type=int, default=4)
    ap.add_argument('--radius', type=int, default=2)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    samples, _ = load_gt_dataset(args.dataset, REPO / 'benchmark_data', require_cam=True)
    scenes = group_samples_by_scene(samples)
    names = list(scenes)[:args.max_scenes]
    model = VideoDepthAnything(**CFGS[args.encoder])
    model.load_state_dict(torch.load(REPO / 'checkpoints' / f'video_depth_anything_{args.encoder}.pth',
                                     map_location='cpu'))
    model = model.cuda().eval()

    acc = None  # per layer accumulators
    for name in names:
        model.transform, model.frame_id_list, model.frame_cache_list, model.id = None, [], [], -1
        frames = scenes[name][:args.frames]
        H0, W0 = frames[0]['rgb'].shape[:2]
        prev = None
        for f in frames:
            with torch.no_grad():
                model.infer_video_depth_one(f['rgb'], input_size=518, device='cuda')
            cur = [t.float().squeeze(1) for t in model.frame_cache_list[-1]]     # (N, C)
            if acc is None:
                acc = [dict(N=c.shape[0], C=c.shape[1], centred=0.0, resid=0.0,
                            skip_frac={t: 0.0 for t in TAUS}, skip_energy={t: 0.0 for t in TAUS},
                            groups=0, mc_best=0.0, mc_zero=0.0) for c in cur]
            if prev is not None:
                for a, c, p in zip(acc, cur, prev):
                    r = c - p
                    rms_layer = (c - c.mean(0, keepdim=True)).pow(2).mean().sqrt()
                    a['centred'] += float((c - c.mean(0, keepdim=True)).pow(2).sum())
                    a['resid'] += float(r.pow(2).sum())
                    g = r.reshape(r.shape[0], -1, 16)
                    g_rms = g.pow(2).mean(-1).sqrt()                      # (N, C/16)
                    g_en = g.pow(2).sum(-1)
                    tot = float(g_en.sum()) + 1e-12
                    for t in TAUS:
                        m = g_rms <= t * rms_layer
                        a['skip_frac'][t] += float(m.float().sum())
                        a['skip_energy'][t] += float(g_en[m].sum()) / tot * g_rms.numel()
                    a['groups'] += g_rms.numel()
                    hw = grid_shape(c.shape[0], W0 / H0)
                    b, z = motion_ratio(c, p, hw, args.block, args.radius)
                    a['mc_best'] += b
                    a['mc_zero'] += z
            prev = cur
        print(f"  {name}: {len(frames)} frames", flush=True)

    out = []
    for i, a in enumerate(acc):
        row = {'layer': i, 'N': a['N'], 'C': a['C'],
               'resid_over_centred': a['resid'] / max(a['centred'], 1e-12),
               'skip_frac': {str(t): a['skip_frac'][t] / a['groups'] for t in TAUS},
               'skip_energy_share': {str(t): a['skip_energy'][t] / a['groups'] for t in TAUS},
               'motion_best_over_zero': a['mc_best'] / max(a['mc_zero'], 1e-12)}
        out.append(row)
        print(f"layer {i} N={a['N']} C={a['C']} resid/centred={row['resid_over_centred']:.3f} "
              f"skip@0.05={row['skip_frac']['0.05']:.2f} skip@0.1={row['skip_frac']['0.1']:.2f} "
              f"(energy share {row['skip_energy_share']['0.1']:.3f}) "
              f"motion best/zero={row['motion_best_over_zero']:.3f}", flush=True)
    with open(args.out, 'w') as fh:
        json.dump({'args': vars(args), 'layers': out}, fh, indent=2)


if __name__ == '__main__':
    main()
