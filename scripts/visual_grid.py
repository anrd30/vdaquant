"""
Side-by-side depth visualisations for several cache / quantization settings.

For a few frames of one Sintel scene, shows RGB, ground truth, and each
setting's prediction (aligned to GT disparity with one least-squares
scale+shift per frame, so colours are comparable), plus |prediction - FP32|.

    python scripts/visual_grid.py --mode stream  --out outputs/visuals/stream_grid.png
    python scripts/visual_grid.py --mode offline --out outputs/visuals/offline_grid.png
"""
import argparse
import gc
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
for p in (REPO, SCRIPT_DIR, REPO / 'Video-Depth-Anything'):
    sys.path.insert(0, str(p))

from datasets_gt import compute_gt_depth_metrics, group_samples_by_scene, load_gt_dataset  # noqa: E402

CFG = dict(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
CKPT = REPO / 'checkpoints' / 'video_depth_anything_vits.pth'

STREAM = [('FP32 cache', 'fp32'),
          ('INT2, 2.5 b', 'int2-g16'),
          ('Delta: INT3 key + 1-bit diff, ~1.75 b', 'int3-g16::d8:bin-g16'),
          ('1-bit, no delta, 1.5 b', 'bin-g16'),
          ('Cache wiped (control)', 'zero')]
OFFLINE = [('FP32', None),
           ('INT3 g64, 3.125 b', 'scalar_g8:64:3'),
           ('E8 g64, 2.625 b', 'lattice_e8:64:3'),
           ('INT2 g16, 2.5 b', 'scalar_g8:16:2'),
           ('E8 b=2 g64, 1.625 b', 'lattice_e8:64:2')]


def stream_preds(spec, frames, upto):
    from video_depth_anything.video_depth_stream import VideoDepthAnything
    from research.models.stream_cache import enable_compressed_cache
    m = VideoDepthAnything(**CFG)
    m.load_state_dict(torch.load(CKPT, map_location='cpu'))
    m = m.cuda().eval()
    parts = spec.split(':')
    mode = parts[0]
    bits = int(parts[1]) if len(parts) > 1 and parts[1] else 3
    dk = int(parts[2][1:]) if len(parts) > 2 and parts[2] else 0
    rmode = parts[3] if len(parts) > 3 else None
    m = enable_compressed_cache(m, mode=mode, bits=bits, seed=0, delta=dk, residual_mode=rmode)
    out = [m.infer_video_depth_one(f['rgb'], input_size=518, device='cuda') for f in frames[:upto + 1]]
    del m
    return out


def offline_preds(spec, window):
    import run_pareto_benchmark_suite as suite
    from video_depth_anything.video_depth import VideoDepthAnything
    from research.models.rotated_attention import apply_rotated_quantization_to_vda
    m = VideoDepthAnything(**CFG)
    m.load_state_dict(torch.load(CKPT, map_location='cpu'))
    m = m.cuda().eval()
    if spec:
        q, g, b = spec.split(':')
        m = apply_rotated_quantization_to_vda(m, bits=int(b), quantizer=q, use_qjl=False,
                                              scale_bits=8, scale_group=int(g), rht_seed=0,
                                              replace_backbone=True, replace_temporal=True,
                                              verbose=False)
    preds = [p.numpy() for p in suite._predict_window(m, [f['rgb'] for f in window])]
    del m
    return preds


def align(pred, gt, gt_range):
    """Least-squares scale+shift of predicted disparity onto GT disparity."""
    valid = (gt > 1e-3) & (gt < gt_range[1])
    gd = np.zeros_like(gt)
    gd[valid] = 1.0 / gt[valid]
    A = np.stack([pred[valid], np.ones(valid.sum())], 1)
    s, t = np.linalg.lstsq(A, gd[valid], rcond=None)[0]
    return s * pred + t, gd, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['stream', 'offline'], default='stream')
    ap.add_argument('--scene', default='bamboo_1')
    ap.add_argument('--frames', default='15,30,45')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    samples, gt_range = load_gt_dataset('sintel', REPO / 'benchmark_data', require_cam=True)
    frames = group_samples_by_scene(samples)[args.scene]
    idx = [int(i) for i in args.frames.split(',')]
    settings = STREAM if args.mode == 'stream' else OFFLINE

    results = {}
    for label, spec in settings:
        print('running', label, flush=True)
        with torch.no_grad():
            if args.mode == 'stream':
                preds = stream_preds(spec, frames, max(idx))
                results[label] = [preds[i] for i in idx]
            else:
                start = max(0, min(idx) - 1)
                window = frames[start:start + 32]
                preds = offline_preds(spec, window)
                results[label] = [preds[i - start] for i in idx]
        gc.collect()
        torch.cuda.empty_cache()

    ref_label = settings[0][0]
    ncol, nrow = 2 * len(idx), len(settings) + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.0 * nrow))
    for c, i in enumerate(idx):
        f = frames[i]
        gt = f['depth']
        _, gd, valid = align(results[ref_label][c], gt, gt_range)
        lo, hi = np.percentile(gd[valid], [2, 98])
        axes[0, 2 * c].imshow(f['rgb'])
        axes[0, 2 * c].set_title(f'RGB frame {i}', fontsize=9)
        axes[0, 2 * c + 1].imshow(np.where(valid, gd, np.nan), cmap='Spectral_r', vmin=lo, vmax=hi)
        axes[0, 2 * c + 1].set_title('Ground truth', fontsize=9)
        ref_al, _, _ = align(results[ref_label][c], gt, gt_range)
        for r, (label, _) in enumerate(settings, start=1):
            al, _, _ = align(results[label][c], gt, gt_range)
            d1 = compute_gt_depth_metrics(torch.from_numpy(results[label][c]).float(),
                                          torch.from_numpy(gt).float(), torch.from_numpy(valid),
                                          gt_range=gt_range, pred_is_disparity=True)['delta1']
            axes[r, 2 * c].imshow(al, cmap='Spectral_r', vmin=lo, vmax=hi)
            axes[r, 2 * c].set_title(f'{label}  d1={d1:.3f}', fontsize=8)
            axes[r, 2 * c + 1].imshow(np.abs(al - ref_al), cmap='magma', vmin=0, vmax=(hi - lo) * 0.25)
            axes[r, 2 * c + 1].set_title(f'|diff vs {ref_label}|', fontsize=8)
    for a in axes.ravel():
        a.axis('off')
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print('saved', args.out)


if __name__ == '__main__':
    main()
