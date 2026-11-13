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

STREAM = [('FP32 cache (32 b)', 'fp32'),
          ('INT2 asym. (2.5 b)', 'int2-g64-asym'),
          ('TCC-1 (2.0 b)', 'int4-g16::d8:bin-g16'),
          ('INT2 asym. (3.0 b)', 'int2-g32-asym'),
          ('TCC-2 (2.9 b)', 'int4-g16::d8:int2-g64-asym'),
          ('1-bit, no prediction', 'bin-g16'),
          ('zero cache', 'zero')]
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


def plot_stream_grid(results, settings, frames, idx, gt_range, out):
    """Paper layout: one column per cache format; per frame a depth row and a
    |difference to the FP32-cache output| row (first column: RGB and GT)."""
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 7.5, 'pdf.fonttype': 42})
    ref_label = settings[0][0]
    h, w = frames[idx[0]]['rgb'].shape[:2]
    ncol, nrow = len(settings) + 1, 2 * len(idx)
    cw = 6.9 / ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.9, nrow * cw * h / w + 0.25), squeeze=False)
    for j, i in enumerate(idx):
        f = frames[i]
        ref_al, gd, valid = align(results[ref_label][j], f['depth'], gt_range)
        lo, hi = np.percentile(gd[valid], [2, 98])
        rd, rf = 2 * j, 2 * j + 1
        axes[rd, 0].imshow(f['rgb'])
        axes[rd, 0].text(0.03, 0.06, f'frame {i}', transform=axes[rd, 0].transAxes, color='white',
                         fontsize=6.5, fontweight='bold')
        axes[rf, 0].imshow(np.where(valid, gd, np.nan), cmap='Spectral_r', vmin=lo, vmax=hi)
        axes[rf, 0].text(0.03, 0.06, 'ground truth', transform=axes[rf, 0].transAxes, color='k', fontsize=6.5)
        scale = np.abs(ref_al[valid]).mean()
        for c, (label, _) in enumerate(settings, start=1):
            al, _, _ = align(results[label][j], f['depth'], gt_range)
            axes[rd, c].imshow(al, cmap='Spectral_r', vmin=lo, vmax=hi)
            diff = np.abs(al - ref_al)
            axes[rf, c].imshow(diff, cmap='magma', vmin=0, vmax=(hi - lo) * 0.25)
            if c > 1:
                axes[rf, c].text(0.97, 0.06, f'{100 * diff[valid].mean() / scale:.1f}%', ha='right',
                                 transform=axes[rf, c].transAxes, color='white', fontsize=6.5,
                                 fontweight='bold')
            else:
                axes[rf, c].text(0.5, 0.5, 'reference', ha='center', va='center',
                                 transform=axes[rf, c].transAxes, color='0.6', fontsize=6.5)
    for c, (label, _) in enumerate([('input / GT', None)] + list(settings)):
        axes[0, c].set_title(label, fontsize=7.5, pad=2)
    for a in axes.ravel():
        a.set_xticks([]), a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(left=0.003, right=0.997, top=0.93, bottom=0.003, wspace=0.02, hspace=0.04)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print('saved', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['stream', 'offline'], default='stream')
    ap.add_argument('--scene', default='bamboo_1')
    ap.add_argument('--frames', default='15,30,45')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dataset', default='sintel')
    ap.add_argument('--preds', help='npz cache of predictions (stream mode): reused if present')
    args = ap.parse_args()

    samples, gt_range = load_gt_dataset(args.dataset, REPO / 'benchmark_data',
                                        require_cam=(args.dataset == 'sintel'))
    scenes = group_samples_by_scene(samples)
    if args.scene not in scenes:
        raise SystemExit(f'scene {args.scene} not in {sorted(scenes)}')
    frames = scenes[args.scene]
    idx = [int(i) for i in args.frames.split(',')]
    settings = STREAM if args.mode == 'stream' else OFFLINE

    if args.mode == 'stream':
        cache = Path(args.preds) if args.preds else None
        if cache and cache.exists():
            z = np.load(cache)
            results = {label: [z[f'{k}_{i}'] for i in idx] for k, (label, _) in enumerate(settings)}
        else:
            results = {}
            for label, spec in settings:
                print('running', label, flush=True)
                with torch.no_grad():
                    preds = stream_preds(spec, frames, max(idx))
                results[label] = [preds[i] for i in idx]
                del preds
                gc.collect()
                torch.cuda.empty_cache()
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, **{f'{k}_{i}': results[label][j] for k, (label, _) in
                                              enumerate(settings) for j, i in enumerate(idx)})
        plot_stream_grid(results, settings, frames, idx, gt_range, args.out)
        return

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
