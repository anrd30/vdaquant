"""
Illustration figures from a dumped streaming cache (scripts/dump_stream_cache.py):
  fig_cache_video.pdf  - RGB frames, the cache entry they produce (PCA of channels),
                         and the change from the previous entry, same scale
  fig_error_time.pdf   - per-frame cache reconstruction error: INT2, INT4,
                         closed-loop 1-bit (with / without keyframes), open-loop 1-bit
  thumb_depth.png      - depth thumbnail for the method diagram
Quantizers are the ones used in the experiments (research/models/stream_cache.py).

usage: python -X utf8 scripts/make_cache_figures.py --dump outputs/figdata/bamboo_1.pt
           --out C:/Users/aniru/Downloads/cvpr2027/paper/figures
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from research.models.stream_cache import scalar_compress, scalar_decompress  # noqa: E402

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'font.size': 8, 'axes.linewidth': 0.6, 'pdf.fonttype': 42, 'mathtext.fontset': 'stix'})
KEYC, RESC = '#2F6DB5', '#E08A1E'


def q(x, spec):
    return scalar_decompress(scalar_compress(x, spec, None), None).float()


def grid_of(n, in_hw):
    """Spatial grid (h, w) of a layer with n positions, from the input size."""
    ph, pw = in_hw[0] // 14, in_hw[1] // 14
    s = (n / (ph * pw)) ** 0.5
    return int(round(ph * s)), int(round(pw * s))


def code_sequence(H, mode, K=8):
    """H (T, N, C) -> reconstruction (T, N, C) under a coding scheme."""
    T = H.shape[0]
    rec = torch.zeros_like(H)
    for t in range(T):
        h = H[t]
        if mode in ('int2-g64-asym', 'int4-g16'):
            rec[t] = q(h, mode)
            continue
        key = (t == 0) or (K is not None and t % K == 0)
        if key:
            rec[t] = q(h, 'int4-g16')
        elif mode == 'closed':
            rec[t] = rec[t - 1] + q(h - rec[t - 1], 'bin-g16')
        elif mode == 'closed2':
            rec[t] = rec[t - 1] + q(h - rec[t - 1], 'int2-g64-asym')
        elif mode == 'open':
            rec[t] = rec[t - 1] + q(h - H[t - 1], 'bin-g16')
    return rec


def rel_err(H, R):
    Hc = H - H.mean(dim=(0, 1), keepdim=True)
    return ((H - R).pow(2).sum(dim=(1, 2)) / Hc.pow(2).sum(dim=(1, 2))).sqrt()


def fig_cache_video(d, out, layer, frames):
    H = d['cache'][layer].float()                       # (T, N, C)
    h, w = grid_of(H.shape[1], d['input_hw'])
    mu = H.mean(dim=(0, 1), keepdim=True)
    X = (H - mu).reshape(-1, H.shape[-1])
    _, _, V = torch.pca_lowrank(X, q=3, center=False)
    P = ((H - mu) @ V).reshape(H.shape[0], h, w, 3).numpy()
    lo, hi = np.percentile(P, 1, axis=(0, 1, 2)), np.percentile(P, 99, axis=(0, 1, 2))
    P = np.clip((P - lo) / (hi - lo), 0, 1)
    energy = (H - mu).norm(dim=-1)                      # per-position size of the entry
    change = (H[1:] - H[:-1]).norm(dim=-1)              # per-position frame-to-frame change
    vmax = float(np.percentile(energy.numpy(), 99))
    ratio = [float(change[t - 1].pow(2).sum() / energy[t].pow(2).sum()) for t in frames]  # variance ratio

    n = len(frames)
    fig, ax = plt.subplots(3, n, figsize=(6.9, 3 * 6.9 / n * h / w + 0.35), squeeze=False)
    for j, t in enumerate(frames):
        ax[0, j].imshow(d['rgb'][t].numpy())
        ax[0, j].set_title(f'frame {t}', fontsize=8, pad=2)
        ax[1, j].imshow(P[t], interpolation='nearest')
        ax[2, j].imshow(change[t - 1].reshape(h, w).numpy(), cmap='magma', vmin=0, vmax=vmax,
                        interpolation='nearest')
        ax[2, j].text(0.98, 0.04, f'{100 * ratio[j]:.0f}%', transform=ax[2, j].transAxes, ha='right',
                      va='bottom', color='white', fontsize=7, fontweight='bold')
    for a in ax.ravel():
        a.set_xticks([]), a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    labels = ['input\nframe', 'cache\nentry $h_t$', 'change\n$h_t - h_{t-1}$']
    for i, lab in enumerate(labels):
        ax[i, 0].text(-0.04, 0.5, lab, transform=ax[i, 0].transAxes, ha='right', va='center',
                      fontsize=7.5, linespacing=1.1)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.93, bottom=0.01, wspace=0.03, hspace=0.06)
    fig.savefig(out / 'fig_cache_video.pdf', dpi=300)
    fig.savefig(out / 'fig_cache_video.png', dpi=200)
    plt.close(fig)
    return {'layer': layer, 'grid': [h, w], 'frames': frames, 'change_ratio': ratio}


def fig_error_time(d, out):
    curves = {}
    for name, mode, K in [('INT2 asym. (2.5 b)', 'int2-g64-asym', None), ('INT4 (4.5 b)', 'int4-g16', None),
                          ('open loop, no keyframes', 'open', None),
                          ('closed loop, no keyframes', 'closed', None),
                          ('TCC-1, $K{=}8$', 'closed', 8), ('TCC-2, $K{=}8$', 'closed2', 8)]:
        errs = []
        for H in d['cache']:
            H = H.float()
            errs.append(rel_err(H, code_sequence(H, mode, K)))
        curves[name] = torch.stack(errs).mean(0).numpy()        # mean over the 8 layers
    style = {'INT2 asym. (2.5 b)': dict(color='0.45', ls='--'), 'INT4 (4.5 b)': dict(color='0.7', ls='--'),
             'open loop, no keyframes': dict(color='#B23A48', ls=':'),
             'closed loop, no keyframes': dict(color=RESC, ls='-.'),
             'TCC-1, $K{=}8$': dict(color=KEYC, ls='-', lw=1.4),
             'TCC-2, $K{=}8$': dict(color='#1B8A5A', ls='-', lw=1.4)}
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    T = len(next(iter(curves.values())))
    for name, c in curves.items():
        ax.plot(np.arange(1, T), 100 * c[1:], label=name, lw=style[name].pop('lw', 1.0), **style[name])
    for t in range(8, T, 8):
        ax.axvline(t, color=KEYC, lw=0.4, alpha=0.35)
    ax.set_xlabel('frame')
    ax.set_ylabel('cache error (%)')
    ax.set_xlim(1, T - 1)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc='lower left', bbox_to_anchor=(-0.02, 1.0),
              handlelength=2.2, columnspacing=0.8, borderaxespad=0.2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(out / 'fig_error_time.pdf')
    fig.savefig(out / 'fig_error_time.png', dpi=200)
    plt.close(fig)
    return {k: {'mean': float(v[1:].mean()), 'max': float(v[1:].max()), 'last': float(v[-1])}
            for k, v in curves.items()}


def thumb_depth(d, out, t):
    x = d['depth'][t].float().numpy()
    x = (x - np.percentile(x, 1)) / (np.percentile(x, 99) - np.percentile(x, 1))
    im = cv2.applyColorMap((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    hh, ww = im.shape[:2]
    cv2.imwrite(str(out / 'thumb_depth.png'), cv2.resize(im, (ww // 3, hh // 3), interpolation=cv2.INTER_AREA))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layer', type=int, default=6)
    ap.add_argument('--frames', default='20,21,22,23,24')
    args = ap.parse_args()
    out = Path(args.out)
    d = torch.load(args.dump)
    facts = {'scene': d['scene'], 'encoder': d['encoder']}
    thumb_depth(d, out, 29)
    facts['cache_video'] = fig_cache_video(d, out, args.layer, [int(f) for f in args.frames.split(',')])
    facts['error_time'] = fig_error_time(d, out)
    facts['change_ratio_all_layers'] = [
        float(((H[1:].float() - H[:-1].float()).pow(2).sum()
               / (H.float() - H.float().mean(dim=(0, 1), keepdim=True)).pow(2).sum()))
        for H in d['cache']]
    (out / 'fig_cache_facts.json').write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts, indent=1))


if __name__ == '__main__':
    main()
