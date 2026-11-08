"""
StreamVGGT tables, figure and numbers for the paper, from the result files only.

    python -X utf8 scripts/make_svg_assets.py --res results/2026-11-07_svg_queue \
        --paper <paper folder>

Reads <res>/res/fidelity_*.json and result_scale.json (official video-depth runs),
<res>/long_bonn/*.json (240-frame runs against the exact reference) and
<res>/res/oom_lean.json, <res>/oom_second.json (frames until out of memory).
Writes tables/tab_svg_*.tex, figures/fig_svg_long.pdf and tables/svg_numbers.json,
and prints every number the text uses.
"""
import argparse
import json
import os

import numpy as np

ROWS = [  # (run name in the official evaluation, label)
    ('plain_int4', 'INT4'),
    ('plain_a2g32', 'INT2 asym., $g{=}32$'),
    ('kivi_2_64_w4', 'KIVI + 4 recent frames'),
    ('delta_int4_a2g64_8', '\\textbf{\\ours-2}, $K{=}8$'),
    ('plain_int2', 'INT2 symmetric'),
    ('plain_a2g64', 'INT2 asym., $g{=}64$'),
    ('kivi_2_64', 'KIVI'),
    ('delta_int4_bin_4', '\\textbf{\\ours-1}, $K{=}4$'),
    ('delta_int4_bin_8', '\\textbf{\\ours-1}, $K{=}8$'),
    ('xstream_2048_4_4', 'XStreamVGGT, 2048 tokens'),
]
LONG = [('plain:int4', 'INT4'), ('delta:int4:a2g64:8', '\\textbf{\\ours-2}'), ('plain:int2', 'INT2 symmetric'),
        ('plain:a2g64', 'INT2 asym., $g{=}64$'), ('kivi:2:64', 'KIVI'), ('delta:int4:bin:4', '\\textbf{\\ours-1}, $K{=}4$'),
        ('delta:int4:bin:8', '\\textbf{\\ours-1}, $K{=}8$'), ('xstream:2048:4:4', 'XStreamVGGT, 2048 tokens')]
PAIRS = [('delta_int4_bin_8', 'plain_a2g64'), ('delta_int4_bin_8', 'plain_int2'), ('delta_int4_bin_8', 'kivi_2_64'),
         ('delta_int4_bin_4', 'plain_a2g64'), ('delta_int4_bin_4', 'kivi_2_64'),
         ('delta_int4_a2g64_8', 'plain_a2g32'), ('delta_int4_a2g64_8', 'kivi_2_64_w4'),
         ('delta_int4_bin_8', 'xstream_2048_4_4')]
NUM = {}


def bootstrap(diff, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def official(res, ds):
    runs = {r['run'][len(ds) + 1:]: r for r in json.load(open(f'{res}/res/fidelity_{ds}.json'))['results']}
    for name, r in runs.items():
        path = f'{res}/res/{ds}_{name}/result_scale.json'
        r['depth'] = json.load(open(path)) if os.path.exists(path) else {}
    return runs


def main_table(res, out):
    lines = ['\\begin{tabular}{lrrrrrr}', '\\toprule',
             ' & Bits & Cache & Dev. & Flick. & AbsRel & $\\delta_1$ \\\\',
             ' & & (MB) & (\\%) & (\\%) & & \\\\']
    for ds, ds_name in (('sintel', 'Sintel (23 scenes)'), ('bonn', 'Bonn (5 scenes, 110 frames)')):
        runs = official(res, ds)
        ref, fp16 = runs['none'], runs['fp16']
        assert ref['n_sequences'] == ref['n_reference']
        lines += ['\\midrule', f'\\multicolumn{{7}}{{l}}{{\\emph{{{ds_name}}}}} \\\\',
                  f"Uncompressed (16-bit) & 16 & {fp16['cache_mb_max']:.0f} & -- & -- & "
                  f"{ref['depth']['Abs Rel']:.4f} & {ref['depth']['δ < 1.25']:.4f} \\\\"]
        NUM[f'{ds}_ref'] = {'cache_mb': fp16['cache_mb_max'], 'peak_mb': ref['peak_mb_max'],
                            'abs_rel': ref['depth']['Abs Rel'], 'delta1': ref['depth']['δ < 1.25']}
        for name, label in ROWS:
            r = runs[name]
            assert r['n_sequences'] == r['n_reference'], (ds, name)
            lines.append(f"\\quad {label} & {r['ideal_bits']:.2f} & {r['cache_mb_max']:.0f} & {r['fid_rel_dev_pct']:.2f} & "
                         f"{r['fid_flicker_pct']:.2f} & {r['depth']['Abs Rel']:.4f} & {r['depth']['δ < 1.25']:.4f} \\\\")
            NUM[f'{ds}_{name}'] = {'bits': r['ideal_bits'], 'cache_mb': r['cache_mb_max'], 'peak_mb': r['peak_mb_max'],
                                   'dev': r['fid_rel_dev_pct'], 'flicker': r['fid_flicker_pct'],
                                   'abs_rel': r['depth']['Abs Rel'], 'delta1': r['depth']['δ < 1.25'],
                                   'compression_vs_16bit': fp16['cache_mb_max'] / r['cache_mb_max'],
                                   'peak_saving_pct': 100 * (1 - r['peak_mb_max'] / ref['peak_mb_max'])}
        for ours, base in PAIRS:
            a, b = runs[ours]['per_sequence'], runs[base]['per_sequence']
            seqs = sorted(set(a) & set(b))
            for key, label in (('fid_rel_dev_pct', 'dev'), ('fid_flicker_pct', 'flicker')):
                x = np.array([a[s][key] for s in seqs])
                y = np.array([b[s][key] for s in seqs])
                lo, hi = bootstrap(y - x)
                NUM[f'{ds}_{ours}_vs_{base}_{label}'] = {
                    'ratio': float(y.mean() / x.mean()), 'wins': int((x < y).sum()), 'n': len(seqs),
                    'ci': [lo, hi], 'excludes_zero': bool(lo > 0 or hi < 0)}
    lines += ['\\bottomrule', '\\end{tabular}', '']
    open(f'{out}/tables/tab_svg_main.tex', 'w', encoding='utf8').write('\n'.join(lines))


def umeyama_ate(pred, gt):
    """RMSE (m) after the similarity transform that best maps pred onto gt."""
    mp, mg = pred.mean(0), gt.mean(0)
    p, g = pred - mp, gt - mg
    u, d, vt = np.linalg.svd(g.T @ p / len(p))
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1
    rot = u @ s @ vt
    scale = np.trace(np.diag(d) @ s) / p.var(0).sum()
    aligned = scale * p @ rot.T + mg
    return float(np.sqrt(((aligned - gt) ** 2).sum(1).mean()))


def long_table(res, out):
    folder = f'{res}/long_bonn'
    load = lambda kv: json.load(open(f"{folder}/{kv.replace(':', '_')}.json"))['scenes']
    ref = load('offload')
    gts = {}
    for scene, r in ref.items():
        gt = r['ground_truth']
        keep = [i for i, g in enumerate(gt) if g is not None]
        gts[scene] = (keep, np.array([gt[i] for i in keep]))

    def ate(scenes):
        vals = []
        for scene, r in scenes.items():
            keep, gt = gts[scene]
            c = np.array(r['centres'])
            keep = [i for i in keep if i < len(c)]
            vals.append(umeyama_ate(c[keep], gt[:len(keep)]))
        return float(np.mean(vals))

    NUM['long_ref_ate_m'] = ate(ref)
    NUM['long_frames'] = sorted({r['frames'] for r in ref.values()})
    lines = ['\\begin{tabular}{lrrrrrrrr}', '\\toprule',
             ' & Bits & \\multicolumn{2}{c}{Dev. (\\%)} & Flick. & Points & Cam. & Rot. & ATE \\\\',
             ' & & 1--110 & 111--240 & (\\%) & (\\%) & (\\%) & ($^\\circ$) & (cm) \\\\', '\\midrule',
             f"Uncompressed & 16 & -- & -- & -- & -- & -- & -- & {100 * NUM['long_ref_ate_m']:.2f} \\\\"]
    series = {}
    for kv, label in LONG:
        sc = load(kv)
        assert set(sc) == set(ref) and all(s['frames'] == ref[k]['frames'] for k, s in sc.items()), kv
        m = lambda key, lo=0, hi=None: float(np.mean([np.mean(s['series'][key][lo:hi]) for s in sc.values()]))
        row = {'bits': float(np.mean([s['bits'] for s in sc.values()])), 'dev': m('dev'),
               'dev_early': m('dev', 0, 110), 'dev_late': m('dev', 110), 'flicker': m('flicker'),
               'pts': m('pts_dev'), 'cam': m('cam_err'), 'rot': m('rot_deg'), 'ate_m': ate(sc),
               'cache_mb': float(np.mean([s['cache_mb'] for s in sc.values()]))}
        NUM[f'long_{kv}'] = row
        lines.append(f"{label} & {row['bits']:.2f} & {row['dev_early']:.2f} & {row['dev_late']:.2f} & "
                     f"{row['flicker']:.2f} & {row['pts']:.2f} & {row['cam']:.1f} & {row['rot']:.2f} & "
                     f"{100 * row['ate_m']:.2f} \\\\")
        n = min(len(s['series']['dev']) for s in sc.values())
        series[kv] = np.mean([s['series']['dev'][:n] for s in sc.values()], 0)
    lines += ['\\bottomrule', '\\end{tabular}', '']
    open(f'{out}/tables/tab_svg_long.tex', 'w', encoding='utf8').write('\n'.join(lines))
    return series


def long_figure(series, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(3.3, 1.5))
    show = [('xstream:2048:4:4', 'XStreamVGGT (4.5 b, pruned)', '#7f7f7f', '-'),
            ('plain:a2g64', 'INT2 asym. (2.5 b)', '#d62728', '-'),
            ('kivi:2:64', 'KIVI (2.5 b)', '#ff7f0e', '-'),
            ('delta:int4:bin:8', 'TCC-1 (1.9 b)', '#1f77b4', '-'),
            ('delta:int4:a2g64:8', 'TCC-2 (2.8 b)', '#2ca02c', '-')]
    w = 10
    for kv, label, color, ls in show:
        y = np.convolve(series[kv], np.ones(w) / w, mode='valid')
        ax.plot(np.arange(len(y)) + w // 2 + 1, y, color=color, ls=ls, lw=1.2, label=label)
    ax.axvline(110, color='k', lw=0.6, ls=':')
    ax.set_xlabel('frame')
    ax.set_ylabel('deviation (%)')
    ax.set_yscale('log')
    ax.legend(fontsize=6, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3,
              handlelength=1.5, columnspacing=0.8)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.2)
    fig.savefig(f'{out}/figures/fig_svg_long.pdf')


def frames_table(res, out):
    rows = {r['kv']: r for r in json.load(open(f'{res}/res/oom_lean.json'))}
    second = {r['kv']: r for r in json.load(open(f'{res}/oom_second.json'))}
    ref_frames = second['none']['frames_reached']
    NUM['oom_none_frames'] = ref_frames
    lines = ['\\begin{tabular}{lrrr}', '\\toprule', 'Cache & Bits & Frames & vs.\\ 16-bit \\\\', '\\midrule',
             f'Uncompressed (16-bit) & 16 & {ref_frames} & 1.0$\\times$ \\\\']
    for kv, label, bits in (('delta:int4:a2g64:8', '\\ours-2', '2.75'), ('plain:a2g64', 'INT2 asym., $g{=}64$', '2.50'),
                            ('delta:int4:bin:8', '\\ours-1', '1.88')):
        r = rows[kv]
        assert r['oom']
        NUM[f'oom_{kv}'] = {'frames': r['frames_reached'], 'ratio': r['frames_reached'] / ref_frames}
        lines.append(f"{label} & {bits} & {r['frames_reached']} & {r['frames_reached'] / ref_frames:.1f}$\\times$ \\\\")
    lines += ['\\bottomrule', '\\end{tabular}', '']
    open(f'{out}/tables/tab_svg_frames.tex', 'w', encoding='utf8').write('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', required=True)
    ap.add_argument('--paper', required=True)
    args = ap.parse_args()
    main_table(args.res, args.paper)
    series = long_table(args.res, args.paper)
    long_figure(series, args.paper)
    frames_table(args.res, args.paper)
    json.dump(NUM, open(f'{args.paper}/tables/svg_numbers.json', 'w'), indent=1, sort_keys=True)
    for k in sorted(NUM):
        print(k, json.dumps(NUM[k], default=float)[:240])


if __name__ == '__main__':
    main()
