"""Build the CVPR supplement (v1): tables from the result JSONs and supp.tex next to the paper.
    python scripts/make_supp.py  ->  <paper>/supp/{supp.tex, tab_*.tex}; compile supp.tex in <paper>.
Every number in the tables is read from results/; nothing is typed by hand."""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

import make_paper_assets as m

REPO = Path(__file__).resolve().parents[1]
R = REPO / 'results'
PAPER = Path('C:/Users/aniru/Downloads/cvpr2027/paper')
OUT = PAPER / 'supp'
SUPP = R / '2026-11-16_supp'
SCN = R / '2026-11-15_scannet_tcc'
TCC1, TCC1K4, TCC2 = m.OURS, m.OURS4, m.TCC2
ROWS = [('fp32', 'FP32 cache'), ('int2-g16', 'INT2 symmetric'), ('int2-g64-asym', 'INT2 asym., $g{=}64$'),
        ('int4-g16@win16', 'INT4, last 16 frames'), (TCC1, '\\textbf{TCC-1}, $K{=}8$'),
        (TCC1K4, '\\textbf{TCC-1}, $K{=}4$'), ('int2-g32-asym', 'INT2 asym., $g{=}32$'),
        (TCC2, '\\textbf{TCC-2}, $K{=}8$')]
ENC = {'vits': 'ViT-S', 'vitb': 'ViT-B', 'vitl': 'ViT-L'}
J = lambda p: {r.get('config', r['mode']): r for r in json.load(open(p))['results']}
f = lambda x, d=2: '--' if x is None else f'{x:.{d}f}'


def dataset_table(name, load):
    """Rows: cache formats. Per encoder: bits, delta1, AbsRel (per-sequence alignment), deviation, flicker."""
    data = {e: load(e) for e in ENC}
    L = [r'\begin{tabular}{l' + 'rrrrr' * 3 + '}', r'\toprule',
         ' & ' + ' & '.join(rf'\multicolumn{{5}}{{c}}{{{ENC[e]}}}' for e in ENC) + r' \\',
         ''.join(rf'\cmidrule(lr){{{2 + 5 * i}-{6 + 5 * i}}}' for i in range(3)),
         'Cache' + r' & bits & $\delta_1$ & Rel & Dev. & Flk.' * 3 + r' \\', r'\midrule']
    for c, lab in ROWS:
        cells = []
        for e in ENC:
            X = data[e]
            if c not in X:
                cells += ['--'] * 5
                continue
            r, b = X[c], X['fp32']
            cells += [f(32 * r['cache_mb'] / b['cache_mb']), f(r['delta1_seq'], 3), f(r['abs_rel_seq'], 3),
                      f(r['fid_rel_dev_pct']), f(r['fid_flicker_pct'])]
        L.append(lab + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / f'tab_{name}.tex').write_text('\n'.join(L), encoding='utf-8')


def splice_table():
    L = [r'\begin{tabular}{llrrrrr}', r'\toprule',
         r'Enc. & Cache & bits & before & frames 1--3 & peak & after \\', r'\midrule']
    for e in ('vits', 'vitl'):
        X = J(SUPP / f'splice_scannet_{e}.json')
        for i, (c, lab) in enumerate([('int2-g64-asym', 'INT2 asym., $g{=}64$'), (TCC1, 'TCC-1, $K{=}8$'),
                                      (TCC1 + ':a0.5', 'TCC-1, $K{=}8$ + adaptive keyframe'),
                                      (TCC2, 'TCC-2, $K{=}8$')]):
            pf = [v['dev_per_frame'] for v in X[c]['per_scene'].values()]
            mean = lambda a, b: float(np.mean([np.mean(p[a:b]) for p in pf]))
            peak = float(np.mean([max(p[85:95]) for p in pf]))
            L.append(f"{ENC[e] if i == 0 else ''} & {lab} & {32 * X[c]['cache_mb'] / X['fp32']['cache_mb']:.2f} & "
                     f"{mean(10, 85):.2f} & {mean(86, 89):.2f} & {peak:.2f} & {mean(95, 170):.2f} \\\\")
        if e == 'vits':
            L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'tab_splice.tex').write_text('\n'.join(L), encoding='utf-8')


def module_table():
    L = [r'\begin{tabular}{lllrrrrr}', r'\toprule',
         r'Data & Enc. & Cache & M1 & M2 & M3 & M4 & Depth \\', r'\midrule']
    for ds in ('sintel', 'bonn'):
        for e in ('vits', 'vitl'):
            X = J(SUPP / f'moderr_{ds}_{e}.json')
            for i, (c, lab) in enumerate([('int2-g16', 'INT2 sym.'), ('int2-g64-asym', 'INT2 asym.'),
                                          (TCC1, 'TCC-1'), (TCC2, 'TCC-2')]):
                me = X[c]['module_rel_err_pct']
                L.append(f"{ds.capitalize() if (e == 'vits' and i == 0) else ''} & {ENC[e] if i == 0 else ''} & {lab} & "
                         + ' & '.join(f'{me[k]:.2f}' for k in sorted(me)) + f" & {X[c]['fid_rel_dev_pct']:.2f} \\\\")
        if ds == 'sintel':
            L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'tab_modules.tex').write_text('\n'.join(L), encoding='utf-8')


def redundancy_table():
    d = json.load(open(SUPP / 'redundancy.json'))
    L = [r'\begin{tabular}{llrrrrrrrr}', r'\toprule',
         r'Data & Enc. & \multicolumn{8}{c}{layer (two per temporal module)} \\', r'\cmidrule(lr){3-10}',
         r' & & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 \\', r'\midrule']
    for ds in ('sintel', 'bonn', 'scannet'):
        for e in ('vits', 'vitl'):
            a = d[ds][e]['1']
            name = {'sintel': 'Sintel', 'bonn': 'Bonn', 'scannet': 'ScanNet'}[ds]
            L.append(f"{name if e == 'vits' else ''} & {ENC[e]} & " + ' & '.join(f'{x:.1f}' for x in a['temporal_pooled'])
                     + r' \\')
            L.append(r' & \quad cross-layer & ' + ' & '.join('' if x is None else f'{x:.1f}' for x in a['cross_layer_mean'])
                     + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'tab_redundancy.tex').write_text('\n'.join(L), encoding='utf-8')


def timing_table():
    d = json.load(open(SUPP / 'timing_k.json'))
    g = defaultdict(list)
    for r in (d['results'] if isinstance(d, dict) else d):
        g[(r['encoder'], r['config'])].append(r)
    cfgs = ['fp32', 'int2-g64-asym'] + [f'int4-g16::d{k}:bin-g16' for k in (2, 4, 8, 16, 32)]
    labs = ['FP32', 'INT2 asym.'] + [f'TCC-1, $K{{=}}{k}$' for k in (2, 4, 8, 16, 32)]
    L = [r'\begin{tabular}{lrrrr}', r'\toprule',
         r' & \multicolumn{2}{c}{ViT-S} & \multicolumn{2}{c}{ViT-L} \\', r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
         r'Cache & ms & MB & ms & MB \\', r'\midrule']
    for c, lab in zip(cfgs, labs):
        cells = []
        for e in ('vits', 'vitl'):
            rs = g[(e, c)]
            cells += [f"{st.median(r['ms_median'] for r in rs):.1f}", f"{rs[0]['cache_mb']:.1f}"]
        L.append(lab + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'tab_timing_k.tex').write_text('\n'.join(L), encoding='utf-8')


def full_compact(ds):
    """Table 1 formats on Bonn or Sintel with flicker and per-frame-aligned delta1 / AbsRel (paper runs)."""
    L = [r'\begin{tabular}{l' + 'rrrrr' * 3 + '}', r'\toprule',
         ' & ' + ' & '.join(rf'\multicolumn{{5}}{{c}}{{{ENC[e]}}}' for e in ENC) + r' \\',
         ''.join(rf'\cmidrule(lr){{{2 + 5 * i}-{6 + 5 * i}}}' for i in range(3)),
         'Cache' + r' & bits & Dev. & Flk. & $\delta_1$ & Rel' * 3 + r' \\', r'\midrule']
    rs = {e: m.pair_rows(ds, e) for e in ENC}
    for cfg, lab in [('fp32', 'FP32 cache')] + [(c, l) for c, l, _ in m.FORMATS]:
        cells = []
        for e in ENC:
            r, b = m.row(rs[e], cfg), m.row(rs[e], 'fp32')
            fid = cfg != 'fp32'
            cells += [f(32 * r['cache_mb'] / b['cache_mb']), f(r['fid_rel_dev_pct']) if fid else '--',
                      f(r['fid_flicker_pct']) if fid else '--', f(r['delta1'], 3), f(r['abs_rel'], 3)]
        L.append(lab + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / f'tab_full_{ds}.tex').write_text('\n'.join(L), encoding='utf-8')


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False,
                         'axes.spines.right': False})
    col = {'INT2 asym., g=64': '#888888', 'TCC-1, K=8': '#d62728', 'TCC-1 + adaptive keyframe': '#ff7f0e',
           'TCC-2, K=8': '#1f77b4'}
    # cut: mean deviation per frame around the cut
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True)
    for ax, e in zip(axs, ('vits', 'vitl')):
        X = J(SUPP / f'splice_scannet_{e}.json')
        for c, lab in (('int2-g64-asym', 'INT2 asym., g=64'), (TCC1, 'TCC-1, K=8'),
                       (TCC1 + ':a0.5', 'TCC-1 + adaptive keyframe'), (TCC2, 'TCC-2, K=8')):
            pf = np.array([v['dev_per_frame'] for v in X[c]['per_scene'].values()])
            x = np.arange(70, 111)
            ax.plot(x - 85, pf[:, 70:111].mean(0), label=lab, color=col[lab], lw=1.2)
        ax.axvline(0, color='k', lw=0.6, ls=':')
        ax.set_title(f'ScanNet cuts, {ENC[e]}')
        ax.set_xlabel('frame relative to the cut')
    axs[0].set_ylabel('deviation (%)')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / 'fig_cut.pdf')
    plt.close(fig)
    # error through the head
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True)
    for ax, ds in zip(axs, ('sintel', 'bonn')):
        X = J(SUPP / f'moderr_{ds}_vits.json')
        for c, lab in (('int2-g16', 'INT2 symmetric'), ('int2-g64-asym', 'INT2 asym., g=64'),
                       (TCC1, 'TCC-1, K=8'), (TCC2, 'TCC-2, K=8')):
            me = X[c]['module_rel_err_pct']
            ys = [me[k] for k in sorted(me)] + [X[c]['fid_rel_dev_pct']]
            ax.plot(range(5), ys, marker='o', ms=3, lw=1.2, label=lab, color=col.get(lab, '#2ca02c'))
        ax.set_xticks(range(5), ['M1', 'M2', 'M3', 'M4', 'depth'])
        ax.set_title(f'{ds.capitalize()}, ViT-S')
    axs[0].set_ylabel('relative error (%)')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / 'fig_modules.pdf')
    plt.close(fig)
    # redundancy per layer
    d = json.load(open(SUPP / 'redundancy.json'))
    fig, ax = plt.subplots(figsize=(6.8, 2.1))
    for ds, c in (('sintel', '#d62728'), ('bonn', '#1f77b4'), ('scannet', '#2ca02c')):
        a = d[ds]['vits']['1']
        ax.plot(range(1, 9), a['temporal_pooled'], marker='o', ms=3, color=c, lw=1.2,
                label={'sintel': 'Sintel', 'bonn': 'Bonn', 'scannet': 'ScanNet'}[ds] + ': previous frame')
        xs = [l + 1 for l, v in enumerate(a['cross_layer_mean']) if v is not None]
        ax.scatter(xs, [v for v in a['cross_layer_mean'] if v is not None], marker='x', color=c, s=18,
                   label='cross-layer (in-sample)')
    ax.set_xlabel('cached layer (two per temporal module)')
    ax.set_ylabel('% of entry variance left')
    ax.set_xticks(range(1, 9))
    ax.legend(frameon=False, fontsize=6.5, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_redundancy.pdf')
    plt.close(fig)


SVG_FMT = [('none', 'Uncompressed (16-bit)'), ('plain:int4', 'INT4'), ('plain:a2g32', 'INT2 asym., $g{=}32$'),
           ('kivi:2:64:w4', 'KIVI + 4 recent frames'), ('delta:int4:a2g64:8', '\\textbf{TCC-2}, $K{=}8$'),
           ('plain:int2', 'INT2 symmetric'), ('plain:a2g64', 'INT2 asym., $g{=}64$'), ('kivi:2:64', 'KIVI'),
           ('delta:int4:bin:4', '\\textbf{TCC-1}, $K{=}4$'), ('delta:int4:bin:8', '\\textbf{TCC-1}, $K{=}8$'),
           ('xstream:2048:4:4', 'XStreamVGGT, 2048 tokens'), ('zero', 'Zero cache')]


def svg_table(ds):
    d = {r['kv']: r for r in json.load(open(R / f'fidelity_{ds}.json'))['results']}
    L = [r'\begin{tabular}{lrrrrrr}', r'\toprule',
         r'Cache & bits & Cache (MB) & Peak (MB) & s/frame & Dev. & Flk. \\', r'\midrule']
    for kv, lab in SVG_FMT:
        if kv not in d:
            continue
        r = d[kv]
        ref = kv == 'none'
        bits_s = '16' if ref else f(r.get('ideal_bits'))
        cache_s = '--' if ref else f(r.get('cache_mb_max'), 0)
        L.append(f"{lab} & {bits_s} & {cache_s} & {f(r.get('peak_mb_max'), 0)} & "
                 f"{f(r.get('seconds_per_frame'))} & {'--' if ref else f(r.get('fid_rel_dev_pct'))} & "
                 f"{'--' if ref else f(r.get('fid_flicker_pct'))} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / f'tab_svg_{ds}.tex').write_text('\n'.join(L), encoding='utf-8')


def boot_table():
    b = json.load(open(R / 'paper_bootstrap.json'))
    L = [r'\begin{tabular}{llllrrr}', r'\toprule',
         r'Model & Data & Enc. & Comparison ($A$ vs.\ $B$) & Dev.\ ratio $B/A$ & 95\% interval & $A$ wins \\',
         r'\midrule']
    for r in b['results']:
        if 'drop2' in r['b'] and not r['enc']:      # StreamVGGT frame-dropping row: not a comparison the paper makes
            continue
        dv = r['dev']
        L.append(f"{r['model']} & {r['data'].capitalize()} & {ENC.get(r['enc'], r['enc'])} & {r['a']} vs.\\ {r['b']} & "
                 f"{dv['ratio']:.2f} & [{dv['ci95'][0]:.2f}, {dv['ci95'][1]:.2f}] & {dv['wins_a']}/{r['n_scenes']} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'tab_boot.tex').write_text('\n'.join(L).replace('_', '\\_').replace('\\\\_', '\\_'), encoding='utf-8')
    return len(b['results']), b['n_boot']


def open_table():
    base = R / '2026-11-02_streaming' / 'streaming_vda'
    L = [r'\begin{tabular}{llrrrrrr}', r'\toprule',
         r' & & \multicolumn{3}{c}{TCC-1, $K{=}8$} & \multicolumn{3}{c}{TCC-2, $K{=}8$} \\',
         r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
         r'Data & Enc. & closed & open & ratio & closed & open & ratio \\', r'\midrule']
    for ds in ('sintel', 'bonn'):
        for e in ENC:
            X = {r['config']: r for r in json.load(open(base / f'{ds}_{e}_open.json'))['results']}
            cells = []
            for c in (TCC1, TCC2):
                a, o = X[c]['fid_rel_dev_pct'], X[c + ':open']['fid_rel_dev_pct']
                cells += [f'{a:.2f}', f'{o:.2f}', f'{o / a:.1f}$\\times$']
            L.append(f"{ds.capitalize() if e == 'vits' else ''} & {ENC[e]} & " + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'tab_open.tex').write_text('\n'.join(L), encoding='utf-8')


def bonn_scenes():
    """Per-scene deviation on Bonn (paper Sec. 4 promises per-scene results instead of intervals)."""
    L = [r'\begin{tabular}{llrrrrr}', r'\toprule',
         r'Enc. & Cache & ' + ' & '.join(m.BONN3).replace('_', r'\_') + r' & mean \\', r'\midrule']
    for e in ENC:
        rs = m.pair_rows('bonn', e)
        for i, (c, lab) in enumerate([('int2-g64-asym', 'INT2 asym., $g{=}64$'), (TCC1, 'TCC-1, $K{=}8$'),
                                      (TCC1K4, 'TCC-1, $K{=}4$'), ('int2-g32-asym', 'INT2 asym., $g{=}32$'),
                                      (TCC2, 'TCC-2, $K{=}8$')]):
            ps = m.row(rs, c)['per_scene']
            v = [ps[s]['fid_rel_dev'] for s in m.BONN3]
            L.append(f"{ENC[e] if i == 0 else ''} & {lab} & " + ' & '.join(f'{x:.2f}' for x in v)
                     + f' & {np.mean(v):.2f} \\\\')
        L.append(r'\midrule')
    L[-1] = r'\bottomrule'
    L.append(r'\end{tabular}')
    (OUT / 'tab_bonn_vda_scenes.tex').write_text('\n'.join(L), encoding='utf-8')
    d = {r['kv']: r for r in json.load(open(R / 'fidelity_bonn.json'))['results']}
    sc = sorted(d['delta:int4:bin:8']['per_sequence'])
    L = [r'\begin{tabular}{l' + 'r' * (len(sc) + 1) + '}', r'\toprule',
         'Cache & ' + ' & '.join(s.replace('_', r'\_') for s in sc) + r' & mean \\', r'\midrule']
    for kv, lab in [('plain:a2g64', 'INT2 asym., $g{=}64$'), ('kivi:2:64', 'KIVI'), ('delta:int4:bin:8', 'TCC-1, $K{=}8$'),
                    ('plain:a2g32', 'INT2 asym., $g{=}32$'), ('delta:int4:a2g64:8', 'TCC-2, $K{=}8$')]:
        v = [d[kv]['per_sequence'][s]['fid_rel_dev_pct'] for s in sc]
        L.append(f'{lab} & ' + ' & '.join(f'{x:.2f}' for x in v) + f' & {np.mean(v):.2f} \\\\')
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'tab_bonn_svg_scenes.tex').write_text('\n'.join(L), encoding='utf-8')


def xs_released():
    """XStreamVGGT run with the authors' released code: summary lines of results/xstream_authors.log
    (server svg_queue/xstream_authors.log), e.g. 'bonn_2048  0.00  4.853  1.149  5/5  0'."""
    import re
    out = {}
    for line in open(R / 'xstream_authors.log', encoding='utf8', errors='replace'):
        mt = re.match(r'^(bonn|sintel)_(\d+)\s+\S+\s+([\d.]+)\s+([\d.]+)\s+(\d+)/(\d+)', line)
        if mt:
            assert mt.group(5) == mt.group(6), line          # every sequence evaluated
            out[(mt.group(1), int(mt.group(2)))] = (float(mt.group(3)), float(mt.group(4)))
    assert len(out) == 4, out
    return dict(sorted(out.items()))


def xstream_table():
    ours = {ds: {r['kv']: r for r in json.load(open(R / '2026-11-12_q5_stac' / 'svg_queue5' / f'fid_xs_{ds}.json'))['results']}
            for ds in ('sintel', 'bonn')}
    L = [r'\begin{tabular}{lrrrrr}', r'\toprule',
         r' & & \multicolumn{2}{c}{Released code} & \multicolumn{2}{c}{Our implementation} \\',
         r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}', r'Data & Tokens & Dev. & Flk. & Dev. & Flk. \\', r'\midrule']
    for (ds, tok), (dv, fl) in xs_released().items():
        o = ours[ds][f'xstream:{tok}:4:4']
        L.append(f"{ds.capitalize()} & {tok} & {dv:.2f} & {fl:.2f} & {o['fid_rel_dev_pct']:.2f} & {o['fid_flicker_pct']:.2f} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'tab_xstream.tex').write_text('\n'.join(L), encoding='utf-8')


RECON_M = [('plain_a2g64', 'INT2 asym., $g{=}64$'), ('kivi_2_64', 'KIVI'), ('delta_int4_bin_8', 'TCC-1, $K{=}8$'),
           ('delta_int4_a2g64_8', 'TCC-2, $K{=}8$')]


def recon_table():
    rf = json.load(open(R / '2026-11-11_timing' / 'recon_fidelity.json'))
    keys = ['dense/7scenes', 'dense/NRGBD', 'sparse/7scenes', 'sparse/NRGBD']
    L = [r'\begin{tabular}{l' + 'rr' * 4 + '}', r'\toprule',
         ' & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{{k.split("/")[0]}, {k.split("/")[1]}}}' for k in keys) + r' \\',
         ''.join(rf'\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(4)),
         'Cache' + ' & mean & median' * 4 + r' \\', r'\midrule']
    for mt, lab in RECON_M:
        L.append(lab + ' & ' + ' & '.join(f"{rf[k]['methods'][mt]['mean']:.2f} & {rf[k]['methods'][mt]['median']:.2f}"
                                          for k in keys) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'tab_recon.tex').write_text('\n'.join(L), encoding='utf-8')


def long_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base = R / '2026-11-12_q5_stac' / 'svg_queue5' / 'longn'
    fig, ax = plt.subplots(figsize=(6.8, 2.2))
    for fn, lab, c in (('plain_a2g64', 'INT2 asym., g=64', '#888888'), ('kivi_2_64', 'KIVI', '#2ca02c'),
                       ('delta_int4_bin_8', 'TCC-1, K=8', '#d62728'), ('delta_int4_a2g64_8', 'TCC-2, K=8', '#1f77b4')):
        d = json.load(open(base / f'{fn}.json'))
        s = d['scenes']['crowd2']
        y = s['series']['dev']
        x = np.linspace(0, s['frames'], len(y))
        ax.plot(x, y, lw=1.1, label=lab, color=c)
    ax.set_xlabel('frame (Bonn crowd2)')
    ax.set_ylabel('deviation (%)')
    ax.legend(frameon=False, fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_long.pdf')
    plt.close(fig)


def qual_figure():
    """RGB, FP32-cache depth, and deviation maps of INT2 asym / TCC-1 / TCC-2 for one frame per dataset
    (VDA-S; predictions saved before the paper deadline in results/2026-11-16_supp_vis)."""
    import glob
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    P = R / '2026-11-16_supp_vis' / 'preds'
    tags = [('fp32_0_None', None), ('int2_g64_asym_0', 'INT2 asym.'), ('int4_g16_8_bin_g16', 'TCC-1'),
            ('int4_g16_8_int2_g64_asym', 'TCC-2')]
    rows = [('sintel', 'Sintel'), ('bonn', 'Bonn'), ('kittiseq', 'KITTI'), ('scannet', 'ScanNet')]
    VMAX = 4.0                                   # one scale for every panel: % of the mean disparity
    data = []
    for ds, dname in rows:
        d = P / f'{ds}_vits'
        scene = sorted(Path(x).name.split('__')[0] for x in glob.glob(str(d / '*__gt.npz')))[0]
        g = np.load(d / f'{scene}__gt.npz')
        k = len(g['idx']) // 2
        pr = {t: np.load(glob.glob(str(d / f'{scene}__cfg_{t}*.npz'))[0])['pred'][k].astype(np.float32) for t, _ in tags}
        ref = pr['fp32_0_None']
        errs = [(lab, 100 * np.abs(pr[t] - ref) / (np.abs(ref).mean() + 1e-8)) for t, lab in tags if lab]
        data.append((dname, g['rgb'][k], ref, errs))
    hr = [r[2].shape[0] / r[2].shape[1] for r in data]
    W = 7.0
    fig = plt.figure(figsize=(W, W / 5.25 * sum(hr) + 0.55))
    gs = fig.add_gridspec(len(rows), 6, height_ratios=hr, width_ratios=[1, 1, 1, 1, 1, 0.05],
                          hspace=0.06, wspace=0.03, left=0.04, right=0.95, top=0.95, bottom=0.01)
    im = None
    for r, (dname, rgb, ref, errs) in enumerate(data):
        ax = fig.add_subplot(gs[r, 0]); ax.imshow(rgb, aspect='auto'); ax.set_ylabel(dname, fontsize=8)
        ax2 = fig.add_subplot(gs[r, 1]); ax2.imshow(ref, cmap='inferno', aspect='auto')
        axs_r = [ax, ax2]
        for c, (lab, e) in enumerate(errs):
            a3 = fig.add_subplot(gs[r, 2 + c])
            im = a3.imshow(e, cmap='magma', vmin=0, vmax=VMAX, aspect='auto')
            a3.text(0.03, 0.06, f'mean {e.mean():.2f}%', color='w', fontsize=6, transform=a3.transAxes)
            axs_r.append(a3)
        for c, a4 in enumerate(axs_r):
            a4.set_xticks([]); a4.set_yticks([])
            if r == 0:
                a4.set_title(['RGB', 'FP32 cache (disparity)', 'INT2 asym., 2.5 b', 'TCC-1, ~2 b', 'TCC-2, ~2.9 b'][c],
                             fontsize=7.5)
    cax = fig.add_subplot(gs[:, 5])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('difference from FP32 cache (% of mean disparity)', fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(OUT / 'fig_qual.pdf', dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    qual_figure()
    bonn_scenes()
    xstream_table()
    recon_table()
    long_figure()
    for ds in ('sintel', 'bonn', 'kitti'):
        svg_table(ds)
    print('bootstrap comparisons / resamples:', boot_table())
    open_table()
    full_compact('bonn')
    full_compact('sintel')
    figures()
    dataset_table('bonn5', lambda e: m.seq_rows('bonn5', e))
    dataset_table('kitti', lambda e: m.seq_rows('kittiseq', e))
    dataset_table('scannet', lambda e: J(SCN / f'scannet_{e}.json'))
    splice_table()
    module_table()
    redundancy_table()
    timing_table()
    print('tables in', OUT)


if __name__ == '__main__':
    main()
