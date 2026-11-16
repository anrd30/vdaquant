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


def main():
    OUT.mkdir(exist_ok=True)
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
