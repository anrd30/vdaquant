"""
Generate every results table (LaTeX) and figure (PDF) of the CVPR paper directly
from the result JSONs, so no number is typed by hand.

    python scripts/make_paper_assets.py --paper C:/Users/aniru/Downloads/cvpr2027/paper

Writes <paper>/tables/*.tex and <paper>/figures/*.pdf, plus <paper>/tables/numbers.json
(every number referenced in the text, for scripts/check_claims.py).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
S = REPO / 'outputs' / 'stream'
OURS = 'int4-g16::d8:bin-g16'          # TCC-1, K=8
OURS4 = 'int4-g16::d4:bin-g16'         # TCC-1, K=4
TCC2 = 'int4-g16::d8:int2-g64-asym'    # TCC-2: 4-level residuals
G64, G32 = 'int2-g64-asym', 'int2-g32-asym'
NUM = {}
KEYF = {('bonn', 'vits'): 'bonn_vits_fidelity', ('bonn', 'vitb'): 'bonn_vitb_key', ('bonn', 'vitl'): 'bonn_vitl_key',
        ('sintel', 'vits'): 'sintel_vits_fidelity', ('sintel', 'vitb'): 'sintel_vitb_fidelity',
        ('sintel', 'vitl'): 'sintel_vitl_fidelity'}
ENCS = ('vits', 'vitb', 'vitl')


def load(name):
    p = S / f'{name}.json'
    return json.load(open(p))['results'] if p.exists() else None


def row(rs, cfg):
    for r in rs or []:
        if r.get('config', r.get('mode')) == cfg:
            return r
    return None


def pair_rows(ds, enc):
    """All configs of one model/dataset pair (fp32 is identical in these files)."""
    rs = []
    for f in (KEYF[ds, enc], f'{ds}_{enc}_int2asym', f'{ds}_{enc}_match'):
        rs += load(f) or []
    return rs


def rel(r, b, k):
    return 100 * (r[k] / b[k] - 1)


def bits(r, b):
    return 32 * r['cache_mb'] / b['cache_mb']


def fmt_pct(x, d=1):
    return f'{x:+.{d}f}' if x is not None else '--'


# ---------------------------------------------------------------- main table
FORMATS = [('zero', 'Zero cache', None), ('int2-g16', 'INT2 symmetric', None),
           (G64, 'INT2 asym., $g{=}64$', 1), (OURS, '\\textbf{TCC-1}, $K{=}8$', 1),
           (OURS4, '\\textbf{TCC-1}, $K{=}4$', 1), (G32, 'INT2 asym., $g{=}32$', 2),
           (TCC2, '\\textbf{TCC-2}, $K{=}8$', 2)]


def main_table(out):
    """One row per cache format; columns Bonn S/B/L and Sintel S/B/L."""
    data = {(ds, e): pair_rows(ds, e) for ds in ('bonn', 'sintel') for e in ENCS}
    order = [(ds, e) for ds in ('bonn', 'sintel') for e in ENCS]

    def val(key, cfg, field):
        rs = data[key]
        r, b = row(rs, cfg), row(rs, 'fp32')
        if field in ('fid_rel_dev_pct', 'fid_flicker_pct'):
            return r[field]
        return rel(r, b, field)

    def cells(cfg, grp, field, fmt):
        out_ = []
        for k in order:
            v = val(k, cfg, field)
            txt = fmt(v)
            if grp is not None and field.startswith('fid'):
                best = min(val(k, c, field) for c, _, g in FORMATS if g == grp)
                if v == best:
                    txt = f'\\textbf{{{txt}}}'
            out_.append(txt)
        return out_

    def bits_of(cfg):
        v = [bits(row(data[k], cfg), row(data[k], 'fp32')) for k in (('bonn', 'vits'), ('sintel', 'vits'))]
        return f'{v[0]:.2f}' if abs(v[0] - v[1]) < 0.005 else f'{v[0]:.2f}/{v[1]:.2f}'

    head = ' & '.join(['S', 'B', 'L'] * 4)
    L = [r'\begin{tabular}{lc' + 'rrr' * 4 + '}', r'\toprule',
         r' & & \multicolumn{6}{c}{Deviation from FP32 output (\%)} & \multicolumn{6}{c}{Flicker deviation (\%)} \\',
         r'\cmidrule(lr){3-8}\cmidrule(lr){9-14}',
         r' & & \multicolumn{3}{c}{Bonn} & \multicolumn{3}{c}{Sintel} & \multicolumn{3}{c}{Bonn} & \multicolumn{3}{c}{Sintel} \\',
         f'Cache format & bits & {head} \\\\', r'\midrule']
    f2 = lambda v: f'{v:.2f}'
    for i, (cfg, lab, g) in enumerate(FORMATS):
        if i in (2, 5):
            L.append(r'\midrule')
        L.append(f'{lab} & {bits_of(cfg)} & ' + ' & '.join(cells(cfg, g, 'fid_rel_dev_pct', f2)
                                                           + cells(cfg, g, 'fid_flicker_pct', f2)) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    (out / 'tab_main.tex').write_text('\n'.join(L), encoding='utf-8')


def full_table(out):
    """Every format on every pair with cache size, delta1 and AbsRel (supplement)."""
    L = [r'\begin{tabular}{llrrrrrr}', r'\toprule',
         r'Enc. & Data & Cache & bits & Fid. & Flick. & $\delta_1$ & AbsRel \\',
         r' & & (MB) & & \% & \% & & \\', r'\midrule']
    for enc in ENCS:
        for ds in ('bonn', 'sintel'):
            rs = pair_rows(ds, enc)
            b = row(rs, 'fp32')
            L.append(f"ViT-{enc[-1].upper()} & {ds.capitalize()} & {b['cache_mb']:.0f} & 32 & "
                     f"-- & -- & {b['delta1']:.4f} & {b['abs_rel']:.4f} \\\\")
            for cfg, lab, _ in FORMATS:
                r = row(rs, cfg)
                L.append(f" & \\quad {lab} & {r['cache_mb']:.0f} & {bits(r, b):.2f} & "
                         f"{r['fid_rel_dev_pct']:.2f} & {r['fid_flicker_pct']:.2f} & {r['delta1']:.4f} & {r['abs_rel']:.4f} \\\\")
            L.append(r'\midrule')
    L[-1] = r'\bottomrule'
    L.append(r'\end{tabular}')
    (out / 'tab_full.tex').write_text('\n'.join(L), encoding='utf-8')


# ---------------------------------------------------------------- memory and speed
def memory_table(out):
    L = [r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
         r' & \multicolumn{2}{c}{FP32} & \multicolumn{3}{c}{TCC-1} & \multicolumn{3}{c}{TCC-2} \\',
         r'\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-9}',
         r'Enc. & cache & peak & cache & ratio & peak & cache & ratio & peak \\', r'\midrule']
    for e in ENCS:
        rs = pair_rows('bonn', e)
        b, o, t = row(rs, 'fp32'), row(rs, OURS), row(rs, TCC2)
        cut = lambda r: 100 * (1 - r['peak_mb'] / b['peak_mb'])
        L.append(f"ViT-{e[-1].upper()} & {b['cache_mb']:.0f} & {b['peak_mb']:.0f} & {o['cache_mb']:.0f} & "
                 f"{b['cache_mb'] / o['cache_mb']:.1f}$\\times$ & {o['peak_mb']:.0f} ($-${cut(o):.0f}\\%) & "
                 f"{t['cache_mb']:.0f} & {b['cache_mb'] / t['cache_mb']:.1f}$\\times$ & "
                 f"{t['peak_mb']:.0f} ($-${cut(t):.0f}\\%) \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    (out / 'tab_memory.tex').write_text('\n'.join(L), encoding='utf-8')


def speed_table(out):
    """A100: per run the median ms/frame over 40 frames; reported: median over the 5 interleaved runs."""
    import statistics
    rs = json.load(open(REPO / 'results' / 'vda_timing_a100.json'))['results']

    def stat(enc, cfg, key):
        v = [r[key] for r in rs if r['encoder'] == enc and r['config'] == cfg and r.get(key) is not None]
        if not v:
            return None
        return max(v) if key == 'peak_mb' else statistics.median(v)

    items = [('fp32', 'FP32'), (G64, 'INT2 asym.'), (OURS, 'TCC-1, $K{=}8$'), (OURS4, 'TCC-1, $K{=}4$'), (TCC2, 'TCC-2')]
    L = [r'\begin{tabular}{l' + 'rr' * 3 + '}', r'\toprule',
         ' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{ViT-{e[-1].upper()}}}' for e in ENCS) + r' \\',
         r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
         'Cache & ' + ' & '.join(['ms', 'FPS'] * 3) + r' \\', r'\midrule']
    for cfg, lab in items:
        c = []
        for e in ENCS:
            ms, fps = (stat(e, cfg, k) for k in ('ms_median', 'fps'))
            c += [f'{ms:.0f}', f'{fps:.1f}'] if ms else ['--'] * 2
        L.append(f'{lab} & ' + ' & '.join(c) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    (out / 'tab_speed.tex').write_text('\n'.join(L), encoding='utf-8')


# ---------------------------------------------------------------- ablations
ABL = [('int4-g16::d4:bin-g16', '$K{=}4$'), ('int4-g16::d8:bin-g16', '$K{=}8$ (default)'),
       ('int4-g16::d16:bin-g16', '$K{=}16$'), ('int4-g16::d32:bin-g16', '$K{=}32$'),
       ('int4-g16::d2:bin-g16', '$K{=}2$'), ('int4-g16::d64:bin-g16', '$K{=}64$'),
       ('int4-g16::d8:bin-g8-s8', 'residual group $g{=}8$'), ('int4-g16::d8:bin-g32-s8', 'residual group $g{=}32$'),
       ('int4-g16::d8:bin-g64-s8', 'residual group $g{=}64$'),
       ('int8-g16::d8:bin-g16', 'INT8 keyframes'), ('int4-g16::d8:int2-g16', 'ternary residuals'),
       (TCC2, '4-level residuals (TCC-2)'),
       ('int4-g16::d8:bin-g16:none', 'no prediction'), ('int4-g16::d8:bin-g16:open', 'open loop'),
       ('int4-g16::d8:bin-g16:extrap2', 'linear extrapolation')]


def ablation_table(out):
    sets = [(n, (load(f'{d}_vits_ablation') or []) + (load(f'{d}_vits_sweep') or []) + (load(f'{d}_vits_match') or []) + (load(f'{d}_vits_open') or []))
            for n, d in (('Sintel', 'sintel'), ('Bonn', 'bonn'))]
    avail = [(n, rs) for n, rs in sets if rs is not None]
    L = [r'\begin{tabular}{l' + 'rrr' * len(avail) + '}', r'\toprule',
         ' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{n}}}' for n, _ in avail) + r' \\',
         'Variant' + ' & bits & Dev.\\%' * len(avail) + r' \\', r'\midrule']
    for cfg, lab in ABL:
        cells = []
        for n, rs in avail:
            b, r = row(rs, 'fp32'), row(rs, cfg)
            if r is None:
                cells.append('-- & --')
                continue
            bt, cov, fid = bits(r, b), rel(r, b, 'tae_covis_percent'), r['fid_rel_dev_pct']
            tag = cfg.replace('int4-g16::', '').replace(':', '_')
            NUM.update({f'abl.{n.lower()}.{tag}.bits': round(bt, 2), f'abl.{n.lower()}.{tag}.cov': round(cov, 1),
                        f'abl.{n.lower()}.{tag}.fid': round(fid, 2)})
            cells.append(f'{bt:.2f} & {fid:.2f}')
        L.append(f'{lab} & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    (out / 'tab_ablation.tex').write_text('\n'.join(L), encoding='utf-8')


# ---------------------------------------------------------------- baselines (ViT-S)
BASE = [('fp16', 'FP16'), ('int8-g64', 'INT8 (per-group)'), ('int4-g64-asym', 'INT4 asym.\\ (KIVI-style)'),
        ('int4-g16', 'INT4'), ('int3-g16', 'INT3'), ('int3-gT', 'INT3, one scale/tensor'),
        (G32, 'INT2 asym., $g{=}32$'), (G64, 'INT2 asym., $g{=}64$'),
        ('int2-g16', 'INT2 symmetric'), ('int2-g16-rht', 'INT2 sym.\\ + Hadamard'), ('bw16:3', '$\\Lambda_{16}$ lattice, 3-bit'),
        ('bw16:2', '$\\Lambda_{16}$ lattice, 2-bit'), ('bin-g16', '1-bit, no prediction'),
        (TCC2, '\\textbf{TCC-2}'), (OURS, '\\textbf{TCC-1}'), ('zero', 'Zero cache')]


def baseline_table(out):
    def gather(prefix_files):
        rows = []
        for f in prefix_files:
            rows += load(f) or []
        return rows
    bonn = gather(['bonn_vits_baselines', 'bonn_vits_part2', 'bonn_vits_part3', 'bonn_vits_fidelity', 'bonn_vits_delta_mixed',
                   'bonn_vits_int2asym', 'bonn_vits_match'])
    sint = gather(['sintel_vits', 'sintel_vits_delta_mixed', 'sintel_vits_fidelity', 'sintel_vits_int2asym',
                   'sintel_vits_match'])
    L = [r'\begin{tabular}{lrrrrr}', r'\toprule',
         r' & bits & \multicolumn{2}{c}{Bonn} & \multicolumn{2}{c}{Sintel} \\',
         r'Cache format & & Dev.\% & AbsRel & Dev.\% & AbsRel \\', r'\midrule']
    bb, bs = row(bonn, 'fp32'), row(sint, 'fp32')
    L.append(f"FP32 & 32 & -- & {bb['abs_rel']:.4f} & -- & {bs['abs_rel']:.4f} \\\\")
    for cfg, lab in BASE:
        rb, rsn = row(bonn, cfg), row(sint, cfg)
        ref = rb or rsn
        if ref is None:
            continue
        bt = bits(ref, bb if rb else bs)
        dv = lambda r: f"{r['fid_rel_dev_pct']:.2f}" if r and r.get('fid_rel_dev_pct') is not None else '--'
        cb, cs = dv(rb), dv(rsn)
        ab = f"{rb['abs_rel']:.4f}" if rb else '--'
        as_ = f"{rsn['abs_rel']:.4f}" if rsn else '--'
        L.append(f'{lab} & {bt:.2f} & {cb} & {ab} & {cs} & {as_} \\\\')
    L += [r'\bottomrule', r'\end{tabular}']
    (out / 'tab_baselines.tex').write_text('\n'.join(L), encoding='utf-8')


# ---------------------------------------------------------------- age analysis
def age_assets(out, figdir):
    fade_name = 'age:recent=int4-g16+mid=int2-g16+old=bin-g16+oldest=bin-g16+anchor=int4-g16'
    L = [r'\begin{tabular}{lrrrr}', r'\toprule',
         r' & \multicolumn{2}{c}{Sintel} & \multicolumn{2}{c}{Bonn} \\',
         r'Perturbation & cov $\Delta$\% & Fid.\% & cov $\Delta$\% & Fid.\% \\', r'\midrule']
    data = {ds: load(f'{ds}_vits_fading') for ds in ('sintel', 'bonn')}
    items = [('age:recent=zero', 'remove recent (1--9)'), ('age:mid=zero', 'remove middle (10--19)'),
             ('age:old=zero', 'remove old (20--29)'), ('age:recent=bin-g16', '1-bit recent'),
             ('age:old=bin-g16', '1-bit old')]
    for cfg, lab in items:
        cells = []
        for ds in ('sintel', 'bonn'):
            b, r = row(data[ds], 'fp32'), row(data[ds], cfg)
            cov, fid = rel(r, b, 'tae_covis_percent'), r['fid_rel_dev_pct']
            NUM[f'age.{ds}.{cfg[4:]}.cov'] = round(cov, 1)
            cells.append(f'{fmt_pct(cov)} & {fid:.2f}')
        L.append(f'{lab} & ' + ' & '.join(cells) + r' \\')
    L.append(r'\midrule')
    alloc = [(fade_name, 'recent-heavy (ours)')]
    rs0 = data['sintel']
    eq = [r['config'] for r in rs0 if r['config'].startswith('age:s0=')]
    alloc += [(eq[0], 'reversed'), (eq[1], 'shuffled \\#1'), (eq[2], 'shuffled \\#2')]
    for cfg, lab in alloc:
        cells = []
        for ds in ('sintel', 'bonn'):
            b, r = row(data[ds], 'fp32'), row(data[ds], cfg)
            cov, fid = rel(r, b, 'tae_covis_percent'), r['fid_rel_dev_pct']
            NUM[f'age.{ds}.alloc.{lab.split()[0]}.cov'] = round(cov, 1)
            cells.append(f'{fmt_pct(cov)} & {fid:.2f}')
        L.append(f'{lab}, 2.79 b & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    (out / 'tab_age.tex').write_text('\n'.join(L), encoding='utf-8')

    # attention mass by age figure
    paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.65), sharey=True)
    for ax, ds in zip(axes, ('sintel', 'bonn')):
        att = json.load(open(S / f'age_attention_{ds}.json'))['per_layer_slot_mass']
        n = len(att)
        for i, (layer, mass) in enumerate(att.items()):
            ages = list(range(29, 0, -1))               # slots 2..30 -> ages 29..1
            vals = [31 * mass[31 - a] for a in ages]
            ax.plot(ages, vals, lw=0.9, color=plt.cm.viridis(i / max(1, n - 1)), label=f'{layer}')
        ax.axhline(1.0, color='k', lw=0.6, ls='--')
        ax.set_xlabel('age of cached frame')
        ax.set_title(ds.capitalize(), fontsize=8, pad=2)
        ax.invert_xaxis()
        tidy(ax)
    axes[0].set_ylabel('attention / uniform')
    axes[0].legend(title='layer', fontsize=5.5, title_fontsize=6, ncol=2, frameon=False,
                   loc='upper left', handlelength=1.2, columnspacing=0.6, labelspacing=0.2)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    fig.savefig(figdir / 'fig_attention_age.pdf')
    plt.close(fig)


# ---------------------------------------------------------------- bits vs fidelity
def paper_style():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'font.size': 7.5, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42})


def tidy(ax):
    ax.grid(alpha=0.25, lw=0.4)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def pareto_fig(figdir):
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.9))
    for ax, ds in zip(axes, ('sintel', 'bonn')):
        pts = []
        for f in (f'{ds}_vits_fidelity', f'{ds}_vits_ablation', f'{ds}_vits_fading', f'{ds}_vits_int2asym',
                  f'{ds}_vits_match', f'{ds}_vits_sweep'):
            rs = load(f)
            if not rs:
                continue
            b = row(rs, 'fp32')
            for r in rs:
                c = r.get('config', r['mode'])
                if c == 'fp32' or r.get('fid_rel_dev_pct') is None or c == 'zero':
                    continue
                bt = r['plan_bits'] if r.get('plan_bits') is not None else bits(r, b)
                kind = ('no prediction' if c.endswith(':none') else
                        'extrapolation predictor' if c.endswith(':extrap2') else
                        'ours' if '::d' in c else
                        'asym INT' if c.endswith('-asym') else
                        'uniform INT' if c.startswith(('int', 'age:all')) else 'age allocation')
                if c.startswith('age:') and not c.startswith('age:all'):
                    continue                              # age-dependent bit plans: not in the paper
                pts.append((bt, r['fid_rel_dev_pct'], kind))
        style = {'ours': ('o', '#2F6DB5'), 'asym INT': ('P', '#B23A48'), 'uniform INT': ('s', '0.55'),
                 'age allocation': ('^', '#3A9D5D'),
                 'no prediction': ('x', 'k'), 'extrapolation predictor': ('D', '#E08A1E')}
        labels = {'ours': 'TCC (ours)', 'asym INT': 'INT2 asym.', 'uniform INT': 'INT sym.', 'age allocation': 'age-based bits',
                  'no prediction': '1-bit, no prediction', 'extrapolation predictor': 'TCC, extrap. predictor'}
        for kind, (m, col) in style.items():
            xs = [p[0] for p in pts if p[2] == kind]
            ys = [p[1] for p in pts if p[2] == kind]
            if xs:
                ax.scatter(xs, ys, marker=m, c=col, s=14, lw=0.9 if m == 'x' else 0.3,
                           edgecolors='white' if m != 'x' else None, label=labels[kind], zorder=3)
        ax.set_xlabel('bits per cached value')
        ax.set_title(ds.capitalize(), fontsize=8, pad=2)
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(FixedLocator([0.3, 1, 3, 10]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        tidy(ax)
    axes[0].set_ylabel('deviation from FP32 (%)')
    h, l = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    for hh, ll in zip(h2, l2):
        if ll not in l:
            h.append(hh), l.append(ll)
    fig.legend(h, l, fontsize=6, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(pad=0.2, w_pad=0.6, rect=(0, 0, 1, 0.83))
    fig.savefig(figdir / 'fig_pareto.pdf')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper', required=True)
    args = ap.parse_args()
    paper = Path(args.paper)
    out, figdir = paper / 'tables', paper / 'figures'
    out.mkdir(exist_ok=True)
    figdir.mkdir(exist_ok=True)
    main_table(out)
    full_table(out)
    memory_table(out)
    speed_table(out)
    ablation_table(out)
    baseline_table(out)
    age_assets(out, figdir)
    pareto_fig(figdir)
    (out / 'numbers.json').write_text(json.dumps(NUM, indent=1, sort_keys=True), encoding='utf-8')
    print('wrote', sorted(p.name for p in out.iterdir()), 'figures', sorted(p.name for p in figdir.glob('fig_*')))


if __name__ == '__main__':
    main()
