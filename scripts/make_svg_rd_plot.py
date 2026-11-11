"""
Memory vs deviation for StreamVGGT on the official benchmark (Sintel, Bonn).
Per-entry quantizers are shown as labelled points; frame dropping, TCC, and TCC
on every n-th frame are each one method at several settings and are drawn as
lines.  Reads fidelity_<dataset>.json written by streamvggt_fidelity.py.

    python scripts/make_svg_rd_plot.py --res <folder with fidelity_*.json> --out fig_svg_rd.pdf
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

QUANT = {'plain_a2g64': 'INT2', 'kivi_2_64': 'KIVI', 'plain_a3g64': 'INT3', 'plain_int4': 'INT4'}
LINES = [('Frame dropping (INT4)', '#7f7f7f', 's', ['drop_4_int4', 'drop_3_int4', 'drop_2_int4']),
         ('TCC-1 ($K$=16, 8, 4)', '#1f77b4', '^', ['delta_int4_bin_16', 'delta_int4_bin_8', 'delta_int4_bin_4']),
         ('TCC-2 ($K$=8, 4)', '#9467bd', 'D', ['delta_int4_a2g64_8', 'delta_int4_a2g64_4']),
         ('TCC on every 2nd / 3rd frame', '#2ca02c', 'v',
          ['delta_int4_bin_8_s3', 'delta_int4_bin_8_s2', 'delta_int4_bin_4_s2', 'delta_int4_a2g64_8_s2'])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.75))
    for ax, ds, title in zip(axes, ('sintel', 'bonn'), ('Sintel (23 scenes)', 'Bonn (5 scenes)')):
        runs = {r['run'][len(ds) + 1:]: r for r in json.load(open(os.path.join(args.res, f'fidelity_{ds}.json')))['results']}
        full = runs['fp16']['cache_mb_max']
        pt = lambda n: (100 * runs[n]['cache_mb_max'] / full, runs[n]['fid_rel_dev_pct'])
        for label, color, marker, names in LINES:
            xs, ys = zip(*sorted(pt(n) for n in names))
            ax.plot(xs, ys, color=color, marker=marker, ms=4, lw=1.1, label=label, zorder=3)
        qx, qy = zip(*[pt(n) for n in QUANT])
        ax.scatter(qx, qy, color='#d62728', marker='o', s=16, label='Per-entry quantizers', zorder=3)
        for n, name in QUANT.items():
            x, y = pt(n)
            ax.annotate(name, (x, y), xytext=(3, 3), textcoords='offset points', fontsize=6.5, color='#d62728')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(FixedLocator([4, 6, 10, 15, 20, 30]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator([1, 2, 4, 8] if ds == 'sintel' else [0.25, 0.5, 1, 2]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('cache size (% of 16-bit)')
        ax.set_title(title)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel('deviation (%)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.5, frameon=False, loc='upper center', ncol=3)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.84))
    fig.savefig(args.out)
    fig.savefig(os.path.splitext(args.out)[0] + '.png', dpi=150)


if __name__ == '__main__':
    main()
