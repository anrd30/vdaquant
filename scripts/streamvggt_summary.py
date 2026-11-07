"""
Summarize StreamVGGT runs: per-scene wins and paired bootstrap for the main
comparisons, plus bits, cache, peak memory, time per frame and depth metrics.

    python scripts/streamvggt_summary.py --raw <folder with fidelity_*.json> --out summary.json

The bootstrap resamples scenes.  The reported interval is for the mean of
(baseline - ours) in percentage points; above zero means ours is closer.
"""
import argparse
import json
import os

import numpy as np

METRICS = (('fid_rel_dev_pct', 'dev'), ('fid_flicker_pct', 'flicker'))
# (ours, baseline)
PAIRS = {
    'sintel': [('delta_int4_bin_8', 'plain_int2'), ('delta_int4_bin_8', 'plain_a2g64'),
               ('delta_int4_bin_8', 'plain_a2g32'), ('delta_int4_bin_4', 'plain_int2'),
               ('delta_int4_bin_4', 'plain_a2g64'), ('delta_int4_a2g64_8', 'plain_a2g32'),
               ('delta_int4_bin_8', 'xstream_12000_4_4'), ('delta_int4_a2g64_8', 'xstream_12000_4_4'),
               ('delta_int4_bin_8', 'xstream_2048_4_4')],
    'bonn': [('delta_int4_bin_8', 'plain_int2'), ('delta_int4_bin_8', 'plain_a2g64'),
             ('delta_int4_bin_8', 'plain_a2g32'), ('delta_int4_bin_4', 'plain_int2'),
             ('delta_int4_bin_4', 'plain_a2g64'), ('delta_int4_a2g64_8', 'plain_a2g32'),
             ('delta_int4_bin_8', 'xstream_47000_4_4'), ('delta_int4_a2g64_8', 'xstream_47000_4_4'),
             ('delta_int4_bin_8', 'xstream_2048_4_4')],
}


def bootstrap(diff, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    means = diff[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_runs(raw, dataset):
    with open(os.path.join(raw, f'fidelity_{dataset}.json')) as fh:
        results = json.load(fh)['results']
    return {r['run'][len(dataset) + 1:]: r for r in results}


def depth_metrics(raw, dataset, name):
    path = os.path.join(raw, f'{dataset}_{name}', 'result_scale.json')
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        m = json.load(fh)
    return {'abs_rel': m.get('Abs Rel'), 'rmse': m.get('RMSE'), 'delta1': m.get('δ < 1.25')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = {}
    for dataset, pairs in PAIRS.items():
        runs = load_runs(args.raw, dataset)
        table = {}
        print(f'\n=== {dataset}: bits dev flick cacheMB peakMB s/frame AbsRel delta1')
        for name in sorted(runs):
            r = runs[name]
            row = {k: r.get(k) for k in ('ideal_bits', 'fid_rel_dev_pct', 'fid_flicker_pct', 'cache_mb_max',
                                         'peak_mb_max', 'seconds_per_frame', 'n_sequences', 'n_reference')}
            row.update(depth_metrics(args.raw, dataset, name))
            table[name] = row
            print(f"{name:34s} {row['ideal_bits'] or 0:5.2f} {row['fid_rel_dev_pct']:7.3f} {row['fid_flicker_pct']:6.3f} "
                  f"{row['cache_mb_max'] or 0:7.0f} {row['peak_mb_max'] or 0:7.0f} {row['seconds_per_frame'] or 0:6.3f} "
                  f"{row.get('abs_rel') or 0:.4f} {row.get('delta1') or 0:.4f}")
        comps = []
        print(f'--- {dataset}: ours vs baseline: ratio, scene wins, 95% CI of mean(baseline-ours)')
        for ours, base in pairs:
            if ours not in runs or base not in runs:
                print('missing', ours, base)
                continue
            a, b = runs[ours]['per_sequence'], runs[base]['per_sequence']
            seqs = sorted(set(a) & set(b))
            for key, label in METRICS:
                x = np.array([a[s][key] for s in seqs])
                y = np.array([b[s][key] for s in seqs])
                lo, hi = bootstrap(y - x)
                c = {'ours': ours, 'baseline': base, 'metric': label,
                     'ratio': float(y.mean() / x.mean()), 'wins': int((x < y).sum()), 'n': len(seqs),
                     'ci_low': lo, 'ci_high': hi}
                comps.append(c)
                print(f"{ours:20s} vs {base:20s} {label:7s} {c['ratio']:5.2f}x wins {c['wins']:2d}/{c['n']:<2d} "
                      f"CI [{lo:6.3f},{hi:6.3f}] {'excl0' if lo > 0 or hi < 0 else 'incl0'}")
        out[dataset] = {'table': table, 'comparisons': comps}
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2)


if __name__ == '__main__':
    main()
