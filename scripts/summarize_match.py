"""Matched-rate comparison and timing summary (queues 2-4).
usage: python -X utf8 scripts/summarize_match.py"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

O = Path(__file__).resolve().parents[1] / 'outputs' / 'stream'
K8, K4, R4 = 'int4-g16::d8:bin-g16', 'int4-g16::d4:bin-g16', 'int4-g16::d8:int2-g64-asym'
G64, G32 = 'int2-g64-asym', 'int2-g32-asym'
PAIRS = [('bonn', 'vits', 'bonn_vits_fidelity'), ('bonn', 'vitb', 'bonn_vitb_key'), ('bonn', 'vitl', 'bonn_vitl_key'),
         ('sintel', 'vits', 'sintel_vits_fidelity'), ('sintel', 'vitb', 'sintel_vitb_fidelity'),
         ('sintel', 'vitl', 'sintel_vitl_fidelity')]


def rows(name):
    return {r['config']: r for r in json.loads((O / f'{name}.json').read_text())['results']}


def boot(d, n=20000):
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a, b, key):
    """b - a per scene; positive means a is lower (better)."""
    s = sorted(k for k in a['per_scene'] if a['per_scene'][k].get(key) is not None
               and b['per_scene'].get(k, {}).get(key) is not None)
    d = np.array([b['per_scene'][k][key] - a['per_scene'][k][key] for k in s])
    lo, hi = boot(d)
    return f'{d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] {int((d > 0).sum())}/{len(s)}'


def main():
    for ds, enc, key in PAIRS:
        A, I, M = rows(key), rows(f'{ds}_{enc}_int2asym'), rows(f'{ds}_{enc}_match')
        f = A['fp32']
        same = abs(f['delta1'] - M['fp32']['delta1']) < 1e-9 and abs(f['delta1'] - I['fp32']['delta1']) < 1e-9
        print(f'== {ds} {enc}  fp32 consistent across files: {same}')
        for name, r in (('TCC K8', A[K8]), ('TCC K4', M[K4]), ('TCC 4-level res', M[R4]),
                        ('INT2 g64', I[G64]), ('INT2 g32', I[G32])):
            print(f"  {name:16s} bits {32 * r['cache_mb'] / f['cache_mb']:.2f} fid {r['fid_rel_dev_pct']:.2f} "
                  f"flick {r['fid_flicker_pct']:.2f} cov {100 * (r['tae_covis_percent'] / f['tae_covis_percent'] - 1):+.1f} "
                  f"tae {100 * (r['tae_percent'] / f['tae_percent'] - 1):+.1f} dd1 {r['delta1'] - f['delta1']:+.4f} "
                  f"dabs {r['abs_rel'] - f['abs_rel']:+.4f} peak {r['peak_mb']:.0f}")
        for lab, a, b in (('K4 vs g64', M[K4], I[G64]), ('4lev vs g32', M[R4], I[G32])):
            print(f"  {lab:12s} fid {paired(a, b, 'fid_rel_dev')} | flick {paired(a, b, 'fid_flicker')}"
                  f" | cov {paired(a, b, 'tae_covis')}")
    print('== timing')
    for fn in ('timing_kernel', 'timing_reference'):
        g = defaultdict(list)
        for r in json.loads((O / f'{fn}.json').read_text())['results']:
            g[(r['encoder'], r['config'], r['decode'])].append(r)
        for (enc, cfg, dec), rs in g.items():
            print(f"  {enc} {dec:9s} {cfg:24s} ms {[round(r['ms_median'], 1) for r in rs]} "
                  f"fps {statistics.mean(r['fps'] for r in rs):.1f} weights {rs[0]['weights_mb']:.0f} "
                  f"cache {rs[0]['cache_mb']:.0f} peak {rs[0]['peak_mb']:.0f}")
    S = rows('bonn_vits_sweep')
    print('== bonn sweep')
    for c, r in S.items():
        if c != 'fp32':
            b = S['fp32']
            print(f"  {c:28s} bits {32 * r['cache_mb'] / b['cache_mb']:.2f} fid {r['fid_rel_dev_pct']:.2f} "
                  f"flick {r['fid_flicker_pct']:.2f} cov {100 * (r['tae_covis_percent'] / b['tae_covis_percent'] - 1):+.1f}")


if __name__ == '__main__':
    main()
