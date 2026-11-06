"""TCC vs KIVI-style INT2 per scene: paired bootstrap of the fidelity gap.
usage: python -X utf8 scripts/compare_tcc_int2asym.py"""
import json
from pathlib import Path

import numpy as np

O = Path(__file__).resolve().parents[1] / 'outputs' / 'stream'
TCC = 'int4-g16::d8:bin-g16'
PAIRS = [('bonn_vits', 'bonn_vits_fidelity', 'bonn_vits_int2asym'),
         ('sintel_vits', 'sintel_vits_fidelity', 'sintel_vits_int2asym'),
         ('bonn_vitb', 'bonn_vitb_key', 'bonn_vitb_int2asym'),
         ('sintel_vitb', 'sintel_vitb_fidelity', 'sintel_vitb_int2asym'),
         ('bonn_vitl', 'bonn_vitl_key', 'bonn_vitl_int2asym'),
         ('sintel_vitl', 'sintel_vitl_fidelity', 'sintel_vitl_int2asym')]


def rows(name):
    p = O / f'{name}.json'
    if not p.exists():
        return None
    return {r['config']: r for r in json.loads(p.read_text())['results']}


def scene_values(row, key):
    return {k: v[key] for k, v in row['per_scene'].items() if v.get(key) is not None}


def boot(d, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    for name, a, b in PAIRS:
        A, B = rows(a), rows(b)
        if not A or not B or 'int2-g64-asym' not in B:
            print(f'{name:12s} (not available)')
            continue
        for key, lab in (('fid_rel_dev', 'fid'), ('fid_flicker', 'flicker'), ('tae_covis', 'covTAE')):
            t, g = scene_values(A[TCC], key), scene_values(B['int2-g64-asym'], key)
            s = sorted(set(t) & set(g))
            d = np.array([g[k] - t[k] for k in s])       # > 0: TCC better
            lo, hi = boot(d)
            print(f'{name:12s} {lab:8s} n={len(s):2d} TCC {np.mean([t[k] for k in s]):7.3f} '
                  f'INT2asym {np.mean([g[k] for k in s]):7.3f} gap {d.mean():+7.3f} '
                  f'CI [{lo:+.3f}, {hi:+.3f}] TCC better on {int((d > 0).sum())}/{len(s)}')


if __name__ == '__main__':
    main()
