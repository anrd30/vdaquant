"""
Paired bootstrap over scenes for every comparison the paper makes (CPU only).

For a comparison "A vs B" on one dataset/backbone: per-scene deviations (and flicker)
of A and B, the ratio of their means (B/A, > 1 means A is closer to the uncompressed
model), a 95% percentile interval from 20,000 paired resamples of scenes (seed 0),
the fraction of resamples where A is better, the number of scenes A wins, and the
standard deviation of the per-scene ratio.  Scenes, not frames, are resampled, since
frames of one video are correlated.  Means are plain means over scenes, as in the paper's tables.

    python -X utf8 scripts/paper_bootstrap.py --out results/paper_bootstrap.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
N_BOOT, SEED = 20000, 0


def vda_rows(ds, enc):
    import make_paper_assets as m
    rows = {}
    for r in m.pair_rows(ds, enc):
        rows.setdefault(r.get('config', r.get('mode')), r)
    for extra in (f'outputs/stream/{ds}_{enc}_drop.json', f'outputs/stream/{ds}_{enc}_open.json'):
        p = os.path.join(ROOT, extra)
        if os.path.exists(p):
            for r in json.load(open(p))['results']:
                rows.setdefault(r.get('config', r.get('mode')), r)
    return {c: {s: (v['fid_rel_dev'], v['fid_flicker'], v['n']) for s, v in r['per_scene'].items()
                if v.get('fid_rel_dev') is not None}
            for c, r in rows.items() if r.get('per_scene')}


def svg_rows(ds):
    d = json.load(open(os.path.join(ROOT, f'results/fidelity_{ds}.json')))
    return {r['kv']: {s: (v['fid_rel_dev_pct'], v['fid_flicker_pct'], v['frames'])
                      for s, v in r['per_sequence'].items()} for r in d['results']}


def compare(a, b):
    scenes = sorted(set(a) & set(b))
    A = np.array([a[s][:2] for s in scenes], float)       # (n, 2): dev, flicker
    B = np.array([b[s][:2] for s in scenes], float)
    w = np.ones(len(scenes))                               # plain mean over scenes, as in the tables
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(scenes), size=(N_BOOT, len(scenes)))
    out = {'n_scenes': len(scenes)}
    for k, name in enumerate(('dev', 'flicker')):
        ma, mb = np.average(A[:, k], weights=w), np.average(B[:, k], weights=w)
        wa, wb = (A[idx, k] * w[idx]).sum(1), (B[idx, k] * w[idx]).sum(1)
        ratios = wb / np.maximum(wa, 1e-12)
        per = B[:, k] / np.maximum(A[:, k], 1e-12)
        out[name] = {'mean_a': ma, 'mean_b': mb, 'ratio': mb / ma,
                     'ci95': [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
                     'p_a_better': float((ratios > 1).mean()), 'wins_a': int((A[:, k] < B[:, k]).sum()),
                     'scene_ratio_sd': float(per.std(ddof=1)) if len(per) > 1 else None}
    return out


VDA = [('TCC-1 K8', 'int4-g16::d8:bin-g16', 'INT2 g64', 'int2-g64-asym'),
       ('TCC-1 K4', 'int4-g16::d4:bin-g16', 'INT2 g64', 'int2-g64-asym'),
       ('TCC-2', 'int4-g16::d8:int2-g64-asym', 'INT2 g32', 'int2-g32-asym'),
       ('TCC-1 K4', 'int4-g16::d4:bin-g16', 'INT4 drop2', 'int4-g16@drop2'),
       ('TCC-1 closed', 'int4-g16::d8:bin-g16', 'TCC-1 open', 'int4-g16::d8:bin-g16:open'),
       ('TCC-2 closed', 'int4-g16::d8:int2-g64-asym', 'TCC-2 open', 'int4-g16::d8:int2-g64-asym:open')]
SVG = [('TCC-1', 'delta:int4:bin:8', 'INT2 g64', 'plain:a2g64'),
       ('TCC-1', 'delta:int4:bin:8', 'KIVI', 'kivi:2:64'),
       ('TCC-1', 'delta:int4:bin:8', 'XStreamVGGT 12000 (equal memory)', 'xstream:12000:4:4'),
       ('TCC-1', 'delta:int4:bin:8', 'INT4 drop2', 'drop:2:int4'),
       ('TCC-1', 'delta:int4:bin:8', 'XStreamVGGT 2048', 'xstream:2048:4:4'),
       ('TCC-2', 'delta:int4:a2g64:8', 'INT2 g64', 'plain:a2g64'),
       ('TCC-2', 'delta:int4:a2g64:8', 'KIVI', 'kivi:2:64'),
       ('TCC-2', 'delta:int4:a2g64:8', 'INT2 g32', 'plain:a2g32'),
       ('TCC-2', 'delta:int4:a2g64:8', 'KIVI + 16-bit window', 'kivi:2:64:w4'),
       ('TCC-2', 'delta:int4:a2g64:8', 'INT3', 'plain:int3'),
       ('TCC-1 closed', 'delta:int4:bin:8', 'TCC-1 open', 'delta:int4:bin:8:o'),
       ('TCC-2 closed', 'delta:int4:a2g64:8', 'TCC-2 open', 'delta:int4:a2g64:8:o')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    os.chdir(ROOT)
    res = []
    for ds in ('sintel', 'bonn', 'kitti'):
        if ds == 'kitti':
            rows = svg_rows(ds)
            for la, a, lb, b in SVG:
                if a in rows and b in rows:
                    res.append({'model': 'StreamVGGT', 'data': ds, 'enc': '', 'a': la, 'b': lb, **compare(rows[a], rows[b])})
            continue
        for enc in ('vits', 'vitb', 'vitl'):
            rows = vda_rows(ds, enc)
            for la, a, lb, b in VDA:
                if a in rows and b in rows:
                    res.append({'model': 'VDA', 'data': ds, 'enc': enc, 'a': la, 'b': lb, **compare(rows[a], rows[b])})
        rows = svg_rows(ds)
        for la, a, lb, b in SVG:
            if a in rows and b in rows:
                res.append({'model': 'StreamVGGT', 'data': ds, 'enc': '', 'a': la, 'b': lb, **compare(rows[a], rows[b])})
    for r in res:
        d, f = r['dev'], r['flicker']
        print(f"{r['model']:10s} {r['data']:6s} {r['enc']:4s} {r['a']:12s} vs {r['b']:32s} n={r['n_scenes']:2d} "
              f"dev {d['mean_a']:.2f} vs {d['mean_b']:.2f} ratio {d['ratio']:.2f} [{d['ci95'][0]:.2f},{d['ci95'][1]:.2f}] "
              f"wins {d['wins_a']}/{r['n_scenes']} | flick ratio {f['ratio']:.2f} [{f['ci95'][0]:.2f},{f['ci95'][1]:.2f}]")
    if args.out:
        json.dump({'n_boot': N_BOOT, 'seed': SEED, 'results': res}, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
