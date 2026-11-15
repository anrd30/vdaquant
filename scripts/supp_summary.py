"""Summarise the supplement queue (results/2026-11-16_supp) and the stride-1 reference runs."""
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / 'results'
S = R / (sys.argv[1] if len(sys.argv) > 1 else '2026-11-16_supp')
TCC1, TCC1K4, TCC2 = 'int4-g16::d8:bin-g16', 'int4-g16::d4:bin-g16', 'int4-g16::d8:int2-g64-asym'
G64, WIN16 = 'int2-g64-asym', 'int4-g16@win16'
J = lambda p: {r['config']: r for r in json.load(open(p))['results']}
bits = lambda R_, c: 32 * R_[c]['cache_mb'] / R_['fp32']['cache_mb']
dev = lambda R_, c: R_[c]['fid_rel_dev_pct']


def stride():
    print('== frame stride: deviation % (bits), delta1_seq change vs FP32')
    ref = R / '2026-11-13_vda_extra2'
    for ds in ('sintel', 'bonn'):
        for enc in ('vits', 'vitl'):
            rows = []
            for st in (1, 2, 4, 8):
                p = (ref / f'{ds}_{enc}.json') if st == 1 else S / f'stride_{ds}_{enc}_s{st}.json'
                if not p.exists():
                    continue
                X = J(p)
                n = len(X['fp32']['per_scene'])
                d1 = lambda c: X[c]['delta1_seq'] - X['fp32']['delta1_seq']
                rows.append(f"  {ds:6s} {enc} s{st} ({n} sc): " + '  '.join(
                    f"{lab} {dev(X, c):5.2f}% ({bits(X, c):.2f}b, d1 {d1(c):+.3f})"
                    for lab, c in (('INT2', G64), ('win16', WIN16), ('TCC1', TCC1), ('TCC1k4', TCC1K4), ('TCC2', TCC2))))
            print('\n'.join(rows))


def splice():
    print('== scene cuts (ScanNet, 85+85 frames): mean deviation % before / at cut (+0..2) / after (+10..84), bits, keyframes')
    for enc in ('vits', 'vitl'):
        X = J(S / f'splice_scannet_{enc}.json')
        for c in [k for k in X if k != 'fp32']:
            pf = [v['dev_per_frame'] for v in X[c]['per_scene'].values()]
            m = lambda a, b: float(np.mean([np.mean(p[a:b]) for p in pf]))
            peak = float(np.mean([max(p[85:95]) for p in pf]))
            rec = float(np.mean([next((i for i in range(85, 170) if p[i] <= 1.5 * np.mean(p[:85])), 170) - 85 for p in pf]))
            print(f"  {enc} {c:30s} before {m(10, 85):5.2f}  at cut {m(85, 88):5.2f} (peak {peak:5.2f})  "
                  f"after {m(95, 170):5.2f}  frames to recover {rec:4.1f}  bits {bits(X, c):.2f}  "
                  f"kf {X[c].get('keyframe_fraction', float('nan')):.3f}  dev all {dev(X, c):.2f}")


def rate():
    print('== rate vs fidelity (ViT-S): fixed K vs adaptive keyframes (K max 32)')
    for ds in ('sintel', 'bonn', 'scannet'):
        X = J(S / f'rate_{ds}_vits.json')
        print(f"  {ds}: " + '  '.join(f"{c.replace('int4-g16::','').replace(':bin-g16','')} {bits(X, c):.2f}b/"
                                      f"{dev(X, c):.2f}%/kf{X[c].get('keyframe_fraction', 0):.2f}"
                                      for c in X if c != 'fp32'))


def moderr():
    print('== error after each temporal module (%, mean over every 4th frame) and depth deviation')
    for ds in ('sintel', 'bonn'):
        for enc in ('vits', 'vitl'):
            X = J(S / f'moderr_{ds}_{enc}.json')
            for c in [k for k in X if k != 'fp32']:
                me = X[c].get('module_rel_err_pct', {})
                print(f"  {ds:6s} {enc} {c:28s} modules {[round(me[k], 2) for k in sorted(me)]}  depth {dev(X, c):.2f}%")


def redundancy():
    print('== redundancy: % of entry variance left by the previous frame (pooled), per layer; cross-layer oracle')
    d = json.load(open(S / 'redundancy.json'))
    for ds, encs in d.items():
        for enc, sts in encs.items():
            for st, a in sts.items():
                t = a['temporal_pooled']
                tf = a['temporal_fit_mean']
                x = a['cross_layer_mean']
                odd = [l for l in range(len(t)) if x[l] is not None]
                print(f"  {ds:7s} {enc} s{st}: temporal {min(t):5.1f}-{max(t):5.1f} (mean {np.mean(t):5.1f}), "
                      f"temporal affine {np.mean(tf):5.1f}; odd layers temporal "
                      f"{[round(a['temporal_mean'][l], 1) for l in odd]} vs cross-layer {[round(x[l], 1) for l in odd]}")


def timing():
    print('== latency vs K (A100)')
    d = json.load(open(S / 'timing_k.json'))
    rs = d['results'] if isinstance(d, dict) and 'results' in d else d
    for r in rs:
        print('  ', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()
                     if k in ('encoder', 'config', 'ms_median', 'ms_mean', 'fps', 'cache_mb', 'peak_mb')})


for f in (stride, splice, rate, moderr, redundancy, timing):
    try:
        f()
    except Exception as e:                     # keep going: one section failing must not hide the others
        print(f'  [{f.__name__} failed: {e!r}]')
