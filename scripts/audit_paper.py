"""
Recompute headline numbers of the paper from the raw result files and check that the
paper states them as written. Prints OK / MISMATCH per claim. Read-only.

    python -X utf8 scripts/audit_paper.py <path to main.tex>
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / 'results'
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_paper_assets as m  # noqa: E402

tex = re.sub(r'\s+', ' ', open(sys.argv[1], encoding='utf8').read())
ok = bad = 0


def check(label, value, text, fmt):
    """`text` must appear in the paper and contain `value` formatted with `fmt`."""
    global ok, bad
    shown = fmt.format(value)
    present = text in tex
    good = present and shown in text
    ok += good
    bad += not good
    print(f"{'OK      ' if good else 'MISMATCH'} {label}: data {shown} | paper '{text}'"
          + ('' if present else '  <- text not in paper'))


svg = {ds: {r['kv']: r for r in json.load(open(R / f'fidelity_{ds}.json'))['results']}
       for ds in ('sintel', 'bonn', 'kitti')}
dev = lambda ds, kv: svg[ds][kv]['fid_rel_dev_pct']
TCC1, TCC2 = 'delta:int4:bin:8', 'delta:int4:a2g64:8'

print('== StreamVGGT official benchmark')
check('Sintel TCC-1', dev('sintel', TCC1), r'(3.41\% over all 23)', '{:.2f}')
check('Bonn TCC-1', dev('bonn', TCC1), r'against 0.78\% and 4.15\%', '{:.2f}')
check('Sintel TCC-1 vs INT2', dev('sintel', 'plain:a2g64') / dev('sintel', TCC1), r'2.2--2.3$\times$ closer', '{:.1f}')
check('Bonn TCC-1 vs INT2', dev('bonn', 'plain:a2g64') / dev('bonn', TCC1), r'2.2--2.3$\times$ closer', '{:.1f}')
check('Sintel TCC-1 vs KIVI', dev('sintel', 'kivi:2:64') / dev('sintel', TCC1), r'$1.8\times$ closer than KIVI', '{:.1f}')
check('Bonn TCC-1 vs KIVI', dev('bonn', 'kivi:2:64') / dev('bonn', TCC1), r'$1.8\times$ closer than KIVI', '{:.1f}')
check('XStream 12000 Sintel dev', dev('sintel', 'xstream:12000:4:4'), r'deviates by 5.26\%', '{:.2f}')
check('XStream 12000 Sintel MB', svg['sintel']['xstream:12000:4:4']['cache_mb_max'], r'(318\,MB, Sintel)', '{:.0f}')
check('Sintel TCC-2 vs INT2 g32', dev('sintel', 'plain:a2g32') / dev('sintel', TCC2), r'$2.8\times$ and $4.2\times$', '{:.1f}')
check('Bonn TCC-2 vs INT2 g32', dev('bonn', 'plain:a2g32') / dev('bonn', TCC2), r'$2.8\times$ and $4.2\times$', '{:.1f}')
check('INT3 Sintel dev', dev('sintel', 'plain:int3'), r'INT3 at 3.5 bits deviates by 2.68\%', '{:.2f}')
check('INT3 Bonn dev', dev('bonn', 'plain:int3'), r'2.68\% and 0.84\%', '{:.2f}')
check('TCC-2 Sintel MB', svg['sintel'][TCC2]['cache_mb_max'], r'456 instead of 738\,MB', '{:.0f}')
check('INT3 Sintel MB', svg['sintel']['plain:int3']['cache_mb_max'], r'456 instead of 738\,MB', '{:.0f}')
check('zero cache Sintel', dev('sintel', 'zero'), r'59\% (Sintel)', '{:.0f}')
check('zero cache Bonn', dev('bonn', 'zero'), r'29\% (Bonn)', '{:.0f}')
check('open loop TCC-1 Sintel', dev('sintel', TCC1 + ':o'), r'3.41\% to 4.86\%', '{:.2f}')
check('open loop TCC-1 Bonn', dev('bonn', TCC1 + ':o'), r'0.78\% to 1.07\%', '{:.2f}')

print('== KITTI')
check('KITTI TCC-1', dev('kitti', TCC1), r'deviates by 2.31\%', '{:.2f}')
check('KITTI TCC-2', dev('kitti', TCC2), r'\ours-2 by 1.20\%', '{:.2f}')
check('KITTI KIVI', dev('kitti', 'kivi:2:64'), r'5.03\% for KIVI', '{:.2f}')
check('KITTI INT2', dev('kitti', 'plain:a2g64'), r'7.06\% for asymmetric INT2', '{:.2f}')
check('KITTI XStream', dev('kitti', 'xstream:2048:4:4'), r'14.48\% for XStreamVGGT', '{:.2f}')
q7 = R / '2026-11-13_q7' / 'media/rvcse22/CSERV/vdaquant/code_run/svg_queue7'
sk = {x['run'].rstrip('/').split('/')[-1]: x for x in json.load(open(q7 / 'stac' / 'fid_official_kitti.json'))['results']}
check('KITTI STAC (own code)', sk['stac_kitti_default']['fid_rel_dev_pct'], r'deviates by 6.41\% from the uncompressed model', '{:.2f}')

print('== bootstrap')
bs = json.load(open(R / 'paper_bootstrap.json'))['results']
get = lambda **k: next(r for r in bs if all(r.get(a) == b for a, b in k.items()))
r = get(model='StreamVGGT', data='kitti', a='TCC-1', b='KIVI')['dev']['ci95']
check('KITTI TCC-1/KIVI CI low', r[0], r'KIVI 1.8--2.7$\times$', '{:.1f}')
check('KITTI TCC-1/KIVI CI high', r[1], r'KIVI 1.8--2.7$\times$', '{:.1f}')
r = get(model='StreamVGGT', data='kitti', a='TCC-2', b='KIVI')['dev']['ci95']
check('KITTI TCC-2/KIVI CI low', r[0], r'and 3.7--4.7$\times$', '{:.1f}')
check('KITTI TCC-2/KIVI CI high', r[1], r'and 3.7--4.7$\times$', '{:.1f}')
r = get(model='StreamVGGT', data='sintel', a='TCC-1', b='INT2 g64')['dev']['ci95']
check('Sintel TCC-1/INT2 CI low', r[0], r'Sintel of 1.7--3.0$\times$', '{:.1f}')
check('Sintel TCC-1/INT2 CI high', r[1], r'Sintel of 1.7--3.0$\times$', '{:.1f}')
r = get(model='StreamVGGT', data='sintel', a='TCC-1', b='KIVI')['dev']['ci95']
check('Sintel TCC-1/KIVI CI low', r[0], r'and 1.35--2.5$\times$', '{:.2f}')
check('Sintel TCC-1/KIVI CI high', r[1], r'and 1.35--2.5$\times$', '{:.1f}')
fd = [get(model='VDA', data=d, enc=e, a='TCC-1 K4', b='INT4 drop2')['dev']['ratio']
      for d in ('sintel', 'bonn') for e in ('vits', 'vitb', 'vitl')]
check('VDA vs drop2 ratio min', min(fd), r'1.2--1.7$\times$ closer to the uncompressed model on all six', '{:.1f}')
check('VDA vs drop2 ratio max', max(fd), r'1.2--1.7$\times$ closer to the uncompressed model on all six', '{:.1f}')
lo = lambda a, b: min(get(model='VDA', data='sintel', enc=e, a=a, b=b)['dev']['ci95'][0] for e in ('vits', 'vitb', 'vitl'))
check('VDA drop2 Sintel CI min', lo('TCC-1 K4', 'INT4 drop2'), r'above $1.06\times$', '{:.2f}')
check('VDA TCC-2 Sintel CI min', lo('TCC-2', 'INT2 g32'), r'lies above $2.6\times$', '{:.1f}')
check('VDA TCC-1 K4 Sintel CI min', lo('TCC-1 K4', 'INT2 g64'), r'above $1.15\times$', '{:.2f}')
ratios = [get(model='VDA', data=d, enc=e, a=a, b=b)['dev']['ratio']
          for d in ('sintel', 'bonn') for e in ('vits', 'vitb', 'vitl')
          for a, b in (('TCC-1 closed', 'TCC-1 open'), ('TCC-2 closed', 'TCC-2 open'))]
check('VDA open loop min', min(ratios), r'1.7--2.7$\times$ less faithful', '{:.1f}')
check('VDA open loop max', max(ratios), r'1.7--2.7$\times$ less faithful', '{:.1f}')

print('== STAC on its scenes')
q = R / '2026-11-12_q5_stac'
for ds, txt in (('bonn', r'deviates by 1.87\% on Bonn'), ('sintel', r'8.92\% on the 14 Sintel scenes')):
    sr = {x['run'].rstrip('/').split('/')[-1]: x for x in json.load(open(q / f'fid_official_{ds}.json'))['results']}
    sc = sorted(sr[f'stac_{ds}_default']['per_sequence'])
    check(f'STAC {ds} ({len(sc)} scenes)', np.mean([sr[f'stac_{ds}_default']['per_sequence'][s]['fid_rel_dev_pct'] for s in sc]), txt, '{:.2f}')
    check(f'TCC-1 on STAC {ds} scenes', np.mean([svg[ds][TCC1]['per_sequence'][s]['fid_rel_dev_pct'] for s in sc]),
          r'against 0.78\% and 4.15\%', '{:.2f}')

print('== long streams')
d = json.load(open(q / 'svg_queue5' / 'longn' / 'delta_int4_bin_8.json'))['scenes']['crowd2']
seg = [np.mean(d['series']['dev'][i:i + 100]) for i in range(0, len(d['series']['dev']), 100)]
check('crowd2 frames', d['frames'], r'streamed to 755 frames', '{:d}')
check('crowd2 TCC-1 min/100', min(seg), r'between 0.44\% and 0.95\%', '{:.2f}')
check('crowd2 TCC-1 max/100', max(seg), r'between 0.44\% and 0.95\%', '{:.2f}')
d = json.load(open(q / 'svg_queue5' / 'longn' / 'kivi_2_64.json'))['scenes']['crowd2']
seg = [np.mean(d['series']['dev'][i:i + 100]) for i in range(0, len(d['series']['dev']), 100)]
check('crowd2 KIVI min/100', min(seg), r'(KIVI: 1.13--2.48\%)', '{:.2f}')
check('crowd2 KIVI max/100', max(seg), r'(KIVI: 1.13--2.48\%)', '{:.2f}')
c3 = {k: json.load(open(q7 / 'long_crowd3' / f'{k}.json'))['scenes']['crowd3']
      for k in ('delta_int4_bin_8', 'delta_int4_a2g64_8', 'kivi_2_64', 'plain_a2g64')}
check('crowd3 frames', c3['delta_int4_bin_8']['frames'], r'streamed to 826 frames', '{:d}')
check('crowd3 TCC-1', np.mean(c3['delta_int4_bin_8']['series']['dev']), r'\ours-1 deviates by 0.43\%', '{:.2f}')
check('crowd3 TCC-2', np.mean(c3['delta_int4_a2g64_8']['series']['dev']), r'\ours-2 by 0.16\%', '{:.2f}')
check('crowd3 KIVI', np.mean(c3['kivi_2_64']['series']['dev']), r'against 1.08\% for KIVI', '{:.2f}')
check('crowd3 INT2', np.mean(c3['plain_a2g64']['series']['dev']), r'1.53\% for INT2', '{:.2f}')
check('crowd3 memory saving', 100 * (1 - c3['delta_int4_bin_8']['cache_mb'] / c3['kivi_2_64']['cache_mb']),
      r'storing 27\% less than KIVI', '{:.0f}')

print('== contraction (TCC-2)')
cs = {'k': [], 'v': []}
for v in json.load(open(q / 'svg_queue5' / 'contraction' / 'delta_int4_a2g64_8.json'))['scenes'].values():
    for layer in v['trace']:
        for w in 'kv':
            t = layer[w]
            cs[w].append((t['quant_err_ratio'] / max(t['frames'], 1), t['quant_err_ratio_max']))
means = [100 * np.mean([a for a, b in cs[w]]) for w in 'kv']
check('TCC-2 mean low', min(means), r'19--23\% on average', '{:.0f}')
check('TCC-2 mean high', max(means), r'19--23\% on average', '{:.0f}')
check('TCC-2 max', 100 * max(b for w in 'kv' for a, b in cs[w]), r'at most 47\%', '{:.0f}')

print('== 3D reconstruction (dense)')
rf = json.load(open(R / '2026-11-11_timing' / 'recon_fidelity.json'))
mean = lambda k, mt: rf[k]['methods'][mt]['mean']
check('7sc TCC-2 vs KIVI', mean('dense/7scenes', 'kivi_2_64') / mean('dense/7scenes', 'delta_int4_a2g64_8'), r'$1.4\times$ and $3.7\times$', '{:.1f}')
check('NRGBD TCC-2 vs KIVI', mean('dense/NRGBD', 'kivi_2_64') / mean('dense/NRGBD', 'delta_int4_a2g64_8'), r'$1.4\times$ and $3.7\times$', '{:.1f}')
check('7sc TCC-2 vs INT2', mean('dense/7scenes', 'plain_a2g64') / mean('dense/7scenes', 'delta_int4_a2g64_8'), r'$2.0\times$ and $5.8\times$', '{:.1f}')
check('NRGBD TCC-2 vs INT2', mean('dense/NRGBD', 'plain_a2g64') / mean('dense/NRGBD', 'delta_int4_a2g64_8'), r'$2.0\times$ and $5.8\times$', '{:.1f}')
check('7sc TCC-1 median', rf['dense/7scenes']['methods']['delta_int4_bin_8']['median'], r'(3.77\% and 1.62\%)', '{:.2f}')
check('NRGBD TCC-1 median', rf['dense/NRGBD']['methods']['delta_int4_bin_8']['median'], r'(3.77\% and 1.62\%)', '{:.2f}')

print('== VDA speed (A100)')
tm = json.load(open(R / 'vda_timing_a100.json'))['results']
med = lambda e, c: st.median(x['ms_median'] for x in tm if x['encoder'] == e and x['config'] == c)
tccs = ('int4-g16::d8:bin-g16', 'int4-g16::d4:bin-g16', 'int4-g16::d8:int2-g64-asym')
add = [med(e, c) - med(e, 'fp32') for e in ('vits', 'vitb', 'vitl') for c in tccs]
check('TCC added ms min', min(add), r'adds 4--11\,ms per frame to the FP32 cache', '{:.0f}')
check('TCC added ms max', max(add), r'adds 4--11\,ms per frame to the FP32 cache', '{:.0f}')
add2 = [med(e, 'int2-g64-asym') - med(e, 'fp32') for e in ('vits', 'vitb', 'vitl')]
check('INT2 added ms min', min(add2), r'(asymmetric INT2: 3--7\,ms)', '{:.0f}')
check('INT2 added ms max', max(add2), r'(asymmetric INT2: 3--7\,ms)', '{:.0f}')
fps = [1000 / med(e, c) for e in ('vits', 'vitb', 'vitl') for c in tccs]
check('TCC FPS min', min(fps), r'runs at 10--20 frames per second', '{:.0f}')
check('TCC FPS max', max(fps), r'runs at 10--20 frames per second', '{:.0f}')

print('== VDA ground-truth metrics')
mx = [0, 0, 0]
for ds in ('sintel', 'bonn'):
    for e in ('vits', 'vitb', 'vitl'):
        rs = m.pair_rows(ds, e)
        b = m.row(rs, 'fp32')
        for c in tccs:
            x = m.row(rs, c)
            mx[0] = max(mx[0], abs(x['delta1'] - b['delta1']))
            mx[1] = max(mx[1], abs(x['abs_rel'] - b['abs_rel']))
        mx[2] = max(mx[2], abs(m.row(rs, 'int2-g64-asym')['abs_rel'] - b['abs_rel']))
check('TCC max |d delta1|', mx[0], r'$\delta_1$ within 0.001 and AbsRel within 0.0004', '{:.3f}')
check('TCC max |d AbsRel|', mx[1], r'$\delta_1$ within 0.001 and AbsRel within 0.0004', '{:.4f}')
check('INT2 max |d AbsRel|', mx[2], r'AbsRel up to 0.0023', '{:.4f}')

print('== zero-cache control (VDA)')
zd1, zdev = [], []
for ds in ('sintel', 'bonn'):
    for e in ('vits', 'vitb', 'vitl'):
        rs = m.pair_rows(ds, e)
        b, z = m.row(rs, 'fp32'), m.row(rs, 'zero')
        zd1.append(abs(z['delta1'] - b['delta1']))
        zdev.append(z['fid_rel_dev_pct'])
check('zero cache max |d delta1|', max(zd1), r'per-frame $\delta_1$ within 0.006', '{:.3f}')
check('zero cache deviation min', min(zdev), r'output moves by 16--38\%', '{:.0f}')
check('zero cache deviation max', max(zdev), r'output moves by 16--38\%', '{:.0f}')

print(f'\n{ok} OK, {bad} MISMATCH')
