"""
Build a flat, named, rounded fact table from the streaming-cache result JSONs, so
that every number in the paper can be traced to (and checked against) a fact.

    python -X utf8 scripts/build_facts.py                 # default file set
    python -X utf8 scripts/build_facts.py outputs/stream/sintel_vitb.json results/.../sintel_vits.json

Outputs (default results/2026-11-02_streaming/analysis):
  facts.json            {'bonn.vits.fp32.covtae': 1.711, 'bonn.vits.fp32.covtae_raw': 1.7107..., ...}
  facts_meta.json       key -> {kind: json|derived|manual, source, decimals}
  inventory_check.md/.json   docs/results_inventory.md (2026-11-02 section) numbers vs facts
  facts_aliases.json    alias -> original stream_eval config label (per group)

Key scheme:  <group>.<alias>.<metric>[_raw|_ci_lo|_ci_hi|_fw...]
  group  : bonn.vits | sintel.vits | sintel.vitb | sintel.vitl | hard.sintel.vits | mem.vits | mem.vitb | mem.vitl
  alias  : fp32, fp16, int3_g16, int3_gT, bin_g16, lam16_b3 (Lambda16, bw16:3), zero,
           delta_int3_bin_K8 (= 'int3-g16::d8:bin-g16'), ...  (see facts_aliases.json)
  metric : delta1, absrel, tae, covtae, covis_fraction, cache_mb, peak_mb, bits,
           covtae_rel_pct / tae_rel_pct (% change vs FP32, scene-mean),
           delta1_diff / absrel_diff (absolute difference vs FP32),
           *_ci_lo / *_ci_hi (95% paired bootstrap over scenes), *_fw (frame-weighted)
Every value has a full-precision twin with suffix `_raw`.
"""
import argparse
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import bootstrap_ci as bc  # noqa: E402


def rnd(x, d):
    """Round half away from zero on the decimal repr (what a human/LaTeX would print)."""
    if x is None:
        return None
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return int(q) if d == 0 else float(q)


class Facts:
    def __init__(self):
        self.f, self.meta, self.conflicts = {}, {}, []

    def add(self, key, val, dec, kind='json', source=''):
        if val is None:
            return
        if key in self.f:
            if abs(self.f[key + '_raw'] - float(val)) > 1e-9 * max(1.0, abs(float(val))):
                self.conflicts.append(f'{key}: kept {self.f[key + "_raw"]!r} from {self.meta[key]["source"]}, '
                                      f'ignored {float(val)!r} from {source}')
            return
        self.f[key] = rnd(val, dec)
        self.f[key + '_raw'] = float(val)
        self.meta[key] = dict(kind=kind, source=source, decimals=dec)


def add_stream_group(F, g, res):
    p = g.name
    src = ','.join(sorted(set(g.src.values())))
    ref = res.get('fp32')
    F.add(f'{p}.n_scenes', len(g.scenes), 0, 'json', src)
    if ref:
        F.add(f'{p}.n_frames', ref['n_frames'], 0, 'json', src)
    for lab, e in res.items():
        k = f'{p}.{e["alias"]}'
        s = e['file']
        F.add(f'{k}.delta1', e['delta1'], 4, 'json', s)
        F.add(f'{k}.absrel', e['abs_rel'], 4, 'json', s)
        F.add(f'{k}.tae', e['tae'], 3, 'json', s)
        F.add(f'{k}.covtae', e['covtae'], 3, 'json', s)
        F.add(f'{k}.covis_fraction', e.get('covis_fraction'), 3, 'json', s)
        F.add(f'{k}.cache_mb', e['cache_mb'] if 'cache_mb' in e else None, 1, 'json', s)
        F.add(f'{k}.peak_mb', e.get('peak_mb'), 1, 'json', s)
        F.add(f'{k}.bits', e['bits'], 2, 'derived' if g.kind == 'stream' else 'json', s)
        if lab == 'fp32':
            continue
        F.add(f'{k}.absrel_diff', e['abs_rel_diff'], 4, 'derived', s)
        F.add(f'{k}.delta1_agg_diff', e['delta1_agg_diff'], 4, 'derived', s)
        if g.kind == 'hard':
            F.add(f'{k}.covtae_rel_pct', e['covtae_rel_point'], 1, 'derived', s)
        for name, dec, st in (('covtae_rel_pct', 1, e.get('covtae_rel')), ('tae_rel_pct', 1, e.get('tae_rel')),
                              ('delta1_diff', 4, e.get('delta1_diff'))):
            if not st or 'sm' not in st:
                continue
            for w, suf in (('sm', ''), ('fw', '_fw')):
                if g.kind == 'hard' and name == 'covtae_rel_pct':
                    continue
                kk = f'{k}.{name}{suf}'
                if kk not in F.f:  # hard covtae_rel_pct already added as a point estimate
                    F.add(kk, st[w], dec, 'derived', s)
                if f'{w}_ci' in st:
                    F.add(f'{kk}_ci_lo', st[f'{w}_ci'][0], dec, 'derived', s)
                    F.add(f'{kk}_ci_hi', st[f'{w}_ci'][1], dec, 'derived', s)
    if g.kind == 'stream':   # bits when the cache is compared with FP16 is occasionally quoted
        pass


def add_mem(F, mem_blocks):
    by_enc = {}
    for blk in mem_blocks:
        enc = blk['encoder']
        for r in blk['rows']:
            lab = bc.label_of(r)
            a = bc.alias_of(lab)
            k = f'mem.{enc}.{a}'
            s = blk['stem']
            F.add(f'{k}.cache_mb', r['cache_mb'], 1, 'json', s)
            F.add(f'{k}.peak_mb', r.get('peak_mb'), 0, 'json', s)
            F.add(f'{k}.steady_alloc_mb', r.get('steady_alloc_mb'), 0, 'json', s)
            F.add(f'{k}.weights_mb', r.get('weights_mb'), 1, 'json', s)
            F.add(f'{k}.step_ms_median', r.get('step_ms_median'), 1, 'json', s)
            F.add(f'{k}.depth_rel_l1_mean', r.get('depth_rel_l1_mean'), 4, 'json', s)
            F.add(f'{k}.depth_rel_l1_max', r.get('depth_rel_l1_max'), 4, 'json', s)
            by_enc.setdefault(enc, {})[a] = r
    # derived ratios
    for enc, rows in by_enc.items():
        ref = rows.get('fp32')
        if not ref:
            continue
        for a, r in rows.items():
            if a == 'fp32' or not r['cache_mb']:
                continue
            F.add(f'mem.{enc}.{a}.cache_ratio_vs_fp32', ref['cache_mb'] / r['cache_mb'], 1, 'derived', 'mem')
            if r.get('peak_mb') and ref.get('peak_mb'):
                F.add(f'mem.{enc}.{a}.peak_ratio_vs_fp32', ref['peak_mb'] / r['peak_mb'], 2, 'derived', 'mem')
    for enc, rows in by_enc.items():  # short names used in the inventory text
        d = rows.get('delta_int3_bin_K8')
        if d and rows.get('fp32'):
            F.add(f'mem.{enc}.delta_cache_ratio', rows['fp32']['cache_mb'] / d['cache_mb'], 1, 'derived', 'mem')


# numbers typed into docs/results_inventory.md that have NO json source in the repo
MANUAL = {
    # stream_delta_probe.py output, Bonn, 150 frames/scene: no json was saved
    'probe.bonn.frames_per_scene': (150, 0),
    'probe.bonn.residual_var_ratio_min': (0.013, 3), 'probe.bonn.residual_var_ratio_max': (0.255, 3),
    'probe.bonn.residual_var_ratio_weighted_min': (0.111, 3),
    'probe.bonn.residual_var_ratio_weighted_max': (0.133, 3),
    'probe.bonn.bits_saving_min': (1.0, 1), 'probe.bonn.bits_saving_max': (2.4, 1),
    # inventory's "packed bits" column of the hard-setting table where it differs from the json's
    # effective_bits_per_scalar (payload + 8/g); NOT verifiable from the json files
    'hard.sintel.vits.e8_g16_b3.bits_inventory': (3.0, 3), 'hard.sintel.vits.e8_g32_b3.bits_inventory': (2.75, 3),
    'hard.sintel.vits.e8_g64_b3.bits_inventory': (2.625, 3), 'hard.sintel.vits.lam16_g64_b3.bits_inventory': (2.625, 3),
    'hard.sintel.vits.e8_g64_b2.bits_inventory': (1.625, 3),
}

# (key, value as printed in docs/results_inventory.md, decimals as printed)
def inventory_claims():
    C = []

    def row(prefix, alias, **kw):
        for m, (v, d) in kw.items():
            C.append((f'{prefix}.{alias}.{m}', v, d))

    B = 'bonn.vits'
    row(B, 'fp32', delta1=(0.9878, 4), absrel=(0.0442, 4), tae=(2.831, 3), covtae=(1.711, 3), cache_mb=(355.9, 1))
    row(B, 'fp16', bits=(16, 0), covtae_rel_pct=(0.0, 1), absrel=(0.0442, 4))
    row(B, 'int8_g64', bits=(8.25, 2), covtae_rel_pct=(0.0, 1), absrel=(0.0442, 4))
    for a in ('int4_g16', 'int4_g64_asym', 'int4_g16_rht'):
        row(B, a, bits=(4.5, 1), covtae_rel_pct=(-0.1, 1), absrel=(0.0442, 4))
    row(B, 'int3_g16', bits=(3.5, 1), covtae_rel_pct=(-0.5, 1), absrel=(0.0442, 4))
    row(B, 'int3_g16_rht', bits=(3.5, 1), covtae_rel_pct=(-0.5, 1), absrel=(0.0443, 4))
    row(B, 'int3_gT', bits=(3.0, 1), covtae_rel_pct=(-0.2, 1), absrel=(0.0444, 4))
    row(B, 'int2_g16', bits=(2.5, 1), covtae_rel_pct=(2.9, 1), absrel=(0.0449, 4))
    row(B, 'int2_g16_rht', bits=(2.5, 1), covtae_rel_pct=(0.8, 1), absrel=(0.0445, 4))
    row(B, 'bin_g16', bits=(1.5, 1), covtae_rel_pct=(35.9, 1), absrel=(0.0447, 4))
    row(B, 'delta_int3_bin_K8', bits=(2.05, 2), covtae_rel_pct=(0.6, 1), absrel=(0.0444, 4))
    row(B, 'delta_int3_bin_K16', bits=(2.20, 2), covtae_rel_pct=(0.8, 1), absrel=(0.0444, 4))
    row(B, 'delta_int4_bin_K8', bits=(2.21, 2), covtae_rel_pct=(0.6, 1), absrel=(0.0444, 4))
    row(B, 'lam16_b3', bits=(3.0, 1), covtae_rel_pct=(0.2, 1), absrel=(0.0443, 4))
    row(B, 'lam16_b2', bits=(2.0, 1), covtae_rel_pct=(13.0, 1), absrel=(0.0448, 4))
    row(B, 'zero', bits=(0, 0), covtae_rel_pct=(35.5, 1), absrel=(0.0460, 4))
    C += [(f'{B}.n_scenes', 3, 0), (f'{B}.n_frames', 2456, 0)]

    S = 'sintel.vits'
    row(S, 'fp32', delta1=(0.7194, 4), absrel=(0.2268, 4), tae=(53.03, 2), covtae=(6.848, 3), cache_mb=(360.0, 1))
    row(S, 'int3_gT', bits=(3.0, 1), covtae_rel_pct=(-0.5, 1), tae_rel_pct=(-0.3, 1), absrel=(0.2277, 4), delta1=(0.7184, 4))
    row(S, 'int3_g16', bits=(3.5, 1), covtae_rel_pct=(-0.4, 1), tae_rel_pct=(-0.3, 1), absrel=(0.2267, 4), delta1=(0.7196, 4))
    row(S, 'int2_g16', bits=(2.5, 1), covtae_rel_pct=(-0.3, 1), tae_rel_pct=(0.3, 1), absrel=(0.2314, 4), delta1=(0.7124, 4))
    row(S, 'int2_g16_rht', bits=(2.5, 1), covtae_rel_pct=(4.6, 1), tae_rel_pct=(0.7, 1), absrel=(0.2224, 4), delta1=(0.7244, 4))
    row(S, 'bin_g16', bits=(1.5, 1), covtae_rel_pct=(35.6, 1), absrel=(0.2223, 4), delta1=(0.7197, 4))
    row(S, 'delta_int3_bin_K8', bits=(1.87, 2), covtae_rel_pct=(0.1, 1), absrel=(0.2267, 4), delta1=(0.7188, 4))
    row(S, 'delta_int3_int2_K8', bits=(2.73, 2), covtae_rel_pct=(0.0, 1), absrel=(0.2271, 4), delta1=(0.7187, 4))
    row(S, 'lam16_b3', bits=(3.0, 1), covtae_rel_pct=(0.4, 1), tae_rel_pct=(0.0, 1), absrel=(0.2272, 4), delta1=(0.7189, 4))
    row(S, 'lam16_b2', bits=(2.0, 1), covtae_rel_pct=(14.5, 1), tae_rel_pct=(0.0, 1), absrel=(0.2242, 4), delta1=(0.7157, 4))
    row(S, 'zero', bits=(0, 0), covtae_rel_pct=(60.1, 1), tae_rel_pct=(2.9, 1), absrel=(0.2200, 4), delta1=(0.7163, 4))
    C += [(f'{S}.n_scenes', 23, 0), (f'{S}.n_frames', 1045, 0)]

    C += [('mem.vits.fp32.cache_mb', 269.0, 1), ('mem.vits.lam16_b3.cache_mb', 25.2, 1),
          ('mem.vits.fp32.peak_mb', 770, 0), ('mem.vits.lam16_b3.peak_mb', 337, 0)]
    for enc, c0, c1, ratio, p0, p1, l_i2, l_d in (('vitb', 538.0, 31.4, 17.1, 1786, 1115, (0.208, 3), (0.0048, 4)),
                                                  ('vitl', 1129.2, 66.0, 17.1, 4150, 2857, (0.033, 3), (0.0071, 4))):
        m = f'mem.{enc}'
        C += [(f'{m}.fp32.cache_mb', c0, 1), (f'{m}.delta_int3_bin_K8.cache_mb', c1, 1),
              (f'{m}.delta_cache_ratio', ratio, 1), (f'{m}.fp32.peak_mb', p0, 0),
              (f'{m}.delta_int3_bin_K8.peak_mb', p1, 0),
              (f'{m}.int2_g16.depth_rel_l1_mean', l_i2[0], l_i2[1]),
              (f'{m}.delta_int3_bin_K8.depth_rel_l1_mean', l_d[0], l_d[1])]
    # "exact (relL1 1e-4)": json has 7.6e-5 (printed as 1e-4 to one significant figure)
    C += [('mem.vits.fp32_kv.depth_rel_l1_mean', 1e-4, 4), ('mem.vits.fp32_kv.cache_mb', 538.0, 1)]
    for k, v, d in (('probe.bonn.residual_var_ratio_min', 0.013, 3), ('probe.bonn.residual_var_ratio_max', 0.255, 3),
                    ('probe.bonn.residual_var_ratio_weighted_min', 0.111, 3),
                    ('probe.bonn.residual_var_ratio_weighted_max', 0.133, 3)):
        C.append((k, v, d))

    H = 'hard.sintel.vits'
    C += [(f'{H}.fp32.delta1', 0.6876, 4), (f'{H}.fp32.covtae', 8.56, 2)]
    for a, bits, d1, cov in (('int3_g16', 3.5, 0.6869, 11.08), ('int3_g64', 3.125, 0.6938, 14.03),
                             ('e8_g16_b3', 3.0, 0.6836, 12.06), ('e8_g32_b3', 2.75, 0.6775, 12.44),
                             ('e8_g64_b3', 2.625, 0.6836, 14.97), ('lam16_g64_b3', 2.625, 0.6225, 52.69),
                             ('int2_g16', 2.5, 0.4187, 6.38), ('int2_g64', 2.125, 0.4069, 5.96),
                             ('e8_g64_b2', 1.625, 0.4252, 7.61), ('int3_tensor', 3.125, 0.4509, 63.88),
                             ('int4_tensor', 4.125, 0.6314, 14.49)):
        C += [(f'{H}.{a}.bits', bits, 3), (f'{H}.{a}.delta1', d1, 4), (f'{H}.{a}.covtae', cov, 2)]
    return C


def check_inventory(F):
    rows = []
    for key, val, dec in inventory_claims():
        raw = F.f.get(key + '_raw')
        if raw is None:
            rows.append(dict(key=key, claimed=val, decimals=dec, status='NO FACT', actual_raw=None))
            continue
        got = rnd(raw, dec)
        ok = abs(got - val) < 1e-12
        rows.append(dict(key=key, claimed=val, decimals=dec, actual_raw=raw, actual_rounded=got,
                         status='ok' if ok else 'MISMATCH'))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*')
    ap.add_argument('--out-dir', default=str(bc.DEFAULT_OUT))
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    files = [Path(x) for x in a.files] or bc.default_files()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    results, mem = bc.compute(files, a.n_boot, a.seed)
    F = Facts()
    aliases = {}
    for g, res in results:
        if not res:
            continue
        add_stream_group(F, g, res)
        aliases[g.name] = {e['alias']: lab for lab, e in res.items()}
    add_mem(F, mem)
    for k, (v, d) in MANUAL.items():
        F.add(k, v, d, 'manual', 'docs/results_inventory.md (no json source)')

    facts = {k: F.f[k] for k in sorted(F.f)}
    (out / 'facts.json').write_text(json.dumps(facts, indent=1), encoding='utf-8')
    (out / 'facts_meta.json').write_text(json.dumps({k: F.meta[k] for k in sorted(F.meta)}, indent=1), encoding='utf-8')
    (out / 'facts_aliases.json').write_text(json.dumps(aliases, indent=1), encoding='utf-8')

    chk = check_inventory(F)
    (out / 'inventory_check.json').write_text(json.dumps(chk, indent=1), encoding='utf-8')
    bad = [c for c in chk if c['status'] != 'ok']
    L = ['# docs/results_inventory.md (2026-11-02 section) vs facts', '',
         f'{len(chk)} printed numbers checked, {len(chk) - len(bad)} reproduced at the printed precision, '
         f'{len(bad)} do not.', '']
    if bad:
        L += ['| fact key | printed in inventory | actual (rounded to printed precision) | actual raw | status |',
              '|---|---|---|---|---|']
        for c in bad:
            L.append(f"| `{c['key']}` | {c['claimed']} | {c.get('actual_rounded')} | {c['actual_raw']} | {c['status']} |")
    L.append('')
    (out / 'inventory_check.md').write_text('\n'.join(L), encoding='utf-8')

    n_vals = sum(1 for k in facts if not k.endswith('_raw'))
    print(f'{n_vals} facts ({len(facts)} keys incl. _raw) -> {out / "facts.json"}')
    print('\n'.join(L))
    if F.conflicts:
        print('conflicting duplicate facts (first kept):')
        print('\n'.join('  ' + c for c in F.conflicts))
    (out / 'facts_conflicts.txt').write_text('\n'.join(F.conflicts), encoding='utf-8')


if __name__ == '__main__':
    main()
