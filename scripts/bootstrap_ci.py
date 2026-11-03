"""
Paired-bootstrap confidence intervals for the streaming temporal-cache results.

Reads the JSON files written by ``scripts/stream_eval.py`` (and the
``run_pareto_benchmark_suite.py`` "hard setting" files) and, for each
(dataset, encoder) group, computes for every config the paired difference to the
FP32 reference over scenes:

  * cov-TAE   relative change in %  (100 * (mean_cfg / mean_fp32 - 1))
  * raw TAE   relative change in %
  * delta1    absolute difference
with 95% percentile CIs from a paired bootstrap over scenes (default 10,000
resamples, fixed seed; the SAME resampled scene indices are used for every
config and metric of a group, so the intervals are paired across configs too).
AbsRel is only stored as an aggregate, so it is tabulated without a CI.

Two aggregation weightings are reported:
  scene-mean   : plain mean over scenes (what stream_eval stores for TAE / cov-TAE)
  frame-weight : weighted by the number of frames per scene (what stream_eval's
                 aggregate delta1 corresponds to: it pools all frames)

Bits per value = 32 * cache_mb / fp32_cache_mb (streaming files) or the stored
``effective_bits_per_scalar`` (hard-setting files).

Usage (CPU only)::

    python -X utf8 scripts/bootstrap_ci.py                       # default file set
    python -X utf8 scripts/bootstrap_ci.py outputs/stream/sintel_vitb.json \
        results/2026-11-02_streaming/streaming_vda/sintel_vits.json

Outputs (``--out-dir``, default results/2026-11-02_streaming/analysis):
  bootstrap_<group>.md, bootstrap_<group>_scenemean.tex, bootstrap_<group>_framew.tex,
  bootstrap_ci_all.md, bootstrap_ci.json
"""
import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
RES = REPO / 'results' / '2026-11-02_streaming'
DEFAULT_OUT = RES / 'analysis'

# --------------------------------------------------------------------------- labels

HARD_ALIASES = {
    ('int_g16', '3bit'): 'int3_g16', ('int_g16', '2bit'): 'int2_g16',
    ('int_g64', '3bit'): 'int3_g64', ('int_g64', '4bit'): 'int4_g64',
    ('int_g64b2', '2bit'): 'int2_g64',
    ('int_tensor', '3bit'): 'int3_tensor', ('int_tensor', '4bit'): 'int4_tensor',
    ('e8_g16', '3bit'): 'e8_g16_b3', ('e8_g16', '2bit'): 'e8_g16_b2',
    ('e8_g32', '3bit'): 'e8_g32_b3', ('e8_g64', '3bit'): 'e8_g64_b3',
    ('e8_g64b2', '2bit'): 'e8_g64_b2',
    ('bw16_g64', '3bit'): 'lam16_g64_b3',
}


def label_of(r):
    """Config label of a result row: 'config' if present, else mode[:bits]."""
    c = r.get('config')
    if c:
        return c
    m, b = r.get('mode'), r.get('bits')
    return f'{m}:{b}' if b is not None else m


def _base_alias(mode, bits, strip_g16=False):
    if mode.startswith('bw16'):
        a = 'lam16' + mode[4:].replace('-', '_') + (f'_b{bits}' if bits is not None else '')
    else:
        a = re.sub(r'[^0-9A-Za-z]+', '_', mode)
    return a.replace('_g16', '') if strip_g16 else a


def alias_of(label):
    """Short snake_case name for a stream_eval config label.

    'int3-g16::d8:bin-g16' -> delta_int3_bin_K8, 'bw16:3' -> lam16_b3, 'int3-gT' -> int3_gT.
    Config grammar (stream_eval.py): mode[:bits][:dK][:residual_mode]."""
    parts = label.split(':')
    mode = parts[0]
    bits = int(parts[1]) if len(parts) > 1 and parts[1] else None
    dk = int(parts[2][1:]) if len(parts) > 2 and parts[2].startswith('d') else None
    rmode = parts[3] if len(parts) > 3 and parts[3] else None
    if dk is None:
        return _base_alias(mode, bits)
    body = 'delta_' + _base_alias(mode, bits, strip_g16=True)
    if rmode:
        body += '_' + _base_alias(rmode, None, strip_g16=True)
    return f'{body}_K{dk}'


# --------------------------------------------------------------------------- loading

def load_file(path):
    """Return dict(kind, dataset, encoder, rows=[row...], path, stem). kind in
    {'stream' (per-scene accuracy), 'mem' (no accuracy), 'hard'}."""
    path = Path(path)
    d = json.load(open(path, encoding='utf-8'))
    stem = path.stem
    res = d['results']
    if isinstance(res, dict):  # run_pareto_benchmark_suite format
        out = []
        for dataset, cfgs in res.items():
            rows = []
            for key, v in cfgs.items():
                row = dict(v)
                row['_key'] = key
                row['_stem'] = stem
                rows.append(row)
            out.append(dict(kind='hard', dataset=dataset, encoder=d.get('encoder', 'vits'),
                            rows=rows, path=path, stem=stem, eval_mode=d.get('eval_mode')))
        return out
    a = d.get('args', {})
    has_acc = any(r.get('per_scene') for r in res)
    kind = 'stream' if has_acc else 'mem'
    return [dict(kind=kind, dataset=a.get('dataset'), encoder=a.get('encoder'),
                 rows=list(res), path=path, stem=stem, args=a)]


def default_files():
    files = sorted((RES / 'streaming_vda').glob('*.json'))
    files += sorted((RES / 'hard_setting_lattice_vs_int').glob('*.json'))
    files += [p for p in sorted((REPO / 'outputs' / 'stream').glob('sintel_vit[bl].json'))]
    return files


class Group:
    def __init__(self, kind, dataset, encoder, scenes):
        self.kind, self.dataset, self.encoder, self.scenes = kind, dataset, encoder, scenes
        self.rows = OrderedDict()   # label -> row
        self.src = {}               # label -> file stem
        self.notes = []
        self.name = None


def build_groups(files):
    """Group accuracy rows by (kind, dataset, encoder, scene set)."""
    groups = OrderedDict()
    mem = []
    for f in files:
        for blk in load_file(f):
            if blk['kind'] == 'mem':
                mem.append(blk)
                continue
            if blk['kind'] == 'stream':
                scenes = tuple(sorted(blk['rows'][0]['per_scene']))
            else:
                scenes = tuple(sorted(blk['rows'][0].get('per_scene_tae', {})))
            key = (blk['kind'], blk['dataset'], blk['encoder'], scenes)
            g = groups.setdefault(key, Group(*key))
            for r in blk['rows']:
                lab = label_of(r) if blk['kind'] == 'stream' else \
                    HARD_ALIASES.get((blk['stem'], r['_key']), f"{blk['stem']}:{r['_key']}")
                if blk['kind'] == 'hard' and r['_key'] == 'FP32_Baseline':
                    lab = 'fp32'
                if lab in g.rows:
                    old = g.rows[lab]
                    k1 = 'tae_covis_percent'
                    if abs((old.get(k1) or 0) - (r.get(k1) or 0)) > 1e-9 or \
                            abs((old.get('delta1') or 0) - (r.get('delta1') or 0)) > 1e-9:
                        g.notes.append(f"CONFLICT: '{lab}' appears in {g.src[lab]} and {blk['stem']} "
                                       f"with different values; kept the first")
                    continue
                g.rows[lab] = r
                g.src[lab] = blk['stem']
    # naming: main group of a (kind,dataset,encoder) keeps the plain name
    by_base = {}
    for k, g in groups.items():
        by_base.setdefault(k[:3], []).append(g)
    for (kind, ds, enc), gs in by_base.items():
        gs.sort(key=lambda g: -len(g.scenes))
        for i, g in enumerate(gs):
            base = f'hard.{ds}.{enc}' if kind == 'hard' else f'{ds}.{enc}'
            g.name = base if i == 0 else f'{base}.{len(g.scenes)}sc'
    return list(groups.values()), mem


# --------------------------------------------------------------------------- statistics

def _ci(samples, lo=2.5, hi=97.5):
    a, b = np.percentile(samples, [lo, hi])
    return float(a), float(b)


def paired_stat(c, r, n, U, kind):
    """c, r: per-scene arrays (nan = missing). Returns dict with scene-mean and
    frame-weighted point estimates + bootstrap CIs. kind 'rel' (% change of
    ratio of means) or 'diff' (difference of means)."""
    valid = ~(np.isnan(c) | np.isnan(r))
    m = int(valid.sum())
    out = dict(n_scenes=m)
    if m == 0:
        return out
    vi = np.flatnonzero(valid)
    cv, rv, nv = c[vi], r[vi], n[vi]

    def est(cc, rr, ww):
        if kind == 'rel':
            return 100.0 * ((cc * ww).sum(-1) / (rr * ww).sum(-1) - 1.0)
        return (cc * ww).sum(-1) / ww.sum(-1) - (rr * ww).sum(-1) / ww.sum(-1)

    ones = np.ones_like(cv)
    out['sm'] = float(est(cv, rv, ones))
    out['fw'] = float(est(cv, rv, nv))
    if m >= 2:
        idx = vi[(U[:, :m] * m).astype(int)]          # (B, m) indices into full arrays
        ci = c[idx]; ri = r[idx]; wi = n[idx]
        out['sm_ci'] = _ci(est(ci, ri, np.ones_like(ci)))
        out['fw_ci'] = _ci(est(ci, ri, wi))
    return out


def scene_arrays(row, scenes, key):
    return np.array([np.nan if (row['per_scene'].get(s, {}) or {}).get(key) is None
                     else row['per_scene'][s][key] for s in scenes], dtype=float)


def analyse_stream_group(g, n_boot, seed):
    ref = g.rows.get('fp32')
    res = OrderedDict()
    if ref is None:
        g.notes.append('no fp32 reference row in this group: relative statistics skipped')
        return res
    scenes = list(ref['per_scene'])
    ns = np.array([ref['per_scene'][s]['n'] for s in scenes], dtype=float)
    rng = np.random.default_rng(seed)
    U = rng.random((n_boot, len(scenes)))
    r_cov = scene_arrays(ref, scenes, 'tae_covis'); r_tae = scene_arrays(ref, scenes, 'tae')
    r_d1 = scene_arrays(ref, scenes, 'delta1')
    ref_cache = ref['cache_mb']
    for lab, row in g.rows.items():
        e = dict(alias=alias_of(lab), file=g.src[lab], bits=32.0 * row['cache_mb'] / ref_cache,
                 cache_mb=row['cache_mb'], delta1=row['delta1'], abs_rel=row['abs_rel'],
                 tae=row['tae_percent'], covtae=row['tae_covis_percent'],
                 covis_fraction=row.get('covis_fraction'), peak_mb=row.get('peak_mb'),
                 n_frames=row['n_frames'], abs_rel_diff=row['abs_rel'] - ref['abs_rel'],
                 delta1_agg_diff=row['delta1'] - ref['delta1'])
        c_cov = scene_arrays(row, scenes, 'tae_covis'); c_tae = scene_arrays(row, scenes, 'tae')
        c_d1 = scene_arrays(row, scenes, 'delta1')
        if lab != 'fp32':
            e['covtae_rel'] = paired_stat(c_cov, r_cov, ns, U, 'rel')
            e['tae_rel'] = paired_stat(c_tae, r_tae, ns, U, 'rel')
            e['delta1_diff'] = paired_stat(c_d1, r_d1, ns, U, 'diff')
        # consistency of stored aggregates with per-scene values
        v = ~np.isnan(c_cov)
        e['chk_covtae_scene_mean'] = float(np.nanmean(c_cov)) - row['tae_covis_percent'] if v.any() else None
        vd = ~np.isnan(c_d1)
        e['chk_delta1_frame_w'] = float((c_d1[vd] * ns[vd]).sum() / ns[vd].sum()) - row['delta1'] if vd.any() else None
        miss = [s for s, x in zip(scenes, c_cov) if np.isnan(x)]
        if miss:
            e['missing_scenes'] = miss
            g.notes.append(f"'{lab}': scenes without cov-TAE (skipped): {', '.join(miss)}")
        res[lab] = e
    skips = {row.get('n_skipped_frames') or 0 for row in g.rows.values()}
    if skips != {0}:
        g.notes.append(
            f"{'/'.join(str(x) for x in sorted(skips))} frames skipped in delta1/AbsRel (no valid GT) but still "
            f"counted in per-scene n, so frame-weighted delta1 uses approximate weights (max |frame-weighted delta1 - "
            f"stored aggregate| = {max(abs(e['chk_delta1_frame_w'] or 0) for e in res.values()):.4f}; the paired "
            f"differences are much less affected because ref and config share the weights). "
            f"TAE / cov-TAE are computed on all frames, so n is the right weight there.")
    g.notes.append(
        f"consistency: max |scene-mean cov-TAE - stored aggregate| = "
        f"{max(abs(e['chk_covtae_scene_mean'] or 0) for e in res.values()):.2e}")
    return res


def analyse_hard_group(g, n_boot, seed):
    ref = g.rows.get('fp32')
    res = OrderedDict()
    if ref is None:
        g.notes.append('no FP32_Baseline row'); return res
    scenes = sorted(ref['per_scene_tae'])
    rng = np.random.default_rng(seed)
    U = rng.random((n_boot, len(scenes)))
    ns = np.ones(len(scenes))
    r_tae = np.array([ref['per_scene_tae'][s] for s in scenes], dtype=float)
    for lab, row in g.rows.items():
        e = dict(alias=lab, file=g.src[lab], bits=row['effective_bits_per_scalar'],
                 delta1=row['delta1'], abs_rel=row['abs_rel'], tae=row['tae_percent'],
                 covtae=row['tae_covis_percent'], covis_fraction=row.get('covis_fraction'),
                 n_frames=row.get('n_images'), mem_savings_x=row.get('mem_savings_x'),
                 abs_rel_diff=row['abs_rel'] - ref['abs_rel'], delta1_agg_diff=row['delta1'] - ref['delta1'],
                 covtae_rel_point=100.0 * (row['tae_covis_percent'] / ref['tae_covis_percent'] - 1.0))
        c_tae = np.array([row['per_scene_tae'][s] for s in scenes], dtype=float)
        if lab != 'fp32':
            e['tae_rel'] = paired_stat(c_tae, r_tae, ns, U, 'rel')
        e['chk_tae_scene_mean'] = float(c_tae.mean()) - row['tae_percent']
        e['collapsed'] = bool(row['delta1'] < ref['delta1'] - 0.1)
        res[lab] = e
    g.notes.append('hard setting: per-scene data is raw TAE only (8 scenes, no per-scene cov-TAE / delta1); '
                   'cov-TAE and delta1 shown as aggregates without CI')
    return res


# --------------------------------------------------------------------------- formatting

def f(x, dec, signed=False):
    if x is None:
        return '-'
    s = f'{x:.{dec}f}'
    if float(s) == 0:
        return s.lstrip('-')          # never print "-0.0" / "+0.0"
    return f'{x:+.{dec}f}' if signed else s


def ci_md(st, w, dec):
    if not st or w not in st:
        return '-'
    pt = f(st[w], dec, True)
    c = st.get(w + '_ci')
    return f'{pt} [{f(c[0], dec, True)}, {f(c[1], dec, True)}]' if c else f'{pt} [n/a]'


def tex(x, dec, signed=False):
    s = f(x, dec, signed)
    if s == '-':
        return '--'
    return '$' + s + '$' if s[0] in '+-' else s


def ci_tex(st, w, dec):
    if not st or w not in st:
        return '--'
    c = st.get(w + '_ci')
    pt = tex(st[w], dec, True)
    return f'{pt} [{tex(c[0], dec, True)}, {tex(c[1], dec, True)}]' if c else f'{pt} [n/a]'


def esc(s):
    return s.replace('_', r'\_')


def sort_rows(res):
    return sorted(res.items(), key=lambda kv: (-(kv[1]['bits'] if kv[1]['bits'] is not None else 0)))


def md_stream(g, res, n_boot, seed):
    ref = res['fp32']
    L = [f'### {g.name}', '',
         f'{len(g.scenes)} scenes, {ref["n_frames"]} frames; FP32 reference: delta1 {ref["delta1"]:.4f}, '
         f'AbsRel {ref["abs_rel"]:.4f}, TAE {ref["tae"]:.3f}, cov-TAE {ref["covtae"]:.3f}, '
         f'cache {ref["cache_mb"]:.1f} MB. 95% paired percentile bootstrap over scenes '
         f'({n_boot} resamples, seed {seed}). Positive TAE change = less consistent.', '']
    if len(g.scenes) < 5:
        L += [f'> WARNING: only {len(g.scenes)} scenes: the bootstrap has at most '
              f'{len(g.scenes) ** len(g.scenes)} distinct resamples, so the CIs are coarse '
              f'and should not be read as precise.', '']
    L += ['**Scene-mean aggregation** (TAE metrics as stored by stream_eval)', '',
          '| Config | bits | cov-TAE | cov-TAE chg % [95% CI] | raw TAE chg % [95% CI] | delta1 chg [95% CI] | AbsRel (chg) |',
          '|---|---|---|---|---|---|---|']
    for lab, e in sort_rows(res):
        a = f'`{e["alias"]}`' + ('' if e['alias'] == lab else f' (`{lab}`)')
        if lab == 'fp32':
            L.append(f'| {a} | {e["bits"]:.2f} | {e["covtae"]:.3f} | ref | ref | ref | {e["abs_rel"]:.4f} |')
            continue
        L.append(f'| {a} | {e["bits"]:.2f} | {e["covtae"]:.3f} | {ci_md(e["covtae_rel"], "sm", 1)} | '
                 f'{ci_md(e["tae_rel"], "sm", 1)} | {ci_md(e["delta1_diff"], "sm", 4)} | '
                 f'{e["abs_rel"]:.4f} ({f(e["abs_rel_diff"], 4, True)}) |')
    L += ['', '**Frame-weighted aggregation** (weights = frames per scene; delta1 matches stream_eval\'s pooled aggregate)', '',
          '| Config | bits | cov-TAE chg % [95% CI] | raw TAE chg % [95% CI] | delta1 chg [95% CI] | stored delta1 chg (no CI) |',
          '|---|---|---|---|---|---|']
    for lab, e in sort_rows(res):
        if lab == 'fp32':
            continue
        L.append(f'| `{e["alias"]}` | {e["bits"]:.2f} | {ci_md(e["covtae_rel"], "fw", 1)} | '
                 f'{ci_md(e["tae_rel"], "fw", 1)} | {ci_md(e["delta1_diff"], "fw", 4)} | '
                 f'{f(e["delta1_agg_diff"], 4, True)} |')
    L.append('')
    return L


def tex_stream(g, res, w):
    wname = 'scene-mean' if w == 'sm' else 'frame-weighted'
    L = [r'\begin{table}[t]', r'\centering', r'\small',
         r'\caption{' + f'{g.name}: change vs.\\ FP32 (paired bootstrap 95\\% CI over {len(g.scenes)} scenes, '
         f'{wname} aggregation). TAE: relative change in \\%; $\\delta_1$: absolute difference.' + '}',
         r'\label{tab:boot_' + g.name.replace('.', '_') + '_' + ('sm' if w == 'sm' else 'fw') + '}',
         r'\begin{tabular}{lrrlll}', r'\toprule',
         r'Config & bits & cov-TAE & cov-TAE chg (\%) & raw TAE chg (\%) & $\delta_1$ chg \\', r'\midrule']
    for lab, e in sort_rows(res):
        if lab == 'fp32':
            L.append(f'\\texttt{{{esc(e["alias"])}}} & {e["bits"]:.2f} & {e["covtae"]:.3f} & ref & ref & ref \\\\')
            continue
        L.append(f'\\texttt{{{esc(e["alias"])}}} & {e["bits"]:.2f} & {e["covtae"]:.3f} & '
                 f'{ci_tex(e["covtae_rel"], w, 1)} & {ci_tex(e["tae_rel"], w, 1)} & '
                 f'{ci_tex(e["delta1_diff"], w, 4)} \\\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    return L


def md_hard(g, res, n_boot, seed):
    ref = res['fp32']
    L = [f'### {g.name}', '',
         f'{len(g.scenes)} scenes (offline T=32, backbone + temporal quantized). FP32: delta1 {ref["delta1"]:.4f}, '
         f'AbsRel {ref["abs_rel"]:.4f}, TAE {ref["tae"]:.2f}, cov-TAE {ref["covtae"]:.3f}. '
         f'Bits are the stored effective_bits_per_scalar. Raw-TAE CI: paired bootstrap over scenes '
         f'({n_boot} resamples, seed {seed}); cov-TAE has no per-scene data here.', '',
         '| Config | bits | delta1 | AbsRel | cov-TAE | cov-TAE chg % (point) | raw TAE | raw TAE chg % [95% CI] | note |',
         '|---|---|---|---|---|---|---|---|---|']
    for lab, e in sort_rows(res):
        if lab == 'fp32':
            L.append(f'| `fp32` | {e["bits"]:.2f} | {e["delta1"]:.4f} | {e["abs_rel"]:.4f} | {e["covtae"]:.2f} | ref | {e["tae"]:.2f} | ref | |')
            continue
        note = 'collapsed output: read with delta1' if e['collapsed'] else ''
        L.append(f'| `{lab}` | {e["bits"]:.3f} | {e["delta1"]:.4f} | {e["abs_rel"]:.4f} | {e["covtae"]:.2f} | '
                 f'{f(e["covtae_rel_point"], 1, True)} | {e["tae"]:.2f} | {ci_md(e["tae_rel"], "sm", 1)} | {note} |')
    L.append('')
    return L


def tex_hard(g, res):
    L = [r'\begin{table}[t]', r'\centering', r'\small',
         r'\caption{' + f'{g.name}: backbone and temporal cache quantized (offline, T=32). Raw-TAE change vs.\\ FP32 with '
         f'paired bootstrap 95\\% CI over {len(g.scenes)} scenes; cov-TAE has no per-scene data.' + '}',
         r'\label{tab:boot_' + g.name.replace('.', '_') + '}',
         r'\begin{tabular}{lrrrrl}', r'\toprule',
         r'Config & bits & $\delta_1$ & AbsRel & cov-TAE & raw TAE chg (\%) \\', r'\midrule']
    for lab, e in sort_rows(res):
        if lab == 'fp32':
            L.append(f'\\texttt{{fp32}} & {e["bits"]:.2f} & {e["delta1"]:.4f} & {e["abs_rel"]:.4f} & {e["covtae"]:.2f} & ref \\\\')
            continue
        star = '$^\\ast$' if e['collapsed'] else ''
        L.append(f'\\texttt{{{esc(lab)}}} & {e["bits"]:.3f} & {e["delta1"]:.4f} & {e["abs_rel"]:.4f} & '
                 f'{e["covtae"]:.2f}{star} & {ci_tex(e["tae_rel"], "sm", 1)} \\\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    return L


# --------------------------------------------------------------------------- main

def compute(files, n_boot=10000, seed=0):
    groups, mem = build_groups(files)
    out = []
    for g in groups:
        res = analyse_hard_group(g, n_boot, seed) if g.kind == 'hard' else analyse_stream_group(g, n_boot, seed)
        out.append((g, res))
    return out, mem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*', help='stream_eval / hard-setting JSON files (default: the 2026-11-02 set '
                                             '+ outputs/stream/sintel_vit{b,l}.json if present)')
    ap.add_argument('--out-dir', default=str(DEFAULT_OUT))
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    files = [Path(x) for x in a.files] or default_files()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print('inputs:'); [print('  ', p) for p in files]
    results, _ = compute(files, a.n_boot, a.seed)

    all_md = ['# Paired bootstrap CIs vs FP32 (streaming temporal-cache results)', '',
              f'Generated by scripts/bootstrap_ci.py ({a.n_boot} resamples, seed {a.seed}). '
              'Scene-mean = mean over scenes (stream_eval stores TAE this way); frame-weighted = weights are '
              'frames per scene (stream_eval\'s aggregate delta1 pools all frames). AbsRel is stored only as an '
              'aggregate, so it has no CI. CIs are percentile intervals over scene resamples; with very few scenes '
              '(Bonn: 3) they are coarse.', '']
    js = {}
    for g, res in results:
        if not res:
            print(f'[{g.name}] skipped: {"; ".join(g.notes)}')
            continue
        L = md_hard(g, res, a.n_boot, a.seed) if g.kind == 'hard' else md_stream(g, res, a.n_boot, a.seed)
        if g.notes:
            L += ['Notes:'] + [f'- {n}' for n in dict.fromkeys(g.notes)] + ['']
        stem = f'bootstrap_{g.name}'
        (out_dir / f'{stem}.md').write_text('\n'.join(L), encoding='utf-8')
        if g.kind == 'hard':
            (out_dir / f'{stem}.tex').write_text('\n'.join(tex_hard(g, res)), encoding='utf-8')
        else:
            (out_dir / f'{stem}_scenemean.tex').write_text('\n'.join(tex_stream(g, res, 'sm')), encoding='utf-8')
            (out_dir / f'{stem}_framew.tex').write_text('\n'.join(tex_stream(g, res, 'fw')), encoding='utf-8')
        all_md += L
        js[g.name] = dict(kind=g.kind, dataset=g.dataset, encoder=g.encoder, scenes=list(g.scenes),
                          notes=list(dict.fromkeys(g.notes)), rows=res)
    (out_dir / 'bootstrap_ci_all.md').write_text('\n'.join(all_md), encoding='utf-8')
    (out_dir / 'bootstrap_ci.json').write_text(json.dumps(js, indent=1, default=float), encoding='utf-8')
    print('\n'.join(all_md))
    print(f'\nwrote outputs to {out_dir}')


if __name__ == '__main__':
    main()
