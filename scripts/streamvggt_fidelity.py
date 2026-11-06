"""
Fidelity of compressed StreamVGGT runs to the uncompressed run, using the same
definition as scripts/stream_eval.py (no ground truth, no alignment):

    rel_dev : mean_t  mean|d_t - r_t| / mean|r_t|
    flicker : mean_t  mean|(d_t - d_{t-1}) - (r_t - r_{t-1})| / mean|r_t|

Inputs are the frame_*.npy depth maps written by StreamVGGT's video-depth
launcher, one folder per sequence.

    python scripts/streamvggt_fidelity.py --ref eval_results/video_depth/sintel_none \
        --runs eval_results/video_depth/sintel_* --out sintel_fidelity.json

A sequence missing from a run (the launcher skips sequences that run out of
memory) is listed under 'missing' and is left out of that run's means.
"""
import argparse
import glob
import json
import os

import numpy as np


def frames(seq_dir):
    return sorted(glob.glob(os.path.join(seq_dir, 'frame_*.npy')))


def fidelity(pred_paths, ref_paths):
    """(rel_dev %, flicker %) for one sequence."""
    dev, flick = [], []
    prev_d = prev_r = None
    for pd, pr in zip(pred_paths, ref_paths):
        d = np.load(pd).astype(np.float32)
        r = np.load(pr).astype(np.float32)
        denom = np.abs(r).mean() + 1e-8
        dev.append(np.abs(d - r).mean() / denom)
        if prev_d is not None:
            flick.append(np.abs((d - prev_d) - (r - prev_r)).mean() / denom)
        prev_d, prev_r = d, r
    return 100 * float(np.mean(dev)), 100 * float(np.mean(flick)) if flick else 0.0


def sequences(run_dir):
    return sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(run_dir, '*', 'frame_0000.npy')))


def load_stats(run_dir):
    path = os.path.join(run_dir, 'kv_stats.json')
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        st = json.load(fh)
    seqs = st.get('sequences', [])
    if not seqs:
        return {'kv': st.get('kv')}
    return {'kv': st.get('kv'),
            'ideal_bits': float(np.mean([s['ideal_bits'] for s in seqs])),
            'cache_mb_max': max(s['cache_bytes'] for s in seqs) / 2 ** 20,
            'peak_mb_max': max(s['peak_mb'] for s in seqs),
            'seconds_per_frame': sum(s['seconds'] for s in seqs) / max(1, sum(s['frames'] for s in seqs))}


def compare(run_dir, ref_dir):
    ref_seqs = sequences(ref_dir)
    per_seq, missing = {}, []
    for seq in ref_seqs:
        pred = frames(os.path.join(run_dir, seq))
        ref = frames(os.path.join(ref_dir, seq))
        if not pred or len(pred) != len(ref):
            missing.append(seq)
            continue
        dev, flk = fidelity(pred, ref)
        per_seq[seq] = {'frames': len(ref), 'fid_rel_dev_pct': dev, 'fid_flicker_pct': flk}
    out = {'run': os.path.basename(run_dir.rstrip('/\\')), 'n_sequences': len(per_seq),
           'n_reference': len(ref_seqs), 'missing': missing, **load_stats(run_dir)}
    if per_seq:
        out['fid_rel_dev_pct'] = float(np.mean([v['fid_rel_dev_pct'] for v in per_seq.values()]))
        out['fid_flicker_pct'] = float(np.mean([v['fid_flicker_pct'] for v in per_seq.values()]))
    out['per_sequence'] = per_seq
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if not sequences(args.ref):
        raise SystemExit(f'no frame_0000.npy under {args.ref}')
    runs = [r for r in args.runs if os.path.isdir(r)]
    results = [compare(r, args.ref) for r in runs]
    with open(args.out, 'w') as fh:
        json.dump({'reference': args.ref, 'results': results}, fh, indent=2)
    print(f"{'run':34s} {'bits':>5s} {'dev%':>8s} {'flick%':>8s} {'seqs':>7s} {'peakMB':>8s}")
    for r in results:
        print(f"{r['run'][:34]:34s} {r.get('ideal_bits') or 0:5.2f} {r.get('fid_rel_dev_pct', float('nan')):8.3f} "
              f"{r.get('fid_flicker_pct', float('nan')):8.3f} {r['n_sequences']:3d}/{r['n_reference']:<3d} "
              f"{r.get('peak_mb_max') or 0:8.0f}")


if __name__ == '__main__':
    main()
