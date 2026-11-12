"""
Fidelity of StreamVGGT's 3D points to the uncompressed model on the mv_recon
benchmark (7-Scenes, NRGBD), from the per-scene point maps the official
evaluator saves (before its ICP step).  Deviation = mean |p - p_ref| / mean |p_ref|
over the valid mask, in percent, per scene; reported as the mean over scenes.

    python scripts/recon_fidelity.py <svg_recon dir> [--out recon_fidelity.json]
"""
import argparse
import glob
import json
import os

import numpy as np

METHODS = ['plain_a2g64', 'kivi_2_64', 'delta_int4_bin_8', 'delta_int4_a2g64_8']
LABEL = {'plain_a2g64': 'INT2 asym (2.5 b)', 'kivi_2_64': 'KIVI (2.5 b)',
         'delta_int4_bin_8': 'TCC-1 (1.9 b)', 'delta_int4_a2g64_8': 'TCC-2 (2.75 b)'}


def load(path):
    d = np.load(path, allow_pickle=True).item()
    return d['pts_all'].astype(np.float64), d['masks_all'] > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = {}
    for setting in ('sparse', 'dense'):
        for data in ('7scenes', 'NRGBD'):
            refs = sorted(glob.glob(os.path.join(args.root, f'{setting}_none', data, '*.npy')))
            if not refs:
                continue
            row = {}
            for m in METHODS:
                devs = {}
                for r in refs:
                    p = os.path.join(args.root, f'{setting}_{m}', data, os.path.basename(r))
                    if not os.path.exists(p):
                        continue
                    pr, mr = load(r)
                    pc, mc = load(p)
                    if pr.shape != pc.shape:
                        print('shape mismatch', p, pr.shape, pc.shape)
                        continue
                    mask = mr & mc
                    devs[os.path.basename(r)[:-4]] = float(
                        100 * np.abs(pc[mask] - pr[mask]).mean() / (np.abs(pr[mask]).mean() + 1e-12))
                row[m] = {'mean': float(np.mean(list(devs.values()))) if devs else None,
                          'median': float(np.median(list(devs.values()))) if devs else None,
                          'scenes': devs, 'frames': None}
            n_frames = [load(r)[0].shape[0] for r in refs]
            out[f'{setting}/{data}'] = {'n_scenes': len(refs), 'frames_per_scene': n_frames, 'methods': row}
            print(f'== {setting} {data}: {len(refs)} scenes, frames per scene {min(n_frames)}-{max(n_frames)}')
            for m in METHODS:
                v = row[m]
                if v['mean'] is not None:
                    print(f'   {LABEL[m]:20s} point deviation mean {v["mean"]:6.2f}%  median {v["median"]:6.2f}%')
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
