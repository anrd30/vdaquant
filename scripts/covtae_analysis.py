"""
cov-TAE v2: every TAE variant and every control, computed offline from saved predictions.

Inputs: the .npz/.json files written by run_pareto_benchmark_suite.py --eval-mode temporal
--dump-preds DIR (raw predicted disparity per frame, float16), plus the dataset's ground truth,
intrinsics and poses (loaded with the same loader, so frame order matches; checked).

Each "variant" is a list of per-frame disparities per scene that goes through the SAME pipeline:
one least-squares scale/shift per scene in disparity (as the main script and VDA's eval_tae.py),
inversion and clipping to [1e-3, max depth], then bidirectional reprojection between consecutive
frames with the per-target z-buffer of the main script.

Variants
  <tag>            each dumped model configuration (e.g. fp32, 2bit, fp32 with window 1)
  fp32 perturbed   blur (sigma px), per-frame scale jitter (log-normal sigma), per-pixel noise,
                   per-frame constant (median) disparity -- applied to the raw disparity
  gt controls      ground truth itself, and the same perturbations applied to ground truth

Metrics per direction, all |d2 - proj| / d2 over pixels where proj > 0 and d2 > 0, plus:
  tae          no further mask (VDA eval_tae.py without --mask)
  tae_gtvalid  target ground truth inside (1e-3, max)
  tae_noclip   neither depth at the clipping bounds
  cov<tau>     ground-truth co-visibility mask (reprojected GT agrees with target GT within tau)
  noncov       complement of cov0.05 (the excluded pixels only)
  cov_gtden    cov0.05 with ground truth in the denominator
  top1share    share of the tae error sum carried by the 1% largest pixel errors
Scene value = mean over pairs and both directions (as eval_tae.py); medians over pairs are kept too.

    python scripts/covtae_analysis.py --dataset sintel --encoder vits --dumps DIR --out FILE.json
"""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets_gt import group_samples_by_scene, load_gt_dataset  # noqa: E402

TAUS = (0.01, 0.02, 0.05, 0.1, 0.2)
LO = 1e-3
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


# ------------------------------------------------------------------ geometry
def reproject(d1, R, t, K, zbuffer=True):
    """Depth of frame-1 pixels seen from frame 2 (0 where nothing lands). zbuffer=True mirrors
    run_pareto_benchmark_suite._tae_geometric_single (nearest surface wins); zbuffer=False is VDA's
    eval_tae.py exactly (depth_proj[Y, X] = Z: last write wins, order not defined on GPU)."""
    H, W = d1.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    yy, xx = torch.meshgrid(torch.arange(H, dtype=d1.dtype, device=d1.device),
                            torch.arange(W, dtype=d1.dtype, device=d1.device), indexing='ij')
    P = torch.stack(((xx - cx) * d1 / fx, (yy - cy) * d1 / fy, d1), -1).reshape(-1, 3) @ R.T + t
    Z = P[:, 2]
    u = torch.round(P[:, 0] * fx / Z + cx).long()
    v = torch.round(P[:, 1] * fy / Z + cy).long()
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not zbuffer:
        out = torch.zeros((H, W), dtype=d1.dtype, device=d1.device)
        out[v[ok], u[ok]] = Z[ok]
        return out
    flat = torch.full((H * W,), float('inf'), dtype=d1.dtype, device=d1.device)
    if ok.any():
        flat.scatter_reduce_(0, v[ok] * W + u[ok], Z[ok], reduce='amin', include_self=True)
    flat[torch.isinf(flat)] = 0.0
    return flat.reshape(H, W)


def rel_pose(p1, p2):
    T21 = np.linalg.inv(p2) @ p1
    T12 = np.linalg.inv(T21)
    f = lambda T: (torch.tensor(T[:3, :3], device=DEV), torch.tensor(T[:3, 3], device=DEV))
    return f(T21), f(T12)


# ------------------------------------------------------------------ alignment
def align_scene(disps, gts, dmax):
    """One scale/shift per scene in disparity (main script's _pool_align_scene_disparity),
    then invert and clip to [1e-3, dmax]."""
    p, g = [], []
    for d, gt in zip(disps, gts):
        m = (gt > LO) & (gt < dmax)
        if m.any():
            p.append(d[m])
            g.append(1.0 / gt[m].clamp(min=1e-8))
    p, g = torch.cat(p), torch.cat(g)
    A = torch.stack((p, torch.ones_like(p)), 1)
    s, t = torch.linalg.lstsq(A, g.unsqueeze(1)).solution[:, 0].tolist()
    return [(1.0 / (s * d + t).clamp(min=LO)).clamp(LO, dmax) for d in disps], (s, t)


def delta1_metrics(disps, gts, depths, dmax):
    """delta1 with the scene alignment used for TAE, and with one alignment per frame."""
    sc, pf = [], []
    for d, gt, dep in zip(disps, gts, depths):
        m = (gt > LO) & (gt < dmax)
        if m.sum() < 10:
            continue
        g = gt[m]
        sc.append(float((torch.maximum(dep[m] / g, g / dep[m]) < 1.25).double().mean()))
        A = torch.stack((d[m], torch.ones_like(d[m])), 1)
        s, t = torch.linalg.lstsq(A, (1.0 / g).unsqueeze(1)).solution[:, 0].tolist()
        dd = (1.0 / (s * d[m] + t).clamp(min=LO)).clamp(LO, dmax)
        pf.append(float((torch.maximum(dd / g, g / dd) < 1.25).double().mean()))
    return (float(np.mean(sc)) if sc else None), (float(np.mean(pf)) if pf else None)


# ------------------------------------------------------------------ perturbations (on disparity)
def gblur(x, sigma):
    r = int(np.ceil(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2)
    k = k / k.sum()
    y = F.pad(x[None, None], (r, r, r, r), mode='replicate')
    y = F.conv2d(y, k.view(1, 1, 1, -1))
    return F.conv2d(y, k.view(1, 1, -1, 1))[0, 0]


def perturb(disps, kind, amount, seed):
    g = torch.Generator(device='cpu').manual_seed(seed)
    out = []
    for d in disps:
        if kind == 'blur':
            out.append(gblur(d, amount))
        elif kind == 'jit':        # one multiplicative scale per frame
            out.append(d * float(torch.exp(amount * torch.randn(1, generator=g))))
        elif kind == 'noise':      # independent multiplicative noise per pixel
            out.append(d * torch.exp(amount * torch.randn(d.shape, generator=g)).to(d))
        elif kind == 'const':      # every pixel of a frame set to the frame's median
            out.append(torch.full_like(d, float(d[d > 0].median()) if (d > 0).any() else 1.0))
        else:
            raise ValueError(kind)
    return out


# ------------------------------------------------------------------ per-scene evaluation
def gt_codes(gts, Ks, poses):
    """Per pair and direction: smallest tau index at which the pixel is co-visible (255 = never)."""
    codes = []
    for i in range(len(gts) - 1):
        (R21, t21), (R12, t12) = rel_pose(poses[i], poses[i + 1])
        K = torch.tensor(Ks[i], device=DEV, dtype=torch.float64)
        pair = []
        for src, dst, R, t in ((gts[i], gts[i + 1], R21, t21), (gts[i + 1], gts[i], R12, t12)):
            proj = reproject(src, R, t, K)
            rel = (proj - dst).abs() / dst.clamp(min=1e-8)
            ok = (proj > 0) & (dst > 0)
            c = torch.full(dst.shape, 255, dtype=torch.uint8, device=DEV)
            for k in reversed(range(len(TAUS))):
                c[ok & (rel < TAUS[k])] = k
            pair.append(c)
        codes.append(pair)
    return codes


def scene_tae(depths, gts, Ks, poses, codes, dmax, depths_up=None):
    """depths_up: the same prediction aligned exactly as eval_tae.py (raw prediction clipped to >= 1e-3
    before the fit); used for 'tae_upstream' with eval_tae.py's last-write-wins reprojection."""
    keys = ['tae', 'tae_gtvalid', 'tae_noclip', 'noncov', 'cov_gtden', 'top1share', 'tae_upstream'] + \
           [f'cov{t:g}' for t in TAUS]
    per = {k: [] for k in keys}
    frac = {f'cov{t:g}': [] for t in TAUS}
    i05 = TAUS.index(0.05)
    for i in range(len(depths) - 1):
        (R21, t21), (R12, t12) = rel_pose(poses[i], poses[i + 1])
        K = torch.tensor(Ks[i], device=DEV, dtype=torch.float64)
        vals = {k: [] for k in keys}
        if depths_up is not None:
            for a1, a2, R, t in ((depths_up[i], depths_up[i + 1], R21, t21), (depths_up[i + 1], depths_up[i], R12, t12)):
                pu = reproject(a1, R, t, K, zbuffer=False)
                mu = (pu > 0) & (a2 > 0)
                vals['tae_upstream'].append(float(((a2 - pu).abs() / a2)[mu].mean()) if mu.any() else 0.0)
        for d1, d2, g2, R, t, c in ((depths[i], depths[i + 1], gts[i + 1], R21, t21, codes[i][0]),
                                    (depths[i + 1], depths[i], gts[i], R12, t12, codes[i][1])):
            proj = reproject(d1, R, t, K)
            base = (proj > 0) & (d2 > 0)
            err = (d2 - proj).abs() / d2
            inr = lambda x: (x > LO * 1.01) & (x < dmax * 0.999)

            def mean_over(m, e=err):
                return float(e[m].mean()) if m.any() else 0.0    # eval_tae.py returns 0 when empty
            vals['tae'].append(mean_over(base))
            vals['tae_gtvalid'].append(mean_over(base & (g2 > LO) & (g2 < dmax)))
            vals['tae_noclip'].append(mean_over(base & inr(d2) & inr(proj)))
            cov05 = c <= i05
            vals['noncov'].append(mean_over(base & ~cov05))
            vals['cov_gtden'].append(mean_over(base & cov05 & (g2 > 0), (d2 - proj).abs() / g2.clamp(min=1e-8)))
            e = err[base]
            if e.numel():
                k = max(1, e.numel() // 100)
                vals['top1share'].append(float(e.topk(k).values.sum() / e.sum().clamp(min=1e-30)))
            else:
                vals['top1share'].append(0.0)
            for j, tau in enumerate(TAUS):
                m = c <= j
                vals[f'cov{tau:g}'].append(mean_over(base & m))
                frac[f'cov{tau:g}'].append(float(m.double().mean()))
        for k in keys:
            if vals[k]:
                per[k].append(sum(vals[k]) / 2.0)
    out = {k: {'mean': 100 * float(np.mean(v)), 'median': 100 * float(np.median(v))} for k, v in per.items() if v}
    out['top1share'] = {kk: vv / 100 for kk, vv in out['top1share'].items()}   # a share, not a percent
    out['covis_fraction'] = {k: float(np.mean(v)) for k, v in frac.items()}
    out['n_pairs'] = len(depths) - 1
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--dumps', required=True)
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-controls', action='store_true')
    args = ap.parse_args()

    dumps = sorted(Path(args.dumps).glob(f'{args.dataset}_{args.encoder}_w*_*.json'))
    if not dumps:
        raise SystemExit(f'no dumps for {args.dataset}/{args.encoder} in {args.dumps}')
    meta0 = json.loads(dumps[0].read_text())
    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parents[1] / 'benchmark_data'
    samples, gt_range = load_gt_dataset(args.dataset, data_dir, max_samples=meta0['max_samples'], require_cam=True)
    scenes = group_samples_by_scene(samples)
    dmax = float(gt_range[1])
    names = [s for s in scenes if s in meta0['scenes']]
    print(f'{args.dataset}/{args.encoder}: {len(names)} scenes, dumps: {[d.stem for d in dumps]}', flush=True)

    npz = {d.stem: np.load(d.with_suffix('.npz')) for d in dumps}
    result = {'dataset': args.dataset, 'encoder': args.encoder, 'gt_range': list(gt_range), 'taus': list(TAUS),
              'variants': {}}
    t0 = time.time()
    for scene in names:
        fr = scenes[scene]
        for stem, d in zip([x.stem for x in dumps], dumps):
            n = json.loads(d.read_text())['scenes'][scene]
            assert n == len(fr), (stem, scene, n, len(fr))
        gts = [torch.from_numpy(np.asarray(f['depth'], dtype=np.float64)).to(DEV) for f in fr]
        Ks = [np.asarray(f['K'], dtype=np.float64) for f in fr]
        poses = [np.asarray(f['pose'], dtype=np.float64) for f in fr]
        codes = gt_codes(gts, Ks, poses)
        seed = zlib.crc32(scene.encode())

        # variants are built one at a time (a long Bonn scene holds ~2.4 GB per variant)
        load = lambda stem: [torch.from_numpy(npz[stem][f'{scene}|{i:05d}'].astype(np.float64)).to(DEV)
                             for i in range(len(fr))]
        variants = []
        for stem in npz:
            tag = stem[len(f'{args.dataset}_{args.encoder}_'):]       # e.g. w32_fp32, w32_2bit, w1_fp32
            variants.append((tag, lambda stem=stem: load(stem)))
        base_stem = f'{args.dataset}_{args.encoder}_w32_fp32'
        if not args.no_controls:
            gt_disp = lambda: [torch.where(g > 0, 1.0 / g.clamp(min=1e-8), torch.zeros_like(g)) for g in gts]
            srcs = [('gt', gt_disp)] + ([('fp32', lambda: load(base_stem))] if base_stem in npz else [])
            for src_name, src in srcs:
                if src_name == 'gt':
                    variants.append(('gt', src))
                for s in ((0.5, 1, 2, 4, 8) if src_name == 'fp32' else (2, 8)):
                    variants.append((f'{src_name}+blur{s:g}', lambda src=src, s=s: perturb(src(), 'blur', float(s), seed)))
                for s in (0.01, 0.02, 0.05, 0.1):
                    variants.append((f'{src_name}+jit{s:g}', lambda src=src, s=s: perturb(src(), 'jit', s, seed)))
                if src_name == 'fp32':
                    for s in (0.02, 0.05):
                        variants.append((f'{src_name}+noise{s:g}', lambda src=src, s=s: perturb(src(), 'noise', s, seed)))
                variants.append((f'{src_name}+const', lambda src=src: perturb(src(), 'const', 0, seed)))

        for tag, make in variants:
            disps = make()
            depths, st = align_scene(disps, gts, dmax)
            depths_up, _ = align_scene([d.clamp(min=LO) for d in disps], gts, dmax)   # eval_tae.py clips first
            r = scene_tae(depths, gts, Ks, poses, codes, dmax, depths_up)
            del depths_up
            r['delta1_scene_aligned'], r['delta1_frame_aligned'] = delta1_metrics(disps, gts, depths, dmax)
            r['align'] = st
            result['variants'].setdefault(tag, {})[scene] = r
            del disps, depths
        print(f'  {scene}: {len(fr)} frames, {len(variants)} variants, {time.time() - t0:.0f}s', flush=True)
        del codes, gts
        torch.cuda.empty_cache() if DEV == 'cuda' else None

    # aggregates over scenes: mean and median of scene values
    agg = {}
    for tag, per_scene in result['variants'].items():
        a = {}
        for k in next(iter(per_scene.values())):
            v = [ps[k] for ps in per_scene.values()]
            if isinstance(v[0], dict) and 'mean' in v[0]:
                sv = [x['mean'] for x in v]
                a[k] = {'mean_of_scenes': float(np.mean(sv)), 'median_of_scenes': float(np.median(sv))}
            elif isinstance(v[0], (int, float)) and v[0] is not None:
                a[k] = float(np.mean([x for x in v if x is not None]))
        agg[tag] = a
    result['aggregate'] = agg
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result))
    print('wrote', args.out, f'({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
