"""
Streaming-mode accuracy / temporal-consistency evaluation with a compressed
temporal cache.

VDA is run frame by frame through ``infer_video_depth_one`` (its streaming
path, which keeps the last frames' temporal-attention inputs in a cache).
Metrics reuse the existing suite code unchanged:
  * per-frame affine-invariant delta1 / AbsRel  (datasets_gt.compute_gt_depth_metrics)
  * scene-pooled geometric TAE and co-visibility TAE
    (run_pareto_benchmark_suite.compute_tae_geometric_for_scene)

    python scripts/stream_eval.py --dataset sintel --encoder vits \
        --configs fp32,fp16,bw16:4,bw16:3 --out outputs/stream/sintel_vits.json
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
for p in (REPO, SCRIPT_DIR, REPO / 'Video-Depth-Anything'):
    sys.path.insert(0, str(p))

from datasets_gt import compute_gt_depth_metrics, group_samples_by_scene, load_gt_dataset  # noqa: E402
import run_pareto_benchmark_suite as _suite  # noqa: E402
from run_pareto_benchmark_suite import compute_tae_geometric_for_scene  # noqa: E402


def _pool_align_lowmem(pred_disps, gt_depths, gt_range):
    """Same least-squares fit as the suite's _pool_align_scene_disparity
    (scale, shift minimising ||s*pred + t - 1/gt||^2 over all valid pixels of
    the scene), solved via 2x2 normal equations accumulated per frame instead
    of one lstsq over every pixel (which needs several GB of RAM on long scenes)."""
    S = torch.zeros(2, 2, dtype=torch.float64)
    v = torch.zeros(2, dtype=torch.float64)
    n_valid = 0
    for pred, gt in zip(pred_disps, gt_depths):
        mask = (gt > 1e-3) & (gt < gt_range[1])
        if not mask.any():
            continue
        p = pred[mask].double()
        g = 1.0 / gt[mask].double().clamp(min=1e-8)
        S += torch.tensor([[float((p * p).sum()), float(p.sum())],
                           [float(p.sum()), float(p.numel())]], dtype=torch.float64)
        v += torch.tensor([float((p * g).sum()), float(g.sum())], dtype=torch.float64)
        n_valid += p.numel()
    if n_valid == 0:
        raise ValueError("No valid pixels across the scene for pooled disparity alignment")
    sol = torch.linalg.solve(S, v)
    return float(sol[0]), float(sol[1])


_suite._pool_align_scene_disparity = _pool_align_lowmem


def seq_aligned_metrics(preds, gts, gt_range, shift=True):
    """delta1 and AbsRel with ONE alignment per sequence (instead of per frame): the
    predicted disparity is mapped by s*d + t (shift=True, VDA's video protocol) or
    s*d (shift=False), fitted by least squares over all valid pixels of the sequence,
    then converted to depth and clipped to gt_range. Averaged over frames."""
    gts_t = [torch.from_numpy(g).float() for g in gts]
    if shift:
        s, t = _pool_align_lowmem(preds, gts_t, gt_range)
    else:
        num = den = 0.0
        for p, g in zip(preds, gts_t):
            m = (g > 1e-3) & (g < gt_range[1])
            num += float((p[m].double() * (1.0 / g[m].double())).sum())
            den += float((p[m].double() ** 2).sum())
        s, t = num / max(den, 1e-12), 0.0
    d1, ar = [], []
    for p, g in zip(preds, gts_t):
        m = (g > 1e-3) & (g < gt_range[1])
        if not m.any():
            continue
        disp = (s * p.double() + t).clamp(min=1.0 / gt_range[1])
        dep = (1.0 / disp).clamp(gt_range[0], gt_range[1])[m]
        gg = g[m].double()
        d1.append(float((torch.maximum(dep / gg, gg / dep) < 1.25).double().mean()))
        ar.append(float(((dep - gg).abs() / gg).mean()))
    return float(np.mean(d1)), float(np.mean(ar))

from video_depth_anything.video_depth_stream import VideoDepthAnything  # noqa: E402
from research.models.stream_cache import cache_nbytes, enable_compressed_cache  # noqa: E402

CFGS = {
    'vits': dict(encoder='vits', features=64, out_channels=[48, 96, 192, 384]),
    'vitb': dict(encoder='vitb', features=128, out_channels=[96, 192, 384, 768]),
    'vitl': dict(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]),
}


# FP32 predictions per scene (fp16 copies), filled by the first 'fp32' config of
# this process and used for ground-truth-free fidelity metrics of later configs.
_REF = {}


def fidelity(preds, ref):
    """Fidelity of a compressed run to the uncompressed (FP32-cache) run.
    rel_dev  : mean_t  mean|d_t - r_t| / mean|r_t|
    flicker  : mean_t  mean|(d_t - d_{t-1}) - (r_t - r_{t-1})| / mean|r_t|
    Both in %, no ground truth and no alignment (same model, same output scale)."""
    dev, flick = [], []
    prev_d = prev_r = None
    for d, r in zip(preds, ref):
        d = d.numpy().astype(np.float32)
        r = r.astype(np.float32)
        denom = np.abs(r).mean() + 1e-8
        dev.append(np.abs(d - r).mean() / denom)
        if prev_d is not None:
            flick.append(np.abs((d - prev_d) - (r - prev_r)).mean() / denom)
        prev_d, prev_r = d, r
    return 100 * float(np.mean(dev)), 100 * float(np.mean(flick)) if flick else 0.0


def reset_stream(model):
    model.transform = None
    model.frame_id_list = []
    model.frame_cache_list = []
    model.id = -1


_LINFIT = {}
_MOD_REF = {}


def linfit_params(args):
    """Fit the 'linfit' predictor (a*rec[t-1] + b*rec[t-2] + c per layer and channel) on the FP32 caches
    of another dataset (args.linfit_calib), at most 60 frames per scene and args.linfit_frames in all."""
    key = (args.encoder, args.linfit_calib, args.linfit_frames)
    if key in _LINFIT:
        return _LINFIT[key]
    from research.models.stream_cache import fit_linear_predictor
    assert args.linfit_calib and args.linfit_calib != args.dataset, 'calibrate on a different dataset'
    samples, _ = load_gt_dataset(args.linfit_calib, REPO / 'benchmark_data', max_samples=None, require_cam=False)
    model = VideoDepthAnything(**CFGS[args.encoder])
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'), strict=True)
    model = model.cuda().eval()
    seqs, total = [], 0
    with torch.no_grad():
        for name, frames in group_samples_by_scene(samples).items():
            frames = frames[:min(60, args.linfit_frames - total)]
            if len(frames) < 4:
                break
            reset_stream(model)
            seq = []
            for f in frames:
                model.infer_video_depth_one(f['rgb'], input_size=args.size, device='cuda')
                seq.append([t.detach().float().cpu() for t in model.frame_cache_list[-1]])
            seqs.append(seq)
            total += len(frames)
    del model
    torch.cuda.empty_cache()
    params = fit_linear_predictor(seqs)
    _LINFIT[key] = params
    print(f"    linfit: fitted on {args.linfit_calib}, {len(seqs)} scenes, {total} frames; "
          f"mean a {np.mean([float(p[0].mean()) for p in params]):.3f} "
          f"b {np.mean([float(p[1].mean()) for p in params]):.3f}", flush=True)
    return params


def _module_hooks(model, store):
    """Keep the latest output of each temporal module (VDA's head has four, two attention layers each)."""
    hs = []
    for k, mod in enumerate(model.head.motion_modules):
        def hook(_m, _inp, out, k=k):
            store[k] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
        hs.append(mod.register_forward_hook(hook))
    return hs


def run_config(args, mode, bits, scenes, gt_range, delta=0, residual_mode=None, age_spec=None,
               predictor='prev', drop=1, window=0, adaptive=None):
    gc.collect()
    torch.cuda.empty_cache()
    params = linfit_params(args) if predictor == 'linfit' else None
    model = VideoDepthAnything(**CFGS[args.encoder])
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'), strict=True)
    model = model.cuda().eval()
    if age_spec:
        from research.models.stream_age import enable_age_quant, parse_age_spec, plan_bits
        plan = parse_age_spec(age_spec)
        model = enable_age_quant(model, plan)
        mode = 'age'
    else:
        model = enable_compressed_cache(model, mode=mode, bits=bits, seed=args.seed, delta=delta,
                                        residual_mode=residual_mode, drop=drop, window=window,
                                        **({'predictor': predictor} if delta else {}),
                                        **({'predictor_params': params} if params is not None else {}),
                                        **({'adaptive': adaptive} if adaptive is not None else {}))
    mod_out = {}
    hooks = _module_hooks(model, mod_out) if args.module_err else []
    mod_err = {}

    acc, tae, tae_cov, cov_frac, per_scene = [], [], [], [], {}
    seq_all = []
    is_ref = mode == 'fp32' and not delta and drop == 1 and not window
    fid_dev, fid_flk = [], []
    n_skip, peak, cache_mb, t0 = 0, 0, 0.0, time.time()
    for name, frames in scenes.items():
        reset_stream(model)
        torch.cuda.reset_peak_memory_stats()
        preds = []
        for fi, f in enumerate(frames):
            d = model.infer_video_depth_one(f['rgb'], input_size=args.size, device='cuda')
            p = torch.from_numpy(np.ascontiguousarray(d)).float()
            if tuple(p.shape) != tuple(f['depth'].shape):   # ScanNet: 1296x968 colour, 640x480 depth (as eval_tae.py)
                p = F.interpolate(p[None, None], size=f['depth'].shape, mode='bilinear', align_corners=False)[0, 0]
            preds.append(p)
            if args.module_err and fi % args.module_err == 0:
                for k, o in mod_out.items():
                    if is_ref:
                        _MOD_REF[(name, fi, k)] = o.half().cpu()
                    elif (name, fi, k) in _MOD_REF:
                        r = _MOD_REF[(name, fi, k)].to(o.device).float()
                        mod_err.setdefault(k, []).append(float((o.float() - r).norm() / (r.norm() + 1e-12)))
        peak = max(peak, torch.cuda.max_memory_allocated())
        cache_mb = max(cache_mb, cache_nbytes(model) / 2 ** 20)
        if is_ref:
            _REF[name] = [p.numpy().astype(np.float16) for p in preds]
        fid = fidelity(preds, _REF[name]) if (name in _REF and not is_ref) else None
        if fid:
            fid_dev.append(fid[0])
            fid_flk.append(fid[1])
        if args.save_preds:                                   # qualitative figures (supplement)
            import cv2
            sd = Path(args.save_preds) / f"{args.dataset}_{args.encoder}"
            sd.mkdir(parents=True, exist_ok=True)
            tag = 'cfg_' + ''.join(ch if ch.isalnum() else '_' for ch in (age_spec or f'{mode}_{delta}_{residual_mode}_{predictor}_{adaptive}_{window}'))
            keep = list(range(0, len(preds), args.save_every))
            np.savez_compressed(sd / f"{name}__{tag}.npz", idx=np.array(keep),
                                pred=np.stack([preds[i].numpy().astype(np.float16) for i in keep]))
            gfile = sd / f"{name}__gt.npz"
            if not gfile.exists():
                rgb = [cv2.resize(np.asarray(frames[i]['rgb']), (frames[i]['depth'].shape[1], frames[i]['depth'].shape[0]))
                       for i in keep]
                np.savez_compressed(gfile, idx=np.array(keep), rgb=np.stack(rgb).astype(np.uint8),
                                    gt=np.stack([frames[i]['depth'].astype(np.float16) for i in keep]))
        dev_frames = None
        if args.per_frame and name in _REF and not is_ref:
            dev_frames = [round(100 * float(np.abs(d.numpy() - r.astype(np.float32)).mean()
                                            / (np.abs(r.astype(np.float32)).mean() + 1e-8)), 4)
                          for d, r in zip(preds, _REF[name])]
        gts = [f['depth'] for f in frames]
        scene_acc = []
        for pred, gt in zip(preds, gts):
            gt_t = torch.from_numpy(gt).float()
            valid = torch.from_numpy((gt > 1e-3) & (gt < gt_range[1]))
            try:
                scene_acc.append(compute_gt_depth_metrics(pred, gt_t, valid, gt_range=gt_range,
                                                          pred_is_disparity=True))
            except ValueError:
                n_skip += 1
        acc.extend(scene_acc)
        seq_ss = seq_aligned_metrics(preds, gts, gt_range, shift=True)
        seq_s = seq_aligned_metrics(preds, gts, gt_range, shift=False)
        seq_all.append(seq_ss + seq_s)
        try:
            if args.no_tae:
                raise ValueError('TAE skipped')
            r = compute_tae_geometric_for_scene(preds, gts, [f['K'] for f in frames],
                                                [f['pose'] for f in frames], gt_range,
                                                covis_tau=args.covis_tau)
            tae.append(r['tae_percent'])
            tae_cov.append(r['tae_covis_percent'])
            cov_frac.append(r['covis_fraction'])
            per_scene[name] = {
                'delta1': float(np.mean([m['delta1'] for m in scene_acc])) if scene_acc else None,
                'tae': r['tae_percent'], 'tae_covis': r['tae_covis_percent'], 'n': len(frames),
                'delta1_seq': seq_ss[0], 'abs_rel_seq': seq_ss[1], 'delta1_seq_scale': seq_s[0], 'abs_rel_seq_scale': seq_s[1],
                'fid_rel_dev': fid[0] if fid else None, 'fid_flicker': fid[1] if fid else None}
        except (ValueError, TypeError, AttributeError):     # no camera data (e.g. KITTI drives)
            per_scene[name] = {
                'delta1': float(np.mean([m['delta1'] for m in scene_acc])) if scene_acc else None,
                'tae': None, 'tae_covis': None, 'n': len(frames),
                'delta1_seq': seq_ss[0], 'abs_rel_seq': seq_ss[1], 'delta1_seq_scale': seq_s[0], 'abs_rel_seq_scale': seq_s[1],
                'fid_rel_dev': fid[0] if fid else None, 'fid_flicker': fid[1] if fid else None}
        if dev_frames is not None:
            per_scene[name]['dev_per_frame'] = dev_frames
        print(f"    {name:24s} n={len(frames):3d} d1={per_scene[name]['delta1']}", flush=True)

    res = {
        'mode': mode, 'bits': bits if mode == 'bw16' else None,
        'delta1': float(np.mean([m['delta1'] for m in acc])),
        'abs_rel': float(np.mean([m['abs_rel'] for m in acc])),
        'delta1_seq': float(np.mean([x[0] for x in seq_all])),
        'abs_rel_seq': float(np.mean([x[1] for x in seq_all])),
        'delta1_seq_scale': float(np.mean([x[2] for x in seq_all])),
        'abs_rel_seq_scale': float(np.mean([x[3] for x in seq_all])),
        'tae_percent': float(np.mean(tae)) if tae else None,
        'tae_covis_percent': float(np.mean(tae_cov)) if tae_cov else None,
        'covis_fraction': float(np.mean(cov_frac)) if cov_frac else None,
        'fid_rel_dev_pct': float(np.mean(fid_dev)) if fid_dev else None,
        'fid_flicker_pct': float(np.mean(fid_flk)) if fid_flk else None,
        'n_frames': len(acc), 'n_skipped_frames': n_skip,
        'peak_mb': peak / 2 ** 20, 'cache_mb': cache_mb,
        'minutes': (time.time() - t0) / 60, 'per_scene': per_scene,
        'plan_bits': plan_bits(plan) if age_spec else None,
    }
    enc = getattr(model, '_delta_encoder', None)
    if enc is not None:
        res['keyframe_fraction'] = enc.n_intra / max(enc.n_entries, 1)
    if adaptive is not None:
        res['adaptive'] = adaptive
    if args.module_err and mod_err:
        res['module_rel_err_pct'] = {str(k): 100 * float(np.mean(v)) for k, v in sorted(mod_err.items())}
    for h in hooks:
        h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='sintel', choices=['sintel', 'bonn', 'bonn5', 'kittiseq', 'scannet'])
    ap.add_argument('--encoder', default='vits', choices=list(CFGS))
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--configs', default='fp32,fp16,bw16:4,bw16:3')
    ap.add_argument('--size', type=int, default=518)
    ap.add_argument('--covis-tau', type=float, default=0.05)
    ap.add_argument('--max-scenes', type=int, default=1000)
    ap.add_argument('--max-samples', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-tae', action='store_true', help='skip the camera-based TAE (slow)')
    ap.add_argument('--out', default=None)
    # supplement experiments (all off by default; the paper's runs use none of them)
    ap.add_argument('--stride', type=int, default=1, help='keep every n-th frame of each scene')
    ap.add_argument('--splice', type=int, default=0,
                    help='n>0: streams of the first n frames of scene 2i followed by the first n of scene 2i+1 '
                         '(an abrupt cut); ground-truth metrics across the cut are not meaningful, use fidelity')
    ap.add_argument('--per-frame', action='store_true', help='store the per-frame deviation of every scene')
    ap.add_argument('--module-err', type=int, default=0,
                    help='n>0: relative error of each temporal module output against FP32, every n-th frame')
    ap.add_argument('--linfit-calib', default=None,
                    help="dataset whose FP32 caches fit the 'linfit' predictor (use a different one than --dataset)")
    ap.add_argument('--linfit-frames', type=int, default=150)
    ap.add_argument('--save-preds', default=None,
                    help='dir: save float16 depth of every --save-every-th frame per scene and config (+ RGB, GT)')
    ap.add_argument('--save-every', type=int, default=40)
    args = ap.parse_args()
    args.ckpt = args.ckpt or str(REPO / 'checkpoints' / f'video_depth_anything_{args.encoder}.pth')

    samples, gt_range = load_gt_dataset(args.dataset, REPO / 'benchmark_data',
                                        max_samples=args.max_samples, require_cam=True)
    groups = group_samples_by_scene(samples)
    scenes = {k: groups[k] for k in list(groups)[:args.max_scenes]}
    if args.stride > 1:
        scenes = {k: v[::args.stride] for k, v in scenes.items() if len(v[::args.stride]) >= 4}
    if args.splice:
        names = list(scenes)
        scenes = {f'{a}+{b}': scenes[a][:args.splice] + scenes[b][:args.splice]
                  for a, b in zip(names[0::2], names[1::2])}
    print(f"[{args.dataset}] {len(scenes)} scenes, {sum(len(v) for v in scenes.values())} frames, "
          f"gt_range={gt_range}", flush=True)

    results = []
    for cfg in args.configs.split(','):
        if cfg.startswith('age:'):
            print(f"  config {cfg}", flush=True)
            r = run_config(args, 'age', 3, scenes, gt_range, age_spec=cfg)
            r['config'] = cfg
            results.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != 'per_scene'}), flush=True)
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                with open(args.out, 'w') as fh:
                    json.dump({'args': vars(args), 'results': results}, fh, indent=2)
            continue
        spec, drop = (cfg.split('@drop') + ['1'])[:2]    # e.g. int4-g16@drop2: keep every 2nd frame
        drop = int(drop)
        spec, window = (spec.split('@win') + ['0'])[:2]  # e.g. int4-g16@win16: keep the last 16 frames
        window = int(window)
        parts = spec.split(':')
        mode = parts[0]
        b = int(parts[1]) if len(parts) > 1 and parts[1] else 3
        dk = int(parts[2][1:]) if len(parts) > 2 and parts[2] else 0
        rmode = parts[3] if len(parts) > 3 and parts[3] else None
        pred, adaptive = 'prev', None                        # predictor: prev|none|extrap2|linfit|open
        for tok in parts[4:]:                                # aX: adaptive keyframe at residual/variance > X
            if tok.startswith('a') and tok[1:].replace('.', '', 1).isdigit():
                adaptive = float(tok[1:])
            elif tok:
                pred = tok
        print(f"  config {cfg}", flush=True)
        r = run_config(args, mode, b, scenes, gt_range, delta=dk, residual_mode=rmode, predictor=pred,
                       drop=drop, window=window, adaptive=adaptive)
        r['config'] = cfg
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != 'per_scene'}), flush=True)
        if args.out:  # save after every config so a crash keeps finished results
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, 'w') as fh:
                json.dump({'args': vars(args), 'results': results}, fh, indent=2)

    print(f"\n{args.dataset} {args.encoder} streaming")
    print(f"{'config':14s} {'delta1':>7s} {'AbsRel':>7s} {'TAE':>7s} {'covTAE':>7s} {'cacheMB':>8s} {'peakMB':>7s}")
    for r in results:
        name = r.get('config', r['mode'])
        print(f"{name:14s} {r['delta1']:7.4f} {r['abs_rel']:7.4f} {r['tae_percent'] or 0:7.2f} "
              f"{r['tae_covis_percent'] or 0:7.2f} {r['cache_mb']:8.1f} {r['peak_mb']:7.1f} "
              f"fid_dev={r.get('fid_rel_dev_pct') or 0:.3f}% fid_flicker={r.get('fid_flicker_pct') or 0:.3f}%")


if __name__ == '__main__':
    main()
