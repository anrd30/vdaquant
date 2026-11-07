"""
Quality of StreamVGGT with a compressed cache on videos longer than the official
benchmarks, measured frame by frame against the uncompressed model.

The reference is the 'offload' cache: the exact model, with past keys and values
kept in host memory, so it does not run out of GPU memory on a long video.  Depth,
world points and camera poses are compared without ground truth; camera centres
and, where available, ground-truth positions are saved for trajectory error.

    python scripts/streamvggt_long.py --weights ckpt/checkpoints.pth --bf16 \
        --frames-dirs DATA/rgbd_bonn_crowd2/rgb --max-frames 600 --out OUT \
        --kv offload,plain:a2g64,delta:int4:bin:8,delta:int4:a2g64:8

Each config runs in its own process and writes OUT/<config>.json after every
scene.  Finished configs are skipped.  --no-compare measures time and residual
statistics only (with --trace) and needs no reference.
"""
import argparse
import gc
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

VDAQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_METRICS = ('dev', 'flicker', 'pts_dev', 'cam_err', 'rot_deg')


def tag(kv):
    return kv.replace(':', '_').replace('+', '-')


def scene_name(frames_dir):
    base = os.path.basename(frames_dir.rstrip('/\\'))
    if base.startswith('rgb'):
        return os.path.basename(os.path.dirname(frames_dir.rstrip('/\\'))).replace('rgbd_bonn_', '')
    return base


def frame_paths(frames_dir, n):
    paths = sorted(glob.glob(os.path.join(frames_dir, '*.png')) + glob.glob(os.path.join(frames_dir, '*.jpg')))
    return paths[:n]


def ground_truth(frames_dir, paths, tolerance=0.05):
    """Positions from a TUM groundtruth.txt beside the frames, nearest in time."""
    gt_file = os.path.join(os.path.dirname(frames_dir.rstrip('/\\')), 'groundtruth.txt')
    if not os.path.exists(gt_file):
        return None
    try:
        rows = np.loadtxt(gt_file, comments='#')
        stamps = [float(os.path.splitext(os.path.basename(p))[0]) for p in paths]
    except ValueError:
        return None
    out = []
    for t in stamps:
        i = int(np.abs(rows[:, 0] - t).argmin())
        out.append(rows[i, 1:4].tolist() if abs(rows[i, 0] - t) < tolerance else None)
    return out


def cameras(pose_enc, hw):
    """(S, 9) encodings -> camera centres (S, 3) and rotations (S, 3, 3)."""
    import torch
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
    ext, _ = pose_encoding_to_extri_intri(torch.as_tensor(pose_enc, dtype=torch.float32)[None], hw,
                                          build_intrinsics=False)
    rot, t = ext[0, :, :, :3], ext[0, :, :, 3]
    return (-rot.transpose(1, 2) @ t[..., None])[..., 0].numpy(), rot.numpy()


def rotation_deg(a, b):
    c = (np.trace(a @ b.T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


class Reference:
    """Frames of the reference run, on disk."""

    def __init__(self, folder):
        self.folder = folder
        pose = np.load(os.path.join(folder, 'pose.npy'))
        self.hw = tuple(np.load(os.path.join(folder, 'hw.npy')).tolist())
        self.centres, self.rots = cameras(pose, self.hw)
        self.extent = max(float(np.linalg.norm(self.centres - self.centres[0], axis=1).max()), 1e-6)
        self.n = len(pose)

    def frame(self, i):
        return (np.load(os.path.join(self.folder, f'd_{i:05d}.npy')),
                np.load(os.path.join(self.folder, f'p_{i:05d}.npy')))


def stats_of(caches):
    bits = [c.ideal_bits() for c in caches if hasattr(c, 'ideal_bits')]
    return (sum(c.nbytes() for c in caches) / 2 ** 20, float(np.mean(bits)) if bits else None)


def run_scene(model, frames_dir, args, kv, device):
    import torch
    from research.models import streamvggt_cache as kc
    from streamvggt.utils.load_fn import load_and_preprocess_images

    scene = scene_name(frames_dir)
    paths = frame_paths(frames_dir, args.max_frames)
    is_ref = kv == args.ref and not args.no_compare
    ref_dir = os.path.join(args.ref_dir, scene)
    ref = None
    if is_ref:
        os.makedirs(ref_dir, exist_ok=True)
    elif not args.no_compare:
        ref = Reference(ref_dir)
    keep_dir = None
    if scene in args.keep:
        keep_dir = os.path.join(args.out, 'depth', tag(kv), scene)
        os.makedirs(keep_dir, exist_ok=True)

    cuda = device == 'cuda'
    past = [None] * model.aggregator.depth
    past_cam = [None] * model.camera_head.trunk_depth
    series = {k: [] for k in REF_METRICS}
    ms, agg_ms, curve, poses = [], [], [], []
    prev = None
    oom = False
    kc._LIVE.pin = []
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i, path in enumerate(paths):
            images = load_and_preprocess_images([path]).to(device)[None]      # (1, 1, 3, H, W)
            if cuda:
                torch.cuda.synchronize()
            t0 = time.time()
            try:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.bf16):
                    tokens, start, past = model.aggregator(images, past_key_values=past, use_cache=True,
                                                           past_frame_idx=i)
                    if cuda:
                        torch.cuda.synchronize()
                    t1 = time.time()
                    with torch.cuda.amp.autocast(enabled=False):
                        pose, past_cam = model.camera_head(tokens, past_key_values_camera=past_cam, use_cache=True)
                        depth, _ = model.depth_head(tokens, images=images, patch_start_idx=start)
                        points, _ = model.point_head(tokens, images=images, patch_start_idx=start)
                if cuda:
                    torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                oom = True
                break
            ms.append(1000 * (time.time() - t0))
            agg_ms.append(1000 * (t1 - t0))
            d = depth[0, 0, ..., 0].float().cpu().numpy()
            p = points[0, 0].float().cpu().numpy()
            poses.append(pose[-1][0, 0].float().cpu().numpy())
            del tokens, depth, points, pose
            if is_ref:
                np.save(os.path.join(ref_dir, f'd_{i:05d}.npy'), d)
                np.save(os.path.join(ref_dir, f'p_{i:05d}.npy'), p)
            if keep_dir:
                np.save(os.path.join(keep_dir, f'd_{i:05d}.npy'), d.astype(np.float16))
            if ref is not None and i < ref.n:
                r, rp = ref.frame(i)
                denom = float(np.abs(r).mean()) + 1e-8
                series['dev'].append(100 * float(np.abs(d - r).mean()) / denom)
                if prev is not None:
                    series['flicker'].append(100 * float(np.abs((d - prev[0]) - (r - prev[1])).mean()) / denom)
                series['pts_dev'].append(100 * float(np.abs(p - rp).mean()) / (float(np.abs(rp).mean()) + 1e-8))
                prev = (d, r)
            if (i + 1) % args.log_every == 0:
                mb, _ = stats_of(kc._LIVE.pin)
                peak = torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else 0.0
                curve.append({'frames': i + 1, 'cache_mb': mb, 'peak_mb': peak, 'ms': ms[-1]})
                recent = series['dev'][-args.log_every:]
                print(f"  [{kv}] {scene} {curve[-1]}" + (f" dev {np.mean(recent):.3f}" if recent else ''), flush=True)

    hw = tuple(int(x) for x in images.shape[-2:])
    n = len(ms)
    pose_arr = np.array(poses, dtype=np.float32).reshape(-1, 9)
    if is_ref:
        np.save(os.path.join(ref_dir, 'pose.npy'), pose_arr)
        np.save(os.path.join(ref_dir, 'hw.npy'), np.array(hw))
    centres = rots = None
    if len(pose_arr):
        centres, rots = cameras(pose_arr, hw)
    if ref is not None and centres is not None:
        m = min(n, ref.n)
        series['cam_err'] = [100 * float(x) / ref.extent
                             for x in np.linalg.norm(centres[:m] - ref.centres[:m], axis=1)]
        series['rot_deg'] = [rotation_deg(rots[i], ref.rots[i]) for i in range(m)]
    cache_mb, bits = stats_of(kc._LIVE.pin)
    trace = [c.trace for c in kc._LIVE.pin if getattr(c, 'trace', None)]
    result = {'frames': n, 'frames_available': len(paths), 'oom': oom, 'hw': hw, 'series': series,
              'ms': ms, 'aggregator_ms': agg_ms, 'curve': curve, 'cache_mb': cache_mb, 'bits': bits,
              'peak_mb': torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else 0.0,
              'centres': centres.tolist() if centres is not None else [],
              'ground_truth': ground_truth(frames_dir, paths[:n]) if is_ref else None}
    if trace:
        result['trace'] = trace
    kc._LIVE.pin = None
    del past, past_cam
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return scene, result


def run_config(args, kv):
    import torch
    from research.models import streamvggt_cache as kc
    kc.TRACE = args.trace
    kc.install(kv)
    from streamvggt.models.streamvggt import StreamVGGT

    device = args.device
    model = StreamVGGT()
    if not args.random_init:
        model.load_state_dict(torch.load(args.weights, map_location='cpu'), strict=True)
    model = model.to(device).eval()
    out = {'kv': kv, 'scenes': {}}
    path = os.path.join(args.out, f'{tag(kv)}.json')
    for frames_dir in args.frames_dirs:
        scene, res = run_scene(model, frames_dir, args, kv, device)
        out['scenes'][scene] = res
        with open(path + '.tmp', 'w') as fh:
            json.dump(out, fh)
        os.replace(path + '.tmp', path)
        dev = res['series']['dev']
        tail = dev[-50:]
        print(f"{kv} {scene}: {res['frames']}/{res['frames_available']} frames"
              f"{' OOM' if res['oom'] else ''} bits {res['bits']} cache {res['cache_mb']:.0f} MB "
              f"peak {res['peak_mb']:.0f} MB median {np.median(res['ms'][-50:]) if res['ms'] else 0:.0f} ms"
              + (f" dev {np.mean(dev):.3f}% last50 {np.mean(tail):.3f}%" if dev else ''), flush=True)
    open(os.path.join(args.out, f'{tag(kv)}.done'), 'w').close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='')
    ap.add_argument('--frames-dirs', nargs='+', required=True)
    ap.add_argument('--kv', default='offload,plain:a2g64,delta:int4:bin:8,delta:int4:a2g64:8')
    ap.add_argument('--ref', default='offload')
    ap.add_argument('--ref-dir', default=None)
    ap.add_argument('--max-frames', type=int, default=240)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--keep', nargs='*', default=[], help='scenes whose depth maps are kept, for videos')
    ap.add_argument('--out', required=True)
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--no-compare', action='store_true')
    ap.add_argument('--trace', action='store_true')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--random-init', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--single', default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)
    args.ref_dir = os.path.abspath(args.ref_dir or os.path.join(args.out, 'ref'))
    args.frames_dirs = [os.path.abspath(d) for d in args.frames_dirs]
    svg_src = os.environ.get('STREAMVGGT_SRC', os.path.abspath('.'))
    for p in (svg_src, VDAQ):
        sys.path.insert(0, p)

    if args.single:                       # child process: one config
        run_config(args, args.single)
        return
    os.makedirs(args.out, exist_ok=True)  # fail now, not after hours of work
    probe = os.path.join(args.out, '.writable')
    open(probe, 'w').close()
    os.remove(probe)
    for d in args.frames_dirs:
        if not frame_paths(d, 1):
            raise SystemExit(f'no frames in {d}')
    failed = []
    for kv in args.kv.split(','):         # fresh process per config: clean GPU state
        if os.path.exists(os.path.join(args.out, f'{tag(kv)}.done')):
            print('skip', kv, '(done)', flush=True)
            continue
        print(f"[{time.strftime('%H:%M')}] {kv}", flush=True)
        cmd = [sys.executable, '-u', os.path.abspath(__file__), '--single', kv] + \
              [a for a in sys.argv[1:]]
        if subprocess.run(cmd, env=os.environ).returncode:
            failed.append(kv)
            print('FAILED', kv, flush=True)
    if failed:
        raise SystemExit('failed: ' + ' '.join(failed))


if __name__ == '__main__':
    main()
