"""Supplement: how redundant is VDA's temporal cache, per layer, dataset, and frame stride, and is the redundancy
temporal or across layers?

For every layer l of the FP32 cache (8 = 4 temporal modules x 2 attention layers) and every scene:
  temporal       sum_t ||h_t - h_{t-1}||^2 / sum_t ||h_t - mu_t||^2    (mu_t: per-channel mean over tokens)
                 = the energy a previous-frame predictor (TCC) leaves, relative to the entry's variance
  temporal_fit   the same with the best per-channel affine map a*h_{t-1}+b (least squares on the scene itself)
  cross_layer    the best per-channel affine map from the other layer of the same module at the same frame
                 (AQUA-KV / CORAL-KV style), fitted on the scene itself, i.e. an oracle that favours it;
                 only within a module, where the two layers have the same shape (modules differ in channels
                 and resolution)
Ratios are in %. The model is rerun for every stride (it sees different history).

    python scripts/supp_redundancy.py --datasets sintel bonn scannet --encoders vits vitl --strides 1 2 4 8 \
        --out results/supp_redundancy.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO))
import stream_eval as se  # noqa: E402  (VDA import path, model configs, reset_stream)
from datasets_gt import group_samples_by_scene, load_gt_dataset  # noqa: E402


class AffineSSE:
    """Per-channel least-squares y ~ a*x + b from running sums; also the plain sums for the ratios."""

    def __init__(self):
        self.s = None

    def add(self, x, y):
        x, y = x.double().reshape(-1, x.shape[-1]), y.double().reshape(-1, y.shape[-1])
        v = torch.stack([x.sum(0), y.sum(0), (x * x).sum(0), (x * y).sum(0), (y * y).sum(0),
                         torch.full_like(x[0], x.shape[0])])
        self.s = v if self.s is None else self.s + v

    def sse_fit(self):
        sx, sy, sxx, sxy, syy, n = self.s
        cov = sxy - sx * sy / n
        var = (sxx - sx * sx / n).clamp(min=1e-30)
        return float((syy - sy * sy / n - cov * cov / var).clamp(min=0).sum())


def scene_stats(model, frames, size):
    se.reset_stream(model)
    L = None
    prev = None
    e_diff = e_var = None
    tfit = xfit = None
    with torch.no_grad():
        for f in frames:
            model.infer_video_depth_one(f['rgb'], input_size=size, device='cuda')
            cur = [t.detach().float() for t in model.frame_cache_list[-1]]
            if L is None:
                L = len(cur)
                e_diff, e_var, e_vt = np.zeros(L), np.zeros(L), np.zeros(L)
                tfit = [AffineSSE() for _ in range(L)]
                xfit = [AffineSSE() for _ in range(L)]
            for l, h in enumerate(cur):
                hv = h.reshape(-1, h.shape[-1]).double()
                v = float((hv - hv.mean(0, keepdim=True)).pow(2).sum())
                e_var[l] += v
                if prev is not None:
                    e_vt[l] += v
                    e_diff[l] += float((h.double() - prev[l].double()).pow(2).sum())
                    tfit[l].add(prev[l], h)
                if l % 2 == 1 and cur[l - 1].shape == h.shape:      # other layer of the same module
                    xfit[l].add(cur[l - 1], h)
            prev = cur
    n = len(frames)
    var_t = e_vt                                                     # variance of the frames with a predecessor
    return dict(temporal=[100 * a / b for a, b in zip(e_diff, var_t)],
                temporal_fit=[100 * t.sse_fit() / b for t, b in zip(tfit, var_t)],
                cross_layer=[100 * x.sse_fit() / b if x.s is not None else None for x, b in zip(xfit, e_var)],
                e_diff=e_diff.tolist(), e_var=var_t.tolist(), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['sintel', 'bonn', 'scannet'])
    ap.add_argument('--encoders', nargs='+', default=['vits', 'vitl'])
    ap.add_argument('--strides', nargs='+', type=int, default=[1, 2, 4, 8])
    ap.add_argument('--max-per-scene', type=int, default=300)
    ap.add_argument('--size', type=int, default=518)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = json.load(open(a.out)) if Path(a.out).exists() else {}
    for ds in a.datasets:
        samples, _ = load_gt_dataset(ds, REPO / 'benchmark_data', max_samples=None, require_cam=False)
        scenes = {k: v[:a.max_per_scene] for k, v in group_samples_by_scene(samples).items()}
        for enc in a.encoders:
            model = se.VideoDepthAnything(**se.CFGS[enc])
            model.load_state_dict(torch.load(REPO / 'checkpoints' / f'video_depth_anything_{enc}.pth',
                                             map_location='cpu'), strict=True)
            model = model.cuda().eval()
            for st in a.strides:
                t0, per = time.time(), {}
                for name, frames in scenes.items():
                    fr = frames[::st]
                    if len(fr) >= 4:
                        per[name] = scene_stats(model, fr, a.size)
                if not per:
                    continue
                L = len(next(iter(per.values()))['temporal'])
                pool = lambda key: [100 * sum(p['e_diff'][l] for p in per.values()) /
                                    sum(p['e_var'][l] for p in per.values()) for l in range(L)] \
                    if key == 'temporal' else None
                agg = dict(
                    temporal_pooled=pool('temporal'),
                    temporal_mean=[float(np.mean([p['temporal'][l] for p in per.values()])) for l in range(L)],
                    temporal_fit_mean=[float(np.mean([p['temporal_fit'][l] for p in per.values()])) for l in range(L)],
                    cross_layer_mean=[float(np.mean([p['cross_layer'][l] for p in per.values()]))
                                      if per[next(iter(per))]['cross_layer'][l] is not None else None
                                      for l in range(L)],
                    per_scene={k: dict(temporal=p['temporal'], temporal_fit=p['temporal_fit'],
                                       cross_layer=p['cross_layer'], n=p['n']) for k, p in per.items()},
                    n_scenes=len(per), minutes=(time.time() - t0) / 60)
                out.setdefault(ds, {}).setdefault(enc, {})[str(st)] = agg
                print(f"{ds} {enc} stride {st}: {len(per)} scenes, temporal % per layer "
                      f"{[round(x, 1) for x in agg['temporal_pooled']]}, cross-layer (oracle) "
                      f"{[None if x is None else round(x, 1) for x in agg['cross_layer_mean']]}", flush=True)
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                json.dump(out, open(a.out, 'w'), indent=1)
            del model
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
