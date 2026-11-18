"""Helpers for reproducing VDA's own ScanNet TAE (scripts/covtae_vda_official.sh).
    extract <scannet export root> <benchmark dir>   VDA's extract_scannet (benchmark/dataset_extract) on our export
    gtpred  <benchmark dir> <pred dir>              ground-truth disparity saved where VDA's infer.py saves predictions
    swapjson <benchmark dir>                        eval_tae.py reads scannet/scannet_video.json, which VDA's extractor
                                                    writes without poses; the TAE list is scannet_video_tae.json
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
cmd = sys.argv[1]
if cmd == 'extract':
    sys.path.insert(0, str(REPO / 'Video-Depth-Anything' / 'benchmark' / 'dataset_extract'))
    from dataset_extract_scannet import extract_scannet
    extract_scannet(root=sys.argv[2], sample_len=-1, datatset_name='scannet', saved_dir=sys.argv[3])
elif cmd == 'gtpred':
    bench, pred = Path(sys.argv[2]) / 'scannet', Path(sys.argv[3]) / 'scannet'
    data = json.load(open(bench / 'scannet_video_tae.json'))['scannet']
    n = 0
    for scene in data:
        for _, frames in scene.items():
            for fr in frames:
                d = cv2.imread(str(bench / fr['gt_depth']), -1).astype(np.float64) / fr['factor']
                disp = np.where(d > 0, 1.0 / np.maximum(d, 1e-8), 0.0).astype(np.float32)
                out = pred / fr['image'].replace('.jpg', '.npy').replace('.png', '.npy')
                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, disp)
                n += 1
    print(f'gtpred: {n} frames')
elif cmd == 'swapjson':
    b = Path(sys.argv[2]) / 'scannet'
    if not (b / 'scannet_video_90.json').exists():
        shutil.copy(b / 'scannet_video.json', b / 'scannet_video_90.json')
    shutil.copy(b / 'scannet_video_tae.json', b / 'scannet_video.json')
    print('eval json = scannet_video_tae.json (original kept as scannet_video_90.json)')
else:
    raise SystemExit(__doc__)
