"""Export frames from a ScanNet .sens file (Python 3, no dependencies beyond numpy and OpenCV).

Writes the layout of ScanNet's own SensReader, which VDA's benchmark/dataset_extract_scannet.py reads:
    <out>/color/<i>.jpg      color frame, the JPEG bytes stored in the .sens file (not re-encoded)
    <out>/depth/<i>.png      uint16 depth in millimetres (depth_shift 1000)
    <out>/pose/<i>.txt       4x4 camera-to-world pose (may contain -inf where tracking failed)
    <out>/intrinsic/intrinsic_{color,depth}.txt, extrinsic_{color,depth}.txt   4x4 matrices

    python scripts/scannet_sens_export.py --sens scene0707_00.sens --out benchmark_data/scannet/scene0707_00 \
        --max-frames 510

.sens layout (ScanNet SensReader, version 4): header, then per frame a 4x4 float32 camera-to-world pose,
color and depth timestamps (uint64), color and depth byte counts (uint64), the color bytes, the depth bytes.
"""
import argparse
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np

COLOR_JPEG, DEPTH_ZLIB = 2, 1


def _read(f, fmt):
    n = struct.calcsize(fmt)
    b = f.read(n)
    if len(b) != n:
        raise EOFError('truncated .sens file')
    return struct.unpack(fmt, b)


def read_header(f):
    (version,) = _read(f, '<I')
    if version != 4:
        raise ValueError(f'unsupported .sens version {version} (expected 4)')
    (strlen,) = _read(f, '<Q')
    sensor = f.read(strlen).decode('utf8', 'replace')
    mats = [np.array(_read(f, '<16f'), dtype=np.float32).reshape(4, 4) for _ in range(4)]
    color_comp, depth_comp = _read(f, '<ii')
    cw, ch, dw, dh = _read(f, '<IIII')
    (depth_shift,) = _read(f, '<f')
    (n_frames,) = _read(f, '<Q')
    return dict(sensor=sensor, intrinsic_color=mats[0], extrinsic_color=mats[1], intrinsic_depth=mats[2],
                extrinsic_depth=mats[3], color_comp=color_comp, depth_comp=depth_comp, color_size=(cw, ch),
                depth_size=(dw, dh), depth_shift=depth_shift, n_frames=n_frames)


def read_frames(f, hdr, max_frames=None):
    n = hdr['n_frames'] if max_frames is None else min(hdr['n_frames'], max_frames)
    dw, dh = hdr['depth_size']
    for i in range(n):
        pose = np.array(_read(f, '<16f'), dtype=np.float32).reshape(4, 4)
        _read(f, '<QQ')                                    # timestamps (unused)
        csize, dsize = _read(f, '<QQ')
        color = f.read(csize)
        depth = f.read(dsize)
        if len(color) != csize or len(depth) != dsize:
            raise EOFError(f'truncated .sens file at frame {i}')
        if hdr['depth_comp'] == DEPTH_ZLIB:
            depth = zlib.decompress(depth)
        elif hdr['depth_comp'] != 0:
            raise ValueError(f'unsupported depth compression {hdr["depth_comp"]}')
        d = np.frombuffer(depth, dtype=np.uint16).reshape(dh, dw)
        yield i, pose, color, d


def export(sens, out, max_frames=None):
    out = Path(out)
    for sub in ('color', 'depth', 'pose', 'intrinsic'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    with open(sens, 'rb') as f:
        hdr = read_header(f)
        if hdr['color_comp'] != COLOR_JPEG:
            raise ValueError(f'unsupported color compression {hdr["color_comp"]} (expected JPEG)')
        if abs(hdr['depth_shift'] - 1000.0) > 1e-3:
            raise ValueError(f'depth_shift {hdr["depth_shift"]} (loader assumes millimetres)')
        for name in ('intrinsic_color', 'extrinsic_color', 'intrinsic_depth', 'extrinsic_depth'):
            np.savetxt(out / 'intrinsic' / f'{name}.txt', hdr[name], fmt='%.9g')
        n_bad = n = 0
        for i, pose, color, depth in read_frames(f, hdr, max_frames):
            (out / 'color' / f'{i}.jpg').write_bytes(color)
            if not cv2.imwrite(str(out / 'depth' / f'{i}.png'), depth):
                raise IOError(f'could not write depth frame {i}')
            np.savetxt(out / 'pose' / f'{i}.txt', pose, fmt='%.9g')
            n_bad += not np.isfinite(pose).all()
            n += 1
    print(f'{Path(sens).name}: {n} of {hdr["n_frames"]} frames -> {out} (color {hdr["color_size"]}, '
          f'depth {hdr["depth_size"]}, {n_bad} frames without a valid pose)')
    return n, n_bad


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sens', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-frames', type=int, default=510, help='VDA extracts the first 510 frames')
    a = ap.parse_args()
    export(a.sens, a.out, a.max_frames)
