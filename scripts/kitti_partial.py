"""
Fetch only the KITTI files the StreamVGGT/MonST3R video-depth benchmark reads, with HTTP
range requests into the official zips (no full download): for each of the 13 validation
drives, the first 110 ground-truth depth maps (val/<seq>/proj_depth/groundtruth/image_02)
and the matching left colour images (<date>/<seq>/image_02/data). Files are written with
their paths inside the zips under --dst, which is the layout prepare_kitti.py expects.

    python scripts/kitti_partial.py --dst <monst3r>/data/kitti [--threads 8]
"""
import argparse
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

from remotezip import RemoteZip

S3 = 'https://s3.eu-central-1.amazonaws.com/avg-kitti'
DRIVES = ['2011_09_26_drive_0002', '2011_09_26_drive_0005', '2011_09_26_drive_0013', '2011_09_26_drive_0020',
          '2011_09_26_drive_0023', '2011_09_26_drive_0036', '2011_09_26_drive_0079', '2011_09_26_drive_0095',
          '2011_09_26_drive_0113', '2011_09_28_drive_0037', '2011_09_29_drive_0026', '2011_09_30_drive_0016',
          '2011_10_03_drive_0047']
N = 110


def fetch(url, names, dst, threads):
    """Extract `names` from the remote zip at `url`, one RemoteZip per worker."""
    todo = [n for n in names if not os.path.exists(os.path.join(dst, n))]
    if not todo:
        return 0
    chunks = [todo[i::threads] for i in range(threads)]

    def work(chunk):
        done, tries = 0, 0
        pending = list(chunk)
        while pending and tries < 20:          # the server drops connections: reopen and continue
            try:
                with RemoteZip(url) as z:
                    while pending:
                        z.extract(pending[0], dst)
                        pending.pop(0)
                        done += 1
            except Exception as e:             # RemoteIOError, timeouts, resets
                tries += 1
                print(f'retry {tries} ({len(pending)} left): {str(e)[:80]}', flush=True)
                time.sleep(min(60, 5 * tries))
        if pending:
            raise RuntimeError(f'{len(pending)} files not fetched from {url}')
        return done
    with ThreadPoolExecutor(threads) as ex:
        return sum(ex.map(work, [c for c in chunks if c]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dst', required=True)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    # 1) depth maps: first N of each validation drive (prepare_kitti.py takes sorted()[:110])
    ann = f'{S3}/data_depth_annotated.zip'
    with RemoteZip(ann) as z:
        all_names = z.namelist()
    frames = {}
    depth_names = []
    for d in DRIVES:
        pre = f'val/{d}_sync/proj_depth/groundtruth/image_02/'
        sel = sorted(n for n in all_names if n.startswith(pre) and n.endswith('.png'))[:N]
        if len(sel) < N:
            print(f'WARNING {d}: only {len(sel)} depth maps in the val split')
        frames[d] = [os.path.basename(n) for n in sel]
        depth_names += sel
    print(f'depth maps: {len(depth_names)} (e.g. {depth_names[0]})', flush=True)
    print('fetched', fetch(ann, depth_names, args.dst, args.threads), 'depth maps', flush=True)

    # 2) the matching colour images from each raw drive zip
    for d in DRIVES:
        date = d[:10]
        names = [f'{date}/{d}_sync/image_02/data/{f}' for f in frames[d]]
        url = f'{S3}/raw_data/{d}/{d}_sync.zip'
        try:
            got = fetch(url, names, args.dst, args.threads)
        except (KeyError, zipfile.BadZipFile) as e:
            print(f'FAILED {d}: {e}', flush=True)
            continue
        print(f'{d}: {got} images', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
