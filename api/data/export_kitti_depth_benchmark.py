"""Script to copy the Kitti Benchmark depth maps into the Kitti Raw Sync folder structure."""
import shutil

from tqdm import tqdm

import src.devkits.kitti_raw as kr
from src import DATA_PATHS


def main():
    TARGET_DIR = 'depth_benchmark'
    K_RAW, K_DEPTH = DATA_PATHS['kitti_raw'], DATA_PATHS['kitti_depth']
    print(f'-> Exporting Kitti Benchmark from "{K_DEPTH}" to "{K_RAW}"...')

    ROOT = K_RAW/TARGET_DIR
    ROOT.mkdir(exist_ok=True)
    for seq in kr.SEQS: (ROOT/seq).mkdir(exist_ok=True)

    for mode in ('train', 'val'):
        for dir in tqdm(sorted((K_DEPTH/mode).iterdir())):
            seq = next(s for s in kr.SEQS if dir.stem.startswith(s))
            shutil.copytree(dir, ROOT/seq/dir.stem, dirs_exist_ok=True)


if __name__ == '__main__':
    main()
