"""Script to export the Kitti ground-truth evaluation targets.

This dataset will produce a targets file with the following variables.
    - depth: (b,) (h, w) Ground-truth depths (each of a different size).
    - K: (b, 4, 4) Camera intrinsic parameters.
"""
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

import src.devkits.kitti_raw as kr


def save(file: Path, **kwargs) -> None:
    """Save a list of arrays as a npz file."""
    print(f'\n -> Saving to "{file}"...')
    np.savez_compressed(file,  **kwargs)


def export_kitti(depth_split: str,
                 mode: str,
                 use_velo_depth: bool = False,
                 save_stem: Optional[str] = None,
                 overwrite: bool = False) -> None:
    """Export the ground truth LiDAR depth images for a given Kitti test split.

    :param depth_split: (str) Kitti depth split to load.
    :param mode: (str) Split mode to use. {'train', 'val', 'test'}
    :param use_velo_depth: (bool) If `True`, load the raw velodyne depth. Only used for legacy reasons!
    :param save_stem: (Optional[str]) Exported depth file stem (i.e. no suffix).
    :param overwrite: (bool) If `True`, overwrite existing exported files.
    """
    print(f'\n-> Exporting ground truth depths for KITTI "{depth_split}/{mode}"...')

    split_file = kr.get_split_file(depth_split, mode='test')
    lines = [line.split() for line in kr.load_split(split_file)]
    items = [{
        'seq': l[0],
        'cam': 2 if l[2] == 'l' else 3,
        'stem': int(l[1])
    } for l in lines]

    save_file = split_file.parent/f'{save_stem}.npz'
    if not overwrite and save_file.is_file():
        raise FileExistsError(f'Target file "{save_file}" already exists. Set flag `--overwrite 1` to overwrite')

    depths, Ks = [], []
    for d in tqdm(items):
        cam2cam, _, velo2cam = kr.load_calib(d['seq'].split('/')[0])

        if use_velo_depth:
            file = kr.get_velodyne_file(d['seq'], d['stem'])

            # NOTE: By default, when using LiDAR ground truth, we also use the raw LiDAR depth.
            # This matches what original papers (Eigen, Garg, Monodepth...) used, even though it is incorrect.
            # In all other cases, the improved benchmark depth should be used.
            depth = kr.load_depth_velodyne(file, velo2cam, cam2cam, cam=d['cam'], use_velo_depth=use_velo_depth)
        else:
            file = kr.get_depth_file(d['seq'], f'image_0{d["cam"]}', d['stem'])
            depth = kr.load_depth(file)

        depths.append(depth)
        Ks.append(cam2cam[f'K_0{d["cam"]}'])

    depths = np.array(depths, dtype=object)  # Each image is a slightly different size.
    save(save_file, depth=depths, K=Ks)


if __name__ == '__main__':
    parser = ArgumentParser('Script to export a target depth dataset as a npz file.')
    parser.add_argument('--split', required=True, choices={'eigen', 'eigen_benchmark', 'eigen_zhou'}, help='Kitti depth split to load.')
    parser.add_argument('--mode', default='test', choices={'train', 'val', 'test'}, help='Mode to use.')
    parser.add_argument('--use-velo-depth', default=None, type=int, help='If 1, use the original Kitti Velodyne LiDAR.')
    parser.add_argument('--save-stem', default=None, help='Exported targets file stem (i.e. without suffix).')
    parser.add_argument('--overwrite', default=0, type=int, help='If 1, overwrite existing exported files.')
    args = parser.parse_args()

    # NOTE: Kitti Eigen is the only dataset that should be generated using the original velodyne.
    # Furthermore, it is also expected to use the raw depth from the velodyne.
    # This is incorrect, and is only kept for compatibility with the commonly reported Kitti Eigen evaluation.
    # You should be using the Kitti Eigen-Benchmark split anyway!

    if args.use_velo_depth is None: args.use_velo_depth = args.split == 'eigen'
    if args.save_stem is None: args.save_stem = f'targets_{args.mode}'
    export_kitti(args.split, args.mode, args.use_velo_depth, args.save_stem, args.overwrite)
