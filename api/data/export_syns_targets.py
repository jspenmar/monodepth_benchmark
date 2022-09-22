"""Script to export the SYNS ground-truth evaluation targets.

NOTE: Only the `val` ground-truth is publicly available!

This dataset will produce a targets file with the following variables.
    - depth: (b, h, w) Ground-truth depths.
    - edge: (b, h, w) Ground-truth depth discontinuities.
    - K: (b, 4, 4) Camera intrinsic parameters.
    - cat: (b,) Scene category (for additional grouping during evaluation).
    - subcat: (b,) Scene subcategory (for additional grouping during evaluation).
"""
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from src.datasets import SYNSPatchesDataset


def save(file: Path, **kwargs) -> None:
    """Save a list of arrays as a npz file."""
    print(f'-> Saving to "{file}"...')
    np.savez_compressed(file,  **kwargs)


def export_syns(mode, save_stem: Optional[str] = None, overwrite: bool = False) -> None:
    """Export the ground truth LiDAR depth images for SYNS.

    :param save_stem: (Optional[str]) Exported depth file stem (i.e. no suffix).
    :param overwrite: (bool) If `True`, overwrite existing exported files.
    """
    print(f'-> Exporting ground truth depths for SYNS "{mode}"...')

    dataset = SYNSPatchesDataset(mode, use_depth=True, use_edges=True, as_torch=False)

    save_file = dataset.split_file.parent/f'{save_stem}.npz'
    if not overwrite and save_file.is_file():
        raise FileExistsError(f'Target file "{save_file}" already exists. Set flag `--overwrite 1` to overwrite')

    depths, edges, Ks, cats, subcats = [], [], [], [], []
    for _, y, m in tqdm(dataset):
        depths.append(y['depth'].squeeze())
        Ks.append(y['K'])
        edges.append(y['edges'].squeeze())
        cats.append(m['cat'])
        subcats.append(m['subcat'])

    save(save_file, depth=depths, K=Ks, edge=edges, cat=cats, subcat=subcats)


if __name__ == '__main__':
    parser = ArgumentParser('Script to export a target depth dataset as a npz file.')
    parser.add_argument('--mode', default='val', choices={'val'}, help='Split mode to use.')
    parser.add_argument('--save-stem', default=None, help='Exported targets file stem (i.e. without suffix).')
    parser.add_argument('--overwrite', default=0, type=int, help='If 1, overwrite existing exported files.')
    args = parser.parse_args()

    if args.save_stem is None: args.save_stem = f'targets_{args.mode}'
    export_syns(args.mode, args.save_stem, args.overwrite)
