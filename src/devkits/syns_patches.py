from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from numpy.typing import NDArray

from src.utils.io import readlines
from . import PATHS

__all__ = [
    'get_scenes', 'get_scene_files', 'get_info_file',
    'get_image_file', 'get_depth_file', 'get_edges_file',
    'load_info', 'load_category', 'load_split',
]


# CONSTANTS
# -----------------------------------------------------------------------------
if 'syns_patches' not in PATHS:
    raise KeyError(f"Missing 'syns_patches' root paths. Got {list(PATHS)}.")

SUBDIRS = [
    'images', 'masks', 'depths', 'edges',
    'edges_01', 'edges_01_log', 'edges_01_inv',
    'edges_02', 'edges_02_log', 'edges_02_inv',
    'edges_03', 'edges_03_log', 'edges_03_inv',
]


KITTI_FOV = (25.46, 84.10)
KITTI_SHAPE = (376, 1242)
# -----------------------------------------------------------------------------


# PATH BUILDING
# -----------------------------------------------------------------------------
def get_split_file(mode: str) -> Path:
    """Get scene information file based on the scene number."""
    file = PATHS['syns_patches']/'splits'/f'{mode}_files.txt'
    return file


def get_scenes() -> list[Path]:
    """Get paths to each of the scenes."""
    return sorted(path for path in (PATHS['syns_patches']).iterdir() if path.is_dir() and path.stem != 'splits')


def get_scene_files(scene_dir: Path) -> dict[str, Sequence[Path]]:
    """Get paths to all subdir files for a given scene."""
    files = {key: sorted((scene_dir/key).iterdir()) for key in SUBDIRS if (scene_dir/key).is_dir()}
    return files


def get_info_file(scene: str) -> Path:
    """Get scene information file based on the scene number."""
    paths = (PATHS['syns_patches']/scene).iterdir()
    return next(f for f in paths if f.suffix == '.txt')


def get_image_file(scene: str, file: str) -> Path:
    """Get image filename based on scene and item number."""
    return PATHS['syns_patches']/scene/'images'/file


def get_depth_file(scene: str, file: str) -> Path:
    """Get image filename based on scene and item number."""
    return (PATHS['syns_patches']/scene/'depths'/file).with_suffix('.npy')


def get_edges_file(scene: str, subdir: str, file: str) -> Path:
    """Get image filename based on scene and item number."""
    assert 'edges' in subdir, f'Must provide an "edges" directory. ({subdir})'
    assert subdir in SUBDIRS, f'Non-existent edges directory. ({subdir} vs. {[s for s in SUBDIRS if "edges" in s]})'
    return PATHS['syns_patches']/scene/subdir/file
# -----------------------------------------------------------------------------


# LOADING
# -----------------------------------------------------------------------------
def load_info(scene: str) -> Sequence[str]:
    """Load the scene information."""
    file = get_info_file(scene)
    info = readlines(file, encoding='latin-1')
    return info


def load_category(scene: str) -> tuple[str, str]:
    """Load the scene category and subcategory."""
    info = load_info(scene)
    category = info[1].replace('Scene Category: ', '')
    try:
        cat, subcat = category.split(': ')
    except ValueError:
        cat, subcat = category.split(' - ')

    return cat, subcat


def load_split(mode) -> tuple[Path, list[list[str]]]:
    """Load the list of scenes and filenames that are part of the test split.

    Test split file is given as "SEQ ITEM":
    ```
    01 00.png
    10 11.png
    ```
    """
    file = get_split_file(mode)
    lines = readlines(file)
    lines = [l.split(' ') for l in lines]
    return file, lines


def load_intrinsics() -> NDArray:
    """Computes the virtual camera intrinsics for the `Kitti` based SYNS Patches.

    We compute this based on the desired FOV, using basic trigonometry.

    :return: (ndarray) (4, 4) Camera intrinsic parameters.
    """
    Fy, Fx = KITTI_FOV
    h, w = KITTI_SHAPE

    cx, cy = w//2, h//2
    fx = cx / np.tan(np.deg2rad(Fx)/2)
    fy = cy / np.tan(np.deg2rad(Fy)/2)

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)
    return K
# -----------------------------------------------------------------------------


def main():
    import matplotlib.pyplot as plt

    for scene in get_scenes():
        print(scene.stem)
        _, axs11 = plt.subplots(3, 3)
        plt.tight_layout()

        _, axs12 = plt.subplots(3, 3)
        plt.tight_layout()

        _, axs21 = plt.subplots(3, 3)
        plt.tight_layout()

        _, axs22 = plt.subplots(3, 3)
        plt.tight_layout()

        axs1 = np.concatenate((axs11.flatten(), axs12.flatten()))
        axs2 = np.concatenate((axs21.flatten(), axs22.flatten()))

        images = sorted((scene/'images').iterdir())
        depths = sorted((scene/'depth_images').iterdir())

        [ax.cla() for ax in axs1]
        [ax.cla() for ax in axs2]

        [ax.imshow(Image.open(f)) for ax, f in zip(axs1, images)]
        [ax.imshow(Image.open(f)) for ax, f in zip(axs2, depths)]
        [ax.set_title(i) for i, ax in enumerate(axs1)]

        plt.show()


if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
