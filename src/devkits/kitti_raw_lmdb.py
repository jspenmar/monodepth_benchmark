from pathlib import Path

import numpy as np

from src.external_libs import ImageDatabase, LabelDatabase
from src.tools import T_from_Rt
from . import PATHS, kitti_raw as kr

__all__ = [
    'get_split_file', 'load_calib', 'load_images', 'load_poses', 'load_oxts',
    'load_velos', 'load_velo_depths', 'load_depths', 'load_hints',
]


# CONSTANTS
# -----------------------------------------------------------------------------
# if 'kitti_raw_lmdb' not in PATHS:
#     raise KeyError(f"Missing 'kitti_raw_lmdb' root paths. Got {list(PATHS)}.")
# -----------------------------------------------------------------------------


# DATABASES
# -----------------------------------------------------------------------------
class PoseDatabase(LabelDatabase):
    """Database wrapper to convert OXTS data into a 4x4 transform matrix."""
    def __getitem__(self, item):
        oxts = super().__getitem__(item)
        pose = [kr.oxts2pose(o) for o in oxts] if isinstance(item, list) else kr.oxts2pose(oxts)
        return pose


class DepthVeloDatabase(LabelDatabase):
    """Database wrapper to convert a pointcloud into a sparse depth map.

    :param calib_db: (LabelDatabase) Calibration information database for the given sequence.
    :param use_velo_depth: (bool) If `True`, use the raw Velodyne depth instead of its projection.
    :param interp: (bool) If `True`, use linear interpolation to fill in the sparse LiDAR map.
    """
    def __init__(self, calib_db: LabelDatabase, *args, use_velo_depth: bool = False, interp: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.calib_db = calib_db
        self.use_velo_depth = use_velo_depth
        self.interp = interp

        self.Pi = np.eye(4)
        self.Pi[:3, :3] = self.calib_db['cam2cam/R_rect_00']
        self.Tr = T_from_Rt(self.calib_db['velo2cam/R'], self.calib_db['velo2cam/T'])

    def points2depth(self, pts: np.ndarray, cam: int) -> np.ndarray:
        return kr.points2depth(
            pts=pts,
            size=self.calib_db[f'cam2cam/S_rect_{cam:02}'],
            Pi=self.calib_db[f'cam2cam/P_rect_{cam:02}'] @ self.Pi,
            Tr=self.Tr,
            use_velo_depth=self.use_velo_depth,
            interp=self.interp,
        )

    def __getitem__(self, item: tuple[str, int]) -> np.ndarray:
        """Retrieve the given item.

        :param item: (tuple[str, int]) Database item as the key & camera, e.g. ('0000000000', 2)
        :return: (ndarray) (h, w) The loaded depth map.
        """
        item, cam = item
        pts = super().__getitem__(item)
        depth = [self.points2depth(p, cam) for p in pts] if isinstance(item, list) else self.points2depth(pts, cam)
        return depth


class DepthDatabase(ImageDatabase):
    """Database wrapper to convert a depth benchmark image into a float32 array."""
    @staticmethod
    def img2depth(img):
        return np.array(img, dtype=np.float32)/256.

    def __getitem__(self, item):
        img = super().__getitem__(item)
        depth = [self.img2depth(i) for i in img] if isinstance(item, list) else self.img2depth(img)
        return depth
# -----------------------------------------------------------------------------


# PATHS
# -----------------------------------------------------------------------------
def get_split_file(split: str, mode: str) -> Path:
    """Get depth split file based on the split and mode."""
    return PATHS['kitti_raw_lmdb']/'splits'/split/f'{mode}_files.txt'


def get_calib_path(seq: str) -> Path:
    """Load a sequence calibration database."""
    path = PATHS['kitti_raw_lmdb']/seq/'calibration'
    return path


def get_images_path(seq: str, drive: str, cam: str) -> Path:
    """Load a sequence/drive/cam image database."""
    path = PATHS['kitti_raw_lmdb']/seq/drive/cam/'data'
    return path


def get_velos_path(seq: str, drive: str) -> Path:
    """Load a sequence/drive Velodyne pointcloud database."""
    path = PATHS['kitti_raw_lmdb']/seq/drive/'velodyne_points'/'data'
    return path


def get_hints_path(seq: str, drive: str, cam: str) -> Path:
    """Load a sequence/drive/cam fused depth hints database."""
    path = PATHS['kitti_raw_lmdb']/'depth_hints'/seq/drive/cam
    return path


def get_depths_path(seq: str, drive: str, cam: str) -> Path:
    """Load a sequence/drive/cam depth benchmark database."""
    path = PATHS['kitti_raw_lmdb']/'depth_benchmark'/seq/drive/'proj_depth'/'groundtruth'/cam
    return path


def get_oxts_path(seq: str, drive: str) -> Path:
    """Load a sequence/drive OXTS database."""
    path = PATHS['kitti_raw_lmdb']/seq/drive/'oxts'/'data'
    return path
# -----------------------------------------------------------------------------


# LOADING
# -----------------------------------------------------------------------------
def load_calib(seq: str) -> LabelDatabase:
    """Load a sequence calibration database."""
    path = get_calib_path(seq)
    return LabelDatabase(path)


def load_images(seq: str, drive: str, cam: str) -> ImageDatabase:
    """Load a sequence/drive/cam image database."""
    path = get_images_path(seq, drive, cam)
    return ImageDatabase(path)


def load_velos(seq: str, drive: str) -> LabelDatabase:
    """Load a sequence/drive Velodyne pointcloud database."""
    path = get_velos_path(seq, drive)
    return LabelDatabase(path)


def load_velo_depths(seq: str, drive: str, calib_db: LabelDatabase, use_velo_depth: bool = False, interp: bool = False) -> DepthVeloDatabase:
    """Load a sequence/drive depth map database."""
    path = get_velos_path(seq, drive)
    return DepthVeloDatabase(calib_db=calib_db, path=path, use_velo_depth=use_velo_depth, interp=interp)


def load_hints(seq: str, drive: str, cam: str) -> LabelDatabase:
    """Load a sequence/drive/cam fused depth hints database."""
    path = get_hints_path(seq, drive, cam)
    return LabelDatabase(path)


def load_depths(seq: str, drive: str, cam: str) -> DepthDatabase:
    """Load a sequence/drive/cam depth benchmark database."""
    path = get_depths_path(seq, drive, cam)
    return DepthDatabase(path)


def load_oxts(seq: str, drive: str) -> LabelDatabase:
    """Load a sequence/drive OXTS database."""
    path = get_oxts_path(seq, drive)
    return LabelDatabase(path)


def load_poses(seq: str, drive: str) -> PoseDatabase:
    """Load a sequence/drive 4x4 transform matrix database."""
    path = get_oxts_path(seq, drive)
    return PoseDatabase(path)
# -----------------------------------------------------------------------------


# MAIN
# -----------------------------------------------------------------------------
def main():
    import matplotlib.pyplot as plt

    seq = '2011_09_26'
    drive = '2011_09_26_drive_0001_sync'

    calib_db = load_calib(seq)

    velo_db = load_velos(seq, drive)
    depth_db = load_velo_depths(seq, drive, calib_db, interp=True)
    left_db = load_images(seq, drive, 'l')
    right_db = load_images(seq, drive, 'r')
    pose_db = load_poses(seq, drive)

    # item = 30
    # key = f'{item:010}'
    _, axs = plt.subplots(2, 1)
    plt.tight_layout()

    for key in left_db:
        [ax.cla() for ax in axs]
        axs[0].imshow(left_db[key])

        depth = depth_db[(key, 2)]
        disp = (depth > 0)/depth.clip(min=1e-3)
        axs[1].imshow(disp, cmap='turbo', vmin=0, vmax=np.percentile(disp[disp > 0], 95))
        plt.pause(0.1)

    plt.show()


if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
