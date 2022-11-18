import warnings
from collections import Counter
from os import PathLike
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator

from src.tools import T_from_Rt
from src.typing import ArrDict
from src.utils.io import readlines
from . import PATHS

__all__ = [
    'SEQS', 'OXTS',
    'get_split_file', 'get_image_file', 'get_pose_file',
    'get_velodyne_file', 'get_hint_file', 'get_depth_file',
    'load_calib', 'load_split', 'load_oxts', 'load_pose', 'load_poses',
    'load_velo', 'load_depth_velodyne', 'load_depth',
    'project_velo', 'interp_velo', 'points2depth', 'oxts2pose',
]


# CONSTANTS
# -----------------------------------------------------------------------------
# if 'kitti_raw' not in PATHS:
#     raise KeyError(f"Missing 'kitti_raw' root paths. Got {list(PATHS)}.")

SEQS: tuple[str, ...] = ('2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03')
OXTS: tuple[str, ...] = (
    'lat', 'lon', 'alt',
    'roll', 'pitch', 'yaw',
    'vn', 've', 'vf', 'vl', 'vu',
    'ax', 'ay', 'az', 'af', 'al', 'au',
    'wx', 'wy', 'wz', 'wf', 'wl', 'wu',
    'pos_accuracy', 'vel_accuracy',
    'navstat', 'numsats', 'posmode', 'velmode', 'orimode'
)
# -----------------------------------------------------------------------------


# PATH BUILDING
# -----------------------------------------------------------------------------
def get_image_file(seq: str, cam: str, stem: int) -> Path:
    """Get image filename based on sequence, camera and item number."""
    return PATHS['kitti_raw']/seq/cam/'data'/f'{stem:010d}.png'


def get_pose_file(seq: str, stem: int) -> Path:
    """Get pose filename based on sequence and item number."""
    return PATHS['kitti_raw']/seq/'oxts'/'data'/f'{stem:010d}.txt'


def get_velodyne_file(seq: str, stem: int) -> Path:
    """Get velodyne filename based on sequence and item number."""
    return PATHS['kitti_raw']/seq/'velodyne_points'/'data'/f'{stem:010d}.bin'


def get_hint_file(seq: str, cam: str, stem: int) -> Path:
    """Get fused Semi-Global Block Matching depth estimate filename."""
    return PATHS['kitti_raw']/'depth_hints'/seq/cam/f'{stem:010d}.npy'


def get_depth_file(seq: str, cam: str, stem: int) -> Path:
    """Get fused Semi-Global Block Matching depth estimate filename."""
    return PATHS['kitti_raw']/'depth_benchmark'/seq/'proj_depth'/'groundtruth'/cam/f'{stem:010d}.png'


def get_split_file(split: str, mode: str) -> Path:
    """Get depth split file based on the split and mode."""
    return PATHS['kitti_raw']/'splits'/split/f'{mode}_files.txt'
# -----------------------------------------------------------------------------


# LOADING
# -----------------------------------------------------------------------------
def load_dict(file: PathLike, header: int = 0, strip: str = ':', shape: Optional[Sequence[int]] = None) -> ArrDict:
    """Load file as a dict of ndarrays.

    Typically serves as an intermediate step to calibration loading, stored as
        P0: 1 0 0 0 0 1 0 0 0 0 0 1
        P1: 1 0 0 0 0 1 0 0 0 0 0 1

    :param file: (PathLike) File to load.
    :param header: (int) Number of header lines to ignore.
    :param strip: (str) String or character to strip from each key.
    :param shape: (None|Sequence[int]) Target shape for ALL arrays, otherwise returns flattened arrays.
    :return: (ArrayDict) Dict mapping strings to float arrays with desired shape.
    """
    lines = [l.split() for l in readlines(file)][header:]

    d = {}
    for l in lines:
        h, arr = l[0], np.array(l[1:], dtype=np.float32)
        if shape: arr = arr.reshape(shape)
        d[h.strip(strip)] = arr

    return d


def load_calib(seq: str) -> tuple[ArrDict, ArrDict, ArrDict]:
    """Load calibrations for a given RawSync sequence.

    Calibration is given as three files: cam_to_cam, imu_to_velo and velo_to_cam.
    See Kitti docs to find more about each file.

        - `Rotations` (R) are reshaped to (3, 3).
        - `Intrinsics` (K) are reshaped to (3, 3).
        - `Image sizes` (S) are converted to integers.
        - `Projections` (P) are converted to homogeneous form (4, 4).

    :param seq: (str) RawSync sequence to load.
    :return: (tuple[ArrayDict]) Calibration dicts mapping strings to ndarray with shapes as above.
    """
    cam2cam_file = PATHS['kitti_raw']/seq/'calib_cam_to_cam.txt'
    imu2velo_file = PATHS['kitti_raw']/seq/'calib_imu_to_velo.txt'
    velo2cam_file = PATHS['kitti_raw']/seq/'calib_velo_to_cam.txt'

    cam2cam = load_dict(cam2cam_file, header=1)
    imu2velo = load_dict(imu2velo_file, header=1)
    velo2cam = load_dict(velo2cam_file, header=1)
    homo = np.array([0, 0, 0, 1], dtype=float)[None]

    for d in (cam2cam, imu2velo, velo2cam):
        for k, v in d.items():
            if   'R' in k: d[k] = v.reshape(3, 3)                     # Rotation matrix
            elif 'K' in k: d[k] = v.reshape(3, 3)                     # Intrinsics matrix
            elif 'S' in k: d[k] = v.astype(int)                       # Image size
            elif 'P' in k: d[k] = np.vstack((v.reshape(3, 4), homo))  # Projection matrix

    return cam2cam, imu2velo, velo2cam


def load_split(file: PathLike) -> list[str]:
    """Load depth split lines."""
    return readlines(file)


def load_velo(file: PathLike) -> NDArray:
    """Load velodyne points from file.

    Original file provides points as (x, y, z, reflectance).
    We replace reflectance with a homogeneous coordinate.

    :param file: (Pathlike) File to load.
    :return: (ndarray) (n, c=4) Velodyne points as (x, y, z, 1).
    """
    pts = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
    pts[:, 3] = 1.0  # Convert reflectance to homogeneous
    return pts


def load_oxts(file: PathLike) -> dict[str, float]:
    """Load OxTS file given the expected keys in OXTS_KEYS."""
    return dict(zip(OXTS, np.loadtxt(file, dtype=np.float32)))


def load_pose(file: PathLike) -> NDArray:
    """Load OxTS file and convert into a 4x4 pose matrix."""
    oxts = load_oxts(file)
    pose = oxts2pose(oxts)
    return pose


def load_poses(seq: str, drive: int) -> NDArray:
    """Load raw sync drive poses.

    Unlike the Kitti Odometry dataset, each pose is provided in a separate file.

    :param seq: (str) Raw sync sequence (e.g. '2011_09_26').
    :param drive: (int) Drive within the sequence.
    :return: (ndarray) (n, 4, 4) storing poses for each image.
    """
    oxts_dir = PATHS['kitti_raw']/seq/f'{seq}_drive_{drive:04}_sync'/'oxts'/'data'
    poses = np.stack([load_pose(file) for file in sorted(oxts_dir.iterdir())])
    return poses


def load_depth_velodyne(file: PathLike,
                        velo2cam: ArrDict,
                        cam2cam: ArrDict,
                        cam: int,
                        use_velo_depth: bool = False,
                        interpolate: bool = False) -> NDArray:
    """Load depth map for a given velodyne file.

    :param file: (PathLike) File to load.
    :param velo2cam: (ArrayDict) velo_to_cam components from calibration.
    :param cam2cam: (ArrayDict) cam_to_cam components from calibration.
    :param cam: (int) Camera to project onto [0, 3].
    :param use_velo_depth: (bool) If `True`, use the raw Velodyne depth instead of its projection.
    :param interpolate: (bool) If `True`, use linear interpolation to fill in the sparse LiDAR map.
    :return: (ndarray) (h, w) Sparse depth map.
    """
    Pi = np.eye(4)
    Pi[:3, :3] = cam2cam['R_rect_00']
    Pi = cam2cam[f'P_rect_{cam:02}'] @ Pi

    depth = points2depth(
        pts=load_velo(file),
        size=cam2cam[f'S_rect_{cam:02}'],
        Pi=Pi,
        Tr=T_from_Rt(velo2cam['R'], velo2cam['T']),
        use_velo_depth=use_velo_depth,
        interp=interpolate,
    )
    return depth


def load_depth(file: PathLike) -> NDArray:
    """Load an updated Kitti depth benchmark file.

    :param file: (PathLike) File to load.
    :return: (ndarray) (h, w) Loaded depth map.
    """
    depth = Image.open(file)
    depth = np.array(depth, dtype=np.float32)/256.  # NOTE: 256, NOT 255
    return depth
# -----------------------------------------------------------------------------


# CONVERSIONS
# -----------------------------------------------------------------------------
def oxts2pose(oxts: dict[str, float]) -> NDArray:
    """Convert data in OxTS format into a 4x4 transform matrix.

    Adapted from the Kitti raw devkit: `convertOxtsToPose`.
    Translation is given as Mercator coordinates.
    Rotation is given as roll, pitch, yaw.

    :param oxts: (dict) OxTS data.
    :return: (ndarray) (4, 4) Pose.
    """
    # Translation from GPS
    earth_radius = 6378137
    scale = np.cos(oxts['lat'] * np.pi/180.0)
    mercator_x = scale * oxts['lon'] * np.pi * earth_radius/180
    mercator_y = scale * earth_radius * np.log(np.tan((90 + oxts['lat']) * np.pi/360))
    t = np.array([mercator_x, mercator_y, oxts['alt']])

    # Rotation from roll, pitch, yaw
    rx, ry, rz = oxts['roll'], oxts['pitch'], oxts['yaw']
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    T = T_from_Rt(R, t)
    return T


def points2depth(pts: NDArray,
                 size: Sequence[int],
                 Pi: NDArray,
                 Tr: NDArray,
                 use_velo_depth: bool,
                 interp: bool) -> NDArray:
    """Convert a pointcloud into a sparse depth map.

    NOTE: `use_velo_depth` should never be used... It is here for backwards compatibility.

    :param pts: (ndarray) (n, 4) Loaded pointcloud
    :param size: (int, int) Image size as (w, h)
    :param Pi: (ndarray) (4, 4) Camera projection matrix.
    :param Tr: (ndarray) (4, 4) Transform from LiDAR to camera reference frame.
    :param use_velo_depth: (bool) If `True`, use the raw Velodyne depth instead of its projection.
    :param interp: (bool) If `True`, use linear interpolation to fill in the sparse LiDAR map.
    :return: (ndarray) (h, w) Sparse depth map.
    """
    if use_velo_depth:
        warnings.warn('Using raw velodyne depth.... This is incorrect and should only be used to '
                      'generate the incorrect ground-truth in Kitti Eigen. ')

    w, h = size
    pc_img, valid = project_velo(pts, Pi, Tr, size, use_velo_depth=use_velo_depth)
    pc_img = pc_img[valid]

    if interp:
        depth = interp_velo((h, w), pc_img)
        return depth

    # Create depth map
    xs, ys = pc_img[:, :2].T.astype(int)
    depth = np.zeros((h, w), dtype=np.float32)
    depth[ys, xs] = pc_img[:, 2]

    # Apply Z-buffering
    # NOTE: In my mind, this should be equivalent to `np.ravel_multi_index((xs, ys), image_size)`. Apparently it isn't.
    idxs = ys * (w - 1) + xs - 1  # Flatten 2-D indexes to 1-D.
    dup = (item for item, count in Counter(idxs).items() if count > 1)
    for idx in dup:
        pts = np.where(idxs == idx)[0]
        x, y = xs[pts[0]], ys[pts[0]]
        depth[y, x] = pc_img[pts, 2].min()
    depth = depth.clip(min=0)

    return depth


def project_velo(pts: NDArray,
                 P: NDArray,
                 Tr: NDArray,
                 size: Sequence[int],
                 T: NDArray = np.eye(4),
                 use_velo_depth: bool = False) -> tuple[NDArray, NDArray]:
    """Project velodyne points onto an image.

    :param pts: (ndarray) (n, c=4) LiDAR points with each entry as (x, y, z, 1).
    :param P: (ndarray) (4, 4) Camera projection matrix.
    :param Tr: (ndarray) (4, 4) Transform from LiDAR to camera reference frame.
    :param size: (Sequence[int]) Image size as (w, h).
    :param T: (ndarray) (4, 4) Additional transform to apply. Default is Identity.
    :param use_velo_depth: (bool) If `True`, use the raw Velodyne depth instead of its projection.
    :return: (Tuple[ndarray, ndarray]) ((n, c=3), (n,)) Image points as (u, v, depth) and valid bool mask.
    """
    w, h = size

    pts_img = P @ Tr @ T @ pts.T  # (4, n)
    pts_img = pts_img[:3]  # (3, n) Remove homogeneous
    pts_img[:2] /= pts_img[-1][None]  # Remove depth scaling
    if use_velo_depth: pts_img[2] = pts[:, 0]

    # Check in-bounds
    pts_img[:2] = pts_img[:2].round() - 1  # -1 to get exact same value as Kitti matlab code
    valid_u = (pts_img[0] >= 0) & (pts_img[0] < w)
    valid_v = (pts_img[1] >= 0) & (pts_img[1] < h)
    valid_z = pts[:, 0] >= 0  # Remove points behind image plane

    pts_img = pts_img.T
    mask_valid = valid_u & valid_v & valid_z
    return pts_img, mask_valid


def interp_velo(shape: tuple[int, int], pts: NDArray) -> NDArray:
    """Linearly interpolate the LiDAR into a dense depth map.
    From https://github.com/hunse/kitti

    :param shape: (int, int) Image resolution as (h, w).
    :param pts: (ndarray) (n, 3) LiDAR pointcloud as (x, y, depth).
    :return: (ndarray) (h, w) Dense interpolated depth.
    """
    h, w = shape
    xy, d = pts[:, 1::-1], pts[:, 2]
    interp = LinearNDInterpolator(xy, d, fill_value=0)

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.vstack((ys.flatten(), xs.flatten())).T
    depth = interp(coords).reshape(shape)
    return depth
# -----------------------------------------------------------------------------


# MAIN
# -----------------------------------------------------------------------------
def main():
    """Example usage for "raw_sync" tools"""
    import matplotlib.pyplot as plt
    from PIL import Image
    np.set_printoptions(precision=4)

    # Get filelists
    seq = '2011_09_26'
    drive_dir = PATHS['kitti_raw']/seq/'2011_09_26_drive_0001_sync'

    cam2cam, imu2velo, velo2cam = load_calib(seq)
    image_dir = drive_dir/'image_02'/'data'
    velo_dir = drive_dir/'velodyne_points'/'data'

    # Load depth
    image_files = sorted(image_dir.iterdir())
    velo_files = sorted(velo_dir.iterdir())

    idx = 30
    _, (ax1, ax2, ax3) = plt.subplots(3, 1)
    plt.tight_layout()

    ax1.imshow(Image.open(image_files[idx]))
    depth = load_depth_velodyne(velo_files[idx], velo2cam, cam2cam, cam=2, interpolate=False)
    ax2.imshow(depth, cmap='turbo')

    plt.show()


if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
