from pathlib import Path

import cv2
import numpy as np
import skimage.transform as skit
from PIL import Image

import src.devkits.kitti_raw_lmdb as kr
from src import register
from src.utils import io
from . import KittiRawDataset, KittiRawItem

__all__ = ['KittiRawLMDBDataset']


@register('kitti_lmdb')
class KittiRawLMDBDataset(KittiRawDataset):
    """Kitti Depth based on the kitti_raw_sync dataset.

    LMDB variant of KittiRawDataset. This is designed to be a drop-in replacement that can help with IO load.
    As such, we only need to provide wrappers around the loading functions in the same format as the original dataset.

    The _databases are loaded as required and added to a cached dict.

    Attributes:
    :param split: (str) Kitti depth split to use (eigen, eigen_zhou, eigen_full, benchmark, odom).
    :param mode: (str) Dataset mode (core, val, test).
    :param size: (Sequence[int]) Target image training size as (w, h).
    :param supp_idxs: (int | Sequence[int]) Indexes of the support images to load.
    :param use_depth: (bool) If `True`, load ground truth LiDAR depth maps.
    :param use_hints: (bool) If `True`, load precomputed fused SGBM depth maps.
    :param use_benchmark: (bool) If `True`, load corrected ground truth depth maps.
    :param use_strong_aug: (bool) If `False`, use only colour jittering augmentations.
    :param as_torch: (bool) If `True`, convert (x, y, meta) to torch.
    :param use_aug: (bool) If `True`, call 'self.augment' during __getitem__.
    :param log_time: (bool) If `True`, log time taken to load/augment each item.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_dbs = {}
        self.depth_dbs = {}
        self.poses_dbs = {}
        self.hints_dbs = {}
        self.calib_dbs = {}

        self.preload()

    def preload(self) -> None:
        """Create all LMDBs required by the dataset split."""
        drives = set(item['seq'] for item in self.items)

        for d in drives:
            self.image_dbs[f'{d}/image_02'] = kr.load_images(*d.split('/'), 'image_02')
            self.image_dbs[f'{d}/image_03'] = kr.load_images(*d.split('/'), 'image_03')

        if self.use_hints:
            for d in drives:
                self.hints_dbs[f'{d}/image_02'] = kr.load_hints(*d.split('/'), 'image_02')
                self.hints_dbs[f'{d}/image_03'] = kr.load_hints(*d.split('/'), 'image_03')

        if self.use_depth:
            if self.use_benchmark:
                for d in drives:
                    self.depth_dbs[f'{d}/image_02'] = kr.load_depths(*d.split('/'), 'image_02')
                    self.depth_dbs[f'{d}/image_03'] = kr.load_depths(*d.split('/'), 'image_03')
            else:
                seqs = set(seq.split('/')[0] for seq in drives)
                self.calib_dbs = {s: kr.load_calib(s) for s in seqs}

                for d in drives:
                    s, d2 = d.split('/')
                    self.depth_dbs[d] = kr.load_velo_depths(s, d2, self.calib_dbs[s])

    def parse_items(self) -> tuple[Path, list[KittiRawItem]]:
        """Helper to parse each dataset item as a sequence, camera and file number."""
        file = kr.get_split_file(self.depth_split, self.mode)
        lines = [line.split() for line in io.readlines(file)]
        items = [{
            'seq': line[0],
            'cam': self.side2cam[line[2]],
            'stem': int(line[1]),
        } for line in lines]
        return file, items

    def load_image(self, data: KittiRawItem, offset: int = 0) -> Image:
        """Load and resize a single image.

        :param data: (KittRawItem) Data representing the item's sequence, camera and number.
        :param offset: (int) Additional offset to apply to the item number.
        :return: (Image) (self.w, self.h) Loaded PIL image.
        """
        k = f'{data["stem"] + offset:010}'
        kdb = f'{data["seq"]}/{data["cam"]}'

        db = self.image_dbs[kdb]
        if k not in db: raise FileNotFoundError(f'Could not find specified file "{kdb}/{k}" with "{offset=}"')

        image = db[k].resize(self.size, resample=Image.BILINEAR)
        return image

    def load_depth(self, data: KittiRawItem) -> np.ndarray:
        """Load ground truth LiDAR depth.

        :param data: (KittRawItem) Data representing the item's sequence, camera and number.
        :return: (ndarray) (h, w, 1) Loaded depth map. NOTE: Shape can vary for each item.
        """
        if self.use_benchmark:
            k = f'{data["stem"]:010}'
            kdb = f'{data["seq"]}/{data["cam"]}'
            depth = self.depth_dbs[kdb][k]
        else:
            k = (f'{data["stem"]:010}', int(data['cam'][-2:]))
            kdb = data["seq"]
            depth = self.depth_dbs[kdb][k]

        depth = skit.resize(depth, (self.h_full, self.w_full), order=0, preserve_range=True, mode='constant')
        return depth[..., None]

    def load_hint(self, data: KittiRawItem) -> np.ndarray:
        """Load a precomputed fusion of SGBM predictions.

        :param data: (KittRawItem) Data representing the item's sequence, camera and number.
        :return: (array) (h, w, 1) (320, 1024) Loaded fused SGBM depth map.
        """
        k = f'{data["stem"]:010}'
        kdb = f'{data["seq"]}/{data["cam"]}'
        depth = cv2.resize(self.hints_dbs[kdb][k], dsize=self.size, interpolation=cv2.INTER_NEAREST)
        return depth[..., None]


def main():
    dataset = KittiRawLMDBDataset(
        split='benchmark2', mode='train',
        size=(640, 192), supp_idxs=None,
        use_depth=True, interpolate_depth=False,
        use_depth_hints=False, use_poses=False,
        use_strong_aug=False,
        as_torch=False, use_aug=False, log_timings=False,
    )
    print(dataset)
    dataset.play(fps=100)


if __name__ == '__main__':
    main()
