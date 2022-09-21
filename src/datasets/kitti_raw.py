import random
from contextlib import suppress
from pathlib import Path
from typing import Optional, Sequence, TypedDict, Union

import cv2
import numpy as np
import skimage.transform as skit
from PIL import Image
from matplotlib import pyplot as plt

import src.devkits.kitti_raw as kr
from src import register
from src.tools import ops, viz
from src.typing import Axes, BatchData
from src.utils import io
from . import BaseDataset, get_augmentations

__all__ = ['KittiRawDataset', 'KittiRawItem']


class KittiRawItem(TypedDict):
    seq: str
    cam: str
    stem: int


@register('kitti')
class KittiRawDataset(BaseDataset):
    """Kitti Depth based on the kitti_raw_sync dataset.

    See each function for details.

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
    def __init__(self,
                 split: str,
                 mode: str,
                 size: tuple[int, int] = (640, 192),
                 supp_idxs: Optional[Union[int, Sequence[int]]] = (-1, 1),
                 use_depth: bool = True,
                 use_hints: bool = False,
                 use_benchmark: bool = False,
                 use_strong_aug: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.depth_split = split
        self.mode = mode
        self.w, self.h = self.size = size
        self.supp_idxs = [] if supp_idxs is None else supp_idxs
        self.use_depth = use_depth
        self.use_hints = use_hints
        self.use_benchmark = use_benchmark
        self.use_strong_aug = use_strong_aug

        self.w_full, self.h_full = self.size_full = 1242, 376

        if isinstance(self.supp_idxs, int): self.supp_idxs = [self.supp_idxs]

        if self.h > self.w:
            raise ValueError(f'Target image height={self.h} is greater than image width={self.w}. '
                             f'Did you pass these in the correct order? Expected (width, height).')

        self.side2cam = {'l': 'image_02', 'r': 'image_03'}
        self.cam2stereo = {'image_02': 'image_03', 'image_03': 'image_02'}

        # NOTE: This might seem counterintuitive, but it makes sense.
        # This transform represents the direction in which the PIXELS move in, NOT the camera.
        self.cam2sign = {'image_02': -1, 'image_03': 1}

        self.split_file, self.items = self.parse_items()

        # Average intrinsics for all cameras
        self.K = np.array([
            [0.58, 0, 0.5, 0],
            [0, 1.92, 0.5, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.K[0, :] *= self.w
        self.K[1, :] *= self.h

        # Augmentations
        self.prob_flip, self.prob_photo = 0.5, (0.7 if self.use_strong_aug else 0.5)
        self.photo = get_augmentations(strong=self.use_strong_aug)

    def __len__(self) -> int:
        """Number of items in dataset."""
        return len(self.items)

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

    def load(self, item: int, x: dict, y: dict, m: dict) -> BatchData:
        """Load single item in dataset.

        NOTE: Items in each dict will be converted into `torch.Tensors` if `self.as_torch=True`.

        :param item: (int) Dataset item to load.
        :param x: {
            images: (ndarray) (h, w, c=3) Target image for depth estimation.
            supp_imgs: (Optional[ndarray]) (n, h, w, c=3) Support images for relative pose estimation.
            supp_idxs: (Optional[ndarray]) (n,) Support image indexes w.r.t `images`.
        }
        :param y: {
            imgs: (ndarray) (h, w, c=3) x['imgs'] (NO AUGMENTATIONS).
            supp_imgs: (Optional[ndarray]) (n, h, w, c=3) x['supp_imgs'] (NO AUGMENTATIONS).
            K: (ndarray) (4, 4) Camera intrinsic parameters.
            T_stereo: (Optional[ndarray]) (4, 4) Ground truth stereo transform. Only present if `0` in `supp_idxs`
            depth: (Optional[ndarray]) (h, w, 1) Ground truth LiDAR depth map.
            depth_hints: (Optional[ndarray]) (h, w, 1) Proxy stereo depth map.
        }
        :param m: {
            items: (str) Loaded dataset item.
            aug (str): Augmentations applied to current item.
            errors: (List[str]): List of errors when loading previous items.
            data_timer (MultiLevelTimer): Timing information for current item.
        }
        """
        d = self.items[item]
        m['stem'] = f"{d['seq']}/{d['cam']}/{d['stem']:010}"

        # LOAD TARGET IMAGE
        with self.timer('Image'):
            img = self.load_image(d)
            x['imgs'] = io.pil2np(img)
            y['imgs'] = x['imgs'].copy()  # Non-augmented copy

        # LOAD SUPPORT IMAGES
        if self.supp_idxs:
            x['supp_idxs'] = np.array(self.supp_idxs)
            with self.timer('Support'):
                supp = []
                for i in self.supp_idxs:
                    self.logger.debug(f'Loading support image: {i}')
                    if i == 0:
                        self.logger.debug(f'Loading stereo pair')
                        d_st = d.copy()
                        d_st['cam'] = self.cam2stereo[d['cam']]
                        supp.append(self.load_image(d_st, offset=i))

                        T = np.eye(4, dtype=np.float32)
                        T[0, 3] = self.cam2sign[d['cam']] * 0.1  # Arbitrary baseline
                        y['T_stereo'] = T
                    else:
                        supp.append(self.load_image(d, offset=i))

                x['supp_imgs'] = np.stack([io.pil2np(img) for img in supp])
                y['supp_imgs'] = x['supp_imgs'].copy()

        # CAMERA INTRINSICS
        y['K'] = self.K

        # LOAD DEPTH
        if self.use_depth:
            with self.timer('Depth'):
                y['depth'] = self.load_depth(d)

        # LOAD DEPTH HINT
        if self.use_hints:
            with self.timer('DepthHint'):
                y['depth_hints'] = self.load_hint(d)

        return x, y, m

    def load_image(self, data: KittiRawItem, offset: int = 0) -> Image:
        """Load and resize a single image.

        :param data: (KittRawItem) Data representing the item's sequence, camera and number.
        :param offset: (int) Additional offset to apply to the item number.
        :return: (Image) (self.w, self.h) Loaded PIL image.
        """
        file = kr.get_image_file(data['seq'], data['cam'], data['stem'] + offset)
        if not file.is_file():
            raise FileNotFoundError(f'Could not find specified file "{file}" with "{offset=}"')

        return Image.open(file).resize(self.size, resample=Image.BILINEAR)

    def load_depth(self, data: KittiRawItem) -> np.ndarray:
        """Load ground truth LiDAR depth.

        :param data: (KittRawItem) Data representing the item's sequence, camera and number.
        :return: (ndarray) (h, w, 1) Loaded depth map. NOTE: Shape can vary for each item.
        """
        if self.use_benchmark:
            file = kr.get_depth_file(data['seq'], data['cam'], data['stem'])
            depth = kr.load_depth(file)
        else:
            file = kr.get_velodyne_file(data['seq'], data['stem'])
            seq = data['seq'].split('/')[0]  # data['seq'] as 'sequence/drive'
            cam2cam, _, velo2cam = kr.load_calib(seq)  # TODO: Preload all and keep as dict?
            depth = kr.load_depth_velodyne(file, velo2cam, cam2cam, cam=int(data['cam'][-2:]))

        depth = skit.resize(depth, (self.h_full, self.w_full), order=0, preserve_range=True, mode='constant')
        return depth[..., None]

    def load_hint(self, data: KittiRawItem) -> np.ndarray:
        """Load a precomputed fusion of SGBM predictions.

        :param data: (KittRawItem) Data representing the item's sequence, camera and number.
        :return: (array) (h, w, 1) (320, 1024) Loaded fused SGBM depth map.
        """
        file = kr.get_hint_file(data['seq'], data['cam'], data['stem'])
        if not file.is_file():
            raise FileNotFoundError(f'Could not find specified depth hint file "{file}".')
        depth = np.load(file)
        depth = cv2.resize(depth, dsize=self.size, interpolation=cv2.INTER_NEAREST)
        return depth[..., None]

    def augment(self, x: dict, y: dict, m: dict) -> BatchData:
        """Apply augmentations to a single dataset item.

        Currently supported augmentations are "horizontal flipping" and "colour jittering".

        NOTE: All images need to be flipped, and all images use the same colour jittering.
        We do not apply colour jittering to the images used to compute the loss in `y`.
        """
        if random.random() <= self.prob_flip:
            self.logger.debug('Triggered Augmentation: Horizontal flip')
            m['augs'] += '[FlipLR]'
            x['imgs'] = np.ascontiguousarray(np.fliplr(x['imgs']))
            y['imgs'] = np.ascontiguousarray(np.fliplr(y['imgs']))

            if self.supp_idxs:
                x['supp_imgs'] = np.ascontiguousarray(np.flip(x['supp_imgs'], axis=-2))
                y['supp_imgs'] = np.ascontiguousarray(np.flip(y['supp_imgs'], axis=-2))
                if 'T_stereo' in y: y['T_stereo'][0, 3] *= -1

            if self.use_depth: y['depth'] = np.ascontiguousarray(np.fliplr(y['depth']))

            if self.use_hints: y['depth_hints'] = np.ascontiguousarray(np.fliplr(y['depth_hints']))

        if random.random() <= self.prob_photo:
            self.logger.debug('Triggered Augmentation: Photometric')

            imgs = x['imgs'][None]
            if self.supp_idxs: imgs = np.concatenate((imgs, x['supp_imgs']))

            if self.use_strong_aug: imgs = (255*imgs).astype(np.uint8)
            imgs = ops.to_numpy(self.photo(ops.to_torch(imgs)))
            if self.use_strong_aug: imgs = (imgs/255.).astype(np.float32)

            x['imgs'] = imgs[0]
            if self.supp_idxs: x['supp_imgs'] = imgs[1:]

            with suppress(AttributeError):
                m['augs'] += str([p.name for p in self.photo._params])

        return x, y, m

    def transform(self, x: dict, y: dict, m: dict) -> BatchData:
        """Apply ImageNet standarization to the images processed by the network `x`."""
        x['imgs'] = ops.standardize(x['imgs'])
        if self.supp_idxs: x['supp_imgs'] = ops.standardize(x['supp_imgs'])
        return x, y, m

    @classmethod
    def collate_fn(cls, batch: Sequence[BatchData]) -> BatchData:
        """Function to collate multiple dataset items.

        `x['supp_idxs']` is flattened as a single Tensor, since indexes are the same for all items. (n, )
        `x['supp_imgs']` are transposed to reflect the number of support images as the first dimension. (n, b, ...)
        `y['supp_imgs']` as above. (n, b, ...)
        `y['depth']` is maintained as a list of tensors since each item may have different (h, w). List[(h, w, 1), ...]

        :param batch: (Sequence[DatasetItem]) List of dataset items, each with `x`, `y`, `meta`.
        :return: (Dataset) Collated batch, where all items are stacked into a single tensor.
        """
        x, y, m = list(zip(*batch))
        x, y, m = super().collate_fn(x), super().collate_fn(y), super().collate_fn(m)

        if 'supp_idxs' in x:
            x['supp_idxs'] = x['supp_idxs'][0]  # Keep a single list of support idxs
            x['supp_imgs'] = x['supp_imgs'].transpose(0, 1)  # (n, b, 3, h, w)
            y['supp_imgs'] = y['supp_imgs'].transpose(0, 1)

        return x, y, m

    def create_axs(self) -> Axes:
        """Create the axis structure required for plotting."""
        _, axs = plt.subplots(1 + len(self.supp_idxs) + self.use_depth + self.use_hints)
        if isinstance(axs, plt.Axes): axs = np.array([axs])
        plt.tight_layout()
        return axs

    def show(self, x: dict, y: dict, m: dict, axs: Optional[Axes] = None) -> None:
        """Show a single dataset item."""
        axs = self.create_axs() if axs is None else axs

        # axs[0].imshow(x['imgs'])
        axs[0].imshow(y['imgs'])
        if self.supp_idxs:
            # [ax.imshow(im) for ax, im in zip(axs[1:], x['supp_imgs'])]
            [ax.imshow(im) for ax, im in zip(axs[1:], y['supp_imgs'])]

        if self.use_depth: axs[-1-self.use_hints].imshow(viz.rgb_from_disp(y['depth'], invert=True))
        if self.use_hints: axs[-1].imshow(viz.rgb_from_disp(y['depth_hints'], invert=True))


if __name__ == '__main__':
    dataset = KittiRawDataset(
        split='eigen', mode='train',
        size=(640, 192), supp_idxs=None,
        use_depth=True,
        use_benchmark=False,
        as_torch=False, use_aug=True, log_time=False,
    )
    print(dataset)
    dataset.play(fps=60)
