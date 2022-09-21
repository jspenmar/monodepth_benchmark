from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import src.devkits.syns_patches as syp
from src import register
from src.tools import ops, viz
from src.typing import Axes, BatchData
from src.utils import io
from . import BaseDataset

__all__ = ['SYNSPatchesDataset']


@register('syns_patches')
class SYNSPatchesDataset(BaseDataset):
    """SYNS Patches dataset based on SYNS panorama images/LiDAR.

    See each function for details.

    Attributes:
    :param mode: (str) Split mode to load. {val, test, all}
    :param size: (Sequence[int]) Target image training size as (w, h).
    :param use_depth: (bool) If `True`, load ground truth LiDAR depth maps.
    :param use_edges: (bool) If `True`, load ground truth LiDAR depth maps.
    :param as_torch: (bool) If `True`, convert (x, y, meta) to torch.
    :param use_aug: (bool) If `True`, call 'self.augment' during __getitem__.
    :param log_time: (bool) If `True`, log time taken to load/augment each item.
    """
    def __init__(self,
                 mode: str,
                 size: tuple[int, int] = (640, 192),
                 use_depth: bool = True,
                 use_edges: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.w, self.h = self.size = size
        self.use_depth = use_depth
        self.use_edges = use_edges
        self.edges_dir = 'edges'

        if self.use_aug: raise ValueError('SYNS Patches is a testing dataset, no augmentations should be applied.')

        if self.mode == 'test' and (self.use_depth or self.use_edges):
            raise ValueError('Cannot use ground truth depth when loading the testing split!')

        self.w_full, self.h_full = self.size_full = 1242, 376
        self.split_file, self.items = self.parse_items()

        if self.h > self.w:
            raise ValueError(f'Target image height={self.h} is greater than image width={self.w}. '
                             f'Did you pass these in the correct order? Expected (width, height).')

    def __len__(self) -> int:
        """Number of items in dataset."""
        return len(self.items)

    def parse_items(self) -> tuple[Path, list[tuple[str, str]]]:
        """Return the list of items in the dataset."""
        return syp.load_split(self.mode)

    def load(self, item: int, x: dict, y: dict, m: dict) -> BatchData:
        """Load single item in dataset.

        NOTE: Items in each dict will be converted into `torch.Tensors` if `self.as_torch=True`.

        :param item: (int) Dataset item to load.
        :param x: {
            imgs: (ndarray) (h, w, 3) Target image for depth estimation.
        }
        :param y: {
            images: (ndarray) (h, w, 3) x['imgs'] (NO AUGMENTATIONS).
            K: (ndarray) (4, 4) Camera intrinsics parameters.
            depth: (Optional[ndarray]) (h, w, 1) Ground truth LiDAR depth map.
            edges: (Optional[ndarray]) (h, w, 1) Ground truth depth edges.
        }
        :param m: {
            items: (str) Loaded dataset item.
            category: (str) Image category label.
            subcategory: (str) Image subcategory label.
            aug (str): Augmentations applied to current item.
            errors: (List[str]): List of errors when loading previous items.
            data_timer (MultiLevelTimer): Timing information for current item.
        }
        """
        d = self.items[item]
        m['cat'], m['subcat'] = syp.load_category(d[0])

        with self.timer('Image'):
            img, img_res = self.load_image(d)
            x['imgs'] = io.pil2np(img_res)
            y['imgs'] = io.pil2np(img)
            # y['imgs'] = x['imgs'].copy()

        if self.use_depth:
            with self.timer('Depth'):
                y['depth'] = self.load_depth(d)

        if self.edges_dir:
            with self.timer('Edges'):
                edges = self.load_edges(d)
                y['edges'] = io.pil2np(edges)[..., None].astype(bool)

        y['K'] = syp.load_intrinsics()
        return x, y, m

    def load_image(self, data: tuple[str, str]) -> tuple[Image, Image]:
        """Load and resize a single image.

        :param data: (str, str) Data representing the item's scene and file number.
        :return: (Image) (self.w, self.h) Loaded PIL image.
        """
        file = syp.get_image_file(*data)
        img = Image.open(file)
        img_res = img.resize(self.size, resample=Image.BILINEAR)
        return img, img_res

    def load_depth(self, data: tuple[str, str]) -> np.ndarray:
        """Load a single depth map.

        :param data: (str, str) Data representing the item's scene and file number.
        :return: (ndarray) (self.full_w, self.full_h) Loaded numpy depth map.
        """
        file = syp.get_depth_file(*data)
        depth = np.load(file)
        return depth

    def load_edges(self, data: tuple[str, str]) -> Image:
        """Load a single depth edge map.

        :param data: (str, str) Data representing the item's scene and file number.
        :return: (Image) (self.full_w, self.full_h) Loaded PIL edge map.
        """
        file = syp.get_edges_file(data[0], self.edges_dir, data[1])
        edges = Image.open(file)
        return edges

    def transform(self, x: dict, y: dict, m: dict) -> BatchData:
        """Apply ImageNet standarization to the images processed by the network `x`."""
        x['imgs'] = ops.standardize(x['imgs'])
        return x, y, m

    def create_axs(self) -> Axes:
        """Create the axis structure required for plotting."""
        _, axs = plt.subplots(1 + self.use_depth + (self.edges_dir is not None))
        if isinstance(axs, plt.Axes): axs = np.array([axs])
        plt.tight_layout()
        return axs

    def show(self, x: dict, y: dict, m: dict, axs: Optional[Axes] = None) -> None:
        """Show a single dataset item."""
        axs = self.create_axs() if axs is None else axs

        axs[0].imshow(y['imgs'])
        if self.use_depth: axs[1].imshow(viz.rgb_from_disp(y['depth'], invert=True))
        if self.edges_dir: axs[-1].imshow(y['edges'])


if __name__ == '__main__':
    dataset = SYNSPatchesDataset(size=(640, 192), as_torch=False, use_aug=False)
    print(dataset)
    dataset.play(30)
