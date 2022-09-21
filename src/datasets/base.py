import inspect
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Optional, Sequence

import kornia.augmentation as ka
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from src.tools import ops
from src.typing import Axes, BatchData
from src.utils import MultiLevelTimer, TrivialAugmentWide, default_collate, delegates, get_logger, retry_new_on_error

__all__ = ['BaseDataset', 'get_augmentations']


def get_augmentations(strong=True):
    if strong:
        tfm = TrivialAugmentWide()
    else:
        tfm = ka.ColorJitter(
            brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1),
            p=1.0, same_on_batch=True, keepdim=True,
        )

    return tfm


class BaseDataset(ABC, Dataset):
    """Base dataset class that all others should inherit from.

    The idea is to provide a common structure and format for data to follow. Additionally, provide some nice
    functionality and automation for the more boring stuff. Datasets are defined as providing the following dictionaries
    for each item:
        - x: Inputs to the network (typically 'imgs').
        - y: Additional data required for loss computation (e.g. 'labels') or for logging (e.g. non-augmented images).
        - meta: Metadata for the given item, typically for logging.

    BaseDataset will automatically add the following fields to 'meta':
        - items: Item number (i.e. argument to  '__getitem__').
        - errors: If 'retry_exc' and NOT silent, log the exception messages caught.
        - aug: If 'use_aug', child class should add a list of the aug performed.

    The additional features/utilities provided include:
        - A logger to be used for logging.
        - A timer which, if enabled, times load/augment for an item. Can also be used in the child class.
        - Functionality to automatically 'retry' if the current item fails to load. This aims to replace "hacky"
          methods for manually filtering/blacklisting items, whilst being easy to enable & customize.
        - This functionality if wrapped in __getitem__, meaning that child classes only need to provide a 'load' method,
          which loads the data and sorts it in the corresponding (x, y, meta) dicts.
        - Tools for visualizing/playing the dataset to inspect and sanity check it.

    Attributes:
    :param as_torch: (bool) If `True`, convert (x, y, meta) to torch.
    :param use_aug: (bool) If `True`, call 'self.augment' during __getitem__.
    :param log_time: (bool) If `True`, log time taken to load/augment each item.

    Utilities:
    :attr logger: (Logger) Logger with parent CogvisDataset to use for logging.
    :attr timer: (MultiLevelTimer) If 'log_timings', timer to use for timing blocks.

    Methods:
    :method __len__: (abstract) Number of items in dataset.
    :method __getitem__: Retrieve a given item in the dataset. Should not be modified.
    :method load: (abstract) Load a single raw dataset item.
    :method augment: (override) Apply augmentations to a single dataset item. Default: No-op.
    :method transform: (override) Apply common transforms to a single dataset item. Default: No-op.
    :method to_torch: (override) Convert (x, y, meta) to torch. Default: Convert and permute (>=3D).
    :method collate_fn: (override) Collate a batch in a DataLoader. Default: PyTorch base collate.
    :method create_axs: (override) Create matplotlib axes to display a dataset item. Default: Single axis.
    :method show: (abstract) Display a single dataset item.
    :method play: Iterate over dataset and display each item.
    """
    def __init__(self, as_torch: bool = True, use_aug: bool = False, log_time: bool = True):
        self.logger.debug("Initializing BaseDataset")
        self.as_torch = as_torch
        self.use_aug = use_aug
        self.log_time = log_time

        # Timer setup - 'nullcontext' allows for a cleaner 'getitem' without too many conditionals
        self.timer = MultiLevelTimer(name=self.__class__.__qualname__, as_ms=True, precision=4) if self.log_time \
            else nullcontext

        if self.use_aug: self.logger.info(f"Dataset augmentations ENABLED")
        if self.log_time: self.logger.info(f"Logging dataset loading times...")

    def __init_subclass__(cls, retry_exc=None, silent=False, max_retries=10, use_blacklist=False, **kwargs):
        """Subclass initializer. We wrap the subclass init to replace kwargs."""
        super().__init_subclass__(**kwargs)
        cls.logger = get_logger(f'BaseDataset.{cls.__qualname__}')
        cls.__init__ = delegates(cls.__base__.__init__)(cls.__init__)  # Replace kwargs in signature.
        cls.__getitem__ = retry_new_on_error(
            cls.__getitem__,
            exc=retry_exc,
            silent=silent,
            max=max_retries,
            use_blacklist=use_blacklist
        )

    def __repr__(self) -> str:
        sig = inspect.signature(self.__init__)
        kw = {k: getattr(self, k) for k in sig.parameters if hasattr(self, k)}
        kw = ', '.join(f'{k}={v}' for k, v in kw.items())
        return f'{self.__class__.__qualname__}({kw})'

    @abstractmethod
    def __len__(self) -> int:
        """Number of items in the dataset."""

    def __getitem__(self, item: int) -> BatchData:
        """Generic dataset __getitem__. Loads, augments, times and converts data to torch (if required)."""
        self.logger.debug(f"Loading item {item}...")
        x, y, m = {}, {}, {'items': str(item)}

        with self.timer('Total'):
            with self.timer('Load'):
                x, y, m = self.load(item, x, y, m)

            if self.use_aug:
                m['augs'] = ''
                with self.timer('Augment'):
                    x, y, m = self.augment(x, y, m)

            with self.timer('Transform'):
                x, y, m = self.transform(x, y, m)

            if self.as_torch:
                with self.timer('ToTorch'):
                    x, y, m = self.to_torch(x, y, m)

        if self.log_time:
            m['data_timer'] = self.timer.copy()
            self.logger.debug(str(self.timer))
            self.timer.reset()

        return x, y, m

    @abstractmethod
    def load(self, item: int, x: dict, y: dict, m: dict) -> BatchData:
        """Load data for a single 'item'. MUST return (x, y, m)."""

    def augment(self, x: dict, y: dict, m: dict) -> BatchData:
        """Augment a loaded item. Default is a no-op."""
        return x, y, m

    def transform(self, x: dict, y: dict, m: dict) -> BatchData:
        """Transform a loaded item. Default is a no-op."""
        return x, y, m

    def to_torch(self, x: dict, y: dict, m: dict) -> BatchData:
        """Convert (x, y, m) to torch Tensors. Default converts to torch and permutes >=3D tensors."""
        return ops.to_torch((x, y, m))

    @classmethod
    def collate_fn(cls, batch: Sequence[BatchData]):
        """Function to collate multiple dataset items. By default uses the PyTorch collator."""
        return default_collate(batch)

    def create_axs(self) -> Axes:
        """Create the axis structure required for plotting. Assumes data will be in numpy format."""
        _, ax = plt.subplots()
        return ax

    @abstractmethod
    def show(self, x: dict, y: dict, m: dict, axs: Optional[Axes] = None) -> None:
        """Show a single dataset item. Should call 'create_axs' if 'axs' is None."""

    def play(self,
             fps: float = 30,
             skip: int = 1,
             reverse: bool = False,
             fullscreen: bool = False,
             axs: Optional[Axes] = None) -> None:
        """Iterate through dataset at the required fps and show each item."""
        if self.as_torch: raise ValueError('Dataset must not be in torch format when playing.')

        axs = self.create_axs() if axs is None else axs
        fig = plt.gcf()
        if fullscreen: fig.canvas.manager.full_screen_toggle()

        items = range(len(self)-1, 0, -skip) if reverse else range(0, len(self), skip)
        for i in items:
            x, y, m = self[i]
            axs.cla() if isinstance(axs, plt.Axes) else [ax.cla() for ax in axs.flatten()]
            self.show(x, y, m, axs)
            fig.suptitle(str(i))
            plt.pause(1/fps)
        plt.show(block=False)
