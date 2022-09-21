from os import PathLike
from typing import Optional, Sequence, Type, TypedDict, Union

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset

__all__ = [
    'TimerData', 'Axes', 'BatchData', 'TensorDict', 'LossData',
    'Metrics', 'ArrDict', 'ModDict', 'DataDict', 'SchedDict',
    'BaseNetCfg', 'BaseLossCfg',
    'NetCfg', 'LossCfg',
    'DataCfg', 'LoaderCfg',
    'OptCfg', 'SchedCfg',
    'TrainCfg', 'MonoDepthCfg'
]

Axes = Union[plt.Axes, NDArray]

Metrics = dict[str, Union[str, float]]
ArrDict = dict[str, NDArray]
TensorDict = dict[Union[str, int], Tensor]
ModDict = dict[str, Type[nn.Module]]
DataDict = dict[str, Type[Dataset]]
SchedDict = dict[str, Type[_LRScheduler]]

TimerData = dict[str, Union[int, float]]
BatchData = tuple[dict, dict, dict]
LossData = tuple[torch.Tensor, TensorDict]


# TRAINER CONFIGS
class BaseNetCfg(TypedDict):
    """Confing for a base network."""
    type: str  # Network class to use.


class BaseLossCfg(TypedDict):
    """Config for a loss without parameters. We only require a weighting factor."""
    weight: float  # Loss weighting factor.


class NetCfg(TypedDict):
    """Config dict for a collection of networks."""
    depth: BaseNetCfg  # Config for the depth estimation network.
    pose: BaseNetCfg  # Config for the pose estimation network


class LossCfg(TypedDict):
    """Config dict for a collection of losses."""
    recon: BaseLossCfg  # Config for the reconstruction loss
    smoooth: Optional[BaseLossCfg]  # Config for the smoothness loss


class DataCfg(TypedDict):
    """Config dict for a collection of BaseDataset. Configs in {core, val, test} override values in main config."""
    type: str  # Dataset class to use. {kitti}
    mode: str  # Dataset slip to use (should be overriden in sub-config).
    size: Sequence[int]  # Target image size as (w, h).
    supp_idxs: Optional[Sequence[int]]  # Indexes of the support frames to load. (default = (-1, 1))
    use_depth: Optional[bool]  # If `True`, load ground truth depth maps. (default = False)
    use_hints: Optional[bool]  # If `True`, load proxy depth hints. (default = False)
    use_benchmark: Optional[bool]  # If `True`, load corrected ground truth depth maps. (default = False)
    use_strong_aug: Optional[bool]  # If `True`, use TrivialAugmentWide augmentations. (default = False)
    as_torch: Optional[bool]  # If `True`, convert (x, y, meta) to torch. (default = True)
    use_aug: Optional[bool]  # If `True`, apply augmentations. (default = False) (recommended = {core: True, val: False})
    log_time: Optional[bool]  # If `True`, log dataset timing. (default = True)

    train: Optional['DataCfg']  # Train config to override defaults.
    val: Optional['DataCfg']  # Val config to override defaults.
    test: Optional['DataCfg']  # Test config to override defaults.


class LoaderCfg(TypedDict):
    """Config dict for a torch DataLoader."""
    batch_size: int  # Number of images per batch.
    num_workers: Optional[int]  # Number of multiprocessing workers. (default = None) (recommended = num_cpus)
    drop_last: Optional[bool]  # If `True`, drop last batch with inconsistent size. (default = False) (recommended = {core: True, val: True, test: False})
    shuffle: Optional[bool]   # If `True`, randomly order dataset items. (default = False) (recommended = {core: True, val: False, test: False})
    pin_memory: Optional[bool]  # If `True`, pin memory on GPU. (default = True) (recommended = True)

    train: Optional['LoaderCfg']  # Train config to override defaults.
    val: Optional['LoaderCfg']  # Val config to override defaults.
    test: Optional['LoaderCfg']  # Test config to override defaults.


class OptCfg(TypedDict):
    """Config dict for a torch Optimizer."""
    type: Optional[str]  # Optimizer class to use (mutually exclusive with `opt`).
    opt: Optional[str]  # Optimizer class to use (mutually exclusive with `type`).
    lr: float  # Base learning rate.

    # Remaining keys are specific to the selected Optimizer.


class SchedCfg(TypedDict):
    """Config dict for a torch LRScheduler."""
    type: str  # Scheduler class to use. {steplr}

    # Remaining keys are specific to the selected Scheduler.


class TrainCfg(TypedDict):
    """Config dict for training options."""
    max_epochs: bool  # Max number of training epochs.
    resume_training: Optional[bool]  # If `True`, look for existing checkpoints and resume training. (default = False)
    load_ckpt: Optional[PathLike]  # Path to load pretrained weights (does not resume training). (default = None)
    log_every_n_steps: Optional[int]  # Logging interval for scalars. (default = 100)
    monitor: Optional[str]  # Metric to monitor when saving checkpoints & stopping. (default = 'AbsRel')

    benchmark: Optional[bool]  # If `True`, set torch.backends.cudnn.benchmark. (default = False)
    gradient_clip_val: Optional[float]  # Gradient norm clipping value. (default = None) (recommended = 1)
    precision: Optional[int]  # Training batch gradients to accumulate. (default = 32) (recommended = 16)
    accumulate_grad_batches: Optional[int]  # Training batch gradients to accumulate. (default = 1) (recommended = 1)

    swa: Optional[bool]  # If `True`, enable Stochastic Weight Averaging callback. (default = False)
    early_stopping: Optional[bool]  # If `True`, enable early stopping callback. (default = False)

    min_depth: Optional[float]  # Minimum depth when scaling the network prediction (default = 0.1)
    max_depth: Optional[float]  # Maximum depth when scaling the network prediction (default = 100)


class MonoDepthCfg(TypedDict):
    """Monocular depth trainer config. See each sub-class for details."""
    net: NetCfg
    loss: LossCfg
    dataset: DataCfg
    loader: LoaderCfg
    optimizer: OptCfg
    scheduler: SchedCfg
    trainer: TrainCfg
