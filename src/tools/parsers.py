from collections import OrderedDict
from typing import Iterable, Type, TypeVar, Union

import torch
from timm.optim.optim_factory import create_optimizer_v2
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src.registry import DATA_REG, LOSS_REG, NET_REG, SCHED_REG
from src.typing import DataCfg, LoaderCfg, LossCfg, NetCfg, OptCfg, SchedCfg
from src.utils import metrics

__all__ = ['get_net', 'get_loss', 'get_ds', 'get_dl', 'get_opt', 'get_sched', 'get_metrics']


T = TypeVar('T')


def get_cls(cls_dict: dict[str, Type[T]], /, *args, type: str, **kwargs) -> T:
    """Instantiate an arbitrary class from a collection.

    Including `type` makes it a keyword-only argument. This has the double benefit of forcing the user to pass it as a
    keyword argument, as well as popping it from the config kwargs.

    :param cls_dict: (Dict[str, cls]) Dict containing the mappings to all possible classes to choose from.
    :param args: (tuple) Args to forward to target class.
    :param type: (str) Key of the target class. Must be present as a keyword-only argument.
    :param kwargs: (dict) Kwargs to forward to target class.
    :return: Instance of the target class with the desired arguments.
    """
    try:
        return cls_dict[type](*args, **kwargs)
    except Exception as e:
        raise ValueError(f'Error using "{type}" in {list(cls_dict)}') from e


def get_net(cfg: NetCfg) -> nn.ModuleDict:
    """Instantiate the target networks from a config dict.

    The depth estimation algorithm typically consists of multiple networks, commonly at least `depth` and `pose`.
    We're assuming that, within a given category, we can use different classes interchangeably.
    For instance, all `depth` networks take a single image as input and produce a multi-scale output, while all
    `pose` networks take multiple images and produce relative poses for each pair.

    New types and classes can be added to `NETWORKS` accordingly.

    :param cfg: (Dict[str, Dict[str, Any]]) Target network `types` and kwargs to forward to them.
    :return:
    """
    nets = {k: get_cls(NET_REG, type=k, **kw) for k, kw in cfg.items() if kw is not None}
    return nn.ModuleDict(OrderedDict(nets))


def get_loss(cfg: LossCfg) -> tuple[nn.ModuleDict, nn.ParameterDict]:
    """Instantiate the target losses from a config dict.

    :param cfg: (Dict[str, Dict[str, Any]]) Target loss `types` and kwargs to forward to them.
    :return:
    """
    losses, weights = nn.ModuleDict(), nn.ParameterDict()
    for k, kw in cfg.items():
        if kw is None: continue
        weights[k] = nn.Parameter(torch.as_tensor(kw.pop('weight', 1)), requires_grad=False)
        losses[k] = LOSS_REG[k](**kw)

    return losses, weights


def get_ds(cfg: DataCfg) -> Dataset:
    """Instantiate the target data from a config dict.

    :param cfg: (Dict[str, Any]]) Target loss `types` and kwargs to forward to them.
    :return:
    """
    ds = get_cls(DATA_REG, **cfg)
    return ds


def get_dl(mode: str, cfg_ds: DataCfg, cfg_dl: LoaderCfg) -> DataLoader:
    """Instantiate the target dataset loader from a config dict.

    Supports the presence of a common config, which gets overriden by specific cfg within each mode.
    Example:
    ```
        dataset:
            type: kitti
            depth_split: eigen_zhou

        core:
            mode: core
            aug: True

        val:
            mode: val
            aug: False
    ```

    By default we set `pin_memory = True` and `collate_fn = dataset.collate_fn`.
    This assumes we are using a `BaseDataset`.

    :param mode: (str) Dataset split: {'train', 'val', 'test'}.
    :param cfg_ds: (Dict[str, Any]]) Target dataset `type` and kwargs to forward to it (contains all modes).
    :param cfg_dl: (Dict[str, Any]]) Kwargs to forward to each dataloader (contains all modes).
    :return:
    """
    cfg = {k: v for k, v in cfg_ds.items() if k not in {'train', 'val', 'test'}}
    cfg.update(cfg_ds.get(mode, {}))
    ds = get_ds(cfg)

    cfg = {k: v for k, v in cfg_dl.items() if k not in {'train', 'val', 'test'}}
    cfg['pin_memory'] = cfg.get('pin_memory', True)
    cfg['collate_fn'] = ds.collate_fn
    cfg.update(cfg_dl.get(mode, {}))

    dl = DataLoader(ds, **cfg)
    return dl


def get_opt(parameters: Union[Iterable, nn.Module], cfg: OptCfg) -> optim.Optimizer:
    """Instantiate the target learning rate scheduler from a config dict.

    Serves as a wrapper for `timm` `create_optimizer_v2` to maintain consistency in the export interface.

    :param parameters: (Iterable | nn.Module) Parameters to forward to the optimizer (in any `torch` format)
    :param cfg: (Dict[str, Any]) Target optimizer `type` and kwargs to forward to it.
    :return:
    """
    if 'type' in cfg:
        cfg['opt'] = cfg.pop('type')
    elif 'opt' not in cfg:
        raise KeyError('Must provide a configuration key `type` or `opt` when instantiating an optimizer.')

    if cfg.pop('frozen_bn', False):
        if not isinstance(parameters, nn.Module):
            raise ValueError('Cannot freeze batch norm parameters unless given nn.Module')

        for m in parameters.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(False)

    return create_optimizer_v2(parameters, **cfg)


def get_sched(opt: optim.Optimizer, cfg: SchedCfg) -> optim.lr_scheduler._LRScheduler:
    """Instantiate the target learning rate scheduler from a config dict.

    TODO: Deprecate in favour of `timm` schedulers?

    :param opt: (optim.Optimizer) Optimizer to forward to the LR scheduler.
    :param cfg: (Dict[str, Any]) Target scheduler `type` and kwargs to forward to it.
    :return:
    """
    sched = get_cls(SCHED_REG, opt, **cfg)
    return sched


def get_metrics() -> nn.ModuleDict:
    """Instantiate the collection of depth metrics to monitor."""
    return nn.ModuleDict({
        'MAE': metrics.MAE(),
        'RMSE': metrics.RMSE(),
        'LogSI': metrics.ScaleInvariant(mode='log'),
        'AbsRel': metrics.AbsRel(),
        'Acc': metrics.DeltaAcc(delta=1.25),
    })
