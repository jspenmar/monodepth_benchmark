# Source Code

This directory contains the source code for the project, excluding the main API scripts the end user should interact with.

---

## Structure
- [`core`](./core): Package containing core library components such as `MonoDepthModule` or `DepthEvaluator`. **Can depend on any custom package. Only API scripts should depend on them.**
- [`datasets`](./datasets): Package containing PyTorch dataset. 
- [`devkits`](./devkits): Package containing basic loading tools for datasets.
- [`external_libs`](./external_libs): Package containing external libraries from other developers.
- [`losses`](./losses): Package containing training losses.
- [`networks`](./networks): Package containing network architectures (including contribution decoders). 
- [`regularizers`](./regularizers): Package containing regularizer losses.
- [`tools`](./tools): Package containing more advanced utilities. **They should depend only on each other and `utils`.** 
- [`utils`](./utils): Package containing basic utilities. **They should not have any custom dependencies!**
- [`__init__.py`](./__init__.py): `src` package init. 
- [`paths.py`](./paths.py): File containing path management tools. 
- [`registry.py`](./registry.py): File containing the tools for registering models & datasets for training.
- [`typing.py`](./typing.py): File containing custom type hints. 

> Please take into account the notes regarding dependencies when deciding where to incorporate custom code. 

---

## Core
Contains the core library components required to train and evaluate a Monocular depth estimation network.

* [`evaluator`](./core/evaluator.py): Tools for computing predictions over a dataset and evaluating, such as `predict_depths` and `MonoDepthEvaluator`.
* [`handlers`](./core/handlers.py): Handlers that wrap multi-scale loss computation during training.
* [`image_logger`](./core/image_logger.py): PyTorch Lightning callback for logging images after each epoch.
* [`metrics`](./core/metrics.py): Functions for computing the various sets of evaluation metrics, such as `eigen`, `benchmark`, `pointcloud` and `ibims`. 
* [`trainer`](./core/trainer.py): Main PyTorch Lightning module for training, `MonoDepthModule`.

### MonoDepthModule Structure
The `MonoDepthModule` used for training is implemented using [PyTorch Lightning](https://www.pytorchlightning.ai), which wraps the optimization procedure and provides hooks to various steps.
See their [docs](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html) for background info about how the code is organized and what hooks are available. 
Overall, the module forward pass is split into:
1. `forward`: Computes the network predictions. 
2. `forward_postprocess`: Prepares the predictions for loss computation. E.g. upsampling to common resolution & converting to depth.
3. `forward_loss`: Computes the optimization loss and produces auxiliary outputs for logging. 
4. `compute_metrics`: Computes the metrics for logging and validation performance tracking.
5. `log_dict`: Logs scalars every `n` steps.
6. `image_logger`: Logs images at the end of each epoch.

To add a new network/loss to the training procedure:
1. Implement it in the respective module.
2. [Add it to the `registry`](#registry).
3. Add a new `if` block to the corresponding forward step based on the `registry` key.
4. If adding a loss, add the corresponding wrapper to [`handlers`](./core/handlers.py).
5. Add auxiliary inputs to `fwd` or `loss_dict` for logging.
6. Add logging to [`image_logger`](./core/image_logger.py) based on the auxiliary inputs. 

[Configs](../cfg) consist of: networks, losses, datasets, loaders, optimizers, schedulers and trainer. 
For an example covering most of the avilable options see [this file](../cfg/default.yaml).
* Networks and losses use dictionaries, where the keys correspond to the `registry` keys. Remaining parameters are `kwargs` to the respective class.
* Losses must add an additional parameter `weight`, which controls the scaling factor in the total loss.
* Datasets, optimizers and schedulers add a `type` argument corresponding to the `registry` keys.
* Datasets/loaders allow for different configs based on the `train` & `val` mode, overriding the original parameters.

```yaml
# -----------------------------------------------------------------------------
net:
  # Depth estimation network.
  depth:
    enc_name: 'convnext_base'  # Choose from `timm` encoders.
    pretrained: True
    dec_name: 'monodepth'  # Choose from custom decoders.
    
  # Pose estimation network (for use with purely monocular models).
  pose:
    enc_name: 'resnet18'  # Typically ResNet18 for efficiency.
    pretrained: True
# -----------------------------------------------------------------------------
loss:
  # Image-based reconstruction loss.
  img_recon:
    weight: 1
    loss_name: 'ssim'
# -----------------------------------------------------------------------------
dataset:
  type: 'kitti_lmdb'
  split: 'eigen_zhou'  # Can also use `eigen_benchmark`.
  size: [ 640, 192 ]  # Training images resolution.
  supp_idxs: [ -1, 1, 0 ]  # Support frames for reconstruction loss. Relative to the target frame. 0 for stereo.
  use_depth: True  # Needed to evaluate performance throughout training.

  train: {mode: 'train', use_aug: True}
  val: {mode: 'test', use_benchmark: True, use_aug: False}
# -----------------------------------------------------------------------------
loader:
  batch_size: 8
  num_workers: 8
  drop_last: True

  train: { shuffle: True }
  val: { shuffle: False }
# -----------------------------------------------------------------------------
optimizer:
  type: 'adam'  # Choose from any optimizer available from `timm`.
  lr: 0.0001
# -----------------------------------------------------------------------------
scheduler:
  type: 'steplr'
  step_size: 15
  gamma: 0.1
# -----------------------------------------------------------------------------
trainer:
  max_epochs: 30
  resume_training: True  # Will begin training from scratch if no checkpoints are found. Otherwise resume.
  monitor: 'AbsRel'  # Monitor metric to save `best` checkpoint.

  min_depth: 0.1  # Min depth to scale sigmoid disparity.
  max_depth: 100  # Max depth to scale sigmoid disparity.

  benchmark: True  # Pytorch cudnn benchmark.
# -----------------------------------------------------------------------------
``` 

---

## Datasets
Contains PyTorch datasets required for training and/or evaluating. 

* [`base`](./datasets/base.py): `BaseDataset` that all other datasets should inherit from, provides utilities for logging, loading and visualizing.
* [`kitti_raw`](./datasets/kitti_raw.py): `KittiRawDataset` for loading Kitti Raw Sync.
* [`kitti_raw_lmdb`](./datasets/kitti_raw_lmdb.py): `KittiRawLmdbDataset` for loading Kitti Raw Sync (LMDB). 
* [`syns_patches`](./datasets/syns_patches.py): `SYNSPatchesDataset` for loading SYNS-Patches.

All datasets should inherit from `BaseDataset` and implement/override the following methods.

```python
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
```

Datasets must return batches as three dictionaries: 
* `x`: Contains data required for the network forward pass. E.g. images, indexes of support frames. 
* `y`: Contains auxiliary data required for loss/metric computation. E.g. depth, edges, non-augmented images.
* `m`: Contains metadata about the loaded batch. E.g. loaded indexes, augmentations applied or errors while loading.

---

## Devkits
Contains low-level tools for loading and interacting with the available datasets.

* [`kitti_raw`](./devkits/kitti_raw.py): Tools for loading Kitti Raw Sync.
* [`kitti_raw_lmdb`](./devkits/kitti_raw_lmdb.py): Tools for loading Kitti Raw Sync (LMDBs).
* [`syns_patches`](./devkits/syns_patches.py): Tools for loading SYNS-Patches.

---

## External Libs
Contains libraries from other developers.

* [Databases](https://github.com/GuillaumeRochette/LMDatabase): Tools for creating LMDB datasets.
* [Chamfer Distance](https://github.com/chrdiller/pyTorchChamferDistance): C++/CUDA implementation of the Chamfer distance.

---

## Losses
The main available losses are:

* [`ReconstructionLoss`](./losses/reconstruction.py): Base view synthesis loss. Additionally used for feature-based view synthesis and autoencoder image reconstruction.
* [`RegressionLoss`](./losses/regression.py): Proxy depth regression loss. Additionally used for virtual stereo consistency.

> **NOTE:** Each of these incorporates multiple different contributions based on the available input configuration. 
> Check out the respective documentation for additional details.

New losses should be added as per the instructions in the [registry](#registry).
Losses must return a tuple consisting of

```python
"""
:return (tuple) (
    loss: (Tensor) (,) Scalar loss value.
    loss_dict: (TensorDict) Dictionary containing intermediate loss outputs used for TensorBoard logging.
)
"""
```

---

## Networks
The main available networks are:

* [`depth`](./networks/depth.py): Predicts a dense disparity map from a single image.
* [`pose`](./networks/pose.py): Predicts the relative pose between two images in axis-angle format. 
* [`autoencoder`](./networks/autoencoder.py): Converts the input image into a compact feature representation, which can be used to reconstruct the image.
Used primarily to learn a feature representation complementary to the image-based reconstruction loss. 

These networks use any of the pretrained encoders available in [`timm`](./https://github.com/rwightman/pytorch-image-models).
New networks should be added as per the instructions in the [registry](#registry).

Networks producing dense outputs (depth & autoencoder) additionally require a dense decoder:

* [`cadepth`](./networks/decoders/cadepth.py): Adds self-attention and channel-wise skip connections. From [CA-Depth](https://arxiv.org/abs/2112.13047).
* [`ddvnet`](./networks/decoders/ddvnet.py). Predicts depth as a discrete disparity volume. From [Johnston](https://arxiv.org/abs/2003.13951).
* [`diffnet`](./networks/decoders/diffnet.py). Adds self-attention and channel-wise attention skip-connections. From [DiffNet](https://arxiv.org/abs/2110.09482).
* [`hrdepth`](./networks/decoders/hrdepth.py). Adds progressive skip connections & SqueezeExcitation. From [HRDepth](https://arxiv.org/abs/2110.09482).
* [`monodepth`](./networks/decoders/monodepth.py). Default Conv+ELU+BilinearUpsample. From [Monodepth](https://arxiv.org/abs/1609.03677).
* [`superdepth`](./networks/decoders/superdepth.py). Conv+ELU+PixelShuffle. From [SuperDepth](https://arxiv.org/abs/1810.01849).

Custom decoders should be added to [`DECODERS`](./networks/decoders/__init__.py).
Currently, all decoders are required to have roughly the same argument structure.
This could probably be improved by using additional `**kwargs` in the main network initializers.

```python
"""
:param num_ch_enc: (Sequence[int]) List of channels per encoder stage.
:param enc_sc: (Sequence[int]) List of downsampling factor per encoder stage.
:param upsample_mode: (str) Torch upsampling mode. {'nearest', 'bilinear'...}
:param use_skip: (bool) If `True`, add skip connections from corresponding encoder stage.
:param out_sc: (Sequence[int]) List of multi-scale output downsampling factor as 2**s.
:param out_ch: (int) Number of output channels.
:param out_act: (str) Activation to apply to each output stage.
"""
```

---

## Regularizers
Regularizers are meant to prevent suboptimal or degenerate representations, rather than driving the optimization. 
The main available regularizers are:

* [`MaskReg`](./regularizers/mask.py): Explainability mask regularization. From [SfM-Learner](https://arxiv.org/abs/1704.07813).
* [`OccReg`](./regularizers/occlusion.py): Disparity occlusion regularization. From [DVSO](https://arxiv.org/abs/1807.02570).
* [`SmoothReg`](./regularizers/smooth.py): Disparity smoothness regularization. From multiple contributions.
* [`FeatPeakReg`](./regularizers/smooth.py): First-order feature peakiness regularization. From [FeatDepth](https://arxiv.org/abs/2007.10603).
* [`FeatSmoothReg`](./regularizers/smooth.py): Second-order feature smoothness regularization. From [FeatDepth](https://arxiv.org/abs/2007.10603).

New regularizers should be added as per the instructions in the [registry](#registry).
They must also follow the output format required by the [losses](#losses).

---

## Tools
A collection of more advanced utilities only depend on each other or on [`utils`](#utils).

* [`geometry`](./tools/geometry.py): Depth scaling/conversion and view synthesis tools, such as `extract_edges`, `to_scaled`, `to_inv`, `T_from_AAt`, `ViewSynth`...
* [`ops`](./tools/ops.py): Collection of PyTorch operations, such as `to_torch`, `to_numpy`, `allow_np` , `interpolate_like`, `expand_dim`...
* [`parsers`](./tools/parsers.py): Tools for instantiating classes from config dicts. 
* [`table_formatter`](./tools/table_formatter.py): `TableFormatter` to convert dataframes into LaTeX tables.
* [`viz`](./tools/viz.py): Visualizations tools `rgb_from_disp` & `rgb_from_feat`.

---

## Utils
A collection of basic utilities that do not depend on any other custom code from this library.

* [`autoaugment`](./utils/autoaugment.py): `TrivialAugmentWide` from PyTorch, modified to only produce photometric augmentations.
* [`callbacks`](./utils/callbacks.py): Custom PyTorch Lighning callbacks, incliding progress bars and anomaly detection.
* [`collate`](./utils/collate.py): `default_collate` from PyTorch, modified to accept `MultiLevelTimer`.
* [`deco`](./utils/deco.py): Custom decorators, including `opt_args_deco`, `delegates`, map_container` & `retry_new_on_error`.
* [`io`](./utils/io): YAML loading/writing tools and image conversion.
* [`metrics`](./utils/metrics.py): PyTorch Lightning metrics for use during training.
* [`misc`](./utils/misc.py): Collection of random utilities, `flatten_dict`, `sort_dict`, `get_logger` & `apply_cmap`.
* [`timers`](./utils/timers.py): `MultiLevelTimer` to allow for nested timing blocks.

---

## Paths
Path management for datasets and storing/loading checkpoints is done based on predefined locations in `DATA_ROOTS` & `MODEL_ROOTS`.
This alleviates the need to provide long and repeated paths, remaining flexible to datasets being stored in different locations (e.g. local scratch spaces).
Instructions for setting up custom roots can be found in the [main README](../README.md#paths).

This file additionally provides some utilities for finding dataset & model paths within the available roots: `find_data_dir` & `find_model_file`.
These functions will return the input path if it is an absolute path to an existing file/directory.
Otherwise, they will search the available roots and return the first existing path. 

---

## Registry
New network, losses or datasets should be added to the registry via the [`register`](./registry.py) decorator. 
This makes these classes accessible to the [`parsers`](./tools/parsers.py), and in turn to the config files.

```python
import torch.nn as nn

from src import register

@register(name='awesome', type='loss')
class MyAwesomeLoss(nn.Module):
    def forward(self, pred, target):
        err = (pred - target).abs().mean(dim=1, keepdim=True)
        loss = err.mean()
        return loss, {'l1_error': err}
```

* `type` selects the relevant registry, but can typically be omitted and guessed from the class name.
* `name` represents the identifier used in the configs and module forward pass.
Multiple aliases can be registered by providing a `tuple`, useful when losses share the same underlying computations but require different inputs or preprocessing in `MonoDepthModule`.
An example is the base [`ReconstructionLoss`](./losses/reconstruction.py), which can be used with either images or dense feature maps.

---

## Typing
Contains custom type hints and details on the config formats.
Some highlights include:

* `ArrDict`: Dict mapping from `str` to `np.ndarray`.
* `TensorDict`: Dict mapping from `str` to `torch.Tensor`.
* `BatchData`: Return type expected by dataset, consisting of `x` (network inputs), `y` (auxiliary data for losses) & `m` (batch metadata).
* `LossData`: Return type expected by losses, consisting of a scalar loss and a dictionary with intermediate tensors for logging.
* `MonoDepthCfg`: Config structure required by the `MonoDepthModule`.

---
