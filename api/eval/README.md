# Evaluation

---

## Export Predictions
When evaluating a given network, we first need to compute the predictions on the target dataset.
The basic script to do this is 
```shell
python api/data/export_depth.py --ckpt-file models/<EXPERIMENT>/<MODEL>/<SEED>/ckpts/best.ckpt \
--cfg-file cfg/export/<DATASET>.yaml \
--save-file models/<EXPERIMENT>/<MODEL>/<SEED>/results/pred_<DATASET>_best.npz
```

Note that `ckpt_file` can be provided as either an absolute path (e.g. the example above) or as a path relative to one of the `MODEL_ROOTS` in [`PATHS.yaml`](../../PATHS.yaml).
For instance, the path above could be replaced with `<EXP>/<MODEL>/<SEED>/ckpts/best.ckpt` (i.e. ommiting `./models`).
`save-file` must be provided as an absolute path to the target file, but the `.npz` will be automatically added.

Configs mimic those from the [`MonoDepthModule`](../../src/core/trainer.py). 
Note that when exporting predictions, only the images are required.
This means that optional fields such as `use_depth` or `use_edges` can typically be disabled.

```yaml
# -----------------------------------------------------------------------------
dataset:
  type: 'kitti_lmdb'
  split: 'eigen_benchmark'
  mode: 'test'
  supp_idxs: ~
  use_depth: False
# -----------------------------------------------------------------------------
```

> **NOTE:** This is the script required to generate challenge submissions.
> The config to use is `cfg/export/syns_test.yaml`.

---

## Evaluate Predictions

The next step is to compute the metrics on each of the network predictions. 
This can be done with the following command.

```shell
python api/data/eval_depth.py --mode stereo \
--pred-file models/<EXPERIMENT>/<MODEL>/<SEED>/results/pred_<DATASET>_best.npz \
--cfg-file cfg/eval/<DATASET>.yaml \
--save-file models/<EXPERIMENT>/<MODEL>/<SEED>/results/metrics_<DATASET>_best.yaml
```

Both steps can be combined into one command, which removes the need for saving intermediate predictions.
```shell
python api/data/eval_depth.py --mode stereo \
--ckpt-file models/<EXPERIMENT>/<MODEL>/<SEED>/ckpts/best.ckpt  \
--cfg-file cfg/eval/<DATASET>.yaml \
--save-file models/<EXPERIMENT>/<MODEL>/<SEED>/results/metrics_<DATASET>_best.yaml
```

Similar to exporting predictions, the config mimics those from `MonoDepthModule`. 
An additional optional parameter `target_stem` can be added to indicate the targets file to load. 
Otherwise, this will be automatically set based on the dataset split mode.
An additional `args` dict should be added, including:
* `min`: `0.001` 
* `max`: `80` if `kitti_eigen` else `100`
* `use_eigen_crop`: `True` if `kitti_eigen` else `False`
* metrics: `[eigen, pointcloud]` if `kitti_eigen`, `[benchmark, pointcloud, ibims]` if `syns_*`, else `[benchmark, pointcloud]` 
```yaml
# -----------------------------------------------------------------------------
dataset:
  type: 'kitti_lmdb'
  split: 'eigen_benchmark'
  mode: 'test'
  supp_idxs: ~
  use_depth: False
  target_stem: 'targets_test'
# -----------------------------------------------------------------------------
args:
  min: 0.001
  max: 100
  use_eigen_crop: False
  metrics: ['benchmark', 'pointcloud']
# -----------------------------------------------------------------------------
```

When evaluating on Kitti, the metrics will be saved as the average over the whole dataset. 
With SYNS, we save each image individually to allow for additional introspection based on the different scene categories. 
This is automatically determined by the script.

---

## Comparing Methods
Once the models have been trained and evaluated, you can easily generate the results tables via the notebooks: [`compare_kitti`](./compare_kitti.ipynb) & [`compare_syns`](./compare_syns.ipynb).
These notebooks will load all found metrics files for the give split, eval mode, random seeds and such.
We report performance for each model as the average performance over all random seeds. 
The StdDev can also be used to identify outliers that failed to train for some reason.

---
