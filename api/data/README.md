# Dataset Downloading & Preprocessing

Assume the project to be located in (change as required)
```shell
REPO_ROOT=/path/to/monodepth_benchmark
```

---

## Paths
Datasets are expected to be in `$REPO_ROOT/data` by default.
Path management is done in [`src/paths.py`](../../src/paths.py).
The expected datasets are 
```python
datas: dict[str, str] = {
    'kitti_raw': 'kitti_raw_sync',
    'kitti_raw_lmdb': 'kitti_raw_sync_lmdb',
    'kitti_depth': 'kitti_depth_benchmark',
    'syns_patches': 'syns_patches',
}
```
where `keys` are the dataset identifiers in the `registry` and `values` are the stems of the path where the dataset is stored.
E.g. `kitti_raw` would be stored in `$REPO_ROOT/data/kitti_raw_sync`. 

If you wish to store datasets in other locations, create the file [`$REPO_ROOT/PATHS.yaml`](../../PATHS.yaml).
This file should not be tracked by Git, since it might contain sensitive information about your machine/server.
Populate the file with the following contents:

```yaml
# -----------------------------------------------------------------------------
MODEL_ROOTS: []

DATA_ROOTS:
  - /path/to/dataroot1
  - /path/to/dataroot2
  - /path/to/dataroot3
# -----------------------------------------------------------------------------
```
> **NOTE:** Multiple roots may be useful if training in an HPC cluster where data has to be copied locally.  

Replace the paths with the desired location(s). 
Once again, the dataset would be expected to be located in `/path/to/dataroot1/kitti_raw_sync`.
These roots should be listed in preference order, and the first existing dataset directory will be selected.

> **NOTE:** This procedure can be applied to change the default locations in which to find pretrained models. 
> Simply add the desired paths to `MODEL_ROOTS` instead.

---

## Download
Before downloading the datasets, create a placeholder directory for each benchmark dataset.
This will prevent errors from the `devkits` thinking the datasets are unavailable. 
```shell
cd $REPO_ROOT
mkdir -p data/{kitti_raw_sync,kitti_raw_sync_lmdb,kitti_depth_benchmark,syns_patches}
```

### Kitti Raw Sync
This is the base Kitti Raw Sync dataset, including the training images and velodyne LiDAR.
The zipfiles will be extracted and deleted automatically.
```shell
cd $REPO_ROOT/data
mkdir kitti_raw_sync; cd kitti_raw_sync
../../api/data/download_kitti_raw_sync.sh  # Ensure file is executable with `chdmod +x ...`
```

The expected dataset size, given by `du -shc ./*` is 
```text
62G     2011_09_26
24G     2011_09_28
6.0G    2011_09_29
51G     2011_09_30
43G     2011_10_03
184G    total
```

### Kitti Depth Benchmark 
This is the updated Kitti Benchmark dataset, including the corrected ground-truth depth maps.
The zipfiles will be extracted and deleted automatically.
```shell
cd $REPO_ROOT/data
mkdir kitti_depth_benchmark; cd kitti_depth_benchmark
../../api/data/download_kitti_bench.sh  # Ensure file is executable with `chdmod +x ...`

python ../../api/data/export_kitti_depth_benchmark.py  # Copy benchmark data to `kitti_raw_sync`
```

The expected dataset size, given by `du -shc ./*` is
```text
1.9G     depth_selection
160K     devkit
13G      train
1.1G     val
16G      total
```

The dowloaded version of `kitti_depth_benchmark` can be deleted after copying it to `kitti_raw_sync`.
Otherwise, you might be able to use it to submit to the official [Kitti Benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion).

### SYNS-Patches
The SYNS-Patches dataset can currently be downloaded from the [MDEC CodaLab website](https://codalab.lisn.upsaclay.fr/competitions/7811#participate-get_starting_kit).

> NOTE: The ground-truth for both `val` and `test` sets are not publicly released! This is to prevent overfitting from repeated evaluation. 
> If you wish to evaluate on this dataset, consider participating in the [challenge](https://codalab.lisn.upsaclay.fr/competitions/7811)!

```shell
cd $REPO_ROOT/data
../../api/data/download_syns_patches.sh  # Ensure file is executable with `chdmod +x ...`
```

The expected dataset size, given by `du -shc .` (subdirs omitted for brevity) is
```text
974M      total
```

---

## Evaluation Targets
In this section we will generate the ground-truth depth targets used for evaluation. 
First, copy the splits provided in `$ROOT/api/data` to the corresponding dataset.
Then export the targets for each test/val split and each dataset.

```shell
cd $ROOT
cp -r api/data/splits_kitti data/kitti_raw_sync/splits
cp -r api/data/splits_syns data/syns_patches/splits
api/data/export_targets.sh  # Ensure file is executable with `chdmod +x ...`
```
> **NOTE:** The test/val set for SYNS-Patches are held out.
> To generate the predictions for your submission please refer to [this section](../eval/README.md#export-predictions)

Expected number of images: 
- Kitti Eigen (test): 697
- Kitti Eigen Zhou (test): 700
- Kitti Eigen Benchmark (test): 652
- SYNS Patches (val): 400

---

## Depth Hints (Optional)
If you wish to train using proxy depth supervision from SGBM predictions, generate them using the following commands.
We follow the procedure from [DepthHints](https://arxiv.org/abs/1909.09051) and compute the hints using multiple hyperparameters and the min reconstruction loss.

```shell
api/data/export_hints.sh  # If you wish to process all splits below...

python api/data/export_kitti_hints.py --split eigen_zhou --mode train
python api/data/export_kitti_hints.py --split eigen_zhou --mode val

python api/data/export_kitti_hints.py --split eigen --mode train
python api/data/export_kitti_hints.py --split eigen --mode val

python api/data/export_kitti_hints.py --split eigen_benchmark --mode train
python api/data/export_kitti_hints.py --split eigen_benchmark --mode val
```

> **NOTE:** This takes quite a while (overnight), so only generate hints for the splits you are going actively use.
> This will cause errors when running the tests, but you can check that the desired splits have all the expected hints.

> **NOTE:** The script will check if a given hint already exists and skip it. 
> So don't worry if you need to interrupt it and carry on with your life :) 

---

## Generate LMDBs (Optional)
Once you have finished downloading and preprocessing the Kitti dataset, you can optionally convert it into LMDB format.
This should make the data load faster, as well as reduce the load on the filesystem. 
You might find this beneficial if you are training in a cluster with limited bandwidth. 

```shell
cd $REPO_ROOT/data
mkdir kitti_raw_sync_lmdb; cd kitti_raw_sync_lmdb
python ../../api/data/export_kitti_lmdb.py  --use-hints 1 --use-benchmark 1   # Change if needed to skip hints.
```

> **NOTE:** This takes quite a while (a few hours), but the script will check if a given database already exists and skip it.
> So don't worry if you need to interrupt it and carry on with your life :)

---

## Tests

You can test that all datasets have been loaded/preprocessed correctly using:
```shell
cd $REPO_ROOT
python -m pytest tests/test_data/test_kitti_raw.py
python -m pytest tests/test_data/test_kitti_raw_lmdb.py
python -m pytest tests/test_data/test_syns_patches.py
```
> **NOTE:** Some tests will fail unless all depth hints and benchmark depths have been processed.
> Ignore these errors if not needed.

---
