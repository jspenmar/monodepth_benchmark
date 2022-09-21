"""Model & dataset path finder.

Objective is to provide a flexible way of managing paths to data and pretrained models.
By default, we assume data is stored in `/.../monodepth_benchmark/data`,
while models should be in `/.../monodepth_benchmark/models`.
Each user can provide a custom config file in `/path/to/repo/monodepth_benchmark/PATHS.yaml`
(which should not be tracked by Git...) with additional directories in which to find models/data.

Roots should be listed in order of preference. I.e. the first existing path will be given priority.
"""
import warnings
from pathlib import Path

from src.utils.io import load_yaml

__all__ = ['MODEL_PATHS', 'DATA_PATHS', 'MODEL_ROOTS', 'DATA_ROOTS', 'find_data_dir', 'find_model_file']


# HELPERS
# -----------------------------------------------------------------------------
_msg = 'Additional roots file "{file}" does not exist! ' \
       'To silence this warning, create the specified file with the following contents (without backquotes):\n' \
       '```\n' \
       '# -----------------------------------------------------------------------------\n' \
       'MODEL_ROOTS: []\n' \
       'DATA_ROOTS: []\n' \
       '# -----------------------------------------------------------------------------\n' \
       '```\n\n'


def _load_roots():
    """Helper to load the additional model & data roots from the repo config."""
    file = REPO_ROOT/'PATHS.yaml'
    if file.is_file():
        paths = load_yaml(file)
        model_roots = [Path(p) for p in paths['MODEL_ROOTS']]
        data_roots = [Path(p) for p in paths['DATA_ROOTS']]
    else:
        warnings.warn(_msg.format(file=file))
        model_roots, data_roots = [], []
    return model_roots, data_roots


def _build_paths(names: dict[str, str], roots: list[Path]):
    """Helper to build the paths from a list of possible `roots`.
    NOTE: This returns the FIRST found path given by the order of roots. I.e. ordered by priority.
    """
    paths = {}
    for k, v in names.items():
        try:
            paths[k] = next(p for r in roots if (p := r/v).exists())
            print(f'Found path "{k}": {paths[k]}')
        except StopIteration:
            warnings.warn(f'No valid path found for "{k}"!')

    return paths
# -----------------------------------------------------------------------------


# CONSTANTS
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parents[1]  # Path to `/.../monodepth_benchmark`

MODEL_ROOTS, DATA_ROOTS = _load_roots()
MODEL_ROOTS.append(REPO_ROOT/'models')
DATA_ROOTS.append(REPO_ROOT/'data')

models: dict[str, str] = {}

datas: dict[str, str] = {
    'kitti_raw': 'kitti_raw_sync',
    'kitti_raw_lmdb': 'kitti_raw_sync_lmdb',
    'kitti_depth': 'kitti_depth_benchmark',
    'syns_patches': 'syns_patches',
}
# -----------------------------------------------------------------------------


# BUILD PATHS
# -----------------------------------------------------------------------------
def find_model_file(name: str) -> Path:
    """Helper to find a model file in the available roots."""
    if (p := Path(name)).is_file(): return p

    try: return next(p for r in MODEL_ROOTS if (p := r/name).is_file())
    except StopIteration: raise FileNotFoundError(f'No valid path found for {name} in {MODEL_ROOTS}...')


def find_data_dir(name: str) -> Path:
    """Helper to find a dataset directory in the available roots."""
    if (p := Path(name)).is_dir(): return p

    try: return next(p for r in DATA_ROOTS if (p := r/name).is_file())
    except StopIteration: raise FileNotFoundError(f'No valid path found for {name} in {DATA_ROOTS}...')


MODEL_PATHS: dict[str, Path] = _build_paths(models, MODEL_ROOTS)
DATA_PATHS: dict[str, Path] = _build_paths(datas, DATA_ROOTS)
# -----------------------------------------------------------------------------
