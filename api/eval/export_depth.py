"""Script to export network predictions on a target dataset."""
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from src import MonoDepthModule, predict_depths
from src.paths import find_model_file
from src.tools import ops, parsers
from src.utils.io import load_yaml


def save_preds(file: Path, preds: NDArray) -> None:
    """Helper to save network predictions to a NPZ file. Required for submitted to the challenge."""
    file.parent.mkdir(exist_ok=True, parents=True)
    print(f'-> Saving network predictions to "{file}"...')
    np.savez_compressed(file, pred=preds)


def compute_eval_preds(ckpt_file: Union[str, Path], cfg: dict, overwrite: bool = False) -> NDArray:
    """Compute network predictions required for evaluation.

    The confing in `cfg_dataset` is equivalent to that used by the `Trainer`.
    Note that in most cases, additional outputs, such as depth or edges can be omitted.
    Furthermore, image `size` is determined by the pretrained checkpoint.

    The config stored in `ckpt_file` is used to automatically determine:
        - Image size for network input.
        - Initial disparity scaling range.

    NOTE: The output disparities are NOT in metric depth. They are just scaled to the range expected by the network
    during training. We still need to apply fixed scaling (stereo) or median scaling (mono). This is done in the
    evaluation script by the `DepthEvaluator`.

    :param ckpt_file: (Path) Path to pretrained model checkpoint. Path can be absolute or relative to `MODEL_ROOTS`.
    :param cfg: (dict) Loaded YAML dataset config.
    :return: (ndarray) (b, h, w) Array containing unscaled network predictions for each dataset item.
    """
    device = ops.get_device()

    ckpt_file = find_model_file(ckpt_file)
    if not (ckpt_file.parent/'finished').is_file() and not overwrite:
        print(f'-> Training for "{ckpt_file}" has not finished...')
        print('-> Set `--overwrite 1` to run this evaluation anyway...')
        exit()

    hparams_file = str(ckpt_file.parents[1] / 'hparams.yaml')
    print(f'\n-> Loading model weights from "{ckpt_file}"...')
    mod = MonoDepthModule.load_from_checkpoint(ckpt_file, hparams_file=hparams_file, strict=False).eval()
    mod.freeze()

    cfg.update({
        'size': mod.cfg['dataset']['size'],
        'as_torch': True,
        'use_aug': False,
        'log_time': False,
    })
    ds = parsers.get_ds(cfg)
    dl = DataLoader(ds, batch_size=12, num_workers=4, collate_fn=ds.collate_fn, pin_memory=True)

    print(f'\n-> Computing predictions...')
    preds = predict_depths(
        mod.nets['depth'].to(device), dl, device=device,
        min=mod.min_depth, max=mod.max_depth, use_stereo_blend=False,
    )
    preds = ops.to_numpy(preds).squeeze()
    return preds


if __name__ == '__main__':
    parser = ArgumentParser(description='Script to export network predictions on a target dataset.')
    parser.add_argument('--ckpt-file', required=True, type=str, help='Absolute/relative path to pretrained model checkpoint.')
    parser.add_argument('--cfg-file', required=True, type=Path, help='Path to dataset config to compute predictions for.')
    parser.add_argument('--save-file', default=None, type=Path, help='Path to file to save prediction in.')
    parser.add_argument('--overwrite', default=0, type=int, help='If 1, overwrite existing prediction files.')
    args = parser.parse_args()

    if args.save_file and args.save_file.is_file() and not args.overwrite:
        print(f'-> Evaluation file already exists "{args.save_file}"...')
        print('-> Set `--overwrite 1` to run this evaluation anyway...')
        exit()

    cfg = load_yaml(args.cfg_file)['dataset']
    preds = compute_eval_preds(args.ckpt_file, cfg, args.overwrite)
    if args.save_file: save_preds(args.save_file, preds)
