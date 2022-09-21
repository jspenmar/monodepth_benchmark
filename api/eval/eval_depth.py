"""Script to evaluate network predictions on a target dataset."""
from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from export_depth import compute_eval_preds
from src import MonoDepthEvaluator
from src.tools import parsers
from src.typing import Metrics
from src.utils.io import load_yaml, write_yaml


def save_metrics(file: Path, metrics: Sequence[Metrics]):
    """Helper to save metrics. If any strings are present, save metrics separately. Otherwise save means."""
    print(f'\n-> Saving results to "{file}"...')
    file.parent.mkdir(exist_ok=True, parents=True)
    use_mean = all((isinstance(v, float) for v in metrics[0].values()))
    if use_mean: metrics = {k: float(np.array([m[k] for m in metrics]).mean()) for k in metrics[0]}
    write_yaml(file, metrics, mkdir=True)


def compute_eval_metrics(preds: NDArray, mode: str, cfg_file: Path) -> Sequence[Metrics]:
    """Compute evaluation metrics from network predictions.
    Predictions must be unscaled (see `compute_eval_preds`).

    :param preds: (NDArray) (b, h, w) Precomputed unscaled network predictions.
    :param mode: (str) Evaluation mode, which determines prediction scaling. {stereo, mono}
    :param cfg_file: (Path) Path to YAML config file.
    :return: (list[Metrics]) Metrics computed for each dataset item.
    """
    cfg = load_yaml(cfg_file)
    cfg_ds, cfg_args = cfg['dataset'], cfg['args']

    target_stem = cfg_ds.pop('target_stem', f'targets_{cfg.get("mode", "test")}')
    ds = parsers.get_ds(cfg_ds)
    target_file = ds.split_file.parent/f'{target_stem}.npz'
    print(f'\n-> Loading targets from "{target_file}"...')
    data = np.load(target_file, allow_pickle=True)

    evaluator = MonoDepthEvaluator(mode=mode, **cfg_args)
    metrics = evaluator.run(preds, data)
    return metrics


if __name__ == '__main__':
    parser = ArgumentParser(description='Script to evaluate network predictions on a target dataset.')
    parser.add_argument('--mode', required=True, choices={'stereo', 'mono'}, help='Evaluation mode, determining the prediction scaling.')
    parser.add_argument('--pred-file', default=None, type=Path, help='Optional `npz` path to precomputed predictions.')
    parser.add_argument('--ckpt-file', default=None, type=Path, help='Optional path to model ckpt to compute predictions.')
    parser.add_argument('--cfg-file', required=True, type=Path, help='Path to YAML eval config.')
    parser.add_argument('--save-file', default=None, type=Path, help='Path to YAML file to save evaluation metrics.')
    parser.add_argument('--overwrite', default=0, type=int, help='If 1, overwrite existing metrics files.')
    args = parser.parse_args()

    if args.save_file and args.save_file.is_file() and not args.overwrite:
        print(f'-> Evaluation file already exists "{args.save_file}"...')
        print('-> Set `--overwrite 1` to run this evaluation anyway...')
        exit()

    if args.pred_file:
        preds = np.load(args.pred_file)['pred']
    else:
        if not args.ckpt_file:
            raise ValueError('Must provide either a `--pred-file` with precomputed predictions '
                             'or a `--ckpt-file to compute predictions from!')
        cfg = load_yaml(args.cfg_file)['dataset']
        cfg.pop('target_stem')
        preds = compute_eval_preds(args.ckpt_file, cfg, args.overwrite)

    metrics = compute_eval_metrics(preds, args.mode, args.cfg_file)
    if args.save_file: save_metrics(args.save_file, metrics)
