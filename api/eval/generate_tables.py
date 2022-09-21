"""Script to generate LaTeX tables from a series of models.
Highlights best and next best models.
"""
import pandas as pd

from src import MODEL_ROOTS
from src.tools import TableFormatter
from src.utils.io import load_yaml


def load_dfs(d):
    df = pd.json_normalize([load_yaml(f) for fs in d.values() for f in fs])
    df.index = [f'{m}' for m, fs in d.items() for i, _ in enumerate(fs)]
    return df


def main():
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    root = MODEL_ROOTS[-1]
    exp = 'benchmark'
    split = 'eigen_benchmark'
    mode = '*'
    ckpt_name = 'best'
    res = 'results'
    fname = f'kitti_{split}_{ckpt_name}_{mode}.yaml'

    metric_type = [-1, -1, -1, -1, +1, +1, +1, -1, +1, +1, +1, +1] if split == 'eigen' \
        else [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1]

    models = []
    if not models:
        fs = sorted(root.glob(f'{exp}/**/{res}/{fname}'))
        models = sorted({file.parents[2].stem for file in fs})

    print('Evalutation Models:', models)
    eval_files = {m: sorted(root.glob(f'{exp}/{m}/**/{res}/{fname}')) for m in models}

    df = load_dfs(eval_files)
    df2 = df.groupby(level=0)

    df_mean = df2.agg('mean').reindex(models)
    df_mean.columns.name = 'Mean'

    df_std = df2.agg('std').reindex(models)
    df_std.columns.name = 'StdDev'

    print(TableFormatter.from_df(df_mean, metrics=metric_type).to_latex(precision=4))


if __name__ == '__main__':
    main()
