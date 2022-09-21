from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.utils.io import load_yaml

__all__ = ['TableFormatter']


class TableFormatter:
    """Class to format a table as a LaTeX `booktabs` table.

    :param header: (Sequence[str]) (m,) Header elements represented as strings.
    :param labels: (Sequence[str]) (n,) Row names represented as strings.
    :param body: (Sequence[Sequence[float]]) (n, m) Table data for each `tag` and each `header`.
    :param metrics: (None|Sequence[int]) Value for each col indicating if a high/low value is better (+1/-1).
    """
    def __init__(self,
                 header: Sequence[str],
                 labels: Sequence[Union[str, Sequence[str]]],
                 body: Sequence[Sequence[float]],
                 metrics: Optional[Union[int, Sequence[int]]] = None):
        self.header = header
        self.labels = labels
        self.body = np.array(body)
        self.metrics = np.array(metrics)[None]
        if self.metrics.ndim == 1:
            self.metrics = self.metrics[None].repeat(len(header), axis=1)

        if not isinstance(self.labels[0], str):
            self.labels = [' '.join(l) for l in self.labels]

        shape = len(self.labels), len(self.header)
        if shape != self.shape:
            raise ValueError(f'Shape mismatch. ({shape} vs. {self.shape})')

        if self.metrics.shape[1] != self.shape[1]:
            raise ValueError(f'Metric type mismatch. ({self.metrics.shape[1]} vs. {self.shape[1]})')

        self.best_mask, self.nbest_mask = self._get_best()

    @classmethod
    def from_files(cls, files: Sequence[Path], key: Optional[Callable[[Path], str]] = None, metrics: Optional[Union[int, Sequence[int]]] = None):
        """Classmethod to create a table from a list of files.

        :param files: (Sequence[Path]) Sequence of YAML files containing results.
        :param key: (Optional[Callable[[Path], str]]) Function to convert a file name into a tag for each row.
        :param metrics: (Optional[Sequence[int]]) Value for each col indicating whether a high/low value is better (+1/-1).
        :return:
        """
        assert len(files), 'Must provide files to create table.'
        if key is None:
            key = lambda x: x.parents[2].name  # File are usually: .../<MODEL_NAME>/version_*/results/kitti_*.yaml

        return cls(
            header=list(load_yaml(files[0])),
            labels=list(map(key, files)),
            body=[list(load_yaml(f).values()) for f in files],
            metrics=metrics,
        )

    @classmethod
    def from_df(cls, df: pd.DataFrame, metrics: Optional[Union[int, Sequence[int]]] = None):
        """Classmethod to create a table from a `DataFrame`.

        :param df: (pd.DataFrame) Pandas dataframe to create the table.
        :param metrics: (Optional[Sequence[int]]) Value for each col indicating if a high/low value is better (+1/-1).
        :return:
        """
        return cls(header=df.columns, labels=df.index, body=df.to_numpy(), metrics=metrics)

    @classmethod
    def from_dict(cls, data):
        return cls(header=np.array(list(data)), labels=['Values'], body=np.array(list(data.values()))[None], metrics=None)

    def __str__(self) -> str:
        """Format as a Latex table using default parameters."""
        return self.to_latex()

    @property
    def shape(self) -> tuple[int, int]:
        """Table shape as (rows, cols)."""
        return self.body.shape

    def _to_row(self, label: str, data: Sequence[str]) -> str:
        """Create a table row."""
        return f'{label} & {" & ".join(data)} \\\\ \n'

    def _get_best(self) -> tuple[NDArray, NDArray]:
        """Get a mask indicating the `best` and `next best` performing row per column."""
        if self.metrics[0, 0] is None:
            return np.zeros_like(self.body, dtype=bool), np.zeros_like(self.body, dtype=bool)

        body = self.body * self.metrics
        best = body.max(axis=0, keepdims=True)
        best_mask = np.equal(body, best)

        if self.shape[0] > 1:  # If we have more than one row...
            body[best_mask] = -np.inf  # Mask the best item
            nbest = body.max(axis=0, keepdims=True)
            nbest_mask = np.equal(body, nbest)
        else:
            nbest_mask = np.zeros_like(body, dtype=bool)

        return best_mask, nbest_mask

    def _get_col_width(self, width: Optional[Union[int, Sequence[int]]], header: Sequence[str], body: NDArray) -> Sequence[int]:
        """Get width for each column: dynamic, fixed or specified. """
        if width is None:  # Dynamic width.
            width = np.concatenate(([list(map(len, header))], np.vectorize(len)(body)), axis=0).max(0)

        elif isinstance(width, int):  # Fixed width.
            width = [width]*self.shape[1]

        elif len(width) != self.shape[1]:
            raise ValueError('Non-matching columns.')

        return width

    def to_latex(self, caption: str = 'CAPTION', precision: int = 2, width: int = None) -> str:
        """Create a Latex booktags table.

        :param caption: (str) Table caption.
        :param precision: (int) Precision when rounding table `body`.
        :param width: (int) Row character width.
        :return: (str) LaTeX table represented as a string.
        """
        header = [h.replace('_', ' ') for h in self.header]
        labels = [l.replace('_', ' ') for l in self.labels]

        body = np.vectorize(lambda i: f'{i:.{precision}f}')(self.body).astype('<U16')
        body[self.best_mask] = [f'\\best{{{i}}}' for i in body[self.best_mask]]
        body[self.nbest_mask] = [f'\\nbest{{{i}}}' for i in body[self.nbest_mask]]

        ws = self._get_col_width(width, header, body)
        header = [f"{h:>{w}}" for h, w in zip(header, ws)]
        body = np.stack([np.vectorize(lambda i: f'{i:>{w}}')(col) for w, col in zip(ws, body.T)]).T

        table = (
            '\\begin{table}\n'
            '\\renewcommand{\\arraystretch}{1.2}\n'
            '\\centering\n'
            '\\caption{' + caption + '}\n'
            '\\begin{tabular}{@{}' + 'l'*(len(header)+1) + '@{}}\n'
            '\\toprule\n'
        )

        n = max(map(len, self.labels))
        table += self._to_row(label=' ' * n, data=header)
        table += '\\midrule\n'
        for tag, row in zip(labels, body):
            table += self._to_row(label=f'{tag:>{n}}', data=row)

        table += (
            '\\bottomrule\n'
            '\\end{tabular}\n'
            '\\end{table}\n'
        )
        return table
