from typing import Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.tools import TableFormatter, blend_stereo, to_inv, to_scaled
from src.typing import ArrDict, Metrics
from .metrics import metrics_benchmark, metrics_eigen, metrics_ibims, metrics_pointcloud

__all__ = ['predict_depths', 'MonoDepthEvaluator']


@torch.no_grad()
def predict_depths(net: nn.Module,
                   loader: DataLoader,
                   device: torch.device,
                   min: float = 0.1,
                   max: float = 100,
                   use_stereo_blend: bool = False) -> Tensor:
    """Compute dataset scaled disparity predictions for a trained network.

    :param net: (nn.Module) Pretrained depth estimation network.
    :param loader: (DataLoader) Dataset to compute predictions for.
    :param device: (torch.device) Device on which to compute predictions.
    :param min: (float) Min depth used to scale sigmoid disparities.
    :param max: (float) Max depth used to scale sigmoid disparities.
    :param use_stereo_blend: (bool) If `True`, apply virtual stereo blending from the flipped image.
    :return: (Tensor) (b, 1, h, w) Scaled predicted disparity for each dataset item.
    """
    preds = []
    for x, *_ in tqdm(loader):
        imgs = x['imgs'].to(device)
        if use_stereo_blend: imgs = torch.cat((imgs, imgs.flip(dims=[-1])))

        disp = net(imgs)['disp'][0]  # (b, 1, h, w)
        if use_stereo_blend:
            b = disp.shape[0]//2
            disp = blend_stereo(disp[:b], disp[b:].flip(dims=[-1]))

        preds.append(disp.cpu())

    preds = torch.cat(preds)
    preds = to_scaled(preds, min=min, max=max)[0]
    return preds


class MonoDepthEvaluator:
    """Class to evaluate unscaled network depth predictions.

    NOTE:
        - Pointcloud metrics can only be computed when camera intrinsics are present.
        - IBIMS metrics can only be computed when depth edges are present.

    :param mode: (str) Evaluation mode. {stereo, mono}
    :param metrics: (list[str]) List of metric sets to compute. {eigen, benchmark, pointcloud, ibims}
    :param min: (float) Min ground-truth depth to evaluate.
    :param max: (float) Max ground-truth depth to evaluate.
    :param use_eigen_crop: (bool) If `True` use border cropping. Should only be used with the Kitti Eigen split.
    """
    STEREO_SF = 5.4  # Fixed Kitti scaling, given that we train using an arbitrary baseline of 0.1 vs. the real 54cm.

    def __init__(self,
                 mode: str,
                 metrics: Sequence[str] = ('benchmark', 'pointcloud'),
                 min: float = 1e-3,
                 max: float = 100,
                 use_eigen_crop: bool = False):
        self.mode = mode
        self.metrics = metrics
        self.min = min
        self.max = max
        self.use_eigen_crop = use_eigen_crop

    @staticmethod
    def _get_eigen_mask(shape: tuple[int, int]) -> NDArray:
        """Helper to get the border masking introduced by Eigen."""
        h, w = shape
        crop = np.array([0.40810811 * h, 0.99189189 * h, 0.03594771 * w, 0.96405229 * w], dtype=int)
        mask = np.zeros((h, w), dtype=bool)
        mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        return mask

    def _get_mask(self, target: NDArray) -> NDArray:
        """Helper to mask ground-truth depth based on the selected range and Eigen crop."""
        mask = target > self.min
        if self.max: mask &= target < self.max
        if self.use_eigen_crop: mask &= self._get_eigen_mask(target.shape)
        return mask

    def _get_ratio(self, pred: NDArray, target: NDArray) -> float:
        """Helper to get the prediction scaling ratio based on the evaluation mode. Stereo=fixed, Mono=median."""
        return self.STEREO_SF if self.mode == 'stereo' else float(np.median(target)/np.median(pred))

    def _upsample(self, pred: NDArray, target: NDArray) -> NDArray:
        """Helper to upsample the prediction to the full target resolution."""
        h, w = target.shape
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        return pred

    def _summarize(self, metrics: Sequence[Metrics]) -> None:
        """Helper to report mean performance of the whole dataset."""
        print(f'\n-> Summarizing results...')
        keys = (k for k, v in metrics[0].items() if isinstance(v, float))
        mean_metrics = {k: float(np.array([d[k] for d in metrics if k in d]).mean()) for k in keys}
        print(TableFormatter.from_dict(mean_metrics).to_latex(precision=4))

    def _eval_single(self,
                     pred: NDArray,
                     target: NDArray,
                     mask: NDArray,
                     K: Optional[NDArray],
                     cat: Optional[str],
                     subcat: Optional[str],
                     metrics: Sequence[str]) -> Metrics:
        """Helper to compute metrics from a single prediction."""
        if mask.sum() == 0: return {}
        pred_mask, target_mask = pred[mask], target[mask]

        r = self._get_ratio(pred_mask, target_mask)
        pred, pred_mask = (r*pred).clip(self.min, self.max), (r*pred_mask).clip(self.min, self.max)

        ms = {'Ratio': r}
        if cat: ms['Cat'] = str(cat)
        if subcat: ms['SubCat'] = str(subcat)

        for m in metrics:
            if m == 'eigen':  ms.update(metrics_eigen(pred_mask, target_mask))
            elif m == 'benchmark': ms.update(metrics_benchmark(pred_mask, target_mask))
            elif m == 'pointcloud': ms.update(metrics_pointcloud(pred, target, mask, K))
            elif m == 'ibims': ms.update(metrics_ibims(pred, target, mask))

        return ms

    def run(self, preds: NDArray, data: ArrDict) -> Sequence[Metrics]:
        """Compute evaluation metrics over a whole dataset, specified by the target `data`.

        :param preds: (ndarray) (b, h, w) Unscaled disparity predictions, where `b=len(dataset)`.
        :param data: (ArrDict) Network targets (depth, *K, *edge, *cat, *subcat) loaded from an `.npz` file.
        :return: (list(Metrics)) Computed metrics for each dataset item.
        """
        targets, Ks, edges = data['depth'], data.get('K'), data.get('edge')
        cats, subcats = data.get('cat'), data.get('subcat')

        if (a := len(preds)) != (b := len(targets)): raise ValueError(f'Non-matching preds and targets! ({a} vs. {b})')

        ts = [t.astype(np.float32) for t in targets]
        # ts = ts[:100]  # For testing on a few examples.
        ms = [self._get_mask(t) for t in ts]
        ps = [to_inv(self._upsample(p, t)) for p, t in zip(preds, ts)]
        if Ks is None: Ks = [None]*len(ts)
        if cats is None: cats = [None]*len(ts)
        if subcats is None: subcats = [None]*len(ts)

        print('\n-> Computing metrics...')
        metrics = [self._eval_single(p, t, m, K, c1, c2, [m for m in self.metrics if m != 'ibims'])
                   for p, t, m, K, c1, c2 in zip(tqdm(ps), ts, ms, Ks, cats, subcats)]
        if edges is not None:
            print('\n-> Computing edges-based metrics...')
            ms = [m1 & m2 for m1, m2 in zip(ms, edges)]
            metrics_edge = [self._eval_single(p, t, m, K, c1, c2, self.metrics)
                            for p, t, m, K, c1, c2 in zip(tqdm(ps), ts, ms, Ks, cats, subcats)]
            metrics_edge = [{f'{k}-Edges': v for k, v in m.items()} for m in metrics_edge]
            metrics = [{**m1, **m2} for m1, m2 in zip(metrics, metrics_edge)]

        assert len(metrics) == len(ts), f'Non-matching metrics and targets! ({len(metrics)} vs. {len(ts)})'
        metrics = [m for m in metrics if m]
        self._summarize(metrics)
        return metrics
