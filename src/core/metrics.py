from functools import wraps

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from torch import Tensor

from src.external_libs import ChamferDistance
from src.tools import BackprojectDepth, extract_edges, ops

__all__ = ['metrics_eigen', 'metrics_benchmark', 'metrics_pointcloud', 'metrics_ibims']


# HELPERS
# -----------------------------------------------------------------------------
def to_float(fn):
    """Helper to convert all metrics into floats."""
    @wraps(fn)
    def wrapper(*a, **kw):
        return {k: float(v) for k, v in fn(*a, **kw).items()}
    return wrapper
# -----------------------------------------------------------------------------


# EIGEN
# -----------------------------------------------------------------------------
@to_float
def metrics_eigen(pred: NDArray, target: NDArray) -> dict[str, float]:
    """Compute Kitti Eigen depth prediction metrics.
    From Eigen (https://arxiv.org/abs/1406.2283)

    NOTE: The `sq_rel` error is incorrect! The correct error is `((err_sq ** 2) / target**2).mean()`
    We use the incorrect metric for backward compatibility with the common Eigen benchmark.
    This metric has been incorrectly reported since the benchmark was introduced.

    :param pred: (ndarray) (n,) Masked predicted depth.
    :param target: (ndarray) (n,) Masked ground truth depth.
    :return: (dict) Computed depth metrics.
    """
    err = np.abs(pred - target)
    err_rel = err/target

    err_sq = err ** 2
    err_sq_rel = err_sq/target

    err_log_sq = (np.log(pred) - np.log(target)) ** 2

    thresh = np.maximum((target/pred), (pred/target))

    return {
        'AbsRel': err_rel.mean(),
        'SqRel': err_sq_rel.mean(),
        'RMSE': np.sqrt(err_sq.mean()),
        'LogRMSE': np.sqrt(err_log_sq.mean()),
        '$\\delta < 1.25$': (thresh < 1.25).mean(),
        '$\\delta < 1.25^2$': (thresh < 1.25 ** 2).mean(),
        '$\\delta < 1.25^3$': (thresh < 1.25 ** 3).mean(),
    }
# -----------------------------------------------------------------------------


# BENCHMARK
# -----------------------------------------------------------------------------
@to_float
def metrics_benchmark(pred: NDArray, target: NDArray) -> dict[str, float]:
    """Compute Kitti Benchmark depth prediction metrics.
    From Kitti (https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_depth.zip devkit/cpp/evaluate_depth.cpp L19-120)

    Base errors are reported as `m`.
    Inv errors are reported as `1/km`.
    Log errors are reported as `100*log(m)`.
    Relative errors are reported as `%`.
    This roughly aligns the significant figures for all metrics.

    :param pred: (ndarray) (n,) Masked predicted depth.
    :param target: (ndarray) (n,) Masked ground truth depth.
    :return: (dict) Computed depth metrics.
    """
    err = np.abs(pred - target)  # Units: m
    err_sq = err ** 2

    err_inv = 1000 * np.abs(1/pred - 1/target)  # Units: 1/km
    err_inv_sq = err_inv ** 2

    # NOTE: This is a DIRECTIONAL error! This is required for the SI Log loss
    # Objective is to not penalize the prediction if the errors are consistently in the same direction.
    # I.e. if the prediction could be aligned by applying a constant scale factor.
    err_log = 100 * (np.log(pred) - np.log(target))  # Units: log(m)*100
    err_log_sq = err_log ** 2

    err_rel = 100 * (err/target)  # Units: %
    err_rel_sq = 100 * (err_sq/target**2)

    return {
        'MAE': err.mean(),
        'RMSE': np.sqrt(err_sq.mean()),
        'InvMAE': err_inv.mean(),
        'InvRMSE': np.sqrt(err_inv_sq.mean()),
        'LogMAE': np.abs(err_log).mean(),
        'LogRMSE': np.sqrt(err_log_sq.mean()),
        'LogSI': np.sqrt(err_log_sq.mean() - err_log.mean() ** 2),
        'AbsRel': err_rel.mean(),
        'SqRel': err_rel_sq.mean(),
    }
# -----------------------------------------------------------------------------


# POINTCLOUD
# -----------------------------------------------------------------------------
def _metrics_pointcloud(pred: Tensor, target: Tensor, th: float) -> tuple[Tensor, Tensor]:
    """Helper to compute F-Score and IoU with different correctness thresholds."""
    P = (pred < th).float().mean()  # Precision - How many predicted points are close enough to GT?
    R = (target < th).float().mean()  # Recall - How many GT points have a predicted point close enough?
    if (P < 1e-3) and (R < 1e-3): return P, P  # No points are correct.

    f = 2*P*R / (P + R)
    iou = P*R / (P + R - (P*R))
    return f, iou


@to_float
def metrics_pointcloud(pred: NDArray, target: NDArray, mask: NDArray, K: NDArray) -> dict[str, float]:
    """Compute pointcloud-based prediction metrics.
    From Ornek: (https://arxiv.org/abs/2203.08122)

    These metrics are computed on the GPU, since Chamfer distance has quadratic complexity.
    Following the original _paper, we set the default threshold of a correct point to 10cm.
    An extra threshold is added at 20cm for informative purposes, but is not typically reported.

    :param pred: (ndarray) (h, w) Predicted depth.
    :param target: (ndarray) (h, w) Ground truth depth.
    :param mask: (ndarray) (h, w) Mask of valid pixels.
    :param K: (ndarray) (4, 4) Camera intrinsic parameters.
    :return: (dict) Computed depth metrics.
    """
    device = ops.get_device()
    pred, target, K = ops.to_torch((pred, target, K), device=device)
    K_inv = K.inverse()[None]

    backproj = BackprojectDepth(pred.shape).to(device)
    pred_pts = backproj(pred[None, None], K_inv)[:, :3, mask.flatten()]
    target_pts = backproj(target[None, None], K_inv)[:, :3, mask.flatten()]

    pred_nn, target_nn = ChamferDistance()(pred_pts.permute(0, 2, 1), target_pts.permute(0, 2, 1))
    pred_nn, target_nn = pred_nn.sqrt(), target_nn.sqrt()

    f1, iou1 = _metrics_pointcloud(pred_nn, target_nn, th=0.1)
    f2, iou2 = _metrics_pointcloud(pred_nn, target_nn, th=0.2)
    return {
        'Chamfer': pred_nn.mean() + target_nn.mean(),
        'F-Score': 100 * f1,
        'IoU': 100 * iou1,
        'F-Score-20': 100 * f2,
        'IoU-20': 100 * iou2,
    }
# -----------------------------------------------------------------------------


# EDGES
# -----------------------------------------------------------------------------
@to_float
def metrics_ibims(pred: NDArray, target: NDArray, mask: NDArray) -> dict[str, float]:
    """Compute edge-based prediction metrics.
    From IBIMS: (https://arxiv.org/abs/1805.01328v1)

    The main metrics of interest are the edge accuracy and completeness. However, we also provide the directed error.
    Edge accuracy measures how close the predicted edges are wrt the ground truth edges.
    Meanwhile, edge completeness measures how close the ground-truth edges are from the predicted ones.

    :param pred: (ndarray) (h, w) Predicted depth.
    :param target: (ndarray) (h, w) Ground truth depth.
    :param mask: (ndarray) (h, w) Mask of valid & edges pixels.
    :param K: (ndarray) (4, 4) Camera intrinsic parameters.
    :return: (dict) Computed depth metrics.
    """
    th_dir = 10  # Plane at 10 meters
    pred_dir = np.where(pred <= th_dir, 1, 0)
    target_dir = np.where(target <= th_dir, 1, 0)
    err_dir = pred_dir - target_dir

    th_edges = 10
    D_target = ndimage.distance_transform_edt(1 - mask)  # Distance of each pixel to ground truth edges

    pred_edges = extract_edges(pred, preprocess='log', sigma=1)
    D_pred = ndimage.distance_transform_edt(1 - pred_edges)  # Distance of each pixel to predicted edges
    pred_edges = pred_edges & (D_target < th_edges)  # Predicted edges close enough to real ones.

    return {
        'DirAcc': 100 * (err_dir == 0).mean(),  # Accurate order
        'Dir (-)': 100 * (err_dir == 1).mean(),  # Pred depth was underestimated
        'Dir (+)': 100 * (err_dir == -1).mean(),  # Pred depth was overestimated
        'EdgeAcc': D_target[pred_edges].mean() if pred_edges.sum() else th_edges,  # Distance from pred to target
        'EdgeComp': D_pred[mask].mean() if pred_edges.sum() else th_edges,  # Distance from target to pred
    }
# -----------------------------------------------------------------------------
