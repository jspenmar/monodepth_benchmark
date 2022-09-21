import torch
from torch import Tensor
from torchmetrics import Metric

__all__ = ['MAE', 'RMSE', 'ScaleInvariant', 'AbsRel', 'SqRel', 'DeltaAcc']


MODES = {'raw', 'log', 'inv'}


class BaseMetric(Metric):
    higher_is_better = False
    full_state_update = False

    """Base class for depth estimation metrics."""
    def __init__(self, mode: str = 'raw', **kwargs):
        super().__init__(**kwargs)
        assert mode in MODES
        self.mode: str = mode
        self.sf: int = {'raw': 1, 'log': 100, 'inv': 1000}[self.mode]

        self.add_state('metric', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def _preprocess(self, input, /):
        """Convert input into log-depth or disparity."""
        if self.mode == 'raw':   pass
        elif self.mode == 'log': input = input.log()
        elif self.mode == 'inv': input = 1/input.clip(min=1e-3)
        return input

    def _compute(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute an error metric for a single pair.

        :param pred: (Tensor) (b, n) Predicted depth.
        :param target: (Tensor) (b, n) Target depth.
        :return: (Tensor) (b,) Computed metric.
        """
        raise NotImplementedError

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Compute an error metric for a whole batch of predictions and update the state.

        :param pred: (Tensor) (b, n) Predicted depths masked with NaNs.
        :param target: (Tensor) (b, n) Target depths masked with NaNs.
        :return:
        """
        self.metric += self.sf * self._compute(self._preprocess(pred), self._preprocess(target)).sum()
        self.total += pred.shape[0]

    def compute(self) -> Tensor:
        """Compute the average metric given the current state."""
        return self.metric / self.total


class MAE(BaseMetric):
    """Compute the mean absolute error."""
    def _compute(self, pred: Tensor, target: Tensor) -> Tensor:
        return (pred - target).abs().nanmean(dim=1)


class RMSE(BaseMetric):
    """Compute the root mean squared error."""
    def _compute(self, pred: Tensor, target: Tensor) -> Tensor:
        return (pred - target).pow(2).nanmean(dim=1).sqrt()


class ScaleInvariant(BaseMetric):
    """Compute the scale invariant error."""
    def _compute(self, pred: Tensor, target: Tensor) -> Tensor:
        err = pred - target
        return (err.pow(2).nanmean(dim=1) - err.nanmean(dim=1).pow(2)).sqrt()


class AbsRel(BaseMetric):
    """Compute the absolute relative error."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sf = 100  # As %

    def _compute(self, pred: Tensor, target: Tensor) -> Tensor:
        return ((pred - target).abs() / target).nanmean(dim=1)


class SqRel(BaseMetric):
    """Compute the absolute relative squared error."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sf = 100  # As %

    def _compute(self, pred: Tensor, target: Tensor) -> Tensor:
        return ((pred - target).pow(2) / target.pow(2)).nanmean(dim=1)


class DeltaAcc(BaseMetric):
    higher_is_better = True

    """Compute the accuracy for a given error threshold."""
    def __init__(self, delta: float, **kwargs):
        super().__init__(**kwargs)
        assert self.mode == 'raw', 'Accuracy should only be computed using raw depths.'
        self.delta: float = delta
        self.sf = 100  # As %

    def _compute(self, pred: Tensor, target: Tensor) -> Tensor:
        thresh = torch.max(target/pred, pred/target)
        return (thresh < self.delta).nansum(dim=1) / thresh.nansum(dim=1)
