import signal
from pathlib import Path

import pytorch_lightning.callbacks as plc

__all__ = ['RichProgressBar', 'TQDMProgressBar', 'TrainingManager', 'DetectAnomaly']


class RichProgressBar(plc.RichProgressBar):
    """Progress bar that removes all `grad norms` from display."""
    def get_metrics(self, trainer, pl_module):
        m = super().get_metrics(trainer, pl_module)
        m = {k: v for k, v in m.items() if 'grad' not in k}
        return m


class TQDMProgressBar(plc.TQDMProgressBar):
    """Progress bar that removes all `grad norms` from display."""
    def get_metrics(self, trainer, pl_module):
        m = super().get_metrics(trainer, pl_module)
        m = {k: v for k, v in m.items() if 'grad' not in k}
        return m


class TrainingManager(plc.Callback):
    """Callback to save a dummy file as an indicator when training has started/finished."""
    # FIXME: Unsure if there are edge cases where `training` is not deleted...
    def __init__(self, ckpt_dir: Path):
        super().__init__()
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        self.fstart = ckpt_dir/'training'
        if self.fstart.is_file():
            # NOTE: Since exception happens in `__init__` we won't call `self.on_exception`.
            # This is the desired behaviour!
            raise ValueError(f'Training already in progress! ({self.fstart})')

        self.fend = ckpt_dir/'finished'
        if self.fend.is_file():
            raise ValueError(f'Training already finished! ({self.fend})')

        signal.signal(signal.SIGTERM, self._on_sigterm)

    def _cleanup(self):
        print('-> Deleting "training" file...')
        if self.fstart.is_file():
            self.fstart.unlink()
        print('-> Done! Exiting...')

    def _on_sigterm(self, signum, frame):
        """Signature required by `signal.signal`."""
        raise SystemExit  # Ensure we call `self._cleanup`.

    def on_exception(self, trainer, pl_module, exception):
        self._cleanup()

    def on_fit_start(self, trainer, pl_module):
        print('-> Creating "training" file...')
        self.fstart.touch()

    def on_fit_end(self, trainer, pl_module):
        self._cleanup()
        print('-> Creating "finished"" file...')
        self.fend.touch()


class DetectAnomaly(plc.Callback):
    """Check for NaN/infinite loss at each core step. Replacement for `detect_anomaly=True`."""
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if not (loss := outputs['loss']).isfinite():
            raise ValueError(f'Detected NaN/Infinite loss: "{loss}"')
