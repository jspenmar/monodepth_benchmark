from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc

from src.core import HeavyLogger, MonoDepthModule
from src.paths import find_model_file
from src.typing import MonoDepthCfg
from src.utils import callbacks as cb, io


def main():
    # ARGUMENT PARSING
    # ------------------------------------------------------------------------------------------------------------------
    parser = ArgumentParser(description='Monocular depth trainer.')
    parser.add_argument('--cfg-file', '-c', required=True, type=Path, help='Path to YAML config file to load.')
    parser.add_argument('--cfg-default', '-d', default=None, type=Path, help='Default YAML config file to overwrite.')
    parser.add_argument('--ckpt-dir', '-o', default=Path('/tmp'), type=Path, help='Root path to store checkpoint in.')
    parser.add_argument('--name', '-n', required=True, type=str, help='Model name for use during saving.')
    parser.add_argument('--version', '-v', default=0, type=int, help='Model version number for use during saving.')
    parser.add_argument('--seed', '-s', default=42, type=int, help='Random generator seed.')
    args = parser.parse_args()

    fs = [f, args.cfg_file] if (f := args.cfg_default) else [args.cfg_file]
    cfg: MonoDepthCfg = io.load_merge_yaml(*fs)
    # ------------------------------------------------------------------------------------------------------------------

    # LOGGER
    # ------------------------------------------------------------------------------------------------------------------
    logger = pl.loggers.TensorBoardLogger(
        save_dir=args.ckpt_dir, name=args.name, version=f'{args.version:03}', default_hp_metric=False,
    )
    # ------------------------------------------------------------------------------------------------------------------

    # CALLBACKS
    # ------------------------------------------------------------------------------------------------------------------
    monitor = cfg['trainer'].get('monitor', 'AbsRel')
    mode = 'max' if 'Acc' in monitor else 'min'
    cb_ckpt = plc.ModelCheckpoint(
        dirpath=Path(logger.log_dir, 'models'), filename='best',
        auto_insert_metric_name=False,  # Removes '=' from filename
        monitor=f'val_metrics/{monitor}', mode=mode,
        save_last=True, save_top_k=1, verbose=True,
    )

    cbks = [
        cb_ckpt,
        plc.LearningRateMonitor(logging_interval='epoch'),
        plc.RichModelSummary(max_depth=2),
        cb.TQDMProgressBar(),
        cb.TrainingManager(Path(cb_ckpt.dirpath)),
        cb.DetectAnomaly(),
        HeavyLogger(),
    ]

    if cfg['trainer'].get('swa'):  # FIXME: Not Tested!
        cbks.append(plc.StochasticWeightAveraging(swa_epoch_start=0.5, annealing_epochs=5, swa_lrs=None))

    if cfg['trainer'].get('early_stopping'):
        cbks.append(plc.EarlyStopping(monitor=f'val_metrics/{monitor}', mode=mode, patience=5))
    # ------------------------------------------------------------------------------------------------------------------

    # CREATE MODULE
    # ------------------------------------------------------------------------------------------------------------------
    pl.seed_everything(args.seed)

    # Load model weights from pretrained model.
    if path := cfg['trainer'].get('load_ckpt'):
        path = find_model_file(path)
        print(f'Loading model from checkpoint: {path}')
        # FIXME: You might need to change `strict=False` if the cfg of the base model is not compatible!
        model = MonoDepthModule.load_from_checkpoint(path, cfg=cfg, strict=True)
    else:
        model = MonoDepthModule(cfg)

    # Resume training from earlier checkpoint.
    resume_path = None
    if cfg['trainer'].get('resume_training'):
        print('Resuming training...')
        if (path := Path(cb_ckpt.dirpath, 'last.ckpt')).is_file(): resume_path = path
        else: print(f'No previous checkpoint found in "{path.parent}". Beginning training from scratch...')
    # ------------------------------------------------------------------------------------------------------------------

    # CREATE TRAINER
    # ------------------------------------------------------------------------------------------------------------------
    num_batches = 10
    # max_epochs = config['trainer']['max_epochs']
    max_epochs = 50
    trainer = pl.Trainer(
        gpus=1, auto_select_gpus=False,

        max_epochs=max_epochs,
        limit_train_batches=num_batches, limit_val_batches=num_batches,
        # accumulate_grad_batches=cfg['trainer'].get('accumulate_grad_batches', None),
        log_every_n_steps=num_batches,

        benchmark=cfg['trainer'].get('benchmark', False),
        precision=cfg['trainer'].get('precision', 32),
        gradient_clip_val=cfg['trainer'].get('gradient_clip_val', None),
        # track_grad_norm=2,
        # detect_anomaly=True,

        logger=logger, callbacks=cbks, enable_model_summary=False,

        # fast_dev_run=True,
        # profiler='simple',
    )
    # ------------------------------------------------------------------------------------------------------------------

    # FIT
    # ------------------------------------------------------------------------------------------------------------------
    trainer.fit(model, ckpt_path=resume_path)
    # ------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
