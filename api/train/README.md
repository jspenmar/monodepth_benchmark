# Training

---

## Train
To train a network, simply run

```shell
python api/train/train.py --cfg-file cfg/<EXP>/<MODEL>.yaml --cfg-default cfg/<EXP>/default.yaml --ckpt-dir ./models/<EXP> \
--name <MODEL> --version 42 --seed 42
```

- `cfg-file`: Path to trainer config file, containing details for networks, losses, optimizers...
- `cfg-default`: Path to config file containing default parameters, which can be overwritten by `cfg-file`. Useful when ablating single parameters.
- `ckpt-dir`: Path to root directory to store checkpoints. Final checkpoints will be saved to `ckpt-dir/<NAME>/<VERSION>/ckpts`
- `name`: Name used when saving models in `ckpt-dir`, as above. 
- `version`: Version used when saving in `ckpt-dir`, as above. Typically set to the same value as the random seed.
- `seed`: Random seed used to initialize the world. Note that reproducibility still isn't fully guaranteed.

For details on creating config files see [this README](../../src/README.md).
A good place to start is the [default config](../../cfg/default.yaml), which illustrates the various available parameters.

When testing for improvements in proposed contributions, it is typically worth training multiple models (at least 3) with different random seeds.
Performance is then compared via the average metrics obtained by each instance of the contribution models.
Note that this behaviour is already incorporated into the [generate_tables](../eval/generate_tables.py) script.
For no reason in particular, I tend to train models using seeds `42`, `195` & `335`. 

### Checkpoints
By default, the training script saves a `best.ckpt` and `last.ckpt` checkpoints.

- `last.ckpt` is saved after every epoch and should be used when resuming interrupted training. 
- `best.ckpt` is the checkpoint for the model that produced the best performance throughout training. 

The 'best' performance is determined by tracking only one specific metric, typically set to `AbsRel`.
This behaviour can be changed in the `trainer` section of the config files.

The hyperparameters used in the experiment will also be saved to `./models/<EXP>/<NAME>/<VERSION>/hparams.yaml`.
These are usually not required to reload a model, since they are also saved as part of the model `state_dict`, but they can be useful if argument names change and need to be hotfixed.

### Logging
By default, the training script logs progress to TensorBoard, located at the root of the checkpoint directory.
Scalars (e.g. losses and metrics) are logged every 100 steps, while images and other artifacts are only logged at the end of each epoch.
To monitor the performance run
```shell
tensorboard --bind_all --logdir ./models/<EXP>  # `--bind_all` for use when ssh tunnelling.
```

---

## Dev
We provide the similar [train_dev](./train_dev.py) for use when developing and debugging. 
The main differences w.r.t. [train](./train.py) are:

- Use of `TQDMProgressBar`, to allow debugging with `breakpoint()`.
- Checkpoints default to `/tmp` instead of `./models`.
- Defaults to 5 training epochs.
- Defaults to 10 train/val batches per epoch.
- Easy enabling of grad norm tracking and gradient anomaly detection.

---
