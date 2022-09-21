import pytorch_lightning.callbacks as plc
import torch
from torchvision.utils import make_grid

from src.tools import ops, viz
from src.typing import BatchData, TensorDict

__all__ = ['HeavyLogger']


class HeavyLogger(plc.Callback):
    """Image logger for the forward step at the end of each epoch.

    Assumes the `pl_module` will have cached the latest batch for each `mode` in `pl_module.current_batch[mode]`.
    The logger will then run a forward pass using `pl_module.step`, which is expected to return:
        - loss: (Unused) Total loss.
        - loss_dict: Artifacts produced by losses.
        - fwd: Network outputs.

    See the documentation in `MonoDepthModule` for additional details on what each component contains.
    """
    def __init__(self, n_imgs: int = 6, n_rows: int = 2):
        self.n = n_imgs  # Max number of images to log.
        self.n_rows = n_rows  # Number of COLUMNS in grid image (yes, for some reason these are flipped).

        self.step = None
        self.logger = None
        self.mode = None

    def make_grid(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Create a grid image from the input.

        :param x: (Tensor) (b, (1|3), h, w) Input images to convert into a grid.
        :param kwargs: (dict) `kwargs` accepted by `torchvision.make_grid` (excluding `nrow`)
        :return: (Tensor) (1, 3, h*n, w*2) Gridded images, where `n` self.n/2.
        """
        return make_grid(x[:self.n], nrow=self.n_rows, **kwargs)[None]

    def write_text(self, d: dict[str, str]) -> None:
        """Log each key in a dict containing strings, prepending `mode` to each key."""
        for k, v in d.items(): self.logger.add_text(f'{self.mode}_{k}', v, global_step=self.step)

    def write_images(self, d):
        """Log each key in a dict containing images, prepending `mode` to each key."""
        for k, v in d.items(): self.logger.add_images(f'{self.mode}_{k}', v, global_step=self.step)

    @torch.no_grad()
    def on_epoch_end(self, trainer, pl_module) -> None:
        """Callback function to log images at the end of each epoch."""
        self.mode = 'train' if pl_module.training else 'val'
        self.step = pl_module.current_epoch
        self.logger = pl_module.logger.experiment

        batch = pl_module.current_batch[self.mode]
        _, loss_dict, fwd = pl_module.step(batch, mode=self.mode)

        self.log_batch(batch)
        self.log_fwd(fwd)
        self.log_loss(loss_dict, batch[0])

    def log_batch(self, batch: BatchData) -> None:
        """Log the input batch to the network.
        See `MonoDepthTrainer` for details.

        :param batch: (DatasetItem) An input batch consisting of (`x`, `y`, `meta`).
        :return:
        """
        x, y, m = batch

        # AUGMENTED IMAGES
        if self.mode == 'train':
            self.write_images({
                'imgs_aug/target': self.make_grid(ops.unstandardize(x['imgs'])),

                **{f'imgs_aug/supp_{idx}': self.make_grid(ops.unstandardize(images))
                   for idx, images in zip(x['supp_idxs'], x['supp_imgs'])},
            })

        # NON-AUGMENTED IMAGES
        self.write_images({
            'imgs/target': self.make_grid(y['imgs']),

            **{f'imgs/supp_{idx}': self.make_grid(images)
               for idx, images in zip(x['supp_idxs'], y['supp_imgs'])},
        })

        # LIDAR DEPTH
        if (depth := y.get('depth')) is not None:
            self.write_images({'depth/lidar': self.make_grid(viz.rgb_from_disp(depth, invert=True))})

        # DEPTH HINTS
        if (depth := y.get('depth_hints')) is not None:
            self.write_images({'depth/hints': self.make_grid(viz.rgb_from_disp(depth, invert=True))})

        # METADATA
        self.write_text({
            'items': ' - '.join(m['items']),
            **({'errors': ' - '.join(m['errors'])} if any(m.get('errors', [])) else {}),
            **({'aug': ' - '.join(m['augs'])} if any(m.get('augs', [])) else {}),
        })

    def log_fwd(self, fwd: TensorDict) -> None:
        """Log the network outputs.
        See `MonoDepthTrainer` for details.

        :param fwd: (TensorDict) Outputs produced by the networks. (See `MonoDepthModule.forward`)
        :return:
        """
        mask_idx = 0  # Log only mask for first support image
        self.write_images({
            **{f'feats/target_{s}': self.make_grid(viz.rgb_from_feat(f))
               for s, f in enumerate(fwd.get('depth_feats', []))},

            **{f'disp/target_{s}': self.make_grid(viz.rgb_from_disp(disp))
               for s, disp in fwd['disp'].items()},

            **{f'disp/stereo_{s}': self.make_grid(viz.rgb_from_disp(disp))
               for s, disp in fwd.get('disp_stereo', {}).items()},

            **{f'masks/mask_{mask_idx}_{s}': self.make_grid(mask[:, mask_idx][:, None])
               for s, mask in fwd.get('mask', {}).items()},

            **{f'imgs/autoenc_{s}': self.make_grid(image)
               for s, image in fwd.get('autoenc_imgs', {}).items()},

            **{f'feats/autoenc_{s}': self.make_grid(viz.rgb_from_feat(f))
               for s, f in enumerate(fwd.get('autoenc_feats', []))},
        })

    def log_loss(self, loss_dict: TensorDict, x: TensorDict) -> None:
        """Log the intermediate outputs produced by the losses.
        See `MonoDepthTrainer` for details.

        :param loss_dict: (TensorDict) Outputs produced by the losses. (See `MonoDepthModule.forward_loss`)
        :param x: (TensorDict) Batch inputs to the network.
        :return:
        """
        # SUPPORT IMAGES/FEATURES WARPED
        self.write_images({
            **{f'imgs_warp/supp_{idx}': self.make_grid(images)
               for idx, images in zip(x['supp_idxs'], loss_dict.get('supp_imgs_warp', []))},

            **{f'imgs_warp/stereo_supp_{idx}': self.make_grid(images)
               for idx, images in zip(x['supp_idxs'], loss_dict.get('stereo_supp_imgs_warp', []))},

            **{f'feats_warp/supp_{idx}': self.make_grid(viz.rgb_from_feat(feats))
               for idx, feats in zip(x['supp_idxs'], loss_dict.get('supp_feats_warp', []))},
        })

        # STEREO CONSISTENCY
        if (disp := loss_dict.get('disps_warp')) is not None:
            disp_stereo = loss_dict['stereo_disps_warp']
            self.write_images({
                'disp/warp': self.make_grid(viz.rgb_from_disp(disp)),
                'disp/stereo_warp': self.make_grid(viz.rgb_from_disp(disp_stereo))
            })

        # AUTOMASK
        if (mask := loss_dict.get('automask')) is not None:
            self.write_images({'masks/automask': self.make_grid(mask)})

        if (mask := loss_dict.get('stereo_automask')) is not None:
            self.write_images({'masks/stereo_automask': self.make_grid(mask)})

        if (mask := loss_dict.get('mask_regr')) is not None:
            self.write_images({'masks/hints': self.make_grid(mask)})

        # SPATIAL GRADIENTS
        if 'image_grad' in loss_dict:
            self.write_images({
                'grad/image': self.make_grid(loss_dict['image_grad'], normalize=True),
                'grad/disp': self.make_grid(loss_dict['disp_grad'], normalize=True)
            })

        if 'stereo_image_grad' in loss_dict:
            self.write_images({
                'grad/stereo_image': self.make_grid(loss_dict['stereo_image_grad'], normalize=True),
                'grad/stereo_disp': self.make_grid(loss_dict['stereo_disp_grad'], normalize=True)
            })
