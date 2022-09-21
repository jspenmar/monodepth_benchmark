from typing import Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from src.tools import T_from_AAt, ViewSynth, ops, parsers, to_scaled
from src.typing import BatchData, MonoDepthCfg, TensorDict
from src.utils import MultiLevelTimer, flatten_dict
from . import handlers as h

__all__ = ['MonoDepthModule']


class MonoDepthModule(pl.LightningModule):
    """A trainer class for monocular depth estimation.

    Self-supervised monocular depth estimation usually consists of the following:
        - Depth estimation network: Produces a multi-scale sigmoid disparity in the range [0, 1]
        - Pose estimation network: Produces the relative transform between two input images (i.e. support frames)
        - Reconstruction loss: Given depth and pose we can reconstruct the target view from a support frame.

    :param cfg: (MonoDepthConfig) Trainer configuration (see `src.typing.MonoDepthConfig` or example cfg)
    """
    def __init__(self, cfg: MonoDepthCfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.losses: nn.ModuleDict
        self.weights: nn.ParameterDict

        self.nets: nn.ModuleDict = parsers.get_net(self.cfg['net'])
        self.losses, self.weights = parsers.get_loss(self.cfg['loss'])
        self.metrics: nn.ModuleDict = parsers.get_metrics()

        self.synth = ViewSynth(shape=(
            self.cfg['dataset']['size'][1],  # h
            self.cfg['dataset']['size'][0],  # w
        ))

        self.current_batch: dict[str, Sequence] = {'train': [], 'val': []}  # For image logging.
        self.scales = self.nets['depth'].out_scales
        self.n_scales = len(self.scales)
        self.min_depth = cfg['trainer'].get('min_depth', 0.1)
        self.max_depth = cfg['trainer'].get('max_depth', 100)

        self.timer = MultiLevelTimer(name=self.__class__.__qualname__, as_ms=True, precision=4)

    def train_dataloader(self) -> DataLoader:
        """Return the dataloader for the training dataset."""
        return parsers.get_dl('train', self.cfg['dataset'], self.cfg['loader'])

    def val_dataloader(self) -> DataLoader:
        """Return the dataloader for the validation dataset."""
        return parsers.get_dl('val', self.cfg['dataset'], self.cfg['loader'])

    def configure_optimizers(self):
        """Create optimizer & scheduler."""
        out = {'optimizer': parsers.get_opt(self.nets, self.cfg['optimizer'])}

        if cfg := self.cfg.get('scheduler'):
            if monitor := cfg.pop('monitor', None): out['monitor'] = monitor
            out['lr_scheduler'] = parsers.get_sched(out['optimizer'], cfg)

        return out

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        """Speed up zero grad by setting parameters to `None`."""
        optimizer.zero_grad(set_to_none=True)

    def forward(self, x: TensorDict) -> TensorDict:
        """Run networks forward pass.

        NOTE: The `virtual stereo` prediction has two channels, not one.
        This is because we assume that the target frame is the central image in a trinocular setting.
        The first channel corresponds to the virtual left prediction, while the second channel is the right prediction.

        :param x: (TensorDict) Batch inputs required for network forward pass. (See `self._step`)
        :return: fwd: (TensorDict) {
            depth_feats: (list[Tensor]) List of intermediate depth encoder features.
            disp: (TensorDict) {s: (b, 1, h/2**s, w/2**s)} Predicted sigmoid disparity at each scale.

            (Optional)
            (If using `mask` prediction)
            mask: (TensorDict) {s: (b, n, h/2**s, w/2**s)} Predicted photometric mask at each scale.

            (If using `virtual stereo` prediction)
            disp_stereo: (TensorDict) {s: (b, 2, h/2**s, w/2**s)} Predicted disparity at each scale for the STEREO pair.

            (If using `mask` & `virtual stereo` prediction)
            mask_stereo: (TensorDict) {s: (b, n, h/2**s, w/2**s)} Predicted photometric mask for the stereo pair.

            (If using `pose` prediction network)
            T_{idx}: (b, 4, 4) Transform from each target frame to each support frame (excluding stereo)

            (If using `autoencoder` network)
            autoenc_imgs: (TensorDict) {s: (b, 3, h/2**s, w/2**s)} Autoencoder target image predictions.
            supp_autoenc_imgs: (TensorDict) {s: (n, b, 3, h/2**s, w/2**s)} Autoencoder support image predictions.
            autoenc_feats: (list[Tensor]) List of intermediate autoencoder target features.
            supp_autoenc_feats: (list[Tensor]) List of intermediate autoencoder support features.
        }
        """
        fwd = {}
        for key, net in self.nets.items():
            # DEPTH ESTIMATION
            # Multi-scale depth prediction: `disp_{0-3}`.
            if key == 'depth':
                fwd.update(net(x['imgs']))

            # POSE ESTIMATION
            # Relative poses wrt each support frame `T_{idx}`.
            # We always predict a forward pose, so image order is reversed and the pose inverted when `idx < 0`.
            # NOTE: Stereo pose is added during loss computation if required.
            elif key == 'pose':
                imgs = [[x['imgs'], img] if i > 0 else [img, x['imgs']]
                        for i, img in zip(x['supp_idxs'], x['supp_imgs']) if i != 0]
                imgs = torch.stack([torch.cat(img, dim=1) for img in imgs])

                Ts = net(imgs.flatten(0, 1))
                Ts = T_from_AAt(aa=Ts['R'][:, 0], t=Ts['t'][:, 0]).unflatten(0, imgs.shape[:2])

                for i, T in zip(x['supp_idxs'], Ts):
                    if i == 0: continue
                    fwd[f'T_{i}'] = T if i > 0 else T.inverse()

            # AUTOENCODER IMAGE RECONSTRUCTION
            elif key == 'autoencoder':
                fwd.update(net(x['imgs']))

                fwd2 = net(x['supp_imgs'].flatten(0, 1))
                fwd2 = ops.op(fwd2, fn='unflatten', dim=0, sizes=(len(x['supp_idxs']), -1))
                fwd.update({f'supp_{k}': v for k, v in fwd2.items()})

            else:
                raise KeyError(f'Unrecognized key: {key}.')
        return fwd

    def forward_postprocess(self, fwd: TensorDict, x: TensorDict, y: TensorDict) -> TensorDict:
        """Run network forward postprocessing.

        - Upsample (stereo) disparity & mask predictions.
        - Convert upsampled disparity to scaled depth.
        - Index the correct cam for the virtual stereo prediction.
        - Concatenate predicted & stereo support motion.

        :param fwd: (TensorDict) Network forward pass. (See `self.forward`)
        :param x: (TensorDict) Batch inputs required for network forward pass. (See `self._step`)
        :param y: (TensorDict) Batch inputs required for loss forward pass. (See `self._step`)
        :return: fwd: (TensorDict) Updated `fwd` with {
            disp_up: (TensorDict) {s: (b, 1, h, w)} Upsampled sigmoid disparity predictions.
            depth_up: (TensorDict) {s: (b, 1, h, w)} Upsampled scaled depth predictions.
            Ts: (TensorDict) (n, b, 4, 4) Predicted and/or stereo motion w.r.t. the target frame.

            (Optional)
            (If using `mask` prediction)
            mask_up: (TensorDict) {s: (b, 1, h, w)} Upsampled photometric mask predictions.

            (If using `virtual stereo` prediction)
            idx_stereo: (int) Index of the support frame corresponding to the support frame.
            disp_stereo: (TensorDict) {s: (b, 1, h/2**s, w/2**s)} Sigmoid stereo disparity predictions.
            disp_stereo_up: (TensorDict) {s: (b, 1, h, w)} Upsampled sigmoid stereo disparity predictions.
            depth_stereo_up: (TensorDict) {s: (b, 1, h, w)} Upsampled stereo scaled depth predictions.

            (If using `mask` and `virtual stereo` prediction)
            mask_stereo_up: (TensorDict) {s: (b, 1, h, w)} Upsampled photometric mask predictions for the STEREO pair.

            (If using `autoencoder` network)
            autoenc_imgs_up: (TensorDict) {s: (b, 3, h, w)} Upsampled autoencoder target image predictions.
            supp_autoenc_imgs_up: (TensorDict) {s: (n, b, 3, h, w)} Upsampled autoencoder support image predictions.
        }
        """
        # UPSAMPLE & CONVERT TO DEPTH
        fwd_new = {}
        for k, v in fwd.items():
            k_new = f'{k}_up'

            if 'disp' in k:  # {s: (b, 1, h/2**s, w/2**s)}
                disp = ops.op(v, fn=ops.interpolate_like, other=x['imgs'], mode='bilinear')
                depth = ops.op(disp, fn=to_scaled, min=self.min_depth, max=self.max_depth)

                fwd_new[k_new] = disp
                fwd_new[k_new.replace('disp', 'depth')] = {k2: v2[1] for k2, v2 in depth.items()}

            elif 'mask' in k:  # {s: (b, 1, h/2**s, w/2**s)}
                fwd_new[k_new] = ops.op(v, fn=ops.interpolate_like, other=x['imgs'], mode='bilinear')

            elif k == 'autoenc_imgs':  # {s: (b, 1, h/2**s, w/2**s)}
                fwd_new[k_new] = ops.op(v, fn=ops.interpolate_like, other=x['imgs'], mode='bilinear')

            elif k == 'supp_autoenc_imgs':  # {s: (n, b, 1, h/2**s, w/2**s)} (n=supp_imgs)
                v = ops.op(v, fn='flatten', start_dim=0, end_dim=1)
                fwd_new[k_new] = ops.op(v, fn=ops.interpolate_like, other=x['imgs'], mode='bilinear')
                fwd_new[k_new] = ops.op(fwd_new[k_new], fn='unflatten', dim=0, sizes=(len(x['supp_idxs']), -1))

        fwd.update(fwd_new)

        # VIRTUAL STEREO
        if 'disp_stereo' in fwd:
            assert 'T_stereo' in y, 'Missing stereo transform.'
 
            x['idx_stereo'] = next(i for i in x['supp_idxs'] if i == 0)  # Index of stereo images in support
            idx = (y['T_stereo'][:, 0, 3] > 0).long()  # 0 if target=l virtual=r, 1 if target=r virtual=l

            for k in {'disp_stereo', 'disp_stereo_up', 'depth_stereo_up'}:
                fwd[k] = {s: torch.stack([d[i] for i, d in zip(idx, depth)])[:, None] for s, depth in fwd[k].items()}

        # CONCATENATE POSES
        fwd['Ts'] = torch.stack([(y['T_stereo'] if i == 0 else fwd[f'T_{i}']) for i in x['supp_idxs']])
        return fwd

    def forward_loss(self, fwd: TensorDict, x: TensorDict, y: TensorDict) -> tuple[Tensor, TensorDict]:
        """Run loss forward pass.

        :param fwd: (TensorDict) Network forward pass. (See `self.forward` & `self.forward_postprocess`)
        :param x: (TensorDict) Batch inputs required for network forward pass. (See `self._step`)
        :param y: (TensorDict) Batch inputs required for loss forward pass. (See `self._step`)
        :return: (
            loss: (Tensor) (,) Total loss for optimization.
            loss_dict: {
                supp_imgs_warp: (Tensor) (n, b, 3, h, w) Support frames warped to match the target frame.

                (Optional)
                (If using automasking in `reconstruction` loss)
                automask: (Tensor) (b, 1, h, w) Boolean mask indicating pixels NOT removed by the automasking procedure.

                (If using `feature_reconstruction` loss)
                supp_feats_warp: (Tensor) (n, b, c, h, w) The warped support features to match the target frame.

                (If using `stereo_consistency` loss)
                disps_warp: (Tensor) (b*2, c, h, w) The warped disparities (first half corresponds to the virtual stereo).

                (If using proxy depth `regression` loss)
                automask_hints: (Tensor) (b, 1, h, w)Boolean mask indicating pixels NOT removed by invalid depths & automasking.

                (If using `smoothness` regularization)
                disp_grad: (Tensor) (b, 1, h, w) Disparity spatial gradients.
                image_grad: (Tensor) (b, 1, h, w) Image spatial gradients.
            }
        )
        """
        if 'idx_stereo' in x: y['imgs_stereo'] = y['supp_imgs'][x['idx_stereo']]
        loss, loss_dict = 0., {}

        for k, crit in self.losses.items():
            l2, ld2 = None, None  # Stereo loss & dict

            # IMAGED-BASED RECONSTRUCTION LOSS
            if k == 'img_recon':
                l, ld = h.image_recon(
                    crit, self.synth, depths=fwd['depth_up'], masks=fwd.get('mask_up'),
                    imgs=y['imgs'], supp_imgs=y['supp_imgs'], Ts=fwd['Ts'], Ks=y['K'],
                )

                if 'disp_stereo' in fwd:  # VIRTUAL STEREO
                    l2, ld2 = h.image_recon(
                        crit, self.synth, depths=fwd['depth_stereo_up'], masks=fwd.get('mask_stereo_up'),
                        imgs=y['imgs_stereo'], supp_imgs=y['imgs'][None], Ts=y['T_stereo'].inverse()[None], Ks=y['K'],
                    )

            # FEATURE-BASED RECONSTRUCTION LOSS
            elif k == 'feat_recon':
                feat, supp_feat = self.extract_features(fwd, x, y)
                l, ld = h.feat_recon(
                    crit, self.synth, depths=fwd['depth_up'], masks=fwd.get('mask_up'),
                    feats=feat, supp_feats=supp_feat, Ts=fwd['Ts'], Ks=y['K']
                )

            # AUTOENCODER IMAGE RECONSTRUCTION
            elif k == 'autoenc_recon':
                l, ld = h.autoenc_recon(
                    crit, preds=fwd['autoenc_imgs_up'], targets=y['imgs'],
                    supp_preds=fwd['supp_autoenc_imgs_up'], supp_targets=y['supp_imgs'],
                )

            # VIRTUAL STEREO CONSISTENCY
            elif k == 'stereo_const':
                assert 'disp_stereo' in fwd, 'Missing virtual stereo prediction "disp_stereo".'
                assert 'T_stereo' in y, 'Missing stereo pair "T_stereo".'
                l, ld = h.stereo_const(
                    crit, self.synth, disps=fwd['disp_up'], depths=fwd['depth_up'],
                    disps_stereo=fwd['disp_stereo_up'], depths_stereo=fwd['depth_stereo_up'],
                    T_stereo=y['T_stereo'], K=y['K'],
                )

            # PROXY DEPTH REGRESSION
            elif k == 'depth_regr':
                assert 'depth_hints' in y, 'Missing proxy depth prediction "depth_hints".'
                l, ld = h.depth_regr(
                    crit, self.synth, photo=self.losses['img_recon'].compute_photo,
                    depths=fwd['depth_up'], targets=y['depth_hints'], imgs=y['imgs'], supp_imgs=y['supp_imgs'],
                    Ts=fwd['Ts'], Ks=y['K'],
                )

            # DISPARITY SMOOTHNESS REGULARIZATION
            elif k == 'disp_smooth':
                l, ld = h.disp_smooth(crit, fwd['disp'], y['imgs'])
                if 'disp_stereo' in fwd:
                    l2, ld2 = h.disp_smooth(crit, fwd['disp_stereo'], y['imgs_stereo'])  # VIRTUAL STEREO

            # FEATURE FIRST-ORDER NON-SMOOTHNESS
            elif k == 'feat_peaky':
                l, ld = h.feat_smooth(crit, fwd['autoenc_feats'], y['imgs'], fwd['supp_autoenc_feats'], y['supp_imgs'])

            # FEATURE SECOND-ORDER NON-SMOOTHNESS
            elif k == 'feat_smooth':
                l, ld = h.feat_smooth(crit, fwd['autoenc_feats'], y['imgs'], fwd['supp_autoenc_feats'], y['supp_imgs'])

            # OCCLUSION REGULARIZATION
            elif k == 'disp_occ':
                l, ld = h.disp_occ(crit, fwd['disp'])
                if 'disp_stereo' in fwd: l += h.disp_occ(crit, fwd['disp_stereo'])[0]  # VIRTUAL STEREO

            # PREDICTIVE MASK REGULARIZATION
            elif k == 'disp_mask':
                assert 'mask' in fwd, 'Missing masks in predictions.'
                l, ld = h.disp_mask(crit, fwd['mask'])
                if 'mask_stereo' in fwd: l += h.disp_mask(crit, fwd['mask_stereo'])[0]  # VIRTUAL STEREO

            else:
                raise ValueError(f'Missing loss key: "{k}"')

            loss += self.weights[k] * l
            loss_dict[f'loss_{k}'] = l
            loss_dict.update(ld)

            if l2 is not None:
                assert ld2 is not None
                loss += self.weights[k] * l2
                loss_dict[f'loss_stereo_{k}'] = l2
                loss_dict.update({f'stereo_{k}': v for k, v in ld2.items()})

        return loss, loss_dict

    def step(self, batch: BatchData, mode: str = 'train') -> tuple[Tensor, TensorDict, TensorDict]:
        """Run a single training step.

        - Compute network forward pass
        - Post-process network outputs
        - Compute loss forward pass
        - Compute depth metrics
        - Log scalars

        :param batch: A single training batch consisting of (
            x: (TensorDict) {
                images: (Tensor) (b, c, h, w) Augmented target images to predict depth.
                supp_imgs: (Tensor) (n, b, c, h, w) Augmented support frames to compute relative pose.
                supp_idxs: (Tensor) (n,) Index of each support frame w.r.t. the target frame.

                (Optional)
                (If using stereo support frame)
                idx_stereo: (Tensor) (n,) Index of the stereo pair within `supp_idxs`. Added by `self.forward_postprocces`.
            }

            y: (TensorDict) {
                images: (Tensor) (b, c, h, w) Non-augmented (or standardized) target images.
                supp_imgs: (Tensor) (n, b, c, h, w) Non-augmented (or standardized) support frames.
                K: (Tensor) (b, 4, 4) Camera intrinsic parameters.

                (Optional)
                (If using stereo support frame)
                T_stereo: (Tensor) (b, 4, 4) Transform to the stereo pair.

                (If using depth validation)
                depth: (Tensor)(b, 1, h, w) Ground-truth LiDAR depth.

                (If using proxy depth supervision)
                depth_hints: (Tensor) (b, 1, h, w) Proxy stereo depth map.
            }

            m: (dict) {
                items: (str) Loaded dataset item.
                aug (list[str]): Augmentations applied to current item.
                errors: (list[str]): List of errors when loading previous items.
                data_timer (list[MultiLevelTimer]): Timing information for each item in the batch.
            }
        )

        :param mode: (str) Training phase {core, val}.
        :return: (
            loss: (Tensor) Total loss for optimization.
            loss_dict: (TensorDict) Intermediate outputs produced by the loss. (See `self.forward_loss`)
            fwd: (TensorDict) Network forward pass. (See `self.forward` & `self.forward_postprocess`)
        )
        """
        x, y, m = batch

        with self.timer('Total'):
            with self.timer('Forward'): fwd = self.forward(x)
            with self.timer('Post-Process'): fwd = self.forward_postprocess(fwd, x, y)
            with self.timer('Loss'): loss, loss_dict = self.forward_loss(fwd, x, y)
            with self.timer('Metrics'): metrics = self.compute_metrics(fwd['depth_up'][0].detach(), y['depth']) \
                if 'depth' in y else {}

        self.log_dict(flatten_dict({
            f'{mode}_losses/loss': loss,
            f'{mode}_losses': {k: v for k, v in loss_dict.items() if 'loss_' in k},
            f'{mode}_timer/Data': m['data_timer'][0].mean_elapsed(m['data_timer']),
            f'{mode}_timer/Module': self.timer.to_dict(),
            f'{mode}_metrics': metrics,
        }))
        self.timer.reset()

        return loss, loss_dict, fwd

    def training_step(self, batch: BatchData, batch_idx: int) -> Tensor:
        """Run forward training step & cache batch."""
        self.current_batch['train'] = batch
        return self.step(batch, mode='train')[0]

    def validation_step(self, batch: BatchData, batch_idx: int) -> Tensor:
        """Run forward validation step & cache batch."""
        self.current_batch['val'] = batch
        return self.step(batch, mode='val')[0]

    @torch.no_grad()
    def compute_metrics(self, pred: Tensor, target: Tensor) -> TensorDict:
        """Compute depth metrics for a dataset batch.

        :param pred: (Tensor) (b, 1, h, w) Scaled network depth predictions.
        :param target: (Tensor) (b, 1, h, w) Ground-truth LiDAR depth.
        :return: metrics: (TensorDict) Average metrics across batch.
        """
        min, max = self.min_depth, self.max_depth
        pred = ops.interpolate_like(pred, target, mode='bilinear', align_corners=False).clamp(min, max)

        mask = target > 0
        target = target.where(mask, target.new_tensor(torch.nan))
        pred = pred.where(mask, pred.new_tensor(torch.nan))

        pred, target = pred.flatten(1), target.flatten(1)
        r = target.nanmedian(dim=1, keepdim=True).values / pred.nanmedian(dim=1, keepdim=True).values
        pred *= r

        pred.clamp_(min, max), target.clamp_(min, max)
        metrics = {k: metric(pred, target) for k, metric in self.metrics.items()}
        return metrics

    @torch.no_grad()
    def extract_features(self, fwd: TensorDict, x: TensorDict, y: TensorDict) -> tuple[Tensor, Tensor]:
        if 'autoencoder' in self.nets:
            feat = fwd['autoenc_feats']
            supp_feat = fwd['supp_autoenc_feats']
        else:
            feat = fwd['depth_feats']
            supp_feat = self.nets['depth'].encoder(x['supp_imgs'].flatten(0, 1))
            supp_feat = ops.op(supp_feat, fn='unflatten', dim=0, sizes=(len(x['supp_idxs']), -1))

        return feat, supp_feat
