from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from src import losses, regularizers as regs
from src.tools import ViewSynth, ops
from src.typing import LossData, TensorDict

__all__ = [
    'image_recon', 'feat_recon', 'autoenc_recon',
    'stereo_const', 'depth_regr',
    'disp_smooth', 'disp_occ', 'disp_mask', 'feat_smooth',
]


def image_recon(crit: losses.ReconstructionLoss,
                synth: ViewSynth,
                depths: TensorDict,
                masks: Optional[TensorDict],
                imgs: Tensor,
                supp_imgs: Tensor,
                Ts: Tensor,
                Ks: Tensor) -> LossData:
    """Compute the reconstruction loss based on the synthesized support images.

    NOTE: `loss_dict` contains only outputs from the highest resolution depth prediction (i.e. scale=0).

    :param crit: (ReconstructionLoss) The reconstruction loss to apply.
    :param synth: (ViewSynthesis) The view synthesis module.
    :param depths: (TensorDict) {s: (b, 1, h, w)} Dict mapping to the multiscale upsampled depth prediction.
    :param masks: (Optional[TensorDict]) {s: (b, 1, h, w)} Dict mapping to the multiscale upsampled mask prediction.
    :param imgs: (Tensor) (b, 3, h, w) Target images corresponding to the depth predictions.
    :param supp_imgs: (Tensor) (n, b, 3, h, w) Support frames to the target frame (prev|next|stereo).
    :param Ts: (Tensor) (n, b, 4, 4) Predicted (or baseline) motion from the target frame to each support frame.
    :param Ks: (Tensor) (b, 4, 4) Camera intrinsic parameters.
    :return: (
        loss: (Tensor) (,) Computed reconstruction loss.
        loss_dict: {
            supp_imgs_warp: (Tensor) (n, b, 3, h, w) The warped support frames.

            (Optional)
            (If using static pixel automasking)
            automask: (Tensor) (b, 1, h, w) Boolean mask indicating pixels NOT removed by the automasking procedure.
        }
    )
    """
    n_supp = supp_imgs.shape[0]
    n_sc = len(depths)

    depths = torch.stack(list(depths.values())).flatten(0, 1)  # (s*b, 1, h, w)
    masks = torch.stack(list(masks.values())).flatten(0, 1) if masks is not None else None  # (s*b, 1, h, w)
    imgs = ops.expand_dim(imgs, n_sc, insert=True).flatten(0, 1)  # (s*b, 3, h, w)

    depths = ops.expand_dim(depths, n_supp, insert=True)  # (n, s*b, 3, h, w)
    supp_imgs = ops.expand_dim(supp_imgs, n_sc, dim=1, insert=True).flatten(1, 2)  # (n, s*b, 3, h, w)
    Ts = ops.expand_dim(Ts, n_sc, dim=1, insert=True)  # (n, s*b, 4, 4)
    Ks = ops.expand_dim(Ks, (n_supp, n_sc), dim=(0, 1), insert=True)  # (n, s*b, 4, 4)

    supp_imgs_warp = synth(
        input=supp_imgs.flatten(0, 1), depth=depths.flatten(0, 1),
        T=Ts.flatten(0, 2), K=Ks.flatten(0, 2),
    )[0].unflatten(0, (n_supp, -1))  # (n, s*b, 3, h, w)

    l, ld = crit(supp_imgs_warp, imgs, source=supp_imgs, mask=masks)

    ld = ops.op(ld, fn='unflatten', dim=0, sizes=(n_sc, -1))  # (s, b, *)
    ld = {key: val[0] for key, val in ld.items()}  # Only scale 0
    ld['supp_imgs_warp'] = supp_imgs_warp.unflatten(1, (n_sc, -1))[:, 0]
    return l, ld


def feat_recon(crit: losses.ReconstructionLoss,
               synth: ViewSynth,
               depths: TensorDict,
               masks: Optional[TensorDict],
               feats: Sequence[Tensor],
               supp_feats: Sequence[Tensor],
               Ts: Tensor,
               Ks: Tensor) -> LossData:
    """Compute the feature-based reconstruction loss based on the synthesized support images.

    NOTE: This loss is only computed using the highest resolution depth map, using x4 downsampled features.

    :param crit: (ReconstructionLoss) The reconstruction loss to apply.
    :param synth: (ViewSynthesis) The view synthesis module.
    :param depths: (TensorDict) {s: (b, 1, h, w)} Dict mapping to the multiscale upsampled depth prediction.
    :param masks: (Optional[TensorDict]) {s: (b, 1, h, w)} Dict mapping to the multiscale upsampled mask prediction.
    :param feats: (Sequence[Tensor]) List of multiscale target image encoder features.
    :param supp_feats: (Sequence[Tensor]) (n, b, c, h, w) List of multiscale support image encoder features.
    :param Ts: (Tensor) (n, b, 4, 4) Predicted (or baseline) motion from the target frame to each support frame.
    :param Ks: (Tensor) (b, 4, 4) Camera instrinsic parameters.
    :return: (
        loss: (Tensor) (,) Computed reconstruction loss.
        loss_dict: {
            supp_feats_warp: (Tensor) (n, b, c, h, w) The warped support features.
        }
    )
    """
    if isinstance(feats, list):
        s = -4  # [*2, 4, 8, 16, 32] -> 4
        feats = feats[s]
        supp_feats = supp_feats[s]  # (n*b, c, h, w)

    # Do not propagate gradients through features! Unsure if needed, but just in case.
    feats, supp_feats = feats.detach(), supp_feats.detach()
    with torch.no_grad():
        feats = ops.interpolate_like(feats, depths[0], mode='bilinear')  # (b, c, h, w)

        n = supp_feats.shape[0]
        supp_feats = ops.interpolate_like(supp_feats.flatten(0, 1), depths[0], mode='bilinear')
        supp_feats = supp_feats.unflatten(0, (n, -1))  # (n, b, c, h, w)

    masks = {0: masks[0]} if masks is not None else None
    l, ld = image_recon(
        crit, synth,
        depths={0: depths[0]}, masks=masks,
        imgs=feats, supp_imgs=supp_feats,
        Ts=Ts, Ks=Ks,
    )
    ld = {'supp_feats_warp': ld.pop('supp_imgs_warp')}
    return l, ld


def autoenc_recon(crit: losses.ReconstructionLoss,
                  preds: TensorDict,
                  targets: Tensor,
                  supp_preds: TensorDict,
                  supp_targets: Tensor) -> LossData:
    """Compute the autoencoder image reconstruction loss.

    :param crit: (ReconstructionLoss) The reconstruction loss to apply.
    :param preds: (TensorDict) {s: (b, 3, h, w)} Autoencoder image predictions.
    :param targets: (Tensor) (b, 3, h, w) Target images corresponding to the depth predictions.
    :param supp_preds: (TensorDict) {s: (b, 3, h, w)} Autoencoder support image predictions.
    :param supp_targets: (Tensor) (n, b, 3, h, w) Support frames to the target frame (prev|next|stereo).
    :return: (
        loss: (Tensor) (,) Computed regression loss.
        loss_dict: {}
    )
    """
    n_sc = len(preds)

    preds = torch.stack(list(preds.values())).flatten(0, 1)  # (s*b, 3, h, w)
    supp_preds = torch.stack(list(supp_preds.values())).flatten(0, 2)  # (s*b*n, 3, h, w)

    targets = ops.expand_dim(targets, n_sc, insert=True).flatten(0, 1)
    supp_targets = ops.expand_dim(supp_targets, n_sc, insert=True).flatten(0, 2)  # (s*b*n, 3, h, w)

    l, _ = crit(torch.cat((preds, supp_preds)), torch.cat((targets, supp_targets)))
    return l, {}


def stereo_const(crit: losses.RegressionLoss,
                 synth: ViewSynth,
                 disps: TensorDict,
                 depths: TensorDict,
                 disps_stereo: TensorDict,
                 depths_stereo: TensorDict,
                 T_stereo: Tensor,
                 K: Tensor) -> LossData:
    """Compute the virtual stereo consistency loss.

    :param crit: (RegressionLoss) The reconstruction loss to apply.
    :param synth: (ViewSynthesis) The view synthesis module.
    :param disps: (TensorDict) {s: (b, 1, h, w)} Dict mapping to the multiscale upsampled disparity prediction.
    :param depths: (TensorDict) {s: (b, 1, h, w)} Dict mapping to the multiscale upsampled depth prediction.
    :param disps_stereo: (TensorDict) {s: (b, 1, h, w)} Dict mapping to the multiscale upsampled virtual stereo disparity prediction.
    :param depths_stereo: (TensorDict) {s: (b, 1, h, w)} Dict mapping to the multiscale upsampled virtual stereo depth prediction.
    :param T_stereo: (b, 4, 4) Motion from the target image to the stereo pair.
    :param K: (Tensor) (b, 4, 4) Camera instrinsic parameters.
    :return: (
        loss: (Tensor) (,) Computed virtual stereo consistency loss.
        loss_dict: {
            disps_warp: (Tensor) (b*2, c, h, w) The warped disparities (first half corresponds to the virtual stereo).
        }
    )
    """
    n_sc = len(disps)

    disps = torch.stack(list(disps.values())).flatten(0, 1)  # (s*b, 1, h, w)
    depths = torch.stack(list(depths.values())).flatten(0, 1)
    disps_stereo = torch.stack(list(disps_stereo.values())).flatten(0, 1)
    depths_stereo = torch.stack(list(depths_stereo.values())).flatten(0, 1)

    T_stereo = ops.expand_dim(T_stereo, n_sc, dim=0, insert=True).flatten(0, 1)
    K = ops.expand_dim(K, (2, n_sc), dim=(0, 1), insert=True)

    all_disps = torch.cat((disps_stereo, disps))
    all_disps_warp, all_depths_proj, _ = synth(
        input=torch.cat((disps_stereo, disps)), depth=torch.cat((depths, depths_stereo)),
        T=torch.cat((T_stereo, T_stereo.inverse())), K=K.flatten(0, 2),
    )  # (2*s*b, 1, h, w)

    stereo_disp_warp, disp_warp = all_disps_warp.chunk(2)
    l, _ = crit(all_disps, all_disps_warp)
    ld = {
        'disps_warp': disp_warp.unflatten(0, (n_sc, -1))[0],  # Only first scale
        'stereo_disps_warp': stereo_disp_warp.unflatten(0, (n_sc, -1))[0],
    }
    return l, ld


def depth_regr(crit: losses.RegressionLoss,
               synth: ViewSynth,
               photo: losses.photometric.PhotoError,
               depths: TensorDict,
               targets: Tensor,
               imgs: Tensor,
               supp_imgs: Tensor,
               Ts: Tensor,
               Ks: Tensor) -> LossData:
    """Compute the proxy depth regression loss.

    :param crit: (RegressionLoss) The reconstruction loss to apply.
    :param synth: (ViewSynthesis) The view synthesis module.
    :param photo: (PhotometricLoss) Module to compute the dense photometric error.
    :param depths: (TensorDict) {s: (b, 1, h, w)} Dict mapping to the multiscale upsampled depth prediction.
    :param targets: (Tensor) (b, 1, h, w) The proxy ground truth depth map to regress.
    :param imgs: (Tensor) (b, 3, h, w) Target images corresponding to the depth predictions.
    :param supp_imgs: (Tensor) (n, b, 3, h, w) Support frames to the target frame (prev|next|stereo).
    :param Ts: (Tensor) (n, b, 4, 4) Predicted (or baseline) motion from the target frame to each support frame.
    :param Ks: (Tensor) (b, 4, 4) Camera instrinsic parameters.
    :return: (
        loss: (Tensor) (,) Computed regression loss.
        loss_dict: {
            automask_hints: (Tensor) (b, 1, h, w) Boolean mask indicating pixels NOT removed by invalid depths & automasking.
        }
    )
    """
    ld = {}
    n_sc = len(depths)

    imgs = ops.expand_dim(imgs, n_sc, insert=True).flatten(0, 1)  # (s*b, 3, h, w)
    depths = torch.stack(list(depths.values())).flatten(0, 1)
    targets = ops.expand_dim(targets, n_sc, insert=True).flatten(0, 1)
    masks = targets > 0

    if crit.use_automask:
        n_supp = supp_imgs.shape[0]
        supp_imgs = ops.expand_dim(supp_imgs, n_sc, dim=1, insert=True).flatten(1, 2)  # (n, s*b, 3, h, w)
        Ts = ops.expand_dim(Ts, n_sc, dim=1, insert=True)  # (n, s, b, 4, 4)
        Ks = ops.expand_dim(Ks, (n_supp, n_sc), dim=(0, 1), insert=True)  # (s*b, 4, 4)

        supp_hints_warp = synth(
            input=supp_imgs.flatten(0, 1), depth=ops.expand_dim(targets, n_supp, insert=True).flatten(0, 1),
            T=Ts.flatten(0, 2), K=Ks.flatten(0, 2),
        )[0].unflatten(0, (n_supp, -1))  # (n, s*b, 3, h, w)

        supp_imgs_warp = synth(
            input=supp_imgs.flatten(0, 1), depth=ops.expand_dim(depths, n_supp, insert=True).flatten(0, 1),
            T=Ts.flatten(0, 2), K=Ks.flatten(0, 2),
        )[0].unflatten(0, (n_supp, -1))  # (n, s*b, 3, h, w)

        automask = photo(supp_imgs_warp, imgs) > photo(supp_hints_warp, imgs)
        ld['automask_hints'] = automask.unflatten(0, (n_sc, -1))[0]
        masks &= automask

    # breakpoint()
    l, ld = crit(depths, targets, masks)
    ld = {'mask_regr': ld['mask_regr'].unflatten(0, (n_sc, -1))[0]}
    return l, ld


def disp_smooth(crit: regs.SmoothReg, disps: TensorDict, imgs: Tensor) -> LossData:
    """Compute the disparity smoothness regularization.

    NOTE: This regularization is computed on the original network predictions, i.e. without upsampling.

    :param crit: (SmoothnessReg) The smoothness regularization to apply.
    :param disps: (TensorDict) {s: (b, 1, h/s**2, w/s**2)} Dict mapping to the multiscale disparity prediction.
    :param imgs: (Tensor) (b, 3, h, w) Target images corresponding to the depth predictions.
    :return: (
        loss: (,) Computed disparity smoothness regularization.
        loss_dict: {
            disp_grad: (Tensor) (b, 1, h, w) Disparity spatial gradients.
            image_grad: (Tensor) (b, 1, h, w) Image spatial gradients.
        }
    )
    """
    ls = {s: crit(disp, ops.interpolate_like(imgs, disp, mode='bilinear')) for s, disp in disps.items()}
    l = torch.stack([val[0]/2**s for s, val in ls.items()]).mean()
    ld = ls[0][1]  # Loss dict for first scale
    return l, ld


def feat_smooth(crit: Union[regs.FeatSmoothReg, regs.FeatPeakReg],
                feats: Sequence[Tensor],
                imgs: Tensor,
                supp_feats: Sequence[Tensor],
                supp_imgs: Tensor) -> LossData:
    """Compute the feature smoothness/peakiness regularization.

    NOTE: This regularization is computed on the original network predictions, i.e. without upsampling.
    NOTE: This can be used for both first-order discriminative and second-order smoothness regularizations

    :param crit: (FeatureSmoothnessReg, FeatureDiscriminationReg) The smoothness regularization to apply.
    :param feats: (Sequence[Tensor]) List of multiscale target image encoder features.
    :param imgs: (Tensor) (b, 3, h, w) Target images corresponding to the depth predictions.
    :param supp_feats: (Sequence[Tensor]) (n, b, c, h, w) List of multiscale support image encoder features.
    :param supp_imgs: (Tensor) (n, b, 3, h, w) Support frames to the target frame (prev|next|stereo).
    :return: (
        loss: (,) Computed feature smoothness regularization.
        loss_dict: {}
    )
    """
    ls = {s: crit(feat, ops.interpolate_like(imgs, feat, mode='bilinear')) for s, feat in enumerate(feats)}
    l = torch.stack([v[0]/2**s for s, v in ls.items()]).mean()

    supp_imgs = supp_imgs.flatten(0, 1)
    supp_feats = ops.op(supp_feats, fn='flatten', start_dim=0, end_dim=1)
    ls = {s: crit(feat, ops.interpolate_like(supp_imgs, feat, mode='bilinear')) for s, feat in enumerate(supp_feats)}
    l += torch.stack([v[0]/2**s for s, v in ls.items()]).mean()
    return l, {}


def disp_occ(crit: regs.OccReg, disps: TensorDict) -> LossData:
    """Compute the disparity occlusion regularization.

    NOTE: This regularization is computed on the original network predictions, i.e. without upsampling.

    :param crit: (OcclusionReg) The occlusion regularization to apply.
    :param disps: (TensorDict) {s: (b, 1, h/s**2, w/s**2)} Dict mapping to the multiscale disparity prediction.
    :return: (
        loss: (Tensor) (,) Computed disparity occlusion loss.
        loss_dict: {}
    )
    """
    ls = ops.op(disps, fn=crit)
    l = torch.stack([v[0] for v in ls.values()]).mean()
    ld = ls[0][1]  # Loss dict for first scale
    return l, ld


def disp_mask(crit: regs.MaskReg, masks: TensorDict) -> LossData:
    """Compute the predictive mask regularization.

    NOTE: This regularization is computed on the original network predictions, i.e. without upsampling.

    :param crit: (MaskReg) The mask regularization to apply.
    :param masks: (TensorDict) {s: (b, 1, h/s**2, w/s**2)} Dict mapping to the multiscale mask prediction.
    :return: (
        loss: (Tensor) (,) Computed mask regularization loss.
        loss_dict: {}
    )
    """
    ls = ops.op(masks, fn=crit)
    l = torch.stack([v[0] for v in ls.values()]).mean()
    ld = ls[0][1]  # Loss dict for first scale
    return l, ld
