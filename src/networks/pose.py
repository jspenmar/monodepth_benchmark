import timm
import torch
import torch.nn as nn

from src import register
from src.typing import TensorDict

__all__ = ['PoseNet']


@register('pose')
class PoseNet(nn.Module):
    """Relative pose prediction network.
    From SfM-Learner (https://arxiv.org/abs/1704.07813)

    This network predicts the relative pose between two images, concatenated channelwise.
    It consists of a ResNet encoder (with duplicated and scaled input weights) and a simple regression decoder.
    Pose is predicted as axis-angle rotation and a translation vector.

    The objective is to predict the relative pose between two images.
    The network consists of a ResNet encoder (with duplicated weights and scaled for the input images), plus a simple
    regression decoder.
    Pose is predicted as an axis-angle rotation and a translation vector.

    NOTE: Translation is not in metric scale unless training with stereo + mono.

    :param enc_name: (str) `timm` encoder key (check `timm.list_models()`).
    :param pretrained: (bool) If `True`, returns an encoder pretrained on ImageNet.
    """
    def __init__(self, enc_name: str = 'resnet18', pretrained: bool = False):
        super().__init__()
        self.enc_name = enc_name
        self.pretrained = pretrained

        self.n_imgs = 2
        self.encoder = timm.create_model(enc_name, in_chans=3 * self.n_imgs, features_only=True, pretrained=pretrained)
        self.n_chenc = self.encoder.feature_info.channels()

        self.squeeze = self.block(self.n_chenc[-1], 256, kernel_size=1)
        self.decoder = nn.Sequential(
            self.block(256, 256, kernel_size=3, stride=1, padding=1),
            self.block(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 6 * self.n_imgs, kernel_size=1),
        )

    @staticmethod
    def block(in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = 0) -> nn.Module:
        """Conv + ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> TensorDict:
        """Pose network forward pass.

        :param x: (Tensor) (b, 2*3, h, w) Channel-wise concatenated input images.
        :return: (dict[str, Tensor]) {
            R: (b, 2, 3) Predicted rotation in axis-angle (direction=axis, magnitude=angle).
            t: (b, 2, 3) Predicted translation.
        }
        """
        feat = self.encoder(x)
        out = self.decoder(self.squeeze(feat[-1]))
        out = 0.01 * out.mean(dim=(2, 3)).view(-1, self.n_imgs, 6)
        return {
            'R': out[..., :3],  # Axis-angle (Direction + Magnitude)
            't': out[..., 3:]
        }
