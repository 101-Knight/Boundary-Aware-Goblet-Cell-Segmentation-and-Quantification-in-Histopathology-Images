import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from cra import CRA

class UNetPP_CRABoundary(nn.Module):
    def __init__(self):

        super().__init__()

        self.backbone = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=16
        )

        self.cra = CRA(
            channels=16
        )

        self.seg_head = nn.Conv2d(
            16,
            1,
            kernel_size=1
        )

        self.boundary_head = nn.Conv2d(
            16,
            1,
            kernel_size=1
        )

    def forward(self, x):

        features = self.backbone(x)

        refined = self.cra(
            features,
            features
        )

        seg_out = self.seg_head(
            refined
        )

        boundary_out = self.boundary_head(
            refined
        )

        return seg_out, boundary_out
