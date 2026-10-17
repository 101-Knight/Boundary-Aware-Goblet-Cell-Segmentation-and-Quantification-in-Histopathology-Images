import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetPPBoundary(nn.Module):

    def __init__(self):

        super().__init__()

        self.mask_model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )

        self.boundary_model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )

    def forward(self, x):

        mask_logits = self.mask_model(x)

        boundary_logits = self.boundary_model(x)

        return mask_logits, boundary_logits