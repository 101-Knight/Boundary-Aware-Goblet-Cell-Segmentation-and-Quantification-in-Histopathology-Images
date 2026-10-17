import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from arp import ARP

class UNetPP_ARP(nn.Module):
    def __init__(self):

        super().__init__()

        self.backbone = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=16
        )

        self.arp = ARP(
            channels=16
        )

        self.final_conv = nn.Conv2d(
            16,
            1,
            kernel_size=1
        )

    def forward(self, x):

        features = self.backbone(x)

        features = self.arp(
            features
        )

        masks = self.final_conv(
            features
        )

        return masks