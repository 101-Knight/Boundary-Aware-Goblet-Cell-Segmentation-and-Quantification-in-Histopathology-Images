import torch
import torch.nn as nn
import torch.nn.functional as F

class CRA(nn.Module):

    def __init__(
        self,
        channels
    ):

        super().__init__()

        self.conv_low = nn.Conv2d(
            channels,
            channels,
            1
        )

        self.conv_high = nn.Conv2d(
            channels,
            channels,
            1
        )

    def forward(
        self,
        low_feat,
        high_feat
    ):

        high_feat = F.interpolate(
            high_feat,
            size=low_feat.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        low_feat = self.conv_low(
            low_feat
        )

        high_feat = self.conv_high(
            high_feat
        )

        relation = torch.sigmoid(
            low_feat * high_feat
        )

        fused = (
            low_feat
            +
            relation * high_feat
        )

        return fused