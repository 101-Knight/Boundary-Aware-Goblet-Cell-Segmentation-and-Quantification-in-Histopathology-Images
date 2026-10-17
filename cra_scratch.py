import torch
import torch.nn as nn


class CRA(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.low_proj = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.high_proj = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        low_feature,
        high_feature
    ):

        low = self.low_proj(
            low_feature
        )

        high = self.high_proj(
            high_feature
        )

        fused = torch.cat(
            [low, high],
            dim=1
        )

        output = self.fuse(
            fused
        )

        return output