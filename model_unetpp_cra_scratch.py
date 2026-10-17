import torch
import torch.nn as nn

from model_unetpp_features import (
UNetPPFeatures
)

from cra_scratch import CRA

class UNetPP_CRA_Scratch(nn.Module):
    def __init__(self):

        super().__init__()

        self.backbone = UNetPPFeatures()

        self.cra = CRA(
            channels=16
        )

        self.segmentation_head = nn.Sequential(

            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                16
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                16,
                1,
                kernel_size=1
            )
        )

    def forward(
        self,
        x
    ):

        decoder_features = self.backbone(
            x
        )

        refined_features = self.cra(
            decoder_features,
            decoder_features
        )

        masks = self.segmentation_head(
            refined_features
        )

        return masks

if __name__ == "__main__":
    model = UNetPP_CRA_Scratch()

    dummy = torch.randn(
        1,
        3,
        256,
        256
    )

    with torch.no_grad():

        output = model(
            dummy
        )

    print(
        "Output Shape:",
        output.shape
    )