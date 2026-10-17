import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetPPFeatures(nn.Module):

    def __init__(self):

        super().__init__()

        model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1
        )

        self.encoder = model.encoder
        self.decoder = model.decoder

    def forward(self, x):

        features = self.encoder(x)

        decoder_output = self.decoder(features)

        return decoder_output