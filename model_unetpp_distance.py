import torch.nn as nn
import segmentation_models_pytorch as smp

class UNetPPDistance(nn.Module):

    def __init__(self):

        super().__init__()

        self.mask_model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            classes=1
        )

        self.distance_model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            classes=1
        )

    def forward(self, x):

        mask_logits = self.mask_model(x)

        distance_map = self.distance_model(x)

        return (
            mask_logits,
            distance_map
        )