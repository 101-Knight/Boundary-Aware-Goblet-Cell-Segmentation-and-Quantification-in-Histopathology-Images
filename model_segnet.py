import torch
import torch.nn as nn
from torchvision import models


class SegNet(nn.Module):

    def __init__(self, num_classes=1):

        super().__init__()

        vgg16 = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
        )

        features = list(vgg16.features.children())

        self.encoder = nn.Sequential(*features)

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                512, 512,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                512, 256,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                256, 128,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                num_classes,
                kernel_size=1
            )
        )

    def forward(self, x):

        x = self.encoder(x)

        x = self.decoder(x)

        return x


def get_model():

    return SegNet(num_classes=1)


if __name__ == "__main__":

    model = get_model()

    x = torch.randn(
        1,
        3,
        256,
        256
    )

    y = model(x)

    print("Output shape:", y.shape)