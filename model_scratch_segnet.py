import torch
import torch.nn as nn


class SegNet(nn.Module):

    def __init__(self, num_classes=1):

        super().__init__()

        # Encoder

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.MaxPool2d(
            2,
            stride=2,
            return_indices=True
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.pool2 = nn.MaxPool2d(
            2,
            stride=2,
            return_indices=True
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.pool3 = nn.MaxPool2d(
            2,
            stride=2,
            return_indices=True
        )

        # Decoder

        self.unpool3 = nn.MaxUnpool2d(
            2,
            stride=2
        )

        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.unpool2 = nn.MaxUnpool2d(
            2,
            stride=2
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.unpool1 = nn.MaxUnpool2d(
            2,
            stride=2
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                num_classes,
                1
            )
        )

    def forward(self, x):

        x1 = self.enc1(x)
        size1 = x1.size()
        x1, ind1 = self.pool1(x1)

        x2 = self.enc2(x1)
        size2 = x2.size()
        x2, ind2 = self.pool2(x2)

        x3 = self.enc3(x2)
        size3 = x3.size()
        x3, ind3 = self.pool3(x3)

        x = self.unpool3(
            x3,
            ind3,
            output_size=size3
        )

        x = self.dec3(x)

        x = self.unpool2(
            x,
            ind2,
            output_size=size2
        )

        x = self.dec2(x)

        x = self.unpool1(
            x,
            ind1,
            output_size=size1
        )

        x = self.dec1(x)

        return x


def get_model():
    return SegNet(num_classes=1)