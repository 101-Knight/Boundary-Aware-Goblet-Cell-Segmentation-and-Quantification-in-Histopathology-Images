import random

from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

from torchvision import transforms

class LabeledDataset(Dataset):
    def __init__(self, image_dir, mask_dir):

        self.images = sorted(
            Path(image_dir).glob("*")
        )

        self.mask_dir = Path(mask_dir)

        self.img_transform = transforms.ToTensor()

        self.mask_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = self.images[idx]

        mask_path = (
            self.mask_dir /
            f"{image_path.stem}.png"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")

        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        return image, mask

class UnlabeledDataset(Dataset):
    def __init__(self, image_dir):

        self.images = sorted(
            Path(image_dir).glob("*")
        )

        self.weak_aug = transforms.Compose([
            transforms.ToTensor()
        ])

        self.strong_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3
            ),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(
            self.images[idx]
        ).convert("RGB")

        weak = self.weak_aug(image)

        if random.random() > 0.5:
            strong = self.strong_aug(image)
        else:
            strong = transforms.ToTensor()(image)

        return weak, strong