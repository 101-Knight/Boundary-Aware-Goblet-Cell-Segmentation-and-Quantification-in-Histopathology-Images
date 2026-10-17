from torch.utils.data import Dataset
from pathlib import Path

from PIL import Image
from torchvision import transforms

import random

class SSLDataset(Dataset):

    def __init__(
        self,
        labeled_image_dir,
        labeled_mask_dir,
        unlabeled_image_dir
    ):

        self.labeled_images = sorted(
            Path(labeled_image_dir).glob("*")
        )

        self.mask_dir = Path(
            labeled_mask_dir
        )

        self.unlabeled_images = sorted(
            Path(unlabeled_image_dir).glob("*")
        )

        self.transform = transforms.ToTensor()

    def __len__(self):

        return len(
            self.labeled_images
        )

    def __getitem__(self, idx):

        image_path = self.labeled_images[idx]

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

        unlabeled_path = random.choice(
            self.unlabeled_images
        )

        unlabeled_image = Image.open(
            unlabeled_path
        ).convert("RGB")

        image = self.transform(image)

        mask = self.transform(mask)

        unlabeled_image = self.transform(
            unlabeled_image
        )

        return (
            image,
            mask,
            unlabeled_image
        )