from torch.utils.data import Dataset

from pathlib import Path
from PIL import Image

import cv2
import numpy as np
import torch

from torchvision import transforms

class GobletCellBoundaryDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir
    ):

        self.images = sorted(
            Path(image_dir).glob("*")
        )

        self.mask_dir = Path(mask_dir)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.images)

    def create_boundary(
        self,
        mask
    ):

        kernel = np.ones(
            (3,3),
            np.uint8
        )

        dilated = cv2.dilate(
            mask,
            kernel,
            iterations=1
        )

        eroded = cv2.erode(
            mask,
            kernel,
            iterations=1
        )

        boundary = (
            dilated - eroded
        )

        boundary = (
            boundary > 0
        ).astype(np.uint8)

        return boundary

    def __getitem__(
        self,
        idx
    ):

        image_path = self.images[idx]

        mask_path = (
            self.mask_dir
            /
            f"{image_path.stem}.png"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")
        
        mask = mask.resize(
            (256, 256),
            Image.NEAREST
        )

        mask = np.array(mask)

        mask = (
            mask > 127
        ).astype(np.uint8)

        boundary = self.create_boundary(
            mask
        )

        image = self.transform(
            image
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        boundary = torch.tensor(
            boundary,
            dtype=torch.float32
        ).unsqueeze(0)

        return (
            image,
            mask,
            boundary
        )