from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

class BoundaryDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        boundary_dir
    ):

        self.images = sorted(
            Path(image_dir).glob("*")
        )

        self.mask_dir = Path(mask_dir)
        self.boundary_dir = Path(boundary_dir)

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = self.images[idx]

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            self.mask_dir /
            f"{image_path.stem}.png"
        ).convert("L")

        boundary = Image.open(
            self.boundary_dir /
            f"{image_path.stem}.png"
        ).convert("L")

        image = self.transform(image)

        mask = (
            self.transform(mask) > 0
        ).float()

        boundary = (
            self.transform(boundary) > 0
        ).float()

        return image, mask, boundary