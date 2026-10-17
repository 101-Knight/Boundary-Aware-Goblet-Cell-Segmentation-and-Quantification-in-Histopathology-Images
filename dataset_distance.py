from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

class DistanceDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        distance_dir
    ):

        self.images = sorted(
            Path(image_dir).glob("*")
        )

        self.mask_dir = Path(mask_dir)
        self.distance_dir = Path(distance_dir)

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

        distance = Image.open(
            self.distance_dir /
            f"{image_path.stem}.png"
        ).convert("L")

        image = self.transform(image)

        mask = (
            self.transform(mask) > 0
        ).float()

        distance = (
            self.transform(distance)
        ).float()

        return image, mask, distance