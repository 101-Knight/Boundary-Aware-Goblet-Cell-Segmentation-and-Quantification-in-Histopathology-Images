from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms

class GobletCellDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.images = sorted(Path(image_dir).glob("*"))
        self.mask_dir = Path(mask_dir)

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.mask_dir / f"{image_path.stem}.png"

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask