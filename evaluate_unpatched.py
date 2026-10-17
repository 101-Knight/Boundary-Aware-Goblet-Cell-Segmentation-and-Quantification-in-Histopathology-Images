import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from pathlib import Path
from PIL import Image

from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    jaccard_score
)

from model import get_model

# =====================================
# DEVICE
# =====================================

device = torch.device(
    "xpu"
    if hasattr(torch, "xpu") and torch.xpu.is_available()
    else "cpu"
)

print("Using:", device)

# =====================================
# DATASET
# =====================================

class UnpatchedDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir
    ):

        self.images = sorted(
            Path(image_dir).glob("*")
        )

        self.mask_dir = Path(mask_dir)

        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (256, 256),
                interpolation=Image.NEAREST
            ),
            transforms.ToTensor()
        ])

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

        image = self.image_transform(image)

        mask = self.mask_transform(mask)

        return image, mask

# =====================================
# DATA
# =====================================

dataset = UnpatchedDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\unpatched\test\images",
    mask_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\unpatched\test\masks"
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

print("Test Images:", len(dataset))

# =====================================
# MODEL
# =====================================

model = get_model()

model.load_state_dict(
    torch.load(
        "best_unetplusplus_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

# =====================================
# METRICS
# =====================================

acc_scores = []
prec_scores = []
rec_scores = []
dice_scores = []
iou_scores = []

# =====================================
# EVALUATION
# =====================================

with torch.no_grad():

    for images, masks in loader:

        images = images.to(device)

        outputs = model(images)

        preds = torch.sigmoid(outputs)

        preds = (preds > 0.5).float()

        pred_np = (
        preds
        .cpu()
        .numpy()
        .flatten()
        .astype(np.uint8)
        )

        mask_np = (
            masks
            .cpu()
            .numpy()
            .flatten()
        )

        mask_np = (mask_np > 0.5).astype(np.uint8)
        print(
        "Mask unique values:",
        np.unique(mask_np)[:20]
        )

        print(
            "Pred unique values:",
            np.unique(pred_np)
        )
        acc = accuracy_score(
            mask_np,
            pred_np
        )

        prec = precision_score(
            mask_np,
            pred_np,
            zero_division=0
        )

        rec = recall_score(
            mask_np,
            pred_np,
            zero_division=0
        )

        iou = jaccard_score(
            mask_np,
            pred_np,
            zero_division=0
        )

        intersection = np.sum(
            pred_np * mask_np
        )

        dice = (
            2 * intersection + 1e-7
        ) / (
            pred_np.sum()
            + mask_np.sum()
            + 1e-7
        )

        acc_scores.append(acc)
        prec_scores.append(prec)
        rec_scores.append(rec)
        dice_scores.append(dice)
        iou_scores.append(iou)

# =====================================
# RESULTS
# =====================================

print("\n===== Unpatched Test Results =====")

print(
    f"Accuracy : {np.mean(acc_scores):.4f}"
)

print(
    f"Precision: {np.mean(prec_scores):.4f}"
)

print(
    f"Recall   : {np.mean(rec_scores):.4f}"
)

print(
    f"Dice     : {np.mean(dice_scores):.4f}"
)

print(
    f"IoU      : {np.mean(iou_scores):.4f}"
)