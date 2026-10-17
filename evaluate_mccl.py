import torch
import numpy as np

from torch.utils.data import DataLoader

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
jaccard_score
)

from dataset import GobletCellDataset
from model import get_model

device = torch.device(
"xpu"
if hasattr(torch, "xpu")
and torch.xpu.is_available()
else "cpu"
)

print("Using:", device)

dataset = GobletCellDataset(
image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\images",
mask_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\masks"
)

loader = DataLoader(
dataset,
batch_size=1,
shuffle=False
)

print("Test Images:", len(dataset))

model = get_model()

model.load_state_dict(
    torch.load(
    "best_mccl_student.pth",
    map_location=device
    )
)

model.to(device)
model.eval()

acc_scores = []
prec_scores = []
rec_scores = []
dice_scores = []
iou_scores = []

with torch.no_grad():

    for images, masks in loader:

        images = images.to(device)

        outputs = model(images)

        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

        pred_np = (
            preds.cpu()
            .numpy()
            .flatten()
            .astype(np.uint8)
        )

        mask_np = (
            masks.cpu()
            .numpy()
            .flatten()
        )

        mask_np = (
            mask_np > 0.5
        ).astype(np.uint8)

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

print("\n===== MCCL Results =====")

print(f"Accuracy : {np.mean(acc_scores):.4f}")
print(f"Precision: {np.mean(prec_scores):.4f}")
print(f"Recall   : {np.mean(rec_scores):.4f}")
print(f"Dice     : {np.mean(dice_scores):.4f}")
print(f"IoU      : {np.mean(iou_scores):.4f}")