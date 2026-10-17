import torch
import numpy as np

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    jaccard_score
)

from dataset_cra_boundary import GobletCellBoundaryDataset
from model_unetpp_cra_boundary import UNetPP_CRABoundary

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

test_dataset = GobletCellBoundaryDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\unpatched\test\images",
    mask_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\unpatched\test\masks"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

print("Test Images:", len(test_dataset))

# =====================================
# MODEL
# =====================================

model = UNetPP_CRABoundary()

model.load_state_dict(
    torch.load(
        "best_unetpp_cra_boundary.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

# =====================================
# STORAGE
# =====================================

all_acc = []
all_precision = []
all_recall = []
all_dice = []
all_iou = []

# =====================================
# EVALUATION
# =====================================

with torch.no_grad():

    for idx, (images, masks, boundaries) in enumerate(test_loader):

        images = images.to(device)

        seg_pred, boundary_pred = model(images)

        seg_pred = torch.sigmoid(seg_pred)

        pred_mask = (
            seg_pred > 0.5
        ).float()

        pred_np = (
            pred_mask.cpu()
            .numpy()
            .flatten()
            .astype(np.uint8)
        )

        gt_np = (
            masks.cpu()
            .numpy()
            .flatten()
        )

        gt_np = (
            gt_np > 0.5
        ).astype(np.uint8)

        acc = accuracy_score(
            gt_np,
            pred_np
        )

        precision = precision_score(
            gt_np,
            pred_np,
            zero_division=0
        )

        recall = recall_score(
            gt_np,
            pred_np,
            zero_division=0
        )

        iou = jaccard_score(
            gt_np,
            pred_np,
            zero_division=0
        )

        intersection = np.sum(
            pred_np * gt_np
        )

        dice = (
            2.0 * intersection
        ) / (
            np.sum(pred_np)
            +
            np.sum(gt_np)
            +
            1e-8
        )

        all_acc.append(acc)
        all_precision.append(precision)
        all_recall.append(recall)
        all_iou.append(iou)
        all_dice.append(dice)

        if idx % 20 == 0:

            print(
                f"Image {idx+1}/{len(test_dataset)} "
                f"| Dice={dice:.4f} "
                f"| IoU={iou:.4f}"
            )

# =====================================
# RESULTS
# =====================================

avg_acc = np.mean(all_acc)
avg_precision = np.mean(all_precision)
avg_recall = np.mean(all_recall)
avg_dice = np.mean(all_dice)
avg_iou = np.mean(all_iou)

print("\n===== CRA + Boundary Results =====")

print(
    f"Accuracy : {avg_acc:.4f}"
)

print(
    f"Precision: {avg_precision:.4f}"
)

print(
    f"Recall   : {avg_recall:.4f}"
)

print(
    f"Dice     : {avg_dice:.4f}"
)

print(
    f"IoU      : {avg_iou:.4f}"
)