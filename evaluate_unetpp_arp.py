import torch
import numpy as np

from torch.utils.data import DataLoader

from dataset import GobletCellDataset
from model_unetpp_arp import UNetPP_ARP

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

# =====================================
# MODEL
# =====================================

model = UNetPP_ARP()

model.load_state_dict(
    torch.load(
        "best_unetpp_arp.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

# =====================================
# METRIC STORAGE
# =====================================

accuracy_scores = []
precision_scores = []
recall_scores = []
dice_scores = []
iou_scores = []

# =====================================
# EVALUATION
# =====================================

with torch.no_grad():

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

        pred = preds.cpu().numpy().flatten()
        gt = masks.cpu().numpy().flatten()

        tp = np.sum((pred == 1) & (gt == 1))
        tn = np.sum((pred == 0) & (gt == 0))
        fp = np.sum((pred == 1) & (gt == 0))
        fn = np.sum((pred == 0) & (gt == 1))

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        precision = tp / (tp + fp + 1e-8)

        recall = tp / (tp + fn + 1e-8)

        dice = (
            2 * tp
        ) / (
            2 * tp + fp + fn + 1e-8
        )

        iou = tp / (
            tp + fp + fn + 1e-8
        )

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        dice_scores.append(dice)
        iou_scores.append(iou)

# =====================================
# RESULTS
# =====================================

print("\n===== Average Results =====")

print(
    f"Accuracy : {np.mean(accuracy_scores):.4f}"
)

print(
    f"Precision: {np.mean(precision_scores):.4f}"
)

print(
    f"Recall   : {np.mean(recall_scores):.4f}"
)

print(
    f"Dice     : {np.mean(dice_scores):.4f}"
)

print(
    f"IoU      : {np.mean(iou_scores):.4f}"
)