import torch
from torch.utils.data import DataLoader

from dataset import GobletCellDataset
from model_deeplabv3 import get_model

# ======================================
# DEVICE
# ======================================

if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

print("Using device:", device)

# ======================================
# DATASET
# ======================================

test_dataset = GobletCellDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\images",
    mask_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\masks"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

print("Test Images:", len(test_dataset))

# ======================================
# MODEL
# ======================================

model = get_model()

model.load_state_dict(
    torch.load(
        "best_deeplabv3_pretrained.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

eps = 1e-7

accuracy_sum = 0
precision_sum = 0
recall_sum = 0
dice_sum = 0
iou_sum = 0

# ======================================
# EVALUATION
# ======================================

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

        preds_flat = preds.view(-1)
        masks_flat = masks.view(-1)

        TP = (preds_flat * masks_flat).sum()

        FP = (
            preds_flat *
            (1 - masks_flat)
        ).sum()

        FN = (
            (1 - preds_flat) *
            masks_flat
        ).sum()

        TN = (
            (1 - preds_flat) *
            (1 - masks_flat)
        ).sum()

        accuracy = (
            (TP + TN)
            /
            (TP + TN + FP + FN + eps)
        )

        precision = (
            TP /
            (TP + FP + eps)
        )

        recall = (
            TP /
            (TP + FN + eps)
        )

        dice = (
            2 * TP
            /
            (2 * TP + FP + FN + eps)
        )

        iou = (
            TP /
            (TP + FP + FN + eps)
        )

        accuracy_sum += accuracy.item()
        precision_sum += precision.item()
        recall_sum += recall.item()
        dice_sum += dice.item()
        iou_sum += iou.item()

# ======================================
# FINAL RESULTS
# ======================================

n = len(test_dataset)

print("\n===== DeepLabV3 Evaluation Results =====")

print(
    f"Accuracy : {accuracy_sum/n:.4f}"
)

print(
    f"Precision: {precision_sum/n:.4f}"
)

print(
    f"Recall   : {recall_sum/n:.4f}"
)

print(
    f"Dice     : {dice_sum/n:.4f}"
)

print(
    f"IoU      : {iou_sum/n:.4f}"
)