import torch
from torch.utils.data import DataLoader

from dataset import GobletCellDataset
from model_segnet import get_model

# =====================================
# DEVICE
# =====================================

if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using:", device)

# =====================================
# DATASET
# =====================================

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

# =====================================
# MODEL
# =====================================

model = get_model()

model.load_state_dict(
    torch.load(
        "best_segnet_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

# =====================================
# METRICS
# =====================================

eps = 1e-7

total_accuracy = 0
total_precision = 0
total_recall = 0
total_dice = 0
total_iou = 0

with torch.no_grad():

    for idx, (images, masks) in enumerate(test_loader):

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
            TP
            /
            (TP + FP + eps)
        )

        recall = (
            TP
            /
            (TP + FN + eps)
        )

        dice = (
            2 * TP
            /
            (2 * TP + FP + FN + eps)
        )

        iou = (
            TP
            /
            (TP + FP + FN + eps)
        )

        total_accuracy += accuracy.item()
        total_precision += precision.item()
        total_recall += recall.item()
        total_dice += dice.item()
        total_iou += iou.item()

        print(
            f"Processed {idx+1}/{len(test_dataset)}",
            end="\r"
        )

# =====================================
# RESULTS
# =====================================

n = len(test_dataset)

print("\n")
print("====================================")
print("SEGNET TEST RESULTS")
print("====================================")

print(f"Accuracy : {total_accuracy/n:.4f}")
print(f"Precision: {total_precision/n:.4f}")
print(f"Recall   : {total_recall/n:.4f}")
print(f"Dice     : {total_dice/n:.4f}")
print(f"IoU      : {total_iou/n:.4f}")