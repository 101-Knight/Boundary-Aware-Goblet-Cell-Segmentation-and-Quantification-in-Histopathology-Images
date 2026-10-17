import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import GobletCellDataset
from model_deeplabv3plus import get_model

# ======================================
# SETTINGS
# ======================================

MODEL_PATH = "best_deeplabv3plus_pretrained.pth"

OUTPUT_DIR = Path(
    r"E:\Goblet Cell\Predictions\Original Data\Patched_img_level\Pretrained DeepLabV3+\Batch_size at 16"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_DIR = OUTPUT_DIR / "Predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)

CSV_FILE = OUTPUT_DIR / "metrics_16_deeplabv3plus.csv"

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
# LOAD MODEL
# ======================================

model = get_model()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)
model.eval()

# ======================================
# METRICS STORAGE
# ======================================

results = []

eps = 1e-7

# ======================================
# PREDICTION LOOP
# ======================================

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

        results.append({
            "Image_ID": idx + 1,
            "Accuracy": accuracy.item(),
            "Precision": precision.item(),
            "Recall": recall.item(),
            "Dice": dice.item(),
            "IoU": iou.item()
        })

        image_np = (
            images[0]
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )

        mask_np = (
            masks[0]
            .cpu()
            .squeeze()
            .numpy()
        )

        pred_np = (
            preds[0]
            .cpu()
            .squeeze()
            .numpy()
        )

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(image_np)

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(mask_np)

        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(pred_np)

        save_path = (
            PREDICTIONS_DIR /
            f"prediction_{idx+1}.png"
        )

        plt.savefig(
            save_path,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved Image {idx+1}/{len(test_dataset)}"
        )

# ======================================
# SAVE CSV
# ======================================

df = pd.DataFrame(results)

df.to_csv(
    CSV_FILE,
    index=False
)

print("\nCSV Saved:")
print(CSV_FILE)

# ======================================
# AVERAGE RESULTS
# ======================================

print("\n===== Average Results =====")

print(
    f"Accuracy : {df['Accuracy'].mean():.4f}"
)

print(
    f"Precision: {df['Precision'].mean():.4f}"
)

print(
    f"Recall   : {df['Recall'].mean():.4f}"
)

print(
    f"Dice     : {df['Dice'].mean():.4f}"
)

print(
    f"IoU      : {df['IoU'].mean():.4f}"
)