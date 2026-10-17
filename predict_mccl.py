import os
import cv2
import torch
import numpy as np
import pandas as pd

from PIL import Image

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
jaccard_score
)

from torch.utils.data import DataLoader

from dataset import GobletCellDataset
from model import get_model

# =====================================

# DEVICE

# =====================================

device = torch.device(
"xpu"
if hasattr(torch, "xpu")
and torch.xpu.is_available()
else "cpu"
)

print("Using:", device)

# =====================================

# OUTPUT FOLDERS

# =====================================

SAVE_ROOT = r"E:\Goblet Cell\Predictions\Original Data\Patched_img_level\MCCL\50%-50%"

MASK_DIR = os.path.join(
SAVE_ROOT,
"predicted_masks"
)

COMPARE_DIR = os.path.join(
SAVE_ROOT,
"comparison"
)

os.makedirs(
MASK_DIR,
exist_ok=True
)

os.makedirs(
COMPARE_DIR,
exist_ok=True
)

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

model = get_model()

model.load_state_dict(
torch.load(
"best_mccl_student.pth",
map_location=device
)
)

model.to(device)
model.eval()

# =====================================

# STORAGE

# =====================================

all_metrics = []

# =====================================

# PREDICTION LOOP

# =====================================

with torch.no_grad():
    for idx, (images, masks) in enumerate(loader):

        images = images.to(device)

        outputs = model(images)

        preds = torch.sigmoid(
            outputs
        )

        preds = (
            preds > 0.5
        ).float()

        # -------------------------
        # ORIGINAL IMAGE
        # -------------------------

        image_np = (
            images[0]
            .cpu()
            .permute(1,2,0)
            .numpy()
        )

        image_np = (
            image_np * 255
        ).astype(np.uint8)

        # -------------------------
        # GT MASK
        # -------------------------

        gt_mask = (
            masks[0]
            .cpu()
            .squeeze()
            .numpy()
        )

        gt_mask_bin = (
            gt_mask > 0.5
        ).astype(np.uint8)

        gt_mask_vis = (
            gt_mask_bin * 255
        ).astype(np.uint8)

        # -------------------------
        # PRED MASK
        # -------------------------

        pred_mask = (
            preds[0]
            .cpu()
            .squeeze()
            .numpy()
        )

        pred_mask_bin = (
            pred_mask > 0.5
        ).astype(np.uint8)

        pred_mask_vis = (
            pred_mask_bin * 255
        ).astype(np.uint8)

        # -------------------------
        # SAVE PRED MASK
        # -------------------------

        Image.fromarray(
            pred_mask_vis
        ).save(
            os.path.join(
                MASK_DIR,
                f"pred_{idx}.png"
            )
        )

        # -------------------------
        # METRICS
        # -------------------------

        gt_flat = gt_mask_bin.flatten()

        pred_flat = pred_mask_bin.flatten()

        acc = accuracy_score(
            gt_flat,
            pred_flat
        )

        prec = precision_score(
            gt_flat,
            pred_flat,
            zero_division=0
        )

        rec = recall_score(
            gt_flat,
            pred_flat,
            zero_division=0
        )

        iou = jaccard_score(
            gt_flat,
            pred_flat,
            zero_division=0
        )

        intersection = np.sum(
            gt_flat * pred_flat
        )

        dice = (
            2 * intersection + 1e-7
        ) / (
            gt_flat.sum()
            + pred_flat.sum()
            + 1e-7
        )

        all_metrics.append(
            [
                idx,
                acc,
                prec,
                rec,
                dice,
                iou
            ]
        )

        # -------------------------
        # COMPARISON IMAGE
        # -------------------------

        gt_rgb = cv2.cvtColor(
            gt_mask_vis,
            cv2.COLOR_GRAY2BGR
        )

        pred_rgb = cv2.cvtColor(
            pred_mask_vis,
            cv2.COLOR_GRAY2BGR
        )

        comparison = np.hstack(
            [
                image_np,
                gt_rgb,
                pred_rgb
            ]
        )

        cv2.putText(
            comparison,
            "Image",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            comparison,
            "Ground Truth",
            (
                image_np.shape[1]+10,
                30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            comparison,
            "Prediction",
            (
                image_np.shape[1]*2+10,
                30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imwrite(
            os.path.join(
                COMPARE_DIR,
                f"compare_{idx}.png"
            ),
            comparison
        )

        if idx % 20 == 0:

            print(
                f"Processed {idx+1}/{len(dataset)}"
            )

# =====================================

# SAVE CSV

# =====================================

df = pd.DataFrame(
all_metrics,
columns=[
"Image",
"Accuracy",
"Precision",
"Recall",
"Dice",
"IoU"
]
)

csv_path = os.path.join(
SAVE_ROOT,
"metrics_mccl.csv"
)

df.to_csv(
csv_path,
index=False
)

print("\nDone.")
print("Predictions :", MASK_DIR)
print("Comparison  :", COMPARE_DIR)
print("Metrics CSV :", csv_path)

print("\nAverage Metrics")

print(
"Accuracy:",
df["Accuracy"].mean()
)

print(
"Precision:",
df["Precision"].mean()
)

print(
"Recall:",
df["Recall"].mean()
)

print(
"Dice:",
df["Dice"].mean()
)

print(
"IoU:",
df["IoU"].mean()
)
