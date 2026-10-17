import os
import cv2
import torch
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
jaccard_score
)

from torch.utils.data import DataLoader

from dataset import GobletCellDataset
from model_unetpp_cra_scratch import (
UNetPP_CRA_Scratch
)

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
# OUTPUT FOLDERS
# =====================================

save_root = r"E:\Goblet Cell\Predictions\Original Data\Patched_img_level\Trained CRA"

pred_dir = os.path.join(
save_root,
"predicted_masks"
)

comp_dir = os.path.join(
save_root,
"comparison"
)

os.makedirs(
pred_dir,
exist_ok=True
)

os.makedirs(
comp_dir,
exist_ok=True
)

# =====================================
# MODEL
# =====================================

model = UNetPP_CRA_Scratch()

model.load_state_dict(
torch.load(
"best_unetpp_cra_scratch.pth",
map_location=device
)
)

model.to(device)
model.eval()

# =====================================
# STORAGE
# =====================================

results = []

# =====================================
# LOOP
# =====================================

with torch.no_grad():
    for idx, (images, masks) in enumerate(test_loader):

        images = images.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        preds = (
            probs > 0.5
        ).float()

        pred_np = (
            preds[0]
            .cpu()
            .squeeze()
            .numpy()
        )

        mask_np = (
            masks[0]
            .cpu()
            .squeeze()
            .numpy()
        )

        image_np = (
            images[0]
            .cpu()
            .permute(1,2,0)
            .numpy()
        )

        image_np = (
            image_np * 255
        ).astype(np.uint8)

        pred_flat = pred_np.flatten()

        mask_flat = (
            mask_np > 0.5
        ).astype(np.uint8).flatten()

        acc = accuracy_score(
            mask_flat,
            pred_flat
        )

        precision = precision_score(
            mask_flat,
            pred_flat,
            zero_division=0
        )

        recall = recall_score(
            mask_flat,
            pred_flat,
            zero_division=0
        )

        iou = jaccard_score(
            mask_flat,
            pred_flat,
            zero_division=0
        )

        intersection = (
            pred_flat * mask_flat
        ).sum()

        dice = (
            2.0 * intersection
        ) / (
            pred_flat.sum()
            + mask_flat.sum()
            + 1e-8
        )

        results.append([
            idx,
            acc,
            precision,
            recall,
            dice,
            iou
        ])

        pred_save = (
            pred_np * 255
        ).astype(np.uint8)

        cv2.imwrite(
            os.path.join(
                pred_dir,
                f"pred_{idx}.png"
            ),
            pred_save
        )

        gt_vis = (
            mask_np * 255
        ).astype(np.uint8)

        pred_vis = (
            pred_np * 255
        ).astype(np.uint8)

        gt_vis = cv2.cvtColor(
            gt_vis,
            cv2.COLOR_GRAY2BGR
        )

        pred_vis = cv2.cvtColor(
            pred_vis,
            cv2.COLOR_GRAY2BGR
        )

        comparison = np.hstack([
            image_np,
            gt_vis,
            pred_vis
        ])

        cv2.imwrite(
            os.path.join(
                comp_dir,
                f"compare_{idx}.png"
            ),
            comparison
        )

        if idx % 20 == 0:

            print(
                f"Processed "
                f"{idx+1}/{len(test_dataset)}"
            )

# =====================================
# CSV
# =====================================

df = pd.DataFrame(
results,
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
save_root,
"metrics.csv"
)

df.to_csv(
csv_path,
index=False
)

print("\nSaved predictions.")

print(
"Metrics CSV:",
csv_path
)

print(
"Comparison Images:",
comp_dir
)

print(
"Predicted Masks:",
pred_dir
)