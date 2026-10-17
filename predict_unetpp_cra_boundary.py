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
# PATHS
# =====================================

IMAGE_DIR = r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\unpatched\test\images"

MASK_DIR = r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\unpatched\test\masks"

MODEL_PATH = "best_unetpp_cra_boundary.pth"

SAVE_DIR = Path(
    r"E:\Goblet Cell\Predictions\Original Data\Unpatched\CRA + Boundary + Unet++"
)

# =====================================
# OUTPUT FOLDERS
# =====================================

pred_mask_dir = SAVE_DIR / "predicted_masks"
boundary_dir = SAVE_DIR / "boundary_predictions"
comparison_dir = SAVE_DIR / "comparison"

pred_mask_dir.mkdir(
    parents=True,
    exist_ok=True
)

boundary_dir.mkdir(
    parents=True,
    exist_ok=True
)

comparison_dir.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================
# DATASET
# =====================================

dataset = GobletCellBoundaryDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR
)

print("Test Images:", len(dataset))

# =====================================
# MODEL
# =====================================

model = UNetPP_CRABoundary()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)
model.eval()

# =====================================
# METRICS STORAGE
# =====================================

results = []

# =====================================
# DICE FUNCTION
# =====================================

def dice_score(
    pred,
    target
):

    intersection = (
        pred * target
    ).sum()

    return (
        2.0 * intersection + 1e-6
    ) / (
        pred.sum()
        + target.sum()
        + 1e-6
    )

# =====================================
# IOU FUNCTION
# =====================================

def iou_score(
    pred,
    target
):

    intersection = (
        pred * target
    ).sum()

    union = (
        pred + target
    ) > 0

    union = union.sum()

    return (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )

# =====================================
# PREDICTION LOOP
# =====================================

with torch.no_grad():

    for idx in range(len(dataset)):

        image, mask, boundary_gt = dataset[idx]

        image_tensor = (
            image.unsqueeze(0)
            .to(device)
        )

        seg_pred, boundary_pred = (
            model(image_tensor)
        )

        seg_pred = torch.sigmoid(
            seg_pred
        )

        boundary_pred = torch.sigmoid(
            boundary_pred
        )

        seg_pred = (
            seg_pred > 0.5
        ).float()

        boundary_pred = (
            boundary_pred > 0.5
        ).float()

        pred_mask = (
            seg_pred[0]
            .cpu()
            .squeeze()
            .numpy()
            .astype(np.uint8)
        )

        pred_boundary = (
            boundary_pred[0]
            .cpu()
            .squeeze()
            .numpy()
            .astype(np.uint8)
        )

        gt_mask = (
            mask.squeeze()
            .numpy()
            .astype(np.uint8)
        )

        # ==========================
        # METRICS
        # ==========================

        gt_flat = gt_mask.flatten()

        pred_flat = pred_mask.flatten()

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

        dice = dice_score(
            pred_mask,
            gt_mask
        )

        iou = iou_score(
            pred_mask,
            gt_mask
        )

        results.append([
            idx,
            acc,
            prec,
            rec,
            dice,
            iou
        ])

        # ==========================
        # SAVE MASK
        # ==========================

        Image.fromarray(
            pred_mask * 255
        ).save(
            pred_mask_dir /
            f"pred_{idx}.png"
        )

        # ==========================
        # SAVE BOUNDARY
        # ==========================

        Image.fromarray(
            pred_boundary * 255
        ).save(
            boundary_dir /
            f"boundary_{idx}.png"
        )

        # ==========================
        # COMPARISON IMAGE
        # ==========================

        original = (
            image.permute(1,2,0)
            .numpy()
            * 255
        ).astype(np.uint8)

        gt_vis = (
            gt_mask * 255
        ).astype(np.uint8)

        pred_vis = (
            pred_mask * 255
        ).astype(np.uint8)

        boundary_vis = (
            pred_boundary * 255
        ).astype(np.uint8)

        gt_vis = cv2.cvtColor(
            gt_vis,
            cv2.COLOR_GRAY2BGR
        )

        pred_vis = cv2.cvtColor(
            pred_vis,
            cv2.COLOR_GRAY2BGR
        )

        boundary_vis = cv2.cvtColor(
            boundary_vis,
            cv2.COLOR_GRAY2BGR
        )

        comparison = np.hstack([
            original,
            gt_vis,
            pred_vis,
            boundary_vis
        ])

        cv2.putText(
            comparison,
            "Original",
            (10,25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        cv2.putText(
            comparison,
            "GT",
            (270,25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        cv2.putText(
            comparison,
            "Prediction",
            (520,25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        cv2.putText(
            comparison,
            "Boundary",
            (780,25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        cv2.imwrite(
            str(
                comparison_dir /
                f"compare_{idx}.png"
            ),
            comparison
        )

        if idx % 20 == 0:

            print(
                f"Processed {idx}/{len(dataset)}"
            )

# =====================================
# SAVE CSV
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

csv_path = (
    SAVE_DIR /
    "metrics.csv"
)

df.to_csv(
    csv_path,
    index=False
)

# =====================================
# SUMMARY
# =====================================

print("\n========================")
print("CRA + BOUNDARY RESULTS")
print("========================")

print(
    "Accuracy :",
    df["Accuracy"].mean()
)

print(
    "Precision:",
    df["Precision"].mean()
)

print(
    "Recall   :",
    df["Recall"].mean()
)

print(
    "Dice     :",
    df["Dice"].mean()
)

print(
    "IoU      :",
    df["IoU"].mean()
)

print(
    "\nSaved to:",
    SAVE_DIR
)