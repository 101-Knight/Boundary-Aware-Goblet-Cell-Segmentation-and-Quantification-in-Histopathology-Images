import cv2
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image

from torchvision import transforms

from model_unetpp_arp import UNetPP_ARP
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score
)

# =====================================
# DEVICE
# =====================================

if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

print("Using:", device)

# =====================================
# MODEL
# =====================================

model = UNetPP_ARP()

model.load_state_dict(
    torch.load(
        "best_unetpp_arp_ssl_5.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model Loaded")

# =====================================
# DATASET PATHS
# =====================================

image_dir = Path(
    r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\images"
)

mask_dir = Path(
    r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\masks"
)

# =====================================
# OUTPUT FOLDER
# =====================================

output_dir = Path(
    r"E:\Goblet Cell\Predictions\Original Data\Patched_img_level\Trained ARP + Unet++\5%-95%"
)

comparison_dir = (
    output_dir /
    "comparisons"
)

output_dir.mkdir(
    exist_ok=True
)

comparison_dir.mkdir(
    exist_ok=True
)

# =====================================
# TRANSFORM
# =====================================

transform = transforms.ToTensor()

# =====================================
# RESULTS
# =====================================

results = []

# =====================================
# INFERENCE
# =====================================

with torch.no_grad():

    image_paths = sorted(
        image_dir.glob("*")
    )

    print(
        "Total Images:",
        len(image_paths)
    )

    for idx, image_path in enumerate(
        image_paths
    ):

        print(
            f"Processing "
            f"{idx+1}/{len(image_paths)}"
        )

        mask_path = (
            mask_dir /
            f"{image_path.stem}.png"
        )

        # =============================
        # IMAGE
        # =============================

        image = Image.open(
            image_path
        ).convert("RGB")

        image_np = np.array(
            image
        )

        image_tensor = transform(
            image
        ).unsqueeze(0)

        image_tensor = image_tensor.to(
            device
        )

        # =============================
        # PREDICTION
        # =============================

        output = model(
            image_tensor
        )

        pred_mask = (
            torch.sigmoid(
                output
            ) > 0.5
        )

        pred_mask = (
            pred_mask.squeeze()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        # =============================
        # GROUND TRUTH
        # =============================

        gt_mask = np.array(
            Image.open(
                mask_path
            ).convert("L")
        )

        gt_mask = (
            gt_mask > 127
        ).astype(np.uint8)

        # =============================
        # METRICS
        # =============================

        # =====================================
        # FLATTEN MASKS
        # =====================================

        gt_flat = gt_mask.flatten()

        pred_flat = pred_mask.flatten()

        # =====================================
        # ACCURACY
        # =====================================

        acc = accuracy_score(
            gt_flat,
            pred_flat
        )

        # =====================================
        # PRECISION
        # =====================================

        prec = precision_score(
            gt_flat,
            pred_flat,
            zero_division=0
        )

        # =====================================
        # RECALL
        # =====================================

        rec = recall_score(
            gt_flat,
            pred_flat,
            zero_division=0
        )

        # =====================================
        # DICE
        # =====================================

        intersection = np.logical_and(
            pred_mask,
            gt_mask
        ).sum()

        dice = (
            2 * intersection
        ) / (
            pred_mask.sum()
            +
            gt_mask.sum()
            +
            1e-8
        )

        # =====================================
        # IOU
        # =====================================

        union = np.logical_or(
            pred_mask,
            gt_mask
        ).sum()

        iou = (
            intersection
        ) / (
            union
            +
            1e-8
        )

        # =====================================
        # SAVE RESULTS
        # =====================================

        results.append(
            [
                image_path.name,
                acc,
                prec,
                rec,
                dice,
                iou
            ]
        )

        # =============================
        # VISUALIZATION
        # =============================

        gt_vis = np.zeros_like(
            image_np
        )

        gt_vis[:, :, 1] = (
            gt_mask * 255
        )

        pred_vis = np.zeros_like(
            image_np
        )

        pred_vis[:, :, 2] = (
            pred_mask * 255
        )

        overlay = image_np.copy()

        overlay[gt_mask == 1] = [
            0,
            255,
            0
        ]

        overlay[pred_mask == 1] = [
            255,
            0,
            0
        ]

        # =============================
        # COMPARISON PANEL
        # =============================

        comparison = np.hstack(
            [
                image_np,
                gt_vis,
                pred_vis,
                overlay
            ]
        )

        h = image_np.shape[0]
        w = image_np.shape[1]

        label_bar = np.ones(
            (
                60,
                comparison.shape[1],
                3
            ),
            dtype=np.uint8
        ) * 255

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(
            label_bar,
            "Original",
            (20,40),
            font,
            1,
            (0,0,0),
            2
        )

        cv2.putText(
            label_bar,
            "Ground Truth",
            (w + 20,40),
            font,
            1,
            (0,0,0),
            2
        )

        cv2.putText(
            label_bar,
            "Prediction",
            (2*w + 20,40),
            font,
            1,
            (0,0,0),
            2
        )

        cv2.putText(
            label_bar,
            "Overlay",
            (3*w + 20,40),
            font,
            1,
            (0,0,0),
            2
        )

        comparison = np.vstack(
            [
                label_bar,
                comparison
            ]
        )

        save_path = (
            comparison_dir /
            f"{image_path.stem}.png"
        )

        cv2.imwrite(
            str(save_path),
            cv2.cvtColor(
                comparison,
                cv2.COLOR_RGB2BGR
            )
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
    output_dir /
    "arp_ssl_metrics.csv"
)
print(df.head())
print(df.shape)

df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8"
)

print("\n================================")
print("Prediction Complete")
print("================================")

print(
    "Average Accuracy:",
    df["Accuracy"].mean()
)

print(
    "Average Precision:",
    df["Precision"].mean()
)

print(
    "Average Recall:",
    df["Recall"].mean()
)

print(
    "Average Dice:",
    df["Dice"].mean()
)

print(
    "Average IoU:",
    df["IoU"].mean()
)