import torch
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from torch.utils.data import DataLoader

from dataset_boundary import BoundaryDataset
from model_unetpp_boundary import UNetPPBoundary

# =====================================
# SETTINGS
# =====================================

MODEL_PATH = "best_unetpp_boundary.pth"

OUTPUT_DIR = Path(
    r"E:\Goblet Cell\Predictions\Original Data\Patched_img_level\Pretrained Boundary Unet++"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PRED_DIR = OUTPUT_DIR / "Predictions"
PRED_DIR.mkdir(exist_ok=True)

CSV_FILE = OUTPUT_DIR / "metrics_unetpp_boundary.csv"

# =====================================
# DEVICE
# =====================================

if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

print("Using:", device)

# =====================================
# DATASET
# =====================================

test_dataset = BoundaryDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\images",
    mask_dir=r"E:\Goblet Cell\binary_test_masks",
    boundary_dir=r"E:\Goblet Cell\boundary_test_masks"
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

model = UNetPPBoundary()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)
model.eval()

# =====================================
# METRICS
# =====================================

results = []

eps = 1e-7

# =====================================
# PREDICTION LOOP
# =====================================

with torch.no_grad():

    for idx, (images, masks, boundaries) in enumerate(test_loader):

        images = images.to(device)
        masks = masks.to(device)

        mask_pred, boundary_pred = model(images)

        mask_pred = torch.sigmoid(mask_pred)
        boundary_pred = torch.sigmoid(boundary_pred)

        mask_pred = (
            mask_pred > 0.5
        ).float()

        boundary_pred = (
            boundary_pred > 0.5
        ).float()

        # ==========================
        # REFINE MASK
        # ==========================

        refined_mask = (
            mask_pred - boundary_pred
        )

        refined_mask[
            refined_mask < 0
        ] = 0

        # ==========================
        # METRICS
        # ==========================

        pred_flat = refined_mask.view(-1)
        mask_flat = masks.view(-1)

        TP = (
            pred_flat * mask_flat
        ).sum()

        FP = (
            pred_flat *
            (1 - mask_flat)
        ).sum()

        FN = (
            (1 - pred_flat) *
            mask_flat
        ).sum()

        TN = (
            (1 - pred_flat) *
            (1 - mask_flat)
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

        results.append({

            "Image_ID": idx + 1,

            "Accuracy":
            accuracy.item(),

            "Precision":
            precision.item(),

            "Recall":
            recall.item(),

            "Dice":
            dice.item(),

            "IoU":
            iou.item()

        })

        # ==========================
        # SAVE FIGURE
        # ==========================

        image_np = (
            images[0]
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )

        gt_mask_np = (
            masks[0]
            .cpu()
            .squeeze()
            .numpy()
        )

        pred_mask_np = (
            mask_pred[0]
            .cpu()
            .squeeze()
            .numpy()
        )

        boundary_np = (
            boundary_pred[0]
            .cpu()
            .squeeze()
            .numpy()
        )

        refined_np = (
            refined_mask[0]
            .cpu()
            .squeeze()
            .numpy()
        )

        plt.figure(
            figsize=(20, 4)
        )

        plt.subplot(1, 5, 1)
        plt.title("Image")
        plt.imshow(image_np)

        plt.subplot(1, 5, 2)
        plt.title("GT Mask")
        plt.imshow(gt_mask_np)

        plt.subplot(1, 5, 3)
        plt.title("Mask Pred")
        plt.imshow(pred_mask_np)

        plt.subplot(1, 5, 4)
        plt.title("Boundary Pred")
        plt.imshow(boundary_np)

        plt.subplot(1, 5, 5)
        plt.title("Refined Mask")
        plt.imshow(refined_np)

        save_path = (
            PRED_DIR
            /
            f"prediction_{idx+1}.png"
        )

        plt.savefig(
            save_path,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved "
            f"{idx+1}/{len(test_dataset)}"
        )

# =====================================
# SAVE CSV
# =====================================

df = pd.DataFrame(results)

df.to_csv(
    CSV_FILE,
    index=False
)

print("\nCSV Saved:")
print(CSV_FILE)

# =====================================
# AVERAGE METRICS
# =====================================

print("\n===== Average Results =====")

print(
    f"Accuracy : "
    f"{df['Accuracy'].mean():.4f}"
)

print(
    f"Precision: "
    f"{df['Precision'].mean():.4f}"
)

print(
    f"Recall   : "
    f"{df['Recall'].mean():.4f}"
)

print(
    f"Dice     : "
    f"{df['Dice'].mean():.4f}"
)

print(
    f"IoU      : "
    f"{df['IoU'].mean():.4f}"
)