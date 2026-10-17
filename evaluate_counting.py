import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from torch.utils.data import DataLoader

from dataset import GobletCellDataset
from model import get_model

# =====================================
# OUTPUT PATH
# =====================================

OUTPUT_DIR = Path(
    r"E:\Goblet Cell\Clean images\Predictions\Pretrained unet++"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PRED_DIR = OUTPUT_DIR / "Watershed_Predictions"

PRED_DIR.mkdir(
    exist_ok=True
)

CSV_FILE = OUTPUT_DIR / "counting_results.csv"

# =====================================
# DEVICE
# =====================================

device = torch.device(
    "xpu"
    if hasattr(torch, "xpu")
    and torch.xpu.is_available()
    else "cpu"
)

print("Using Device:", device)

# =====================================
# DATASET
# =====================================

dataset = GobletCellDataset(
    image_dir=r"E:\Goblet Cell\Clean images\Test\images",
    mask_dir=r"E:\Goblet Cell\Clean images\Test\masks"
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

print("Test Images:", len(dataset))

# =====================================
# LOAD MODEL
# =====================================

model = get_model()

model.load_state_dict(
    torch.load(
        "best_unetplusplus_model.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

# =====================================
# WATERSHED FUNCTION
# =====================================

def watershed_count(mask):

    binary = mask > 0

    distance = ndimage.distance_transform_edt(
        binary
    )

    coords = peak_local_max(
        distance,
        footprint=np.ones((50, 50)),
        labels=binary
    )

    markers = np.zeros_like(
        distance,
        dtype=int
    )

    for i, coord in enumerate(coords):

        markers[
            coord[0],
            coord[1]
        ] = i + 1

    labels = watershed(
        -distance,
        markers,
        mask=binary
    )

    return labels.max(), labels

# =====================================
# EVALUATION
# =====================================

results = []

gt_counts = []
pred_counts = []

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

        # -----------------------------
        # NUMPY CONVERSION
        # -----------------------------

        image_np = (
            images[0]
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )

        pred_np = (
            preds[0]
            .cpu()
            .squeeze()
            .numpy()
        )

        gt_np = (
            masks[0]
            .cpu()
            .squeeze()
            .numpy()
        )

        # -----------------------------
        # WATERSHED COUNTS
        # -----------------------------

        gt_count, gt_labels = (
            watershed_count(gt_np)
        )

        pred_count, pred_labels = (
            watershed_count(pred_np)
        )

        error = abs(
            gt_count - pred_count
        )

        gt_counts.append(
            gt_count
        )

        pred_counts.append(
            pred_count
        )

        results.append({

            "Image_ID":
            idx + 1,

            "GT_Count":
            gt_count,

            "Pred_Count":
            pred_count,

            "Absolute_Error":
            error
        })

        # -----------------------------
        # SAVE FIGURE
        # -----------------------------

        plt.figure(
            figsize=(15, 8)
        )

        plt.subplot(2, 3, 1)
        plt.title("Image")
        plt.imshow(image_np)

        plt.subplot(2, 3, 2)
        plt.title(
            f"GT Mask"
        )
        plt.imshow(gt_np)

        plt.subplot(2, 3, 3)
        plt.title(
            f"Pred Mask"
        )
        plt.imshow(pred_np)

        plt.subplot(2, 3, 5)
        plt.title(
            f"GT Count = {gt_count}"
        )
        plt.imshow(gt_labels)

        plt.subplot(2, 3, 6)
        plt.title(
            f"Pred Count = {pred_count}"
        )
        plt.imshow(pred_labels)

        save_path = (
            PRED_DIR /
            f"count_{idx+1}.png"
        )

        plt.savefig(
            save_path,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Processed "
            f"{idx+1}/{len(dataset)}"
        )

# =====================================
# CSV
# =====================================

df = pd.DataFrame(
    results
)

df.to_csv(
    CSV_FILE,
    index=False
)

print("\nCSV Saved:")
print(CSV_FILE)

# =====================================
# FINAL METRICS
# =====================================

gt_counts = np.array(
    gt_counts
)

pred_counts = np.array(
    pred_counts
)

mae = np.mean(
    np.abs(
        pred_counts -
        gt_counts
    )
)

rmse = np.sqrt(
    np.mean(
        (
            pred_counts -
            gt_counts
        ) ** 2
    )
)

valid = gt_counts > 0

mape = np.mean(
    np.abs(
        pred_counts[valid] -
        gt_counts[valid]
    )
    /
    gt_counts[valid]
) * 100

errors = np.abs(pred_counts - gt_counts)
print("\n==========================")
print("GOBLET CELL COUNTING")
print("==========================")

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"MAPE : {mape:.2f}%"
)
print("Mean GT Count:", np.mean(gt_counts))
print("Mean Pred Count:", np.mean(pred_counts))

print("Median Error:", np.median(errors))
print("Max Error:", np.max(errors))
print("Min Error:", np.min(errors))

# --------------------------------
# WORST IMAGES
# --------------------------------

errors = np.abs(pred_counts - gt_counts)

sorted_idx = np.argsort(errors)[::-1]

print("\n===== WORST 10 IMAGES =====")

for i in range(10):

    idx = sorted_idx[i]

    print(
        f"Image {idx+1}: "
        f"GT={gt_counts[idx]}, "
        f"Pred={pred_counts[idx]}, "
        f"Error={errors[idx]}"
    )

signed_error = pred_counts - gt_counts

print("Mean Signed Error:",
      np.mean(signed_error))