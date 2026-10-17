import torch
import numpy as np

from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from torch.utils.data import DataLoader

from dataset_distance import DistanceDataset
from model_unetpp_distance import UNetPPDistance

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

dataset = DistanceDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\images",
    mask_dir=r"E:\Goblet Cell\binary_test_masks",
    distance_dir=r"E:\Goblet Cell\distance_test_maps"
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

model = UNetPPDistance()

model.load_state_dict(
    torch.load(
        "best_unetpp_distance.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

# =====================================
# STORAGE
# =====================================

pred_counts = []
gt_counts = []

# =====================================
# COUNT FUNCTION
# =====================================

def count_cells(pred_mask, pred_distance):

    pred_mask = pred_mask.astype(np.uint8)

    if pred_mask.sum() == 0:
        return 0

    coords = peak_local_max(
        pred_distance,
        footprint=np.ones((50, 50)),
        labels=pred_mask
    )

    if len(coords) == 0:
        return 0

    markers = np.zeros_like(
        pred_distance,
        dtype=np.int32
    )

    markers[
        tuple(coords.T)
    ] = np.arange(
        1,
        len(coords) + 1
    )

    labels = watershed(
        -pred_distance,
        markers,
        mask=pred_mask.astype(bool)
    )

    return labels.max()

# =====================================
# EVALUATION
# =====================================

with torch.no_grad():

    for idx, (images, masks, gt_distance) in enumerate(loader):

        images = images.to(device)

        mask_pred, distance_pred = model(images)

        mask_pred = (
            torch.sigmoid(mask_pred) > 0.5
        ).float()

        distance_pred = (
            torch.sigmoid(distance_pred)
        )

        pred_mask = (
            mask_pred[0]
            .cpu()
            .squeeze()
            .numpy()
        )

        pred_distance = (
            distance_pred[0]
            .cpu()
            .squeeze()
            .numpy()
        )

        gt_mask = (
            masks[0]
            .cpu()
            .squeeze()
            .numpy()
        )

        # -------------------------
        # Predicted Count
        # -------------------------

        pred_count = count_cells(
            pred_mask,
            pred_distance
        )

        # -------------------------
        # Ground Truth Count
        # -------------------------

        _, gt_count = ndimage.label(
            gt_mask > 0
        )

        pred_counts.append(pred_count)
        gt_counts.append(gt_count)

        if idx % 20 == 0:

            print(
                f"Image {idx+1}/{len(dataset)} | "
                f"GT={gt_count} | "
                f"Pred={pred_count}"
            )

        if idx < 5:

            print(
                "Distance Min:",
                pred_distance.min(),
                "Distance Max:",
                pred_distance.max(),
                "Distance Mean:",
                pred_distance.mean()
            )
# =====================================
# METRICS
# =====================================

pred_counts = np.array(pred_counts)
gt_counts = np.array(gt_counts)

errors = np.abs(
    pred_counts - gt_counts
)

mae = np.mean(errors)

rmse = np.sqrt(
    np.mean(
        (pred_counts - gt_counts) ** 2
    )
)

valid = gt_counts > 0

if np.sum(valid) > 0:

    mape = np.mean(
        errors[valid]
        /
        gt_counts[valid]
    ) * 100

else:

    mape = 0

signed_error = np.mean(
    pred_counts - gt_counts
)

# =====================================
# RESULTS
# =====================================

print("\n==========================")
print("DISTANCE-AWARE COUNTING")
print("==========================")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")

print(
    "Mean GT Count:",
    np.mean(gt_counts)
)

print(
    "Mean Pred Count:",
    np.mean(pred_counts)
)

print(
    "Median Error:",
    np.median(errors)
)

print(
    "Max Error:",
    np.max(errors)
)

print(
    "Min Error:",
    np.min(errors)
)

print(
    "Mean Signed Error:",
    signed_error
)