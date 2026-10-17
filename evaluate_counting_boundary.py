import torch
import numpy as np

from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from torch.utils.data import DataLoader

from dataset_boundary import BoundaryDataset
from model_unetpp_boundary import UNetPPBoundary

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

dataset = BoundaryDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\images",
    mask_dir=r"E:\Goblet Cell\binary_test_masks",
    boundary_dir=r"E:\Goblet Cell\boundary_test_masks"
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

model = UNetPPBoundary()

model.load_state_dict(
    torch.load(
        "best_unetpp_boundary.pth",
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
# HELPER FUNCTION
# =====================================

def watershed_count(binary_mask):

    binary_mask = binary_mask.astype(np.uint8)

    if binary_mask.sum() == 0:
        return 0

    distance = ndimage.distance_transform_edt(
        binary_mask
    )

    coords = peak_local_max(
        distance,
        footprint=np.ones((50, 50)),
        labels=binary_mask
    )

    if len(coords) == 0:
        return 0

    markers = np.zeros_like(
        distance,
        dtype=np.int32
    )

    markers[
        tuple(coords.T)
    ] = np.arange(
        1,
        len(coords) + 1
    )

    labels = watershed(
        -distance,
        markers,
        mask=binary_mask.astype(bool)
    )

    return labels.max()

# =====================================
# LOOP
# =====================================

with torch.no_grad():

    for idx, (images, masks, boundaries) in enumerate(loader):

        try:

            images = images.to(device)

            # -------------------------
            # PREDICTION
            # -------------------------

            mask_pred, boundary_pred = model(images)

            mask_pred = (
                torch.sigmoid(mask_pred) > 0.5
            ).float()

            boundary_pred = (
                torch.sigmoid(boundary_pred) > 0.5
            ).float()

            refined_mask = (
                mask_pred - boundary_pred
            )

            refined_mask[
                refined_mask < 0
            ] = 0

            pred_np = (
                refined_mask[0]
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

            # -------------------------
            # COUNTS
            # -------------------------

            pred_count = watershed_count(
                pred_np
            )

            gt_count = watershed_count(
                gt_np
            )

            pred_counts.append(
                pred_count
            )

            gt_counts.append(
                gt_count
            )

            if idx % 20 == 0:

                print(
                    f"Image {idx+1}/{len(dataset)} | "
                    f"GT={gt_count} | "
                    f"Pred={pred_count}"
                )

        except Exception as e:

            print(
                f"Error on image {idx+1}: {e}"
            )

            continue

# =====================================
# CHECK
# =====================================

print("\nProcessed Images:", len(pred_counts))

if len(pred_counts) == 0:

    print(
        "No valid counts generated."
    )

    exit()

# =====================================
# NUMPY ARRAYS
# =====================================

pred_counts = np.array(
    pred_counts
)

gt_counts = np.array(
    gt_counts
)

errors = np.abs(
    pred_counts - gt_counts
)

signed_errors = (
    pred_counts - gt_counts
)

# =====================================
# METRICS
# =====================================

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

# =====================================
# RESULTS
# =====================================

print("\n==========================")
print("GOBLET CELL COUNTING")
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
    np.mean(signed_errors)
)

# =====================================
# WORST 10 IMAGES
# =====================================

sorted_idx = np.argsort(
    errors
)[::-1]

print("\n===== WORST 10 IMAGES =====")

for i in range(
    min(10, len(sorted_idx))
):

    idx = sorted_idx[i]

    print(
        f"Image {idx+1}: "
        f"GT={gt_counts[idx]}, "
        f"Pred={pred_counts[idx]}, "
        f"Error={errors[idx]}"
    )