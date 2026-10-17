import torch
import numpy as np

from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from torch.utils.data import DataLoader

from dataset import GobletCellDataset
from model_unetpp_arp import UNetPP_ARP

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

model = UNetPP_ARP()

model.load_state_dict(
    torch.load(
        "best_unetpp_arp.pth",
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
# COUNTING LOOP
# =====================================

with torch.no_grad():

    for idx, (images, masks) in enumerate(loader):

        images = images.to(device)

        outputs = model(images)

        preds = torch.sigmoid(outputs)

        preds = (
            preds > 0.5
        ).float()

        pred_mask = (
            preds[0]
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

        # ==========================
        # PREDICTED CELL COUNT
        # ==========================

        distance = ndimage.distance_transform_edt(
            pred_mask
        )

        coords = peak_local_max(
            distance,
            footprint=np.ones((50, 50)),
            labels=pred_mask.astype(np.uint8)
        )

        markers = np.zeros_like(
            distance,
            dtype=np.int32
        )

        if len(coords) > 0:

            markers[
                tuple(coords.T)
            ] = np.arange(
                1,
                len(coords) + 1
            )

            labels = watershed(
                -distance,
                markers,
                mask=pred_mask.astype(bool)
            )

            pred_count = labels.max()

        else:

            pred_count = 0

        # ==========================
        # GT CELL COUNT
        # ==========================

        gt_distance = ndimage.distance_transform_edt(
            gt_mask
        )

        gt_coords = peak_local_max(
            gt_distance,
            footprint=np.ones((50, 50)),
            labels=gt_mask.astype(np.uint8)
        )

        gt_markers = np.zeros_like(
            gt_distance,
            dtype=np.int32
        )

        if len(gt_coords) > 0:

            gt_markers[
                tuple(gt_coords.T)
            ] = np.arange(
                1,
                len(gt_coords) + 1
            )

            gt_labels = watershed(
                -gt_distance,
                gt_markers,
                mask=gt_mask.astype(bool)
            )

            gt_count = gt_labels.max()

        else:

            gt_count = 0

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

# =====================================
# METRICS
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

mae = np.mean(
    errors
)

rmse = np.sqrt(
    np.mean(
        (pred_counts - gt_counts) ** 2
    )
)

valid = gt_counts > 0

if np.sum(valid) > 0:

    mape = (
        np.mean(
            errors[valid]
            / gt_counts[valid]
        ) * 100
    )

else:

    mape = 0

signed_error = np.mean(
    pred_counts - gt_counts
)

# =====================================
# RESULTS
# =====================================

print("\n==========================")
print("ARP COUNTING")
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

# =====================================
# WORST 10 IMAGES
# =====================================

worst = np.argsort(errors)[-10:]

print("\n===== WORST 10 IMAGES =====")

for i in worst[::-1]:

    print(
        f"Image {i+1}: "
        f"GT={gt_counts[i]}, "
        f"Pred={pred_counts[i]}, "
        f"Error={errors[i]}"
    )