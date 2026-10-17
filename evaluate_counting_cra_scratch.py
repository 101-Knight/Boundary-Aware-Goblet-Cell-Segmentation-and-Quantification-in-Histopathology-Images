import torch
import numpy as np

from scipy import ndimage
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

dataset = GobletCellDataset(
image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\images",
mask_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\masks"
)

loader = DataLoader(
dataset,
batch_size=1,
shuffle=False
)

print(
"Test Images:",
len(dataset)
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

pred_counts = []
gt_counts = []

# =====================================
# LOOP
# =====================================

with torch.no_grad():
    for idx, (
        images,
        masks
    ) in enumerate(loader):

        images = images.to(device)

        outputs = model(
            images
        )

        preds = (
            torch.sigmoid(outputs)
            > 0.5
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
        # CONNECTED COMPONENTS
        # ==========================

        _, pred_count = ndimage.label(
            pred_mask > 0
        )

        _, gt_count = ndimage.label(
            gt_mask > 0
        )

        pred_counts.append(
            pred_count
        )

        gt_counts.append(
            gt_count
        )

        if idx % 20 == 0:

            print(
                f"Image {idx+1}/{len(dataset)} "
                f"| GT={gt_count} "
                f"| Pred={pred_count}"
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
(
pred_counts
-
gt_counts
) ** 2
)
)

valid = gt_counts > 0

if np.sum(valid) > 0:
    mape = (
        np.mean(
            errors[valid]
            /
            gt_counts[valid]
        )
        * 100
    )

else:
    mape = 0

signed_error = (
pred_counts
-
gt_counts
)

# =====================================
# RESULTS
# =====================================

print("\n==========================")
print("CRA SCRATCH COUNTING")
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
np.mean(signed_error)
)