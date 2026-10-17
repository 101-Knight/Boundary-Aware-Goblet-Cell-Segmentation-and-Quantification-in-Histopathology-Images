import torch
import numpy as np
from skimage.segmentation import watershed
from scipy import ndimage
from scipy.ndimage import maximum_filter

from torch.utils.data import DataLoader

from dataset import GobletCellDataset
from model import get_model

device = torch.device(
"xpu"
if hasattr(torch, "xpu")
and torch.xpu.is_available()
else "cpu"
)

print("Using:", device)

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

model = get_model()

model.load_state_dict(
torch.load(
"best_mccl_student.pth",
map_location=device
)
)

model.to(device)
model.eval()

pred_counts = []
gt_counts = []

with torch.no_grad():

    for images, masks in loader:

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

        distance = ndimage.distance_transform_edt(
            pred_mask
        )

        from skimage.feature import peak_local_max

        coords = peak_local_max(
            distance,
            footprint=np.ones((50,50)),
            labels=pred_mask.astype(np.uint8)
        )

        if len(coords) == 0:

            pred_count = 0

        else:

            markers = np.zeros_like(
                distance,
                dtype=np.int32
            )

            markers[
                tuple(coords.T)
            ] = np.arange(
                1,
                len(coords)+1
            )

            labels = watershed(
                -distance,
                markers,
                mask=pred_mask.astype(bool)
            )

            pred_count = labels.max()

        _, gt_count = ndimage.label(
            gt_mask > 0
        )

        pred_counts.append(pred_count)
        gt_counts.append(gt_count)

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

mape = (
np.mean(
errors[valid]
/
gt_counts[valid]
) * 100
)

print("\n==========================")
print("MCCL COUNTING")
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
np.mean(
pred_counts - gt_counts
)
)