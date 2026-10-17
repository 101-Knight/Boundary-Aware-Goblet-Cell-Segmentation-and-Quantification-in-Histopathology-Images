import os
import cv2
import torch
import numpy as np
import pandas as pd

from PIL import Image
from scipy import ndimage

from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from torch.utils.data import DataLoader

from dataset import GobletCellDataset
from model_scratch import get_model

# =====================================
# PATHS
# =====================================

IMAGE_DIR = r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\images"

MASK_DIR = r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\masks"

OUTPUT_DIR = r"E:\Goblet Cell\Predictions\Original Data\Patched_img_level\Morphology"

MASK_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    "predicted_masks"
)

os.makedirs(
    MASK_OUTPUT_DIR,
    exist_ok=True
)

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "morphology_results.csv"
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
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

# =====================================
# MODEL
# =====================================

model = get_model()

model.load_state_dict(
    torch.load(
        "best_unetplusplus_scratch.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

print("Model Loaded")

# =====================================
# RESULTS
# =====================================

results = []

# =====================================
# LOOP
# =====================================

with torch.no_grad():

    for idx, (image, _) in enumerate(loader):

        image = image.to(device)

        output = model(image)

        pred = torch.sigmoid(output)

        pred = (
            pred > 0.5
        ).float()

        pred_mask = (
            pred[0]
            .cpu()
            .squeeze()
            .numpy()
        )

        binary_mask = (
            pred_mask > 0
        ).astype(np.uint8)

        # ---------------------------------
        # Save Prediction
        # ---------------------------------

        save_name = dataset.images[idx].stem + ".png"

        Image.fromarray(
            binary_mask * 255
        ).save(
            os.path.join(
                MASK_OUTPUT_DIR,
                save_name
            )
        )

        # ---------------------------------
        # Watershed Counting
        # ---------------------------------

        distance = ndimage.distance_transform_edt(
            binary_mask
        )

        coords = peak_local_max(
            distance,
            footprint=np.ones((50, 50)),
            labels=binary_mask
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
                mask=binary_mask
            )

        else:

            labels = np.zeros_like(
                binary_mask
            )

        cell_count = labels.max()

        # ---------------------------------
        # Area Statistics
        # ---------------------------------

        areas = []

        for label_id in range(
            1,
            labels.max() + 1
        ):

            area = np.sum(
                labels == label_id
            )

            areas.append(area)

        if len(areas) > 0:

            mean_area = np.mean(areas)

            median_area = np.median(areas)

            largest_area = np.max(areas)

            smallest_area = np.min(areas)

        else:

            mean_area = 0
            median_area = 0
            largest_area = 0
            smallest_area = 0

        # ---------------------------------
        # Coverage
        # ---------------------------------

        total_goblet_area = np.sum(
            binary_mask
        )

        image_area = (
            binary_mask.shape[0]
            *
            binary_mask.shape[1]
        )

        coverage_percent = (
            total_goblet_area
            /
            image_area
        ) * 100

        cell_density = (
            cell_count
            /
            image_area
        )

        # ---------------------------------
        # Store
        # ---------------------------------

        results.append({

            "Image":
            save_name,

            "Cell_Count":
            cell_count,

            "Mean_Cell_Area":
            round(mean_area, 2),

            "Median_Cell_Area":
            round(median_area, 2),

            "Largest_Cell_Area":
            round(largest_area, 2),

            "Smallest_Cell_Area":
            round(smallest_area, 2),

            "Total_Goblet_Area":
            round(total_goblet_area, 2),

            "Coverage_Percent":
            round(coverage_percent, 2),

            "Cell_Density":
            round(cell_density, 6)
        })

        if idx % 20 == 0:

            print(
                f"Processed {idx+1}/{len(dataset)}"
            )

# =====================================
# SAVE CSV
# =====================================

df = pd.DataFrame(results)

df.to_csv(
    CSV_PATH,
    index=False
)

print("\nSaved CSV:")
print(CSV_PATH)

print("\nSaved Masks:")
print(MASK_OUTPUT_DIR)

print("\nDone.")