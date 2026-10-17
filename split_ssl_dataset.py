import random
import shutil
import cv2
import numpy as np

from pathlib import Path
from PIL import Image

# =====================================
# PATHS
# =====================================

image_dir = Path(
    r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\train\images"
)

mask_dir = Path(
    r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\train\masks"
)

# =====================================
# SSL OUTPUT
# =====================================

labeled_img_dir = Path(
    r"E:\Goblet Cell\SSL_Data\labeled\images"
)

labeled_mask_dir = Path(
    r"E:\Goblet Cell\SSL_Data\labeled\masks"
)

labeled_boundary_dir = Path(
    r"E:\Goblet Cell\SSL_Data\labeled\boundaries"
)

unlabeled_img_dir = Path(
    r"E:\Goblet Cell\SSL_Data\unlabeled\images"
)

# =====================================
# CREATE DIRECTORIES
# =====================================

for folder in [
    labeled_img_dir,
    labeled_mask_dir,
    labeled_boundary_dir,
    unlabeled_img_dir
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )

# =====================================
# BOUNDARY CREATION
# =====================================

def create_boundary(mask):

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    dilated = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    eroded = cv2.erode(
        mask,
        kernel,
        iterations=1
    )

    boundary = (
        dilated - eroded
    )

    boundary = (
        boundary > 0
    ).astype(np.uint8)

    return boundary * 255

# =====================================
# READ IMAGES
# =====================================

images = sorted(
    list(image_dir.glob("*"))
)

print(
    "Total Images:",
    len(images)
)

# =====================================
# SHUFFLE
# =====================================

random.seed(42)

random.shuffle(images)

# =====================================
# CHANGE RATIO HERE
# =====================================

LABELED_RATIO = 0.05

num_labeled = int(
    LABELED_RATIO * len(images)
)

labeled_images = images[:num_labeled]

unlabeled_images = images[num_labeled:]

print(
    "Labeled:",
    len(labeled_images)
)

print(
    "Unlabeled:",
    len(unlabeled_images)
)

# =====================================
# LABELED DATA
# =====================================

for img_path in labeled_images:

    mask_path = (
        mask_dir /
        f"{img_path.stem}.png"
    )

    shutil.copy(
        img_path,
        labeled_img_dir / img_path.name
    )

    shutil.copy(
        mask_path,
        labeled_mask_dir / mask_path.name
    )

    # ----------------------------
    # Generate Boundary Mask
    # ----------------------------

    mask = Image.open(
        mask_path
    ).convert("L")

    mask = np.array(mask)

    mask = (
        mask > 127
    ).astype(np.uint8)

    boundary = create_boundary(
        mask
    )

    boundary_name = (
        labeled_boundary_dir /
        f"{img_path.stem}.png"
    )

    cv2.imwrite(
        str(boundary_name),
        boundary
    )

# =====================================
# UNLABELED DATA
# =====================================

for img_path in unlabeled_images:

    shutil.copy(
        img_path,
        unlabeled_img_dir / img_path.name
    )

print("\nSSL Dataset Created Successfully")

print(
    "Boundary masks generated:",
    len(list(labeled_boundary_dir.glob("*")))
)