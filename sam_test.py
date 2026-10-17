import torch
import cv2
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator

# =====================================
# DEVICE
# =====================================

if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = "xpu"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)

# =====================================
# LOAD SAM
# =====================================

sam_checkpoint = r"E:\Goblet Cell\SAM\sam_vit_b_01ec64.pth"

sam = sam_model_registry["vit_b"](
    checkpoint=sam_checkpoint
)

sam.to(device)

print("SAM loaded successfully!")

# =====================================
# LOAD IMAGE
# =====================================

image_path = r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\images"

import os

first_image = sorted(os.listdir(image_path))[0]

image = cv2.imread(
    os.path.join(image_path, first_image)
)

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

print("Image loaded:", first_image)

# =====================================
# GENERATE MASKS
# =====================================

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

print("Number of masks found:", len(masks))

# =====================================
# SHOW LARGEST MASK
# =====================================

largest_mask = max(
    masks,
    key=lambda x: x["area"]
)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1,2,2)
plt.title("Largest SAM Mask")
plt.imshow(largest_mask["segmentation"])

plt.show()