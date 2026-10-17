from pathlib import Path
from PIL import Image
import numpy as np
from scipy.ndimage import distance_transform_edt

MASK_DIR = r"E:\Goblet Cell\binary_test_masks"
OUTPUT_DIR = r"E:\Goblet Cell\distance_test_maps"

Path(OUTPUT_DIR).mkdir(exist_ok=True)

for mask_file in Path(MASK_DIR).glob("*.png"):

    mask = np.array(
        Image.open(mask_file)
    )

    mask = mask > 0

    dist = distance_transform_edt(mask)

    if dist.max() > 0:
        dist = dist / dist.max()

    dist = (dist * 255).astype(np.uint8)

    Image.fromarray(dist).save(
        Path(OUTPUT_DIR) /
        mask_file.name
    )

print("Done")