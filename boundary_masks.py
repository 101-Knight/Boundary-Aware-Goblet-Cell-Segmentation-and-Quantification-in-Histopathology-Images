from pathlib import Path
from PIL import Image
import numpy as np
from scipy.ndimage import binary_dilation

INPUT_DIR = r"E:\Goblet Cell\binary_test_masks"
OUTPUT_DIR = r"E:\Goblet Cell\boundary_test_masks"

Path(OUTPUT_DIR).mkdir(exist_ok=True)

for mask_path in Path(INPUT_DIR).glob("*.png"):

    mask = np.array(
        Image.open(mask_path)
    )

    mask = mask > 0

    dilated = binary_dilation(
        mask,
        iterations=2
    )

    boundary = (
        dilated.astype(np.uint8)
        - mask.astype(np.uint8)
    )

    Image.fromarray(
        boundary * 255
    ).save(
        Path(OUTPUT_DIR) /
        mask_path.name
    )

print("Boundary masks created.")