from PIL import Image
import numpy as np
from pathlib import Path

input_dir = Path(
    r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\masks"
)

output_dir = Path(
    r"E:\Goblet Cell\binary_test_masks"
)

output_dir.mkdir(exist_ok=True)

for mask_path in input_dir.glob("*.png"):

    mask = Image.open(mask_path)

    mask = np.array(mask)

    binary = (mask > 0).astype(np.uint8)

    binary *= 255

    Image.fromarray(binary).save(
        output_dir / mask_path.name
    )

print("Done")