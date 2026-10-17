from PIL import Image
import numpy as np

mask = Image.open(
    r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\train\masks\fbs01_0.png"
)

mask = np.array(mask)

print(np.unique(mask))