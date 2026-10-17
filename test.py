from pathlib import Path
from PIL import Image

image_dir = Path(r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\train\images")
mask_dir = Path(r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\train\masks")

images = sorted(image_dir.glob("*"))
masks = sorted(mask_dir.glob("*"))

print("Images:", len(images))
print("Masks:", len(masks))

img = Image.open(images[0])
mask = Image.open(masks[0])

print("Image size:", img.size)
print("Mask size:", mask.size)